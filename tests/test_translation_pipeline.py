from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import ANY, patch

import vid_to_sub_app.cli.translation as translation_module
from vid_to_sub_app.cli.main import build_parser, main
from vid_to_sub_app.cli.manifest import ProcessResult
from vid_to_sub_app.cli.runner import _assess_stage1_quality, process_one, run_stage1, run_stage2
from vid_to_sub_app.cli.stage_artifact import (
    ARTIFACT_FILENAME_SUFFIX,
    ARTIFACT_SCHEMA_VERSION,
    StageArtifact,
    artifact_path_for,
    build_stage_artifact_metadata,
    load_stage_artifact,
    write_stage_artifact,
)
from vid_to_sub_app.cli.translation import (
    TranslationContractError,
    _build_retry_schedule,
    postprocess_translated_segments_openai_compatible,
    translate_segments_openai_compatible,
)
from vid_to_sub_app.shared.constants import EVENT_PREFIX


class _FakeHTTPResponse:
    def __init__(
        self,
        payload: dict[str, Any],
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.status = status
        self.headers = headers or {}

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _chat_payload(content: str, *, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "id": "resp-123",
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
    }


def _event_payloads(buffer: io.StringIO) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in buffer.getvalue().splitlines():
        if not line.startswith(EVENT_PREFIX + " "):
            continue
        payloads.append(json.loads(line.split(" ", 1)[1]))
    return payloads


class TranslationPipelineTests(unittest.TestCase):
    def test_translation_max_concurrency_defaults_to_one_for_invalid_env(self) -> None:
        with patch.dict(
            os.environ,
            {"VID_TO_SUB_TRANSLATION_MAX_CONCURRENCY": "not-a-number"},
            clear=False,
        ):
            self.assertEqual(1, translation_module._translation_max_concurrency())

    def test_retry_schedule_uses_expected_ladder(self) -> None:
        self.assertEqual([100, 50, 20, 10, 5, 1], _build_retry_schedule(100))
        self.assertEqual([20, 10, 5, 1], _build_retry_schedule(20))

    def test_translation_requests_run_inside_bounded_semaphore(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        class _RecordingSemaphore:
            def __init__(self) -> None:
                self.enter_count = 0
                self.exit_count = 0
                self.active_count = 0

            def __enter__(self) -> "_RecordingSemaphore":
                self.enter_count += 1
                self.active_count += 1
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                self.exit_count += 1
                self.active_count -= 1
                return False

        semaphore = _RecordingSemaphore()

        def fake_urlopen_with_timeout(request, timeout_sec):
            self.assertEqual(1, semaphore.active_count)
            self.assertGreater(timeout_sec, 0)
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "안녕"}], ensure_ascii=False
                    )
                )
            )

        with (
            patch(
                "vid_to_sub_app.cli.translation._TRANSLATION_SEMAPHORE",
                semaphore,
            ),
            patch(
                "vid_to_sub_app.cli.translation._urlopen_with_timeout",
                side_effect=fake_urlopen_with_timeout,
            ),
        ):
            with redirect_stdout(io.StringIO()):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                )

        self.assertEqual(["안녕"], [segment["text"] for segment in translated_segments])
        self.assertEqual(1, semaphore.enter_count)
        self.assertEqual(1, semaphore.exit_count)
        self.assertEqual(0, semaphore.active_count)

    def test_translate_segments_openai_compatible_preserves_segment_boundaries(
        self,
    ) -> None:
        segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello"},
            {"start": 1.5, "end": 3.0, "text": "World"},
        ]
        captured_payloads: list[dict[str, Any]] = []

        def fake_urlopen(request):
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [
                            {"segment_number": 1, "text": "안녕"},
                            {"segment_number": 2, "text": "세상"},
                        ],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                translated_segments, info = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                )

        self.assertEqual(
            [
                {"start": 0.0, "end": 1.5, "text": "안녕"},
                {"start": 1.5, "end": 3.0, "text": "세상"},
            ],
            translated_segments,
        )
        self.assertEqual("gpt-4.1-mini", info["model"])
        self.assertEqual(100, info["chunk_size"])
        self.assertIn("segment_number", captured_payloads[0]["messages"][0]["content"])

    def test_translate_accepts_structured_output_object_response(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "World"},
        ]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        {
                            "items": [
                                {"segment_number": 1, "text": "안녕"},
                                {"segment_number": 2, "text": "세상"},
                            ]
                        },
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                )

        self.assertEqual(
            ["안녕", "세상"], [segment["text"] for segment in translated_segments]
        )

    def test_translation_falls_back_after_structured_output_capability_error(
        self,
    ) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
        ]
        captured_payloads: list[dict[str, Any]] = []
        stderr_buffer = io.StringIO()

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            captured_payloads.append(payload)
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_number = user_payload["segment_numbers"][0]
            if len(captured_payloads) == 1:
                raise urllib.error.HTTPError(
                    url=request.full_url,
                    code=400,
                    msg="bad request",
                    hdrs=Message(),
                    fp=io.BytesIO(
                        json.dumps(
                            {
                                "error": {
                                    "type": "invalid_request_error",
                                    "code": "unsupported_parameter",
                                    "message": (
                                        "response_format json_schema is not supported "
                                        "for this model"
                                    ),
                                }
                            }
                        ).encode("utf-8")
                    ),
                )
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [
                            {
                                "segment_number": segment_number,
                                "text": f"ok-{segment_number}",
                            }
                        ],
                        ensure_ascii=False,
                    )
                )
            )

        with (
            patch(
                "vid_to_sub_app.cli.translation.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr_buffer),
        ):
            translated_segments, _ = translate_segments_openai_compatible(
                segments=segments,
                target_language="ko",
                translation_model="gpt-4.1-mini",
                translation_base_url="https://translation.example/v1",
                translation_api_key="secret",
                source_language="en",
                chunk_size=1,
            )

        self.assertEqual(
            ["ok-1", "ok-2"], [segment["text"] for segment in translated_segments]
        )
        self.assertEqual(
            [True, False, False],
            ["response_format" in payload for payload in captured_payloads],
        )
        fallback_event = next(
            event
            for event in _event_payloads(stderr_buffer)
            if event["event"] == "translation_structured_output_fallback"
        )
        self.assertEqual(400, fallback_event["http_status"])
        self.assertEqual([1], fallback_event["segment_numbers"])
        self.assertEqual("translate", fallback_event["stage"])

    def test_translation_does_not_fallback_for_malformed_structured_output_schema(
        self,
    ) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "one"}]
        captured_payloads: list[dict[str, Any]] = []
        stderr_buffer = io.StringIO()

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            captured_payloads.append(payload)
            raise urllib.error.HTTPError(
                url=request.full_url,
                code=400,
                msg="bad request",
                hdrs=Message(),
                fp=io.BytesIO(
                    json.dumps(
                        {
                            "error": {
                                "type": "invalid_request_error",
                                "code": "invalid_schema",
                                "message": (
                                    "response_format json_schema failed to validate schema"
                                ),
                            }
                        }
                    ).encode("utf-8")
                ),
            )

        with (
            patch(
                "vid_to_sub_app.cli.translation.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ),
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr_buffer),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=1,
                )

        self.assertIn("invalid_schema", str(ctx.exception))
        self.assertEqual(1, len(captured_payloads))
        self.assertIn("response_format", captured_payloads[0])
        self.assertFalse(
            any(
                event["event"] == "translation_structured_output_fallback"
                for event in _event_payloads(stderr_buffer)
            )
        )

    def test_translate_accepts_legacy_string_array_only_when_counts_match(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "World"},
        ]

        def fake_urlopen(request):
            return _FakeHTTPResponse(_chat_payload('["안녕", "세상"]'))

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                )

        self.assertEqual(
            ["안녕", "세상"], [segment["text"] for segment in translated_segments]
        )

    def test_translation_retries_after_retryable_http_error(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        calls = 0
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        def fake_urlopen(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise urllib.error.HTTPError(
                    url="https://translation.example/v1/chat/completions",
                    code=429,
                    msg="rate limited",
                    hdrs=Message(),
                    fp=io.BytesIO(b'{"error":"rate limited"}'),
                )
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "안녕"}], ensure_ascii=False
                    )
                )
            )

        with (
            patch(
                "vid_to_sub_app.cli.translation.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ),
            patch("vid_to_sub_app.cli.translation.time.sleep") as sleep,
            redirect_stdout(stdout_buffer),
            redirect_stderr(stderr_buffer),
        ):
            translated_segments, _ = translate_segments_openai_compatible(
                segments=segments,
                target_language="ko",
                translation_model="gpt-4.1-mini",
                translation_base_url="https://translation.example/v1",
                translation_api_key="secret",
                source_language="en",
                chunk_size=1,
            )

        self.assertEqual(["안녕"], [segment["text"] for segment in translated_segments])
        self.assertEqual(2, calls)
        sleep.assert_called_once_with(1.0)
        retry_event = next(
            event
            for event in _event_payloads(stderr_buffer)
            if event["event"] == "translation_http_retry"
        )
        self.assertEqual(429, retry_event["http_status"])
        self.assertEqual("http_429", retry_event["reason"])
        self.assertEqual("translate", retry_event["stage"])
        self.assertGreater(retry_event["payload_chars"], 0)

    def test_translation_retries_after_transient_url_error(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        calls = 0
        stderr_buffer = io.StringIO()

        def fake_urlopen(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise urllib.error.URLError("temporary dns failure")
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "안녕"}], ensure_ascii=False
                    )
                )
            )

        with (
            patch(
                "vid_to_sub_app.cli.translation.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ),
            patch("vid_to_sub_app.cli.translation.time.sleep") as sleep,
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr_buffer),
        ):
            translated_segments, _ = translate_segments_openai_compatible(
                segments=segments,
                target_language="ko",
                translation_model="gpt-4.1-mini",
                translation_base_url="https://translation.example/v1",
                translation_api_key="secret",
                source_language="en",
                chunk_size=1,
            )

        self.assertEqual(["안녕"], [segment["text"] for segment in translated_segments])
        self.assertEqual(2, calls)
        sleep.assert_called_once_with(1.0)
        retry_event = next(
            event
            for event in _event_payloads(stderr_buffer)
            if event["event"] == "translation_http_retry"
        )
        self.assertIsNone(retry_event["http_status"])
        self.assertIn("temporary dns failure", retry_event["reason"])
        self.assertEqual("translate", retry_event["stage"])
        self.assertGreater(retry_event["payload_chars"], 0)

    def test_translation_retries_after_legacy_string_array_count_mismatch(
        self,
    ) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
        ]
        request_segment_numbers: list[list[int]] = []
        stderr_buffer = io.StringIO()

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_segment_numbers.append(segment_numbers)
            if segment_numbers == [1, 2]:
                return _FakeHTTPResponse(_chat_payload('["only-one"]'))
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps([f"ok-{segment_numbers[0]}"], ensure_ascii=False)
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr_buffer):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=2,
                )

        self.assertEqual([[1, 2], [1], [2]], request_segment_numbers)
        self.assertEqual(
            ["ok-1", "ok-2"], [segment["text"] for segment in translated_segments]
        )
        contract_event = next(
            event
            for event in _event_payloads(stderr_buffer)
            if event["event"] == "translation_batch_contract_error"
        )
        self.assertEqual("legacy_string_array_count_mismatch", contract_event["reason"])
        self.assertEqual(1, contract_event["parsed_count"])
        self.assertEqual(["only-one"], contract_event["parsed_sample"])
        self.assertEqual(1, contract_event["next_retry_size"])

    def test_translate_reorders_object_array_by_segment_number(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
            {"start": 2.0, "end": 3.0, "text": "three"},
        ]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [
                            {"segment_number": 3, "text": "셋"},
                            {"segment_number": 1, "text": "하나"},
                            {"segment_number": 2, "text": "둘"},
                        ],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                )

        self.assertEqual(
            ["하나", "둘", "셋"], [segment["text"] for segment in translated_segments]
        )

    def test_translate_raises_on_missing_segment_number(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
        ]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "하나"}], ensure_ascii=False
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(TranslationContractError) as ctx:
                    translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=2,
                    )

        message = str(ctx.exception)
        self.assertIn("missing_segment_numbers=[2]", message)
        self.assertIn("requested_count=2", message)
        self.assertIn("configured_chunk_size=2", message)

    def test_translate_raises_on_duplicate_segment_number(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "one"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [
                            {"segment_number": 1, "text": "하나"},
                            {"segment_number": 1, "text": "dup"},
                        ],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(TranslationContractError) as ctx:
                    translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=1,
                    )

        self.assertIn("duplicate_segment_numbers=[1]", str(ctx.exception))

    def test_translate_raises_on_unexpected_segment_number(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "one"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 99, "text": "unexpected"}],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(TranslationContractError) as ctx:
                    translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=1,
                    )

        self.assertIn("unexpected_segment_numbers=[99]", str(ctx.exception))

    def test_translate_raises_on_non_object_array_item(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
        ]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [
                            {"segment_number": 1, "text": "ok"},
                            ["not-an-object"],
                        ],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(TranslationContractError) as ctx:
                    translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=2,
                    )

        self.assertIn(
            "model_response_items_must_be_objects_or_strings", str(ctx.exception)
        )

    def test_translate_raises_on_non_int_segment_number(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "one"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": "1", "text": "bad-id"}],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(TranslationContractError) as ctx:
                    translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=1,
                    )

        self.assertIn("segment_number_must_be_int", str(ctx.exception))

    def test_translate_raises_on_non_string_text(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "one"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": 42}],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(TranslationContractError) as ctx:
                    translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=1,
                    )

        self.assertIn("text_must_be_string", str(ctx.exception))

    def test_blank_segments_are_skipped_and_reinserted(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "   "},
            {"start": 2.0, "end": 3.0, "text": "World"},
        ]
        request_payloads: list[dict[str, Any]] = []
        stderr_buffer = io.StringIO()

        def fake_urlopen(request):
            request_payloads.append(json.loads(request.data.decode("utf-8")))
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [
                            {"segment_number": 1, "text": "안녕"},
                            {"segment_number": 3, "text": "세상"},
                        ],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr_buffer):
                translated_segments, info = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=10,
                )

        user_payload = json.loads(request_payloads[0]["messages"][1]["content"])
        self.assertEqual([1, 3], user_payload["segment_numbers"])
        self.assertEqual(["Hello", "World"], user_payload["subtitles"])
        batch_result = next(
            event
            for event in _event_payloads(stderr_buffer)
            if event["event"] == "translation_batch_result"
        )
        self.assertEqual(2, batch_result["requested_count"])
        self.assertEqual(0, batch_result["blank_or_whitespace_count"])
        self.assertEqual(2, batch_result["effective_sent_count"])
        self.assertEqual(
            ["안녕", "   ", "세상"],
            [segment["text"] for segment in translated_segments],
        )
        self.assertEqual(1, info["blank_segment_count"])

    def test_translation_retries_with_smaller_batches(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
            {"start": 2.0, "end": 3.0, "text": "three"},
        ]
        request_segment_numbers: list[list[int]] = []

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_segment_numbers.append(segment_numbers)
            if segment_numbers == [1, 2, 3]:
                return _FakeHTTPResponse(
                    _chat_payload(
                        json.dumps(
                            [
                                {"segment_number": 1, "text": "하나"},
                                {"segment_number": 2, "text": "둘"},
                            ],
                            ensure_ascii=False,
                        )
                    )
                )
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [
                            {
                                "segment_number": segment_numbers[0],
                                "text": f"번역-{segment_numbers[0]}",
                            }
                        ],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=3,
                )

        self.assertEqual([[1, 2, 3], [1], [2], [3]], request_segment_numbers)
        self.assertEqual(
            ["번역-1", "번역-2", "번역-3"],
            [segment["text"] for segment in translated_segments],
        )

    def test_translation_transport_errors_shrink_batches(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
            {"start": 2.0, "end": 3.0, "text": "three"},
        ]
        request_segment_numbers: list[list[int]] = []
        client_request_ids: list[str | None] = []
        stderr_buffer = io.StringIO()

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_segment_numbers.append(segment_numbers)
            client_request_ids.append(request.headers.get("X-client-request-id"))
            if segment_numbers == [1, 2, 3]:
                raise urllib.error.HTTPError(
                    url=request.full_url,
                    code=429,
                    msg="rate limited",
                    hdrs=Message(),
                    fp=io.BytesIO(b'{"error":"rate limited"}'),
                )
            items = [
                {"segment_number": number, "text": f"ok-{number}"}
                for number in segment_numbers
            ]
            return _FakeHTTPResponse(
                _chat_payload(json.dumps(items, ensure_ascii=False))
            )

        with (
            patch(
                "vid_to_sub_app.cli.translation.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ),
            patch("vid_to_sub_app.cli.translation.time.sleep") as sleep,
            redirect_stdout(io.StringIO()),
            redirect_stderr(stderr_buffer),
        ):
            translated_segments, _ = translate_segments_openai_compatible(
                segments=segments,
                target_language="ko",
                translation_model="gpt-4.1-mini",
                translation_base_url="https://translation.example/v1",
                translation_api_key="secret",
                source_language="en",
                chunk_size=3,
            )

        self.assertEqual([[1, 2, 3], [1], [2], [3]], request_segment_numbers)
        self.assertEqual(
            ["ok-1", "ok-2", "ok-3"], [s["text"] for s in translated_segments]
        )
        self.assertEqual(0, sleep.call_count)
        self.assertTrue(all(client_request_ids))
        self.assertEqual(4, len(set(client_request_ids)))
        transport_event = next(
            event
            for event in _event_payloads(stderr_buffer)
            if event["event"] == "translation_batch_transport_error"
        )
        self.assertEqual("translate", transport_event["stage"])
        self.assertEqual(429, transport_event["http_status"])
        self.assertEqual(1, transport_event["next_retry_size"])
        self.assertGreater(transport_event["payload_chars"], 0)

    def test_translation_single_item_transport_errors_keep_http_retries(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "one"}]
        request_attempts = 0

        def fake_urlopen(request):
            nonlocal request_attempts
            request_attempts += 1
            if request_attempts < 4:
                raise urllib.error.HTTPError(
                    url=request.full_url,
                    code=429,
                    msg="rate limited",
                    hdrs=Message(),
                    fp=io.BytesIO(b'{"error":"rate limited"}'),
                )
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "ok-1"}], ensure_ascii=False
                    )
                )
            )

        with (
            patch(
                "vid_to_sub_app.cli.translation.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ),
            patch("vid_to_sub_app.cli.translation.time.sleep") as sleep,
            redirect_stdout(io.StringIO()),
        ):
            translated_segments, _ = translate_segments_openai_compatible(
                segments=segments,
                target_language="ko",
                translation_model="gpt-4.1-mini",
                translation_base_url="https://translation.example/v1",
                translation_api_key="secret",
                source_language="en",
                chunk_size=3,
            )

        self.assertEqual(4, request_attempts)
        self.assertEqual(3, sleep.call_count)
        self.assertEqual(["ok-1"], [segment["text"] for segment in translated_segments])

    def test_translation_retries_after_json_parse_failure(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
        ]
        request_segment_numbers: list[list[int]] = []

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_segment_numbers.append(segment_numbers)
            if segment_numbers == [1, 2]:
                return _FakeHTTPResponse(_chat_payload("not json at all"))
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [
                            {
                                "segment_number": segment_numbers[0],
                                "text": f"ok-{segment_numbers[0]}",
                            }
                        ],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=2,
                )

        self.assertEqual([[1, 2], [1], [2]], request_segment_numbers)
        self.assertEqual(
            ["ok-1", "ok-2"],
            [segment["text"] for segment in translated_segments],
        )

    def test_translation_retries_after_finish_reason_length_on_multi_item_batch(
        self,
    ) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
        ]
        request_segment_numbers: list[list[int]] = []
        stderr_buffer = io.StringIO()

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_segment_numbers.append(segment_numbers)
            if segment_numbers == [1, 2]:
                return _FakeHTTPResponse(
                    _chat_payload(
                        json.dumps(
                            [
                                {"segment_number": 1, "text": "one-ko"},
                                {"segment_number": 2, "text": "two-ko"},
                            ],
                            ensure_ascii=False,
                        ),
                        finish_reason="length",
                    )
                )
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [
                            {
                                "segment_number": segment_numbers[0],
                                "text": f"ok-{segment_numbers[0]}",
                            }
                        ],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr_buffer):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=2,
                )

        self.assertEqual([[1, 2], [1], [2]], request_segment_numbers)
        self.assertEqual(
            ["ok-1", "ok-2"],
            [segment["text"] for segment in translated_segments],
        )
        contract_event = next(
            event
            for event in _event_payloads(stderr_buffer)
            if event["event"] == "translation_batch_contract_error"
        )
        self.assertEqual("length", contract_event["finish_reason"])
        self.assertEqual(1, contract_event["next_retry_size"])

    def test_best_effort_mode_falls_back_to_source_text_at_size_one(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "keep me"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(_chat_payload("[]", finish_reason="length"))

        stderr_buffer = io.StringIO()
        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr_buffer):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=1,
                    translation_mode="best-effort",
                )

        self.assertEqual("keep me", translated_segments[0]["text"])
        self.assertIn("fell back to source text", stderr_buffer.getvalue())

    def test_strict_mode_raises_at_size_one(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "fail me"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(_chat_payload("[]", finish_reason="length"))

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                with self.assertRaises(TranslationContractError) as ctx:
                    translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=1,
                        translation_mode="strict",
                    )

        self.assertIn("finish_reason=length", str(ctx.exception))

    def test_batch_logs_emit_without_debug_and_raw_response_requires_exact_one(
        self,
    ) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "안녕"}], ensure_ascii=False
                    )
                )
            )

        no_debug_stderr = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "vid_to_sub_app.cli.translation.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ),
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(no_debug_stderr):
                translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=1,
                )

        no_debug_events = _event_payloads(no_debug_stderr)
        no_debug_result = next(
            event
            for event in no_debug_events
            if event["event"] == "translation_batch_result"
        )
        self.assertIsNone(no_debug_result["raw_response_file"])
        self.assertEqual(1, no_debug_result["chunk_size"])
        self.assertEqual(
            "https://translation.example/v1/chat/completions",
            no_debug_result["endpoint"],
        )

        truthy_text_stderr = io.StringIO()
        with (
            patch.dict(
                os.environ, {"VID_TO_SUB_TRANSLATION_DEBUG": "true"}, clear=False
            ),
            patch(
                "vid_to_sub_app.cli.translation.urllib.request.urlopen",
                side_effect=fake_urlopen,
            ),
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(truthy_text_stderr):
                translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=1,
                )

        truthy_text_events = _event_payloads(truthy_text_stderr)
        truthy_text_result = next(
            event
            for event in truthy_text_events
            if event["event"] == "translation_batch_result"
        )
        self.assertIsNone(truthy_text_result["raw_response_file"])

        with tempfile.TemporaryDirectory() as tmpdir:
            exact_debug_stderr = io.StringIO()
            with (
                patch.dict(
                    os.environ, {"VID_TO_SUB_TRANSLATION_DEBUG": "1"}, clear=False
                ),
                patch(
                    "vid_to_sub_app.cli.translation.tempfile.gettempdir",
                    return_value=tmpdir,
                ),
                patch(
                    "vid_to_sub_app.cli.translation.urllib.request.urlopen",
                    side_effect=fake_urlopen,
                ),
            ):
                with (
                    redirect_stdout(io.StringIO()),
                    redirect_stderr(exact_debug_stderr),
                ):
                    translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=1,
                    )

            exact_debug_events = _event_payloads(exact_debug_stderr)
            exact_debug_result = next(
                event
                for event in exact_debug_events
                if event["event"] == "translation_batch_result"
            )
            raw_path = Path(exact_debug_result["raw_response_file"])
            self.assertTrue(raw_path.exists())

    def test_batch_logs_include_response_metadata_fields(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        stderr_buffer = io.StringIO()

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "안녕"}], ensure_ascii=False
                    )
                ),
                status=206,
                headers={"x-request-id": "req-456"},
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr_buffer):
                translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=1,
                )

        batch_result = next(
            event
            for event in _event_payloads(stderr_buffer)
            if event["event"] == "translation_batch_result"
        )
        self.assertEqual(206, batch_result["http_status"])
        self.assertEqual("req-456", batch_result["request_id"])
        self.assertEqual("resp-123", batch_result["response_id"])
        self.assertEqual(
            {"completion_tokens": 10, "prompt_tokens": 20, "total_tokens": 30},
            batch_result["usage"],
        )
        self.assertEqual(
            "https://translation.example/v1/chat/completions",
            batch_result["endpoint"],
        )

    def test_max_payload_chars_splits_translation_batches(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "A" * 40},
            {"start": 1.0, "end": 2.0, "text": "B" * 40},
            {"start": 2.0, "end": 3.0, "text": "C" * 40},
        ]
        request_segment_numbers: list[list[int]] = []

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_segment_numbers.append(segment_numbers)
            items = [
                {"segment_number": number, "text": f"ok-{number}"}
                for number in segment_numbers
            ]
            return _FakeHTTPResponse(
                _chat_payload(json.dumps(items, ensure_ascii=False))
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                translated_segments, info = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=3,
                    max_payload_chars=100,
                )

        self.assertEqual([[1], [2], [3]], request_segment_numbers)
        self.assertEqual(
            ["ok-1", "ok-2", "ok-3"], [s["text"] for s in translated_segments]
        )
        self.assertEqual(100, info["max_payload_chars"])

    def test_translate_resume_skips_completed_segments_and_tracks_resumed_count(
        self,
    ) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
            {"start": 2.0, "end": 3.0, "text": "three"},
        ]
        request_segment_numbers: list[list[int]] = []
        batch_successes: list[tuple[list[int], list[str]]] = []

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_segment_numbers.append(segment_numbers)
            items = [
                {"segment_number": number, "text": f"ok-{number}"}
                for number in segment_numbers
            ]
            return _FakeHTTPResponse(
                _chat_payload(json.dumps(items, ensure_ascii=False))
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                translated_segments, info = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=3,
                    resume_text_by_number={2: "cached-2"},
                    on_batch_success=lambda numbers, texts: batch_successes.append(
                        (list(numbers), list(texts))
                    ),
                )

        self.assertEqual([[1, 3]], request_segment_numbers)
        self.assertEqual(
            ["ok-1", "cached-2", "ok-3"],
            [segment["text"] for segment in translated_segments],
        )
        self.assertEqual(1, info["resumed_segment_count"])
        self.assertEqual([([1, 3], ["ok-1", "ok-3"])], batch_successes)

    def test_retry_logs_use_actual_chunk_size_and_unique_batch_ids(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "   "},
            {"start": 2.0, "end": 3.0, "text": "three"},
        ]

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            if segment_numbers == [1, 3]:
                return _FakeHTTPResponse(
                    _chat_payload(
                        json.dumps(
                            [
                                {"segment_number": 1, "text": "하나"},
                            ],
                            ensure_ascii=False,
                        )
                    )
                )
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [
                            {
                                "segment_number": segment_numbers[0],
                                "text": f"ok-{segment_numbers[0]}",
                            }
                        ],
                        ensure_ascii=False,
                    )
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            stderr_buffer = io.StringIO()
            with (
                patch.dict(
                    os.environ, {"VID_TO_SUB_TRANSLATION_DEBUG": "1"}, clear=False
                ),
                patch(
                    "vid_to_sub_app.cli.translation.tempfile.gettempdir",
                    return_value=tmpdir,
                ),
                patch(
                    "vid_to_sub_app.cli.translation.urllib.request.urlopen",
                    side_effect=fake_urlopen,
                ),
            ):
                with redirect_stdout(io.StringIO()), redirect_stderr(stderr_buffer):
                    translated_segments, _ = translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=3,
                    )

            self.assertEqual(
                ["ok-1", "   ", "ok-3"],
                [segment["text"] for segment in translated_segments],
            )
            events = _event_payloads(stderr_buffer)
            contract_event = next(
                event
                for event in events
                if event["event"] == "translation_batch_contract_error"
            )
            result_events = [
                event
                for event in events
                if event["event"] == "translation_batch_result"
            ]
            self.assertEqual(2, contract_event["chunk_size"])
            self.assertEqual(2, contract_event["requested_count"])
            self.assertEqual(0, contract_event["blank_or_whitespace_count"])
            self.assertEqual(1, contract_event["next_retry_size"])
            self.assertEqual([1, 1], [event["chunk_size"] for event in result_events])
            self.assertEqual(
                [2, 2], [event["requested_count"] for event in result_events]
            )
            self.assertEqual(
                [0, 0],
                [event["blank_or_whitespace_count"] for event in result_events],
            )
            self.assertEqual(
                len(result_events),
                len({event["batch_id"] for event in result_events}),
            )
            raw_paths = [
                Path(str(event["raw_response_file"])) for event in result_events
            ]
            self.assertEqual(len(raw_paths), len({str(path) for path in raw_paths}))
            self.assertTrue(all(path.exists() for path in raw_paths))

    def test_postprocess_uses_translation_config_as_fallback(self) -> None:
        captured_requests = []

        def fake_urlopen(request):
            captured_requests.append(request)
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "정제된 가사"}],
                        ensure_ascii=False,
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                final_segments, info = (
                    postprocess_translated_segments_openai_compatible(
                        source_segments=[{"start": 0.0, "end": 2.0, "text": "青い風"}],
                        translated_segments=[
                            {"start": 0.0, "end": 2.0, "text": "푸른 바람"}
                        ],
                        target_language="ko",
                        postprocess_mode="auto",
                        postprocess_model=None,
                        postprocess_base_url=None,
                        postprocess_api_key=None,
                        source_language="ja",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="translation-secret",
                        chunk_size=7,
                        max_payload_chars=4321,
                        translation_mode="best-effort",
                    )
                )

        self.assertEqual(
            [{"start": 0.0, "end": 2.0, "text": "정제된 가사"}],
            final_segments,
        )
        self.assertEqual("auto", info["mode"])
        self.assertEqual(7, info["chunk_size"])
        self.assertEqual(4321, info["max_payload_chars"])
        self.assertEqual("best-effort", info["translation_mode"])
        self.assertEqual(
            "https://translation.example/v1/chat/completions",
            captured_requests[0].full_url,
        )
        payload = json.loads(captured_requests[0].data.decode("utf-8"))
        self.assertIn(
            "silently fall back to contextual correction",
            payload["messages"][0]["content"],
        )
        self.assertEqual(
            "application/json", captured_requests[0].headers["Content-type"]
        )
        self.assertTrue(
            captured_requests[0].headers["X-client-request-id"].startswith("pp-")
        )

    def test_postprocess_resume_skips_completed_segments_and_tracks_resumed_count(
        self,
    ) -> None:
        request_segment_numbers: list[list[int]] = []
        batch_successes: list[tuple[list[int], list[str]]] = []

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_segment_numbers.append(segment_numbers)
            items = [
                {"segment_number": number, "text": f"final-{number}"}
                for number in segment_numbers
            ]
            return _FakeHTTPResponse(
                _chat_payload(json.dumps(items, ensure_ascii=False))
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()):
                final_segments, info = (
                    postprocess_translated_segments_openai_compatible(
                        source_segments=[
                            {"start": 0.0, "end": 1.0, "text": "one"},
                            {"start": 1.0, "end": 2.0, "text": "two"},
                            {"start": 2.0, "end": 3.0, "text": "three"},
                        ],
                        translated_segments=[
                            {"start": 0.0, "end": 1.0, "text": "draft-1"},
                            {"start": 1.0, "end": 2.0, "text": "draft-2"},
                            {"start": 2.0, "end": 3.0, "text": "draft-3"},
                        ],
                        target_language="ko",
                        postprocess_mode="auto",
                        postprocess_model=None,
                        postprocess_base_url=None,
                        postprocess_api_key=None,
                        source_language="en",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        chunk_size=3,
                        resume_text_by_number={2: "cached-final-2"},
                        on_batch_success=lambda numbers, texts: batch_successes.append(
                            (list(numbers), list(texts))
                        ),
                    )
                )

        self.assertEqual([[1, 3]], request_segment_numbers)
        self.assertEqual(
            ["final-1", "cached-final-2", "final-3"],
            [segment["text"] for segment in final_segments],
        )
        self.assertEqual(1, info["resumed_segment_count"])
        self.assertEqual([([1, 3], ["final-1", "final-3"])], batch_successes)

    def test_translation_fingerprint_distinguishes_worker_identity(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "안녕"}], ensure_ascii=False
                    )
                )
            )

        first_stdout = io.StringIO()
        second_stdout = io.StringIO()

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with patch(
                "vid_to_sub_app.cli.translation._git_output",
                side_effect=["abc1234", "M x", "def5678", None],
            ):
                with patch(
                    "vid_to_sub_app.cli.translation.socket.gethostname",
                    side_effect=["worker-03", "worker-07"],
                ):
                    with patch(
                        "vid_to_sub_app.cli.translation.sys.argv",
                        ["vid_to_sub.py", "--manifest-stdin", "--translate-to", "ko"],
                    ):
                        with patch(
                            "vid_to_sub_app.cli.translation.vid_to_sub_app.__file__",
                            "/srv/app-a/vid_to_sub_app/__init__.py",
                        ):
                            with (
                                redirect_stdout(first_stdout),
                                redirect_stderr(io.StringIO()),
                            ):
                                translate_segments_openai_compatible(
                                    segments=segments,
                                    target_language="ko",
                                    translation_model="gpt-4.1-mini",
                                    translation_base_url="https://translation.example/v1",
                                    translation_api_key="secret",
                                    source_language="en",
                                    chunk_size=20,
                                )
                        with patch(
                            "vid_to_sub_app.cli.translation.vid_to_sub_app.__file__",
                            "/srv/app-b/vid_to_sub_app/__init__.py",
                        ):
                            with (
                                redirect_stdout(second_stdout),
                                redirect_stderr(io.StringIO()),
                            ):
                                translate_segments_openai_compatible(
                                    segments=segments,
                                    target_language="ko",
                                    translation_model="gpt-4.1-mini",
                                    translation_base_url="https://translation.example/v1",
                                    translation_api_key="secret",
                                    source_language="en",
                                    chunk_size=20,
                                )

        first_line = first_stdout.getvalue().splitlines()[0]
        second_line = second_stdout.getvalue().splitlines()[0]
        self.assertIn('"event": "translation_execution_fingerprint"', first_line)
        self.assertIn('"git_sha": "abc1234"', first_line)
        self.assertIn(
            '"package_path": "/srv/app-a/vid_to_sub_app/__init__.py"', first_line
        )
        self.assertIn('"hostname": "worker-03"', first_line)
        self.assertIn('"distributed": true', first_line)
        self.assertIn('"translation_chunk_size": 20', first_line)
        self.assertIn(
            '"endpoint": "https://translation.example/v1/chat/completions"',
            first_line,
        )
        self.assertIn('"git_sha": "def5678"', second_line)
        self.assertIn(
            '"package_path": "/srv/app-b/vid_to_sub_app/__init__.py"', second_line
        )
        self.assertIn('"hostname": "worker-07"', second_line)
        self.assertNotEqual(first_line, second_line)

    def test_main_rejects_postprocess_without_translation_target(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["--manifest-stdin", "--postprocess-translation"])

        self.assertEqual(2, ctx.exception.code)

    def test_stage1_only_flag_skips_translation(self) -> None:
        video_path = Path("/tmp/movie.mp4")
        manifest = {
            "found_total": 1,
            "skipped": 0,
            "folders": [
                {
                    "folder_hash": "folder-hash",
                    "folder_path": str(video_path.parent),
                    "total_files": 1,
                    "completed_files": 0,
                    "status": "queued",
                    "is_completed": False,
                }
            ],
            "entries": [
                {
                    "video_path": str(video_path),
                    "folder_hash": "folder-hash",
                    "folder_path": str(video_path.parent),
                }
            ],
        }
        stage1_result = ProcessResult(
            success=True,
            video_path=str(video_path),
            folder_hash="folder-hash",
            folder_path=str(video_path.parent),
            worker_id=0,
            elapsed_sec=0.1,
            artifact_path="/tmp/movie.stage1.json",
        )

        with (
            patch("vid_to_sub_app.cli.main.discover_videos", return_value=[video_path]),
            patch(
                "vid_to_sub_app.cli.main.build_run_manifest",
                return_value=manifest,
            ),
            patch("vid_to_sub_app.cli.main.persist_folder_manifest_state"),
            patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            patch(
                "vid_to_sub_app.cli.main.run_stage1",
                return_value=stage1_result,
            ) as run_stage1_mock,
            patch("vid_to_sub_app.cli.main.run_stage2") as run_stage2_mock,
        ):
            exit_code = main([str(video_path), "--stage1-only"])

        self.assertEqual(0, exit_code)
        run_stage1_mock.assert_called_once()
        run_stage2_mock.assert_not_called()

    def test_stage1_only_quality_hold_returns_success(self) -> None:
        video_path = Path("/tmp/movie.mp4")
        manifest = {
            "found_total": 1,
            "skipped": 0,
            "folders": [
                {
                    "folder_hash": "folder-hash",
                    "folder_path": str(video_path.parent),
                    "total_files": 1,
                    "completed_files": 0,
                    "status": "queued",
                    "is_completed": False,
                }
            ],
            "entries": [
                {
                    "video_path": str(video_path),
                    "folder_hash": "folder-hash",
                    "folder_path": str(video_path.parent),
                }
            ],
        }
        stage1_result = ProcessResult(
            success=False,
            video_path=str(video_path),
            folder_hash="folder-hash",
            folder_path=str(video_path.parent),
            worker_id=0,
            stage="quality_hold",
            elapsed_sec=0.1,
            error="empty_transcript",
            artifact_path="/tmp/movie.stage1.json",
        )

        with (
            patch("vid_to_sub_app.cli.main.discover_videos", return_value=[video_path]),
            patch(
                "vid_to_sub_app.cli.main.build_run_manifest",
                return_value=manifest,
            ),
            patch("vid_to_sub_app.cli.main.persist_folder_manifest_state"),
            patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            patch(
                "vid_to_sub_app.cli.main.run_stage1",
                return_value=stage1_result,
            ) as run_stage1_mock,
            patch("vid_to_sub_app.cli.main.run_stage2") as run_stage2_mock,
        ):
            exit_code = main([str(video_path), "--stage1-only"])

        self.assertEqual(0, exit_code)
        run_stage1_mock.assert_called_once()
        run_stage2_mock.assert_not_called()

    def test_translate_from_artifact_flag_skips_transcription(self) -> None:
        artifact_path = Path("/tmp/movie.stage1.json")
        stage2_result = ProcessResult(
            success=True,
            video_path="/tmp/movie.mp4",
            folder_hash="folder-hash",
            folder_path="/tmp",
            worker_id=0,
            elapsed_sec=0.1,
            output_paths=["/tmp/movie.ko.srt"],
            artifact_path=str(artifact_path),
        )

        with (
            patch(
                "vid_to_sub_app.cli.main.run_stage2",
                return_value=stage2_result,
            ) as run_stage2_mock,
            patch("vid_to_sub_app.cli.main.run_stage1") as run_stage1_mock,
        ):
            exit_code = main(
                [
                    "--translate-from-artifact",
                    str(artifact_path),
                    "--translate-to",
                    "ko",
                ]
            )

        self.assertEqual(0, exit_code)
        run_stage2_mock.assert_called_once_with(artifact_path.resolve(), ANY)
        run_stage1_mock.assert_not_called()

    def test_translate_from_artifact_emits_stage2_job_events(self) -> None:
        artifact_path = Path("/tmp/movie.stage1.json")
        artifact: StageArtifact = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_path": "/tmp/movie.mp4",
            "output_base": "/tmp",
            "source_fingerprint": "10:1700000000",
            "backend": "faster-whisper",
            "device": "cpu",
            "model": "large-v3",
            "language": "en",
            "language_probability": 0.99,
            "duration": 1.0,
            "quality": {"suspicious": False, "reasons": []},
            "target_lang": "ko",
            "formats": ["srt"],
            "primary_outputs": ["/tmp/movie.srt"],
            "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
            "stage_status": {
                "transcription_complete": True,
                "translation_pending": True,
                "translation_complete": False,
                "translation_failed": False,
                "translation_error": None,
            },
        }
        stage2_result = ProcessResult(
            success=True,
            video_path="/tmp/movie.mp4",
            folder_hash="folder-hash",
            folder_path="/tmp",
            worker_id=0,
            elapsed_sec=0.1,
            language="en",
            output_paths=["/tmp/movie.ko.srt"],
            segments=1,
            artifact_path=str(artifact_path),
            artifact_metadata=build_stage_artifact_metadata(artifact_path, artifact),
        )

        stdout_buf = io.StringIO()
        with (
            patch(
                "vid_to_sub_app.cli.main.load_stage_artifact",
                return_value=artifact,
            ),
            patch(
                "vid_to_sub_app.cli.main.run_stage2",
                return_value=stage2_result,
            ),
            redirect_stdout(stdout_buf),
        ):
            exit_code = main(
                [
                    "--translate-from-artifact",
                    str(artifact_path),
                    "--translate-to",
                    "ko",
                ]
            )

        payloads = _event_payloads(stdout_buf)
        started = next(
            payload for payload in payloads if payload.get("event") == "job_started"
        )
        finished = next(
            payload for payload in payloads if payload.get("event") == "job_finished"
        )

        self.assertEqual(0, exit_code)
        self.assertEqual("/tmp/movie.mp4", started.get("video_path"))
        self.assertEqual("/tmp/movie.mp4", finished.get("video_path"))
        self.assertEqual("done", finished.get("status"))
        self.assertEqual(str(artifact_path), finished.get("artifact_path"))

    def test_mutually_exclusive_stage_flags(self) -> None:
        with patch.object(
            argparse.ArgumentParser,
            "error",
            side_effect=SystemExit(2),
        ) as parser_error:
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        "--stage1-only",
                        "--translate-from-artifact",
                        "/tmp/movie.stage1.json",
                    ]
                )

        self.assertEqual(2, ctx.exception.code)
        parser_error.assert_called_once_with(
            "--stage1-only and --translate-from-artifact are mutually exclusive."
        )


class TestStageArtifact(unittest.TestCase):
    def test_write_and_load_roundtrip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = (temp_path / "movie.mp4").resolve()
            output_dir = (temp_path / "out").resolve()
            output_dir.mkdir()
            source_path.write_bytes(b"")
            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source_path),
                "output_base": str(output_dir),
                "source_fingerprint": "0:123",
                "backend": "faster-whisper",
                "device": "cuda",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.0,
                "quality": {"suspicious": False, "reasons": []},
                "target_lang": "ko",
                "formats": ["srt", "json"],
                "primary_outputs": [
                    str((output_dir / "movie.srt").resolve()),
                    str((output_dir / "movie.json").resolve()),
                ],
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Hello"},
                ],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": False,
                    "translation_complete": True,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }

            artifact_path = write_stage_artifact(artifact, output_dir, source_path)
            loaded_artifact = load_stage_artifact(artifact_path)

        self.assertEqual(output_dir / f"movie{ARTIFACT_FILENAME_SUFFIX}", artifact_path)
        self.assertEqual(artifact, loaded_artifact)

    def test_load_unsupported_version_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            artifact_path = temp_path / f"movie{ARTIFACT_FILENAME_SUFFIX}"
            artifact_path.write_text(
                json.dumps(
                    {
                        "schema_version": "99",
                        "source_path": str((temp_path / "movie.mp4").resolve()),
                        "output_base": str(temp_path.resolve()),
                        "source_fingerprint": "0:0",
                        "backend": "faster-whisper",
                        "device": "cpu",
                        "model": "large-v3",
                        "language": None,
                        "target_lang": None,
                        "formats": ["srt"],
                        "primary_outputs": [],
                        "segments": [],
                        "stage_status": {
                            "transcription_complete": False,
                            "translation_pending": True,
                            "translation_complete": False,
                            "translation_failed": False,
                            "translation_error": None,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "99"):
                load_stage_artifact(artifact_path)

    def test_artifact_path_for_source_only(self) -> None:
        source_path = Path("/tmp/videos/movie.mp4")

        self.assertEqual(
            Path("/tmp/videos") / f"movie{ARTIFACT_FILENAME_SUFFIX}",
            artifact_path_for(source_path, None),
        )

    def test_artifact_path_for_with_output_dir(self) -> None:
        source_path = Path("/tmp/videos/movie.mp4")
        output_dir = Path("/tmp/output")

        self.assertEqual(
            output_dir / f"movie{ARTIFACT_FILENAME_SUFFIX}",
            artifact_path_for(source_path, output_dir),
        )


class TestRunnerStageSplit(unittest.TestCase):
    def _make_args(self, **overrides: object) -> argparse.Namespace:
        defaults: dict[str, object] = {
            "workers": 1,
            "backend": "faster-whisper",
            "model": "large-v3",
            "device": "cpu",
            "language": None,
            "content_type": "auto",
            "beam_size": 5,
            "compute_type": "int8",
            "hf_token": None,
            "diarize": False,
            "whisper_cpp_model_path": None,
            "verbose": False,
            "translate_to": "ko",
            "translation_model": "gpt-4.1-mini",
            "translation_base_url": "https://translation.example/v1",
            "translation_api_key": "secret",
            "postprocess_translation": False,
            "postprocess_mode": "auto",
            "postprocess_model": None,
            "postprocess_base_url": None,
            "postprocess_api_key": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_stage1_writes_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = (temp_path / "movie.mp4").resolve()
            video_path.write_bytes(b"video")
            output_dir = (temp_path / "out").resolve()
            task = {
                "video_path": str(video_path),
                "folder_hash": "folder-hash",
                "folder_path": str(video_path.parent),
            }
            args = self._make_args()
            segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
            info = {"language": "en", "duration": 1.0}
            written_paths = [output_dir / "movie.srt"]
            artifact_path = output_dir / f"movie{ARTIFACT_FILENAME_SUFFIX}"

            with (
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    return_value=(segments, info),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=written_paths,
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_stage_artifact",
                    return_value=artifact_path,
                ) as write_artifact,
            ):
                result = run_stage1(task, args, frozenset({"srt"}), output_dir, 2, 0)

        metadata = result.artifact_metadata
        assert metadata is not None
        self.assertTrue(result.success)
        self.assertEqual(str(artifact_path), result.artifact_path)
        self.assertEqual(
            build_stage_artifact_metadata(artifact_path).get("path"),
            metadata.get("path"),
        )
        self.assertEqual("ko", metadata.get("target_lang"))
        self.assertTrue(metadata.get("transcription_complete"))
        self.assertTrue(metadata.get("translation_pending"))
        self.assertFalse(metadata.get("translation_complete"))
        self.assertFalse(metadata.get("translation_failed"))
        write_artifact.assert_called_once()

    def test_stage1_passes_content_type_to_transcribe(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = (temp_path / "movie.mp4").resolve()
            video_path.write_bytes(b"video")
            output_dir = (temp_path / "out").resolve()
            task = {
                "video_path": str(video_path),
                "folder_hash": "folder-hash",
                "folder_path": str(video_path.parent),
            }
            args = self._make_args(content_type="music", language="ja")
            segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
            info = {"language": "ja", "duration": 1.0}
            written_paths = [output_dir / "movie.srt"]
            artifact_path = output_dir / f"movie{ARTIFACT_FILENAME_SUFFIX}"

            with (
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    return_value=(segments, info),
                ) as mock_transcribe,
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=written_paths,
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_stage_artifact",
                    return_value=artifact_path,
                ),
            ):
                result = run_stage1(task, args, frozenset({"srt"}), output_dir, 2, 0)

        self.assertTrue(result.success)
        self.assertEqual("music", mock_transcribe.call_args.kwargs["content_type"])

    def test_stage1_auto_retries_with_music_preset_when_initial_quality_is_suspicious(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = (temp_path / "movie.mp4").resolve()
            video_path.write_bytes(b"video")
            output_dir = (temp_path / "out").resolve()
            task = {
                "video_path": str(video_path),
                "folder_hash": "folder-hash",
                "folder_path": str(video_path.parent),
            }
            args = self._make_args(
                content_type="auto", language=None, translate_to=None
            )
            initial_segments = [
                {"start": 0.0, "end": 1.0, "text": "la la"},
                {"start": 1.0, "end": 2.0, "text": "la la"},
            ]
            initial_info = {
                "language": "ja",
                "language_probability": 0.41,
                "duration": 2.0,
                "content_type": "auto",
            }
            retried_segments = [
                {"start": 0.0, "end": 1.0, "text": "青い風"},
                {"start": 1.0, "end": 2.0, "text": "光る海"},
            ]
            retried_info = {
                "language": "ja",
                "language_probability": 0.98,
                "duration": 2.0,
                "content_type": "music",
            }
            written_paths = [output_dir / "movie.srt"]
            artifact_path = output_dir / f"movie{ARTIFACT_FILENAME_SUFFIX}"

            with (
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    side_effect=[
                        (initial_segments, initial_info),
                        (retried_segments, retried_info),
                    ],
                ) as mock_transcribe,
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=written_paths,
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_stage_artifact",
                    return_value=artifact_path,
                ),
            ):
                result = run_stage1(task, args, frozenset({"srt"}), output_dir, 2, 0)

        self.assertTrue(result.success, result.error)
        self.assertEqual(2, mock_transcribe.call_count)
        self.assertEqual(
            "auto", mock_transcribe.call_args_list[0].kwargs["content_type"]
        )
        self.assertEqual(
            "music", mock_transcribe.call_args_list[1].kwargs["content_type"]
        )
        self.assertEqual("ja", mock_transcribe.call_args_list[1].kwargs["language"])

    def test_stage1_holds_suspicious_outputs_and_keeps_canonical_primary_names(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = (temp_path / "movie.mp4").resolve()
            video_path.write_bytes(b"video")
            output_dir = (temp_path / "out").resolve()
            task = {
                "video_path": str(video_path),
                "folder_hash": "folder-hash",
                "folder_path": str(video_path.parent),
            }
            args = self._make_args(
                content_type="speech",
                language="en",
                translate_to=None,
            )
            suspicious_segments = [
                {"start": 0.0, "end": 1.0, "text": "uh"},
                {"start": 1.0, "end": 2.0, "text": "uh"},
            ]
            suspicious_info = {
                "language": "en",
                "language_probability": 0.31,
                "duration": 2.0,
                "content_type": "speech",
            }

            with patch(
                "vid_to_sub_app.cli.runner.transcribe",
                return_value=(suspicious_segments, suspicious_info),
            ):
                result = run_stage1(
                    task,
                    args,
                    frozenset({"json", "srt"}),
                    output_dir,
                    2,
                    0,
                )

            assert result.artifact_path is not None
            artifact = load_stage_artifact(Path(result.artifact_path))
            self.assertFalse(result.success)
            self.assertEqual("quality_hold", result.stage)
            self.assertIn("withheld", result.error or "")
            self.assertFalse((output_dir / "movie.srt").exists())
            self.assertFalse((output_dir / "movie.json").exists())
            self.assertTrue((output_dir / "movie.stage1.suspicious.srt").exists())
            self.assertTrue((output_dir / "movie.stage1.suspicious.json").exists())
            self.assertEqual(
                [
                    str(output_dir / "movie.srt"),
                    str(output_dir / "movie.json"),
                ],
                artifact["primary_outputs"],
            )
            self.assertTrue(artifact["quality"]["output_held"])
            assert result.artifact_metadata is not None
            self.assertTrue(result.artifact_metadata.get("stage1_output_held"))

    def test_process_one_continues_to_stage2_after_quality_hold_when_translation_requested(
        self,
    ) -> None:
        task = {
            "video_path": "/tmp/movie.mp4",
            "folder_hash": "folder-hash",
            "folder_path": "/tmp",
        }
        args = self._make_args(translate_to="ko", force_translate=True)
        stage1_result = ProcessResult(
            success=False,
            video_path="/tmp/movie.mp4",
            folder_hash="folder-hash",
            folder_path="/tmp",
            worker_id=0,
            language="en",
            video_duration=1.0,
            output_paths=["/tmp/movie.stage1.suspicious.srt"],
            segments=1,
            elapsed_sec=0.1,
            error="held",
            stage="quality_hold",
            artifact_path="/tmp/movie.stage1.json",
        )
        stage2_result = ProcessResult(
            success=True,
            video_path="/tmp/movie.mp4",
            folder_hash="folder-hash",
            folder_path="/tmp",
            worker_id=0,
            language="en",
            output_paths=["/tmp/movie.ko.srt"],
            segments=1,
            elapsed_sec=0.2,
            artifact_path="/tmp/movie.stage1.json",
        )

        with (
            patch("vid_to_sub_app.cli.runner.run_stage1", return_value=stage1_result),
            patch(
                "vid_to_sub_app.cli.runner.run_stage2", return_value=stage2_result
            ) as run_stage2_mock,
        ):
            result = process_one(task, args, frozenset({"srt"}), None, 2, 0)

        run_stage2_mock.assert_called_once_with(Path("/tmp/movie.stage1.json"), args)
        self.assertTrue(result.success)
        self.assertEqual(
            ["/tmp/movie.stage1.suspicious.srt", "/tmp/movie.ko.srt"],
            result.output_paths,
        )

    def test_stage2_does_not_call_transcribe(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = (temp_path / "movie.mp4").resolve()
            output_dir = (temp_path / "out").resolve()
            output_dir.mkdir()
            source_path.write_bytes(b"video")
            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source_path),
                "output_base": str(output_dir),
                "source_fingerprint": f"{source_path.stat().st_size}:{int(source_path.stat().st_mtime)}",
                "backend": "faster-whisper",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.0,
                "quality": {"suspicious": False, "reasons": []},
                "target_lang": "ko",
                "formats": ["srt"],
                "primary_outputs": [str(output_dir / "movie.srt")],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": True,
                    "translation_complete": False,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = write_stage_artifact(artifact, output_dir, source_path)
            args = self._make_args()
            translated_segments = [{"start": 0.0, "end": 1.0, "text": "안녕"}]
            translated_written = [output_dir / "movie.ko.srt"]

            with (
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    side_effect=AssertionError("transcribe should not run"),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                    return_value=(translated_segments, {"model": "gpt-4.1-mini"}),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=translated_written,
                ),
            ):
                result = run_stage2(artifact_path, args)

            loaded_artifact = load_stage_artifact(artifact_path)

        metadata = result.artifact_metadata
        assert metadata is not None
        self.assertTrue(result.success)
        self.assertEqual(str(artifact_path), result.artifact_path)
        self.assertEqual(str(artifact_path), metadata.get("path"))
        self.assertEqual("ko", metadata.get("target_lang"))
        self.assertFalse(metadata.get("translation_pending"))
        self.assertTrue(metadata.get("translation_complete"))
        self.assertFalse(metadata.get("translation_failed"))
        self.assertIsNone(metadata.get("translation_error"))
        self.assertTrue(loaded_artifact["stage_status"]["translation_complete"])

    def test_process_one_no_translation_skips_stage2(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = (temp_path / "movie.mp4").resolve()
            video_path.write_bytes(b"video")
            output_dir = (temp_path / "out").resolve()
            task = {
                "video_path": str(video_path),
                "folder_hash": "folder-hash",
                "folder_path": str(video_path.parent),
            }
            args = self._make_args(translate_to=None)
            segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
            info = {"language": "en", "duration": 1.0}
            written_paths = [output_dir / "movie.srt"]
            artifact_path = output_dir / f"movie{ARTIFACT_FILENAME_SUFFIX}"

            with (
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    return_value=(segments, info),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=written_paths,
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_stage_artifact",
                    return_value=artifact_path,
                ),
                patch(
                    "vid_to_sub_app.cli.runner.translate_segments_openai_compatible"
                ) as translate,
            ):
                result = process_one(task, args, frozenset({"srt"}), output_dir, 2, 0)

        self.assertTrue(result.success)
        translate.assert_not_called()


    def test_assess_stage1_quality_zero_segments_reports_empty_transcript_not_high_repetition(
        self,
    ) -> None:
        """Zero segments must not falsely trigger high_repetition.

        unique_ratio = 0/1 = 0.0 → repeated_ratio = 1.0 when non_empty_texts is empty.
        The guard introduced for this edge case should emit empty_transcript instead.
        """
        quality = _assess_stage1_quality([], {"language_probability": 1.0})
        self.assertIn("empty_transcript", quality["reasons"])
        self.assertNotIn("high_repetition", quality["reasons"])
        self.assertTrue(quality["suspicious"])

    def test_assess_stage1_quality_zero_segments_with_no_info_still_not_high_repetition(
        self,
    ) -> None:  
        """Same guard applies when language_probability is absent from info."""
        quality = _assess_stage1_quality([], {})
        self.assertIn("empty_transcript", quality["reasons"])
        self.assertNotIn("high_repetition", quality["reasons"])

    def test_assess_stage1_quality_all_empty_text_segments_reports_empty_transcript(
        self,
    ) -> None:
        """Segments whose text strips to empty string count as zero non-empty texts."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "  "},
            {"start": 1.0, "end": 2.0, "text": ""},
        ]
        quality = _assess_stage1_quality(segments, {"language_probability": 1.0})
        self.assertIn("empty_transcript", quality["reasons"])
        self.assertNotIn("high_repetition", quality["reasons"])

    def test_assess_stage1_quality_genuine_repetition_still_flagged(
        self,
    ) -> None:
        """Real high-repetition content must still be caught after the guard."""
        segments = [
            {"start": float(i), "end": float(i + 1), "text": "Валерий Сюткин"}
            for i in range(10)
        ]
        quality = _assess_stage1_quality(segments, {"language_probability": 1.0})
        self.assertIn("high_repetition", quality["reasons"])
        self.assertNotIn("empty_transcript", quality["reasons"])
        self.assertTrue(quality["suspicious"])

class TestRunStage2Idempotency(unittest.TestCase):
    """run_stage2 must skip translation when output files already exist."""

    def _make_args(self, tmpdir: Path) -> argparse.Namespace:
        args = argparse.Namespace()
        args.translate_to = "ko"
        args.translation_model = "test-model"
        args.translation_base_url = "http://localhost/v1"
        args.translation_api_key = "test-key"
        args.postprocess_translation = False
        args.force_translate = False
        return args

    def _write_source(self, source: Path, payload: bytes = b"video") -> str:
        source.write_bytes(payload)
        source_stat = source.stat()
        return f"{source_stat.st_size}:{int(source_stat.st_mtime)}"

    def test_run_stage2_blocks_suspicious_artifact_without_force_translate(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            source_fingerprint = self._write_source(source)
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": source_fingerprint,
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.41,
                "duration": 1.0,
                "quality": {
                    "suspicious": True,
                    "reasons": ["low_language_probability"],
                },
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": True,
                    "translation_complete": False,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible"
            ) as mock_translate:
                result = run_stage2(artifact_path, args)

            loaded_artifact = load_stage_artifact(artifact_path)

        self.assertFalse(result.success)
        self.assertIn("--force-translate", result.error or "")
        mock_translate.assert_not_called()
        self.assertTrue(loaded_artifact["stage_status"]["translation_failed"])
        self.assertFalse(loaded_artifact["stage_status"]["translation_complete"])
        self.assertEqual(
            result.error, loaded_artifact["stage_status"]["translation_error"]
        )

    def test_run_stage2_allows_suspicious_artifact_when_force_translate_enabled(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            source_fingerprint = self._write_source(source)
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": source_fingerprint,
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.41,
                "duration": 1.0,
                "quality": {
                    "suspicious": True,
                    "reasons": ["low_language_probability"],
                },
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": True,
                    "translation_complete": False,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            args.force_translate = True
            translated_segs = [{"start": 0.0, "end": 1.0, "text": "안녕"}]
            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                return_value=(translated_segs, {}),
            ) as mock_translate:
                result = run_stage2(artifact_path, args)

        mock_translate.assert_called_once()
        self.assertTrue(result.success, result.error)

    def test_run_stage2_skips_when_translation_complete_and_output_exists(
        self,
    ) -> None:
        """run_stage2 returns success without calling the translation API when
        translation_complete=True and the translated file already exists on disk."""
        import argparse
        from unittest.mock import patch

        from vid_to_sub_app.cli.runner import run_stage2
        from vid_to_sub_app.cli.stage_artifact import write_stage_artifact

        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            source_fingerprint = self._write_source(source)
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")
            # Translated file already exists
            translated = tmpdir / "movie.ko.srt"
            translated.write_text("1\n00:00:00,000 --> 00:00:01,000\n안녕")

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": source_fingerprint,
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.0,
                "quality": {"suspicious": False, "reasons": []},
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": False,
                    "translation_complete": True,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible"
            ) as mock_translate:
                result = run_stage2(artifact_path, args)

            metadata = result.artifact_metadata
            assert metadata is not None
            self.assertTrue(result.success, result.error)
            mock_translate.assert_not_called()
            self.assertEqual(str(artifact_path), metadata.get("path"))
            self.assertTrue(metadata.get("translation_complete"))
            self.assertFalse(metadata.get("translation_failed"))
            self.assertIsNone(metadata.get("translation_error"))

    def test_run_stage2_proceeds_when_output_missing(self) -> None:
        """run_stage2 proceeds normally when translation_complete=True but
        the translated file does not exist on disk."""
        import argparse
        from unittest.mock import patch

        from vid_to_sub_app.cli.runner import run_stage2
        from vid_to_sub_app.cli.stage_artifact import write_stage_artifact

        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            source_fingerprint = self._write_source(source)
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")
            # Translated file does NOT exist — should NOT be idempotent-skipped

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": source_fingerprint,
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.0,
                "quality": {"suspicious": False, "reasons": []},
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": False,
                    "translation_complete": True,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            translated_segs = [{"start": 0.0, "end": 1.0, "text": "안녕"}]
            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                return_value=(translated_segs, {}),
            ) as mock_translate:
                result = run_stage2(artifact_path, args)

            metadata = result.artifact_metadata
            assert metadata is not None
            mock_translate.assert_called_once()
            self.assertTrue(result.success, result.error)
            self.assertEqual(str(artifact_path), metadata.get("path"))
            self.assertTrue(metadata.get("translation_complete"))
            self.assertFalse(metadata.get("translation_failed"))

    def test_run_stage2_persists_partial_progress_on_translation_failure(self) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            source_fingerprint = self._write_source(source)
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")
            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": source_fingerprint,
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.0,
                "quality": {"suspicious": False, "reasons": []},
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Hello"},
                    {"start": 1.0, "end": 2.0, "text": "World"},
                ],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": True,
                    "translation_complete": False,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)
            args = self._make_args(tmpdir)
            progress_path = tmpdir / "movie.stage2.progress.jsonl"

            def fake_translate(**kwargs):
                kwargs["on_batch_success"]([1], ["안녕"])
                raise RuntimeError("boom")

            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                side_effect=fake_translate,
            ):
                result = run_stage2(artifact_path, args)

            self.assertFalse(result.success)
            self.assertEqual("boom", result.error)
            self.assertTrue(progress_path.exists())
            progress_lines = progress_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(1, len(progress_lines))
            self.assertEqual(
                {
                    "stage": "translate",
                    "items": [{"segment_number": 1, "text": "안녕"}],
                },
                json.loads(progress_lines[0]),
            )
            loaded_artifact = load_stage_artifact(artifact_path)
            self.assertTrue(loaded_artifact["stage_status"]["translation_failed"])
            self.assertEqual(
                "boom", loaded_artifact["stage_status"]["translation_error"]
            )

    def test_run_stage2_resumes_from_progress_and_clears_sidecar_on_success(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            source_fingerprint = self._write_source(source)
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")
            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": source_fingerprint,
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.0,
                "quality": {"suspicious": False, "reasons": []},
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Hello"},
                    {"start": 1.0, "end": 2.0, "text": "World"},
                ],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": True,
                    "translation_complete": False,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)
            progress_path = tmpdir / "movie.stage2.progress.jsonl"
            progress_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "stage": "translate",
                                "items": [{"segment_number": 1, "text": "cached-1"}],
                            }
                        ),
                        json.dumps(
                            {
                                "stage": "postprocess",
                                "items": [{"segment_number": 1, "text": "polished-1"}],
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            args = self._make_args(tmpdir)
            args.postprocess_translation = True
            args.postprocess_mode = "auto"
            args.postprocess_model = None
            args.postprocess_base_url = None
            args.postprocess_api_key = None
            translate_resume: dict[int, str] | None = None
            postprocess_resume: dict[int, str] | None = None

            def fake_translate(**kwargs):
                nonlocal translate_resume
                translate_resume = kwargs["resume_text_by_number"]
                kwargs["on_batch_success"]([2], ["fresh-2"])
                return (
                    [
                        {"start": 0.0, "end": 1.0, "text": "cached-1"},
                        {"start": 1.0, "end": 2.0, "text": "fresh-2"},
                    ],
                    {"model": "test-model", "resumed_segment_count": 1},
                )

            def fake_postprocess(**kwargs):
                nonlocal postprocess_resume
                postprocess_resume = kwargs["resume_text_by_number"]
                kwargs["on_batch_success"]([2], ["polished-2"])
                return (
                    [
                        {"start": 0.0, "end": 1.0, "text": "polished-1"},
                        {"start": 1.0, "end": 2.0, "text": "polished-2"},
                    ],
                    {"model": "post-model", "resumed_segment_count": 1},
                )

            with (
                patch(
                    "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                    side_effect=fake_translate,
                ),
                patch(
                    "vid_to_sub_app.cli.runner.postprocess_translated_segments_openai_compatible",
                    side_effect=fake_postprocess,
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=[tmpdir / "movie.ko.srt"],
                ),
            ):
                result = run_stage2(artifact_path, args)

            self.assertTrue(result.success, result.error)
            self.assertEqual({1: "cached-1"}, translate_resume)
            self.assertEqual({1: "polished-1"}, postprocess_resume)
            self.assertFalse(progress_path.exists())
            loaded_artifact = load_stage_artifact(artifact_path)
            self.assertTrue(loaded_artifact["stage_status"]["translation_complete"])
            self.assertFalse(loaded_artifact["stage_status"]["translation_failed"])

    def test_run_stage2_prefers_cli_translate_to_over_artifact_target(self) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            source_fingerprint = self._write_source(source)
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": source_fingerprint,
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.0,
                "quality": {"suspicious": False, "reasons": []},
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": True,
                    "translation_complete": False,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            args.translate_to = "ja"
            translated_segs = [{"start": 0.0, "end": 1.0, "text": "こんにちは"}]
            with (
                patch(
                    "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                    return_value=(translated_segs, {}),
                ) as mock_translate,
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=[tmpdir / "movie.ja.srt"],
                ) as mock_write_outputs,
            ):
                result = run_stage2(artifact_path, args)

            loaded_artifact = load_stage_artifact(artifact_path)

        self.assertTrue(result.success, result.error)
        self.assertEqual("ja", mock_translate.call_args.kwargs["target_language"])
        self.assertEqual(".ja", mock_write_outputs.call_args.kwargs["name_suffix"])
        self.assertEqual("ja", loaded_artifact["target_lang"])

    def test_run_stage2_blocks_source_fingerprint_mismatch_without_force_translate(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            _ = self._write_source(source)
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": "0:0",
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.0,
                "quality": {"suspicious": False, "reasons": []},
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": True,
                    "translation_complete": False,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible"
            ) as mock_translate:
                result = run_stage2(artifact_path, args)

            loaded_artifact = load_stage_artifact(artifact_path)

        self.assertFalse(result.success)
        self.assertIn("source fingerprint mismatch", (result.error or "").lower())
        self.assertIn("--force-translate", result.error or "")
        mock_translate.assert_not_called()
        self.assertTrue(loaded_artifact["stage_status"]["translation_failed"])
        self.assertFalse(loaded_artifact["stage_status"]["translation_complete"])

    def test_run_stage2_allows_source_fingerprint_mismatch_with_force_translate(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            _ = self._write_source(source)
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": "0:0",
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.0,
                "quality": {"suspicious": False, "reasons": []},
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": True,
                    "translation_complete": False,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            args.force_translate = True
            translated_segs = [{"start": 0.0, "end": 1.0, "text": "안녕"}]
            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                return_value=(translated_segs, {}),
            ) as mock_translate:
                result = run_stage2(artifact_path, args)

        mock_translate.assert_called_once()
        self.assertTrue(result.success, result.error)

    def test_fingerprint_emits_all_required_fields(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "안녕"}], ensure_ascii=False
                    )
                )
            )

        stdout_buf = io.StringIO()
        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(stdout_buf), redirect_stderr(io.StringIO()):
                translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=10,
                )

        fp_events = [
            e
            for e in _event_payloads(stdout_buf)
            if e.get("event") == "translation_execution_fingerprint"
        ]
        self.assertTrue(fp_events, "no fingerprint event emitted")
        fp = fp_events[0]
        for field in (
            "exec_path",
            "cwd",
            "python",
            "argv",
            "repo_root",
            "git_dirty",
            "endpoint_host",
            "model",
            "git_sha",
            "package_path",
            "hostname",
            "translation_chunk_size",
            "distributed",
            "endpoint",
        ):
            self.assertIn(field, fp, f"fingerprint missing field: {field}")
        self.assertEqual("gpt-4.1-mini", fp["model"])
        self.assertEqual(10, fp["translation_chunk_size"])
        self.assertEqual(
            "https://translation.example/v1/chat/completions",
            fp["endpoint"],
        )
        self.assertEqual(
            "translation.example",
            fp["endpoint_host"],
        )

    def test_batch_log_emits_all_required_fields(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
        stderr_buf = io.StringIO()

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(
                    json.dumps(
                        [{"segment_number": 1, "text": "안녕"}], ensure_ascii=False
                    )
                )
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr_buf):
                translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=5,
                )

        batch_events = [
            e
            for e in _event_payloads(stderr_buf)
            if e.get("event") == "translation_batch_result"
        ]
        self.assertTrue(batch_events, "no batch_result event emitted")
        ev = batch_events[0]
        for field in (
            "attempt",
            "batch_start_idx",
            "batch_end_idx",
            "segment_numbers",
            "endpoint_host",
            "model",
            "temperature",
            "max_output_tokens",
            "http_status",
            "response_id",
            "request_id",
            "finish_reason",
            "usage",
            "raw_response_file",
            "parsed_count",
            "parsed_sample",
            "requested_count",
            "blank_or_whitespace_count",
            "effective_sent_count",
            "chunk_size",
            "configured_chunk_size",
        ):
            self.assertIn(field, ev, f"batch_result missing field: {field}")
        self.assertEqual(1, ev["attempt"])
        self.assertEqual(1, ev["batch_start_idx"])
        self.assertEqual(1, ev["batch_end_idx"])
        self.assertEqual([1], ev["segment_numbers"])
        self.assertEqual("translation.example", ev["endpoint_host"])
        self.assertEqual("gpt-4.1-mini", ev["model"])
        self.assertEqual(0, ev["temperature"])

    def test_contract_error_includes_full_fingerprint_context(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "fail"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(_chat_payload("[]", finish_reason="length"))

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(TranslationContractError) as ctx:
                    translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=1,
                        translation_mode="strict",
                    )

        msg = str(ctx.exception)
        for field_prefix in (
            "cwd=",
            "python=",
            "argv=",
            "repo_root=",
            "git_sha=",
            "git_dirty=",
            "translation_chunk_size=",
            "distributed=",
            "exec_path=",
            "package_path=",
            "endpoint_host=",
            "model=",
        ):
            self.assertIn(field_prefix, msg, f"contract error missing: {field_prefix}")

    def test_translate_raises_on_object_item_missing_segment_number_key(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "one"}]

        def fake_urlopen(request):
            return _FakeHTTPResponse(
                _chat_payload(json.dumps([{"text": "missing-key"}], ensure_ascii=False))
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(TranslationContractError) as ctx:
                    translate_segments_openai_compatible(
                        segments=segments,
                        target_language="ko",
                        translation_model="gpt-4.1-mini",
                        translation_base_url="https://translation.example/v1",
                        translation_api_key="secret",
                        source_language="en",
                        chunk_size=1,
                    )

        self.assertIn("segment_number_must_be_int", str(ctx.exception))

    def test_translation_retries_when_100_sent_99_returned(self) -> None:
        segments = [
            {"start": float(i), "end": float(i + 1), "text": f"seg-{i + 1}"}
            for i in range(100)
        ]
        request_sizes: list[int] = []
        stderr_buf = io.StringIO()

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_sizes.append(len(segment_numbers))
            if len(segment_numbers) == 100:
                # Return only 99 — deliberately omit segment 50
                items = [
                    {"segment_number": n, "text": f"ok-{n}"}
                    for n in segment_numbers
                    if n != 50
                ]
                return _FakeHTTPResponse(
                    _chat_payload(json.dumps(items, ensure_ascii=False))
                )
            # All smaller batches succeed
            items = [{"segment_number": n, "text": f"ok-{n}"} for n in segment_numbers]
            return _FakeHTTPResponse(
                _chat_payload(json.dumps(items, ensure_ascii=False))
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr_buf):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=100,
                )

        # First request was 100, which triggered a retry at next size
        self.assertEqual(100, request_sizes[0])
        self.assertGreater(len(request_sizes), 1)
        contract_events = [
            e
            for e in _event_payloads(stderr_buf)
            if e["event"] == "translation_batch_contract_error"
        ]
        self.assertTrue(
            contract_events, "no contract error emitted for 100\u219299 mismatch"
        )
        self.assertEqual(
            "segment_number_contract_mismatch", contract_events[0]["reason"]
        )
        self.assertNotEqual([], contract_events[0]["parsed_sample"])
        self.assertEqual(100, len(translated_segments))

    def test_translation_retries_on_multi_item_duplicate_segment_number(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
        ]
        request_segment_numbers: list[list[int]] = []
        stderr_buf = io.StringIO()

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_segment_numbers.append(segment_numbers)
            if segment_numbers == [1, 2]:
                # Return duplicate segment_number 1 — triggers retry
                return _FakeHTTPResponse(
                    _chat_payload(
                        json.dumps(
                            [
                                {"segment_number": 1, "text": "dup-one-a"},
                                {"segment_number": 1, "text": "dup-one-b"},
                            ],
                            ensure_ascii=False,
                        )
                    )
                )
            items = [{"segment_number": n, "text": f"ok-{n}"} for n in segment_numbers]
            return _FakeHTTPResponse(
                _chat_payload(json.dumps(items, ensure_ascii=False))
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr_buf):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=2,
                )

        # Initial [1,2] failed with duplicate, retried as [1] and [2]
        self.assertEqual([[1, 2], [1], [2]], request_segment_numbers)
        self.assertEqual(["ok-1", "ok-2"], [s["text"] for s in translated_segments])
        contract_events = [
            e
            for e in _event_payloads(stderr_buf)
            if e["event"] == "translation_batch_contract_error"
        ]
        self.assertTrue(contract_events)
        self.assertEqual(
            "segment_number_contract_mismatch", contract_events[0]["reason"]
        )

    def test_translation_retries_on_multi_item_unexpected_segment_number(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "one"},
            {"start": 1.0, "end": 2.0, "text": "two"},
        ]
        request_segment_numbers: list[list[int]] = []
        stderr_buf = io.StringIO()

        def fake_urlopen(request):
            payload = json.loads(request.data.decode("utf-8"))
            user_payload = json.loads(payload["messages"][1]["content"])
            segment_numbers = user_payload["segment_numbers"]
            request_segment_numbers.append(segment_numbers)
            if segment_numbers == [1, 2]:
                # Return segment 99 which was never requested — triggers retry
                return _FakeHTTPResponse(
                    _chat_payload(
                        json.dumps(
                            [
                                {"segment_number": 1, "text": "one-ko"},
                                {"segment_number": 99, "text": "unexpected"},
                            ],
                            ensure_ascii=False,
                        )
                    )
                )
            items = [{"segment_number": n, "text": f"ok-{n}"} for n in segment_numbers]
            return _FakeHTTPResponse(
                _chat_payload(json.dumps(items, ensure_ascii=False))
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            with redirect_stdout(io.StringIO()), redirect_stderr(stderr_buf):
                translated_segments, _ = translate_segments_openai_compatible(
                    segments=segments,
                    target_language="ko",
                    translation_model="gpt-4.1-mini",
                    translation_base_url="https://translation.example/v1",
                    translation_api_key="secret",
                    source_language="en",
                    chunk_size=2,
                )

        # Initial [1,2] failed with unexpected 99, retried as [1] and [2]
        self.assertEqual([[1, 2], [1], [2]], request_segment_numbers)
        self.assertEqual(["ok-1", "ok-2"], [s["text"] for s in translated_segments])
        contract_events = [
            e
            for e in _event_payloads(stderr_buf)
            if e["event"] == "translation_batch_contract_error"
        ]
        self.assertTrue(contract_events)
        self.assertEqual(
            "segment_number_contract_mismatch", contract_events[0]["reason"]
        )


if __name__ == "__main__":
    _ = unittest.main()


class TestOverwriteTranslationFlag(unittest.TestCase):
    """U9: --overwrite-translation bypasses the idempotency guard in run_stage2."""

    def _make_args(self, output_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            translate_to="ko",
            translation_model="gpt-4o-mini",
            translation_base_url="https://example.com/v1",
            translation_api_key="key",
            postprocess_translation=False,
            overwrite_translation=True,
        )

    def _write_source(self, source: Path, payload: bytes = b"video") -> str:
        source.write_bytes(payload)
        source_stat = source.stat()
        return f"{source_stat.st_size}:{int(source_stat.st_mtime)}"

    def test_overwrite_translation_reruns_despite_translation_complete(self) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            source_fingerprint = self._write_source(source)
            srt = tmpdir / "movie.srt"
            srt.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8"
            )
            translated_srt = tmpdir / "movie.ko.srt"
            translated_srt.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\n\uc548\ub155\n", encoding="utf-8"
            )

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": source_fingerprint,
                "backend": "faster-whisper",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "language_probability": 0.99,
                "duration": 1.0,
                "quality": {"suspicious": False, "reasons": []},
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": False,
                    "translation_complete": True,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            translated_segs = [{"start": 0.0, "end": 1.0, "text": "\uc548\ub155"}]
            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                return_value=(translated_segs, {}),
            ) as mock_translate:
                result = run_stage2(artifact_path, args)

            # Even though translation_complete=True and output file exists,
            # --overwrite-translation=True must cause the API to be called again.
            mock_translate.assert_called_once()
            self.assertTrue(result.success, result.error)


class TestParserFlags(unittest.TestCase):
    def test_content_type_flag_parses(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["/tmp/input", "--content-type", "music"])

        self.assertEqual("music", args.content_type)

    def test_force_translate_flag_parses(self) -> None:
        parser = build_parser()

        args = parser.parse_args(["/tmp/input", "--force-translate"])

        self.assertTrue(args.force_translate)


class TestLegacyInlineRollback(unittest.TestCase):
    """U11: VID_TO_SUB_LEGACY_INLINE=1 restores old inline transcription+translation."""

    def test_legacy_inline_env_var_calls_inline_path(self) -> None:
        import os

        """When VID_TO_SUB_LEGACY_INLINE=1, process_one should use the inline path,
        not write any stage artifact, and return successfully."""
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            video = tmpdir / "movie.mp4"
            video.write_bytes(b"video")
            task = {
                "video_path": str(video),
                "folder_hash": "folder-hash",
                "folder_path": str(tmpdir),
            }
            args = argparse.Namespace(
                backend="faster-whisper",
                model="large-v3",
                device="cpu",
                language=None,
                content_type="speech",
                beam_size=5,
                compute_type=None,
                hf_token=None,
                diarize=False,
                whisper_cpp_model_path=None,
                translate_to=None,
                translation_model=None,
                translation_base_url=None,
                translation_api_key=None,
                postprocess_translation=False,
                postprocess_mode="auto",
                postprocess_model=None,
                postprocess_base_url=None,
                postprocess_api_key=None,
                workers=1,
                verbose=False,
            )
            formats = frozenset({"srt"})
            fake_segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
            fake_info = {
                "backend": "faster-whisper",
                "language": "en",
                "duration": 1.0,
                "model": "large-v3",
            }
            fake_srt = tmpdir / "movie.srt"
            fake_srt.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8"
            )

            with (
                patch.dict(os.environ, {"VID_TO_SUB_LEGACY_INLINE": "1"}),
                patch("vid_to_sub_app.cli.runner._LEGACY_INLINE", True),
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    return_value=(fake_segments, fake_info),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=[fake_srt],
                ),
            ):
                result = process_one(task, args, formats, tmpdir, 2, 0)

            # No artifact should be written in legacy inline mode
            artifact_files = list(tmpdir.glob("*.stage1.json"))
            self.assertEqual(
                [], artifact_files, "Legacy inline mode must not write stage artifacts"
            )
            self.assertTrue(result.success, result.error)
            self.assertIsNone(
                result.artifact_path, "Legacy inline result must have no artifact_path"
            )

    def test_legacy_inline_rejects_translation_requests(self) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            video = tmpdir / "movie.mp4"
            video.write_bytes(b"video")
            task = {
                "video_path": str(video),
                "folder_hash": "folder-hash",
                "folder_path": str(tmpdir),
            }
            args = argparse.Namespace(
                backend="faster-whisper",
                model="large-v3",
                device="cpu",
                language="en",
                content_type="speech",
                beam_size=5,
                compute_type=None,
                hf_token=None,
                diarize=False,
                whisper_cpp_model_path=None,
                translate_to="ko",
                translation_model="gpt-4.1-mini",
                translation_base_url="https://translation.example/v1",
                translation_api_key="secret",
                postprocess_translation=False,
                postprocess_mode="auto",
                postprocess_model=None,
                postprocess_base_url=None,
                postprocess_api_key=None,
                workers=1,
                verbose=False,
            )

            with (
                patch("vid_to_sub_app.cli.runner._LEGACY_INLINE", True),
                patch(
                    "vid_to_sub_app.cli.runner._process_one_inline",
                    side_effect=AssertionError("inline path must be blocked"),
                ),
            ):
                result = process_one(task, args, frozenset({"srt"}), tmpdir, 2, 0)

        self.assertFalse(result.success)
        self.assertEqual("legacy_inline", result.stage)
        self.assertIn("--translate-to", result.error or "")

    def test_legacy_inline_rejects_auto_content_type(self) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            video = tmpdir / "movie.mp4"
            video.write_bytes(b"video")
            task = {
                "video_path": str(video),
                "folder_hash": "folder-hash",
                "folder_path": str(tmpdir),
            }
            args = argparse.Namespace(
                backend="faster-whisper",
                model="large-v3",
                device="cpu",
                language=None,
                content_type="auto",
                beam_size=5,
                compute_type=None,
                hf_token=None,
                diarize=False,
                whisper_cpp_model_path=None,
                translate_to=None,
                translation_model=None,
                translation_base_url=None,
                translation_api_key=None,
                postprocess_translation=False,
                postprocess_mode="auto",
                postprocess_model=None,
                postprocess_base_url=None,
                postprocess_api_key=None,
                workers=1,
                verbose=False,
            )

            with (
                patch("vid_to_sub_app.cli.runner._LEGACY_INLINE", True),
                patch(
                    "vid_to_sub_app.cli.runner._process_one_inline",
                    side_effect=AssertionError("inline path must be blocked"),
                ),
            ):
                result = process_one(task, args, frozenset({"srt"}), tmpdir, 2, 0)

        self.assertFalse(result.success)
        self.assertEqual("legacy_inline", result.stage)
        self.assertIn("--content-type auto", result.error or "")


class TestParallelLoopEventParity(unittest.TestCase):
    """PR0 characterization tests: freeze job_finished/folder_finished event payloads.

    Both _run_stage1_parallel (main.py) and run_parallel (runner.py) emit the same
    event types. This test locks in the set of keys emitted by each loop so that
    PR6 (CLI orchestration centralization) cannot silently drop fields that the TUI
    consumes.
    """

    _REQUIRED_JOB_FINISHED_KEYS = frozenset(
        {
            "video_path",
            "worker_id",
            "status",
            "stage",
            "error",
            "elapsed_sec",
            "language",
            "video_duration",
            "output_paths",
            "segments",
            "artifact_path",
            "artifact_metadata",
            "folder_hash",
            "folder_path",
            "folder_total_files",
            "folder_completed_files",
            "folder_status",
            "folder_completed",
        }
    )

    _REQUIRED_FOLDER_FINISHED_KEYS = frozenset(
        {
            "folder_hash",
            "folder_path",
            "total_files",
            "folder_total_files",
            "folder_completed_files",
            "folder_status",
            "folder_completed",
        }
    )

    def _capture_events(
        self,
        run_fn,  # _run_stage1_parallel or run_parallel
        args: argparse.Namespace,
        manifest: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run run_fn and collect emitted events, returning (job_events, folder_events)."""
        import sys

        from vid_to_sub_app.shared.constants import EVENT_PREFIX

        job_events: list[dict[str, Any]] = []
        folder_events: list[dict[str, Any]] = []

        def capture_stdout(line: str) -> None:
            line = line.strip()
            if not line.startswith(EVENT_PREFIX):
                return
            try:
                payload = json.loads(line[len(EVENT_PREFIX) :])
            except json.JSONDecodeError:
                return
            if payload.get("event") == "job_finished":
                job_events.append(payload)
            elif payload.get("event") == "folder_finished":
                folder_events.append(payload)

        buf = io.StringIO()
        with redirect_stdout(buf):
            run_fn(manifest, args, frozenset({"srt"}), None)

        for line in buf.getvalue().splitlines():
            capture_stdout(line)

        return job_events, folder_events

    def _make_args(self, tmpdir: Path) -> argparse.Namespace:
        from vid_to_sub_app.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(
            [
                str(tmpdir),
                "--stage1-only",
                "--translate-to",
                "ko",
                "--workers",
                "1",
                "--backend-threads",
                "2",
            ]
        )
        return args

    def _make_manifest_and_video(self, tmpdir: Path) -> tuple[dict[str, Any], Path]:
        """Create one fake video and a corresponding run manifest."""
        video = tmpdir / "clip.mp4"
        video.write_bytes(b"fake")
        manifest = {
            "version": 1,
            "entries": [
                {
                    "video_path": str(video),
                    "folder_path": str(tmpdir),
                    "folder_hash": "abc123",
                }
            ],
        }
        return manifest, video

    def _run_stage1_parallel_via_import(self, manifest, args, formats, output_dir):
        from vid_to_sub_app.cli.main import _run_stage1_parallel

        return _run_stage1_parallel(manifest, args, formats, output_dir)

    def _run_parallel_via_import(self, manifest, args, formats, output_dir):
        from vid_to_sub_app.cli.runner import run_parallel

        return run_parallel(manifest, args, formats, output_dir)

    def _fake_run_stage1(
        self, task, args, formats, output_dir, backend_threads, worker_id
    ):
        """Minimal stub: returns a successful ProcessResult without actual transcription."""
        fake_srt = Path(str(task["folder_path"])) / "clip.srt"
        fake_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi", encoding="utf-8")
        return ProcessResult(
            success=True,
            video_path=str(task["video_path"]),
            folder_hash=str(task["folder_hash"]),
            folder_path=str(task["folder_path"]),
            worker_id=worker_id,
            stage="stage1",
            output_paths=[str(fake_srt)],
        )

    def _fake_process_one(
        self, task, args, formats, output_dir, backend_threads, worker_id
    ):
        """Minimal stub: returns a successful ProcessResult without actual transcription."""
        fake_srt = Path(str(task["folder_path"])) / "clip.srt"
        fake_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi", encoding="utf-8")
        return ProcessResult(
            success=True,
            video_path=str(task["video_path"]),
            folder_hash=str(task["folder_hash"]),
            folder_path=str(task["folder_path"]),
            worker_id=worker_id,
            stage="full",
            output_paths=[str(fake_srt)],
        )

    def test_run_stage1_parallel_emits_required_job_finished_keys(self) -> None:
        """_run_stage1_parallel must emit job_finished events with all required fields.

        Locks in the event contract before PR6 consolidates the two worker loops.
        """
        from vid_to_sub_app.cli import manifest as manifest_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            args = self._make_args(td)
            manifest, _ = self._make_manifest_and_video(td)

            with (
                patch(
                    "vid_to_sub_app.cli.main.run_stage1",
                    side_effect=self._fake_run_stage1,
                ),
                patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            ):
                job_events, folder_events = self._capture_events(
                    self._run_stage1_parallel_via_import, args, manifest
                )

        self.assertEqual(1, len(job_events), "expected exactly one job_finished event")
        job = job_events[0]
        missing = self._REQUIRED_JOB_FINISHED_KEYS - set(job.keys())
        self.assertEqual(
            set(),
            missing,
            f"job_finished event missing required keys: {missing}",
        )

        # folder_finished should also fire when the single folder completes
        self.assertGreaterEqual(
            len(folder_events), 1, "expected at least one folder_finished event"
        )
        folder = folder_events[0]
        missing_f = self._REQUIRED_FOLDER_FINISHED_KEYS - set(folder.keys())
        self.assertEqual(
            set(),
            missing_f,
            f"folder_finished event missing required keys: {missing_f}",
        )

    def test_run_parallel_emits_required_job_finished_keys(self) -> None:
        """run_parallel must emit job_finished events with all required fields.

        The set of emitted keys must be compatible with what _run_stage1_parallel
        emits, since both are consumed by the TUI progress machinery.
        """
        from vid_to_sub_app.cli import manifest as manifest_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            args = self._make_args(td)
            # run_parallel takes a full run, not stage1-only, so remove stage1-only
            args.stage1_only = False
            manifest, _ = self._make_manifest_and_video(td)

            with (
                patch(
                    "vid_to_sub_app.cli.runner.process_one",
                    side_effect=self._fake_process_one,
                ),
                patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            ):
                job_events, folder_events = self._capture_events(
                    self._run_parallel_via_import, args, manifest
                )

        self.assertEqual(1, len(job_events), "expected exactly one job_finished event")
        job = job_events[0]
        missing = self._REQUIRED_JOB_FINISHED_KEYS - set(job.keys())
        self.assertEqual(
            set(),
            missing,
            f"job_finished event missing required keys: {missing}",
        )

        self.assertGreaterEqual(
            len(folder_events), 1, "expected at least one folder_finished event"
        )
        folder = folder_events[0]
        missing_f = self._REQUIRED_FOLDER_FINISHED_KEYS - set(folder.keys())
        self.assertEqual(
            set(),
            missing_f,
            f"folder_finished event missing required keys: {missing_f}",
        )

    def test_run_stage1_parallel_and_run_parallel_emit_same_job_finished_keys(
        self,
    ) -> None:
        """Parity contract: both loops must emit an identical set of job_finished keys.

        This is the core guard for PR6: centralizing the worker loop must not silently
        drop any key that the other loop emits.
        """
        from vid_to_sub_app.cli import manifest as manifest_mod

        with tempfile.TemporaryDirectory() as tmpdir1:
            td1 = Path(tmpdir1)
            args1 = self._make_args(td1)
            manifest1, _ = self._make_manifest_and_video(td1)

            with (
                patch(
                    "vid_to_sub_app.cli.main.run_stage1",
                    side_effect=self._fake_run_stage1,
                ),
                patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            ):
                job_events_s1, _ = self._capture_events(
                    self._run_stage1_parallel_via_import, args1, manifest1
                )

        with tempfile.TemporaryDirectory() as tmpdir2:
            td2 = Path(tmpdir2)
            args2 = self._make_args(td2)
            args2.stage1_only = False
            manifest2, _ = self._make_manifest_and_video(td2)

            with (
                patch(
                    "vid_to_sub_app.cli.runner.process_one",
                    side_effect=self._fake_process_one,
                ),
                patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            ):
                job_events_par, _ = self._capture_events(
                    self._run_parallel_via_import, args2, manifest2
                )

        self.assertTrue(
            job_events_s1, "_run_stage1_parallel emitted no job_finished events"
        )
        self.assertTrue(job_events_par, "run_parallel emitted no job_finished events")

        keys_s1 = set(job_events_s1[0].keys())
        keys_par = set(job_events_par[0].keys())

        only_in_stage1 = keys_s1 - keys_par
        only_in_parallel = keys_par - keys_s1

        self.assertEqual(
            set(),
            only_in_stage1,
            f"Keys only in _run_stage1_parallel job_finished (not in run_parallel): {only_in_stage1}",
        )
        self.assertEqual(
            set(),
            only_in_parallel,
            f"Keys only in run_parallel job_finished (not in _run_stage1_parallel): {only_in_parallel}",
        )

    def test_run_stage1_parallel_treats_quality_hold_as_withheld(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            td = Path(tmpdir)
            args = self._make_args(td)
            manifest, _ = self._make_manifest_and_video(td)
            quality_hold_result = ProcessResult(
                success=False,
                video_path=str((td / "clip.mp4").resolve()),
                folder_hash="abc123",
                folder_path=str(td),
                worker_id=0,
                stage="quality_hold",
                elapsed_sec=0.1,
                error="empty_transcript",
                artifact_path=str(td / "clip.stage1.json"),
            )

            with (
                patch(
                    "vid_to_sub_app.cli.main.run_stage1",
                    return_value=quality_hold_result,
                ),
                patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            ):
                ok, err, suspicious_holds, artifact_paths = (
                    self._run_stage1_parallel_via_import(
                        manifest, args, frozenset({"srt"}), None
                    )
                )

        self.assertEqual(0, ok)
        self.assertEqual(0, err)
        self.assertEqual(1, suspicious_holds)
        self.assertEqual([str(td / "clip.stage1.json")], artifact_paths)

    def test_job_finished_payload_values_are_correct(self) -> None:
        """job_finished event payload must contain correct, non-empty values (not just keys).

        Guards against implementations that emit required keys but fill them with
        placeholder values (None, empty string, 0) that the TUI would silently misrender.
        Checks both _run_stage1_parallel and run_parallel for identical value semantics.
        """
        from vid_to_sub_app.cli import manifest as manifest_mod

        # --- stage1 path ---
        with tempfile.TemporaryDirectory() as tmpdir1:
            td1 = Path(tmpdir1)
            args1 = self._make_args(td1)
            manifest1, video1 = self._make_manifest_and_video(td1)

            with (
                patch(
                    "vid_to_sub_app.cli.main.run_stage1",
                    side_effect=self._fake_run_stage1,
                ),
                patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            ):
                job_events_s1, _ = self._capture_events(
                    self._run_stage1_parallel_via_import, args1, manifest1
                )

        self.assertEqual(1, len(job_events_s1))
        job_s1 = job_events_s1[0]

        # video_path must be the actual path of the queued video
        self.assertEqual(str(video1), job_s1["video_path"],
                         "stage1 job_finished: video_path must equal the input video")
        # folder_hash must be the non-empty hash from the manifest fixture
        self.assertEqual("abc123", job_s1["folder_hash"],
                         "stage1 job_finished: folder_hash must match manifest")
        # status on success must be 'done'
        self.assertEqual("done", job_s1["status"],
                         "stage1 job_finished: status must be 'done' on success")
        # worker_id must be an integer (0-based index)
        self.assertIsInstance(job_s1["worker_id"], int,
                              "stage1 job_finished: worker_id must be an int")

        # --- run_parallel path ---
        with tempfile.TemporaryDirectory() as tmpdir2:
            td2 = Path(tmpdir2)
            args2 = self._make_args(td2)
            args2.stage1_only = False
            manifest2, video2 = self._make_manifest_and_video(td2)

            with (
                patch(
                    "vid_to_sub_app.cli.runner.process_one",
                    side_effect=self._fake_process_one,
                ),
                patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            ):
                job_events_par, _ = self._capture_events(
                    self._run_parallel_via_import, args2, manifest2
                )

        self.assertEqual(1, len(job_events_par))
        job_par = job_events_par[0]

        self.assertEqual(str(video2), job_par["video_path"],
                         "run_parallel job_finished: video_path must equal the input video")
        self.assertEqual("abc123", job_par["folder_hash"],
                         "run_parallel job_finished: folder_hash must match manifest")
        self.assertEqual("done", job_par["status"],
                         "run_parallel job_finished: status must be 'done' on success")
        self.assertIsInstance(job_par["worker_id"], int,
                              "run_parallel job_finished: worker_id must be an int")

    def test_error_path_emits_job_finished_with_error_fields(self) -> None:
        """When the worker function raises, job_finished must fire with status='failed'.

        Guards both loops: _run_stage1_parallel (via main.run_stage1 side_effect) and
        run_parallel (via runner.process_one side_effect).  The error field must contain
        the exception message; the keys must match the full required contract even on
        the failure path.
        """
        from vid_to_sub_app.cli import manifest as manifest_mod

        def _raise_stage1(task, args, formats, output_dir, backend_threads, worker_id):
            raise RuntimeError("simulated_stage1_failure")

        def _raise_process_one(task, args, formats, output_dir, backend_threads, worker_id):
            raise RuntimeError("simulated_process_one_failure")

        # --- stage1 error path ---
        with tempfile.TemporaryDirectory() as tmpdir1:
            td1 = Path(tmpdir1)
            args1 = self._make_args(td1)
            manifest1, video1 = self._make_manifest_and_video(td1)

            with (
                patch(
                    "vid_to_sub_app.cli.main.run_stage1",
                    side_effect=_raise_stage1,
                ),
                patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            ):
                job_events_s1, _ = self._capture_events(
                    self._run_stage1_parallel_via_import, args1, manifest1
                )

        self.assertEqual(1, len(job_events_s1),
                         "stage1 error path: expected exactly one job_finished event")
        err_s1 = job_events_s1[0]

        # All required keys must be present on the error path too
        missing_s1 = self._REQUIRED_JOB_FINISHED_KEYS - set(err_s1.keys())
        self.assertEqual(set(), missing_s1,
                         f"stage1 error path: missing keys {missing_s1}")
        self.assertEqual("failed", err_s1["status"],
                         "stage1 error path: status must be 'failed'")
        self.assertIn("simulated_stage1_failure", str(err_s1["error"]),
                      "stage1 error path: error field must contain the exception message")
        self.assertIsInstance(err_s1["worker_id"], int,
                              "stage1 error path: worker_id must be an int")

        # --- run_parallel error path ---
        with tempfile.TemporaryDirectory() as tmpdir2:
            td2 = Path(tmpdir2)
            args2 = self._make_args(td2)
            args2.stage1_only = False
            manifest2, video2 = self._make_manifest_and_video(td2)

            with (
                patch(
                    "vid_to_sub_app.cli.runner.process_one",
                    side_effect=_raise_process_one,
                ),
                patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            ):
                job_events_par, _ = self._capture_events(
                    self._run_parallel_via_import, args2, manifest2
                )

        self.assertEqual(1, len(job_events_par),
                         "run_parallel error path: expected exactly one job_finished event")
        err_par = job_events_par[0]

        missing_par = self._REQUIRED_JOB_FINISHED_KEYS - set(err_par.keys())
        self.assertEqual(set(), missing_par,
                         f"run_parallel error path: missing keys {missing_par}")
        self.assertEqual("failed", err_par["status"],
                         "run_parallel error path: status must be 'failed'")
        self.assertIn("simulated_process_one_failure", str(err_par["error"]),
                      "run_parallel error path: error field must contain the exception message")
        self.assertIsInstance(err_par["worker_id"], int,
                              "run_parallel error path: worker_id must be an int")

        # Parity: both loops must emit the same set of keys on the error path
        self.assertEqual(
            set(err_s1.keys()),
            set(err_par.keys()),
            "Error-path parity: both loops must emit an identical set of job_finished keys",
        )


class TestWarnModelMemory(unittest.TestCase):
    """_warn_model_memory_if_needed must warn iff faster-whisper + multiple workers."""

    def _make_args(self, backend: str, model: str, workers: int) -> argparse.Namespace:
        return argparse.Namespace(backend=backend, model=model, workers=workers)

    def _run(
        self, backend: str, model: str, workers: int, available_vram: float | None = None
    ) -> str:
        from vid_to_sub_app.cli.runner import _warn_model_memory_if_needed
        import io
        buf = io.StringIO()
        with unittest.mock.patch("sys.stderr", buf):
            with unittest.mock.patch(
                "vid_to_sub_app.shared.env.detect_cuda_total_memory_gb",
                return_value=available_vram,
            ):
                try:
                    _warn_model_memory_if_needed(self._make_args(backend, model, workers), workers)
                except Exception:
                    pass
        return buf.getvalue()

    def test_no_warning_for_single_worker(self) -> None:
        out = self._run("faster-whisper", "large-v3", 1)
        self.assertEqual("", out)

    def test_no_warning_for_non_faster_whisper_backend(self) -> None:
        out = self._run("whisper-cpp", "large-v3", 4)
        self.assertEqual("", out)

    def test_generic_warning_when_no_vram_data(self) -> None:
        out = self._run("faster-whisper", "unknown-model", 2, available_vram=None)
        self.assertIn("[WARN]", out)
        self.assertIn("2 workers", out)

    def test_oom_warning_when_workers_exceed_vram(self) -> None:
        # large-v3 needs 10 GiB; 3 workers = 30 GiB > 8 GiB available
        out = self._run("faster-whisper", "large-v3", 3, available_vram=8.0)
        self.assertIn("[WARN]", out)
        self.assertIn("OOM", out)

    def test_no_oom_warning_when_workers_fit_in_vram(self) -> None:
        # tiny needs 1 GiB; 2 workers = 2 GiB < 16 GiB available → no VRAM OOM warn
        # (may still emit generic multi-worker warning; we just check no OOM msg)
        out = self._run("faster-whisper", "tiny", 2, available_vram=16.0)
        self.assertNotIn("OOM", out)


class TestRemoteCommandSecretExclusion(unittest.TestCase):
    """_build_remote_command must not include SECRET_ENV_KEYS in the shell env prefix.

    API keys passed via CLI args (--translation-api-key etc.) must NOT also appear
    as inline shell variable assignments where they are visible in process listings.
    """

    def _make_run_mixin_with_env(
        self,
        run_env: dict[str, str],
    ):
        """Return a minimal RunMixin-like object with a patched _build_run_env."""
        from vid_to_sub_app.tui.mixins.run_mixin import RunMixin
        from vid_to_sub_app.tui.models import RemoteResourceProfile

        class _Stub(RunMixin):
            # Satisfy the minimum mixin contract for _build_cli_args + _build_remote_command
            _selected_paths: list[str] = []
            _run_last_shell: str = ""
            _run_shell_collapsed: bool = False
            _detected_ggml_models: dict[str, str] = {}

            def _val(self, wid: str) -> str:
                return ""

            def _sel(self, wid: str, fallback: str = "") -> str:
                return fallback

            def _chk(self, wid: str) -> bool:
                return False

            def _sw(self, wid: str) -> bool:
                return False

            def _resolved_wcpp_model_path(self) -> str:
                return ""

            def _build_run_env(self, config=None) -> dict[str, str]:
                return dict(run_env)

            def _refresh_live_panels(self) -> None:
                pass

            def _log(self, text: str) -> None:
                pass

        return _Stub()

    def _get_remote_cmd_str(
        self,
        run_env: dict[str, str],
    ) -> str:
        from vid_to_sub_app.tui.models import RemoteResourceProfile

        stub = self._make_run_mixin_with_env(run_env)
        profile = RemoteResourceProfile(
            name="test-remote",
            ssh_target="user@host",
            remote_workdir="/srv/vid_to_sub",
            slots=1,
            path_map={},
            env={},
            python_bin="python3",
            script_path="vid_to_sub.py",
        )
        parts = stub._build_remote_command(profile, None, dry_run=False)
        # The last element is the full remote shell command string
        return parts[-1]

    def test_translation_api_key_not_in_env_prefix(self) -> None:
        from vid_to_sub_app.shared.constants import ENV_TRANSLATION_API_KEY
        run_env = {
            "VID_TO_SUB_TRANSLATION_BASE_URL": "https://api.example.com/v1",
            ENV_TRANSLATION_API_KEY: "super-secret-key",
        }
        cmd_str = self._get_remote_cmd_str(run_env)
        # The secret MUST NOT appear as an inline env var assignment
        self.assertNotIn("super-secret-key", cmd_str.split("python3")[0],
            "API key leaked into shell env prefix (visible in ps / shell history)")

    def test_non_secret_vars_remain_in_env_prefix(self) -> None:
        run_env = {
            "VID_TO_SUB_TRANSLATION_BASE_URL": "https://api.example.com/v1",
            "VID_TO_SUB_TRANSLATION_API_KEY": "secret",
        }
        cmd_str = self._get_remote_cmd_str(run_env)
        self.assertIn("VID_TO_SUB_TRANSLATION_BASE_URL", cmd_str)

    def test_postprocess_api_key_not_in_env_prefix(self) -> None:
        from vid_to_sub_app.shared.constants import ENV_POSTPROCESS_API_KEY
        run_env = {
            "VID_TO_SUB_POSTPROCESS_BASE_URL": "https://post.example.com/v1",
            ENV_POSTPROCESS_API_KEY: "post-secret-key",
        }
        cmd_str = self._get_remote_cmd_str(run_env)
        self.assertNotIn("post-secret-key", cmd_str.split("python3")[0])

    def test_agent_api_key_not_in_env_prefix(self) -> None:
        from vid_to_sub_app.shared.constants import ENV_AGENT_API_KEY
        run_env = {
            "VID_TO_SUB_AGENT_BASE_URL": "https://agent.example.com/v1",
            ENV_AGENT_API_KEY: "agent-secret-key",
        }
        cmd_str = self._get_remote_cmd_str(run_env)
        self.assertNotIn("agent-secret-key", cmd_str.split("python3")[0])


class TestRemoteArtifactProvenance(unittest.TestCase):
    """_materialize_remote_stage1_artifact must preserve remote provenance and
    emit a warning when local fingerprint differs from remote.
    """

    def _make_stub_mixin(self, log_sink: list):
        from vid_to_sub_app.tui.mixins.run_mixin import RunMixin

        _captured = log_sink

        class _Stub(RunMixin):
            _selected_paths: list = []
            _run_last_shell: str = ""
            _run_shell_collapsed: bool = False
            _detected_ggml_models: dict = {}
            _run_output_dir: str = ""

            def _val(self, wid):
                return ""

            def _sel(self, wid, fallback=""):
                return fallback

            def _chk(self, wid):
                return False

            def _sw(self, wid):
                return False

            def _resolved_wcpp_model_path(self):
                return ""

            def _build_run_env(self, config=None):
                return {}

            def _refresh_live_panels(self):
                pass

            def _log(self, text):
                _captured.append(text)

            def call_from_thread(self, fn, *args, **kwargs):
                fn(*args, **kwargs)

        stub = _Stub()
        stub._run_output_dir = ""
        return stub

    def _make_profile(self):
        from vid_to_sub_app.tui.models import RemoteResourceProfile
        return RemoteResourceProfile(
            name="test-box",
            ssh_target="user@host",
            remote_workdir="/srv/vid_to_sub",
            slots=1,
            path_map={},
            env={},
            python_bin="python3",
            script_path="vid_to_sub.py",
        )

    def _make_artifact_json(self, source_path, source_fingerprint):
        import json
        return json.dumps({
            "schema_version": "1",
            "source_path": source_path,
            "output_base": "/remote/out",
            "source_fingerprint": source_fingerprint,
            "backend": "faster-whisper",
            "device": "cuda",
            "model": "large-v3",
            "content_type": None,
            "language": "en",
            "language_probability": 0.99,
            "duration": 60.0,
            "quality": {},
            "target_lang": None,
            "formats": ["srt"],
            "primary_outputs": [],
            "segments": [],
            "stage_status": {"transcription_complete": True},
        })

    def test_provenance_fields_saved_in_quality(self):
        """After remap, quality must contain remote_source_path + remote_source_fingerprint."""
        import tempfile, json, subprocess as _sp
        from pathlib import Path

        log = []
        stub = self._make_stub_mixin(log)
        profile = self._make_profile()

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "movie.mp4"
            source.write_bytes(b"fake-video-content")

            remote_fp = "9999:1700000000"
            artifact_json = self._make_artifact_json("/remote/media/movie.mp4", remote_fp)

            def _fake_scp(cmd, *, check, stdout, stderr, text):
                Path(cmd[-1]).write_text(artifact_json, encoding="utf-8")
                return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

            with unittest.mock.patch("subprocess.run", side_effect=_fake_scp):
                stub._run_output_dir = ""
                result = stub._materialize_remote_stage1_artifact(profile, str(source))

            self.assertIsNotNone(result, "Materialization should succeed")
            data = json.loads(Path(result).read_text())
            quality = data["quality"]
            self.assertEqual(quality["remote_source_path"], "/remote/media/movie.mp4")
            self.assertEqual(quality["remote_source_fingerprint"], remote_fp)
            self.assertIn("local_source_fingerprint", quality)
            self.assertTrue(quality["artifact_fetched_from_remote"])

    def test_provenance_mismatch_emits_warning(self):
        """When remote and local fingerprints differ, a yellow warning must be logged."""
        import tempfile, subprocess as _sp
        from pathlib import Path

        log = []
        stub = self._make_stub_mixin(log)
        profile = self._make_profile()

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "movie.mp4"
            source.write_bytes(b"different-local-content")

            remote_fp = "99999:9999999999"
            artifact_json = self._make_artifact_json("/remote/media/movie.mp4", remote_fp)

            def _fake_scp(cmd, *, check, stdout, stderr, text):
                Path(cmd[-1]).write_text(artifact_json, encoding="utf-8")
                return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

            with unittest.mock.patch("subprocess.run", side_effect=_fake_scp):
                stub._materialize_remote_stage1_artifact(profile, str(source))

            all_logs = " ".join(log)
            self.assertIn("remote provenance mismatch", all_logs.lower())
            self.assertIn("test-box", all_logs)

    def test_matching_fingerprints_no_mismatch_warning(self):
        """When fingerprints match (same file), no mismatch warning is emitted."""
        import tempfile, subprocess as _sp
        from pathlib import Path
        from vid_to_sub_app.cli.stage_artifact import fingerprint_source_path

        log = []
        stub = self._make_stub_mixin(log)
        profile = self._make_profile()

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "movie.mp4"
            source.write_bytes(b"real-video-bytes")
            local_fp = fingerprint_source_path(source)

            artifact_json = self._make_artifact_json("/remote/media/movie.mp4", local_fp)

            def _fake_scp(cmd, *, check, stdout, stderr, text):
                Path(cmd[-1]).write_text(artifact_json, encoding="utf-8")
                return _sp.CompletedProcess(cmd, 0, stdout="", stderr="")

            with unittest.mock.patch("subprocess.run", side_effect=_fake_scp):
                stub._materialize_remote_stage1_artifact(profile, str(source))

            all_logs = " ".join(log)
            self.assertNotIn("provenance mismatch", all_logs.lower())

    def test_scp_failure_returns_none_and_logs(self):
        """When scp fails, the method returns None and logs a failure message."""
        import tempfile, subprocess as _sp
        from pathlib import Path

        log = []
        stub = self._make_stub_mixin(log)
        profile = self._make_profile()

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "movie.mp4"
            source.write_bytes(b"video")

            def _failing_scp(cmd, *, check, stdout, stderr, text):
                return _sp.CompletedProcess(cmd, 1, stdout="connection refused")

            with unittest.mock.patch("subprocess.run", side_effect=_failing_scp):
                result = stub._materialize_remote_stage1_artifact(profile, str(source))

            self.assertIsNone(result)
            all_logs = " ".join(log)
            self.assertIn("failed to fetch", all_logs.lower())



class TestDiscoveryMetric(unittest.TestCase):
    """[METRIC] stage=discovery must be emitted to stderr after video discovery."""

    def _run_main(self, paths: list[str], extra_args: list[str] | None = None) -> str:
        """Run main() with the given paths and return stderr output."""
        import io
        from unittest.mock import patch
        from vid_to_sub_app.cli.main import main

        extra = extra_args or []
        argv = paths + extra
        buf = io.StringIO()
        with patch("sys.stderr", buf):
            try:
                main(argv)
            except SystemExit:
                pass
        return buf.getvalue()

    def test_discovery_metric_emitted_to_stderr(self) -> None:
        """After video discovery, [METRIC] stage=discovery must appear on stderr."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create one fake video so discovery finds something
            fake_video = Path(tmpdir) / "clip.mp4"
            fake_video.write_bytes(b"fake")

            # Patch the actual transcription to avoid running whisper
            with unittest.mock.patch(
                "vid_to_sub_app.cli.main.run_parallel",
                return_value=(1, 0),
            ), unittest.mock.patch(
                "vid_to_sub_app.cli.manifest.persist_folder_manifest_state",
            ):
                stderr_output = self._run_main([tmpdir])

        self.assertIn(
            "[METRIC] stage=discovery",
            stderr_output,
            "Expected [METRIC] stage=discovery in stderr after video discovery",
        )
        self.assertIn(
            "files_found=",
            stderr_output,
            "Expected files_found= in discovery metric",
        )
        self.assertIn(
            "elapsed_ms=",
            stderr_output,
            "Expected elapsed_ms= in discovery metric",
        )

    def test_discovery_metric_not_emitted_for_manifest_stdin(self) -> None:
        """When --manifest-stdin is used, no discovery occurs so no discovery metric."""
        import io
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from vid_to_sub_app.cli.main import main

        with tempfile.TemporaryDirectory() as tmpdir:
            fake_video = Path(tmpdir) / "clip.mp4"
            fake_video.write_bytes(b"fake")
            manifest = {
                "version": 1,
                "entries": [{
                    "video_path": str(fake_video),
                    "folder_path": tmpdir,
                    "folder_hash": "abc123",
                }],
            }
            stdin_data = json.dumps(manifest)

            stderr_buf = io.StringIO()
            with patch("sys.stdin", io.StringIO(stdin_data)), \
                 patch("sys.stderr", stderr_buf), \
                 patch("vid_to_sub_app.cli.main.run_parallel", return_value=(1, 0)), \
                 patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"):
                try:
                    main(["--manifest-stdin"])
                except SystemExit:
                    pass

        stderr_output = stderr_buf.getvalue()
        self.assertNotIn(
            "stage=discovery",
            stderr_output,
            "No discovery metric expected when using --manifest-stdin",
        )


if __name__ == "__main__":
    unittest.main()
