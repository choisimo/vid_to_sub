from __future__ import annotations

import json
import io
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

from vid_to_sub_app.cli.main import main
from vid_to_sub_app.cli.translation import (
    postprocess_translated_segments_openai_compatible,
    translate_segments_openai_compatible,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class TranslationPipelineTests(unittest.TestCase):
    def test_translate_segments_openai_compatible_preserves_segment_boundaries(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello"},
            {"start": 1.5, "end": 3.0, "text": "World"},
        ]
        captured_payloads: list[dict] = []

        def fake_urlopen(request):
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return _FakeHTTPResponse(
                {"choices": [{"message": {"content": '["안녕", "세상"]'}}]}
            )

        with patch("vid_to_sub_app.cli.translation.urllib.request.urlopen", side_effect=fake_urlopen):
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
        self.assertIn("first-pass subtitle translation agent", captured_payloads[0]["messages"][0]["content"])

    def test_postprocess_uses_translation_config_as_fallback(self) -> None:
        captured_requests = []

        def fake_urlopen(request):
            captured_requests.append(request)
            return _FakeHTTPResponse(
                {"choices": [{"message": {"content": '["정제된 가사"]'}}]}
            )

        with patch("vid_to_sub_app.cli.translation.urllib.request.urlopen", side_effect=fake_urlopen):
            final_segments, info = postprocess_translated_segments_openai_compatible(
                source_segments=[{"start": 0.0, "end": 2.0, "text": "青い風"}],
                translated_segments=[{"start": 0.0, "end": 2.0, "text": "푸른 바람"}],
                target_language="ko",
                postprocess_mode="auto",
                postprocess_model=None,
                postprocess_base_url=None,
                postprocess_api_key=None,
                source_language="ja",
                translation_model="gpt-4.1-mini",
                translation_base_url="https://translation.example/v1",
                translation_api_key="translation-secret",
            )

        self.assertEqual(
            [{"start": 0.0, "end": 2.0, "text": "정제된 가사"}],
            final_segments,
        )
        self.assertEqual("auto", info["mode"])
        self.assertEqual(
            "https://translation.example/v1/chat/completions",
            captured_requests[0].full_url,
        )
        payload = json.loads(captured_requests[0].data.decode("utf-8"))
        self.assertIn("silently fall back to contextual correction", payload["messages"][0]["content"])

    def test_main_rejects_postprocess_without_translation_target(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["--manifest-stdin", "--postprocess-translation"])

        self.assertEqual(2, ctx.exception.code)


if __name__ == "__main__":
    _ = unittest.main()
