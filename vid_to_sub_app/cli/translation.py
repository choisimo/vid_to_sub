from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

import vid_to_sub_app
from vid_to_sub_app.shared.constants import (
    ENV_POSTPROCESS_API_KEY,
    ENV_POSTPROCESS_BASE_URL,
    ENV_POSTPROCESS_MODEL,
    ENV_TRANSLATION_API_KEY,
    ENV_TRANSLATION_BASE_URL,
    ENV_TRANSLATION_DEBUG,
    ENV_TRANSLATION_MODEL,
    EVENT_PREFIX,
    ROOT_DIR,
    TRANSLATION_HTTP_RETRY_ATTEMPTS,
    TRANSLATION_HTTP_RETRY_BACKOFF_SEC,
    TRANSLATION_HTTP_RETRYABLE_STATUS_CODES,
    TRANSLATION_HTTP_TIMEOUT_SEC,
    TRANSLATION_RETRY_SCHEDULE,
)


_BATCH_COUNTER = count(1)


def _translation_max_concurrency() -> int:
    raw_value = os.getenv("VID_TO_SUB_TRANSLATION_MAX_CONCURRENCY", "1").strip()
    try:
        return max(1, int(raw_value))
    except ValueError:
        return 1


_TRANSLATION_SEMAPHORE = threading.BoundedSemaphore(_translation_max_concurrency())


@dataclass(frozen=True)
class ProviderResponse:
    message: str
    response_id: str | None
    request_id: str | None
    finish_reason: str | None
    usage: Any
    http_status: int | None
    response_payload: Any
    raw_response_file: str | None


@dataclass(frozen=True)
class ParsedTranslationPayload:
    texts: list[str]
    parsed_count: int
    parsed_sample: list[Any]
    payload_kind: str


@dataclass
class StructuredOutputPolicy:
    enabled: bool = True
    disabled_reason: str | None = None


class TranslationPayloadError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        parsed_count: int | None = None,
        parsed_sample: list[Any] | None = None,
        missing_segment_numbers: list[int] | None = None,
        duplicate_segment_numbers: list[int] | None = None,
        unexpected_segment_numbers: list[int] | None = None,
    ) -> None:
        super().__init__(message)
        self.parsed_count = parsed_count
        self.parsed_sample = parsed_sample or []
        self.missing_segment_numbers = missing_segment_numbers or []
        self.duplicate_segment_numbers = duplicate_segment_numbers or []
        self.unexpected_segment_numbers = unexpected_segment_numbers or []


class TranslationContractError(RuntimeError):
    pass


class RetryableProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class StructuredOutputFallbackRequired(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def extract_json_array(text: str) -> list[str]:
    payload = _extract_json_payload(text)
    if not isinstance(payload, list) or not all(
        isinstance(item, str) for item in payload
    ):
        raise ValueError("Model response is not a JSON string array")
    return payload


def _extract_json_payload(text: str) -> Any:
    candidates: list[tuple[int, str, str]] = []
    array_start = text.find("[")
    if array_start != -1:
        candidates.append((array_start, "[", "]"))
    object_start = text.find("{")
    if object_start != -1:
        candidates.append((object_start, "{", "}"))

    if not candidates:
        raise ValueError("No JSON payload found in model response")

    start, _open_char, close_char = min(candidates, key=lambda item: item[0])
    end = text.rfind(close_char)
    if end == -1 or end < start:
        raise ValueError("No JSON payload found in model response")
    return json.loads(text[start : end + 1])


def _structured_output_response_format(stage: str) -> dict[str, Any]:
    schema_name = (
        "subtitle_postprocess_batch"
        if stage == "postprocess"
        else "subtitle_translation_batch"
    )
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "segment_number": {"type": "integer"},
                                "text": {"type": "string"},
                            },
                            "required": ["segment_number", "text"],
                        },
                    }
                },
                "required": ["items"],
            },
        },
    }


def _build_chat_completion_payload(
    *,
    model: str,
    system_prompt: str,
    user_payload: dict[str, Any],
    stage: str,
    structured_output_enabled: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
    }
    if structured_output_enabled:
        payload["response_format"] = _structured_output_response_format(stage)
    return payload


def _should_fallback_structured_outputs(
    *, status_code: int | None, body: str, payload: dict[str, Any]
) -> bool:
    if "response_format" not in payload or status_code != 400:
        return False

    normalized_body = body.lower()
    malformed_schema_terms = (
        "invalid_schema",
        "schema_validation_error",
        "failed to validate schema",
        "invalid json schema",
        "json schema is invalid",
        "response_format schema",
    )
    if any(term in normalized_body for term in malformed_schema_terms):
        return False

    capability_terms = (
        "response_format",
        "json_schema",
        "json schema",
        "structured output",
        "structured outputs",
    )
    support_boundary_terms = (
        "unsupported",
        "not supported",
        "unsupported_parameter",
        "does not support",
        "only supported",
        "not available",
        "not enabled",
        "unsupported value",
    )
    return any(term in normalized_body for term in capability_terms) and any(
        term in normalized_body for term in support_boundary_terms
    )


def _resolve_chat_config(
    *,
    label: str,
    model_arg: Optional[str],
    base_url_arg: Optional[str],
    api_key_arg: Optional[str],
    model_env: str,
    base_url_env: str,
    api_key_env: str,
    fallback_model: str | None = None,
    fallback_base_url: str | None = None,
    fallback_api_key: str | None = None,
) -> tuple[str, str, str]:
    model = model_arg or os.getenv(model_env) or fallback_model
    base_url = base_url_arg or os.getenv(base_url_env) or fallback_base_url
    api_key = api_key_arg or os.getenv(api_key_env) or fallback_api_key

    if not model:
        raise RuntimeError(
            f"{label} model is not configured. Set the matching CLI option, "
            f"{model_env}, or a fallback translation model."
        )
    if not base_url:
        raise RuntimeError(
            f"{label} base URL is not configured. Set the matching CLI option, "
            f"{base_url_env}, or a fallback translation base URL."
        )
    if not api_key:
        raise RuntimeError(
            f"{label} API key is not configured. Set the matching CLI option, "
            f"{api_key_env}, or a fallback translation API key."
        )

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/chat/completions"
    return model, endpoint, api_key


def _request_chat_completion(
    *,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    error_prefix: str,
    debug_enabled: bool,
    batch_id: str,
    attempt: int,
    stage: str,
    payload_chars: int,
) -> ProviderResponse:
    payload_json = json.dumps(payload, ensure_ascii=False)
    request = urllib.request.Request(
        endpoint,
        data=payload_json.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "vid_to_sub/1.0 (+https://local.cli)",
            "X-Client-Request-Id": batch_id,
        },
    )
    response_payload: Any = None
    http_status: int | None = None
    headers: Any = None
    max_http_attempts = max(1, int(TRANSLATION_HTTP_RETRY_ATTEMPTS))
    with _TRANSLATION_SEMAPHORE:
        for http_attempt in range(1, max_http_attempts + 1):
            try:
                with _urlopen_with_timeout(
                    request, TRANSLATION_HTTP_TIMEOUT_SEC
                ) as response:
                    raw_text = response.read().decode("utf-8")
                    response_payload = json.loads(raw_text)
                    http_status = getattr(response, "status", None)
                    headers = getattr(response, "headers", None)
                    break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if _should_fallback_structured_outputs(
                    status_code=exc.code, body=body, payload=payload
                ):
                    raise StructuredOutputFallbackRequired(
                        f"{error_prefix} structured outputs fallback: HTTP {exc.code}: {body}",
                        status_code=exc.code,
                    ) from exc
                if _should_retry_http_request(
                    status_code=exc.code, http_attempt=http_attempt
                ):
                    _emit_http_retry_log(
                        batch_id=batch_id,
                        attempt=attempt,
                        http_attempt=http_attempt,
                        endpoint=endpoint,
                        reason=f"http_{exc.code}",
                        status_code=exc.code,
                        retry_in_sec=_retry_backoff_delay(http_attempt),
                        stage=stage,
                        payload_chars=payload_chars,
                    )
                    time.sleep(_retry_backoff_delay(http_attempt))
                    continue
                if exc.code in TRANSLATION_HTTP_RETRYABLE_STATUS_CODES:
                    raise RetryableProviderError(
                        f"{error_prefix} HTTP {exc.code}: {body}",
                        status_code=exc.code,
                    ) from exc
                raise RuntimeError(f"{error_prefix} HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
                reason = getattr(exc, "reason", exc)
                if _should_retry_http_request(
                    status_code=None, http_attempt=http_attempt
                ):
                    _emit_http_retry_log(
                        batch_id=batch_id,
                        attempt=attempt,
                        http_attempt=http_attempt,
                        endpoint=endpoint,
                        reason=str(reason),
                        status_code=None,
                        retry_in_sec=_retry_backoff_delay(http_attempt),
                        stage=stage,
                        payload_chars=payload_chars,
                    )
                    time.sleep(_retry_backoff_delay(http_attempt))
                    continue
                raise RetryableProviderError(
                    f"{error_prefix} request failed: {reason}"
                ) from exc

    if response_payload is None:
        raise RuntimeError(f"{error_prefix} request failed without a response payload")

    raw_response_file = _write_debug_response_file(
        enabled=debug_enabled,
        batch_id=batch_id,
        attempt=attempt,
        payload=response_payload,
    )
    try:
        choice = response_payload["choices"][0]
        message = choice["message"]["content"]
        if not isinstance(message, str):
            raise TypeError("message content is not a string")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Could not parse {error_prefix.lower()} response envelope: {response_payload}"
        ) from exc

    request_id = None
    if headers is not None:
        request_id = (
            headers.get("x-request-id")
            or headers.get("request-id")
            or headers.get("openai-request-id")
        )

    return ProviderResponse(
        message=message,
        response_id=_optional_str(response_payload.get("id")),
        request_id=_optional_str(request_id),
        finish_reason=_optional_str(choice.get("finish_reason")),
        usage=response_payload.get("usage"),
        http_status=http_status,
        response_payload=response_payload,
        raw_response_file=raw_response_file,
    )


def _urlopen_with_timeout(
    request: urllib.request.Request,
    timeout_sec: float,
):
    try:
        return urllib.request.urlopen(request, timeout=timeout_sec)
    except TypeError as exc:
        if "timeout" not in str(exc):
            raise
        return urllib.request.urlopen(request)


def _retry_backoff_delay(http_attempt: int) -> float:
    delays = TRANSLATION_HTTP_RETRY_BACKOFF_SEC
    if not delays:
        return 0.0
    index = min(max(0, http_attempt - 1), len(delays) - 1)
    return float(delays[index])


def _should_retry_http_request(
    *,
    status_code: int | None,
    http_attempt: int,
) -> bool:
    if http_attempt >= max(1, int(TRANSLATION_HTTP_RETRY_ATTEMPTS)):
        return False
    if status_code is None:
        return True
    return status_code in TRANSLATION_HTTP_RETRYABLE_STATUS_CODES


def _emit_http_retry_log(
    *,
    batch_id: str,
    attempt: int,
    http_attempt: int,
    endpoint: str,
    reason: str,
    status_code: int | None,
    retry_in_sec: float,
    stage: str,
    payload_chars: int,
) -> None:
    _emit_batch_log(
        {
            "event": "translation_http_retry",
            "batch_id": batch_id,
            "attempt": attempt,
            "http_attempt": http_attempt,
            "endpoint": endpoint,
            "endpoint_host": urlsplit(endpoint).netloc or endpoint,
            "http_status": status_code,
            "reason": reason,
            "retry_in_sec": retry_in_sec,
            "stage": stage,
            "payload_chars": payload_chars,
            "timeout_sec": TRANSLATION_HTTP_TIMEOUT_SEC,
        }
    )


def _parse_translation_payload(
    *,
    message: str,
    expected_segment_numbers: list[int],
) -> ParsedTranslationPayload:
    payload = _extract_json_payload(message)
    if isinstance(payload, dict):
        payload = payload.get("items")
        if not isinstance(payload, list):
            raise TranslationPayloadError("model_response_object_missing_items_array")
    if not isinstance(payload, list):
        raise TranslationPayloadError("model_response_is_not_a_json_array")

    sample = payload[: min(3, len(payload))]
    if all(isinstance(item, str) for item in payload):
        if len(payload) != len(expected_segment_numbers):
            raise TranslationPayloadError(
                "legacy_string_array_count_mismatch",
                parsed_count=len(payload),
                parsed_sample=sample,
            )
        return ParsedTranslationPayload(
            texts=[item.strip() for item in payload],
            parsed_count=len(payload),
            parsed_sample=sample,
            payload_kind="legacy_string_array",
        )

    numbers_to_text: dict[int, str] = {}
    duplicate_numbers: list[int] = []
    unexpected_numbers: list[int] = []
    expected_number_set = set(expected_segment_numbers)

    for item in payload:
        if not isinstance(item, dict):
            raise TranslationPayloadError(
                "model_response_items_must_be_objects_or_strings",
                parsed_count=len(payload),
                parsed_sample=sample,
            )
        segment_number = item.get("segment_number")
        text = item.get("text")
        if not isinstance(segment_number, int):
            raise TranslationPayloadError(
                "segment_number_must_be_int",
                parsed_count=len(payload),
                parsed_sample=sample,
            )
        if not isinstance(text, str):
            raise TranslationPayloadError(
                "text_must_be_string",
                parsed_count=len(payload),
                parsed_sample=sample,
            )
        if segment_number in numbers_to_text:
            duplicate_numbers.append(segment_number)
            continue
        if segment_number not in expected_number_set:
            unexpected_numbers.append(segment_number)
            continue
        numbers_to_text[segment_number] = text.strip()

    missing_numbers = [
        segment_number
        for segment_number in expected_segment_numbers
        if segment_number not in numbers_to_text
    ]
    if missing_numbers or duplicate_numbers or unexpected_numbers:
        raise TranslationPayloadError(
            "segment_number_contract_mismatch",
            parsed_count=len(payload),
            parsed_sample=sample,
            missing_segment_numbers=missing_numbers,
            duplicate_segment_numbers=sorted(set(duplicate_numbers)),
            unexpected_segment_numbers=sorted(set(unexpected_numbers)),
        )

    return ParsedTranslationPayload(
        texts=[
            numbers_to_text[segment_number]
            for segment_number in expected_segment_numbers
        ],
        parsed_count=len(payload),
        parsed_sample=sample,
        payload_kind="segment_object_array",
    )


def _debug_response_persistence_enabled(name: str) -> bool:
    return os.getenv(name, "").strip() == "1"


def _make_batch_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")[:-3]
    slug = "pp" if prefix.lower().startswith("postprocess") else "tr"
    return f"{slug}-{stamp}-{next(_BATCH_COUNTER):06d}"


def _write_debug_response_file(
    *,
    enabled: bool,
    batch_id: str,
    attempt: int,
    payload: Any,
) -> str | None:
    if not enabled:
        return None
    debug_dir = Path(tempfile.gettempdir()) / "vid_to_sub" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    raw_response_path = debug_dir / f"{batch_id}.attempt-{attempt}.raw.json"
    raw_response_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(raw_response_path)


def _emit_batch_log(payload: dict[str, Any]) -> None:
    print(
        f"{EVENT_PREFIX} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value or None


def _build_execution_fingerprint(
    *,
    endpoint: str,
    model: str,
    chunk_size: int,
    stage: str,
) -> dict[str, Any]:
    package_path = getattr(vid_to_sub_app, "__file__", None)
    return {
        "event": "translation_execution_fingerprint",
        "stage": stage,
        "exec_path": str(Path(sys.argv[0]).resolve()) if sys.argv else None,
        "cwd": str(Path.cwd()),
        "python": sys.executable,
        "argv": list(sys.argv),
        "hostname": socket.gethostname(),
        "repo_root": str(ROOT_DIR),
        "git_sha": _git_output("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "package_path": str(Path(package_path).resolve()) if package_path else None,
        "translation_chunk_size": chunk_size,
        "endpoint": endpoint,
        "endpoint_host": urlsplit(endpoint).netloc or endpoint,
        "model": model,
        "distributed": "--manifest-stdin" in sys.argv,
    }


def _emit_execution_fingerprint(
    *,
    endpoint: str,
    model: str,
    chunk_size: int,
    stage: str,
) -> dict[str, Any]:
    payload = _build_execution_fingerprint(
        endpoint=endpoint,
        model=model,
        chunk_size=chunk_size,
        stage=stage,
    )
    print(
        f"{EVENT_PREFIX} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )
    return payload


def _format_contract_error(
    *,
    error_prefix: str,
    batch_id: str,
    attempt: int,
    batch_segment_numbers: list[int],
    requested_count: int,
    configured_chunk_size: int,
    endpoint: str,
    model: str,
    provider_response: ProviderResponse | None,
    payload_error: Exception,
    fingerprint: dict[str, Any],
) -> TranslationContractError:
    parsed_count = None
    parsed_sample: list[Any] = []
    missing_numbers: list[int] = []
    duplicate_numbers: list[int] = []
    unexpected_numbers: list[int] = []
    if isinstance(payload_error, TranslationPayloadError):
        parsed_count = payload_error.parsed_count
        parsed_sample = payload_error.parsed_sample
        missing_numbers = payload_error.missing_segment_numbers
        duplicate_numbers = payload_error.duplicate_segment_numbers
        unexpected_numbers = payload_error.unexpected_segment_numbers

    lines = [f"{error_prefix.replace(' API', '')}ContractError:"]
    lines.extend(
        [
            f"  batch_id={batch_id}",
            f"  attempt={attempt}",
            f"  batch_range={batch_segment_numbers[0]}-{batch_segment_numbers[-1]}",
            f"  segment_numbers={batch_segment_numbers}",
            f"  requested_count={requested_count}",
            f"  parsed_count={parsed_count}",
            f"  configured_chunk_size={configured_chunk_size}",
            f"  exec_path={fingerprint.get('exec_path')}",
            f"  cwd={fingerprint.get('cwd')}",
            f"  python={fingerprint.get('python')}",
            f"  argv={fingerprint.get('argv')}",
            f"  repo_root={fingerprint.get('repo_root')}",
            f"  git_sha={fingerprint.get('git_sha')}",
            f"  git_dirty={fingerprint.get('git_dirty')}",
            f"  package_path={fingerprint.get('package_path')}",
            f"  host={fingerprint.get('hostname')}",
            f"  translation_chunk_size={fingerprint.get('translation_chunk_size')}",
            f"  distributed={fingerprint.get('distributed')}",
            f"  endpoint={endpoint}",
            f"  endpoint_host={urlsplit(endpoint).netloc or endpoint}",
            f"  model={model}",
            f"  finish_reason={provider_response.finish_reason if provider_response else None}",
            f"  http_status={provider_response.http_status if provider_response else None}",
            f"  response_id={provider_response.response_id if provider_response else None}",
            f"  request_id={provider_response.request_id if provider_response else None}",
            f"  usage={provider_response.usage if provider_response else None}",
            f"  missing_segment_numbers={missing_numbers}",
            f"  duplicate_segment_numbers={duplicate_numbers}",
            f"  unexpected_segment_numbers={unexpected_numbers}",
            f"  parsed_sample={parsed_sample}",
            f"  raw_response_file={provider_response.raw_response_file if provider_response else None}",
            f"  reason={payload_error}",
        ]
    )
    return TranslationContractError("\n".join(lines))


def _build_retry_schedule(chunk_size: int) -> list[int]:
    schedule = [size for size in TRANSLATION_RETRY_SCHEDULE if size <= chunk_size]
    schedule.append(chunk_size)
    unique = sorted(set(max(1, int(size)) for size in schedule), reverse=True)
    if unique[-1] != 1:
        unique.append(1)
    return unique


def _next_retry_size(batch_len: int, schedule: list[int]) -> int | None:
    for size in schedule:
        if size < batch_len:
            return size
    return None


def _estimate_item_chars(item: dict[str, Any], *, stage: str, text_key: str) -> int:
    if stage == "postprocess":
        return len(str(item.get("source", ""))) + len(str(item.get("draft", ""))) + 96
    return len(str(item.get(text_key, ""))) + 48


def _iter_budgeted_batches(
    items: list[dict[str, Any]],
    *,
    stage: str,
    text_key: str,
    max_items: int,
    max_payload_chars: int,
):
    batch: list[dict[str, Any]] = []
    total_chars = 0

    for item in items:
        item_chars = _estimate_item_chars(item, stage=stage, text_key=text_key)
        if batch and (
            len(batch) >= max_items or total_chars + item_chars > max_payload_chars
        ):
            yield batch
            batch = []
            total_chars = 0

        batch.append(item)
        total_chars += item_chars

    if batch:
        yield batch


def _translate_batch(
    *,
    batch: list[dict[str, Any]],
    model: str,
    endpoint: str,
    api_key: str,
    system_prompt: str,
    build_user_payload: Callable[[list[dict[str, Any]], int], dict[str, Any]],
    error_prefix: str,
    configured_chunk_size: int,
    retry_schedule: list[int],
    translation_mode: str,
    debug_enabled: bool,
    fingerprint: dict[str, Any],
    stage: str,
    attempt: int,
    requested_count: int | None = None,
    blank_or_whitespace_count: int = 0,
    on_batch_success: Callable[[list[int], list[str]], None] | None = None,
    structured_output_policy: StructuredOutputPolicy | None = None,
) -> list[str]:
    batch_id = _make_batch_id(error_prefix)
    segment_numbers = [int(item["segment_number"]) for item in batch]
    user_payload = build_user_payload(batch, segment_numbers[0] - 1)
    policy = structured_output_policy or StructuredOutputPolicy()
    payload = _build_chat_completion_payload(
        model=model,
        system_prompt=system_prompt,
        user_payload=user_payload,
        stage=stage,
        structured_output_enabled=policy.enabled,
    )
    payload_chars = len(json.dumps(payload, ensure_ascii=False))
    actual_batch_size = len(batch)
    batch_requested_count = (
        actual_batch_size
        if requested_count is None
        else max(actual_batch_size, requested_count)
    )
    provider_response: ProviderResponse | None = None
    try:
        try:
            provider_response = _request_chat_completion(
                endpoint=endpoint,
                api_key=api_key,
                payload=payload,
                error_prefix=error_prefix,
                debug_enabled=debug_enabled,
                batch_id=batch_id,
                attempt=attempt,
                stage=stage,
                payload_chars=payload_chars,
            )
        except StructuredOutputFallbackRequired as exc:
            policy.enabled = False
            policy.disabled_reason = str(exc)
            _emit_batch_log(
                {
                    "event": "translation_structured_output_fallback",
                    "batch_id": batch_id,
                    "attempt": attempt,
                    "stage": stage,
                    "batch_start_idx": segment_numbers[0],
                    "batch_end_idx": segment_numbers[-1],
                    "segment_numbers": segment_numbers,
                    "requested_count": batch_requested_count,
                    "blank_or_whitespace_count": blank_or_whitespace_count,
                    "effective_sent_count": actual_batch_size,
                    "chunk_size": actual_batch_size,
                    "configured_chunk_size": configured_chunk_size,
                    "endpoint": endpoint,
                    "endpoint_host": urlsplit(endpoint).netloc or endpoint,
                    "model": model,
                    "http_status": exc.status_code,
                    "reason": str(exc),
                }
            )
            payload = _build_chat_completion_payload(
                model=model,
                system_prompt=system_prompt,
                user_payload=user_payload,
                stage=stage,
                structured_output_enabled=False,
            )
            payload_chars = len(json.dumps(payload, ensure_ascii=False))
            provider_response = _request_chat_completion(
                endpoint=endpoint,
                api_key=api_key,
                payload=payload,
                error_prefix=error_prefix,
                debug_enabled=debug_enabled,
                batch_id=batch_id,
                attempt=attempt,
                stage=stage,
                payload_chars=payload_chars,
            )
        parsed = _parse_translation_payload(
            message=provider_response.message,
            expected_segment_numbers=segment_numbers,
        )
        if provider_response.finish_reason == "length":
            raise TranslationPayloadError(
                "finish_reason_length",
                parsed_count=parsed.parsed_count,
                parsed_sample=parsed.parsed_sample,
            )
    except RetryableProviderError as exc:
        next_size = _next_retry_size(len(batch), retry_schedule)
        _emit_batch_log(
            {
                "event": "translation_batch_transport_error",
                "batch_id": batch_id,
                "attempt": attempt,
                "stage": stage,
                "batch_start_idx": segment_numbers[0],
                "batch_end_idx": segment_numbers[-1],
                "segment_numbers": segment_numbers,
                "requested_count": batch_requested_count,
                "blank_or_whitespace_count": blank_or_whitespace_count,
                "effective_sent_count": actual_batch_size,
                "chunk_size": actual_batch_size,
                "configured_chunk_size": configured_chunk_size,
                "payload_chars": payload_chars,
                "endpoint": endpoint,
                "endpoint_host": urlsplit(endpoint).netloc or endpoint,
                "model": model,
                "http_status": exc.status_code,
                "reason": str(exc),
                "next_retry_size": next_size,
            }
        )
        if next_size is not None:
            outputs: list[str] = []
            for start_idx in range(0, len(batch), next_size):
                sub_batch = batch[start_idx : start_idx + next_size]
                outputs.extend(
                    _translate_batch(
                        batch=sub_batch,
                        model=model,
                        endpoint=endpoint,
                        api_key=api_key,
                        system_prompt=system_prompt,
                        build_user_payload=build_user_payload,
                        error_prefix=error_prefix,
                        configured_chunk_size=configured_chunk_size,
                        retry_schedule=retry_schedule,
                        translation_mode=translation_mode,
                        debug_enabled=debug_enabled,
                        fingerprint=fingerprint,
                        stage=stage,
                        attempt=attempt + 1,
                        requested_count=batch_requested_count,
                        blank_or_whitespace_count=blank_or_whitespace_count,
                        on_batch_success=on_batch_success,
                        structured_output_policy=policy,
                    )
                )
            return outputs
        raise
    except (ValueError, TranslationPayloadError) as exc:
        contract_error = _format_contract_error(
            error_prefix=error_prefix,
            batch_id=batch_id,
            attempt=attempt,
            batch_segment_numbers=segment_numbers,
            requested_count=batch_requested_count,
            configured_chunk_size=configured_chunk_size,
            endpoint=endpoint,
            model=model,
            provider_response=provider_response,
            payload_error=exc,
            fingerprint=fingerprint,
        )
        next_size = _next_retry_size(len(batch), retry_schedule)
        _emit_batch_log(
            {
                "event": "translation_batch_contract_error",
                "batch_id": batch_id,
                "attempt": attempt,
                "batch_start_idx": segment_numbers[0],
                "batch_end_idx": segment_numbers[-1],
                "segment_numbers": segment_numbers,
                "requested_count": batch_requested_count,
                "blank_or_whitespace_count": blank_or_whitespace_count,
                "effective_sent_count": actual_batch_size,
                "chunk_size": actual_batch_size,
                "configured_chunk_size": configured_chunk_size,
                "stage": stage,
                "payload_chars": payload_chars,
                "endpoint": endpoint,
                "endpoint_host": urlsplit(endpoint).netloc or endpoint,
                "model": model,
                "temperature": 0,
                "max_output_tokens": None,
                "http_status": provider_response.http_status
                if provider_response
                else None,
                "response_id": provider_response.response_id
                if provider_response
                else None,
                "request_id": provider_response.request_id
                if provider_response
                else None,
                "finish_reason": provider_response.finish_reason
                if provider_response
                else None,
                "usage": provider_response.usage if provider_response else None,
                "raw_response_file": provider_response.raw_response_file
                if provider_response
                else None,
                "parsed_count": getattr(exc, "parsed_count", None),
                "parsed_sample": getattr(exc, "parsed_sample", []),
                "reason": str(exc),
                "next_retry_size": next_size,
            }
        )
        if next_size is not None:
            outputs: list[str] = []
            for start_idx in range(0, len(batch), next_size):
                sub_batch = batch[start_idx : start_idx + next_size]
                outputs.extend(
                    _translate_batch(
                        batch=sub_batch,
                        model=model,
                        endpoint=endpoint,
                        api_key=api_key,
                        system_prompt=system_prompt,
                        build_user_payload=build_user_payload,
                        error_prefix=error_prefix,
                        configured_chunk_size=configured_chunk_size,
                        retry_schedule=retry_schedule,
                        translation_mode=translation_mode,
                        debug_enabled=debug_enabled,
                        fingerprint=fingerprint,
                        stage=stage,
                        attempt=attempt + 1,
                        requested_count=batch_requested_count,
                        blank_or_whitespace_count=blank_or_whitespace_count,
                        on_batch_success=on_batch_success,
                        structured_output_policy=policy,
                    )
                )
            return outputs
        if translation_mode == "best-effort" and len(batch) == 1:
            fallback_text = str(batch[0]["fallback_text"])
            print(
                f"[WARN] {error_prefix} fell back to source text for segment {segment_numbers[0]} after contract failure.",
                file=sys.stderr,
                flush=True,
            )
            return [fallback_text]
        raise contract_error from exc

    _emit_batch_log(
        {
            "event": "translation_batch_result",
            "batch_id": batch_id,
            "attempt": attempt,
            "batch_start_idx": segment_numbers[0],
            "batch_end_idx": segment_numbers[-1],
            "segment_numbers": segment_numbers,
            "requested_count": batch_requested_count,
            "blank_or_whitespace_count": blank_or_whitespace_count,
            "effective_sent_count": actual_batch_size,
            "chunk_size": actual_batch_size,
            "configured_chunk_size": configured_chunk_size,
            "stage": stage,
            "payload_chars": payload_chars,
            "endpoint": endpoint,
            "endpoint_host": urlsplit(endpoint).netloc or endpoint,
            "model": model,
            "temperature": 0,
            "max_output_tokens": None,
            "http_status": provider_response.http_status,
            "response_id": provider_response.response_id,
            "request_id": provider_response.request_id,
            "finish_reason": provider_response.finish_reason,
            "usage": provider_response.usage,
            "raw_response_file": provider_response.raw_response_file,
            "parsed_count": parsed.parsed_count,
            "parsed_sample": parsed.parsed_sample,
            "payload_kind": parsed.payload_kind,
        }
    )
    if on_batch_success is not None:
        on_batch_success(segment_numbers, parsed.texts)
    return parsed.texts


def _run_subtitle_agent_batches(
    *,
    items: list[dict[str, Any]],
    text_key: str,
    model: str,
    endpoint: str,
    api_key: str,
    system_prompt: str,
    build_user_payload: Callable[[list[dict[str, Any]], int], dict[str, Any]],
    error_prefix: str,
    chunk_size: int,
    max_payload_chars: int,
    translation_mode: str,
    stage: str,
    on_batch_success: Callable[[list[int], list[str]], None] | None = None,
) -> list[str]:
    if not items:
        return []

    debug_enabled = _debug_response_persistence_enabled(ENV_TRANSLATION_DEBUG)
    structured_output_policy = StructuredOutputPolicy()
    fingerprint = _emit_execution_fingerprint(
        endpoint=endpoint,
        model=model,
        chunk_size=chunk_size,
        stage=stage,
    )
    retry_schedule = _build_retry_schedule(chunk_size)
    outputs: list[str] = []
    for batch in _iter_budgeted_batches(
        items,
        stage=stage,
        text_key=text_key,
        max_items=chunk_size,
        max_payload_chars=max_payload_chars,
    ):
        batch_translatable, batch_blank_values = _split_blank_items(batch, text_key)
        if not batch_translatable:
            continue
        outputs.extend(
            _translate_batch(
                batch=batch_translatable,
                model=model,
                endpoint=endpoint,
                api_key=api_key,
                system_prompt=system_prompt,
                build_user_payload=build_user_payload,
                error_prefix=error_prefix,
                configured_chunk_size=chunk_size,
                retry_schedule=retry_schedule,
                translation_mode=translation_mode,
                debug_enabled=debug_enabled,
                fingerprint=fingerprint,
                stage=stage,
                attempt=1,
                requested_count=len(batch),
                blank_or_whitespace_count=len(batch_blank_values),
                on_batch_success=on_batch_success,
                structured_output_policy=structured_output_policy,
            )
        )
    return outputs


def _split_blank_items(
    items: list[dict[str, Any]], text_key: str
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    translatable: list[dict[str, Any]] = []
    blank_values: dict[int, str] = {}
    for item in items:
        segment_number = int(item["segment_number"])
        text = str(item[text_key])
        if text.strip():
            translatable.append(item)
        else:
            blank_values[segment_number] = text
    return translatable, blank_values


def translate_segments_openai_compatible(
    segments: list[dict[str, Any]],
    target_language: str,
    translation_model: str | None,
    translation_base_url: str | None,
    translation_api_key: str | None,
    source_language: str | None,
    chunk_size: int = 100,
    max_payload_chars: int = 16000,
    translation_mode: str = "strict",
    resume_text_by_number: dict[int, str] | None = None,
    on_batch_success: Callable[[list[int], list[str]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, endpoint, api_key = _resolve_chat_config(
        label="Translation",
        model_arg=translation_model,
        base_url_arg=translation_base_url,
        api_key_arg=translation_api_key,
        model_env=ENV_TRANSLATION_MODEL,
        base_url_env=ENV_TRANSLATION_BASE_URL,
        api_key_env=ENV_TRANSLATION_API_KEY,
    )

    indexed_segments = [
        {
            "segment_number": index + 1,
            "text": str(segment["text"]),
            "fallback_text": str(segment["text"]),
        }
        for index, segment in enumerate(segments)
    ]
    translatable_items, blank_values = _split_blank_items(indexed_segments, "text")
    translated_by_number = dict(blank_values)
    resumed_count = 0
    if resume_text_by_number:
        for segment_number, text in resume_text_by_number.items():
            if 1 <= int(segment_number) <= len(segments):
                translated_by_number[int(segment_number)] = str(text)
        resumed_count = sum(
            1
            for item in translatable_items
            if item["segment_number"] in translated_by_number
        )

    remaining_items = [
        item
        for item in translatable_items
        if item["segment_number"] not in translated_by_number
    ]
    if remaining_items:
        translated_texts = _run_subtitle_agent_batches(
            items=remaining_items,
            text_key="text",
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            system_prompt=(
                "You are the first-pass subtitle translation agent. Return only a JSON array. "
                "Preferred contract: each array item is an object with keys segment_number and text. "
                "Legacy compatibility: a plain string array is temporarily accepted only when item count "
                "exactly matches the request. Do not merge or split subtitle lines. Preserve line breaks "
                "inside each item when possible. Preserve proper nouns when they should stay untranslated. "
                "When the source feels lyrical or poetic, keep the target text natural rather than "
                "mechanically literal."
            ),
            build_user_payload=lambda batch, start_idx: {
                "target_language": target_language,
                "source_language": source_language,
                "segment_numbers": [item["segment_number"] for item in batch],
                "subtitles": [item["text"] for item in batch],
                "response_contract": [
                    {
                        "segment_number": batch[0]["segment_number"],
                        "text": "translated subtitle text",
                    }
                ],
            },
            error_prefix="Translation API",
            chunk_size=chunk_size,
            max_payload_chars=max_payload_chars,
            translation_mode=translation_mode,
            stage="translate",
            on_batch_success=on_batch_success,
        )
        translated_by_number.update(
            {
                item["segment_number"]: translated_text
                for item, translated_text in zip(remaining_items, translated_texts)
            }
        )

    translated_segments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": translated_by_number[index + 1],
        }
        for index, segment in enumerate(segments)
    ]

    info = {
        "backend": "openai-compatible-translation",
        "model": model,
        "target_language": target_language,
        "source_language": source_language,
        "chunk_size": chunk_size,
        "max_payload_chars": max_payload_chars,
        "mode": translation_mode,
        "blank_segment_count": len(blank_values),
        "resumed_segment_count": resumed_count,
    }
    return translated_segments, info


def postprocess_translated_segments_openai_compatible(
    *,
    source_segments: list[dict[str, Any]],
    translated_segments: list[dict[str, Any]],
    target_language: str,
    postprocess_mode: str,
    postprocess_model: str | None,
    postprocess_base_url: str | None,
    postprocess_api_key: str | None,
    source_language: str | None,
    translation_model: str | None,
    translation_base_url: str | None,
    translation_api_key: str | None,
    chunk_size: int = 100,
    max_payload_chars: int = 12000,
    translation_mode: str = "strict",
    resume_text_by_number: dict[int, str] | None = None,
    on_batch_success: Callable[[list[int], list[str]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, endpoint, api_key = _resolve_chat_config(
        label="Postprocess",
        model_arg=postprocess_model,
        base_url_arg=postprocess_base_url,
        api_key_arg=postprocess_api_key,
        model_env=ENV_POSTPROCESS_MODEL,
        base_url_env=ENV_POSTPROCESS_BASE_URL,
        api_key_env=ENV_POSTPROCESS_API_KEY,
        fallback_model=translation_model or os.getenv(ENV_TRANSLATION_MODEL),
        fallback_base_url=translation_base_url or os.getenv(ENV_TRANSLATION_BASE_URL),
        fallback_api_key=translation_api_key or os.getenv(ENV_TRANSLATION_API_KEY),
    )

    if len(source_segments) != len(translated_segments):
        raise RuntimeError(
            "Postprocess requires the same number of source and translated segments."
        )

    mode_instruction = {
        "auto": (
            "If your runtime offers web search, MCP, retrieval, or similar tools, you may consult "
            "authoritative references when the lines look like lyrics, quotations, or OCR-noisy "
            "text. If such tools are unavailable, silently fall back to contextual correction and "
            "natural polishing."
        ),
        "web_lookup": (
            "Prefer authoritative external references for lyric-like or OCR-corrupted lines when "
            "your runtime offers web search, MCP, or retrieval tools. If those tools are "
            "unavailable, silently fall back to contextual correction and natural polishing."
        ),
        "context_polish": (
            "Do not rely on external lookup. Improve only from the provided source lines, draft "
            "translation, and surrounding context."
        ),
    }.get(postprocess_mode, "")

    items = [
        {
            "segment_number": index + 1,
            "source": str(source["text"]),
            "draft": str(translated["text"]),
            "fallback_text": str(translated["text"]),
            "start": source["start"],
            "end": source["end"],
        }
        for index, (source, translated) in enumerate(
            zip(source_segments, translated_segments)
        )
    ]
    translatable_items, blank_values = _split_blank_items(items, "draft")
    final_text_by_number = dict(blank_values)
    resumed_count = 0
    if resume_text_by_number:
        for segment_number, text in resume_text_by_number.items():
            if 1 <= int(segment_number) <= len(translated_segments):
                final_text_by_number[int(segment_number)] = str(text)
        resumed_count = sum(
            1
            for item in translatable_items
            if item["segment_number"] in final_text_by_number
        )

    remaining_items = [
        item
        for item in translatable_items
        if item["segment_number"] not in final_text_by_number
    ]
    if remaining_items:
        postprocessed_texts = _run_subtitle_agent_batches(
            items=remaining_items,
            text_key="draft",
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            system_prompt=(
                "You are the subtitle post-editing agent. Improve a first-pass translation while "
                "keeping the same item count and order. Return only a JSON array. Preferred contract: "
                "each array item is an object with keys segment_number and text. Legacy compatibility: "
                "a plain string array is temporarily accepted only when item count exactly matches the "
                "request. Do not merge or split subtitle items. Keep each line aligned with its source "
                "line, natural in the target language, and consistent in tone across the batch. "
                f"{mode_instruction}"
            ),
            build_user_payload=lambda batch, start_idx: {
                "task": "postprocess_subtitle_translation",
                "mode": postprocess_mode,
                "target_language": target_language,
                "source_language": source_language,
                "segment_numbers": [item["segment_number"] for item in batch],
                "segments": [
                    {
                        "source_text": item["source"],
                        "draft_translation": item["draft"],
                    }
                    for item in batch
                ],
                "response_contract": [
                    {
                        "segment_number": batch[0]["segment_number"],
                        "text": "postprocessed subtitle text",
                    }
                ],
            },
            error_prefix="Postprocess API",
            chunk_size=chunk_size,
            max_payload_chars=max_payload_chars,
            translation_mode=translation_mode,
            stage="postprocess",
            on_batch_success=on_batch_success,
        )
        final_text_by_number.update(
            {
                item["segment_number"]: final_text
                for item, final_text in zip(remaining_items, postprocessed_texts)
            }
        )

    final_segments = [
        {
            "start": translated["start"],
            "end": translated["end"],
            "text": final_text_by_number[index + 1],
        }
        for index, translated in enumerate(translated_segments)
    ]
    info = {
        "backend": "openai-compatible-postprocess",
        "model": model,
        "mode": postprocess_mode,
        "target_language": target_language,
        "source_language": source_language,
        "chunk_size": chunk_size,
        "max_payload_chars": max_payload_chars,
        "translation_mode": translation_mode,
        "blank_segment_count": len(blank_values),
        "resumed_segment_count": resumed_count,
    }
    return final_segments, info
