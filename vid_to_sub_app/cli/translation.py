from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

from vid_to_sub_app.shared.constants import (
    ENV_POSTPROCESS_API_KEY,
    ENV_POSTPROCESS_BASE_URL,
    ENV_POSTPROCESS_MODEL,
    ENV_TRANSLATION_API_KEY,
    ENV_TRANSLATION_BASE_URL,
    ENV_TRANSLATION_MODEL,
)


def extract_json_array(text: str) -> list[str]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found in model response")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError("Model response is not a JSON string array")
    return payload


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


def _request_json_array(
    *,
    endpoint: str,
    api_key: str,
    payload: dict[str, Any],
    error_prefix: str,
) -> list[str]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "vid_to_sub/1.0 (+https://local.cli)",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{error_prefix} HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{error_prefix} request failed: {exc.reason}") from exc

    try:
        message = response_payload["choices"][0]["message"]["content"]
        return extract_json_array(message)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not parse {error_prefix.lower()} response as JSON string array: "
            f"{response_payload}"
        ) from exc


def _run_subtitle_agent_batches(
    *,
    items: list[dict],
    model: str,
    endpoint: str,
    api_key: str,
    system_prompt: str,
    build_user_payload,
    error_prefix: str,
    chunk_size: int,
) -> list[str]:
    outputs: list[str] = []
    for start_idx in range(0, len(items), chunk_size):
        batch = items[start_idx : start_idx + chunk_size]
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        build_user_payload(batch, start_idx),
                        ensure_ascii=False,
                    ),
                },
            ],
        }
        batch_outputs = _request_json_array(
            endpoint=endpoint,
            api_key=api_key,
            payload=payload,
            error_prefix=error_prefix,
        )
        if len(batch_outputs) != len(batch):
            raise RuntimeError(
                f"{error_prefix} item count mismatch. "
                f"Expected {len(batch)}, got {len(batch_outputs)}."
            )
        outputs.extend(item.strip() for item in batch_outputs)
    return outputs


def translate_segments_openai_compatible(
    segments: list[dict],
    target_language: str,
    translation_model: Optional[str],
    translation_base_url: Optional[str],
    translation_api_key: Optional[str],
    source_language: Optional[str],
    chunk_size: int = 100,
) -> tuple[list[dict], dict]:
    model, endpoint, api_key = _resolve_chat_config(
        label="Translation",
        model_arg=translation_model,
        base_url_arg=translation_base_url,
        api_key_arg=translation_api_key,
        model_env=ENV_TRANSLATION_MODEL,
        base_url_env=ENV_TRANSLATION_BASE_URL,
        api_key_env=ENV_TRANSLATION_API_KEY,
    )

    translated_texts = _run_subtitle_agent_batches(
        items=segments,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        system_prompt=(
            "You are the first-pass subtitle translation agent. Return only a JSON array of "
            "translated strings. Keep the same item count and order. Do not merge or split "
            "subtitle lines. Preserve line breaks inside each item when possible. Preserve "
            "proper nouns when they should stay untranslated. When the source feels lyrical or "
            "poetic, keep the target text natural rather than mechanically literal."
        ),
        build_user_payload=lambda batch, start_idx: {
            "target_language": target_language,
            "source_language": source_language,
            "segment_numbers": list(range(start_idx + 1, start_idx + len(batch) + 1)),
            "subtitles": [seg["text"] for seg in batch],
        },
        error_prefix="Translation API",
        chunk_size=chunk_size,
    )

    translated_segments = [
        {
            "start": segment["start"],
            "end": segment["end"],
            "text": translated_text,
        }
        for segment, translated_text in zip(segments, translated_texts)
    ]

    info = {
        "backend": "openai-compatible-translation",
        "model": model,
        "target_language": target_language,
        "source_language": source_language,
    }
    return translated_segments, info


def postprocess_translated_segments_openai_compatible(
    *,
    source_segments: list[dict],
    translated_segments: list[dict],
    target_language: str,
    postprocess_mode: str,
    postprocess_model: Optional[str],
    postprocess_base_url: Optional[str],
    postprocess_api_key: Optional[str],
    source_language: Optional[str],
    translation_model: Optional[str],
    translation_base_url: Optional[str],
    translation_api_key: Optional[str],
    chunk_size: int = 100,
) -> tuple[list[dict], dict]:
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
        raise RuntimeError("Postprocess requires the same number of source and translated segments.")

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
            "source": source["text"],
            "draft": translated["text"],
            "start": source["start"],
            "end": source["end"],
        }
        for source, translated in zip(source_segments, translated_segments)
    ]
    postprocessed_texts = _run_subtitle_agent_batches(
        items=items,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        system_prompt=(
            "You are the subtitle post-editing agent. Improve a first-pass translation while "
            "keeping the same item count and order. Return only a JSON array of final subtitle "
            "strings. Do not merge or split subtitle items. Keep each line aligned with its "
            "source line, natural in the target language, and consistent in tone across the batch. "
            f"{mode_instruction}"
        ),
        build_user_payload=lambda batch, start_idx: {
            "task": "postprocess_subtitle_translation",
            "mode": postprocess_mode,
            "target_language": target_language,
            "source_language": source_language,
            "segment_numbers": list(range(start_idx + 1, start_idx + len(batch) + 1)),
            "segments": [
                {
                    "source_text": item["source"],
                    "draft_translation": item["draft"],
                }
                for item in batch
            ],
        },
        error_prefix="Postprocess API",
        chunk_size=chunk_size,
    )

    final_segments = [
        {
            "start": translated["start"],
            "end": translated["end"],
            "text": final_text,
        }
        for translated, final_text in zip(translated_segments, postprocessed_texts)
    ]
    info = {
        "backend": "openai-compatible-postprocess",
        "model": model,
        "mode": postprocess_mode,
        "target_language": target_language,
        "source_language": source_language,
    }
    return final_segments, info
