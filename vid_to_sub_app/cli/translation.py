from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from vid_to_sub_app.shared.constants import (
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


def translate_segments_openai_compatible(
    segments: list[dict],
    target_language: str,
    translation_model: Optional[str],
    translation_base_url: Optional[str],
    translation_api_key: Optional[str],
    source_language: Optional[str],
    chunk_size: int = 100,
) -> tuple[list[dict], dict]:
    model = translation_model or os.getenv(ENV_TRANSLATION_MODEL)
    base_url = translation_base_url or os.getenv(ENV_TRANSLATION_BASE_URL)
    api_key = translation_api_key or os.getenv(ENV_TRANSLATION_API_KEY)

    if not model:
        raise RuntimeError(
            "Translation model is not configured. Set --translation-model or "
            f"{ENV_TRANSLATION_MODEL}."
        )
    if not base_url:
        raise RuntimeError(
            "Translation base URL is not configured. Set --translation-base-url or "
            f"{ENV_TRANSLATION_BASE_URL}."
        )
    if not api_key:
        raise RuntimeError(
            "Translation API key is not configured. Set --translation-api-key or "
            f"{ENV_TRANSLATION_API_KEY}."
        )

    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/chat/completions"

    translated_segments: list[dict] = []
    for start_idx in range(0, len(segments), chunk_size):
        batch = segments[start_idx : start_idx + chunk_size]
        payload = {
            "model": model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You translate subtitle lines. Return only a JSON array of translated strings. "
                        "Keep the same item count and order. Do not merge or split subtitle lines. "
                        "Preserve line breaks inside each item when possible."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "target_language": target_language,
                            "source_language": source_language,
                            "subtitles": [seg["text"] for seg in batch],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        }
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
            raise RuntimeError(f"Translation API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Translation API request failed: {exc.reason}") from exc

        try:
            message = response_payload["choices"][0]["message"]["content"]
            translated_texts = extract_json_array(message)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not parse translation response as JSON string array: {response_payload}"
            ) from exc

        if len(translated_texts) != len(batch):
            raise RuntimeError(
                "Translation item count mismatch. "
                f"Expected {len(batch)}, got {len(translated_texts)}."
            )

        for segment, translated_text in zip(batch, translated_texts):
            translated_segments.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translated_text.strip(),
                }
            )

    info = {
        "backend": "openai-compatible-translation",
        "model": model,
        "target_language": target_language,
        "source_language": source_language,
    }
    return translated_segments, info
