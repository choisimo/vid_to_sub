#!/usr/bin/env python3
"""
vid_to_sub.py — Recursive video-to-subtitle CLI
================================================
Recursively finds video files under one or more input paths and generates
subtitle files using whisper.cpp on CPU by default, or faster-whisper,
openai-whisper, and whisperx as optional backends.

Usage
-----
    python vid_to_sub.py [OPTIONS] PATH [PATH ...]

Examples
--------
    # Transcribe every video under ./movies with whisper.cpp large-v3 on CPU
    python vid_to_sub.py ./movies

    # Override backend/model and output SRT only
    python vid_to_sub.py ./movies --backend faster-whisper --model turbo --format srt

    # Use openai-whisper backend, multiple roots, word-level timestamps
    python vid_to_sub.py ./movies ./series --backend whisper --model medium

    # Use whisperx backend with speaker diarization
    python vid_to_sub.py ./movies --backend whisperx --hf-token YOUR_HF_TOKEN

    # Dry-run: list discovered files without transcribing
    python vid_to_sub.py ./movies --dry-run

Dependencies (see requirements.txt)
------------------------------------
    whisper.cpp      — default backend via external whisper-cli binary
    faster-whisper   — optional backend
    openai-whisper   — optional backend  (pip install openai-whisper)
    whisperx         — optional backend  (pip install whisperx)
    ffmpeg           — runtime requirement (must be on PATH)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import timedelta
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".ts",
        ".mts",
        ".m2ts",
        ".mpeg",
        ".mpg",
        ".3gp",
        ".ogv",
        ".rmvb",
        ".vob",
        ".divx",
    }
)

SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {"srt", "vtt", "txt", "tsv", "json", "all"}
)

# faster-whisper model identifiers (also valid for whisper / whisperx)
KNOWN_MODELS: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
    # distil models (faster-whisper)
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
)

DEFAULT_MODEL = "large-v3"
DEFAULT_BACKEND = "whisper-cpp"
DEFAULT_FORMAT = "srt"
DEFAULT_DEVICE = "cpu"

ENV_TRANSLATION_BASE_URL = "VID_TO_SUB_TRANSLATION_BASE_URL"
ENV_TRANSLATION_API_KEY = "VID_TO_SUB_TRANSLATION_API_KEY"
ENV_TRANSLATION_MODEL = "VID_TO_SUB_TRANSLATION_MODEL"
ENV_WHISPER_CPP_BIN = "VID_TO_SUB_WHISPER_CPP_BIN"
ENV_WHISPER_CPP_MODEL = "VID_TO_SUB_WHISPER_CPP_MODEL"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_seconds(seconds: float) -> str:
    """Return a human-readable duration string."""
    return str(timedelta(seconds=round(seconds)))


def _srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    ms = int(round((seconds % 1) * 1000))
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def _vtt_timestamp(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp format HH:MM:SS.mmm."""
    return _srt_timestamp(seconds).replace(",", ".")


def _tsv_row(idx: int, start: float, end: float, text: str) -> str:
    return f"{int(start * 1000)}\t{int(end * 1000)}\t{text.strip()}"


def _parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def parse_srt(text: str) -> list[dict]:
    segments: list[dict] = []
    blocks = re.split(r"\r?\n\r?\n+", text.strip())
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timing_idx = 1 if re.fullmatch(r"\d+", lines[0]) else 0
        if timing_idx >= len(lines):
            continue
        timing = lines[timing_idx]
        if " --> " not in timing:
            continue
        start_raw, end_raw = timing.split(" --> ", 1)
        text_lines = lines[timing_idx + 1 :]
        if not text_lines:
            continue
        segments.append(
            {
                "start": _parse_srt_timestamp(start_raw),
                "end": _parse_srt_timestamp(end_raw),
                "text": "\n".join(text_lines).strip(),
            }
        )
    return segments


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_videos(roots: Sequence[str | Path], recursive: bool = True) -> List[Path]:
    """
    Discover video files under *roots*.

    Parameters
    ----------
    roots:
        Paths to files or directories.
    recursive:
        If True (default) use rglob for directories; otherwise only the
        immediate contents are considered.

    Returns
    -------
    Sorted, deduplicated list of absolute Path objects.
    """
    found: set[Path] = set()
    for raw in roots:
        p = Path(raw).resolve()
        if not p.exists():
            print(f"[WARN] Path does not exist, skipping: {p}", file=sys.stderr)
            continue
        if p.is_file():
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                found.add(p)
            else:
                print(
                    f"[WARN] File extension '{p.suffix}' not in known video list, "
                    f"skipping: {p}",
                    file=sys.stderr,
                )
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in p.glob(pattern):
                if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
                    found.add(child)
        else:
            print(f"[WARN] Not a file or directory, skipping: {p}", file=sys.stderr)

    return sorted(found)


# ---------------------------------------------------------------------------
# Subtitle writers
# ---------------------------------------------------------------------------


def _segments_to_srt(segments) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_srt_timestamp(seg['start'])} --> {_srt_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _segments_to_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_vtt_timestamp(seg['start'])} --> {_vtt_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def _segments_to_txt(segments) -> str:
    return "\n".join(seg["text"].strip() for seg in segments)


def _segments_to_tsv(segments) -> str:
    rows = ["start\tend\ttext"]
    for seg in segments:
        rows.append(_tsv_row(0, seg["start"], seg["end"], seg["text"]))
    return "\n".join(rows)


def _segments_to_json(segments, info: dict | None = None) -> str:
    payload = {"segments": segments}
    if info:
        payload["info"] = info
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_outputs(
    video_path: Path,
    segments: list[dict],
    formats: frozenset[str],
    output_dir: Optional[Path],
    info: dict | None = None,
    name_suffix: str = "",
) -> list[Path]:
    """
    Write subtitle/transcript files next to the video (or into *output_dir*).

    Returns list of written paths.
    """
    base_dir = output_dir if output_dir else video_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    written: list[Path] = []

    writers = {
        "srt": ("srt", _segments_to_srt),
        "vtt": ("vtt", _segments_to_vtt),
        "txt": ("txt", _segments_to_txt),
        "tsv": ("tsv", _segments_to_tsv),
        "json": ("json", lambda s: _segments_to_json(s, info)),
    }

    active = (
        set(writers.keys()) if "all" in formats else (formats & set(writers.keys()))
    )

    for fmt, (ext, fn) in writers.items():
        if fmt not in active:
            continue
        out_path = base_dir / f"{stem}{name_suffix}.{ext}"
        content = fn(segments)
        out_path.write_text(content, encoding="utf-8")
        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# Backend: faster-whisper
# ---------------------------------------------------------------------------


def _resolve_device_fw(device: str) -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper."""
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", "float16"
        except ImportError:
            pass
        return "cpu", "int8"
    if device == "cuda":
        return "cuda", "float16"
    if device == "mps":
        # faster-whisper does not support MPS; fall back to CPU
        print(
            "[WARN] faster-whisper does not support MPS; using CPU int8.",
            file=sys.stderr,
        )
        return "cpu", "int8"
    return "cpu", "int8"


def transcribe_faster_whisper(
    video: Path,
    model_name: str,
    device: str,
    language: Optional[str],
    beam_size: int,
    compute_type: Optional[str],
) -> tuple[list[dict], dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "[ERROR] faster-whisper is not installed.\n"
            "  Install with: pip install faster-whisper",
            file=sys.stderr,
        )
        sys.exit(1)

    dev, default_ct = _resolve_device_fw(device)
    ct = compute_type or default_ct

    model = WhisperModel(model_name, device=dev, compute_type=ct)
    segs_raw, info_raw = model.transcribe(
        str(video),
        language=language,
        beam_size=beam_size,
        word_timestamps=False,
    )

    segments: list[dict] = []
    for seg in segs_raw:  # generator — must consume
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})

    info = {
        "language": info_raw.language,
        "language_probability": round(info_raw.language_probability, 4),
        "duration": round(info_raw.duration, 3),
        "backend": "faster-whisper",
        "model": model_name,
    }
    return segments, info


# ---------------------------------------------------------------------------
# Backend: openai-whisper
# ---------------------------------------------------------------------------


def transcribe_openai_whisper(
    video: Path,
    model_name: str,
    device: str,
    language: Optional[str],
    beam_size: int,
) -> tuple[list[dict], dict]:
    try:
        import whisper as ow
    except ImportError:
        print(
            "[ERROR] openai-whisper is not installed.\n"
            "  Install with: pip install openai-whisper",
            file=sys.stderr,
        )
        sys.exit(1)

    dev = "cpu" if device == "auto" else device
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                dev = "cuda"
        except ImportError:
            pass

    model = ow.load_model(model_name, device=dev)
    result = model.transcribe(
        str(video),
        language=language,
        beam_size=beam_size,
    )

    segments: list[dict] = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in result.get("segments", [])
    ]
    info = {
        "language": result.get("language"),
        "backend": "openai-whisper",
        "model": model_name,
    }
    return segments, info


# ---------------------------------------------------------------------------
# Backend: whisperx
# ---------------------------------------------------------------------------


def transcribe_whisperx(
    video: Path,
    model_name: str,
    device: str,
    language: Optional[str],
    beam_size: int,
    compute_type: Optional[str],
    hf_token: Optional[str],
    diarize: bool,
) -> tuple[list[dict], dict]:
    try:
        import whisperx
    except ImportError:
        print(
            "[ERROR] whisperx is not installed.\n  Install with: pip install whisperx",
            file=sys.stderr,
        )
        sys.exit(1)

    dev = "cpu" if device == "auto" else device
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                dev = "cuda"
        except ImportError:
            pass

    ct = compute_type or ("float16" if dev == "cuda" else "int8")

    model = whisperx.load_model(model_name, dev, compute_type=ct, language=language)
    audio = whisperx.load_audio(str(video))
    result = model.transcribe(audio, batch_size=16, language=language)

    # Alignment
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=dev
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            dev,
            return_char_alignments=False,
        )
    except Exception as exc:
        print(
            f"[WARN] Alignment failed ({exc}); using unaligned segments.",
            file=sys.stderr,
        )

    # Diarization
    if diarize:
        if not hf_token:
            print(
                "[WARN] --hf-token is required for diarization; skipping diarization.",
                file=sys.stderr,
            )
        else:
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token, device=dev
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as exc:
                print(
                    f"[WARN] Diarization failed ({exc}); continuing without it.",
                    file=sys.stderr,
                )

    segments: list[dict] = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in result.get("segments", [])
    ]
    info = {
        "language": result.get("language"),
        "backend": "whisperx",
        "model": model_name,
    }
    return segments, info


# ---------------------------------------------------------------------------
# Backend: whisper.cpp
# ---------------------------------------------------------------------------


def extract_audio_for_whisper_cpp(video: Path, wav_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required on PATH for whisper.cpp backend") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "ffmpeg audio extraction failed") from exc


def _default_whisper_cpp_model_path(model_name: str) -> Path:
    env_path = os.getenv(ENV_WHISPER_CPP_MODEL)
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path("models") / f"ggml-{model_name}.bin"


def transcribe_whisper_cpp(
    video: Path,
    model_name: str,
    device: str,
    language: Optional[str],
    threads: int,
    whisper_cpp_model_path: Optional[str],
) -> tuple[list[dict], dict]:
    if device not in {"auto", "cpu"}:
        print(
            "[WARN] whisper.cpp backend is CPU-oriented here; using CPU.",
            file=sys.stderr,
        )

    whisper_cpp_bin = os.getenv(ENV_WHISPER_CPP_BIN, "whisper-cli")
    model_path = (
        Path(whisper_cpp_model_path).expanduser().resolve()
        if whisper_cpp_model_path
        else _default_whisper_cpp_model_path(model_name).resolve()
    )
    if not model_path.exists():
        raise RuntimeError(
            "whisper.cpp model file not found. Set --whisper-cpp-model-path or "
            f"{ENV_WHISPER_CPP_MODEL}. Expected: {model_path}"
        )

    with tempfile.TemporaryDirectory(prefix="vid_to_sub_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        wav_path = tmpdir_path / f"{video.stem}.wav"
        output_prefix = tmpdir_path / video.stem
        extract_audio_for_whisper_cpp(video, wav_path)

        command = [
            whisper_cpp_bin,
            "-m",
            str(model_path),
            "-f",
            str(wav_path),
            "-osrt",
            "-of",
            str(output_prefix),
            "-t",
            str(max(1, threads)),
        ]
        if language:
            command.extend(["-l", language])

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "whisper.cpp CLI not found. Set "
                f"{ENV_WHISPER_CPP_BIN} or install `whisper-cli` on PATH"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(exc.stderr.strip() or "whisper.cpp transcription failed") from exc

        srt_path = output_prefix.with_suffix(".srt")
        if not srt_path.exists():
            raise RuntimeError(f"whisper.cpp did not produce expected SRT: {srt_path}")

        segments = parse_srt(srt_path.read_text(encoding="utf-8"))
        info = {
            "language": language or "auto",
            "backend": "whisper-cpp",
            "model": model_name,
            "model_path": str(model_path),
        }
        return segments, info


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def _extract_json_array(text: str) -> list[str]:
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
                "Authorization": f"Bearer {api_key}",
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
            translated_texts = _extract_json_array(message)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not parse translation response as JSON string array: {response_payload}"
            ) from exc

        if len(translated_texts) != len(batch):
            raise RuntimeError(
                "Translation item count mismatch. "
                f"Expected {len(batch)}, got {len(translated_texts)}."
            )

        for seg, translated_text in zip(batch, translated_texts):
            translated_segments.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
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


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def transcribe(
    video: Path,
    backend: str,
    model_name: str,
    device: str,
    language: Optional[str],
    beam_size: int,
    compute_type: Optional[str],
    hf_token: Optional[str],
    diarize: bool,
    whisper_cpp_model_path: Optional[str],
    threads: int,
) -> tuple[list[dict], dict]:
    if backend == "whisper-cpp":
        return transcribe_whisper_cpp(
            video,
            model_name,
            device,
            language,
            threads,
            whisper_cpp_model_path,
        )
    if backend == "faster-whisper":
        return transcribe_faster_whisper(
            video, model_name, device, language, beam_size, compute_type
        )
    if backend == "whisper":
        return transcribe_openai_whisper(video, model_name, device, language, beam_size)
    if backend == "whisperx":
        return transcribe_whisperx(
            video,
            model_name,
            device,
            language,
            beam_size,
            compute_type,
            hf_token,
            diarize,
        )
    raise ValueError(f"Unknown backend: {backend!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vid_to_sub",
        description="Recursively transcribe video files to subtitle/transcript files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input/output ---
    p.add_argument(
        "paths",
        nargs="+",
        metavar="PATH",
        help="One or more video files or directories to process.",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Write output files to DIR instead of next to the source video. "
            "Directory will be created if it does not exist."
        ),
    )
    p.add_argument(
        "--no-recurse",
        action="store_true",
        default=False,
        help="Do not recurse into subdirectories.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip a video if its primary output file already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List discovered video files without transcribing.",
    )

    # --- Transcription ---
    p.add_argument(
        "--backend",
        choices=["whisper-cpp", "faster-whisper", "whisper", "whisperx"],
        default=DEFAULT_BACKEND,
        help=f"Transcription backend (default: {DEFAULT_BACKEND}).",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=(
            f"Model identifier (default: {DEFAULT_MODEL}). "
            f"Known values: {', '.join(KNOWN_MODELS)}"
        ),
    )
    p.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="Source language code, e.g. 'en', 'ja', 'fr'. Auto-detect if omitted.",
    )
    p.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["auto", "cpu", "cuda", "mps"],
        help=(
            "Compute device (default: cpu). "
            "whisper.cpp uses CPU in this tool. "
            "Other backends may use auto/cuda/mps when selected."
        ),
    )
    p.add_argument(
        "--compute-type",
        default=None,
        metavar="TYPE",
        help=(
            "Quantization type for faster-whisper / whisperx "
            "(e.g. float16, int8_float16, int8). "
            "Auto-selected based on device if omitted."
        ),
    )
    p.add_argument(
        "--beam-size",
        type=int,
        default=5,
        metavar="N",
        help="Beam size for decoding (default: 5).",
    )

    # --- Output formats ---
    p.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=sorted(SUPPORTED_FORMATS),
        metavar="FMT",
        help=(
            "Output format(s): srt, vtt, txt, tsv, json, all. "
            "Repeat to produce multiple formats. Default: srt. "
            "Use 'all' for every format."
        ),
    )

    # --- whisper.cpp / whisperx extras ---
    p.add_argument(
        "--whisper-cpp-model-path",
        default=None,
        metavar="PATH",
        help=(
            "Path to whisper.cpp model file. "
            f"Defaults to {ENV_WHISPER_CPP_MODEL} or ./models/ggml-<model>.bin"
        ),
    )
    p.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="Hugging Face token required for whisperx diarization.",
    )
    p.add_argument(
        "--diarize",
        action="store_true",
        default=False,
        help="Enable speaker diarization (whisperx backend only; requires --hf-token).",
    )

    # --- Translation ---
    p.add_argument(
        "--translate-to",
        default=None,
        metavar="LANG",
        help="Translate subtitle text to target language while keeping timings unchanged.",
    )
    p.add_argument(
        "--translation-model",
        default=None,
        metavar="MODEL",
        help=f"Override translation model (default from {ENV_TRANSLATION_MODEL}).",
    )
    p.add_argument(
        "--translation-base-url",
        default=None,
        metavar="URL",
        help=(
            "OpenAI-compatible base URL or /chat/completions URL for translation "
            f"(default from {ENV_TRANSLATION_BASE_URL})."
        ),
    )
    p.add_argument(
        "--translation-api-key",
        default=None,
        metavar="KEY",
        help=f"Override translation API key (default from {ENV_TRANSLATION_API_KEY}).",
    )

    # --- Misc ---
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel transcription workers (default: 1). "
            "Only useful when processing many files on CPU; "
            "GPU users should keep this at 1."
        ),
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print segment-level progress.",
    )
    p.add_argument(
        "--list-models",
        action="store_true",
        default=False,
        help="Print known model identifiers and exit.",
    )

    return p


def _primary_output_exists(
    video: Path, formats: frozenset[str], output_dir: Optional[Path]
) -> bool:
    """Check if at least one expected output file already exists."""
    base_dir = output_dir if output_dir else video.parent
    active_formats = (
        {"srt", "vtt", "txt", "tsv", "json"}
        if "all" in formats
        else (formats - {"all"})
    )
    for fmt in active_formats:
        candidate = base_dir / f"{video.stem}.{fmt}"
        if candidate.exists():
            return True
    return False


def process_one(
    video: Path,
    args: argparse.Namespace,
    formats: frozenset[str],
    output_dir: Optional[Path],
    worker_id: int = 0,
) -> bool:
    """
    Transcribe *video* and write outputs.

    Returns True on success, False on error.
    """
    prefix = f"[{worker_id}] " if args.workers > 1 else ""
    print(f"{prefix}▶  {video}", flush=True)
    t0 = time.monotonic()

    try:
        segments, info = transcribe(
            video=video,
            backend=args.backend,
            model_name=args.model,
            device=args.device,
            language=args.language,
            beam_size=args.beam_size,
            compute_type=args.compute_type,
            hf_token=args.hf_token,
            diarize=args.diarize,
            whisper_cpp_model_path=args.whisper_cpp_model_path,
            threads=args.workers,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"{prefix}[ERROR] Transcription failed: {exc}", file=sys.stderr)
        return False

    if args.verbose:
        for s in segments:
            print(
                f"  {_srt_timestamp(s['start'])} --> {_srt_timestamp(s['end'])}  {s['text'].strip()}"
            )

    try:
        written = write_outputs(video, segments, formats, output_dir, info)
        translated_written: list[Path] = []
        if args.translate_to:
            translated_segments, translation_info = translate_segments_openai_compatible(
                segments=segments,
                target_language=args.translate_to,
                translation_model=args.translation_model,
                translation_base_url=args.translation_base_url,
                translation_api_key=args.translation_api_key,
                source_language=info.get("language"),
            )
            translated_info = dict(info)
            translated_info["translation"] = translation_info
            translated_written = write_outputs(
                video,
                translated_segments,
                formats,
                output_dir,
                translated_info,
                name_suffix=f".{args.translate_to}",
            )
    except Exception as exc:
        print(f"{prefix}[ERROR] Output/translation failed: {exc}", file=sys.stderr)
        return False

    elapsed = time.monotonic() - t0
    lang = info.get("language", "?")
    dur = info.get("duration")
    dur_str = f"  video={_fmt_seconds(dur)}" if dur else ""
    print(
        f"{prefix}   ✓ [{lang}]{dur_str}  wall={_fmt_seconds(elapsed)}  "
        + "  ".join(str(w.name) for w in (written + translated_written)),
        flush=True,
    )
    return True


def run_parallel(
    videos: list[Path],
    args: argparse.Namespace,
    formats: frozenset[str],
    output_dir: Optional[Path],
) -> tuple[int, int]:
    """Run transcription across multiple workers using ThreadPoolExecutor."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ok = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_one, v, args, formats, output_dir, i % args.workers): v
            for i, v in enumerate(videos)
        }
        for fut in as_completed(futures):
            try:
                success = fut.result()
            except Exception as exc:
                print(f"[ERROR] Worker exception: {exc}", file=sys.stderr)
                success = False
            if success:
                ok += 1
            else:
                err += 1
    return ok, err


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        print("Known model identifiers:")
        for m in KNOWN_MODELS:
            print(f"  {m}")
        return 0

    # Resolve output formats
    raw_formats = args.formats or [DEFAULT_FORMAT]
    formats: frozenset[str] = frozenset(raw_formats)

    # Resolve output directory
    output_dir: Optional[Path] = (
        Path(args.output_dir).resolve() if args.output_dir else None
    )

    # Discover videos
    recursive = not args.no_recurse
    videos = discover_videos(args.paths, recursive=recursive)

    if not videos:
        print("[WARN] No video files found.", file=sys.stderr)
        return 1

    print(f"Found {len(videos)} video file(s).", flush=True)

    if args.dry_run:
        for v in videos:
            print(f"  {v}")
        return 0

    # Filter already-processed files
    if args.skip_existing:
        before = len(videos)
        videos = [
            v for v in videos if not _primary_output_exists(v, formats, output_dir)
        ]
        skipped = before - len(videos)
        if skipped:
            print(f"Skipping {skipped} already-processed file(s).")
        if not videos:
            print("Nothing to do.")
            return 0

    # Run
    t_total = time.monotonic()
    if args.workers > 1:
        ok, err = run_parallel(videos, args, formats, output_dir)
    else:
        ok = err = 0
        for video in videos:
            success = process_one(video, args, formats, output_dir)
            if success:
                ok += 1
            else:
                err += 1

    elapsed_total = time.monotonic() - t_total
    print(
        f"\nDone. {ok} succeeded, {err} failed. "
        f"Total wall time: {_fmt_seconds(elapsed_total)}",
        flush=True,
    )
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
