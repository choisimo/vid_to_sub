from __future__ import annotations

import json
import re
import subprocess
from datetime import timedelta
from pathlib import Path
from typing import Optional

from vid_to_sub_app.shared.constants import FORMATS

WHISPER_CPP_SEGMENT_RE = re.compile(
    r"\[(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})\]"
)


def fmt_seconds(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


def srt_timestamp(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, rem = divmod(millis, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def tsv_row(start: float, end: float, text: str) -> str:
    return f"{start:.3f}\t{end:.3f}\t{text.strip()}"


def parse_media_timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)


def parse_srt_timestamp(value: str) -> float:
    return parse_media_timestamp(value.replace(",", "."))


def parse_whisper_cpp_progress_seconds(line: str) -> Optional[float]:
    match = WHISPER_CPP_SEGMENT_RE.search(line)
    if not match:
        return None
    _, end = match.groups()
    return parse_media_timestamp(end)


def probe_media_duration(video: Path) -> Optional[float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    value = result.stdout.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_srt(text: str) -> list[dict]:
    blocks = re.split(r"\n\s*\n", text.strip(), flags=re.MULTILINE)
    segments: list[dict] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        idx_offset = 1 if re.fullmatch(r"\d+", lines[0]) else 0
        if idx_offset >= len(lines):
            continue

        timestamp_line = lines[idx_offset]
        if "-->" not in timestamp_line:
            continue
        start_raw, end_raw = [part.strip() for part in timestamp_line.split("-->", 1)]
        text_lines = lines[idx_offset + 1 :]
        if not text_lines:
            continue

        segments.append(
            {
                "start": parse_srt_timestamp(start_raw),
                "end": parse_srt_timestamp(end_raw),
                "text": "\n".join(text_lines).strip(),
            }
        )

    return segments


def segments_to_srt(segments) -> str:
    lines: list[str] = []
    for idx, seg in enumerate(segments, 1):
        lines.append(str(idx))
        lines.append(f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def segments_to_vtt(segments) -> str:
    lines = ["WEBVTT", ""]
    for idx, seg in enumerate(segments, 1):
        lines.append(str(idx))
        lines.append(f"{vtt_timestamp(seg['start'])} --> {vtt_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


def segments_to_txt(segments) -> str:
    return "\n".join(seg["text"].strip() for seg in segments)


def segments_to_tsv(segments) -> str:
    rows = ["start\tend\ttext"]
    for seg in segments:
        rows.append(tsv_row(seg["start"], seg["end"], seg["text"]))
    return "\n".join(rows)


def segments_to_json(segments, info: dict | None = None) -> str:
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
    base_dir = output_dir if output_dir else video_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    written: list[Path] = []

    writers = {
        "srt": ("srt", segments_to_srt),
        "vtt": ("vtt", segments_to_vtt),
        "txt": ("txt", segments_to_txt),
        "tsv": ("tsv", segments_to_tsv),
        "json": ("json", lambda s: segments_to_json(s, info)),
    }
    active = set(FORMATS) if "all" in formats else (formats & set(FORMATS))

    for fmt, (ext, writer) in writers.items():
        if fmt not in active:
            continue
        out_path = base_dir / f"{stem}{name_suffix}.{ext}"
        out_path.write_text(writer(segments), encoding="utf-8")
        written.append(out_path)

    return written
