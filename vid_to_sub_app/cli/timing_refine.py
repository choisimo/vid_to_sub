from __future__ import annotations

import math
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .transcription import extract_audio_for_whisper_cpp

_FRAME_HOP_SEC = 0.01


@dataclass(frozen=True)
class _RefineProfile:
    enabled: bool
    disabled_reason: str | None
    min_gap_sec: float
    hangover_sec: float
    min_segment_sec: float
    min_trim_sec: float
    search_back_sec: float
    energy_threshold_ratio: float
    min_dynamic_range: float
    word_guard_slack_sec: float
    max_word_overshoot_sec: float
    confidence_threshold: float


def _normalize_content_type(value: object) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in {"auto", "speech", "music"}:
        return "auto"
    return normalized


def _profile_for_content_type(content_type: str) -> _RefineProfile:
    normalized = _normalize_content_type(content_type)
    if normalized == "music":
        return _RefineProfile(
            enabled=False,
            disabled_reason="content_type_music",
            min_gap_sec=0.08,
            hangover_sec=0.12,
            min_segment_sec=0.25,
            min_trim_sec=0.05,
            search_back_sec=1.2,
            energy_threshold_ratio=0.30,
            min_dynamic_range=0.16,
            word_guard_slack_sec=0.03,
            max_word_overshoot_sec=0.16,
            confidence_threshold=0.70,
        )
    return _RefineProfile(
        enabled=True,
        disabled_reason=None,
        min_gap_sec=0.08,
        hangover_sec=0.12,
        min_segment_sec=0.25,
        min_trim_sec=0.05,
        search_back_sec=1.2,
        energy_threshold_ratio=0.30,
        min_dynamic_range=0.16,
        word_guard_slack_sec=0.03,
        max_word_overshoot_sec=0.16,
        confidence_threshold=0.68,
    )


def refine_segment_timing(
    video: Path,
    segments: list[dict[str, Any]],
    info: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    info_map = dict(info or {})
    content_type = _normalize_content_type(info_map.get("content_type"))
    profile = _profile_for_content_type(content_type)
    stats: dict[str, Any] = {
        "enabled": profile.enabled,
        "applied": False,
        "content_type": content_type,
        "total_segments": len(segments),
        "trimmed_segments": 0,
        "median_trim_ms": 0,
        "p95_trim_ms": 0,
        "low_confidence_segments": 0,
        "word_anchor_segments": 0,
        "acoustic_anchor_segments": 0,
        "disabled_reason": profile.disabled_reason,
        "error": None,
    }

    copied_segments = [dict(segment) for segment in segments]
    if not copied_segments or not profile.enabled:
        return copied_segments, stats

    try:
        with tempfile.TemporaryDirectory(prefix="vid_to_sub_timing_") as tmpdir:
            wav_path = Path(tmpdir) / f"{video.stem}.timing.wav"
            extract_audio_for_whisper_cpp(video, wav_path)
            with wave.open(str(wav_path), "rb") as reader:
                if reader.getnchannels() != 1:
                    raise RuntimeError("timing refine expects mono PCM audio")
                if reader.getsampwidth() != 2:
                    raise RuntimeError("timing refine expects 16-bit PCM audio")
                if reader.getframerate() != 16000:
                    raise RuntimeError("timing refine expects 16 kHz PCM audio")
                refined_segments = _refine_with_reader(
                    reader=reader,
                    segments=copied_segments,
                    profile=profile,
                    stats=stats,
                )
    except Exception as exc:
        stats["error"] = str(exc)
        return copied_segments, stats

    trim_values = [
        int(segment.get("timing_trim_ms") or 0)
        for segment in refined_segments
        if int(segment.get("timing_trim_ms") or 0) > 0
    ]
    if trim_values:
        stats["applied"] = True
        stats["median_trim_ms"] = int(round(_percentile(trim_values, 0.5)))
        stats["p95_trim_ms"] = int(round(_percentile(trim_values, 0.95)))
    return refined_segments, stats


def _refine_with_reader(
    *,
    reader: wave.Wave_read,
    segments: list[dict[str, Any]],
    profile: _RefineProfile,
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    refined: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        normalized = dict(segment)
        start = _as_float(normalized.get("start"))
        raw_end = _as_float(normalized.get("end"))
        if start is None or raw_end is None or raw_end <= start:
            refined.append(normalized)
            continue

        next_start = _next_segment_start(segments, index)
        hard_cap = raw_end
        if next_start is not None:
            guarded_cap = max(start + profile.min_segment_sec, next_start - profile.min_gap_sec)
            hard_cap = min(raw_end, guarded_cap)

        word_end = _segment_word_end(normalized)
        if word_end is not None:
            stats["word_anchor_segments"] = int(stats["word_anchor_segments"]) + 1

        normalized["raw_end"] = round(raw_end, 3)
        if word_end is not None:
            normalized["word_end"] = round(word_end, 3)

        anchor_end = min(hard_cap, word_end) if word_end is not None else hard_cap
        search_start = max(start, anchor_end - profile.search_back_sec)
        if hard_cap - search_start < _FRAME_HOP_SEC:
            normalized["timing_confidence"] = 0.0
            normalized["timing_refined"] = False
            stats["low_confidence_segments"] = int(stats["low_confidence_segments"]) + 1
            refined.append(normalized)
            continue

        frames = _window_levels(reader, search_start, hard_cap)
        if not frames:
            normalized["timing_confidence"] = 0.0
            normalized["timing_refined"] = False
            stats["low_confidence_segments"] = int(stats["low_confidence_segments"]) + 1
            refined.append(normalized)
            continue

        levels = [level for _frame_start, _frame_end, level in frames]
        noise_floor = _percentile(levels, 0.2)
        peak_level = _percentile(levels, 0.9)
        dynamic_range = max(0.0, peak_level - noise_floor)
        threshold = noise_floor + (dynamic_range * profile.energy_threshold_ratio)

        last_speech_index: int | None = None
        if dynamic_range >= profile.min_dynamic_range:
            for frame_index in range(len(frames) - 1, -1, -1):
                if frames[frame_index][2] >= threshold:
                    last_speech_index = frame_index
                    break

        confidence = 0.0
        acoustic_end: float | None = None
        candidate_end = raw_end
        if last_speech_index is not None:
            acoustic_end = frames[last_speech_index][1]
            stats["acoustic_anchor_segments"] = int(stats["acoustic_anchor_segments"]) + 1
            tail_levels = [level for _s, _e, level in frames[last_speech_index + 1 :]]
            tail_silence_ratio = (
                sum(1 for level in tail_levels if level < threshold) / len(tail_levels)
                if tail_levels
                else 1.0
            )
            dynamic_score = min(1.0, dynamic_range / max(profile.min_dynamic_range * 2.0, 0.001))
            anchor_score = 1.0 if word_end is not None else 0.65
            confidence = min(
                1.0,
                (0.5 * dynamic_score) + (0.3 * tail_silence_ratio) + (0.2 * anchor_score),
            )
            candidate_end = min(hard_cap, acoustic_end + profile.hangover_sec)
            if word_end is not None:
                candidate_end = min(
                    candidate_end,
                    min(hard_cap, word_end + profile.max_word_overshoot_sec),
                )
                candidate_end = max(
                    candidate_end,
                    min(word_end, hard_cap) - profile.word_guard_slack_sec,
                )
            candidate_end = max(start + profile.min_segment_sec, candidate_end)

        if acoustic_end is not None:
            normalized["acoustic_end"] = round(acoustic_end, 3)
        normalized["timing_confidence"] = round(confidence, 4)

        trim_sec = max(0.0, raw_end - candidate_end)
        if trim_sec >= profile.min_trim_sec and confidence >= profile.confidence_threshold:
            normalized["end"] = round(candidate_end, 3)
            normalized["timing_refined"] = True
            normalized["timing_trim_ms"] = int(round(trim_sec * 1000))
            stats["trimmed_segments"] = int(stats["trimmed_segments"]) + 1
        else:
            normalized["timing_refined"] = False
            if confidence < profile.confidence_threshold:
                stats["low_confidence_segments"] = int(stats["low_confidence_segments"]) + 1

        refined.append(normalized)
    return refined


def _window_levels(
    reader: wave.Wave_read,
    start_sec: float,
    end_sec: float,
) -> list[tuple[float, float, float]]:
    sample_rate = reader.getframerate()
    sample_width = reader.getsampwidth()
    frame_samples = max(1, int(round(sample_rate * _FRAME_HOP_SEC)))
    start_frame = max(0, int(math.floor(start_sec * sample_rate)))
    end_frame = max(start_frame + 1, int(math.ceil(end_sec * sample_rate)))
    reader.setpos(start_frame)
    raw = reader.readframes(end_frame - start_frame)
    if not raw:
        return []

    chunk_bytes = frame_samples * sample_width
    frames: list[tuple[float, float, float]] = []
    byte_offset = 0
    while byte_offset < len(raw):
        chunk = raw[byte_offset : byte_offset + chunk_bytes]
        if len(chunk) < sample_width:
            break
        samples_in_chunk = len(chunk) // sample_width
        frame_start = start_sec + ((byte_offset / sample_width) / sample_rate)
        frame_end = frame_start + (samples_in_chunk / sample_rate)
        rms = _frame_rms(chunk, sample_width)
        level = math.log10(1.0 + rms)
        frames.append((frame_start, frame_end, level))
        byte_offset += chunk_bytes
    return frames


def _next_segment_start(segments: list[dict[str, Any]], index: int) -> float | None:
    for next_index in range(index + 1, len(segments)):
        next_start = _as_float(segments[next_index].get("start"))
        if next_start is not None:
            return next_start
    return None


def _segment_word_end(segment: dict[str, Any]) -> float | None:
    direct = _as_float(segment.get("word_end"))
    if direct is not None:
        return direct

    words = segment.get("words")
    if not isinstance(words, list):
        return None
    for word in reversed(words):
        if not isinstance(word, dict):
            continue
        end = _as_float(word.get("end"))
        if end is not None:
            return end
    return None


def _percentile(values: list[float] | list[int], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = max(0.0, min(1.0, quantile)) * (len(sorted_values) - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]
    fraction = index - lower
    return (
        (sorted_values[lower] * (1.0 - fraction))
        + (sorted_values[upper] * fraction)
    )


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _frame_rms(chunk: bytes, sample_width: int) -> float:
    if sample_width != 2:
        raise RuntimeError(f"Unsupported PCM width for timing refine: {sample_width}")

    samples = memoryview(chunk).cast("h")
    if not samples:
        return 0.0
    total = 0.0
    for sample in samples:
        total += float(sample * sample)
    return math.sqrt(total / len(samples))
