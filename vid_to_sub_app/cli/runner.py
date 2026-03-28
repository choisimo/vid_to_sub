from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from vid_to_sub_app.shared.constants import (
    EVENT_PREFIX,
    FASTER_WHISPER_MODEL_MIN_VRAM_GB,
)

from .discovery import hash_video_folder
from .manifest import FolderAwareScheduler, ProcessResult
from .output import (
    fmt_seconds,
    planned_output_paths,
    probe_media_duration,
    probe_media_metadata,
    srt_timestamp,
    write_outputs,
)
from .stage_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    StageArtifact,
    build_stage_artifact_metadata,
    fingerprint_source_path,
    load_stage_artifact,
    verify_artifact_source,
    write_stage_artifact,
)
from .timing_refine import refine_segment_timing
from .transcription import transcribe
from .translation import (
    postprocess_translated_segments_openai_compatible,
    translate_segments_openai_compatible,
)

# ---------------------------------------------------------------------------
# Rollback toggle: set VID_TO_SUB_LEGACY_INLINE=1 to skip the stage-artifact
# handoff and run transcription + translation in a single inline pass (the
# pre-split behaviour). Useful as a quick rollback knob without code changes.
# ---------------------------------------------------------------------------
_LEGACY_INLINE = os.environ.get("VID_TO_SUB_LEGACY_INLINE", "").strip() == "1"
_STAGE2_PROGRESS_SUFFIX = ".stage2.progress.jsonl"
_LEGACY_INLINE_WARNING_LOCK = threading.Lock()
_LEGACY_INLINE_WARNING_EMITTED = False


def _assess_stage1_quality(
    segments: list[dict[str, Any]], info: dict[str, Any]
) -> dict[str, Any]:
    texts = [str(segment.get("text", "")).strip().lower() for segment in segments]
    non_empty_texts = [text for text in texts if text]
    unique_ratio = len(set(non_empty_texts)) / max(1, len(non_empty_texts))
    repeated_ratio = 1.0 - unique_ratio
    noise_like_ratio = sum(
        1
        for text in non_empty_texts
        if len(text) <= 2 or text in {".", ",", "...", "-"}
    ) / max(1, len(non_empty_texts))
    lang_prob = info.get("language_probability")

    reasons: list[str] = []
    if isinstance(lang_prob, (int, float)) and float(lang_prob) < 0.80:
        reasons.append("low_language_probability")
    if not non_empty_texts:
        reasons.append("empty_transcript")
    elif repeated_ratio > 0.35:
        reasons.append("high_repetition")
    if noise_like_ratio > 0.20:
        reasons.append("noise_like_segments")

    quality = {
        "language_probability": round(float(lang_prob), 4)
        if isinstance(lang_prob, (int, float))
        else None,
        "unique_ratio": round(unique_ratio, 4),
        "repeated_ratio": round(repeated_ratio, 4),
        "noise_like_ratio": round(noise_like_ratio, 4),
        "suspicious": bool(reasons),
        "reasons": reasons,
    }
    timing_refine = info.get("timing_refine")
    if isinstance(timing_refine, dict):
        quality["timing_refine"] = timing_refine
    return quality


def _stage1_quality_sort_key(quality: dict[str, Any]) -> tuple[int, int, float, float]:
    lang_prob = quality.get("language_probability")
    normalized_lang_prob = (
        float(lang_prob) if isinstance(lang_prob, (int, float)) else 0.0
    )
    return (
        1 if quality.get("suspicious") else 0,
        len(quality.get("reasons") or []),
        -normalized_lang_prob,
        float(quality.get("repeated_ratio") or 0.0)
        + float(quality.get("noise_like_ratio") or 0.0),
    )


def _requested_content_type(args: argparse.Namespace) -> str:
    return str(getattr(args, "content_type", "auto") or "auto").strip().lower()


def _music_safe_retry_supported(args: argparse.Namespace) -> bool:
    return str(getattr(args, "backend", "") or "") in {"faster-whisper", "whisper"}


def _run_stage1_transcription(
    *,
    video: Path,
    args: argparse.Namespace,
    backend_threads: int,
    progress_callback=None,
    language: str | None = None,
    content_type: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return transcribe(
        video=video,
        backend=args.backend,
        model_name=args.model,
        device=args.device,
        language=args.language if language is None else language,
        content_type=_requested_content_type(args)
        if content_type is None
        else content_type,
        beam_size=args.beam_size,
        compute_type=args.compute_type,
        hf_token=args.hf_token,
        diarize=args.diarize,
        whisper_cpp_model_path=args.whisper_cpp_model_path,
        threads=backend_threads,
        progress_callback=progress_callback,
    )


def _refine_stage1_timing(
    *,
    video: Path,
    segments: list[dict[str, Any]],
    info: dict[str, Any],
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    timing_started_at = time.monotonic()
    refined_segments, timing_stats = refine_segment_timing(video, segments, info)
    timing_elapsed_ms = (time.monotonic() - timing_started_at) * 1000.0
    info = dict(info)
    info["timing_refine"] = timing_stats
    print(
        f"{prefix}[METRIC] stage=timing_refine file={video.name} "
        f"elapsed_ms={timing_elapsed_ms:.1f} "
        f"trimmed_segments={int(timing_stats.get('trimmed_segments') or 0)} "
        f"median_trim_ms={int(timing_stats.get('median_trim_ms') or 0)} "
        f"p95_trim_ms={int(timing_stats.get('p95_trim_ms') or 0)} "
        f"low_confidence_segments={int(timing_stats.get('low_confidence_segments') or 0)}",
        file=sys.stderr,
        flush=True,
    )
    return refined_segments, info


def _maybe_retry_stage1_with_music_preset(
    *,
    video: Path,
    args: argparse.Namespace,
    backend_threads: int,
    segments: list[dict[str, Any]],
    info: dict[str, Any],
    quality: dict[str, Any],
    prefix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if (
        _requested_content_type(args) != "auto"
        or not quality.get("suspicious")
        or str(info.get("content_type") or "").strip().lower() == "music"
    ):
        return segments, info, quality

    retry_details: dict[str, Any] = {
        "attempted": False,
        "selected": False,
        "reason": "stage1_suspicious_under_auto",
    }
    info = dict(info)
    quality = dict(quality)
    retry_language = str(args.language or info.get("language") or "").strip() or None
    if not _music_safe_retry_supported(args):
        retry_details["blocked_reason"] = "backend_does_not_support_music_preset"
        info["auto_music_retry"] = retry_details
        quality["auto_music_retry"] = retry_details
        return segments, info, quality
    if not retry_language:
        retry_details["blocked_reason"] = "missing_language_for_music_preset"
        info["auto_music_retry"] = retry_details
        quality["auto_music_retry"] = retry_details
        return segments, info, quality

    retry_details["attempted"] = True
    retry_details["retry_language"] = retry_language
    print(
        f"{prefix}[WARN] Stage 1 transcript looked suspicious under --content-type auto; "
        f"retrying with music-safe decoding using language '{retry_language}'.",
        file=sys.stderr,
        flush=True,
    )
    retried_segments, retried_info = _run_stage1_transcription(
        video=video,
        args=args,
        backend_threads=backend_threads,
        language=retry_language,
        content_type="music",
    )
    retried_segments, retried_info = _refine_stage1_timing(
        video=video,
        segments=retried_segments,
        info=retried_info,
        prefix=prefix,
    )
    retried_quality = _assess_stage1_quality(retried_segments, retried_info)
    retry_details["retried_quality"] = {
        "suspicious": bool(retried_quality.get("suspicious")),
        "reasons": list(retried_quality.get("reasons") or []),
        "language_probability": retried_quality.get("language_probability"),
        "repeated_ratio": retried_quality.get("repeated_ratio"),
        "noise_like_ratio": retried_quality.get("noise_like_ratio"),
    }

    if _stage1_quality_sort_key(retried_quality) < _stage1_quality_sort_key(quality):
        retry_details["selected"] = True
        retried_info = dict(retried_info)
        retried_info["auto_music_retry"] = retry_details
        retried_quality = dict(retried_quality)
        retried_quality["auto_music_retry"] = retry_details
        print(
            f"{prefix}[WARN] Music-safe retry improved Stage 1 quality; using the retried transcript.",
            file=sys.stderr,
            flush=True,
        )
        return retried_segments, retried_info, retried_quality

    retry_details["selected"] = False
    info["auto_music_retry"] = retry_details
    quality["auto_music_retry"] = retry_details
    return segments, info, quality


def _stage1_quality_warning(
    *,
    reasons: list[str],
    held_paths: list[Path],
) -> str:
    reason_suffix = f" Reasons: {', '.join(reasons)}." if reasons else ""
    held_suffix = (
        " Held outputs: " + ", ".join(path.name for path in held_paths) + "."
        if held_paths
        else ""
    )
    return (
        "Stage 1 transcript quality is suspicious; canonical outputs were withheld."
        f"{reason_suffix}{held_suffix}"
    )


_PRE_ASR_MUSIC_PROMOTION_THRESHOLD = 0.75


def legacy_inline_requested() -> bool:
    return _LEGACY_INLINE


def legacy_inline_block_reason(args: argparse.Namespace) -> str | None:
    if not _LEGACY_INLINE:
        return None

    blocked_flags: list[str] = []
    if bool(getattr(args, "translate_to", None)):
        blocked_flags.append("--translate-to")
    if _requested_content_type(args) == "auto":
        blocked_flags.append("--content-type auto")
    if not blocked_flags:
        return None

    if len(blocked_flags) == 1:
        blocked_summary = blocked_flags[0]
    else:
        blocked_summary = " and ".join(blocked_flags)
    return (
        "VID_TO_SUB_LEGACY_INLINE=1 is not allowed with "
        f"{blocked_summary} because it bypasses the stage artifact handoff, "
        "suspicious-output hold, and Stage 2 safety gates. Disable "
        "VID_TO_SUB_LEGACY_INLINE to continue."
    )


def emit_legacy_inline_warning_once() -> None:
    global _LEGACY_INLINE_WARNING_EMITTED
    if not _LEGACY_INLINE:
        return

    with _LEGACY_INLINE_WARNING_LOCK:
        if _LEGACY_INLINE_WARNING_EMITTED:
            return
        print(
            "[WARN] VID_TO_SUB_LEGACY_INLINE=1 bypasses stage artifacts, suspicious-output "
            "hold, Stage 2 resume state, and Stage 2 policy gates. Use only as a local "
            "dev/test rollback switch.",
            file=sys.stderr,
            flush=True,
        )
        _LEGACY_INLINE_WARNING_EMITTED = True


def _normalized_hint_text(value: object) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[_\-.]+", " ", text)


def _hint_tokens(value: object) -> set[str]:
    return {
        token
        for token in re.split(r"[^0-9a-z]+", _normalized_hint_text(value))
        if token
    }


def _collect_pre_asr_hint_tag_dicts(
    metadata: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(metadata, dict):
        return []

    tag_dicts: list[tuple[str, dict[str, Any]]] = []
    format_payload = metadata.get("format")
    if isinstance(format_payload, dict):
        format_tags = format_payload.get("tags")
        if isinstance(format_tags, dict):
            tag_dicts.append(("format_tag", format_tags))

    for index, stream in enumerate(metadata.get("streams") or []):
        if not isinstance(stream, dict):
            continue
        stream_tags = stream.get("tags")
        if isinstance(stream_tags, dict):
            tag_dicts.append((f"stream_tag:{index}", stream_tags))

    for index, chapter in enumerate(metadata.get("chapters") or []):
        if not isinstance(chapter, dict):
            continue
        chapter_tags = chapter.get("tags")
        if isinstance(chapter_tags, dict):
            tag_dicts.append((f"chapter_tag:{index}", chapter_tags))

    return tag_dicts


def predict_auto_content_type(
    video: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if _requested_content_type(args) != "auto":
        return {}

    reasons: list[str] = []
    seen_reasons: set[str] = set()
    score = 0.0

    def add_reason(reason: str, weight: float) -> None:
        nonlocal score
        if reason in seen_reasons:
            return
        seen_reasons.add(reason)
        reasons.append(reason)
        score = min(1.0, score + weight)

    filename_source = f"{video.stem} {video.name} {video.parent.name}"
    source_texts: list[tuple[str, object]] = [("filename", filename_source)]
    metadata = probe_media_metadata(video)
    for source_name, tag_dict in _collect_pre_asr_hint_tag_dicts(metadata):
        for tag_key, tag_value in tag_dict.items():
            if not str(tag_value or "").strip():
                continue
            source_texts.append((f"{source_name}:{str(tag_key).lower()}", tag_value))

    for source_name, text in source_texts:
        normalized = _normalized_hint_text(text)
        tokens = _hint_tokens(text)
        if "karaoke" in tokens:
            add_reason(f"{source_name}:karaoke", 0.9)
        if "instrumental" in tokens:
            add_reason(f"{source_name}:instrumental", 0.9)
        if "lyric" in tokens or "lyrics" in tokens:
            add_reason(f"{source_name}:lyrics", 0.8)
        if "music video" in normalized:
            add_reason(f"{source_name}:music_video", 0.75)
        if "mv" in tokens:
            add_reason(f"{source_name}:mv", 0.45)
        if "ost" in tokens:
            add_reason(f"{source_name}:ost", 0.45)
        if "concert" in tokens:
            add_reason(f"{source_name}:concert", 0.55)
        if "live" in tokens:
            add_reason(f"{source_name}:live", 0.35)
        if "opening" in tokens or "op" in tokens:
            add_reason(f"{source_name}:opening", 0.3)
        if "ending" in tokens or "ed" in tokens:
            add_reason(f"{source_name}:ending", 0.3)

    for source_name, tag_dict in _collect_pre_asr_hint_tag_dicts(metadata):
        if str(tag_dict.get("artist") or "").strip():
            add_reason(f"{source_name}:artist", 0.35)
        if str(tag_dict.get("album") or "").strip():
            add_reason(f"{source_name}:album", 0.35)

    blocked_reason: str | None = None
    promoted = False
    if reasons:
        if score < _PRE_ASR_MUSIC_PROMOTION_THRESHOLD:
            blocked_reason = "hint_below_promotion_threshold"
        elif not _music_safe_retry_supported(args):
            blocked_reason = "backend_does_not_support_music_preset"
        elif not str(getattr(args, "language", "") or "").strip():
            blocked_reason = "missing_language_for_music_preset"
        else:
            promoted = True

    return {
        "pre_asr_content_hint": "music" if reasons else None,
        "pre_asr_hint_score": round(score, 4),
        "pre_asr_hint_reasons": reasons,
        "pre_asr_promoted": promoted,
        "pre_asr_promotion_blocked_reason": blocked_reason,
    }


def _log_pre_asr_content_hint(prefix: str, hint: dict[str, Any]) -> None:
    if not hint or not hint.get("pre_asr_hint_reasons"):
        return

    score = float(hint.get("pre_asr_hint_score") or 0.0)
    reasons = ", ".join(str(reason) for reason in hint["pre_asr_hint_reasons"])
    if hint.get("pre_asr_promoted"):
        print(
            f"{prefix}[INFO] Pre-ASR music hint promoted auto -> music "
            f"(score={score:.2f}; reasons={reasons}).",
            file=sys.stderr,
            flush=True,
        )
        return

    blocked_reason = str(hint.get("pre_asr_promotion_blocked_reason") or "not_promoted")
    print(
        f"{prefix}[INFO] Pre-ASR music hint recorded without promotion "
        f"(score={score:.2f}; blocked={blocked_reason}; reasons={reasons}).",
        file=sys.stderr,
        flush=True,
    )


def _resolved_postprocess_chunk_size(args: argparse.Namespace) -> int:
    return max(
        1,
        int(
            getattr(args, "postprocess_chunk_size", None)
            or getattr(args, "translation_chunk_size", 100)
        ),
    )


def _stage2_progress_path(artifact_path: Path) -> Path:
    artifact_name = artifact_path.name
    if artifact_name.endswith(".stage1.json"):
        stem = artifact_name[: -len(".stage1.json")]
    else:
        stem = artifact_path.stem
    return artifact_path.with_name(f"{stem}{_STAGE2_PROGRESS_SUFFIX}")


def _load_stage2_progress(
    progress_path: Path, segment_count: int
) -> dict[str, dict[int, str]]:
    progress: dict[str, dict[int, str]] = {"translate": {}, "postprocess": {}}
    if not progress_path.exists():
        return progress

    try:
        raw_lines = progress_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(
            f"[WARN] Unable to read stage2 progress file {progress_path}: {exc}",
            file=sys.stderr,
        )
        return progress

    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                f"[WARN] Ignoring malformed stage2 progress line {line_number} in {progress_path}: {exc}",
                file=sys.stderr,
            )
            continue

        stage = str(record.get("stage") or "")
        if stage not in progress:
            print(
                f"[WARN] Ignoring unknown stage2 progress stage on line {line_number} in {progress_path}: {stage}",
                file=sys.stderr,
            )
            continue

        items = record.get("items")
        if not isinstance(items, list):
            print(
                f"[WARN] Ignoring malformed stage2 progress items on line {line_number} in {progress_path}",
                file=sys.stderr,
            )
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            raw_segment_number = item.get("segment_number")
            if raw_segment_number is None:
                continue
            try:
                segment_number = int(raw_segment_number)
            except (TypeError, ValueError):
                continue
            if segment_number < 1 or segment_number > segment_count:
                continue
            progress[stage][segment_number] = str(item.get("text") or "")

    return progress


def _append_stage2_progress(
    progress_path: Path,
    *,
    stage: str,
    segment_numbers: list[int],
    texts: list[str],
) -> None:
    if len(segment_numbers) != len(texts):
        raise ValueError("stage2 progress append requires matching segment/text counts")

    record = {
        "stage": stage,
        "items": [
            {"segment_number": segment_number, "text": text}
            for segment_number, text in zip(segment_numbers, texts)
        ],
    }
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _clear_stage2_progress(progress_path: Path) -> None:
    try:
        progress_path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        print(
            f"[WARN] Unable to remove stage2 progress file {progress_path}: {exc}",
            file=sys.stderr,
        )


def _expected_translated_outputs(
    source_path: Path,
    output_dir: Path,
    formats: list[str],
    target_language: str,
) -> list[Path]:
    return planned_output_paths(
        source_path,
        frozenset(str(fmt) for fmt in formats),
        output_dir,
        name_suffix=f".{target_language}",
    )


def translation_capable(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "translate_to", None)) and bool(
        (getattr(args, "translation_base_url", None) or "").strip()
    )


def emit_progress_event(event: str, **payload: object) -> None:
    message = {"event": event, **payload}
    print(
        f"{EVENT_PREFIX} {json.dumps(message, ensure_ascii=False, sort_keys=True)}",
        flush=True,
    )


def primary_output_exists(
    video: Path,
    formats: frozenset[str],
    output_dir: Path | None,
) -> bool:
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


def _process_one_inline(
    task: dict[str, str],
    args: argparse.Namespace,
    formats: frozenset[str],
    output_dir: Path | None,
    backend_threads: int,
    worker_id: int = 0,
) -> ProcessResult:
    """Legacy single-pass path: transcription + inline translation, no artifact.

    Activated by VID_TO_SUB_LEGACY_INLINE=1. Mirrors the original process_one()
    behaviour before the stage-split refactor.  Used as a rollback escape hatch.
    """
    video = Path(task["video_path"])
    folder_hash = str(task.get("folder_hash") or hash_video_folder(video.parent))
    folder_path = str(task.get("folder_path") or str(video.parent))
    prefix = f"[{worker_id}] " if args.workers > 1 else ""
    print(f"{prefix}▶  {video}", flush=True)
    started_at = time.monotonic()
    video_duration = probe_media_duration(video)

    emit_progress_event(
        "job_started",
        video_path=str(video),
        worker_id=worker_id,
        folder_hash=folder_hash,
        folder_path=folder_path,
        video_duration=video_duration,
    )

    try:
        segments, info = transcribe(
            video=video,
            backend=args.backend,
            model_name=args.model,
            device=args.device,
            language=args.language,
            content_type=getattr(args, "content_type", "auto"),
            beam_size=args.beam_size,
            compute_type=args.compute_type,
            hf_token=args.hf_token,
            diarize=args.diarize,
            whisper_cpp_model_path=args.whisper_cpp_model_path,
            threads=backend_threads,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"{prefix}[ERROR] Transcription failed: {exc}", file=sys.stderr)
        return ProcessResult(
            success=False,
            video_path=str(video),
            folder_hash=folder_hash,
            folder_path=folder_path,
            worker_id=worker_id,
            error=str(exc),
            stage="transcribe",
            elapsed_sec=round(time.monotonic() - started_at, 3),
        )

    all_paths: list[str] = []
    try:
        written = write_outputs(video, segments, formats, output_dir, info)
        all_paths.extend(str(p) for p in written)
    except Exception as exc:
        print(f"{prefix}[ERROR] Output failed: {exc}", file=sys.stderr)
        return ProcessResult(
            success=False,
            video_path=str(video),
            folder_hash=folder_hash,
            folder_path=folder_path,
            worker_id=worker_id,
            error=str(exc),
            stage="output",
            elapsed_sec=round(time.monotonic() - started_at, 3),
        )

    if translation_capable(args):
        try:
            translated_segments, _tinfo = translate_segments_openai_compatible(
                segments=segments,
                target_language=args.translate_to,
                translation_model=args.translation_model,
                translation_base_url=args.translation_base_url,
                translation_api_key=args.translation_api_key,
                source_language=info.get("language"),
                chunk_size=max(1, int(getattr(args, "translation_chunk_size", 100))),
                max_payload_chars=max(
                    2000, int(getattr(args, "translation_max_payload_chars", 16000))
                ),
                translation_mode=getattr(args, "translation_mode", "strict"),
            )
            if getattr(args, "postprocess_translation", False):
                translated_segments, _ = (
                    postprocess_translated_segments_openai_compatible(
                        source_segments=segments,
                        translated_segments=translated_segments,
                        target_language=args.translate_to,
                        postprocess_mode=getattr(args, "postprocess_mode", "auto"),
                        postprocess_model=getattr(args, "postprocess_model", None),
                        postprocess_base_url=getattr(
                            args, "postprocess_base_url", None
                        ),
                        postprocess_api_key=getattr(args, "postprocess_api_key", None),
                        source_language=info.get("language"),
                        translation_model=args.translation_model,
                        translation_base_url=args.translation_base_url,
                        translation_api_key=args.translation_api_key,
                        chunk_size=_resolved_postprocess_chunk_size(args),
                        max_payload_chars=max(
                            2000,
                            int(getattr(args, "postprocess_max_payload_chars", 12000)),
                        ),
                        translation_mode=getattr(args, "translation_mode", "strict"),
                    )
                )
            translated_written = write_outputs(
                video,
                translated_segments,
                formats,
                output_dir,
                info,
                name_suffix=f".{args.translate_to}",
            )
            all_paths.extend(str(p) for p in translated_written)
        except Exception as exc:
            print(f"{prefix}[ERROR] Translation failed: {exc}", file=sys.stderr)

    elapsed = time.monotonic() - started_at
    language = info.get("language", "?")
    duration = info.get("duration")
    duration_str = f"  video={fmt_seconds(duration)}" if duration else ""
    print(
        f"{prefix}   \u2713 [{language}]{duration_str}  wall={fmt_seconds(elapsed)}  "
        + "  ".join(Path(path).name for path in all_paths),
        flush=True,
    )
    return ProcessResult(
        success=True,
        video_path=str(video),
        folder_hash=folder_hash,
        folder_path=folder_path,
        worker_id=worker_id,
        elapsed_sec=round(elapsed, 3),
        language=language,
        video_duration=float(duration) if isinstance(duration, (int, float)) else None,
        output_paths=all_paths,
        segments=len(segments),
    )


def process_one(
    task: dict[str, str],
    args: argparse.Namespace,
    formats: frozenset[str],
    output_dir: Path | None,
    backend_threads: int,
    worker_id: int = 0,
) -> ProcessResult:
    if _LEGACY_INLINE:
        block_reason = legacy_inline_block_reason(args)
        if block_reason:
            video = Path(task["video_path"])
            folder_hash = str(
                task.get("folder_hash") or hash_video_folder(video.parent)
            )
            folder_path = str(task.get("folder_path") or str(video.parent))
            print(f"[ERROR] {block_reason}", file=sys.stderr, flush=True)
            return ProcessResult(
                success=False,
                video_path=str(video),
                folder_hash=folder_hash,
                folder_path=folder_path,
                worker_id=worker_id,
                error=block_reason,
                stage="legacy_inline",
            )
        emit_legacy_inline_warning_once()
        return _process_one_inline(
            task, args, formats, output_dir, backend_threads, worker_id
        )
    stage1_result = run_stage1(
        task, args, formats, output_dir, backend_threads, worker_id
    )
    can_continue_after_stage1_hold = (
        not stage1_result.success
        and stage1_result.stage == "quality_hold"
        and bool(args.translate_to)
        and bool(stage1_result.artifact_path)
    )
    if not stage1_result.success and not can_continue_after_stage1_hold:
        return stage1_result

    if not args.translate_to:
        return _finalize_process_result(stage1_result, None, args)

    if not stage1_result.artifact_path:
        return ProcessResult(
            success=False,
            video_path=stage1_result.video_path,
            folder_hash=stage1_result.folder_hash,
            folder_path=stage1_result.folder_path,
            worker_id=stage1_result.worker_id,
            error="Stage 1 completed without producing an artifact.",
            stage="artifact",
            elapsed_sec=stage1_result.elapsed_sec,
            language=stage1_result.language,
            video_duration=stage1_result.video_duration,
            output_paths=stage1_result.output_paths,
            segments=stage1_result.segments,
            artifact_path=stage1_result.artifact_path,
            artifact_metadata=stage1_result.artifact_metadata,
        )

    stage2_result = run_stage2(Path(stage1_result.artifact_path), args)
    return _finalize_process_result(stage1_result, stage2_result, args)


def run_stage1(
    task: dict[str, str],
    args: argparse.Namespace,
    formats: frozenset[str],
    output_dir: Path | None,
    backend_threads: int,
    worker_id: int,
) -> ProcessResult:
    video = Path(task["video_path"])
    folder_hash = str(task.get("folder_hash") or hash_video_folder(video.parent))
    folder_path = str(task.get("folder_path") or str(video.parent))
    prefix = f"[{worker_id}] " if args.workers > 1 else ""
    print(f"{prefix}\u25b6  {video}", flush=True)
    started_at = time.monotonic()
    video_duration = probe_media_duration(video)
    _probe_elapsed = time.monotonic() - started_at
    print(
        f"{prefix}[METRIC] stage=probe file={video.name} elapsed_ms={_probe_elapsed * 1000:.1f}",
        file=sys.stderr,
        flush=True,
    )
    last_progress_seconds = -1.0

    def report_progress(progress_seconds: float) -> None:
        nonlocal last_progress_seconds
        if progress_seconds <= last_progress_seconds:
            return
        last_progress_seconds = progress_seconds
        progress_ratio = None
        if video_duration is not None and video_duration > 0:
            progress_ratio = max(0.0, min(1.0, progress_seconds / video_duration))
        emit_progress_event(
            "job_progress",
            video_path=str(video),
            worker_id=worker_id,
            folder_hash=folder_hash,
            folder_path=folder_path,
            video_duration=video_duration,
            progress_seconds=round(progress_seconds, 3),
            progress_ratio=progress_ratio,
        )

    emit_progress_event(
        "job_started",
        video_path=str(video),
        worker_id=worker_id,
        folder_hash=folder_hash,
        folder_path=folder_path,
        video_duration=video_duration,
    )
    pre_asr_hint = predict_auto_content_type(video, args)
    _log_pre_asr_content_hint(prefix, pre_asr_hint)
    initial_content_type = (
        "music"
        if pre_asr_hint.get("pre_asr_promoted")
        else _requested_content_type(args)
    )

    try:
        segments, info = _run_stage1_transcription(
            video=video,
            args=args,
            backend_threads=backend_threads,
            content_type=initial_content_type,
            progress_callback=report_progress
            if args.backend == "whisper-cpp"
            else None,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"{prefix}[ERROR] Transcription failed: {exc}", file=sys.stderr)
        return ProcessResult(
            success=False,
            video_path=str(video),
            folder_hash=folder_hash,
            folder_path=folder_path,
            worker_id=worker_id,
            error=str(exc),
            stage="transcribe",
            elapsed_sec=round(time.monotonic() - started_at, 3),
        )

    _asr_elapsed = time.monotonic() - started_at
    print(
        f"{prefix}[METRIC] stage=asr file={video.name} elapsed_ms={_asr_elapsed * 1000:.1f}",
        file=sys.stderr,
        flush=True,
    )
    segments, info = _refine_stage1_timing(
        video=video,
        segments=segments,
        info=info,
        prefix=prefix,
    )

    if args.backend == "whisper-cpp" and video_duration is not None:
        report_progress(video_duration)
    if pre_asr_hint.get("pre_asr_promoted") and not info.get("content_type"):
        info = dict(info)
        info["content_type"] = "music"

    quality = _assess_stage1_quality(segments, info)
    try:
        segments, info, quality = _maybe_retry_stage1_with_music_preset(
            video=video,
            args=args,
            backend_threads=backend_threads,
            segments=segments,
            info=info,
            quality=quality,
            prefix=prefix,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(
            f"{prefix}[WARN] Music-safe retry failed; keeping the original Stage 1 transcript: {exc}",
            file=sys.stderr,
            flush=True,
        )
        info = dict(info)
        quality = dict(quality)
        retry_details = dict(info.get("auto_music_retry") or {})
        retry_details.update(
            {
                "attempted": True,
                "selected": False,
                "error": str(exc),
            }
        )
        info["auto_music_retry"] = retry_details
        quality["auto_music_retry"] = retry_details

    if args.verbose:
        for segment in segments:
            print(
                f"  {srt_timestamp(segment['start'])} --> {srt_timestamp(segment['end'])}  "
                f"{segment['text'].strip()}"
            )

    try:
        output_base = output_dir if output_dir else video.parent
        canonical_outputs = planned_output_paths(video, formats, output_dir)
        written: list[Path]
        stage1_success = True
        stage1_error: str | None = None
        stage1_stage: str | None = None
        if quality.get("suspicious"):
            written = write_outputs(
                video,
                segments,
                formats,
                output_dir,
                info,
                name_suffix=".stage1.suspicious",
            )
            reasons = [str(reason) for reason in quality.get("reasons") or []]
            stage1_error = _stage1_quality_warning(
                reasons=reasons,
                held_paths=written,
            )
            stage1_success = False
            stage1_stage = "quality_hold"
            print(f"{prefix}[WARN] {stage1_error}", file=sys.stderr, flush=True)
        else:
            written = write_outputs(video, segments, formats, output_dir, info)

        quality = dict(quality)
        if pre_asr_hint:
            quality.update(pre_asr_hint)
        quality["requested_content_type"] = _requested_content_type(args)
        quality["effective_content_type"] = str(
            info.get("content_type") or _requested_content_type(args)
        )
        quality["output_held"] = not stage1_success
        quality["held_output_paths"] = (
            [str(path) for path in written] if not stage1_success else []
        )
        quality["warning"] = stage1_error
        artifact: StageArtifact = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_path": str(video),
            "output_base": str(output_base),
            "source_fingerprint": fingerprint_source_path(video),
            "backend": str(info.get("backend") or args.backend),
            "device": str(info.get("device") or args.device),
            "model": str(info.get("model") or args.model),
            "content_type": str(
                info.get("content_type") or quality["effective_content_type"]
            ),
            "language": info.get("language"),
            "language_probability": (
                round(float(info["language_probability"]), 4)
                if isinstance(info.get("language_probability"), (int, float))
                else None
            ),
            "duration": (
                round(float(info["duration"]), 3)
                if isinstance(info.get("duration"), (int, float))
                else None
            ),
            "quality": quality,
            "target_lang": args.translate_to,
            "formats": sorted(formats),
            "primary_outputs": (
                [str(path) for path in written]
                if stage1_success
                else [str(path) for path in canonical_outputs]
            ),
            "segments": segments,
            "stage_status": {
                "transcription_complete": True,
                "stage1_output_held": not stage1_success,
                "stage1_output_warning": stage1_error,
                "translation_pending": bool(args.translate_to),
                "translation_complete": False,
                "translation_failed": False,
                "translation_error": None,
            },
        }
        artifact_path = write_stage_artifact(artifact, output_dir, video)
        artifact_metadata = build_stage_artifact_metadata(artifact_path, artifact)
    except Exception as exc:
        print(f"{prefix}[ERROR] Output/artifact failed: {exc}", file=sys.stderr)
        return ProcessResult(
            success=False,
            video_path=str(video),
            folder_hash=folder_hash,
            folder_path=folder_path,
            worker_id=worker_id,
            error=str(exc),
            stage="output",
            elapsed_sec=round(time.monotonic() - started_at, 3),
        )

    language = info.get("language", "?")
    duration = info.get("duration")
    elapsed = time.monotonic() - started_at
    return ProcessResult(
        success=stage1_success,
        video_path=str(video),
        folder_hash=folder_hash,
        folder_path=folder_path,
        worker_id=worker_id,
        elapsed_sec=round(elapsed, 3),
        language=language,
        video_duration=float(duration) if isinstance(duration, (int, float)) else None,
        output_paths=[str(path) for path in written],
        segments=len(segments),
        error=stage1_error,
        stage=stage1_stage,
        artifact_path=str(artifact_path),
        artifact_metadata=artifact_metadata,
    )


def run_stage2(artifact_path: Path, args: argparse.Namespace) -> ProcessResult:
    started_at = time.monotonic()

    try:
        artifact = load_stage_artifact(artifact_path)
        artifact_metadata = build_stage_artifact_metadata(artifact_path, artifact)
    except Exception as exc:
        print(f"[ERROR] Stage 2 setup failed: {exc}", file=sys.stderr)
        return ProcessResult(
            success=False,
            video_path=str(artifact_path),
            folder_hash=hash_video_folder(artifact_path.parent),
            folder_path=str(artifact_path.parent),
            worker_id=0,
            error=str(exc),
            stage="translate",
            elapsed_sec=round(time.monotonic() - started_at, 3),
            artifact_path=str(artifact_path),
            artifact_metadata=build_stage_artifact_metadata(artifact_path),
        )

    source_path = Path(artifact["source_path"])
    output_dir = Path(artifact["output_base"])
    progress_path = _stage2_progress_path(artifact_path)
    folder_hash = hash_video_folder(source_path.parent)
    folder_path = str(source_path.parent)
    target_language = args.translate_to or artifact.get("target_lang")
    quality = artifact.get("quality") or {}
    if quality.get("suspicious") and not getattr(args, "force_translate", False):
        reasons = quality.get("reasons") or []
        reason_suffix = (
            f" Reasons: {', '.join(str(reason) for reason in reasons)}."
            if reasons
            else ""
        )
        error = (
            "Stage 1 transcript quality is suspicious; refusing Stage 2 translation "
            "without --force-translate."
            f"{reason_suffix}"
        )
        artifact["stage_status"]["translation_pending"] = False
        artifact["stage_status"]["translation_complete"] = False
        artifact["stage_status"]["translation_failed"] = True
        artifact["stage_status"]["translation_error"] = error
        write_stage_artifact(artifact, output_dir, source_path)
        artifact_metadata = build_stage_artifact_metadata(artifact_path, artifact)
        return ProcessResult(
            success=False,
            video_path=str(source_path),
            folder_hash=folder_hash,
            folder_path=folder_path,
            worker_id=0,
            error=error,
            stage="translate",
            elapsed_sec=round(time.monotonic() - started_at, 3),
            language=artifact.get("language"),
            segments=len(artifact["segments"]),
            artifact_path=str(artifact_path),
            artifact_metadata=artifact_metadata,
        )
    source_verification_error = verify_artifact_source(artifact)
    if source_verification_error and not getattr(args, "force_translate", False):
        error = (
            f"{source_verification_error}. Re-run Stage 1 or pass "
            "--force-translate to bypass replay verification."
        )
        artifact["stage_status"]["translation_pending"] = False
        artifact["stage_status"]["translation_complete"] = False
        artifact["stage_status"]["translation_failed"] = True
        artifact["stage_status"]["translation_error"] = error
        write_stage_artifact(artifact, output_dir, source_path)
        artifact_metadata = build_stage_artifact_metadata(artifact_path, artifact)
        return ProcessResult(
            success=False,
            video_path=str(source_path),
            folder_hash=folder_hash,
            folder_path=folder_path,
            worker_id=0,
            error=error,
            stage="translate",
            elapsed_sec=round(time.monotonic() - started_at, 3),
            language=artifact.get("language"),
            segments=len(artifact["segments"]),
            artifact_path=str(artifact_path),
            artifact_metadata=artifact_metadata,
        )
    if source_verification_error:
        print(
            f"[WARN] {source_verification_error}; proceeding because "
            "--force-translate was set.",
            file=sys.stderr,
        )
    if not target_language:
        error = "Stage artifact does not include a translation target."
        artifact["stage_status"]["translation_pending"] = False
        artifact["stage_status"]["translation_complete"] = False
        artifact["stage_status"]["translation_failed"] = True
        artifact["stage_status"]["translation_error"] = error
        write_stage_artifact(artifact, output_dir, source_path)
        artifact_metadata = build_stage_artifact_metadata(artifact_path, artifact)
        return ProcessResult(
            success=False,
            video_path=str(source_path),
            folder_hash=folder_hash,
            folder_path=folder_path,
            worker_id=0,
            error=error,
            stage="translate",
            elapsed_sec=round(time.monotonic() - started_at, 3),
            language=artifact.get("language"),
            segments=len(artifact["segments"]),
            artifact_path=str(artifact_path),
            artifact_metadata=artifact_metadata,
        )

    # Idempotency guard: skip if translation already completed and outputs exist.
    stage_status = artifact.get("stage_status") or {}
    if getattr(args, "overwrite_translation", False):
        _clear_stage2_progress(progress_path)
    if stage_status.get("translation_complete") and not getattr(
        args, "overwrite_translation", False
    ):
        expected_translated = _expected_translated_outputs(
            source_path,
            output_dir,
            list(artifact.get("formats") or []),
            str(target_language),
        )
        existing_outputs = [path for path in expected_translated if path.exists()]
        if existing_outputs:
            # idempotency: ALL expected translated outputs must exist, not just any.
            all_exist = expected_translated and all(
                ep.exists() for ep in expected_translated
            )
            if all_exist:
                artifact["target_lang"] = str(target_language)
                write_stage_artifact(artifact, output_dir, source_path)
                _clear_stage2_progress(progress_path)
                artifact_metadata = build_stage_artifact_metadata(
                    artifact_path, artifact
                )
                return ProcessResult(
                    success=True,
                    video_path=str(source_path),
                    folder_hash=folder_hash,
                    folder_path=folder_path,
                    worker_id=0,
                    elapsed_sec=round(time.monotonic() - started_at, 3),
                    language=artifact.get("language"),
                    output_paths=[str(ep) for ep in expected_translated],
                    segments=len(artifact["segments"]),
                    artifact_path=str(artifact_path),
                    artifact_metadata=artifact_metadata,
                )

    stage2_progress = _load_stage2_progress(progress_path, len(artifact["segments"]))

    try:
        translated_segments, translation_info = translate_segments_openai_compatible(
            segments=artifact["segments"],
            target_language=target_language,
            translation_model=args.translation_model,
            translation_base_url=args.translation_base_url,
            translation_api_key=args.translation_api_key,
            source_language=artifact.get("language"),
            chunk_size=max(1, int(getattr(args, "translation_chunk_size", 100))),
            max_payload_chars=max(
                2000, int(getattr(args, "translation_max_payload_chars", 16000))
            ),
            translation_mode=getattr(args, "translation_mode", "strict"),
            resume_text_by_number=stage2_progress["translate"],
            on_batch_success=lambda segment_numbers, texts: _append_stage2_progress(
                progress_path,
                stage="translate",
                segment_numbers=segment_numbers,
                texts=texts,
            ),
        )
        if args.postprocess_translation:
            translated_segments, postprocess_info = (
                postprocess_translated_segments_openai_compatible(
                    source_segments=artifact["segments"],
                    translated_segments=translated_segments,
                    target_language=target_language,
                    postprocess_mode=args.postprocess_mode,
                    postprocess_model=args.postprocess_model,
                    postprocess_base_url=args.postprocess_base_url,
                    postprocess_api_key=args.postprocess_api_key,
                    source_language=artifact.get("language"),
                    translation_model=args.translation_model,
                    translation_base_url=args.translation_base_url,
                    translation_api_key=args.translation_api_key,
                    chunk_size=_resolved_postprocess_chunk_size(args),
                    max_payload_chars=max(
                        2000,
                        int(getattr(args, "postprocess_max_payload_chars", 12000)),
                    ),
                    translation_mode=getattr(args, "translation_mode", "strict"),
                    resume_text_by_number=stage2_progress["postprocess"],
                    on_batch_success=lambda segment_numbers, texts: (
                        _append_stage2_progress(
                            progress_path,
                            stage="postprocess",
                            segment_numbers=segment_numbers,
                            texts=texts,
                        )
                    ),
                )
            )
            translation_info["postprocess"] = postprocess_info
        translated_info = {
            "backend": artifact["backend"],
            "device": artifact["device"],
            "model": artifact["model"],
            "language": artifact.get("language"),
            "translation": translation_info,
        }
        translated_written = write_outputs(
            source_path,
            translated_segments,
            frozenset(artifact["formats"]),
            output_dir,
            translated_info,
            name_suffix=f".{target_language}",
        )
        _translation_elapsed = time.monotonic() - started_at
        _video_name = source_path.name
        print(
            f"[METRIC] stage=translation file={_video_name} elapsed_ms={_translation_elapsed * 1000:.1f}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        artifact["stage_status"]["translation_pending"] = False
        artifact["stage_status"]["translation_complete"] = False
        artifact["stage_status"]["translation_failed"] = True
        artifact["stage_status"]["translation_error"] = str(exc)
        write_stage_artifact(artifact, output_dir, source_path)
        artifact_metadata = build_stage_artifact_metadata(artifact_path, artifact)
        print(f"[ERROR] Translation/postprocess failed: {exc}", file=sys.stderr)
        return ProcessResult(
            success=False,
            video_path=str(source_path),
            folder_hash=folder_hash,
            folder_path=folder_path,
            worker_id=0,
            error=str(exc),
            stage="translate",
            elapsed_sec=round(time.monotonic() - started_at, 3),
            language=artifact.get("language"),
            output_paths=list(artifact["primary_outputs"]),
            segments=len(artifact["segments"]),
            artifact_path=str(artifact_path),
            artifact_metadata=artifact_metadata,
        )

    artifact["target_lang"] = str(target_language)
    artifact["stage_status"]["translation_pending"] = False
    artifact["stage_status"]["translation_complete"] = True
    artifact["stage_status"]["translation_failed"] = False
    artifact["stage_status"]["translation_error"] = None
    write_stage_artifact(artifact, output_dir, source_path)
    _clear_stage2_progress(progress_path)
    artifact_metadata = build_stage_artifact_metadata(artifact_path, artifact)

    return ProcessResult(
        success=True,
        video_path=str(source_path),
        folder_hash=folder_hash,
        folder_path=folder_path,
        worker_id=0,
        elapsed_sec=round(time.monotonic() - started_at, 3),
        language=artifact.get("language"),
        output_paths=[str(path) for path in translated_written],
        segments=len(artifact["segments"]),
        artifact_path=str(artifact_path),
        artifact_metadata=artifact_metadata,
    )


def _finalize_process_result(
    stage1_result: ProcessResult,
    stage2_result: ProcessResult | None,
    args: argparse.Namespace,
) -> ProcessResult:
    if stage2_result is not None and not stage2_result.success:
        elapsed = (stage1_result.elapsed_sec or 0.0) + (
            stage2_result.elapsed_sec or 0.0
        )
        return ProcessResult(
            success=False,
            video_path=stage1_result.video_path,
            folder_hash=stage1_result.folder_hash,
            folder_path=stage1_result.folder_path,
            worker_id=stage1_result.worker_id,
            language=stage1_result.language,
            video_duration=stage1_result.video_duration,
            output_paths=(stage1_result.output_paths or [])
            + (stage2_result.output_paths or []),
            segments=stage1_result.segments,
            elapsed_sec=round(elapsed, 3),
            error=stage2_result.error,
            stage=stage2_result.stage,
            artifact_path=stage1_result.artifact_path,
            artifact_metadata=stage2_result.artifact_metadata
            or stage1_result.artifact_metadata,
        )

    elapsed = stage1_result.elapsed_sec or 0.0
    output_paths = list(stage1_result.output_paths or [])
    if stage2_result is not None:
        elapsed += stage2_result.elapsed_sec or 0.0
        output_paths.extend(stage2_result.output_paths or [])

    prefix = f"[{stage1_result.worker_id}] " if args.workers > 1 else ""
    language = stage1_result.language or "?"
    duration = stage1_result.video_duration
    duration_str = f"  video={fmt_seconds(duration)}" if duration else ""
    print(
        f"{prefix}   ✓ [{language}]{duration_str}  wall={fmt_seconds(elapsed)}  "
        + "  ".join(Path(path).name for path in output_paths),
        flush=True,
    )
    return ProcessResult(
        success=True,
        video_path=stage1_result.video_path,
        folder_hash=stage1_result.folder_hash,
        folder_path=stage1_result.folder_path,
        worker_id=stage1_result.worker_id,
        elapsed_sec=round(elapsed, 3),
        language=stage1_result.language,
        video_duration=stage1_result.video_duration,
        output_paths=output_paths,
        segments=stage1_result.segments,
        artifact_path=stage1_result.artifact_path,
        artifact_metadata=(
            stage2_result.artifact_metadata
            if stage2_result is not None
            else stage1_result.artifact_metadata
        ),
    )


def _warn_model_memory_if_needed(args: argparse.Namespace, worker_total: int) -> None:
    """Print a warning when multiple workers may each load a separate model copy.

    faster-whisper uses thread-local model caching, so every worker thread loads
    its own model instance.  At worker_total > 1 the VRAM / RAM footprint
    multiplies linearly, which can silently exhaust available memory.
    """
    if str(getattr(args, "backend", "") or "") != "faster-whisper":
        return
    if worker_total < 2:
        return

    model_name = str(getattr(args, "model", "") or "")
    min_vram = FASTER_WHISPER_MODEL_MIN_VRAM_GB.get(model_name)
    if min_vram is not None:
        estimated_total = min_vram * worker_total
        try:
            from vid_to_sub_app.shared.env import detect_cuda_total_memory_gb
            available = detect_cuda_total_memory_gb()
        except Exception:
            available = None
        if available is not None and estimated_total > available:
            print(
                f"[WARN] faster-whisper: {worker_total} workers × ~{min_vram:.0f} GiB per model"
                f" ≈ {estimated_total:.0f} GiB estimated; only {available:.1f} GiB total VRAM detected"
                " (actual free VRAM will be less). Consider --workers 1 or a smaller --model to avoid OOM errors.",
                file=sys.stderr,
                flush=True,
            )
            return
    # Generic warning when no VRAM data available
    print(
        f"[WARN] faster-whisper: {worker_total} workers will each load a separate model copy"
        " (thread-local cache). Memory usage scales with worker count.",
        file=sys.stderr,
        flush=True,
    )


def _run_worker_loop(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    formats: frozenset[str],
    output_dir: Path | None,
    work_fn: Callable[..., ProcessResult],
    *,
    collect_artifacts: bool = False,
    count_suspicious_holds_as_errors: bool = True,
) -> tuple[int, int, int, list[str]]:
    """Unified parallel worker loop shared by run_parallel and _run_stage1_parallel.

    Parameters
    ----------
    work_fn:
        Callable with signature ``(task, args, formats, output_dir, backend_threads,
        worker_id) -> ProcessResult``.  Typically ``process_one`` or ``run_stage1``.
    collect_artifacts:
        When True, ``result.artifact_path`` values are accumulated and returned
        as the third element of the tuple (used by the stage-1-only path).

    Returns
    -------
    (ok, err, suspicious_holds, artifact_paths)

    ``err`` can exclude ``quality_hold`` results when
    ``count_suspicious_holds_as_errors`` is ``False``.
    """
    from concurrent.futures import ThreadPoolExecutor

    scheduler = FolderAwareScheduler(manifest)
    backend_threads = max(2, int(args.backend_threads))
    worker_total = max(1, int(args.workers))
    _warn_model_memory_if_needed(args, worker_total)
    counter_lock = threading.Lock()
    ok = 0
    err = 0
    suspicious_holds = 0
    artifact_paths: list[str] = []

    def worker_loop(worker_id: int) -> None:
        nonlocal ok, err, suspicious_holds
        while True:
            task = scheduler.claim_next()
            if task is None:
                return
            try:
                result = work_fn(
                    task,
                    args,
                    formats,
                    output_dir,
                    backend_threads,
                    worker_id,
                )
            except Exception as exc:
                result = ProcessResult(
                    success=False,
                    video_path=str(task["video_path"]),
                    folder_hash=str(task["folder_hash"]),
                    folder_path=str(task["folder_path"]),
                    worker_id=worker_id,
                    error=str(exc),
                    stage="worker",
                )
                print(f"[ERROR] Worker exception: {exc}", file=sys.stderr)

            folder_snapshot = scheduler.complete(result)
            emit_progress_event(
                "job_finished",
                video_path=result.video_path,
                worker_id=result.worker_id,
                status="done" if result.success else "failed",
                stage=result.stage,
                error=result.error,
                elapsed_sec=result.elapsed_sec,
                language=result.language,
                video_duration=result.video_duration,
                output_paths=result.output_paths or [],
                segments=result.segments,
                artifact_path=result.artifact_path,
                artifact_metadata=result.artifact_metadata,
                folder_hash=folder_snapshot["folder_hash"],
                folder_path=folder_snapshot["folder_path"],
                folder_total_files=folder_snapshot["total_files"],
                folder_completed_files=folder_snapshot["completed_files"],
                folder_status=folder_snapshot["status"],
                folder_completed=folder_snapshot["is_completed"],
            )
            if folder_snapshot["is_completed"]:
                emit_progress_event(
                    "folder_finished",
                    folder_hash=folder_snapshot["folder_hash"],
                    folder_path=folder_snapshot["folder_path"],
                    total_files=folder_snapshot["total_files"],
                    folder_total_files=folder_snapshot["total_files"],
                    folder_completed_files=folder_snapshot["completed_files"],
                    folder_status=folder_snapshot["status"],
                    folder_completed=folder_snapshot["is_completed"],
                )

            with counter_lock:
                if collect_artifacts and result.artifact_path:
                    artifact_paths.append(result.artifact_path)
                if result.success:
                    ok += 1
                elif (
                    result.stage == "quality_hold" and not count_suspicious_holds_as_errors
                ):
                    suspicious_holds += 1
                else:
                    err += 1

    with ThreadPoolExecutor(max_workers=worker_total) as pool:
        futures = [
            pool.submit(worker_loop, worker_id) for worker_id in range(worker_total)
        ]
        for future in futures:
            future.result()

    artifact_paths.sort()
    return ok, err, suspicious_holds, artifact_paths


def run_parallel(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    formats: frozenset[str],
    output_dir: Path | None,
) -> tuple[int, int]:
    ok, err, _, _ = _run_worker_loop(
        manifest, args, formats, output_dir, process_one, collect_artifacts=False
    )
    return ok, err
