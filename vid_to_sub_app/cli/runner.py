from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from vid_to_sub_app.shared.constants import EVENT_PREFIX

from .discovery import hash_video_folder
from .manifest import FolderAwareScheduler, ProcessResult
from .output import fmt_seconds, probe_media_duration, srt_timestamp, write_outputs
from .stage_artifact import (
    ARTIFACT_SCHEMA_VERSION,
    StageArtifact,
    build_stage_artifact_metadata,
    load_stage_artifact,
    write_stage_artifact,
)
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
        return _process_one_inline(
            task, args, formats, output_dir, backend_threads, worker_id
        )
    stage1_result = run_stage1(
        task, args, formats, output_dir, backend_threads, worker_id
    )
    if not stage1_result.success:
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
    print(f"{prefix}▶  {video}", flush=True)
    started_at = time.monotonic()
    video_duration = probe_media_duration(video)
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
            threads=backend_threads,
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

    if args.backend == "whisper-cpp" and video_duration is not None:
        report_progress(video_duration)

    if args.verbose:
        for segment in segments:
            print(
                f"  {srt_timestamp(segment['start'])} --> {srt_timestamp(segment['end'])}  "
                f"{segment['text'].strip()}"
            )

    try:
        written = write_outputs(video, segments, formats, output_dir, info)
        output_base = output_dir if output_dir else video.parent
        source_stat = video.stat()
        artifact: StageArtifact = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "source_path": str(video),
            "output_base": str(output_base),
            "source_fingerprint": f"{source_stat.st_size}:{int(source_stat.st_mtime)}",
            "backend": str(info.get("backend") or args.backend),
            "device": str(info.get("device") or args.device),
            "model": str(info.get("model") or args.model),
            "language": info.get("language"),
            "target_lang": args.translate_to,
            "formats": sorted(formats),
            "primary_outputs": [str(path) for path in written],
            "segments": segments,
            "stage_status": {
                "transcription_complete": True,
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
        success=True,
        video_path=str(video),
        folder_hash=folder_hash,
        folder_path=folder_path,
        worker_id=worker_id,
        elapsed_sec=round(elapsed, 3),
        language=language,
        video_duration=float(duration) if isinstance(duration, (int, float)) else None,
        output_paths=[str(path) for path in written],
        segments=len(segments),
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
    folder_hash = hash_video_folder(source_path.parent)
    folder_path = str(source_path.parent)
    target_language = artifact.get("target_lang") or args.translate_to
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
    if stage_status.get("translation_complete") and not getattr(
        args, "overwrite_translation", False
    ):
        existing_outputs = [
            p
            for p in (artifact.get("primary_outputs") or [])
            if Path(str(p))
            .with_name(Path(str(p)).stem + f".{target_language}" + Path(str(p)).suffix)
            .exists()
        ]
        if existing_outputs:
            return ProcessResult(
                success=True,
                video_path=str(source_path),
                folder_hash=folder_hash,
                folder_path=folder_path,
                worker_id=0,
                elapsed_sec=round(time.monotonic() - started_at, 3),
                language=artifact.get("language"),
                output_paths=[
                    str(
                        Path(str(p)).with_name(
                            Path(str(p)).stem
                            + f".{target_language}"
                            + Path(str(p)).suffix
                        )
                    )
                    for p in (artifact.get("primary_outputs") or [])
                ],
                segments=len(artifact["segments"]),
                artifact_path=str(artifact_path),
                artifact_metadata=artifact_metadata,
            )

    try:
        translated_segments, translation_info = translate_segments_openai_compatible(
            segments=artifact["segments"],
            target_language=target_language,
            translation_model=args.translation_model,
            translation_base_url=args.translation_base_url,
            translation_api_key=args.translation_api_key,
            source_language=artifact.get("language"),
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

    artifact["stage_status"]["translation_pending"] = False
    artifact["stage_status"]["translation_complete"] = True
    artifact["stage_status"]["translation_failed"] = False
    artifact["stage_status"]["translation_error"] = None
    write_stage_artifact(artifact, output_dir, source_path)
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


def run_parallel(
    manifest: dict[str, Any],
    args: argparse.Namespace,
    formats: frozenset[str],
    output_dir: Path | None,
) -> tuple[int, int]:
    from concurrent.futures import ThreadPoolExecutor

    scheduler = FolderAwareScheduler(manifest)
    backend_threads = max(2, int(args.backend_threads))
    counter_lock = threading.Lock()
    ok = 0
    err = 0

    def worker_loop(worker_id: int) -> None:
        nonlocal ok, err
        while True:
            task = scheduler.claim_next()
            if task is None:
                return
            try:
                result = process_one(
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
                if result.success:
                    ok += 1
                else:
                    err += 1

    worker_total = max(1, int(args.workers))
    with ThreadPoolExecutor(max_workers=worker_total) as pool:
        futures = [
            pool.submit(worker_loop, worker_id) for worker_id in range(worker_total)
        ]
        for future in futures:
            future.result()
    return ok, err
