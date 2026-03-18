from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from vid_to_sub_app.shared.constants import EVENT_PREFIX

from .discovery import hash_video_folder
from .manifest import FolderAwareScheduler, ProcessResult
from .output import fmt_seconds, probe_media_duration, srt_timestamp, write_outputs
from .transcription import transcribe
from .translation import translate_segments_openai_compatible


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


def process_one(
    task: dict[str, str],
    args: argparse.Namespace,
    formats: frozenset[str],
    output_dir: Path | None,
    backend_threads: int,
    worker_id: int = 0,
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
            progress_callback=report_progress if args.backend == "whisper-cpp" else None,
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

    elapsed = time.monotonic() - started_at
    language = info.get("language", "?")
    duration = info.get("duration")
    duration_str = f"  video={fmt_seconds(duration)}" if duration else ""
    print(
        f"{prefix}   ✓ [{language}]{duration_str}  wall={fmt_seconds(elapsed)}  "
        + "  ".join(str(path.name) for path in (written + translated_written)),
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
        output_paths=[str(path) for path in written + translated_written],
        segments=len(segments),
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
        futures = [pool.submit(worker_loop, worker_id) for worker_id in range(worker_total)]
        for future in futures:
            future.result()
    return ok, err
