from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from vid_to_sub_app.shared.constants import (
    DEFAULT_FORMAT,
    DEFAULT_MODEL,
    ENV_TRANSLATION_API_KEY,
    ENV_TRANSLATION_BASE_URL,
    ENV_TRANSLATION_MODEL,
    ENV_WHISPER_CPP_MODEL,
    KNOWN_MODELS,
    POSTPROCESS_MODES,
    SUPPORTED_FORMATS,
    TRANSLATION_MODES,
)
from vid_to_sub_app.shared.env import (
    load_env_from_sqlite,
    load_project_env_fallback,
    resolve_runtime_backend_and_device,
    resolve_runtime_backend_threads,
    resolve_runtime_model,
)
from vid_to_sub_app.shared.secrets import hydrate_secret_env

from .discovery import discover_videos, hash_video_folder
from .manifest import (
    FolderAwareScheduler,
    ProcessResult,
    build_run_manifest,
    load_manifest_from_stdin,
    persist_folder_manifest_state,
)
from .output import fmt_seconds
from .runner import (
    emit_legacy_inline_warning_once,
    emit_progress_event,
    legacy_inline_block_reason,
    legacy_inline_requested,
    primary_output_exists,
    run_parallel,
    run_stage1,
    run_stage2,
)
from .stage_artifact import artifact_path_for, load_stage_artifact


def _emit_stage2_job_events(
    artifact_path: Path,
    result: ProcessResult,
) -> None:
    source_path = Path(result.video_path)
    emit_progress_event(
        "job_finished",
        video_path=str(source_path),
        worker_id=result.worker_id,
        status="done" if result.success else "failed",
        stage=result.stage,
        error=result.error,
        elapsed_sec=result.elapsed_sec,
        language=result.language,
        video_duration=result.video_duration,
        output_paths=result.output_paths or [],
        segments=result.segments,
        artifact_path=result.artifact_path or str(artifact_path),
        artifact_metadata=result.artifact_metadata,
        folder_hash=result.folder_hash,
        folder_path=result.folder_path,
        folder_total_files=1,
        folder_completed_files=1,
        folder_status="completed" if result.success else "failed",
        folder_completed=True,
    )
    emit_progress_event(
        "folder_finished",
        folder_hash=result.folder_hash,
        folder_path=result.folder_path,
        total_files=1,
        folder_total_files=1,
        folder_completed_files=1,
        folder_status="completed" if result.success else "failed",
        folder_completed=True,
    )


def build_parser() -> argparse.ArgumentParser:
    runtime_default_backend, runtime_default_device = (
        resolve_runtime_backend_and_device()
    )
    runtime_default_model = resolve_runtime_model(
        runtime_default_backend,
        runtime_default_device,
        DEFAULT_MODEL,
    )
    parser = argparse.ArgumentParser(
        prog="vid_to_sub",
        description="Recursively transcribe video files to subtitle/transcript files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="Input video files or directories to scan recursively.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        metavar="DIR",
        default=None,
        help=(
            "Write output files to DIR instead of next to the source video. "
            "Directory will be created if it does not exist."
        ),
    )
    parser.add_argument(
        "--no-recurse",
        action="store_true",
        default=False,
        help="Do not recurse into subdirectories; only process the top-level paths given.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip videos that already have a primary output file (checked by filename).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the job queue without running any transcription or translation.",
    )
    parser.add_argument(
        "--backend",
        choices=["whisper-cpp", "faster-whisper", "whisper", "whisperx"],
        default=runtime_default_backend,
        help="Transcription backend to use. Defaults to the fastest available backend on the current hardware.",
    )
    parser.add_argument(
        "--model",
        default=runtime_default_model,
        metavar="MODEL",
        help="Model name, e.g. large-v3, medium, small. Defaults to the best model that fits detected VRAM.",
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="Source audio language code (e.g. en, ja). Auto-detected when omitted.",
    )
    parser.add_argument(
        "--content-type",
        choices=["auto", "speech", "music"],
        default="auto",
        help=(
            "Audio content hint for transcription. Use 'music' for lyric-heavy or "
            "music-dominant inputs; this enables safer decoding on supported backends "
            "and requires --language. 'auto' now retries suspicious Stage 1 outputs "
            "with the music preset when possible and withholds canonical outputs if "
            "the result still looks unsafe."
        ),
    )
    parser.add_argument(
        "--device",
        default=runtime_default_device,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device. 'auto' selects CUDA > MPS > CPU in that priority order.",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        metavar="TYPE",
        help="Quantisation type passed to faster-whisper, e.g. int8, float16. Backend-specific.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        metavar="N",
        help="Beam width used by faster-whisper and openai-whisper. Larger values improve accuracy at the cost of speed.",
    )
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=sorted(SUPPORTED_FORMATS),
        metavar="FMT",
        help="Output format(s) to write. May be repeated. Choices: "
        + ", ".join(sorted(SUPPORTED_FORMATS))
        + ". Defaults to srt.",
    )
    parser.add_argument(
        "--whisper-cpp-model-path",
        default=None,
        metavar="PATH",
        help="Explicit path to the GGML model file for the whisper-cpp backend.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        metavar="TOKEN",
        help="Hugging Face token required for whisperX diarization models.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        default=False,
        help="Enable speaker diarization via whisperX (requires --hf-token).",
    )
    parser.add_argument(
        "--translate-to",
        default=None,
        metavar="LANG",
        help="Translate subtitles to this language code (e.g. ko, fr) using an OpenAI-compatible API.",
    )
    parser.add_argument(
        "--translation-model",
        default=None,
        metavar="MODEL",
        help="Model name for the translation API. Falls back to VID_TO_SUB_TRANSLATION_MODEL env var.",
    )
    parser.add_argument(
        "--translation-base-url",
        default=None,
        metavar="URL",
        help="Base URL for the OpenAI-compatible translation API (e.g. https://host/v1).",
    )
    parser.add_argument(
        "--translation-api-key",
        default=None,
        metavar="KEY",
        help="Bearer token for the translation API. Falls back to VID_TO_SUB_TRANSLATION_API_KEY.",
    )
    parser.add_argument(
        "--translation-chunk-size",
        type=int,
        default=100,
        metavar="N",
        help="Maximum subtitle items per translation batch before contract-aware retries split the work.",
    )
    parser.add_argument(
        "--postprocess-chunk-size",
        type=int,
        default=None,
        metavar="N",
        help="Maximum subtitle items per postprocess batch. Defaults to --translation-chunk-size.",
    )
    parser.add_argument(
        "--translation-max-payload-chars",
        type=int,
        default=16000,
        metavar="N",
        help="Approximate character budget per translation batch payload before adaptive splitting.",
    )
    parser.add_argument(
        "--postprocess-max-payload-chars",
        type=int,
        default=12000,
        metavar="N",
        help="Approximate character budget per postprocess batch payload before adaptive splitting.",
    )
    parser.add_argument(
        "--translation-mode",
        choices=TRANSLATION_MODES,
        default="strict",
        metavar="MODE",
        help="Translation contract mode. 'strict' fails on malformed batches; 'best-effort' retries smaller batches before giving up.",
    )
    parser.add_argument(
        "--postprocess-translation",
        action="store_true",
        default=False,
        help="Run a second agent pass to correct and polish the translated subtitles.",
    )
    parser.add_argument(
        "--postprocess-mode",
        choices=POSTPROCESS_MODES,
        default="auto",
        metavar="MODE",
    )
    parser.add_argument(
        "--postprocess-model",
        default=None,
        metavar="MODEL",
        help="Model for the post-editing pass. Falls back to the translation model when omitted.",
    )
    parser.add_argument(
        "--postprocess-base-url",
        default=None,
        metavar="URL",
        help="Base URL for the post-editing API. Falls back to the translation base URL when omitted.",
    )
    parser.add_argument(
        "--postprocess-api-key",
        default=None,
        metavar="KEY",
        help="API key for the post-editing pass. Falls back to the translation API key when omitted.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel transcription workers.",
    )
    parser.add_argument(
        "--backend-threads",
        type=int,
        default=None,
        metavar="N",
        help="CPU threads allocated per worker. Defaults to all available threads divided across workers.",
    )
    parser.add_argument(
        "--manifest-stdin",
        action="store_true",
        default=False,
        help="Read a JSON job manifest from stdin instead of discovering videos from PATH arguments.",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        default=False,
        help="Transcribe only and write a .stage1.json artifact next to each output. Use --translate-from-artifact later to translate without re-transcribing.",
    )
    parser.add_argument(
        "--translate-from-artifact",
        metavar="PATH",
        default=None,
        help="Translate a previously written .stage1.json artifact (Stage 2 only). Uses the artifact's saved target language unless you override it with --translate-to.",
    )
    parser.add_argument(
        "--force-translate",
        action="store_true",
        default=False,
        help="Allow Stage 2 translation to run even when the Stage 1 artifact is flagged as suspicious or its source fingerprint cannot be verified.",
    )
    parser.add_argument(
        "--overwrite-translation",
        action="store_true",
        default=False,
        help="Re-run Stage 2 even if the artifact already records translation_complete=True.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print extra progress detail during transcription.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        default=False,
        help="Print built-in model identifiers and exit.",
    )

    return parser


def _run_stage1_parallel(
    manifest: dict[str, object],
    args: argparse.Namespace,
    formats: frozenset[str],
    output_dir: Path | None,
) -> tuple[int, int, list[str]]:
    from concurrent.futures import ThreadPoolExecutor

    scheduler = FolderAwareScheduler(manifest)
    backend_threads = max(2, int(args.backend_threads))
    counter_lock = threading.Lock()
    ok = 0
    err = 0
    artifact_paths: list[str] = []

    def worker_loop(worker_id: int) -> None:
        nonlocal ok, err
        while True:
            task = scheduler.claim_next()
            if task is None:
                return
            try:
                result = run_stage1(
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
                if result.artifact_path:
                    artifact_paths.append(result.artifact_path)
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

    artifact_paths.sort()
    return ok, err, artifact_paths


def _print_dry_run_plan(
    entries: list[dict[str, str]],
    args: argparse.Namespace,
    output_dir: Path | None,
) -> None:
    for entry in entries:
        video_path = Path(entry["video_path"])
        if args.translate_to and not args.stage1_only:
            print(f"  Stage 1: {video_path}")
            print(f"  Stage 2: {artifact_path_for(video_path, output_dir)}")
            continue
        if args.stage1_only:
            print(f"  Stage 1: {video_path}")
            continue
        print(f"  {video_path}")


def main(argv: Optional[list[str]] = None) -> int:
    bootstrap_env_db = None
    try:
        from vid_to_sub_app.db import Database

        bootstrap_env_db = Database()
        load_project_env_fallback(override=False)
        hydrate_secret_env()
        bootstrap_env_db.migrate_secret_settings_to_env()
        load_env_from_sqlite(bootstrap_env_db.get_all, override=True)
    except Exception:
        load_project_env_fallback(override=False)
        hydrate_secret_env()
    finally:
        if bootstrap_env_db is not None:
            bootstrap_env_db.close()

    parser = build_parser()
    args = parser.parse_args(argv)

    artifact_target_lang: str | None = None

    if args.stage1_only and args.translate_from_artifact:
        parser.error(
            "--stage1-only and --translate-from-artifact are mutually exclusive."
        )

    if args.stage1_only and args.translate_to:
        # Keep args.translate_to so run_stage1() records target_lang in the artifact.
        # Stage1-only never executes translation — the artifact carries the intent
        # forward so a later --translate-from-artifact run can proceed without
        # the user needing to re-specify --translate-to.
        print(
            "[INFO] --stage1-only: translation will not run now; "
            "target_lang recorded in artifact for later stage-2 replay.",
            file=sys.stderr,
        )

    if args.translate_from_artifact and not args.translate_to:
        try:
            artifact_target_lang = load_stage_artifact(
                Path(args.translate_from_artifact)
            ).get("target_lang")
        except Exception as exc:
            parser.error(
                "--translate-from-artifact requires --translate-to or an artifact "
                f"with target_lang: {exc}"
            )
        if not artifact_target_lang:
            parser.error(
                "--translate-from-artifact requires --translate-to or an artifact "
                "with target_lang."
            )

    effective_translate_to = args.translate_to or artifact_target_lang
    if (
        legacy_inline_requested()
        and not args.stage1_only
        and not args.translate_from_artifact
    ):
        legacy_inline_error = legacy_inline_block_reason(args)
        if legacy_inline_error:
            print(f"[ERROR] {legacy_inline_error}", file=sys.stderr)
            return 2
        emit_legacy_inline_warning_once()

    if (
        not args.stage1_only
        and args.postprocess_translation
        and not effective_translate_to
    ):
        parser.error("--postprocess-translation requires --translate-to.")

    if args.list_models:
        print("Known model identifiers:")
        for model in KNOWN_MODELS:
            print(f"  {model}")
        return 0

    raw_formats = args.formats or [DEFAULT_FORMAT]
    formats: frozenset[str] = frozenset(raw_formats)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    args.workers = max(1, int(args.workers))
    args.translation_chunk_size = max(1, int(args.translation_chunk_size))
    args.postprocess_chunk_size = max(
        1,
        int(args.postprocess_chunk_size or args.translation_chunk_size),
    )
    args.translation_max_payload_chars = max(
        2000,
        int(args.translation_max_payload_chars),
    )
    args.postprocess_max_payload_chars = max(
        2000,
        int(args.postprocess_max_payload_chars),
    )
    if args.backend_threads is None:
        args.backend_threads = resolve_runtime_backend_threads(
            args.backend,
            args.device,
            args.workers,
        )
    else:
        args.backend_threads = max(1, int(args.backend_threads))

    if args.translate_from_artifact:
        artifact_path = Path(args.translate_from_artifact).expanduser().resolve()
        stage2_source_path: Path | None = None
        try:
            stage2_artifact = load_stage_artifact(artifact_path)
            source_raw = str(stage2_artifact.get("source_path") or "").strip()
            if source_raw:
                stage2_source_path = Path(source_raw).expanduser().resolve()
        except Exception:
            stage2_source_path = None
        emit_progress_event(
            "queue_prepared",
            backend=args.backend,
            model=args.model,
            dry_run=bool(args.dry_run),
            total_found=1,
            total_queued=1,
            skipped=0,
            workers=1,
            backend_threads=args.backend_threads,
        )
        if args.dry_run:
            target_language = args.translate_to or artifact_target_lang or "?"
            print("Found 1 video file(s).", flush=True)
            print(f"  Stage 2: {artifact_path} -> {target_language}")
            emit_progress_event("run_finished", total=1, ok=0, err=0, elapsed_sec=0.0)
            return 0

        started_at = time.monotonic()
        if stage2_source_path is not None:
            emit_progress_event(
                "job_started",
                video_path=str(stage2_source_path),
                worker_id=0,
                folder_hash=hash_video_folder(stage2_source_path.parent),
                folder_path=str(stage2_source_path.parent),
            )
        result = run_stage2(artifact_path, args)
        _emit_stage2_job_events(artifact_path, result)
        elapsed_total = time.monotonic() - started_at
        ok = 1 if result.success else 0
        err = 0 if result.success else 1
        emit_progress_event(
            "run_finished",
            total=1,
            ok=ok,
            err=err,
            elapsed_sec=round(elapsed_total, 3),
        )
        if result.success and result.output_paths:
            print(
                "   ✓ " + "  ".join(Path(path).name for path in result.output_paths),
                flush=True,
            )
        print(
            f"\nDone. {ok} succeeded, {err} failed. "
            f"Total wall time: {fmt_seconds(elapsed_total)}",
            flush=True,
        )
        return 0 if err == 0 else 1

    if args.manifest_stdin:
        try:
            manifest = load_manifest_from_stdin()
        except ValueError as exc:
            print(f"[ERROR] {exc}")
            return 2
    else:
        if not args.paths:
            parser.error("PATH is required unless --manifest-stdin is used.")
        recursive = not args.no_recurse
        videos = discover_videos(args.paths, recursive=recursive)
        found_total = len(videos)
        skipped = 0

        if not videos:
            print("[WARN] No video files found.")
            return 1

        if args.skip_existing:
            before = len(videos)
            videos = [
                video
                for video in videos
                if not primary_output_exists(video, formats, output_dir)
            ]
            skipped = before - len(videos)
            if skipped:
                print(f"Skipping {skipped} already-processed file(s).")

        manifest = build_run_manifest(videos, found_total=found_total, skipped=skipped)

    persist_folder_manifest_state(manifest)
    entries = manifest.get("entries", [])
    total_queued = len(entries)
    total_found = int(manifest.get("found_total", total_queued))
    skipped = int(manifest.get("skipped", 0))

    emit_progress_event(
        "queue_prepared",
        backend=args.backend,
        model=args.model,
        dry_run=bool(args.dry_run),
        total_found=total_found,
        total_queued=total_queued,
        skipped=skipped,
        workers=max(1, args.workers),
        backend_threads=args.backend_threads,
    )

    print(f"Found {total_queued} video file(s).", flush=True)

    if args.dry_run:
        dry_run_entries = [entry for entry in entries if isinstance(entry, dict)]
        _print_dry_run_plan(dry_run_entries, args, output_dir)
        emit_progress_event(
            "run_finished", total=total_queued, ok=0, err=0, elapsed_sec=0.0
        )
        return 0

    if not entries:
        print("Nothing to do.")
        emit_progress_event("run_finished", total=0, ok=0, err=0, elapsed_sec=0.0)
        return 0

    started_at = time.monotonic()
    artifact_paths: list[str] = []
    if args.stage1_only:
        ok, err, artifact_paths = _run_stage1_parallel(
            manifest, args, formats, output_dir
        )
    else:
        ok, err = run_parallel(manifest, args, formats, output_dir)
    elapsed_total = time.monotonic() - started_at
    emit_progress_event(
        "run_finished",
        total=total_queued,
        ok=ok,
        err=err,
        elapsed_sec=round(elapsed_total, 3),
    )
    if args.stage1_only and artifact_paths:
        print("\nStage 1 artifacts:", flush=True)
        for artifact_path in artifact_paths:
            print(f"  {artifact_path}", flush=True)
    print(
        f"\nDone. {ok} succeeded, {err} failed. "
        f"Total wall time: {fmt_seconds(elapsed_total)}",
        flush=True,
    )
    return 0 if err == 0 else 1
