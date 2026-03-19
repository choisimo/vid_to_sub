from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from vid_to_sub_app.shared.constants import (
    DEFAULT_FORMAT,
    DEFAULT_MODEL,
    POSTPROCESS_MODES,
    ENV_TRANSLATION_API_KEY,
    ENV_TRANSLATION_BASE_URL,
    ENV_TRANSLATION_MODEL,
    ENV_WHISPER_CPP_MODEL,
    KNOWN_MODELS,
    SUPPORTED_FORMATS,
)
from vid_to_sub_app.shared.env import (
    resolve_runtime_model,
    load_project_env,
    resolve_runtime_backend_and_device,
    resolve_runtime_backend_threads,
)

from .discovery import discover_videos
from .manifest import build_run_manifest, load_manifest_from_stdin, persist_folder_manifest_state
from .output import fmt_seconds
from .runner import emit_progress_event, primary_output_exists, run_parallel


def build_parser() -> argparse.ArgumentParser:
    runtime_default_backend, runtime_default_device = resolve_runtime_backend_and_device()
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

    parser.add_argument("paths", nargs="*", metavar="PATH")
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
    parser.add_argument("--no-recurse", action="store_true", default=False)
    parser.add_argument("--skip-existing", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--backend",
        choices=["whisper-cpp", "faster-whisper", "whisper", "whisperx"],
        default=runtime_default_backend,
    )
    parser.add_argument("--model", default=runtime_default_model, metavar="MODEL")
    parser.add_argument("--language", default=None, metavar="LANG")
    parser.add_argument(
        "--device",
        default=runtime_default_device,
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--compute-type", default=None, metavar="TYPE")
    parser.add_argument("--beam-size", type=int, default=5, metavar="N")
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=sorted(SUPPORTED_FORMATS),
        metavar="FMT",
    )
    parser.add_argument("--whisper-cpp-model-path", default=None, metavar="PATH")
    parser.add_argument("--hf-token", default=None, metavar="TOKEN")
    parser.add_argument("--diarize", action="store_true", default=False)
    parser.add_argument("--translate-to", default=None, metavar="LANG")
    parser.add_argument("--translation-model", default=None, metavar="MODEL")
    parser.add_argument("--translation-base-url", default=None, metavar="URL")
    parser.add_argument("--translation-api-key", default=None, metavar="KEY")
    parser.add_argument("--postprocess-translation", action="store_true", default=False)
    parser.add_argument(
        "--postprocess-mode",
        choices=POSTPROCESS_MODES,
        default="auto",
        metavar="MODE",
    )
    parser.add_argument("--postprocess-model", default=None, metavar="MODEL")
    parser.add_argument("--postprocess-base-url", default=None, metavar="URL")
    parser.add_argument("--postprocess-api-key", default=None, metavar="KEY")
    parser.add_argument("--workers", type=int, default=1, metavar="N")
    parser.add_argument("--backend-threads", type=int, default=None, metavar="N")
    parser.add_argument("--manifest-stdin", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("--list-models", action="store_true", default=False)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    load_project_env(override=False)
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.postprocess_translation and not args.translate_to:
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
    if args.backend_threads is None:
        args.backend_threads = resolve_runtime_backend_threads(
            args.backend,
            args.device,
            args.workers,
        )
    else:
        args.backend_threads = max(1, int(args.backend_threads))

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
        for entry in entries:
            if isinstance(entry, dict):
                print(f"  {entry['video_path']}")
        emit_progress_event("run_finished", total=total_queued, ok=0, err=0, elapsed_sec=0.0)
        return 0

    if not entries:
        print("Nothing to do.")
        emit_progress_event("run_finished", total=0, ok=0, err=0, elapsed_sec=0.0)
        return 0

    started_at = time.monotonic()
    ok, err = run_parallel(manifest, args, formats, output_dir)
    elapsed_total = time.monotonic() - started_at
    emit_progress_event(
        "run_finished",
        total=total_queued,
        ok=ok,
        err=err,
        elapsed_sec=round(elapsed_total, 3),
    )
    print(
        f"\nDone. {ok} succeeded, {err} failed. "
        f"Total wall time: {fmt_seconds(elapsed_total)}",
        flush=True,
    )
    return 0 if err == 0 else 1
