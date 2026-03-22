from __future__ import annotations

import argparse
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from textual import work

from vid_to_sub_app.cli import (
    apply_runtime_path_map_to_manifest,
    build_run_manifest,
    discover_videos,
)
from vid_to_sub_app.cli.runner import (
    primary_output_exists as _primary_output_exists,
    translation_capable,
)
from vid_to_sub_app.shared.constants import ROOT_DIR

from ..helpers import (
    DEFAULT_BACKEND,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    ENV_POST_KEY,
    ENV_POST_MOD,
    ENV_POST_URL,
    ENV_TRANS_KEY,
    ENV_TRANS_MOD,
    ENV_TRANS_URL,
    ENV_WCPP_BIN,
    ENV_WCPP_MODEL,
    EXECUTION_MODES,
    ExecutorPlan,
    FORMATS,
    RemoteResourceProfile,
    RunConfig,
    RunJobState,
    _clamp_ratio,
    _colorize,
    _coerce_positive_int,
    _fmt_elapsed,
    _mask,
    _progress_bar_markup_ratio,
    group_paths_by_video_folder,
    map_path_for_remote,
    parse_progress_event,
    partition_folder_groups_by_capacity,
    resolve_runtime_backend_threads,
)
from ..state import db as _db


SCRIPT_PATH = ROOT_DIR / "vid_to_sub.py"


class RunMixin:
    # ── Command builder ───────────────────────────────────────────────────

    def _build_cli_args(
        self,
        paths: Sequence[str] | None,
        *,
        dry_run: bool = False,
        remote_profile: RemoteResourceProfile | None = None,
        config: RunConfig | None = None,
        manifest_stdin: bool = False,
        stage: str = "full",  # "full" | "stage1" | "stage2" (injected by plan builder)
    ) -> list[str]:
        if not paths and not manifest_stdin:
            raise ValueError("No input paths — use Browse tab to select")

        map_path = (
            (lambda raw: map_path_for_remote(raw, remote_profile.path_map))
            if remote_profile
            else (lambda raw: raw)
        )
        cmd: list[str] = [map_path(path) for path in (paths or [])]
        if manifest_stdin:
            cmd.append("--manifest-stdin")

        output_dir = (
            config.output_dir if config else (self._val("inp-output-dir") or None)
        )
        if output_dir:
            v = output_dir
            cmd += ["--output-dir", map_path(v)]

        no_recurse = config.no_recurse if config else self._chk("chk-no-recurse")
        if no_recurse:
            cmd.append("--no-recurse")
        skip_existing = (
            config.skip_existing if config else self._chk("chk-skip-existing")
        )
        if skip_existing:
            cmd.append("--skip-existing")
        if dry_run:
            cmd.append("--dry-run")
        verbose = config.verbose if config else self._chk("chk-verbose")
        if verbose:
            cmd.append("--verbose")

        formats = config.formats if config else self._selected_formats()
        if "all" in formats:
            cmd += ["--format", "all"]
        else:
            fmts = [fmt for fmt in FORMATS if fmt in formats]
            if not fmts:
                fmts = ["srt"]
            for fmt in fmts:
                cmd += ["--format", fmt]

        backend = (
            config.backend if config else self._sel("sel-backend", DEFAULT_BACKEND)
        )
        model = config.model if config else self._sel("sel-model", DEFAULT_MODEL)
        cmd += ["--backend", backend]
        cmd += ["--model", model]
        device = config.device if config else self._sel("sel-device", DEFAULT_DEVICE)
        cmd += ["--device", device]

        language = config.language if config else (self._val("inp-language") or None)
        if language:
            v = language
            cmd += ["--language", v]
        content_type = (
            config.content_type
            if config is not None
            else self._sel("sel-content-type", "auto")
        )
        if content_type and content_type != "auto":
            cmd += ["--content-type", content_type]
        compute_type = (
            config.compute_type if config else (self._val("inp-compute-type") or None)
        )
        if compute_type:
            v = compute_type
            cmd += ["--compute-type", v]

        beam = config.beam_size if config else (self._val("inp-beam-size") or "5")
        if beam != "5":
            cmd += ["--beam-size", beam]

        worker_count = (
            remote_profile.slots
            if remote_profile is not None
            else (
                config.local_workers
                if config is not None
                else _coerce_positive_int(self._val("inp-workers") or "1", default=1)
            )
        )
        if worker_count != 1:
            cmd += ["--workers", str(worker_count)]
        backend_threads = resolve_runtime_backend_threads(backend, device, worker_count)
        cmd += ["--backend-threads", str(backend_threads)]

        whisper_cpp_model_path = (
            config.whisper_cpp_model_path
            if config is not None
            else self._resolved_wcpp_model_path()
        )
        if backend == "whisper-cpp" and whisper_cpp_model_path:
            v = whisper_cpp_model_path
            cmd += ["--whisper-cpp-model-path", map_path(v)]

        force_translate = (
            config.force_translate
            if config is not None
            else self._sw("sw-force-translate")
        )
        if force_translate:
            cmd.append("--force-translate")
        translate_enabled = (
            config.translate_enabled if config is not None else self._sw("sw-translate")
        )
        if translate_enabled:
            translate_to = (
                config.translate_to
                if config is not None
                else (self._val("inp-translate-to") or None)
            )
            translation_base_url = (
                config.translation_base_url
                if config is not None
                else (self._val("inp-trans-url") or None)
            )
            if translation_capable(
                argparse.Namespace(
                    translate_to=translate_to,
                    translation_base_url=translation_base_url,
                )
            ):
                if translate_to:
                    cmd += ["--translate-to", translate_to]
                translation_model = (
                    config.translation_model
                    if config is not None
                    else (self._val("inp-trans-model") or None)
                )
                if translation_model:
                    cmd += ["--translation-model", translation_model]
                if translation_base_url:
                    cmd += ["--translation-base-url", translation_base_url]
                translation_api_key = (
                    config.translation_api_key
                    if config is not None
                    else (self._val("inp-trans-key") or None)
                )
                if translation_api_key:
                    cmd += ["--translation-api-key", translation_api_key]
                # Forward translation batching options
                translation_chunk_size = (
                    config.translation_chunk_size
                    if config is not None
                    else max(
                        1,
                        _coerce_positive_int(
                            self._val("inp-translation-chunk-size") or "100",
                            default=100,
                        ),
                    )
                )
                if translation_chunk_size != 100:
                    cmd += ["--translation-chunk-size", str(translation_chunk_size)]
                translation_mode = (
                    config.translation_mode
                    if config is not None
                    else self._sel("sel-translation-mode", "strict")
                )
                if translation_mode and translation_mode != "strict":
                    cmd += ["--translation-mode", translation_mode]
                postprocess_enabled = (
                    config.postprocess_enabled
                    if config is not None
                    else self._sw("sw-postprocess")
                )
                if postprocess_enabled:
                    cmd.append("--postprocess-translation")
                    postprocess_mode = (
                        config.postprocess_mode
                        if config is not None
                        else self._sel("sel-postprocess-mode", "auto")
                    )
                    if postprocess_mode and postprocess_mode != "auto":
                        cmd += ["--postprocess-mode", postprocess_mode]
                    postprocess_model = (
                        config.postprocess_model
                        if config is not None
                        else (self._val("inp-post-model") or None)
                    )
                    if postprocess_model:
                        cmd += ["--postprocess-model", postprocess_model]
                    postprocess_base_url = (
                        config.postprocess_base_url
                        if config is not None
                        else (self._val("inp-post-url") or None)
                    )
                    if postprocess_base_url:
                        cmd += ["--postprocess-base-url", postprocess_base_url]
                    postprocess_api_key = (
                        config.postprocess_api_key
                        if config is not None
                        else (self._val("inp-post-key") or None)
                    )
                    if postprocess_api_key:
                        cmd += ["--postprocess-api-key", postprocess_api_key]

        diarize = config.diarize if config is not None else self._sw("sw-diarize")
        if diarize:
            cmd.append("--diarize")
        hf_token = (
            config.hf_token
            if config is not None
            else (self._val("inp-hf-token") or None)
        )
        if hf_token:
            v = hf_token
            cmd += ["--hf-token", v]

        # Stage-split injection: override translation behaviour based on plan stage.
        # "stage1" — suppress translation, add --stage1-only so artifact is written.
        # "stage2" — do not pass path args; caller is responsible for
        #           --translate-from-artifact injection.
        if stage == "stage1":
            # Remove any --translate-to that was already added above and add
            # --stage1-only so the process writes .stage1.json artifacts.
            cleaned: list[str] = []
            skip_next = False
            for tok in cmd:
                if skip_next:
                    skip_next = False
                    continue
                if tok == "--translate-to":
                    skip_next = True
                    continue
                cleaned.append(tok)
            cmd = cleaned
            if "--stage1-only" not in cmd:
                cmd.append("--stage1-only")
            # Keep --translate-to so the artifact records target_lang for stage2.
            translate_to = (
                config.translate_to
                if config is not None
                else (self._val("inp-translate-to") or None)
            )
            if translate_to:
                cmd += ["--translate-to", translate_to]

        return cmd

    def _build_cmd(self, dry_run: bool = False) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT_PATH),
            *self._build_cli_args(self._selected_paths, dry_run=dry_run),
        ]

    def _build_stage2_cmd(
        self, artifact_path: str, *, source_path: str, target_lang: str
    ) -> list[str]:
        cli_args = self._build_cli_args([source_path], dry_run=False)
        if cli_args and cli_args[0] == source_path:
            cli_args = cli_args[1:]

        clean: list[str] = []
        skip_next = False
        for tok in cli_args:
            if skip_next:
                skip_next = False
                continue
            if tok in ("--translate-to", "--translate-from-artifact"):
                skip_next = True
                continue
            if tok == "--stage1-only":
                continue
            clean.append(tok)

        clean += [
            "--translate-from-artifact",
            artifact_path,
            "--translate-to",
            target_lang,
        ]
        return [sys.executable, str(SCRIPT_PATH), *clean]

    def _update_cmd_preview(self) -> None:
        try:
            if self._active_worker and self._active_worker.is_running:
                return
            mode = self._sel(
                "sel-execution-mode", _db.get("tui.execution_mode") or "local"
            )
            cmd = self._build_cmd()
            display = " ".join(_mask(cmd[2:]))
            if mode == "distributed" and self._remote_resources:
                self._run_last_shell = (
                    f"[dim]Distributed[/] [cyan]local + {len(self._remote_resources)} remote[/] · "
                    f"{display}"
                )
            else:
                self._run_last_shell = f"[dim]$[/dim] [cyan]vid_to_sub[/cyan] {display}"
            self._refresh_live_panels()
        except ValueError as exc:
            self._run_last_shell = f"[dim]{exc}[/]"
            self._refresh_live_panels()
        except Exception:
            pass

    # ── Run & Kill ────────────────────────────────────────────────────────

    def _selected_formats(self) -> frozenset[str]:
        if self._chk("fmt-all"):
            return frozenset({"all"})
        selected = [fmt for fmt in FORMATS if self._chk(f"fmt-{fmt}")]
        return frozenset(selected or ["srt"])

    def _snapshot_run_config(self, dry_run: bool) -> RunConfig:
        if not self._selected_paths:
            raise ValueError("No input paths — use Browse tab to select")

        self._run_request_id += 1
        translate_enabled = self._sw("sw-translate")
        return RunConfig(
            request_id=self._run_request_id,
            selected_paths=list(self._selected_paths),
            output_dir=self._val("inp-output-dir") or None,
            formats=self._selected_formats(),
            no_recurse=self._chk("chk-no-recurse"),
            skip_existing=self._chk("chk-skip-existing"),
            dry_run=dry_run,
            verbose=self._chk("chk-verbose"),
            backend=self._sel("sel-backend", DEFAULT_BACKEND),
            model=self._sel("sel-model", DEFAULT_MODEL),
            device=self._sel("sel-device", DEFAULT_DEVICE),
            language=self._val("inp-language") or None,
            content_type=self._sel("sel-content-type", "auto"),
            compute_type=self._val("inp-compute-type") or None,
            beam_size=self._val("inp-beam-size") or "5",
            local_workers=_coerce_positive_int(
                self._val("inp-workers") or "1", default=1
            ),
            whisper_cpp_model_path=self._resolved_wcpp_model_path() or None,
            translate_enabled=translate_enabled,
            translate_to=self._val("inp-translate-to") or None,
            force_translate=self._sw("sw-force-translate"),
            translation_model=self._val("inp-trans-model") or None,
            translation_base_url=self._val("inp-trans-url") or None,
            translation_api_key=self._val("inp-trans-key") or None,
            postprocess_enabled=self._sw("sw-postprocess"),
            postprocess_mode=self._sel("sel-postprocess-mode", "auto"),
            postprocess_model=self._val("inp-post-model") or None,
            postprocess_base_url=self._val("inp-post-url") or None,
            postprocess_api_key=self._val("inp-post-key") or None,
            translation_chunk_size=max(
                1,
                _coerce_positive_int(
                    self._val("inp-translation-chunk-size") or "100", default=100
                ),
            ),
            translation_mode=self._sel("sel-translation-mode", "strict"),
            diarize=self._sw("sw-diarize"),
            hf_token=self._val("inp-hf-token") or None,
            execution_mode=self._sel(
                "sel-execution-mode",
                _db.get("tui.execution_mode") or "local",
            ),
            remote_resources=list(self._remote_resources),
            run_env=self._build_run_env(),
        )

    def _discover_videos_for_run(
        self,
        config: RunConfig | None = None,
    ) -> tuple[list[str], int, int]:
        selected_paths = config.selected_paths if config else self._selected_paths
        if not selected_paths:
            raise ValueError("No input paths — use Browse tab to select")

        videos = discover_videos(
            selected_paths,
            recursive=not (
                config.no_recurse if config else self._chk("chk-no-recurse")
            ),
        )
        found_total = len(videos)
        skipped = 0

        if config:
            output_dir = (
                Path(config.output_dir).resolve() if config.output_dir else None
            )
            formats = config.formats
            skip_existing = config.skip_existing
        else:
            output_dir = (
                Path(self._val("inp-output-dir")).resolve()
                if self._val("inp-output-dir")
                else None
            )
            formats = self._selected_formats()
            skip_existing = self._chk("chk-skip-existing")

        if skip_existing:
            filtered = [
                video
                for video in videos
                if not _primary_output_exists(video, formats, output_dir)
            ]
            skipped = found_total - len(filtered)
            videos = filtered

        return [str(video) for video in videos], found_total, skipped

    def _build_run_env(self, config: RunConfig | None = None) -> dict[str, str]:
        if config is not None:
            return dict(config.run_env)

        env = os.environ.copy()
        # Inject SQLite-backed env vars
        env.update(_db.get_env_dict())
        # whisper-cli binary override from Transcribe tab
        if v := self._val("inp-wcpp-bin"):
            env[ENV_WCPP_BIN] = v
        backend = self._sel("sel-backend", DEFAULT_BACKEND)
        if backend == "whisper-cpp":
            resolved_model = self._resolved_wcpp_model_path()
            stored_model = _db.get(ENV_WCPP_MODEL)
            if resolved_model:
                env[ENV_WCPP_MODEL] = resolved_model
            elif stored_model and stored_model in self._detected_ggml_models.values():
                env.pop(ENV_WCPP_MODEL, None)
        # Translation overrides
        for wid, evar in [
            ("inp-trans-url", ENV_TRANS_URL),
            ("inp-trans-key", ENV_TRANS_KEY),
            ("inp-trans-model", ENV_TRANS_MOD),
            ("inp-post-url", ENV_POST_URL),
            ("inp-post-key", ENV_POST_KEY),
            ("inp-post-model", ENV_POST_MOD),
        ]:
            if v := self._val(wid):
                env[evar] = v
        return env

    def _build_remote_command(
        self,
        profile: RemoteResourceProfile,
        paths: Sequence[str] | None,
        *,
        dry_run: bool,
        config: RunConfig | None = None,
        stage: str = "full",
    ) -> list[str]:
        base_env = {
            key: value
            for key, value in self._build_run_env(config).items()
            if key.startswith("VID_TO_SUB_") and value
        }
        mapped_env: dict[str, str] = {}
        for key, value in base_env.items():
            if value.startswith("/") or value.startswith("~"):
                mapped_env[key] = map_path_for_remote(value, profile.path_map)
            else:
                mapped_env[key] = value
        mapped_env.update(profile.env)

        script_path = profile.script_path or str(
            Path(profile.remote_workdir) / "vid_to_sub.py"
        )
        cli_args = self._build_cli_args(
            paths,
            dry_run=dry_run,
            remote_profile=profile,
            config=config,
            manifest_stdin=True,
            stage=stage,
        )
        env_prefix = " ".join(
            f"{key}={shlex.quote(value)}"
            for key, value in sorted(mapped_env.items())
            if value
        )
        remote_parts = [profile.python_bin, script_path, *cli_args]
        remote_command = shlex.join(remote_parts)
        if env_prefix:
            remote_command = env_prefix + " " + remote_command
        remote_command = f"cd {shlex.quote(profile.remote_workdir)} && {remote_command}"
        return profile.ssh_command_prefix() + [remote_command]

    def _build_executor_plans(
        self,
        videos: Sequence[str],
        *,
        dry_run: bool,
        config: RunConfig | None = None,
    ) -> list[ExecutorPlan]:
        local_capacity = (
            config.local_workers
            if config is not None
            else _coerce_positive_int(self._val("inp-workers") or "1", default=1)
        )
        mode = (
            config.execution_mode
            if config is not None
            else self._sel(
                "sel-execution-mode", _db.get("tui.execution_mode") or "local"
            )
        )
        remote_resources = (
            config.remote_resources if config is not None else self._remote_resources
        )
        translate_enabled_flag = (
            config.translate_enabled if config is not None else self._sw("sw-translate")
        )
        translate_to_flag = (
            config.translate_to
            if config is not None
            else (self._val("inp-translate-to") or None)
        )
        translation_base_url_flag = (
            config.translation_base_url
            if config is not None
            else (self._val("inp-trans-url") or None)
        )
        planned_stage = (
            "stage1"
            if translation_capable(
                argparse.Namespace(
                    translate_to=translate_to_flag,
                    translation_base_url=translation_base_url_flag,
                )
            )
            and translate_enabled_flag
            else "full"
        )

        if mode != "distributed" or not remote_resources:
            manifest = build_run_manifest(videos)
            local_cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                *self._build_cli_args(
                    None,
                    dry_run=dry_run,
                    config=config,
                    manifest_stdin=True,
                    stage=planned_stage,
                ),
            ]
            return [
                ExecutorPlan(
                    name="local",
                    kind="local",
                    label="local",
                    cmd=local_cmd,
                    env=self._build_run_env(config),
                    assigned_paths=list(videos),
                    capacity=local_capacity,
                    manifest=manifest,
                    stdin_payload=json.dumps(manifest, ensure_ascii=False),
                    stage=planned_stage,
                )
            ]

        capacities = ["local", local_capacity]
        capacities = [("local", local_capacity)]
        capacities.extend((profile.name, profile.slots) for profile in remote_resources)
        assignments = partition_folder_groups_by_capacity(
            group_paths_by_video_folder(videos),
            capacities,
        )
        plans: list[ExecutorPlan] = []

        local_videos = assignments.get("local") or []
        if local_videos:
            local_manifest = build_run_manifest(local_videos)
            plans.append(
                ExecutorPlan(
                    name="local",
                    kind="local",
                    label="local",
                    cmd=[
                        sys.executable,
                        str(SCRIPT_PATH),
                        *self._build_cli_args(
                            None,
                            dry_run=dry_run,
                            config=config,
                            manifest_stdin=True,
                            stage=planned_stage,
                        ),
                    ],
                    env=self._build_run_env(config),
                    assigned_paths=list(local_videos),
                    capacity=local_capacity,
                    manifest=local_manifest,
                    stdin_payload=json.dumps(local_manifest, ensure_ascii=False),
                    stage=planned_stage,
                )
            )

        for profile in remote_resources:
            assigned = assignments.get(profile.name) or []
            if not assigned:
                continue
            base_manifest = build_run_manifest(assigned)
            remote_manifest = apply_runtime_path_map_to_manifest(
                base_manifest,
                lambda raw: map_path_for_remote(raw, profile.path_map),
            )
            plans.append(
                ExecutorPlan(
                    name=profile.name,
                    kind="remote",
                    label=profile.name,
                    cmd=self._build_remote_command(
                        profile,
                        None,
                        dry_run=dry_run,
                        config=config,
                        stage=planned_stage,
                    ),
                    env=None,
                    assigned_paths=list(assigned),
                    capacity=profile.slots,
                    manifest=remote_manifest,
                    stdin_payload=json.dumps(remote_manifest, ensure_ascii=False),
                    stage=planned_stage,
                )
            )

        return plans

    def _apply_progress_event(self, executor: str, event: dict[str, Any]) -> None:
        event_name = str(event.get("event") or "").strip()
        video_path = str(event.get("video_path") or "").strip()
        key = f"{executor}:{video_path}" if video_path else ""
        folder_hash = str(event.get("folder_hash") or "").strip()
        folder_path = str(event.get("folder_path") or "").strip()

        def event_float(name: str) -> float | None:
            try:
                raw_value = event.get(name)
                return float(raw_value) if raw_value is not None else None
            except (TypeError, ValueError):
                return None

        def sync_folder_state_from_event() -> None:
            if not folder_hash or not folder_path:
                return
            try:
                total_files = int(
                    event.get("folder_total_files", event.get("total_files", 0))
                )
            except (TypeError, ValueError):
                total_files = 0
            try:
                completed_files = int(event.get("folder_completed_files", 0))
            except (TypeError, ValueError):
                completed_files = 0
            status = str(
                event.get("folder_status")
                or ("completed" if event.get("folder_completed") else "running")
            ).strip()
            _db.upsert_folder_queue_state(
                folder_hash,
                folder_path,
                status=status or "running",
                total_files=total_files,
                completed_files=completed_files,
                is_completed=bool(event.get("folder_completed")),
            )

        if event_name == "job_started" and video_path:
            video_duration = event_float("video_duration")
            if key not in self._active_jobs:
                job_id = _db.create_job(
                    video_path=video_path,
                    backend=self._run_backend,
                    model=self._run_model,
                    output_dir=self._run_output_dir,
                    language=self._run_language,
                    target_lang=self._run_target_lang,
                )
                self._active_jobs[key] = RunJobState(
                    video_path=video_path,
                    executor=executor,
                    job_id=job_id,
                    started_at=time.monotonic(),
                    video_duration=video_duration,
                    progress_seconds=0.0 if video_duration is not None else None,
                    progress_ratio=0.0 if video_duration is not None else None,
                )
            else:
                job = self._active_jobs[key]
                job.video_duration = video_duration
                if video_duration is not None and job.progress_seconds is None:
                    job.progress_seconds = 0.0
                if video_duration is not None and job.progress_ratio is None:
                    job.progress_ratio = 0.0
            self._refresh_history()
            self._refresh_live_panels()
            self._show_hist_detail(self._hist_key)
            return

        if event_name == "job_progress" and video_path:
            job = self._active_jobs.get(key)
            if job is None:
                return
            video_duration = event_float("video_duration")
            progress_seconds = event_float("progress_seconds")
            progress_ratio = event_float("progress_ratio")
            if video_duration is not None:
                job.video_duration = video_duration
            if progress_seconds is not None:
                job.progress_seconds = progress_seconds
            if (
                progress_ratio is None
                and job.video_duration
                and job.progress_seconds is not None
            ):
                progress_ratio = job.progress_seconds / job.video_duration
            if progress_ratio is not None:
                job.progress_ratio = _clamp_ratio(progress_ratio)
            self._refresh_history()
            self._refresh_live_panels()
            self._show_hist_detail(self._hist_key)
            return

        if event_name == "job_finished" and video_path:
            sync_folder_state_from_event()
            status = str(event.get("status") or "failed").strip()
            job = self._active_jobs.pop(key, None)
            error = str(event.get("error") or "").strip() or None
            output_paths = event.get("output_paths")
            if not isinstance(output_paths, list):
                output_paths = []
            wall_sec = event_float("elapsed_sec")
            video_dur = event_float("video_duration")
            try:
                raw_segments = event.get("segments")
                segments = int(raw_segments) if raw_segments is not None else None
            except (TypeError, ValueError):
                segments = None
            artifact_metadata = event.get("artifact_metadata")
            if not isinstance(artifact_metadata, dict):
                artifact_metadata = None

            if job and job.job_id is not None:
                _db.finish_job(
                    job.job_id,
                    "done" if status == "done" else "failed",
                    output_paths=[str(path) for path in output_paths],
                    error=error,
                    wall_sec=wall_sec,
                    video_dur=video_dur,
                    segments=segments,
                    artifact_path=event.get("artifact_path") or None,
                    artifact_metadata=artifact_metadata,
                )
            elif status != "done":
                job_id = _db.create_job(
                    video_path=video_path,
                    backend=self._run_backend,
                    model=self._run_model,
                    output_dir=self._run_output_dir,
                    language=self._run_language,
                    target_lang=self._run_target_lang,
                )
                _db.finish_job(
                    job_id,
                    "failed",
                    error=error
                    or f"{executor} failed before job tracking was established.",
                    wall_sec=wall_sec,
                )

            pending = self._pending_paths.get(executor)
            if pending is not None:
                pending.discard(video_path)

            if status == "done":
                self._run_completed += 1
            else:
                self._run_failed += 1
                if error:
                    stage = str(event.get("stage") or "run").strip()
                    self._log(
                        f"[bold red]✕ {executor} [{stage}] {Path(video_path).name}: {error}[/]"
                    )

            self._refresh_history()
            self._refresh_live_panels()
            self._show_hist_detail(self._hist_key)
            return

        if event_name == "folder_finished":
            sync_folder_state_from_event()
            self._refresh_live_panels()
            return

        if event_name == "run_finished" and self._run_dry_run:
            pending = self._pending_paths.get(executor)
            if pending is not None:
                pending.clear()
            self._run_completed = max(
                0,
                self._run_total_queued
                - sum(len(paths) for paths in self._pending_paths.values()),
            )
            self._refresh_live_panels()

    def _finalize_executor_failure(self, plan: ExecutorPlan, rc: int) -> None:
        pending = list(self._pending_paths.pop(plan.label, set()))
        if not pending:
            self._refresh_live_panels()
            return

        for video_path in pending:
            active_key = f"{plan.label}:{video_path}"
            active = self._active_jobs.pop(active_key, None)
            error = f"{plan.label} exited with code {rc}"
            if active and active.job_id is not None:
                _db.finish_job(
                    active.job_id,
                    "failed",
                    error=error,
                )
            else:
                job_id = _db.create_job(
                    video_path=video_path,
                    backend=self._run_backend,
                    model=self._run_model,
                    output_dir=self._run_output_dir,
                    language=self._run_language,
                    target_lang=self._run_target_lang,
                )
                _db.finish_job(
                    job_id,
                    "failed",
                    error=f"{error} before starting the job.",
                )
                error = f"{error} before starting the job."
            self._log(
                f"[bold red]✕ {plan.label} [executor] {Path(video_path).name}: {error}[/]"
            )
            self._run_failed += 1

        self._refresh_history()
        self._refresh_live_panels()

    def _resolve_stage1_artifact_path(self, source_path: str) -> Path:
        from vid_to_sub_app.cli.stage_artifact import artifact_path_for

        source = Path(source_path)
        output_dir = (
            Path(self._run_output_dir).expanduser().resolve()
            if self._run_output_dir
            else None
        )
        candidate = artifact_path_for(source, output_dir)
        if candidate.exists():
            return candidate

        fallback = source.with_suffix(".stage1.json")
        if fallback.exists():
            return fallback

        legacy = source.parent / f"{source.stem}.stage1.json"
        if legacy.exists():
            return legacy

        return candidate

    def _collect_stage1_artifacts(self, plans: list[ExecutorPlan]) -> list[str]:
        artifact_paths: list[str] = []
        for plan in plans:
            if getattr(plan, "stage", "full") != "stage1":
                continue
            for source_path in plan.assigned_paths:
                artifact_path = self._resolve_stage1_artifact_path(source_path)
                if artifact_path.exists():
                    artifact_paths.append(str(artifact_path))
        return sorted(set(artifact_paths))

    def _trigger_translate_from_artifact(
        self, artifact_path: str, target_lang_override: str | None = None
    ) -> None:
        """Start a Stage-2 translation run from a pre-existing stage artifact."""
        try:
            from vid_to_sub_app.cli.stage_artifact import load_stage_artifact

            artifact = load_stage_artifact(Path(artifact_path))
            source_path = str(artifact.get("source_path") or artifact_path)
        except Exception as exc:
            self._log(f"[red]Cannot read artifact {artifact_path}: {exc}[/]")
            return

        target_lang = str(
            target_lang_override
            or artifact.get("target_lang")
            or self._val("inp-translate-to")
            or ""
        ).strip()
        if not target_lang:
            self._log(
                "[yellow]Artifact has no target_lang. "
                "Set a translation target in the form and re-run.[/]"
            )
            return

        cmd = self._build_stage2_cmd(
            artifact_path,
            source_path=source_path,
            target_lang=target_lang,
        )

        self._clear_runtime_logs()
        self._log(
            f"[bold cyan]Stage-2 translation[/] from artifact {Path(artifact_path).name}"
        )
        self._reset_run_state(preserve_shell=True)
        self._run_last_shell = (
            f"[cyan]Stage-2 translate[/] · artifact={Path(artifact_path).name}"
        )
        self._run_total_queued = 1
        self._pending_paths = {"local": {source_path}}
        self._refresh_live_panels()
        from ..helpers import ExecutorPlan

        plan = ExecutorPlan(
            name="local",
            kind="local",
            label="local",
            cmd=cmd,
            env=self._build_run_env(),
            assigned_paths=[source_path],
            capacity=1,
            manifest={"entries": [], "folders": []},
            stdin_payload=None,
            stage="stage2",
        )
        self._active_worker = self._stream([plan])

    def _trigger(self, dry_run: bool = False) -> None:
        try:
            config = self._snapshot_run_config(dry_run)
        except ValueError as exc:
            self._log(f"[bold red]✕ {exc}[/]")
            return

        self._clear_runtime_logs()
        self._log(
            "[bold cyan]Preparing run[/] discovering video files and building execution plan…"
        )

        self._reset_run_state(preserve_shell=True)
        if config.execution_mode == "distributed" and config.remote_resources:
            remote_slots = sum(profile.slots for profile in config.remote_resources)
            self._run_last_shell = (
                f"[cyan]Preparing distributed run[/] · local={config.local_workers} worker(s) "
                f"· remote={remote_slots} slot(s)"
            )
        else:
            self._run_last_shell = (
                f"[cyan]Preparing local run[/] · {config.local_workers} worker(s)"
            )
        self._refresh_live_panels()
        self._active_worker = self._prepare_run(config)

    def _abort_prepared_run(self, request_id: int, message: str) -> None:
        if request_id != self._run_request_id:
            return
        self._log(message)
        self._reset_run_state()
        self._refresh_live_panels()

    def _start_prepared_run(
        self,
        config: RunConfig,
        videos: list[str],
        found_total: int,
        skipped: int,
        plans: list[ExecutorPlan],
    ) -> None:
        if config.request_id != self._run_request_id:
            return

        self._clear_runtime_logs()
        if config.execution_mode == "distributed" and len(plans) > 1:
            self._log(
                f"[bold cyan]Distributed run[/] {len(videos)} video(s) across {len(plans)} executor(s)"
            )
        elif plans[0].kind == "remote":
            self._log(
                f"[bold cyan]Remote run[/] {len(videos)} video(s) via {plans[0].label}"
            )
        else:
            self._log(
                "[bold cyan]$ " + " ".join(_mask(plans[0].cmd[2:])) + "[/bold cyan]"
            )
        for plan in plans:
            self._log(
                f"[dim]{plan.label}[/] {len(plan.assigned_paths)} file(s) · capacity={plan.capacity}"
            )

        self._reset_run_state(preserve_shell=True)
        self._run_started_at = time.monotonic()
        self._run_total_found = found_total
        self._run_total_queued = len(videos)
        self._run_skipped = skipped
        self._run_backend = config.backend
        self._run_model = config.model
        self._run_language = config.language
        self._run_target_lang = (
            config.translate_to if config.translate_enabled else None
        )
        self._run_output_dir = config.output_dir
        self._run_dry_run = config.dry_run
        self._pending_paths = {plan.label: set(plan.assigned_paths) for plan in plans}
        queued_folders: list[dict[str, Any]] = []
        for plan in plans:
            for folder in plan.manifest.get("folders", []):
                if not isinstance(folder, dict):
                    continue
                queued_folders.append(
                    {
                        "folder_hash": str(folder.get("folder_hash") or ""),
                        "folder_path": str(folder.get("folder_path") or ""),
                        "status": "queued",
                        "total_files": int(folder.get("total_files", 0)),
                        "completed_files": 0,
                        "is_completed": False,
                    }
                )
        _db.upsert_folder_queue_states(queued_folders)
        if config.execution_mode == "distributed" and len(plans) > 1:
            remote_slots = sum(plan.capacity for plan in plans if plan.kind == "remote")
            self._run_last_shell = (
                f"[cyan]Distributed[/] {len(videos)} file(s) · local={config.local_workers} worker(s) "
                f"· remote={remote_slots} slot(s)"
            )
        elif plans[0].kind == "remote":
            self._run_last_shell = (
                f"[cyan]Remote[/] {plans[0].label} · {len(videos)} file(s)"
            )
        else:
            self._run_last_shell = f"[dim]$[/dim] [cyan]vid_to_sub[/cyan] {' '.join(_mask(plans[0].cmd[2:]))}"
        self._refresh_live_panels()
        self._active_worker = self._stream(plans)

    @work(thread=True, exclusive=True, exit_on_error=False, name="prepare-run")
    def _prepare_run(self, config: RunConfig) -> None:
        try:
            videos, found_total, skipped = self._discover_videos_for_run(config)
        except ValueError as exc:
            self.call_from_thread(
                self._abort_prepared_run,
                config.request_id,
                f"[bold red]✕ {exc}[/]",
            )
            return

        if config.request_id != self._run_request_id:
            return
        if not videos:
            self.call_from_thread(
                self._abort_prepared_run,
                config.request_id,
                "[bold yellow]No video files matched the current selection and filters.[/]",
            )
            return

        plans = self._build_executor_plans(
            videos,
            dry_run=config.dry_run,
            config=config,
        )
        if config.request_id != self._run_request_id:
            return
        if not plans:
            self.call_from_thread(
                self._abort_prepared_run,
                config.request_id,
                "[bold red]✕ Could not build any execution plan.[/]",
            )
            return

        self.call_from_thread(
            self._start_prepared_run,
            config,
            videos,
            found_total,
            skipped,
            plans,
        )

    @work(thread=True, exclusive=True, exit_on_error=False, name="_stream")
    def _stream(self, plans: list[ExecutorPlan]) -> None:
        output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue(maxsize=2048)
        launched: list[
            tuple[
                ExecutorPlan,
                subprocess.Popen[str],
                threading.Thread,
                threading.Thread | None,
            ]
        ] = []
        multi_executor = len(plans) > 1
        completed_labels: list[str] = []
        failed_executors: list[tuple[ExecutorPlan, int]] = []
        stream_completed = False

        def _pump_stdout(executor_name: str, proc: subprocess.Popen[str]) -> None:
            assert proc.stdout
            try:
                for raw in iter(proc.stdout.readline, ""):
                    output_queue.put((executor_name, raw.rstrip("\n")))
            except Exception:
                pass
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                output_queue.put((executor_name, None))

        def _pump_stdin(proc: subprocess.Popen[str], payload: str) -> None:
            if proc.stdin is None:
                return
            try:
                proc.stdin.write(payload)
            except BrokenPipeError:
                pass
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        try:
            for plan in plans:
                try:
                    proc = subprocess.Popen(
                        plan.cmd,
                        stdin=subprocess.PIPE
                        if plan.stdin_payload is not None
                        else None,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=plan.env,
                    )
                except FileNotFoundError as exc:
                    self.call_from_thread(
                        self._log, f"[bold red]✕ {plan.label}: not found: {exc}[/]"
                    )
                    self.call_from_thread(self._finalize_executor_failure, plan, 127)
                    continue
                except Exception as exc:
                    self.call_from_thread(
                        self._log, f"[bold red]✕ {plan.label}: {exc}[/]"
                    )
                    self.call_from_thread(self._finalize_executor_failure, plan, 1)
                    continue

                self._procs[plan.label] = proc
                if plan.kind == "local":
                    self._proc = proc

                reader = threading.Thread(
                    target=_pump_stdout,
                    args=(plan.label, proc),
                    daemon=True,
                )
                reader.start()
                writer: threading.Thread | None = None
                if plan.stdin_payload is not None:
                    writer = threading.Thread(
                        target=_pump_stdin,
                        args=(proc, plan.stdin_payload),
                        daemon=True,
                    )
                    writer.start()
                launched.append((plan, proc, reader, writer))

            active_streams = len(launched)
            while active_streams > 0:
                try:
                    executor, line = output_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    active_streams -= 1
                    continue

                event = parse_progress_event(line)
                if event:
                    self.call_from_thread(self._apply_progress_event, executor, event)
                    continue

                rendered = _colorize(line)
                if multi_executor:
                    rendered = f"[bold blue]{executor}[/] {rendered}"
                self.call_from_thread(self._log, rendered)

            stream_completed = True
        finally:
            # Always reap launched subprocesses, even if a UI callback raises or the
            # worker is cancelled mid-stream, so exited children don't remain zombies.
            for plan, proc, reader, writer in launched:
                reader.join(timeout=1)
                if writer is not None:
                    writer.join(timeout=1)

                rc = proc.poll()
                if rc is None and not stream_completed:
                    try:
                        proc.terminate()
                    except OSError:
                        pass
                    rc = proc.poll()

                try:
                    rc = proc.wait(timeout=1) if rc is None else proc.wait()
                except subprocess.TimeoutExpired:
                    # Terminate timed out — escalate to SIGKILL.
                    try:
                        proc.kill()
                    except OSError:
                        pass
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass

                self._procs.pop(plan.label, None)
                if self._proc is proc:
                    self._proc = None

                if rc is not None and rc != 0:
                    failed_executors.append((plan, rc))
                elif rc is not None and multi_executor:
                    completed_labels.append(plan.label)

        had_error = bool(failed_executors)
        for plan, rc in failed_executors:
            self.call_from_thread(
                self._log, f"\n[bold red]✕ {plan.label} exited with code {rc}[/]"
            )
            self.call_from_thread(self._finalize_executor_failure, plan, rc)
        for label in completed_labels:
            self.call_from_thread(self._log, f"[green]{label} finished (exit 0)[/]")

        if had_error:
            self.call_from_thread(
                self._log,
                "\n[bold red]✕ One or more executors failed. Review the log and history.[/]",
            )
        else:
            self.call_from_thread(self._log, "\n[bold green]✓ Completed (exit 0)[/]")

        # Stage handoff: when stage1-only plans complete successfully, collect
        # their artifacts and launch a separate local stage2 translation pass.
        if not had_error:
            stage1_plans = [
                plan for plan in plans if getattr(plan, "stage", "full") == "stage1"
            ]
            if stage1_plans:
                has_remote_stage1 = any(
                    getattr(plan, "kind", "local") != "local" for plan in stage1_plans
                )
                artifact_paths = self._collect_stage1_artifacts(stage1_plans)
                if artifact_paths:
                    self.call_from_thread(
                        self._log,
                        f"[bold cyan]Stage handoff:[/] {len(artifact_paths)} stage-1 artifact(s) found. "
                        "Launching local stage-2 translation…",
                    )
                    self.call_from_thread(
                        self._schedule_stage2_from_artifacts, artifact_paths
                    )
                elif has_remote_stage1:
                    self.call_from_thread(
                        self._log,
                        "[yellow]Stage handoff: distributed stage-1 completed, but automatic "
                        "stage-2 only scans locally visible .stage1.json artifacts. Copy "
                        "artifacts locally or run --translate-from-artifact manually.[/]",
                    )
                else:
                    self.call_from_thread(
                        self._log,
                        "[yellow]Stage handoff: stage-1 completed but no .stage1.json artifacts "
                        "were found locally. Run --translate-from-artifact manually.[/]",
                    )

    def _schedule_stage2_from_artifacts(self, artifact_paths: list[str]) -> None:
        """Launch local Stage-2 translation for a batch of stage-1 artifacts.

        Called on the main thread (via call_from_thread) after a stage1 run
        completes successfully. Each artifact must contain a ``target_lang``
        field; artifacts that lack one are skipped with a warning. All valid
        artifacts are submitted as a single multi-executor _stream() call so
        that progress tracking works correctly.
        """
        from pathlib import Path as _Path
        from vid_to_sub_app.cli.stage_artifact import load_stage_artifact
        from ..helpers import ExecutorPlan

        plans: list[ExecutorPlan] = []
        for artifact_path in artifact_paths:
            try:
                artifact = load_stage_artifact(_Path(artifact_path))
                target_lang = str(
                    artifact.get("target_lang") or self._val("inp-translate-to") or ""
                ).strip()
                source_path = str(artifact.get("source_path") or artifact_path)
            except Exception as exc:
                self._log(
                    f"[yellow]Skipping artifact {artifact_path}: cannot read ({exc})[/]"
                )
                continue
            if not target_lang:
                self._log(
                    f"[yellow]Skipping artifact {_Path(artifact_path).name}: no target_lang.[/]"
                )
                continue

            cmd = self._build_stage2_cmd(
                artifact_path,
                source_path=source_path,
                target_lang=target_lang,
            )

            label = f"stage2:{len(plans) + 1}:{_Path(source_path).stem}"
            plans.append(
                ExecutorPlan(
                    name=label,
                    kind="local",
                    label=label,
                    cmd=cmd,
                    env=self._build_run_env(),
                    assigned_paths=[source_path],
                    capacity=1,
                    manifest={"entries": [], "folders": []},
                    stdin_payload=None,
                    stage="stage2",
                )
            )

        if not plans:
            self._log(
                "[yellow]Stage handoff: no valid artifacts with target_lang. "
                "Nothing to translate.[/]"
            )
            return

        self._log(
            f"[bold cyan]Stage handoff:[/] launching local stage-2 for "
            f"{len(plans)} artifact(s)…"
        )
        self._reset_run_state(preserve_shell=True)
        self._run_last_shell = f"[cyan]Stage-2 translate[/] · {len(plans)} artifact(s) from stage-1 handoff"
        self._run_total_queued = len(plans)
        self._pending_paths = {plan.label: set(plan.assigned_paths) for plan in plans}
        self._refresh_live_panels()
        self._active_worker = self._stream(plans)
