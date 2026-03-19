from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Protocol, cast

from textual import work
from textual.css.query import NoMatches
from textual.widgets import DataTable, Input, Select, Static, Switch

from vid_to_sub_app.cli.output import parse_srt_timestamp, srt_timestamp
from vid_to_sub_app.cli.stage_artifact import load_stage_artifact
from vid_to_sub_app.cli.translation import translate_segments_openai_compatible

from ..helpers import (
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    ENV_TRANS_KEY,
    ENV_TRANS_MOD,
    ENV_TRANS_URL,
    _compact_progress_markup,
    _fmt_elapsed,
    _progress_bar_markup_ratio,
    filter_subtitle_paths,
    resolve_copy_dest,
)
from ..state import db as _db


SRT_BLOCK_RE = re.compile(
    r"(?ms)^\s*\d+\s*\n\s*"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"(.*?)(?=\n{2,}|\Z)"
)


class _HistoryApp(Protocol):
    _active_jobs: dict[str, Any]
    _hist_key: str | None
    _hist_select_mode: bool
    _hist_selected: set[str]
    _selected_paths: list[str]

    def call_from_thread(self, callback: Any, *args: Any) -> Any: ...
    def query_one(self, selector: str, expect_type: type[Any] | None = None) -> Any: ...
    def _log(self, text: str) -> None: ...
    def _refresh_recent_paths(self) -> None: ...
    def _refresh_sel_paths(self) -> None: ...
    def _trigger(self, dry_run: bool) -> None: ...
    def _val(self, wid: str) -> str: ...


def _history_app(instance: object) -> _HistoryApp:
    return cast(_HistoryApp, instance)


class HistoryMixin:
    # -- History tab -----------------------------------------------------------

    def _update_history_count(self, total_jobs: int | None = None) -> None:
        app = _history_app(self)
        if total_jobs is None:
            total_jobs = len(_db.get_jobs())
        selected = len(app._hist_selected)
        label = f"[dim]{total_jobs} job(s)[/]"
        if app._hist_select_mode or selected:
            label = f"[dim]{total_jobs} job(s) · {selected} selected[/]"
        try:
            app.query_one("#hist-count", Static).update(label)
        except NoMatches:
            pass

    def _toggle_history_selection(self, key: str) -> None:
        app = _history_app(self)
        if key in app._hist_selected:
            app._hist_selected.remove(key)
        else:
            app._hist_selected.add(key)
        self._update_history_count()

    def _toggle_history_select_mode(self) -> None:
        app = _history_app(self)
        app._hist_select_mode = not app._hist_select_mode
        if not app._hist_select_mode:
            app._hist_selected.clear()
        self._update_history_count()

    def _reset_history_selection(self) -> None:
        app = _history_app(self)
        app._hist_select_mode = False
        app._hist_selected.clear()
        self._update_history_count()

    def _parse_srt_segments(self, text: str) -> list[dict[str, Any]]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        segments: list[dict[str, Any]] = []
        for match in SRT_BLOCK_RE.finditer(normalized):
            body = match.group(3).strip()
            if not body:
                continue
            segments.append(
                {
                    "start": parse_srt_timestamp(match.group(1)),
                    "end": parse_srt_timestamp(match.group(2)),
                    "text": body,
                }
            )
        return segments

    def _segments_to_srt_text(self, segments: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, segment in enumerate(segments, 1):
            lines.append(str(index))
            lines.append(
                f"{srt_timestamp(segment['start'])} --> {srt_timestamp(segment['end'])}"
            )
            lines.append(str(segment["text"]).strip())
            lines.append("")
        return "\n".join(lines)

    def _init_history_table(self) -> None:
        app = _history_app(self)
        try:
            t = app.query_one("#hist-table", DataTable)
            t.cell_padding = 0
            t.add_column("ID", key="id", width=4)
            t.add_column("Date", key="date", width=10)
            t.add_column("File", key="file", width=22)
            t.add_column("Backend", key="backend", width=10)
            t.add_column("Model", key="model", width=11)
            t.add_column("Status", key="status", width=15)
            t.add_column("Time", key="time", width=6)
        except NoMatches:
            pass

    def _refresh_history(self) -> None:
        app = _history_app(self)
        try:
            t = app.query_one("#hist-table", DataTable)
        except NoMatches:
            return
        t.clear(columns=False)
        jobs = _db.get_jobs()
        valid_keys = {str(job["id"]) for job in jobs}
        app._hist_selected.intersection_update(valid_keys)
        if app._hist_key and app._hist_key not in valid_keys:
            app._hist_key = None
        active_by_job_id = {
            state.job_id: state
            for state in app._active_jobs.values()
            if state.job_id is not None
        }

        for job in jobs:
            status = job["status"]
            active = active_by_job_id.get(job["id"])
            if status == "running" and active is not None:
                status_cell = _compact_progress_markup(active.progress_ratio)
            else:
                status_cell = (
                    f"[green]✓ {status}[/]"
                    if status == "done"
                    else f"[red]✗ {status}[/]"
                    if status == "failed"
                    else f"[yellow]⟳ {status}[/]"
                    if status == "running"
                    else status
                )
            name = Path(job["video_path"]).name
            if len(name) > 24:
                name = name[:21] + "…"
            wall = f"{job['wall_sec']:.0f}s" if job.get("wall_sec") else "-"
            t.add_row(
                str(job["id"]),
                job["created_at"][:10],
                name,
                job["backend"] or "-",
                job["model"] or "-",
                status_cell,
                wall,
                key=str(job["id"]),
            )
        self._update_history_count(len(jobs))

    def _show_hist_detail(self, key: str | None) -> None:
        app = _history_app(self)
        if not key:
            return
        jobs = _db.get_jobs()
        job = next((j for j in jobs if str(j["id"]) == key), None)
        if not job:
            return
        active = next(
            (state for state in app._active_jobs.values() if state.job_id == job["id"]),
            None,
        )
        lines = [
            f"[bold]ID {job['id']}[/]  {job['created_at']}  "
            f"backend={job['backend']}  model={job['model']}  status={job['status']}",
            f"File: {job['video_path']}",
        ]
        if job.get("language") or job.get("target_lang"):
            lines.append(
                f"Language: {job.get('language') or 'auto'}   Translate to: {job.get('target_lang') or '-'}"
            )
        if active is not None:
            progress_seconds = max(0.0, active.progress_seconds or 0.0)
            progress_ratio = (
                active.progress_ratio
                if active.progress_ratio is not None
                else (
                    progress_seconds / active.video_duration
                    if active.video_duration and active.video_duration > 0
                    else 0.0
                )
            )
            lines.append(
                "Progress: " + _progress_bar_markup_ratio(progress_ratio, width=30)
            )
            if active.video_duration is not None:
                lines.append(
                    f"Transcribed: {_fmt_elapsed(progress_seconds)} / "
                    f"{_fmt_elapsed(active.video_duration)}"
                )
        if job.get("output_dir"):
            lines.append(f"Output dir: {job['output_dir']}")
        if job.get("wall_sec"):
            lines.append(f"Wall time: {_fmt_elapsed(job['wall_sec'])}")
        if job.get("error"):
            lines.append(f"[red]Error: {job['error']}[/]")

        stage_artifact_path: str | None = None
        try:
            output_paths = json.loads(job.get("output_paths") or "[]")
            if output_paths:
                lines.append(
                    "Outputs: "
                    + "  ".join(Path(output_path).name for output_path in output_paths)
                )
            stage_artifact_path = next(
                (
                    output_path
                    for output_path in output_paths
                    if Path(output_path).name.lower().endswith(".stage1.json")
                ),
                None,
            )
        except (json.JSONDecodeError, TypeError):
            stage_artifact_path = None

        if stage_artifact_path:
            lines.append(f"Stage artifact: {Path(stage_artifact_path).name}")
            artifact_path = Path(stage_artifact_path)
            if artifact_path.exists():
                try:
                    stage_status = (
                        load_stage_artifact(artifact_path).get("stage_status") or {}
                    )
                    stage_state = None
                    if stage_status.get("translation_failed"):
                        stage_state = "failed"
                    elif stage_status.get("translation_complete"):
                        stage_state = "complete"
                    elif stage_status.get("translation_pending"):
                        stage_state = "pending"
                    if stage_state:
                        lines.append(f"Stage status: translate={stage_state}")
                except (OSError, ValueError, TypeError):
                    pass

        try:
            app.query_one("#hist-detail", Static).update("\n".join(lines))
        except NoMatches:
            pass

    def _load_history_job(self, job_id: int) -> None:
        app = _history_app(self)
        job = next((item for item in _db.get_jobs() if item["id"] == job_id), None)
        if not job:
            app._log(f"[red]History row #{job_id} was not found.[/]")
            return

        app._selected_paths = [job["video_path"]]
        app._refresh_sel_paths()
        app._refresh_recent_paths()

        for wid, value in [
            ("inp-output-dir", job.get("output_dir") or ""),
            ("inp-language", job.get("language") or ""),
        ]:
            try:
                app.query_one(f"#{wid}", Input).value = value
            except NoMatches:
                pass

        try:
            app.query_one("#sel-backend", Select).value = (
                job.get("backend") or DEFAULT_BACKEND
            )
            app.query_one("#sel-model", Select).value = (
                job.get("model") or DEFAULT_MODEL
            )
        except NoMatches:
            pass

    def _rerun_history_job(self, job_id: int) -> None:
        """Load a history job's settings into the form then immediately trigger a run."""
        app = _history_app(self)
        self._load_history_job(job_id)
        app._trigger(dry_run=False)

    def _translate_history_job(self, job_id: int) -> None:
        app = _history_app(self)
        job = next((item for item in _db.get_jobs() if item["id"] == job_id), None)
        if not job:
            app._log(f"[red]History row #{job_id} was not found.[/]")
            return
        if job.get("status") != "done":
            app._log("[yellow]Only completed jobs can be translated from History.[/]")
            return

        try:
            subtitle_paths = filter_subtitle_paths(
                json.loads(job.get("output_paths") or "[]")
            )
        except (json.JSONDecodeError, TypeError):
            subtitle_paths = []
        if not subtitle_paths:
            app._log("[yellow]No subtitle outputs found for this job.[/]")
            return

        existing_paths: list[str] = []
        missing_paths: list[str] = []
        for subtitle_path in subtitle_paths:
            if Path(subtitle_path).exists():
                existing_paths.append(subtitle_path)
            else:
                missing_paths.append(subtitle_path)
        if not existing_paths:
            app._log("[red]No subtitle files exist on disk — cannot translate.[/]")
            return

        target_lang = (
            job.get("target_lang") or app._val("inp-translate-to") or ""
        ).strip()
        if not target_lang:
            app._log("[red]No target language is configured for this job.[/]")
            return

        if missing_paths:
            app._log(f"[yellow]Skipped {len(missing_paths)} missing file(s).[/]")
        self._trigger_translate_from_paths(existing_paths, target_lang)

    @work(thread=True, exclusive=False, exit_on_error=False, name="history-translate")
    def _trigger_translate_from_paths(
        self, subtitle_paths: list[str], target_lang: str
    ) -> None:
        app = _history_app(self)
        translation_base_url = _db.get(ENV_TRANS_URL) or None
        translation_api_key = _db.get(ENV_TRANS_KEY) or None
        translation_model = _db.get(ENV_TRANS_MOD) or None
        translated_count = 0

        for subtitle_path in subtitle_paths:
            source = Path(subtitle_path)
            suffix = source.suffix.lower() or "<none>"
            if suffix != ".srt":
                app.call_from_thread(app._log, f"[dim]Skipping {suffix} format[/]")
                continue

            try:
                segments = self._parse_srt_segments(source.read_text(encoding="utf-8"))
                if not segments:
                    app.call_from_thread(
                        app._log,
                        f"[yellow]No subtitle segments found in {source.name}.[/]",
                    )
                    continue
                translated_segments, _info = translate_segments_openai_compatible(
                    segments=segments,
                    target_language=target_lang,
                    translation_model=translation_model,
                    translation_base_url=translation_base_url,
                    translation_api_key=translation_api_key,
                    source_language=None,
                )
                destination = source.with_name(
                    f"{source.stem}.{target_lang}{source.suffix}"
                )
                destination.write_text(
                    self._segments_to_srt_text(translated_segments),
                    encoding="utf-8",
                )
                translated_count += 1
            except (OSError, RuntimeError, ValueError) as exc:
                app.call_from_thread(app._log, f"[red]Translation failed: {exc}[/]")

        app.call_from_thread(
            app._log,
            f"[green]Translation complete: {translated_count} file(s) translated.[/]",
        )

    def _copy_selected_subtitles(self, dest_dir: Path) -> None:
        app = _history_app(self)
        if not app._hist_selected:
            app._log("[yellow]No history rows are selected.[/]")
            return

        dest_dir = dest_dir.expanduser()
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            app._log(f"[red]Could not prepare copy destination: {exc}[/]")
            return

        jobs_by_key = {str(job["id"]): job for job in _db.get_jobs()}
        existing_names: set[str] = set()
        copied = 0
        errors: list[str] = []

        for key in sorted(app._hist_selected):
            job = jobs_by_key.get(key)
            if not job:
                errors.append(f"history row #{key} was not found")
                continue
            try:
                subtitle_paths = filter_subtitle_paths(
                    json.loads(job.get("output_paths") or "[]")
                )
            except (json.JSONDecodeError, TypeError):
                errors.append(f"history row #{key} has invalid output paths")
                continue

            for subtitle_path in subtitle_paths:
                source = Path(subtitle_path)
                destination = resolve_copy_dest(source, dest_dir, existing_names)
                try:
                    shutil.copy2(source, destination)
                    copied += 1
                except OSError as exc:
                    errors.append(f"{source}: {exc}")

        app._log(
            f"[green]Copied {copied} file(s) to {dest_dir}. {len(errors)} error(s).[/]"
        )
        self._reset_history_selection()
