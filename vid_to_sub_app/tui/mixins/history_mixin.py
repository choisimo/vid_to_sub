from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from textual.css.query import NoMatches
from textual.widgets import DataTable, Input, Select, Static, Switch

from ..helpers import DEFAULT_BACKEND, DEFAULT_MODEL, _compact_progress_markup, _fmt_elapsed, _progress_bar_markup_ratio
from ..state import db as _db


class HistoryMixin:
    """History-tab mixin.

    Requires (must be provided by the host class):
        - self._active_jobs: dict[str, RunJobState]
        - self._selected_paths: list[str]
        - self._hist_key: str | None
        - self._trigger(dry_run: bool) -> None
        - self._log(text: str) -> None
        - self._val(wid: str) -> str
        - self._refresh_sel_paths() -> None
        - self._refresh_recent_paths() -> None
        - self._refresh_live_panels() -> None

    Provides:
        - _init_history_table() -> None
        - _refresh_history() -> None
        - _show_hist_detail(key: str | None) -> None
        - _load_history_job(job_id: int) -> None
        - _rerun_history_job(job_id: int) -> None
        - _action_hist_refresh() -> None
        - _action_hist_load() -> None
        - _action_hist_rerun() -> None
        - _action_hist_clear() -> None
        - _action_hist_delete() -> None
    """
    # ── History tab ───────────────────────────────────────────────────────

    def _init_history_table(self) -> None:
        try:
            t = self.query_one("#hist-table", DataTable)
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
        try:
            t = self.query_one("#hist-table", DataTable)
        except NoMatches:
            return
        t.clear(columns=False)
        jobs = _db.get_jobs()
        active_by_job_id = {
            state.job_id: state
            for state in self._active_jobs.values()
            if state.job_id is not None
        }

        for job in jobs:
            s = job["status"]
            active = active_by_job_id.get(job["id"])
            if s == "running" and active is not None:
                s_cell = _compact_progress_markup(active.progress_ratio)
            else:
                s_cell = (
                    f"[green]✓ {s}[/]"
                    if s == "done"
                    else f"[red]✗ {s}[/]"
                    if s == "failed"
                    else f"[yellow]⟳ {s}[/]"
                    if s == "running"
                    else s
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
                s_cell,
                wall,
                key=str(job["id"]),
            )
        try:
            self.query_one("#hist-count", Static).update(f"[dim]{len(jobs)} job(s)[/]")
        except NoMatches:
            pass

    def _show_hist_detail(self, key: str | None) -> None:
        if not key:
            return
        jobs = _db.get_jobs()
        job = next((j for j in jobs if str(j["id"]) == key), None)
        if not job:
            return
        active = next(
            (state for state in self._active_jobs.values() if state.job_id == job["id"]),
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
        try:
            ops = json.loads(job.get("output_paths") or "[]")
            if ops:
                lines.append("Outputs: " + "  ".join(Path(o).name for o in ops))
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            self.query_one("#hist-detail", Static).update("\n".join(lines))
        except NoMatches:
            pass

    def _load_history_job(self, job_id: int) -> None:
        job = next((item for item in _db.get_jobs() if item["id"] == job_id), None)
        if not job:
            self._log(f"[red]History row #{job_id} was not found.[/]")
            return

        self._selected_paths = [job["video_path"]]
        self._refresh_sel_paths()
        self._refresh_recent_paths()

        for wid, value in [
            ("inp-output-dir", job.get("output_dir") or ""),
            ("inp-language", job.get("language") or ""),
        ]:
            try:
                self.query_one(f"#{wid}", Input).value = value
            except NoMatches:
                pass

        try:
            self.query_one("#sel-backend", Select).value = job.get("backend") or DEFAULT_BACKEND
            self.query_one("#sel-model", Select).value = job.get("model") or DEFAULT_MODEL
        except NoMatches:
            pass

    def _rerun_history_job(self, job_id: int) -> None:
        """Load a history job's settings into the form then immediately trigger a run."""
        self._load_history_job(job_id)
        self._trigger(dry_run=False)

