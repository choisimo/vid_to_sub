from __future__ import annotations

import json
import os
from typing import Mapping

from textual.css.query import NoMatches
from textual.widgets import DataTable, Input, Select, Static, TextArea

from vid_to_sub_app.shared.env import load_project_env

from ..helpers import (
    ENV_AGENT_KEY,
    ENV_AGENT_MOD,
    ENV_AGENT_URL,
    ENV_FILE,
    ENV_POST_KEY,
    ENV_POST_MOD,
    ENV_POST_URL,
    ENV_TRANS_KEY,
    ENV_TRANS_MOD,
    ENV_TRANS_URL,
    ENV_WCPP_BIN,
    ENV_WCPP_MODEL,
    parse_remote_resources,
)
from ..state import db as _db


class SettingsMixin:
    """Settings-tab mixin — form persistence, SSH connection management, and env sync.

    Requires (must be provided by the host class):
        - self._ssh_selected_id: int | None
        - self._remote_resources: list[RemoteResourceProfile]
        - self._val(wid: str) -> str
        - self._sel(wid: str, fallback: str) -> str
        - self._sw(wid: str) -> bool
        - self.query_one(selector, type) — Textual widget accessor
        - self.notify(message: str, **kw) — Textual notification
        # Cross-mixin: called after state-changing operations
        - self._refresh_remote_state() -> None          # app.py canonical call-site
        - self._update_agent_config_status() -> None    # AgentMixin
        - self._update_cmd_preview() -> None            # RunMixin
        - self._sync_transcribe_overrides_from_settings(...) -> None  # self
        - self._sync_run_defaults_from_settings(...) -> None          # self

    Provides:
        - _load_settings_form() -> None
        - _prefill_transcribe() -> None
        - _sync_transcribe_overrides_from_settings(prev, new) -> None
        - _sync_run_defaults_from_settings(prev, new) -> None
        - _save_settings() -> None
        - _export_env() -> None
        - _apply_db_to_env() -> None
        - _migrate_env_to_db() -> None
        - _import_env_to_sqlite() -> None
        - _init_ssh_table() -> None
        - _refresh_ssh_table() -> None
        - _ssh_read_form() -> dict[str, str]
        - _ssh_parse_json_field(value, field_name) -> dict | None
        - _ssh_set_status(msg: str) -> None
        - _ssh_clear_form() -> None
        - _ssh_fill_form_from_row(row: dict) -> None
        - _ssh_add_connection() -> None
        - _ssh_update_connection() -> None
        - _ssh_delete_connection() -> None
        - _action_stg_reload() -> None
        - _action_stg_save() -> None
        - _action_export_env() -> None
        - _action_import_env() -> None
        - _action_ssh_add() -> None
        - _action_ssh_update() -> None
        - _action_ssh_delete() -> None
    """

    # ── Settings ──────────────────────────────────────────────────────────

    def _load_settings_form(self) -> None:
        pairs = [
            ("stg-wcpp-bin", ENV_WCPP_BIN),
            ("stg-wcpp-model", ENV_WCPP_MODEL),
            ("stg-trans-url", ENV_TRANS_URL),
            ("stg-trans-key", ENV_TRANS_KEY),
            ("stg-trans-model", ENV_TRANS_MOD),
            ("stg-post-url", ENV_POST_URL),
            ("stg-post-key", ENV_POST_KEY),
            ("stg-post-model", ENV_POST_MOD),
            ("stg-agent-url", ENV_AGENT_URL),
            ("stg-agent-key", ENV_AGENT_KEY),
            ("stg-agent-model", ENV_AGENT_MOD),
            ("stg-build-dir", "tui.build_dir"),
            ("stg-install-dir", "tui.install_dir"),
            ("stg-model-dir", "tui.model_dir"),
            ("stg-browse-root", "tui.browse_root"),
            ("stg-default-output-dir", "tui.default_output_dir"),
            ("stg-default-translate-to", "tui.default_translate_to"),
        ]
        for wid, key in pairs:
            try:
                self.query_one(f"#{wid}", Input).value = _db.get(key)
            except NoMatches:
                pass
        try:
            self.query_one("#stg-remote-resources", TextArea).text = _db.get(
                "tui.remote_resources", "[]"
            )
        except NoMatches:
            pass
        try:
            mode = _db.get("tui.execution_mode") or "local"
            self.query_one("#sel-execution-mode", Select).value = mode
        except NoMatches:
            pass

    def _prefill_transcribe(self) -> None:
        """Pre-fill run form fields from settings / env vars."""
        for wid, key in [
            ("inp-trans-url", ENV_TRANS_URL),
            ("inp-trans-key", ENV_TRANS_KEY),
            ("inp-trans-model", ENV_TRANS_MOD),
            ("inp-post-url", ENV_POST_URL),
            ("inp-post-key", ENV_POST_KEY),
            ("inp-post-model", ENV_POST_MOD),
            ("inp-wcpp-bin", ENV_WCPP_BIN),
        ]:
            try:
                val = _db.get(key) or os.environ.get(key, "")
                if val:
                    self.query_one(f"#{wid}", Input).value = val
            except NoMatches:
                pass
        for wid, key in [
            ("inp-output-dir", "tui.default_output_dir"),
            ("inp-translate-to", "tui.default_translate_to"),
        ]:
            try:
                val = _db.get(key).strip()
                if val:
                    self.query_one(f"#{wid}", Input).value = val
            except NoMatches:
                pass

    def _sync_transcribe_overrides_from_settings(
        self,
        previous_values: Mapping[str, str],
        updated_values: Mapping[str, str],
    ) -> None:
        pairs = [
            ("inp-trans-url", ENV_TRANS_URL),
            ("inp-trans-key", ENV_TRANS_KEY),
            ("inp-trans-model", ENV_TRANS_MOD),
            ("inp-post-url", ENV_POST_URL),
            ("inp-post-key", ENV_POST_KEY),
            ("inp-post-model", ENV_POST_MOD),
            ("inp-wcpp-bin", ENV_WCPP_BIN),
        ]
        for wid, key in pairs:
            try:
                widget = self.query_one(f"#{wid}", Input)
            except NoMatches:
                continue

            current = widget.value.strip()
            previous = (previous_values.get(key) or "").strip()
            if current and current != previous:
                continue

            widget.value = (updated_values.get(key) or "").strip()

    def _sync_run_defaults_from_settings(
        self,
        previous_values: Mapping[str, str],
        updated_values: Mapping[str, str],
    ) -> None:
        pairs = [
            ("inp-output-dir", "tui.default_output_dir"),
            ("inp-translate-to", "tui.default_translate_to"),
        ]
        for wid, key in pairs:
            try:
                widget = self.query_one(f"#{wid}", Input)
            except NoMatches:
                continue

            current = widget.value.strip()
            previous = (previous_values.get(key) or "").strip()
            if current and current != previous:
                continue

            widget.value = (updated_values.get(key) or "").strip()

    def _save_settings(self) -> None:
        pairs = [
            ("stg-wcpp-bin", ENV_WCPP_BIN),
            ("stg-wcpp-model", ENV_WCPP_MODEL),
            ("stg-trans-url", ENV_TRANS_URL),
            ("stg-trans-key", ENV_TRANS_KEY),
            ("stg-trans-model", ENV_TRANS_MOD),
            ("stg-post-url", ENV_POST_URL),
            ("stg-post-key", ENV_POST_KEY),
            ("stg-post-model", ENV_POST_MOD),
            ("stg-agent-url", ENV_AGENT_URL),
            ("stg-agent-key", ENV_AGENT_KEY),
            ("stg-agent-model", ENV_AGENT_MOD),
            ("stg-build-dir", "tui.build_dir"),
            ("stg-install-dir", "tui.install_dir"),
            ("stg-model-dir", "tui.model_dir"),
            ("stg-browse-root", "tui.browse_root"),
            ("stg-default-output-dir", "tui.default_output_dir"),
            ("stg-default-translate-to", "tui.default_translate_to"),
        ]
        data: dict[str, str] = {}
        for wid, key in pairs:
            try:
                data[key] = self.query_one(f"#{wid}", Input).value.strip()
            except NoMatches:
                pass
        try:
            data["tui.remote_resources"] = self.query_one(
                "#stg-remote-resources", TextArea
            ).text.strip() or "[]"
        except NoMatches:
            pass
        data["tui.execution_mode"] = self._sel("sel-execution-mode", "local")
        try:
            parse_remote_resources(data.get("tui.remote_resources", "[]"))
        except ValueError as exc:
            try:
                self.query_one("#stg-status", Static).update(f"[red]✗ {exc}[/]")
            except NoMatches:
                pass
            return
        previous_values = {key: _db.get(key) for _, key in pairs}
        _db.set_many(data)
        self._apply_db_to_env()
        self._sync_transcribe_overrides_from_settings(previous_values, data)
        self._sync_run_defaults_from_settings(previous_values, data)
        self._refresh_remote_state()
        self._update_agent_config_status()
        self._update_cmd_preview()
        try:
            self.query_one("#stg-status", Static).update("[green]✓ Saved to SQLite[/]")
        except NoMatches:
            pass

    def _export_env(self) -> None:
        """Write current settings to .env for backwards compatibility."""
        env_map = {
            ENV_WCPP_BIN: "stg-wcpp-bin",
            ENV_WCPP_MODEL: "stg-wcpp-model",
            ENV_TRANS_URL: "stg-trans-url",
            ENV_TRANS_KEY: "stg-trans-key",
            ENV_TRANS_MOD: "stg-trans-model",
            ENV_POST_URL: "stg-post-url",
            ENV_POST_KEY: "stg-post-key",
            ENV_POST_MOD: "stg-post-model",
            ENV_AGENT_URL: "stg-agent-url",
            ENV_AGENT_KEY: "stg-agent-key",
            ENV_AGENT_MOD: "stg-agent-model",
        }
        lines: list[str] = []
        for env_key, wid in env_map.items():
            try:
                val = self.query_one(f"#{wid}", Input).value.strip()
            except NoMatches:
                val = _db.get(env_key)
            escaped = val.replace('"', '\\"')
            lines.append(f'{env_key}="{escaped}"\n' if val else f"# {env_key}=\n")
        try:
            ENV_FILE.write_text("".join(lines), encoding="utf-8")
            try:
                self.query_one("#stg-status", Static).update(
                    f"[green]✓ Exported to {ENV_FILE}[/]"
                )
            except NoMatches:
                pass
        except OSError as e:
            try:
                self.query_one("#stg-status", Static).update(
                    f"[red]✗ Export failed: {e}[/]"
                )
            except NoMatches:
                pass

    def _apply_db_to_env(self) -> None:
        """Push VID_TO_SUB_* settings from SQLite into os.environ.

        SQLite is the authoritative source; this always overrides whatever
        was set by shell or a previous .env load.
        """
        from vid_to_sub_app.shared.env import load_env_from_sqlite
        load_env_from_sqlite(_db.get_all, override=True)

    def _migrate_env_to_db(self) -> None:
        """One-time migration: import .env into SQLite if DB keys are blank.

        Uses import_env_file_to_sqlite (overwrite=False) so it only fills keys
        that have no value in DB yet.  After this, SQLite is the source of truth
        and .env is no longer automatically consulted.
        """
        from vid_to_sub_app.shared.env import import_env_file_to_sqlite
        import_env_file_to_sqlite(
            ENV_FILE,
            _db.set_many,
            _db.get_all,
            overwrite=False,
        )
    def _import_env_to_sqlite(self) -> None:
        """Import .env file into SQLite (overwrite=False: only fills blank keys).

        This is how users can 'use' their .env: after import, SQLite becomes the
        source of truth and .env is no longer consulted automatically.
        """
        from vid_to_sub_app.shared.env import import_env_file_to_sqlite
        if not ENV_FILE.exists():
            try:
                self.query_one("#stg-status", Static).update(
                    f"[yellow]No .env file found at {ENV_FILE}[/]"
                )
            except NoMatches:
                pass
            return
        written = import_env_file_to_sqlite(
            ENV_FILE,
            _db.set_many,
            _db.get_all,
            overwrite=False,
        )
        self._apply_db_to_env()
        self._load_settings_form()
        self._update_agent_config_status()
        msg = (
            f"[green]✓ Imported {len(written)} key(s) from .env to SQLite[/]"
            if written
            else "[yellow]No new keys to import (all DB keys already set)[/]"
        )
        try:
            self.query_one("#stg-status", Static).update(msg)
        except NoMatches:
            pass

    # ── SSH connection management ─────────────────────────────────────

    def _init_ssh_table(self) -> None:
        """Initialize the SSH connections DataTable columns."""
        try:
            table = self.query_one("#ssh-conn-table", DataTable)
            table.add_columns("ID", "Label", "Host", "User", "Port", "Workdir", "Slots", "Enabled")
        except (NoMatches, Exception):
            pass

    def _refresh_ssh_table(self) -> None:
        """Reload SSH connections from DB into the table widget."""
        try:
            table = self.query_one("#ssh-conn-table", DataTable)
        except NoMatches:
            return
        table.clear()
        connections = _db.get_ssh_connections()
        for row in connections:
            table.add_row(
                str(row.get("id", "")),
                str(row.get("label") or ""),
                str(row.get("host") or ""),
                str(row.get("user") or ""),
                str(row.get("port") or 22),
                str(row.get("remote_workdir") or "")[:40],
                str(row.get("slots") or 1),
                "✔" if row.get("enabled", True) else "✘",
                key=str(row.get("id", "")),
            )
        hint = f"[dim]{len(connections)} connection(s)[/]" if connections else "[dim]No SSH connections configured[/]"
        try:
            self.query_one("#ssh-conn-hint", Static).update(hint)
        except NoMatches:
            pass

    def _ssh_read_form(self) -> dict[str, str]:
        """Read all SSH form input fields into a plain dict."""
        return {
            "label": self._val("ssh-label"),
            "host": self._val("ssh-host"),
            "user": self._val("ssh-user"),
            "port": self._val("ssh-port") or "22",
            "key_path": self._val("ssh-key-path"),
            "remote_workdir": self._val("ssh-remote-workdir"),
            "python_bin": self._val("ssh-python-bin") or "python3",
            "script_path": self._val("ssh-script-path"),
            "slots": self._val("ssh-slots") or "1",
            "path_map": self._val("ssh-path-map"),
            "env_json": self._val("ssh-env-json"),
        }

    def _ssh_parse_json_field(self, value: str, field_name: str) -> dict[str, str] | None:
        """Parse a JSON dict field from the SSH form. Returns None on error."""
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"{field_name} must be a JSON object")
            return {str(k): str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, ValueError) as exc:
            self._ssh_set_status(f"[red]✗ {field_name}: {exc}[/]")
            return None

    def _ssh_set_status(self, msg: str) -> None:
        try:
            self.query_one("#ssh-form-status", Static).update(msg)
        except NoMatches:
            pass

    def _ssh_clear_form(self) -> None:
        """Reset SSH form to default values."""
        self._ssh_selected_id = None
        for wid, default in [
            ("ssh-label", ""),
            ("ssh-host", ""),
            ("ssh-user", ""),
            ("ssh-port", "22"),
            ("ssh-key-path", ""),
            ("ssh-remote-workdir", ""),
            ("ssh-python-bin", "python3"),
            ("ssh-script-path", ""),
            ("ssh-slots", "1"),
            ("ssh-path-map", ""),
            ("ssh-env-json", ""),
        ]:
            try:
                self.query_one(f"#{wid}", Input).value = default
            except NoMatches:
                pass
        self._ssh_set_status("")

    def _ssh_fill_form_from_row(self, row: dict) -> None:
        """Fill SSH form fields from a DB row dict."""
        import json as _json
        self._ssh_selected_id = row.get("id")
        for wid, key, default in [
            ("ssh-label", "label", ""),
            ("ssh-host", "host", ""),
            ("ssh-user", "user", ""),
            ("ssh-port", "port", "22"),
            ("ssh-key-path", "key_path", ""),
            ("ssh-remote-workdir", "remote_workdir", ""),
            ("ssh-python-bin", "python_bin", "python3"),
            ("ssh-script-path", "script_path", ""),
            ("ssh-slots", "slots", "1"),
        ]:
            try:
                self.query_one(f"#{wid}", Input).value = str(row.get(key) or default)
            except NoMatches:
                pass
        # JSON dict fields
        for wid, key in [("ssh-path-map", "path_map"), ("ssh-env-json", "env")]:
            raw = row.get(key, {})
            text = _json.dumps(raw, ensure_ascii=False) if raw else ""
            try:
                self.query_one(f"#{wid}", Input).value = text
            except NoMatches:
                pass

    def _ssh_add_connection(self) -> None:
        """Read form and create a new SSH connection in DB."""
        form = self._ssh_read_form()
        if not form["host"]:
            self._ssh_set_status("[red]✗ Host is required[/]")
            return
        if not form["remote_workdir"]:
            self._ssh_set_status("[red]✗ Remote workdir is required[/]")
            return
        path_map = self._ssh_parse_json_field(form["path_map"], "path_map")
        if path_map is None:
            return
        env = self._ssh_parse_json_field(form["env_json"], "env")
        if env is None:
            return
        try:
            port = max(1, int(form["port"] or "22"))
            slots = max(1, int(form["slots"] or "1"))
        except ValueError:
            self._ssh_set_status("[red]✗ Port and Slots must be integers[/]")
            return
        _db.add_ssh_connection(
            host=form["host"],
            label=form["label"],
            user=form["user"],
            port=port,
            key_path=form["key_path"],
            remote_workdir=form["remote_workdir"],
            python_bin=form["python_bin"],
            script_path=form["script_path"],
            slots=slots,
            path_map=path_map,
            env=env,
        )
        self._refresh_ssh_table()
        self._refresh_remote_state()
        self._ssh_set_status(f"[green]✓ Added connection: {form['label'] or form['host']}[/]")

    def _ssh_update_connection(self) -> None:
        """Update the selected SSH connection with current form values."""
        if self._ssh_selected_id is None:
            self._ssh_set_status("[yellow]Select a connection from the table first[/]")
            return
        form = self._ssh_read_form()
        if not form["host"]:
            self._ssh_set_status("[red]✗ Host is required[/]")
            return
        path_map = self._ssh_parse_json_field(form["path_map"], "path_map")
        if path_map is None:
            return
        env = self._ssh_parse_json_field(form["env_json"], "env")
        if env is None:
            return
        try:
            port = max(1, int(form["port"] or "22"))
            slots = max(1, int(form["slots"] or "1"))
        except ValueError:
            self._ssh_set_status("[red]✗ Port and Slots must be integers[/]")
            return
        _db.update_ssh_connection(
            self._ssh_selected_id,
            host=form["host"],
            label=form["label"],
            user=form["user"],
            port=port,
            key_path=form["key_path"],
            remote_workdir=form["remote_workdir"],
            python_bin=form["python_bin"],
            script_path=form["script_path"],
            slots=slots,
            path_map=path_map,
            env=env,
        )
        self._refresh_ssh_table()
        self._refresh_remote_state()
        self._ssh_set_status(f"[green]✓ Updated connection ID {self._ssh_selected_id}[/]")

    def _ssh_delete_connection(self) -> None:
        if self._ssh_selected_id is None:
            self._ssh_set_status("[yellow]Select a connection to delete.[/]")
            return
        _db.delete_ssh_connection(self._ssh_selected_id)
        self._ssh_selected_id = None
        self._ssh_clear_form()
        self._refresh_ssh_table()
        self._refresh_remote_state()
        self._ssh_set_status("[green]✓ Connection deleted.[/]")
