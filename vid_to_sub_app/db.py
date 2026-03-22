#!/usr/bin/env python3
"""
db.py — SQLite persistence layer for vid_to_sub
=================================================
Stores:
  settings        — all configuration (replaces .env); SQLite is primary source of truth
  ssh_connections — SSH remote server connection profiles (structured, not raw JSON)
  jobs            — per-file transcription history
  recent_paths    — recently used input paths
"""

from __future__ import annotations

import atexit
import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vid_to_sub_app.cli.stage_artifact import StageArtifactMetadata
else:
    StageArtifactMetadata = dict[str, Any]

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "vid_to_sub.db"

_SCHEMA = """\
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    video_path   TEXT NOT NULL,
    output_dir   TEXT,
    output_paths TEXT NOT NULL DEFAULT '[]',
    backend      TEXT NOT NULL DEFAULT 'whisper-cpp',
    model        TEXT NOT NULL DEFAULT 'large-v3',
    language     TEXT,
    target_lang  TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    error        TEXT,
    wall_sec     REAL,
    video_dur    REAL,
    segments     INTEGER,
    artifact_path TEXT,
    artifact_metadata TEXT
);

CREATE TABLE IF NOT EXISTS recent_paths (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    path     TEXT UNIQUE NOT NULL,
    kind     TEXT NOT NULL DEFAULT 'directory',
    used_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS folder_queue_state (
    folder_hash     TEXT PRIMARY KEY,
    folder_path     TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'queued',
    total_files     INTEGER NOT NULL DEFAULT 0,
    completed_files INTEGER NOT NULL DEFAULT 0,
    is_completed    INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ssh_connections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    label        TEXT NOT NULL DEFAULT '',
    host         TEXT NOT NULL,
    user         TEXT NOT NULL DEFAULT '',
    port         INTEGER NOT NULL DEFAULT 22,
    key_path     TEXT NOT NULL DEFAULT '',
    remote_workdir TEXT NOT NULL,
    python_bin   TEXT NOT NULL DEFAULT 'python3',
    script_path  TEXT NOT NULL DEFAULT '',
    slots        INTEGER NOT NULL DEFAULT 1,
    path_map     TEXT NOT NULL DEFAULT '{}',
    env          TEXT NOT NULL DEFAULT '{}',
    enabled      INTEGER NOT NULL DEFAULT 1,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ssh_connections_label ON ssh_connections(label);


CREATE INDEX IF NOT EXISTS idx_jobs_ts   ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_recent_ts ON recent_paths(used_at DESC);
CREATE INDEX IF NOT EXISTS idx_folder_queue_state_updated_at
    ON folder_queue_state(updated_at DESC);
"""

# Keys that map directly to environment variables consumed by vid_to_sub.py
ENV_KEYS = (
    "VID_TO_SUB_WHISPER_CPP_BIN",
    "VID_TO_SUB_WHISPER_CPP_MODEL",
    "VID_TO_SUB_TRANSLATION_BASE_URL",
    "VID_TO_SUB_TRANSLATION_API_KEY",
    "VID_TO_SUB_TRANSLATION_MODEL",
    "VID_TO_SUB_POSTPROCESS_BASE_URL",
    "VID_TO_SUB_POSTPROCESS_API_KEY",
    "VID_TO_SUB_POSTPROCESS_MODEL",
    "VID_TO_SUB_AGENT_BASE_URL",
    "VID_TO_SUB_AGENT_API_KEY",
    "VID_TO_SUB_AGENT_MODEL",
)

TUI_DEFAULT_TRANSLATE_ENABLED_KEY = "tui.default_translate_enabled"

_DEFAULT_SETTINGS: dict[str, str] = {
    # Environment variable settings
    "VID_TO_SUB_WHISPER_CPP_BIN": "",
    "VID_TO_SUB_WHISPER_CPP_MODEL": "",
    "VID_TO_SUB_TRANSLATION_BASE_URL": "",
    "VID_TO_SUB_TRANSLATION_API_KEY": "",
    "VID_TO_SUB_TRANSLATION_MODEL": "",
    "VID_TO_SUB_POSTPROCESS_BASE_URL": "",
    "VID_TO_SUB_POSTPROCESS_API_KEY": "",
    "VID_TO_SUB_POSTPROCESS_MODEL": "",
    "VID_TO_SUB_AGENT_BASE_URL": "",
    "VID_TO_SUB_AGENT_API_KEY": "",
    "VID_TO_SUB_AGENT_MODEL": "",
    # TUI-specific settings
    "tui.build_dir": str(Path.home() / ".cache" / "vid_to_sub_build"),
    "tui.install_dir": str(Path.home() / ".local" / "bin"),
    "tui.model_dir": str(ROOT_DIR / "models"),
    "tui.browse_root": str(Path.home()),
    "tui.default_backend": "whisper-cpp",
    "tui.default_model": "large-v3",
    "tui.default_device": "cpu",
    "tui.default_language": "",
    "tui.default_output_dir": "",
    TUI_DEFAULT_TRANSLATE_ENABLED_KEY: "1",
    "tui.default_translate_to": "ko",
    "tui.default_formats": '["srt"]',
    "tui.execution_mode": "local",
    "tui.remote_resources": "[]",
    # SSH connection manager (structured, see ssh_connections table)
    # legacy: tui.remote_resources kept for backward compat but new code uses ssh_connections table
}


class Database:
    """Thread-safe SQLite wrapper using per-thread connections."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self._path = path
        self._local = threading.local()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._connections_lock = threading.Lock()
        self._init_schema()
        atexit.register(self.close)

    # ── Internal ──────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            with self._connections_lock:
                self._connections[threading.get_ident()] = conn
        return conn

    def close(self) -> None:
        with self._connections_lock:
            connections = list(self._connections.values())
            self._connections.clear()
        for conn in connections:
            try:
                conn.close()
            except Exception:
                pass
        self._local.conn = None

    def _init_schema(self) -> None:
        self._conn().executescript(_SCHEMA)
        # Backward-safe column additions for existing databases
        for stmt in (
            "ALTER TABLE jobs ADD COLUMN artifact_path TEXT",
            "ALTER TABLE jobs ADD COLUMN artifact_metadata TEXT",
        ):
            try:
                self._conn().execute(stmt)
                self._conn().commit()
            except Exception:
                pass  # column already exists

    def _normalize_artifact_metadata(
        self,
        raw_metadata: Any,
        artifact_path: str | None,
    ) -> StageArtifactMetadata | None:
        metadata: StageArtifactMetadata | None = None
        if isinstance(raw_metadata, str) and raw_metadata.strip():
            try:
                decoded = json.loads(raw_metadata)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                metadata = cast(
                    StageArtifactMetadata,
                    cast(
                        object,
                        {
                            str(key): value
                            for key, value in decoded.items()
                            if isinstance(key, str)
                        },
                    ),
                )
        elif isinstance(raw_metadata, dict):
            metadata = cast(
                StageArtifactMetadata,
                cast(
                    object,
                    {
                        str(key): value
                        for key, value in raw_metadata.items()
                        if isinstance(key, str)
                    },
                ),
            )

        resolved_path = str(artifact_path or "").strip()
        if metadata is None:
            return {"path": resolved_path} if resolved_path else None
        if resolved_path and not str(metadata.get("path") or "").strip():
            metadata["path"] = resolved_path
        return metadata or None

    # ── Settings ──────────────────────────────────────────────────────────

    def get(self, key: str, default: str = "") -> str:
        row = (
            self._conn()
            .execute("SELECT value FROM settings WHERE key=?", (key,))
            .fetchone()
        )
        return row["value"] if row else default

    def get_setting(self, key: str, default: str | bool = "") -> str | bool:
        row = (
            self._conn()
            .execute("SELECT value FROM settings WHERE key=?", (key,))
            .fetchone()
        )
        if row is None:
            return default
        if isinstance(default, bool):
            return row["value"] == "1"
        return row["value"]

    def set(self, key: str, value: str) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, value)
        )
        self._conn().commit()

    def set_setting(self, key: str, value: str | bool) -> None:
        if isinstance(value, bool):
            self.set(key, "1" if value else "0")
            return
        self.set(key, value)

    def get_all(self) -> dict[str, str]:
        rows = self._conn().execute("SELECT key,value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def set_many(self, data: dict[str, str]) -> None:
        self._conn().executemany(
            "INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)",
            list(data.items()),
        )
        self._conn().commit()

    def seed_defaults(self) -> None:
        """Insert default values only for keys that do not yet exist."""
        existing = set(self.get_all())
        missing = {k: v for k, v in _DEFAULT_SETTINGS.items() if k not in existing}
        if missing:
            self.set_many(missing)

    def get_env_dict(self) -> dict[str, str]:
        """Return only the VID_TO_SUB_ settings with non-empty values."""
        all_s = self.get_all()
        return {k: v for k, v in all_s.items() if k in ENV_KEYS and v}

    # ── Jobs ──────────────────────────────────────────────────────────────

    def create_job(
        self,
        video_path: str,
        backend: str,
        model: str,
        output_dir: str | None = None,
        language: str | None = None,
        target_lang: str | None = None,
    ) -> int:
        cur = self._conn().execute(
            """INSERT INTO jobs
               (created_at,video_path,output_dir,backend,model,language,target_lang,status)
               VALUES(?,?,?,?,?,?,?,'running')""",
            (
                datetime.now().isoformat(timespec="seconds"),
                video_path,
                output_dir,
                backend,
                model,
                language,
                target_lang,
            ),
        )
        self._conn().commit()
        job_id = cur.lastrowid
        if job_id is None:
            raise RuntimeError("Failed to create job row")
        return int(job_id)

    def finish_job(
        self,
        job_id: int,
        status: str,
        output_paths: list[str] | None = None,
        error: str | None = None,
        wall_sec: float | None = None,
        video_dur: float | None = None,
        segments: int | None = None,
        artifact_path: str | None = None,
        artifact_metadata: StageArtifactMetadata | dict[str, Any] | None = None,
    ) -> None:
        self._conn().execute(
            """UPDATE jobs
               SET status=?,output_paths=?,error=?,wall_sec=?,video_dur=?,segments=?,
                   artifact_path=?,artifact_metadata=?
               WHERE id=?""",
            (
                status,
                json.dumps(output_paths or []),
                error,
                wall_sec,
                video_dur,
                segments,
                artifact_path,
                json.dumps(artifact_metadata)
                if artifact_metadata is not None
                else None,
                job_id,
            ),
        )
        self._conn().commit()

    def get_jobs(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = (
            self._conn()
            .execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            .fetchall()
        )
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            metadata = self._normalize_artifact_metadata(
                item.get("artifact_metadata"),
                item.get("artifact_path"),
            )
            item["artifact_metadata"] = metadata
            if not item.get("artifact_path") and metadata is not None:
                item["artifact_path"] = metadata.get("path")
            normalized_rows.append(item)
        return normalized_rows

    def delete_job(self, job_id: int) -> None:
        self._conn().execute("DELETE FROM jobs WHERE id=?", (job_id,))
        self._conn().commit()

    def clear_jobs(self) -> None:
        self._conn().execute("DELETE FROM jobs")
        self._conn().commit()

    # ── Folder Queue State ────────────────────────────────────────────────

    def upsert_folder_queue_state(
        self,
        folder_hash: str,
        folder_path: str,
        *,
        status: str,
        total_files: int,
        completed_files: int,
        is_completed: bool,
    ) -> None:
        self._conn().execute(
            """INSERT INTO folder_queue_state
               (folder_hash,folder_path,status,total_files,completed_files,is_completed,updated_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(folder_hash) DO UPDATE SET
                   folder_path=excluded.folder_path,
                   status=excluded.status,
                   total_files=excluded.total_files,
                   completed_files=excluded.completed_files,
                   is_completed=excluded.is_completed,
                   updated_at=excluded.updated_at""",
            (
                folder_hash,
                folder_path,
                status,
                max(0, int(total_files)),
                max(0, int(completed_files)),
                1 if is_completed else 0,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        self._conn().commit()

    def upsert_folder_queue_states(self, rows: list[dict[str, Any]]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        payload = [
            (
                str(row["folder_hash"]),
                str(row["folder_path"]),
                str(row.get("status") or "queued"),
                max(0, int(row.get("total_files", 0))),
                max(0, int(row.get("completed_files", 0))),
                1 if bool(row.get("is_completed")) else 0,
                now,
            )
            for row in rows
            if row.get("folder_hash") and row.get("folder_path")
        ]
        if not payload:
            return
        self._conn().executemany(
            """INSERT INTO folder_queue_state
               (folder_hash,folder_path,status,total_files,completed_files,is_completed,updated_at)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(folder_hash) DO UPDATE SET
                   folder_path=excluded.folder_path,
                   status=excluded.status,
                   total_files=excluded.total_files,
                   completed_files=excluded.completed_files,
                   is_completed=excluded.is_completed,
                   updated_at=excluded.updated_at""",
            payload,
        )
        self._conn().commit()

    def get_folder_queue_states(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = (
            self._conn()
            .execute(
                """SELECT *
                   FROM folder_queue_state
                   ORDER BY updated_at DESC, folder_path ASC
                   LIMIT ?""",
                (limit,),
            )
            .fetchall()
        )
        return [dict(r) for r in rows]

    # ── Recent Paths ──────────────────────────────────────────────────────

    def touch_path(self, path: str, kind: str = "directory") -> None:
        self._conn().execute(
            """INSERT INTO recent_paths(path,kind,used_at) VALUES(?,?,?)
               ON CONFLICT(path)
               DO UPDATE SET used_at=excluded.used_at, kind=excluded.kind""",
            (path, kind, datetime.now().isoformat(timespec="seconds")),
        )
        self._conn().commit()

    def get_recent_paths(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = (
            self._conn()
            .execute(
                "SELECT * FROM recent_paths ORDER BY used_at DESC LIMIT ?", (limit,)
            )
            .fetchall()
        )
        return [dict(r) for r in rows]

    def remove_path(self, path: str) -> None:
        self._conn().execute("DELETE FROM recent_paths WHERE path=?", (path,))
        self._conn().commit()

    # ── SSH Connections ───────────────────────────────────────────────────

    def add_ssh_connection(
        self,
        host: str,
        *,
        label: str = "",
        user: str = "",
        port: int = 22,
        key_path: str = "",
        remote_workdir: str = "",
        python_bin: str = "python3",
        script_path: str = "",
        slots: int = 1,
        path_map: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        enabled: bool = True,
    ) -> int:
        """Insert a new SSH connection profile.  Returns the new row id."""
        now = datetime.now().isoformat(timespec="seconds")
        cur = self._conn().execute(
            """INSERT INTO ssh_connections
               (label,host,user,port,key_path,remote_workdir,python_bin,script_path,
                slots,path_map,env,enabled,created_at,updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                label.strip(),
                host.strip(),
                user.strip(),
                max(1, int(port)),
                key_path.strip(),
                remote_workdir.strip(),
                python_bin.strip() or "python3",
                script_path.strip(),
                max(1, int(slots)),
                json.dumps(path_map or {}),
                json.dumps(env or {}),
                1 if enabled else 0,
                now,
                now,
            ),
        )
        self._conn().commit()
        conn_id = cur.lastrowid
        if conn_id is None:
            raise RuntimeError("Failed to create SSH connection row")
        return int(conn_id)

    def update_ssh_connection(
        self,
        conn_id: int,
        *,
        label: str | None = None,
        host: str | None = None,
        user: str | None = None,
        port: int | None = None,
        key_path: str | None = None,
        remote_workdir: str | None = None,
        python_bin: str | None = None,
        script_path: str | None = None,
        slots: int | None = None,
        path_map: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Partial-update an SSH connection profile by id."""
        fields: list[str] = []
        values: list[Any] = []

        def _add(col: str, val: Any, transform: Any = None) -> None:
            if val is None:
                return
            fields.append(f"{col}=?")
            values.append(transform(val) if transform else val)

        _add("label", label, str.strip if label is not None else None)
        _add("host", host, str.strip if host is not None else None)
        _add("user", user, str.strip if user is not None else None)
        _add("port", port, lambda p: max(1, int(p)))
        _add("key_path", key_path, str.strip if key_path is not None else None)
        _add(
            "remote_workdir",
            remote_workdir,
            str.strip if remote_workdir is not None else None,
        )
        _add("python_bin", python_bin, lambda p: p.strip() or "python3")
        _add("script_path", script_path, str.strip if script_path is not None else None)
        _add("slots", slots, lambda s: max(1, int(s)))
        if path_map is not None:
            fields.append("path_map=?")
            values.append(json.dumps(path_map))
        if env is not None:
            fields.append("env=?")
            values.append(json.dumps(env))
        if enabled is not None:
            fields.append("enabled=?")
            values.append(1 if enabled else 0)

        if not fields:
            return

        fields.append("updated_at=?")
        values.append(datetime.now().isoformat(timespec="seconds"))
        values.append(conn_id)
        self._conn().execute(
            f"UPDATE ssh_connections SET {', '.join(fields)} WHERE id=?",
            values,
        )
        self._conn().commit()

    def delete_ssh_connection(self, conn_id: int) -> None:
        self._conn().execute("DELETE FROM ssh_connections WHERE id=?", (conn_id,))
        self._conn().commit()

    def get_ssh_connections(
        self, *, enabled_only: bool = False
    ) -> list[dict[str, Any]]:
        """Return all SSH connection profiles, newest first."""
        if enabled_only:
            rows = (
                self._conn()
                .execute(
                    "SELECT * FROM ssh_connections WHERE enabled=1 ORDER BY label, id",
                )
                .fetchall()
            )
        else:
            rows = (
                self._conn()
                .execute(
                    "SELECT * FROM ssh_connections ORDER BY label, id",
                )
                .fetchall()
            )
        result = []
        for r in rows:
            row = dict(r)
            # Deserialize JSON fields
            for field in ("path_map", "env"):
                raw = row.get(field, "{}")
                try:
                    row[field] = json.loads(raw) if raw else {}
                except (json.JSONDecodeError, TypeError):
                    row[field] = {}
            result.append(row)
        return result

    def get_ssh_connection(self, conn_id: int) -> dict[str, Any] | None:
        row = (
            self._conn()
            .execute("SELECT * FROM ssh_connections WHERE id=?", (conn_id,))
            .fetchone()
        )
        if row is None:
            return None
        result = dict(row)
        for field in ("path_map", "env"):
            raw = result.get(field, "{}")
            try:
                result[field] = json.loads(raw) if raw else {}
            except (json.JSONDecodeError, TypeError):
                result[field] = {}
        return result
