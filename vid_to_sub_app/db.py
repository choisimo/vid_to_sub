#!/usr/bin/env python3
"""
db.py — SQLite persistence layer for vid_to_sub
=================================================
Stores:
  settings     — all configuration (replaces .env)
  jobs         — per-file transcription history
  recent_paths — recently used input paths
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

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
    segments     INTEGER
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
    "VID_TO_SUB_AGENT_BASE_URL",
    "VID_TO_SUB_AGENT_API_KEY",
    "VID_TO_SUB_AGENT_MODEL",
)

_DEFAULT_SETTINGS: dict[str, str] = {
    # Environment variable settings
    "VID_TO_SUB_WHISPER_CPP_BIN": "",
    "VID_TO_SUB_WHISPER_CPP_MODEL": "",
    "VID_TO_SUB_TRANSLATION_BASE_URL": "",
    "VID_TO_SUB_TRANSLATION_API_KEY": "",
    "VID_TO_SUB_TRANSLATION_MODEL": "",
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
    "tui.default_translate_to": "ko",
    "tui.default_formats": '["srt"]',
    "tui.execution_mode": "local",
    "tui.remote_resources": "[]",
}


class Database:
    """Thread-safe SQLite wrapper using per-thread connections."""

    def __init__(self, path: Path = DB_PATH) -> None:
        self._path = path
        self._local = threading.local()
        self._init_schema()

    # ── Internal ──────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        self._conn().executescript(_SCHEMA)
        self._conn().commit()

    # ── Settings ──────────────────────────────────────────────────────────

    def get(self, key: str, default: str = "") -> str:
        row = (
            self._conn()
            .execute("SELECT value FROM settings WHERE key=?", (key,))
            .fetchone()
        )
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, value)
        )
        self._conn().commit()

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
        return cur.lastrowid  # type: ignore[return-value]

    def finish_job(
        self,
        job_id: int,
        status: str,
        output_paths: list[str] | None = None,
        error: str | None = None,
        wall_sec: float | None = None,
        video_dur: float | None = None,
        segments: int | None = None,
    ) -> None:
        self._conn().execute(
            """UPDATE jobs
               SET status=?,output_paths=?,error=?,wall_sec=?,video_dur=?,segments=?
               WHERE id=?""",
            (
                status,
                json.dumps(output_paths or []),
                error,
                wall_sec,
                video_dur,
                segments,
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
        return [dict(r) for r in rows]

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
