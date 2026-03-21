"""
tui/models.py — Pure data classes and conversion helpers for the TUI layer.

All dataclasses here are:
  - Import-free of Textual / UI primitives (no circular deps)
  - Used by both helpers.py (logic) and app.py (UI binding)

Design:
  SSHConnection    — structured SSH server profile (backed by ssh_connections table)
  RemoteResourceProfile — runtime execution profile derived from SSHConnection
  ExecutorPlan     — immutable execution plan for one worker slot
  RunJobState      — mutable in-flight job state
  RunConfig        — full snapshot of user-chosen run parameters
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# SSH connection profile (DB-backed)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SSHConnection:
    """Structured SSH server profile stored in ``ssh_connections`` table.

    All fields map 1:1 to DB columns so that round-trip serialization is
    trivial.  Use :func:`ssh_connection_from_row` and
    :func:`ssh_connection_to_row` for DB I/O.
    """

    id: int | None  # None for unsaved (new) connections
    label: str  # human-readable name shown in the UI
    host: str  # hostname or IP
    user: str  # SSH username (empty = use system default)
    port: int  # SSH port (default 22)
    key_path: str  # path to private key file (empty = agent/default)
    remote_workdir: str  # working directory on the remote host
    python_bin: str  # python executable on the remote host
    script_path: (
        str  # path to vid_to_sub.py on the remote host (empty = workdir/vid_to_sub.py)
    )
    slots: int  # parallel workers on this host
    path_map: dict[str, str]  # local → remote path prefix mapping
    env: dict[str, str]  # per-host env var overrides
    enabled: bool  # whether this connection is active

    @property
    def display_name(self) -> str:
        """Short label for UI drop-downs and tables."""
        if self.label:
            return self.label
        user_part = f"{self.user}@" if self.user else ""
        port_part = f":{self.port}" if self.port != 22 else ""
        return f"{user_part}{self.host}{port_part}"

    @property
    def ssh_target(self) -> str:
        """``user@host`` string suitable for the ssh command."""
        if self.user:
            return f"{self.user}@{self.host}"
        return self.host

    def to_remote_resource_profile(self) -> RemoteResourceProfile:
        """Convert to the runtime execution representation."""
        return RemoteResourceProfile(
            name=self.display_name,
            ssh_target=self.ssh_target,
            remote_workdir=self.remote_workdir,
            python_bin=self.python_bin or "python3",
            script_path=self.script_path,
            slots=max(1, self.slots),
            path_map=dict(self.path_map),
            env=dict(self.env),
            ssh_key_path=self.key_path,
            ssh_port=self.port,
            connection_id=self.id,
        )


def ssh_connection_from_row(row: dict[str, Any]) -> SSHConnection:
    """Construct :class:`SSHConnection` from a db.get_ssh_connection() dict."""
    return SSHConnection(
        id=row.get("id"),
        label=str(row.get("label") or ""),
        host=str(row.get("host") or ""),
        user=str(row.get("user") or ""),
        port=int(row.get("port") or 22),
        key_path=str(row.get("key_path") or ""),
        remote_workdir=str(row.get("remote_workdir") or ""),
        python_bin=str(row.get("python_bin") or "python3"),
        script_path=str(row.get("script_path") or ""),
        slots=max(1, int(row.get("slots") or 1)),
        path_map=_ensure_str_dict(row.get("path_map", {})),
        env=_ensure_str_dict(row.get("env", {})),
        enabled=bool(row.get("enabled", True)),
    )


def _ensure_str_dict(value: Any) -> dict[str, str]:
    """Coerce any value to ``dict[str, str]``, handling JSON strings."""
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Runtime execution profile (derived from SSHConnection or legacy JSON)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RemoteResourceProfile:
    """Runtime profile for a single remote executor.

    Derived from :class:`SSHConnection` via
    :meth:`SSHConnection.to_remote_resource_profile`, or from legacy raw JSON
    via :func:`parse_remote_resources` for backward compatibility.
    """

    name: str
    ssh_target: str
    remote_workdir: str
    python_bin: str = "python3"
    script_path: str = ""
    slots: int = 1
    path_map: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    # Extended fields (only populated from SSHConnection)
    ssh_key_path: str = ""
    ssh_port: int = 22
    connection_id: int | None = None

    def ssh_command_prefix(self) -> list[str]:
        """Build the ``ssh [options] target`` prefix for subprocess calls."""
        cmd = ["ssh"]
        if self.ssh_port != 22:
            cmd += ["-p", str(self.ssh_port)]
        if self.ssh_key_path:
            cmd += ["-i", self.ssh_key_path]
        cmd.append(self.ssh_target)
        return cmd


# ---------------------------------------------------------------------------
# Executor plan (immutable, per worker slot)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExecutorPlan:
    """Immutable execution plan assigned to one worker/executor."""

    name: str
    kind: str  # "local" | "remote"
    label: str  # human-readable label for progress display
    cmd: list[str]
    env: dict[str, str] | None
    assigned_paths: list[str]
    capacity: int
    manifest: dict[str, Any]
    stdin_payload: str | None = None
    # Stage annotation: "full" | "stage1" | "stage2"
    # "full"   — transcription + inline translation (default, all existing plans)
    # "stage1" — transcription only; writes .stage1.json artifacts
    # "stage2" — translation only from pre-existing .stage1.json artifacts
    stage: str = "full"


# ---------------------------------------------------------------------------
# In-flight job state (mutable, tracked during a run)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunJobState:
    """Mutable state for one in-flight transcription job."""

    video_path: str
    executor: str
    job_id: int | None
    started_at: float
    status: str = "running"
    video_duration: float | None = None
    progress_seconds: float | None = None
    progress_ratio: float | None = None


# ---------------------------------------------------------------------------
# Run configuration snapshot (immutable, captured at run start)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RunConfig:
    """Full snapshot of user-chosen run parameters.

    Created once at run start and threaded through the executor pipeline so
    that UI changes during execution do not affect the active run.
    """

    request_id: int
    selected_paths: list[str]
    output_dir: str | None
    formats: frozenset[str]
    no_recurse: bool
    skip_existing: bool
    dry_run: bool
    verbose: bool
    backend: str
    model: str
    device: str
    language: str | None
    compute_type: str | None
    beam_size: str
    local_workers: int
    whisper_cpp_model_path: str | None
    translate_enabled: bool
    translate_to: str | None
    translation_model: str | None
    translation_base_url: str | None
    translation_api_key: str | None
    postprocess_enabled: bool
    postprocess_mode: str
    postprocess_model: str | None
    postprocess_base_url: str | None
    postprocess_api_key: str | None
    diarize: bool
    hf_token: str | None
    execution_mode: str
    remote_resources: list[RemoteResourceProfile]
    run_env: dict[str, str]
