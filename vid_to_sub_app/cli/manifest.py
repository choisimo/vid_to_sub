from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from vid_to_sub_app.db import Database

from .discovery import hash_video_folder
from .stage_artifact import StageArtifactMetadata

_folder_state_db = Database()


def build_run_manifest(
    videos: Sequence[str | Path],
    *,
    found_total: int | None = None,
    skipped: int = 0,
) -> dict[str, Any]:
    folders: dict[str, dict[str, Any]] = {}
    for raw_video in sorted(
        str(Path(video).expanduser().resolve()) for video in videos
    ):
        video_path = Path(raw_video)
        folder_path = str(video_path.parent)
        folder_hash = hash_video_folder(folder_path)
        folder = folders.setdefault(
            folder_hash,
            {
                "folder_hash": folder_hash,
                "folder_path": folder_path,
                "videos": [],
            },
        )
        folder["videos"].append(raw_video)

    ordered_folders = sorted(
        folders.values(),
        key=lambda item: (str(item["folder_path"]), str(item["folder_hash"])),
    )
    entries: list[dict[str, str]] = []
    max_videos = max((len(folder["videos"]) for folder in ordered_folders), default=0)
    for offset in range(max_videos):
        for folder in ordered_folders:
            videos_in_folder = folder["videos"]
            if offset >= len(videos_in_folder):
                continue
            entries.append(
                {
                    "video_path": videos_in_folder[offset],
                    "folder_path": str(folder["folder_path"]),
                    "folder_hash": str(folder["folder_hash"]),
                }
            )

    return {
        "found_total": found_total if found_total is not None else len(entries),
        "skipped": skipped,
        "folders": [
            {
                "folder_hash": str(folder["folder_hash"]),
                "folder_path": str(folder["folder_path"]),
                "total_files": len(folder["videos"]),
                "completed_files": 0,
                "status": "queued",
                "is_completed": False,
            }
            for folder in ordered_folders
        ],
        "entries": entries,
    }


def apply_runtime_path_map_to_manifest(
    manifest: dict[str, Any],
    path_mapper,
) -> dict[str, Any]:
    mapped_entries: list[dict[str, str]] = []
    for entry in manifest.get("entries", []):
        if not isinstance(entry, dict):
            continue
        video_path = str(entry.get("video_path") or "").strip()
        folder_path = str(entry.get("folder_path") or "").strip()
        folder_hash = str(entry.get("folder_hash") or "").strip()
        if not video_path or not folder_path or not folder_hash:
            continue
        mapped_entries.append(
            {
                "video_path": str(path_mapper(video_path)),
                "folder_path": folder_path,
                "folder_hash": folder_hash,
            }
        )

    mapped_folders: list[dict[str, Any]] = []
    for folder in manifest.get("folders", []):
        if not isinstance(folder, dict):
            continue
        folder_hash = str(folder.get("folder_hash") or "").strip()
        folder_path = str(folder.get("folder_path") or "").strip()
        if not folder_hash or not folder_path:
            continue
        mapped_folders.append(
            {
                "folder_hash": folder_hash,
                "folder_path": folder_path,
                "total_files": int(folder.get("total_files", 0)),
                "completed_files": int(folder.get("completed_files", 0)),
                "status": str(folder.get("status") or "queued"),
                "is_completed": bool(folder.get("is_completed")),
            }
        )

    return {
        "found_total": int(manifest.get("found_total", len(mapped_entries))),
        "skipped": int(manifest.get("skipped", 0)),
        "folders": mapped_folders,
        "entries": mapped_entries,
    }


def _normalize_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    folders: dict[str, dict[str, Any]] = {}
    for raw_folder in payload.get("folders", []):
        if not isinstance(raw_folder, dict):
            continue
        folder_hash = str(raw_folder.get("folder_hash") or "").strip()
        folder_path = str(raw_folder.get("folder_path") or "").strip()
        if not folder_hash or not folder_path:
            continue
        folders[folder_hash] = {
            "folder_hash": folder_hash,
            "folder_path": folder_path,
            "total_files": 0,
            "completed_files": max(0, int(raw_folder.get("completed_files", 0))),
            "status": str(raw_folder.get("status") or "queued"),
            "is_completed": bool(raw_folder.get("is_completed")),
        }

    entries: list[dict[str, str]] = []
    for raw_entry in payload.get("entries", []):
        if not isinstance(raw_entry, dict):
            continue
        video_path = str(raw_entry.get("video_path") or "").strip()
        folder_path = str(raw_entry.get("folder_path") or "").strip()
        folder_hash = str(raw_entry.get("folder_hash") or "").strip()
        if not video_path:
            continue
        runtime_video_path = str(Path(video_path).expanduser().resolve())
        resolved_folder_path = folder_path or str(Path(runtime_video_path).parent)
        resolved_folder_hash = folder_hash or hash_video_folder(resolved_folder_path)
        if resolved_folder_hash not in folders:
            folders[resolved_folder_hash] = {
                "folder_hash": resolved_folder_hash,
                "folder_path": resolved_folder_path,
                "total_files": 0,
                "completed_files": 0,
                "status": "queued",
                "is_completed": False,
            }
        folders[resolved_folder_hash]["total_files"] += 1
        entries.append(
            {
                "video_path": runtime_video_path,
                "folder_path": resolved_folder_path,
                "folder_hash": resolved_folder_hash,
            }
        )

    return {
        "found_total": int(payload.get("found_total", len(entries))),
        "skipped": int(payload.get("skipped", 0)),
        "folders": list(folders.values()),
        "entries": entries,
    }


def load_manifest_from_stdin() -> dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        raise ValueError("Manifest stdin was empty.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest stdin is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Manifest payload must be a JSON object.")
    manifest = _normalize_manifest(payload)
    if not manifest["entries"]:
        raise ValueError("Manifest did not contain any video entries.")
    return manifest


def persist_folder_manifest_state(manifest: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for folder in manifest.get("folders", []):
        if not isinstance(folder, dict):
            continue
        folder_hash = str(folder.get("folder_hash") or "").strip()
        folder_path = str(folder.get("folder_path") or "").strip()
        if not folder_hash or not folder_path:
            continue
        rows.append(
            {
                "folder_hash": folder_hash,
                "folder_path": folder_path,
                "status": str(folder.get("status") or "queued"),
                "total_files": max(0, int(folder.get("total_files", 0))),
                "completed_files": max(0, int(folder.get("completed_files", 0))),
                "is_completed": bool(folder.get("is_completed")),
            }
        )
    _folder_state_db.upsert_folder_queue_states(rows)


@dataclass(slots=True)
class ProcessResult:
    success: bool
    video_path: str
    folder_hash: str
    folder_path: str
    worker_id: int
    language: str | None = None
    video_duration: float | None = None
    output_paths: list[str] | None = None
    segments: int | None = None
    elapsed_sec: float | None = None
    error: str | None = None
    stage: str | None = None
    artifact_path: str | None = None
    artifact_metadata: StageArtifactMetadata | None = None


class FolderAwareScheduler:
    def __init__(self, manifest: dict[str, Any]) -> None:
        self._entries: list[dict[str, str]] = [
            entry for entry in manifest.get("entries", []) if isinstance(entry, dict)
        ]
        self._lock = threading.Lock()
        self._next_index = 0
        self._folders: dict[str, dict[str, Any]] = {}
        for folder in manifest.get("folders", []):
            if not isinstance(folder, dict):
                continue
            folder_hash = str(folder.get("folder_hash") or "").strip()
            folder_path = str(folder.get("folder_path") or "").strip()
            if not folder_hash or not folder_path:
                continue
            self._folders[folder_hash] = {
                "folder_hash": folder_hash,
                "folder_path": folder_path,
                "total_files": max(0, int(folder.get("total_files", 0))),
                "completed_files": max(0, int(folder.get("completed_files", 0))),
                "failed_files": 0,
                "status": str(folder.get("status") or "queued"),
                "is_completed": bool(folder.get("is_completed")),
            }
        seeded_hashes = set(self._folders)
        for entry in self._entries:
            folder_hash = str(entry["folder_hash"])
            if folder_hash not in self._folders:
                self._folders[folder_hash] = {
                    "folder_hash": folder_hash,
                    "folder_path": str(entry["folder_path"]),
                    "total_files": 0,
                    "completed_files": 0,
                    "failed_files": 0,
                    "status": "queued",
                    "is_completed": False,
                }
            if folder_hash not in seeded_hashes:
                self._folders[folder_hash]["total_files"] += 1

    def claim_next(self) -> dict[str, str] | None:
        with self._lock:
            if self._next_index >= len(self._entries):
                return None
            entry = dict(self._entries[self._next_index])
            self._next_index += 1
            folder = self._folders[entry["folder_hash"]]
            if folder["status"] == "queued":
                folder["status"] = "running"
                folder["is_completed"] = False
                persist_folder_manifest_state({"folders": [dict(folder)]})
            return entry

    def complete(self, result: ProcessResult) -> dict[str, Any]:
        with self._lock:
            folder = self._folders[result.folder_hash]
            if result.success:
                folder["completed_files"] += 1
            else:
                folder["failed_files"] += 1
            processed = folder["completed_files"] + folder["failed_files"]
            if folder["failed_files"] > 0:
                folder["status"] = "failed"
                folder["is_completed"] = False
            elif processed >= folder["total_files"]:
                folder["status"] = "completed"
                folder["is_completed"] = True
            else:
                folder["status"] = "running"
                folder["is_completed"] = False
            snapshot = {
                "folder_hash": folder["folder_hash"],
                "folder_path": folder["folder_path"],
                "total_files": folder["total_files"],
                "completed_files": folder["completed_files"],
                "status": folder["status"],
                "is_completed": folder["is_completed"],
            }
            persist_folder_manifest_state({"folders": [snapshot]})
            return snapshot
