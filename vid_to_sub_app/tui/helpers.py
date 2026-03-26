from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vid_to_sub_app.cli.discovery import hash_video_folder
from vid_to_sub_app.shared.constants import (
    BACKENDS,
    ENV_AGENT_API_KEY,
    ENV_AGENT_BASE_URL,
    ENV_AGENT_MODEL,
    ENV_POSTPROCESS_API_KEY,
    ENV_POSTPROCESS_BASE_URL,
    ENV_POSTPROCESS_MODEL,
    ENV_TRANSLATION_API_KEY,
    ENV_TRANSLATION_BASE_URL,
    ENV_TRANSLATION_MODEL,
    ENV_WHISPER_CPP_BIN,
    ENV_WHISPER_CPP_MODEL,
    EVENT_PREFIX,
    EXECUTION_MODES,
    FORMATS,
    HF_MODEL_BASE,
    KNOWN_MODELS,
    MODEL_SEARCH_DIRS,
    PIP_REQUIREMENT_FILES,
    POSTPROCESS_MODES,
    ROOT_DIR,
    SEARCH_RESULT_LIMIT,
    SUBTITLE_OUTPUT_EXTENSIONS,
    SYSTEM_PACKAGE_MAP,
    VIDEO_EXTENSIONS,
)
from vid_to_sub_app.shared.constants import (
    DEFAULT_BACKEND as BASE_DEFAULT_BACKEND,
)
from vid_to_sub_app.shared.constants import (
    DEFAULT_DEVICE as BASE_DEFAULT_DEVICE,
)
from vid_to_sub_app.shared.constants import (
    DEFAULT_MODEL as BASE_DEFAULT_MODEL,
)
from vid_to_sub_app.shared.env import (
    discover_ggml_models as shared_discover_ggml_models,
)
from vid_to_sub_app.shared.env import (
    find_whisper_cpp_bin,
    resolve_runtime_backend_and_device,
    resolve_runtime_backend_threads,
    resolve_runtime_model,
)
from vid_to_sub_app.shared.env import (
    preferred_ggml_model_path as shared_preferred_ggml_model_path,
)

from .models import (
    ExecutorPlan,
    RemoteResourceProfile,
    RunConfig,
    RunJobState,
    SSHConnection,
    ssh_connection_from_row,
)
from .state import db as _db

DetectResult = dict[str, tuple[bool, str]]
SCRIPT_PATH = ROOT_DIR / "vid_to_sub.py"
ENV_FILE = ROOT_DIR / ".env"
VIDEO_INPUT_EXTENSIONS = VIDEO_EXTENSIONS
ENV_WCPP_BIN = ENV_WHISPER_CPP_BIN
ENV_WCPP_MODEL = ENV_WHISPER_CPP_MODEL
ENV_TRANS_URL = ENV_TRANSLATION_BASE_URL
ENV_TRANS_KEY = ENV_TRANSLATION_API_KEY
ENV_TRANS_MOD = ENV_TRANSLATION_MODEL
ENV_POST_URL = ENV_POSTPROCESS_BASE_URL
ENV_POST_KEY = ENV_POSTPROCESS_API_KEY
ENV_POST_MOD = ENV_POSTPROCESS_MODEL
ENV_AGENT_URL = ENV_AGENT_BASE_URL
ENV_AGENT_KEY = ENV_AGENT_API_KEY
ENV_AGENT_MOD = ENV_AGENT_MODEL
DEFAULT_BACKEND, DEFAULT_DEVICE = resolve_runtime_backend_and_device(
    BASE_DEFAULT_BACKEND,
    BASE_DEFAULT_DEVICE,
)
DEFAULT_MODEL = resolve_runtime_model(
    DEFAULT_BACKEND,
    DEFAULT_DEVICE,
    BASE_DEFAULT_MODEL,
)


def detect_all() -> DetectResult:
    """Scan system for required / optional dependencies."""
    results: DetectResult = {}
    ggml_models = discover_ggml_models()

    # ffmpeg
    p = shutil.which("ffmpeg")
    results["ffmpeg"] = (bool(p), p or "not in PATH")

    # whisper-cli
    found_wbin = find_whisper_cpp_bin(_db.get(ENV_WCPP_BIN))
    results["whisper-cli"] = (bool(found_wbin), found_wbin or "not found")

    # ggml model
    wmodel = _db.get(ENV_WCPP_MODEL)
    if wmodel and Path(wmodel).exists():
        results["ggml-model"] = (True, wmodel)
    else:
        found_model = preferred_ggml_model_path(ggml_models)
        results["ggml-model"] = (
            bool(found_model),
            found_model or "no ggml-*.bin found",
        )

    # cmake & git (needed for building)
    for tool in ("cmake", "git"):
        tp = shutil.which(tool)
        results[tool] = (bool(tp), tp or "not in PATH")

    # Python packages
    for pkg, modname in [
        ("faster-whisper", "faster_whisper"),
        ("whisper", "whisper"),
        ("whisperx", "whisperx"),
    ]:
        spec = importlib.util.find_spec(modname)
        results[pkg] = (spec is not None, "installed" if spec else "not installed")

    return results


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _opts(values: Sequence[str], default: str) -> list[tuple[str, str]]:
    ordered = [default] + [v for v in values if v != default]
    return [(v, v) for v in ordered]


def _colorize(line: str) -> str:
    if "[ERROR]" in line:
        return f"[bold red]{line}[/]"
    if "[WARN]" in line:
        return f"[yellow]{line}[/]"
    if "\u2713" in line or "succeeded" in line.lower():
        return f"[green]{line}[/]"
    if line.startswith("\u25b6"):
        return f"[cyan]{line}[/]"
    if line.startswith("Found") or "Done." in line:
        return f"[bold]{line}[/]"
    return line


def _mask(cmd: list[str]) -> list[str]:
    SENSITIVE = {"--translation-api-key", "--postprocess-api-key", "--hf-token"}
    out, skip = [], False
    for tok in cmd:
        if skip:
            out.append("***")
            skip = False
        elif tok in SENSITIVE:
            out.append(tok)
            skip = True
        else:
            out.append(tok)
    return out


def filter_subtitle_paths(output_paths: list[str]) -> list[str]:
    subtitle_paths: list[str] = []
    for path_str in output_paths:
        path = Path(path_str)
        if path.name.lower().endswith(".stage1.json"):
            continue
        if path.suffix.lower() in SUBTITLE_OUTPUT_EXTENSIONS:
            subtitle_paths.append(path_str)
    return subtitle_paths


def resolve_copy_dest(source: Path, dest_dir: Path, existing: set[str]) -> Path:
    candidate_name = source.name
    if candidate_name not in existing:
        existing.add(candidate_name)
        return dest_dir / candidate_name

    index = 1
    while True:
        candidate_name = f"{source.stem} ({index}){source.suffix}"
        if candidate_name not in existing:
            existing.add(candidate_name)
            return dest_dir / candidate_name
        index += 1


def _candidate_model_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    configured = _db.get("tui.model_dir")
    raw_dirs = [configured] if configured else []
    raw_dirs.extend(MODEL_SEARCH_DIRS)
    for raw in raw_dirs:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = ROOT_DIR / path
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        dirs.append(path)
    return dirs


def discover_ggml_models(
    search_dirs: Sequence[str | Path] | None = None,
) -> dict[str, str]:
    """Return whisper.cpp model names mapped to absolute ggml file paths."""
    dirs = (
        [Path(p) for p in search_dirs]
        if search_dirs is not None
        else _candidate_model_dirs()
    )
    return shared_discover_ggml_models(dirs)


def summarize_ggml_models(models: dict[str, str], limit: int = 4) -> str:
    if not models:
        return "no ggml-*.bin found"
    names = sorted(models)
    preview = ", ".join(names[:limit])
    remainder = len(names) - limit
    suffix = f" (+{remainder} more)" if remainder > 0 else ""
    return f"{len(names)} detected: {preview}{suffix}"


def preferred_ggml_model_path(models: dict[str, str]) -> str:
    return shared_preferred_ggml_model_path(models, preferred_model=DEFAULT_MODEL)


def _search_terms(query: str) -> list[str]:
    return [term for term in re.split(r"\s+", query.strip().lower()) if term]


def _matches_search(path: Path, root: Path, terms: Sequence[str]) -> bool:
    try:
        rel = str(path.relative_to(root)).lower()
    except ValueError:
        rel = str(path).lower()
    name = path.name.lower()
    return all(term in name or term in rel for term in terms)


def discover_input_matches(
    root: Path, query: str, limit: int = SEARCH_RESULT_LIMIT
) -> list[str]:
    """Search *root* recursively for directories and video files matching *query*."""
    if limit < 1:
        return []

    root = root.expanduser()
    terms = _search_terms(query)
    if not terms or not root.is_dir():
        return []

    matches: list[str] = []
    seen: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        current_dir = Path(dirpath)

        if current_dir != root and _matches_search(current_dir, root, terms):
            resolved = str(current_dir.resolve())
            if resolved not in seen:
                matches.append(resolved)
                seen.add(resolved)
                if len(matches) >= limit:
                    return matches

        for filename in filenames:
            candidate = current_dir / filename
            if candidate.suffix.lower() not in VIDEO_INPUT_EXTENSIONS:
                continue
            if not _matches_search(candidate, root, terms):
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            matches.append(resolved)
            seen.add(resolved)
            if len(matches) >= limit:
                return matches

    return matches


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _sample_directory_videos(path: Path, limit: int = 5) -> list[str]:
    samples: list[str] = []
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames.sort()
        filenames.sort()
        current_dir = Path(dirpath)
        for filename in filenames:
            candidate = current_dir / filename
            if candidate.suffix.lower() not in VIDEO_INPUT_EXTENSIONS:
                continue
            try:
                label = str(candidate.relative_to(path))
            except ValueError:
                label = candidate.name
            samples.append(label)
            if len(samples) >= limit:
                return samples
    return samples


def build_search_preview(path: Path) -> str:
    if not path.exists():
        return f"Path no longer exists:\n{path}"

    try:
        stat = path.stat()
    except OSError as exc:
        return f"Could not read path metadata:\n{path}\n{exc}"
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))

    if path.is_file():
        sibling_outputs = sorted(
            candidate.name
            for candidate in path.parent.glob(f"{path.stem}.*")
            if candidate.is_file()
            and candidate != path
            and candidate.suffix.lower().lstrip(".") in FORMATS
        )
        lines = [
            "Video file preview",
            f"Name: {path.name}",
            f"Path: {path}",
            f"Size: {_format_bytes(stat.st_size)}",
            f"Modified: {mtime}",
            f"Parent: {path.parent}",
        ]
        if sibling_outputs:
            lines.append("Existing sidecar outputs:")
            lines.extend(f"  - {name}" for name in sibling_outputs[:6])
        else:
            lines.append("Existing sidecar outputs: none")
        return "\n".join(lines)

    try:
        entries = list(path.iterdir())
    except OSError as exc:
        return f"Could not inspect directory:\n{path}\n{exc}"
    dir_count = sum(1 for entry in entries if entry.is_dir())
    file_count = sum(1 for entry in entries if entry.is_file())
    video_samples = _sample_directory_videos(path, limit=5)
    lines = [
        "Directory preview",
        f"Name: {path.name or str(path)}",
        f"Path: {path}",
        f"Modified: {mtime}",
        f"Immediate entries: {dir_count} dir(s), {file_count} file(s)",
    ]
    if video_samples:
        lines.append("Sample videos:")
        lines.extend(f"  - {sample}" for sample in video_samples)
    else:
        lines.append("Sample videos: none found")
    return "\n".join(lines)


def detect_package_manager() -> str | None:
    for manager in SYSTEM_PACKAGE_MAP:
        if shutil.which(manager):
            return manager
    return None


def packages_for_manager(manager: str, tools: Sequence[str]) -> list[str]:
    mapping = SYSTEM_PACKAGE_MAP.get(manager, {})
    packages: list[str] = []
    seen: set[str] = set()
    for tool in tools:
        for package in mapping.get(tool, ()):
            if package not in seen:
                packages.append(package)
                seen.add(package)
    return packages


def build_system_install_commands(
    manager: str,
    tools: Sequence[str],
    *,
    use_sudo: bool,
) -> list[list[str]]:
    packages = packages_for_manager(manager, tools)
    if not packages:
        return []

    prefix = ["sudo", "-n"] if use_sudo and manager != "brew" else []
    if manager == "apt-get":
        return [
            prefix + ["apt-get", "update"],
            prefix + ["apt-get", "install", "-y", *packages],
        ]
    if manager in {"dnf", "yum"}:
        return [prefix + [manager, "install", "-y", *packages]]
    if manager == "pacman":
        return [prefix + ["pacman", "-Sy", "--noconfirm", *packages]]
    if manager == "brew":
        return [["brew", "install", *packages]]
    return []


def normalize_chat_endpoint(base_url: str) -> str:
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = endpoint + "/chat/completions"
    return endpoint


def extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Agent response must be a JSON object")
    return payload


def _fmt_elapsed(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _progress_ratio(completed: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, completed / total))


def _clamp_ratio(ratio: float | None) -> float:
    if ratio is None:
        return 0.0
    return max(0.0, min(1.0, ratio))


def _progress_bar_markup_ratio(ratio: float | None, width: int = 28) -> str:
    ratio = _clamp_ratio(ratio)
    filled = min(width, int(round(ratio * width)))
    empty = max(0, width - filled)
    bar = "[green]" + ("█" * filled) + "[/]" + "[dim]" + ("─" * empty) + "[/]"
    return f"{bar} [bold]{ratio * 100:5.1f}%[/]"


def _progress_bar_markup(completed: int, total: int, width: int = 28) -> str:
    return _progress_bar_markup_ratio(_progress_ratio(completed, total), width=width)


def _compact_progress_markup(ratio: float | None, width: int = 6) -> str:
    ratio = _clamp_ratio(ratio)
    filled = min(width, int(round(ratio * width)))
    empty = max(0, width - filled)
    bar = "█" * filled + "─" * empty
    return f"[yellow]{bar} {ratio * 100:4.1f}%[/]"


# RemoteResourceProfile, ExecutorPlan, RunJobState, RunConfig are defined in
# .models and re-exported here for backward compatibility.
# (old definitions removed — use models.py as the single source of truth)


def _coerce_positive_int(value: Any, default: int = 1) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def parse_remote_resources(raw: str) -> list[RemoteResourceProfile]:
    cleaned = raw.strip()
    if not cleaned:
        return []

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid remote resource JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise ValueError("Remote resources must be a JSON array")

    profiles: list[RemoteResourceProfile] = []
    for idx, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        if item.get("enabled", True) is False:
            continue

        ssh_target = str(item.get("ssh_target") or "").strip()
        remote_workdir = str(item.get("remote_workdir") or "").strip()
        if not ssh_target or not remote_workdir:
            continue

        name = str(item.get("name") or f"remote-{idx}").strip() or f"remote-{idx}"
        path_map: dict[str, str] = {}
        raw_path_map = item.get("path_map")
        if isinstance(raw_path_map, dict):
            for local_prefix, remote_prefix in raw_path_map.items():
                if not isinstance(local_prefix, str) or not isinstance(
                    remote_prefix, str
                ):
                    continue
                local_prefix = local_prefix.strip()
                remote_prefix = remote_prefix.strip()
                if local_prefix and remote_prefix:
                    path_map[local_prefix] = remote_prefix

        env: dict[str, str] = {}
        raw_env = item.get("env")
        if isinstance(raw_env, dict):
            for key, value in raw_env.items():
                if not isinstance(key, str):
                    continue
                key = key.strip()
                if key:
                    env[key] = str(value)

        profiles.append(
            RemoteResourceProfile(
                name=name,
                ssh_target=ssh_target,
                remote_workdir=remote_workdir,
                python_bin=str(item.get("python_bin") or "python3").strip()
                or "python3",
                script_path=str(item.get("script_path") or "").strip(),
                slots=_coerce_positive_int(item.get("slots"), default=1),
                path_map=path_map,
                env=env,
            )
        )

    return profiles


def build_remote_resource_labels(
    resources: Sequence[RemoteResourceProfile],
) -> dict[str, str]:
    name_counts = Counter(profile.name for profile in resources)
    return {
        profile.executor_key: profile.rendered_name(
            disambiguate=name_counts[profile.name] > 1
        )
        for profile in resources
    }


def merge_remote_resource_profiles(
    primary_profiles: Sequence[RemoteResourceProfile],
    legacy_profiles: Sequence[RemoteResourceProfile],
) -> tuple[list[RemoteResourceProfile], list[str]]:
    merged: list[RemoteResourceProfile] = []
    kept_by_key: dict[str, RemoteResourceProfile] = {}
    aliases_by_key: dict[str, list[tuple[str, RemoteResourceProfile]]] = {}

    for source_label, profiles in (
        ("SSH connections", primary_profiles),
        ("legacy remote JSON", legacy_profiles),
    ):
        for profile in profiles:
            key = profile.executor_key
            aliases_by_key.setdefault(key, []).append((source_label, profile))
            if key in kept_by_key:
                continue
            kept_by_key[key] = profile
            merged.append(profile)

    warnings: list[str] = []
    for key, entries in sorted(aliases_by_key.items()):
        if len(entries) < 2:
            continue
        kept_profile = kept_by_key[key]
        alias_names = ", ".join(
            sorted(
                {
                    profile.rendered_name()
                    for _, profile in entries
                    if profile.rendered_name()
                }
            )
        )
        warnings.append(
            "Remote executor aliases collapse onto the same ssh target/workdir: "
            f"{alias_names} -> {kept_profile.rendered_name(disambiguate=True)}."
        )

    name_groups: dict[str, list[RemoteResourceProfile]] = {}
    for profile in merged:
        name_groups.setdefault(profile.name, []).append(profile)
    colliding_names = {
        name: profiles for name, profiles in name_groups.items() if len(profiles) > 1
    }
    if colliding_names:
        rendered = "; ".join(
            f"{name}: "
            + ", ".join(sorted(profile.target_descriptor() for profile in profiles))
            for name, profiles in sorted(colliding_names.items())
        )
        warnings.append(
            "Remote profiles reuse the same display name across different executors: "
            f"{rendered}. Distributed scheduling now keys off executor identity, "
            "and progress labels include target info when needed."
        )

    return merged, warnings


def summarize_remote_resources(resources: Sequence[RemoteResourceProfile]) -> str:
    if not resources:
        return "Local execution only."
    total_slots = sum(profile.slots for profile in resources)
    labels = build_remote_resource_labels(resources)
    names = ", ".join(labels[profile.executor_key] for profile in resources[:3])
    extra = len(resources) - 3
    suffix = f" (+{extra} more)" if extra > 0 else ""
    return (
        f"{len(resources)} remote resource(s), {total_slots} slot(s): {names}{suffix}"
    )


def map_path_for_remote(path: str, path_map: dict[str, str]) -> str:
    if not path:
        return ""
    source = Path(path).expanduser()
    for local_prefix, remote_prefix in sorted(
        path_map.items(), key=lambda item: len(item[0]), reverse=True
    ):
        local_path = Path(local_prefix).expanduser()
        try:
            relative = source.relative_to(local_path)
        except ValueError:
            continue
        return str(Path(remote_prefix).expanduser() / relative)
    return str(source)


def partition_paths_by_capacity(
    paths: Sequence[str],
    capacities: Sequence[tuple[str, int]],
) -> dict[str, list[str]]:
    assignments = {name: [] for name, capacity in capacities if capacity > 0}
    schedule: list[str] = []
    for name, capacity in capacities:
        schedule.extend([name] * max(0, capacity))
    if not schedule:
        return assignments
    for idx, path in enumerate(paths):
        assignments[schedule[idx % len(schedule)]].append(path)
    return assignments


def group_paths_by_video_folder(paths: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for raw_path in sorted(paths):
        resolved = str(Path(raw_path).expanduser().resolve())
        folder_path = str(Path(resolved).parent)
        folder_hash = hash_video_folder(folder_path)
        group = groups.setdefault(
            folder_hash,
            {
                "folder_hash": folder_hash,
                "folder_path": folder_path,
                "videos": [],
            },
        )
        group["videos"].append(resolved)
    return sorted(
        groups.values(),
        key=lambda item: (str(item["folder_path"]), str(item["folder_hash"])),
    )


def partition_folder_groups_by_capacity(
    groups: Sequence[dict[str, Any]],
    capacities: Sequence[tuple[str, int]],
) -> dict[str, list[str]]:
    assignments = {name: [] for name, capacity in capacities if capacity > 0}
    capacity_map = {
        name: max(1, capacity) for name, capacity in capacities if capacity > 0
    }
    load_map = {name: 0 for name in assignments}
    if not assignments:
        return assignments

    ordered_groups = sorted(
        groups,
        key=lambda item: (
            -len(item.get("videos", [])),
            str(item.get("folder_path") or ""),
        ),
    )
    for group in ordered_groups:
        target = min(
            assignments,
            key=lambda name: (
                load_map[name] / capacity_map[name],
                load_map[name],
                name,
            ),
        )
        videos = [str(video) for video in group.get("videos", [])]
        assignments[target].extend(videos)
        load_map[target] += len(videos)
    return assignments


def parse_progress_event(line: str) -> dict[str, Any] | None:
    if not line.startswith(EVENT_PREFIX):
        return None
    payload = line.removeprefix(EVENT_PREFIX).strip()
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
