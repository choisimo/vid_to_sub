from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .constants import (
    DEFAULT_MODEL,
    ENV_FILE,
    ENV_WHISPER_CPP_BIN,
    ENV_WHISPER_CPP_MODEL,
    MODEL_SEARCH_DIRS,
    ROOT_DIR,
    WHISPER_CLI_CANDIDATES,
)


def parse_env_assignment(line: str) -> Optional[tuple[str, str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[7:].lstrip()
    if "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    elif " #" in value:
        value = value.split(" #", 1)[0].rstrip()

    return key, value


def load_env_file(env_path: Path, *, override: bool = False) -> bool:
    if not env_path.is_file():
        return False

    loaded = False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_assignment(raw_line)
        if parsed is None:
            continue
        key, value = parsed
        if override or key not in os.environ:
            os.environ[key] = value
        loaded = True
    return loaded


def load_project_env(*, override: bool = False) -> bool:
    if not ENV_FILE.is_file():
        return False
    if load_dotenv is not None:
        load_dotenv(ENV_FILE, override=override)
        return True
    return load_env_file(ENV_FILE, override=override)


def resolve_executable(candidate: str) -> Optional[str]:
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    candidate_path = Path(candidate).expanduser()
    if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
        return str(candidate_path.resolve())
    return None


def find_whisper_cpp_bin(configured_bin: Optional[str] = None) -> Optional[str]:
    raw_candidate = configured_bin
    if raw_candidate is None:
        raw_candidate = os.getenv(ENV_WHISPER_CPP_BIN, "")
    candidate = raw_candidate.strip()
    if candidate:
        resolved = resolve_executable(candidate)
        if resolved:
            return resolved

    resolved = resolve_executable("whisper-cli")
    if resolved:
        return resolved

    for fallback in WHISPER_CLI_CANDIDATES:
        resolved = resolve_executable(fallback)
        if resolved:
            return resolved
    return None


def candidate_model_dirs(
    search_dirs: Sequence[str | Path] | None = None,
) -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    raw_dirs = list(search_dirs) if search_dirs is not None else list(MODEL_SEARCH_DIRS)
    for raw_dir in raw_dirs:
        path = Path(raw_dir).expanduser()
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
    found: dict[str, str] = {}
    for raw_dir in candidate_model_dirs(search_dirs):
        if not raw_dir.is_dir():
            continue
        for candidate in sorted(raw_dir.glob("ggml-*.bin")):
            model_name = candidate.name.removeprefix("ggml-").removesuffix(".bin")
            if model_name and model_name not in found:
                found[model_name] = str(candidate.resolve())
    return found


def preferred_ggml_model_path(
    models: dict[str, str],
    preferred_model: str = DEFAULT_MODEL,
) -> str:
    if preferred_model in models:
        return models[preferred_model]
    if models:
        return models[sorted(models)[0]]
    return ""


def find_whisper_cpp_model_path(
    model_name: str,
    configured_model_path: Optional[str] = None,
    *,
    search_dirs: Sequence[str | Path] | None = None,
    strict_configured: bool = False,
) -> Optional[str]:
    raw_candidate = configured_model_path
    if raw_candidate is None:
        raw_candidate = os.getenv(ENV_WHISPER_CPP_MODEL, "")
    candidate = raw_candidate.strip()
    if candidate:
        candidate_path = Path(candidate).expanduser()
        if candidate_path.is_file():
            return str(candidate_path.resolve())
        if strict_configured:
            return None

    models = discover_ggml_models(search_dirs)
    return models.get(model_name)
