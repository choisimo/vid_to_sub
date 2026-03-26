from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional, Sequence

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .constants import (
    DEFAULT_BACKEND,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    ENV_FILE,
    ENV_WHISPER_CPP_BIN,
    ENV_WHISPER_CPP_MODEL,
    FASTER_WHISPER_MODEL_FALLBACKS,
    FASTER_WHISPER_MODEL_MIN_VRAM_GB,
    MODEL_SEARCH_DIRS,
    ROOT_DIR,
    SECRET_ENV_KEYS,
    WHISPER_CLI_CANDIDATES,
)

# ---------------------------------------------------------------------------
# SQLite-first env loading
# ---------------------------------------------------------------------------
#
# load_env_from_sqlite() / import_env_file_to_sqlite() are the canonical
# entry points for env bootstrapping.  .env is a *backup* only — it is never
# loaded unless explicitly imported via import_env_file_to_sqlite().
#
# Callers that need a live DB handle pass it as a protocol-compatible getter.
# This avoids circular imports (env.py must not import from db.py).
#
# Protocol: SettingsProvider = Callable[[], dict[str, str]]


def load_env_from_sqlite(
    get_all: Callable[[], dict[str, str]],
    *,
    override: bool = True,
) -> int:
    """Push VID_TO_SUB_* settings from SQLite into os.environ.

    This is the **primary** env bootstrap path.  Call early in startup before
    any subprocess or os.getenv access.

    Args:
        get_all: callable that returns ``{key: value}`` from the settings
                 table (i.e. ``db.get_all``).
        override: when True (default) SQLite values always win over any value
                  that may already be in os.environ (e.g. leftover from a
                  previous .env load).  Pass False only when you want to
                  preserve values set by the shell before startup.

    Returns:
        Number of keys injected into os.environ.
    """
    settings = get_all()
    count = 0
    for key, value in settings.items():
        if not key.startswith("VID_TO_SUB_"):
            continue
        if key in SECRET_ENV_KEYS:
            continue
        if value:
            if override or key not in os.environ:
                os.environ[key] = value
                count += 1
        else:
            # Explicit blank in DB means "clear" — remove whatever .env put there.
            if override:
                os.environ.pop(key, None)
    return count


def import_env_file_to_sqlite(
    env_path: Path,
    set_many: Callable[[dict[str, str]], None],
    get_all: Callable[[], dict[str, str]],
    *,
    overwrite: bool = False,
) -> dict[str, str]:
    """Parse *env_path* and upsert matching keys into SQLite.

    This is how .env values enter the system.  After import the .env file
    itself is no longer consulted — SQLite becomes the source of truth.

    Args:
        env_path: path to the .env file to read.
        set_many: callable that writes ``{key: value}`` to the settings table
                  (i.e. ``db.set_many``).
        get_all:  callable that reads all settings (i.e. ``db.get_all``).
        overwrite: when True, import even for keys that already have a
                   non-empty value in SQLite.  When False (default) only
                   keys that are blank or missing in SQLite are updated.

    Returns:
        Dict of keys that were actually written to SQLite.
    """
    if not env_path.is_file():
        return {}

    parsed: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        assignment = parse_env_assignment(raw_line)
        if assignment is None:
            continue
        key, value = assignment
        parsed[key] = value

    if not parsed:
        return {}

    existing = get_all() if not overwrite else {}
    to_write: dict[str, str] = {}
    for key, value in parsed.items():
        if key in SECRET_ENV_KEYS:
            continue
        if overwrite or not (existing.get(key) or "").strip():
            to_write[key] = value

    if to_write:
        set_many(to_write)
    return to_write


def load_project_env_fallback(*, override: bool = False) -> bool:
    """Load .env into os.environ **only as a last-resort fallback**.

    Prefer :func:`load_env_from_sqlite` for all production code paths.
    This function exists solely for the CLI bootstrap before the DB is
    available (e.g. ``vid_to_sub.py --help`` without TUI).

    SQLite always wins: if a key is already in os.environ (set by
    load_env_from_sqlite) it will *not* be overwritten here (override=False).
    """
    return load_project_env(override=False)


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
    """Low-level: load .env into os.environ.

    .. warning::
        Direct callers should prefer :func:`load_env_from_sqlite` for the
        primary bootstrap and :func:`load_project_env_fallback` everywhere
        else.  This function does *not* consult SQLite.
    """
    if not ENV_FILE.is_file():
        return False
    if load_dotenv is not None:
        load_dotenv(ENV_FILE, override=override)
        return True
    return load_env_file(ENV_FILE, override=override)


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def available_cpu_threads() -> int:
    process_cpu_count = getattr(os, "process_cpu_count", None)
    detected = process_cpu_count() if callable(process_cpu_count) else os.cpu_count()
    return max(1, int(detected or 1))


def detect_torch_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    try:
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return "cpu"


def nvidia_gpu_available() -> bool:
    for device_path in (Path("/dev/nvidiactl"), Path("/dev/nvidia0")):
        if device_path.exists():
            return True

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False

    return result.returncode == 0 and bool(result.stdout.strip())


def detect_best_device() -> str:
    torch_device = detect_torch_device()
    if torch_device != "cpu":
        return torch_device
    if nvidia_gpu_available():
        return "cuda"
    return "cpu"


def detect_cuda_total_memory_gb() -> float | None:
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        try:
            if torch.cuda.is_available():
                device_count = max(1, int(torch.cuda.device_count()))
                totals = [
                    float(torch.cuda.get_device_properties(index).total_memory)
                    / (1024**3)
                    for index in range(device_count)
                ]
                if totals:
                    return max(totals)
        except Exception:
            pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    values: list[float] = []
    for raw_line in result.stdout.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            mib = float(stripped)
        except ValueError:
            continue
        values.append(mib / 1024.0)
    return max(values) if values else None


def detect_cuda_free_memory_gb() -> float | None:
    """Return the free (available) VRAM in GiB for the best GPU.

    Prefers torch.cuda.mem_get_info() which returns (free, total) for the
    currently selected device.  Falls back to nvidia-smi --query-gpu=memory.free
    for environments without torch.  Returns None when no CUDA device is found.

    This is the preferred signal for model selection because it reflects
    current runtime pressure — unlike total VRAM it accounts for other
    processes already occupying GPU memory.
    """
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]

    if torch is not None:
        try:
            if torch.cuda.is_available():
                device_count = max(1, int(torch.cuda.device_count()))
                free_values: list[float] = []
                for index in range(device_count):
                    try:
                        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
                        free_values.append(float(free_bytes) / (1024**3))
                    except Exception:
                        pass
                if free_values:
                    return max(free_values)
        except Exception:
            pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None

    if result.returncode != 0:
        return None

    free_vals: list[float] = []
    for raw_line in result.stdout.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            mib = float(stripped)
        except ValueError:
            continue
        free_vals.append(mib / 1024.0)
    return max(free_vals) if free_vals else None


def faster_whisper_model_candidates(
    model_name: str,
    *,
    available_vram_gb: float | None = None,
) -> list[str]:
    raw_candidates = FASTER_WHISPER_MODEL_FALLBACKS.get(model_name, (model_name,))
    deduped_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in raw_candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped_candidates.append(candidate)

    if available_vram_gb is None:
        return deduped_candidates

    fitting_candidates = [
        candidate
        for candidate in deduped_candidates
        if FASTER_WHISPER_MODEL_MIN_VRAM_GB.get(candidate, 0.0) <= available_vram_gb
    ]
    return fitting_candidates


def resolve_runtime_model(
    backend: str,
    requested_device: str,
    preferred_model: str = DEFAULT_MODEL,
) -> str:
    resolved_device = resolve_runtime_backend_device(backend, requested_device)
    if backend != "faster-whisper" or resolved_device != "cuda":
        return preferred_model

    # Prefer free VRAM (reflects current GPU load) over total VRAM.
    # Fall back to total when free is unavailable (e.g. no torch, nvidia-smi absent).
    available_vram_gb = detect_cuda_free_memory_gb()
    if available_vram_gb is None:
        available_vram_gb = detect_cuda_total_memory_gb()
    candidates = faster_whisper_model_candidates(
        preferred_model,
        available_vram_gb=available_vram_gb,
    )
    return candidates[0] if candidates else preferred_model


def resolve_runtime_backend_and_device(
    default_backend: str = DEFAULT_BACKEND,
    default_device: str = DEFAULT_DEVICE,
) -> tuple[str, str]:
    best_device = detect_best_device()
    torch_device = detect_torch_device()

    if best_device == "cuda":
        if module_available("faster_whisper"):
            return "faster-whisper", "auto"
        if torch_device == "cuda" and module_available("whisperx"):
            return "whisperx", "auto"
        if torch_device == "cuda" and module_available("whisper"):
            return "whisper", "auto"
        return default_backend, default_device

    if torch_device == "mps":
        if module_available("whisperx"):
            return "whisperx", "auto"
        if module_available("whisper"):
            return "whisper", "auto"

    return default_backend, default_device


def resolve_runtime_backend_device(backend: str, requested_device: str) -> str:
    if backend == "whisper-cpp":
        return "cpu"

    if requested_device == "auto":
        if backend == "faster-whisper":
            detected_device = detect_best_device()
            return "cuda" if detected_device == "cuda" else "cpu"
        return detect_torch_device()

    if backend == "faster-whisper" and requested_device == "mps":
        return "cpu"

    return requested_device


def resolve_runtime_backend_threads(
    backend: str,
    requested_device: str,
    worker_count: int = 1,
    *,
    default_gpu_threads: int = 2,
) -> int:
    resolved_device = resolve_runtime_backend_device(backend, requested_device)
    if resolved_device != "cpu":
        return max(1, int(default_gpu_threads))

    workers = max(1, int(worker_count))
    return max(1, available_cpu_threads() // workers)


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
