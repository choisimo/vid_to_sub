from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"

VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mkv",
        ".mov",
        ".avi",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".ts",
        ".mts",
        ".m2ts",
        ".mpeg",
        ".mpg",
        ".3gp",
        ".ogv",
        ".rmvb",
        ".vob",
        ".divx",
    }
)

FORMATS: list[str] = ["srt", "vtt", "txt", "tsv", "json"]
SUPPORTED_FORMATS: frozenset[str] = frozenset({*FORMATS, "all"})

KNOWN_MODELS: tuple[str, ...] = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
    "distil-small.en",
    "distil-medium.en",
    "distil-large-v2",
    "distil-large-v3",
)

DEFAULT_MODEL = "large-v3"
DEFAULT_BACKEND = "whisper-cpp"
DEFAULT_FORMAT = "srt"
DEFAULT_DEVICE = "cpu"

BACKENDS = ["whisper-cpp", "faster-whisper", "whisper", "whisperx"]
DEVICES = ["cpu", "auto", "cuda", "mps"]
EXECUTION_MODES = ["local", "distributed"]
SEARCH_RESULT_LIMIT = 60

ENV_TRANSLATION_BASE_URL = "VID_TO_SUB_TRANSLATION_BASE_URL"
ENV_TRANSLATION_API_KEY = "VID_TO_SUB_TRANSLATION_API_KEY"
ENV_TRANSLATION_MODEL = "VID_TO_SUB_TRANSLATION_MODEL"
ENV_WHISPER_CPP_BIN = "VID_TO_SUB_WHISPER_CPP_BIN"
ENV_WHISPER_CPP_MODEL = "VID_TO_SUB_WHISPER_CPP_MODEL"
ENV_AGENT_BASE_URL = "VID_TO_SUB_AGENT_BASE_URL"
ENV_AGENT_API_KEY = "VID_TO_SUB_AGENT_API_KEY"
ENV_AGENT_MODEL = "VID_TO_SUB_AGENT_MODEL"

EVENT_PREFIX = "@@VID_TO_SUB_EVENT@@"

WHISPER_CLI_CANDIDATES: tuple[str, ...] = (
    "/usr/local/bin/whisper-cli",
    "/usr/bin/whisper-cli",
    str(Path.home() / ".local/bin/whisper-cli"),
    str(Path.home() / "whisper.cpp/build/bin/whisper-cli"),
    str(Path.home() / "whisper.cpp/main"),
    str(ROOT_DIR / "whisper.cpp/build/bin/whisper-cli"),
)

MODEL_SEARCH_DIRS: tuple[str, ...] = (
    str(ROOT_DIR / "models"),
    str(Path.home() / ".cache" / "whisper"),
    str(Path.home() / "models"),
    "/models",
    "/opt/models",
)

HF_MODEL_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

PIP_REQUIREMENT_FILES: dict[str, tuple[str, str]] = {
    "base": ("requirements.txt", "base TUI/runtime requirements"),
    "faster-whisper": (
        "requirements-faster-whisper.txt",
        "faster-whisper backend requirements",
    ),
    "whisper": ("requirements-whisper.txt", "openai-whisper backend requirements"),
    "whisperx": ("requirements-whisperx.txt", "whisperX backend requirements"),
}

SYSTEM_PACKAGE_MAP: dict[str, dict[str, tuple[str, ...]]] = {
    "apt-get": {
        "ffmpeg": ("ffmpeg",),
        "git": ("git",),
        "cmake": ("cmake",),
        "whisper-build": ("build-essential", "pkg-config"),
    },
    "dnf": {
        "ffmpeg": ("ffmpeg",),
        "git": ("git",),
        "cmake": ("cmake",),
        "whisper-build": ("gcc-c++", "make", "pkgconf-pkg-config"),
    },
    "yum": {
        "ffmpeg": ("ffmpeg",),
        "git": ("git",),
        "cmake": ("cmake",),
        "whisper-build": ("gcc-c++", "make", "pkgconfig"),
    },
    "pacman": {
        "ffmpeg": ("ffmpeg",),
        "git": ("git",),
        "cmake": ("cmake",),
        "whisper-build": ("base-devel", "pkgconf"),
    },
    "brew": {
        "ffmpeg": ("ffmpeg",),
        "git": ("git",),
        "cmake": ("cmake",),
        "whisper-build": ("pkg-config",),
    },
}
