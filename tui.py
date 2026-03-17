#!/home/nodove/workspace/vid_to_sub/.venv/bin/python3
"""
tui.py — vid_to_sub TUI v2
===========================
5-tab terminal UI backed by SQLite.

Tabs
----
  1  Browse     — DirectoryTree browser + selected input paths
  2  Setup      — Auto-detect dependencies + auto-install
  3  Transcribe — All job settings (backend / model / formats / translation)
  4  History    — SQLite-backed job history
  5  Settings   — Persistent configuration (SQLite)

Keyboard shortcuts
------------------
  Ctrl+R   Run
  Ctrl+D   Dry Run
  Ctrl+K   Kill
  Ctrl+S   Save settings
  Ctrl+Q   Quit
  1–5      Switch tabs
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from dotenv import load_dotenv
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.worker import Worker, WorkerState

from db import Database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
SCRIPT_PATH = SCRIPT_DIR / "vid_to_sub.py"
ENV_FILE = SCRIPT_DIR / ".env"

DEFAULT_BACKEND = "whisper-cpp"
DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cpu"

KNOWN_MODELS: list[str] = [
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
]
BACKENDS = ["whisper-cpp", "faster-whisper", "whisper", "whisperx"]
DEVICES = ["cpu", "auto", "cuda", "mps"]
FORMATS = ["srt", "vtt", "txt", "tsv", "json"]
SEARCH_RESULT_LIMIT = 60

VIDEO_INPUT_EXTENSIONS: frozenset[str] = frozenset(
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

ENV_WCPP_BIN = "VID_TO_SUB_WHISPER_CPP_BIN"
ENV_WCPP_MODEL = "VID_TO_SUB_WHISPER_CPP_MODEL"
ENV_TRANS_URL = "VID_TO_SUB_TRANSLATION_BASE_URL"
ENV_TRANS_KEY = "VID_TO_SUB_TRANSLATION_API_KEY"
ENV_TRANS_MOD = "VID_TO_SUB_TRANSLATION_MODEL"

# Common locations to search for whisper-cli
WHISPER_CLI_CANDIDATES: list[str] = [
    "/usr/local/bin/whisper-cli",
    "/usr/bin/whisper-cli",
    str(Path.home() / ".local/bin/whisper-cli"),
    str(Path.home() / "whisper.cpp/build/bin/whisper-cli"),
    str(Path.home() / "whisper.cpp/main"),
    str(SCRIPT_DIR / "whisper.cpp/build/bin/whisper-cli"),
]

# Common dirs that may contain ggml model files
MODEL_SEARCH_DIRS: list[str] = [
    str(SCRIPT_DIR / "models"),
    str(Path.home() / ".cache/whisper"),
    str(Path.home() / "models"),
    "/models",
    "/opt/models",
]

HF_MODEL_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

# Module-level database singleton
_db = Database()

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

DetectResult = dict[str, tuple[bool, str]]  # name -> (found, detail)


def detect_all() -> DetectResult:
    """Scan system for required / optional dependencies."""
    results: DetectResult = {}
    ggml_models = discover_ggml_models()

    # ffmpeg
    p = shutil.which("ffmpeg")
    results["ffmpeg"] = (bool(p), p or "not in PATH")

    # whisper-cli
    wbin = _db.get(ENV_WCPP_BIN)
    if wbin and Path(wbin).exists():
        results["whisper-cli"] = (True, wbin)
    else:
        found: str | None = shutil.which("whisper-cli")
        if not found:
            found = next((c for c in WHISPER_CLI_CANDIDATES if Path(c).exists()), None)
        results["whisper-cli"] = (bool(found), found or "not found")

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
        ("whisperx", "whisperx"),
    ]:
        spec = importlib.util.find_spec(modname)
        results[pkg] = (spec is not None, "installed" if spec else "not installed")

    return results


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _opts(values: list[str], default: str) -> list[tuple[str, str]]:
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
    SENSITIVE = {"--translation-api-key", "--hf-token"}
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


def _candidate_model_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()
    configured = _db.get("tui.model_dir")
    raw_dirs = [configured] if configured else []
    raw_dirs.extend(MODEL_SEARCH_DIRS)
    for raw in raw_dirs:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = SCRIPT_DIR / path
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
    found: dict[str, str] = {}
    dirs = [Path(p) for p in search_dirs] if search_dirs is not None else _candidate_model_dirs()
    for raw_dir in dirs:
        path = raw_dir.expanduser()
        if not path.is_absolute():
            path = SCRIPT_DIR / path
        if not path.is_dir():
            continue
        for candidate in sorted(path.glob("ggml-*.bin")):
            model_name = candidate.name.removeprefix("ggml-").removesuffix(".bin")
            if model_name and model_name not in found:
                found[model_name] = str(candidate.resolve())
    return found


def summarize_ggml_models(models: dict[str, str], limit: int = 4) -> str:
    if not models:
        return "no ggml-*.bin found"
    names = sorted(models)
    preview = ", ".join(names[:limit])
    remainder = len(names) - limit
    suffix = f" (+{remainder} more)" if remainder > 0 else ""
    return f"{len(names)} detected: {preview}{suffix}"


def preferred_ggml_model_path(models: dict[str, str]) -> str:
    if DEFAULT_MODEL in models:
        return models[DEFAULT_MODEL]
    if models:
        first_name = sorted(models)[0]
        return models[first_name]
    return ""


def _search_terms(query: str) -> list[str]:
    return [term for term in re.split(r"\s+", query.strip().lower()) if term]


def _matches_search(path: Path, root: Path, terms: Sequence[str]) -> bool:
    try:
        rel = str(path.relative_to(root)).lower()
    except ValueError:
        rel = str(path).lower()
    name = path.name.lower()
    return all(term in name or term in rel for term in terms)


def discover_input_matches(root: Path, query: str, limit: int = SEARCH_RESULT_LIMIT) -> list[str]:
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


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
/* ── Global ─────────────────────────────────────── */
Screen { layout: vertical; background: $background; }
#tab-area { height: 1fr; min-height: 20; }

/* ── Scrollable tab body ─────────────────────────── */
.tab-body { padding: 0 2 1 2; }

/* ── Section titles ──────────────────────────────── */
.stitle {
    text-style: bold;
    color: $accent;
    margin-top: 1;
    padding: 0 0 0 1;
    border-bottom: solid $accent;
    height: 2;
}

/* ── Form rows ───────────────────────────────────── */
.frow {
    height: auto;
    margin-bottom: 1;
    align: left middle;
}
.flabel {
    width: 32;
    padding-right: 1;
    color: $text-muted;
    text-align: right;
    overflow: hidden;
}
.fwidget { width: 1fr; }

/* ── Check / switch rows ─────────────────────────── */
.crow {
    height: auto;
    padding: 0 1;
    margin-bottom: 1;
    align: left middle;
}
.crow Checkbox { margin-right: 3; }
.crow Switch   { margin-right: 2; }
.crow Label    { margin-right: 1; color: $text-muted; }

/* ── Format checkboxes ───────────────────────────── */
#fmt-row { height: auto; padding: 0 1; margin-bottom: 1; }
#fmt-row Checkbox { margin-right: 2; }

/* ── Browse tab ──────────────────────────────────── */
#browse-split { layout: horizontal; height: 1fr; }
#tree-pane {
    width: 43%;
    height: 1fr;
    border-right: solid $primary;
}
#tree-nav {
    height: 3;
    layout: horizontal;
    padding: 0 1;
}
#tree-nav Input { width: 1fr; }
#tree-nav Button { width: 7; margin-left: 1; }
DirectoryTree { height: 1fr; }
#paths-pane { width: 57%; height: 1fr; padding: 1 2; layout: vertical; }
#sel-paths-box {
    height: 1fr;
    overflow-y: auto;
    border: solid $panel;
    padding: 0 1;
    min-height: 4;
}
.sel-row { height: 3; align: left middle; }
.sel-label { width: 1fr; overflow: hidden; }
.sel-btn { width: 5; min-width: 5; }
#browse-actions { height: 3; margin-top: 1; align: left middle; }
#browse-actions Button { margin-right: 1; }
#manual-add-row { height: 3; align: left middle; margin-top: 0; }
#manual-add-row Input { width: 1fr; }
#manual-add-row Button { width: 5; margin-left: 1; }
#search-row { height: 3; align: left middle; margin-top: 1; }
#search-row Input { width: 1fr; }
#search-row Button { width: 8; margin-left: 1; }
#search-status { height: auto; padding: 0 1; margin-bottom: 1; color: $text-muted; }
#search-results-box {
    height: auto;
    max-height: 12;
    overflow-y: auto;
    border: solid $panel;
    padding: 0 1;
    min-height: 4;
}
.search-row { height: 2; align: left middle; }
.search-label { width: 1fr; overflow: hidden; }
.search-go { width: 5; margin-right: 1; }
.search-add { width: 5; }
#outdir-row { height: auto; margin-top: 1; layout: horizontal; align: left middle; }
#outdir-label { width: 14; color: $text-muted; }
#inp-output-dir { width: 1fr; }
#behavior-row { height: auto; padding: 0 1; margin-top: 0; margin-bottom: 0; }
#behavior-row Checkbox { margin-right: 3; }
#recent-box {
    height: auto;
    max-height: 9;
    overflow-y: auto;
    border: solid $panel;
    padding: 0 1;
}
.recent-row { height: 2; align: left middle; }
.recent-label { width: 1fr; overflow: hidden; color: $text-muted; }
.recent-add { width: 5; }

/* ── Setup tab ───────────────────────────────────── */
.det-row {
    height: 3;
    align: left middle;
    padding: 0 1;
    border-bottom: solid $panel;
}
.det-name { width: 20; }
.det-icon { width: 4; }
.det-info { width: 1fr; overflow: hidden; color: $text-muted; }
.det-btn  { width: 22; }
.inst-row { height: 3; align: left middle; layout: horizontal; padding: 0 1; }
.inst-label { width: 18; color: $text-muted; }
.inst-field { width: 1fr; }
#setup-log {
    height: 12;
    border: solid $panel;
    background: $surface-darken-1;
    margin: 0 1;
}

/* ── Translation fields ──────────────────────────── */
#trans-fields.hidden { display: none; }

/* ── History tab ─────────────────────────────────── */
#hist-pane { layout: vertical; height: 1fr; }
#hist-actions { height: 3; align: left middle; padding: 0 1; }
#hist-actions Button { margin-right: 1; }
#hist-table { height: 1fr; }
#hist-detail {
    height: 5;
    border-top: solid $primary;
    padding: 0 1;
    color: $text-muted;
    overflow-y: auto;
}

/* ── Settings tab ────────────────────────────────── */
#stg-actions { height: 3; align: left middle; margin-top: 1; }
#stg-actions Button { margin-right: 1; }
#stg-status { height: 1; color: $success; padding: 0 1; }

/* ── Bottom panel ────────────────────────────────── */
#bottom { height: auto; border-top: solid $primary; padding: 0 1; }
#cmd-preview {
    height: 2;
    background: $surface;
    border: solid $panel;
    padding: 0 1;
    margin-top: 1;
    margin-bottom: 1;
    overflow-x: auto;
    color: $text-muted;
}
#run-btns { height: 3; margin-bottom: 1; align: left middle; }
#run-btns Button { margin-right: 1; }
#log { height: 14; border: solid $panel; background: $surface-darken-1; }
"""


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class VidToSubApp(App):
    """vid_to_sub TUI v2 — DirectoryTree + SQLite + auto-install."""

    TITLE = "vid_to_sub"
    SUB_TITLE = "Video → Subtitle  v2"
    CSS = _CSS
    BINDINGS = [
        Binding("ctrl+r", "run", "▶ Run", priority=True),
        Binding("ctrl+d", "dry_run", "⚡ Dry", priority=True),
        Binding("ctrl+k", "kill", "✕ Kill", priority=True, show=False),
        Binding("ctrl+s", "save_settings", "💾 Save", priority=True, show=False),
        Binding("ctrl+q", "quit_app", "Quit"),
        Binding("1", "tab('tab-browse')", show=False),
        Binding("2", "tab('tab-setup')", show=False),
        Binding("3", "tab('tab-transcribe')", show=False),
        Binding("4", "tab('tab-history')", show=False),
        Binding("5", "tab('tab-settings')", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        _db.seed_defaults()
        self._selected_paths: list[str] = []
        self._search_results: list[str] = []
        self._tree_selection: Path | None = None
        self._detect_results: DetectResult = {}
        self._detected_ggml_models: dict[str, str] = {}
        self._active_worker: Worker | None = None
        self._proc: subprocess.Popen | None = None
        self._hist_key: str | None = None
        self._active_jobs: dict[str, int] = {}  # video_path -> db job id
        self._job_t0: float = 0.0

    # ── compose ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()

        with Vertical(id="tab-area"):
            with TabbedContent(initial="tab-browse"):
                # ── Tab 1: Browse ─────────────────────────────────────
                with TabPane("1 Browse", id="tab-browse"):
                    with Horizontal(id="browse-split"):
                        # Left: directory tree
                        with Vertical(id="tree-pane"):
                            with Horizontal(id="tree-nav"):
                                yield Input(
                                    value=str(Path.home()),
                                    id="tree-root",
                                    placeholder="Root path…",
                                )
                                yield Button("Go", id="btn-tree-go")
                            yield DirectoryTree(str(Path.home()), id="dir-tree")

                        # Right: paths management
                        with Vertical(id="paths-pane"):
                            yield Static("Selected Input Paths", classes="stitle")
                            with Vertical(id="sel-paths-box"):
                                yield Static(
                                    "[dim]No paths selected — browse left[/]",
                                    id="sel-empty",
                                )

                            with Horizontal(id="browse-actions"):
                                yield Button(
                                    "＋ Add Selected",
                                    id="btn-add-sel",
                                    variant="success",
                                )
                                yield Button(
                                    "✕ Clear All",
                                    id="btn-clear-paths",
                                    variant="error",
                                )

                            with Horizontal(id="manual-add-row"):
                                yield Input(
                                    placeholder="Or type/paste path here…",
                                    id="inp-manual-path",
                                )
                                yield Button(
                                    "＋", id="btn-manual-add", variant="default"
                                )

                            yield Static("Quick Search", classes="stitle")
                            with Horizontal(id="search-row"):
                                yield Input(
                                    placeholder="Find directories or video files under root…",
                                    id="inp-path-search",
                                )
                                yield Button("Find", id="btn-search-paths")
                                yield Button("Clear", id="btn-clear-search")
                            yield Static(
                                "[dim]Type at least 2 characters. Results stay inside the current root path.[/]",
                                id="search-status",
                                markup=True,
                            )
                            with Vertical(id="search-results-box"):
                                yield Static(
                                    "[dim]Type a keyword to search the current root.[/]",
                                    id="search-empty",
                                    markup=True,
                                )

                            with Horizontal(id="outdir-row"):
                                yield Label("Output dir:", id="outdir-label")
                                yield Input(
                                    placeholder="(next to each video)",
                                    id="inp-output-dir",
                                )

                            with Horizontal(id="behavior-row", classes="crow"):
                                yield Checkbox(
                                    "No recurse", id="chk-no-recurse", value=False
                                )
                                yield Checkbox(
                                    "Skip existing", id="chk-skip-existing", value=False
                                )

                            yield Static("Recent Paths", classes="stitle")
                            with Vertical(id="recent-box"):
                                yield Static("[dim]None yet[/]", id="recent-empty")

                # ── Tab 2: Setup ──────────────────────────────────────
                with TabPane("2 Setup", id="tab-setup"):
                    with ScrollableContainer(classes="tab-body"):
                        yield Static("System Detection", classes="stitle")
                        # Each row is a Static updated by _update_detect_ui
                        for comp in (
                            "ffmpeg",
                            "whisper-cli",
                            "ggml-model",
                            "cmake",
                            "git",
                            "faster-whisper",
                            "whisperx",
                        ):
                            yield Static(
                                f"[dim]{comp}  scanning…[/]",
                                id=f"det-{comp}",
                                classes="det-row",
                            )

                        with Horizontal(classes="crow"):
                            yield Button(
                                "🔍 Re-detect", id="btn-redetect", variant="default"
                            )

                        yield Static("Build whisper.cpp", classes="stitle")
                        with Horizontal(classes="inst-row"):
                            yield Label("Build dir:", classes="inst-label")
                            yield Input(id="inp-build-dir", classes="inst-field")
                        with Horizontal(classes="inst-row"):
                            yield Label("Install to:", classes="inst-label")
                            yield Input(id="inp-install-dir", classes="inst-field")
                        with Horizontal(classes="inst-row"):
                            yield Button(
                                "🔨 Build & Install whisper-cli",
                                id="btn-build-whisper",
                                variant="warning",
                            )

                        yield Static("Download Model", classes="stitle")
                        with Horizontal(classes="inst-row"):
                            yield Label("Model:", classes="inst-label")
                            yield Select(
                                _opts(KNOWN_MODELS, "large-v3"),
                                id="sel-dl-model",
                                classes="inst-field",
                                allow_blank=False,
                            )
                        with Horizontal(classes="inst-row"):
                            yield Label("Save to:", classes="inst-label")
                            yield Input(id="inp-model-dir", classes="inst-field")
                        with Horizontal(classes="inst-row"):
                            yield Button(
                                "⬇  Download Model",
                                id="btn-download-model",
                                variant="warning",
                            )

                        yield Static("Python Packages", classes="stitle")
                        with Horizontal(classes="crow"):
                            yield Button(
                                "pip install faster-whisper",
                                id="btn-pip-fw",
                                variant="default",
                            )
                            yield Button(
                                "pip install whisperx",
                                id="btn-pip-wx",
                                variant="default",
                            )

                        yield Static("Install / Build Log", classes="stitle")
                        yield RichLog(
                            id="setup-log",
                            highlight=True,
                            markup=True,
                            auto_scroll=True,
                            wrap=True,
                            max_lines=3000,
                        )

                # ── Tab 3: Transcribe (consolidated) ─────────────────
                with TabPane("3 Transcribe", id="tab-transcribe"):
                    with ScrollableContainer(classes="tab-body"):
                        yield Static("Backend & Model", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label("Backend", classes="flabel")
                            yield Select(
                                _opts(BACKENDS, DEFAULT_BACKEND),
                                id="sel-backend",
                                classes="fwidget",
                                allow_blank=False,
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Model", classes="flabel")
                            yield Select(
                                _opts(KNOWN_MODELS, DEFAULT_MODEL),
                                id="sel-model",
                                classes="fwidget",
                                allow_blank=False,
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Language", classes="flabel")
                            yield Input(
                                placeholder="auto-detect  (e.g. en ja ko zh)",
                                id="inp-language",
                                classes="fwidget",
                            )

                        yield Static("Hardware", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label("Device", classes="flabel")
                            yield Select(
                                _opts(DEVICES, DEFAULT_DEVICE),
                                id="sel-device",
                                classes="fwidget",
                                allow_blank=False,
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Compute type", classes="flabel")
                            yield Input(
                                placeholder="auto  (int8 | float16 | int8_float16)",
                                id="inp-compute-type",
                                classes="fwidget",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Beam size", classes="flabel")
                            yield Input(
                                value="5",
                                id="inp-beam-size",
                                type="integer",
                                classes="fwidget",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Workers", classes="flabel")
                            yield Input(
                                value="1",
                                id="inp-workers",
                                type="integer",
                                classes="fwidget",
                            )

                        yield Static("Output Formats", classes="stitle")
                        with Horizontal(id="fmt-row"):
                            for fmt in FORMATS:
                                yield Checkbox(
                                    fmt, id=f"fmt-{fmt}", value=(fmt == "srt")
                                )
                            yield Checkbox("all", id="fmt-all", value=False)
                        with Horizontal(classes="crow"):
                            yield Checkbox("Verbose", id="chk-verbose", value=False)

                        yield Static(
                            "Translation  (OpenAI-compatible)", classes="stitle"
                        )
                        with Horizontal(classes="crow"):
                            yield Label("Enable translation")
                            yield Switch(id="sw-translate", value=False)
                        with Vertical(id="trans-fields", classes="hidden"):
                            with Horizontal(classes="frow"):
                                yield Label("Translate to", classes="flabel")
                                yield Input(
                                    placeholder="ko  ja  fr  zh…",
                                    id="inp-translate-to",
                                    classes="fwidget",
                                )
                            with Horizontal(classes="frow"):
                                yield Label("Translation model", classes="flabel")
                                yield Input(
                                    placeholder="(from Settings)",
                                    id="inp-trans-model",
                                    classes="fwidget",
                                )
                            with Horizontal(classes="frow"):
                                yield Label("Base URL", classes="flabel")
                                yield Input(
                                    placeholder="(from Settings)",
                                    id="inp-trans-url",
                                    classes="fwidget",
                                )
                            with Horizontal(classes="frow"):
                                yield Label("API key", classes="flabel")
                                yield Input(
                                    placeholder="(from Settings)",
                                    id="inp-trans-key",
                                    password=True,
                                    classes="fwidget",
                                )

                        yield Static("Advanced", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label("whisper-cli override", classes="flabel")
                            yield Input(
                                placeholder="(from Settings / auto-detect)",
                                id="inp-wcpp-bin",
                                classes="fwidget",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("GGML model override", classes="flabel")
                            yield Input(
                                placeholder="(from Settings / auto-detect)",
                                id="inp-wcpp-model",
                                classes="fwidget",
                            )
                        yield Static(
                            "[dim]Auto-detect scans known model directories and matches ggml-<model>.bin.[/]",
                            id="wcpp-model-status",
                            markup=True,
                        )
                        with Horizontal(classes="crow"):
                            yield Label("Diarize (whisperX only)")
                            yield Switch(id="sw-diarize", value=False)
                        with Horizontal(classes="frow"):
                            yield Label("HuggingFace token", classes="flabel")
                            yield Input(
                                placeholder="hf_…",
                                id="inp-hf-token",
                                password=True,
                                classes="fwidget",
                            )

                # ── Tab 4: History ────────────────────────────────────
                with TabPane("4 History", id="tab-history"):
                    with Vertical(id="hist-pane"):
                        with Horizontal(id="hist-actions"):
                            yield Button(
                                "🔄 Refresh", id="btn-hist-refresh", variant="default"
                            )
                            yield Button(
                                "🗑  Clear All", id="btn-hist-clear", variant="error"
                            )
                            yield Button(
                                "❌ Delete Row", id="btn-hist-delete", variant="warning"
                            )
                            yield Static("", id="hist-count")
                        yield DataTable(
                            id="hist-table",
                            cursor_type="row",
                            zebra_stripes=True,
                        )
                        yield Static(
                            "[dim]Select a row to view details[/]",
                            id="hist-detail",
                            markup=True,
                        )

                # ── Tab 5: Settings ───────────────────────────────────
                with TabPane("5 Settings", id="tab-settings"):
                    with ScrollableContainer(classes="tab-body"):
                        yield Static(
                            f"[dim]  DB: {_db._path}[/]",
                            id="stg-db-path",
                            markup=True,
                        )

                        yield Static("whisper.cpp", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label(ENV_WCPP_BIN, classes="flabel")
                            yield Input(
                                id="stg-wcpp-bin",
                                classes="fwidget",
                                placeholder="path/to/whisper-cli",
                            )
                        with Horizontal(classes="frow"):
                            yield Label(ENV_WCPP_MODEL, classes="flabel")
                            yield Input(
                                id="stg-wcpp-model",
                                classes="fwidget",
                                placeholder="path/to/ggml-large-v3.bin",
                            )

                        yield Static("Translation API", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label(ENV_TRANS_URL, classes="flabel")
                            yield Input(
                                id="stg-trans-url",
                                classes="fwidget",
                                placeholder="https://api.openai.com/v1",
                            )
                        with Horizontal(classes="frow"):
                            yield Label(ENV_TRANS_KEY, classes="flabel")
                            yield Input(
                                id="stg-trans-key",
                                classes="fwidget",
                                placeholder="sk-…",
                                password=True,
                            )
                        with Horizontal(classes="frow"):
                            yield Label(ENV_TRANS_MOD, classes="flabel")
                            yield Input(
                                id="stg-trans-model",
                                classes="fwidget",
                                placeholder="gpt-4.1-mini",
                            )

                        yield Static("Build & Installation", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label("Build dir", classes="flabel")
                            yield Input(id="stg-build-dir", classes="fwidget")
                        with Horizontal(classes="frow"):
                            yield Label("Install dir", classes="flabel")
                            yield Input(id="stg-install-dir", classes="fwidget")
                        with Horizontal(classes="frow"):
                            yield Label("Model dir", classes="flabel")
                            yield Input(id="stg-model-dir", classes="fwidget")
                        with Horizontal(classes="frow"):
                            yield Label("Browse root", classes="flabel")
                            yield Input(id="stg-browse-root", classes="fwidget")

                        yield Static("", id="stg-status", markup=True)
                        with Horizontal(id="stg-actions"):
                            yield Button(
                                "💾 Save", id="btn-stg-save", variant="primary"
                            )
                            yield Button(
                                "🔄 Reload", id="btn-stg-reload", variant="default"
                            )
                            yield Button(
                                "📤 Export .env", id="btn-export-env", variant="default"
                            )

        # ── Bottom panel (always visible) ─────────────────────────────────
        with Vertical(id="bottom"):
            yield Static(
                "[dim]Browse → select paths → Ctrl+R to run[/]",
                id="cmd-preview",
                markup=True,
            )
            with Horizontal(id="run-btns"):
                yield Button("▶ Run", id="btn-run", variant="success")
                yield Button("⚡ Dry Run", id="btn-dry-run", variant="warning")
                yield Button("✕ Kill", id="btn-kill", variant="error", disabled=True)
            yield RichLog(
                id="log",
                highlight=True,
                markup=True,
                auto_scroll=True,
                wrap=True,
                max_lines=5000,
            )

        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        self._migrate_env_to_db()
        self._apply_db_to_env()
        self._load_settings_form()
        self._prefill_transcribe()
        self._load_setup_inputs()
        self._init_history_table()
        self._refresh_history()
        self._refresh_recent_paths()
        self._run_detection()
        self._update_cmd_preview()
        # Navigate tree to configured root
        root = _db.get("tui.browse_root") or str(Path.home())
        try:
            self.query_one("#dir-tree", DirectoryTree).path = Path(root)
            self.query_one("#tree-root", Input).value = root
        except (NoMatches, Exception):
            pass
        self._update_wcpp_model_status()

    # ── Event handlers ────────────────────────────────────────────────────

    def on_input_changed(self, event: Input.Changed) -> None:
        self._update_cmd_preview()
        if event.input.id == "inp-path-search":
            query = event.value.strip()
            if not query:
                self._clear_path_search(
                    "[dim]Type a keyword to search the current root.[/]"
                )
            elif len(query) < 2:
                self._clear_path_search(
                    "[dim]Type at least 2 characters to start searching.[/]"
                )
            else:
                self._start_path_search()
        elif event.input.id in {"inp-wcpp-model", "tree-root"}:
            self._update_wcpp_model_status()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "tree-root":
            self._goto_tree(event.value.strip())
        elif event.input.id == "inp-manual-path":
            path = event.value.strip()
            if path:
                self._add_path(path)
                event.input.value = ""
        elif event.input.id == "inp-path-search":
            self._start_path_search()

    def on_select_changed(self, _: Select.Changed) -> None:
        self._update_wcpp_model_status()
        self._update_cmd_preview()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "fmt-all" and event.value:
            for fmt in FORMATS:
                try:
                    self.query_one(f"#fmt-{fmt}", Checkbox).value = False
                except NoMatches:
                    pass
        elif (
            event.checkbox.id
            and event.checkbox.id.startswith("fmt-")
            and event.checkbox.id != "fmt-all"
            and event.value
        ):
            try:
                self.query_one("#fmt-all", Checkbox).value = False
            except NoMatches:
                pass
        self._update_cmd_preview()

    def on_switch_changed(self, event: Switch.Changed) -> None:
        if event.switch.id == "sw-translate":
            try:
                flds = self.query_one("#trans-fields")
                if event.value:
                    flds.remove_class("hidden")
                else:
                    flds.add_class("hidden")
            except NoMatches:
                pass
        self._update_cmd_preview()

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        self._tree_selection = event.path

    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        self._tree_selection = event.path

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "hist-table":
            key = event.row_key
            self._hist_key = str(key.value) if key else None
            self._show_hist_detail(self._hist_key)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        state = event.state
        try:
            rb = self.query_one("#btn-run", Button)
            db = self.query_one("#btn-dry-run", Button)
            kb = self.query_one("#btn-kill", Button)
        except NoMatches:
            return
        if state == WorkerState.RUNNING:
            rb.disabled = True
            db.disabled = True
            kb.disabled = False
            self.sub_title = "Running…"
        else:
            rb.disabled = False
            db.disabled = False
            kb.disabled = True
            label = {
                WorkerState.CANCELLED: "Cancelled",
                WorkerState.ERROR: "Error",
            }.get(state, "Idle")
            self.sub_title = label
            # refresh history when a transcription worker finishes
            if event.worker.name == "_stream":
                self._refresh_history()

    def on_button_pressed(self, event: Button.Pressed) -> None:  # noqa: C901
        bid = event.button.id or ""

        # ── Browse ─────────────────────────────────────────────
        if bid == "btn-tree-go":
            try:
                v = self.query_one("#tree-root", Input).value.strip()
                self._goto_tree(v)
            except NoMatches:
                pass
        elif bid == "btn-add-sel":
            if self._tree_selection:
                self._add_path(str(self._tree_selection))
        elif bid == "btn-manual-add":
            try:
                inp = self.query_one("#inp-manual-path", Input)
                path = inp.value.strip()
                if path:
                    self._add_path(path)
                    inp.value = ""
            except NoMatches:
                pass
        elif bid == "btn-clear-paths":
            self._selected_paths.clear()
            self._refresh_sel_paths()
            self._update_cmd_preview()
        elif bid == "btn-search-paths":
            self._start_path_search()
        elif bid == "btn-clear-search":
            try:
                self.query_one("#inp-path-search", Input).value = ""
            except NoMatches:
                pass
            self._clear_path_search("[dim]Type a keyword to search the current root.[/]")
        elif bid.startswith("selrm-"):
            try:
                idx = int(bid.removeprefix("selrm-"))
                if 0 <= idx < len(self._selected_paths):
                    self._selected_paths.pop(idx)
                    self._refresh_sel_paths()
                    self._update_cmd_preview()
            except ValueError:
                pass
        elif bid.startswith("radd-"):
            try:
                idx = int(bid.removeprefix("radd-"))
                recent = _db.get_recent_paths(limit=20)
                if idx < len(recent):
                    self._add_path(recent[idx]["path"])
            except (ValueError, IndexError):
                pass
        elif bid.startswith("sgo-"):
            try:
                idx = int(bid.removeprefix("sgo-"))
                if 0 <= idx < len(self._search_results):
                    target = Path(self._search_results[idx])
                    self._tree_selection = target
                    self._goto_tree(str(target if target.is_dir() else target.parent))
            except ValueError:
                pass
        elif bid.startswith("sadd-"):
            try:
                idx = int(bid.removeprefix("sadd-"))
                if 0 <= idx < len(self._search_results):
                    self._tree_selection = Path(self._search_results[idx])
                    self._add_path(self._search_results[idx])
            except ValueError:
                pass

        # ── Setup ──────────────────────────────────────────────
        elif bid == "btn-redetect":
            self._run_detection()
        elif bid == "btn-build-whisper":
            self._save_setup_build_fields()
            self._build_whisper_cpp()
        elif bid == "btn-download-model":
            self._save_setup_model_fields()
            model = self._sel("sel-dl-model", "large-v3")
            self._download_model(model)
        elif bid == "btn-pip-fw":
            self._pip_install("faster-whisper")
        elif bid == "btn-pip-wx":
            self._pip_install("whisperx")

        # ── Run panel ──────────────────────────────────────────
        elif bid == "btn-run":
            self.action_run()
        elif bid == "btn-dry-run":
            self.action_dry_run()
        elif bid == "btn-kill":
            self.action_kill()

        # ── History ────────────────────────────────────────────
        elif bid == "btn-hist-refresh":
            self._refresh_history()
        elif bid == "btn-hist-clear":
            _db.clear_jobs()
            self._hist_key = None
            self._refresh_history()
        elif bid == "btn-hist-delete":
            if self._hist_key:
                _db.delete_job(int(self._hist_key))
                self._hist_key = None
                self._refresh_history()

        # ── Settings ───────────────────────────────────────────
        elif bid == "btn-stg-save":
            self._save_settings()
        elif bid == "btn-stg-reload":
            self._load_settings_form()
        elif bid == "btn-export-env":
            self._export_env()

    # ── Actions ───────────────────────────────────────────────────────────

    def action_run(self) -> None:
        self._trigger(dry_run=False)

    def action_dry_run(self) -> None:
        self._trigger(dry_run=True)

    def action_kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log("[yellow]⚡ Process terminated.[/]")
        if self._active_worker and self._active_worker.is_running:
            self._active_worker.cancel()

    def action_save_settings(self) -> None:
        self._save_settings()

    def action_tab(self, tab: str) -> None:
        try:
            self.query_one(TabbedContent).active = tab
        except NoMatches:
            pass

    async def action_quit_app(self) -> None:
        self.action_kill()
        self.exit()

    # ── Browse tab ────────────────────────────────────────────────────────

    def _goto_tree(self, raw: str) -> None:
        p = Path(raw).expanduser()
        target = p if p.is_dir() else (p.parent if p.exists() else None)
        if target and target.is_dir():
            try:
                self.query_one("#dir-tree", DirectoryTree).path = target
                self.query_one("#tree-root", Input).value = str(target)
            except NoMatches:
                pass
            if len(self._val("inp-path-search")) >= 2:
                self._start_path_search()

    def _add_path(self, path: str) -> None:
        if path and path not in self._selected_paths:
            self._selected_paths.append(path)
            kind = "directory" if Path(path).is_dir() else "file"
            _db.touch_path(path, kind)
            self._refresh_sel_paths()
            self._refresh_recent_paths()
            self._update_cmd_preview()

    def _refresh_sel_paths(self) -> None:
        try:
            box = self.query_one("#sel-paths-box")
        except NoMatches:
            return
        box.remove_children()
        if not self._selected_paths:
            box.mount(
                Static(
                    "[dim]No paths selected — browse left[/]",
                    markup=True,
                )
            )
            return
        for i, p in enumerate(self._selected_paths):
            row = Horizontal(classes="sel-row")
            box.mount(row)
            row.mount(Static(p, classes="sel-label", markup=False))
            row.mount(Button("✕", id=f"selrm-{i}", variant="error", classes="sel-btn"))

    def _refresh_recent_paths(self) -> None:
        try:
            box = self.query_one("#recent-box")
        except NoMatches:
            return
        box.remove_children()
        recent = _db.get_recent_paths(limit=12)
        if not recent:
            box.mount(Static("[dim]None yet[/]", markup=True))
            return
        for i, r in enumerate(recent):
            row = Horizontal(classes="recent-row")
            box.mount(row)
            row.mount(Static(r["path"], classes="recent-label", markup=False))
            row.mount(
                Button("＋", id=f"radd-{i}", classes="recent-add", variant="default")
            )

    def _clear_path_search(self, status: str) -> None:
        self._search_results = []
        try:
            self.query_one("#search-status", Static).update(status)
        except NoMatches:
            pass
        self._refresh_search_results()

    def _refresh_search_results(self) -> None:
        try:
            box = self.query_one("#search-results-box")
        except NoMatches:
            return
        box.remove_children()
        if not self._search_results:
            box.mount(Static("[dim]No search results yet.[/]", markup=True))
            return
        for i, path in enumerate(self._search_results):
            target = Path(path)
            label = path + ("/" if target.is_dir() else "")
            row = Horizontal(classes="search-row")
            box.mount(row)
            row.mount(Button("Go", id=f"sgo-{i}", classes="search-go"))
            row.mount(Static(label, classes="search-label", markup=False))
            row.mount(
                Button("＋", id=f"sadd-{i}", classes="search-add", variant="default")
            )

    def _start_path_search(self) -> None:
        query = self._val("inp-path-search")
        if len(query) < 2:
            self._clear_path_search("[dim]Type at least 2 characters to start searching.[/]")
            return
        root = Path(self._val("tree-root") or str(Path.home())).expanduser()
        if not root.is_dir():
            self._clear_path_search(f"[red]Search root not found:[/] {root}")
            return
        try:
            self.query_one("#search-status", Static).update(
                f"[cyan]Searching under[/] {root}"
            )
        except NoMatches:
            pass
        self._search_input_paths(str(root), query)

    @work(thread=True, exclusive=True, exit_on_error=False, name="path-search")
    def _search_input_paths(self, root: str, query: str) -> None:
        root_path = Path(root)
        results = discover_input_matches(root_path, query, limit=SEARCH_RESULT_LIMIT)
        if results:
            count_label = f"{len(results)} result(s)"
            if len(results) >= SEARCH_RESULT_LIMIT:
                count_label += f" (showing first {SEARCH_RESULT_LIMIT})"
            status = (
                f"[green]{count_label}[/] for [bold]{query}[/] under {root_path}"
            )
        else:
            status = f"[yellow]No matches[/] for [bold]{query}[/] under {root_path}"
        self.call_from_thread(self._apply_path_search_results, results, status)

    def _apply_path_search_results(self, results: list[str], status: str) -> None:
        self._search_results = results
        try:
            self.query_one("#search-status", Static).update(status)
        except NoMatches:
            pass
        self._refresh_search_results()

    # ── Setup tab ─────────────────────────────────────────────────────────

    def _load_setup_inputs(self) -> None:
        pairs = [
            ("inp-build-dir", "tui.build_dir"),
            ("inp-install-dir", "tui.install_dir"),
            ("inp-model-dir", "tui.model_dir"),
        ]
        for wid, key in pairs:
            try:
                self.query_one(f"#{wid}", Input).value = _db.get(key)
            except NoMatches:
                pass

    def _save_setup_build_fields(self) -> None:
        for wid, key in [
            ("inp-build-dir", "tui.build_dir"),
            ("inp-install-dir", "tui.install_dir"),
        ]:
            v = self._val(wid)
            if v:
                _db.set(key, v)

    def _save_setup_model_fields(self) -> None:
        v = self._val("inp-model-dir")
        if v:
            _db.set("tui.model_dir", v)

    def _setup_log(self, text: str) -> None:
        try:
            self.query_one("#setup-log", RichLog).write(text)
        except NoMatches:
            pass

    def _run_cmd(self, cmd: list[str]) -> bool:
        """Run a subprocess, stream to setup-log. Returns True on success. Call from worker."""
        self.call_from_thread(self._setup_log, f"[dim]$ {' '.join(cmd)}[/]")
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout
            for line in iter(proc.stdout.readline, ""):
                s = line.rstrip()
                if s:
                    self.call_from_thread(self._setup_log, s)
            proc.stdout.close()
            rc = proc.wait()
            if rc != 0:
                self.call_from_thread(self._setup_log, f"[red]✗ exit {rc}[/]")
            return rc == 0
        except FileNotFoundError as e:
            self.call_from_thread(self._setup_log, f"[red]✗ Not found: {e}[/]")
            return False

    @work(thread=True, exclusive=False, exit_on_error=False)
    def _run_detection(self) -> None:
        self._detected_ggml_models = discover_ggml_models()
        results = detect_all()
        self._detect_results = results
        self.call_from_thread(self._update_detect_ui, results)

    def _update_detect_ui(self, results: DetectResult) -> None:
        for name, (found, detail) in results.items():
            icon = "[green]✅[/]" if found else "[red]❌[/]"
            if name == "ggml-model" and self._detected_ggml_models:
                detail = summarize_ggml_models(self._detected_ggml_models)
            dshort = detail[:55] + "…" if len(detail) > 56 else detail
            try:
                self.query_one(f"#det-{name}", Static).update(
                    f"{icon} [bold]{name:<20}[/] {dshort}"
                )
            except NoMatches:
                pass

        # Auto-fill settings if not already set
        wbin = results.get("whisper-cli", (False, ""))[1]
        if results.get("whisper-cli", (False,))[0] and not _db.get(ENV_WCPP_BIN):
            _db.set(ENV_WCPP_BIN, wbin)
            os.environ[ENV_WCPP_BIN] = wbin
            for wid in ("stg-wcpp-bin", "inp-wcpp-bin"):
                try:
                    self.query_one(f"#{wid}", Input).value = wbin
                except NoMatches:
                    pass

        wmod = results.get("ggml-model", (False, ""))[1]
        if results.get("ggml-model", (False,))[0] and not _db.get(ENV_WCPP_MODEL):
            _db.set(ENV_WCPP_MODEL, wmod)
            os.environ[ENV_WCPP_MODEL] = wmod
            for wid in ("stg-wcpp-model",):
                try:
                    self.query_one(f"#{wid}", Input).value = wmod
                except NoMatches:
                    pass
        self._update_wcpp_model_status()

    def _resolved_wcpp_model_path(self) -> str:
        if manual := self._val("inp-wcpp-model"):
            return manual
        selected_model = self._sel("sel-model", DEFAULT_MODEL)
        return self._detected_ggml_models.get(selected_model, "")

    def _update_wcpp_model_status(self) -> None:
        backend = self._sel("sel-backend", DEFAULT_BACKEND)
        manual = self._val("inp-wcpp-model")
        selected_model = self._sel("sel-model", DEFAULT_MODEL)
        auto_path = self._detected_ggml_models.get(selected_model, "")

        if backend != "whisper-cpp":
            status = "[dim]GGML auto-detect is used only with the whisper-cpp backend.[/]"
        elif manual:
            status = f"[yellow]Manual override[/] {manual}"
        elif auto_path:
            filename = Path(auto_path).name
            status = f"[green]Auto[/] {filename} -> {auto_path}"
        elif self._detected_ggml_models:
            available = ", ".join(sorted(self._detected_ggml_models)[:4])
            extra = len(self._detected_ggml_models) - 4
            more = f" (+{extra} more)" if extra > 0 else ""
            status = (
                f"[yellow]No ggml-{selected_model}.bin detected.[/] "
                f"Available: {available}{more}"
            )
        else:
            status = (
                f"[yellow]No GGML model detected for {selected_model}.[/] "
                "Use Setup -> Download Model or set a manual override."
            )

        try:
            self.query_one("#wcpp-model-status", Static).update(status)
        except NoMatches:
            pass

    @work(thread=True, exclusive=False, exit_on_error=False, name="build-whisper")
    def _build_whisper_cpp(self) -> None:
        log = lambda m: self.call_from_thread(self._setup_log, m)

        build_dir = _db.get("tui.build_dir") or str(
            Path.home() / ".cache/vid_to_sub_build"
        )
        install_dir = _db.get("tui.install_dir") or str(Path.home() / ".local/bin")
        build_root = Path(build_dir).expanduser().resolve()
        install_d = Path(install_dir).expanduser().resolve()
        repo_dir = build_root / "whisper.cpp"

        if not shutil.which("git"):
            log("[red]✗ git not found in PATH — needed for cloning[/]")
            return
        if not shutil.which("cmake"):
            log("[red]✗ cmake not found in PATH — needed for building[/]")
            return

        build_root.mkdir(parents=True, exist_ok=True)

        if not (repo_dir / ".git").exists():
            log("[cyan]Cloning whisper.cpp (shallow)…[/]")
            ok = self._run_cmd(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "https://github.com/ggerganov/whisper.cpp",
                    str(repo_dir),
                ]
            )
            if not ok:
                log("[red]✗ Clone failed[/]")
                return
        else:
            log("[cyan]Pulling latest whisper.cpp…[/]")
            self._run_cmd(["git", "-C", str(repo_dir), "pull", "--ff-only"])

        log("[cyan]CMake configure…[/]")
        cmake_build = repo_dir / "build"
        ok = self._run_cmd(
            [
                "cmake",
                "-B",
                str(cmake_build),
                "-DCMAKE_BUILD_TYPE=Release",
                str(repo_dir),
            ]
        )
        if not ok:
            log("[red]✗ cmake configure failed[/]")
            return

        nproc = str(os.cpu_count() or 4)
        log(f"[cyan]Building with {nproc} threads…[/]")
        ok = self._run_cmd(
            [
                "cmake",
                "--build",
                str(cmake_build),
                "-j",
                nproc,
                "--target",
                "whisper-cli",
            ]
        )
        if not ok:
            log("[yellow]whisper-cli target not found, building all…[/]")
            ok = self._run_cmd(["cmake", "--build", str(cmake_build), "-j", nproc])
        if not ok:
            log("[red]✗ Build failed[/]")
            return

        # Locate binary
        candidates = [
            cmake_build / "bin" / "whisper-cli",
            cmake_build / "whisper-cli",
            cmake_build / "bin" / "main",
            cmake_build / "main",
        ]
        binary = next((c for c in candidates if c.exists()), None)
        if not binary:
            log("[red]✗ Built binary not found — check cmake output above[/]")
            return

        install_d.mkdir(parents=True, exist_ok=True)
        dest = install_d / "whisper-cli"
        try:
            shutil.copy2(str(binary), str(dest))
            os.chmod(str(dest), 0o755)
        except PermissionError:
            log(f"[red]✗ Permission denied writing to {dest}[/]")
            log(f"[yellow]  Tip: set Install dir to ~/.local/bin[/]")
            return

        _db.set(ENV_WCPP_BIN, str(dest))
        os.environ[ENV_WCPP_BIN] = str(dest)

        def _update_bin_inputs() -> None:
            for wid in ("stg-wcpp-bin", "inp-wcpp-bin"):
                try:
                    self.query_one(f"#{wid}", Input).value = str(dest)
                except NoMatches:
                    pass

        self.call_from_thread(_update_bin_inputs)

        log(f"[green]✅ whisper-cli installed → {dest}[/]")
        self.call_from_thread(self._run_detection)

    @work(thread=True, exclusive=False, exit_on_error=False, name="download-model")
    def _download_model(self, model_name: str) -> None:
        log = lambda m: self.call_from_thread(self._setup_log, m)

        model_dir_s = _db.get("tui.model_dir") or str(SCRIPT_DIR / "models")
        model_dir = Path(model_dir_s).expanduser()
        if not model_dir.is_absolute():
            model_dir = SCRIPT_DIR / model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        filename = f"ggml-{model_name}.bin"
        dest = model_dir / filename

        if dest.exists():
            size = dest.stat().st_size / 1024 / 1024
            log(f"[yellow]⚠ {filename} already at {dest} ({size:.0f} MB) — skipping[/]")
            _db.set(ENV_WCPP_MODEL, str(dest))
            os.environ[ENV_WCPP_MODEL] = str(dest)
            self.call_from_thread(self._run_detection)
            return

        url = f"{HF_MODEL_BASE}/{filename}"
        log(f"[cyan]Downloading {filename}…[/]")
        log(f"[dim]  {url}[/]")

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vid_to_sub/2"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                total_mb = total / 1024 / 1024
                downloaded = 0
                last_pct = -1
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            if pct >= last_pct + 5:
                                log(
                                    f"  [{pct:3d}%] {downloaded / 1024 / 1024:.0f}/{total_mb:.0f} MB"
                                )
                                last_pct = pct
        except urllib.error.URLError as e:
            log(f"[red]✗ Download failed: {e}[/]")
            if dest.exists():
                dest.unlink()
            return
        except OSError as e:
            log(f"[red]✗ Write error: {e}[/]")
            if dest.exists():
                dest.unlink()
            return

        size = dest.stat().st_size / 1024 / 1024
        _db.set(ENV_WCPP_MODEL, str(dest))
        os.environ[ENV_WCPP_MODEL] = str(dest)
        log(f"[green]✅ {filename} saved → {dest} ({size:.0f} MB)[/]")
        self.call_from_thread(self._run_detection)

    @work(thread=True, exclusive=False, exit_on_error=False, name="pip-install")
    def _pip_install(self, package: str) -> None:
        self.call_from_thread(self._setup_log, f"[cyan]Installing {package}…[/]")
        ok = self._run_cmd([sys.executable, "-m", "pip", "install", package])
        if ok:
            self.call_from_thread(self._setup_log, f"[green]✅ {package} installed[/]")
        self.call_from_thread(self._run_detection)

    # ── Command builder ───────────────────────────────────────────────────

    def _build_cmd(self, dry_run: bool = False) -> list[str]:
        if not self._selected_paths:
            raise ValueError("No input paths — use Browse tab to select")

        cmd: list[str] = [sys.executable, str(SCRIPT_PATH)]
        cmd.extend(self._selected_paths)

        if v := self._val("inp-output-dir"):
            cmd += ["--output-dir", v]

        if self._chk("chk-no-recurse"):
            cmd.append("--no-recurse")
        if self._chk("chk-skip-existing"):
            cmd.append("--skip-existing")
        if dry_run:
            cmd.append("--dry-run")
        if self._chk("chk-verbose"):
            cmd.append("--verbose")

        if self._chk("fmt-all"):
            cmd += ["--format", "all"]
        else:
            fmts = [f for f in FORMATS if self._chk(f"fmt-{f}")]
            if not fmts:
                fmts = ["srt"]
            for f in fmts:
                cmd += ["--format", f]

        backend = self._sel("sel-backend", DEFAULT_BACKEND)
        model = self._sel("sel-model", DEFAULT_MODEL)
        cmd += ["--backend", backend]
        cmd += ["--model", model]
        cmd += ["--device", self._sel("sel-device", DEFAULT_DEVICE)]

        if v := self._val("inp-language"):
            cmd += ["--language", v]
        if v := self._val("inp-compute-type"):
            cmd += ["--compute-type", v]

        beam = self._val("inp-beam-size") or "5"
        if beam != "5":
            cmd += ["--beam-size", beam]

        workers = self._val("inp-workers") or "1"
        if workers != "1":
            cmd += ["--workers", workers]

        if backend == "whisper-cpp" and (v := self._resolved_wcpp_model_path()):
            cmd += ["--whisper-cpp-model-path", v]

        if self._sw("sw-translate"):
            if v := self._val("inp-translate-to"):
                cmd += ["--translate-to", v]
            if v := self._val("inp-trans-model"):
                cmd += ["--translation-model", v]
            if v := self._val("inp-trans-url"):
                cmd += ["--translation-base-url", v]
            if v := self._val("inp-trans-key"):
                cmd += ["--translation-api-key", v]

        if self._sw("sw-diarize"):
            cmd.append("--diarize")
        if v := self._val("inp-hf-token"):
            cmd += ["--hf-token", v]

        return cmd

    def _update_cmd_preview(self) -> None:
        try:
            cmd = self._build_cmd()
            display = " ".join(_mask(cmd[2:]))
            self.query_one("#cmd-preview", Static).update(
                f"[dim]$[/dim] [cyan]vid_to_sub[/cyan] {display}"
            )
        except ValueError as exc:
            try:
                self.query_one("#cmd-preview", Static).update(f"[dim]{exc}[/]")
            except NoMatches:
                pass
        except Exception:
            pass

    # ── Run & Kill ────────────────────────────────────────────────────────

    def _build_run_env(self) -> dict[str, str]:
        env = os.environ.copy()
        # Inject SQLite-backed env vars
        env.update(_db.get_env_dict())
        # whisper-cli binary override from Transcribe tab
        if v := self._val("inp-wcpp-bin"):
            env[ENV_WCPP_BIN] = v
        backend = self._sel("sel-backend", DEFAULT_BACKEND)
        if backend == "whisper-cpp":
            resolved_model = self._resolved_wcpp_model_path()
            stored_model = _db.get(ENV_WCPP_MODEL)
            if resolved_model:
                env[ENV_WCPP_MODEL] = resolved_model
            elif stored_model and stored_model in self._detected_ggml_models.values():
                env.pop(ENV_WCPP_MODEL, None)
        # Translation overrides
        for wid, evar in [
            ("inp-trans-url", ENV_TRANS_URL),
            ("inp-trans-key", ENV_TRANS_KEY),
            ("inp-trans-model", ENV_TRANS_MOD),
        ]:
            if v := self._val(wid):
                env[evar] = v
        return env

    def _trigger(self, dry_run: bool = False) -> None:
        try:
            cmd = self._build_cmd(dry_run=dry_run)
        except ValueError as exc:
            self._log(f"[bold red]✕ {exc}[/]")
            return

        log = self.query_one("#log", RichLog)
        log.clear()
        log.write("[bold cyan]$ " + " ".join(_mask(cmd[2:])) + "[/bold cyan]")

        self._active_jobs.clear()
        self._job_t0 = time.monotonic()
        self._active_worker = self._stream(cmd)

    @work(thread=True, exclusive=True, exit_on_error=False, name="_stream")
    def _stream(self, cmd: list[str]) -> None:
        env = self._build_run_env()

        # Extract job metadata from cmd
        backend = DEFAULT_BACKEND
        model = DEFAULT_MODEL
        t_lang: str | None = None
        for i, tok in enumerate(cmd):
            if tok == "--backend" and i + 1 < len(cmd):
                backend = cmd[i + 1]
            elif tok == "--model" and i + 1 < len(cmd):
                model = cmd[i + 1]
            elif tok == "--translate-to" and i + 1 < len(cmd):
                t_lang = cmd[i + 1]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except FileNotFoundError as exc:
            self.call_from_thread(self._log, f"[bold red]✕ Not found: {exc}[/]")
            return
        except Exception as exc:
            self.call_from_thread(self._log, f"[bold red]✕ {exc}[/]")
            return

        self._proc = proc
        active: dict[str, int] = {}  # video_path -> job_id

        assert proc.stdout
        for raw in iter(proc.stdout.readline, ""):
            stripped = raw.rstrip("\n")
            if not stripped:
                continue
            self.call_from_thread(self._log, _colorize(stripped))

            # ▶  /path/to/video.mp4
            if stripped.startswith("▶  ") or stripped.startswith("[0] ▶  "):
                vpath = re.sub(r"^\[\d+\] ", "", stripped).lstrip("▶").strip()
                jid = _db.create_job(
                    video_path=vpath,
                    backend=backend,
                    model=model,
                    output_dir=self._val("inp-output-dir") or None,
                    language=self._val("inp-language") or None,
                    target_lang=t_lang,
                )
                active[vpath] = jid

            #    ✓ [en]  video=0:01:23  wall=0:00:15  video.srt
            elif "✓" in stripped:
                for vp, jid in list(active.items()):
                    if Path(vp).stem in stripped:
                        m = re.search(r"wall=(\d+:\d+:\d+)", stripped)
                        wsec: float | None = None
                        if m:
                            p = m.group(1).split(":")
                            wsec = int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
                        _db.finish_job(jid, "done", wall_sec=wsec)
                        del active[vp]
                        break

            # [ERROR] Transcription failed: …
            elif "[ERROR]" in stripped:
                for vp, jid in list(active.items()):
                    _db.finish_job(jid, "failed", error=stripped[:500])
                    del active[vp]

        proc.stdout.close()
        rc = proc.wait()
        self._proc = None

        # Close any jobs still running
        status = "done" if rc == 0 else "failed"
        for vp, jid in active.items():
            _db.finish_job(jid, status)

        if rc == 0:
            self.call_from_thread(self._log, "\n[bold green]✓ Completed (exit 0)[/]")
        else:
            self.call_from_thread(self._log, f"\n[bold red]✕ Exited with code {rc}[/]")

    # ── History tab ───────────────────────────────────────────────────────

    def _init_history_table(self) -> None:
        try:
            t = self.query_one("#hist-table", DataTable)
            t.add_column("ID", key="id", width=5)
            t.add_column("Date", key="date", width=11)
            t.add_column("File", key="file", width=26)
            t.add_column("Backend", key="backend", width=13)
            t.add_column("Model", key="model", width=13)
            t.add_column("Status", key="status", width=10)
            t.add_column("Time", key="time", width=8)
        except NoMatches:
            pass

    def _refresh_history(self) -> None:
        try:
            t = self.query_one("#hist-table", DataTable)
        except NoMatches:
            return
        t.clear(columns=False)
        jobs = _db.get_jobs()

        for job in jobs:
            s = job["status"]
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
        lines = [
            f"[bold]ID {job['id']}[/]  {job['created_at']}  "
            f"backend={job['backend']}  model={job['model']}",
            f"File: {job['video_path']}",
        ]
        if job.get("output_dir"):
            lines.append(f"Output dir: {job['output_dir']}")
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

    # ── Settings ──────────────────────────────────────────────────────────

    def _load_settings_form(self) -> None:
        pairs = [
            ("stg-wcpp-bin", ENV_WCPP_BIN),
            ("stg-wcpp-model", ENV_WCPP_MODEL),
            ("stg-trans-url", ENV_TRANS_URL),
            ("stg-trans-key", ENV_TRANS_KEY),
            ("stg-trans-model", ENV_TRANS_MOD),
            ("stg-build-dir", "tui.build_dir"),
            ("stg-install-dir", "tui.install_dir"),
            ("stg-model-dir", "tui.model_dir"),
            ("stg-browse-root", "tui.browse_root"),
        ]
        for wid, key in pairs:
            try:
                self.query_one(f"#{wid}", Input).value = _db.get(key)
            except NoMatches:
                pass

    def _prefill_transcribe(self) -> None:
        """Pre-fill Transcribe tab from settings / env vars."""
        for wid, key in [
            ("inp-trans-url", ENV_TRANS_URL),
            ("inp-trans-key", ENV_TRANS_KEY),
            ("inp-trans-model", ENV_TRANS_MOD),
            ("inp-wcpp-bin", ENV_WCPP_BIN),
        ]:
            try:
                val = _db.get(key) or os.environ.get(key, "")
                if val:
                    self.query_one(f"#{wid}", Input).value = val
            except NoMatches:
                pass

    def _save_settings(self) -> None:
        pairs = [
            ("stg-wcpp-bin", ENV_WCPP_BIN),
            ("stg-wcpp-model", ENV_WCPP_MODEL),
            ("stg-trans-url", ENV_TRANS_URL),
            ("stg-trans-key", ENV_TRANS_KEY),
            ("stg-trans-model", ENV_TRANS_MOD),
            ("stg-build-dir", "tui.build_dir"),
            ("stg-install-dir", "tui.install_dir"),
            ("stg-model-dir", "tui.model_dir"),
            ("stg-browse-root", "tui.browse_root"),
        ]
        data: dict[str, str] = {}
        for wid, key in pairs:
            try:
                data[key] = self.query_one(f"#{wid}", Input).value.strip()
            except NoMatches:
                pass
        _db.set_many(data)
        self._apply_db_to_env()
        self._run_detection()
        self._update_wcpp_model_status()
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
        """Push VID_TO_SUB_ settings from DB into os.environ."""
        for key, val in _db.get_env_dict().items():
            if val:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    def _migrate_env_to_db(self) -> None:
        """One-time migration: if .env exists and DB keys are empty, import them."""
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=False)
        env_keys = [
            ENV_WCPP_BIN,
            ENV_WCPP_MODEL,
            ENV_TRANS_URL,
            ENV_TRANS_KEY,
            ENV_TRANS_MOD,
        ]
        updates: dict[str, str] = {}
        for key in env_keys:
            if not _db.get(key):
                val = os.environ.get(key, "")
                if val:
                    updates[key] = val
        if updates:
            _db.set_many(updates)

    # ── Widget accessors ──────────────────────────────────────────────────

    def _val(self, wid: str) -> str:
        try:
            return self.query_one(f"#{wid}", Input).value.strip()
        except NoMatches:
            return ""

    def _sel(self, wid: str, fallback: str = "") -> str:
        try:
            v = self.query_one(f"#{wid}", Select).value
            return str(v) if v is not Select.BLANK else fallback
        except NoMatches:
            return fallback

    def _chk(self, wid: str) -> bool:
        try:
            return bool(self.query_one(f"#{wid}", Checkbox).value)
        except NoMatches:
            return False

    def _sw(self, wid: str) -> bool:
        try:
            return bool(self.query_one(f"#{wid}", Switch).value)
        except NoMatches:
            return False

    def _log(self, text: str) -> None:
        try:
            self.query_one("#log", RichLog).write(text)
        except NoMatches:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if not SCRIPT_PATH.exists():
        print(f"Error: vid_to_sub.py not found at {SCRIPT_PATH}", file=sys.stderr)
        sys.exit(1)
    app = VidToSubApp()
    app.run()


if __name__ == "__main__":
    main()
