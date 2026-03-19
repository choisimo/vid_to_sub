from __future__ import annotations

import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    TextArea,
)
from textual.worker import Worker, WorkerState

from vid_to_sub_app.cli import (
    apply_runtime_path_map_to_manifest,
    build_run_manifest,
    discover_videos,
    hash_video_folder,
)
from vid_to_sub_app.cli.runner import primary_output_exists as _primary_output_exists
from vid_to_sub_app.shared.constants import DEVICES, ROOT_DIR
from vid_to_sub_app.shared.env import find_whisper_cpp_bin, load_project_env

from .helpers import (
    BACKENDS,
    DEFAULT_BACKEND,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DetectResult,
    ENV_AGENT_KEY,
    ENV_AGENT_MOD,
    ENV_AGENT_URL,
    ENV_POST_KEY,
    ENV_POST_MOD,
    ENV_POST_URL,
    ENV_FILE,
    ENV_TRANS_KEY,
    ENV_TRANS_MOD,
    ENV_TRANS_URL,
    ENV_WCPP_BIN,
    ENV_WCPP_MODEL,
    EXECUTION_MODES,
    ExecutorPlan,
    FORMATS,
    HF_MODEL_BASE,
    KNOWN_MODELS,
    POSTPROCESS_MODES,
    PIP_REQUIREMENT_FILES,
    RemoteResourceProfile,
    RunConfig,
    RunJobState,
    SSHConnection,
    ssh_connection_from_row,
    SEARCH_RESULT_LIMIT,
    _colorize,
    _coerce_positive_int,
    _compact_progress_markup,
    _fmt_elapsed,
    _mask,
    _opts,
    _progress_bar_markup,
    _progress_bar_markup_ratio,
    build_search_preview,
    build_system_install_commands,
    detect_all,
    detect_package_manager,
    discover_ggml_models,
    discover_input_matches,
    extract_json_payload,
    group_paths_by_video_folder,
    map_path_for_remote,
    normalize_chat_endpoint,
    packages_for_manager,
    parse_progress_event,
    parse_remote_resources,
    partition_folder_groups_by_capacity,
    summarize_ggml_models,
    summarize_remote_resources,
    resolve_runtime_backend_threads,
)
from .state import db as _db
from .styles import _CSS
from .mixins import (
    AgentMixin,
    BrowseMixin,
    HistoryMixin,
    RunMixin,
    SettingsMixin,
    SetupMixin,
)

SCRIPT_DIR = ROOT_DIR
SCRIPT_PATH = ROOT_DIR / "vid_to_sub.py"


class VidToSubApp(
    BrowseMixin,
    SetupMixin,
    RunMixin,
    HistoryMixin,
    SettingsMixin,
    AgentMixin,
    App,
):
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
        Binding("6", "tab('tab-agent')", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        _db.seed_defaults()
        self._selected_paths: list[str] = []
        self._search_results: list[str] = []
        self._search_preview_path: str | None = None
        self._tree_selection: Path | None = None
        self._detect_results: DetectResult = {}
        self._detected_ggml_models: dict[str, str] = {}
        self._remote_resources: list[RemoteResourceProfile] = []
        self._ssh_selected_id: int | None = None  # selected SSH connection id
        self._agent_plan: dict[str, Any] | None = None
        self._active_worker: Worker | None = None
        self._proc: subprocess.Popen | None = None
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._hist_key: str | None = None
        self._active_jobs: dict[str, RunJobState] = {}
        self._pending_paths: dict[str, set[str]] = {}
        self._run_started_at: float | None = None
        self._run_total_found = 0
        self._run_total_queued = 0
        self._run_skipped = 0
        self._run_completed = 0
        self._run_failed = 0
        self._run_backend = DEFAULT_BACKEND
        self._run_model = DEFAULT_MODEL
        self._run_language: str | None = None
        self._run_target_lang: str | None = None
        self._run_output_dir: str | None = None
        self._run_dry_run = False
        self._run_last_shell = "[dim]Browse → select paths → Ctrl+R to run[/]"
        self._run_shell_collapsed = True
        self._run_request_id = 0

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
                            yield Static("Selected Paths", classes="stitle")
                            with Vertical(id="sel-paths-box"):
                                yield Static(
                                    "[dim]No paths selected — browse left[/]",
                                    id="sel-empty",
                                )

                            with Horizontal(id="browse-actions"):
                                yield Button(
                                    "Add Sel",
                                    id="btn-add-sel",
                                    variant="success",
                                )
                                yield Button(
                                    "Clear",
                                    id="btn-clear-paths",
                                    variant="error",
                                )

                            with Horizontal(id="manual-add-row"):
                                yield Input(
                                    placeholder="Or type/paste path here…",
                                    id="inp-manual-path",
                                )
                                yield Button("Add", id="btn-manual-add", variant="default")

                            yield Static("Quick Search", classes="stitle")
                            with Horizontal(id="search-row"):
                                yield Input(
                                    placeholder="Find directories or video files under root…",
                                    id="inp-path-search",
                                )
                                yield Button("Find", id="btn-search-paths")
                                yield Button("Reset", id="btn-clear-search")
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
                            yield Static(
                                "Search result preview will appear here.",
                                id="search-preview",
                                markup=False,
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
                            "whisper",
                            "whisperx",
                        ):
                            yield Static(
                                f"[dim]{comp}  scanning…[/]",
                                id=f"det-{comp}",
                                classes="det-row",
                            )

                        with Horizontal(classes="crow"):
                            yield Button("Detect", id="btn-redetect", variant="default")

                        yield Static("Automatic Setup", classes="stitle")
                        with Horizontal(classes="crow"):
                            yield Button(
                                "Auto",
                                id="btn-auto-setup",
                                variant="primary",
                            )
                            yield Button(
                                "Full",
                                id="btn-full-setup",
                                variant="warning",
                            )
                        yield Static(
                            "[dim]Best effort: install missing system tools, Python deps, build whisper.cpp, and download the selected model.[/]",
                            id="auto-setup-status",
                            markup=True,
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
                                "Build whisper-cli",
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
                                "Download",
                                id="btn-download-model",
                                variant="warning",
                            )

                        yield Static("Python Packages", classes="stitle")
                        with Horizontal(classes="crow"):
                            yield Button(
                                "pip fw",
                                id="btn-pip-fw",
                                variant="default",
                            )
                            yield Button(
                                "pip whisper",
                                id="btn-pip-whisper",
                                variant="default",
                            )
                            yield Button(
                                "pip wx",
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
                            yield Label("Local workers", classes="flabel")
                            yield Input(
                                value="1",
                                id="inp-workers",
                                type="integer",
                                classes="fwidget",
                            )

                        yield Static("Execution", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label("Mode", classes="flabel")
                            yield Select(
                                _opts(EXECUTION_MODES, "local"),
                                id="sel-execution-mode",
                                classes="fwidget",
                                allow_blank=False,
                            )
                        yield Static(
                            "[dim]Local mode runs one process here. Distributed mode splits discovered files across local workers and SSH resources from Settings.[/]",
                            classes="hint",
                            markup=True,
                        )
                        yield Static("", id="remote-status", markup=True)

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
                            with Horizontal(classes="crow"):
                                yield Label("Enable post-edit agent")
                                yield Switch(id="sw-postprocess", value=False)
                            with Vertical(id="post-fields", classes="hidden"):
                                with Horizontal(classes="frow"):
                                    yield Label("Post-edit mode", classes="flabel")
                                    yield Select(
                                        _opts(list(POSTPROCESS_MODES), "auto"),
                                        id="sel-postprocess-mode",
                                        classes="fwidget",
                                        value="auto",
                                    )
                                with Horizontal(classes="frow"):
                                    yield Label("Post-edit model", classes="flabel")
                                    yield Input(
                                        placeholder="(from Settings / translation model)",
                                        id="inp-post-model",
                                        classes="fwidget",
                                    )
                                with Horizontal(classes="frow"):
                                    yield Label("Base URL", classes="flabel")
                                    yield Input(
                                        placeholder="(from Settings / translation URL)",
                                        id="inp-post-url",
                                        classes="fwidget",
                                    )
                                with Horizontal(classes="frow"):
                                    yield Label("API key", classes="flabel")
                                    yield Input(
                                        placeholder="(from Settings / translation API key)",
                                        id="inp-post-key",
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
                            yield Button("Refresh", id="btn-hist-refresh", variant="default")
                            yield Button("Load", id="btn-hist-load", variant="primary")
                            yield Button("Rerun", id="btn-hist-rerun", variant="success")
                            yield Button("Clear", id="btn-hist-clear", variant="error")
                            yield Button("Delete", id="btn-hist-delete", variant="warning")
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

                        yield Static("Post-edit API", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label(ENV_POST_URL, classes="flabel")
                            yield Input(
                                id="stg-post-url",
                                classes="fwidget",
                                placeholder="(blank = use translation URL)",
                            )
                        with Horizontal(classes="frow"):
                            yield Label(ENV_POST_KEY, classes="flabel")
                            yield Input(
                                id="stg-post-key",
                                classes="fwidget",
                                placeholder="(blank = use translation API key)",
                                password=True,
                            )
                        with Horizontal(classes="frow"):
                            yield Label(ENV_POST_MOD, classes="flabel")
                            yield Input(
                                id="stg-post-model",
                                classes="fwidget",
                                placeholder="(blank = use translation model)",
                            )

                        yield Static("AI Agent", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label(ENV_AGENT_URL, classes="flabel")
                            yield Input(
                                id="stg-agent-url",
                                classes="fwidget",
                                placeholder="(blank = use translation URL)",
                            )
                        with Horizontal(classes="frow"):
                            yield Label(ENV_AGENT_KEY, classes="flabel")
                            yield Input(
                                id="stg-agent-key",
                                classes="fwidget",
                                placeholder="(blank = use translation API key)",
                                password=True,
                            )
                        with Horizontal(classes="frow"):
                            yield Label(ENV_AGENT_MOD, classes="flabel")
                            yield Input(
                                id="stg-agent-model",
                                classes="fwidget",
                                placeholder="(blank = use translation model)",
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

                        yield Static("Run Defaults", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label("Output dir", classes="flabel")
                            yield Input(
                                id="stg-default-output-dir",
                                classes="fwidget",
                                placeholder="(blank = next to each video)",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Translate to", classes="flabel")
                            yield Input(
                                id="stg-default-translate-to",
                                classes="fwidget",
                                placeholder="ko",
                            )

                        yield Static("Remote Resources", classes="stitle")
                        yield Static(
                            "[dim]JSON array of SSH resources. Distributed mode assumes shared storage or explicit path_map prefixes. Nothing runs remotely until you switch execution mode and press Run.[/]",
                            classes="hint",
                            markup=True,
                        )
                        yield TextArea(
                            id="stg-remote-resources",
                            soft_wrap=False,
                            language="json",
                            placeholder=(
                                '[\n'
                                '  {\n'
                                '    "name": "gpu-box",\n'
                                '    "ssh_target": "user@gpu-host",\n'
                                '    "remote_workdir": "/home/user/vid_to_sub",\n'
                                '    "slots": 2,\n'
                                '    "path_map": {"/mnt/media": "/srv/media"},\n'
                                '    "env": {"VID_TO_SUB_WHISPER_CPP_MODEL": "/models/ggml-large-v3.bin"}\n'
                                "  }\n"
                                "]"
                            ),
                        )

                        yield Static("SSH Connections", classes="stitle")
                        yield Static(
                            "[dim]Define SSH servers for remote transcription. Each connection can be selected — in the Transcribe tab — as a remote executor.[/]",
                            classes="hint",
                            markup=True,
                        )
                        yield DataTable(
                            id="ssh-conn-table",
                            cursor_type="row",
                            zebra_stripes=True,
                        )
                        yield Static("", id="ssh-conn-hint", markup=True)

                        yield Static("SSH Connection Form", classes="stitle")
                        with Horizontal(classes="frow"):
                            yield Label("Label", classes="flabel")
                            yield Input(
                                id="ssh-label",
                                classes="fwidget",
                                placeholder="gpu-server (optional display name)",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Host", classes="flabel")
                            yield Input(
                                id="ssh-host",
                                classes="fwidget",
                                placeholder="192.168.1.100 or gpu-box.example.com",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("User", classes="flabel")
                            yield Input(
                                id="ssh-user",
                                classes="fwidget",
                                placeholder="ubuntu (blank = current user)",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Port", classes="flabel")
                            yield Input(
                                id="ssh-port",
                                classes="fwidget",
                                placeholder="22",
                                value="22",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Private key", classes="flabel")
                            yield Input(
                                id="ssh-key-path",
                                classes="fwidget",
                                placeholder="~/.ssh/id_rsa (blank = SSH agent)",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Remote workdir", classes="flabel")
                            yield Input(
                                id="ssh-remote-workdir",
                                classes="fwidget",
                                placeholder="/home/user/vid_to_sub",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Python bin", classes="flabel")
                            yield Input(
                                id="ssh-python-bin",
                                classes="fwidget",
                                placeholder="python3",
                                value="python3",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Script path", classes="flabel")
                            yield Input(
                                id="ssh-script-path",
                                classes="fwidget",
                                placeholder="(blank = remote_workdir/vid_to_sub.py)",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Slots", classes="flabel")
                            yield Input(
                                id="ssh-slots",
                                classes="fwidget",
                                placeholder="1",
                                value="1",
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Path map (JSON)", classes="flabel")
                            yield Input(
                                id="ssh-path-map",
                                classes="fwidget",
                                placeholder='{"local prefix": "remote prefix"} or blank',
                            )
                        with Horizontal(classes="frow"):
                            yield Label("Env overrides (JSON)", classes="flabel")
                            yield Input(
                                id="ssh-env-json",
                                classes="fwidget",
                                placeholder='{"VID_TO_SUB_WHISPER_CPP_MODEL": "/models/ggml-large-v3.bin"}',
                            )
                        yield Static("", id="ssh-form-status", markup=True)
                        with Horizontal(id="ssh-form-actions"):
                            yield Button("Add Connection", id="btn-ssh-add", variant="primary")
                            yield Button("Update Selected", id="btn-ssh-update", variant="default")
                            yield Button("Delete Selected", id="btn-ssh-delete", variant="warning")
                            yield Button("Clear Form", id="btn-ssh-clear", variant="default")
                        yield Static("", id="stg-status", markup=True)
                        with Horizontal(id="stg-actions"):
                            yield Button("Save", id="btn-stg-save", variant="primary")
                            yield Button("Reload", id="btn-stg-reload", variant="default")
                            yield Button("Export .env", id="btn-export-env", variant="default")
                            yield Button("Import from .env", id="btn-import-env", variant="default")

                # ── Tab 6: Agent ──────────────────────────────────────
                with TabPane("6 Agent", id="tab-agent"):
                    with ScrollableContainer(classes="tab-body"):
                        yield Static("Prompt", classes="stitle")
                        yield Static(
                            "[dim]Uses OpenAI-compatible chat completions. The agent can inspect live run status and recent history, then propose safe actions that only execute after you review and apply them.[/]",
                            id="agent-help",
                            markup=True,
                        )
                        yield Static("", id="agent-config", markup=True)
                        yield Static("", id="agent-live", markup=True)
                        yield TextArea(
                            id="agent-prompt",
                            soft_wrap=True,
                            placeholder=(
                                "Ask for setup help, run analysis, or history control, for example:\n"
                                "- Analyze the current run and elapsed time\n"
                                "- Load the last failed job into the form\n"
                                "- Rerun job 42 with the current settings after reviewing the plan"
                            ),
                        )
                        with Horizontal(id="agent-actions"):
                            yield Button("Ask", id="btn-agent-plan", variant="primary")
                            yield Button(
                                "Apply",
                                id="btn-agent-apply",
                                variant="success",
                                disabled=True,
                            )
                            yield Button("Clear", id="btn-agent-clear")
                        yield Static(
                            "[dim]No agent plan yet.[/]",
                            id="agent-status",
                            markup=True,
                        )
                        yield RichLog(
                            id="agent-log",
                            highlight=True,
                            markup=True,
                            auto_scroll=True,
                            wrap=True,
                            max_lines=4000,
                        )

        # ── Bottom panel (always visible) ─────────────────────────────────
        with Vertical(id="bottom"):
            with Horizontal(id="run-toolbar"):
                with Horizontal(id="run-btns"):
                    yield Button("Run", id="btn-run", variant="success")
                    yield Button("Dry", id="btn-dry-run", variant="warning")
                    yield Button("Kill", id="btn-kill", variant="error", disabled=True)
                    yield Button("Cmd ▸", id="btn-toggle-run-shell", variant="default")
                yield Static("", id="run-overview", markup=True)
            with Vertical(id="run-command-panel", classes="collapsed"):
                yield Static(
                    "[dim]Browse → select paths → Ctrl+R to run[/]",
                    id="run-shell",
                    markup=True,
                )
            yield Static(_progress_bar_markup(0, 0), id="run-progress", markup=True)
            with Vertical(id="run-active-box"):
                yield Static("[dim]Current jobs will appear here when a run starts.[/]", markup=True)
            yield RichLog(
                id="log",
                highlight=True,
                markup=True,
                auto_scroll=True,
                wrap=True,
                max_lines=5000,
            )

        yield Footer(compact=True, show_command_palette=False)

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
        self._init_ssh_table()
        self._refresh_ssh_table()
        # Navigate tree to configured root
        root = _db.get("tui.browse_root") or str(Path.home())
        try:
            self.query_one("#dir-tree", DirectoryTree).path = Path(root)
            self.query_one("#tree-root", Input).value = root
        except (NoMatches, Exception):
            pass
        self._update_wcpp_model_status()
        self._load_remote_resources()
        self._update_remote_status()
        self._refresh_live_panels()
        self._update_agent_config_status()
        self._sync_run_command_panel()
        self.set_interval(1.0, self._refresh_live_panels)

    def _sync_run_command_panel(self) -> None:
        try:
            panel = self.query_one("#run-command-panel")
            toggle = self.query_one("#btn-toggle-run-shell", Button)
        except NoMatches:
            return

        if self._run_shell_collapsed:
            panel.add_class("collapsed")
            toggle.label = "Cmd ▸"
        else:
            panel.remove_class("collapsed")
            toggle.label = "Cmd ▾"

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
        self._update_remote_status()
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
                    try:
                        if self.query_one("#sw-postprocess", Switch).value:
                            self.query_one("#post-fields").remove_class("hidden")
                    except NoMatches:
                        pass
                else:
                    flds.add_class("hidden")
                    try:
                        self.query_one("#post-fields").add_class("hidden")
                    except NoMatches:
                        pass
            except NoMatches:
                pass
        elif event.switch.id == "sw-postprocess":
            try:
                flds = self.query_one("#post-fields")
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
            self._refresh_live_panels()
        elif event.data_table.id == "ssh-conn-table":
            key = event.row_key
            conn_id_str = str(key.value) if key else None
            if conn_id_str:
                try:
                    conn_id = int(conn_id_str)
                    row = _db.get_ssh_connection(conn_id)
                    if row:
                        self._ssh_fill_form_from_row(row)
                        self._ssh_set_status(f"[dim]Loaded connection ID {conn_id}[/]")
                except (ValueError, TypeError):
                    pass

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
            run_busy = bool(self._active_worker and self._active_worker.is_running)
            rb.disabled = run_busy
            db.disabled = run_busy
            kb.disabled = run_busy
            if run_busy:
                self.sub_title = "Running…"
            else:
                if event.worker.name == "prepare-run":
                    self._reset_run_state()
                label = {
                    WorkerState.CANCELLED: "Cancelled",
                    WorkerState.ERROR: "Error",
                }.get(state, "Idle")
                self.sub_title = label
            # refresh history when a transcription worker finishes
            if event.worker.name == "_stream":
                self._refresh_history()
        self._refresh_live_panels()

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
                    self._set_search_preview(self._search_results[idx])
                    self._tree_selection = target
                    self._goto_tree(str(target if target.is_dir() else target.parent))
            except ValueError:
                pass
        elif bid.startswith("spv-"):
            try:
                idx = int(bid.removeprefix("spv-"))
                if 0 <= idx < len(self._search_results):
                    self._set_search_preview(self._search_results[idx])
            except ValueError:
                pass
        elif bid.startswith("sadd-"):
            try:
                idx = int(bid.removeprefix("sadd-"))
                if 0 <= idx < len(self._search_results):
                    self._set_search_preview(self._search_results[idx])
                    self._tree_selection = Path(self._search_results[idx])
                    self._add_path(self._search_results[idx])
            except ValueError:
                pass

        # ── Setup ──────────────────────────────────────────────
        elif bid == "btn-redetect":
            self._run_detection()
        elif bid == "btn-auto-setup":
            self._save_setup_build_fields()
            self._save_setup_model_fields()
            model = self._sel("sel-dl-model", DEFAULT_MODEL)
            self._auto_setup(False, model)
        elif bid == "btn-full-setup":
            self._save_setup_build_fields()
            self._save_setup_model_fields()
            model = self._sel("sel-dl-model", DEFAULT_MODEL)
            self._auto_setup(True, model)
        elif bid == "btn-build-whisper":
            self._save_setup_build_fields()
            self._build_whisper_cpp()
        elif bid == "btn-download-model":
            self._save_setup_model_fields()
            model = self._sel("sel-dl-model", "large-v3")
            self._download_model(model)
        elif bid == "btn-pip-fw":
            self._pip_install("faster-whisper")
        elif bid == "btn-pip-whisper":
            self._pip_install("whisper")
        elif bid == "btn-pip-wx":
            self._pip_install("whisperx")

        # ── Run panel ──────────────────────────────────────────
        elif bid == "btn-run":
            self.action_run()
        elif bid == "btn-dry-run":
            self.action_dry_run()
        elif bid == "btn-kill":
            self.action_kill()
        elif bid == "btn-toggle-run-shell":
            self._run_shell_collapsed = not self._run_shell_collapsed
            self._sync_run_command_panel()

        # ── History ────────────────────────────────────────────
        elif bid == "btn-hist-refresh":
            self._refresh_history()
            self._refresh_live_panels()
        elif bid == "btn-hist-load":
            if self._hist_key:
                self._load_history_job(int(self._hist_key))
        elif bid == "btn-hist-rerun":
            if self._hist_key:
                self._rerun_history_job(int(self._hist_key))
        elif bid == "btn-hist-clear":
            _db.clear_jobs()
            self._hist_key = None
            self._refresh_history()
            self._refresh_live_panels()
        elif bid == "btn-hist-delete":
            if self._hist_key:
                _db.delete_job(int(self._hist_key))
                self._hist_key = None
                self._refresh_history()
                self._refresh_live_panels()

        # ── Settings ───────────────────────────────────────────
        elif bid == "btn-stg-save":
            self._save_settings()
        elif bid == "btn-stg-reload":
            self._load_settings_form()
            self._load_remote_resources()
            self._update_remote_status()
            self._update_agent_config_status()
        elif bid == "btn-export-env":
            self._export_env()
        elif bid == "btn-import-env":
            self._import_env_to_sqlite()
        # ── SSH Connections ─────────────────────────────────────
        elif bid == "btn-ssh-add":
            self._ssh_add_connection()
        elif bid == "btn-ssh-update":
            self._ssh_update_connection()
        elif bid == "btn-ssh-delete":
            self._ssh_delete_connection()
        elif bid == "btn-ssh-clear":
            self._ssh_clear_form()
            self._request_agent_plan()
        elif bid == "btn-agent-apply":
            self._apply_agent_plan()
        elif bid == "btn-agent-clear":
            self._clear_agent_ui()

    # ── Actions ───────────────────────────────────────────────────────────

    def action_run(self) -> None:
        self._trigger(dry_run=False)

    def action_dry_run(self) -> None:
        self._trigger(dry_run=True)

    def action_kill(self) -> None:
        terminated = False
        for name, proc in list(self._procs.items()):
            if proc.poll() is None:
                proc.terminate()
                terminated = True
                self._log(f"[yellow]⚡ Terminated {name}.[/]")
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            terminated = True
        if terminated:
            self._refresh_live_panels()
        if self._active_worker and self._active_worker.is_running:
            self._run_request_id += 1
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

    # ── Live state ─────────────────────────────────────────────────────────

    def _reset_run_state(self, *, preserve_shell: bool = False) -> None:
        self._active_jobs.clear()
        self._pending_paths.clear()
        self._proc = None
        self._procs.clear()
        self._run_started_at = None
        self._run_total_found = 0
        self._run_total_queued = 0
        self._run_skipped = 0
        self._run_completed = 0
        self._run_failed = 0
        self._run_backend = DEFAULT_BACKEND
        self._run_model = DEFAULT_MODEL
        self._run_language = None
        self._run_target_lang = None
        self._run_output_dir = None
        self._run_dry_run = False
        if not preserve_shell:
            self._run_last_shell = "[dim]Browse → select paths → Ctrl+R to run[/]"

    def _selected_history_job(self) -> dict[str, Any] | None:
        if not self._hist_key:
            return None
        return next((job for job in _db.get_jobs() if str(job["id"]) == self._hist_key), None)

    def _refresh_live_panels(self) -> None:
        running = bool(self._active_worker and self._active_worker.is_running)
        elapsed = (
            time.monotonic() - self._run_started_at if self._run_started_at is not None else None
        )
        processed = self._run_completed + self._run_failed
        remaining = max(0, self._run_total_queued - processed)
        overview = (
            f"[bold]{'Running' if running else 'Idle'}[/]  "
            f"queued={self._run_total_queued or '-'}  "
            f"done={self._run_completed}  "
            f"failed={self._run_failed}  "
            f"left={remaining if self._run_total_queued else '-'}  "
            f"elapsed={_fmt_elapsed(elapsed)}"
        )
        try:
            self.query_one("#run-overview", Static).update(overview)
            self.query_one("#run-shell", Static).update(self._run_last_shell)
            self.query_one("#run-progress", Static).update(
                _progress_bar_markup(processed, self._run_total_queued)
            )
        except NoMatches:
            pass

        try:
            box = self.query_one("#run-active-box")
        except NoMatches:
            box = None
        if box is not None and box.is_attached:
            box.remove_children()
            if self._active_jobs:
                for job in sorted(self._active_jobs.values(), key=lambda item: item.started_at):
                    run_for = max(0.0, time.monotonic() - job.started_at)
                    label = (
                        f"[cyan]{job.executor}[/] "
                        f"{Path(job.video_path).name}  "
                        f"[dim]{_fmt_elapsed(run_for)}[/]"
                    )
                    if (
                        job.video_duration is not None
                        or job.progress_seconds is not None
                        or job.progress_ratio is not None
                    ):
                        progress_seconds = max(0.0, job.progress_seconds or 0.0)
                        progress_ratio = (
                            job.progress_ratio
                            if job.progress_ratio is not None
                            else (
                                progress_seconds / job.video_duration
                                if job.video_duration and job.video_duration > 0
                                else 0.0
                            )
                        )
                        progress_meta = f"[dim]{_fmt_elapsed(progress_seconds)}[/]"
                        if job.video_duration is not None:
                            progress_meta = (
                                f"[dim]{_fmt_elapsed(progress_seconds)} / "
                                f"{_fmt_elapsed(job.video_duration)}[/]"
                            )
                        label = (
                            label
                            + "\n"
                            + _progress_bar_markup_ratio(progress_ratio, width=20)
                            + f"  {progress_meta}"
                        )
                    box.mount(Static(label, classes="run-active-row", markup=True))
            elif running:
                box.mount(
                    Static(
                        "[dim]Preparing executors or waiting for the first file to start…[/]",
                        markup=True,
                    )
                )
            else:
                box.mount(
                    Static(
                        "[dim]Current jobs will appear here when a run starts.[/]",
                        markup=True,
                    )
                )

        selected = self._selected_history_job()
        live_lines = [
            f"[dim]Run:[/] {'active' if running else 'idle'}",
            f"[dim]Queued:[/] {self._run_total_queued or 0}   [dim]Processed:[/] {processed}",
            f"[dim]Elapsed:[/] {_fmt_elapsed(elapsed)}",
        ]
        if selected:
            live_lines.append(
                f"[dim]Selected history row:[/] #{selected['id']} {Path(selected['video_path']).name}"
            )
        try:
            self.query_one("#agent-live", Static).update("   ".join(live_lines))
        except NoMatches:
            pass

    def _load_remote_resources(self) -> None:
        """Load remote resources from DB.

        Primary: ssh_connections table (structured, UI-managed).
        Fallback/supplement: tui.remote_resources legacy JSON field.
        Both sources are merged; DB connections take priority.
        """
        profiles: list[RemoteResourceProfile] = []

        # 1. Structured SSH connections from DB (primary source)
        for row in _db.get_ssh_connections(enabled_only=True):
            conn = ssh_connection_from_row(row)
            profiles.append(conn.to_remote_resource_profile())

        # 2. Legacy raw JSON (tui.remote_resources) for backward compat
        raw = _db.get("tui.remote_resources")
        if raw and raw.strip() not in ("", "[]"):
            try:
                legacy = parse_remote_resources(raw)
                # Only add profiles whose name isn't already taken by a DB connection
                existing_names = {p.name for p in profiles}
                for lp in legacy:
                    if lp.name not in existing_names:
                        profiles.append(lp)
            except ValueError as exc:
                try:
                    self.query_one("#stg-status", Static).update(f"[yellow]Legacy remote JSON warning: {exc}[/]")
                except NoMatches:
                    pass
            self._ssh_set_status("[yellow]Select a connection from the table first[/]")
            return
        _db.delete_ssh_connection(self._ssh_selected_id)
        self._ssh_selected_id = None
        self._ssh_clear_form()
        self._refresh_ssh_table()
        self._load_remote_resources()
        self._update_remote_status()
        self._ssh_set_status("[green]✓ Connection deleted[/]")


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
