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
    PIP_REQUIREMENT_FILES,
    RemoteResourceProfile,
    RunConfig,
    RunJobState,
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
)
from .state import db as _db
from .styles import _CSS

SCRIPT_DIR = ROOT_DIR
SCRIPT_PATH = ROOT_DIR / "vid_to_sub.py"

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

                        yield Static("", id="stg-status", markup=True)
                        with Horizontal(id="stg-actions"):
                            yield Button("Save", id="btn-stg-save", variant="primary")
                            yield Button("Reload", id="btn-stg-reload", variant="default")
                            yield Button("Export .env", id="btn-export-env", variant="default")

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
        elif bid == "btn-agent-plan":
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
        raw = _db.get("tui.remote_resources")
        try:
            self._remote_resources = parse_remote_resources(raw)
        except ValueError as exc:
            self._remote_resources = []
            try:
                self.query_one("#stg-status", Static).update(f"[red]✗ {exc}[/]")
            except NoMatches:
                pass

    def _update_remote_status(self) -> None:
        mode = self._sel("sel-execution-mode", _db.get("tui.execution_mode") or "local")
        summary = summarize_remote_resources(self._remote_resources)
        if mode == "distributed":
            if self._remote_resources:
                status = f"[green]Distributed mode[/] · {summary}"
            else:
                status = (
                    "[yellow]Distributed mode selected, but no valid remote resources are configured. "
                    "Run will fall back to local execution.[/]"
                )
        else:
            status = f"[dim]Local mode.[/] {summary}"
        try:
            self.query_one("#remote-status", Static).update(status)
        except NoMatches:
            pass

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
            row.mount(Button("X", id=f"selrm-{i}", variant="error", classes="sel-btn"))

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
                Button("+", id=f"radd-{i}", classes="recent-add", variant="default")
            )

    def _clear_path_search(self, status: str) -> None:
        self._search_results = []
        self._search_preview_path = None
        try:
            self.query_one("#search-status", Static).update(status)
        except NoMatches:
            pass
        self._refresh_search_results()
        self._refresh_search_preview()

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
            row.mount(
                Button("View", id=f"spv-{i}", classes="search-preview-btn")
            )
            row.mount(Static(label, classes="search-label", markup=False))
            row.mount(
                Button("+", id=f"sadd-{i}", classes="search-add", variant="default")
            )

    def _refresh_search_preview(self) -> None:
        try:
            preview = self.query_one("#search-preview", Static)
        except NoMatches:
            return
        if not self._search_preview_path:
            preview.update("Search result preview will appear here.")
            return
        preview.update(build_search_preview(Path(self._search_preview_path)))

    def _set_search_preview(self, path: str | None) -> None:
        self._search_preview_path = path
        self._refresh_search_preview()

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
        self._search_preview_path = results[0] if results else None
        try:
            self.query_one("#search-status", Static).update(status)
        except NoMatches:
            pass
        self._refresh_search_results()
        self._refresh_search_preview()

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

    def _capture_detection_state(self) -> DetectResult:
        self._detected_ggml_models = discover_ggml_models()
        results = detect_all()
        self._detect_results = results
        return results

    def _refresh_detection_from_worker(self) -> DetectResult:
        results = self._capture_detection_state()
        self.call_from_thread(self._update_detect_ui, results)
        return results

    @work(thread=True, exclusive=False, exit_on_error=False)
    def _run_detection(self) -> None:
        self._refresh_detection_from_worker()

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

    def _set_setup_status(self, message: str) -> None:
        try:
            self.query_one("#auto-setup-status", Static).update(message)
        except NoMatches:
            pass

    def _system_install_tools(self, tools: Sequence[str], reason: str) -> bool:
        log = lambda m: self.call_from_thread(self._setup_log, m)
        manager = detect_package_manager()
        wanted = list(dict.fromkeys(tools))

        if not manager:
            log("[yellow]⚠ No supported package manager detected for auto-install.[/]")
            self.call_from_thread(
                self._set_setup_status,
                "[yellow]Auto-install skipped: no supported package manager detected.[/]",
            )
            return False

        use_sudo = manager != "brew" and hasattr(os, "geteuid") and os.geteuid() != 0
        if use_sudo and not shutil.which("sudo"):
            log("[red]✗ sudo is not available, so system package auto-install cannot run.[/]")
            self.call_from_thread(
                self._set_setup_status,
                "[red]Auto-install failed: sudo is required for system packages.[/]",
            )
            return False

        commands = build_system_install_commands(manager, wanted, use_sudo=use_sudo)
        if not commands:
            log(
                f"[yellow]⚠ No package mapping for {', '.join(wanted)} on {manager}. Skipping auto-install.[/]"
            )
            return False

        log(
            f"[cyan]Auto-installing system packages via {manager} for {reason}: {', '.join(wanted)}[/]"
        )
        if use_sudo:
            log("[dim]Using sudo -n; password prompts are not supported in this TUI worker.[/]")
        self.call_from_thread(
            self._set_setup_status,
            f"[cyan]Auto-installing system packages via {manager}…[/]",
        )

        for cmd in commands:
            if not self._run_cmd(cmd):
                log("[red]✗ System package auto-install failed.[/]")
                self.call_from_thread(
                    self._set_setup_status,
                    "[red]System package auto-install failed. Check setup log.[/]",
                )
                self._refresh_detection_from_worker()
                return False

        log("[green]✅ System package auto-install completed.[/]")
        self._refresh_detection_from_worker()
        return True

    def _install_requirement_target(self, target: str) -> bool:
        log = lambda m: self.call_from_thread(self._setup_log, m)
        spec = PIP_REQUIREMENT_FILES.get(target)
        if not spec:
            log(f"[red]✗ Unknown pip install target: {target}[/]")
            return False
        filename, label = spec
        req_path = SCRIPT_DIR / filename
        if not req_path.exists():
            log(f"[red]✗ Requirement file not found: {req_path}[/]")
            return False
        self.call_from_thread(self._set_setup_status, f"[cyan]Installing {label}…[/]")
        log(f"[cyan]Installing {label}…[/]")
        ok = self._run_cmd([sys.executable, "-m", "pip", "install", "-r", str(req_path)])
        if ok:
            log(f"[green]✅ Installed {label}[/]")
        self._refresh_detection_from_worker()
        return ok

    def _build_whisper_cpp_sync(self) -> bool:
        log = lambda m: self.call_from_thread(self._setup_log, m)

        build_dir = _db.get("tui.build_dir") or str(
            Path.home() / ".cache/vid_to_sub_build"
        )
        install_dir = _db.get("tui.install_dir") or str(Path.home() / ".local/bin")
        build_root = Path(build_dir).expanduser().resolve()
        install_d = Path(install_dir).expanduser().resolve()
        repo_dir = build_root / "whisper.cpp"

        missing_tools: list[str] = []
        if not shutil.which("git"):
            missing_tools.append("git")
        if not shutil.which("cmake"):
            missing_tools.append("cmake")
        if missing_tools:
            log(
                f"[yellow]⚠ Missing build tools: {', '.join(missing_tools)} — attempting auto-install…[/]"
            )
            self._system_install_tools(
                [*missing_tools, "whisper-build"],
                "whisper.cpp build prerequisites",
            )

        if not shutil.which("git"):
            log("[red]✗ git not found in PATH — needed for cloning[/]")
            return False
        if not shutil.which("cmake"):
            log("[red]✗ cmake not found in PATH — needed for building[/]")
            return False

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
                return False
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
            return False

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
            return False

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
            return False

        install_d.mkdir(parents=True, exist_ok=True)
        dest = install_d / "whisper-cli"
        try:
            shutil.copy2(str(binary), str(dest))
            os.chmod(str(dest), 0o755)
        except PermissionError:
            log(f"[red]✗ Permission denied writing to {dest}[/]")
            log(f"[yellow]  Tip: set Install dir to ~/.local/bin[/]")
            return False

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
        self.call_from_thread(
            self._set_setup_status,
            f"[green]whisper-cli installed → {dest}[/]",
        )
        self._refresh_detection_from_worker()
        return True

    @work(thread=True, exclusive=False, exit_on_error=False, name="build-whisper")
    def _build_whisper_cpp(self) -> None:
        self._build_whisper_cpp_sync()

    def _download_model_sync(self, model_name: str) -> bool:
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
            self.call_from_thread(
                self._set_setup_status,
                f"[green]{filename} already available.[/]",
            )
            self._refresh_detection_from_worker()
            return True

        url = f"{HF_MODEL_BASE}/{filename}"
        log(f"[cyan]Downloading {filename}…[/]")
        log(f"[dim]  {url}[/]")
        self.call_from_thread(
            self._set_setup_status,
            f"[cyan]Downloading {filename}…[/]",
        )

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
            self.call_from_thread(
                self._set_setup_status,
                f"[red]Download failed for {filename}. Check setup log.[/]",
            )
            return False
        except OSError as e:
            log(f"[red]✗ Write error: {e}[/]")
            if dest.exists():
                dest.unlink()
            self.call_from_thread(
                self._set_setup_status,
                f"[red]Could not write {filename}. Check setup log.[/]",
            )
            return False

        size = dest.stat().st_size / 1024 / 1024
        _db.set(ENV_WCPP_MODEL, str(dest))
        os.environ[ENV_WCPP_MODEL] = str(dest)
        log(f"[green]✅ {filename} saved → {dest} ({size:.0f} MB)[/]")
        self.call_from_thread(
            self._set_setup_status,
            f"[green]{filename} saved → {dest}[/]",
        )
        self._refresh_detection_from_worker()
        return True

    @work(thread=True, exclusive=False, exit_on_error=False, name="download-model")
    def _download_model(self, model_name: str) -> None:
        self._download_model_sync(model_name)

    @work(thread=True, exclusive=False, exit_on_error=False, name="pip-install")
    def _pip_install(self, package: str) -> None:
        if package in PIP_REQUIREMENT_FILES:
            self._install_requirement_target(package)
            return
        self.call_from_thread(self._setup_log, f"[cyan]Installing {package}…[/]")
        ok = self._run_cmd([sys.executable, "-m", "pip", "install", package])
        if ok:
            self.call_from_thread(self._setup_log, f"[green]✅ {package} installed[/]")
        self._refresh_detection_from_worker()

    def _auto_setup_sync(self, full: bool, model_name: str) -> None:
        log = lambda m: self.call_from_thread(self._setup_log, m)
        mode_label = "full install" if full else "auto setup"
        self.call_from_thread(
            self._set_setup_status,
            f"[cyan]Running {mode_label}…[/]",
        )
        log(f"[bold cyan]Starting {mode_label}[/]")
        self._refresh_detection_from_worker()

        essential_tools = ["ffmpeg", "git", "cmake", "whisper-build"]
        self._system_install_tools(essential_tools, "default whisper.cpp pipeline")
        self._install_requirement_target("base")

        if full:
            for target in ("faster-whisper", "whisper", "whisperx"):
                self._install_requirement_target(target)

        self._build_whisper_cpp_sync()
        self._download_model_sync(model_name)
        results = self._refresh_detection_from_worker()

        missing = [
            name
            for name in ("ffmpeg", "whisper-cli", "ggml-model")
            if not results.get(name, (False, ""))[0]
        ]
        if missing:
            summary = f"[yellow]{mode_label.title()} incomplete: missing {', '.join(missing)}[/]"
            log(summary)
            self.call_from_thread(self._set_setup_status, summary)
        else:
            summary = f"[green]✅ {mode_label.title()} complete[/]"
            log(summary)
            self.call_from_thread(self._set_setup_status, summary)

    @work(thread=True, exclusive=False, exit_on_error=False, name="auto-setup")
    def _auto_setup(self, full: bool, model_name: str) -> None:
        self._auto_setup_sync(full, model_name)

    def _effective_agent_config(self) -> tuple[str, str, str]:
        model = (
            _db.get(ENV_AGENT_MOD)
            or _db.get(ENV_TRANS_MOD)
            or os.environ.get(ENV_AGENT_MOD, "")
            or os.environ.get(ENV_TRANS_MOD, "")
        )
        base_url = (
            _db.get(ENV_AGENT_URL)
            or _db.get(ENV_TRANS_URL)
            or os.environ.get(ENV_AGENT_URL, "")
            or os.environ.get(ENV_TRANS_URL, "")
        )
        api_key = (
            _db.get(ENV_AGENT_KEY)
            or _db.get(ENV_TRANS_KEY)
            or os.environ.get(ENV_AGENT_KEY, "")
            or os.environ.get(ENV_TRANS_KEY, "")
        )
        return model, base_url, api_key

    def _update_agent_config_status(self) -> None:
        model, base_url, api_key = self._effective_agent_config()
        status = (
            f"[dim]Effective agent model:[/] {model or '(missing)'}   "
            f"[dim]base URL:[/] {base_url or '(missing)'}   "
            f"[dim]API key:[/] {'set' if api_key else 'missing'}"
        )
        try:
            self.query_one("#agent-config", Static).update(status)
        except NoMatches:
            pass

    def _set_agent_status(self, message: str) -> None:
        try:
            self.query_one("#agent-status", Static).update(message)
        except NoMatches:
            pass

    def _agent_log_write(self, message: str) -> None:
        try:
            self.query_one("#agent-log", RichLog).write(message)
        except NoMatches:
            pass

    def _update_agent_apply_state(self) -> None:
        try:
            button = self.query_one("#btn-agent-apply", Button)
            button.disabled = not bool(self._agent_plan and self._agent_plan.get("actions"))
        except NoMatches:
            pass

    def _clear_agent_ui(self) -> None:
        self._agent_plan = None
        try:
            self.query_one("#agent-prompt", TextArea).text = ""
            self.query_one("#agent-log", RichLog).clear()
        except NoMatches:
            pass
        self._set_agent_status("[dim]No agent plan yet.[/]")
        self._update_agent_apply_state()
        self._update_agent_config_status()

    def _normalize_agent_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary = str(payload.get("summary") or "No summary provided.").strip()
        analysis = str(payload.get("analysis") or "").strip()
        normalized_actions: list[dict[str, Any]] = []
        raw_actions = payload.get("actions")
        known_job_ids = {int(job["id"]) for job in _db.get_jobs()}

        if isinstance(raw_actions, list):
            for raw_action in raw_actions:
                if not isinstance(raw_action, dict):
                    continue
                action_type = str(raw_action.get("type") or "").strip()
                if action_type == "run_detection":
                    normalized_actions.append({"type": "run_detection"})
                elif action_type == "auto_setup":
                    mode = str(raw_action.get("mode") or "essential").strip().lower()
                    model = str(raw_action.get("model") or DEFAULT_MODEL).strip()
                    normalized_actions.append(
                        {
                            "type": "auto_setup",
                            "mode": "full" if mode == "full" else "essential",
                            "model": model if model in KNOWN_MODELS else DEFAULT_MODEL,
                        }
                    )
                elif action_type == "build_whisper_cpp":
                    normalized_actions.append({"type": "build_whisper_cpp"})
                elif action_type == "download_model":
                    model = str(raw_action.get("model") or DEFAULT_MODEL).strip()
                    normalized_actions.append(
                        {
                            "type": "download_model",
                            "model": model if model in KNOWN_MODELS else DEFAULT_MODEL,
                        }
                    )
                elif action_type == "pip_install":
                    target = str(raw_action.get("target") or "").strip()
                    if target in PIP_REQUIREMENT_FILES:
                        normalized_actions.append(
                            {"type": "pip_install", "target": target}
                        )
                elif action_type in {
                    "refresh_history",
                    "kill_active_run",
                }:
                    normalized_actions.append({"type": action_type})
                elif action_type in {
                    "load_history_job",
                    "rerun_history_job",
                    "delete_history_job",
                }:
                    try:
                        job_id = int(raw_action.get("job_id"))
                    except (TypeError, ValueError):
                        continue
                    if job_id in known_job_ids:
                        normalized_actions.append(
                            {"type": action_type, "job_id": job_id}
                        )

        return {
            "summary": summary,
            "analysis": analysis,
            "actions": normalized_actions,
        }

    def _render_agent_plan(self, plan: dict[str, Any]) -> None:
        try:
            log = self.query_one("#agent-log", RichLog)
            log.clear()
        except NoMatches:
            log = None

        if log is not None:
            log.write(f"[bold]Summary[/] {plan['summary']}")
            if plan.get("analysis"):
                log.write(f"\n[bold]Analysis[/] {plan['analysis']}")
            actions = plan.get("actions") or []
            if actions:
                log.write("\n[bold]Actions[/]")
                for idx, action in enumerate(actions, start=1):
                    line = f"{idx}. {action['type']}"
                    if action["type"] == "auto_setup":
                        line += f" mode={action['mode']} model={action['model']}"
                    elif action["type"] == "download_model":
                        line += f" model={action['model']}"
                    elif action["type"] == "pip_install":
                        line += f" target={action['target']}"
                    elif action["type"].endswith("_history_job"):
                        line += f" job_id={action['job_id']}"
                    log.write(line)
            else:
                log.write("\n[yellow]No executable actions returned.[/]")

        if plan.get("actions"):
            self._set_agent_status("[green]Agent plan ready. Review and apply when ready.[/]")
        else:
            self._set_agent_status("[yellow]Agent returned guidance only.[/]")
        self._update_agent_apply_state()

    def _agent_context(self) -> dict[str, Any]:
        detections = {
            name: {"found": found, "detail": detail}
            for name, (found, detail) in self._detect_results.items()
        }
        recent_jobs = [
            {
                "id": job["id"],
                "created_at": job["created_at"],
                "video_path": job["video_path"],
                "backend": job["backend"],
                "model": job["model"],
                "status": job["status"],
                "language": job.get("language"),
                "target_lang": job.get("target_lang"),
                "error": job.get("error"),
            }
            for job in _db.get_jobs(limit=20)
        ]
        active_run = {
            "is_running": bool(self._active_worker and self._active_worker.is_running),
            "started_at": self._run_started_at,
            "elapsed_sec": (
                round(time.monotonic() - self._run_started_at, 3)
                if self._run_started_at is not None
                else None
            ),
            "total_found": self._run_total_found,
            "total_queued": self._run_total_queued,
            "completed": self._run_completed,
            "failed": self._run_failed,
            "active_jobs": [
                {
                    "executor": job.executor,
                    "video_path": job.video_path,
                    "started_at": job.started_at,
                    "run_for_sec": round(time.monotonic() - job.started_at, 3),
                }
                for job in self._active_jobs.values()
            ],
        }
        selected_job = self._selected_history_job()
        return {
            "detections": detections,
            "active_run": active_run,
            "selected_history_job": (
                {
                    "id": selected_job["id"],
                    "video_path": selected_job["video_path"],
                    "status": selected_job["status"],
                }
                if selected_job
                else None
            ),
            "recent_jobs": recent_jobs,
            "execution_mode": self._sel(
                "sel-execution-mode", _db.get("tui.execution_mode") or "local"
            ),
            "remote_resources": [profile.name for profile in self._remote_resources],
            "selected_download_model": self._sel("sel-dl-model", DEFAULT_MODEL),
            "selected_backend": self._sel("sel-backend", DEFAULT_BACKEND),
            "selected_model": self._sel("sel-model", DEFAULT_MODEL),
            "known_models": KNOWN_MODELS,
            "allowed_actions": [
                {"type": "run_detection"},
                {"type": "auto_setup", "mode": "essential|full", "model": "KNOWN_MODELS"},
                {"type": "build_whisper_cpp"},
                {"type": "download_model", "model": "KNOWN_MODELS"},
                {
                    "type": "pip_install",
                    "target": list(PIP_REQUIREMENT_FILES),
                },
                {"type": "refresh_history"},
                {"type": "load_history_job", "job_id": "recent_jobs[].id"},
                {"type": "rerun_history_job", "job_id": "recent_jobs[].id"},
                {"type": "delete_history_job", "job_id": "recent_jobs[].id"},
                {"type": "kill_active_run"},
            ],
        }

    def _request_agent_plan(self) -> None:
        try:
            prompt = self.query_one("#agent-prompt", TextArea).text.strip()
            self.query_one("#agent-log", RichLog).clear()
        except NoMatches:
            return
        if not prompt:
            self._set_agent_status("[red]Enter a request for the agent first.[/]")
            return
        self._agent_plan = None
        self._update_agent_apply_state()
        self._set_agent_status("[cyan]Requesting agent plan…[/]")
        self._update_agent_config_status()
        self._fetch_agent_plan(prompt)

    @work(thread=True, exclusive=True, exit_on_error=False, name="agent-plan")
    def _fetch_agent_plan(self, prompt: str) -> None:
        log = lambda m: self.call_from_thread(self._agent_log_write, m)
        model, base_url, api_key = self._effective_agent_config()
        if not model or not base_url or not api_key:
            self.call_from_thread(
                self._set_agent_status,
                "[red]Agent settings are incomplete. Configure agent or translation API settings first.[/]",
            )
            return

        endpoint = normalize_chat_endpoint(base_url)
        context = self._agent_context()
        payload = {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a bounded operations agent for the vid_to_sub TUI. "
                        "Return JSON only with keys summary, analysis, actions. "
                        "Actions must be an array using only these types: "
                        "run_detection, auto_setup, build_whisper_cpp, download_model, pip_install, "
                        "refresh_history, load_history_job, rerun_history_job, delete_history_job, kill_active_run. "
                        "Never invent shell commands or unsupported action types. "
                        "Prefer empty actions when the request is informational. "
                        "History-destructive or run-control actions must be conservative and should only be proposed "
                        "when clearly asked for or strongly justified by the current context."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"request": prompt, "context": context},
                        ensure_ascii=False,
                    ),
                },
            ],
        }

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "vid_to_sub/agent",
            },
        )

        log(f"[dim]POST {endpoint}[/]")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self.call_from_thread(
                self._set_agent_status,
                f"[red]Agent API HTTP {exc.code}. Check agent log.[/]",
            )
            log(f"[red]HTTP {exc.code}[/] {body}")
            return
        except urllib.error.URLError as exc:
            self.call_from_thread(
                self._set_agent_status,
                f"[red]Agent API request failed: {exc.reason}[/]",
            )
            return
        except json.JSONDecodeError as exc:
            self.call_from_thread(
                self._set_agent_status,
                "[red]Agent API returned invalid JSON.[/]",
            )
            log(f"[red]JSON decode failed[/] {exc}")
            return

        try:
            message = response_payload["choices"][0]["message"]["content"]
            if isinstance(message, list):
                message = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in message
                )
            plan = self._normalize_agent_plan(extract_json_payload(str(message)))
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.call_from_thread(
                self._set_agent_status,
                "[red]Could not parse the agent response as a valid plan. Check agent log.[/]",
            )
            log(f"[red]Raw response[/] {response_payload}")
            log(f"[red]Parse error[/] {exc}")
            return

        self._agent_plan = plan
        self.call_from_thread(self._render_agent_plan, plan)

    def _apply_agent_plan(self) -> None:
        if not self._agent_plan or not self._agent_plan.get("actions"):
            self._set_agent_status("[yellow]There is no executable agent plan to apply.[/]")
            return
        self._set_agent_status("[cyan]Applying agent plan…[/]")
        self._execute_agent_plan(list(self._agent_plan["actions"]))

    @work(thread=True, exclusive=True, exit_on_error=False, name="agent-apply")
    def _execute_agent_plan(self, actions: list[dict[str, Any]]) -> None:
        log = lambda m: self.call_from_thread(self._agent_log_write, m)
        for idx, action in enumerate(actions, start=1):
            action_type = action["type"]
            log(f"[bold cyan]Step {idx}/{len(actions)}[/] {action_type}")
            if action_type == "run_detection":
                self._refresh_detection_from_worker()
            elif action_type == "auto_setup":
                self._auto_setup_sync(
                    action.get("mode") == "full",
                    str(action.get("model") or DEFAULT_MODEL),
                )
            elif action_type == "build_whisper_cpp":
                self._build_whisper_cpp_sync()
            elif action_type == "download_model":
                self._download_model_sync(str(action.get("model") or DEFAULT_MODEL))
            elif action_type == "pip_install":
                target = str(action.get("target") or "")
                if target in PIP_REQUIREMENT_FILES:
                    self._install_requirement_target(target)
            elif action_type == "refresh_history":
                self.call_from_thread(self._refresh_history)
            elif action_type == "load_history_job":
                job_id = int(action["job_id"])
                self.call_from_thread(self._load_history_job, job_id)
            elif action_type == "rerun_history_job":
                job_id = int(action["job_id"])
                self.call_from_thread(self._rerun_history_job, job_id)
            elif action_type == "delete_history_job":
                job_id = int(action["job_id"])
                _db.delete_job(job_id)
                self.call_from_thread(self._refresh_history)
            elif action_type == "kill_active_run":
                self.call_from_thread(self.action_kill)

        self._refresh_detection_from_worker()
        self.call_from_thread(
            self._set_agent_status,
            "[green]Agent plan applied. Review setup and logs.[/]",
        )

    # ── Command builder ───────────────────────────────────────────────────

    def _build_cli_args(
        self,
        paths: Sequence[str] | None,
        *,
        dry_run: bool = False,
        remote_profile: RemoteResourceProfile | None = None,
        config: RunConfig | None = None,
        manifest_stdin: bool = False,
    ) -> list[str]:
        if not paths and not manifest_stdin:
            raise ValueError("No input paths — use Browse tab to select")

        map_path = (
            (lambda raw: map_path_for_remote(raw, remote_profile.path_map))
            if remote_profile
            else (lambda raw: raw)
        )
        cmd: list[str] = [map_path(path) for path in (paths or [])]
        if manifest_stdin:
            cmd.append("--manifest-stdin")

        output_dir = config.output_dir if config else (self._val("inp-output-dir") or None)
        if output_dir:
            v = output_dir
            cmd += ["--output-dir", map_path(v)]

        no_recurse = config.no_recurse if config else self._chk("chk-no-recurse")
        if no_recurse:
            cmd.append("--no-recurse")
        skip_existing = config.skip_existing if config else self._chk("chk-skip-existing")
        if skip_existing:
            cmd.append("--skip-existing")
        if dry_run:
            cmd.append("--dry-run")
        verbose = config.verbose if config else self._chk("chk-verbose")
        if verbose:
            cmd.append("--verbose")

        formats = config.formats if config else self._selected_formats()
        if "all" in formats:
            cmd += ["--format", "all"]
        else:
            fmts = [fmt for fmt in FORMATS if fmt in formats]
            if not fmts:
                fmts = ["srt"]
            for fmt in fmts:
                cmd += ["--format", fmt]

        backend = config.backend if config else self._sel("sel-backend", DEFAULT_BACKEND)
        model = config.model if config else self._sel("sel-model", DEFAULT_MODEL)
        cmd += ["--backend", backend]
        cmd += ["--model", model]
        device = config.device if config else self._sel("sel-device", DEFAULT_DEVICE)
        cmd += ["--device", device]

        language = config.language if config else (self._val("inp-language") or None)
        if language:
            v = language
            cmd += ["--language", v]
        compute_type = config.compute_type if config else (self._val("inp-compute-type") or None)
        if compute_type:
            v = compute_type
            cmd += ["--compute-type", v]

        beam = config.beam_size if config else (self._val("inp-beam-size") or "5")
        if beam != "5":
            cmd += ["--beam-size", beam]

        worker_count = (
            str(remote_profile.slots)
            if remote_profile is not None
            else (
                str(config.local_workers)
                if config is not None
                else (self._val("inp-workers") or "1")
            )
        )
        if worker_count != "1":
            cmd += ["--workers", worker_count]
        cmd += ["--backend-threads", "2"]

        whisper_cpp_model_path = (
            config.whisper_cpp_model_path
            if config is not None
            else self._resolved_wcpp_model_path()
        )
        if backend == "whisper-cpp" and whisper_cpp_model_path:
            v = whisper_cpp_model_path
            cmd += ["--whisper-cpp-model-path", map_path(v)]

        translate_enabled = (
            config.translate_enabled if config is not None else self._sw("sw-translate")
        )
        if translate_enabled:
            translate_to = (
                config.translate_to
                if config is not None
                else (self._val("inp-translate-to") or None)
            )
            if translate_to:
                v = translate_to
                cmd += ["--translate-to", v]
            translation_model = (
                config.translation_model
                if config is not None
                else (self._val("inp-trans-model") or None)
            )
            if translation_model:
                v = translation_model
                cmd += ["--translation-model", v]
            translation_base_url = (
                config.translation_base_url
                if config is not None
                else (self._val("inp-trans-url") or None)
            )
            if translation_base_url:
                v = translation_base_url
                cmd += ["--translation-base-url", v]
            translation_api_key = (
                config.translation_api_key
                if config is not None
                else (self._val("inp-trans-key") or None)
            )
            if translation_api_key:
                v = translation_api_key
                cmd += ["--translation-api-key", v]

        diarize = config.diarize if config is not None else self._sw("sw-diarize")
        if diarize:
            cmd.append("--diarize")
        hf_token = config.hf_token if config is not None else (self._val("inp-hf-token") or None)
        if hf_token:
            v = hf_token
            cmd += ["--hf-token", v]

        return cmd

    def _build_cmd(self, dry_run: bool = False) -> list[str]:
        return [
            sys.executable,
            str(SCRIPT_PATH),
            *self._build_cli_args(self._selected_paths, dry_run=dry_run),
        ]

    def _update_cmd_preview(self) -> None:
        try:
            if self._active_worker and self._active_worker.is_running:
                return
            mode = self._sel("sel-execution-mode", _db.get("tui.execution_mode") or "local")
            cmd = self._build_cmd()
            display = " ".join(_mask(cmd[2:]))
            if mode == "distributed" and self._remote_resources:
                self._run_last_shell = (
                    f"[dim]Distributed[/] [cyan]local + {len(self._remote_resources)} remote[/] · "
                    f"{display}"
                )
            else:
                self._run_last_shell = f"[dim]$[/dim] [cyan]vid_to_sub[/cyan] {display}"
            self._refresh_live_panels()
        except ValueError as exc:
            self._run_last_shell = f"[dim]{exc}[/]"
            self._refresh_live_panels()
        except Exception:
            pass

    # ── Run & Kill ────────────────────────────────────────────────────────

    def _selected_formats(self) -> frozenset[str]:
        if self._chk("fmt-all"):
            return frozenset({"all"})
        selected = [fmt for fmt in FORMATS if self._chk(f"fmt-{fmt}")]
        return frozenset(selected or ["srt"])

    def _snapshot_run_config(self, dry_run: bool) -> RunConfig:
        if not self._selected_paths:
            raise ValueError("No input paths — use Browse tab to select")

        self._run_request_id += 1
        translate_enabled = self._sw("sw-translate")
        return RunConfig(
            request_id=self._run_request_id,
            selected_paths=list(self._selected_paths),
            output_dir=self._val("inp-output-dir") or None,
            formats=self._selected_formats(),
            no_recurse=self._chk("chk-no-recurse"),
            skip_existing=self._chk("chk-skip-existing"),
            dry_run=dry_run,
            verbose=self._chk("chk-verbose"),
            backend=self._sel("sel-backend", DEFAULT_BACKEND),
            model=self._sel("sel-model", DEFAULT_MODEL),
            device=self._sel("sel-device", DEFAULT_DEVICE),
            language=self._val("inp-language") or None,
            compute_type=self._val("inp-compute-type") or None,
            beam_size=self._val("inp-beam-size") or "5",
            local_workers=_coerce_positive_int(self._val("inp-workers") or "1", default=1),
            whisper_cpp_model_path=self._resolved_wcpp_model_path() or None,
            translate_enabled=translate_enabled,
            translate_to=self._val("inp-translate-to") or None,
            translation_model=self._val("inp-trans-model") or None,
            translation_base_url=self._val("inp-trans-url") or None,
            translation_api_key=self._val("inp-trans-key") or None,
            diarize=self._sw("sw-diarize"),
            hf_token=self._val("inp-hf-token") or None,
            execution_mode=self._sel(
                "sel-execution-mode",
                _db.get("tui.execution_mode") or "local",
            ),
            remote_resources=list(self._remote_resources),
            run_env=self._build_run_env(),
        )

    def _discover_videos_for_run(
        self,
        config: RunConfig | None = None,
    ) -> tuple[list[str], int, int]:
        selected_paths = config.selected_paths if config else self._selected_paths
        if not selected_paths:
            raise ValueError("No input paths — use Browse tab to select")

        videos = discover_videos(
            selected_paths,
            recursive=not (config.no_recurse if config else self._chk("chk-no-recurse")),
        )
        found_total = len(videos)
        skipped = 0

        if config:
            output_dir = Path(config.output_dir).resolve() if config.output_dir else None
            formats = config.formats
            skip_existing = config.skip_existing
        else:
            output_dir = (
                Path(self._val("inp-output-dir")).resolve()
                if self._val("inp-output-dir")
                else None
            )
            formats = self._selected_formats()
            skip_existing = self._chk("chk-skip-existing")

        if skip_existing:
            filtered = [
                video
                for video in videos
                if not _primary_output_exists(video, formats, output_dir)
            ]
            skipped = found_total - len(filtered)
            videos = filtered

        return [str(video) for video in videos], found_total, skipped

    def _build_run_env(self, config: RunConfig | None = None) -> dict[str, str]:
        if config is not None:
            return dict(config.run_env)

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

    def _build_remote_command(
        self,
        profile: RemoteResourceProfile,
        paths: Sequence[str] | None,
        *,
        dry_run: bool,
        config: RunConfig | None = None,
    ) -> list[str]:
        base_env = {
            key: value
            for key, value in self._build_run_env(config).items()
            if key.startswith("VID_TO_SUB_") and value
        }
        mapped_env: dict[str, str] = {}
        for key, value in base_env.items():
            if value.startswith("/") or value.startswith("~"):
                mapped_env[key] = map_path_for_remote(value, profile.path_map)
            else:
                mapped_env[key] = value
        mapped_env.update(profile.env)

        script_path = profile.script_path or str(Path(profile.remote_workdir) / "vid_to_sub.py")
        cli_args = self._build_cli_args(
            paths,
            dry_run=dry_run,
            remote_profile=profile,
            config=config,
            manifest_stdin=True,
        )
        env_prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in sorted(mapped_env.items()) if value
        )
        remote_parts = [profile.python_bin, script_path, *cli_args]
        remote_command = shlex.join(remote_parts)
        if env_prefix:
            remote_command = env_prefix + " " + remote_command
        remote_command = (
            f"cd {shlex.quote(profile.remote_workdir)} && {remote_command}"
        )
        return ["ssh", profile.ssh_target, remote_command]

    def _build_executor_plans(
        self,
        videos: Sequence[str],
        *,
        dry_run: bool,
        config: RunConfig | None = None,
    ) -> list[ExecutorPlan]:
        local_capacity = (
            config.local_workers
            if config is not None
            else _coerce_positive_int(self._val("inp-workers") or "1", default=1)
        )
        mode = (
            config.execution_mode
            if config is not None
            else self._sel("sel-execution-mode", _db.get("tui.execution_mode") or "local")
        )
        remote_resources = config.remote_resources if config is not None else self._remote_resources

        if mode != "distributed" or not remote_resources:
            manifest = build_run_manifest(videos)
            local_cmd = [
                sys.executable,
                str(SCRIPT_PATH),
                *self._build_cli_args(
                    None,
                    dry_run=dry_run,
                    config=config,
                    manifest_stdin=True,
                ),
            ]
            return [
                ExecutorPlan(
                    name="local",
                    kind="local",
                    label="local",
                    cmd=local_cmd,
                    env=self._build_run_env(config),
                    assigned_paths=list(videos),
                    capacity=local_capacity,
                    manifest=manifest,
                    stdin_payload=json.dumps(manifest, ensure_ascii=False),
                )
            ]

        capacities = [("local", local_capacity)]
        capacities.extend((profile.name, profile.slots) for profile in remote_resources)
        assignments = partition_folder_groups_by_capacity(
            group_paths_by_video_folder(videos),
            capacities,
        )
        plans: list[ExecutorPlan] = []

        local_videos = assignments.get("local") or []
        if local_videos:
            local_manifest = build_run_manifest(local_videos)
            plans.append(
                ExecutorPlan(
                    name="local",
                    kind="local",
                    label="local",
                    cmd=[
                        sys.executable,
                        str(SCRIPT_PATH),
                        *self._build_cli_args(
                            None,
                            dry_run=dry_run,
                            config=config,
                            manifest_stdin=True,
                        ),
                    ],
                    env=self._build_run_env(config),
                    assigned_paths=list(local_videos),
                    capacity=local_capacity,
                    manifest=local_manifest,
                    stdin_payload=json.dumps(local_manifest, ensure_ascii=False),
                )
            )

        for profile in remote_resources:
            assigned = assignments.get(profile.name) or []
            if not assigned:
                continue
            base_manifest = build_run_manifest(assigned)
            remote_manifest = apply_runtime_path_map_to_manifest(
                base_manifest,
                lambda raw: map_path_for_remote(raw, profile.path_map),
            )
            plans.append(
                ExecutorPlan(
                    name=profile.name,
                    kind="remote",
                    label=profile.name,
                    cmd=self._build_remote_command(
                        profile,
                        None,
                        dry_run=dry_run,
                        config=config,
                    ),
                    env=None,
                    assigned_paths=list(assigned),
                    capacity=profile.slots,
                    manifest=remote_manifest,
                    stdin_payload=json.dumps(remote_manifest, ensure_ascii=False),
                )
            )

        return plans

    def _apply_progress_event(self, executor: str, event: dict[str, Any]) -> None:
        event_name = str(event.get("event") or "").strip()
        video_path = str(event.get("video_path") or "").strip()
        key = f"{executor}:{video_path}" if video_path else ""
        folder_hash = str(event.get("folder_hash") or "").strip()
        folder_path = str(event.get("folder_path") or "").strip()

        def event_float(name: str) -> float | None:
            try:
                return (
                    float(event.get(name))
                    if event.get(name) is not None
                    else None
                )
            except (TypeError, ValueError):
                return None

        def sync_folder_state_from_event() -> None:
            if not folder_hash or not folder_path:
                return
            try:
                total_files = int(event.get("folder_total_files", event.get("total_files", 0)))
            except (TypeError, ValueError):
                total_files = 0
            try:
                completed_files = int(event.get("folder_completed_files", 0))
            except (TypeError, ValueError):
                completed_files = 0
            status = str(event.get("folder_status") or ("completed" if event.get("folder_completed") else "running")).strip()
            _db.upsert_folder_queue_state(
                folder_hash,
                folder_path,
                status=status or "running",
                total_files=total_files,
                completed_files=completed_files,
                is_completed=bool(event.get("folder_completed")),
            )

        if event_name == "job_started" and video_path:
            video_duration = event_float("video_duration")
            if key not in self._active_jobs:
                job_id = _db.create_job(
                    video_path=video_path,
                    backend=self._run_backend,
                    model=self._run_model,
                    output_dir=self._run_output_dir,
                    language=self._run_language,
                    target_lang=self._run_target_lang,
                )
                self._active_jobs[key] = RunJobState(
                    video_path=video_path,
                    executor=executor,
                    job_id=job_id,
                    started_at=time.monotonic(),
                    video_duration=video_duration,
                    progress_seconds=0.0 if video_duration is not None else None,
                    progress_ratio=0.0 if video_duration is not None else None,
                )
            else:
                job = self._active_jobs[key]
                job.video_duration = video_duration
                if video_duration is not None and job.progress_seconds is None:
                    job.progress_seconds = 0.0
                if video_duration is not None and job.progress_ratio is None:
                    job.progress_ratio = 0.0
            self._refresh_history()
            self._refresh_live_panels()
            self._show_hist_detail(self._hist_key)
            return

        if event_name == "job_progress" and video_path:
            job = self._active_jobs.get(key)
            if job is None:
                return
            video_duration = event_float("video_duration")
            progress_seconds = event_float("progress_seconds")
            progress_ratio = event_float("progress_ratio")
            if video_duration is not None:
                job.video_duration = video_duration
            if progress_seconds is not None:
                job.progress_seconds = progress_seconds
            if progress_ratio is None and job.video_duration and job.progress_seconds is not None:
                progress_ratio = job.progress_seconds / job.video_duration
            if progress_ratio is not None:
                job.progress_ratio = _clamp_ratio(progress_ratio)
            self._refresh_history()
            self._refresh_live_panels()
            self._show_hist_detail(self._hist_key)
            return

        if event_name == "job_finished" and video_path:
            sync_folder_state_from_event()
            status = str(event.get("status") or "failed").strip()
            job = self._active_jobs.pop(key, None)
            error = str(event.get("error") or "").strip() or None
            output_paths = event.get("output_paths")
            if not isinstance(output_paths, list):
                output_paths = []
            wall_sec = event_float("elapsed_sec")
            video_dur = event_float("video_duration")
            try:
                segments = int(event.get("segments")) if event.get("segments") is not None else None
            except (TypeError, ValueError):
                segments = None

            if job and job.job_id is not None:
                _db.finish_job(
                    job.job_id,
                    "done" if status == "done" else "failed",
                    output_paths=[str(path) for path in output_paths],
                    error=error,
                    wall_sec=wall_sec,
                    video_dur=video_dur,
                    segments=segments,
                )
            elif status != "done":
                job_id = _db.create_job(
                    video_path=video_path,
                    backend=self._run_backend,
                    model=self._run_model,
                    output_dir=self._run_output_dir,
                    language=self._run_language,
                    target_lang=self._run_target_lang,
                )
                _db.finish_job(
                    job_id,
                    "failed",
                    error=error or f"{executor} failed before job tracking was established.",
                    wall_sec=wall_sec,
                )

            pending = self._pending_paths.get(executor)
            if pending is not None:
                pending.discard(video_path)

            if status == "done":
                self._run_completed += 1
            else:
                self._run_failed += 1

            self._refresh_history()
            self._refresh_live_panels()
            self._show_hist_detail(self._hist_key)
            return

        if event_name == "folder_finished":
            sync_folder_state_from_event()
            self._refresh_live_panels()
            return

        if event_name == "run_finished" and self._run_dry_run:
            pending = self._pending_paths.get(executor)
            if pending is not None:
                pending.clear()
            self._run_completed = max(
                0,
                self._run_total_queued
                - sum(len(paths) for paths in self._pending_paths.values()),
            )
            self._refresh_live_panels()

    def _finalize_executor_failure(self, plan: ExecutorPlan, rc: int) -> None:
        pending = list(self._pending_paths.pop(plan.label, set()))
        if not pending:
            self._refresh_live_panels()
            return

        for video_path in pending:
            active_key = f"{plan.label}:{video_path}"
            active = self._active_jobs.pop(active_key, None)
            if active and active.job_id is not None:
                _db.finish_job(
                    active.job_id,
                    "failed",
                    error=f"{plan.label} exited with code {rc}",
                )
            else:
                job_id = _db.create_job(
                    video_path=video_path,
                    backend=self._run_backend,
                    model=self._run_model,
                    output_dir=self._run_output_dir,
                    language=self._run_language,
                    target_lang=self._run_target_lang,
                )
                _db.finish_job(
                    job_id,
                    "failed",
                    error=f"{plan.label} exited with code {rc} before starting the job.",
                )
            self._run_failed += 1

        self._refresh_history()
        self._refresh_live_panels()

    def _trigger(self, dry_run: bool = False) -> None:
        try:
            config = self._snapshot_run_config(dry_run)
        except ValueError as exc:
            self._log(f"[bold red]✕ {exc}[/]")
            return

        log = self.query_one("#log", RichLog)
        log.clear()
        log.write(
            "[bold cyan]Preparing run[/] discovering video files and building execution plan…"
        )

        self._reset_run_state(preserve_shell=True)
        if (
            config.execution_mode == "distributed"
            and config.remote_resources
        ):
            remote_slots = sum(profile.slots for profile in config.remote_resources)
            self._run_last_shell = (
                f"[cyan]Preparing distributed run[/] · local={config.local_workers} worker(s) "
                f"· remote={remote_slots} slot(s)"
            )
        else:
            self._run_last_shell = (
                f"[cyan]Preparing local run[/] · {config.local_workers} worker(s)"
            )
        self._refresh_live_panels()
        self._active_worker = self._prepare_run(config)

    def _abort_prepared_run(self, request_id: int, message: str) -> None:
        if request_id != self._run_request_id:
            return
        self._log(message)
        self._reset_run_state()
        self._refresh_live_panels()

    def _start_prepared_run(
        self,
        config: RunConfig,
        videos: list[str],
        found_total: int,
        skipped: int,
        plans: list[ExecutorPlan],
    ) -> None:
        if config.request_id != self._run_request_id:
            return

        log = self.query_one("#log", RichLog)
        log.clear()
        if config.execution_mode == "distributed" and len(plans) > 1:
            log.write(
                f"[bold cyan]Distributed run[/] {len(videos)} video(s) across {len(plans)} executor(s)"
            )
        elif plans[0].kind == "remote":
            log.write(
                f"[bold cyan]Remote run[/] {len(videos)} video(s) via {plans[0].label}"
            )
        else:
            log.write("[bold cyan]$ " + " ".join(_mask(plans[0].cmd[2:])) + "[/bold cyan]")
        for plan in plans:
            log.write(
                f"[dim]{plan.label}[/] {len(plan.assigned_paths)} file(s) · capacity={plan.capacity}"
            )

        self._reset_run_state(preserve_shell=True)
        self._run_started_at = time.monotonic()
        self._run_total_found = found_total
        self._run_total_queued = len(videos)
        self._run_skipped = skipped
        self._run_backend = config.backend
        self._run_model = config.model
        self._run_language = config.language
        self._run_target_lang = (
            config.translate_to if config.translate_enabled else None
        )
        self._run_output_dir = config.output_dir
        self._run_dry_run = config.dry_run
        self._pending_paths = {
            plan.label: set(plan.assigned_paths)
            for plan in plans
        }
        queued_folders: list[dict[str, Any]] = []
        for plan in plans:
            for folder in plan.manifest.get("folders", []):
                if not isinstance(folder, dict):
                    continue
                queued_folders.append(
                    {
                        "folder_hash": str(folder.get("folder_hash") or ""),
                        "folder_path": str(folder.get("folder_path") or ""),
                        "status": "queued",
                        "total_files": int(folder.get("total_files", 0)),
                        "completed_files": 0,
                        "is_completed": False,
                    }
                )
        _db.upsert_folder_queue_states(queued_folders)
        if config.execution_mode == "distributed" and len(plans) > 1:
            remote_slots = sum(
                plan.capacity for plan in plans if plan.kind == "remote"
            )
            self._run_last_shell = (
                f"[cyan]Distributed[/] {len(videos)} file(s) · local={config.local_workers} worker(s) "
                f"· remote={remote_slots} slot(s)"
            )
        elif plans[0].kind == "remote":
            self._run_last_shell = f"[cyan]Remote[/] {plans[0].label} · {len(videos)} file(s)"
        else:
            self._run_last_shell = (
                f"[dim]$[/dim] [cyan]vid_to_sub[/cyan] {' '.join(_mask(plans[0].cmd[2:]))}"
            )
        self._refresh_live_panels()
        self._active_worker = self._stream(plans)

    @work(thread=True, exclusive=True, exit_on_error=False, name="prepare-run")
    def _prepare_run(self, config: RunConfig) -> None:
        try:
            videos, found_total, skipped = self._discover_videos_for_run(config)
        except ValueError as exc:
            self.call_from_thread(
                self._abort_prepared_run,
                config.request_id,
                f"[bold red]✕ {exc}[/]",
            )
            return

        if config.request_id != self._run_request_id:
            return
        if not videos:
            self.call_from_thread(
                self._abort_prepared_run,
                config.request_id,
                "[bold yellow]No video files matched the current selection and filters.[/]",
            )
            return

        plans = self._build_executor_plans(
            videos,
            dry_run=config.dry_run,
            config=config,
        )
        if config.request_id != self._run_request_id:
            return
        if not plans:
            self.call_from_thread(
                self._abort_prepared_run,
                config.request_id,
                "[bold red]✕ Could not build any execution plan.[/]",
            )
            return

        self.call_from_thread(
            self._start_prepared_run,
            config,
            videos,
            found_total,
            skipped,
            plans,
        )

    @work(thread=True, exclusive=True, exit_on_error=False, name="_stream")
    def _stream(self, plans: list[ExecutorPlan]) -> None:
        output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        launched: list[
            tuple[
                ExecutorPlan,
                subprocess.Popen[str],
                threading.Thread,
                threading.Thread | None,
            ]
        ] = []
        multi_executor = len(plans) > 1

        def _pump_stdout(executor_name: str, proc: subprocess.Popen[str]) -> None:
            assert proc.stdout
            for raw in iter(proc.stdout.readline, ""):
                output_queue.put((executor_name, raw.rstrip("\n")))
            proc.stdout.close()
            output_queue.put((executor_name, None))

        def _pump_stdin(proc: subprocess.Popen[str], payload: str) -> None:
            if proc.stdin is None:
                return
            try:
                proc.stdin.write(payload)
            finally:
                proc.stdin.close()

        for plan in plans:
            try:
                proc = subprocess.Popen(
                    plan.cmd,
                    stdin=subprocess.PIPE if plan.stdin_payload is not None else None,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=plan.env,
                )
            except FileNotFoundError as exc:
                self.call_from_thread(
                    self._log, f"[bold red]✕ {plan.label}: not found: {exc}[/]"
                )
                self.call_from_thread(self._finalize_executor_failure, plan, 127)
                continue
            except Exception as exc:
                self.call_from_thread(
                    self._log, f"[bold red]✕ {plan.label}: {exc}[/]"
                )
                self.call_from_thread(self._finalize_executor_failure, plan, 1)
                continue

            self._procs[plan.label] = proc
            if plan.kind == "local":
                self._proc = proc

            reader = threading.Thread(
                target=_pump_stdout,
                args=(plan.label, proc),
                daemon=True,
            )
            reader.start()
            writer: threading.Thread | None = None
            if plan.stdin_payload is not None:
                writer = threading.Thread(
                    target=_pump_stdin,
                    args=(proc, plan.stdin_payload),
                    daemon=True,
                )
                writer.start()
            launched.append((plan, proc, reader, writer))

        active_streams = len(launched)
        while active_streams > 0:
            try:
                executor, line = output_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                active_streams -= 1
                continue

            event = parse_progress_event(line)
            if event:
                self.call_from_thread(self._apply_progress_event, executor, event)
                continue

            rendered = _colorize(line)
            if multi_executor:
                rendered = f"[bold blue]{executor}[/] {rendered}"
            self.call_from_thread(self._log, rendered)

        had_error = False
        for plan, proc, reader, writer in launched:
            reader.join(timeout=1)
            if writer is not None:
                writer.join(timeout=1)
            rc = proc.wait()
            self._procs.pop(plan.label, None)
            if self._proc is proc:
                self._proc = None
            if rc != 0:
                had_error = True
                self.call_from_thread(
                    self._log, f"\n[bold red]✕ {plan.label} exited with code {rc}[/]"
                )
                self.call_from_thread(self._finalize_executor_failure, plan, rc)
            elif multi_executor:
                self.call_from_thread(
                    self._log, f"[green]{plan.label} finished (exit 0)[/]"
                )

        if had_error:
            self.call_from_thread(
                self._log,
                "\n[bold red]✕ One or more executors failed. Review the log and history.[/]",
            )
        else:
            self.call_from_thread(self._log, "\n[bold green]✓ Completed (exit 0)[/]")

    # ── History tab ───────────────────────────────────────────────────────

    def _init_history_table(self) -> None:
        try:
            t = self.query_one("#hist-table", DataTable)
            t.cell_padding = 0
            t.add_column("ID", key="id", width=4)
            t.add_column("Date", key="date", width=10)
            t.add_column("File", key="file", width=22)
            t.add_column("Backend", key="backend", width=10)
            t.add_column("Model", key="model", width=11)
            t.add_column("Status", key="status", width=15)
            t.add_column("Time", key="time", width=6)
        except NoMatches:
            pass

    def _refresh_history(self) -> None:
        try:
            t = self.query_one("#hist-table", DataTable)
        except NoMatches:
            return
        t.clear(columns=False)
        jobs = _db.get_jobs()
        active_by_job_id = {
            state.job_id: state
            for state in self._active_jobs.values()
            if state.job_id is not None
        }

        for job in jobs:
            s = job["status"]
            active = active_by_job_id.get(job["id"])
            if s == "running" and active is not None:
                s_cell = _compact_progress_markup(active.progress_ratio)
            else:
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
        active = next(
            (state for state in self._active_jobs.values() if state.job_id == job["id"]),
            None,
        )
        lines = [
            f"[bold]ID {job['id']}[/]  {job['created_at']}  "
            f"backend={job['backend']}  model={job['model']}  status={job['status']}",
            f"File: {job['video_path']}",
        ]
        if job.get("language") or job.get("target_lang"):
            lines.append(
                f"Language: {job.get('language') or 'auto'}   Translate to: {job.get('target_lang') or '-'}"
            )
        if active is not None:
            progress_seconds = max(0.0, active.progress_seconds or 0.0)
            progress_ratio = (
                active.progress_ratio
                if active.progress_ratio is not None
                else (
                    progress_seconds / active.video_duration
                    if active.video_duration and active.video_duration > 0
                    else 0.0
                )
            )
            lines.append(
                "Progress: " + _progress_bar_markup_ratio(progress_ratio, width=30)
            )
            if active.video_duration is not None:
                lines.append(
                    f"Transcribed: {_fmt_elapsed(progress_seconds)} / "
                    f"{_fmt_elapsed(active.video_duration)}"
                )
        if job.get("output_dir"):
            lines.append(f"Output dir: {job['output_dir']}")
        if job.get("wall_sec"):
            lines.append(f"Wall time: {_fmt_elapsed(job['wall_sec'])}")
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

    def _load_history_job(self, job_id: int) -> None:
        job = next((item for item in _db.get_jobs() if item["id"] == job_id), None)
        if not job:
            self._log(f"[red]History row #{job_id} was not found.[/]")
            return

        self._selected_paths = [job["video_path"]]
        self._refresh_sel_paths()
        self._refresh_recent_paths()

        for wid, value in [
            ("inp-output-dir", job.get("output_dir") or ""),
            ("inp-language", job.get("language") or ""),
        ]:
            try:
                self.query_one(f"#{wid}", Input).value = value
            except NoMatches:
                pass

        try:
            self.query_one("#sel-backend", Select).value = job.get("backend") or DEFAULT_BACKEND
            self.query_one("#sel-model", Select).value = job.get("model") or DEFAULT_MODEL
        except NoMatches:
            pass

        translate_target = job.get("target_lang") or ""
        try:
            self.query_one("#sw-translate", Switch).value = bool(translate_target)
            self.query_one("#inp-translate-to", Input).value = translate_target
            fields = self.query_one("#trans-fields")
            if translate_target:
                fields.remove_class("hidden")
            else:
                fields.add_class("hidden")
        except NoMatches:
            pass

        self._update_cmd_preview()
        self.action_tab("tab-transcribe")
        self._log(
            f"[green]Loaded history row #{job_id} into the Transcribe form.[/]"
        )

    def _rerun_history_job(self, job_id: int) -> None:
        self._load_history_job(job_id)
        self.action_run()

    # ── Settings ──────────────────────────────────────────────────────────

    def _load_settings_form(self) -> None:
        pairs = [
            ("stg-wcpp-bin", ENV_WCPP_BIN),
            ("stg-wcpp-model", ENV_WCPP_MODEL),
            ("stg-trans-url", ENV_TRANS_URL),
            ("stg-trans-key", ENV_TRANS_KEY),
            ("stg-trans-model", ENV_TRANS_MOD),
            ("stg-agent-url", ENV_AGENT_URL),
            ("stg-agent-key", ENV_AGENT_KEY),
            ("stg-agent-model", ENV_AGENT_MOD),
            ("stg-build-dir", "tui.build_dir"),
            ("stg-install-dir", "tui.install_dir"),
            ("stg-model-dir", "tui.model_dir"),
            ("stg-browse-root", "tui.browse_root"),
            ("stg-default-output-dir", "tui.default_output_dir"),
            ("stg-default-translate-to", "tui.default_translate_to"),
        ]
        for wid, key in pairs:
            try:
                self.query_one(f"#{wid}", Input).value = _db.get(key)
            except NoMatches:
                pass
        try:
            self.query_one("#stg-remote-resources", TextArea).text = _db.get(
                "tui.remote_resources", "[]"
            )
        except NoMatches:
            pass
        try:
            mode = _db.get("tui.execution_mode") or "local"
            self.query_one("#sel-execution-mode", Select).value = mode
        except NoMatches:
            pass

    def _prefill_transcribe(self) -> None:
        """Pre-fill run form fields from settings / env vars."""
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
        for wid, key in [
            ("inp-output-dir", "tui.default_output_dir"),
            ("inp-translate-to", "tui.default_translate_to"),
        ]:
            try:
                val = _db.get(key).strip()
                if val:
                    self.query_one(f"#{wid}", Input).value = val
            except NoMatches:
                pass

    def _sync_transcribe_overrides_from_settings(
        self,
        previous_values: Mapping[str, str],
        updated_values: Mapping[str, str],
    ) -> None:
        pairs = [
            ("inp-trans-url", ENV_TRANS_URL),
            ("inp-trans-key", ENV_TRANS_KEY),
            ("inp-trans-model", ENV_TRANS_MOD),
            ("inp-wcpp-bin", ENV_WCPP_BIN),
        ]
        for wid, key in pairs:
            try:
                widget = self.query_one(f"#{wid}", Input)
            except NoMatches:
                continue

            current = widget.value.strip()
            previous = (previous_values.get(key) or "").strip()
            if current and current != previous:
                continue

            widget.value = (updated_values.get(key) or "").strip()

    def _sync_run_defaults_from_settings(
        self,
        previous_values: Mapping[str, str],
        updated_values: Mapping[str, str],
    ) -> None:
        pairs = [
            ("inp-output-dir", "tui.default_output_dir"),
            ("inp-translate-to", "tui.default_translate_to"),
        ]
        for wid, key in pairs:
            try:
                widget = self.query_one(f"#{wid}", Input)
            except NoMatches:
                continue

            current = widget.value.strip()
            previous = (previous_values.get(key) or "").strip()
            if current and current != previous:
                continue

            widget.value = (updated_values.get(key) or "").strip()

    def _save_settings(self) -> None:
        pairs = [
            ("stg-wcpp-bin", ENV_WCPP_BIN),
            ("stg-wcpp-model", ENV_WCPP_MODEL),
            ("stg-trans-url", ENV_TRANS_URL),
            ("stg-trans-key", ENV_TRANS_KEY),
            ("stg-trans-model", ENV_TRANS_MOD),
            ("stg-agent-url", ENV_AGENT_URL),
            ("stg-agent-key", ENV_AGENT_KEY),
            ("stg-agent-model", ENV_AGENT_MOD),
            ("stg-build-dir", "tui.build_dir"),
            ("stg-install-dir", "tui.install_dir"),
            ("stg-model-dir", "tui.model_dir"),
            ("stg-browse-root", "tui.browse_root"),
            ("stg-default-output-dir", "tui.default_output_dir"),
            ("stg-default-translate-to", "tui.default_translate_to"),
        ]
        data: dict[str, str] = {}
        for wid, key in pairs:
            try:
                data[key] = self.query_one(f"#{wid}", Input).value.strip()
            except NoMatches:
                pass
        try:
            data["tui.remote_resources"] = self.query_one(
                "#stg-remote-resources", TextArea
            ).text.strip() or "[]"
        except NoMatches:
            pass
        data["tui.execution_mode"] = self._sel("sel-execution-mode", "local")
        try:
            parse_remote_resources(data.get("tui.remote_resources", "[]"))
        except ValueError as exc:
            try:
                self.query_one("#stg-status", Static).update(f"[red]✗ {exc}[/]")
            except NoMatches:
                pass
            return
        previous_values = {key: _db.get(key) for _, key in pairs}
        _db.set_many(data)
        self._apply_db_to_env()
        self._sync_transcribe_overrides_from_settings(previous_values, data)
        self._sync_run_defaults_from_settings(previous_values, data)
        self._load_remote_resources()
        self._run_detection()
        self._update_wcpp_model_status()
        self._update_remote_status()
        self._update_agent_config_status()
        self._update_cmd_preview()
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
            ENV_AGENT_URL: "stg-agent-url",
            ENV_AGENT_KEY: "stg-agent-key",
            ENV_AGENT_MOD: "stg-agent-model",
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
            load_project_env(override=False)
        env_keys = [
            ENV_WCPP_BIN,
            ENV_WCPP_MODEL,
            ENV_TRANS_URL,
            ENV_TRANS_KEY,
            ENV_TRANS_MOD,
            ENV_AGENT_URL,
            ENV_AGENT_KEY,
            ENV_AGENT_MOD,
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
