#!/usr/bin/env python3
"""
tui.py — Textual 8.x TUI for vid_to_sub
=========================================
Full-featured terminal UI for configuring and running vid_to_sub.

Tabs
----
  1  Paths & Output    — input paths, output dir, recursion, output formats
  2  Transcription     — backend, model, device, beam-size, workers, language
  3  whisper.cpp       — binary path, model path, active env display
  4  Translation       — on/off, target lang, API URL/key/model, diarize, HF token
  5  Environment       — read/edit/save all 5 env vars to .env

Bottom panel (always visible)
------------------------------
  Command preview | ▶ Run | ⚡ Dry Run | ✕ Kill | scrollable RichLog

Keyboard shortcuts
------------------
  Ctrl+R   Run
  Ctrl+D   Dry Run
  Ctrl+K   Kill running process
  Ctrl+S   Save .env
  Ctrl+Q   Quit
  1–5      Switch tabs

Usage
-----
  /home/nodove/workspace/.venv/bin/python tui.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

from dotenv import load_dotenv
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Checkbox,
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

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
SCRIPT_PATH = SCRIPT_DIR / "vid_to_sub.py"
ENV_FILE = SCRIPT_DIR / ".env"

# Mirror constants from vid_to_sub (avoids an import that pulls in optional deps)
DEFAULT_BACKEND = "whisper-cpp"
DEFAULT_MODEL = "large-v3"
DEFAULT_DEVICE = "cpu"
DEFAULT_FORMAT = "srt"

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

ENV_WHISPER_CPP_BIN = "VID_TO_SUB_WHISPER_CPP_BIN"
ENV_WHISPER_CPP_MODEL = "VID_TO_SUB_WHISPER_CPP_MODEL"
ENV_TRANSLATION_BASE_URL = "VID_TO_SUB_TRANSLATION_BASE_URL"
ENV_TRANSLATION_API_KEY = "VID_TO_SUB_TRANSLATION_API_KEY"
ENV_TRANSLATION_MODEL = "VID_TO_SUB_TRANSLATION_MODEL"

# env-var → widget-id mapping (Environment tab)
ENV_WIDGET: dict[str, str] = {
    ENV_WHISPER_CPP_BIN: "env-wcpp-bin",
    ENV_WHISPER_CPP_MODEL: "env-wcpp-model",
    ENV_TRANSLATION_BASE_URL: "env-trans-url",
    ENV_TRANSLATION_API_KEY: "env-trans-key",
    ENV_TRANSLATION_MODEL: "env-trans-model",
}

# ---------------------------------------------------------------------------
# CSS (inline — no external file required)
# ---------------------------------------------------------------------------

CSS = """
/* ── global ─────────────────────────────────────────── */
Screen { layout: vertical; background: $background; }

/* ── tab area ────────────────────────────────────────── */
#tab-area { height: 1fr; min-height: 18; }

/* ── scrollable form body inside each tab ────────────── */
.tab-scroll { padding: 1 2; }

/* ── section headers ─────────────────────────────────── */
.section-title {
    text-style: bold;
    color: $accent;
    margin-top: 1;
    margin-bottom: 0;
    padding: 0 0 0 1;
    border-bottom: solid $accent;
    height: 2;
}

/* ── form rows ───────────────────────────────────────── */
.form-row {
    height: auto;
    margin-bottom: 1;
    align: left middle;
}
.form-label {
    width: 26;
    padding-right: 1;
    color: $text-muted;
    text-align: right;
}
.form-widget { width: 1fr; }

/* ── checkboxes / switches in a row ─────────────────── */
.check-row {
    height: auto;
    padding: 0 1;
    margin-bottom: 1;
    align: left middle;
}
.check-row Checkbox { margin-right: 3; }
.check-row Switch   { margin-right: 2; }
.check-row Label    { margin-right: 1; color: $text-muted; }

/* ── format checkboxes ───────────────────────────────── */
#format-row { height: auto; padding: 0 1; margin-bottom: 1; }
#format-row Checkbox { margin-right: 2; }

/* ── path list ───────────────────────────────────────── */
#paths-container { height: auto; }
.path-entry { margin-bottom: 1; }
#path-buttons { height: auto; margin-bottom: 1; }
#path-buttons Button { margin-right: 1; }

/* ── bottom panel ────────────────────────────────────── */
#bottom-panel {
    height: auto;
    border-top: solid $primary;
    padding: 0 1;
}
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
#run-buttons { height: 3; margin-bottom: 1; align: left middle; }
#run-buttons Button { margin-right: 1; }
#log { height: 14; border: solid $panel; background: $surface-darken-1; }

/* ── env tab status ──────────────────────────────────── */
#env-status { height: 1; margin-top: 1; color: $success; }
#env-buttons { height: 3; align: left middle; margin-top: 1; }
#env-buttons Button { margin-right: 1; }

/* ── whisper.cpp env display ─────────────────────────── */
.env-display {
    color: $text-muted;
    height: 1;
    padding: 0 1;
    margin-bottom: 0;
}

/* ── translation fields visibility ───────────────────── */
#translation-fields.hidden { display: none; }

/* ── about tab ───────────────────────────────────────── */
#about-text { padding: 2 3; color: $text-muted; }
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opts(values: list[str], default: str) -> list[tuple[str, str]]:
    """Select options: default first, then the rest."""
    ordered = [default] + [v for v in values if v != default]
    return [(v, v) for v in ordered]


def _colorize(line: str) -> str:
    """Apply Rich markup to known output patterns."""
    if "[ERROR]" in line:
        return f"[bold red]{line}[/]"
    if "[WARN]" in line:
        return f"[yellow]{line}[/]"
    if "\u2713" in line or "succeeded" in line:
        return f"[green]{line}[/]"
    if line.startswith("\u25b6"):
        return f"[cyan]{line}[/]"
    if line.startswith("Found") or "Done." in line:
        return f"[bold]{line}[/]"
    return line


def _mask_cmd(cmd: list[str]) -> list[str]:
    """Replace values of sensitive flags with *** for display."""
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


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class VidToSubTUI(App):
    """Interactive Textual TUI for vid_to_sub."""

    TITLE = "vid_to_sub"
    SUB_TITLE = "Video \u2192 Subtitle"
    CSS = CSS

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("ctrl+r", "run", "\u25b6 Run", priority=True),
        Binding("ctrl+d", "dry_run", "\u26a1 Dry Run", priority=True),
        Binding("ctrl+k", "kill", "\u2715 Kill", priority=True, show=False),
        Binding("ctrl+s", "save_env", "\U0001f4be .env", priority=True, show=False),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("1", "show_tab('tab-paths')", show=False),
        Binding("2", "show_tab('tab-transcription')", show=False),
        Binding("3", "show_tab('tab-whisper-cpp')", show=False),
        Binding("4", "show_tab('tab-translation')", show=False),
        Binding("5", "show_tab('tab-env')", show=False),
    ]

    # internal state
    _proc: subprocess.Popen | None = None
    _path_count: int = 1
    _active_worker: Worker | None = None

    # ── compose ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:  # noqa: C901
        yield Header()

        with Vertical(id="tab-area"):
            with TabbedContent(initial="tab-paths"):
                # ── Tab 1: Paths & Output ─────────────────────────────
                with TabPane("1 Paths & Output", id="tab-paths"):
                    with ScrollableContainer(classes="tab-scroll"):
                        yield Static("Input Paths", classes="section-title")
                        with Vertical(id="paths-container"):
                            yield Input(
                                placeholder="/path/to/video.mp4  or  /directory",
                                id="path-0",
                                classes="path-entry",
                            )
                        with Horizontal(id="path-buttons"):
                            yield Button(
                                "\uff0b Add Path", id="btn-add-path", variant="default"
                            )
                            yield Button(
                                "\uff0d Remove Last",
                                id="btn-remove-path",
                                variant="default",
                            )

                        yield Static("Output", classes="section-title")
                        with Horizontal(classes="form-row"):
                            yield Label("Output directory", classes="form-label")
                            yield Input(
                                placeholder="(next to each video by default)",
                                id="inp-output-dir",
                                classes="form-widget",
                            )

                        yield Static("Behaviour", classes="section-title")
                        with Horizontal(classes="check-row"):
                            yield Checkbox(
                                "No recurse", id="chk-no-recurse", value=False
                            )
                            yield Checkbox(
                                "Skip existing", id="chk-skip-existing", value=False
                            )
                            yield Checkbox("Verbose", id="chk-verbose", value=False)

                        yield Static(
                            "Output Formats  (select one or more)",
                            classes="section-title",
                        )
                        with Horizontal(id="format-row"):
                            for fmt in FORMATS:
                                yield Checkbox(
                                    fmt, id=f"fmt-{fmt}", value=(fmt == "srt")
                                )
                            yield Checkbox("all", id="fmt-all", value=False)

                # ── Tab 2: Transcription ──────────────────────────────
                with TabPane("2 Transcription", id="tab-transcription"):
                    with ScrollableContainer(classes="tab-scroll"):
                        yield Static("Backend & Model", classes="section-title")
                        with Horizontal(classes="form-row"):
                            yield Label("Backend", classes="form-label")
                            yield Select(
                                _opts(BACKENDS, DEFAULT_BACKEND),
                                id="sel-backend",
                                classes="form-widget",
                                allow_blank=False,
                            )
                        with Horizontal(classes="form-row"):
                            yield Label("Model", classes="form-label")
                            yield Select(
                                _opts(KNOWN_MODELS, DEFAULT_MODEL),
                                id="sel-model",
                                classes="form-widget",
                                allow_blank=False,
                            )

                        yield Static("Hardware", classes="section-title")
                        with Horizontal(classes="form-row"):
                            yield Label("Device", classes="form-label")
                            yield Select(
                                _opts(DEVICES, DEFAULT_DEVICE),
                                id="sel-device",
                                classes="form-widget",
                                allow_blank=False,
                            )
                        with Horizontal(classes="form-row"):
                            yield Label("Compute type", classes="form-label")
                            yield Input(
                                placeholder="auto  (e.g. int8, float16, int8_float16)",
                                id="inp-compute-type",
                                classes="form-widget",
                            )

                        yield Static("Decoding", classes="section-title")
                        with Horizontal(classes="form-row"):
                            yield Label("Beam size", classes="form-label")
                            yield Input(
                                value="5",
                                id="inp-beam-size",
                                type="integer",
                                classes="form-widget",
                            )
                        with Horizontal(classes="form-row"):
                            yield Label("Workers / threads", classes="form-label")
                            yield Input(
                                value="1",
                                id="inp-workers",
                                type="integer",
                                classes="form-widget",
                            )
                        with Horizontal(classes="form-row"):
                            yield Label("Language", classes="form-label")
                            yield Input(
                                placeholder="auto-detect  (e.g. en, ja, ko, zh)",
                                id="inp-language",
                                classes="form-widget",
                            )

                # ── Tab 3: whisper.cpp ────────────────────────────────
                with TabPane("3 whisper.cpp", id="tab-whisper-cpp"):
                    with ScrollableContainer(classes="tab-scroll"):
                        yield Static("Binary & Model File", classes="section-title")
                        with Horizontal(classes="form-row"):
                            yield Label("whisper-cli binary", classes="form-label")
                            yield Input(
                                placeholder="whisper-cli  (on PATH, or set env var)",
                                id="inp-wcpp-bin",
                                classes="form-widget",
                            )
                        with Horizontal(classes="form-row"):
                            yield Label("GGML model file", classes="form-label")
                            yield Input(
                                placeholder="./models/ggml-large-v3.bin  (or set env var)",
                                id="inp-wcpp-model",
                                classes="form-widget",
                            )

                        yield Static(
                            "Active Environment Values", classes="section-title"
                        )
                        yield Static("", id="envdisp-bin", classes="env-display")
                        yield Static("", id="envdisp-model", classes="env-display")

                        yield Static("Setup Guide", classes="section-title")
                        yield Static(
                            "  Build:   git clone https://github.com/ggerganov/whisper.cpp\n"
                            "           cmake -B build && cmake --build build -j\n\n"
                            "  Models:  https://huggingface.co/ggerganov/whisper.cpp/tree/main\n"
                            "           bash models/download-ggml-model.sh large-v3\n\n"
                            "  Set binary path + model path in the Environment tab, then Save .env.",
                            id="about-wcpp",
                        )

                # ── Tab 4: Translation & whisperX ─────────────────────
                with TabPane("4 Translation", id="tab-translation"):
                    with ScrollableContainer(classes="tab-scroll"):
                        yield Static(
                            "Translation (OpenAI-compatible API)",
                            classes="section-title",
                        )
                        with Horizontal(classes="check-row"):
                            yield Label("Enable translation")
                            yield Switch(id="sw-enable-translation", value=False)

                        with Vertical(id="translation-fields", classes="hidden"):
                            with Horizontal(classes="form-row"):
                                yield Label("Translate to", classes="form-label")
                                yield Input(
                                    placeholder="target language code  e.g. ko, ja, fr, zh",
                                    id="inp-translate-to",
                                    classes="form-widget",
                                )
                            with Horizontal(classes="form-row"):
                                yield Label("Translation model", classes="form-label")
                                yield Input(
                                    placeholder="(from VID_TO_SUB_TRANSLATION_MODEL)",
                                    id="inp-trans-model",
                                    classes="form-widget",
                                )
                            with Horizontal(classes="form-row"):
                                yield Label("Base URL", classes="form-label")
                                yield Input(
                                    placeholder="(from VID_TO_SUB_TRANSLATION_BASE_URL)",
                                    id="inp-trans-url",
                                    classes="form-widget",
                                )
                            with Horizontal(classes="form-row"):
                                yield Label("API key", classes="form-label")
                                yield Input(
                                    placeholder="(from VID_TO_SUB_TRANSLATION_API_KEY)",
                                    id="inp-trans-key",
                                    password=True,
                                    classes="form-widget",
                                )

                        yield Static("whisperX / Diarization", classes="section-title")
                        with Horizontal(classes="check-row"):
                            yield Label("Enable diarization  (whisperX only)")
                            yield Switch(id="sw-diarize", value=False)
                        with Horizontal(classes="form-row"):
                            yield Label("HuggingFace token", classes="form-label")
                            yield Input(
                                placeholder="hf_...  (required for diarization)",
                                id="inp-hf-token",
                                password=True,
                                classes="form-widget",
                            )

                # ── Tab 5: Environment ────────────────────────────────
                with TabPane("5 Environment", id="tab-env"):
                    with ScrollableContainer(classes="tab-scroll"):
                        yield Static(
                            "Environment Variables  (.env)", classes="section-title"
                        )
                        yield Static(
                            "  Saved to / loaded from:  " + str(ENV_FILE),
                            classes="env-display",
                        )

                        yield Static("whisper.cpp", classes="section-title")
                        with Horizontal(classes="form-row"):
                            yield Label(ENV_WHISPER_CPP_BIN, classes="form-label")
                            yield Input(
                                placeholder="path/to/whisper-cli",
                                id="env-wcpp-bin",
                                classes="form-widget",
                            )
                        with Horizontal(classes="form-row"):
                            yield Label(ENV_WHISPER_CPP_MODEL, classes="form-label")
                            yield Input(
                                placeholder="path/to/ggml-large-v3.bin",
                                id="env-wcpp-model",
                                classes="form-widget",
                            )

                        yield Static("Translation API", classes="section-title")
                        with Horizontal(classes="form-row"):
                            yield Label(ENV_TRANSLATION_BASE_URL, classes="form-label")
                            yield Input(
                                placeholder="https://api.openai.com/v1",
                                id="env-trans-url",
                                classes="form-widget",
                            )
                        with Horizontal(classes="form-row"):
                            yield Label(ENV_TRANSLATION_API_KEY, classes="form-label")
                            yield Input(
                                placeholder="sk-...",
                                id="env-trans-key",
                                password=True,
                                classes="form-widget",
                            )
                        with Horizontal(classes="form-row"):
                            yield Label(ENV_TRANSLATION_MODEL, classes="form-label")
                            yield Input(
                                placeholder="gpt-4.1-mini",
                                id="env-trans-model",
                                classes="form-widget",
                            )

                        yield Static("", id="env-status")
                        with Horizontal(id="env-buttons"):
                            yield Button(
                                "\U0001f4be Save .env",
                                id="btn-save-env",
                                variant="primary",
                            )
                            yield Button(
                                "\U0001f4c2 Load .env",
                                id="btn-load-env",
                                variant="default",
                            )

                # ── Tab 6: About ──────────────────────────────────────
                with TabPane("? About", id="tab-about"):
                    yield Static(
                        "\n".join(
                            [
                                "  vid_to_sub \u2014 Video \u2192 Subtitle",
                                "",
                                "  Recursively transcribes video files under a directory",
                                "  and writes .srt / .vtt / .txt / .tsv / .json subtitle files.",
                                "",
                                "  Default pipeline:",
                                "    whisper.cpp  \u2192  large-v3  \u2192  CPU  \u2192  .srt",
                                "",
                                "  Supported backends:",
                                "    \u2022 whisper-cpp      (default, no Python deps beyond ffmpeg)",
                                "    \u2022 faster-whisper   (pip install faster-whisper)",
                                "    \u2022 openai-whisper   (pip install openai-whisper)",
                                "    \u2022 whisperX         (pip install whisperx)",
                                "",
                                "  Translation:",
                                "    Any OpenAI-compatible /chat/completions endpoint.",
                                "    Timings are preserved exactly \u2014 only text is translated.",
                                "    Translated files get a language suffix: movie.ko.srt",
                                "",
                                "  Keyboard shortcuts:",
                                "    Ctrl+R   Run",
                                "    Ctrl+D   Dry Run  (lists files without transcribing)",
                                "    Ctrl+K   Kill running process",
                                "    Ctrl+S   Save .env",
                                "    Ctrl+Q   Quit",
                                "    1\u20135      Switch to tab",
                            ]
                        ),
                        id="about-text",
                    )

        # ── Bottom panel (always visible) ─────────────────────────────────
        with Vertical(id="bottom-panel"):
            yield Static(
                "[dim]Configure above, then press [bold]Ctrl+R[/] to run[/]",
                id="cmd-preview",
            )
            with Horizontal(id="run-buttons"):
                yield Button("\u25b6 Run", id="btn-run", variant="success")
                yield Button("\u26a1 Dry Run", id="btn-dry-run", variant="warning")
                yield Button(
                    "\u2715 Kill", id="btn-kill", variant="error", disabled=True
                )
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
        # Load .env on startup if it exists
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=True)
        self._populate_env_tab()
        self._update_wcpp_env_display()
        # Pre-fill Translation tab from env
        self._prefill_translation_tab()
        self._update_cmd_preview()

    # ── Event handlers ────────────────────────────────────────────────────

    def on_input_changed(self, _: Input.Changed) -> None:
        self._update_cmd_preview()

    def on_select_changed(self, _: Select.Changed) -> None:
        self._update_cmd_preview()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        # "all" format checkbox logic
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
        if event.switch.id == "sw-enable-translation":
            fields = self.query_one("#translation-fields")
            if event.value:
                fields.remove_class("hidden")
            else:
                fields.add_class("hidden")
        self._update_cmd_preview()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-run":
            self.action_run()
        elif bid == "btn-dry-run":
            self.action_dry_run()
        elif bid == "btn-kill":
            self.action_kill()
        elif bid == "btn-save-env":
            self._save_env()
        elif bid == "btn-load-env":
            self._load_env()
        elif bid == "btn-add-path":
            self._add_path_input()
        elif bid == "btn-remove-path":
            self._remove_last_path()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        state = event.state
        try:
            run_btn = self.query_one("#btn-run", Button)
            kill_btn = self.query_one("#btn-kill", Button)
            dry_btn = self.query_one("#btn-dry-run", Button)
        except NoMatches:
            return
        if state == WorkerState.RUNNING:
            run_btn.disabled = True
            dry_btn.disabled = True
            kill_btn.disabled = False
            self.sub_title = "Running\u2026"
        else:
            run_btn.disabled = False
            dry_btn.disabled = False
            kill_btn.disabled = True
            if state == WorkerState.CANCELLED:
                self.sub_title = "Cancelled"
            elif state == WorkerState.ERROR:
                self.sub_title = "Error"
            else:
                self.sub_title = "Idle"

    # ── Actions (keyboard shortcuts) ──────────────────────────────────────

    def action_run(self) -> None:
        self._trigger(dry_run=False)

    def action_dry_run(self) -> None:
        self._trigger(dry_run=True)

    def action_kill(self) -> None:
        self._kill()

    def action_save_env(self) -> None:
        self._save_env()

    def action_show_tab(self, tab: str) -> None:
        try:
            self.query_one(TabbedContent).active = tab
        except NoMatches:
            pass

    def action_quit(self) -> None:
        self._kill()
        self.exit()

    # ── Path management ───────────────────────────────────────────────────

    def _add_path_input(self) -> None:
        self._path_count += 1
        new_inp = Input(
            placeholder=f"path {self._path_count}",
            id=f"path-{self._path_count - 1}",
            classes="path-entry",
        )
        self.query_one("#paths-container").mount(new_inp)
        new_inp.focus()
        self._update_cmd_preview()

    def _remove_last_path(self) -> None:
        entries = list(self.query(".path-entry"))
        if len(entries) > 1:
            entries[-1].remove()
            self._path_count = max(1, self._path_count - 1)
            self._update_cmd_preview()

    # ── Command builder ───────────────────────────────────────────────────

    def _build_cmd(self, dry_run: bool = False) -> list[str]:
        """
        Build the full command list from current form values.
        Raises ValueError if required fields are missing.
        """
        cmd: list[str] = [sys.executable, str(SCRIPT_PATH)]

        # --- paths (required) ---
        paths = [
            inp.value.strip() for inp in self.query(".path-entry") if inp.value.strip()
        ]
        if not paths:
            raise ValueError("At least one input path is required.")
        cmd.extend(paths)

        # --- output dir ---
        if v := self._val("inp-output-dir"):
            cmd += ["--output-dir", v]

        # --- flags ---
        if self._chk("chk-no-recurse"):
            cmd.append("--no-recurse")
        if self._chk("chk-skip-existing"):
            cmd.append("--skip-existing")
        if dry_run:
            cmd.append("--dry-run")
        if self._chk("chk-verbose"):
            cmd.append("--verbose")

        # --- formats ---
        if self._chk("fmt-all"):
            cmd += ["--format", "all"]
        else:
            fmts = [f for f in FORMATS if self._chk(f"fmt-{f}")]
            if not fmts:
                fmts = ["srt"]  # fallback default
            for f in fmts:
                cmd += ["--format", f]

        # --- transcription ---
        cmd += ["--backend", self._sel("sel-backend", DEFAULT_BACKEND)]
        cmd += ["--model", self._sel("sel-model", DEFAULT_MODEL)]
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

        # --- whisper.cpp ---
        if v := self._val("inp-wcpp-model"):
            cmd += ["--whisper-cpp-model-path", v]

        # --- translation ---
        if self._sw("sw-enable-translation"):
            if v := self._val("inp-translate-to"):
                cmd += ["--translate-to", v]
            if v := self._val("inp-trans-model"):
                cmd += ["--translation-model", v]
            if v := self._val("inp-trans-url"):
                cmd += ["--translation-base-url", v]
            if v := self._val("inp-trans-key"):
                cmd += ["--translation-api-key", v]

        # --- whisperX / diarize ---
        if self._sw("sw-diarize"):
            cmd.append("--diarize")
        if v := self._val("inp-hf-token"):
            cmd += ["--hf-token", v]

        return cmd

    # ── Run & Kill ────────────────────────────────────────────────────────

    def _trigger(self, dry_run: bool = False) -> None:
        try:
            cmd = self._build_cmd(dry_run=dry_run)
        except ValueError as exc:
            self._log(f"[bold red]\u2715 {exc}[/]")
            return

        log = self.query_one("#log", RichLog)
        log.clear()
        log.write(
            "[bold cyan]$ "
            + " ".join(_mask_cmd(cmd[2:]))  # hide "python script.py" prefix
            + "[/bold cyan]"
        )
        self._active_worker = self._stream(cmd)

    @work(thread=True, exclusive=True, exit_on_error=False)
    def _stream(self, cmd: list[str]) -> None:
        """Spawn vid_to_sub as subprocess and stream output line-by-line to RichLog."""
        env = os.environ.copy()

        # Honour wcpp binary override from Tab 3 at runtime
        bin_override = ""
        try:
            bin_override = self.query_one("#inp-wcpp-bin", Input).value.strip()
        except NoMatches:
            pass
        if bin_override:
            env[ENV_WHISPER_CPP_BIN] = bin_override

        try:
            proc = subprocess.Popen(
                [str(c) for c in cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
        except FileNotFoundError as exc:
            self.call_from_thread(self._log, f"[bold red]\u2715 Not found: {exc}[/]")
            return
        except Exception as exc:
            self.call_from_thread(self._log, f"[bold red]\u2715 {exc}[/]")
            return

        self._proc = proc

        assert proc.stdout
        for raw_line in iter(proc.stdout.readline, ""):
            stripped = raw_line.rstrip("\n")
            if stripped:
                self.call_from_thread(self._log, _colorize(stripped))

        proc.stdout.close()
        rc = proc.wait()
        self._proc = None

        if rc == 0:
            self.call_from_thread(
                self._log, "\n[bold green]\u2713 Completed (exit 0)[/]"
            )
        else:
            self.call_from_thread(
                self._log, f"\n[bold red]\u2715 Exited with code {rc}[/]"
            )

    def _kill(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log("[yellow]\u26a1 Process terminated.[/]")
        if self._active_worker and self._active_worker.is_running:
            self._active_worker.cancel()

    # ── Command preview ───────────────────────────────────────────────────

    def _update_cmd_preview(self) -> None:
        try:
            cmd = self._build_cmd()
            display = " ".join(_mask_cmd(cmd[2:]))  # drop "python script.py"
            self.query_one("#cmd-preview").update(
                f"[dim]$[/dim] [cyan]vid_to_sub[/cyan] {display}"
            )
        except ValueError:
            self.query_one("#cmd-preview").update(
                "[dim]Add at least one input path above[/]"
            )
        except Exception:
            pass  # widgets may not be mounted yet on startup

    # ── Environment tab ───────────────────────────────────────────────────

    def _populate_env_tab(self) -> None:
        for var, wid in ENV_WIDGET.items():
            try:
                self.query_one(f"#{wid}", Input).value = os.environ.get(var, "")
            except NoMatches:
                pass

    def _prefill_translation_tab(self) -> None:
        """Pre-fill translation tab inputs from env vars (if not already set)."""
        pairs = [
            ("inp-trans-url", ENV_TRANSLATION_BASE_URL),
            ("inp-trans-key", ENV_TRANSLATION_API_KEY),
            ("inp-trans-model", ENV_TRANSLATION_MODEL),
        ]
        for wid, var in pairs:
            try:
                w = self.query_one(f"#{wid}", Input)
                if not w.value:
                    w.value = os.environ.get(var, "")
            except NoMatches:
                pass

    def _update_wcpp_env_display(self) -> None:
        bin_val = os.environ.get(ENV_WHISPER_CPP_BIN, "(not set)")
        model_val = os.environ.get(ENV_WHISPER_CPP_MODEL, "(not set)")
        try:
            self.query_one("#envdisp-bin").update(
                f"  {ENV_WHISPER_CPP_BIN} = [bold]{bin_val}[/]"
            )
            self.query_one("#envdisp-model").update(
                f"  {ENV_WHISPER_CPP_MODEL} = [bold]{model_val}[/]"
            )
        except NoMatches:
            pass

    def _save_env(self) -> None:
        lines: list[str] = []
        for var, wid in ENV_WIDGET.items():
            try:
                val = self.query_one(f"#{wid}", Input).value.strip()
            except NoMatches:
                val = ""
            escaped = val.replace('"', '\\"')
            lines.append(f'{var}="{escaped}"\n' if val else f"# {var}=\n")
        try:
            ENV_FILE.write_text("".join(lines), encoding="utf-8")
        except OSError as exc:
            self._env_status(f"[red]\u2715 Save failed: {exc}[/]")
            return
        # Push into os.environ so subprocess inherits without restart
        for var, wid in ENV_WIDGET.items():
            try:
                val = self.query_one(f"#{wid}", Input).value.strip()
            except NoMatches:
                val = ""
            if val:
                os.environ[var] = val
            else:
                os.environ.pop(var, None)
        self._update_wcpp_env_display()
        self._env_status("[bold green]\u2713 Saved to .env[/]")

    def _load_env(self) -> None:
        if not ENV_FILE.exists():
            self._env_status("[yellow]\u26a0 .env not found[/]")
            return
        load_dotenv(ENV_FILE, override=True)
        self._populate_env_tab()
        self._prefill_translation_tab()
        self._update_wcpp_env_display()
        self._env_status("[bold green]\u2713 Loaded from .env[/]")

    def _env_status(self, msg: str) -> None:
        try:
            self.query_one("#env-status").update(msg)
        except NoMatches:
            pass

    # ── Small widget accessors ────────────────────────────────────────────

    def _val(self, wid: str) -> str:
        try:
            return self.query_one(f"#{wid}", Input).value.strip()
        except NoMatches:
            return ""

    def _sel(self, wid: str, fallback: str = "") -> str:
        try:
            v = self.query_one(f"#{wid}", Select).value
            return str(v) if v is not Select.NULL else fallback
        except NoMatches:
            return fallback

    def _chk(self, wid: str) -> bool:
        try:
            return self.query_one(f"#{wid}", Checkbox).value
        except NoMatches:
            return False

    def _sw(self, wid: str) -> bool:
        try:
            return self.query_one(f"#{wid}", Switch).value
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
    app = VidToSubTUI()
    app.run()


if __name__ == "__main__":
    main()
