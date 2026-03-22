from __future__ import annotations

_CSS = """
/* ── Global ─────────────────────────────────────── */
Screen {
    layout: vertical;
    background: $background;
    color: $foreground;
}
Header {
    background: $panel;
    color: $foreground;
    border-bottom: hkey $panel;
}
HeaderIcon {
    width: 3;
    padding: 0 1 0 0;
    color: $accent;
}
HeaderClockSpace {
    width: 0;
    padding: 0;
}
HeaderTitle {
    padding: 0 1;
    content-align: left middle;
}
Footer {
    background: $panel;
    color: $text-muted;
}
Tabs {
    height: 1;
    background: $panel;
}
Tab {
    padding: 0 1;
    color: $text-muted;
}
Tab.-active {
    background: $primary;
    color: $foreground;
    text-style: bold;
}
Underline {
    display: none;
    height: 0;
}
Button {
    min-width: 4;
    height: 1;
    padding: 0 0;
    border: none !important;
    background: $surface-darken-1;
    color: $foreground;
    text-style: bold;
}
Button:hover {
    background: $surface-lighten-1;
}
Button:focus {
    background: $primary;
    color: $foreground;
    text-style: reverse bold;
}
Button.-primary,
Button.-primary:hover,
Button.-primary:focus {
    background: $primary-muted;
    color: $text-primary;
}
Button.-success,
Button.-success:hover,
Button.-success:focus {
    background: $success-muted;
    color: $text-success;
}
Button.-warning,
Button.-warning:hover,
Button.-warning:focus {
    background: $warning-muted;
    color: $text-warning;
}
Button.-error,
Button.-error:hover,
Button.-error:focus {
    background: $error-muted;
    color: $text-error;
}
Input {
    height: 1;
    padding: 0 1;
    border: none !important;
    background: $surface-darken-1;
}
Input:focus {
    border: none !important;
    background: $surface-lighten-1;
}
SelectCurrent {
    border: none !important;
    padding: 0 1;
    background: $surface-darken-1;
}
Select:focus > SelectCurrent {
    border: none !important;
    background: $surface-lighten-1;
}
Select > SelectOverlay {
    max-height: 10;
    border: none;
    background: $surface-darken-1;
}
TextArea {
    border: none !important;
    padding: 0 1;
    background: $surface-darken-1;
}
TextArea:focus {
    border: none !important;
    background: $surface;
}
RichLog {
    border: none;
    padding: 0 1;
    background: $surface-darken-1;
}
#tab-area { height: 1fr; min-height: 18; }

/* ── Scrollable tab body ─────────────────────────── */
.tab-body { padding: 0 1; }

/* ── Section titles ──────────────────────────────── */
.stitle {
    text-style: bold;
    color: $accent;
    margin-top: 0;
    padding: 0 0 0 1;
    background: $surface-darken-1;
    height: 1;
}

/* ── Form rows ───────────────────────────────────── */
.frow {
    height: auto;
    margin-bottom: 0;
    align: left middle;
}
.flabel {
    width: 16;
    padding-right: 1;
    color: $text-muted;
    text-align: right;
    overflow: hidden;
}
.fwidget { width: 1fr; }
.hint {
    height: auto;
    padding: 0 1;
    color: $text-muted;
}

/* ── Check / switch rows ─────────────────────────── */
.crow {
    height: auto;
    padding: 0 1;
    margin-bottom: 0;
    align: left middle;
}
.crow Checkbox { margin-right: 1; }
.crow Switch   { margin-right: 1; }
.crow Label    { margin-right: 1; color: $text-muted; }

/* ── Format checkboxes ───────────────────────────── */
#fmt-row { height: auto; padding: 0 1; margin-bottom: 0; }
#fmt-row Checkbox { margin-right: 1; }

/* ── Browse tab ──────────────────────────────────── */
#browse-split { layout: horizontal; height: 1fr; }
#tree-pane {
    width: 40%;
    height: 1fr;
    border-right: solid $primary;
}
#tree-nav {
    height: auto;
    min-height: 1;
    layout: horizontal;
    padding: 0 1 0 0;
    align: left middle;
}
#tree-nav Input { width: 1fr; }
#tree-nav Button {
    width: 4;
    min-width: 4;
    min-height: 1;
    margin-left: 1;
}
DirectoryTree { height: 1fr; }
#paths-pane { width: 60%; height: 1fr; padding: 0 1; layout: vertical; }
#sel-paths-box {
    height: 1fr;
    overflow-y: auto;
    border: solid $panel;
    padding: 0 1;
    min-height: 3;
}
.sel-row { height: auto; min-height: 1; align: left middle; }
.sel-label { width: 1fr; overflow: hidden; }
.sel-btn { width: 3; min-width: 3; }
#browse-actions { height: auto; min-height: 1; margin-top: 0; align: left middle; }
#browse-actions Button { margin-right: 1; }
#manual-add-row { height: auto; min-height: 1; align: left middle; margin-top: 0; }
#manual-add-row Input { width: 1fr; }
#manual-add-row Button { width: 5; min-width: 5; min-height: 1; margin-left: 1; }
#search-row { height: auto; min-height: 1; align: left middle; margin-top: 0; }
#search-row Input { width: 1fr; }
#search-row Button {
    width: auto;
    min-width: 6;
    min-height: 1;
    margin-left: 1;
}
#search-status { height: auto; padding: 0 1; margin-bottom: 0; color: $text-muted; }
#search-results-box {
    height: auto;
    max-height: 8;
    overflow-y: auto;
    border: solid $panel;
    padding: 0 1;
    min-height: 3;
}
#search-preview {
    height: auto;
    max-height: 6;
    overflow-y: auto;
    border: solid $panel;
    padding: 0 1;
    min-height: 4;
    margin-top: 0;
}
.search-row { height: auto; min-height: 1; align: left middle; }
.search-label { width: 1fr; overflow: hidden; }
.search-go { width: 4; min-width: 4; min-height: 1; margin-right: 1; }
.search-preview-btn { width: 6; min-width: 6; min-height: 1; margin-right: 1; }
.search-add { width: 3; min-width: 3; min-height: 1; }
#outdir-row { height: auto; margin-top: 0; layout: horizontal; align: left middle; }
#outdir-label { width: 9; color: $text-muted; }
#inp-output-dir { width: 1fr; }
#behavior-row { height: auto; padding: 0 1; margin-top: 0; margin-bottom: 0; }
#behavior-row Checkbox { margin-right: 1; }
#recent-box {
    height: auto;
    max-height: 6;
    overflow-y: auto;
    border: solid $panel;
    padding: 0 1;
}
.recent-row { height: auto; min-height: 1; align: left middle; }
.recent-label { width: 1fr; overflow: hidden; color: $text-muted; }
.recent-add { width: 3; min-width: 3; min-height: 1; }

/* ── Setup tab ───────────────────────────────────── */
.det-row {
    height: auto;
    min-height: 1;
    align: left middle;
    padding: 0 1;
    border-bottom: solid $panel;
}
.det-name { width: 16; }
.det-icon { width: 2; }
.det-info { width: 1fr; overflow: hidden; color: $text-muted; }
.det-btn  { width: 16; }
.inst-row { height: auto; min-height: 1; align: left middle; layout: horizontal; padding: 0 1; }
.inst-label { width: 12; color: $text-muted; }
.inst-field { width: 1fr; }
#auto-setup-status {
    height: auto;
    padding: 0 1;
    margin-bottom: 0;
    color: $text-muted;
}
#setup-log {
    height: 8;
    margin: 0 1 0 1;
}

/* ── Translation fields ──────────────────────────── */
#trans-fields.hidden { display: none; }
#remote-status {
    margin: 0 1;
    min-height: 1;
    border: solid $panel;
    padding: 0 1;
}

/* ── History tab ─────────────────────────────────── */
#hist-pane { layout: vertical; height: 1fr; }
#hist-actions { height: auto; min-height: 1; align: left middle; padding: 0 1; }
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
#stg-actions { height: auto; min-height: 1; align: left middle; margin-top: 0; }
#stg-actions Button { margin-right: 1; }
#stg-status { height: auto; color: $success; padding: 0 1; }
#stg-remote-resources {
    height: 8;
    margin: 0 1;
}

/* ── Agent tab ───────────────────────────────────── */
#agent-help, #agent-config, #agent-status, #agent-live {
    height: auto;
    padding: 0 1;
    color: $text-muted;
}
#agent-prompt { height: 5; margin: 0 1; }
#agent-actions { height: auto; min-height: 1; align: left middle; padding: 0 1; }
#agent-actions Button { margin-right: 1; }
#agent-log {
    height: 1fr;
    min-height: 10;
    margin: 0 1;
}

/* ── Bottom panel ────────────────────────────────── */
#bottom {
    height: 12;
    border-top: solid $primary;
    padding: 0 1;
    layout: vertical;
}
#bottom.hidden-on-logs {
    display: none;
}
#run-toolbar { height: auto; min-height: 1; margin-top: 0; align: left middle; }
#run-overview {
    width: 1fr;
    height: auto;
    min-height: 1;
    padding: 0 1;
    margin-left: 1;
    color: $text-muted;
    background: $surface-darken-1;
}
#run-shell {
    height: auto;
    padding: 0 1;
    margin-top: 0;
    color: $text-muted;
    max-height: 4;
    overflow-y: auto;
    background: $surface-darken-1;
}
#run-command-panel {
    height: auto;
    margin-bottom: 0;
}
#run-command-panel.collapsed {
    display: none;
}
#run-progress {
    height: 1;
    padding: 0 1;
    margin-bottom: 0;
}
#run-active-box {
    height: 3;
    min-height: 2;
    padding: 0 1;
    margin-bottom: 0;
    overflow-y: auto;
    background: $surface-darken-1;
}
.run-active-row {
    height: 1;
    color: $text-muted;
}
#run-btns { height: auto; min-height: 1; margin-bottom: 0; align: left middle; }
#run-btns Button { margin-right: 1; }
#btn-toggle-run-shell { min-width: 8; }
#log-full { height: 1fr; }
#log { height: 1fr; min-height: 4; background: $surface-darken-1; }
"""


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
