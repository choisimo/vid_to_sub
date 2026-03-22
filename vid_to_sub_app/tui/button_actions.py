"""Button ID constants and action dispatch metadata for VidToSubApp.

This module centralises every button identity and the runtime rules attached
to it so that ``on_button_pressed`` can be a thin, dumb dispatcher instead of
a growing elif chain.

Structure
---------
* ``ButtonId``   – StrEnum of every static ``Button`` id in the compose tree.
                   Dynamic ids (``selrm-*``, ``radd-*``, ``sgo-*``, etc.) are
                   handled by prefix checks in the dispatcher and are therefore
                   intentionally excluded.
* ``ActionSpec`` – Frozen dataclass that attaches metadata to each button:
                   which handler method to call, and what post-processing the
                   dispatcher should run automatically.
* Domain maps   – One dict per logical tab/domain, each mapping a ``ButtonId``
                   to an ``ActionSpec``.
* ``ALL_ACTIONS`` – Merged view used by the dispatcher; built at import time.
                    Duplicate ids across domain maps raise ``AssertionError``
                    so misregistrations are caught on first import, not at
                    click-time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import auto
from sys import version_info

# StrEnum backport for Python < 3.11
if version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Button IDs
# ---------------------------------------------------------------------------


class ButtonId(StrEnum):
    """Static button identifiers as they appear in ``Button(id=...)``.

    Keep in alphabetical order within each section.
    """

    # ── Browse tab ──────────────────────────────────────────────────
    TREE_GO = "btn-tree-go"
    ADD_SEL = "btn-add-sel"
    MANUAL_ADD = "btn-manual-add"
    CLEAR_PATHS = "btn-clear-paths"
    SEARCH_PATHS = "btn-search-paths"
    CLEAR_SEARCH = "btn-clear-search"

    # ── Setup tab ───────────────────────────────────────────────────
    REDETECT = "btn-redetect"
    AUTO_SETUP = "btn-auto-setup"
    FULL_SETUP = "btn-full-setup"
    BUILD_WHISPER = "btn-build-whisper"
    DOWNLOAD_MODEL = "btn-download-model"
    PIP_FW = "btn-pip-fw"
    PIP_WHISPER = "btn-pip-whisper"
    PIP_WX = "btn-pip-wx"

    # ── Run bar (persistent footer) ─────────────────────────────────
    RUN = "btn-run"
    DRY_RUN = "btn-dry-run"
    KILL = "btn-kill"
    TOGGLE_RUN_SHELL = "btn-toggle-run-shell"

    # ── History tab ─────────────────────────────────────────────────
    HIST_REFRESH = "btn-hist-refresh"
    HIST_LOAD = "btn-hist-load"
    HIST_RERUN = "btn-hist-rerun"
    HIST_TRANSLATE = "btn-hist-translate"
    HIST_SELECT_MODE = "btn-hist-select-mode"
    HIST_COPY = "btn-hist-copy"
    HIST_CLEAR = "btn-hist-clear"
    HIST_DELETE = "btn-hist-delete"

    # ── Settings / SSH tab ──────────────────────────────────────────
    STG_SAVE = "btn-stg-save"
    STG_RELOAD = "btn-stg-reload"
    EXPORT_ENV = "btn-export-env"
    IMPORT_ENV = "btn-import-env"
    SSH_ADD = "btn-ssh-add"
    SSH_UPDATE = "btn-ssh-update"
    SSH_DELETE = "btn-ssh-delete"
    SSH_CLEAR = "btn-ssh-clear"

    # ── Agent tab ───────────────────────────────────────────────────
    AGENT_PLAN = "btn-agent-plan"
    AGENT_APPLY = "btn-agent-apply"
    AGENT_CLEAR = "btn-agent-clear"


# ---------------------------------------------------------------------------
# Action specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSpec:
    """Metadata attached to a button action.

    Parameters
    ----------
    handler_name:
        Name of the method to call on ``VidToSubApp``.  Must exist at app
        startup or ``_validate_button_handlers()`` will raise.
    refresh_remote_after:
        When *True* the dispatcher calls ``_refresh_remote_state()`` after the
        handler returns.  Set this for any action that modifies SSH connections
        or the remote-resources list.
    """

    handler_name: str
    refresh_remote_after: bool = False


# ---------------------------------------------------------------------------
# Domain-separated action maps
# ---------------------------------------------------------------------------

BROWSE_ACTIONS: dict[ButtonId, ActionSpec] = {
    ButtonId.TREE_GO: ActionSpec("_action_tree_go"),
    ButtonId.ADD_SEL: ActionSpec("_action_add_sel"),
    ButtonId.MANUAL_ADD: ActionSpec("_action_manual_add"),
    ButtonId.CLEAR_PATHS: ActionSpec("_action_clear_paths"),
    ButtonId.SEARCH_PATHS: ActionSpec("_start_path_search"),
    ButtonId.CLEAR_SEARCH: ActionSpec("_action_clear_search"),
}

SETUP_ACTIONS: dict[ButtonId, ActionSpec] = {
    ButtonId.REDETECT: ActionSpec("_run_detection"),
    ButtonId.AUTO_SETUP: ActionSpec("_action_auto_setup"),
    ButtonId.FULL_SETUP: ActionSpec("_action_full_setup"),
    ButtonId.BUILD_WHISPER: ActionSpec("_action_build_whisper"),
    ButtonId.DOWNLOAD_MODEL: ActionSpec("_action_download_model"),
    ButtonId.PIP_FW: ActionSpec("_action_pip_fw"),
    ButtonId.PIP_WHISPER: ActionSpec("_action_pip_whisper"),
    ButtonId.PIP_WX: ActionSpec("_action_pip_wx"),
}

RUN_ACTIONS: dict[ButtonId, ActionSpec] = {
    ButtonId.RUN: ActionSpec("action_run"),
    ButtonId.DRY_RUN: ActionSpec("action_dry_run"),
    ButtonId.KILL: ActionSpec("action_kill"),
    ButtonId.TOGGLE_RUN_SHELL: ActionSpec("_action_toggle_run_shell"),
}

HISTORY_ACTIONS: dict[ButtonId, ActionSpec] = {
    ButtonId.HIST_REFRESH: ActionSpec("_action_hist_refresh"),
    ButtonId.HIST_LOAD: ActionSpec("_action_hist_load"),
    ButtonId.HIST_RERUN: ActionSpec("_action_hist_rerun"),
    ButtonId.HIST_TRANSLATE: ActionSpec("_action_hist_translate"),
    ButtonId.HIST_SELECT_MODE: ActionSpec("_action_hist_select_mode"),
    ButtonId.HIST_COPY: ActionSpec("_action_hist_copy"),
    ButtonId.HIST_CLEAR: ActionSpec("_action_hist_clear"),
    ButtonId.HIST_DELETE: ActionSpec("_action_hist_delete"),
}

SETTINGS_ACTIONS: dict[ButtonId, ActionSpec] = {
    ButtonId.STG_SAVE: ActionSpec("_save_settings", refresh_remote_after=True),
    ButtonId.STG_RELOAD: ActionSpec("_action_stg_reload", refresh_remote_after=True),
    ButtonId.EXPORT_ENV: ActionSpec("_export_env"),
    ButtonId.IMPORT_ENV: ActionSpec("_import_env_to_sqlite"),
    ButtonId.SSH_ADD: ActionSpec("_ssh_add_connection", refresh_remote_after=True),
    ButtonId.SSH_UPDATE: ActionSpec(
        "_ssh_update_connection", refresh_remote_after=True
    ),
    ButtonId.SSH_DELETE: ActionSpec(
        "_ssh_delete_connection", refresh_remote_after=True
    ),
    ButtonId.SSH_CLEAR: ActionSpec("_ssh_clear_form"),
}

AGENT_ACTIONS: dict[ButtonId, ActionSpec] = {
    ButtonId.AGENT_PLAN: ActionSpec("_request_agent_plan"),
    ButtonId.AGENT_APPLY: ActionSpec("_apply_agent_plan"),
    ButtonId.AGENT_CLEAR: ActionSpec("_clear_agent_ui"),
}

# ---------------------------------------------------------------------------
# Merged lookup (validated at import time)
# ---------------------------------------------------------------------------

_DOMAIN_MAPS: list[dict[ButtonId, ActionSpec]] = [
    BROWSE_ACTIONS,
    SETUP_ACTIONS,
    RUN_ACTIONS,
    HISTORY_ACTIONS,
    SETTINGS_ACTIONS,
    AGENT_ACTIONS,
]

ALL_ACTIONS: dict[ButtonId, ActionSpec] = {}
for _domain in _DOMAIN_MAPS:
    for _bid, _spec in _domain.items():
        assert _bid not in ALL_ACTIONS, (
            f"Duplicate ButtonId registration: {_bid!r} already registered as "
            f"{ALL_ACTIONS[_bid].handler_name!r}"
        )
        ALL_ACTIONS[_bid] = _spec

del _domain, _bid, _spec  # clean up loop vars from module namespace
