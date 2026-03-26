from __future__ import annotations

from importlib import import_module

from .helpers import (
    ENV_POST_KEY,
    ENV_POST_MOD,
    ENV_POST_URL,
    ENV_TRANS_KEY,
    ENV_TRANS_MOD,
    ENV_TRANS_URL,
    RunConfig,
    RunJobState,
    SSHConnection,
    ssh_connection_from_row,
    build_search_preview,
    build_system_install_commands,
    detect_all,
    discover_ggml_models,
    discover_input_matches,
    extract_json_payload,
    map_path_for_remote,
    normalize_chat_endpoint,
    packages_for_manager,
    group_paths_by_video_folder,
    parse_progress_event,
    parse_remote_resources,
    partition_folder_groups_by_capacity,
    partition_paths_by_capacity,
    summarize_ggml_models,
)

_APP_EXPORTS = ("VidToSubApp", "main")

__all__ = [
    "ENV_POST_KEY",
    "ENV_POST_MOD",
    "ENV_POST_URL",
    "ENV_TRANS_KEY",
    "ENV_TRANS_MOD",
    "ENV_TRANS_URL",
    "RunConfig",
    "RunJobState",
    "SSHConnection",
    "ssh_connection_from_row",
    "build_search_preview",
    "build_system_install_commands",
    "detect_all",
    "discover_ggml_models",
    "discover_input_matches",
    "extract_json_payload",
    "map_path_for_remote",
    "normalize_chat_endpoint",
    "packages_for_manager",
    "group_paths_by_video_folder",
    "parse_progress_event",
    "parse_remote_resources",
    "partition_folder_groups_by_capacity",
    "partition_paths_by_capacity",
    "summarize_ggml_models",
    *_APP_EXPORTS,
]


def _load_app_module():
    return import_module("vid_to_sub_app.tui.app")


def __getattr__(name: str):
    if name in _APP_EXPORTS:
        return getattr(_load_app_module(), name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
