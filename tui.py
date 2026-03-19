#!/usr/bin/env python3
from __future__ import annotations

from importlib import import_module

PUBLIC_EXPORTS = [
    "ENV_POST_KEY",
    "ENV_POST_MOD",
    "ENV_POST_URL",
    "ENV_TRANS_KEY",
    "ENV_TRANS_MOD",
    "ENV_TRANS_URL",
    "RunConfig",
    "RunJobState",
    "VidToSubApp",
    "build_search_preview",
    "build_system_install_commands",
    "detect_all",
    "discover_ggml_models",
    "discover_input_matches",
    "extract_json_payload",
    "group_paths_by_video_folder",
    "main",
    "map_path_for_remote",
    "normalize_chat_endpoint",
    "packages_for_manager",
    "parse_progress_event",
    "parse_remote_resources",
    "partition_folder_groups_by_capacity",
    "partition_paths_by_capacity",
    "summarize_ggml_models",
]

__all__ = PUBLIC_EXPORTS


def _load_public_module():
    return import_module("vid_to_sub_app.tui")


if __name__ != "__main__":
    _public = _load_public_module()
    for _name in PUBLIC_EXPORTS:
        globals()[_name] = getattr(_public, _name)


def __getattr__(name: str):
    if name in PUBLIC_EXPORTS:
        return getattr(_load_public_module(), name)
    raise AttributeError(name)


def main() -> int:
    from init_checker import bootstrap_runtime

    bootstrap_runtime(requirement_groups=("base",))
    module = _load_public_module()
    return int(module.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
