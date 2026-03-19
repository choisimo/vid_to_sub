from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from db import Database
from vid_to_sub import (
    ENV_WHISPER_CPP_BIN,
    ENV_WHISPER_CPP_MODEL,
    EVENT_PREFIX,
    build_parser,
    build_run_manifest,
    find_whisper_cpp_model_path,
    find_whisper_cpp_bin,
    hash_video_folder,
    load_project_env,
    parse_whisper_cpp_progress_seconds,
    transcribe_whisper_cpp,
)
from vid_to_sub_app.cli.transcription import (
    resolve_device_fw,
    transcribe_faster_whisper,
    transcribe_openai_whisper,
)
from vid_to_sub_app.cli.translation import translate_segments_openai_compatible
from vid_to_sub_app.shared.env import (
    faster_whisper_model_candidates,
    resolve_runtime_backend_and_device,
    resolve_runtime_model,
    resolve_runtime_backend_threads,
)
from tui import (
    ENV_POST_KEY,
    ENV_POST_MOD,
    ENV_POST_URL,
    ENV_TRANS_KEY,
    ENV_TRANS_MOD,
    ENV_TRANS_URL,
    RunConfig,
    RunJobState,
    VidToSubApp,
    build_search_preview,
    build_system_install_commands,
    discover_ggml_models,
    discover_input_matches,
    extract_json_payload,
    map_path_for_remote,
    normalize_chat_endpoint,
    parse_progress_event,
    parse_remote_resources,
    partition_folder_groups_by_capacity,
    partition_paths_by_capacity,
    packages_for_manager,
    group_paths_by_video_folder,
    summarize_ggml_models,
)


class TuiHelperTests(unittest.TestCase):
    def test_discover_input_matches_returns_matching_dirs_and_video_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            season_dir = root / "Season One"
            season_dir.mkdir()
            episode = season_dir / "Episode 01.mp4"
            episode.write_bytes(b"video")
            (season_dir / "notes.txt").write_text("ignore", encoding="utf-8")

            matches = discover_input_matches(root, "season")

            self.assertIn(str(season_dir.resolve()), matches)
            self.assertNotIn(str((season_dir / "notes.txt").resolve()), matches)

            episode_matches = discover_input_matches(root, "episode 01")
            self.assertEqual([str(episode.resolve())], episode_matches)

    def test_discover_input_matches_respects_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for idx in range(5):
                folder = root / f"clip-{idx}"
                folder.mkdir()
                (folder / f"clip-{idx}.mp4").write_bytes(b"video")

            matches = discover_input_matches(root, "clip", limit=3)

            self.assertEqual(3, len(matches))

    def test_build_search_preview_for_video_file_includes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "movie.mp4"
            video.write_bytes(b"video-data")
            (root / "movie.srt").write_text("1", encoding="utf-8")

            preview = build_search_preview(video)

            self.assertIn("Video file preview", preview)
            self.assertIn("movie.mp4", preview)
            self.assertIn("movie.srt", preview)

    def test_build_search_preview_for_directory_lists_sample_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            folder = root / "Season 1"
            folder.mkdir()
            (folder / "Episode 01.mp4").write_bytes(b"video")
            (folder / "Episode 02.mkv").write_bytes(b"video")

            preview = build_search_preview(folder)

            self.assertIn("Directory preview", preview)
            self.assertIn("Episode 01.mp4", preview)
            self.assertIn("Episode 02.mkv", preview)

    def test_discover_ggml_models_prefers_first_directory_for_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            primary = root / "primary"
            secondary = root / "secondary"
            primary.mkdir()
            secondary.mkdir()
            (primary / "ggml-large-v3.bin").write_bytes(b"a")
            (secondary / "ggml-large-v3.bin").write_bytes(b"b")
            (secondary / "ggml-small.bin").write_bytes(b"c")

            found = discover_ggml_models([primary, secondary])

            self.assertEqual(
                str((primary / "ggml-large-v3.bin").resolve()),
                found["large-v3"],
            )
            self.assertEqual(
                str((secondary / "ggml-small.bin").resolve()),
                found["small"],
            )

    def test_summarize_ggml_models_includes_count(self) -> None:
        summary = summarize_ggml_models(
            {
                "base": "/tmp/ggml-base.bin",
                "large-v3": "/tmp/ggml-large-v3.bin",
                "small": "/tmp/ggml-small.bin",
            }
        )

        self.assertIn("3 detected", summary)
        self.assertIn("large-v3", summary)

    def test_packages_for_manager_and_install_commands(self) -> None:
        packages = packages_for_manager("apt-get", ["cmake", "git", "whisper-build"])
        self.assertIn("cmake", packages)
        self.assertIn("git", packages)
        self.assertIn("build-essential", packages)

        commands = build_system_install_commands(
            "apt-get",
            ["ffmpeg", "cmake"],
            use_sudo=True,
        )
        self.assertEqual(["sudo", "-n", "apt-get", "update"], commands[0])
        self.assertIn("ffmpeg", commands[1])
        self.assertIn("cmake", commands[1])

    def test_extract_json_payload_accepts_code_fence(self) -> None:
        payload = extract_json_payload(
            """```json
            {"summary":"ok","analysis":"done","actions":[]}
            ```"""
        )

        self.assertEqual("ok", payload["summary"])

    def test_normalize_chat_endpoint_appends_chat_completions(self) -> None:
        self.assertEqual(
            "https://example.com/v1/chat/completions",
            normalize_chat_endpoint("https://example.com/v1"),
        )

    def test_parse_remote_resources_filters_invalid_entries(self) -> None:
        resources = parse_remote_resources(
            """
            [
              {"name":"gpu-a","ssh_target":"user@gpu-a","remote_workdir":"/srv/vid_to_sub","slots":2},
              {"name":"disabled","ssh_target":"user@gpu-b","remote_workdir":"/srv/app","enabled":false},
              {"name":"broken","ssh_target":"","remote_workdir":"/srv/app"}
            ]
            """
        )

        self.assertEqual(1, len(resources))
        self.assertEqual("gpu-a", resources[0].name)
        self.assertEqual(2, resources[0].slots)

    def test_map_path_for_remote_uses_longest_prefix(self) -> None:
        mapped = map_path_for_remote(
            "/mnt/media/library/movie.mkv",
            {
                "/mnt/media": "/remote/media",
                "/mnt/media/library": "/remote/library",
            },
        )

        self.assertEqual("/remote/library/movie.mkv", mapped)

    def test_partition_paths_by_capacity_distributes_weighted_round_robin(self) -> None:
        assignments = partition_paths_by_capacity(
            ["a", "b", "c", "d", "e", "f"],
            [("local", 1), ("gpu", 2)],
        )

        self.assertEqual(["a", "d"], assignments["local"])
        self.assertEqual(["b", "c", "e", "f"], assignments["gpu"])

    def test_build_run_manifest_round_robins_by_hashed_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            alpha = root / "alpha"
            beta = root / "beta"
            alpha.mkdir()
            beta.mkdir()
            a1 = alpha / "a1.mp4"
            a2 = alpha / "a2.mp4"
            b1 = beta / "b1.mp4"
            for video in (a1, a2, b1):
                video.write_bytes(b"video")

            manifest = build_run_manifest([str(a1), str(a2), str(b1)])

            self.assertEqual(
                [str(a1.resolve()), str(b1.resolve()), str(a2.resolve())],
                [entry["video_path"] for entry in manifest["entries"]],
            )
            self.assertEqual(2, len(manifest["folders"]))
            self.assertEqual(hash_video_folder(alpha), manifest["folders"][0]["folder_hash"])
            self.assertEqual(2, manifest["folders"][0]["total_files"])

    def test_partition_folder_groups_by_capacity_keeps_folder_videos_together(self) -> None:
        groups = [
            {
                "folder_hash": "a",
                "folder_path": "/tmp/a",
                "videos": ["/tmp/a/1.mp4", "/tmp/a/2.mp4"],
            },
            {
                "folder_hash": "b",
                "folder_path": "/tmp/b",
                "videos": ["/tmp/b/1.mp4"],
            },
        ]

        assignments = partition_folder_groups_by_capacity(
            groups,
            [("local", 1), ("gpu", 1)],
        )

        combined = [*assignments["local"], *assignments["gpu"]]

        self.assertCountEqual(
            ["/tmp/a/1.mp4", "/tmp/a/2.mp4", "/tmp/b/1.mp4"],
            combined,
        )
        self.assertTrue(
            assignments["local"] in (
                ["/tmp/a/1.mp4", "/tmp/a/2.mp4"],
                ["/tmp/b/1.mp4"],
            )
        )
        self.assertTrue(
            assignments["gpu"] in (
                ["/tmp/a/1.mp4", "/tmp/a/2.mp4"],
                ["/tmp/b/1.mp4"],
            )
        )

    def test_group_paths_by_video_folder_hashes_resolved_parent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            season = root / "Season 01"
            season.mkdir()
            ep1 = season / "ep1.mp4"
            ep2 = season / "ep2.mp4"
            ep1.write_bytes(b"video")
            ep2.write_bytes(b"video")

            groups = group_paths_by_video_folder([str(ep2), str(ep1)])

            self.assertEqual(1, len(groups))
            self.assertEqual(hash_video_folder(season), groups[0]["folder_hash"])
            self.assertEqual(
                [str(ep1.resolve()), str(ep2.resolve())],
                groups[0]["videos"],
            )

    def test_parse_progress_event_extracts_json_payload(self) -> None:
        event = parse_progress_event(
            f'{EVENT_PREFIX} {{"event":"job_finished","video_path":"/tmp/a.mp4","status":"done"}}'
        )

        self.assertEqual("job_finished", event["event"])
        self.assertEqual("/tmp/a.mp4", event["video_path"])

    def test_parse_whisper_cpp_progress_seconds_reads_segment_end_timestamp(self) -> None:
        seconds = parse_whisper_cpp_progress_seconds(
            "[00:00:03.000 --> 00:00:11.250]   hello world"
        )

        self.assertAlmostEqual(11.25, seconds or 0.0, places=3)

    def test_discover_videos_for_run_uses_snapshot_config(self) -> None:
        app = VidToSubApp()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            kept = root / "keep.mp4"
            skipped = root / "skip.mp4"
            kept.write_bytes(b"video")
            skipped.write_bytes(b"video")
            (root / "skip.srt").write_text("1", encoding="utf-8")

            config = RunConfig(
                request_id=1,
                selected_paths=[str(root)],
                output_dir=None,
                formats=frozenset({"srt"}),
                no_recurse=False,
                skip_existing=True,
                dry_run=False,
                verbose=False,
                backend="whisper-cpp",
                model="large-v3",
                device="cpu",
                language=None,
                compute_type=None,
                beam_size="5",
                local_workers=2,
                whisper_cpp_model_path=None,
                translate_enabled=False,
                translate_to=None,
                translation_model=None,
                translation_base_url=None,
                translation_api_key=None,
                postprocess_enabled=False,
                postprocess_mode="auto",
                postprocess_model=None,
                postprocess_base_url=None,
                postprocess_api_key=None,
                diarize=False,
                hf_token=None,
                execution_mode="local",
                remote_resources=[],
                run_env={},
            )

            videos, found_total, skipped_count = app._discover_videos_for_run(config)

            self.assertEqual(2, found_total)
            self.assertEqual(1, skipped_count)
            self.assertEqual([str(kept.resolve())], videos)

    def test_build_executor_plans_uses_snapshot_config(self) -> None:
        app = VidToSubApp()
        config = RunConfig(
            request_id=7,
            selected_paths=["/tmp/input.mp4"],
            output_dir="/tmp/out",
            formats=frozenset({"srt", "json"}),
            no_recurse=True,
            skip_existing=True,
            dry_run=False,
            verbose=True,
            backend="faster-whisper",
            model="turbo",
            device="cpu",
            language="en",
            compute_type="int8",
            beam_size="7",
            local_workers=3,
            whisper_cpp_model_path=None,
            translate_enabled=True,
            translate_to="ko",
            translation_model="gpt-4.1-mini",
            translation_base_url="https://example.com/v1",
            translation_api_key="secret",
            postprocess_enabled=True,
            postprocess_mode="web_lookup",
            postprocess_model="gpt-4.1",
            postprocess_base_url="https://post.example/v1",
            postprocess_api_key="post-secret",
            diarize=False,
            hf_token=None,
            execution_mode="local",
            remote_resources=[],
            run_env={
                "VID_TO_SUB_TRANSLATION_MODEL": "gpt-4.1-mini",
                "VID_TO_SUB_POSTPROCESS_MODEL": "gpt-4.1",
            },
        )

        plans = app._build_executor_plans(
            ["/tmp/input.mp4"],
            dry_run=False,
            config=config,
        )

        self.assertEqual(1, len(plans))
        self.assertEqual("local", plans[0].label)
        self.assertIn("--manifest-stdin", plans[0].cmd)
        self.assertIn("--workers", plans[0].cmd)
        self.assertIn("--backend-threads", plans[0].cmd)
        self.assertIn("--translate-to", plans[0].cmd)
        self.assertIn("--postprocess-translation", plans[0].cmd)
        self.assertIn("--postprocess-mode", plans[0].cmd)
        self.assertIn("--postprocess-model", plans[0].cmd)
        self.assertEqual(
            {
                "VID_TO_SUB_TRANSLATION_MODEL": "gpt-4.1-mini",
                "VID_TO_SUB_POSTPROCESS_MODEL": "gpt-4.1",
            },
            plans[0].env,
        )
        self.assertNotIn("/tmp/input.mp4", plans[0].cmd)
        self.assertIsNotNone(plans[0].stdin_payload)
        self.assertIn("/tmp/input.mp4", plans[0].stdin_payload or "")

    def test_sync_transcribe_overrides_updates_inherited_values_only(self) -> None:
        app = VidToSubApp()
        widgets = {
            "#inp-trans-url": SimpleNamespace(value="https://old.example/v1"),
            "#inp-trans-key": SimpleNamespace(value="manual-key"),
            "#inp-trans-model": SimpleNamespace(value="old-model"),
            "#inp-post-url": SimpleNamespace(value="https://old-post.example/v1"),
            "#inp-post-key": SimpleNamespace(value="manual-post-key"),
            "#inp-post-model": SimpleNamespace(value="old-post-model"),
            "#inp-wcpp-bin": SimpleNamespace(value=""),
        }

        def query_one(selector: str, *_args):
            return widgets[selector]

        with patch.object(app, "query_one", side_effect=query_one):
            app._sync_transcribe_overrides_from_settings(
                previous_values={
                    ENV_TRANS_URL: "https://old.example/v1",
                    ENV_TRANS_KEY: "saved-key",
                    ENV_TRANS_MOD: "old-model",
                    ENV_POST_URL: "https://old-post.example/v1",
                    ENV_POST_KEY: "saved-post-key",
                    ENV_POST_MOD: "old-post-model",
                    ENV_WHISPER_CPP_BIN: "/old/bin/whisper-cli",
                },
                updated_values={
                    ENV_TRANS_URL: "https://new.example/v1",
                    ENV_TRANS_KEY: "new-key",
                    ENV_TRANS_MOD: "new-model",
                    ENV_POST_URL: "https://new-post.example/v1",
                    ENV_POST_KEY: "new-post-key",
                    ENV_POST_MOD: "new-post-model",
                    ENV_WHISPER_CPP_BIN: "/new/bin/whisper-cli",
                },
            )

        self.assertEqual("https://new.example/v1", widgets["#inp-trans-url"].value)
        self.assertEqual("manual-key", widgets["#inp-trans-key"].value)
        self.assertEqual("new-model", widgets["#inp-trans-model"].value)
        self.assertEqual("https://new-post.example/v1", widgets["#inp-post-url"].value)
        self.assertEqual("manual-post-key", widgets["#inp-post-key"].value)
        self.assertEqual("new-post-model", widgets["#inp-post-model"].value)
        self.assertEqual("/new/bin/whisper-cli", widgets["#inp-wcpp-bin"].value)

    def test_sync_run_defaults_updates_inherited_values_only(self) -> None:
        app = VidToSubApp()
        widgets = {
            "#inp-output-dir": SimpleNamespace(value="/old/output"),
            "#inp-translate-to": SimpleNamespace(value="ja"),
        }

        def query_one(selector: str, *_args):
            return widgets[selector]

        with patch.object(app, "query_one", side_effect=query_one):
            app._sync_run_defaults_from_settings(
                previous_values={
                    "tui.default_output_dir": "/old/output",
                    "tui.default_translate_to": "ko",
                },
                updated_values={
                    "tui.default_output_dir": "/new/output",
                    "tui.default_translate_to": "en",
                },
            )

        self.assertEqual("/new/output", widgets["#inp-output-dir"].value)
        self.assertEqual("ja", widgets["#inp-translate-to"].value)

    def test_prefill_transcribe_reads_sqlite_settings(self) -> None:
        app = VidToSubApp()
        widgets = {
            "#inp-trans-url": SimpleNamespace(value=""),
            "#inp-trans-key": SimpleNamespace(value=""),
            "#inp-trans-model": SimpleNamespace(value=""),
            "#inp-post-url": SimpleNamespace(value=""),
            "#inp-post-key": SimpleNamespace(value=""),
            "#inp-post-model": SimpleNamespace(value=""),
            "#inp-wcpp-bin": SimpleNamespace(value=""),
            "#inp-output-dir": SimpleNamespace(value=""),
            "#inp-translate-to": SimpleNamespace(value=""),
        }

        def query_one(selector: str, *_args):
            return widgets[selector]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_db = Database(Path(tmpdir) / "state.db")
            temp_db.set_many(
                {
                    ENV_TRANS_URL: "https://sqlite.example/v1",
                    ENV_TRANS_KEY: "sqlite-key",
                    ENV_TRANS_MOD: "sqlite-model",
                    ENV_POST_URL: "https://sqlite-post.example/v1",
                    ENV_POST_KEY: "sqlite-post-key",
                    ENV_POST_MOD: "sqlite-post-model",
                    ENV_WHISPER_CPP_BIN: "/sqlite/bin/whisper-cli",
                    "tui.default_output_dir": "/sqlite/output",
                    "tui.default_translate_to": "ko",
                }
            )

            with patch("vid_to_sub_app.tui.app._db", temp_db), patch.object(
                app, "query_one", side_effect=query_one
            ):
                app._prefill_transcribe()

        self.assertEqual("https://sqlite.example/v1", widgets["#inp-trans-url"].value)
        self.assertEqual("sqlite-key", widgets["#inp-trans-key"].value)
        self.assertEqual("sqlite-model", widgets["#inp-trans-model"].value)
        self.assertEqual("https://sqlite-post.example/v1", widgets["#inp-post-url"].value)
        self.assertEqual("sqlite-post-key", widgets["#inp-post-key"].value)
        self.assertEqual("sqlite-post-model", widgets["#inp-post-model"].value)
        self.assertEqual("/sqlite/bin/whisper-cli", widgets["#inp-wcpp-bin"].value)
        self.assertEqual("/sqlite/output", widgets["#inp-output-dir"].value)
        self.assertEqual("ko", widgets["#inp-translate-to"].value)

    def test_build_run_env_reads_sqlite_settings(self) -> None:
        app = VidToSubApp()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_db = Database(Path(tmpdir) / "state.db")
            temp_db.set_many(
                {
                    ENV_TRANS_URL: "https://sqlite.example/v1",
                    ENV_TRANS_KEY: "sqlite-key",
                    ENV_TRANS_MOD: "sqlite-model",
                    ENV_POST_URL: "https://sqlite-post.example/v1",
                    ENV_POST_KEY: "sqlite-post-key",
                    ENV_POST_MOD: "sqlite-post-model",
                    ENV_WHISPER_CPP_BIN: "/sqlite/bin/whisper-cli",
                }
            )

            values = {
                "inp-wcpp-bin": "",
                "inp-trans-url": "",
                "inp-trans-key": "",
                "inp-trans-model": "",
                "inp-post-url": "",
                "inp-post-key": "",
                "inp-post-model": "",
            }
            with patch("vid_to_sub_app.tui.app._db", temp_db), patch.object(
                app, "_val", side_effect=lambda wid: values.get(wid, "")
            ), patch.object(
                app, "_sel", return_value="faster-whisper"
            ), patch.object(
                app, "_resolved_wcpp_model_path", return_value=None
            ):
                env = app._build_run_env()

        self.assertEqual("https://sqlite.example/v1", env[ENV_TRANS_URL])
        self.assertEqual("sqlite-key", env[ENV_TRANS_KEY])
        self.assertEqual("sqlite-model", env[ENV_TRANS_MOD])
        self.assertEqual("https://sqlite-post.example/v1", env[ENV_POST_URL])
        self.assertEqual("sqlite-post-key", env[ENV_POST_KEY])
        self.assertEqual("sqlite-post-model", env[ENV_POST_MOD])
        self.assertEqual("/sqlite/bin/whisper-cli", env[ENV_WHISPER_CPP_BIN])

    def test_refresh_history_shows_progress_for_running_job(self) -> None:
        app = VidToSubApp()
        app._active_jobs["local:/tmp/a.mp4"] = RunJobState(
            video_path="/tmp/a.mp4",
            executor="local",
            job_id=11,
            started_at=0.0,
            video_duration=100.0,
            progress_seconds=42.0,
            progress_ratio=0.42,
        )

        class FakeTable:
            def __init__(self) -> None:
                self.rows: list[tuple[str, ...]] = []

            def clear(self, *, columns: bool = False) -> None:
                self.rows.clear()

            def add_row(self, *args, **_kwargs) -> None:
                self.rows.append(args)

        table = FakeTable()
        count_updates: list[str] = []

        def query_one(selector: str, *_args):
            if selector == "#hist-table":
                return table
            if selector == "#hist-count":
                return SimpleNamespace(update=lambda value: count_updates.append(value))
            raise AssertionError(f"Unexpected selector: {selector}")

        job = {
            "id": 11,
            "created_at": "2026-03-18T13:57:49",
            "video_path": "/tmp/a.mp4",
            "backend": "whisper-cpp",
            "model": "large-v3",
            "status": "running",
            "wall_sec": None,
        }

        with patch("vid_to_sub_app.tui.app._db.get_jobs", return_value=[job]), patch.object(
            app, "query_one", side_effect=query_one
        ):
            app._refresh_history()

        self.assertEqual(1, len(table.rows))
        self.assertIn("42.0%", table.rows[0][5])
        self.assertTrue(count_updates)

    def test_show_hist_detail_includes_progress_for_active_job(self) -> None:
        app = VidToSubApp()
        app._active_jobs["local:/tmp/a.mp4"] = RunJobState(
            video_path="/tmp/a.mp4",
            executor="local",
            job_id=11,
            started_at=0.0,
            video_duration=100.0,
            progress_seconds=42.0,
            progress_ratio=0.42,
        )
        updates: list[str] = []
        job = {
            "id": 11,
            "created_at": "2026-03-18T13:57:49",
            "video_path": "/tmp/a.mp4",
            "backend": "whisper-cpp",
            "model": "large-v3",
            "status": "running",
            "language": "auto",
            "target_lang": "ko",
            "output_dir": None,
            "wall_sec": None,
            "error": None,
            "output_paths": "[]",
        }

        with patch("vid_to_sub_app.tui.app._db.get_jobs", return_value=[job]), patch.object(
            app,
            "query_one",
            return_value=SimpleNamespace(update=lambda value: updates.append(value)),
        ):
            app._show_hist_detail("11")

        self.assertTrue(updates)
        self.assertIn("Progress:", updates[-1])
        self.assertIn("42.0%", updates[-1])
        self.assertIn("00:00:42 / 00:01:40", updates[-1])

    def test_sync_run_command_panel_updates_visibility_and_label(self) -> None:
        app = VidToSubApp()

        class FakePanel:
            def __init__(self) -> None:
                self.classes: set[str] = set()

            def add_class(self, name: str) -> None:
                self.classes.add(name)

            def remove_class(self, name: str) -> None:
                self.classes.discard(name)

        panel = FakePanel()
        button = SimpleNamespace(label="")

        def query_one(selector: str, *_args):
            if selector == "#run-command-panel":
                return panel
            if selector == "#btn-toggle-run-shell":
                return button
            raise AssertionError(f"Unexpected selector: {selector}")

        with patch.object(app, "query_one", side_effect=query_one):
            app._run_shell_collapsed = True
            app._sync_run_command_panel()
            self.assertIn("collapsed", panel.classes)
            self.assertEqual("Cmd ▸", button.label)

            app._run_shell_collapsed = False
            app._sync_run_command_panel()
            self.assertNotIn("collapsed", panel.classes)
            self.assertEqual("Cmd ▾", button.label)

    def test_build_parser_leaves_backend_threads_auto_by_default(self) -> None:
        args = build_parser().parse_args(["--manifest-stdin"])

        self.assertIsNone(args.backend_threads)
        self.assertTrue(args.manifest_stdin)

    def test_resolve_runtime_backend_and_device_prefers_faster_whisper_on_cuda(self) -> None:
        with patch("vid_to_sub_app.shared.env.detect_best_device", return_value="cuda"), patch(
            "vid_to_sub_app.shared.env.detect_torch_device", return_value="cuda"
        ), patch(
            "vid_to_sub_app.shared.env.module_available",
            side_effect=lambda name: name == "faster_whisper",
        ):
            resolved = resolve_runtime_backend_and_device()

        self.assertEqual(("faster-whisper", "auto"), resolved)

    def test_resolve_runtime_backend_and_device_prefers_mps_backend_when_available(self) -> None:
        with patch("vid_to_sub_app.shared.env.detect_best_device", return_value="mps"), patch(
            "vid_to_sub_app.shared.env.detect_torch_device", return_value="mps"
        ), patch(
            "vid_to_sub_app.shared.env.module_available",
            side_effect=lambda name: name == "whisper",
        ):
            resolved = resolve_runtime_backend_and_device()

        self.assertEqual(("whisper", "auto"), resolved)

    def test_build_parser_uses_runtime_backend_and_device_defaults(self) -> None:
        with patch(
            "vid_to_sub_app.cli.main.resolve_runtime_backend_and_device",
            return_value=("faster-whisper", "auto"),
        ), patch(
            "vid_to_sub_app.cli.main.resolve_runtime_model",
            return_value="small",
        ):
            args = build_parser().parse_args(["--manifest-stdin"])

        self.assertEqual("faster-whisper", args.backend)
        self.assertEqual("auto", args.device)
        self.assertEqual("small", args.model)

    def test_faster_whisper_model_candidates_filter_by_vram(self) -> None:
        candidates = faster_whisper_model_candidates(
            "large-v3",
            available_vram_gb=4.5,
        )

        self.assertEqual(["small", "base", "tiny"], candidates)

    def test_resolve_runtime_model_picks_first_fitting_cuda_candidate(self) -> None:
        with patch(
            "vid_to_sub_app.shared.env.resolve_runtime_backend_device",
            return_value="cuda",
        ), patch(
            "vid_to_sub_app.shared.env.detect_cuda_total_memory_gb",
            return_value=4.5,
        ):
            resolved = resolve_runtime_model("faster-whisper", "auto", "large-v3")

        self.assertEqual("small", resolved)

    def test_resolve_device_fw_auto_prefers_cuda(self) -> None:
        with patch("vid_to_sub_app.cli.transcription.detect_best_device", return_value="cuda"):
            resolved = resolve_device_fw("auto")

        self.assertEqual(("cuda", "float16"), resolved)

    def test_resolve_runtime_backend_threads_uses_all_cpu_threads_by_default(self) -> None:
        with patch("vid_to_sub_app.shared.env.available_cpu_threads", return_value=12):
            resolved = resolve_runtime_backend_threads("whisper-cpp", "cpu", worker_count=1)

        self.assertEqual(12, resolved)

    def test_resolve_runtime_backend_threads_splits_cpu_threads_across_workers(self) -> None:
        with patch("vid_to_sub_app.shared.env.available_cpu_threads", return_value=12):
            resolved = resolve_runtime_backend_threads(
                "faster-whisper",
                "cpu",
                worker_count=3,
            )

        self.assertEqual(4, resolved)

    def test_transcribe_openai_whisper_auto_uses_detected_torch_device(self) -> None:
        loaded: dict[str, str] = {}

        class FakeModel:
            def transcribe(self, _video_path: str, **_kwargs):
                return {"segments": [], "language": "en"}

        fake_whisper = SimpleNamespace(
            load_model=lambda model_name, device: loaded.update(  # type: ignore[arg-type]
                {"model": model_name, "device": device}
            )
            or FakeModel()
        )

        with patch.dict("sys.modules", {"whisper": fake_whisper}), patch(
            "vid_to_sub_app.cli.transcription.detect_torch_device",
            return_value="mps",
        ):
            _segments, info = transcribe_openai_whisper(
                video=Path("/tmp/sample.mp4"),
                model_name="base",
                device="auto",
                language=None,
                beam_size=5,
                threads=8,
            )

        self.assertEqual("mps", loaded["device"])
        self.assertEqual("openai-whisper", info["backend"])

    def test_transcribe_faster_whisper_cpu_uses_requested_thread_count(self) -> None:
        captured: dict[str, object] = {}

        class FakeSegment:
            start = 0.0
            end = 1.0
            text = "hello"

        class FakeInfo:
            language = "en"
            language_probability = 0.99
            duration = 1.0

        class FakeWhisperModel:
            def __init__(self, *args, **kwargs) -> None:
                captured.update(kwargs)

            def transcribe(self, *_args, **_kwargs):
                return [FakeSegment()], FakeInfo()

        with patch.dict("sys.modules", {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)}):
            segments, info = transcribe_faster_whisper(
                video=Path("/tmp/sample.mp4"),
                model_name="base",
                device="cpu",
                language=None,
                beam_size=5,
                compute_type=None,
                threads=10,
            )

        self.assertEqual(10, captured["cpu_threads"])
        self.assertEqual("faster-whisper", info["backend"])
        self.assertEqual("hello", segments[0]["text"])

    def test_transcribe_faster_whisper_logs_model_loading_hint(self) -> None:
        class FakeSegment:
            start = 0.0
            end = 1.0
            text = "hello"

        class FakeInfo:
            language = "en"
            language_probability = 0.99
            duration = 1.0

        class FakeWhisperModel:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def transcribe(self, *_args, **_kwargs):
                return [FakeSegment()], FakeInfo()

        captured_stdout = io.StringIO()
        with patch.dict(
            "sys.modules",
            {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)},
        ), redirect_stdout(captured_stdout):
            _segments, _info = transcribe_faster_whisper(
                video=Path("/tmp/sample.mp4"),
                model_name="large-v3",
                device="cpu",
                language=None,
                beam_size=5,
                compute_type=None,
                threads=4,
            )

        self.assertIn("Loading faster-whisper model 'large-v3'", captured_stdout.getvalue())

    def test_transcribe_faster_whisper_reports_incomplete_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "models--Systran--faster-whisper-large-v3"
            snapshot_dir = cache_root / "snapshots" / "abc123"
            blob_dir = cache_root / "blobs"
            snapshot_dir.mkdir(parents=True)
            blob_dir.mkdir(parents=True)
            incomplete_blob = blob_dir / "broken-model.incomplete"
            incomplete_blob.write_bytes(b"partial")

            class FakeWhisperModel:
                def __init__(self, *_args, **_kwargs) -> None:
                    raise RuntimeError(
                        f"Unable to open file 'model.bin' in model '{snapshot_dir}'"
                    )

            with patch.dict(
                "sys.modules",
                {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)},
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    transcribe_faster_whisper(
                        video=Path("/tmp/sample.mp4"),
                        model_name="large-v3",
                        device="cpu",
                        language=None,
                        beam_size=5,
                        compute_type=None,
                        threads=4,
                    )

        message = str(ctx.exception)
        self.assertIn("cache is incomplete or corrupted", message)
        self.assertIn(str(cache_root), message)
        self.assertIn(incomplete_blob.name, message)

    def test_transcribe_faster_whisper_auto_falls_back_to_cpu_after_cuda_oom(self) -> None:
        attempts: list[tuple[str, str]] = []

        class FakeSegment:
            start = 0.0
            end = 1.0
            text = "hello"

        class FakeInfo:
            language = "en"
            language_probability = 0.99
            duration = 1.0

        class FakeWhisperModel:
            def __init__(self, model_name: str, *, device: str, **_kwargs) -> None:
                attempts.append((model_name, device))
                if device == "cuda":
                    raise RuntimeError("CUDA failed with error out of memory")

            def transcribe(self, *_args, **_kwargs):
                return [FakeSegment()], FakeInfo()

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with patch.dict(
            "sys.modules",
            {"faster_whisper": SimpleNamespace(WhisperModel=FakeWhisperModel)},
        ), patch(
            "vid_to_sub_app.cli.transcription.detect_best_device",
            return_value="cuda",
        ), patch(
            "vid_to_sub_app.cli.transcription.detect_cuda_total_memory_gb",
            return_value=4.5,
        ), redirect_stdout(captured_stdout), redirect_stderr(captured_stderr):
            segments, info = transcribe_faster_whisper(
                video=Path("/tmp/sample.mp4"),
                model_name="large-v3",
                device="auto",
                language=None,
                beam_size=5,
                compute_type=None,
                threads=4,
            )

        self.assertEqual(
            [
                ("small", "cuda"),
                ("base", "cuda"),
                ("tiny", "cuda"),
                ("tiny", "cpu"),
            ],
            attempts,
        )
        self.assertEqual("hello", segments[0]["text"])
        self.assertEqual("faster-whisper", info["backend"])
        self.assertEqual("tiny", info["model"])
        self.assertIn("falling back to CPU int8", captured_stderr.getvalue())

    def test_translate_segments_openai_compatible_sends_accept_and_user_agent(self) -> None:
        captured_headers: dict[str, str] = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self) -> bytes:
                payload = {"choices": [{"message": {"content": "[\"안녕하세요\"]"}}]}
                return json.dumps(payload, ensure_ascii=False).encode("utf-8")

        def fake_urlopen(request):
            nonlocal captured_headers
            captured_headers = dict(request.header_items())
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            translated, info = translate_segments_openai_compatible(
                segments=[{"start": 0.0, "end": 1.0, "text": "hello"}],
                target_language="ko",
                translation_model="gpt-5.4",
                translation_base_url="https://example.com/v1",
                translation_api_key="secret",
                source_language="en",
            )

        self.assertEqual("안녕하세요", translated[0]["text"])
        self.assertEqual("openai-compatible-translation", info["backend"])
        self.assertEqual("application/json", captured_headers["Content-type"])
        self.assertEqual("application/json", captured_headers["Accept"])
        self.assertIn("vid_to_sub/1.0", captured_headers["User-agent"])

    def test_find_whisper_cpp_bin_ignores_empty_override_and_uses_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback = Path(tmpdir) / "whisper-cli"
            fallback.write_text("#!/bin/sh\n", encoding="utf-8")
            fallback.chmod(0o755)

            with patch("vid_to_sub_app.shared.env.shutil.which", return_value=None), patch(
                "vid_to_sub_app.shared.env.WHISPER_CLI_CANDIDATES",
                (str(fallback),),
            ):
                found = find_whisper_cpp_bin("")

        self.assertEqual(str(fallback.resolve()), found)

    def test_find_whisper_cpp_bin_falls_back_from_invalid_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            fallback = Path(tmpdir) / "whisper-cli"
            fallback.write_text("#!/bin/sh\n", encoding="utf-8")
            fallback.chmod(0o755)

            with patch("vid_to_sub_app.shared.env.shutil.which", return_value=None), patch(
                "vid_to_sub_app.shared.env.WHISPER_CLI_CANDIDATES",
                (str(fallback),),
            ):
                found = find_whisper_cpp_bin("/done")

        self.assertEqual(str(fallback.resolve()), found)

    def test_find_whisper_cpp_model_path_falls_back_from_invalid_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "ggml-large-v3.bin"
            model.write_bytes(b"model")

            found = find_whisper_cpp_model_path(
                "large-v3",
                "/missing/model.bin",
                search_dirs=[root],
            )

        self.assertEqual(str(model.resolve()), found)

    def test_find_whisper_cpp_model_path_requires_valid_explicit_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model = root / "ggml-large-v3.bin"
            model.write_bytes(b"model")

            found = find_whisper_cpp_model_path(
                "large-v3",
                "/missing/model.bin",
                search_dirs=[root],
                strict_configured=True,
            )

        self.assertIsNone(found)

    def test_load_project_env_reads_repo_dotenv_without_overriding_existing_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_file = Path(tmpdir) / ".env"
            env_file.write_text(
                'VID_TO_SUB_WHISPER_CPP_BIN="/from-dotenv"\n'
                'VID_TO_SUB_WHISPER_CPP_MODEL="/from-dotenv-model"\n',
                encoding="utf-8",
            )

            with patch("vid_to_sub_app.shared.env.ENV_FILE", env_file), patch(
                "vid_to_sub_app.shared.env.load_dotenv",
                None,
            ), patch.dict(
                os.environ,
                {ENV_WHISPER_CPP_BIN: "/keep-existing"},
                clear=False,
            ):
                os.environ.pop(ENV_WHISPER_CPP_MODEL, None)
                loaded = load_project_env()

                self.assertTrue(loaded)
                self.assertEqual("/keep-existing", os.environ[ENV_WHISPER_CPP_BIN])
                self.assertEqual("/from-dotenv-model", os.environ[ENV_WHISPER_CPP_MODEL])

    def test_transcribe_whisper_cpp_uses_fallback_when_env_override_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "clip.mp4"
            model = root / "ggml-large-v3.bin"
            fallback = root / "whisper-cli"
            video.write_bytes(b"video")
            model.write_bytes(b"model")
            fallback.write_text("#!/bin/sh\n", encoding="utf-8")
            fallback.chmod(0o755)

            def fake_extract(_video: Path, wav_path: Path) -> None:
                wav_path.write_bytes(b"wav")

            def fake_run(command: list[str], check: bool):
                self.assertTrue(check)
                self.assertEqual(str(fallback.resolve()), command[0])
                output_prefix = Path(command[command.index("-of") + 1])
                output_prefix.with_suffix(".srt").write_text(
                    "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with patch.dict(os.environ, {ENV_WHISPER_CPP_BIN: ""}, clear=False), patch(
                "vid_to_sub_app.shared.env.shutil.which",
                return_value=None,
            ), patch(
                "vid_to_sub_app.shared.env.WHISPER_CLI_CANDIDATES",
                (str(fallback),),
            ), patch(
                "vid_to_sub_app.cli.transcription.extract_audio_for_whisper_cpp",
                side_effect=fake_extract,
            ), patch(
                "vid_to_sub_app.cli.transcription.subprocess.run",
                side_effect=fake_run,
            ):
                segments, info = transcribe_whisper_cpp(
                    video=video,
                    model_name="large-v3",
                    device="cpu",
                    language=None,
                    threads=2,
                    whisper_cpp_model_path=str(model),
                )

        self.assertEqual("hello", segments[0]["text"])
        self.assertEqual(str(model.resolve()), info["model_path"])

    def test_transcribe_whisper_cpp_falls_back_from_invalid_env_bin_and_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "clip.mp4"
            model = root / "ggml-large-v3.bin"
            fallback = root / "whisper-cli"
            video.write_bytes(b"video")
            model.write_bytes(b"model")
            fallback.write_text("#!/bin/sh\n", encoding="utf-8")
            fallback.chmod(0o755)

            def fake_extract(_video: Path, wav_path: Path) -> None:
                wav_path.write_bytes(b"wav")

            def fake_run(command: list[str], check: bool):
                self.assertTrue(check)
                self.assertEqual(str(fallback.resolve()), command[0])
                self.assertIn(str(model.resolve()), command)
                output_prefix = Path(command[command.index("-of") + 1])
                output_prefix.with_suffix(".srt").write_text(
                    "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with patch.dict(
                os.environ,
                {
                    ENV_WHISPER_CPP_BIN: "/done",
                    ENV_WHISPER_CPP_MODEL: "/done-model",
                },
                clear=False,
            ), patch(
                "vid_to_sub_app.shared.env.shutil.which",
                return_value=None,
            ), patch(
                "vid_to_sub_app.shared.env.WHISPER_CLI_CANDIDATES",
                (str(fallback),),
            ), patch(
                "vid_to_sub_app.shared.env.MODEL_SEARCH_DIRS",
                (str(root),),
            ), patch(
                "vid_to_sub_app.cli.transcription.extract_audio_for_whisper_cpp",
                side_effect=fake_extract,
            ), patch(
                "vid_to_sub_app.cli.transcription.subprocess.run",
                side_effect=fake_run,
            ):
                segments, info = transcribe_whisper_cpp(
                    video=video,
                    model_name="large-v3",
                    device="cpu",
                    language=None,
                    threads=2,
                    whisper_cpp_model_path=None,
                )

        self.assertEqual("hello", segments[0]["text"])
        self.assertEqual(str(model.resolve()), info["model_path"])

    def test_transcribe_whisper_cpp_reports_progress_from_segment_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "clip.mp4"
            model = root / "ggml-large-v3.bin"
            fallback = root / "whisper-cli"
            video.write_bytes(b"video")
            model.write_bytes(b"model")
            fallback.write_text("#!/bin/sh\n", encoding="utf-8")
            fallback.chmod(0o755)

            def fake_extract(_video: Path, wav_path: Path) -> None:
                wav_path.write_bytes(b"wav")

            class FakePopen:
                def __init__(self, command: list[str], **_kwargs) -> None:
                    self.command = command
                    output_prefix = Path(command[command.index("-of") + 1])
                    output_prefix.with_suffix(".srt").write_text(
                        "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
                        encoding="utf-8",
                    )
                    self.stdout = io.StringIO(
                        "[00:00:00.000 --> 00:00:01.500]   hello\n"
                    )

                def wait(self) -> int:
                    return 0

            progress: list[float] = []

            with patch("vid_to_sub_app.shared.env.shutil.which", return_value=None), patch(
                "vid_to_sub_app.shared.env.WHISPER_CLI_CANDIDATES",
                (str(fallback),),
            ), patch(
                "vid_to_sub_app.cli.transcription.extract_audio_for_whisper_cpp",
                side_effect=fake_extract,
            ), patch(
                "vid_to_sub_app.cli.transcription.subprocess.Popen",
                side_effect=lambda *args, **kwargs: FakePopen(*args, **kwargs),
            ):
                with redirect_stdout(io.StringIO()):
                    segments, info = transcribe_whisper_cpp(
                        video=video,
                        model_name="large-v3",
                        device="cpu",
                        language=None,
                        threads=2,
                        whisper_cpp_model_path=str(model),
                        progress_callback=progress.append,
                    )

        self.assertEqual("hello", segments[0]["text"])
        self.assertEqual(str(model.resolve()), info["model_path"])
        self.assertEqual([1.5], progress)

    def test_transcribe_whisper_cpp_accepts_wav_suffixed_srt_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "clip.mp4"
            model = root / "ggml-large-v3.bin"
            fallback = root / "whisper-cli"
            video.write_bytes(b"video")
            model.write_bytes(b"model")
            fallback.write_text("#!/bin/sh\n", encoding="utf-8")
            fallback.chmod(0o755)

            def fake_extract(_video: Path, wav_path: Path) -> None:
                wav_path.write_bytes(b"wav")

            def fake_run(command: list[str], check: bool):
                self.assertTrue(check)
                wav_path = Path(command[command.index("-f") + 1])
                wav_path.parent.joinpath(f"{wav_path.name}.srt").write_text(
                    "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            with patch("vid_to_sub_app.shared.env.shutil.which", return_value=None), patch(
                "vid_to_sub_app.shared.env.WHISPER_CLI_CANDIDATES",
                (str(fallback),),
            ), patch(
                "vid_to_sub_app.cli.transcription.extract_audio_for_whisper_cpp",
                side_effect=fake_extract,
            ), patch(
                "vid_to_sub_app.cli.transcription.subprocess.run",
                side_effect=fake_run,
            ):
                segments, info = transcribe_whisper_cpp(
                    video=video,
                    model_name="large-v3",
                    device="cpu",
                    language=None,
                    threads=2,
                    whisper_cpp_model_path=str(model),
                )

        self.assertEqual("hello", segments[0]["text"])
        self.assertEqual(str(model.resolve()), info["model_path"])

    def test_database_folder_queue_state_upsert_tracks_completion_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir) / "state.db")

            db.upsert_folder_queue_state(
                "hash-a",
                "/tmp/videos/a",
                status="queued",
                total_files=3,
                completed_files=0,
                is_completed=False,
            )
            db.upsert_folder_queue_state(
                "hash-a",
                "/tmp/videos/a",
                status="completed",
                total_files=3,
                completed_files=3,
                is_completed=True,
            )

            rows = db.get_folder_queue_states()

            self.assertEqual(1, len(rows))
            self.assertEqual("hash-a", rows[0]["folder_hash"])
            self.assertEqual(1, rows[0]["is_completed"])
            self.assertEqual(3, rows[0]["completed_files"])

    def test_refresh_live_panels_skips_unattached_run_box(self) -> None:
        app = VidToSubApp()
        fake_updates: list[str] = []
        fake_box = SimpleNamespace(
            is_attached=False,
            remove_children=lambda: fake_updates.append("remove"),
            mount=lambda *_args, **_kwargs: fake_updates.append("mount"),
        )
        fake_static = SimpleNamespace(update=lambda *_args, **_kwargs: None)

        def query_one(selector: str, *_args):
            if selector == "#run-active-box":
                return fake_box
            if selector in {
                "#run-overview",
                "#run-shell",
                "#run-progress",
                "#agent-live",
            }:
                return fake_static
            raise AssertionError(f"Unexpected selector: {selector}")

        with patch.object(app, "query_one", side_effect=query_one):
            app._refresh_live_panels()

        self.assertEqual([], fake_updates)

    def test_agent_plan_normalization_filters_unsupported_actions(self) -> None:
        app = VidToSubApp()

        plan = app._normalize_agent_plan(
            {
                "summary": "ok",
                "analysis": "done",
                "actions": [
                    {"type": "auto_setup", "mode": "FULL", "model": "large-v3"},
                    {"type": "download_model", "model": "small"},
                    {"type": "pip_install", "target": "whisperx"},
                    {"type": "shell", "command": "rm -rf /"},
                ],
            }
        )

        self.assertEqual(3, len(plan["actions"]))
        self.assertEqual("full", plan["actions"][0]["mode"])
        self.assertEqual("download_model", plan["actions"][1]["type"])

    def test_agent_plan_normalization_allows_history_actions_for_known_jobs(self) -> None:
        app = VidToSubApp()

        with patch("vid_to_sub_app.tui.app._db.get_jobs", return_value=[{"id": 42}]):
            plan = app._normalize_agent_plan(
                {
                    "summary": "ok",
                    "actions": [
                        {"type": "load_history_job", "job_id": 42},
                        {"type": "delete_history_job", "job_id": 999},
                        {"type": "kill_active_run"},
                    ],
                }
            )

        self.assertEqual(
            [
                {"type": "load_history_job", "job_id": 42},
                {"type": "kill_active_run"},
            ],
            plan["actions"],
        )

    def test_stream_reaps_process_when_ui_callback_raises(self) -> None:
        app = VidToSubApp()

        class FakeProc:
            def __init__(self) -> None:
                self.stdout = io.StringIO("plain output\n")
                self.stdin = None
                self.wait_calls = 0
                self.terminate_calls = 0

            def poll(self) -> int | None:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                self.wait_calls += 1
                return 0

            def terminate(self) -> None:
                self.terminate_calls += 1

        proc = FakeProc()
        plan = SimpleNamespace(
            label="local",
            kind="local",
            cmd=["python", "vid_to_sub.py"],
            env=None,
            stdin_payload=None,
        )

        def raise_from_ui(_callback, *_args):
            raise RuntimeError("ui callback failed")

        with patch("vid_to_sub_app.tui.app.subprocess.Popen", return_value=proc), patch.object(
            app, "call_from_thread", side_effect=raise_from_ui
        ), patch.object(app, "_log", return_value=None), patch.object(
            app, "_apply_progress_event", return_value=None
        ), patch.object(app, "_finalize_executor_failure", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "ui callback failed"):
                VidToSubApp._stream.__wrapped__(app, [plan])

        self.assertEqual(1, proc.wait_calls)
        self.assertEqual({}, app._procs)
        self.assertIsNone(app._proc)
        self.assertEqual(0, proc.terminate_calls)

    def test_stream_reaps_process_when_stdout_reader_crashes(self) -> None:
        app = VidToSubApp()
        logs: list[str] = []

        class BrokenStdout:
            def __init__(self) -> None:
                self.closed = False

            def readline(self) -> str:
                raise RuntimeError("read failed")

            def close(self) -> None:
                self.closed = True

        class FakeProc:
            def __init__(self) -> None:
                self.stdout = BrokenStdout()
                self.stdin = None
                self.wait_calls = 0

            def poll(self) -> int | None:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                self.wait_calls += 1
                return 0

            def terminate(self) -> None:
                raise AssertionError("terminate should not be called")

        proc = FakeProc()
        plan = SimpleNamespace(
            label="local",
            kind="local",
            cmd=["python", "vid_to_sub.py"],
            env=None,
            stdin_payload=None,
        )

        with patch("vid_to_sub_app.tui.app.subprocess.Popen", return_value=proc), patch.object(
            app,
            "call_from_thread",
            side_effect=lambda callback, *args: callback(*args),
        ), patch.object(app, "_log", side_effect=logs.append), patch.object(
            app, "_apply_progress_event", return_value=None
        ), patch.object(app, "_finalize_executor_failure", return_value=None):
            VidToSubApp._stream.__wrapped__(app, [plan])

        self.assertTrue(proc.stdout.closed)
        self.assertEqual(1, proc.wait_calls)
        self.assertEqual({}, app._procs)
        self.assertIsNone(app._proc)
        self.assertIn("\n[bold green]✓ Completed (exit 0)[/]", logs)


if __name__ == "__main__":
    unittest.main()
