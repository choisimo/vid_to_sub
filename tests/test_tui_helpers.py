from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tui import (
    VidToSubApp,
    build_system_install_commands,
    discover_ggml_models,
    discover_input_matches,
    extract_json_payload,
    normalize_chat_endpoint,
    packages_for_manager,
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


if __name__ == "__main__":
    unittest.main()
