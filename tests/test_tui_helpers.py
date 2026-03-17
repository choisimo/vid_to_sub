from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tui import (
    discover_ggml_models,
    discover_input_matches,
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


if __name__ == "__main__":
    unittest.main()
