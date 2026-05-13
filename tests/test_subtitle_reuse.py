from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vid_to_sub_app.cli.output import parse_srt
from vid_to_sub_app.cli.runner import run_stage1
from vid_to_sub_app.cli.stage_artifact import load_stage_artifact
from vid_to_sub_app.cli.subtitle_reuse import (
    find_sidecar_subtitles,
    opensubtitles_hash,
    parse_ass,
    parse_vtt,
    sync_segments_to_reference,
)


SRT_TEXT = """1
00:00:01,000 --> 00:00:02,000
hello world

2
00:00:03,000 --> 00:00:04,000
goodbye world
"""


class SubtitleReuseTests(unittest.TestCase):
    def test_parse_vtt_and_ass_into_segments(self) -> None:
        vtt_segments = parse_vtt(
            """WEBVTT

00:00:01.000 --> 00:00:02.500
hello

2
00:00:03.000 --> 00:00:04.000
goodbye
"""
        )
        self.assertEqual(2, len(vtt_segments))
        self.assertEqual(1.0, vtt_segments[0]["start"])
        self.assertEqual("goodbye", vtt_segments[1]["text"])

        ass_segments = parse_ass(
            """[Script Info]
Title: sample
[Events]
Format: Layer, Start, End, Style, Text
Dialogue: 0,0:00:01.00,0:00:02.25,Default,{\\i1}hello\\Nworld
"""
        )
        self.assertEqual(1, len(ass_segments))
        self.assertEqual(2.25, ass_segments[0]["end"])
        self.assertEqual("hello\nworld", ass_segments[0]["text"])

    def test_opensubtitles_hash_is_deterministic_for_large_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video = Path(tmpdir) / "sample.mkv"
            video.write_bytes(bytes(range(256)) * 800)

            first = opensubtitles_hash(video)
            second = opensubtitles_hash(video)

        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        self.assertEqual(16, len(first or ""))

    def test_find_sidecar_subtitles_scores_matching_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "Movie.Name.2024.mkv"
            video.write_bytes(b"fake")
            subtitle = root / "Movie.Name.2024.en.srt"
            subtitle.write_text(SRT_TEXT, encoding="utf-8")

            candidates = find_sidecar_subtitles(video, ["en"])

        self.assertEqual(1, len(candidates))
        self.assertEqual("local", candidates[0].provider)
        self.assertEqual("en", candidates[0].language)
        self.assertGreaterEqual(candidates[0].score, 0.90)

    def test_sync_segments_to_reference_commits_offset(self) -> None:
        source = parse_srt(SRT_TEXT)
        reference = [
            {"start": 11.0, "end": 12.0, "text": "hello world"},
            {"start": 13.0, "end": 14.0, "text": "goodbye world"},
        ]

        synced, info = sync_segments_to_reference(source, reference)

        self.assertTrue(info["committed"])
        self.assertAlmostEqual(11.0, synced[0]["start"], places=2)
        self.assertAlmostEqual(10.0, info["offset"], places=2)

    def test_run_stage1_reuses_local_subtitle_without_transcription(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video = root / "Movie.Name.2024.mkv"
            video.write_bytes(b"fake")
            (root / "Movie.Name.2024.en.srt").write_text(SRT_TEXT, encoding="utf-8")
            args = SimpleNamespace(
                workers=1,
                reuse_existing_subtitles="local",
                subtitle_providers=None,
                subtitle_languages="en",
                subtitle_max_candidates=10,
                subtitle_min_score=0.50,
                subtitle_sync_mode="off",
                translate_to=None,
                backend="whisper-cpp",
                model="large-v3",
                device="cpu",
                language="en",
                content_type="auto",
                beam_size=5,
                compute_type=None,
                hf_token=None,
                diarize=False,
                whisper_cpp_model_path=None,
            )

            with patch("vid_to_sub_app.cli.runner._run_stage1_transcription") as transcribe:
                transcribe.side_effect = AssertionError("transcription should not run")
                result = run_stage1(
                    {"video_path": str(video), "folder_path": str(root), "folder_hash": "abc"},
                    args,
                    frozenset({"srt"}),
                    None,
                    1,
                    0,
                )

            output_exists = (root / "Movie.Name.2024.srt").exists()
            artifact = load_stage_artifact(root / "Movie.Name.2024.stage1.json")

            self.assertTrue(result.success)
            self.assertEqual("subtitle_reuse", result.stage)
            self.assertTrue(output_exists)
            self.assertEqual("subtitle-reuse", artifact["backend"])
            self.assertEqual("local", artifact["quality"]["provider"])


if __name__ == "__main__":
    unittest.main()
