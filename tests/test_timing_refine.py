from __future__ import annotations

import math
import shutil
import struct
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vid_to_sub_app.cli.timing_refine import refine_segment_timing


def _write_test_wav(
    path: Path,
    *,
    sample_rate: int = 16000,
    duration_sec: float = 1.4,
    speech_regions: list[tuple[float, float, float]] | None = None,
) -> None:
    total_samples = int(sample_rate * duration_sec)
    samples = [0] * total_samples
    for start_sec, end_sec, amplitude in speech_regions or []:
        start_index = max(0, int(start_sec * sample_rate))
        end_index = min(total_samples, int(end_sec * sample_rate))
        for index in range(start_index, end_index):
            t = index / sample_rate
            value = amplitude * math.sin(2.0 * math.pi * 220.0 * t)
            samples[index] = int(max(-32767, min(32767, value)))

    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


class TimingRefineTests(unittest.TestCase):
    def test_refine_segment_timing_trims_speech_tail(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "fixture.wav"
            video_path = temp_path / "movie.mp4"
            video_path.write_bytes(b"video")
            _write_test_wav(
                wav_path,
                speech_regions=[
                    (0.00, 0.34, 12000.0),
                    (0.86, 1.02, 10000.0),
                ],
            )
            segments = [
                {
                    "start": 0.0,
                    "end": 0.78,
                    "text": "hello",
                    "words": [{"word": "hello", "start": 0.0, "end": 0.34}],
                },
                {
                    "start": 0.82,
                    "end": 1.08,
                    "text": "world",
                    "words": [{"word": "world", "start": 0.86, "end": 1.02}],
                },
            ]
            info = {"content_type": "speech"}

            with patch(
                "vid_to_sub_app.cli.timing_refine.extract_audio_for_whisper_cpp",
                side_effect=lambda _video, out_path: shutil.copyfile(wav_path, out_path),
            ):
                refined, stats = refine_segment_timing(video_path, segments, info)

        self.assertEqual(2, len(refined))
        self.assertTrue(refined[0]["timing_refined"])
        self.assertAlmostEqual(0.78, refined[0]["raw_end"], places=3)
        self.assertLess(refined[0]["end"], 0.60)
        self.assertLessEqual(refined[0]["end"], 0.74)
        self.assertGreaterEqual(refined[0]["end"], 0.31)
        self.assertIn("acoustic_end", refined[0])
        self.assertGreaterEqual(refined[0]["timing_confidence"], 0.68)
        self.assertGreaterEqual(stats["trimmed_segments"], 1)
        self.assertGreater(stats["median_trim_ms"], 0)
        self.assertGreaterEqual(stats["word_anchor_segments"], 1)
        self.assertGreaterEqual(stats["acoustic_anchor_segments"], 1)

    def test_refine_segment_timing_skips_music_profile(self) -> None:
        segments = [{"start": 0.0, "end": 1.0, "text": "lyric"}]
        with patch(
            "vid_to_sub_app.cli.timing_refine.extract_audio_for_whisper_cpp"
        ) as mock_extract:
            refined, stats = refine_segment_timing(
                Path("/tmp/movie.mp4"),
                segments,
                {"content_type": "music"},
            )

        self.assertEqual(segments, refined)
        self.assertFalse(stats["enabled"])
        self.assertEqual("content_type_music", stats["disabled_reason"])
        mock_extract.assert_not_called()

if __name__ == "__main__":
    unittest.main()
