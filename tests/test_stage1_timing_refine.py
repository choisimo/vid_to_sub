from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from vid_to_sub_app.cli.runner import run_stage1
from vid_to_sub_app.cli.stage_artifact import load_stage_artifact


class TimingRefineStage1Tests(unittest.TestCase):
    @staticmethod
    def _make_args(**overrides: object) -> argparse.Namespace:
        defaults: dict[str, object] = {
            "backend": "faster-whisper",
            "model": "small",
            "device": "cpu",
            "language": "en",
            "content_type": "speech",
            "beam_size": 5,
            "compute_type": None,
            "hf_token": None,
            "diarize": False,
            "whisper_cpp_model_path": None,
            "workers": 1,
            "verbose": False,
            "translate_to": None,
            "translation_model": None,
            "translation_base_url": None,
            "translation_api_key": None,
            "postprocess": False,
            "postprocess_model": None,
            "postprocess_base_url": None,
            "postprocess_api_key": None,
            "postprocess_mode": "auto",
            "force_translate": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_run_stage1_persists_timing_refine_stats_and_segments(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = (temp_path / "movie.mp4").resolve()
            output_dir = (temp_path / "out").resolve()
            video_path.write_bytes(b"video")
            task = {
                "video_path": str(video_path),
                "folder_hash": "folder-hash",
                "folder_path": str(video_path.parent),
            }
            args = self._make_args()
            raw_segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
            refined_segments = [
                {
                    "start": 0.0,
                    "end": 0.58,
                    "raw_end": 1.0,
                    "timing_refined": True,
                    "timing_trim_ms": 420,
                    "timing_confidence": 0.91,
                    "text": "Hello",
                }
            ]
            info = {"language": "en", "duration": 1.0, "content_type": "speech"}
            timing_stats = {
                "enabled": True,
                "applied": True,
                "content_type": "speech",
                "trimmed_segments": 1,
                "median_trim_ms": 420,
                "p95_trim_ms": 420,
                "low_confidence_segments": 0,
            }

            with (
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    return_value=(raw_segments, info),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.refine_segment_timing",
                    return_value=(refined_segments, timing_stats),
                ),
            ):
                result = run_stage1(task, args, frozenset({"json"}), output_dir, 1, 0)

            self.assertTrue(result.success, result.error)
            assert result.artifact_path is not None
            artifact = load_stage_artifact(Path(result.artifact_path))
            self.assertEqual(0.58, artifact["segments"][0]["end"])
            self.assertTrue(artifact["segments"][0]["timing_refined"])
            self.assertEqual(1, artifact["quality"]["timing_refine"]["trimmed_segments"])


if __name__ == "__main__":
    unittest.main()
