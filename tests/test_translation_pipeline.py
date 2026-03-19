from __future__ import annotations

import json
import argparse
import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import ANY, patch

from vid_to_sub_app.cli.main import main
from vid_to_sub_app.cli.manifest import ProcessResult
from vid_to_sub_app.cli.runner import process_one, run_stage1, run_stage2
from vid_to_sub_app.cli.stage_artifact import (
    ARTIFACT_FILENAME_SUFFIX,
    ARTIFACT_SCHEMA_VERSION,
    StageArtifact,
    artifact_path_for,
    load_stage_artifact,
    write_stage_artifact,
)
from vid_to_sub_app.cli.translation import (
    postprocess_translated_segments_openai_compatible,
    translate_segments_openai_compatible,
)


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class TranslationPipelineTests(unittest.TestCase):
    def test_translate_segments_openai_compatible_preserves_segment_boundaries(
        self,
    ) -> None:
        segments = [
            {"start": 0.0, "end": 1.5, "text": "Hello"},
            {"start": 1.5, "end": 3.0, "text": "World"},
        ]
        captured_payloads: list[dict[str, list[dict[str, str]]]] = []

        def fake_urlopen(request):
            captured_payloads.append(json.loads(request.data.decode("utf-8")))
            return _FakeHTTPResponse(
                {"choices": [{"message": {"content": '["안녕", "세상"]'}}]}
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            translated_segments, info = translate_segments_openai_compatible(
                segments=segments,
                target_language="ko",
                translation_model="gpt-4.1-mini",
                translation_base_url="https://translation.example/v1",
                translation_api_key="secret",
                source_language="en",
            )

        self.assertEqual(
            [
                {"start": 0.0, "end": 1.5, "text": "안녕"},
                {"start": 1.5, "end": 3.0, "text": "세상"},
            ],
            translated_segments,
        )
        self.assertEqual("gpt-4.1-mini", info["model"])
        self.assertIn(
            "first-pass subtitle translation agent",
            captured_payloads[0]["messages"][0]["content"],
        )

    def test_postprocess_uses_translation_config_as_fallback(self) -> None:
        captured_requests = []

        def fake_urlopen(request):
            captured_requests.append(request)
            return _FakeHTTPResponse(
                {"choices": [{"message": {"content": '["정제된 가사"]'}}]}
            )

        with patch(
            "vid_to_sub_app.cli.translation.urllib.request.urlopen",
            side_effect=fake_urlopen,
        ):
            final_segments, info = postprocess_translated_segments_openai_compatible(
                source_segments=[{"start": 0.0, "end": 2.0, "text": "青い風"}],
                translated_segments=[{"start": 0.0, "end": 2.0, "text": "푸른 바람"}],
                target_language="ko",
                postprocess_mode="auto",
                postprocess_model=None,
                postprocess_base_url=None,
                postprocess_api_key=None,
                source_language="ja",
                translation_model="gpt-4.1-mini",
                translation_base_url="https://translation.example/v1",
                translation_api_key="translation-secret",
            )

        self.assertEqual(
            [{"start": 0.0, "end": 2.0, "text": "정제된 가사"}],
            final_segments,
        )
        self.assertEqual("auto", info["mode"])
        self.assertEqual(
            "https://translation.example/v1/chat/completions",
            captured_requests[0].full_url,
        )
        payload = json.loads(captured_requests[0].data.decode("utf-8"))
        self.assertIn(
            "silently fall back to contextual correction",
            payload["messages"][0]["content"],
        )

    def test_main_rejects_postprocess_without_translation_target(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                main(["--manifest-stdin", "--postprocess-translation"])

        self.assertEqual(2, ctx.exception.code)

    def test_stage1_only_flag_skips_translation(self) -> None:
        video_path = Path("/tmp/movie.mp4")
        manifest = {
            "found_total": 1,
            "skipped": 0,
            "folders": [
                {
                    "folder_hash": "folder-hash",
                    "folder_path": str(video_path.parent),
                    "total_files": 1,
                    "completed_files": 0,
                    "status": "queued",
                    "is_completed": False,
                }
            ],
            "entries": [
                {
                    "video_path": str(video_path),
                    "folder_hash": "folder-hash",
                    "folder_path": str(video_path.parent),
                }
            ],
        }
        stage1_result = ProcessResult(
            success=True,
            video_path=str(video_path),
            folder_hash="folder-hash",
            folder_path=str(video_path.parent),
            worker_id=0,
            elapsed_sec=0.1,
            artifact_path="/tmp/movie.stage1.json",
        )

        with (
            patch("vid_to_sub_app.cli.main.discover_videos", return_value=[video_path]),
            patch(
                "vid_to_sub_app.cli.main.build_run_manifest",
                return_value=manifest,
            ),
            patch("vid_to_sub_app.cli.main.persist_folder_manifest_state"),
            patch("vid_to_sub_app.cli.manifest.persist_folder_manifest_state"),
            patch(
                "vid_to_sub_app.cli.main.run_stage1",
                return_value=stage1_result,
            ) as run_stage1_mock,
            patch("vid_to_sub_app.cli.main.run_stage2") as run_stage2_mock,
        ):
            exit_code = main([str(video_path), "--stage1-only"])

        self.assertEqual(0, exit_code)
        run_stage1_mock.assert_called_once()
        run_stage2_mock.assert_not_called()

    def test_translate_from_artifact_flag_skips_transcription(self) -> None:
        artifact_path = Path("/tmp/movie.stage1.json")
        stage2_result = ProcessResult(
            success=True,
            video_path="/tmp/movie.mp4",
            folder_hash="folder-hash",
            folder_path="/tmp",
            worker_id=0,
            elapsed_sec=0.1,
            output_paths=["/tmp/movie.ko.srt"],
            artifact_path=str(artifact_path),
        )

        with (
            patch(
                "vid_to_sub_app.cli.main.run_stage2",
                return_value=stage2_result,
            ) as run_stage2_mock,
            patch("vid_to_sub_app.cli.main.run_stage1") as run_stage1_mock,
        ):
            exit_code = main(
                [
                    "--translate-from-artifact",
                    str(artifact_path),
                    "--translate-to",
                    "ko",
                ]
            )

        self.assertEqual(0, exit_code)
        run_stage2_mock.assert_called_once_with(artifact_path.resolve(), ANY)
        run_stage1_mock.assert_not_called()

    def test_mutually_exclusive_stage_flags(self) -> None:
        with patch.object(
            argparse.ArgumentParser,
            "error",
            side_effect=SystemExit(2),
        ) as parser_error:
            with self.assertRaises(SystemExit) as ctx:
                main(
                    [
                        "--stage1-only",
                        "--translate-from-artifact",
                        "/tmp/movie.stage1.json",
                    ]
                )

        self.assertEqual(2, ctx.exception.code)
        parser_error.assert_called_once_with(
            "--stage1-only and --translate-from-artifact are mutually exclusive."
        )


class TestStageArtifact(unittest.TestCase):
    def test_write_and_load_roundtrip(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = (temp_path / "movie.mp4").resolve()
            output_dir = (temp_path / "out").resolve()
            output_dir.mkdir()
            source_path.write_bytes(b"")
            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source_path),
                "output_base": str(output_dir),
                "source_fingerprint": "0:123",
                "backend": "faster-whisper",
                "device": "cuda",
                "model": "large-v3",
                "language": "en",
                "target_lang": "ko",
                "formats": ["srt", "json"],
                "primary_outputs": [
                    str((output_dir / "movie.srt").resolve()),
                    str((output_dir / "movie.json").resolve()),
                ],
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Hello"},
                ],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": False,
                    "translation_complete": True,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }

            artifact_path = write_stage_artifact(artifact, output_dir, source_path)
            loaded_artifact = load_stage_artifact(artifact_path)

        self.assertEqual(output_dir / f"movie{ARTIFACT_FILENAME_SUFFIX}", artifact_path)
        self.assertEqual(artifact, loaded_artifact)

    def test_load_unsupported_version_raises(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            artifact_path = temp_path / f"movie{ARTIFACT_FILENAME_SUFFIX}"
            artifact_path.write_text(
                json.dumps(
                    {
                        "schema_version": "99",
                        "source_path": str((temp_path / "movie.mp4").resolve()),
                        "output_base": str(temp_path.resolve()),
                        "source_fingerprint": "0:0",
                        "backend": "faster-whisper",
                        "device": "cpu",
                        "model": "large-v3",
                        "language": None,
                        "target_lang": None,
                        "formats": ["srt"],
                        "primary_outputs": [],
                        "segments": [],
                        "stage_status": {
                            "transcription_complete": False,
                            "translation_pending": True,
                            "translation_complete": False,
                            "translation_failed": False,
                            "translation_error": None,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "99"):
                load_stage_artifact(artifact_path)

    def test_artifact_path_for_source_only(self) -> None:
        source_path = Path("/tmp/videos/movie.mp4")

        self.assertEqual(
            Path("/tmp/videos") / f"movie{ARTIFACT_FILENAME_SUFFIX}",
            artifact_path_for(source_path, None),
        )

    def test_artifact_path_for_with_output_dir(self) -> None:
        source_path = Path("/tmp/videos/movie.mp4")
        output_dir = Path("/tmp/output")

        self.assertEqual(
            output_dir / f"movie{ARTIFACT_FILENAME_SUFFIX}",
            artifact_path_for(source_path, output_dir),
        )


class TestRunnerStageSplit(unittest.TestCase):
    def _make_args(self, **overrides: object) -> argparse.Namespace:
        defaults: dict[str, object] = {
            "workers": 1,
            "backend": "faster-whisper",
            "model": "large-v3",
            "device": "cpu",
            "language": None,
            "beam_size": 5,
            "compute_type": "int8",
            "hf_token": None,
            "diarize": False,
            "whisper_cpp_model_path": None,
            "verbose": False,
            "translate_to": "ko",
            "translation_model": "gpt-4.1-mini",
            "translation_base_url": "https://translation.example/v1",
            "translation_api_key": "secret",
            "postprocess_translation": False,
            "postprocess_mode": "auto",
            "postprocess_model": None,
            "postprocess_base_url": None,
            "postprocess_api_key": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_stage1_writes_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = (temp_path / "movie.mp4").resolve()
            video_path.write_bytes(b"video")
            output_dir = (temp_path / "out").resolve()
            task = {
                "video_path": str(video_path),
                "folder_hash": "folder-hash",
                "folder_path": str(video_path.parent),
            }
            args = self._make_args()
            segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
            info = {"language": "en", "duration": 1.0}
            written_paths = [output_dir / "movie.srt"]
            artifact_path = output_dir / f"movie{ARTIFACT_FILENAME_SUFFIX}"

            with (
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    return_value=(segments, info),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=written_paths,
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_stage_artifact",
                    return_value=artifact_path,
                ) as write_artifact,
            ):
                result = run_stage1(task, args, frozenset({"srt"}), output_dir, 2, 0)

        self.assertTrue(result.success)
        self.assertEqual(str(artifact_path), result.artifact_path)
        write_artifact.assert_called_once()

    def test_stage2_does_not_call_transcribe(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            source_path = (temp_path / "movie.mp4").resolve()
            output_dir = (temp_path / "out").resolve()
            output_dir.mkdir()
            source_path.write_bytes(b"video")
            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source_path),
                "output_base": str(output_dir),
                "source_fingerprint": f"{source_path.stat().st_size}:{int(source_path.stat().st_mtime)}",
                "backend": "faster-whisper",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "target_lang": "ko",
                "formats": ["srt"],
                "primary_outputs": [str(output_dir / "movie.srt")],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": True,
                    "translation_complete": False,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = write_stage_artifact(artifact, output_dir, source_path)
            args = self._make_args()
            translated_segments = [{"start": 0.0, "end": 1.0, "text": "안녕"}]
            translated_written = [output_dir / "movie.ko.srt"]

            with (
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    side_effect=AssertionError("transcribe should not run"),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                    return_value=(translated_segments, {"model": "gpt-4.1-mini"}),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=translated_written,
                ),
            ):
                result = run_stage2(artifact_path, args)

            loaded_artifact = load_stage_artifact(artifact_path)

        self.assertTrue(result.success)
        self.assertEqual(str(artifact_path), result.artifact_path)
        self.assertTrue(loaded_artifact["stage_status"]["translation_complete"])

    def test_process_one_no_translation_skips_stage2(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = (temp_path / "movie.mp4").resolve()
            video_path.write_bytes(b"video")
            output_dir = (temp_path / "out").resolve()
            task = {
                "video_path": str(video_path),
                "folder_hash": "folder-hash",
                "folder_path": str(video_path.parent),
            }
            args = self._make_args(translate_to=None)
            segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
            info = {"language": "en", "duration": 1.0}
            written_paths = [output_dir / "movie.srt"]
            artifact_path = output_dir / f"movie{ARTIFACT_FILENAME_SUFFIX}"

            with (
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    return_value=(segments, info),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=written_paths,
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_stage_artifact",
                    return_value=artifact_path,
                ),
                patch(
                    "vid_to_sub_app.cli.runner.translate_segments_openai_compatible"
                ) as translate,
            ):
                result = process_one(task, args, frozenset({"srt"}), output_dir, 2, 0)

        self.assertTrue(result.success)
        translate.assert_not_called()


class TestRunStage2Idempotency(unittest.TestCase):
    """run_stage2 must skip translation when output files already exist."""

    def _make_args(self, tmpdir: Path) -> argparse.Namespace:
        args = argparse.Namespace()
        args.translate_to = "ko"
        args.translation_model = "test-model"
        args.translation_base_url = "http://localhost/v1"
        args.translation_api_key = "test-key"
        args.postprocess_translation = False
        return args

    def test_run_stage2_skips_when_translation_complete_and_output_exists(
        self,
    ) -> None:
        """run_stage2 returns success without calling the translation API when
        translation_complete=True and the translated file already exists on disk."""
        import argparse
        from unittest.mock import patch
        from vid_to_sub_app.cli.runner import run_stage2
        from vid_to_sub_app.cli.stage_artifact import write_stage_artifact

        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            # Write a fake source video path (no need for it to exist)
            source = tmpdir / "movie.mp4"
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")
            # Translated file already exists
            translated = tmpdir / "movie.ko.srt"
            translated.write_text("1\n00:00:00,000 --> 00:00:01,000\n안녕")

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": "1024:1700000000",
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": False,
                    "translation_complete": True,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible"
            ) as mock_translate:
                result = run_stage2(artifact_path, args)

            self.assertTrue(result.success, result.error)
            mock_translate.assert_not_called()

    def test_run_stage2_proceeds_when_output_missing(self) -> None:
        """run_stage2 proceeds normally when translation_complete=True but
        the translated file does not exist on disk."""
        import argparse
        from unittest.mock import patch
        from vid_to_sub_app.cli.runner import run_stage2
        from vid_to_sub_app.cli.stage_artifact import write_stage_artifact

        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello")
            # Translated file does NOT exist — should NOT be idempotent-skipped

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": "1024:1700000000",
                "backend": "whisper-cpp",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": False,
                    "translation_complete": True,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            translated_segs = [{"start": 0.0, "end": 1.0, "text": "안녕"}]
            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                return_value=(translated_segs, {}),
            ) as mock_translate:
                result = run_stage2(artifact_path, args)

            mock_translate.assert_called_once()
            self.assertTrue(result.success, result.error)

if __name__ == "__main__":
    _ = unittest.main()


class TestOverwriteTranslationFlag(unittest.TestCase):
    """U9: --overwrite-translation bypasses the idempotency guard in run_stage2."""

    def _make_args(self, output_dir: Path) -> argparse.Namespace:
        return argparse.Namespace(
            translate_to="ko",
            translation_model="gpt-4o-mini",
            translation_base_url="https://example.com/v1",
            translation_api_key="key",
            postprocess_translation=False,
            overwrite_translation=True,
        )

    def test_overwrite_translation_reruns_despite_translation_complete(self) -> None:
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source = tmpdir / "movie.mp4"
            source.write_bytes(b"video")
            srt = tmpdir / "movie.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            translated_srt = tmpdir / "movie.ko.srt"
            translated_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\n\uc548\ub155\n", encoding="utf-8")

            artifact: StageArtifact = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "source_path": str(source),
                "output_base": str(tmpdir),
                "source_fingerprint": "0:0",
                "backend": "faster-whisper",
                "device": "cpu",
                "model": "large-v3",
                "language": "en",
                "target_lang": "ko",
                "formats": ["srt"],
                "segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}],
                "primary_outputs": [str(srt)],
                "stage_status": {
                    "transcription_complete": True,
                    "translation_pending": False,
                    "translation_complete": True,
                    "translation_failed": False,
                    "translation_error": None,
                },
            }
            artifact_path = tmpdir / "movie.stage1.json"
            write_stage_artifact(artifact, tmpdir, source)

            args = self._make_args(tmpdir)
            translated_segs = [{"start": 0.0, "end": 1.0, "text": "\uc548\ub155"}]
            with patch(
                "vid_to_sub_app.cli.runner.translate_segments_openai_compatible",
                return_value=(translated_segs, {}),
            ) as mock_translate:
                result = run_stage2(artifact_path, args)

            # Even though translation_complete=True and output file exists,
            # --overwrite-translation=True must cause the API to be called again.
            mock_translate.assert_called_once()
            self.assertTrue(result.success, result.error)


class TestLegacyInlineRollback(unittest.TestCase):
    """U11: VID_TO_SUB_LEGACY_INLINE=1 restores old inline transcription+translation."""

    def test_legacy_inline_env_var_calls_inline_path(self) -> None:
        import os
        """When VID_TO_SUB_LEGACY_INLINE=1, process_one should use the inline path,
        not write any stage artifact, and return successfully."""
        with TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            video = tmpdir / "movie.mp4"
            video.write_bytes(b"video")
            task = {
                "video_path": str(video),
                "folder_hash": "folder-hash",
                "folder_path": str(tmpdir),
            }
            args = argparse.Namespace(
                backend="faster-whisper",
                model="large-v3",
                device="cpu",
                language=None,
                beam_size=5,
                compute_type=None,
                hf_token=None,
                diarize=False,
                whisper_cpp_model_path=None,
                translate_to=None,
                translation_model=None,
                translation_base_url=None,
                translation_api_key=None,
                postprocess_translation=False,
                postprocess_mode="auto",
                postprocess_model=None,
                postprocess_base_url=None,
                postprocess_api_key=None,
                workers=1,
                verbose=False,
            )
            formats = frozenset({"srt"})
            fake_segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}]
            fake_info = {"backend": "faster-whisper", "language": "en", "duration": 1.0, "model": "large-v3"}
            fake_srt = tmpdir / "movie.srt"
            fake_srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")

            with (
                patch.dict(os.environ, {"VID_TO_SUB_LEGACY_INLINE": "1"}),
                patch("vid_to_sub_app.cli.runner._LEGACY_INLINE", True),
                patch(
                    "vid_to_sub_app.cli.runner.transcribe",
                    return_value=(fake_segments, fake_info),
                ),
                patch(
                    "vid_to_sub_app.cli.runner.write_outputs",
                    return_value=[fake_srt],
                ),
            ):
                result = process_one(task, args, formats, tmpdir, 2, 0)

            # No artifact should be written in legacy inline mode
            artifact_files = list(tmpdir.glob("*.stage1.json"))
            self.assertEqual([], artifact_files,
                             "Legacy inline mode must not write stage artifacts")
            self.assertTrue(result.success, result.error)
            self.assertIsNone(result.artifact_path,
                              "Legacy inline result must have no artifact_path")

