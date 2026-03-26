from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from vid_to_sub_app.cli.stage_artifact import artifact_path_for, load_stage_artifact
from vid_to_sub_app.shared.constants import (
    ENV_TRANSLATION_API_KEY,
    ENV_TRANSLATION_BASE_URL,
    ENV_TRANSLATION_MODEL,
    ENV_WHISPER_CPP_BIN,
    ENV_WHISPER_CPP_MODEL,
)
from vid_to_sub_app.shared.env import find_whisper_cpp_bin

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_MEDIA_ENV = "VID_TO_SUB_SMOKE_MEDIA"
SMOKE_BACKEND_ENV = "VID_TO_SUB_SMOKE_BACKEND"
SMOKE_MODEL_ENV = "VID_TO_SUB_SMOKE_MODEL"
SMOKE_SOURCE_LANGUAGE_ENV = "VID_TO_SUB_SMOKE_SOURCE_LANGUAGE"
SMOKE_TARGET_LANGUAGE_ENV = "VID_TO_SUB_SMOKE_TARGET_LANGUAGE"
SMOKE_CONTENT_TYPE_ENV = "VID_TO_SUB_SMOKE_CONTENT_TYPE"
SMOKE_TIMEOUT_ENV = "VID_TO_SUB_SMOKE_TIMEOUT_SEC"
SMOKE_FORCE_TRANSLATE_ENV = "VID_TO_SUB_SMOKE_FORCE_TRANSLATE"
SMOKE_TRANSLATION_BASE_URL_ENV = "VID_TO_SUB_SMOKE_TRANSLATION_BASE_URL"
SMOKE_TRANSLATION_API_KEY_ENV = "VID_TO_SUB_SMOKE_TRANSLATION_API_KEY"
SMOKE_TRANSLATION_MODEL_ENV = "VID_TO_SUB_SMOKE_TRANSLATION_MODEL"
SMOKE_WHISPER_CPP_BIN_ENV = "VID_TO_SUB_SMOKE_WHISPER_CPP_BIN"
SMOKE_WHISPER_CPP_MODEL_ENV = "VID_TO_SUB_SMOKE_WHISPER_CPP_MODEL_PATH"


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


class RealPipelineSmokeTests(unittest.TestCase):
    """Opt-in end-to-end smoke for real media, ffmpeg, a whisper backend, and translation."""

    def test_cli_transcribe_and_translate_real_media_sample(self) -> None:
        media = _env(SMOKE_MEDIA_ENV)
        if not media:
            self.skipTest(
                f"set {SMOKE_MEDIA_ENV} to a real media file to enable this smoke test"
            )

        media_path = Path(media).expanduser().resolve()
        if not media_path.is_file():
            self.skipTest(f"{SMOKE_MEDIA_ENV} does not point to a readable file: {media}")

        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg is not available on PATH")

        backend = _env(SMOKE_BACKEND_ENV) or "whisper-cpp"
        model = _env(SMOKE_MODEL_ENV)
        if not model:
            self.skipTest(f"set {SMOKE_MODEL_ENV} for the configured smoke backend")

        source_language = _env(SMOKE_SOURCE_LANGUAGE_ENV) or "en"
        target_language = _env(SMOKE_TARGET_LANGUAGE_ENV) or "ko"
        content_type = _env(SMOKE_CONTENT_TYPE_ENV) or "speech"
        timeout_sec = int(_env(SMOKE_TIMEOUT_ENV) or "1800")

        translation_base_url = _env(SMOKE_TRANSLATION_BASE_URL_ENV) or _env(
            ENV_TRANSLATION_BASE_URL
        )
        translation_api_key = _env(SMOKE_TRANSLATION_API_KEY_ENV) or _env(
            ENV_TRANSLATION_API_KEY
        )
        translation_model = _env(SMOKE_TRANSLATION_MODEL_ENV) or _env(
            ENV_TRANSLATION_MODEL
        )
        missing_translation = [
            name
            for name, value in (
                (ENV_TRANSLATION_BASE_URL, translation_base_url),
                (ENV_TRANSLATION_API_KEY, translation_api_key),
                (ENV_TRANSLATION_MODEL, translation_model),
            )
            if not value
        ]
        if missing_translation:
            self.skipTest(
                "translation smoke test requires: " + ", ".join(missing_translation)
            )

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env[ENV_TRANSLATION_BASE_URL] = translation_base_url
        env[ENV_TRANSLATION_API_KEY] = translation_api_key
        env[ENV_TRANSLATION_MODEL] = translation_model

        if backend == "whisper-cpp":
            whisper_cpp_bin = _env(SMOKE_WHISPER_CPP_BIN_ENV) or find_whisper_cpp_bin(
                _env(ENV_WHISPER_CPP_BIN) or None
            )
            whisper_cpp_model = _env(SMOKE_WHISPER_CPP_MODEL_ENV) or _env(
                ENV_WHISPER_CPP_MODEL
            )
            if not whisper_cpp_bin:
                self.skipTest(
                    "whisper-cpp smoke test requires whisper-cli on PATH or "
                    f"{SMOKE_WHISPER_CPP_BIN_ENV}/{ENV_WHISPER_CPP_BIN}"
                )
            if not whisper_cpp_model:
                self.skipTest(
                    "whisper-cpp smoke test requires "
                    f"{SMOKE_WHISPER_CPP_MODEL_ENV} or {ENV_WHISPER_CPP_MODEL}"
                )
            env[ENV_WHISPER_CPP_BIN] = whisper_cpp_bin
            env[ENV_WHISPER_CPP_MODEL] = whisper_cpp_model

        with tempfile.TemporaryDirectory(prefix="vid-to-sub-smoke-") as tmpdir:
            output_dir = Path(tmpdir)
            command = [
                sys.executable,
                str(REPO_ROOT / "vid_to_sub.py"),
                str(media_path),
                "--output-dir",
                str(output_dir),
                "--backend",
                backend,
                "--model",
                model,
                "--language",
                source_language,
                "--content-type",
                content_type,
                "--translate-to",
                target_language,
                "--format",
                "srt",
                "--workers",
                "1",
            ]
            if _env(SMOKE_FORCE_TRANSLATE_ENV) == "1":
                command.append("--force-translate")

            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
                check=False,
            )

            if result.returncode != 0:
                self.fail(
                    "smoke pipeline failed\n"
                    f"command: {' '.join(command)}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )

            source_srt = output_dir / f"{media_path.stem}.srt"
            translated_srt = output_dir / f"{media_path.stem}.{target_language}.srt"
            artifact_path = artifact_path_for(media_path, output_dir)

            self.assertTrue(source_srt.is_file(), f"missing source subtitle: {source_srt}")
            self.assertTrue(
                translated_srt.is_file(),
                f"missing translated subtitle: {translated_srt}",
            )
            self.assertTrue(
                artifact_path.is_file(),
                f"missing stage artifact: {artifact_path}",
            )

            artifact = load_stage_artifact(artifact_path)
            self.assertTrue(
                artifact["stage_status"]["translation_complete"],
                "stage artifact did not record translation completion",
            )
            self.assertFalse(
                artifact["stage_status"]["translation_failed"],
                "stage artifact recorded translation failure",
            )
            self.assertEqual(target_language, artifact["target_lang"])
            self.assertGreater(
                source_srt.stat().st_size, 0, "source subtitle output is empty"
            )
            self.assertGreater(
                translated_srt.stat().st_size, 0, "translated subtitle output is empty"
            )
