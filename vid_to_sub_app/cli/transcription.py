from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

from vid_to_sub_app.shared.constants import (
    DEFAULT_DEVICE,
    ENV_WHISPER_CPP_BIN,
    ENV_WHISPER_CPP_MODEL,
    ROOT_DIR,
)
from vid_to_sub_app.shared.env import (
    detect_best_device,
    detect_torch_device,
    find_whisper_cpp_bin,
    find_whisper_cpp_model_path,
)

from .output import parse_srt, parse_whisper_cpp_progress_seconds


def configure_torch_cpu_threads(threads: int) -> None:
    if threads < 1:
        return

    try:
        import torch
    except ImportError:
        return

    try:
        torch.set_num_threads(max(1, int(threads)))
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(max(1, min(int(threads), 4)))
    except Exception:
        pass


def resolve_device_fw(device: str) -> tuple[str, str]:
    if device == "auto":
        detected_device = detect_best_device()
        if detected_device == "cuda":
            return "cuda", "float16"
        if detected_device == "mps":
            print(
                "[WARN] faster-whisper does not support MPS; using CPU int8.",
                file=sys.stderr,
            )
            return "cpu", "int8"
        return "cpu", "int8"
    if device == "cuda":
        return "cuda", "float16"
    if device == "mps":
        print(
            "[WARN] faster-whisper does not support MPS; using CPU int8.",
            file=sys.stderr,
        )
        return "cpu", "int8"
    return "cpu", "int8"


def transcribe_faster_whisper(
    video: Path,
    model_name: str,
    device: str,
    language: Optional[str],
    beam_size: int,
    compute_type: Optional[str],
    threads: int,
) -> tuple[list[dict], dict]:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print(
            "[ERROR] faster-whisper is not installed.\n"
            "  Install with: pip install faster-whisper",
            file=sys.stderr,
        )
        sys.exit(1)

    dev, default_compute_type = resolve_device_fw(device)
    ct = compute_type or default_compute_type

    model = WhisperModel(
        model_name,
        device=dev,
        compute_type=ct,
        cpu_threads=max(1, int(threads)) if dev == "cpu" else 0,
    )
    segs_raw, info_raw = model.transcribe(
        str(video),
        language=language,
        beam_size=beam_size,
        word_timestamps=False,
    )

    segments = [{"start": seg.start, "end": seg.end, "text": seg.text} for seg in segs_raw]
    info = {
        "language": info_raw.language,
        "language_probability": round(info_raw.language_probability, 4),
        "duration": round(info_raw.duration, 3),
        "backend": "faster-whisper",
        "model": model_name,
    }
    return segments, info


def transcribe_openai_whisper(
    video: Path,
    model_name: str,
    device: str,
    language: Optional[str],
    beam_size: int,
    threads: int,
) -> tuple[list[dict], dict]:
    try:
        import whisper as ow
    except ImportError:
        print(
            "[ERROR] openai-whisper is not installed.\n"
            "  Install with: pip install openai-whisper",
            file=sys.stderr,
        )
        sys.exit(1)

    dev = detect_torch_device() if device == "auto" else device
    if dev == "cpu":
        configure_torch_cpu_threads(threads)

    model = ow.load_model(model_name, device=dev)
    result = model.transcribe(str(video), language=language, beam_size=beam_size)
    segments = [
        {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
        for segment in result.get("segments", [])
    ]
    info = {
        "language": result.get("language"),
        "backend": "openai-whisper",
        "model": model_name,
    }
    return segments, info


def transcribe_whisperx(
    video: Path,
    model_name: str,
    device: str,
    language: Optional[str],
    beam_size: int,
    compute_type: Optional[str],
    hf_token: Optional[str],
    diarize: bool,
    threads: int,
) -> tuple[list[dict], dict]:
    try:
        import whisperx
    except ImportError:
        print(
            "[ERROR] whisperx is not installed.\n  Install with: pip install whisperx",
            file=sys.stderr,
        )
        sys.exit(1)

    dev = detect_torch_device() if device == "auto" else device
    if dev == "cpu":
        configure_torch_cpu_threads(threads)

    ct = compute_type or ("float16" if dev == "cuda" else "int8")
    model = whisperx.load_model(model_name, dev, compute_type=ct, language=language)
    audio = whisperx.load_audio(str(video))
    result = model.transcribe(audio, batch_size=16, language=language)

    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=dev,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            dev,
            return_char_alignments=False,
        )
    except Exception as exc:
        print(
            f"[WARN] Alignment failed ({exc}); using unaligned segments.",
            file=sys.stderr,
        )

    if diarize:
        if not hf_token:
            print(
                "[WARN] --hf-token is required for diarization; skipping diarization.",
                file=sys.stderr,
            )
        else:
            try:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token,
                    device=dev,
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as exc:
                print(
                    f"[WARN] Diarization failed ({exc}); continuing without it.",
                    file=sys.stderr,
                )

    segments = [
        {"start": segment["start"], "end": segment["end"], "text": segment["text"]}
        for segment in result.get("segments", [])
    ]
    info = {
        "language": result.get("language"),
        "backend": "whisperx",
        "model": model_name,
    }
    return segments, info


def extract_audio_for_whisper_cpp(video: Path, wav_path: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required on PATH for whisper.cpp backend") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg audio extraction failed (exit {exc.returncode})"
        ) from exc


def default_whisper_cpp_model_path(model_name: str) -> Path:
    resolved = find_whisper_cpp_model_path(model_name)
    if resolved:
        return Path(resolved)
    return ROOT_DIR / "models" / f"ggml-{model_name}.bin"


def resolve_whisper_cpp_srt_path(output_prefix: Path, wav_path: Path) -> Path:
    candidates = [
        output_prefix.with_suffix(".srt"),
        output_prefix.parent / f"{output_prefix.name}.wav.srt",
        wav_path.with_suffix(".srt"),
        wav_path.parent / f"{wav_path.name}.srt",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    discovered = sorted(output_prefix.parent.glob("*.srt"))
    if len(discovered) == 1:
        return discovered[0]

    preferred = [
        path
        for path in discovered
        if path.name.startswith(output_prefix.name) or path.name.startswith(wav_path.name)
    ]
    if len(preferred) == 1:
        return preferred[0]

    found = ", ".join(path.name for path in discovered) if discovered else "none"
    expected = ", ".join(path.name for path in candidates)
    raise RuntimeError(
        "whisper.cpp did not produce expected SRT. "
        f"Expected one of: {expected}. Found: {found}"
    )


def transcribe_whisper_cpp(
    video: Path,
    model_name: str,
    device: str,
    language: Optional[str],
    threads: int,
    whisper_cpp_model_path: Optional[str],
    progress_callback: Optional[Callable[[float], None]] = None,
) -> tuple[list[dict], dict]:
    if device not in {"auto", DEFAULT_DEVICE}:
        print(
            "[WARN] whisper.cpp backend is CPU-oriented here; using CPU.",
            file=sys.stderr,
        )

    whisper_cpp_bin = find_whisper_cpp_bin()
    if not whisper_cpp_bin:
        raise RuntimeError(
            "whisper.cpp CLI not found. Set "
            f"{ENV_WHISPER_CPP_BIN} or install `whisper-cli` on PATH"
        )
    model_path_str = find_whisper_cpp_model_path(
        model_name,
        whisper_cpp_model_path,
        strict_configured=bool(whisper_cpp_model_path),
    )
    model_path = (
        Path(model_path_str)
        if model_path_str
        else default_whisper_cpp_model_path(model_name).resolve()
    )
    if not model_path.exists():
        raise RuntimeError(
            "whisper.cpp model file not found. Set --whisper-cpp-model-path or "
            f"{ENV_WHISPER_CPP_MODEL}, or place ggml-{model_name}.bin under "
            f"{ROOT_DIR / 'models'}. Expected: {model_path}"
        )

    with tempfile.TemporaryDirectory(prefix="vid_to_sub_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        wav_path = tmpdir_path / f"{video.stem}.wav"
        output_prefix = tmpdir_path / video.stem
        extract_audio_for_whisper_cpp(video, wav_path)

        command = [
            whisper_cpp_bin,
            "-m",
            str(model_path),
            "-f",
            str(wav_path),
            "-osrt",
            "-of",
            str(output_prefix),
            "-t",
            str(max(1, threads)),
        ]
        if language:
            command.extend(["-l", language])

        try:
            if progress_callback is None:
                subprocess.run(command, check=True)
            else:
                proc = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert proc.stdout is not None
                last_progress_seconds = -1.0
                try:
                    for raw in iter(proc.stdout.readline, ""):
                        line = raw.rstrip("\n")
                        print(line, flush=True)
                        progress_seconds = parse_whisper_cpp_progress_seconds(line)
                        if (
                            progress_seconds is not None
                            and progress_seconds > last_progress_seconds
                        ):
                            progress_callback(progress_seconds)
                            last_progress_seconds = progress_seconds
                finally:
                    proc.stdout.close()

                rc = proc.wait()
                if rc != 0:
                    raise subprocess.CalledProcessError(rc, command)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"whisper.cpp transcription failed (exit {exc.returncode})"
            ) from exc

        srt_path = resolve_whisper_cpp_srt_path(output_prefix, wav_path)

        segments = parse_srt(srt_path.read_text(encoding="utf-8"))
        info = {
            "language": language or "auto",
            "backend": "whisper-cpp",
            "model": model_name,
            "model_path": str(model_path),
        }
        return segments, info


def transcribe(
    video: Path,
    backend: str,
    model_name: str,
    device: str,
    language: Optional[str],
    beam_size: int,
    compute_type: Optional[str],
    hf_token: Optional[str],
    diarize: bool,
    whisper_cpp_model_path: Optional[str],
    threads: int,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> tuple[list[dict], dict]:
    if backend == "whisper-cpp":
        return transcribe_whisper_cpp(
            video,
            model_name,
            device,
            language,
            threads,
            whisper_cpp_model_path,
            progress_callback=progress_callback,
        )
    if backend == "faster-whisper":
        return transcribe_faster_whisper(
            video,
            model_name,
            device,
            language,
            beam_size,
            compute_type,
            threads,
        )
    if backend == "whisper":
        return transcribe_openai_whisper(
            video,
            model_name,
            device,
            language,
            beam_size,
            threads,
        )
    if backend == "whisperx":
        return transcribe_whisperx(
            video,
            model_name,
            device,
            language,
            beam_size,
            compute_type,
            hf_token,
            diarize,
            threads,
        )
    raise ValueError(f"Unknown backend: {backend!r}")
