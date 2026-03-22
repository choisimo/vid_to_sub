from __future__ import annotations

import subprocess
import importlib
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, TypedDict, cast

from vid_to_sub_app.shared.constants import (
    DEFAULT_DEVICE,
    ENV_WHISPER_CPP_BIN,
    ENV_WHISPER_CPP_MODEL,
    ROOT_DIR,
)
from vid_to_sub_app.shared.env import (
    detect_cuda_total_memory_gb,
    detect_best_device,
    detect_torch_device,
    faster_whisper_model_candidates,
    find_whisper_cpp_bin,
    find_whisper_cpp_model_path,
)

from .output import parse_srt, parse_whisper_cpp_progress_seconds

_FasterWhisperRequestedKey = tuple[str, str, str]
_FasterWhisperRuntimeKey = tuple[str, str, str]
_FasterWhisperModelCacheKey = tuple[str, str, str, int]
_faster_whisper_thread_state = threading.local()


class TranscriptSegment(TypedDict):
    start: float
    end: float
    text: str


class _FasterWhisperThreadCache(TypedDict):
    model_class_identity: int
    models: dict[_FasterWhisperModelCacheKey, Any]
    preferred_runtime: dict[_FasterWhisperRequestedKey, _FasterWhisperRuntimeKey]


def _normalize_content_type(content_type: Optional[str]) -> str:
    normalized = (content_type or "auto").strip().lower()
    if normalized not in {"auto", "speech", "music"}:
        raise ValueError(f"Unknown content type: {content_type!r}")
    return normalized


def _music_safe_backend_options(
    backend: str,
    language: Optional[str],
) -> dict[str, Any]:
    if not language:
        raise RuntimeError(
            "--content-type music requires --language because automatic language "
            "detection is unreliable on music-dominant audio."
        )
    if backend == "faster-whisper":
        return {
            "condition_on_previous_text": False,
            "vad_filter": True,
        }
    if backend == "whisper":
        return {
            "condition_on_previous_text": False,
        }
    raise RuntimeError(
        f"--content-type music is not supported for backend '{backend}'. "
        "Use --backend faster-whisper or --backend whisper."
    )


def _transcribe_options_for_content_type(
    backend: str,
    content_type: Optional[str],
    language: Optional[str],
) -> tuple[str, dict[str, Any]]:
    normalized = _normalize_content_type(content_type)
    if normalized != "music":
        return normalized, {}
    return normalized, _music_safe_backend_options(backend, language)


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


def _extract_faster_whisper_model_dir(error_message: str) -> Path | None:
    match = re.search(r"in model ['\"]([^'\"]+)['\"]", error_message)
    if not match:
        return None
    return Path(match.group(1)).expanduser()


def _describe_faster_whisper_load_error(model_name: str, exc: Exception) -> str:
    raw_message = str(exc).strip() or exc.__class__.__name__
    model_dir = _extract_faster_whisper_model_dir(raw_message)
    if "Unable to open file 'model.bin'" not in raw_message or model_dir is None:
        return (
            f"faster-whisper failed while loading model '{model_name}': {raw_message}"
        )

    cache_dir = model_dir
    if model_dir.parent.name == "snapshots":
        cache_dir = model_dir.parent.parent

    message_parts = [
        f"faster-whisper model cache is incomplete or corrupted for '{model_name}'.",
        f"Remove '{cache_dir}' and rerun to download the model again.",
    ]

    blob_dir = cache_dir / "blobs"
    if blob_dir.exists():
        incomplete_blobs = sorted(path.name for path in blob_dir.glob("*.incomplete"))
        if incomplete_blobs:
            displayed = ", ".join(incomplete_blobs[:3])
            remaining = len(incomplete_blobs) - 3
            if remaining > 0:
                displayed = f"{displayed}, +{remaining} more"
            message_parts.append(f"Incomplete blob(s) detected: {displayed}.")

    message_parts.append(f"Original error: {raw_message}")
    return " ".join(message_parts)


def _is_cuda_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _release_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _announce_faster_whisper_load(
    model_name: str, device: str, compute_type: str
) -> None:
    print(
        f"[INFO] Loading faster-whisper model '{model_name}' on {device} ({compute_type}). "
        "First run may take time while the model is downloaded or loaded.",
        flush=True,
    )


def _faster_whisper_cpu_threads(device: str, threads: int) -> int:
    if device != "cpu":
        return 0
    return max(1, int(threads))


def _get_faster_whisper_thread_cache(
    model_class_identity: int,
) -> _FasterWhisperThreadCache:
    state = getattr(_faster_whisper_thread_state, "state", None)
    if (
        state is None
        or not isinstance(state, dict)
        or state.get("model_class_identity") != model_class_identity
    ):
        state = {
            "model_class_identity": model_class_identity,
            "models": {},
            "preferred_runtime": {},
        }
        _faster_whisper_thread_state.state = state
    models = state.get("models")
    preferred_runtime = state.get("preferred_runtime")
    if not isinstance(models, dict) or not isinstance(preferred_runtime, dict):
        refreshed_state: _FasterWhisperThreadCache = {
            "model_class_identity": model_class_identity,
            "models": {},
            "preferred_runtime": {},
        }
        _faster_whisper_thread_state.state = refreshed_state
        return refreshed_state
    return {
        "model_class_identity": model_class_identity,
        "models": cast(dict[_FasterWhisperModelCacheKey, Any], models),
        "preferred_runtime": cast(
            dict[_FasterWhisperRequestedKey, _FasterWhisperRuntimeKey],
            preferred_runtime,
        ),
    }


def _result_mapping(result: object, backend: str) -> Mapping[str, object]:
    if not isinstance(result, Mapping):
        raise RuntimeError(f"{backend} returned an unexpected result payload.")
    return result


def _segments_from_mapping(result: Mapping[str, object]) -> list[TranscriptSegment]:
    segments_raw = result.get("segments", [])
    if not isinstance(segments_raw, list):
        return []

    segments: list[TranscriptSegment] = []
    for segment in segments_raw:
        if not isinstance(segment, Mapping):
            continue
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text")
        if not isinstance(start, (int, float)):
            continue
        if not isinstance(end, (int, float)):
            continue
        if not isinstance(text, str):
            continue
        segments.append({"start": float(start), "end": float(end), "text": text})
    return segments


def _preferred_gpu_candidates(
    candidates: list[str],
    preferred_runtime: _FasterWhisperRuntimeKey | None,
) -> list[str]:
    if not preferred_runtime or preferred_runtime[1] != "cuda":
        return candidates

    preferred_model = preferred_runtime[0]
    if preferred_model not in candidates:
        return [preferred_model, *candidates]

    return [
        preferred_model,
        *[candidate for candidate in candidates if candidate != preferred_model],
    ]


def _coerce_preferred_runtime(value: object) -> _FasterWhisperRuntimeKey | None:
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    model_name, device, compute_type = value
    if not all(isinstance(item, str) for item in value):
        return None
    return model_name, device, compute_type


def transcribe_faster_whisper(
    video: Path,
    model_name: str,
    device: str,
    language: Optional[str],
    content_type: str,
    beam_size: int,
    compute_type: Optional[str],
    threads: int,
) -> tuple[list[TranscriptSegment], dict[str, object]]:
    normalized_content_type, extra_transcribe_options = (
        _transcribe_options_for_content_type("faster-whisper", content_type, language)
    )
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
    requested_runtime: _FasterWhisperRequestedKey = (model_name, dev, ct)
    thread_cache = _get_faster_whisper_thread_cache(id(WhisperModel))
    model_cache = thread_cache["models"]
    preferred_runtime_map = thread_cache["preferred_runtime"]
    candidate_chain = faster_whisper_model_candidates(model_name)
    selected_model = model_name
    segs_raw: Any = []
    info_raw: Any | None = None

    def load_model(
        target_model: str, target_device: str, target_compute_type: str
    ) -> Any:
        cache_key: _FasterWhisperModelCacheKey = (
            target_model,
            target_device,
            target_compute_type,
            _faster_whisper_cpu_threads(target_device, threads),
        )
        cached_model = model_cache.get(cache_key)
        if cached_model is not None:
            return cached_model

        _announce_faster_whisper_load(target_model, target_device, target_compute_type)
        model = WhisperModel(
            target_model,
            device=target_device,
            compute_type=target_compute_type,
            cpu_threads=cache_key[3],
        )
        model_cache[cache_key] = model
        return model

    if dev == "cuda":
        available_vram_gb = detect_cuda_total_memory_gb()
        gpu_candidates = faster_whisper_model_candidates(
            model_name,
            available_vram_gb=available_vram_gb,
        )
        preferred_runtime = _coerce_preferred_runtime(
            preferred_runtime_map.get(requested_runtime)
        )
        gpu_succeeded = False
        if (
            gpu_candidates
            and gpu_candidates[0] != model_name
            and preferred_runtime is None
        ):
            vram_summary = (
                f"{available_vram_gb:.1f} GiB"
                if available_vram_gb is not None
                else "unknown VRAM"
            )
            print(
                f"[WARN] Detected CUDA memory {vram_summary}; "
                f"starting with model '{gpu_candidates[0]}' instead of '{model_name}'.",
                file=sys.stderr,
                flush=True,
            )

        reuse_cpu_fallback = (
            preferred_runtime is not None and preferred_runtime[1] == "cpu"
        )
        if reuse_cpu_fallback:
            assert preferred_runtime is not None
            dev = preferred_runtime[1]
            ct = preferred_runtime[2]
            selected_model = preferred_runtime[0]
        else:
            gpu_candidates = _preferred_gpu_candidates(
                gpu_candidates, preferred_runtime
            )

        if reuse_cpu_fallback:
            model = load_model(selected_model, dev, ct)
            segs_raw, info_raw = model.transcribe(
                str(video),
                language=language,
                beam_size=beam_size,
                word_timestamps=False,
                **extra_transcribe_options,
            )
            gpu_succeeded = True
            preferred_runtime_map[requested_runtime] = (selected_model, dev, ct)
        elif gpu_candidates:
            for index, candidate_model in enumerate(gpu_candidates):
                selected_model = candidate_model
                try:
                    model = load_model(selected_model, dev, ct)
                except Exception as exc:
                    if _is_cuda_oom_error(exc):
                        _release_cuda_cache()
                        next_candidate = (
                            gpu_candidates[index + 1]
                            if index + 1 < len(gpu_candidates)
                            else None
                        )
                        if next_candidate:
                            print(
                                f"[WARN] faster-whisper model '{selected_model}' "
                                "ran out of CUDA memory during model load; "
                                f"retrying with '{next_candidate}'.",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                        break
                    raise RuntimeError(
                        _describe_faster_whisper_load_error(selected_model, exc)
                    ) from exc

                try:
                    segs_raw, info_raw = model.transcribe(
                        str(video),
                        language=language,
                        beam_size=beam_size,
                        word_timestamps=False,
                        **extra_transcribe_options,
                    )
                    gpu_succeeded = True
                    preferred_runtime_map[requested_runtime] = (selected_model, dev, ct)
                    break
                except Exception as exc:
                    if _is_cuda_oom_error(exc):
                        _release_cuda_cache()
                        next_candidate = (
                            gpu_candidates[index + 1]
                            if index + 1 < len(gpu_candidates)
                            else None
                        )
                        if next_candidate:
                            print(
                                f"[WARN] faster-whisper model '{selected_model}' "
                                "ran out of CUDA memory during transcription; "
                                f"retrying with '{next_candidate}'.",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                        break
                    raise
        if not reuse_cpu_fallback and (not gpu_candidates or not gpu_succeeded):
            fallback_model = candidate_chain[-1] if candidate_chain else model_name
            print(
                f"[WARN] faster-whisper GPU candidates exhausted; "
                f"falling back to CPU int8 with '{fallback_model}'.",
                file=sys.stderr,
                flush=True,
            )
            dev = "cpu"
            ct = "int8"
            selected_model = fallback_model
            try:
                model = load_model(selected_model, dev, ct)
            except Exception as exc:
                raise RuntimeError(
                    _describe_faster_whisper_load_error(selected_model, exc)
                ) from exc
            segs_raw, info_raw = model.transcribe(
                str(video),
                language=language,
                beam_size=beam_size,
                word_timestamps=False,
                **extra_transcribe_options,
            )
            preferred_runtime_map[requested_runtime] = (selected_model, dev, ct)
    else:
        preferred_runtime = _coerce_preferred_runtime(
            preferred_runtime_map.get(requested_runtime)
        )
        if (
            preferred_runtime
            and preferred_runtime[1] == dev
            and preferred_runtime[2] == ct
        ):
            selected_model = preferred_runtime[0]
        try:
            model = load_model(selected_model, dev, ct)
        except Exception as exc:
            raise RuntimeError(
                _describe_faster_whisper_load_error(selected_model, exc)
            ) from exc
        segs_raw, info_raw = model.transcribe(
            str(video),
            language=language,
            beam_size=beam_size,
            word_timestamps=False,
            **extra_transcribe_options,
        )
        preferred_runtime_map[requested_runtime] = (selected_model, dev, ct)

    if info_raw is None:
        raise RuntimeError("faster-whisper returned no transcription metadata.")

    segments: list[TranscriptSegment] = [
        {"start": seg.start, "end": seg.end, "text": seg.text} for seg in segs_raw
    ]
    info: dict[str, object] = {
        "language": info_raw.language,
        "language_probability": round(info_raw.language_probability, 4),
        "duration": round(info_raw.duration, 3),
        "backend": "faster-whisper",
        "model": selected_model,
        "content_type": normalized_content_type,
    }
    return segments, info


def transcribe_openai_whisper(
    video: Path,
    model_name: str,
    device: str,
    language: Optional[str],
    content_type: str,
    beam_size: int,
    threads: int,
) -> tuple[list[TranscriptSegment], dict[str, object]]:
    normalized_content_type, extra_transcribe_options = (
        _transcribe_options_for_content_type("whisper", content_type, language)
    )
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
    result = model.transcribe(
        str(video),
        language=language,
        beam_size=beam_size,
        **extra_transcribe_options,
    )
    result_map = _result_mapping(result, "openai-whisper")
    segments = _segments_from_mapping(result_map)
    info: dict[str, object] = {
        "language": result_map.get("language"),
        "backend": "openai-whisper",
        "model": model_name,
        "content_type": normalized_content_type,
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
) -> tuple[list[TranscriptSegment], dict[str, object]]:
    try:
        whisperx = importlib.import_module("whisperx")
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
    result_map = _result_mapping(result, "whisperx")

    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=result_map["language"],
            device=dev,
        )
        result = whisperx.align(
            result_map["segments"],
            align_model,
            metadata,
            audio,
            dev,
            return_char_alignments=False,
        )
        result_map = _result_mapping(result, "whisperx")
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
                result_map = _result_mapping(result, "whisperx")
            except Exception as exc:
                print(
                    f"[WARN] Diarization failed ({exc}); continuing without it.",
                    file=sys.stderr,
                )

    segments = _segments_from_mapping(result_map)
    info: dict[str, object] = {
        "language": result_map.get("language"),
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
        raise RuntimeError(
            "ffmpeg is required on PATH for whisper.cpp backend"
        ) from exc
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
        if path.name.startswith(output_prefix.name)
        or path.name.startswith(wav_path.name)
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
) -> tuple[list[TranscriptSegment], dict[str, object]]:
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

        segments = _segments_from_mapping(
            {"segments": parse_srt(srt_path.read_text(encoding="utf-8"))}
        )
        info: dict[str, object] = {
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
    content_type: str,
    beam_size: int,
    compute_type: Optional[str],
    hf_token: Optional[str],
    diarize: bool,
    whisper_cpp_model_path: Optional[str],
    threads: int,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> tuple[list[TranscriptSegment], dict[str, object]]:
    if backend == "whisper-cpp":
        _transcribe_options_for_content_type("whisper-cpp", content_type, language)
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
            content_type,
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
            content_type,
            beam_size,
            threads,
        )
    if backend == "whisperx":
        _transcribe_options_for_content_type("whisperx", content_type, language)
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
    _normalize_content_type(content_type)
    raise ValueError(f"Unknown backend: {backend!r}")
