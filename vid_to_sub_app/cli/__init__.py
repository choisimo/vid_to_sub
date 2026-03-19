from __future__ import annotations

from .discovery import discover_videos, hash_video_folder
from .main import build_parser, main
from .manifest import (
    FolderAwareScheduler,
    ProcessResult,
    apply_runtime_path_map_to_manifest,
    build_run_manifest,
    load_manifest_from_stdin,
    persist_folder_manifest_state,
)
from .output import (
    fmt_seconds,
    parse_srt,
    parse_whisper_cpp_progress_seconds,
    probe_media_duration,
)
from .runner import (
    emit_progress_event,
    primary_output_exists,
    process_one,
    run_parallel,
    translation_capable,
)
from .transcription import (
    default_whisper_cpp_model_path,
    extract_audio_for_whisper_cpp,
    resolve_device_fw,
    transcribe,
    transcribe_faster_whisper,
    transcribe_openai_whisper,
    transcribe_whisper_cpp,
    transcribe_whisperx,
)
from .translation import (
    extract_json_array,
    postprocess_translated_segments_openai_compatible,
    translate_segments_openai_compatible,
)
