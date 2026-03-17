# vid_to_sub

Recursively transcribe video files to subtitle files.

Default flow:
- transcription: `whisper.cpp` on CPU with `large-v3`
- output: `.srt`
- optional translation: OpenAI-compatible API while preserving the original subtitle timings/frame boundaries exactly

## Features

- Recursive directory scan for common video formats
- Default CPU pipeline via `whisper.cpp` + `ffmpeg`
- Optional backends: `faster-whisper`, `openai-whisper`, `whisperX`
- Output formats: `srt`, `vtt`, `txt`, `tsv`, `json`
- Translation mode that keeps original `start/end` timings and only replaces subtitle text
- OpenAI-compatible translation config via environment variables
- `--skip-existing`, `--dry-run`, `--workers`
- **Interactive TUI** (`tui.py`) — 5-tab terminal UI backed by SQLite

## TUI

```bash
./tui.py
# or
python tui.py
```

Tabs:
- **1 Browse** — DirectoryTree path browser, manual path entry, recent paths
- **2 Setup** — auto-detect deps, build whisper.cpp, download ggml models, pip install
- **3 Transcribe** — all job settings (backend, model, format, translation)
- **4 History** — SQLite-backed job history with status and timing
- **5 Settings** — persistent config (replaces `.env`), export to `.env`

Keyboard shortcuts: `Ctrl+R` run · `Ctrl+D` dry-run · `Ctrl+K` kill · `Ctrl+S` save · `1-5` tabs · `Ctrl+Q` quit

## Requirements

### System

- Python 3.9+
- `ffmpeg` on `PATH`
- For default backend: `whisper-cli` from `whisper.cpp` on `PATH`
- `ggml-large-v3.bin` model file for `whisper.cpp`

### Python packages

```bash
pip install -r requirements.txt
```

Optional backends:

```bash
pip install -r requirements-faster-whisper.txt  # faster-whisper (recommended Python backend)
pip install -r requirements-whisper.txt          # openai-whisper
pip install -r requirements-whisperx.txt         # whisperX + diarization
```

## whisper.cpp setup

Example build/install flow:

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build
cmake --build build -j
```

Then either:
- put `whisper-cli` on `PATH`, or
- set `VID_TO_SUB_WHISPER_CPP_BIN`

For the model file, either:
- place it at `./models/ggml-large-v3.bin`, or
- set `VID_TO_SUB_WHISPER_CPP_MODEL`

Example:

```bash
export VID_TO_SUB_WHISPER_CPP_BIN=/path/to/whisper.cpp/build/bin/whisper-cli
export VID_TO_SUB_WHISPER_CPP_MODEL=/path/to/models/ggml-large-v3.bin
```

## Translation environment variables

Set these to use subtitle translation with an OpenAI-compatible server:

```bash
export VID_TO_SUB_TRANSLATION_BASE_URL=https://your-openai-compatible-host/v1
export VID_TO_SUB_TRANSLATION_API_KEY=your_api_key
export VID_TO_SUB_TRANSLATION_MODEL=your_translation_model
```

Notes:
- base URL can be either the API root like `.../v1` or the full `.../chat/completions` URL
- translated subtitle files are written with a language suffix, for example `movie.ko.srt`
- timings are copied from the original transcription, so subtitle frames stay identical

## Quick start

Default CPU transcription with `whisper.cpp large-v3`:

```bash
python vid_to_sub.py ./videos
```

This writes `movie.srt` next to each video.

Translate to Korean while preserving subtitle frames:

```bash
python vid_to_sub.py ./videos --translate-to ko
```

This writes both:
- `movie.srt`
- `movie.ko.srt`

## Common examples

Use explicit `whisper.cpp` model path:

```bash
python vid_to_sub.py ./videos \
  --backend whisper-cpp \
  --model large-v3 \
  --device cpu \
  --whisper-cpp-model-path /models/ggml-large-v3.bin
```

Translate to Japanese with a specific API model override:

```bash
python vid_to_sub.py ./videos \
  --translate-to ja \
  --translation-model gpt-4.1-mini
```

Write multiple output formats:

```bash
python vid_to_sub.py ./videos \
  --format srt \
  --format json
```

Dry run:

```bash
python vid_to_sub.py ./videos --dry-run
```

Skip files that already have primary outputs:

```bash
python vid_to_sub.py ./videos --skip-existing
```

## CLI

```text
usage: vid_to_sub [-h] [-o DIR] [--no-recurse] [--skip-existing] [--dry-run]
                  [--backend {whisper-cpp,faster-whisper,whisper,whisperx}]
                  [--model MODEL] [--language LANG]
                  [--device {auto,cpu,cuda,mps}] [--compute-type TYPE]
                  [--beam-size N] [--format FMT]
                  [--whisper-cpp-model-path PATH] [--hf-token TOKEN]
                  [--diarize] [--translate-to LANG]
                  [--translation-model MODEL]
                  [--translation-base-url URL]
                  [--translation-api-key KEY]
                  [--workers N] [-v] [--list-models]
                  PATH [PATH ...]
```

## Output naming

Original transcription:
- `movie.srt`
- `movie.vtt`
- `movie.json`

Translated output with `--translate-to ko`:
- `movie.ko.srt`
- `movie.ko.vtt`
- `movie.ko.json`

## Notes

- In this tool, `whisper.cpp` is treated as the default CPU backend.
- `ffmpeg` is used to extract mono 16k WAV audio before calling `whisper-cli`.
- Translation only changes subtitle text; timestamps are preserved exactly from the original subtitle segments.
- `whisperX` diarization still requires Hugging Face access for `pyannote`.
