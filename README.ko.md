# vid_to_sub

[English](README.md) | 한국어

`vid_to_sub`는 비디오 파일을 재귀적으로 찾아 자막 또는 텍스트 파일로 저장하는 도구입니다. 런타임 기본값은 로컬 환경에서 GPU를 활용할 수 있으면 그에 맞는 백엔드를 우선 선택하고, 그렇지 않으면 `ffmpeg` + `whisper.cpp` 기반 CPU 전사로 되돌아가며, 필요하면 OpenAI 호환 API로 자막 번역과 후처리 교정 agent 단계도 추가할 수 있습니다.

## 스크린샷

### 입력 경로 선택과 작업 큐 준비

![Browse 탭](docs/images/tui-browse.png)

`Browse` 탭에서는 폴더나 파일을 추가하고, 현재 루트 기준으로 빠르게 검색하고, 실행 전 출력 디렉터리를 정할 수 있습니다.

### 의존성 점검과 설치

![Setup 탭](docs/images/tui-setup.png)

`Setup` 탭은 `ffmpeg`, `whisper-cli`, GGML 모델, 선택형 Python 백엔드를 점검하고 설치/빌드 작업을 한곳에서 제공합니다.

### 전사 및 번역 설정

![Transcribe 탭](docs/images/tui-transcribe.png)

`Transcribe` 탭에서 백엔드, 모델, 장치, 실행 모드, 출력 형식, 번역 여부, 고급 오버라이드를 설정합니다.

## 지원 기능

- `mp4`, `mkv`, `mov`, `avi`, `webm`, `ts` 등 공통 비디오 포맷 재귀 탐색
- 감지된 CUDA에서는 `faster-whisper`, 지원되는 Torch 장치에서는 `whisperX` 또는 `openai-whisper`, 그 외에는 CPU `whisper.cpp`를 기본으로 선택하는 런타임 기본값
- 선택형 백엔드: `faster-whisper`, `openai-whisper`, `whisperX`
- 출력 형식: `srt`, `vtt`, `txt`, `tsv`, `json`, `all`
- OpenAI 호환 Chat Completions API 기반 번역
- 별도 모델/API 설정이 가능한 후처리 교정 agent
- `auto` 모드에서 웹 검색/MCP 도구가 가능하면 참조 교정을 시도하고, 불가능하면 문맥 기반으로 자연스럽게 다듬는 후처리
- 번역 시 원본 자막의 타임스탬프 경계 유지
- Browse / Setup / Transcribe / History / Settings / Agent / Logs 로 구성된 7탭 TUI
- SQLite 기반 설정 저장 및 실행 이력 관리
- SSH 리소스 프로필 기반 분산 실행

## 진입점

- `python vid_to_sub.py ...`
  재귀 탐색과 배치 전사를 위한 CLI 진입점입니다.
- `python tui.py`
  Textual 기반 TUI 진입점입니다. 첫 실행 시 프로젝트 로컬 `.venv`를 만들고, 누락된 requirement 그룹을 설치한 뒤 다시 실행합니다.
- `python init_checker.py`
  관리용 가상환경을 준비하는 부트스트랩 도우미입니다.

## 요구 사항

### 시스템

- Python 3.9+
- `PATH` 상의 `ffmpeg`
- 기본 백엔드를 위한 `whisper.cpp`의 `whisper-cli`
- 기본 모델용 `ggml-large-v3.bin`

### Python 패키지

기본 패키지:

```bash
pip install -r requirements.txt
```

선택형 백엔드 패키지:

```bash
pip install -r requirements-faster-whisper.txt
pip install -r requirements-whisper.txt
pip install -r requirements-whisperx.txt
```

## 빠른 시작

### 1. 기본 로컬 전사

```bash
python vid_to_sub.py /path/to/videos
```

기본 동작:

- 디렉터리를 재귀적으로 탐색합니다.
- 모델은 `large-v3`를 사용합니다.
- 로컬 환경에서 가장 적합한 백엔드/장치를 자동 선택합니다. 우선순위는 CUDA `faster-whisper`, 그다음 Torch 기반 `whisperX` 또는 `openai-whisper`, 마지막이 CPU `whisper-cpp`입니다.
- CPU로 전사할 때는 가용 CPU 스레드 수를 `--workers` 수에 맞춰 자동 분배합니다.
- 각 원본 파일 옆에 `movie.srt`를 생성합니다.

런타임 기본값을 명시적으로 덮어쓰고 싶다면 `--backend`, `--device`, `--backend-threads`를 사용하면 됩니다.

### 2. 타이밍을 유지한 자막 번역

먼저 번역용 환경 변수를 설정합니다.

```bash
export VID_TO_SUB_TRANSLATION_BASE_URL=https://your-host/v1
export VID_TO_SUB_TRANSLATION_API_KEY=your_api_key
export VID_TO_SUB_TRANSLATION_MODEL=your_model
```

그다음 실행합니다.

```bash
python vid_to_sub.py /path/to/videos --translate-to ko
```

1차 번역 뒤에 별도 교정 agent를 한 번 더 태우려면 다음처럼 실행합니다.

```bash
python vid_to_sub.py /path/to/videos --translate-to ko --postprocess-translation
```

후처리 전략은 `--postprocess-mode auto|web_lookup|context_polish`로 고를 수 있습니다. `auto`는 서빙 agent가 웹 검색이나 MCP 도구를 지원하면 표준 가사/유사 텍스트 확인을 우선 시도하고, 그런 도구가 없으면 자연스럽게 문맥 보정으로 넘어가도록 프롬프트를 구성합니다.

예를 들어 아래 두 파일이 함께 생성됩니다.

- `movie.srt`
- `movie.ko.srt`

번역 시 바뀌는 것은 자막 텍스트뿐이며, `start`/`end` 타임스탬프는 원본 세그먼트 값을 그대로 사용합니다.

### 3. TUI 사용

```bash
python tui.py
```

권장 사용 순서:

1. `Browse`: 입력 폴더/파일을 추가하고, 필요하면 `Output dir`, `No recurse`, `Skip existing`를 조정합니다.
2. `Setup`: 의존성을 탐지하고 Python 백엔드 패키지 설치, `whisper.cpp` 빌드, GGML 모델 다운로드를 진행합니다.
3. `Transcribe`: 백엔드, 모델, 장치, 출력 형식, 번역 대상 언어, 실행 모드를 설정합니다.
4. `Ctrl+R`로 실행하고, `Ctrl+D`로 드라이런, `Ctrl+K`로 중지합니다.
5. `History`에서 이전 작업을 확인하고, `Settings`에 기본값을 저장하고, `Agent`에서 검토 가능한 가이드나 실행 계획을 받을 수 있습니다.

유용한 단축키:

- `Ctrl+R` 실행
- `Ctrl+D` 드라이런
- `Ctrl+K` 중지
- `Ctrl+S` 설정 저장
- `1` 부터 `7`까지 탭 이동
- `Ctrl+Q` 종료

## 자주 쓰는 CLI 예시

출력 파일을 별도 디렉터리에 저장:

```bash
python vid_to_sub.py /path/to/videos -o /path/to/output
```

재귀 탐색 끄기:

```bash
python vid_to_sub.py /path/to/videos --no-recurse
```

이미 기본 출력이 있는 파일 건너뛰기:

```bash
python vid_to_sub.py /path/to/videos --skip-existing
```

실행 없이 큐만 확인:

```bash
python vid_to_sub.py /path/to/videos --dry-run
```

여러 출력 형식 동시 생성:

```bash
python vid_to_sub.py /path/to/videos --format srt --format json
```

명시적인 `whisper.cpp` 모델 경로 사용:

```bash
python vid_to_sub.py /path/to/videos \
  --backend whisper-cpp \
  --model large-v3 \
  --whisper-cpp-model-path /models/ggml-large-v3.bin
```

내장 모델 식별자 목록 확인:

```bash
python vid_to_sub.py --list-models
```

## 환경 변수

### `whisper.cpp`

- `VID_TO_SUB_WHISPER_CPP_BIN`
  `whisper-cli` 실행 파일 경로를 직접 지정합니다.
- `VID_TO_SUB_WHISPER_CPP_MODEL`
  GGML 모델 경로를 직접 지정합니다.

`VID_TO_SUB_WHISPER_CPP_MODEL`이 비어 있으면 프로젝트는 `./models`, `~/.cache/whisper`, `~/models`, `/models`, `/opt/models` 같은 경로를 순서대로 탐색합니다.

### 번역 API

- `VID_TO_SUB_TRANSLATION_BASE_URL`
  `https://host/v1` 같은 API 루트나 전체 `/chat/completions` 엔드포인트를 모두 받을 수 있습니다.
- `VID_TO_SUB_TRANSLATION_API_KEY`
  번역 서비스 Bearer 토큰입니다.
- `VID_TO_SUB_TRANSLATION_MODEL`
  번역에 사용할 모델 이름입니다.

### 후처리 API

- `VID_TO_SUB_POSTPROCESS_BASE_URL`
  자막 후처리 agent 전용 엔드포인트입니다. 비어 있으면 번역 Base URL을 재사용합니다.
- `VID_TO_SUB_POSTPROCESS_API_KEY`
  자막 후처리 agent 전용 Bearer 토큰입니다. 비어 있으면 번역 API 키를 재사용합니다.
- `VID_TO_SUB_POSTPROCESS_MODEL`
  자막 후처리 agent 전용 모델입니다. 비어 있으면 번역 모델을 재사용합니다.

### Agent 탭

- `VID_TO_SUB_AGENT_BASE_URL`
- `VID_TO_SUB_AGENT_API_KEY`
- `VID_TO_SUB_AGENT_MODEL`

TUI에서 이 값들이 비어 있으면 Agent 탭은 번역 API 설정을 그대로 재사용합니다.

## 분산 실행

TUI는 SSH 리소스 프로필 기반 분산 모드를 지원합니다. `Settings -> Remote Resources`에 프로필 JSON을 입력한 뒤, `Transcribe` 탭의 `Execution -> Mode`를 `distributed`로 바꾸고 실행하면 됩니다.

예시 JSON:

```json
[
  {
    "name": "gpu-box",
    "ssh_target": "user@gpu-host",
    "remote_workdir": "/srv/vid_to_sub",
    "slots": 2,
    "path_map": {
      "/mnt/media": "/srv/media"
    },
    "env": {
      "VID_TO_SUB_WHISPER_CPP_MODEL": "/models/ggml-large-v3.bin"
    }
  }
]
```

필드 의미:

- `slots`는 해당 호스트에 배정할 작업량 비중입니다.
- `path_map`은 로컬 경로 prefix를 원격 경로로 치환합니다.
- `env`는 원격 실행 시 덮어쓸 환경 변수입니다.

## 출력 파일 이름

기본 전사 출력:

- `movie.srt`
- `movie.vtt`
- `movie.txt`
- `movie.tsv`
- `movie.json`

`--translate-to ko` 사용 시:

- `movie.ko.srt`
- `movie.ko.vtt`
- `movie.ko.txt`
- `movie.ko.tsv`
- `movie.ko.json`

## 사용 방안

### 빠른 로컬 일괄 전사

CPU 기반 기본 흐름이 필요하면 `whisper.cpp` 기본 설정으로 시작하는 것이 가장 단순합니다.

### 타이밍 유지 번역

기존 자막 구간은 그대로 두고 텍스트만 다른 언어로 바꾸고 싶다면 `--translate-to <lang>`가 맞습니다.

### 운영형 사용

설치 보조, 설정 저장, 이력 확인, 분산 실행까지 한 화면에서 다루려면 `tui.py` 기반 작업 흐름이 가장 적합합니다.

## 참고

- CLI와 TUI는 같은 런타임 백엔드/장치 감지 로직을 사용하므로, 선택형 패키지가 설치된 GPU 호스트에서는 그에 맞는 기본값이 자동으로 잡힙니다.
- `whisperX` 화자 분리는 `--hf-token`이 있어야 실제로 동작하고, 없으면 경고 후 일반 전사만 진행합니다.
- `--skip-existing`는 대상 출력 디렉터리에서 `movie.srt` 같은 기본 출력 파일 존재 여부만 기준으로 판단합니다.
- `Settings` 탭에서 현재 설정을 `.env`로 내보낼 수 있습니다.
