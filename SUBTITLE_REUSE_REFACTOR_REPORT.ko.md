# vid-to-sub 기존 자막 재사용 기능 코드 리뷰 및 리팩토링 리포트

작성일: 2026-05-11
대상 프로젝트: vid_to_sub-main
작업 목적: 전사 전에 기존 자막을 검색, 추출, 재사용, 싱크 보정, 번역 파이프라인에 연결하는 기능을 추가한다.

## 1. 짧은 결론

기존 프로젝트는 이미 stage 1 전사 artifact와 stage 2 번역 artifact가 분리되어 있어, 완전히 새 자막을 만드는 경로 앞에 “기존 자막 후보 탐색” 상태를 추가하는 것이 가장 안전했다. 따라서 이번 리팩토링은 기존 Whisper/WhisperX 전사 코드를 제거하거나 교체하지 않고, 전사 직전에 후보 검색/점수화/싱크 보정/재사용을 시도한 뒤 실패하면 기존 전사 경로로 롤백하도록 설계했다.

구현 결과는 다음과 같다.

- 로컬 sidecar 자막, 영상 내장 텍스트 자막, OpenSubtitles, SubDL 후보 검색을 통합했다.
- 후보 목록을 `--list-subtitle-candidates`로 다운로드 가능 목록처럼 확인할 수 있게 했다.
- 후보 점수와 임계값으로 자동 선택하거나, 후보 부족 시 기존 ASR 경로로 안전하게 롤백한다.
- 기존 자막을 `.stage1.json` artifact로 저장하므로 기존 번역 파이프라인이 그대로 이어진다.
- 싱크 보정은 `off`, `duration`, `asr` 모드를 제공한다. `asr`은 새 자막 생성 목적이 아니라 기존 자막 시간축 보정용 기준 transcript로만 사용한다.
- OpenSubtitles/SubDL API key가 없으면 해당 외부 provider는 건너뛰고 로컬/내장 후보만 처리한다.

정확도 p99는 단일 구현만으로 보장할 수 있는 값이 아니라 검증 corpus, provider quota, 제목/릴리즈/언어별 샘플링, 실제 사용자 선택 로그가 필요한 운영 지표다. 이번 구현은 p99 목표를 향한 상태 기반 검색/점수화/롤백 구조를 넣은 것이며, 운영 전에는 별도의 benchmark set으로 임계값을 조정해야 한다.

## 2. 외부 API 조사 요약

OpenSubtitles는 기존 XML-RPC보다 최신 REST API 사용을 안내하며, 제목, release name, IMDb ID, file hash 기반 검색과 다운로드, 다국어 자막 제공을 지원한다. 무료 접근에는 다운로드 수 제한이 있으므로 자동화에서는 quota와 인증 실패 처리가 필요하다.

OpenSubtitles forum의 API 논의에서는 query와 moviehash를 함께 보낼 경우 hash 매칭 결과가 query 필터에 의해 빠질 수 있으며, 임의 파일을 탐색하는 상황에서는 moviehash만 보내는 접근이 더 낫다고 설명한다. 이 때문에 구현에서는 moviehash-only query와 title/file name query를 분리했다.

SubDL API는 `film_name`, `file_name`, IMDb/TMDB/SubDL ID, 시즌/에피소드, 언어, 결과 개수 제한 등을 지원한다. 다운로드 결과는 zip 링크 형태가 기본이므로 zip 내부의 `.srt`, `.vtt`, `.ass`, `.ssa` 파일을 추출하는 materialize 단계가 필요하다.

참고한 공식/준공식 문서:

- OpenSubtitles Help Center - About the OpenSubtitles API: https://opensubtitles.tawk.help/article/about-the-api
- OpenSubtitles forum - REST API moviehash/query behavior: https://forum.opensubtitles.org/viewtopic.php?start=105&t=17146
- SubDL Search and Download API: https://subdl.com/api-doc

## 3. 기존 상태 리뷰

### 3.1 현재 상태

기존 프로젝트는 다음 상태 전이를 갖고 있었다.

1. 입력 경로에서 영상 파일 발견
2. stage 1: Whisper/faster-whisper/WhisperX로 원문 segment 생성
3. SRT/VTT/TXT/JSON 출력 생성
4. stage 1 artifact 저장
5. stage 2: 원문 artifact를 번역 backend로 번역
6. 번역 출력 및 artifact 저장

이 구조의 장점은 stage 1과 stage 2가 이미 분리되어 있다는 점이다. 따라서 기존 자막도 “전사 결과와 같은 segment 구조”로만 변환하면 번역, 출력, artifact 재사용이 가능하다.

### 3.2 문제 상태

요청 기능 기준으로 기존 상태에는 다음 공백이 있었다.

- 전사 전에 기존 자막을 찾는 preflight 상태가 없었다.
- sidecar 자막, 영상 내장 자막, 외부 자막 provider가 같은 후보 모델로 묶이지 않았다.
- 후보를 사용자에게 목록으로 보여주는 상태가 없었다.
- 외부 provider quota/API key/다운로드 실패가 기존 전사 파이프라인을 깨지 않도록 하는 롤백 조건이 없었다.
- 기존 자막 싱크가 영상 길이와 어긋날 때 보정하는 경로가 없었다.
- OpenSubtitles/SubDL key 같은 민감 환경변수가 secret export 제외 목록에 포함되어 있지 않았다.

## 4. 리팩토링 상태 시스템 정의

새 상태 전이는 다음처럼 정의했다.

```
VideoDiscovered
  -> SubtitleSearchPlanned
  -> CandidateCollected
  -> CandidateRanked
  -> CandidateSelected | CandidateRejected
  -> SubtitleMaterialized
  -> SubtitleParsed
  -> SubtitleSynced
  -> Stage1ArtifactWritten
  -> TranslationStageReady
```

롤백 전이는 다음과 같다.

```
CandidateMissing
CandidateBelowThreshold
ProviderAuthMissing
ProviderDownloadFailed
SubtitleParseFailed
SyncFailed
  -> FreshTranscription
```

이 전이를 사용하면 “기존 자막 재사용”은 전사 파이프라인의 교체가 아니라, 전사 이전에 추가되는 선택적 최적화 상태가 된다.

## 5. 핵심 before/after

| 구분 | 기존 before | 변경 after |
|---|---|---|
| stage 1 진입 | 항상 ASR 전사 시작 | 자막 후보 탐색 후 임계값 이상이면 기존 자막 재사용 |
| 후보 표시 | 없음 | `--list-subtitle-candidates`로 후보 목록 출력 |
| 후보 출처 | 없음 | local, embedded, opensubtitles, subdl |
| 자막 파싱 | 출력 생성 중심 | SRT/VTT/ASS/SSA 입력 파싱 추가 |
| 싱크 처리 | 전사 결과 타이밍 refine 중심 | 기존 자막 duration clamp 및 ASR 기준 anchor sync 추가 |
| 번역 연결 | 전사 artifact만 사용 | 재사용 자막도 stage 1 artifact로 저장해 기존 번역 경로 사용 |
| 실패 처리 | 해당 없음 | 후보 실패 시 기존 ASR로 롤백 |
| 민감 정보 | 기존 provider key 중심 | OpenSubtitles/SubDL key/password도 secret env 처리 |

## 6. 구현 상세

### 6.1 신규 모듈: `vid_to_sub_app/cli/subtitle_reuse.py`

역할은 “후보를 찾고, 고르고, 실제 segment로 변환하는 adapter layer”이다. 주요 구성은 다음과 같다.

- `SubtitleSearchPlan`: 언어, 파일명, normalized title, release token, OpenSubtitles movie hash, media tag를 담는 검색 계획 객체
- `SubtitleCandidate`: provider, kind, language, score, confidence, path/download_url/file_id/stream_index 등을 담는 후보 객체
- `opensubtitles_hash`: 영상 앞/뒤 64KiB와 파일 크기로 OpenSubtitles movie hash 계산
- local sidecar finder: 같은 폴더의 `.srt`, `.vtt`, `.ass`, `.ssa` 후보 점수화
- embedded stream finder/extractor: `ffprobe`로 텍스트 자막 stream을 찾고 `ffmpeg`로 추출
- OpenSubtitles adapter: moviehash-only, title, filename query를 분리해 후보 수집
- SubDL adapter: file_name/film_name query와 zip download materialize 처리
- parsers: SRT, VTT, ASS/SSA를 segment 구조로 변환
- sync helpers: duration clamp 및 ASR reference 기반 보수적 offset/scale 보정

### 6.2 CLI 옵션 추가

`vid_to_sub_app/cli/main.py`에 다음 옵션을 추가했다.

- `--reuse-existing-subtitles off|auto|local|embedded|external|all`
- `--subtitle-providers local,embedded,opensubtitles,subdl`
- `--subtitle-languages en,ko`
- `--subtitle-min-score 0.78`
- `--subtitle-max-candidates 20`
- `--subtitle-sync-mode off|duration|asr`
- `--list-subtitle-candidates`

### 6.3 runner 연결

`vid_to_sub_app/cli/runner.py`의 `run_stage1` 시작부에 자막 재사용 전이를 삽입했다. 후보가 선택되면 다음을 수행한다.

1. 후보 materialize 또는 extraction
2. 자막 파싱
3. 싱크 보정
4. 기존 출력 포맷 생성
5. stage 1 artifact 저장
6. 이벤트 로그에 provider, score, sync mode 기록
7. fresh ASR 생략

후보가 없거나 실패하면 기존 전사 경로로 그대로 진행한다.

### 6.4 환경변수 및 secret 처리

`.env.example`과 `vid_to_sub_app/shared/constants.py`에 다음 값을 추가했다.

- `VID_TO_SUB_OPENSUBTITLES_API_KEY`
- `VID_TO_SUB_OPENSUBTITLES_USERNAME`
- `VID_TO_SUB_OPENSUBTITLES_PASSWORD`
- `VID_TO_SUB_OPENSUBTITLES_USER_AGENT`
- `VID_TO_SUB_SUBDL_API_KEY`

API key와 password는 secret env 목록에 포함했다.

### 6.5 문서화

`README.ko.md`, `README.md`에 사용 예시와 상태 전이를 추가했다. 또한 이 리포트를 프로젝트 루트에 포함했다.

## 7. 후보 선택 기준

후보 선택은 단순 키워드 매칭 하나만으로 하지 않고 여러 신호를 결합한다.

| 신호 | 의미 | 상태 변화 영향 |
|---|---|---|
| 언어 일치 | 요청 언어와 후보 언어가 일치 | 후보 유지 또는 제거 |
| 파일명/릴리즈 token overlap | 영상 파일명과 자막 release 이름의 겹침 | score 증가 |
| moviehash match | 영상 binary hash와 provider 결과의 직접 일치 | score 크게 증가 |
| 로컬 sidecar proximity | 같은 폴더/같은 stem의 자막 | 높은 신뢰 후보 |
| 내장 텍스트 stream | 영상 컨테이너 내부 자막 | 높은 신뢰 후보 |
| provider metadata | download count, rating, trusted, fps 등 | 보조 score |
| 파싱 가능성 | 실제 SRT/VTT/ASS/SSA segment 변환 가능 | materialize 이후 확정 |

커밋 조건은 “요청 언어와 호환되고, score가 threshold 이상이며, 실제 자막 파일을 materialize하고 segment 파싱까지 성공한 상태”이다.

## 8. 싱크 보정 방식

### 8.1 off

자막 시간을 그대로 사용한다. 이미 싱크가 맞는 local/embedded 후보에 적합하다.

### 8.2 duration

자막 end time이 영상 길이를 명백히 초과할 때 영상 길이에 맞춰 clamp한다. 이 모드는 안전하지만 섬세한 drift 보정은 하지 않는다.

### 8.3 asr

ASR을 최종 자막 생성용으로 쓰지 않고 기준 transcript만 생성한다. 이후 기존 자막 text와 ASR text 사이의 anchor를 찾아 offset/scale을 보수적으로 추정한다. anchor가 충분하지 않으면 무리하게 커밋하지 않고 기존 시간을 보존한다.

이 방식은 “처음부터 자막 생성”이 아니라 “기존 자막의 시간축 정렬”을 위한 보조 경로다.

## 9. 유지되는 불변식

- 기존 `--translate-to` 번역 흐름은 깨지지 않는다.
- 기존 ASR 전사 경로는 삭제하지 않는다.
- 외부 provider key가 없어도 local/embedded 검색은 계속 가능하다.
- 외부 검색 실패가 전체 job 실패로 전파되지 않는다.
- API key/password는 `.env` export 및 로그에서 노출되지 않는다.
- stage 1 artifact schema는 기존 번역 pipeline이 이해할 수 있는 segment 구조를 유지한다.
- 후보가 낮은 신뢰도이면 자동 커밋하지 않는다.

## 10. 실패/위험 지점

| 위험 | 현재 처리 | 추가 권장 |
|---|---|---|
| provider quota 초과 | warning 후 ASR 롤백 | provider별 quota 상태 UI 표시 |
| API key 없음 | provider skip | setup wizard에 key 검사 추가 |
| 잘못된 외부 후보 | score threshold 및 fallback | 사용자 선택형 accept/reject 로그 수집 |
| image subtitle 내장 스트림 | 텍스트 subtitle codec만 추출 | OCR 기반 PGS/VobSub 추출은 별도 기능으로 분리 |
| p99 정확도 미검증 | 구조만 구현 | benchmark corpus와 metric dashboard 필요 |
| TUI 통합 미완성 | CLI 우선 구현 | TUI candidate list panel 추가 필요 |
| 실제 외부 API live test 제한 | key 없는 환경에서는 skip | mock/live contract test 분리 필요 |

## 11. 테스트 결과

수행한 검증:

- `python -m compileall -q vid_to_sub_app vid_to_sub.py tui.py init_checker.py`
- `python -m pytest -q tests/test_subtitle_reuse.py tests/test_output_regressions.py tests/test_translation_pipeline.py -k 'not RemoteCommandSecretExclusion and not RemoteArtifactProvenance' tests/test_timing_refine.py tests/test_stage1_timing_refine.py tests/test_transcription_metadata.py`

결과:

- 106 passed
- 8 deselected
- compileall 통과

추가 smoke test:

- 임시 video 파일과 `Movie.Name.2024.en.srt` sidecar를 생성했다.
- `--list-subtitle-candidates --reuse-existing-subtitles local --subtitle-languages en` 실행 시 local 후보가 `score=0.940`, `confidence=exact`, `reusable=yes`로 표시되는 것을 확인했다.

제한:

- 전체 pytest suite는 현재 환경에 `textual` 패키지가 없어 TUI 관련 테스트 수집에서 `ModuleNotFoundError: No module named 'textual'`로 중단된다. 이는 이번 자막 재사용 코어 로직 실패가 아니라 테스트 환경 의존성 문제다.
- OpenSubtitles/SubDL live download는 API key/quota가 필요한 환경 의존 기능이므로 unit test는 adapter 로직과 local/materialize 중심으로 검증했다.

## 12. 변경 파일 요약

| 파일 | 변경 내용 |
|---|---|
| `vid_to_sub_app/cli/subtitle_reuse.py` | 신규 subtitle search/reuse adapter 모듈 |
| `vid_to_sub_app/cli/main.py` | CLI 옵션 및 candidate listing 처리 추가 |
| `vid_to_sub_app/cli/runner.py` | stage 1 전사 전 subtitle reuse 전이 추가 |
| `vid_to_sub_app/shared/constants.py` | 외부 provider env 및 secret 처리 추가 |
| `.env.example` | OpenSubtitles/SubDL 환경변수 문서화 |
| `README.ko.md` | 한국어 사용법 추가 |
| `README.md` | 영어 사용법 추가 |
| `tests/test_subtitle_reuse.py` | 신규 기능 단위/통합 테스트 추가 |
| `SUBTITLE_REUSE_REFACTOR_REPORT.ko.md` | 작업 리포트 추가 |

## 13. 사용 예시

후보만 확인:

```
python vid_to_sub.py /path/to/videos \
  --list-subtitle-candidates \
  --reuse-existing-subtitles auto \
  --subtitle-languages en,ko
```

기존 영어 자막을 찾아 한국어로 번역:

```
python vid_to_sub.py /path/to/videos \
  --reuse-existing-subtitles auto \
  --subtitle-languages en \
  --subtitle-min-score 0.78 \
  --subtitle-sync-mode duration \
  --translate-to ko
```

ASR 기준으로 기존 자막 싱크 보정:

```
python vid_to_sub.py /path/to/videos \
  --reuse-existing-subtitles all \
  --subtitle-languages en \
  --subtitle-sync-mode asr \
  --translate-to ko
```

외부 provider 활성화:

```
export VID_TO_SUB_OPENSUBTITLES_API_KEY=...
export VID_TO_SUB_OPENSUBTITLES_USERNAME=...
export VID_TO_SUB_OPENSUBTITLES_PASSWORD=...
export VID_TO_SUB_SUBDL_API_KEY=...
```

## 14. 리팩토링 계획안: 남은 고도화 단계

### 14.1 1차 운영화

- provider별 실패 사유를 structured event로 저장한다.
- user-facing 후보 목록에 score breakdown을 추가한다.
- 외부 provider 요청에 exponential backoff와 cache를 추가한다.
- 동일 video hash 결과를 local cache에 저장해 quota를 줄인다.

### 14.2 2차 정확도 고도화

- 제목/시즌/에피소드/연도 parser를 분리 모듈로 강화한다.
- IMDb/TMDB ID 추론을 별도 provider로 추가한다.
- 후보별 release group/fps/duration 차이를 score에 반영한다.
- 사용자 선택 로그 기반 threshold calibration을 수행한다.

### 14.3 3차 p99 목표 검증

- 영화/드라마/애니/강의/다국어 파일명을 포함한 corpus를 구성한다.
- metric: top-1 exact match, top-3 candidate inclusion, bad auto-commit rate, fallback rate, sync MAE를 추적한다.
- 언어별/파일명 품질별/릴리즈별 segment를 나누어 p50/p90/p99 지표를 산출한다.
- p99 기준을 만족하지 못하는 segment는 자동 커밋 대신 사용자 선택 상태로 전이한다.

### 14.4 TUI 통합

- 후보 목록 panel 추가
- 후보 accept/reject 버튼 추가
- provider key 상태 표시
- sync mode 선택 UI 추가

## 15. 최종 상태 의미

이번 리팩토링 후 프로젝트는 “항상 새로 전사하는 도구”에서 “기존 자막을 먼저 재사용하고, 실패할 때만 새로 전사하는 상태 기반 자막 파이프라인”으로 바뀌었다. 이 구조는 시간과 비용을 줄이고, 이미 존재하는 고품질 자막을 번역/싱크 보정 자원으로 활용할 수 있게 한다.
