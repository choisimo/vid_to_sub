from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .output import parse_srt, parse_srt_timestamp, probe_media_metadata

SUBTITLE_EXTENSIONS: frozenset[str] = frozenset({".srt", ".vtt", ".ass", ".ssa"})
TEXT_SUBTITLE_CODECS: frozenset[str] = frozenset(
    {
        "subrip",
        "srt",
        "ass",
        "ssa",
        "webvtt",
        "mov_text",
        "text",
    }
)

ENV_OPENSUBTITLES_API_KEY = "VID_TO_SUB_OPENSUBTITLES_API_KEY"
ENV_OPENSUBTITLES_USERNAME = "VID_TO_SUB_OPENSUBTITLES_USERNAME"
ENV_OPENSUBTITLES_PASSWORD = "VID_TO_SUB_OPENSUBTITLES_PASSWORD"
ENV_OPENSUBTITLES_USER_AGENT = "VID_TO_SUB_OPENSUBTITLES_USER_AGENT"
ENV_SUBDL_API_KEY = "VID_TO_SUB_SUBDL_API_KEY"

_OPENSUBTITLES_API_ROOT = "https://api.opensubtitles.com/api/v1"
_SUBDL_API_ROOT = "https://api.subdl.com/api/v1"
_DEFAULT_USER_AGENT = "vid-to-sub/1.0"


@dataclass(slots=True)
class SubtitleSearchPlan:
    languages: list[str]
    filename: str
    normalized_title: str
    title_candidates: list[str]
    release_tokens: list[str]
    movie_hash: str | None
    file_size: int | None
    media_tags: dict[str, str] = field(default_factory=dict)
    embedded_text_snippet: str | None = None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "languages": self.languages,
            "filename": self.filename,
            "normalized_title": self.normalized_title,
            "title_candidates": self.title_candidates,
            "release_tokens": self.release_tokens,
            "movie_hash": self.movie_hash,
            "file_size": self.file_size,
            "media_tags": self.media_tags,
            "embedded_text_snippet": self.embedded_text_snippet,
        }


@dataclass(slots=True)
class SubtitleCandidate:
    provider: str
    kind: str
    id: str
    language: str | None
    title: str
    release: str | None = None
    score: float = 0.0
    confidence: str = "low"
    reasons: list[str] = field(default_factory=list)
    path: str | None = None
    download_url: str | None = None
    file_id: str | None = None
    stream_index: int | None = None
    codec: str | None = None
    hearing_impaired: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        raw = payload.get("raw")
        if isinstance(raw, dict) and len(json.dumps(raw, ensure_ascii=False, default=str)) > 3000:
            payload["raw"] = {"_truncated": True}
        return payload


def parse_language_list(value: str | None, *, fallback: Sequence[str] = ()) -> list[str]:
    raw_items: list[str] = []
    if value:
        raw_items.extend(re.split(r"[,;\s]+", value))
    if not raw_items:
        raw_items.extend(fallback)
    languages: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        normalized = str(item or "").strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in {"*", "any"}:
            lowered = "all"
        if lowered not in seen:
            seen.add(lowered)
            languages.append(lowered)
    return languages or ["all"]


def _normalize_text(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[._+\-]+", " ", text)
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)
    text = re.sub(r"\b(720p|1080p|2160p|480p|x264|x265|h264|h265|hevc|aac|dts|bluray|web[- ]?dl|webrip|hdrip|dvdrip|brrip|proper|repack|extended|unrated|limited|multi|subs?)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(value: object) -> set[str]:
    return {token for token in re.split(r"[^0-9a-z]+", _normalize_text(value)) if token}


def _safe_id(*parts: object) -> str:
    raw = "|".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _language_matches(candidate_language: str | None, requested: Sequence[str]) -> bool:
    if not requested or "all" in requested:
        return True
    if not candidate_language:
        return False
    lowered = candidate_language.lower()
    requested_set = {lang.lower() for lang in requested}
    return lowered in requested_set or lowered[:2] in requested_set


def _title_score(plan: SubtitleSearchPlan, *texts: object) -> tuple[float, list[str]]:
    plan_tokens = set(plan.release_tokens) or _tokens(plan.normalized_title)
    candidate_tokens: set[str] = set()
    for text in texts:
        candidate_tokens.update(_tokens(text))
    if not plan_tokens or not candidate_tokens:
        return 0.0, []
    overlap = len(plan_tokens & candidate_tokens) / max(1, len(plan_tokens | candidate_tokens))
    reasons = ["title_token_overlap"] if overlap >= 0.35 else []
    return overlap, reasons


def _confidence(score: float) -> str:
    if score >= 0.92:
        return "exact"
    if score >= 0.82:
        return "high"
    if score >= 0.68:
        return "medium"
    return "low"


def opensubtitles_hash(path: Path) -> str | None:
    """Return the OpenSubtitles movie hash for a sufficiently large file.

    The algorithm sums 64-bit little-endian integers from the first and last
    64 KiB and adds the file size modulo 2^64.
    """
    chunk_size = 64 * 1024
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size < chunk_size * 2:
        return None

    acc = size
    try:
        with path.open("rb") as handle:
            first = handle.read(chunk_size)
            handle.seek(max(0, size - chunk_size))
            last = handle.read(chunk_size)
    except OSError:
        return None

    for chunk in (first, last):
        for offset in range(0, len(chunk), 8):
            block = chunk[offset : offset + 8]
            if len(block) != 8:
                continue
            acc = (acc + int.from_bytes(block, "little", signed=False)) & 0xFFFFFFFFFFFFFFFF
    return f"{acc:016x}"


def _collect_media_tags(video: Path) -> dict[str, str]:
    payload = probe_media_metadata(video)
    tags: dict[str, str] = {}
    if not isinstance(payload, dict):
        return tags

    def add(prefix: str, mapping: Mapping[str, Any]) -> None:
        for key, value in mapping.items():
            text = str(value or "").strip()
            if text:
                tags[f"{prefix}.{str(key).lower()}"] = text

    format_payload = payload.get("format")
    if isinstance(format_payload, dict) and isinstance(format_payload.get("tags"), dict):
        add("format", format_payload["tags"])
    for index, stream in enumerate(payload.get("streams") or []):
        if isinstance(stream, dict) and isinstance(stream.get("tags"), dict):
            add(f"stream{index}", stream["tags"])
    return tags


def _guess_titles(video: Path, tags: Mapping[str, str]) -> list[str]:
    candidates: list[str] = []
    for tag_key in ("format.title", "format.movie", "format.show", "stream0.title"):
        value = str(tags.get(tag_key) or "").strip()
        if value:
            candidates.append(value)
    stem = video.stem
    candidates.append(stem)
    normalized_stem = _normalize_text(stem)
    if normalized_stem:
        # Drop common release suffixes after year/season patterns while keeping the
        # raw stem as a fallback query.
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", normalized_stem)
        if year_match:
            candidates.append(normalized_stem[: year_match.end()].strip())
        episode_match = re.search(r"\bs\d{1,2}\s*e\d{1,2}\b", normalized_stem)
        if episode_match:
            candidates.append(normalized_stem[: episode_match.end()].strip())
        candidates.append(normalized_stem)

    result: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        normalized = re.sub(r"\s+", " ", str(item or "").strip())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def build_search_plan(
    video: Path,
    *,
    languages: Sequence[str],
    embedded_text_snippet: str | None = None,
) -> SubtitleSearchPlan:
    tags = _collect_media_tags(video)
    title_candidates = _guess_titles(video, tags)
    normalized_title = _normalize_text(title_candidates[0] if title_candidates else video.stem)
    release_tokens = sorted(_tokens(video.stem))
    try:
        file_size = int(video.stat().st_size)
    except OSError:
        file_size = None
    return SubtitleSearchPlan(
        languages=list(languages),
        filename=video.name,
        normalized_title=normalized_title,
        title_candidates=title_candidates,
        release_tokens=release_tokens,
        movie_hash=opensubtitles_hash(video),
        file_size=file_size,
        media_tags=dict(tags),
        embedded_text_snippet=embedded_text_snippet,
    )


def parse_vtt(text: str) -> list[dict[str, Any]]:
    text = re.sub(r"^WEBVTT.*?(\n\s*\n|$)", "", text.strip(), flags=re.IGNORECASE | re.DOTALL)
    blocks = re.split(r"\n\s*\n", text, flags=re.MULTILINE)
    segments: list[dict[str, Any]] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timestamp_index = 0
        if "-->" not in lines[0] and len(lines) > 1:
            timestamp_index = 1
        if timestamp_index >= len(lines) or "-->" not in lines[timestamp_index]:
            continue
        start_raw, end_raw = [part.strip().split()[0] for part in lines[timestamp_index].split("-->", 1)]
        payload_lines = [line for line in lines[timestamp_index + 1 :] if not line.startswith("NOTE")]
        if not payload_lines:
            continue
        try:
            segments.append(
                {
                    "start": parse_srt_timestamp(start_raw.replace(".", ",")),
                    "end": parse_srt_timestamp(end_raw.replace(".", ",")),
                    "text": "\n".join(payload_lines).strip(),
                }
            )
        except ValueError:
            continue
    return segments


_ASS_TIME_RE = re.compile(r"(?P<h>\d+):(?P<m>\d{2}):(?P<s>\d{2})[.](?P<cs>\d{2})")


def _parse_ass_time(value: str) -> float:
    match = _ASS_TIME_RE.fullmatch(value.strip())
    if not match:
        raise ValueError(f"Invalid ASS timestamp: {value}")
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(match.group("cs")) / 100.0
    )


def parse_ass(text: str) -> list[dict[str, Any]]:
    fields: list[str] | None = None
    segments: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip("\ufeff")
        if line.lower().startswith("format:"):
            fields = [part.strip().lower() for part in line.split(":", 1)[1].split(",")]
            continue
        if not line.lower().startswith("dialogue:"):
            continue
        payload = line.split(":", 1)[1].lstrip()
        if fields:
            maxsplit = max(0, len(fields) - 1)
            values = [part.strip() for part in payload.split(",", maxsplit)]
            lookup = {field: values[idx] for idx, field in enumerate(fields) if idx < len(values)}
            start_raw = lookup.get("start")
            end_raw = lookup.get("end")
            text_raw = lookup.get("text")
        else:
            values = [part.strip() for part in payload.split(",", 9)]
            start_raw = values[1] if len(values) > 1 else None
            end_raw = values[2] if len(values) > 2 else None
            text_raw = values[9] if len(values) > 9 else None
        if not start_raw or not end_raw or text_raw is None:
            continue
        cleaned = re.sub(r"\{[^}]*\}", "", text_raw).replace("\\N", "\n").strip()
        if not cleaned:
            continue
        try:
            segments.append({"start": _parse_ass_time(start_raw), "end": _parse_ass_time(end_raw), "text": cleaned})
        except ValueError:
            continue
    return segments


def parse_subtitle_text(text: str, suffix: str | None = None) -> list[dict[str, Any]]:
    ext = (suffix or "").lower()
    if ext == ".vtt" or text.lstrip().upper().startswith("WEBVTT"):
        return parse_vtt(text)
    if ext in {".ass", ".ssa"} or "[Script Info]" in text[:500]:
        return parse_ass(text)
    return parse_srt(text)


def _read_text_guessing(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "cp949", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_subtitle_file(path: Path) -> list[dict[str, Any]]:
    return parse_subtitle_text(_read_text_guessing(path), path.suffix)


def find_sidecar_subtitles(video: Path, languages: Sequence[str]) -> list[SubtitleCandidate]:
    candidates: list[SubtitleCandidate] = []
    if not video.parent.exists():
        return candidates
    video_stem_norm = _normalize_text(video.stem)
    video_tokens = _tokens(video.stem)
    for path in sorted(video.parent.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUBTITLE_EXTENSIONS:
            continue
        if path.name.startswith(video.name):
            pass
        subtitle_stem_norm = _normalize_text(path.stem)
        stem_tokens = _tokens(path.stem)
        related = path.stem.startswith(video.stem) or video.stem.startswith(path.stem)
        token_overlap = len(video_tokens & stem_tokens) / max(1, len(video_tokens | stem_tokens))
        if not related and token_overlap < 0.25 and video_stem_norm not in subtitle_stem_norm:
            continue
        language = _infer_language_from_name(path, languages)
        reasons = ["sidecar_file"]
        score = 0.72 + min(0.2, token_overlap * 0.2)
        if path.stem == video.stem:
            score = 0.94
            reasons.append("same_stem")
        elif path.stem.startswith(video.stem + "."):
            score = max(score, 0.90)
            reasons.append("video_stem_prefix")
        if _language_matches(language, languages):
            score += 0.04
            reasons.append("language_match")
        candidates.append(
            SubtitleCandidate(
                provider="local",
                kind="sidecar",
                id=_safe_id("local", str(path)),
                language=language,
                title=path.name,
                release=path.stem,
                score=min(0.99, score),
                confidence=_confidence(min(0.99, score)),
                reasons=reasons,
                path=str(path),
            )
        )
    return candidates


def _infer_language_from_name(path: Path, requested: Sequence[str]) -> str | None:
    parts = [part.lower() for part in re.split(r"[._\-\s]+", path.stem) if part]
    known = {"ko", "kor", "en", "eng", "ja", "jpn", "jp", "fr", "fre", "fra", "es", "spa", "de", "ger", "deu", "zh", "chi", "zho", "pt", "por", "it", "ita", "ru", "rus", "ar", "ara"}
    aliases = {"eng": "en", "kor": "ko", "jpn": "ja", "jp": "ja", "fre": "fr", "fra": "fr", "spa": "es", "ger": "de", "deu": "de", "chi": "zh", "zho": "zh", "por": "pt", "ita": "it", "rus": "ru", "ara": "ar"}
    requested_set = {lang.lower() for lang in requested}
    for part in reversed(parts):
        if part in aliases:
            return aliases[part]
        if part in known:
            return part
        if part in requested_set:
            return part
    return None


def probe_embedded_subtitle_streams(video: Path) -> list[SubtitleCandidate]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "s",
        "-show_entries",
        "stream=index,codec_name:stream_tags=language,title",
        "-of",
        "json",
        str(video),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return []
    candidates: list[SubtitleCandidate] = []
    for stream in payload.get("streams") or []:
        if not isinstance(stream, dict):
            continue
        index = stream.get("index")
        try:
            stream_index = int(index)
        except (TypeError, ValueError):
            continue
        codec = str(stream.get("codec_name") or "").lower() or None
        tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
        language = str(tags.get("language") or "").strip().lower() or None
        title = str(tags.get("title") or "").strip() or f"embedded stream #{stream_index}"
        text_codec = codec in TEXT_SUBTITLE_CODECS
        score = 0.91 if text_codec else 0.60
        reasons = ["embedded_stream", "text_subtitle_codec" if text_codec else "image_subtitle_codec"]
        candidates.append(
            SubtitleCandidate(
                provider="embedded",
                kind="embedded_stream",
                id=_safe_id("embedded", str(video), stream_index, codec),
                language=language,
                title=title,
                release=video.name,
                score=score,
                confidence=_confidence(score),
                reasons=reasons,
                stream_index=stream_index,
                codec=codec,
                raw={"stream": stream},
            )
        )
    return candidates


def extract_embedded_subtitle(video: Path, candidate: SubtitleCandidate) -> Path:
    if candidate.stream_index is None:
        raise ValueError("Embedded subtitle candidate is missing stream_index.")
    if (candidate.codec or "").lower() not in TEXT_SUBTITLE_CODECS:
        raise ValueError(
            f"Embedded subtitle stream #{candidate.stream_index} uses unsupported codec {candidate.codec!r}; OCR/image subtitle extraction is not implemented."
        )
    out_dir = Path(tempfile.mkdtemp(prefix="vid-to-sub-embedded-"))
    out_path = out_dir / f"{video.stem}.embedded.{candidate.stream_index}.srt"
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-map",
        f"0:{candidate.stream_index}",
        str(out_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to extract embedded subtitles.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg subtitle extraction failed: {detail}") from exc
    return out_path


def _http_json(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    body: Mapping[str, Any] | None = None,
    timeout: float = 20.0,
) -> dict[str, Any]:
    payload: bytes | None = None
    merged_headers = {"Accept": "application/json", **(dict(headers or {}))}
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        merged_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=payload, headers=merged_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to call {url}: {exc.reason}") from exc
    try:
        payload_obj = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"JSON response expected from {url}: {exc}") from exc
    if not isinstance(payload_obj, dict):
        raise RuntimeError(f"JSON object expected from {url}.")
    return payload_obj


def _download_bytes(url: str, *, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _DEFAULT_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while downloading subtitle: {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Unable to download subtitle from {url}: {exc.reason}") from exc


def _opensubtitles_headers(*, token: str | None = None) -> dict[str, str]:
    api_key = os.environ.get(ENV_OPENSUBTITLES_API_KEY, "").strip()
    user_agent = os.environ.get(ENV_OPENSUBTITLES_USER_AGENT, "").strip() or _DEFAULT_USER_AGENT
    headers = {"User-Agent": user_agent, "Accept": "application/json"}
    if api_key:
        headers["Api-Key"] = api_key
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _opensubtitles_login() -> str | None:
    username = os.environ.get(ENV_OPENSUBTITLES_USERNAME, "").strip()
    password = os.environ.get(ENV_OPENSUBTITLES_PASSWORD, "").strip()
    if not username or not password:
        return None
    payload = _http_json(
        f"{_OPENSUBTITLES_API_ROOT}/login",
        method="POST",
        headers=_opensubtitles_headers(),
        body={"username": username, "password": password},
    )
    token = payload.get("token")
    return str(token) if token else None


def search_opensubtitles(plan: SubtitleSearchPlan, *, limit: int = 12) -> list[SubtitleCandidate]:
    api_key = os.environ.get(ENV_OPENSUBTITLES_API_KEY, "").strip()
    if not api_key:
        return []

    queries: list[dict[str, str]] = []
    languages = ",".join(lang for lang in plan.languages if lang != "all")
    if plan.movie_hash:
        params = {"moviehash": plan.movie_hash}
        if languages:
            params["languages"] = languages
        queries.append(params)
    for title in plan.title_candidates[:4]:
        params = {"query": title}
        if languages:
            params["languages"] = languages
        queries.append(params)
    if plan.filename:
        params = {"query": plan.filename}
        if languages:
            params["languages"] = languages
        queries.append(params)

    candidates: list[SubtitleCandidate] = []
    seen: set[str] = set()
    for params in queries:
        query_string = urllib.parse.urlencode(params)
        try:
            payload = _http_json(
                f"{_OPENSUBTITLES_API_ROOT}/subtitles?{query_string}",
                headers=_opensubtitles_headers(),
            )
        except RuntimeError:
            continue
        for item in payload.get("data") or []:
            if not isinstance(item, dict):
                continue
            attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else {}
            files = attributes.get("files") if isinstance(attributes.get("files"), list) else []
            file_id = None
            file_name = None
            if files:
                first_file = files[0]
                if isinstance(first_file, dict):
                    file_id = first_file.get("file_id")
                    file_name = first_file.get("file_name")
            unique_id = str(file_id or item.get("id") or _safe_id(item))
            if unique_id in seen:
                continue
            seen.add(unique_id)
            feature_details = attributes.get("feature_details") if isinstance(attributes.get("feature_details"), dict) else {}
            language = str(attributes.get("language") or "").strip().lower() or None
            release = str(attributes.get("release") or file_name or "").strip() or None
            title = str(feature_details.get("title") or attributes.get("feature_details", {}).get("title") or release or item.get("id") or "OpenSubtitles result")
            score, reasons = _score_external_candidate(plan, language, release, title, attributes)
            if "moviehash" in params:
                score = max(score, 0.88)
                reasons.append("moviehash_query")
            candidates.append(
                SubtitleCandidate(
                    provider="opensubtitles",
                    kind="external",
                    id=unique_id,
                    language=language,
                    title=title,
                    release=release,
                    score=score,
                    confidence=_confidence(score),
                    reasons=reasons,
                    file_id=str(file_id) if file_id is not None else None,
                    hearing_impaired=_boolish(attributes.get("hearing_impaired")),
                    raw={"id": item.get("id"), "attributes": attributes},
                )
            )
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]


def _boolish(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    return None


def _score_external_candidate(
    plan: SubtitleSearchPlan,
    language: str | None,
    release: str | None,
    title: str | None,
    attributes: Mapping[str, Any],
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    title_overlap, overlap_reasons = _title_score(plan, release, title)
    reasons.extend(overlap_reasons)
    score = 0.45 + min(0.25, title_overlap * 0.25)
    if _language_matches(language, plan.languages):
        score += 0.12
        reasons.append("language_match")
    if _boolish(attributes.get("trusted")) or _boolish(attributes.get("from_trusted")):
        score += 0.06
        reasons.append("trusted_uploader")
    try:
        downloads = float(attributes.get("download_count") or attributes.get("downloads") or 0)
    except (TypeError, ValueError):
        downloads = 0.0
    if downloads > 0:
        score += min(0.08, math.log10(downloads + 1) / 100 * 2)
        reasons.append("download_popularity")
    if release and plan.filename and _normalize_text(Path(plan.filename).stem) in _normalize_text(release):
        score += 0.12
        reasons.append("release_name_contains_filename")
    return min(0.98, max(0.0, score)), reasons


def download_opensubtitles_candidate(candidate: SubtitleCandidate) -> Path:
    if not candidate.file_id:
        raise ValueError("OpenSubtitles candidate is missing file_id.")
    token = _opensubtitles_login()
    payload = _http_json(
        f"{_OPENSUBTITLES_API_ROOT}/download",
        method="POST",
        headers=_opensubtitles_headers(token=token),
        body={"file_id": int(candidate.file_id), "sub_format": "srt"},
    )
    link = str(payload.get("link") or "").strip()
    if not link:
        raise RuntimeError("OpenSubtitles download response did not include a link.")
    file_name = str(payload.get("file_name") or candidate.release or f"{candidate.id}.srt")
    return _materialize_download_bytes(_download_bytes(link), file_name)


def search_subdl(plan: SubtitleSearchPlan, *, limit: int = 12) -> list[SubtitleCandidate]:
    api_key = os.environ.get(ENV_SUBDL_API_KEY, "").strip()
    if not api_key:
        return []
    languages = ",".join(lang.upper() for lang in plan.languages if lang != "all")
    queries: list[dict[str, str]] = []
    base = {"api_key": api_key, "subs_per_page": str(min(30, max(1, limit)))}
    if languages:
        base["languages"] = languages
    queries.append({**base, "file_name": plan.filename})
    for title in plan.title_candidates[:3]:
        queries.append({**base, "film_name": title})

    candidates: list[SubtitleCandidate] = []
    seen: set[str] = set()
    for params in queries:
        try:
            payload = _http_json(f"{_SUBDL_API_ROOT}/subtitles?{urllib.parse.urlencode(params)}")
        except RuntimeError:
            continue
        if payload.get("status") is False:
            continue
        for item in payload.get("subtitles") or []:
            if not isinstance(item, dict):
                continue
            url_path = str(item.get("url") or item.get("download_url") or "").strip()
            if url_path.startswith("/"):
                download_url = f"https://dl.subdl.com{url_path}"
            elif url_path.startswith("http"):
                download_url = url_path
            else:
                subtitle_id = item.get("subtitle_id") or item.get("id") or item.get("sd_id")
                download_url = f"https://dl.subdl.com/subtitle/{subtitle_id}.zip" if subtitle_id else None
            unique_id = str(item.get("subtitle_id") or item.get("id") or download_url or _safe_id(item))
            if unique_id in seen:
                continue
            seen.add(unique_id)
            language = str(item.get("lang") or item.get("language") or "").strip().lower() or None
            release = str(item.get("release_name") or item.get("name") or item.get("file_name") or "").strip() or None
            title = str(item.get("movie_name") or item.get("film_name") or release or "SubDL result")
            score, reasons = _score_external_candidate(plan, language, release, title, item)
            candidates.append(
                SubtitleCandidate(
                    provider="subdl",
                    kind="external",
                    id=unique_id,
                    language=language,
                    title=title,
                    release=release,
                    score=score,
                    confidence=_confidence(score),
                    reasons=reasons,
                    download_url=download_url,
                    hearing_impaired=_boolish(item.get("hi") or item.get("hearing_impaired")),
                    raw=item,
                )
            )
    return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]


def download_subdl_candidate(candidate: SubtitleCandidate) -> Path:
    if not candidate.download_url:
        raise ValueError("SubDL candidate is missing download_url.")
    file_name = candidate.release or f"{candidate.id}.zip"
    return _materialize_download_bytes(_download_bytes(candidate.download_url), file_name)


def _materialize_download_bytes(data: bytes, file_name: str) -> Path:
    out_dir = Path(tempfile.mkdtemp(prefix="vid-to-sub-download-"))
    suffix = Path(file_name).suffix.lower()
    if suffix == ".gz" or data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
        suffix = ".srt"
        file_name = Path(file_name).with_suffix(".srt").name
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = [name for name in archive.namelist() if Path(name).suffix.lower() in SUBTITLE_EXTENSIONS]
            if not names:
                raise RuntimeError("Downloaded subtitle ZIP did not contain a supported subtitle file.")
            selected = sorted(names, key=lambda name: (Path(name).suffix.lower() != ".srt", name))[0]
            target = out_dir / Path(selected).name
            target.write_bytes(archive.read(selected))
            return target
    if suffix not in SUBTITLE_EXTENSIONS:
        suffix = ".srt"
    target = out_dir / (Path(file_name).stem + suffix)
    target.write_bytes(data)
    return target


def search_subtitle_candidates(
    video: Path,
    *,
    languages: Sequence[str],
    providers: Sequence[str],
    limit: int = 12,
) -> tuple[SubtitleSearchPlan, list[SubtitleCandidate]]:
    provider_set = {provider.strip().lower() for provider in providers if provider.strip()}
    if "all" in provider_set:
        provider_set = {"local", "embedded", "opensubtitles", "subdl"}
    if "external" in provider_set:
        provider_set.update({"opensubtitles", "subdl"})
    if "auto" in provider_set:
        provider_set.update({"local", "embedded", "opensubtitles", "subdl"})
    plan = build_search_plan(video, languages=languages)
    candidates: list[SubtitleCandidate] = []
    if "local" in provider_set:
        candidates.extend(find_sidecar_subtitles(video, languages))
    if "embedded" in provider_set:
        candidates.extend(probe_embedded_subtitle_streams(video))
    if "opensubtitles" in provider_set:
        candidates.extend(search_opensubtitles(plan, limit=limit))
    if "subdl" in provider_set:
        candidates.extend(search_subdl(plan, limit=limit))
    candidates = _dedupe_candidates(candidates)
    candidates.sort(key=lambda item: item.score, reverse=True)
    return plan, candidates[:limit]


def _dedupe_candidates(candidates: Sequence[SubtitleCandidate]) -> list[SubtitleCandidate]:
    seen: set[tuple[str, str]] = set()
    result: list[SubtitleCandidate] = []
    for candidate in candidates:
        key = (candidate.provider, candidate.id)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def materialize_candidate(video: Path, candidate: SubtitleCandidate) -> Path:
    if candidate.provider == "local" and candidate.path:
        return Path(candidate.path)
    if candidate.provider == "embedded":
        return extract_embedded_subtitle(video, candidate)
    if candidate.provider == "opensubtitles":
        return download_opensubtitles_candidate(candidate)
    if candidate.provider == "subdl":
        return download_subdl_candidate(candidate)
    if candidate.download_url:
        return _materialize_download_bytes(_download_bytes(candidate.download_url), candidate.release or f"{candidate.id}.srt")
    raise ValueError(f"Candidate cannot be materialized: {candidate.provider}/{candidate.id}")


def load_candidate_segments(video: Path, candidate: SubtitleCandidate) -> tuple[list[dict[str, Any]], Path]:
    path = materialize_candidate(video, candidate)
    segments = parse_subtitle_file(path)
    if not segments:
        raise RuntimeError(f"Candidate subtitle did not contain parseable segments: {path}")
    return segments, path


def normalize_subtitle_bounds(
    segments: list[dict[str, Any]],
    *,
    duration: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not segments:
        return segments, {"mode": "duration", "changed": False, "reason": "empty"}
    cleaned: list[dict[str, Any]] = []
    changed = False
    for segment in segments:
        start = max(0.0, float(segment.get("start") or 0.0))
        end = max(start + 0.01, float(segment.get("end") or start + 0.01))
        if duration is not None and duration > 0:
            if start > duration:
                changed = True
                continue
            end = min(duration, end)
        if start != segment.get("start") or end != segment.get("end"):
            changed = True
        item = dict(segment)
        item["start"] = round(start, 3)
        item["end"] = round(end, 3)
        cleaned.append(item)
    return cleaned, {"mode": "duration", "changed": changed, "segments_in": len(segments), "segments_out": len(cleaned)}


def sync_segments_to_reference(
    candidate_segments: list[dict[str, Any]],
    reference_segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Estimate a linear time transform using text-overlap matched segments.

    This is intentionally conservative: if fewer than two high-confidence text
    anchors are found, the function returns the original timings and marks the
    sync as not committed.
    """
    anchors: list[tuple[float, float, float]] = []
    for candidate in candidate_segments:
        candidate_tokens = _tokens(candidate.get("text"))
        if not candidate_tokens:
            continue
        best_score = 0.0
        best_ref: dict[str, Any] | None = None
        for reference in reference_segments:
            ref_tokens = _tokens(reference.get("text"))
            if not ref_tokens:
                continue
            score = len(candidate_tokens & ref_tokens) / max(1, len(candidate_tokens | ref_tokens))
            if score > best_score:
                best_score = score
                best_ref = reference
        if best_ref is not None and best_score >= 0.55:
            candidate_mid = (float(candidate["start"]) + float(candidate["end"])) / 2.0
            ref_mid = (float(best_ref["start"]) + float(best_ref["end"])) / 2.0
            anchors.append((candidate_mid, ref_mid, best_score))
    if len(anchors) < 2:
        return candidate_segments, {
            "mode": "asr",
            "committed": False,
            "reason": "insufficient_text_anchors",
            "anchors": len(anchors),
        }
    anchors.sort(key=lambda item: item[2], reverse=True)
    selected = sorted(anchors[: min(25, len(anchors))], key=lambda item: item[0])
    xs = [item[0] for item in selected]
    ys = [item[1] for item in selected]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= 0:
        scale = 1.0
    else:
        scale = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    if scale < 0.90 or scale > 1.10:
        scale = 1.0
    offset = y_mean - scale * x_mean
    if abs(offset) > 300:
        return candidate_segments, {
            "mode": "asr",
            "committed": False,
            "reason": "estimated_offset_too_large",
            "offset": round(offset, 3),
            "scale": round(scale, 6),
            "anchors": len(selected),
        }
    adjusted: list[dict[str, Any]] = []
    for segment in candidate_segments:
        start = max(0.0, scale * float(segment["start"]) + offset)
        end = max(start + 0.01, scale * float(segment["end"]) + offset)
        item = dict(segment)
        item["start"] = round(start, 3)
        item["end"] = round(end, 3)
        adjusted.append(item)
    return adjusted, {
        "mode": "asr",
        "committed": True,
        "offset": round(offset, 3),
        "scale": round(scale, 6),
        "anchors": len(selected),
        "median_anchor_score": round(sorted(item[2] for item in selected)[len(selected) // 2], 4),
    }


def cleanup_materialized_candidate(path: Path, candidate: SubtitleCandidate) -> None:
    if candidate.provider == "local":
        return
    try:
        root = path.parent
        if root.name.startswith("vid-to-sub-"):
            shutil.rmtree(root, ignore_errors=True)
    except Exception:
        pass
