"""Helpers for history-driven subtitle filtering and bulk copy.

These functions are intentionally decoupled from the TUI so they can be
unit-tested without a textual runtime.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TypedDict

# Subtitle extensions considered "subtitle" outputs vs. raw transcription JSON
_SUBTITLE_EXTENSIONS: frozenset[str] = frozenset({".srt", ".vtt", ".txt", ".tsv"})


class CopyResult(TypedDict):
    source: str
    destination: str | None
    success: bool
    error: str | None


def subtitle_paths_from_output_paths(output_paths_json: str | None) -> list[Path]:
    """Return subtitle-format paths from a jobs.output_paths JSON string.

    Filters to extensions in _SUBTITLE_EXTENSIONS. Returns an empty list for
    null, empty, or malformed input without raising.
    """
    if not output_paths_json:
        return []
    try:
        raw = json.loads(output_paths_json)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(raw, list):
        return []
    return [
        Path(str(p)) for p in raw if Path(str(p)).suffix.lower() in _SUBTITLE_EXTENSIONS
    ]


def bulk_copy_subtitles(
    jobs: list[dict[str, object]],
    destination_dir: Path,
) -> list[CopyResult]:
    """Copy subtitle files from multiple job records to destination_dir.

    Each entry in jobs must be a dict with an ``output_paths`` key (JSON string).
    Files that do not exist are reported as partial failures without aborting
    the remaining copies.  Duplicate filenames are renamed ``name (1).ext``,
    ``name (2).ext``, etc.

    Returns one CopyResult per attempted file.
    """
    results: list[CopyResult] = []
    used_names: dict[str, int] = {}

    for job in jobs:
        raw_paths = job.get("output_paths")
        paths = subtitle_paths_from_output_paths(
            str(raw_paths) if isinstance(raw_paths, str) else None
        )
        for src in paths:
            stem = src.stem
            suffix = src.suffix
```python
            count = used_names.get(src.name, 0)
            dest_name = src.name if count == 0 else f"{stem} ({count}){suffix}"
            # Avoid overwriting existing files in the destination folder
            while (destination_dir / dest_name).exists():
                count += 1
                dest_name = f"{stem} ({count}){suffix}"
            used_names[src.name] = count + 1
            dest = destination_dir / dest_name
            if count == 0:
                dest_name = src.name
            else:
                dest_name = f"{stem} ({count}){suffix}"
            used_names[src.name] = count + 1
            dest = destination_dir / dest_name

            if not src.exists():
                results.append(
                    CopyResult(
                        source=str(src),
                        destination=None,
                        success=False,
                        error="source file not found",
                    )
                )
                continue
            try:
                shutil.copy2(src, dest)
                results.append(
                    CopyResult(
                        source=str(src),
                        destination=str(dest),
                        success=True,
                        error=None,
                    )
                )
            except Exception as exc:
                results.append(
                    CopyResult(
                        source=str(src),
                        destination=str(dest),
                        success=False,
                        error=str(exc),
                    )
                )

    return results
