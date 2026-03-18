from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Sequence

from vid_to_sub_app.shared.constants import VIDEO_EXTENSIONS


def discover_videos(roots: Sequence[str | Path], recursive: bool = True) -> list[Path]:
    found: set[Path] = set()
    for raw in roots:
        path = Path(raw).resolve()
        if not path.exists():
            print(f"[WARN] Path does not exist, skipping: {path}", file=sys.stderr)
            continue
        if path.is_file():
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                found.add(path)
            else:
                print(
                    f"[WARN] File extension '{path.suffix}' not in known video list, "
                    f"skipping: {path}",
                    file=sys.stderr,
                )
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for child in path.glob(pattern):
                if child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS:
                    found.add(child)
        else:
            print(f"[WARN] Not a file or directory, skipping: {path}", file=sys.stderr)

    return sorted(found)


def hash_video_folder(folder: str | Path) -> str:
    resolved = str(Path(folder).expanduser().resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]
