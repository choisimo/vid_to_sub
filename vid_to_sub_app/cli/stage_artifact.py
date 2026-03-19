import json
from pathlib import Path
from typing import Any, TypedDict, cast

ARTIFACT_SCHEMA_VERSION = "1"
ARTIFACT_FILENAME_SUFFIX = ".stage1.json"

StageStatus = dict[str, bool | str | None]


class StageArtifact(TypedDict):
    schema_version: str
    source_path: str
    output_base: str
    source_fingerprint: str
    backend: str
    device: str
    model: str
    language: str | None
    target_lang: str | None
    formats: list[str]
    primary_outputs: list[str]
    segments: list[dict[str, Any]]
    stage_status: StageStatus


def artifact_path_for(source_path: Path, output_dir: Path | None) -> Path:
    artifact_dir = source_path.parent if output_dir is None else output_dir
    return artifact_dir / f"{source_path.stem}{ARTIFACT_FILENAME_SUFFIX}"


def write_stage_artifact(
    artifact: StageArtifact,
    output_dir: Path | None,
    source_path: Path,
) -> Path:
    artifact_path = artifact_path_for(source_path, output_dir)
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact_path


def load_stage_artifact(artifact_path: Path) -> StageArtifact:
    artifact = cast(dict[str, Any] | Any, json.loads(artifact_path.read_text(encoding="utf-8")))
    schema_version = artifact.get("schema_version") if isinstance(artifact, dict) else None
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact schema_version: {schema_version}")
    return cast(StageArtifact, artifact)
