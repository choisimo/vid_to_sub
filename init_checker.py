#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import venv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
VENV_DIR = ROOT_DIR / ".venv"
REEXEC_ENV = "VID_TO_SUB_BOOTSTRAP_REEXEC"
GROUP_ENV = "VID_TO_SUB_BOOTSTRAP_GROUPS"


@dataclass(frozen=True)
class RequirementGroup:
    name: str
    filename: str
    modules: tuple[str, ...]
    optional: bool = False

    @property
    def path(self) -> Path:
        return ROOT_DIR / self.filename


REQUIREMENT_GROUPS: dict[str, RequirementGroup] = {
    "base": RequirementGroup("base", "requirements.txt", ("textual", "dotenv"), optional=False),
    "faster-whisper": RequirementGroup(
        "faster-whisper",
        "requirements-faster-whisper.txt",
        ("faster_whisper",),
        optional=True,
    ),
    "whisper": RequirementGroup(
        "whisper",
        "requirements-whisper.txt",
        ("whisper", "torch", "torchaudio"),
        optional=True,
    ),
    "whisperx": RequirementGroup(
        "whisperx",
        "requirements-whisperx.txt",
        ("whisperx", "torch", "torchaudio"),
        optional=True,
    ),
}


def venv_python_path() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def is_same_interpreter(path_a: Path, path_b: Path) -> bool:
    try:
        return path_a.resolve() == path_b.resolve()
    except FileNotFoundError:
        return False


def ensure_venv() -> Path:
    python_path = venv_python_path()
    if python_path.exists():
        return python_path
    print(f"[bootstrap] creating virtualenv at {VENV_DIR}", file=sys.stderr)
    builder = venv.EnvBuilder(with_pip=True, upgrade_deps=False)
    builder.create(VENV_DIR)
    return python_path


def ensure_pip(python_executable: str) -> None:
    _ = subprocess.run(
        [python_executable, "-m", "ensurepip", "--upgrade"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def missing_modules(modules: Iterable[str]) -> list[str]:
    return [module for module in modules if importlib.util.find_spec(module) is None]


def install_requirements(python_executable: str, group: RequirementGroup) -> bool:
    if not group.path.exists():
        print(
            f"[bootstrap] requirement file missing, skipping: {group.path}",
            file=sys.stderr,
        )
        return False
    print(f"[bootstrap] installing {group.filename}", file=sys.stderr)
    result = subprocess.run(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(group.path),
        ],
        cwd=ROOT_DIR,
    )
    return result.returncode == 0


def resolve_groups(requirement_groups: Sequence[str] | None) -> list[RequirementGroup]:
    names = list(requirement_groups or ("base",))
    groups: list[RequirementGroup] = []
    for name in names:
        group = REQUIREMENT_GROUPS.get(name)
        if group is None:
            raise ValueError(f"Unknown requirement group: {name}")
        groups.append(group)
    return groups


def relaunch(python_executable: Path, argv: Sequence[str], groups: Sequence[str]) -> None:
    env = os.environ.copy()
    env[GROUP_ENV] = ",".join(groups)
    env[REEXEC_ENV] = "1"
    os.execve(str(python_executable), [str(python_executable), *argv], env)


def bootstrap_runtime(
    requirement_groups: Sequence[str] | None = None,
) -> None:
    groups = resolve_groups(
        requirement_groups
        or tuple(filter(None, os.environ.get(GROUP_ENV, "").split(",")))
        or ("base",)
    )
    target_python = ensure_venv()
    current_python = Path(sys.executable)

    if not is_same_interpreter(current_python, target_python):
        relaunch(target_python, sys.argv, [group.name for group in groups])

    ensure_pip(sys.executable)
    installed_any = False
    failed_optional: list[str] = []
    for group in groups:
        if missing_modules(group.modules):
            success = install_requirements(sys.executable, group)
            if success:
                installed_any = True
            elif group.optional:
                failed_optional.append(group.name)
            else:
                raise RuntimeError(
                    f"Failed to install required dependency group '{group.name}'"
                )

    if failed_optional:
        print(
            f"[bootstrap] optional groups skipped: {', '.join(failed_optional)}",
            file=sys.stderr,
        )

    if installed_any and os.environ.get(REEXEC_ENV) != "1":
        relaunch(Path(sys.executable), sys.argv, [group.name for group in groups])


def main() -> int:
    bootstrap_runtime(requirement_groups=tuple(REQUIREMENT_GROUPS))
    print("[bootstrap] environment ready", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
