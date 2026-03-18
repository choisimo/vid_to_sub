from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import init_checker


class InitCheckerTests(unittest.TestCase):
    def test_resolve_groups_rejects_unknown_group(self) -> None:
        with self.assertRaises(ValueError):
            _ = init_checker.resolve_groups(("missing",))

    def test_bootstrap_runtime_relaunches_into_project_venv(self) -> None:
        target_python = Path("/tmp/project/.venv/bin/python")

        with patch.object(init_checker, "ensure_venv", return_value=target_python), patch.object(
            init_checker,
            "is_same_interpreter",
            return_value=False,
        ), patch.object(init_checker, "ensure_pip"), patch.object(
            init_checker, "missing_modules", return_value=[]
        ), patch.object(init_checker, "install_requirements", return_value=True), patch.object(
            init_checker, "relaunch"
        ) as relaunch:
            init_checker.bootstrap_runtime(requirement_groups=("base",))

        relaunch.assert_called_once()
        self.assertEqual(target_python, relaunch.call_args.args[0])

    def test_bootstrap_runtime_installs_missing_groups_and_reexecs(self) -> None:
        current_python = Path(sys.executable)

        with patch.object(init_checker, "ensure_venv", return_value=current_python), patch.object(
            init_checker,
            "is_same_interpreter",
            return_value=True,
        ), patch.object(init_checker, "ensure_pip"), patch.object(
            init_checker,
            "missing_modules",
            side_effect=[["textual"], []],
        ), patch.object(init_checker, "install_requirements", return_value=True) as install_requirements, patch.object(
            init_checker, "relaunch"
        ) as relaunch, patch.dict(os.environ, {}, clear=False):
            init_checker.bootstrap_runtime(requirement_groups=("base", "faster-whisper"))

        install_requirements.assert_called_once_with(
            sys.executable, init_checker.REQUIREMENT_GROUPS["base"]
        )
        relaunch.assert_called_once()

    def test_install_requirements_returns_true_on_success(self) -> None:
        group = init_checker.RequirementGroup("base", "requirements.txt", ("textual",))
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0),
        ):
            result = init_checker.install_requirements(sys.executable, group)
        self.assertTrue(result)

    def test_install_requirements_returns_false_on_failure(self) -> None:
        group = init_checker.RequirementGroup("base", "requirements.txt", ("textual",))
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=1),
        ):
            result = init_checker.install_requirements(sys.executable, group)
        self.assertFalse(result)

    def test_install_requirements_returns_false_for_missing_file(self) -> None:
        group = init_checker.RequirementGroup("fake", "nonexistent.txt", ("fake",))
        result = init_checker.install_requirements(sys.executable, group)
        self.assertFalse(result)

    def test_bootstrap_skips_optional_group_on_pip_failure(self) -> None:
        current_python = Path(sys.executable)
        with patch.object(init_checker, "ensure_venv", return_value=current_python), patch.object(
            init_checker, "is_same_interpreter", return_value=True
        ), patch.object(init_checker, "ensure_pip"), patch.object(
            init_checker, "missing_modules", return_value=["whisperx"]
        ), patch.object(init_checker, "install_requirements", return_value=False), patch.object(
            init_checker, "relaunch"
        ) as relaunch:
            init_checker.bootstrap_runtime(requirement_groups=("whisperx",))
        relaunch.assert_not_called()

    def test_bootstrap_raises_on_required_group_pip_failure(self) -> None:
        current_python = Path(sys.executable)
        with patch.object(init_checker, "ensure_venv", return_value=current_python), patch.object(
            init_checker, "is_same_interpreter", return_value=True
        ), patch.object(init_checker, "ensure_pip"), patch.object(
            init_checker, "missing_modules", return_value=["textual"]
        ), patch.object(init_checker, "install_requirements", return_value=False), self.assertRaises(
            RuntimeError
        ):
            init_checker.bootstrap_runtime(requirement_groups=("base",))


if __name__ == "__main__":
    _ = unittest.main()
