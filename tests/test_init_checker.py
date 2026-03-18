from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import init_checker


class InitCheckerTests(unittest.TestCase):
    def test_resolve_groups_rejects_unknown_group(self) -> None:
        with self.assertRaises(ValueError):
            init_checker.resolve_groups(("missing",))

    def test_bootstrap_runtime_relaunches_into_project_venv(self) -> None:
        target_python = Path("/tmp/project/.venv/bin/python")

        with patch.object(init_checker, "ensure_venv", return_value=target_python), patch.object(
            init_checker,
            "is_same_interpreter",
            return_value=False,
        ), patch.object(init_checker, "relaunch") as relaunch:
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
        ), patch.object(init_checker, "install_requirements") as install_requirements, patch.object(
            init_checker, "relaunch"
        ) as relaunch, patch.dict(os.environ, {}, clear=False):
            init_checker.bootstrap_runtime(requirement_groups=("base", "faster-whisper"))

        self.assertEqual(1, install_requirements.call_count)
        self.assertEqual("base", install_requirements.call_args.args[1].name)
        relaunch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
