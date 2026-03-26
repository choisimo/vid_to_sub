from __future__ import annotations

import unittest
from unittest.mock import patch

from tui import VidToSubApp


class SharedDbLifecycleTests(unittest.TestCase):
    def test_on_unmount_closes_shared_db(self) -> None:
        with (
            patch("vid_to_sub_app.tui.app._db.seed_defaults"),
            patch("vid_to_sub_app.tui.app.close_shared_db") as close_shared_db,
        ):
            app = VidToSubApp()
            app.on_unmount()

        close_shared_db.assert_called_once_with()
