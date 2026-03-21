from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from textual.css.query import NoMatches
from textual.widgets import RichLog, TabbedContent, TabPane

from tui import VidToSubApp


class _FakeLog:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.cleared = False

    def write(self, text: str) -> None:
        self.messages.append(text)

    def clear(self) -> None:
        self.cleared = True


class _FakeBottom:
    def __init__(self) -> None:
        self.classes: set[str] = set()

    def add_class(self, name: str) -> None:
        self.classes.add(name)

    def remove_class(self, name: str) -> None:
        self.classes.discard(name)

    def has_class(self, name: str) -> bool:
        return name in self.classes


class LogsTabStructureTests(unittest.IsolatedAsyncioTestCase):
    async def test_tab_logs_exists_with_dedicated_full_log_widget(self) -> None:
        app = VidToSubApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            tabbed = app.query_one("TabbedContent", TabbedContent)
            tab_logs = app.query_one("#tab-logs", TabPane)
            full_log = app.query_one("#log-full", RichLog)
            bottom_log = app.query_one("#log", RichLog)

            self.assertEqual("tab-browse", tabbed.active)
            self.assertEqual("tab-logs", tab_logs.id)
            self.assertEqual(bottom_log.max_lines, full_log.max_lines)

    async def test_key_7_selects_logs_tab_without_regressing_existing_actions(
        self,
    ) -> None:
        app = VidToSubApp()

        bindings = {binding.key: binding.action for binding in app.BINDINGS}

        self.assertEqual("tab('tab-browse')", bindings["1"])
        self.assertEqual("tab('tab-agent')", bindings["6"])
        self.assertEqual("tab('tab-logs')", bindings["7"])

        async with app.run_test() as pilot:
            await pilot.pause()
            tabbed = app.query_one("TabbedContent", TabbedContent)

            await pilot.press("7")
            await pilot.pause()
            self.assertEqual("tab-logs", tabbed.active)

            await pilot.press("6")
            await pilot.pause()
            self.assertEqual("tab-agent", tabbed.active)


class LogsTabVisibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_keeps_bottom_visible_on_initial_non_logs_tab(self) -> None:
        app = VidToSubApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            tabbed = app.query_one("TabbedContent", TabbedContent)
            bottom = app.query_one("#bottom")

            self.assertEqual("tab-browse", tabbed.active)
            self.assertFalse(bottom.has_class("hidden-on-logs"))

    async def test_browse_logs_browse_toggles_bottom_visibility(self) -> None:
        app = VidToSubApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            tabbed = app.query_one("TabbedContent", TabbedContent)
            bottom = app.query_one("#bottom")

            await pilot.press("7")
            await pilot.pause()
            self.assertEqual("tab-logs", tabbed.active)
            self.assertTrue(bottom.has_class("hidden-on-logs"))

            await pilot.press("1")
            await pilot.pause()
            self.assertEqual("tab-browse", tabbed.active)
            self.assertFalse(bottom.has_class("hidden-on-logs"))

    async def test_direct_tab_activation_event_syncs_bottom_visibility(self) -> None:
        app = VidToSubApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            bottom = app.query_one("#bottom")

            app.on_tabbed_content_tab_activated(
                SimpleNamespace(pane=SimpleNamespace(id="tab-logs"))
            )
            await pilot.pause()
            self.assertTrue(bottom.has_class("hidden-on-logs"))

            app.on_tabbed_content_tab_activated(
                SimpleNamespace(pane=SimpleNamespace(id="tab-browse"))
            )
            await pilot.pause()
            self.assertFalse(bottom.has_class("hidden-on-logs"))


class LogsTabFallbackTests(unittest.TestCase):
    def test_invalid_tab_state_keeps_bottom_visible(self) -> None:
        app = VidToSubApp()
        bottom = _FakeBottom()
        bottom.add_class("hidden-on-logs")

        def query_one(selector: str, *_args):
            if selector == "#bottom":
                return bottom
            raise NoMatches(selector)

        with patch.object(app, "query_one", side_effect=query_one):
            app._sync_bottom_visibility("tab-missing")
            self.assertFalse(bottom.has_class("hidden-on-logs"))

            bottom.add_class("hidden-on-logs")
            app._sync_bottom_visibility("")

        self.assertFalse(bottom.has_class("hidden-on-logs"))

    def test_missing_or_unknown_active_tab_fails_safe_to_visible_bottom(self) -> None:
        app = VidToSubApp()
        bottom = _FakeBottom()
        bottom.add_class("hidden-on-logs")

        def query_one(selector: str, *_args):
            if selector == "#bottom":
                return bottom
            if selector == "TabbedContent":
                raise NoMatches(selector)
            raise NoMatches(selector)

        with patch.object(app, "query_one", side_effect=query_one):
            app._sync_bottom_visibility(None)

        self.assertFalse(bottom.has_class("hidden-on-logs"))


class RuntimeLogFanoutTests(unittest.TestCase):
    def test_log_writes_to_bottom_and_full_runtime_logs(self) -> None:
        app = VidToSubApp()
        bottom_log = _FakeLog()
        full_log = _FakeLog()

        def query_one(selector: str, *_args):
            if selector == "#log":
                return bottom_log
            if selector == "#log-full":
                return full_log
            raise NoMatches(selector)

        with patch.object(app, "query_one", side_effect=query_one):
            app._log("hello")

        self.assertEqual(["hello"], bottom_log.messages)
        self.assertEqual(["hello"], full_log.messages)

    def test_clear_runtime_logs_clears_both_runtime_surfaces(self) -> None:
        app = VidToSubApp()
        bottom_log = _FakeLog()
        full_log = _FakeLog()

        def query_one(selector: str, *_args):
            if selector == "#log":
                return bottom_log
            if selector == "#log-full":
                return full_log
            raise NoMatches(selector)

        with patch.object(app, "query_one", side_effect=query_one):
            app._clear_runtime_logs()

        self.assertTrue(bottom_log.cleared)
        self.assertTrue(full_log.cleared)


class RuntimeLogFailureToleranceTests(unittest.TestCase):
    def test_missing_full_page_log_does_not_break_runtime_logging(self) -> None:
        app = VidToSubApp()
        bottom_log = _FakeLog()

        def query_one(selector: str, *_args):
            if selector == "#log":
                return bottom_log
            if selector == "#log-full":
                raise NoMatches(selector)
            raise NoMatches(selector)

        with patch.object(app, "query_one", side_effect=query_one):
            app._log("still works")

        self.assertEqual(["still works"], bottom_log.messages)


class LogsTabStyleContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_logs_tab_runtime_log_fills_available_space_when_bottom_hidden(
        self,
    ) -> None:
        app = VidToSubApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            bottom = app.query_one("#bottom")
            tabbed = app.query_one("TabbedContent", TabbedContent)
            bottom_log = app.query_one("#log", RichLog)
            full_log = app.query_one("#log-full", RichLog)

            await pilot.press("7")
            await pilot.pause()

            self.assertEqual("tab-logs", tabbed.active)
            self.assertFalse(bottom.display)
            self.assertEqual(0, bottom.region.height)
            self.assertGreater(full_log.region.height, 0)
            self.assertEqual(tabbed.region.width, full_log.region.width)
            self.assertGreaterEqual(full_log.region.height, tabbed.region.height - 1)
            self.assertEqual(0, bottom_log.region.height)
            self.assertEqual(bottom_log.max_lines, full_log.max_lines)

    async def test_non_log_tabs_restore_bottom_runtime_log_layout_after_switching(
        self,
    ) -> None:
        app = VidToSubApp()

        async with app.run_test() as pilot:
            await pilot.pause()
            bottom = app.query_one("#bottom")
            tabbed = app.query_one("TabbedContent", TabbedContent)
            bottom_log = app.query_one("#log", RichLog)
            full_log = app.query_one("#log-full", RichLog)

            await pilot.press("7")
            await pilot.pause()
            await pilot.press("1")
            await pilot.pause()

            self.assertEqual("tab-browse", tabbed.active)
            self.assertTrue(bottom.display)
            self.assertGreater(bottom.region.height, 0)
            self.assertGreater(bottom_log.region.height, 0)
            self.assertLess(bottom_log.region.height, bottom.region.height)
            self.assertLess(bottom_log.region.width, bottom.region.width)
            self.assertEqual(0, full_log.region.height)


class SideLogIsolationTests(unittest.TestCase):
    def test_setup_log_helper_targets_only_setup_log_widget(self) -> None:
        app = VidToSubApp()
        setup_log = _FakeLog()
        selectors: list[str] = []

        def query_one(selector: str, *_args):
            selectors.append(selector)
            if selector == "#setup-log":
                return setup_log
            raise NoMatches(selector)

        with patch.object(app, "query_one", side_effect=query_one):
            app._setup_log("setup message")

        self.assertEqual(["#setup-log"], selectors)
        self.assertEqual(["setup message"], setup_log.messages)

    def test_agent_log_helper_targets_only_agent_log_widget(self) -> None:
        app = VidToSubApp()
        agent_log = _FakeLog()
        selectors: list[str] = []

        def query_one(selector: str, *_args):
            selectors.append(selector)
            if selector == "#agent-log":
                return agent_log
            raise NoMatches(selector)

        with patch.object(app, "query_one", side_effect=query_one):
            app._agent_log_write("agent message")

        self.assertEqual(["#agent-log"], selectors)
        self.assertEqual(["agent message"], agent_log.messages)


if __name__ == "__main__":
    unittest.main()
