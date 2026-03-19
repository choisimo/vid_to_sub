from __future__ import annotations

from pathlib import Path

from textual import work
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.widgets import Button, DirectoryTree, Input, Static

from ..helpers import SEARCH_RESULT_LIMIT, build_search_preview, discover_input_matches
from ..state import db as _db


class BrowseMixin:
    # ── Browse tab ────────────────────────────────────────────────────────

    def _goto_tree(self, raw: str) -> None:
        p = Path(raw).expanduser()
        target = p if p.is_dir() else (p.parent if p.exists() else None)
        if target and target.is_dir():
            try:
                self.query_one("#dir-tree", DirectoryTree).path = target
                self.query_one("#tree-root", Input).value = str(target)
            except NoMatches:
                pass
            if len(self._val("inp-path-search")) >= 2:
                self._start_path_search()

    def _add_path(self, path: str) -> None:
        if path and path not in self._selected_paths:
            self._selected_paths.append(path)
            kind = "directory" if Path(path).is_dir() else "file"
            _db.touch_path(path, kind)
            self._refresh_sel_paths()
            self._refresh_recent_paths()
            self._update_cmd_preview()

    def _refresh_sel_paths(self) -> None:
        try:
            box = self.query_one("#sel-paths-box")
        except NoMatches:
            return
        box.remove_children()
        if not self._selected_paths:
            box.mount(
                Static(
                    "[dim]No paths selected — browse left[/]",
                    markup=True,
                )
            )
            return
        for i, p in enumerate(self._selected_paths):
            row = Horizontal(classes="sel-row")
            box.mount(row)
            row.mount(Static(p, classes="sel-label", markup=False))
            row.mount(Button("X", id=f"selrm-{i}", variant="error", classes="sel-btn"))

    def _refresh_recent_paths(self) -> None:
        try:
            box = self.query_one("#recent-box")
        except NoMatches:
            return
        box.remove_children()
        recent = _db.get_recent_paths(limit=12)
        if not recent:
            box.mount(Static("[dim]None yet[/]", markup=True))
            return
        for i, r in enumerate(recent):
            row = Horizontal(classes="recent-row")
            box.mount(row)
            row.mount(Static(r["path"], classes="recent-label", markup=False))
            row.mount(
                Button("+", id=f"radd-{i}", classes="recent-add", variant="default")
            )

    def _clear_path_search(self, status: str) -> None:
        self._search_results = []
        self._search_preview_path = None
        try:
            self.query_one("#search-status", Static).update(status)
        except NoMatches:
            pass
        self._refresh_search_results()
        self._refresh_search_preview()

    def _refresh_search_results(self) -> None:
        try:
            box = self.query_one("#search-results-box")
        except NoMatches:
            return
        box.remove_children()
        if not self._search_results:
            box.mount(Static("[dim]No search results yet.[/]", markup=True))
            return
        for i, path in enumerate(self._search_results):
            target = Path(path)
            label = path + ("/" if target.is_dir() else "")
            row = Horizontal(classes="search-row")
            box.mount(row)
            row.mount(Button("Go", id=f"sgo-{i}", classes="search-go"))
            row.mount(Button("View", id=f"spv-{i}", classes="search-preview-btn"))
            row.mount(Static(label, classes="search-label", markup=False))
            row.mount(
                Button("+", id=f"sadd-{i}", classes="search-add", variant="default")
            )

    def _refresh_search_preview(self) -> None:
        try:
            preview = self.query_one("#search-preview", Static)
        except NoMatches:
            return
        if not self._search_preview_path:
            preview.update("Search result preview will appear here.")
            return
        preview.update(build_search_preview(Path(self._search_preview_path)))

    def _set_search_preview(self, path: str | None) -> None:
        self._search_preview_path = path
        self._refresh_search_preview()

    def _start_path_search(self) -> None:
        query = self._val("inp-path-search")
        if len(query) < 2:
            self._clear_path_search(
                "[dim]Type at least 2 characters to start searching.[/]"
            )
            return
        root = Path(self._val("tree-root") or str(Path.home())).expanduser()
        if not root.is_dir():
            self._clear_path_search(f"[red]Search root not found:[/] {root}")
            return
        try:
            self.query_one("#search-status", Static).update(
                f"[cyan]Searching under[/] {root}"
            )
        except NoMatches:
            pass
        self._search_input_paths(str(root), query)

    @work(thread=True, exclusive=True, exit_on_error=False, name="path-search")
    def _search_input_paths(self, root: str, query: str) -> None:
        root_path = Path(root)
        results = discover_input_matches(root_path, query, limit=SEARCH_RESULT_LIMIT)
        if results:
            count_label = f"[green]{len(results)} match(es)[/]"
            if len(results) >= SEARCH_RESULT_LIMIT:
                count_label += f" (showing first {SEARCH_RESULT_LIMIT})"
            status = f"{count_label} for [bold]{query}[/] under {root_path}"
        else:
            status = f"[yellow]No matches[/] for [bold]{query}[/] under {root_path}"
        self.call_from_thread(self._apply_path_search_results, results, status)

    def _apply_path_search_results(self, results: list[str], status: str) -> None:
        self._search_results = results
        self._search_preview_path = results[0] if results else None
        try:
            self.query_one("#search-status", Static).update(status)
        except NoMatches:
            pass
        self._refresh_search_results()
        self._refresh_search_preview()
