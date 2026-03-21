## Learning Log

- Task 1: `VidToSubApp` numeric tab switching is centralized in `BINDINGS` plus `action_tab()`, so the logs tab reuses that path with `Binding("7", "tab('tab-logs')", show=False)` and a dedicated `#log-full` RichLog in `tab-logs`.
- Task 2: keyboard tab switches go through `action_tab()`, but mouse tab changes need the Textual `on_tabbed_content_tab_activated()` hook, so `_sync_bottom_visibility()` has to be called from both paths to keep `#bottom` in sync.
- Task 3: the runtime log clear/reset path lives in `vid_to_sub_app/tui/mixins/run_mixin.py`, so fan-out is only consistent when both clearing and writing are centralized through app helpers rather than direct `#log` access.
- Task 4: `#log-full` can rely on scoped CSS in `vid_to_sub_app/tui/styles.py`, which lets the full-page logs tab own its height there and keeps Task 1's inline sizing no longer necessary.
- Task 5: the most stable coverage split is to use mounted Textual tests only where widget state matters (`#bottom` visibility, tab switching, retention parity) and use patched `query_one()` helpers for fan-out and isolation checks, because those contracts depend on selectors and method routing more than on rendered RichLog internals.
- Task 6: both README variants needed edits, but only for the exact facts the implementation changed - the tab count/list and numeric shortcut range - so a grep pass is a cheap way to prove there are no stale `6-tab` or `1`-to-`6` references left after the docs update.
