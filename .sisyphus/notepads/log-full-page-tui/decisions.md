## Decisions

- Task 1 decision: add `TabPane("7 Logs", id="tab-logs")` directly after `tab-agent` in `vid_to_sub_app/tui/app.py`, keep bottom `#log` untouched, and set `#log-full` height inline (`styles.height = "1fr"`) instead of changing shared CSS in this task.
- Task 2 decision: add `_is_logs_tab()` and `_sync_bottom_visibility()` in `vid_to_sub_app/tui/app.py`, and make the helper fail safe by hiding `#bottom` only when the active tab id is exactly `tab-logs`.
- Task 3 decision: keep `_log()` as the only runtime log writer by faning out there to `#log` and `#log-full`, wrap the `#log-full` lookup in `try/except NoMatches`, and route run-mixin clear/write entry points through `_clear_runtime_logs()` plus `_log()`.
- Task 4 decision: move `#log-full` height control into `vid_to_sub_app/tui/styles.py` and hide `#bottom` with a dedicated `#bottom.hidden-on-logs { display: none; }` rule so the widget stays mounted while leaving `#log`, `#setup-log`, and `#agent-log` untouched.
- Task 5 decision: keep `tests/test_tui_logs_tab.py` in repo-standard `unittest` form by mixing mounted `run_test()` checks for tab visibility behavior with small direct helper tests for `_log()`, `_clear_runtime_logs()`, and side-log routing so the regression suite proves the contract without introducing a new test framework.
- Task 6 decision: update only the verified user-facing tab-count and shortcut text in `README.md` and `README.ko.md`, because both files explicitly referenced the old 6-tab / `1`-to-`6` state and no broader docs rewrite was needed.
