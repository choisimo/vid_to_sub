scope_id: scope-log-full-page-tui
plan_id: plan-log-full-page-tui-v1
phase_id: phase-3-plan-generation
contract_version: v1
decision_log_version: v2
risk_register_version: v2

# Dedicated Logs Tab for Textual TUI

## TL;DR
> **Summary**: Add a dedicated `7 Logs` tab inside the existing Textual `TabbedContent` so runtime logs can be viewed in an expanded full-page surface, while preserving the current bottom runtime log panel on tabs 1-6 and hiding that bottom panel only on the logs tab.
> **Deliverables**:
> - New `tab-logs` pane with full-height runtime `RichLog`
> - Centralized bottom-panel visibility sync tied to active tab
> - Runtime log fan-out/history preservation across bottom and full-page log surfaces
> - Focused `unittest` coverage plus executable TUI smoke validation
> **Effort**: Medium
> **Parallel**: Limited - 3 waves
> **Critical Path**: Task 1 -> Task 2 -> Task 3 -> Task 4 -> Task 5

## Context
### Original Request
- Add a log tab so the page can show logs only in a full-page view.
- Keep the existing bottom log tab/panel for the rest of the app.
- Hide the bottom log panel on the dedicated logs page because that page already shows logs full screen.
- Add a TUI page for expanded log viewing.

### Interview Summary
- The repo is a Textual TUI, not a web app.
- The feature is locked to a new `7 Logs` tab inside existing `TabbedContent`, not a separate fullscreen shell mode.
- Verification is locked to `tests-after`: targeted `unittest` coverage plus agent-executed TUI smoke checks.
- Default assumption: the new full-page log mirrors the existing global runtime log stream instead of introducing a second log source.

### Metis Review (gaps addressed)
- Keep `_log()` as the single runtime log sink and fan out from there.
- Keep `#bottom` mounted and hide it via display/layout toggling only when `tab-logs` is active.
- Preserve `#setup-log` and `#agent-log` as separate surfaces; they stay out of scope.
- Add explicit coverage for startup state, tab-switch recovery, fan-out failure tolerance, and max-line retention parity.
- Treat docs updates as optional and minimal only if verified repo docs explicitly state the old tab count.

## Work Objectives
### Core Objective
Add a dedicated runtime logs tab that gives the user a full-page log view in the existing TUI without regressing the current bottom-panel experience on all non-log tabs.

### Deliverables
- `vid_to_sub_app/tui/app.py` updated with `tab-logs`, `7` keybinding, centralized tab-state sync, and runtime-log fan-out behavior.
- `vid_to_sub_app/tui/styles.py` updated with the dedicated logs-tab layout and hidden-bottom styling.
- `tests/test_tui_logs_tab.py` added with focused regression coverage for logs-tab behavior.
- Minimal docs correction for `README.md`, and for `README.ko.md` only if it also explicitly states the outdated tab count or shortcut range.

### Definition of Done (verifiable conditions with commands)
- `python -m unittest tests.test_tui_logs_tab` exits `0`.
- `python -m unittest tests.test_tui_logs_tab tests.test_tui_helpers` exits `0`.
- `timeout 5s python tui.py` produces no traceback before timeout; exit code `124` is acceptable because the TUI remains running.
- The active logs tab hides `#bottom`, while switching back to a non-log tab restores `#bottom`.
- `_log()` writes to both runtime log surfaces without breaking when the full-page log widget is temporarily unavailable.

### Must Have
- Preserve existing tabs 1-6 and add `tab-logs` as the seventh tab.
- Keep the bottom runtime log panel available everywhere except the dedicated logs tab.
- Make the new logs tab show runtime log history, not just future lines.
- Keep runtime log trimming behavior aligned across both runtime log widgets.
- Keep existing setup and agent log behavior unchanged.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No separate fullscreen app mode.
- No new logging backend, persistence, export, search, filtering, or file-backed logs.
- No renaming or renumbering of existing tabs 1-6.
- No unrelated mixin refactors or style cleanup.
- No unification of runtime, setup, and agent logs into one generalized system.
- No docs sweep beyond a minimal factual correction if strictly necessary.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: `tests-after` using repo-standard `unittest`, plus one live TUI smoke pass.
- QA policy: Every task includes agent-executed happy-path and failure-path scenarios.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`
- Smoke baseline: use `timeout 5s python tui.py` and treat "no traceback before timeout" as the binary pass condition.

## Execution Strategy
### Parallel Execution Waves
> This feature is intentionally serialized around `vid_to_sub_app/tui/app.py`, which is the hotspot for tabs, bottom-panel layout, and runtime-log routing.

Wave 1:
- Task 1 - app shell logs-tab structure and binding

Wave 2:
- Task 2 - bottom-panel visibility sync
- Task 3 - runtime-log fan-out and history parity
- Task 4 - logs-tab styling and layout polish

Wave 3:
- Task 5 - focused `unittest` regression coverage
- Task 6 - minimal docs correction for verified tab-count mismatch

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2, 3, 4, and 5.
- Task 2 blocks Task 5.
- Task 3 blocks Task 5.
- Task 4 blocks Task 5.
- Task 6 depends on the final implemented tab count/order and can run once Tasks 1-4 are stable.

### Agent Dispatch Summary (wave -> task count -> categories)
- Wave 1 -> 1 task -> `quick`
- Wave 2 -> 3 tasks -> `quick`, `business-logic`
- Wave 3 -> 2 tasks -> `verification`, `writing`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Add the dedicated `tab-logs` pane and `7` binding in the app shell

  **What to do**: Update `vid_to_sub_app/tui/app.py` so the existing `TabbedContent` gains `TabPane("7 Logs", id="tab-logs")` after the current six tabs. Inside that pane, mount one full-height runtime `RichLog` with a stable ID such as `log-full` and `max_lines` matched to the bottom runtime log. Extend the existing bindings/action path so key `7` activates `tab-logs` using the same action flow as keys `1`-`6`.
  **Must NOT do**: Do not reorder existing tabs, do not remove `#log` from `#bottom`, and do not introduce a second navigation system outside `TabbedContent`.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: bounded shell update concentrated in one file.
  - Skills: `[]` - no special skill needed beyond the existing app pattern.
  - Omitted: `orchestration-contract` - plan already encodes handoff/version metadata.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 3, 4, 5 | Blocked By: none

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/tui/app.py` - existing `TabbedContent` composition, tab IDs `tab-browse` through `tab-agent`, and numeric binding conventions.
  - Pattern: `vid_to_sub_app/tui/styles.py` - existing runtime log sizing and bottom-panel layout conventions.
  - Pattern: `README.md` - current user-facing description of the TUI tab structure.
  - Test: `tests/test_tui_helpers.py` - existing `unittest` style and TUI-adjacent helper assertions.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m unittest tests.test_tui_logs_tab.LogsTabStructureTests` exits `0`.
  - [ ] `tests/test_tui_logs_tab.py` contains a structure test that asserts `tab-logs` is present and key `7` routes to it.
  - [ ] The dedicated logs pane contains a distinct runtime log widget whose `max_lines` matches the bottom runtime log.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Logs tab shell structure exists
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab.LogsTabStructureTests`
    Expected: Exit code 0; assertions confirm `tab-logs`, binding `7`, and full-page runtime log widget exist.
    Evidence: .sisyphus/evidence/task-1-logs-tab-structure.txt

  Scenario: Invalid tab target stays safe
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab.LogsTabFallbackTests`
    Expected: Exit code 0; invalid or missing tab IDs do not crash the app and do not hide `#bottom`.
    Evidence: .sisyphus/evidence/task-1-logs-tab-fallback.txt
  ```

  **Commit**: NO | Message: `feat(tui): add dedicated logs tab runtime view` | Files: `vid_to_sub_app/tui/app.py`, `tests/test_tui_logs_tab.py`

- [ ] 2. Centralize `#bottom` visibility sync for the logs tab only

  **What to do**: In `vid_to_sub_app/tui/app.py`, add one authoritative helper for logs-tab detection and one authoritative helper for bottom-panel layout sync. Use the active `TabbedContent` pane ID as the source of truth, show `#bottom` for tabs 1-6 and any invalid/empty state, and hide `#bottom` only when `tab-logs` is active. Hook this helper into both keyboard-driven tab changes and direct tab activation events so mouse and keyboard flows stay consistent.
  **Must NOT do**: Do not unmount or remove `#bottom`; do not duplicate hide/show logic across multiple event handlers; do not hide the footer or unrelated controls unless the existing layout requires it for the logs tab.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: this task locks the single source of truth for layout state transitions.
  - Skills: `[]` - existing app patterns are sufficient.
  - Omitted: `schema-evolution` - no schema or API migration is involved.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 5 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/tui/app.py` - current `action_tab(...)`, tab bindings, and bottom-panel composition around `#bottom`.
  - Pattern: `vid_to_sub_app/tui/styles.py` - current style hooks for `#bottom` and tab-area sizing.
  - Test: `tests/test_tui_helpers.py` - examples of app-method testing with patched widget lookups.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m unittest tests.test_tui_logs_tab.LogsTabVisibilityTests` exits `0`.
  - [ ] A test covers `browse -> logs -> browse` and asserts `#bottom` hides only on `tab-logs` and reappears afterward.
  - [ ] A test covers an invalid/empty active-tab fallback and asserts `#bottom` remains visible.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Keyboard and direct tab activation both sync bottom visibility
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab.LogsTabVisibilityTests`
    Expected: Exit code 0; tests confirm both action-driven and event-driven activation paths update `#bottom` correctly.
    Evidence: .sisyphus/evidence/task-2-bottom-visibility.txt

  Scenario: Visibility logic fails safe on unexpected tab state
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab.LogsTabFallbackTests`
    Expected: Exit code 0; bottom panel remains visible instead of disappearing on invalid tab input.
    Evidence: .sisyphus/evidence/task-2-bottom-fallback.txt
  ```

  **Commit**: NO | Message: `feat(tui): add dedicated logs tab runtime view` | Files: `vid_to_sub_app/tui/app.py`, `tests/test_tui_logs_tab.py`

- [ ] 3. Fan out runtime log writes and runtime-log clearing to both runtime log surfaces

  **What to do**: Keep `_log()` in `vid_to_sub_app/tui/app.py` as the only runtime-log entry point, and make it write to both the existing bottom runtime log and the new full-page runtime log. If a shared runtime-log clear/reset path exists, update it so both runtime log widgets stay in sync. Ensure the full-page runtime log can show prior history by reusing the same line stream rather than showing only future writes, and make failure handling tolerant when the full-page widget is not mounted yet.
  **Must NOT do**: Do not add a second independent runtime logging backend, do not reroute setup/agent logs into the runtime log, and do not let one missing widget crash `_log()`.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: correctness depends on preserving one logging contract while safely fanning out state.
  - Skills: `[]` - no extra skill is required.
  - Omitted: `release-readiness` - this is not a release-gate task.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: 5 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/tui/app.py` - `_log()` is the canonical runtime log sink and bottom runtime log owner.
  - Pattern: `vid_to_sub_app/tui/mixins/run_mixin.py` - runtime execution writes currently flow through app logging.
  - Pattern: `vid_to_sub_app/tui/mixins/history_mixin.py` - existing actions also route through `app._log(...)`.
  - Pattern: `vid_to_sub_app/tui/mixins/setup_mixin.py` - separate `#setup-log` pattern that must remain untouched.
  - Pattern: `vid_to_sub_app/tui/mixins/agent_mixin.py` - separate `#agent-log` pattern that must remain untouched.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m unittest tests.test_tui_logs_tab.RuntimeLogFanoutTests` exits `0`.
  - [ ] A test asserts `_log("x")` writes to both runtime log widgets.
  - [ ] A test asserts `_log("x")` still succeeds when the full-page log widget is absent or not yet mounted.
  - [ ] A test asserts runtime log clear/reset behavior, if present, updates both runtime log widgets consistently.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Runtime log fan-out preserves both surfaces
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab.RuntimeLogFanoutTests`
    Expected: Exit code 0; both runtime log widgets receive the same line stream and remain in sync.
    Evidence: .sisyphus/evidence/task-3-runtime-log-fanout.txt

  Scenario: Full-page log unavailable does not break runtime logging
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab.RuntimeLogFailureToleranceTests`
    Expected: Exit code 0; `_log()` still writes to the bottom runtime log and raises no exception when the full-page target is missing.
    Evidence: .sisyphus/evidence/task-3-runtime-log-failure.txt
  ```

  **Commit**: NO | Message: `feat(tui): add dedicated logs tab runtime view` | Files: `vid_to_sub_app/tui/app.py`, `tests/test_tui_logs_tab.py`

- [ ] 4. Add dedicated logs-tab layout styling without changing non-log tabs

  **What to do**: Update `vid_to_sub_app/tui/styles.py` so the new full-page runtime log fills the available tab body height, and so the hidden state for `#bottom` removes it from layout only on `tab-logs`. Keep the styling narrowly scoped to the new logs tab, the full-page runtime log widget, and the hide/show mechanism required by Task 2.
  **Must NOT do**: Do not restyle unrelated tabs, do not change setup/agent log appearance, and do not rely on hard-coded terminal sizes.

  **Recommended Agent Profile**:
  - Category: `quick` - Reason: localized styling work with existing Textual CSS patterns.
  - Skills: `[]` - no extra skill is needed.
  - Omitted: `release-readiness` - this is UI layout work, not release gating.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 5 | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/tui/styles.py` - existing IDs/classes for `#bottom`, `#log`, `#setup-log`, and `#agent-log`.
  - Pattern: `vid_to_sub_app/tui/app.py` - widget IDs introduced for the dedicated logs tab and bottom visibility hooks.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m unittest tests.test_tui_logs_tab.LogsTabStyleContractTests` exits `0`.
  - [ ] The dedicated logs tab runtime log fills the available vertical space when `#bottom` is hidden.
  - [ ] Non-log tabs retain the existing bottom runtime log layout after switching away from `tab-logs`.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Logs tab uses full-height runtime log layout
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab.LogsTabStyleContractTests`
    Expected: Exit code 0; style-contract assertions confirm the logs tab layout consumes the tab area while `#bottom` is hidden.
    Evidence: .sisyphus/evidence/task-4-logs-tab-style.txt

  Scenario: Layout does not leak onto non-log tabs
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab.LogsTabVisibilityTests`
    Expected: Exit code 0; after leaving `tab-logs`, non-log tabs render with the original bottom runtime panel visible again.
    Evidence: .sisyphus/evidence/task-4-non-log-layout.txt
  ```

  **Commit**: NO | Message: `feat(tui): add dedicated logs tab runtime view` | Files: `vid_to_sub_app/tui/styles.py`, `tests/test_tui_logs_tab.py`

- [ ] 5. Add focused regression coverage for structure, visibility, fan-out, and unaffected side logs

  **What to do**: Create `tests/test_tui_logs_tab.py` using the repo’s existing `unittest` conventions. Prefer `unittest.IsolatedAsyncioTestCase` with Textual `run_test()` if interactive mounting is required; otherwise keep tests as small direct app/helper tests. Include named test groups covering structure, visibility sync, runtime-log fan-out, startup state, fallback safety, line-retention parity, and regression protection for `#setup-log` / `#agent-log` remaining independent.
  **Must NOT do**: Do not migrate the repo to `pytest`, do not add a new test framework, and do not write vague snapshot tests that fail to prove the runtime-log contract.

  **Recommended Agent Profile**:
  - Category: `verification` - Reason: the goal is proof of behavior with existing repo test patterns.
  - Skills: `[]` - existing `unittest` usage is sufficient.
  - Omitted: `schema-evolution` - irrelevant to test coverage.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: none | Blocked By: 1, 2, 3, 4

  **References** (executor has NO interview context - be exhaustive):
  - Test: `tests/test_tui_helpers.py` - current `unittest` structure and app-helper testing style.
  - Pattern: `vid_to_sub_app/tui/app.py` - structure and helper methods under test.
  - Pattern: `vid_to_sub_app/tui/mixins/setup_mixin.py` - setup log surface behavior to keep unchanged.
  - Pattern: `vid_to_sub_app/tui/mixins/agent_mixin.py` - agent log surface behavior to keep unchanged.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -m unittest tests.test_tui_logs_tab` exits `0`.
  - [ ] `python -m unittest tests.test_tui_logs_tab tests.test_tui_helpers` exits `0`.
  - [ ] Tests cover startup bottom visibility, `browse -> logs -> browse`, `_log()` fan-out, missing full-page widget tolerance, retention parity, and unaffected setup/agent log routing.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Focused regression suite passes
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab tests.test_tui_helpers`
    Expected: Exit code 0; all dedicated logs-tab regression tests and touched helper tests pass.
    Evidence: .sisyphus/evidence/task-5-unittest-suite.txt

  Scenario: Side logs remain isolated from runtime logs tab
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab.SideLogIsolationTests`
    Expected: Exit code 0; setup and agent log helpers still target only their own widgets.
    Evidence: .sisyphus/evidence/task-5-side-log-isolation.txt
  ```

  **Commit**: YES | Message: `test(tui): cover logs tab visibility and fan-out` | Files: `tests/test_tui_logs_tab.py`, `tests/test_tui_helpers.py`

- [ ] 6. Correct the verified user-facing tab-count and shortcut docs mismatch

  **What to do**: Update `README.md` after implementation because it currently states `A 6-tab Textual TUI` and `1` to `6` switch tabs. Change only the factual lines that become wrong after adding `7 Logs`. Also inspect `README.ko.md`; update it only if it contains the same now-outdated tab-count or shortcut wording.
  **Must NOT do**: Do not rewrite unrelated documentation, screenshots, or setup sections. Do not create a docs-only diff unless there is a verified factual mismatch.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: this is a minimal factual docs correction task.
  - Skills: `[]` - no special writing skill is required.
  - Omitted: `release-readiness` - docs-only correction does not need operational gating.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: none | Blocked By: 1

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `README.md` - current English user-facing app description.
  - Pattern: `README.ko.md` - current Korean user-facing app description.
  - Pattern: `vid_to_sub_app/tui/app.py` - final authoritative tab count and order after implementation.

  **Acceptance Criteria** (agent-executable only):
  - [ ] `README.md` matches the implemented tab count/order and shortcut range exactly after the feature lands.
  - [ ] `README.ko.md` changes only if it explicitly contained the same outdated tab-count or shortcut wording.
  - [ ] `python -m unittest tests.test_tui_logs_tab` still exits `0` after any docs correction.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Verified README mismatch corrected minimally
    Tool: Bash
    Steps: Run `python -m unittest tests.test_tui_logs_tab` after the README correction
    Expected: Exit code 0; docs edits stay limited to the verified tab-count and shortcut text.
    Evidence: .sisyphus/evidence/task-6-docs-check.txt

  Scenario: Korean README only changes when mismatch is present
    Tool: Bash
    Steps: Inspect the final diff and confirm `README.ko.md` changed only if it explicitly contained the outdated tab-count or shortcut wording.
    Expected: Binary pass/fail; no speculative docs rewrite appears in the diff.
    Evidence: .sisyphus/evidence/task-6-docs-skip.txt
  ```

  **Commit**: YES | Message: `docs(tui): correct tab count reference` | Files: `README.md`, `README.ko.md`

## Final Verification Wave (MANDATORY - after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit - oracle
- [ ] F2. Code Quality Review - unspecified-high
- [ ] F3. Real Manual QA - unspecified-high (+ interactive_bash for TUI)
- [ ] F4. Scope Fidelity Check - deep

## Commit Strategy
- Commit 1: `feat(tui): add dedicated logs tab runtime view`
  - Includes Tasks 1-4 once app behavior is working and smoke-safe.
- Commit 2: `test(tui): cover logs tab visibility and fan-out`
  - Includes Task 5 after tests are green.
- Commit 3: `docs(tui): correct tab count reference`
  - Create after Task 6 because `README.md` already contains a verified outdated tab-count and shortcut reference.

## Success Criteria
- The app keeps the current bottom runtime log workflow on tabs 1-6.
- The new `7 Logs` tab provides a larger runtime log-only view and hides the bottom panel.
- Runtime logging remains single-source through `_log()` and stays resilient if one target widget is absent.
- Focused regression coverage and the smoke command both pass.
- No setup/agent log regressions and no unrelated TUI architecture changes.

## Handoff to Sisyphus
1. Artifacts
- This plan file
- Draft decisions captured in `.sisyphus/drafts/log-full-page-tui.md` until cleanup
- Research-backed constraints from `vid_to_sub_app/tui/app.py`, `vid_to_sub_app/tui/styles.py`, `tests/test_tui_helpers.py`, and Metis review findings
2. Entry Conditions
- Confirm current selectors and tab IDs in `vid_to_sub_app/tui/app.py` before editing.
- Confirm any runtime-log clear/reset path before changing `_log()` fan-out behavior.
3. Unresolved Assumption / Blocker
- None blocking; use the locked defaults in this plan.
4. Verification Request
- Verify exact current widget IDs and any existing clear helpers before code edits, then execute the defined unit tests and smoke checks after implementation.
