scope_id: scope-translation-history-subtitle-copy
plan_id: plan-translation-history-subtitle-copy
phase_id: phase-plan-generation
contract_version: v1
decision_log_version: v1
risk_register_version: v1

# Default Translation, History Translation, and Subtitle Copy

## TL;DR
> **Summary**: Extend the existing Settings -> Transcribe -> History flow so translation defaults to on, auto-disables at runtime when translation-capable config cannot be resolved, completed History rows can trigger translation directly from persisted subtitle outputs, and multiple History rows can bulk-copy subtitle files into one chosen folder with deterministic duplicate renaming.
> **Deliverables**:
> - Persisted default-translation-on behavior aligned with current settings sync
> - Safe translation capability guard that skips expected config errors
> - History translate action for completed jobs using persisted subtitle outputs
> - History multi-select bulk subtitle copy with partial-failure reporting
> - `unittest` coverage for settings sync, history translation, and copy edge cases
> **Effort**: Medium
> **Parallel**: YES - 3 waves
> **Critical Path**: 1 -> 2 -> 3 -> 5 -> 7 -> F1-F4

## Context
### Original Request
- 기본적으로 번역 활성화되도록 해주고
- AI agent 미설정 시에는 자동 fallback 으로 비활성해서 예상 오류 넘기기
- 이미 완료된 History 내부의 작업 리스트들에 대해서 번역 바로 수행할 수 있도록 기능 개선
- 만약 자막 삭제로 인한 미존재시에는 예외처리 되도록
- history 작업 내역들을 여러 개 선택해서 각각의 자막들만 원하는 곳으로 복사 할 수 있는 기능 구현해서 적용

### Interview Summary
- Copy mode is fixed to one destination folder with automatic duplicate renaming.
- Test strategy is fixed to tests-after using the existing Python `unittest` suite.
- Existing architecture should be extended rather than replaced: settings sync remains the source for new defaults, history actions remain centered in the current history mixin, and output file paths stored in job history remain authoritative for history-driven actions.
- Missing subtitle files must be handled as partial-success cases with visible reporting, not as fatal workflow failures.

### Metis Review (gaps addressed)
- Translation availability is defined by a single `translation_capable` predicate derived from already supported translation config resolution paths, not by a vague "AI agent configured" check.
- `jobs.output_paths` is the authoritative source for history translate/copy actions; no filename guessing is allowed unless the codebase already exposes a trusted helper.
- Multi-select history work must explicitly manage selection reset on refresh/delete to avoid stale row actions.
- Bulk copy and history translation must be per-file tolerant: missing source, unreadable source, unwritable destination, malformed `output_paths`, and duplicate collisions are summarized instead of aborting the whole action.

## Work Objectives
### Core Objective
Deliver a decision-safe enhancement to the existing TUI workflow so users can rely on translation being on by default when possible, can re-run translation from completed history artifacts without manually reloading jobs, and can bulk-copy subtitle outputs from selected history jobs into a chosen folder.

### Deliverables
- New persisted setting and UI sync for default translation enabled state.
- Shared translation capability guard used before emitting translation CLI arguments or executing history translation actions.
- History action for translating completed job subtitle outputs directly from persisted `output_paths`.
- History multi-select state, subtitle filtering, destination selection, duplicate renaming, and copy result summaries.
- Test coverage in `tests/test_tui_helpers.py` and, if helper extraction warrants it, related translation tests in `tests/test_translation_pipeline.py`.

### Definition of Done (verifiable conditions with commands)
- `python -m unittest tests.test_tui_helpers tests.test_translation_pipeline` exits `0` and prints `OK`.
- New tests prove default translation is on for fresh state, but runtime args omit translation flags when translation-capable config cannot be resolved.
- New tests prove completed history jobs can trigger translation from persisted subtitle outputs and skip missing files without aborting the entire action.
- New tests prove multi-select history copy writes files into one destination folder and renames duplicates as `name (1).ext`, `name (2).ext`, and so on.

### Must Have
- Preserve current Settings -> Transcribe sync behavior.
- Preserve current History load/rerun actions while adding new capabilities around them.
- Use `jobs.output_paths` from `vid_to_sub_app/db.py` as the primary artifact list for history-driven operations.
- Limit history translation and bulk copy eligibility to completed jobs.
- Treat missing subtitle files, malformed output metadata, and copy permission failures as explicit partial-success states with user-visible summaries.
- Keep copy/translation artifact filtering in testable helper code rather than embedding all logic inside widget callbacks.

### Must NOT Have
- No schema redesign beyond the minimal key/value setting needed for default translation enabled state.
- No deletion of subtitle files when deleting history rows.
- No filename guessing when `output_paths` is empty or malformed unless an already existing trusted helper is found and referenced during implementation.
- No hard failure for expected missing-config or missing-file cases that can be skipped safely.
- No scope expansion into unrelated transcription, distributed execution, or agent planning features.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after with Python `unittest`
- QA policy: Every task includes agent-executed happy-path and failure-path verification.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.txt`
- Primary verification command: `python -m unittest tests.test_tui_helpers tests.test_translation_pipeline`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. Implementation + tests stay combined inside each task.

Wave 1: translation default contract, translation capability guard, history artifact helper extraction
Wave 2: history rerun hydration, direct history translation, multi-select History state and destination controls
Wave 3: bulk subtitle copy execution and summary reporting

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2 and 4.
- Task 2 blocks Tasks 4 and 5.
- Task 3 blocks Tasks 5 and 7.
- Task 4 does not block later tasks but must land before final verification.
- Task 5 blocks Task 7.
- Task 6 blocks Task 7.
- Task 7 blocks F1-F4.

### Agent Dispatch Summary (wave -> task count -> categories)
- Wave 1 -> 3 tasks -> `business-logic`
- Wave 2 -> 3 tasks -> `business-logic`, `visual-engineering`
- Wave 3 -> 1 task -> `business-logic`
- Final Verification -> 4 tasks -> `oracle`, `unspecified-high`, `deep`

## TODOs
> Implementation + Test = ONE task. Never separate.
> Every task must leave clear evidence in `.sisyphus/evidence/`.

- [ ] 1. Persist the default translation-enabled setting through Settings and Transcribe sync

  **What to do**: Add a minimal key/value setting for default translation enabled state and wire it through the existing Settings -> Transcribe sync path. Add a dedicated Settings switch for the default so users can turn the new default off later, and make fresh Transcribe state honor that setting before any run starts.
  **Must NOT do**: Do not redesign the settings schema, add new tables, or bypass the current settings mixin save/prefill flow.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: touches persisted defaults and UI state contracts.
  - Skills: `[]` - no extra skill required.
  - Omitted: `orchestration-contract` - plan already defines handoff and metadata.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 2, 4 | Blocked By: none

  **References**:
  - Pattern: `vid_to_sub_app/tui/mixins/settings_mixin.py` - existing settings save, prefill, and sync behavior to extend.
  - Pattern: `vid_to_sub_app/tui/app.py` - existing translation switch and settings widgets layout.
  - API/Type: `vid_to_sub_app/db.py` - key/value defaults and job settings persistence.
  - API/Type: `vid_to_sub_app/tui/models.py` - run-state fields already carrying translation state.
  - Test: `tests/test_tui_helpers.py` - current settings/UI state regression patterns.

  **Acceptance Criteria**:
  - [ ] A persisted default key exists for translation enabled state and is read during Settings prefill and save.
  - [ ] Fresh TUI state sets `#sw-translate` on when no explicit saved override disables it.
  - [ ] Saved Settings with translation disabled keep `#sw-translate` off after reload.
  - [ ] `python -m unittest tests.test_tui_helpers` exits `0` with new assertions covering fresh-state on and saved-state off.

  **QA Scenarios**:
  ```text
  Scenario: fresh state defaults translation on
    Tool: Bash
    Steps: Add tests in `tests/test_tui_helpers.py` for a clean settings state, then run `python -m unittest tests.test_tui_helpers`
    Expected: The suite passes and includes coverage proving fresh state enables `#sw-translate`
    Evidence: .sisyphus/evidence/task-1-default-translate-setting.txt

  Scenario: saved override keeps translation off
    Tool: Bash
    Steps: Add tests in `tests/test_tui_helpers.py` for a saved default-off setting, then run `python -m unittest tests.test_tui_helpers`
    Expected: The suite passes and proves the saved override keeps `#sw-translate` off after prefill
    Evidence: .sisyphus/evidence/task-1-default-translate-setting-error.txt
  ```

  **Commit**: NO | Message: `handled in task 2 feature commit` | Files: `vid_to_sub_app/tui/mixins/settings_mixin.py`, `vid_to_sub_app/tui/app.py`, `vid_to_sub_app/db.py`

- [ ] 2. Add a shared translation-capable guard and safe auto-disable before runtime translation starts

  **What to do**: Introduce one reusable `translation_capable` predicate that resolves whether translation can actually run from the already supported translation config sources. Use it before CLI arg emission and before history translation actions so translation stays enabled in the UI by default but auto-disables at execution time when model, base URL, or API key cannot be resolved. Emit a visible non-fatal status message whenever the guard suppresses translation.
  **Must NOT do**: Do not key this behavior off Agent tab configuration alone, and do not allow expected missing-config cases to raise runtime translation errors.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: central runtime guard shared by multiple workflows.
  - Skills: `[]` - no extra skill required.
  - Omitted: `release-readiness` - no deployment or ops change in this task.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: 4, 5 | Blocked By: 1

  **References**:
  - Pattern: `vid_to_sub_app/tui/mixins/run_mixin.py` - current run snapshot and CLI arg emission path.
  - Pattern: `vid_to_sub_app/cli/translation.py` - verified translation config resolution and current error paths.
  - Pattern: `vid_to_sub_app/tui/mixins/agent_mixin.py` - existing fallback relationship between agent and translation settings.
  - Test: `tests/test_tui_helpers.py` - run-argument/state verification patterns.
  - Test: `tests/test_translation_pipeline.py` - translation-path verification patterns.

  **Acceptance Criteria**:
  - [ ] Translation CLI flags are omitted when translation-capable config cannot be resolved, even if the UI switch is on.
  - [ ] A user-visible warning/status message reports that translation was auto-disabled because required config was unavailable.
  - [ ] History translation actions reuse the same guard rather than duplicating config checks.
  - [ ] `python -m unittest tests.test_tui_helpers tests.test_translation_pipeline` exits `0` with new cases covering safe auto-disable behavior.

  **QA Scenarios**:
  ```text
  Scenario: runtime omits translation flags when config is incomplete
    Tool: Bash
    Steps: Add guard-focused tests in `tests/test_tui_helpers.py` and `tests/test_translation_pipeline.py`, then run `python -m unittest tests.test_tui_helpers tests.test_translation_pipeline`
    Expected: The suite passes and proves missing translation config suppresses translation flags without raising translation runtime errors
    Evidence: .sisyphus/evidence/task-2-translation-guard.txt

  Scenario: warning is emitted instead of failing
    Tool: Bash
    Steps: Extend tests to assert the warning/status path for missing config, then run `python -m unittest tests.test_tui_helpers tests.test_translation_pipeline`
    Expected: The suite passes and proves the workflow continues with translation skipped and a concrete warning recorded
    Evidence: .sisyphus/evidence/task-2-translation-guard-error.txt
  ```

  **Commit**: YES | Message: `feat(tui): default translation on with safe auto-disable` | Files: `vid_to_sub_app/tui/mixins/run_mixin.py`, `vid_to_sub_app/tui/mixins/settings_mixin.py`, `vid_to_sub_app/tui/app.py`, `vid_to_sub_app/cli/translation.py`, `tests/test_tui_helpers.py`, `tests/test_translation_pipeline.py`

- [ ] 3. Normalize History output artifacts into reusable subtitle and translate-eligible file lists

  **What to do**: Add a helper layer for history-driven actions that parses `jobs.output_paths`, treats malformed or empty metadata as a handled error state, filters subtitle files for bulk copy to `.srt` and `.vtt`, and filters translate-eligible files to `.srt` only because `parse_srt` already exists. Return structured summaries with copied, skipped, missing, invalid, and renamed counts so both translate and copy actions can report partial success consistently.
  **Must NOT do**: Do not guess filenames when `output_paths` is empty or malformed, and do not treat `.txt`, `.tsv`, or `.json` artifacts as subtitle files for bulk copy.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: shared helper logic with filesystem and metadata edge cases.
  - Skills: `[]` - no extra skill required.
  - Omitted: `schema-evolution` - no schema migration is involved.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5, 7 | Blocked By: none

  **References**:
  - Pattern: `vid_to_sub_app/db.py` - authoritative persisted `output_paths` and job status metadata.
  - Pattern: `vid_to_sub_app/cli/output.py` - verified `parse_srt` helper and output format definitions.
  - Pattern: `vid_to_sub_app/tui/mixins/history_mixin.py` - current history action entry point to extend.
  - Test: `tests/test_tui_helpers.py` - best existing place for history/state helper tests.

  **Acceptance Criteria**:
  - [ ] Empty or malformed `output_paths` is handled without crashes and returns an explicit invalid-artifact result.
  - [ ] Bulk copy eligibility includes only `.srt` and `.vtt` paths that still exist.
  - [ ] History translation eligibility includes only existing `.srt` paths.
  - [ ] `python -m unittest tests.test_tui_helpers` exits `0` with helper coverage for valid, missing, malformed, and mixed-format outputs.

  **QA Scenarios**:
  ```text
  Scenario: valid history outputs classify correctly
    Tool: Bash
    Steps: Add helper tests in `tests/test_tui_helpers.py` for mixed `.srt`, `.vtt`, `.txt`, and `.json` `output_paths`, then run `python -m unittest tests.test_tui_helpers`
    Expected: The suite passes and proves only existing `.srt` and `.vtt` are bulk-copy eligible while only existing `.srt` is translate-eligible
    Evidence: .sisyphus/evidence/task-3-history-output-helper.txt

  Scenario: malformed metadata and missing files are summarized
    Tool: Bash
    Steps: Add helper tests for malformed JSON, empty lists, and deleted files, then run `python -m unittest tests.test_tui_helpers`
    Expected: The suite passes and proves invalid and missing artifacts are counted and reported without aborting the action
    Evidence: .sisyphus/evidence/task-3-history-output-helper-error.txt
  ```

  **Commit**: NO | Message: `handled in task 5 and task 7 feature commits` | Files: `vid_to_sub_app/tui/mixins/history_mixin.py`, `tests/test_tui_helpers.py`

- [ ] 4. Extend History load and rerun to restore translation intent from completed jobs

  **What to do**: Update history hydration so loading or rerunning a completed job restores `target_lang` into the Transcribe form, turns `#sw-translate` on only when the loaded job had a target language and the shared translation-capable guard passes, and otherwise leaves the target filled while reporting that translation will stay disabled until config is available. Keep backend, model, language, selected path, and output directory restoration behavior unchanged.
  **Must NOT do**: Do not silently clear the saved target language, and do not force an immediate rerun into a missing-config failure.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: existing history hydration and rerun semantics need precise extension.
  - Skills: `[]` - no extra skill required.
  - Omitted: `visual-engineering` - widget layout changes are secondary here.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: none | Blocked By: 1, 2

  **References**:
  - Pattern: `vid_to_sub_app/tui/mixins/history_mixin.py` - current `_load_history_job` and `_rerun_history_job` flow.
  - Pattern: `vid_to_sub_app/tui/app.py` - current history button dispatch and translation widget event handling.
  - API/Type: `vid_to_sub_app/db.py` - persisted `target_lang`, `backend`, `model`, and `output_dir` fields.
  - Test: `tests/test_tui_helpers.py` - history hydration and rerun verification patterns.

  **Acceptance Criteria**:
  - [ ] Loading a completed history job restores `target_lang` into the translate target input.
  - [ ] Rerunning a completed history job keeps translation enabled only when the shared translation-capable guard passes.
  - [ ] Missing translation capability yields a warning and a safe rerun without translation flags.
  - [ ] `python -m unittest tests.test_tui_helpers` exits `0` with new history load/rerun coverage for translation intent.

  **QA Scenarios**:
  ```text
  Scenario: history load restores target language and valid translate state
    Tool: Bash
    Steps: Add history hydration tests in `tests/test_tui_helpers.py`, then run `python -m unittest tests.test_tui_helpers`
    Expected: The suite passes and proves a completed job with `target_lang` repopulates the translate target and enables translation when config is resolvable
    Evidence: .sisyphus/evidence/task-4-history-rerun-hydration.txt

  Scenario: history rerun degrades safely when config is missing
    Tool: Bash
    Steps: Add rerun tests for a completed job with `target_lang` but missing translation config, then run `python -m unittest tests.test_tui_helpers`
    Expected: The suite passes and proves rerun proceeds without translation flags and records a warning instead of failing
    Evidence: .sisyphus/evidence/task-4-history-rerun-hydration-error.txt
  ```

  **Commit**: NO | Message: `handled in task 5 feature commit` | Files: `vid_to_sub_app/tui/mixins/history_mixin.py`, `vid_to_sub_app/tui/app.py`, `tests/test_tui_helpers.py`

- [ ] 5. Add a single-row History Translate action for completed jobs using persisted `.srt` outputs

  **What to do**: Add a new single-row history action and button that operates only on completed jobs. For the selected job, use the normalized history artifact helper to locate existing `.srt` outputs, parse them with the existing `parse_srt` path, translate their segment text through the existing translation pipeline, and write translated subtitle outputs using the job's original output naming convention via the existing output writer. When multiple `.srt` files exist for one job, process each eligible file and summarize translated, skipped, and missing counts.
  **Must NOT do**: Do not re-transcribe the source video for this action, do not attempt to translate `.vtt` directly, and do not abort the entire action because one subtitle file is missing.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: combines history metadata, subtitle parsing, translation, and output writing.
  - Skills: `[]` - no extra skill required.
  - Omitted: `safe-refactor` - feature fits current flow without structural migration.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7 | Blocked By: 2, 3

  **References**:
  - Pattern: `vid_to_sub_app/tui/mixins/history_mixin.py` - current history actions to extend with a new translate path.
  - Pattern: `vid_to_sub_app/tui/app.py` - current history button event dispatch.
  - API/Type: `vid_to_sub_app/cli/output.py` - verified `parse_srt` and `write_outputs` reuse path.
  - API/Type: `vid_to_sub_app/cli/translation.py` - existing translation pipeline for segment text.
  - API/Type: `vid_to_sub_app/db.py` - authoritative completed-job metadata and `output_paths`.
  - Test: `tests/test_tui_helpers.py` - best location for history action regression tests.
  - Test: `tests/test_translation_pipeline.py` - translation behavior and output consistency checks.

  **Acceptance Criteria**:
  - [ ] Completed history jobs expose a translate action while non-completed jobs reject the action with a visible status.
  - [ ] Existing `.srt` outputs translate without re-transcribing the source video.
  - [ ] Missing or non-eligible outputs are skipped and included in a partial-success summary.
  - [ ] `python -m unittest tests.test_tui_helpers tests.test_translation_pipeline` exits `0` with new coverage for history translate success and partial-failure paths.

  **QA Scenarios**:
  ```text
  Scenario: completed history job translates existing subtitle outputs
    Tool: Bash
    Steps: Add tests in `tests/test_tui_helpers.py` and `tests/test_translation_pipeline.py` for a completed job with an existing `.srt` output, then run `python -m unittest tests.test_tui_helpers tests.test_translation_pipeline`
    Expected: The suite passes and proves the history action translates subtitle text, writes translated output with the original naming convention, and does not re-enter the transcription path
    Evidence: .sisyphus/evidence/task-5-history-translate.txt

  Scenario: partial success skips missing or unsupported outputs
    Tool: Bash
    Steps: Add tests for a completed job whose `output_paths` include one deleted `.srt` and one `.vtt`, then run `python -m unittest tests.test_tui_helpers tests.test_translation_pipeline`
    Expected: The suite passes and proves the missing `.srt` and unsupported `.vtt` are skipped, counts are summarized, and the action does not abort
    Evidence: .sisyphus/evidence/task-5-history-translate-error.txt
  ```

  **Commit**: YES | Message: `feat(history): add direct translation from completed subtitles` | Files: `vid_to_sub_app/tui/mixins/history_mixin.py`, `vid_to_sub_app/tui/app.py`, `vid_to_sub_app/cli/output.py`, `vid_to_sub_app/cli/translation.py`, `tests/test_tui_helpers.py`, `tests/test_translation_pipeline.py`

- [ ] 6. Add explicit History multi-select state and destination controls for subtitle copy

  **What to do**: Keep the existing single active row behavior, but add a separate marked-row selection model for bulk operations because the current `DataTable` flow is single-row oriented. Introduce marked-history state, a mark/unmark action for the current row, a clear-marks action, and a History copy destination input that can be filled directly or populated from the current directory tree selection. Show the marked count in the History status/detail area so the user can verify which rows are queued for bulk copy.
  **Must NOT do**: Do not replace the existing `_hist_key` single-row behavior, and do not depend on unsupported native multi-select behavior from the current `DataTable` widget.

  **Recommended Agent Profile**:
  - Category: `visual-engineering` - Reason: adds UI controls and stateful interactions on top of the existing History tab.
  - Skills: `[]` - no extra skill required.
  - Omitted: `business-logic` - helper logic is already handled in task 3.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7 | Blocked By: none

  **References**:
  - Pattern: `vid_to_sub_app/tui/app.py` - current History tab button layout, `DataTable`, and button dispatch.
  - Pattern: `vid_to_sub_app/tui/mixins/history_mixin.py` - current single-row history state and detail refresh flow.
  - Pattern: `vid_to_sub_app/tui/mixins/browse_mixin.py` - existing directory tree interaction model to mirror for destination capture.
  - Test: `tests/test_tui_helpers.py` - existing history/state tests to extend.

  **Acceptance Criteria**:
  - [ ] Users can mark and unmark multiple history rows for bulk copy without breaking existing single-row load/rerun/delete behavior.
  - [ ] The History tab exposes a copy destination input and a way to populate it from the current directory-tree selection.
  - [ ] Refreshing or deleting history rows clears or recomputes stale marked selections safely.
  - [ ] `python -m unittest tests.test_tui_helpers` exits `0` with new coverage for marked-row state and stale-selection cleanup.

  **QA Scenarios**:
  ```text
  Scenario: mark and clear multiple rows safely
    Tool: Bash
    Steps: Add multi-select state tests in `tests/test_tui_helpers.py`, then run `python -m unittest tests.test_tui_helpers`
    Expected: The suite passes and proves multiple rows can be marked, unmarked, and cleared without changing the active single-row action target
    Evidence: .sisyphus/evidence/task-6-history-multiselect.txt

  Scenario: refresh or delete drops stale marks
    Tool: Bash
    Steps: Add tests covering table refresh and row deletion after rows were marked, then run `python -m unittest tests.test_tui_helpers`
    Expected: The suite passes and proves stale marked IDs are removed or recomputed before bulk copy actions can run
    Evidence: .sisyphus/evidence/task-6-history-multiselect-error.txt
  ```

  **Commit**: NO | Message: `handled in task 7 feature commit` | Files: `vid_to_sub_app/tui/app.py`, `vid_to_sub_app/tui/mixins/history_mixin.py`, `tests/test_tui_helpers.py`

- [ ] 7. Implement bulk subtitle copy from marked History rows into one destination folder with duplicate renaming

  **What to do**: Add a bulk copy History action that reads marked rows, validates the chosen destination directory, uses the normalized history artifact helper to gather existing `.srt` and `.vtt` files, copies them into the one chosen folder, and resolves filename collisions as `name (1).ext`, `name (2).ext`, and so on. Return and display a summary covering copied, renamed, skipped-missing, skipped-invalid, and permission-failure counts. Allow partial success when some files fail.
  **Must NOT do**: Do not copy `.txt`, `.tsv`, or `.json` artifacts, do not overwrite existing files in place, and do not abort the whole batch because one file cannot be copied.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: filesystem-safe batching, duplicate handling, and user-visible summary logic.
  - Skills: `[]` - no extra skill required.
  - Omitted: `release-readiness` - no deployment-facing change.

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: F1-F4 | Blocked By: 3, 5, 6

  **References**:
  - Pattern: `vid_to_sub_app/tui/app.py` - history button dispatch and visible status updates.
  - Pattern: `vid_to_sub_app/tui/mixins/history_mixin.py` - bulk action entry points and history selection state.
  - Pattern: `vid_to_sub_app/tui/mixins/setup_mixin.py` - existing `shutil.copy2` usage pattern already present in the TUI codebase.
  - Pattern: `vid_to_sub_app/tui/mixins/browse_mixin.py` - existing directory-selection interaction to mirror for destination selection.
  - API/Type: `vid_to_sub_app/db.py` - source job metadata and `output_paths`.
  - Test: `tests/test_tui_helpers.py` - best location for batch copy regression tests.

  **Acceptance Criteria**:
  - [ ] Bulk copy runs only when at least one history row is marked and the destination directory is valid.
  - [ ] Existing `.srt` and `.vtt` files copy into the chosen folder and rename deterministically on collisions.
  - [ ] Missing, invalid, duplicate, and permission-failure cases are summarized without aborting valid copies.
  - [ ] `python -m unittest tests.test_tui_helpers` exits `0` with new coverage for duplicate renaming and partial failures.

  **QA Scenarios**:
  ```text
  Scenario: marked rows copy subtitle files and rename collisions
    Tool: Bash
    Steps: Add copy tests in `tests/test_tui_helpers.py` for marked rows whose outputs include two subtitle files with the same basename, then run `python -m unittest tests.test_tui_helpers`
    Expected: The suite passes and proves the destination folder receives `name.ext` and `name (1).ext` with accurate copied and renamed counts
    Evidence: .sisyphus/evidence/task-7-history-copy.txt

  Scenario: partial failure still copies valid subtitles
    Tool: Bash
    Steps: Add tests for one missing subtitle, one invalid metadata entry, and one valid subtitle file, then run `python -m unittest tests.test_tui_helpers`
    Expected: The suite passes and proves the valid file copies while skipped and failed items are counted and reported without aborting the batch
    Evidence: .sisyphus/evidence/task-7-history-copy-error.txt
  ```

  **Commit**: YES | Message: `feat(history): add bulk subtitle copy from marked jobs` | Files: `vid_to_sub_app/tui/app.py`, `vid_to_sub_app/tui/mixins/history_mixin.py`, `tests/test_tui_helpers.py`

## Final Verification Wave (MANDATORY - after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.
> Never mark F1-F4 as checked before getting user's okay. Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.

- [ ] F1. Plan Compliance Audit

  **What to do**: Run a plan-compliance review against the completed implementation, using the plan file, changed files, task evidence, and test outputs as inputs. Verify that every numbered task was completed within scope, that `.srt`-only history translation and `.srt`/`.vtt`-only bulk copy rules were honored, and that no out-of-plan shortcuts were taken.
  **Must NOT do**: Do not approve based only on green tests; this review must compare implementation behavior to the plan.

  **Recommended Agent Profile**:
  - Category: `oracle` - Reason: highest-rigor compliance audit against the written plan.
  - Skills: `[]` - no extra skill required.
  - Omitted: `release-readiness` - release gating is not the goal of this audit.

  **Parallelization**: Can Parallel: YES | Final Verification | Blocks: completion | Blocked By: 1, 2, 3, 4, 5, 6, 7

  **Acceptance Criteria**:
  - [ ] Oracle reports PASS or PASS WITH KNOWN RISK with no unresolved mismatch against Tasks 1-7.
  - [ ] Any cited mismatch is fixed before the final user-facing verification bundle is presented.
  - [ ] Evidence is saved to `.sisyphus/evidence/f1-plan-compliance.txt`.

  **QA Scenarios**:
  ```text
  Scenario: oracle audits implementation against the plan
    Tool: task
    Steps: Invoke `task(subagent_type="oracle", load_skills=[], run_in_background=false, prompt="Audit the completed implementation against .sisyphus/plans/translation-history-subtitle-copy.md. Review changed files, unittest output, and .sisyphus/evidence/task-*.txt. Return PASS, PASS WITH KNOWN RISK, BLOCKED, or FAIL with exact mismatches.")`
    Expected: Oracle returns PASS or PASS WITH KNOWN RISK and any mismatch is either absent or explicitly fixed before completion
    Evidence: .sisyphus/evidence/f1-plan-compliance.txt
  ```

- [ ] F2. Code Quality Review

  **What to do**: Run an independent code-quality review over the changed implementation with emphasis on state handling, helper extraction, duplicate rename correctness, error summary quality, and regression risk in the existing History and Settings flows.
  **Must NOT do**: Do not limit this review to style nits; it must cover correctness, maintainability, and failure handling.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: broad high-effort review of correctness and maintainability.
  - Skills: `[]` - no extra skill required.
  - Omitted: `schema-evolution` - schema work is not central to this change set.

  **Parallelization**: Can Parallel: YES | Final Verification | Blocks: completion | Blocked By: 1, 2, 3, 4, 5, 6, 7

  **Acceptance Criteria**:
  - [ ] Reviewer reports no unresolved defect in state sync, history action behavior, or duplicate rename logic.
  - [ ] Reviewer calls out any risky edge case, and the implementation is fixed or documented before final sign-off.
  - [ ] Evidence is saved to `.sisyphus/evidence/f2-code-quality.txt`.

  **QA Scenarios**:
  ```text
  Scenario: high-effort reviewer inspects changed implementation
    Tool: task
    Steps: Invoke `task(category="unspecified-high", load_skills=[], run_in_background=false, prompt="Review the completed changes for translation defaults, history subtitle translation, and bulk subtitle copy. Focus on state sync, malformed output_paths handling, duplicate renaming, stale selection cleanup, and user-visible summaries. Return PASS, PASS WITH KNOWN RISK, BLOCKED, or FAIL with exact issues.")`
    Expected: Reviewer returns PASS or PASS WITH KNOWN RISK after any code-quality issues are fixed or explicitly documented
    Evidence: .sisyphus/evidence/f2-code-quality.txt
  ```

- [ ] F3. Real Manual QA

  **What to do**: Run an agent-executed QA pass that exercises the actual user workflows after implementation: default translation on for a new state, safe auto-disable when config is missing, history translation from completed `.srt` output, and bulk copy from marked rows into a destination with rename collisions.
  **Must NOT do**: Do not treat unit tests as a substitute for this workflow verification.

  **Recommended Agent Profile**:
  - Category: `unspecified-high` - Reason: end-to-end behavior verification across TUI state and filesystem side effects.
  - Skills: `[]` - no extra skill required.
  - Omitted: `orchestration-contract` - this is behavior verification, not planning.

  **Parallelization**: Can Parallel: YES | Final Verification | Blocks: completion | Blocked By: 1, 2, 3, 4, 5, 6, 7

  **Acceptance Criteria**:
  - [ ] QA confirms the implemented workflows behave as the plan specifies in actual app/test execution.
  - [ ] Missing config, missing subtitle files, and duplicate rename cases are explicitly exercised.
  - [ ] Evidence is saved to `.sisyphus/evidence/f3-manual-qa.txt`.

  **QA Scenarios**:
  ```text
  Scenario: agent executes final workflow QA
    Tool: task
    Steps: Invoke `task(category="unspecified-high", load_skills=[], run_in_background=false, prompt="Run final QA for the completed implementation. Verify: default translation starts on for fresh state; missing translation config causes safe auto-disable with warning; completed history rows can translate existing .srt outputs; marked rows bulk-copy .srt/.vtt files into one folder with name (1).ext collision handling. Use available commands and test hooks, and return PASS, PASS WITH KNOWN RISK, BLOCKED, or FAIL with evidence notes.")`
    Expected: Reviewer returns PASS or PASS WITH KNOWN RISK after directly exercising all required workflows and edge cases
    Evidence: .sisyphus/evidence/f3-manual-qa.txt
  ```

- [ ] F4. Scope Fidelity Check

  **What to do**: Run a final scope audit to ensure the change set stayed within the requested boundaries: translation defaults, history translation from existing subtitle artifacts, missing-file handling, and bulk subtitle copy only. Verify that unrelated transcription, distributed execution, and deletion semantics were not changed.
  **Must NOT do**: Do not approve if unrelated modules or behavior changed without explicit justification in the plan.

  **Recommended Agent Profile**:
  - Category: `deep` - Reason: slower, evidence-heavy scope and side-effect audit.
  - Skills: `[]` - no extra skill required.
  - Omitted: `release-readiness` - this is scope verification rather than operational release review.

  **Parallelization**: Can Parallel: YES | Final Verification | Blocks: completion | Blocked By: 1, 2, 3, 4, 5, 6, 7

  **Acceptance Criteria**:
  - [ ] Reviewer confirms no unrelated scope expansion or silent behavior drift remains.
  - [ ] Any out-of-scope change is reverted or explicitly justified before final completion.
  - [ ] Evidence is saved to `.sisyphus/evidence/f4-scope-fidelity.txt`.

  **QA Scenarios**:
  ```text
  Scenario: deep reviewer checks scope boundaries
    Tool: task
    Steps: Invoke `task(category="deep", load_skills=[], run_in_background=false, prompt="Audit the completed implementation for scope fidelity against .sisyphus/plans/translation-history-subtitle-copy.md. Confirm only translation defaults, history subtitle translation, missing-file handling, and bulk subtitle copy changed. Flag unrelated behavior drift in transcription, distributed execution, deletion semantics, or agent features. Return PASS, PASS WITH KNOWN RISK, BLOCKED, or FAIL with exact scope violations.")`
    Expected: Reviewer returns PASS or PASS WITH KNOWN RISK only when no unapproved scope drift remains
    Evidence: .sisyphus/evidence/f4-scope-fidelity.txt
  ```

## Commit Strategy
- Commit 1: `feat(tui): default translation on with safe auto-disable`
- Commit 2: `feat(history): add direct translation from completed subtitles`
- Commit 3: `feat(history): add bulk subtitle copy from marked jobs`

## Success Criteria
- Users see translation enabled by default for new work without being forced into predictable missing-config failures.
- Completed history rows support direct translation from persisted subtitle artifacts.
- Selected history rows can bulk-copy existing subtitle files to one chosen folder with deterministic rename-on-collision behavior.
- Missing subtitle files and malformed history metadata are reported clearly without causing unrelated selected files to fail.
- Regression coverage exists for default sync, history translation, and bulk copy behavior.

## Handoff to Sisyphus
- Artifacts:
  - `.sisyphus/plans/translation-history-subtitle-copy.md`
  - `.sisyphus/drafts/translation-history-subtitle-copy.md` until cleanup
  - implementation evidence under `.sisyphus/evidence/`
- Entry conditions:
  - Use the existing TUI architecture in `vid_to_sub_app/tui/app.py`, `vid_to_sub_app/tui/mixins/settings_mixin.py`, `vid_to_sub_app/tui/mixins/history_mixin.py`, `vid_to_sub_app/tui/mixins/run_mixin.py`, and `vid_to_sub_app/tui/mixins/agent_mixin.py`.
  - Treat `vid_to_sub_app/db.py` job records and persisted `output_paths` as the source of truth for history actions.
- Unresolved assumption / blocker:
  - None blocking; the plan fixes copy mode and test strategy, and defaults translation capability to existing translation config resolution rules.
- Verification request:
  - Verify every acceptance criterion with `unittest`, capture evidence files per task, and do not mark final verification complete without explicit user approval.
