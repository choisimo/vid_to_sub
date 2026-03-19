# Transcription Translation Batch Architecture

scope_id: `translation-memory-oom`
plan_id: `transcription-translation-batch-architecture`
phase_id: `planning`
contract_version: `1.0`
decision_log_version: `1.0`
risk_register_version: `1.0`

## TL;DR
> **Summary**: Split the current inline transcription and translation flow into two explicit stages with a file-backed handoff artifact so GPU-heavy ASR completes before any translation or postprocess HTTP work starts.
> **Deliverables**:
> - Stage-1 transcription artifact contract
> - Stage-2 translation execution path and rerun policy
> - Stage-aware CLI/TUI/history/distributed behavior
> - Verification and rollback gates
> **Effort**: Large
> **Parallel**: YES - 2 waves
> **Critical Path**: Task 1 -> Task 2 -> Task 4 -> Task 6 -> Task 8

## Context
### Original Request
Explain why translation appeared to hit OOM, clarify GPU versus system-memory ownership, and provide a concrete batch architecture so translation no longer runs inline with ASR.

### Interview Summary
- The observed `retrying with 'distill-large-v3'` signal points to local transcription fallback behavior, not proof that subtitle translation itself ran on the same GPU process.
- HTTP translation and agent calls are out-of-process from this app, but the app cannot guarantee those endpoints are CPU-only if they are deployed on the same host with GPU access.
- The preferred architecture is a hard stage boundary with manifest/file-backed handoff rather than an immediate DB-first queue redesign.

### Metis Review (gaps addressed)
- Define exact stage-completion semantics before any history/UI changes.
- Make rerun behavior explicit for translation outputs and handoff artifacts.
- Keep CPU-only enforcement out of app guarantees; treat it as deployment policy documented in the plan.
- Add acceptance criteria for distributed execution, idempotency, and rollback.

## Work Objectives
### Core Objective
Create a decision-complete implementation plan for refactoring `vid_to_sub` into a two-stage batch pipeline where Stage 1 performs local transcription only, emits a durable handoff artifact, and Stage 2 performs translation and optional postprocess work from that artifact without re-entering ASR.

### Deliverables
- New stage contract covering artifact schema, versioning, and file locations.
- CLI execution changes for `transcribe-only`, `translate-from-artifact`, and full orchestrated two-stage runs.
- TUI/distributed execution behavior for scheduling Stage 2 onto CPU-only or non-ASR workers.
- Stage-aware history/progress semantics and rerun policy.
- Tests, QA scenarios, rollback triggers, and evidence requirements.

### Definition of Done (verifiable conditions with commands)
- `python -m pytest tests/test_translation_pipeline.py tests/test_tui_helpers.py` passes with new stage-aware coverage.
- A dry-run or preview command shows Stage 1 and Stage 2 separately instead of one inline pipeline.
- A translation-only rerun can consume a Stage-1 artifact without invoking any transcription backend.
- Distributed preview shows Stage 2 plans can target executor groups that do not require ASR/GPU capabilities.

### Must Have
- One durable handoff artifact per source job containing transcription outputs, metadata, and translation-ready segment payload.
- Translation stage never calls transcription code paths when launched from artifact.
- Clear policy for overwrite, skip, resume, and partial failure handling.
- Stage-aware progress/history visible to both CLI and TUI layers.
- CPU-only guidance documented as runtime/deployment policy, not as a false in-app guarantee.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- No claim that HTTP equals CPU-only by itself.
- No hidden fallback from Stage 2 back into ASR.
- No mandatory database migration unless file-backed history extension proves insufficient.
- No scope creep into model-quality tuning, prompt optimization, or container orchestration implementation.
- No breaking change to existing single-stage transcription-only runs.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: tests-after using existing `pytest` coverage, extended with stage-aware cases.
- QA policy: Every task includes agent-executed happy-path and failure-path scenarios.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: contract + runner seam + CLI surface + distributed planning foundation.
Wave 2: persistence/history + tests + docs/help text + rollout/rollback hardening.

### Dependency Matrix (full, all tasks)
- Task 1 blocks Tasks 2-8.
- Task 2 blocks Tasks 3-6.
- Task 3 blocks Task 6.
- Task 4 blocks Tasks 5-6.
- Task 5 blocks Task 8.
- Task 6 blocks Task 8.
- Task 7 depends on Tasks 2 and 4.
- Task 8 depends on Tasks 5, 6, and 7.

### Agent Dispatch Summary (wave -> task count -> categories)
- Wave 1 -> 4 tasks -> `architecture-review`, `business-logic`, `safe-refactor`, `schema-evolution`
- Wave 2 -> 4 tasks -> `business-logic`, `writing`, `release-gate`, `schema-evolution`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [ ] 1. Define Stage Artifact Contract

  **What to do**: Introduce one versioned transcription handoff artifact produced after Stage 1 and consumed by Stage 2. The artifact must live beside other batch outputs or in the selected output directory, contain source path, output base path, source media fingerprint or stable identity, transcription backend/device/model metadata, segment payload needed by `translate_segments_openai_compatible(...)`, original timing-preserving subtitle content, and stage status fields for `transcription_complete`, `translation_pending`, `translation_complete`, and `translation_failed`. Choose JSON as the initial format because current translation payloads are already JSON-shaped in `vid_to_sub_app/cli/translation.py`, and keep schema migration simple with an explicit `schema_version` field.
  **Must NOT do**: Do not introduce a database-first queue as the primary contract, do not make Stage 2 depend on in-memory objects from `process_one()`, and do not encode host-specific absolute paths without preserving path remap compatibility.

  **Recommended Agent Profile**:
  - Category: `schema-evolution` - Reason: define a rollback-safe file schema with explicit versioning and forward compatibility.
  - Skills: [`schema-evolution`, `orchestration-contract`] - why needed: schema versioning plus handoff contract discipline.
  - Omitted: [`release-readiness`] - why not needed: operational gating comes later.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [2, 3, 4, 5, 6, 7, 8] | Blocked By: []

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/cli/manifest.py:17` - existing manifest builder shows file-backed batch orchestration style already used in this repo.
  - Pattern: `vid_to_sub_app/cli/manifest.py:179` - manifest stdin loader shows normalization/validation seam to mirror for artifact loading.
  - Pattern: `vid_to_sub_app/cli/translation.py:76` - translation request payload is already JSON-oriented.
  - Pattern: `vid_to_sub_app/cli/translation.py:105` - translation batching operates over serializable subtitle items.
  - API/Type: `vid_to_sub_app/cli/output.py` - output naming/location rules must remain compatible.
  - API/Type: `vid_to_sub_app/shared/constants.py` - reuse existing constant organization for schema/version constants.
  - Test: `tests/test_translation_pipeline.py` - extend request-shape and no-ASR translation coverage here.

  **Acceptance Criteria** (agent-executable only):
  - [ ] Repository contains one documented artifact schema constant/module and one loader/writer implementation with version validation.
  - [ ] Stage-1 output includes enough data for translation-only execution without reopening source media.
  - [ ] Invalid or unsupported artifact versions fail fast with a deterministic error.
  - [ ] `python -m pytest tests/test_translation_pipeline.py -k artifact` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Stage-1 artifact written successfully
    Tool: Bash
    Steps: Run the new stage-1-only command against a small fixture video; inspect output directory for artifact JSON and primary subtitle file.
    Expected: Artifact exists, includes schema_version and stage status fields, and references the produced transcription outputs.
    Evidence: .sisyphus/evidence/task-1-stage-artifact.txt

  Scenario: Unsupported artifact version rejected
    Tool: Bash
    Steps: Copy a valid artifact, change schema_version to an unsupported value, then run the new stage-2 command against it.
    Expected: Command exits non-zero with a clear unsupported-version error before any translation HTTP request is attempted.
    Evidence: .sisyphus/evidence/task-1-stage-artifact-error.txt
  ```

  **Commit**: YES | Message: `feat(pipeline): add transcription handoff artifact contract` | Files: [`vid_to_sub_app/cli/manifest.py`, `vid_to_sub_app/shared/constants.py`, `tests/test_translation_pipeline.py`]

- [ ] 2. Refactor Runner Into Explicit Stage Boundaries

  **What to do**: Replace the current inline `process_one()` behavior with explicit stage orchestration so transcription completion writes the artifact and returns without automatically entering translation code. Add an orchestrator that can run `stage1 -> stage2` sequentially when requested, but only by invoking a separate translation-stage entrypoint that loads the artifact. Preserve current transcription-only behavior as the default when `--translate-to` is absent.
  **Must NOT do**: Do not leave any hidden translation call inside the Stage-1 completion path, do not break existing output naming, and do not remove current ASR fallback logic in `vid_to_sub_app/cli/transcription.py`.

  **Recommended Agent Profile**:
  - Category: `safe-refactor` - Reason: split a hot path without breaking current transcription behavior.
  - Skills: [`orchestration-contract`] - why needed: maintain a clear handoff boundary between stages.
  - Omitted: [`schema-evolution`] - why not needed: contract is defined in Task 1.

  **Parallelization**: Can Parallel: NO | Wave 1 | Blocks: [3, 5, 6, 7, 8] | Blocked By: [1]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/cli/runner.py:49` - current single-job orchestration lives here.
  - Pattern: `vid_to_sub_app/cli/runner.py:137` - inline translation gate that must be removed from Stage 1.
  - Pattern: `vid_to_sub_app/cli/runner.py:146` - postprocess stage currently chained inline and must move behind artifact-driven Stage 2.
  - Pattern: `vid_to_sub_app/cli/runner.py:289` - local worker pool semantics must still function after stage split.
  - API/Type: `vid_to_sub_app/cli/transcription.py` - ASR implementations remain Stage 1 only.
  - API/Type: `vid_to_sub_app/cli/translation.py` - translation/postprocess remain Stage 2 only.
  - Test: `tests/test_translation_pipeline.py` - add translation-from-artifact no-ASR assertions.

  **Acceptance Criteria** (agent-executable only):
  - [ ] No Stage-2 translation function is reachable from the Stage-1-only code path.
  - [ ] Sequential full-run mode executes Stage 1 then Stage 2 via artifact reload rather than sharing live segment state.
  - [ ] Transcription-only runs still succeed without artifact-consumer requirements.
  - [ ] `python -m pytest tests/test_translation_pipeline.py -k stage` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Full pipeline uses stage boundary
    Tool: Bash
    Steps: Run the new full two-stage command with translation enabled and capture logs.
    Expected: Logs show Stage 1 completion, artifact write, then a distinct Stage 2 startup that loads the artifact before translation begins.
    Evidence: .sisyphus/evidence/task-2-runner-boundary.txt

  Scenario: Stage-1-only run never enters translation
    Tool: Bash
    Steps: Run the stage-1-only command with translation env vars present but no stage-2 invocation.
    Expected: Primary subtitles and artifact are produced, and logs contain no translation HTTP request or postprocess call.
    Evidence: .sisyphus/evidence/task-2-runner-boundary-error.txt
  ```

  **Commit**: YES | Message: `refactor(runner): separate transcription and translation stages` | Files: [`vid_to_sub_app/cli/runner.py`, `vid_to_sub_app/cli/transcription.py`, `vid_to_sub_app/cli/translation.py`, `tests/test_translation_pipeline.py`]

- [ ] 3. Add Stage-Aware CLI Surface And Preview Semantics

  **What to do**: Extend CLI arguments so operators can run Stage 1 only, Stage 2 only from artifact, or the existing user-facing "do both" path that now orchestrates the two stages explicitly. Add clear preview/dry-run output describing stage count, stage order, and artifact locations. Validate incompatible flag combinations early, including `--postprocess-translation` without translation stage input.
  **Must NOT do**: Do not silently reinterpret ambiguous flags, do not require users to pass low-level internal stage identifiers for normal runs, and do not break existing `--translate-to` semantics for users who still want one command.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: CLI contract and validation rules materially affect operator behavior.
  - Skills: [`orchestration-contract`] - why needed: keep flags aligned with stage contracts and handoff expectations.
  - Omitted: [`release-readiness`] - why not needed: release gating is later.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [7, 8] | Blocked By: [1, 2]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/cli/main.py` - CLI argument parsing and validation live here.
  - Pattern: `vid_to_sub_app/cli/main.py:111` - existing translation/postprocess validation must be extended, not bypassed.
  - Pattern: `vid_to_sub_app/cli/manifest.py:179` - reuse manifest-style loading error messages for artifact input.
  - API/Type: `README.md` - CLI examples will need matching help text.
  - Test: `tests/test_translation_pipeline.py` - add argument validation and preview expectations.

  **Acceptance Criteria** (agent-executable only):
  - [ ] CLI exposes one stage-1-only path, one artifact-driven stage-2-only path, and one combined orchestrated path.
  - [ ] `--dry-run` or equivalent preview clearly distinguishes stage execution and artifact reuse.
  - [ ] Invalid flag combinations exit before any work starts.
  - [ ] `python -m pytest tests/test_translation_pipeline.py -k cli` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Preview shows explicit stages
    Tool: Bash
    Steps: Run the new dry-run command with translation enabled.
    Expected: Output lists Stage 1 transcription, artifact path, and Stage 2 translation as separate steps.
    Evidence: .sisyphus/evidence/task-3-cli-preview.txt

  Scenario: Invalid flag combination rejected early
    Tool: Bash
    Steps: Run the CLI with stage-2-only input omitted but postprocess-only flags enabled.
    Expected: Command exits non-zero with a clear validation error and no output files are created.
    Evidence: .sisyphus/evidence/task-3-cli-preview-error.txt
  ```

  **Commit**: YES | Message: `feat(cli): add stage-aware pipeline commands` | Files: [`vid_to_sub_app/cli/main.py`, `vid_to_sub_app/cli/runner.py`, `README.md`, `tests/test_translation_pipeline.py`]

- [ ] 4. Extend Distributed Planning For Stage-Specific Scheduling

  **What to do**: Teach executor planning to distinguish ASR-capable Stage 1 work from translation-only Stage 2 work. Stage 2 must be schedulable onto executors that have no GPU/ASR capability requirement, while still allowing operators to target the same host when desired. Reuse manifest and `ExecutorPlan` patterns so distributed mode can emit separate per-stage command plans, path remaps, and environment injection.
  **Must NOT do**: Do not claim hard CPU-only enforcement unless the remote profile actually sets external runtime controls, do not assume localhost endpoints are safe, and do not require a GPU profile for translation-only plans.

  **Recommended Agent Profile**:
  - Category: `architecture-review` - Reason: this changes scheduling semantics across local/distributed execution.
  - Skills: [`orchestration-contract`, `release-readiness`] - why needed: cross-agent command contract plus resource-boundary guardrails.
  - Omitted: [`schema-evolution`] - why not needed: artifact contract already decided.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [5, 6, 8] | Blocked By: [1]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/tui/models.py:125` - remote resource profile contract starts here.
  - Pattern: `vid_to_sub_app/tui/models.py:163` - `ExecutorPlan` should carry stage-aware dispatch data.
  - Pattern: `vid_to_sub_app/tui/mixins/run_mixin.py` - distributed command planning and env injection already live here.
  - Pattern: `vid_to_sub_app/cli/manifest.py:75` - path remap logic must remain valid when Stage 2 uses artifact paths.
  - API/Type: `vid_to_sub_app/tui/helpers.py` - profile parsing and capability defaults should be updated consistently.
  - Test: `tests/test_tui_helpers.py` - add stage-group assignment and non-GPU Stage 2 expectations.

  **Acceptance Criteria** (agent-executable only):
  - [ ] Distributed preview can emit separate Stage 1 and Stage 2 plans.
  - [ ] Stage 2 can target profiles without GPU assumptions.
  - [ ] Artifact path remapping works for remote Stage 2 execution.
  - [ ] `python -m pytest tests/test_tui_helpers.py -k stage` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Distributed preview separates stage executors
    Tool: Bash
    Steps: Configure one GPU profile and one CPU-only profile, then run distributed dry-run with translation enabled.
    Expected: Preview assigns Stage 1 to the ASR-capable profile and Stage 2 to the CPU-only profile, with artifact path mapping shown.
    Evidence: .sisyphus/evidence/task-4-distributed-stage-planning.txt

  Scenario: Misconfigured Stage-2 executor fails clearly
    Tool: Bash
    Steps: Run distributed stage-2 preview with a profile missing required translation endpoint env vars.
    Expected: Planning fails with a configuration error before remote execution starts.
    Evidence: .sisyphus/evidence/task-4-distributed-stage-planning-error.txt
  ```

  **Commit**: YES | Message: `feat(distributed): schedule translation as separate stage` | Files: [`vid_to_sub_app/tui/models.py`, `vid_to_sub_app/tui/mixins/run_mixin.py`, `vid_to_sub_app/tui/helpers.py`, `tests/test_tui_helpers.py`]

- [ ] 5. Add Stage-Aware Persistence And History Semantics

  **What to do**: Extend job/history tracking so a single logical run can expose stage sub-statuses without forcing an immediate schema rewrite unless necessary. Preferred default: keep one logical job record and add stage progress/state storage through existing manifest or settings-friendly persistence paths first; only add DB columns or related tables if code inspection during implementation proves the current history UI cannot represent stage state without it. Record `pending`, `running`, `complete`, `failed`, and `skipped` per stage, plus artifact path and translation rerun provenance.
  **Must NOT do**: Do not lose current History tab compatibility, do not create duplicate top-level jobs for one user-triggered run by default, and do not require manual cleanup of stale stage state after failures.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: status semantics and rerun policy affect correctness and operator trust.
  - Skills: [`schema-evolution`, `orchestration-contract`] - why needed: only if persistence extension becomes necessary, keep it rollback-safe.
  - Omitted: [`release-readiness`] - why not needed: operational review lands in Task 8.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8] | Blocked By: [1, 4]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/db.py:203` - current job creation starts here.
  - API/Type: `vid_to_sub_app/db.py` - current history schema is coarse and must remain backward compatible.
  - Pattern: `vid_to_sub_app/cli/manifest.py:195` - existing manifest-state persistence may cover stage snapshots without immediate DB migration.
  - API/Type: `vid_to_sub_app/tui/app.py` - History tab rendering must not regress.
  - API/Type: `vid_to_sub_app/tui/mixins/run_mixin.py` - run status updates need stage detail.
  - Test: `tests/test_tui_helpers.py` - extend history/status projection checks if helpers participate.

  **Acceptance Criteria** (agent-executable only):
  - [ ] One user-triggered run can expose distinct Stage 1 and Stage 2 statuses.
  - [ ] Translation-only reruns are represented without fabricating a new transcription completion event.
  - [ ] Failed Stage 2 state preserves a reusable artifact path for retry.
  - [ ] Relevant `pytest` coverage passes for status/history behavior.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: History shows stage progression
    Tool: Bash
    Steps: Execute a full run through Stage 1 and Stage 2, then inspect persisted status data or History rendering output.
    Expected: One logical run shows Stage 1 complete and Stage 2 complete with artifact path retained.
    Evidence: .sisyphus/evidence/task-5-history-stage-status.txt

  Scenario: Stage-2 failure remains retryable
    Tool: Bash
    Steps: Force translation endpoint failure during Stage 2 after Stage 1 artifact creation.
    Expected: Status shows Stage 1 complete and Stage 2 failed, with enough persisted state to retry Stage 2 only.
    Evidence: .sisyphus/evidence/task-5-history-stage-status-error.txt
  ```

  **Commit**: YES | Message: `feat(history): track stage-aware run status` | Files: [`vid_to_sub_app/db.py`, `vid_to_sub_app/tui/app.py`, `vid_to_sub_app/tui/mixins/run_mixin.py`, `tests/test_tui_helpers.py`]

- [ ] 6. Implement Translation-Only Stage Execution And Idempotency Rules

  **What to do**: Add a dedicated Stage-2 execution path that loads the artifact, reconstructs translation batches, writes translated outputs, optionally runs postprocess, and updates artifact/job stage state. Define exact overwrite semantics: if translated outputs already exist, default to skip unless the operator explicitly requests overwrite or rerun; if Stage 2 partially fails, successful translated files remain valid and the artifact records which formats completed. Ensure Stage 2 never calls source discovery or ASR backend resolution.
  **Must NOT do**: Do not reopen media files for content extraction during Stage 2, do not mutate Stage-1 timing data, and do not overwrite translated outputs silently.

  **Recommended Agent Profile**:
  - Category: `business-logic` - Reason: rerun/overwrite semantics and partial-failure behavior are correctness-critical.
  - Skills: [`release-readiness`] - why needed: task must define rollback and partial-failure handling.
  - Omitted: [`schema-evolution`] - why not needed: uses contract from Task 1.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [8] | Blocked By: [1, 2, 4]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/cli/translation.py` - all translation and postprocess HTTP logic belongs here or behind a Stage-2 entrypoint.
  - Pattern: `vid_to_sub_app/cli/translation.py:155` - existing chunk-size defaults must still apply when Stage 2 is artifact-driven.
  - Pattern: `vid_to_sub_app/cli/translation.py:220` - keep postprocess batching compatible with current behavior.
  - Pattern: `vid_to_sub_app/cli/runner.py:171` - preserve translated filename suffix rules.
  - API/Type: `vid_to_sub_app/cli/output.py` - output write behavior and overwrite checks must stay centralized.
  - Test: `tests/test_translation_pipeline.py` - add skip/overwrite/retry coverage here.

  **Acceptance Criteria** (agent-executable only):
  - [ ] Stage 2 can run from artifact alone and produce translated outputs with original timing preserved.
  - [ ] Default behavior skips existing translated outputs unless explicit overwrite/rerun is set.
  - [ ] Partial Stage-2 failure preserves retryable state and does not mark transcription incomplete.
  - [ ] `python -m pytest tests/test_translation_pipeline.py -k rerun` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Translation-only rerun succeeds without ASR
    Tool: Bash
    Steps: Produce a Stage-1 artifact, then run the stage-2-only command while instrumenting logs for backend selection.
    Expected: Translated subtitle file is written and logs show no ASR backend resolution or media transcription call.
    Evidence: .sisyphus/evidence/task-6-translation-only-rerun.txt

  Scenario: Existing translation skipped unless overwrite requested
    Tool: Bash
    Steps: Run Stage 2 twice against the same artifact, first normally and then again without overwrite.
    Expected: Second run exits cleanly with a skip message and does not rewrite translated outputs; explicit overwrite reruns when requested.
    Evidence: .sisyphus/evidence/task-6-translation-only-rerun-error.txt
  ```

  **Commit**: YES | Message: `feat(translation): support artifact-driven reruns` | Files: [`vid_to_sub_app/cli/translation.py`, `vid_to_sub_app/cli/output.py`, `vid_to_sub_app/cli/runner.py`, `tests/test_translation_pipeline.py`]

- [ ] 7. Update TUI And Operator-Facing Help Text For Stage Runs

  **What to do**: Surface the new stage model in CLI help text, README examples, and TUI labels/previews so operators understand that full runs now orchestrate two stages and that translation-only execution consumes an artifact. Where the TUI exposes execution previews or summaries, show which stage uses local ASR resources and which stage depends on translation endpoints/executor selection.
  **Must NOT do**: Do not introduce UI copy that promises GPU isolation the app cannot enforce, and do not overload the TUI with advanced deployment-policy controls that do not exist in code.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: operator correctness depends on accurate guidance and examples.
  - Skills: [`release-readiness`] - why needed: wording must call out operational constraints honestly.
  - Omitted: [`schema-evolution`] - why not needed: no schema work here.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8] | Blocked By: [2, 3]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `README.md` - translation examples and distributed execution notes already live here.
  - API/Type: `vid_to_sub_app/tui/app.py` - visible labels, summaries, and history wording may need updates.
  - API/Type: `vid_to_sub_app/tui/mixins/run_mixin.py` - preview/status strings should align with CLI output.
  - Test: `tests/test_tui_helpers.py` - add assertions only if helper text generation is testable.

  **Acceptance Criteria** (agent-executable only):
  - [ ] README and help text explain the stage boundary and artifact-driven translation reruns.
  - [ ] TUI or preview text distinguishes ASR stage resources from translation endpoint resources.
  - [ ] No user-facing text claims the app itself enforces CPU-only translation inference.
  - [ ] Relevant doc/help tests or snapshot assertions pass if present.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Help text describes stage choices
    Tool: Bash
    Steps: Run the CLI help output and inspect updated README examples.
    Expected: Help and docs show stage-1-only, stage-2-only, and combined flow options with artifact wording.
    Evidence: .sisyphus/evidence/task-7-help-text.txt

  Scenario: Resource-boundary wording stays accurate
    Tool: Bash
    Steps: Search updated docs/help text for CPU-only or GPU-isolation claims.
    Expected: Text states that CPU-only translation is a deployment choice, not an automatic property of HTTP calls.
    Evidence: .sisyphus/evidence/task-7-help-text-error.txt
  ```

  **Commit**: YES | Message: `docs(pipeline): explain stage-based translation flow` | Files: [`README.md`, `vid_to_sub_app/tui/app.py`, `vid_to_sub_app/tui/mixins/run_mixin.py`]

- [ ] 8. Add End-To-End Verification, Rollback Gate, And Release Guardrails

  **What to do**: Add final automated coverage and rollout protections for the new two-stage architecture. Introduce one temporary feature gate or compatibility toggle so the team can revert to the current inline path quickly if stage orchestration causes regressions during rollout. Document failure modes, partial-failure behavior, first rollback trigger, and the minimum evidence set required before enabling the new default.
  **Must NOT do**: Do not ship the new architecture without a rollback path, do not mark release-ready without proving translation-only rerun and distributed preview behavior, and do not remove existing observability or error messages around ASR OOM fallback.

  **Recommended Agent Profile**:
  - Category: `release-gate` - Reason: this task defines rollout safety, evidence, and rollback.
  - Skills: [`release-readiness`, `orchestration-contract`] - why needed: release guardrails plus explicit handoff criteria.
  - Omitted: [`schema-evolution`] - why not needed: no new contract design here.

  **Parallelization**: Can Parallel: NO | Wave 2 | Blocks: [] | Blocked By: [3, 4, 5, 6, 7]

  **References** (executor has NO interview context - be exhaustive):
  - Pattern: `vid_to_sub_app/cli/transcription.py:165` - preserve current OOM fallback visibility in logs.
  - API/Type: `README.md` - rollout/rollback notes belong in operator docs.
  - API/Type: `tests/test_translation_pipeline.py` - final stage-path regression coverage.
  - API/Type: `tests/test_tui_helpers.py` - final distributed/history coverage.

  **Acceptance Criteria** (agent-executable only):
  - [ ] One documented rollback switch restores current inline behavior without reverting unrelated code.
  - [ ] End-to-end automated coverage exercises Stage 1 only, Stage 2 only, combined flow, distributed preview, and failure recovery.
  - [ ] Release notes or operator docs list must-fix-before-release, can-monitor-after-release, fastest rollback path, and first signal to watch.
  - [ ] `python -m pytest tests/test_translation_pipeline.py tests/test_tui_helpers.py` passes.

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Rollback switch restores legacy behavior
    Tool: Bash
    Steps: Enable the compatibility toggle and run a translated job.
    Expected: Execution follows the legacy inline flow, logs that the compatibility path is active, and still preserves current outputs.
    Evidence: .sisyphus/evidence/task-8-rollback-gate.txt

  Scenario: New default survives translation failure cleanly
    Tool: Bash
    Steps: Run the new default two-stage flow while forcing the translation endpoint to fail.
    Expected: Stage 1 completes, Stage 2 fails cleanly, rollback guidance is visible, and retry can start from the artifact.
    Evidence: .sisyphus/evidence/task-8-rollback-gate-error.txt
  ```

  **Commit**: YES | Message: `test(pipeline): add rollout safeguards for staged translation` | Files: [`README.md`, `tests/test_translation_pipeline.py`, `tests/test_tui_helpers.py`, `vid_to_sub_app/cli/runner.py`]


## Final Verification Wave (MANDATORY - after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit - oracle
- [ ] F2. Code Quality Review - unspecified-high
- [ ] F3. Real Manual QA - unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check - deep

## Commit Strategy
- Commit 1: stage contract + CLI runner separation.
- Commit 2: TUI/distributed/history integration.
- Commit 3: tests + docs/help text + rollout safeguards.

## Success Criteria
- OOM explanations in docs/code comments/history messages distinguish ASR GPU fallback from translation HTTP execution.
- Translation reruns are possible without re-transcription.
- Operators can route Stage 2 to CPU-only infrastructure by choosing separate executors/endpoints.
- Existing transcription-only workflows remain intact.
- Rollback path restores current inline behavior behind one feature gate or code-path toggle.

## Handoff to Sisyphus
1. Artifacts:
- `.sisyphus/plans/transcription-translation-batch-architecture.md`
- Existing analysis draft at `.sisyphus/drafts/translation-memory-oom.md`
2. Entry conditions:
- Executor has read this full plan and verified referenced files still exist.
- Any unresolved deployment policy assumptions are treated as external ops work, not app guarantees.
3. Unresolved assumption / blocker:
- None blocking for implementation planning; stage-aware history can remain file-backed first unless a code inspection during implementation proves a DB schema change is unavoidable.
4. Verification request:
- Verify all task references against current repo state before editing code and preserve backward compatibility for transcription-only runs.
