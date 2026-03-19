# Unified Implementation Plan
# Merged from: translation-history-subtitle-copy + transcription-translation-batch-architecture

scope_id: `unified-vid-to-sub`
plan_id: `unified-implementation`
phase_id: `implementation`
contract_version: `1.0`
decision_log_version: `1.0`
risk_register_version: `1.0`

## Merge Decision Log
- Plan A (translation-history-subtitle-copy): TUI features — default translation on, history translate action, bulk subtitle copy.
- Plan B (transcription-translation-batch-architecture): CLI architecture — stage artifact contract, runner split, distributed scheduling, stage-aware history, idempotent rerun.
- Merged pairs: A-T1+T2 → U2 | A-T3+B-T5 → U6 | A-T7+B-T7 → U10 | A-T8+B-T8 → U11
- Test runner: `python -m pytest tests/` (pytest, not unittest, is the current runner per codebase)

## Verified Current State
- `sw-translate` widget defaults to `False` in `vid_to_sub_app/tui/app.py:472`
- `tui.default_translate_to` defaults to `"ko"` in db.py:131 — target language persisted, not enabled state
- No `tui.default_translate_enabled` key exists yet
- `output_paths` is JSON array of full path strings in `jobs` table
- History actions today: refresh, load, rerun, clear, delete — no translate, no copy
- No stage-split CLI args exist in main.py
- No stage artifact contract exists anywhere

## Critical Path
U1 → U3 → U5 → U9 → U11
U2 → U6 → U7 → U10 → U11
U8 → U10

## TODOs

- [ ] U1: Stage Artifact Contract (schema-evolution)
- [ ] U2: Translation Default On + Capability Guard (business-logic)
- [ ] U3: Runner Stage Split (safe-refactor)
- [ ] U4: Distributed Stage Scheduling (architecture-review)
- [ ] U5: Stage-Aware CLI Surface (business-logic)
- [ ] U6: History Artifact Helper + Stage-Aware Persistence (schema-evolution)
- [ ] U7: History Translate Action (business-logic)
- [ ] U8: History Multi-Select + Bulk Subtitle Copy (visual-engineering)
- [ ] U9: Translation-Only Stage Execution + Idempotency (business-logic)
- [ ] U10: TUI/Help Text + Integration Polish (writing)
- [ ] U11: E2E Verification + Rollback Gate (release-gate)

## Final Verification Wave
- [ ] F1. Plan Compliance Audit - oracle
- [ ] F2. Code Quality Review - unspecified-high
- [ ] F3. Real Manual QA - unspecified-high
- [ ] F4. Scope Fidelity Check - deep
