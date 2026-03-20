## 2026-03-20 U2–U9 status audit

- U2 (Translation Default On + Capability Guard): fully implemented. `TUI_DEFAULT_TRANSLATE_ENABLED_KEY` seeded to `"1"` in `vid_to_sub_app/db.py:140`, restored in `_prefill_transcribe` via `settings_mixin.py:134-136`, guarded in `run_mixin.py:182` with `translation_capable()`. No code changes needed.
- U3 (Runner Stage Split): fully implemented. `run_stage1()` at `runner.py:253`, `run_stage2()` at `runner.py:399`, both called from `process_one()` and `main.py` dispatch. `--stage1-only` at `main.py:117`, `--translate-from-artifact` at `main.py:118`. No code changes needed.
- U4 (Distributed Stage Scheduling): flagged as architecture-review scope; existing distributed/SSH mode already present. Not touched.
- U5 (Stage-Aware CLI Surface): confirmed present via `--stage1-only` / `--translate-from-artifact` args and their validation at `main.py:251–273`. No code changes needed.
- U6 (History Artifact Helper + Stage-Aware Persistence): implemented in `history_mixin.py` and `run_mixin.py`; `artifact_path` / `artifact_metadata` persist through `db.finish_job()`.
- U7 (History Translate Action): implemented. `btn-hist-translate` button and `_translate_history_job()` in `history_mixin.py:330`.
- U8 (History Multi-Select + Bulk Subtitle Copy): implemented. `btn-hist-select-mode`, `btn-hist-copy`, `_copy_selected_subtitles()` in `history_mixin.py:421`, `bulk_copy_subtitles()` in `cli/subtitle_copy.py`.
- U9 (Translation-Only Stage Execution + Idempotency): implemented. `translation_complete` short-circuit at `runner.py:450`, `--overwrite-translation` at `main.py:120`.
- U10 (TUI/Help Text + Integration Polish): completed 2026-03-20. Added `help=` strings to all bare `add_argument()` calls in `vid_to_sub_app/cli/main.py`.
- U11 (E2E Verification): `python -m pytest tests/` → 104 passed.

## 2026-03-20 U1 stage artifact contract

- `Database.get_jobs()` in `vid_to_sub_app/db.py` now normalizes additive `artifact_metadata` and backfills `artifact_path` from metadata when older/newer rows only persisted one side of the contract.
- History detail rendering in `vid_to_sub_app/tui/mixins/history_mixin.py` can determine translation stage state from persisted metadata first, which avoids unnecessary file reads and keeps rows readable when artifact files move or disappear.
- Legacy history rows still need the `.stage1.json` scan fallback through `output_paths`, because older rows may have neither `artifact_path` nor `artifact_metadata` persisted.
- Focused regression coverage now lives in `tests/test_tui_helpers.py` for metadata-path backfill, legacy path-only rows, persisted-metadata precedence, and legacy output scan fallback.

## 2026-03-20 settings mixin startup unblock

- The startup crash came from mixin shadowing, not just missing helpers: `SettingsMixin.query_one()` and `SettingsMixin._update_agent_config_status()` overrode working implementations earlier in `VidToSubApp` MRO and raised during `on_mount` settings hydration.
- Safe helper delegation inside `vid_to_sub_app/tui/mixins/settings_mixin.py` is enough to restore startup without touching other mixins, as long as missing widgets keep the existing `NoMatches` default/no-op behavior.
- Smoke verification with `timeout 5s python tui.py` rendered the TUI and exited `124` without any `Traceback`, which confirms the previous startup blocker is cleared.
