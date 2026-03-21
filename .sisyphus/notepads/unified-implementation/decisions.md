## 2026-03-20 U1 stage artifact contract

- Persist `artifact_metadata` as an additive optional JSON column and keep `artifact_path` as a sibling field so existing rows remain readable without a destructive migration.
- Treat persisted `artifact_path` / `artifact_metadata["path"]` as the primary source for history detail display, and only fall back to `.stage1.json` discovery in `output_paths` for legacy rows.
- Treat persisted metadata stage flags as authoritative for history status rendering before attempting to load the artifact file from disk.
- Keep all artifact fields null-safe: rows without metadata still normalize to a minimal path-only metadata dict when `artifact_path` exists, and rows without either field still render via legacy fallback.

## 2026-03-20 settings mixin startup unblock

- Keep the fix local to `vid_to_sub_app/tui/mixins/settings_mixin.py` by replacing stubbed helper methods with safe delegation and local fallbacks instead of changing mixin order or editing other mixins. This preserves the current public behavior while removing the `on_mount` startup hazard caused by MRO shadowing.
