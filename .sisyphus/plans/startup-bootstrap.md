# Startup Bootstrap Fix Plan

## TODOs
- [x] T1 Limit `python tui.py` default bootstrap to core `base` dependencies only.
- [x] T2 Keep optional backend install behavior for explicit setup/optional path and avoid startup-time noise from `whisperx`.
- [x] T3 Add/adjust tests for bootstrap group selection so startup defaults are enforced.
- [x] T4 Verify service launch with `python tui.py` and capture startup log evidence.

## Final Verification Wave
- [x] F1 Verify bootstrap path no longer attempts optional group installation during default launch.
- [x] F2 Verify Textual app imports start without dependency errors.
- [x] F3 Verify optional install failures are still tolerated in optional setup flow.
- [x] F4 Confirm no unintended file scope changes; update plan checkboxes.
