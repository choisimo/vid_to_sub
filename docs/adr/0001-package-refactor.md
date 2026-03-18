# ADR 0001: Package-Oriented Refactor And Bootstrap Wrapper

## Context

- [`/home/nodove/workspace/vid_to_sub/tui.py`](/home/nodove/workspace/vid_to_sub/tui.py) and [`/home/nodove/workspace/vid_to_sub/vid_to_sub.py`](/home/nodove/workspace/vid_to_sub/vid_to_sub.py) previously held most runtime logic in single files.
- `tui.py` imported Textual immediately, so a clean environment could fail before any dependency recovery path existed.
- Shared constants, env parsing, model discovery, and persistence concerns were duplicated or tightly coupled across the CLI and TUI.

## Decision

- Introduce a package root at `vid_to_sub_app/` and move implementation into package modules.
- Keep `tui.py`, `vid_to_sub.py`, and `db.py` as compatibility entrypoints/wrappers so existing user commands remain unchanged.
- Add `init_checker.py` plus `init-checker.py` as a stdlib-only bootstrap layer that:
  - ensures a project-local `.venv`,
  - installs missing requirement groups,
  - relaunches the requested entrypoint under the managed interpreter.
- Split CLI logic into discovery, manifest, output, transcription, translation, runner, and main modules.
- Split TUI support code into package state/helpers/styles, with `app.py` retaining the Textual application class.

## Alternatives Considered

- Keep monolith files and only add a bootstrap wrapper.
  - Rejected because it would not materially improve maintainability or module boundaries.
- Full TUI mixin decomposition in one step.
  - Rejected for now because it would create a much larger migration surface and higher regression risk in a single pass.

## Consequences

- Positive:
  - `python tui.py` becomes the only required user entrypoint.
  - Shared constants/env behavior now have a single source of truth.
  - Future refactors can target smaller package modules instead of editing multi-thousand-line files directly.
- Trade-off:
  - Compatibility wrappers now forward to package modules, so tests and mocks should target implementation modules where behavior matters.
  - Optional backend bootstrap may take longer on first run because requirement installation is automated.

## Rollback

- The rollback boundary is the wrapper layer:
  - restore the previous monolith entrypoints,
  - remove `vid_to_sub_app/`,
  - remove `init_checker.py` and `init-checker.py`.
- Data rollback is not required because SQLite storage remains at the original project root.
