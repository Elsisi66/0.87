# Phase J03 Mapping To Existing Engine

## Reused modules
- Route generation: `scripts.phase_d123_tail_filter.evaluate_baseline_routes()`
- Feature transforms and policy sizing: `scripts.phase_af_ah_sizing_autorun.py`
- Contract lock: `src.execution.ga_exec_3m_opt._validate_and_lock_frozen_artifacts()`

## Sizing-only scope
- All policies map to sizing functions already available (`step`, `smooth`, `streak`, `hybrid`, `ae_s1_anchor`, `streak_control`).
- `cooldown-lite` is represented as state-triggered size-down (no skip) because hard skip paths were previously fragile.

## Out-of-scope in this branch
- Combined 1h+3m GA.
- Hard-gate changes.
- Execution entry/exit mechanic modifications.
