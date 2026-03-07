# Patch Summary

- File changed: `scripts/phase_i_sol_signal_fork.py`
- Why: add forensic traceability and resolve no-op/metric ambiguity safely without changing contract lock, fee model, metrics hashes, or gates.

Changes:
- Added stage-level gate accounting (`signals_before/after`, `removed_by_*`) to prove parameter effects and detect dead knobs.
- Added explicit CVaR computation helper (`_compute_cvar5_trade_notional`) and vector hash (`trade_return_vector_sha256`) per variant to prove variant-specific inputs.
- Added geometric metric split: legacy field + clean field + `ruin_event_fixed` flag (`_geom_mean_return_with_ruin_flag`).
- Updated ranking to use ruin-aware clean geometric metric while preserving legacy field for compatibility.
- Added optional `--vol-mode-filter` to run tiny focused validation matrices without full candidate cartesian explosion.
