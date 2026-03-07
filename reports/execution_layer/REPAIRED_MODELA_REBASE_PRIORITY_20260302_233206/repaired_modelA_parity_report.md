# Repaired Model A Parity Report

- Generated UTC: 2026-03-02T23:35:38.674139+00:00
- Canonical repaired 1h baseline: `/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650/repaired_1h_reference_summary.csv`

## Repaired 1h Reference Contract

- `scripts/backtest_exec_phasec_sol.py:240` defines `_simulate_1h_reference`.
- `scripts/backtest_exec_phasec_sol.py:247` keeps `defer_exit_to_next_bar=True` by default.
- `scripts/backtest_exec_phasec_sol.py:316` sets `eval_start_idx = idx + 1`, so 1h exit evaluation starts on the first full bar after entry.
- `scripts/backtest_exec_phasec_sol.py:390-392` records `entry_parent_bar_time`, `exit_eval_start_time`, and `exit_eval_bar_time` for chronology traceability.

## Model A Audit Wrapper

- `scripts/phase_a_model_a_audit.py:382` defines `simulate_frozen_1h_exit`.
- `scripts/phase_a_model_a_audit.py:423` uses `searchsorted(..., side="right")`, which skips the fill bar before starting 1h exit evaluation.
- `scripts/phase_a_model_a_audit.py:160` calls the repaired `_simulate_1h_reference`, so the comparison baseline is the repaired 1h contract.

## Paper Runtime Parity Check

- `paper_trading/app/model_a_runtime.py:663` defines `_maybe_close_position`.
- `paper_trading/app/model_a_runtime.py:672` returns early when `current_bar_ts <= fill_time`, blocking same-parent-bar 1h exits after fill.
- `paper_trading/app/model_a_runtime.py:674` uses `searchsorted(..., side="right")`, matching the repaired 1h chronology.

## Verdict

- paper_runtime_parity_clean: `1`
- remaining_parity_gaps: `[]`