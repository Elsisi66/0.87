# A1 Model A Contract Report

- Generated UTC: 2026-02-28T01:48:55.969032+00:00
- Frozen subset: `/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv`
- Phase R route source: `/root/analysis/0.87/reports/execution_layer/PHASER_ROUTE_HARNESS_REDESIGN_20260228_005334`
- Freeze lock pass: `1`
- 1h signal source file (from Phase E2 if available): `unavailable`
- 1h reference engine used for this audit: `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference`.
- 1h reference entry semantics:
  - signal candle = `signal_time` from the frozen representative subset
  - action candle = first 1h bar with timestamp `>= signal_time`
  - TP/SL = `signal tp_mult/sl_mult` applied to the 1h action-candle entry price
  - if SL and TP are both touched in the same 1h bar, SL wins
  - if neither is hit, exit at the last 1h close before the fixed horizon
- Current hybrid downstream evaluator: `src/execution/ga_exec_3m_opt.py::_simulate_candidate_signal`.
- Hybrid exit override evidence:
  - it calls `_simulate_dynamic_exit_long` after entry fill, so exit logic is re-simulated on 3m bars
  - the hybrid evaluator exposes downstream exit knobs such as `time_stop_min`, `break_even_*`, `trailing_*`, and `partial_take_*`
- Prior hybrid branch mixes execution and exits: `1`
- Repaired routes reproduced from code: `3` routes
- Repaired route reproduction matches Phase R artifact: `1`
- Route reproduction mismatches (if any): `[]`

## Repaired Route Feasibility

| route_id | route_signal_count | wf_test_signal_count | hard_min_trades_overall | headroom_vs_overall_gate | route_trade_gates_reachable |
| --- | --- | --- | --- | --- | --- |
| route_back_60pct | 720 | 216 | 200 | 16 | 1 |
| route_center_60pct | 720 | 216 | 200 | 16 | 1 |
| route_front_60pct | 720 | 216 | 200 | 16 | 1 |

