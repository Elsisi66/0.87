# NY0 Route Support Upper Bound

- Generated UTC: 2026-02-27T12:34:23.850838+00:00
- Freeze lock pass: `1`
- Representative subset rows: `1200`
- Base bundle contexts loaded: `1200`
- Walkforward config: `train_ratio=0.7`, `wf_splits=5`
- Hard trade gates: `overall>=200`, `symbol>=50`
- Minimum route contexts needed for `overall>=200` under this walkforward setting: `665` (yields `200` test signals)
- Route feasibility verdict: `infeasible`
- Failing routes: `['route1_holdout']`

## Exact Route Definition Used In NX3

- `route1_holdout`: last `max(120, round(0.20 * N_subset))` signals.
- `route2_reslice`: full representative subset.

## Route Upper-Bound Table

| route_id | subset_signal_count | wf_test_signal_count | upper_bound_entries_entry_rate_1 | hard_min_trades_overall | hard_min_trades_symbol | overall_trade_gate_reachable | symbol_trade_gate_reachable | route_trade_gates_reachable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| route1_holdout | 240 | 72 | 72 | 200 | 50 | 0 | 1 | 0 |
| route2_reslice | 1200 | 360 | 360 | 200 | 50 | 1 | 1 | 1 |
