# R0 Route Harness Forensics

- Generated UTC: 2026-02-28T00:54:05.381899+00:00
- NY evidence source: `/root/analysis/0.87/reports/execution_layer/PHASENY_NX_POSTMORTEM_FEASIBILITY_20260227_123357`
- Legacy route constructor source: `scripts/phase_nx_exec_family_discovery.py::build_route_bundles`
- Legacy route definitions are imported from `scripts/phase_af_ah_sizing_autorun.py::route_signal_sets`.
- `route1_holdout` is the last `max(120, round(20% * N_subset))` signals; with `N=1200`, this becomes `240` route signals.
- Under walkforward (`train_ratio=0.7`, `wf_splits=5`), `route1_holdout` yields only `72` scored test signals.
- Hard gate requires `overall>=200`; therefore `route1_holdout` is ex-ante infeasible.
- Failing legacy routes: `['route1_holdout']`

## Legacy Route Support Summary

| route_id | route_signal_count | route_upper_bound_entries | min_split_test_count | max_split_test_count | route_trade_gates_reachable |
| --- | --- | --- | --- | --- | --- |
| route1_holdout | 240 | 72 | 14 | 16 | 0 |
| route2_reslice | 1200 | 360 | 72 | 72 | 1 |

Exact split-level support counts are stored in `phaseR0_route_support_table.csv`.
