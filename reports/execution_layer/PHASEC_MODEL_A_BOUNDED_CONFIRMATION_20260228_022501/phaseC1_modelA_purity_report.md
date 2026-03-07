# C1 Model A Purity Report

- Generated UTC: 2026-02-28T02:25:35.791921+00:00
- Freeze lock pass: `1`
- Wrapper remains pure Model A:
  - 3m entry execution only
  - 1h TP/SL/exit semantics only
  - no dynamic 3m exit mutation
- Forbidden exit knobs blocked: `['tp_mult', 'sl_mult', 'time_stop_min', 'break_even_enabled', 'break_even_trigger_r', 'break_even_offset_bps', 'trailing_enabled', 'trail_start_r', 'trail_step_bps', 'partial_take_enabled', 'partial_take_r', 'partial_take_pct']`
- Repaired route reproduction matches Phase R: `1`
- Route reproduction mismatches: `[]`

## Seed Clusters

| candidate_id | seed_cluster_id | entry_mode | limit_offset_bps | fallback_to_market | fallback_delay_min | max_fill_delay_min |
| --- | --- | --- | --- | --- | --- | --- |
| M1_ENTRY_ONLY_PASSIVE_BASELINE_NOFB | 2 | limit | 0.75 | 0 | 0 | 24 |
| M2_ENTRY_ONLY_MORE_PASSIVE_NOFB | 4 | limit | 1.5 | 0 | 0 | 45 |
| M2_ENTRY_ONLY_MORE_PASSIVE_OFF_02 | 3 | limit | 1.95 | 1 | 12 | 45 |
| M3_ENTRY_ONLY_FASTER | 1 | limit | 0.35 | 1 | 3 | 9 |

## Repaired Route Feasibility

| route_id | route_signal_count | wf_test_signal_count | headroom_vs_overall_gate | route_trade_gates_reachable |
| --- | --- | --- | --- | --- |
| route_back_60pct | 720 | 216 | 16 | 1 |
| route_center_60pct | 720 | 216 | 16 | 1 |
| route_front_60pct | 720 | 216 | 16 | 1 |
