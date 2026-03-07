# B1 Model A Lock Report

- Generated UTC: 2026-02-28T02:05:21.079436+00:00
- Freeze lock pass: `1`
- Wrapper remains identical to Phase A and still enforces:
  - 3m entry execution only
  - 1h exit-only ownership after fill
  - no dynamic 3m exit mutation
- Forbidden exit knobs blocked: `['tp_mult', 'sl_mult', 'time_stop_min', 'break_even_enabled', 'break_even_trigger_r', 'break_even_offset_bps', 'trailing_enabled', 'trail_start_r', 'trail_step_bps', 'partial_take_enabled', 'partial_take_r', 'partial_take_pct']`
- Phase A reference dir: `/root/analysis/0.87/reports/execution_layer/PHASEA_MODEL_A_AUDIT_20260228_014944`
- Repaired route reproduction matches Phase R: `1`
- Route reproduction mismatches: `[]`

## Repaired Route Feasibility

| route_id | route_signal_count | wf_test_signal_count | headroom_vs_overall_gate | route_trade_gates_reachable |
| --- | --- | --- | --- | --- |
| route_back_60pct | 720 | 216 | 16 | 1 |
| route_center_60pct | 720 | 216 | 16 | 1 |
| route_front_60pct | 720 | 216 | 16 | 1 |

## Phase A Anchor Rows

| candidate_id | valid_for_ranking | exec_expectancy_net | delta_expectancy_vs_1h_reference | cvar_improve_ratio | maxdd_improve_ratio | route_pass |
| --- | --- | --- | --- | --- | --- | --- |
| M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.001167739031 | 0.002085664724 | 0.07067856575 | 0.7032237657 | 1 |
| M2_ENTRY_ONLY_MORE_PASSIVE | 1 | 0.0009979046213 | 0.001915830314 | 0.07951338646 | 0.7035408049 | 1 |
| M3_ENTRY_ONLY_FASTER | 1 | 0.001210006245 | 0.002127931938 | 0.08834820718 | 0.7030546976 | 1 |
