# Model B1 Reproduction Report

- Generated UTC: `2026-02-28T14:14:11.966189+00:00`
- Frozen lock pass: `1`
- Primary parity OK vs approved Phase C row: `1`
- Primary candidate: `M3_ENTRY_ONLY_FASTER_C_WIN_02`
- Backup reference (for context only): `M2_ENTRY_ONLY_MORE_PASSIVE_NOFB_C_FB_ON`

## Reproduced Primary Metrics
- expectancy_net: `0.001211949481977627`
- delta_expectancy_vs_1h_reference: `0.0021298751745981886`
- cvar_improve_ratio: `0.10601784861933772`
- maxdd_improve_ratio: `0.7037901248367386`
- entry_rate: `1.0`
- entries_valid: `360`
- taker_share: `0.019444444444444445`
- p95_fill_delay_min: `3.0`

## Parity Diffs vs Approved Phase C Primary
- overall_exec_expectancy_net: `2.710505431213761e-17`
- overall_delta_expectancy_exec_minus_baseline: `8.847089727481716e-17`
- overall_cvar_improve_ratio: `1.3877787807814457e-17`
- overall_maxdd_improve_ratio: `0.0`
- overall_entry_rate: `0.0`
- overall_entries_valid: `0.0`
- overall_exec_taker_share: `4.5102810375396984e-17`
- overall_exec_p95_fill_delay_min: `0.0`
- min_split_delta: `7.37257477290143e-17`

## Route Baseline
| route_id | expectancy_net | entry_rate | entries_valid |
| --- | --- | --- | --- |
| route_back_60pct | 0.0002074420295 | 1 | 216 |
| route_center_60pct | 0.004159875743 | 1 | 216 |
| route_front_60pct | 0.001192356747 | 1 | 216 |
