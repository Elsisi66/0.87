# MB1 Contract Report

- Generated UTC: `2026-02-28T14:31:08.929302+00:00`
- Freeze lock pass: `1`
- Wrapper purity: `3m entry only = 1`, `1h exit only = 1`
- Forbidden exit controls active: `0`
- Model A primary parity OK: `1`
- Route trade gates reachable on repaired harness: `1`
- All local Model B candidates are sizing-only and preserve participation: `1`
- Raw sizing probes: `20`
- Unique probes after duplicate collapse: `20`
- Correlation-adjusted clusters: `2`

## Model A Primary Parity Diffs
- overall_exec_expectancy_net: `2.710505431213761e-17`
- overall_delta_expectancy_exec_minus_baseline: `8.847089727481716e-17`
- overall_cvar_improve_ratio: `1.3877787807814457e-17`
- overall_maxdd_improve_ratio: `0.0`
- overall_entry_rate: `0.0`
- overall_entries_valid: `0.0`
- overall_exec_taker_share: `4.5102810375396984e-17`
- overall_exec_p95_fill_delay_min: `0.0`
- min_split_delta: `7.37257477290143e-17`

## Prior Model B Anchor Reproduction Diffs
- regime_cap_size_delay_tiered: `{'expectancy_net': 5.854691731421724e-18, 'delta_expectancy_vs_modelA': 0.0, 'cvar_improve_ratio': 4.85722573273506e-17, 'maxdd_improve_ratio': 1.214306433183765e-17, 'route_pass': 0.0, 'stress_lite_pass': 0.0, 'bootstrap_pass_rate': 0.0}`
- regime_cap_size_delay_soft: `{'expectancy_net': 6.461844948013606e-17, 'delta_expectancy_vs_modelA': 0.0, 'cvar_improve_ratio': 2.7755575615628914e-17, 'maxdd_improve_ratio': 7.112366251504909e-17, 'route_pass': 0.0, 'stress_lite_pass': 0.0, 'bootstrap_pass_rate': 0.0}`

## Approved Anchors in This Confirmation Run

| variant_id | expectancy_net | delta_expectancy_vs_modelA | cvar_improve_ratio | maxdd_improve_ratio | route_pass | stress_lite_pass | bootstrap_pass_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regime_cap_size_delay_tiered | 0.001234498935 | 2.254945341e-05 | 0.05592952826 | 0.0105036945 | 1 | 1 | 0.95 |
| regime_cap_size_delay_soft | 0.001215277506 | 3.328023951e-06 | 0.0592952826 | 0.006564809061 | 1 | 1 | 0.92 |
| MODEL_A_PRIMARY_BASELINE | 0.001211949482 | 0 | 0 | 0 | 1 | 1 | 1 |
