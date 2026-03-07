# Phase C SOL Decision

- Generated UTC: 2026-02-21T23:48:43.808940+00:00
- Symbol: SOLUSDT
- Phase A contract dir: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310`
- Signal subset hash: `5e719faf676dffba8d7da926314997182d429361495884b8a870c3393c079bbf`
- WF split definition hash: `388ba743b9c16c291385a9ecab6435eabf65eb16f1e1083eee76627193c42c01`

## Best Refined Config

- tp_mult: 1.0
- sl_mult: 0.75
- time_stop_min: 720
- break_even_enabled: 0
- break_even_trigger_r: 0.5
- break_even_offset_bps: 0.0
- partial_take_enabled: 0
- partial_take_r: 0.8
- partial_take_pct: 0.25
- cfg_hash: `a285b86c4c22a26976d4a762`

## Support

- signals_total_test: 600
- trades_total_test: 600
- min_split_trades: 120
- median_split_trades: 120.00

## Rubric

- pass_expectancy (delta_expectancy_best_exit_minus_baseline_exit >= 0.000050): 1
- pass_maxdd_not_worse (delta_maxdd_best_exit_minus_baseline_exit >= -0.020): 1
- pass_cvar_not_worse (delta_cvar5_best_exit_minus_baseline_exit >= -0.000100): 1
- pass_stability: 1
- pass_data_quality: 1
- pass_participation: 1
- Decision: **PASS**

## Deltas

- delta_expectancy_best_exit_minus_baseline_exit: 0.000084
- delta_maxdd_best_exit_minus_baseline_exit: 0.071583
- delta_cvar5_best_exit_minus_baseline_exit: 0.000250
