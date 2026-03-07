# Exit Sweep Decision

- Generated UTC: 2026-02-21T21:44:29.682376+00:00
- symbols: `SOLUSDT`
- configs evaluated: 8

## Best Config

- tp_mult: 1.0
- sl_mult: 0.5
- time_stop_min: 2880.0
- break_even_enabled: 0.0
- break_even_trigger_r: 0.75
- break_even_offset_bps: 5.0
- partial_take_enabled: 0.0
- partial_take_r: 0.6
- partial_take_pct: 0.5

## Rubric

- pass_expectancy (delta >= 0.000050): 0
- pass_maxdd_not_worse (delta >= -0.020): 0
- pass_cvar_not_worse (delta >= -0.000100): 1
- pass_stability: 0
- pass_data_quality: 1
- pass_participation: 1
- Decision: **FAIL**

## Outputs

- results: `/root/analysis/0.87/reports/execution_layer/EXIT_SWEEP_20260221_214408/exit_sweep_results.csv`
- topk: `/root/analysis/0.87/reports/execution_layer/EXIT_SWEEP_20260221_214408/exit_sweep_topk.csv`
- split rollup: `/root/analysis/0.87/reports/execution_layer/EXIT_SWEEP_20260221_214408/walkforward_results_by_split.csv`
- symbol rollup: `/root/analysis/0.87/reports/execution_layer/EXIT_SWEEP_20260221_214408/risk_rollup_by_symbol.csv`
- overall rollup: `/root/analysis/0.87/reports/execution_layer/EXIT_SWEEP_20260221_214408/risk_rollup_overall.csv`
