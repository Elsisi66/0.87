# Walkforward Exec Limit Report: SOLUSDT

- Generated UTC: 2026-02-20T12:27:41.310921+00:00
- params_file: `/root/analysis/0.87/data/metadata/params/SOLUSDT_C13_active_params_long.json`
- execution_config: `/root/analysis/0.87/configs/execution_configs.yaml`
- symbol_config: `{"constraints": {"max_fill_delay_min": 90, "max_taker_share": 0.4, "min_entry_rate": 0.6}, "exec_mode": "exec_limit", "fallback": "market", "fallback_values": ["market"], "k_base": 0.2, "k_values": [0.0, 0.1, 0.2, 0.3], "ladder": 0, "stop_distance_min_pct": 0.0, "tight_constraints": {"max_fill_delay_min": 45, "max_taker_share": 0.25, "min_entry_rate": 0.97, "min_median_entry_improvement_bps": 0.0}, "timeout_bars": 10, "timeout_values": [10, 20, 40], "trend_fast_col": "EMA_50", "trend_min_slope": 0.0, "trend_slow_col": "EMA_120", "use_panic_filter": 1, "use_trend_gate": 0, "use_vol_gate": 0, "use_vol_regime_gate": 1, "vol_gate_values": [0], "vol_regime_lookback_bars": 2160, "vol_regime_max_percentile": 90}`
- gate_config: `{"stop_distance_min_pct": 0.0, "trend_fast_col": "EMA_50", "trend_min_slope": 0.0, "trend_slow_col": "EMA_120", "use_trend_gate": 0, "use_vol_regime_gate": 1, "vol_regime_lookback_bars": 2160, "vol_regime_max_percentile": 90}`
- tight_mode: 1
- total signals: 2000
- train/test: 1400/600
- train tuning CSV: `/root/analysis/0.87/reports/execution_layer/20260220_122741_walkforward_SOLUSDT/SOLUSDT_walkforward_train_tuning.csv`
- test signals CSV: `/root/analysis/0.87/reports/execution_layer/20260220_122741_walkforward_SOLUSDT/SOLUSDT_walkforward_test_signals.csv`
- test summary CSV: `/root/analysis/0.87/reports/execution_layer/20260220_122741_walkforward_SOLUSDT/SOLUSDT_walkforward_test_summary.csv`

## Selected Config (Train)

- k: 0.0000
- timeout_bars: 10
- fallback: market
- use_vol_gate: 0
- passes_constraints: 1

## Test Metrics

- entry_rate: 0.988333
- taker_share: 0.116358
- max_fill_delay_min: 30.00
- median_fill_delay_min: 0.00
- pnl_net_sum: -0.507835
- expectancy_net: -0.000856
- baseline_expectancy_net: -0.000643
- median_entry_improvement_bps: 5.1465

## Constraint Report

- min_entry_rate: 0.970000
- max_fill_delay_min: 45.00
- max_taker_share: 0.250000
- min_median_entry_improvement_bps: 0.0000
- selected_passes_constraints: 1

### Tight Mode

- tight_require_no_missing_data: 1
- tight_require_pass: 1
- fallback forced to `market`

## Top 10 Train Configs

| k | timeout_bars | fallback | use_vol_gate | entry_rate | taker_share | max_fill_delay_min | median_fill_delay_min | expectancy_net | median_entry_improvement_bps | pnl_net_sum | passes_constraints |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000000 | 10 | market | 0 | 0.977143 | 0.118421 | 30.000000 | 0.000000 | -0.000796 | 5.776568 | -1.089335 | 1 |
| 0.000000 | 20 | market | 0 | 0.977143 | 0.118421 | 30.000000 | 0.000000 | -0.000796 | 5.776568 | -1.089335 | 1 |
| 0.000000 | 40 | market | 0 | 0.977143 | 0.118421 | 30.000000 | 0.000000 | -0.000796 | 5.776568 | -1.089335 | 1 |
| 0.100000 | 10 | market | 0 | 0.977143 | 0.118421 | 30.000000 | 0.000000 | -0.000796 | 5.776568 | -1.089335 | 1 |
| 0.100000 | 20 | market | 0 | 0.977143 | 0.118421 | 30.000000 | 0.000000 | -0.000796 | 5.776568 | -1.089335 | 1 |
| 0.100000 | 40 | market | 0 | 0.977143 | 0.118421 | 30.000000 | 0.000000 | -0.000796 | 5.776568 | -1.089335 | 1 |
| 0.200000 | 10 | market | 0 | 0.977143 | 0.118421 | 30.000000 | 0.000000 | -0.000796 | 5.776568 | -1.089335 | 1 |
| 0.200000 | 20 | market | 0 | 0.977143 | 0.118421 | 30.000000 | 0.000000 | -0.000796 | 5.776568 | -1.089335 | 1 |
| 0.200000 | 40 | market | 0 | 0.977143 | 0.118421 | 30.000000 | 0.000000 | -0.000796 | 5.776568 | -1.089335 | 1 |
| 0.300000 | 10 | market | 0 | 0.977143 | 0.118421 | 30.000000 | 0.000000 | -0.000796 | 5.776568 | -1.089335 | 1 |
