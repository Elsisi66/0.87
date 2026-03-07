# Walkforward Exec Limit Report: AVAXUSDT

- Generated UTC: 2026-02-20T12:22:47.910785+00:00
- params_file: `/root/analysis/0.87/data/metadata/params/AVAXUSDT__UNIVERSE_LONG_active_params.json`
- execution_config: `/root/analysis/0.87/configs/execution_configs.yaml`
- symbol_config: `{"constraints": {"max_fill_delay_min": 90, "max_taker_share": 0.4, "min_entry_rate": 0.55}, "exec_mode": "exec_limit", "fallback": "market", "fallback_values": ["market"], "k_base": 0.05, "k_values": [0.0, 0.05, 0.1, 0.2], "ladder": 0, "stop_distance_min_pct": 0.0, "tight_constraints": {"max_fill_delay_min": 45, "max_taker_share": 0.3, "min_entry_rate": 0.95, "min_median_entry_improvement_bps": 0.0}, "timeout_bars": 20, "timeout_values": [10, 20, 40], "trend_fast_col": "EMA_50", "trend_min_slope": 0.0, "trend_slow_col": "EMA_120", "use_panic_filter": 0, "use_trend_gate": 0, "use_vol_gate": 0, "use_vol_regime_gate": 1, "vol_gate_values": [0], "vol_regime_lookback_bars": 2160, "vol_regime_max_percentile": 90}`
- gate_config: `{"stop_distance_min_pct": 0.0, "trend_fast_col": "EMA_50", "trend_min_slope": 0.0, "trend_slow_col": "EMA_120", "use_trend_gate": 0, "use_vol_regime_gate": 1, "vol_regime_lookback_bars": 2160, "vol_regime_max_percentile": 90}`
- tight_mode: 1
- total signals: 1484
- train/test: 1039/445
- train tuning CSV: `/root/analysis/0.87/reports/execution_layer/20260220_122247_walkforward_AVAXUSDT/AVAXUSDT_walkforward_train_tuning.csv`
- test signals CSV: `/root/analysis/0.87/reports/execution_layer/20260220_122247_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_signals.csv`
- test summary CSV: `/root/analysis/0.87/reports/execution_layer/20260220_122247_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_summary.csv`

## Selected Config (Train)

- k: 0.0000
- timeout_bars: 10
- fallback: market
- use_vol_gate: 0
- passes_constraints: 0

## Test Metrics

- entry_rate: 0.968539
- taker_share: 0.155452
- max_fill_delay_min: 30.00
- median_fill_delay_min: 0.00
- pnl_net_sum: -0.284663
- expectancy_net: -0.000660
- baseline_expectancy_net: -0.000381
- median_entry_improvement_bps: 5.7189

## Constraint Report

- min_entry_rate: 0.950000
- max_fill_delay_min: 45.00
- max_taker_share: 0.300000
- min_median_entry_improvement_bps: 0.0000
- selected_passes_constraints: 0

### Tight Mode

- tight_require_no_missing_data: 1
- tight_require_pass: 0
- fallback forced to `market`

## Top 10 Train Configs

| k | timeout_bars | fallback | use_vol_gate | entry_rate | taker_share | max_fill_delay_min | median_fill_delay_min | expectancy_net | median_entry_improvement_bps | pnl_net_sum | passes_constraints |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.000000 | 10 | market | 0 | 0.930703 | 0.100310 | 60.000000 | 0.000000 | -0.000972 | 8.321373 | -0.940282 | 0 |
| 0.000000 | 20 | market | 0 | 0.930703 | 0.100310 | 60.000000 | 0.000000 | -0.000972 | 8.321373 | -0.940282 | 0 |
| 0.000000 | 40 | market | 0 | 0.930703 | 0.100310 | 60.000000 | 0.000000 | -0.000972 | 8.321373 | -0.940282 | 0 |
| 0.050000 | 10 | market | 0 | 0.930703 | 0.100310 | 60.000000 | 0.000000 | -0.000972 | 8.321373 | -0.940282 | 0 |
| 0.050000 | 20 | market | 0 | 0.930703 | 0.100310 | 60.000000 | 0.000000 | -0.000972 | 8.321373 | -0.940282 | 0 |
| 0.050000 | 40 | market | 0 | 0.930703 | 0.100310 | 60.000000 | 0.000000 | -0.000972 | 8.321373 | -0.940282 | 0 |
| 0.100000 | 10 | market | 0 | 0.930703 | 0.100310 | 60.000000 | 0.000000 | -0.000972 | 8.321373 | -0.940282 | 0 |
| 0.100000 | 20 | market | 0 | 0.930703 | 0.100310 | 60.000000 | 0.000000 | -0.000972 | 8.321373 | -0.940282 | 0 |
| 0.100000 | 40 | market | 0 | 0.930703 | 0.100310 | 60.000000 | 0.000000 | -0.000972 | 8.321373 | -0.940282 | 0 |
| 0.200000 | 10 | market | 0 | 0.930703 | 0.100310 | 60.000000 | 0.000000 | -0.000972 | 8.321373 | -0.940282 | 0 |
