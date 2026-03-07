# Walkforward Exec Limit Report: SOLUSDT

- Generated UTC: 2026-02-20T11:17:06.376283+00:00
- params_file: `/root/analysis/0.87/data/metadata/params/SOLUSDT_C13_active_params_long.json`
- execution_config: `/root/analysis/0.87/configs/execution_configs.yaml`
- symbol_config: `{"constraints": {"max_fill_delay_min": 90, "max_taker_share": 0.4, "min_entry_rate": 0.6}, "exec_mode": "exec_limit", "fallback": "market", "fallback_values": ["market"], "k_base": 0.2, "k_values": [0.0, 0.1, 0.2, 0.3], "ladder": 0, "timeout_bars": 10, "timeout_values": [10, 20, 40], "use_panic_filter": 1, "use_vol_gate": 1, "vol_gate_values": [1]}`
- total signals: 2000
- train/test: 1400/600
- train tuning CSV: `/root/analysis/0.87/reports/execution_layer/20260220_111706_walkforward_SOLUSDT/SOLUSDT_walkforward_train_tuning.csv`
- test signals CSV: `/root/analysis/0.87/reports/execution_layer/20260220_111706_walkforward_SOLUSDT/SOLUSDT_walkforward_test_signals.csv`
- test summary CSV: `/root/analysis/0.87/reports/execution_layer/20260220_111706_walkforward_SOLUSDT/SOLUSDT_walkforward_test_summary.csv`

## Selected Config (Train)

- k: 0.0000
- timeout_bars: 10
- fallback: market
- use_vol_gate: 1
- passes_constraints: 1

## Test Metrics

- entry_rate: 0.911667
- taker_share: 0.117002
- max_fill_delay_min: 30.00
- pnl_net_sum: -0.447375
- expectancy_net: -0.000818
- baseline_expectancy_net: -0.000946

## Top 10 Train Configs

| k | timeout_bars | fallback | use_vol_gate | entry_rate | taker_share | max_fill_delay_min | expectancy_net | pnl_net_sum | passes_constraints |
|---|---|---|---|---|---|---|---|---|---|
| 0.000000 | 10 | market | 1 | 0.919286 | 0.120435 | 30.000000 | -0.000864 | -1.112541 | 1 |
| 0.000000 | 20 | market | 1 | 0.919286 | 0.120435 | 30.000000 | -0.000864 | -1.112541 | 1 |
| 0.000000 | 40 | market | 1 | 0.919286 | 0.120435 | 30.000000 | -0.000864 | -1.112541 | 1 |
| 0.100000 | 10 | market | 1 | 0.919286 | 0.120435 | 30.000000 | -0.000864 | -1.112541 | 1 |
| 0.100000 | 20 | market | 1 | 0.919286 | 0.120435 | 30.000000 | -0.000864 | -1.112541 | 1 |
| 0.100000 | 40 | market | 1 | 0.919286 | 0.120435 | 30.000000 | -0.000864 | -1.112541 | 1 |
| 0.200000 | 10 | market | 1 | 0.919286 | 0.120435 | 30.000000 | -0.000864 | -1.112541 | 1 |
| 0.200000 | 20 | market | 1 | 0.919286 | 0.120435 | 30.000000 | -0.000864 | -1.112541 | 1 |
| 0.200000 | 40 | market | 1 | 0.919286 | 0.120435 | 30.000000 | -0.000864 | -1.112541 | 1 |
| 0.300000 | 10 | market | 1 | 0.919286 | 0.120435 | 30.000000 | -0.000864 | -1.112541 | 1 |
