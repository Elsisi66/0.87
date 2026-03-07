# Walkforward Exec Limit Report: NEARUSDT

- Generated UTC: 2026-02-20T07:47:18.863965+00:00
- params_file: `/root/analysis/0.87/data/metadata/params/NEARUSDT_C13_active_params_long.json`
- total signals: 1380
- train/test: 966/414
- train tuning CSV: `/root/analysis/0.87/reports/execution_layer/20260220_074718_walkforward_NEARUSDT/NEARUSDT_walkforward_train_tuning.csv`
- test signals CSV: `/root/analysis/0.87/reports/execution_layer/20260220_074718_walkforward_NEARUSDT/NEARUSDT_walkforward_test_signals.csv`
- test summary CSV: `/root/analysis/0.87/reports/execution_layer/20260220_074718_walkforward_NEARUSDT/NEARUSDT_walkforward_test_summary.csv`

## Selected Config (Train)

- k: 0.2000
- timeout_bars: 10
- fallback: market
- use_vol_gate: 0
- passes_constraints: 1

## Test Metrics

- entry_rate: 0.504831
- taker_share: 0.143541
- max_fill_delay_min: 30.00
- pnl_net_sum: -0.263242
- expectancy_net: -0.001260
- baseline_expectancy_net: -0.001797

## Top 10 Train Configs

| k | timeout_bars | fallback | use_vol_gate | entry_rate | taker_share | max_fill_delay_min | expectancy_net | pnl_net_sum | passes_constraints |
|---|---|---|---|---|---|---|---|---|---|
| 0.200000 | 10 | market | 0 | 0.908903 | 0.110478 | 30.000000 | -0.001113 | -0.976875 | 1 |
| 0.200000 | 20 | market | 0 | 0.908903 | 0.110478 | 30.000000 | -0.001113 | -0.976875 | 1 |
| 0.200000 | 40 | market | 0 | 0.908903 | 0.110478 | 30.000000 | -0.001113 | -0.976875 | 1 |
| 0.500000 | 10 | market | 0 | 0.908903 | 0.110478 | 30.000000 | -0.001113 | -0.976875 | 1 |
| 0.500000 | 20 | market | 0 | 0.908903 | 0.110478 | 30.000000 | -0.001113 | -0.976875 | 1 |
| 0.500000 | 40 | market | 0 | 0.908903 | 0.110478 | 30.000000 | -0.001113 | -0.976875 | 1 |
| 0.800000 | 10 | market | 0 | 0.908903 | 0.110478 | 30.000000 | -0.001113 | -0.976875 | 1 |
| 0.800000 | 20 | market | 0 | 0.908903 | 0.110478 | 30.000000 | -0.001113 | -0.976875 | 1 |
| 0.800000 | 40 | market | 0 | 0.908903 | 0.110478 | 30.000000 | -0.001113 | -0.976875 | 1 |
| 1.100000 | 10 | market | 0 | 0.908903 | 0.110478 | 30.000000 | -0.001113 | -0.976875 | 1 |
