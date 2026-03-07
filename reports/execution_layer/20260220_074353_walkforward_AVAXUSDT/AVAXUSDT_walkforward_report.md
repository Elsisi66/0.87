# Walkforward Exec Limit Report: AVAXUSDT

- Generated UTC: 2026-02-20T07:43:53.956268+00:00
- params_file: `/root/analysis/0.87/data/metadata/params/AVAXUSDT_C13_active_params_long.json`
- total signals: 1624
- train/test: 1137/487
- train tuning CSV: `/root/analysis/0.87/reports/execution_layer/20260220_074353_walkforward_AVAXUSDT/AVAXUSDT_walkforward_train_tuning.csv`
- test signals CSV: `/root/analysis/0.87/reports/execution_layer/20260220_074353_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_signals.csv`
- test summary CSV: `/root/analysis/0.87/reports/execution_layer/20260220_074353_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_summary.csv`

## Selected Config (Train)

- k: 0.2000
- timeout_bars: 10
- fallback: market
- use_vol_gate: 0
- passes_constraints: 1

## Test Metrics

- entry_rate: 0.383984
- taker_share: 0.128342
- max_fill_delay_min: 30.00
- pnl_net_sum: -0.151186
- expectancy_net: -0.000808
- baseline_expectancy_net: -0.000957

## Top 10 Train Configs

| k | timeout_bars | fallback | use_vol_gate | entry_rate | taker_share | max_fill_delay_min | expectancy_net | pnl_net_sum | passes_constraints |
|---|---|---|---|---|---|---|---|---|---|
| 0.200000 | 10 | market | 0 | 0.914688 | 0.092308 | 60.000000 | -0.000932 | -0.968915 | 1 |
| 0.200000 | 20 | market | 0 | 0.914688 | 0.092308 | 60.000000 | -0.000932 | -0.968915 | 1 |
| 0.200000 | 40 | market | 0 | 0.914688 | 0.092308 | 60.000000 | -0.000932 | -0.968915 | 1 |
| 0.500000 | 10 | market | 0 | 0.914688 | 0.092308 | 60.000000 | -0.000932 | -0.968915 | 1 |
| 0.500000 | 20 | market | 0 | 0.914688 | 0.092308 | 60.000000 | -0.000932 | -0.968915 | 1 |
| 0.500000 | 40 | market | 0 | 0.914688 | 0.092308 | 60.000000 | -0.000932 | -0.968915 | 1 |
| 0.800000 | 10 | market | 0 | 0.914688 | 0.092308 | 60.000000 | -0.000932 | -0.968915 | 1 |
| 0.800000 | 20 | market | 0 | 0.914688 | 0.092308 | 60.000000 | -0.000932 | -0.968915 | 1 |
| 0.800000 | 40 | market | 0 | 0.914688 | 0.092308 | 60.000000 | -0.000932 | -0.968915 | 1 |
| 1.100000 | 10 | market | 0 | 0.914688 | 0.092308 | 60.000000 | -0.000932 | -0.968915 | 1 |
