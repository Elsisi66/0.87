# Cost Sensitivity Report (exec_limit)

- Generated UTC: 2026-02-20T11:04:38.510890+00:00
- Symbols: SOLUSDT, AVAXUSDT, NEARUSDT
- Base dir: `/root/analysis/0.87/reports/execution_layer`
- Scenarios per symbol: 81
- Total scenarios: 243
- Pass threshold (exec beats baseline expectancy): 0.70
- Overall pass ratio: 0.818930
- Gate result: PASS

## Per-Symbol Gate Summary

| symbol | scenarios | scenarios_exec_beats_baseline | pass_ratio | avg_exec_expectancy | avg_baseline_expectancy | avg_entry_rate | avg_taker_share | avg_sl_hit_rate_exec |
|---|---|---|---|---|---|---|---|---|
| NEARUSDT | 81 | 78 | 0.962963 | -0.001626 | -0.002264 | 0.504831 | 0.143541 | 0.727273 |
| AVAXUSDT | 81 | 63 | 0.777778 | -0.001173 | -0.001424 | 0.383984 | 0.128342 | 0.631016 |
| SOLUSDT | 81 | 58 | 0.716049 | -0.001181 | -0.001412 | 0.911667 | 0.117002 | 0.645338 |

## Source Files

- `SOLUSDT`: `/root/analysis/0.87/reports/execution_layer/20260220_071048_walkforward_SOLUSDT/SOLUSDT_walkforward_test_signals.csv`
- `AVAXUSDT`: `/root/analysis/0.87/reports/execution_layer/20260220_074353_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_signals.csv`
- `NEARUSDT`: `/root/analysis/0.87/reports/execution_layer/20260220_074718_walkforward_NEARUSDT/NEARUSDT_walkforward_test_signals.csv`

- Detailed CSV: `/root/analysis/0.87/reports/execution_layer/20260220_110354/cost_sensitivity_summary.csv`
- Source list CSV: `/root/analysis/0.87/reports/execution_layer/20260220_110354/cost_sensitivity_sources.csv`
