# Diagnose 1h Bleed: SOLUSDT

- Generated UTC: 2026-02-20T11:29:20.142467+00:00
- Source summary CSV: `/root/analysis/0.87/reports/execution_layer/20260220_111706_walkforward_SOLUSDT/SOLUSDT_walkforward_test_summary.csv`
- Source test-signals CSV: `/root/analysis/0.87/reports/execution_layer/20260220_111706_walkforward_SOLUSDT/SOLUSDT_walkforward_test_signals.csv`
- Params file: `/root/analysis/0.87/data/metadata/params/SOLUSDT_C13_active_params_long.json`
- Valid baseline entries analyzed: 600

## Stop Distance Buckets

| bucket_stop_distance | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| (0.0, 0.0726] | 600 | -0.567306 | -0.000946 | 0.955000 | 0.001667 |

## ATR 1h Quantile Buckets

| bucket_atr_1h | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| (2.448, 3.937] | 103 | -0.162288 | -0.001576 | 0.980583 | 0.000000 |
| (1.983, 2.448] | 102 | -0.148952 | -0.001460 | 0.970588 | 0.000000 |
| (1.0330000000000001, 1.689] | 103 | -0.128421 | -0.001247 | 0.941748 | 0.000000 |
| (1.689, 1.983] | 103 | -0.068809 | -0.000668 | 0.941748 | 0.000000 |
| nan | 189 | -0.058836 | -0.000311 | 0.947090 | 0.005291 |

## Trend Buckets

| bucket_trend | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| trend_up | 398 | -0.479876 | -0.001206 | 0.957286 | 0.000000 |
| trend_down_or_flat | 202 | -0.087430 | -0.000433 | 0.950495 | 0.004950 |

## Volatility Regime Buckets

| bucket_vol_regime | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| 75-90 | 40 | -0.087981 | -0.002200 | 1.000000 | 0.000000 |
| 25-50 | 79 | -0.146632 | -0.001856 | 0.974684 | 0.000000 |
| >90 | 67 | -0.083106 | -0.001240 | 0.970149 | 0.000000 |
| <=25 | 68 | -0.078567 | -0.001155 | 0.941176 | 0.000000 |
| 50-75 | 157 | -0.112184 | -0.000715 | 0.942675 | 0.000000 |
| nan | 189 | -0.058836 | -0.000311 | 0.947090 | 0.005291 |

## Candidate Risk Gates (Evidence-Driven)

- vol_regime_gate candidate: worst bucket `75-90` expectancy=-0.002200
- trend_gate candidate: worst bucket `trend_up` expectancy=-0.001206
- stop_distance_min candidate: worst bucket `(0.0, 0.0726]` expectancy=-0.000946
- atr_quantile warning: worst bucket `(2.448, 3.937]` expectancy=-0.001576

- CSV stop-distance: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_SOLUSDT_stop_distance.csv`
- CSV ATR quantiles: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_SOLUSDT_atr_quantile.csv`
- CSV trend: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_SOLUSDT_trend.csv`
- CSV volatility regime: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_SOLUSDT_vol_regime.csv`
