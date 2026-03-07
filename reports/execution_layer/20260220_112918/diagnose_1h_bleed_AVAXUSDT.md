# Diagnose 1h Bleed: AVAXUSDT

- Generated UTC: 2026-02-20T11:29:21.394654+00:00
- Source summary CSV: `/root/analysis/0.87/reports/execution_layer/20260220_112651_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_summary.csv`
- Source test-signals CSV: `/root/analysis/0.87/reports/execution_layer/20260220_112651_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_signals.csv`
- Params file: `/root/analysis/0.87/data/metadata/params/AVAXUSDT_C13_active_params_long.json`
- Valid baseline entries analyzed: 487

## Stop Distance Buckets

| bucket_stop_distance | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| nan | 487 | -0.251234 | -0.000516 | 0.948665 | 0.020534 |

## ATR 1h Quantile Buckets

| bucket_atr_1h | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| (0.454, 1.419] | 85 | -0.153508 | -0.001806 | 0.964706 | 0.000000 |
| (0.107, 0.261] | 85 | -0.067175 | -0.000790 | 0.964706 | 0.023529 |
| (0.261, 0.32] | 85 | -0.045930 | -0.000540 | 0.941176 | 0.000000 |
| nan | 147 | -0.009977 | -0.000068 | 0.952381 | 0.034014 |
| (0.32, 0.454] | 85 | 0.025356 | 0.000298 | 0.917647 | 0.035294 |

## Trend Buckets

| bucket_trend | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| trend_up | 314 | -0.184069 | -0.000586 | 0.942675 | 0.015924 |
| trend_down_or_flat | 173 | -0.067165 | -0.000388 | 0.959538 | 0.028902 |

## Volatility Regime Buckets

| bucket_vol_regime | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| <=25 | 113 | -0.187140 | -0.001656 | 0.982301 | 0.000000 |
| >90 | 48 | -0.076859 | -0.001601 | 0.979167 | 0.000000 |
| 25-50 | 90 | -0.007167 | -0.000080 | 0.922222 | 0.033333 |
| nan | 147 | -0.009977 | -0.000068 | 0.952381 | 0.034014 |
| 75-90 | 31 | 0.006240 | 0.000201 | 0.903226 | 0.032258 |
| 50-75 | 58 | 0.023670 | 0.000408 | 0.913793 | 0.017241 |

## Candidate Risk Gates (Evidence-Driven)

- vol_regime_gate candidate: worst bucket `<=25` expectancy=-0.001656
- trend_gate candidate: worst bucket `trend_up` expectancy=-0.000586
- stop_distance_min candidate: worst bucket `nan` expectancy=-0.000516
- atr_quantile warning: worst bucket `(0.454, 1.419]` expectancy=-0.001806

- CSV stop-distance: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_AVAXUSDT_stop_distance.csv`
- CSV ATR quantiles: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_AVAXUSDT_atr_quantile.csv`
- CSV trend: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_AVAXUSDT_trend.csv`
- CSV volatility regime: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_AVAXUSDT_vol_regime.csv`
