# Diagnose 1h Bleed: NEARUSDT

- Generated UTC: 2026-02-20T11:29:22.481091+00:00
- Source summary CSV: `/root/analysis/0.87/reports/execution_layer/20260220_112659_walkforward_NEARUSDT/NEARUSDT_walkforward_test_summary.csv`
- Source test-signals CSV: `/root/analysis/0.87/reports/execution_layer/20260220_112659_walkforward_NEARUSDT/NEARUSDT_walkforward_test_signals.csv`
- Params file: `/root/analysis/0.87/data/metadata/params/NEARUSDT_C13_active_params_long.json`
- Valid baseline entries analyzed: 414

## Stop Distance Buckets

| bucket_stop_distance | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| nan | 414 | -0.452457 | -0.001093 | 0.968599 | 0.002415 |

## ATR 1h Quantile Buckets

| bucket_atr_1h | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| (0.104, 0.309] | 104 | -0.208726 | -0.002007 | 0.990385 | 0.000000 |
| (0.0599, 0.104] | 103 | -0.166189 | -0.001613 | 0.990291 | 0.000000 |
| (0.012899999999999998, 0.0378] | 104 | -0.079893 | -0.000768 | 0.942308 | 0.000000 |
| (0.0378, 0.0599] | 103 | 0.002351 | 0.000023 | 0.951456 | 0.009709 |

## Trend Buckets

| bucket_trend | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| trend_down_or_flat | 12 | -0.026394 | -0.002200 | 1.000000 | 0.000000 |
| trend_up | 402 | -0.426062 | -0.001060 | 0.967662 | 0.002488 |

## Volatility Regime Buckets

| bucket_vol_regime | trades | net_pnl_sum | net_expectancy | sl_hit_rate | tp_hit_rate |
|---|---|---|---|---|---|
| >90 | 84 | -0.184760 | -0.002200 | 1.000000 | 0.000000 |
| 75-90 | 56 | -0.087075 | -0.001555 | 0.982143 | 0.000000 |
| <=25 | 111 | -0.164477 | -0.001482 | 0.972973 | 0.000000 |
| 25-50 | 89 | -0.098521 | -0.001107 | 0.966292 | 0.000000 |
| 50-75 | 74 | 0.082376 | 0.001113 | 0.918919 | 0.013514 |

## Candidate Risk Gates (Evidence-Driven)

- vol_regime_gate candidate: worst bucket `>90` expectancy=-0.002200
- trend_gate candidate: worst bucket `trend_down_or_flat` expectancy=-0.002200
- stop_distance_min candidate: worst bucket `nan` expectancy=-0.001093
- atr_quantile warning: worst bucket `(0.104, 0.309]` expectancy=-0.002007

- CSV stop-distance: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_NEARUSDT_stop_distance.csv`
- CSV ATR quantiles: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_NEARUSDT_atr_quantile.csv`
- CSV trend: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_NEARUSDT_trend.csv`
- CSV volatility regime: `/root/analysis/0.87/reports/execution_layer/20260220_112918/diagnose_1h_bleed_NEARUSDT_vol_regime.csv`
