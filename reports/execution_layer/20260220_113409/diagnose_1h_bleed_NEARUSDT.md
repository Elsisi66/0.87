# Diagnose 1h Bleed: NEARUSDT

- Generated UTC: 2026-02-20T11:34:13.610982+00:00
- Source walkforward dir: `/root/analysis/0.87/reports/execution_layer/20260220_113038_walkforward_NEARUSDT`
- Baseline valid test entries: 414
- Baseline net expectancy: -0.001093
- Baseline SL-hit rate: 0.968599

## Stop Distance Quantiles

| stop_distance_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| all | 414 | -0.001093 | -0.452457 | 0.968599 | 0.002415 | 1.000000 |

## ATR(1h) Quantiles

| atr_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| (0.104, 0.309] | 104 | -0.002007 | -0.208726 | 0.990385 | 0.000000 | 0.251208 |
| (0.0599, 0.104] | 103 | -0.001613 | -0.166189 | 0.990291 | 0.000000 | 0.248792 |
| (0.012899999999999998, 0.0378] | 104 | -0.000768 | -0.079893 | 0.942308 | 0.000000 | 0.251208 |
| (0.0378, 0.0599] | 103 | 0.000023 | 0.002351 | 0.951456 | 0.009709 | 0.248792 |

## Trend Proxy

| trend_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| trend_down_or_flat | 12 | -0.002200 | -0.026394 | 1.000000 | 0.000000 | 0.028986 |
| trend_up | 402 | -0.001060 | -0.426062 | 0.967662 | 0.002488 | 0.971014 |

## Vol Regime (ATR Percentile)

| vol_regime_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| p90_100 | 84 | -0.002200 | -0.184760 | 1.000000 | 0.000000 | 0.202899 |
| p00_50 | 200 | -0.001315 | -0.262998 | 0.970000 | 0.000000 | 0.483092 |
| p80_90 | 37 | -0.001224 | -0.045284 | 0.972973 | 0.000000 | 0.089372 |
| p50_80 | 93 | 0.000436 | 0.040585 | 0.935484 | 0.010753 | 0.224638 |

## Suggested Gates

- recommend_vol_regime_gate: 1
- vol_regime_max_percentile: 90.0
- recommend_trend_gate: 0
- trend_fast_col/trend_slow_col: EMA_50/EMA_120
- recommend_stop_distance_min: 0
- stop_distance_min_pct: nan
- notes: High ATR percentile bucket (p90_100) materially underperforms.

## CSV Outputs

- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_NEARUSDT_stop_distance.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_NEARUSDT_atr_quantile.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_NEARUSDT_trend.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_NEARUSDT_vol_regime.csv`
