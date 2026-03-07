# Diagnose 1h Bleed: AVAXUSDT

- Generated UTC: 2026-02-20T11:34:12.258390+00:00
- Source walkforward dir: `/root/analysis/0.87/reports/execution_layer/20260220_112651_walkforward_AVAXUSDT`
- Baseline valid test entries: 487
- Baseline net expectancy: -0.000516
- Baseline SL-hit rate: 0.948665

## Stop Distance Quantiles

| stop_distance_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| all | 487 | -0.000516 | -0.251234 | 0.948665 | 0.020534 | 1.000000 |

## ATR(1h) Quantiles

| atr_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| (0.454, 1.419] | 85 | -0.001806 | -0.153508 | 0.964706 | 0.000000 | 0.174538 |
| (0.107, 0.261] | 85 | -0.000790 | -0.067175 | 0.964706 | 0.023529 | 0.174538 |
| (0.261, 0.32] | 85 | -0.000540 | -0.045930 | 0.941176 | 0.000000 | 0.174538 |
| nan | 147 | -0.000068 | -0.009977 | 0.952381 | 0.034014 | 0.301848 |
| (0.32, 0.454] | 85 | 0.000298 | 0.025356 | 0.917647 | 0.035294 | 0.174538 |

## Trend Proxy

| trend_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| trend_up | 314 | -0.000586 | -0.184069 | 0.942675 | 0.015924 | 0.644764 |
| trend_down_or_flat | 173 | -0.000388 | -0.067165 | 0.959538 | 0.028902 | 0.355236 |

## Vol Regime (ATR Percentile)

| vol_regime_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| p80_90 | 17 | -0.002022 | -0.034381 | 0.941176 | 0.000000 | 0.034908 |
| p90_100 | 48 | -0.001601 | -0.076859 | 0.979167 | 0.000000 | 0.098563 |
| p00_50 | 203 | -0.000957 | -0.194307 | 0.955665 | 0.014778 | 0.416838 |
| nan | 147 | -0.000068 | -0.009977 | 0.952381 | 0.034014 | 0.301848 |
| p50_80 | 72 | 0.000893 | 0.064290 | 0.902778 | 0.027778 | 0.147844 |

## Suggested Gates

- recommend_vol_regime_gate: 1
- vol_regime_max_percentile: 90.0
- recommend_trend_gate: 0
- trend_fast_col/trend_slow_col: EMA_50/EMA_120
- recommend_stop_distance_min: 0
- stop_distance_min_pct: nan
- notes: High ATR percentile bucket (p90_100) materially underperforms.

## CSV Outputs

- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_AVAXUSDT_stop_distance.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_AVAXUSDT_atr_quantile.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_AVAXUSDT_trend.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_AVAXUSDT_vol_regime.csv`
