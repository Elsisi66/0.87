# Diagnose 1h Bleed: SOLUSDT

- Generated UTC: 2026-02-20T11:34:10.941141+00:00
- Source walkforward dir: `/root/analysis/0.87/reports/execution_layer/20260220_111706_walkforward_SOLUSDT`
- Baseline valid test entries: 600
- Baseline net expectancy: -0.000946
- Baseline SL-hit rate: 0.955000

## Stop Distance Quantiles

| stop_distance_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| (0.0, 0.0726] | 600 | -0.000946 | -0.567306 | 0.955000 | 0.001667 | 1.000000 |

## ATR(1h) Quantiles

| atr_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| (2.448, 3.937] | 103 | -0.001576 | -0.162288 | 0.980583 | 0.000000 | 0.171667 |
| (1.983, 2.448] | 102 | -0.001460 | -0.148952 | 0.970588 | 0.000000 | 0.170000 |
| (1.0330000000000001, 1.689] | 103 | -0.001247 | -0.128421 | 0.941748 | 0.000000 | 0.171667 |
| (1.689, 1.983] | 103 | -0.000668 | -0.068809 | 0.941748 | 0.000000 | 0.171667 |
| nan | 189 | -0.000311 | -0.058836 | 0.947090 | 0.005291 | 0.315000 |

## Trend Proxy

| trend_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| trend_up | 398 | -0.001206 | -0.479876 | 0.957286 | 0.000000 | 0.663333 |
| trend_down_or_flat | 202 | -0.000433 | -0.087430 | 0.950495 | 0.004950 | 0.336667 |

## Vol Regime (ATR Percentile)

| vol_regime_bucket | trades | expectancy_net | pnl_net_sum | sl_hit_rate | tp_hit_rate | share |
|---|---|---|---|---|---|---|
| p80_90 | 23 | -0.002200 | -0.050589 | 1.000000 | 0.000000 | 0.038333 |
| p00_50 | 147 | -0.001532 | -0.225200 | 0.959184 | 0.000000 | 0.245000 |
| p90_100 | 67 | -0.001240 | -0.083106 | 0.970149 | 0.000000 | 0.111667 |
| p50_80 | 174 | -0.000860 | -0.149576 | 0.948276 | 0.000000 | 0.290000 |
| nan | 189 | -0.000311 | -0.058836 | 0.947090 | 0.005291 | 0.315000 |

## Suggested Gates

- recommend_vol_regime_gate: 1
- vol_regime_max_percentile: 90.0
- recommend_trend_gate: 0
- trend_fast_col/trend_slow_col: EMA_50/EMA_120
- recommend_stop_distance_min: 0
- stop_distance_min_pct: nan
- notes: High ATR percentile bucket (p90_100) materially underperforms.

## CSV Outputs

- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_SOLUSDT_stop_distance.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_SOLUSDT_atr_quantile.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_SOLUSDT_trend.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_113409/diagnose_1h_bleed_SOLUSDT_vol_regime.csv`
