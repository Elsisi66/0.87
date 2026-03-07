# Signal Layer Diagnostics

- Generated UTC: 2026-02-22T02:20:16.466931+00:00
- E2 standard dir: `/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052`
- Contract hash fee/metrics: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a` / `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
- Representative subset size/hash: 1200 / `fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d`

## Fixed Variant Headline Metrics

- V2R expectancy/return/maxDD/cvar5/PF: -0.001027 / -1.000000 / -1.000000 / -0.002200 / 0.514693
- V3R expectancy/return/maxDD/cvar5/PF: -0.000984 / -0.999999 / -0.999999 / -0.002283 / 0.534313
- V4R expectancy/return/maxDD/cvar5/PF: -0.000868 / -1.000000 / -1.000000 / -0.002037 / 0.539177

## Regime Edge (1h reference)

Best regimes:
| regime_bucket | signals_total | trades_total | expectancy_net | win_rate | profit_factor | max_drawdown_pct | cvar_5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low|up | 259 | 259 | -0.000164 | 0.061776 | 0.920314 | -0.870506 | -0.002200 |
| unknown|down | 466 | 466 | -0.001073 | 0.036481 | 0.493748 | -0.998090 | -0.002200 |
| mid|up | 262 | 262 | -0.001323 | 0.030534 | 0.379572 | -0.978285 | -0.002200 |

Worst regimes:
| regime_bucket | signals_total | trades_total | expectancy_net | win_rate | profit_factor | max_drawdown_pct | cvar_5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| low|down | 10 | 10 | -0.002200 | 0.000000 | 0.000000 | -0.181405 | -0.002200 |
| high|down | 13 | 13 | -0.002200 | 0.000000 | 0.000000 | -0.234241 | -0.002200 |
| mid|down | 10 | 10 | -0.002200 | 0.000000 | 0.000000 | -0.181405 | -0.002200 |

## Signal Clustering

| metric | value | notes |
| --- | --- | --- |
| signals_total | 1200.000000 |  |
| median_gap_min | 180.000000 | inter-signal spacing |
| p95_gap_min | 4806.000000 | inter-signal spacing |
| signals_per_day_median | 3.000000 | burst density |
| signals_per_day_p95 | 10.000000 | burst density |
| max_direction_streak | 1200.000000 | same-direction streak |
| adverse_regime_share | 0.565833 | share of signals in adverse regime |
| expectancy_adverse | -0.001243 | 1h ref |
| expectancy_non_adverse | -0.000747 | 1h ref |

## Entry Delay Sensitivity

| variant | signals_total | trades_total | expectancy_net | total_return | max_drawdown_pct | cvar_5 | profit_factor | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_next_3m_open | 1200 | 1200 | -0.001027 | -1.000000 | -1.000000 | -0.002200 | 0.514693 | 0.037500 |
| next_1h_open_plus_0bar_delay | 1200 | 1200 | -0.001264 | -1.000000 | -1.000000 | -0.002200 | 0.405539 | 0.033333 |
| next_1h_open_plus_1bar_delay | 1200 | 1200 | -0.000817 | -0.999998 | -0.999998 | -0.002200 | 0.613282 | 0.039167 |
| next_1h_open_plus_2bar_delay | 1200 | 1200 | -0.001008 | -1.000000 | -1.000000 | -0.002200 | 0.525495 | 0.034167 |

## Hold Horizon Sensitivity

| horizon_hours | signals_total | trades_total | expectancy_net | total_return | max_drawdown_pct | cvar_5 | profit_factor | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.000000 | 1200.000000 | 1200.000000 | -0.000970 | -0.999999 | -0.999999 | -0.002299 | 0.535863 | 0.052500 |
| 12.000000 | 1200.000000 | 1200.000000 | -0.001027 | -1.000000 | -1.000000 | -0.002200 | 0.514693 | 0.037500 |
| 18.000000 | 1200.000000 | 1200.000000 | -0.001040 | -1.000000 | -1.000000 | -0.002200 | 0.512870 | 0.029167 |
| 24.000000 | 1200.000000 | 1200.000000 | -0.000898 | -1.000000 | -1.000000 | -0.002200 | 0.580851 | 0.025833 |
| 36.000000 | 1200.000000 | 1200.000000 | -0.000870 | -1.000000 | -1.000000 | -0.002200 | 0.595773 | 0.021667 |
| 48.000000 | 1200.000000 | 1200.000000 | -0.000775 | -0.999999 | -0.999999 | -0.002200 | 0.639905 | 0.021667 |

## Ablations

| ablation | signals_total | signals_share_of_rep | trades_total | expectancy_net | total_return | max_drawdown_pct | cvar_5 | profit_factor | win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cycle_detection_only | 1200 | 1.000000 | 1200 | -0.001027 | -1.000000 | -1.000000 | -0.002200 | 0.514693 | 0.037500 |
| cycle_plus_min_filters | 621 | 0.517500 | 621 | -0.001135 | -0.999729 | -0.999723 | -0.002200 | 0.465981 | 0.033816 |
| cycle_plus_min_filters_cooldown4h | 348 | 0.290000 | 348 | -0.001123 | -0.988567 | -0.988553 | -0.002200 | 0.469424 | 0.037356 |
| full_signal_params | 1200 | 1.000000 | 1200 | -0.001027 | -1.000000 | -1.000000 | -0.002200 | 0.514693 | 0.037500 |

## Tail / Distribution Diagnostics

- Tail cutoff (bottom decile pnl_net_pct): -0.002200
- R reach >=0.5 / >=1.0 / >=1.5: 0.9258 / 0.8792 / 0.8142

Top tail-loss regimes:
| regime_bucket | tail_loss_count |
| --- | --- |
| unknown|down | 112 |
| low|up | 73 |
| mid|up | 57 |
| high|up | 39 |
| low|down | 4 |
