# Phase J02 Data Quality Report

- Generated UTC: 2026-02-24T02:53:13.875240+00:00
- Trade-feature rows: `984`
- Unique signals: `420`
- Routes: `['route1_holdout', 'route2_reslice']`
- Exec choices: `['E1', 'E2']`

## Baseline by route/choice

| exec_choice_id | route_id | exec_expectancy_net | exec_cvar_5 | exec_max_drawdown | entries_valid | entry_rate | taker_share | p95_fill_delay_min | max_consecutive_losses | streak_ge5_count | streak_ge10_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | route1_holdout | 6.468296e-06 | -0.0015367996 | -0.049991127 | 70 | 0.97222222 | 0.042857143 | 15.3 | 28 | 5 | 2 |
| E1 | route2_reslice | 0.00027128995 | -0.0017991816 | -0.18691674 | 408 | 0.97142857 | 0.088235294 | 18 | 33 | 22 | 14 |
| E2 | route1_holdout | 6.468296e-06 | -0.0015367996 | -0.049991127 | 70 | 0.97222222 | 0.042857143 | 15.3 | 28 | 5 | 2 |
| E2 | route2_reslice | 0.00027045699 | -0.0017991816 | -0.18726658 | 408 | 0.97142857 | 0.090686275 | 18 | 33 | 22 | 14 |

## Missingness

| column | missing_rate |
| --- | --- |
| prior_rolling_loss_rate_5 | 0.032520325 |
| prior_rolling_tail_count_20 | 0.032520325 |
| pnl_net_trade_notional_dec | 0.028455285 |
| prior_loss_streak_len | 0.028455285 |
| entry_time_utc | 0.028455285 |
| signal_time_utc | 0 |
| split_id | 0 |
| entry_for_labels | 0 |
| signal_id | 0 |
| pre3m_close_to_high_dist_bps | 0 |
| pre3m_realized_vol_12 | 0 |
| pre3m_wick_ratio | 0 |
| route_id | 0 |
| exec_choice_id | 0 |

## Join/Leakage checks

- Timestamp columns normalized via existing validated parsers (`phase_ae_signal_labeling` + `phase_af_ah_sizing_autorun`).
- Prior-state features are inherited from validated AE pipeline and remain pre-entry/prior-only.
