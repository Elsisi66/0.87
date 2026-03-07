# Phase AE Feature Dictionary

- Generated UTC: 2026-02-23T11:12:15.275426+00:00
- Leakage policy: features are restricted to signal-time / pre-entry context; no post-entry outcome fields are included.

## A_1h_signal_context

- `tp_mult`: Signal TP multiplier from 1h layer
- `sl_mult`: Signal SL multiplier from 1h layer
- `atr_percentile_1h`: 1h ATR percentile at signal time
- `trend_up_1h`: 1h trend direction/probability proxy
- `vol_bucket`: Derived 1h volatility regime bucket
- `trend_bucket`: Derived 1h trend regime bucket

## B_3m_preentry_context

- `pre3m_ret_3bars`: 3m return over last ~3 bars before signal
- `pre3m_realized_vol_12`: Realized vol over last ~12 bars (prior-only)
- `pre3m_atr_z`: ATR z-score using prior history
- `pre3m_spread_proxy_bps`: Spread proxy bps at decision bar
- `pre3m_body_bps_abs`: Absolute body size bps at decision bar
- `pre3m_wick_ratio`: Wick ratio proxy at decision bar
- `pre3m_impulse_atr`: Decision-bar impulse normalized by ATR
- `pre3m_trend_slope_12`: Linear slope over prior closes
- `pre3m_accel_6v6`: Acceleration: last6-return mean minus prior6
- `pre3m_upbar_ratio_12`: Up-bar ratio over prior window
- `pre3m_close_to_high_dist_bps`: Distance of close to high
- `pre3m_close_to_low_dist_bps`: Distance of close to low

## C_interaction_context

- `prior_loss_streak_len`: Prior-only current loss streak length
- `prior_rolling_loss_rate_5`: Prior-only rolling loss rate (5)
- `prior_rolling_loss_rate_10`: Prior-only rolling loss rate (10)
- `prior_rolling_loss_rate_20`: Prior-only rolling loss rate (20)
- `prior_rolling_tail_count_5`: Prior-only tail-loss count (5)
- `prior_rolling_tail_count_10`: Prior-only tail-loss count (10)
- `prior_rolling_tail_count_20`: Prior-only tail-loss count (20)

## D_operational_proxies

- `est_limit_distance_bps`: Configured limit distance proxy
- `est_fill_window_bars`: Configured max fill window in bars
- `est_fallback_window_bars`: Configured fallback delay in bars
- `est_taker_risk_proxy`: Pre-entry taker-risk proxy
- `est_fill_delay_risk_proxy`: Pre-entry fill-delay risk proxy
- `cfg_spread_guard_bps`: Execution spread guard setting
- `cfg_vol_threshold`: Execution vol-threshold setting
- `cfg_micro_vol_filter`: Execution micro-vol gate enabled
- `cfg_use_signal_quality_gate`: Execution signal-quality gate enabled
- `cfg_min_signal_quality_gate`: Execution signal-quality threshold
