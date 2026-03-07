# Phase G1 Label Dictionary

- `signal_id`, `signal_time`: 1h signal identity/time from frozen representative subset.
- `exec_choice_id`: execution policy (`E1` or `E2`) used for downstream labeling.
- `route_id`: route partition (`route1_holdout`, `route2_reslice`).
- `subperiod_id`: chronological thirds within route for robustness slicing.
- `entry_for_labels`: 1 if execution outcome valid for label metrics.
- `pnl_net_trade_notional_dec`: downstream executed net return.
- `y_toxic_trade`: AE composite toxic label (tail OR cluster OR large adverse excursion).
- `y_cluster_loss`, `y_tail_loss`, `y_sl_loss`: cluster/tail/SL-specific labels.
- `prior_loss_streak_len`, `prior_rolling_tail_count_20`, `prior_rolling_loss_rate_5`: prior-only context labels.
- `fragility_long_delay`, `fragility_taker`, `fragility_fee_dominated`: execution fragility markers.
- `legacy_score_raw`: current/legacy 1h rank proxy from 1h indicators.
- `f1h_*`: baseline 1h indicator metadata aligned at signal time.
