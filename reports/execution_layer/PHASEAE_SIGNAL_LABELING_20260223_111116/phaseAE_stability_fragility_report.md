# Phase AE Stability and Fragility Report

- Generated UTC: 2026-02-23T11:12:17.819395+00:00
- Leakage check: excluded post-entry outcomes (`pnl`, `exit_reason`, `mae/mfe`, fill results) from feature library.
- Split stability rule-of-thumb: `stable_sign_frac >= 0.60` considered stable.

## Top Numeric Features (y_toxic_trade)

| feature | support | delta_event_rate | risk_ratio | direction | stable_sign_frac | bin_summary |
| --- | --- | --- | --- | --- | --- | --- |
| atr_percentile_1h | 238 | 0.29787234 | 1.4375 | high_risk_high_value | 0.5 | (1.447, 18.929]:48/0.9167; (18.929, 40.216]:47/0.6809; (40.216, 55.564]:48/0.8542; (55.564, 67.967]:47/0.9787; (67.967, 88.955]:48/0.9583 |
| prior_rolling_tail_count_20 | 354 | 0.24 | 1.3157895 | high_risk_high_value | 0.75 | (0.999, 4.0]:79/0.8481; (4.0, 6.0]:119/0.8824; (6.0, 7.0]:73/0.8630; (7.0, 8.0]:50/0.7600; (8.0, 11.0]:33/1.0000 |
| prior_loss_streak_len | 355 | 0.20793651 | 1.2793177 | high_risk_high_value | 0.75 | (-0.001, 2.0]:90/0.7444; (10.0, 17.0]:63/0.9524; (17.0, 33.0]:70/0.9000; (2.0, 5.0]:66/0.8333; (5.0, 10.0]:66/0.9394 |
| pre3m_close_to_high_dist_bps | 355 | 0.18309859 | 1.2363636 | high_risk_high_value | 1 | (-0.001, 3.354]:71/0.7746; (12.991, 20.159]:71/0.9296; (20.159, 58.187]:71/0.9577; (3.354, 8.1]:71/0.8732; (8.1, 12.991]:71/0.7887 |
| pre3m_upbar_ratio_12 | 355 | 0.14914377 | 1.1858561 | high_risk_high_value | 1 | (0.0899, 0.364]:81/0.8025; (0.364, 0.455]:93/0.8387; (0.455, 0.545]:86/0.8721; (0.545, 0.636]:62/0.9516; (0.636, 0.909]:33/0.9091 |
| pre3m_realized_vol_12 | 355 | 0.14084507 | 1.1818182 | high_risk_high_value | 0.6 | (-0.000555, 0.00089]:71/0.8873; (0.00089, 0.00111]:71/0.8451; (0.00111, 0.00134]:71/0.7746; (0.00134, 0.00165]:71/0.9155; (0.00165, 0.00597]:71/0.9014 |
| pre3m_trend_slope_12 | 355 | 0.14084507 | 1.1785714 | high_risk_high_value | 0.8 | (-0.0103, 0.0221]:71/0.7887; (-0.0524, -0.0103]:71/0.8732; (-0.292, -0.0524]:71/0.8451; (0.0221, 0.0644]:71/0.8873; (0.0644, 0.278]:71/0.9296 |
| pre3m_impulse_atr | 355 | 0.12676056 | 1.1607143 | high_risk_high_value | 0.8 | (-0.001, 0.245]:71/0.7887; (0.245, 0.402]:71/0.9155; (0.402, 0.61]:71/0.8592; (0.61, 0.897]:71/0.8592; (0.897, 2.737]:71/0.9014 |
| prior_rolling_tail_count_10 | 354 | 0.1246142 | 1.1552885 | high_risk_low_value | 0.8 | (-0.001, 2.0]:125/0.8560; (2.0, 3.0]:96/0.9271; (3.0, 4.0]:81/0.8025; (4.0, 7.0]:52/0.8654 |
| pre3m_close_to_low_dist_bps | 355 | 0.11267606 | 1.1403509 | high_risk_low_value | 0.8 | (-0.001, 1.987]:71/0.8732; (1.987, 5.496]:71/0.9014; (10.682, 16.591]:71/0.8028; (16.591, 65.615]:71/0.8310; (5.496, 10.682]:71/0.9155 |

## Fragile Effects (split-unstable)

| feature | support | delta_event_rate | stable_sign_frac |
| --- | --- | --- | --- |
| atr_percentile_1h | 238 | 0.29787234 | 0.5 |

## Backup Candidate (E2) Direction Confirmation

| feature | e1_direction | e2_delta_event_rate | e2_direction_consistent | e2_support |
| --- | --- | --- | --- | --- |
| atr_percentile_1h | high_risk_high_value | 0.041666667 | 1 | 238 |
| prior_rolling_tail_count_20 | high_risk_high_value | 0.0073204209 | 1 | 354 |
| prior_loss_streak_len | high_risk_high_value | 0.16464646 | 1 | 355 |
| pre3m_close_to_high_dist_bps | high_risk_high_value | 0.18309859 | 1 | 355 |
| pre3m_upbar_ratio_12 | high_risk_high_value | 0.13437297 | 1 | 355 |
