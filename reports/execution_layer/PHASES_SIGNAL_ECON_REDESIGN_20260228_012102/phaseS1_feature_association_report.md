# S1 Feature Association Report

- Generated UTC: 2026-02-28T01:21:41.916742+00:00
- Ranked likely upstream causes:
  1. Center-route weakness is dominated by fill-risk / delay-risk clusters.
  2. Front-route weakness is dominated by taker-risk / spread-cost bursts.
  3. Cost-killed gross edge remains material in stressed microstructure buckets.
  4. Tail weakness clusters around high ATR-z / impulsive bar conditions.

## toxic_candidate

| feature | corr | mean_when_flag_1 | mean_when_flag_0 | mean_diff |
| --- | --- | --- | --- | --- |
| pre3m_ret_3bars | nan | 2.335496814e-05 | nan | nan |
| pre3m_realized_vol_12 | nan | 0.001395947714 | nan | nan |
| pre3m_atr_z | nan | -0.171172859 | nan | nan |
| pre3m_spread_proxy_bps | nan | 25.98352156 | nan | nan |
| pre3m_body_bps_abs | nan | 14.15908106 | nan | nan |
| pre3m_wick_ratio | nan | 4259259263 | nan | nan |
| pre3m_impulse_atr | nan | 0.6527944052 | nan | nan |
| pre3m_trend_slope_12 | nan | 0.002877600794 | nan | nan |

## weak_post_exec_delta

| feature | corr | mean_when_flag_1 | mean_when_flag_0 | mean_diff |
| --- | --- | --- | --- | --- |
| pre3m_close_to_low_dist_bps | 0.1290706125 | 20.703017 | 12.35543177 | 8.347585232 |
| pre3m_impulse_atr | 0.1126113348 | 0.9279742657 | 0.6403669276 | 0.2876073381 |
| pre3m_upbar_ratio_12 | -0.1091116437 | 0.4220779221 | 0.4998533724 | -0.07777545036 |
| pre3m_ret_3bars | -0.08036142985 | -0.0009424111431 | 6.697021187e-05 | -0.001009381355 |
| pre3m_body_bps_abs | 0.07809628295 | 19.28875858 | 13.92741821 | 5.361340372 |
| pre3m_close_to_high_dist_bps | -0.07451369874 | 7.920204704 | 13.50887753 | -5.588672826 |
| pre3m_trend_slope_12 | -0.06701742832 | -0.02508741259 | 0.004140536882 | -0.02922794947 |
| pre3m_atr_z | -0.05367764846 | -0.5103339338 | -0.1558559072 | -0.3544780267 |

## center_route_failure_bucket

| feature | corr | mean_when_flag_1 | mean_when_flag_0 | mean_diff |
| --- | --- | --- | --- | --- |
| pre3m_spread_proxy_bps | 0.3504772654 | 54.22082999 | 24.3712772 | 29.84955279 |
| est_taker_risk_proxy | 0.3302960207 | 0.3428571429 | 0.03099510604 | 0.3118620368 |
| pre3m_body_bps_abs | 0.329686221 | 33.41835375 | 13.05944886 | 20.3589049 |
| pre3m_close_to_low_dist_bps | 0.2443588079 | 26.1641255 | 11.9482998 | 14.21582569 |
| pre3m_close_to_high_dist_bps | 0.2317284525 | 28.05670449 | 12.4229774 | 15.63372709 |
| pre3m_realized_vol_12 | 0.2221607241 | 0.002115852543 | 0.001354843849 | 0.0007610086934 |
| pre3m_impulse_atr | 0.1856221926 | 1.056201771 | 0.6297613582 | 0.4264404132 |
| est_fill_delay_risk_proxy | 0.167161619 | 0.5714285714 | 0.2169657423 | 0.3544628292 |

## negative_cvar_contrib

| feature | corr | mean_when_flag_1 | mean_when_flag_0 | mean_diff |
| --- | --- | --- | --- | --- |
| est_taker_risk_proxy | -0.07804372787 | 0.01875 | 0.05737704918 | -0.03862704918 |
| pre3m_close_to_high_dist_bps | -0.07137886592 | 11.36634849 | 13.89068451 | -2.524336029 |
| est_fill_delay_risk_proxy | -0.06553771496 | 0.18125 | 0.2540983607 | -0.07284836066 |
| pre3m_spread_proxy_bps | -0.04302005185 | 24.53712267 | 26.45775071 | -1.920628035 |
| pre3m_body_bps_abs | -0.0408424299 | 13.16343642 | 14.48552193 | -1.322085513 |
| pre3m_atr_z | -0.03304221287 | -0.2486573414 | -0.1457681106 | -0.1028892308 |
| pre3m_upbar_ratio_12 | 0.02283538374 | 0.5022727273 | 0.4945976155 | 0.007675111773 |
| pre3m_close_to_low_dist_bps | 0.01979650288 | 13.17077419 | 12.56706619 | 0.6037079938 |
