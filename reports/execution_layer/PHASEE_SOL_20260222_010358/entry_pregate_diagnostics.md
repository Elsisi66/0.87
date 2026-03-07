# Entry Pre-Gate Diagnostics

- Generated UTC: 2026-02-22T01:04:19.735527+00:00
- Symbol: SOLUSDT
- Configs generated: 1944
- Configs evaluated: 480
- Sampling method: deterministic_random_sample_plus_baseline
- Sampling seed: 42

## Control Metrics (Phase C fixed-exit, no entry gate)

- expectancy_net: -0.000559
- pnl_net_sum: -0.335322
- cvar5: -0.001950
- max_drawdown: -0.518469
- trades_total: 600

## Best Candidate

- config_id: 9e16d9ae4f163245
- config: `{"config_id": "9e16d9ae4f163245", "cooldown_h": 0, "max_signals_24h": 999, "overlap_h": 0, "session_mode": "all", "stop_distance_min": 0.001, "trend_required": 1, "vol_max_pct": 100.0, "vol_min_pct": 5.0}`
- delta_expectancy: 0.000049
- delta_maxdd: 0.155162
- delta_cvar5: -0.000000
- entry_rate: 0.5883
- trades_total: 353
- split_median_expectancy_delta: 0.000213
- pass_all: 1

## Top 10 configs

| trend_required | vol_min_pct | vol_max_pct | cooldown_h | max_signals_24h | session_mode | overlap_h | stop_distance_min | config_id | signals_total | trades_total | entry_rate | expectancy_net | expectancy_net_per_signal | pnl_net_sum | cvar_5 | max_drawdown | win_rate | sl_hit_rate | tp_hit_rate | timeout_rate | min_split_trades | median_split_trades | split_median_expectancy_delta | split_min_expectancy_delta | delta_expectancy_best_entry_pregate_minus_phasec_control | delta_maxdd_best_entry_pregate_minus_phasec_control | delta_cvar5_best_entry_pregate_minus_phasec_control | delta_pnl_sum_best_entry_pregate_minus_phasec_control | pass_expectancy | pass_split_median | pass_maxdd_not_worse | pass_cvar_not_worse | pass_participation | pass_min_split_support | pass_data_quality | pass_reproducibility | pass_all | fail_reasons | skip_reasons_json | top_skip_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.000000 | 100.000000 | 0 | 999 | all | 0 | 0.001000 | 9e16d9ae4f163245 | 600 | 353 | 0.588333 | -0.000510 | -0.000300 | -0.180159 | -0.001950 | -0.363306 | 0.045326 | 0.954674 | 0.002833 | 0.042493 | 58 | 70.000000 | 0.000213 | -0.000498 | 0.000049 | 0.155162 | -0.000000 | 0.155162 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | none | {"trend_gate": 213, "vol_gate": 34} | trend_gate:213, vol_gate:34 |
| 0 | 5.000000 | 100.000000 | 0 | 999 | all | 0 | 0.000000 | e72fa884a66918c0 | 600 | 365 | 0.608333 | -0.000558 | -0.000339 | -0.203555 | -0.001950 | -0.386702 | 0.043836 | 0.956164 | 0.002740 | 0.041096 | 62 | 73.000000 | 0.000122 | -0.000503 | 0.000001 | 0.131767 | -0.000000 | 0.131767 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | none | {"vol_gate": 235} | vol_gate:235 |
| 0 | 5.000000 | 100.000000 | 0 | 999 | all | 0 | 0.001000 | 378a5610e5d0cccf | 600 | 365 | 0.608333 | -0.000558 | -0.000339 | -0.203555 | -0.001950 | -0.386702 | 0.043836 | 0.956164 | 0.002740 | 0.041096 | 62 | 73.000000 | 0.000122 | -0.000503 | 0.000001 | 0.131767 | -0.000000 | 0.131767 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | none | {"vol_gate": 235} | vol_gate:235 |
| 1 | 20.000000 | 85.000000 | 2 | 8 | all | 2 | 0.001000 | d08cb36a3e15e3a0 | 600 | 93 | 0.155000 | -0.000147 | -0.000023 | -0.013690 | -0.001950 | -0.047240 | 0.075269 | 0.924731 | 0.000000 | 0.075269 | 11 | 21.000000 | 0.000670 | -0.000888 | 0.000412 | 0.471229 | -0.000000 | 0.321631 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | participation_low,min_split_support_low | {"anti_chop_gate": 159, "cooldown_gate": 45, "trend_gate": 213, "vol_gate": 90} | trend_gate:213, anti_chop_gate:159, vol_gate:90 |
| 1 | 10.000000 | 85.000000 | 4 | 8 | all | 2 | 0.001000 | f26257a0a8765589 | 600 | 76 | 0.126667 | -0.000171 | -0.000022 | -0.012995 | -0.001950 | -0.046791 | 0.078947 | 0.921053 | 0.000000 | 0.078947 | 10 | 16.000000 | 0.000612 | -0.001731 | 0.000388 | 0.471678 | 0.000000 | 0.322327 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | participation_low,min_split_support_low | {"anti_chop_gate": 169, "cooldown_gate": 74, "trend_gate": 213, "vol_gate": 68} | trend_gate:213, anti_chop_gate:169, cooldown_gate:74 |
| 1 | 10.000000 | 85.000000 | 0 | 8 | all | 2 | 0.000000 | afa32b6f6dbb1402 | 600 | 100 | 0.166667 | -0.000177 | -0.000030 | -0.017750 | -0.001950 | -0.047240 | 0.080000 | 0.920000 | 0.000000 | 0.080000 | 11 | 22.000000 | 0.000844 | -0.000991 | 0.000381 | 0.471229 | -0.000000 | 0.317572 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | participation_low,min_split_support_low | {"anti_chop_gate": 169, "overlap_gate": 50, "trend_gate": 213, "vol_gate": 68} | trend_gate:213, anti_chop_gate:169, vol_gate:68 |
| 1 | 20.000000 | 85.000000 | 4 | 8 | all | 2 | 0.000000 | f9bb72810df85830 | 600 | 71 | 0.118333 | -0.000181 | -0.000021 | -0.012834 | -0.001950 | -0.046791 | 0.070423 | 0.929577 | 0.000000 | 0.070423 | 10 | 14.000000 | 0.000812 | -0.001643 | 0.000378 | 0.471678 | -0.000000 | 0.322487 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | participation_low,min_split_support_low | {"anti_chop_gate": 159, "cooldown_gate": 67, "trend_gate": 213, "vol_gate": 90} | trend_gate:213, anti_chop_gate:159, vol_gate:90 |
| 0 | 20.000000 | 85.000000 | 2 | 8 | all | 2 | 0.000000 | 28287128f4383f8d | 600 | 99 | 0.165000 | -0.000256 | -0.000042 | -0.025388 | -0.001950 | -0.048197 | 0.070707 | 0.929293 | 0.000000 | 0.070707 | 11 | 21.000000 | 0.000426 | -0.000888 | 0.000302 | 0.470272 | -0.000000 | 0.309934 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | participation_low,min_split_support_low | {"anti_chop_gate": 162, "cooldown_gate": 47, "vol_gate": 292} | vol_gate:292, anti_chop_gate:162, cooldown_gate:47 |
| 0 | 20.000000 | 85.000000 | 4 | 8 | all | 0 | 0.000000 | 5ea165a98dde45ff | 600 | 75 | 0.125000 | -0.000275 | -0.000034 | -0.020633 | -0.001950 | -0.046791 | 0.066667 | 0.933333 | 0.000000 | 0.066667 | 10 | 14.000000 | 0.000812 | -0.001643 | 0.000284 | 0.471678 | -0.000000 | 0.314689 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | participation_low,min_split_support_low | {"anti_chop_gate": 162, "cooldown_gate": 71, "vol_gate": 292} | vol_gate:292, anti_chop_gate:162, cooldown_gate:71 |
| 0 | 20.000000 | 85.000000 | 4 | 8 | all | 2 | 0.001000 | 8113c8438ef260a8 | 600 | 75 | 0.125000 | -0.000275 | -0.000034 | -0.020633 | -0.001950 | -0.046791 | 0.066667 | 0.933333 | 0.000000 | 0.066667 | 10 | 14.000000 | 0.000812 | -0.001643 | 0.000284 | 0.471678 | -0.000000 | 0.314689 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | participation_low,min_split_support_low | {"anti_chop_gate": 162, "cooldown_gate": 71, "vol_gate": 292} | vol_gate:292, anti_chop_gate:162, cooldown_gate:71 |

## Fail Reason Histogram

| reason | count |
| --- | --- |
| participation_low | 469 |
| min_split_support_low | 439 |
| split_median_negative | 432 |
| expectancy_nonpos | 410 |
| cvar_worse_than_tol | 159 |
| none | 3 |
