# A3 Model A Report

- Generated UTC: 2026-02-28T01:50:29.274080+00:00
- Candidate rows: `5`
- Non-reference rows: `4`

## Results

| candidate_id | valid_for_ranking | exec_expectancy_net | delta_expectancy_vs_1h_reference | cvar_improve_ratio | maxdd_improve_ratio | entry_rate | entries_valid | taker_share | route_pass | center_route_delta | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M0_1H_REFERENCE | 1 | -0.0009179256926 | 0 | 0 | 0 | 1 | 360 | 1 | 1 | 0 |  |
| M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.001167739031 | 0.002085664724 | 0.07067856575 | 0.7032237657 | 1 | 360 | 0.03333333333 | 1 | 0.003345326573 |  |
| M2_ENTRY_ONLY_MORE_PASSIVE | 1 | 0.0009979046213 | 0.001915830314 | 0.07951338646 | 0.7035408049 | 1 | 360 | 0.03055555556 | 1 | 0.003130053339 |  |
| M3_ENTRY_ONLY_FASTER | 1 | 0.001210006245 | 0.002127931938 | 0.08834820718 | 0.7030546976 | 1 | 360 | 0.025 | 1 | 0.003360739849 |  |
| M4_ENTRY_ONLY_MARKET_CONTROL | 0 | 0.001045857291 | 0.001963782983 | 0 | 0.6425977764 | 1 | 360 | 1 | 0 | 0.003330944546 | SOLUSDT:taker_share |
