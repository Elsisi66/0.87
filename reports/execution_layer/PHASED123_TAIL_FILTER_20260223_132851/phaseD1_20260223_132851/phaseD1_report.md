# Phase D1 Report

- Generated UTC: 2026-02-23T13:29:28.330930+00:00
- Decision: **PASS**
- Reason: clear worst-route tail driver identified
- Source Phase C root: `/root/analysis/0.87/reports/execution_layer/PHASEABC_LABEL_REPAIR_20260223_131344`
- Source phaseC dir: `/root/analysis/0.87/reports/execution_layer/PHASEABC_LABEL_REPAIR_20260223_131344/phaseC_20260223_131446`
- Worst route by CVaR: `route2_reslice`
- Worst route CVaR5: `-0.00179918`
- Top-2 risk-decile CVaR loss share (worst route): `0.428571`

## Route-level summary from D1

| route_id | entries | mean_pnl | cvar5 | tail_cut10 | cvar5_cut |
| --- | --- | --- | --- | --- | --- |
| route1_holdout | 70 | 6.6531045e-06 | -0.0015367996 | -0.001449339 | -0.001449339 |
| route2_reslice | 408 | 0.00027926907 | -0.0017991816 | -0.001449339 | -0.0016767367 |

## Tail attribution by route/decile (head)

| route_id | risk_decile | support | tail10_count | tail10_rate | cvar5_count | cvar5_rate | tail10_loss_share | cvar5_loss_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| route1_holdout | (0.0938, 0.125] | 7 | 2 | 0.28571429 | 2 | 0.28571429 | 0.1107326 | 0.1107326 |
| route1_holdout | (0.125, 0.151] | 7 | 1 | 0.14285714 | 1 | 0.14285714 | 0.049403744 | 0.049403744 |
| route1_holdout | (0.151, 0.203] | 7 | 3 | 0.42857143 | 3 | 0.42857143 | 0.14821123 | 0.14821123 |
| route1_holdout | (0.203, 0.263] | 7 | 3 | 0.42857143 | 3 | 0.42857143 | 0.14821123 | 0.14821123 |
| route1_holdout | (0.263, 0.336] | 7 | 2 | 0.28571429 | 2 | 0.28571429 | 0.098807488 | 0.098807488 |
| route1_holdout | (0.336, 0.371] | 7 | 0 | 0 | 0 | 0 | 0 | 0 |
| route1_holdout | (0.371, 0.411] | 7 | 3 | 0.42857143 | 3 | 0.42857143 | 0.14821123 | 0.14821123 |
| route1_holdout | (0.411, 0.495] | 7 | 2 | 0.28571429 | 2 | 0.28571429 | 0.098807488 | 0.098807488 |
| route1_holdout | (0.495, 0.537] | 7 | 2 | 0.28571429 | 2 | 0.28571429 | 0.098807488 | 0.098807488 |
| route1_holdout | (0.537, 0.971] | 7 | 2 | 0.28571429 | 2 | 0.28571429 | 0.098807488 | 0.098807488 |
| route2_reslice | (0.015299999999999998, 0.173] | 41 | 14 | 0.34146341 | 5 | 0.12195122 | 0.11258619 | 0.23809524 |
| route2_reslice | (0.173, 0.227] | 41 | 16 | 0.3902439 | 3 | 0.073170732 | 0.12381925 | 0.14285714 |
| route2_reslice | (0.227, 0.279] | 41 | 14 | 0.34146341 | 4 | 0.097560976 | 0.1107991 | 0.19047619 |
| route2_reslice | (0.279, 0.329] | 40 | 10 | 0.25 | 3 | 0.075 | 0.079397513 | 0.14285714 |
| route2_reslice | (0.329, 0.37] | 41 | 14 | 0.34146341 | 2 | 0.048780488 | 0.10722491 | 0.095238095 |
| route2_reslice | (0.37, 0.401] | 41 | 12 | 0.29268293 | 0 | 0 | 0.088843482 | 0 |
| route2_reslice | (0.401, 0.447] | 40 | 13 | 0.325 | 0 | 0 | 0.096247105 | 0 |
| route2_reslice | (0.447, 0.49] | 41 | 11 | 0.26829268 | 1 | 0.024390244 | 0.083226951 | 0.047619048 |
| route2_reslice | (0.49, 0.558] | 41 | 14 | 0.34146341 | 3 | 0.073170732 | 0.10901201 | 0.14285714 |
| route2_reslice | (0.558, 0.892] | 41 | 12 | 0.29268293 | 0 | 0 | 0.088843482 | 0 |
