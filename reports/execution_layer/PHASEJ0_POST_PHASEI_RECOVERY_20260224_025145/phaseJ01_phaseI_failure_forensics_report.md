# Phase J01 Phase I Failure Forensics Report

- Generated UTC: 2026-02-24T02:51:45.409063+00:00
- Source Phase I dir: `/root/analysis/0.87/reports/execution_layer/PHASEI_EXECAWARE_1H_GA_EXPANSION_20260224_012237`
- Robustness-tested candidates: `9`
- Decision excerpt: `- Classification: **I_NO_GO_FRAGILE_LUCKY_POINTS**`
- Duplicate ratio among valid (I2): `0.777893`
- Effective trials corr-adjusted (I2): `1.017375`
- Avg abs metric corr (I2): `0.982842`

## Top Candidate-Level Failure Modes

| scope_id | fail_count | fail_pct | support_n |
| --- | --- | --- | --- |
| fail_route | 9 | 1 | 9 |
| fail_split | 9 | 1 | 9 |
| fail_stress | 9 | 1 | 9 |
| fail_bootstrap | 9 | 1 | 9 |
| fail_tail_sign | 2 | 0.22222222 | 9 |
| fail_dd_sign | 0 | 0 | 9 |

## Route Failure Concentration

| scope_id | fail_count | fail_pct | support_n |
| --- | --- | --- | --- |
| route1_holdout | 9 | 1 | 9 |
| route2_reslice | 9 | 1 | 9 |

## Stress Scenario Failure Concentration

| scope_id | fail_count | fail_pct | support_n |
| --- | --- | --- | --- |
| S00_base | 9 | 1 | 9 |
| S01_cost125 | 9 | 1 | 9 |
| S02_cost150 | 9 | 1 | 9 |
| S03_slip_p1 | 9 | 1 | 9 |
| S04_slip_p2 | 9 | 1 | 9 |
| S05_lat_entry1 | 9 | 1 | 9 |
| S07_cost125_slip1 | 9 | 1 | 9 |
| S08_trim_head10 | 9 | 1 | 9 |
| S09_trim_tail10 | 9 | 1 | 9 |
| S06_spread15 | 6 | 0.66666667 | 9 |

## Lucky-Point Signatures (head)

| candidate_id | seed_origin | near_h0313_clone | OJ2 | delta_expectancy_vs_exec_baseline | cvar_improve_ratio | maxdd_improve_ratio | entries_valid | entry_rate | route_pass_rate | stress_pass_rate | bootstrap_pass_rate | min_subperiod_delta | min_subperiod_cvar | lucky_point_score | sig_in_sample_outlier | sig_route_fragile | sig_split_fragile | sig_stress_fragile | sig_bootstrap_weak | sig_negative_tail_metric |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I00339 | mutation | False | 0.15390253 | 0.00077682333 | 0.039021456 | 0.45858587 | 205 | 0.99033816 | 0 | 0 | 0.0033333333 | 0 | 0 | 5 | 1 | 1 | 1 | 1 | 1 | 0 |
| I00873 | crossover | True | 0.15340676 | 0.0008731439 | -0.0054602462 | 0.51591569 | 207 | 0.99043062 | 0 | 0.1 | 0 | -0.00099775406 | -0.10769291 | 5 | 0 | 1 | 1 | 1 | 1 | 1 |
| I00509 | crossover | True | 0.15049682 | 0.00084894096 | -0.0054602462 | 0.50629911 | 209 | 0.99052133 | 0 | 0.1 | 0 | -0.00099775406 | -0.10769291 | 5 | 0 | 1 | 1 | 1 | 1 | 1 |
| I00253 | crossover | True | 0.14263908 | 0.00086904659 | 0.0034747021 | 0.46814364 | 200 | 0.98522167 | 0 | 0 | 0 | -0.0010907374 | -0.053846454 | 4 | 0 | 1 | 1 | 1 | 1 | 0 |
| I00442 | mutation | False | 0.14263908 | 0.00086904659 | 0.0034747021 | 0.46814364 | 200 | 0.98522167 | 0 | 0 | 0.013333333 | -0.0010907374 | -0.053846454 | 4 | 0 | 1 | 1 | 1 | 1 | 0 |
| I00002 | near_H0313 | True | 0.14254212 | 0.0007724109 | 0.0034747021 | 0.46814364 | 200 | 0.98522167 | 0 | 0 | 0.0033333333 | -0.0010907374 | -0.053846454 | 4 | 0 | 1 | 1 | 1 | 1 | 0 |
| I00283 | crossover | True | 0.14192737 | 0.00085242236 | 0.0034747021 | 0.46582357 | 200 | 0.99009901 | 0 | 0.1 | 0 | -0.00023000333 | -0.053846454 | 4 | 0 | 1 | 1 | 1 | 1 | 0 |
| I00297 | mutation | False | 0.14154774 | 0.00074255501 | 0.021248079 | 0.44122259 | 208 | 0.99047619 | 0 | 0 | 0.0033333333 | -0.00072356848 | -3.0853556e-14 | 4 | 0 | 1 | 1 | 1 | 1 | 0 |
| I00000 | phaseH_seed_H0020 | False | 0.064104247 | 0.00014809671 | 0.00068253078 | 0.21248278 | 305 | 0.98705502 | 0 | 0 | 0 | -0.0016419045 | -0.071795272 | 4 | 0 | 1 | 1 | 1 | 1 | 0 |
