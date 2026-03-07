# Phase AG Robustness Report

- Generated UTC: 2026-02-23T12:04:11.935411+00:00
- Classification: **C_AG_FAIL_BRITTLE**

## OOS Route Results

| route_id | policy_id | delta_expectancy_vs_flat | cvar_improve_ratio_vs_flat | maxdd_improve_ratio_vs_flat | min_split_expectancy_net_weighted |
| --- | --- | --- | --- | --- | --- |
| route1_holdout | flat_baseline | 0 | 0 | 0 | -0.0013680194 |
| route1_holdout | P005 | 1.6742764e-05 | -0.059668452 | 0.035557759 | -0.001384879 |
| route1_holdout | P011 | 4.7250701e-05 | -0.036388416 | 0.028725592 | -0.001380662 |
| route1_holdout | P030 | -2.5956648e-05 | -0.17227727 | 0.04457855 | -0.0013067352 |
| route2_reslice | flat_baseline | 0 | 0 | 0 | -0.0010923311 |
| route2_reslice | P005 | 5.9424934e-05 | -0.032529253 | 0.016018147 | -0.0011272512 |
| route2_reslice | P011 | 1.6523097e-05 | -0.028865815 | 0.014686219 | -0.0011118351 |
| route2_reslice | P030 | -5.8454644e-06 | -0.028754808 | 0.05350315 | -0.0011154948 |

## Stress Matrix

| policy_id | scenario_id | delta_expectancy_vs_flat | cvar_improve_ratio_vs_flat | maxdd_improve_ratio_vs_flat | budget_norm_error |
| --- | --- | --- | --- | --- | --- |
| P005 | baseline | 4.7621254e-05 | -0.031252847 | 0.017479107 | 0 |
| P005 | fee_mult_1p25 | 4.7477271e-05 | -0.030401578 | 0.01914572 | 0 |
| P005 | fee_mult_1p50 | 4.7333288e-05 | -0.029762285 | 0.029228337 | 0 |
| P005 | score_noise_0p05 | 1.640404e-05 | -0.034449017 | 0.016783796 | 0 |
| P005 | cap_tight_0p6_1p4 | 4.7621254e-05 | -0.031252847 | 0.017479107 | 0 |
| P005 | slope_plus10pct | 5.7951148e-05 | -0.040477973 | 0.021406648 | 0 |
| P005 | slope_minus10pct | 3.6709409e-05 | -0.021508008 | 0.013330303 | 2.220446e-16 |
| P011 | baseline | 2.8295151e-05 | -0.025564298 | 0.015688272 | 0 |
| P011 | fee_mult_1p25 | 2.8130949e-05 | -0.025544367 | 0.010672668 | 0 |
| P011 | fee_mult_1p50 | 2.7966747e-05 | -0.025529399 | 0.0086968773 | 0 |
| P011 | score_noise_0p05 | 1.2872946e-05 | -0.033411626 | 0.015277047 | 0 |
| P011 | cap_tight_0p6_1p4 | 2.8295151e-05 | -0.025564298 | 0.015688272 | 0 |
| P011 | slope_plus10pct | 4.2014872e-05 | -0.037396699 | 0.023511279 | 0 |
| P011 | slope_minus10pct | 1.3923625e-05 | -0.013169756 | 0.0074936065 | 2.220446e-16 |
| P030 | baseline | 5.8639273e-05 | -0.044537524 | 0.051957425 | 0 |
| P030 | fee_mult_1p25 | 5.8459803e-05 | -0.040739319 | 0.023227137 | 0 |
| P030 | fee_mult_1p50 | 5.8280333e-05 | -0.038797267 | 0.0021829331 | 0 |
| P030 | score_noise_0p05 | 5.5464176e-05 | -0.042154677 | 0.051810794 | 0 |
| P030 | cap_tight_0p6_1p4 | 5.8639273e-05 | -0.044537524 | 0.051957425 | 0 |
| P030 | slope_plus10pct | 6.2272303e-05 | -0.050257213 | 0.053271515 | 0 |
| P030 | slope_minus10pct | 5.4985692e-05 | -0.039013621 | 0.050635901 | 0 |

## Policy Robustness Summary

| policy_id | route_pass_count | route_total | stress_retention |
| --- | --- | --- | --- |
| P030 | 0 | 2 | 0 |
| P011 | 0 | 2 | 0 |
| P005 | 0 | 2 | 0 |
