# Phase AF Pilot Report

- Generated UTC: 2026-02-23T12:06:22.209827+00:00
- Classification: **B_AF_GO_WEAK_BUT_PROMISING**
- Policies evaluated (excl. flat): `66`

## Top Policies

| policy_id | policy_family | delta_expectancy_vs_flatsize_baseline | cvar_improve_ratio_vs_flat | maxdd_improve_ratio_vs_flat | weighted_loss_run_burden_reduction_vs_flat | budget_norm_error | candidate_budget_norm_pass | candidate_invariance_pass | backup_delta_expectancy_vs_flat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P030 | hybrid | 5.8639273e-05 | -0.044537524 | 0.051957425 | 0.0030734685 | 0 | 1 | 1 | 5.8725829e-05 |
| flat_baseline | flat | 0 | 0 | 0 | 0 | 0 | 1 | 1 | nan |
| P056 | streak | -2.3332883e-05 | -0.052409179 | 0.047701299 | 0.0077377339 | 2.220446e-16 | 1 | 1 | -2.3282071e-05 |
| P011 | step | 2.8295151e-05 | -0.025564298 | 0.015688272 | 0.0006789747 | 0 | 1 | 1 | 2.8333877e-05 |
| P005 | step | 4.7621254e-05 | -0.031252847 | 0.017479107 | 0.00020825118 | 0 | 1 | 1 | 4.7719709e-05 |
| P038 | streak | -2.6033188e-05 | -0.076168008 | 0.05322175 | 0.0086325571 | 0 | 1 | 1 | -2.5976495e-05 |
| P001 | streak | -5.5709336e-05 | -0.080749609 | 0.05778389 | 0.0078077521 | 0 | 1 | 1 | nan |
| P012 | step | 6.8071935e-05 | -0.063746614 | 0.038012605 | 0.00034068841 | 0 | 1 | 1 | nan |
| P039 | step | 8.176696e-05 | -0.094427075 | 0.066140716 | 0.0027513671 | 0 | 1 | 1 | nan |
| P024 | smooth | 4.4440177e-05 | -0.038060672 | 0.0091270817 | -0.00032308181 | 0 | 1 | 1 | nan |

## Baseline / S1 / Best

| policy_id | exec_expectancy_net_weighted | delta_expectancy_vs_flatsize_baseline | cvar_improve_ratio_vs_flat | maxdd_improve_ratio_vs_flat | max_consecutive_losses | streak_ge5_count | weighted_loss_run_burden | entry_rate | budget_norm_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P030 | 0.00011513118 | 5.8639273e-05 | -0.044537524 | 0.051957425 | 33 | 20 | 0.96597949 | 0.98611111 | 0 |
| flat_baseline | 5.6491907e-05 | 0 | 0 | 0 | 33 | 20 | 0.96895755 | 0.98611111 | 0 |
| S1_anchor | 0.00012821234 | 7.1720435e-05 | -0.10941358 | 0.0099253653 | 33 | 20 | 0.97224905 | 0.98611111 | 0 |
| streak_only_control | -6.0479974e-05 | -0.00011697188 | -0.30343114 | 0.1119046 | 33 | 20 | 0.95299478 | 0.98611111 | -0.0094361172 |
