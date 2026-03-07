# Phase E1 Reproduction Report

- Generated UTC: 2026-02-23T18:52:15.458155+00:00
- Decision: **PASS**
- Reason: top candidate reproduced strict pass
- Phase D source: `/root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324`

## Candidate reproduction table

| policy_id | strict_pass | prior_strict_pass | min_delta_expectancy_vs_flat | prior_min_delta_expectancy_vs_flat | delta_abs_diff | delta_tol | delta_within_tol | min_cvar_improve_ratio_vs_flat | cvar_nonneg | min_maxdd_improve_ratio_vs_flat | maxdd_pos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skip_risk_ge_0.70 | 1 | 1 | 1.4824264e-05 | 1.4824264e-05 | 0 | 5e-05 | 1 | 0 | 1 | 0.010048185 | 1 |
| skip_streak4_cool60m | 1 | 1 | 8.028531e-05 | 8.028531e-05 | 0 | 5e-05 | 1 | 0 | 1 | 0.14307857 | 1 |
| skip_streak5_cool60m | 1 | 1 | 0.00014507789 | 0.00014507789 | 4.0684687e-17 | 5e-05 | 1 | 0 | 1 | 0.14307857 | 1 |
| skip_streak_ge_3 | 0 | 0 | -0.00041865059 | -0.00041865059 | 7.8333607e-17 | 8.3730119e-05 | 1 | 0.056910905 | 1 | 0.57451805 | 1 |
