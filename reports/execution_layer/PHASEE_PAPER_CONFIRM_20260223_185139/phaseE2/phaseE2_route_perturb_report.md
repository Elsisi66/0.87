# Phase E2 Route Perturbation Report

- Generated UTC: 2026-02-23T18:52:16.343510+00:00
- Decision: **PASS**
- Reason: base strict pass + bootstrap >=70% + no route-level CVaR drift below epsilon
- Top candidate `skip_streak5_cool60m` bootstrap pass rate: `0.9700`
- Top candidate min route-level CVaR improve across perturbs: `0.000000`

## Scenario matrix (aggregate)

| scenario_id | policy_id | strict_pass | min_delta_expectancy_vs_flat | min_cvar_improve_ratio_vs_flat | min_maxdd_improve_ratio_vs_flat | min_entries_valid | min_filter_kept_entries_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| base | skip_risk_ge_0.70 | 1 | 1.4824264e-05 | 0 | 0.010048185 | 67 | 0.95714286 |
| maker_bias | skip_risk_ge_0.70 | 1 | 1.280072e-05 | 0 | 0.010082244 | 67 | 0.95714286 |
| taker_bias | skip_risk_ge_0.70 | 1 | 1.8871353e-05 | 0 | 0.022948464 | 67 | 0.95714286 |
| trim_head10 | skip_risk_ge_0.70 | 1 | 1.6471405e-05 | 0 | 0.010048185 | 60 | 0.95238095 |
| trim_tail10 | skip_risk_ge_0.70 | 0 | 0 | 0 | 0 | 63 | 0.99184783 |
| base | skip_streak4_cool60m | 1 | 8.028531e-05 | 0 | 0.14307857 | 59 | 0.79411765 |
| maker_bias | skip_streak4_cool60m | 1 | 4.3104331e-05 | 0 | 0.14592547 | 59 | 0.79411765 |
| taker_bias | skip_streak4_cool60m | 1 | 0.00015464727 | 0 | 0.13921116 | 59 | 0.79411765 |
| trim_head10 | skip_streak4_cool60m | 1 | 0.0001693579 | 0 | 0.14307857 | 55 | 0.77956989 |
| trim_tail10 | skip_streak4_cool60m | 1 | 7.0283521e-05 | 0 | 0.17151228 | 52 | 0.78532609 |
| base | skip_streak5_cool60m | 1 | 0.00014507789 | 0 | 0.14307857 | 61 | 0.81617647 |
| maker_bias | skip_streak5_cool60m | 1 | 0.0001120029 | 0 | 0.14592547 | 61 | 0.81617647 |
| taker_bias | skip_streak5_cool60m | 1 | 0.00021073199 | 0 | 0.13921116 | 61 | 0.81617647 |
| trim_head10 | skip_streak5_cool60m | 1 | 0.00014706037 | 0 | 0.14307857 | 56 | 0.7983871 |
| trim_tail10 | skip_streak5_cool60m | 1 | 0.00014227528 | 0 | 0.17151228 | 54 | 0.80978261 |
