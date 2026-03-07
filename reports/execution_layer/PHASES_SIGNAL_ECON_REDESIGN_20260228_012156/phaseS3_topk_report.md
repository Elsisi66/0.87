# S3 Top-K Report

- Generated UTC: 2026-02-28T01:22:41.912250+00:00
- Prototype count: `12`
- S3 GO candidates: `1`

| prototype_id | family_id | valid_for_ranking | delta_expectancy_vs_exec_baseline | cvar_improve_ratio | maxdd_improve_ratio | route_pass_rate | center_route_delta_min | support_min_entries_route | decision_fail_tags |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| center_guard_moderate | CENTER_GUARD | 1 | 6.496012168e-05 | 0 | 0.007746685507 | 0.6666666667 | 0.00173298755 | 202 |  |
| tail_penalized_moderate | TAIL_PENALIZED | 1 | 6.904276669e-05 | 0 | 0.01549337101 | 0.3333333333 | 0.001730286135 | 200 | route_center_collapse|route_survivability_not_improved |
| tail_penalized_conservative | TAIL_PENALIZED | 1 | 6.477764943e-05 | 0 | 0.007746685507 | 0.3333333333 | 0.001700289672 | 202 | route_center_collapse|route_survivability_not_improved |
| center_guard_conservative | CENTER_GUARD | 1 | 6.070647253e-05 | 0 | 0 | 0.3333333333 | 0.001716482907 | 202 | maxdd_nonpositive|route_center_collapse|route_survivability_not_improved |
| ref_ae_h2 | REFERENCE | 1 | 5.692584142e-05 | 0 | 0.04648011304 | 0.3333333333 | 0.001884556278 | 186 | route_center_collapse|support_starvation|route_survivability_not_improved |
| baseline_active | REFERENCE | 1 | 5.600601457e-05 | 0 | 0 | 0.3333333333 | 0.00165538302 | 204 | maxdd_nonpositive|route_center_collapse|route_survivability_not_improved |
| ref_ae_h1 | REFERENCE | 1 | 4.962738084e-05 | -0.005460246217 | 0.04757099594 | 0 | 0.001648245172 | 197 | cvar_negative|route_center_collapse|support_starvation|route_survivability_not_improved |
| ref_ae_h3 | REFERENCE | 1 | 4.382788647e-05 | 0 | 0.03098674203 | 0.3333333333 | 0.001673426943 | 190 | route_center_collapse|support_starvation|route_survivability_not_improved |
| cost_defense_moderate | COST_DEFENSE | 1 | 9.646497167e-06 | 0 | 0.007746685507 | 0.3333333333 | 0.001600208294 | 202 | route_center_collapse|route_survivability_not_improved |
| cost_defense_conservative | COST_DEFENSE | 1 | 5.571118746e-06 | 0 | -7.417647587e-16 | 0.3333333333 | 0.001585958073 | 203 | maxdd_nonpositive|route_center_collapse|route_survivability_not_improved |
| fill_rescue_conservative | FILL_RESCUE | 1 | -2.302823299e-06 | 0 | -8.901177104e-16 | 0.3333333333 | 0.001587634242 | 202 | delta_nonpositive|maxdd_nonpositive|route_center_collapse|route_survivability_not_improved |
| fill_rescue_moderate | FILL_RESCUE | 1 | -2.309291904e-06 | 0 | -8.901177104e-16 | 0.3333333333 | 0.001609656182 | 201 | delta_nonpositive|maxdd_nonpositive|route_center_collapse|route_survivability_not_improved |
