- Furthest phase reached: D3
- Classification at furthest phase: PASS_GO_PAPER_CANDIDATES
- Mainline status: CONTINUE

- What was proven (1 paragraph plain English):
D3 found at least one strict multi-route passer under unchanged hard gates.

- Top candidates (exact metrics):
| policy_id | policy_type | policy_hash | routes_tested | min_delta_expectancy_vs_flat | min_cvar_improve_ratio_vs_flat | min_maxdd_improve_ratio_vs_flat | mean_delta_expectancy_vs_flat | mean_cvar_improve_ratio_vs_flat | mean_maxdd_improve_ratio_vs_flat | min_entries_valid | min_entry_rate | min_filter_kept_entries_pct | no_pathology | strict_pass | rank_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skip_streak5_cool60m | cooldown_skip | d25328a0269b637804fc7ae3 | 2 | 0.00014507789 | 0 | 0.14307857 | 0.00015665835 | 0.018518612 | 0.20184392 | 61 | 0.79285714 | 0.81617647 | 1 | 1 | 0.28747443 |
| skip_streak4_cool60m | cooldown_skip | 311f06b9d64f880c21d20e16 | 2 | 8.028531e-05 | 0 | 0.14307857 | 0.00014439177 | 0.018518612 | 0.20572089 | 59 | 0.77142857 | 0.79411765 | 1 | 1 | 0.28694382 |
| skip_risk_ge_0.70 | risk_skip | f5149047a987af087ace682e | 2 | 1.4824264e-05 | 0 | 0.010048185 | 3.7606694e-05 | 0 | 0.048511979 | 67 | 0.93055556 | 0.95714286 | 1 | 1 | 0.020252571 |
| skip_streak_ge_3 | streak_skip | f0dd31000b7c93c61e4914a9 | 2 | -0.00041865059 | 0.056910905 | 0.57451805 | -0.00037033487 | 0.097900247 | 0.60050389 | 20 | 0.26190476 | 0.26960784 | 1 | 0 | 1.2591384 |
| skip_streak_ge_4 | streak_skip | b2649693e7927962ad8dd81d | 2 | -0.00041809381 | 0 | 0.43614598 | -0.00034359023 | 0.064815141 | 0.55775113 | 25 | 0.33095238 | 0.34068627 | 1 | 0 | 0.86860361 |

- Failure branch taken (if any):
none

- Is next phase justified? (yes/no)
yes

- Artifact directory (exact path)
/root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324

- Key files list
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseA/phaseA_decision.md
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseA/phaseA_freeze_lock_validation.json
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseA/phaseA_report.md
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseA/phaseA_run_manifest.json
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD1/phaseD1_report.md
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD1/phaseD1_run_manifest.json
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD1/tail_attribution_by_route.csv
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD1/tail_concentration_by_score_bucket.csv
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD2/phaseD2_report.md
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD2/split_stability_tail.json
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD2/tail_label.csv
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD2/tail_risk_score.csv
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD3/invalid_reason_histogram.json
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD3/phaseD3_decision.md
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD3/phaseD3_report.md
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/phaseD3/phaseD3_results.csv
- /root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324/run_manifest.json

- Exact next prompt contents (only if justified)
Phase E paper/shadow confirmation (contract-locked): validate D3 strict-pass filter policy set under route perturbation and stress scenarios. Keep hard gates unchanged, include rollback triggers, and do not run full GA.
