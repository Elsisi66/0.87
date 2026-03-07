# Phase D3 Report

- Generated UTC: 2026-02-23T14:09:25.655052+00:00
- Decision: **PASS_GO_PAPER_CANDIDATES**
- Reason: >=1 strict passer
- Policy budget: `40`
- Evaluated policies (excluding flat): `31`
- Strict passers: `3`

## Top candidates

| policy_id | policy_type | strict_pass | min_delta_expectancy_vs_flat | min_cvar_improve_ratio_vs_flat | min_maxdd_improve_ratio_vs_flat | min_entries_valid | min_filter_kept_entries_pct | rank_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skip_streak5_cool60m | cooldown_skip | 1 | 0.00014507789 | 0 | 0.14307857 | 61 | 0.81617647 | 0.28747443 |
| skip_streak4_cool60m | cooldown_skip | 1 | 8.028531e-05 | 0 | 0.14307857 | 59 | 0.79411765 | 0.28694382 |
| skip_risk_ge_0.70 | risk_skip | 1 | 1.4824264e-05 | 0 | 0.010048185 | 67 | 0.95714286 | 0.020252571 |
| skip_streak_ge_3 | streak_skip | 0 | -0.00041865059 | 0.056910905 | 0.57451805 | 20 | 0.26960784 | 1.2591384 |
| skip_streak_ge_4 | streak_skip | 0 | -0.00041809381 | 0 | 0.43614598 | 25 | 0.34068627 | 0.86860361 |
| skip_streak_ge_6 | streak_skip | 0 | -0.0004214681 | 0 | 0.42436109 | 35 | 0.45588235 | 0.84508985 |
| skip_streak4_cool240m | cooldown_skip | 0 | -0.00044632859 | 0 | 0.33724458 | 42 | 0.54656863 | 0.67052153 |
| skip_streak_ge_5 | streak_skip | 0 | -0.00051874235 | 0 | 0.29118636 | 30 | 0.40196078 | 0.57784012 |
| skip_streak4_cool120m | cooldown_skip | 0 | -0.00023927838 | 0 | 0.26276576 | 49 | 0.67156863 | 0.52359137 |
| skip_streak5_cool120m | cooldown_skip | 0 | -0.00017997062 | 0 | 0.23219796 | 52 | 0.70098039 | 0.46292969 |
| skip_streak5_cool240m | cooldown_skip | 0 | -0.00042819453 | 0 | 0.23098345 | 46 | 0.59068627 | 0.45812552 |
| skip_streak3_cool240m | cooldown_skip | 0 | -0.00060707792 | 0 | 0.20191092 | 38 | 0.48284314 | 0.39852338 |
| skip_tail20_ge_6 | tail_skip | 0 | -0.0006204882 | 0 | 0.18093061 | 52 | 0.34803922 | 0.35674728 |
| skip_tail20_ge_7 | tail_skip | 0 | -0.00062399484 | 0 | 0.17305121 | 62 | 0.53921569 | 0.34086609 |
| skip_streak3_cool120m | cooldown_skip | 0 | -0.00039686205 | 0 | 0.15448926 | 46 | 0.61519608 | 0.30572582 |
| skip_streak3_cool60m | cooldown_skip | 0 | -7.0149543e-05 | 0 | 0.14307857 | 57 | 0.7622549 | 0.28568197 |
| skip_tail20_ge_8 | tail_skip | 0 | -0.00043762264 | 0 | 0.07880658 | 66 | 0.74264706 | 0.15392073 |
| skip_risk0.75_streak3 | risk_streak_combo | 0 | 3.4508071e-06 | 0 | 0 | 67 | 0.95714286 | 5.9526422e-05 |
| skip_risk0.75_streak4 | risk_streak_combo | 0 | 3.4508071e-06 | 0 | 0 | 67 | 0.95714286 | 5.9526422e-05 |
| skip_risk0.75_streak5 | risk_streak_combo | 0 | 3.4508071e-06 | 0 | 0 | 67 | 0.95714286 | 5.9526422e-05 |

## Reality-check

- Reality-check bootstrap: TODO placeholder (deployment-adjacent).
