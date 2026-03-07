# Phase F1 Forensics Report

- Generated UTC: 2026-02-23T19:38:04.274604+00:00
- Source Phase E run: `/root/analysis/0.87/reports/execution_layer/PHASEE_PAPER_CONFIRM_20260223_185139`
- E2 rows: `15`
- E3 rows: `36`
- E3 by-route rows: `72`

## Top Failure Causes (aggregate)

| reason | fail_count |
| --- | --- |
| fail_subperiod_delta | 36 |
| fail_dd | 0 |
| fail_cvar | 0 |
| fail_delta | 0 |
| fail_participation | 0 |
| fail_pathology | 0 |
| fail_subperiod_cvar | 0 |

## Scenario-level breakdown (head)

| scope_id | reason | fail_count | fail_pct | support_n |
| --- | --- | --- | --- | --- |
| S00_base | fail_cvar | 0 | 0 | 3 |
| S00_base | fail_dd | 0 | 0 | 3 |
| S00_base | fail_delta | 0 | 0 | 3 |
| S00_base | fail_participation | 0 | 0 | 3 |
| S00_base | fail_pathology | 0 | 0 | 3 |
| S00_base | fail_subperiod_cvar | 0 | 0 | 3 |
| S00_base | fail_subperiod_delta | 3 | 1 | 3 |
| S01_cost125 | fail_cvar | 0 | 0 | 3 |
| S01_cost125 | fail_dd | 0 | 0 | 3 |
| S01_cost125 | fail_delta | 0 | 0 | 3 |
| S01_cost125 | fail_participation | 0 | 0 | 3 |
| S01_cost125 | fail_pathology | 0 | 0 | 3 |
| S01_cost125 | fail_subperiod_cvar | 0 | 0 | 3 |
| S01_cost125 | fail_subperiod_delta | 3 | 1 | 3 |
| S02_cost150 | fail_cvar | 0 | 0 | 3 |
| S02_cost150 | fail_dd | 0 | 0 | 3 |
| S02_cost150 | fail_delta | 0 | 0 | 3 |
| S02_cost150 | fail_participation | 0 | 0 | 3 |
| S02_cost150 | fail_pathology | 0 | 0 | 3 |
| S02_cost150 | fail_subperiod_cvar | 0 | 0 | 3 |
| S02_cost150 | fail_subperiod_delta | 3 | 1 | 3 |
| S03_slip_p1 | fail_cvar | 0 | 0 | 3 |
| S03_slip_p1 | fail_dd | 0 | 0 | 3 |
| S03_slip_p1 | fail_delta | 0 | 0 | 3 |
| S03_slip_p1 | fail_participation | 0 | 0 | 3 |
| S03_slip_p1 | fail_pathology | 0 | 0 | 3 |
| S03_slip_p1 | fail_subperiod_cvar | 0 | 0 | 3 |
| S03_slip_p1 | fail_subperiod_delta | 3 | 1 | 3 |
| S04_slip_p2 | fail_cvar | 0 | 0 | 3 |
| S04_slip_p2 | fail_dd | 0 | 0 | 3 |

## Policy attribution

| policy_id | family_hint | risk_threshold | streak_depth | cooldown_min | scenario_pass_rate | mean_min_delta | mean_min_cvar | mean_min_dd | min_subperiod_delta | min_subperiod_cvar | min_kept_entries_pct | base_strict_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| skip_risk_ge_0.70 | risk_skip | 0.7 | nan | nan | 0 | 2.1104762e-05 | 0.00027346645 | 0.017331267 | 0 | 0 | 0.95714286 | 1 |
| skip_streak4_cool60m | cooldown_skip | nan | 4 | 60 | 0 | 0.00015931173 | 0 | 0.14152024 | -0.00027052039 | 0 | 0.79411765 | 1 |
| skip_streak5_cool60m | cooldown_skip | nan | 5 | 60 | 0 | 0.00020107748 | 0 | 0.14023157 | -4.5085376e-05 | 0 | 0.81617647 | 1 |
