# LINK Soft Repair Confirmation

- Generated UTC: `2026-03-03T02:03:48.606555+00:00`
- Artifact dir: `/root/analysis/0.87/reports/execution_layer/LINK_SOFT_REPAIR_CONFIRMATION_20260303_020324`

## A) Discovered Code Paths / Artifacts Used

- Repaired 1h baseline dir: `/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650`
- Rebased Model A dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108`
- Entry-quality forensic dir: `/root/analysis/0.87/reports/execution_layer/INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_20260303_012335`
- Latest bounded entry-repair pilot dir: `/root/analysis/0.87/reports/execution_layer/LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_20260303_013409`
- Retention-cliff forensic dir: `/root/analysis/0.87/reports/execution_layer/LINK_RETENTION_CLIFF_FORENSICS_20260303_014856`
- Gap metric definition path: `/root/analysis/0.87/scripts/instant_loser_vs_winner_entry_forensics.py`
- Positive-gap percentile filter reconstruction path: `/root/analysis/0.87/scripts/link_retention_cliff_forensics.py`
- Acceptance logic reused from: `/root/analysis/0.87/scripts/live_coin_bounded_entry_repair_pilot.py`
- Route robustness machinery reused from: `/root/analysis/0.87/scripts/phase_r_route_harness_redesign.py`

## B) Reconstruction Validity Check

| variant_id | reconstruction_valid | gap_metric_definition_match | acceptance_status | trade_count_retention_vs_control |
| --- | --- | --- | --- | --- |
| PCTL_DROP_TOP10_POSGAP | 1 | 1 | 1 | 0.951724 |
| PCTL_DROP_TOP25_POSGAP | 1 | 1 | 1 | 0.886207 |

## C) Strict Confirmation Results Table

| variant_id | acceptance_status | expectancy_delta_vs_control | cvar_delta_vs_control | maxdd_delta_vs_control | trade_count_retention_vs_control | valid_for_ranking | robustness_label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCTL_DROP_TOP10_POSGAP | 1 | 8.92977e-05 | 0 | 0.00184974 | 0.951724 | 1 | fragile |
| PCTL_DROP_TOP25_POSGAP | 1 | 0.000210487 | 0 | 0.0147979 | 0.886207 | 1 | fragile |

## D) Robustness Summary By Split / Route / Seed / Cluster / Time Slice

| variant_id | confirming_splits | split_total | confirming_routes | route_total | confirming_time_slices | time_slice_total | confirming_seeds | seed_total | loss_cluster_confirm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PCTL_DROP_TOP10_POSGAP | 4 | 5 | 1 | 2 | 3 | 3 | 0 | 1 | 1 |
| PCTL_DROP_TOP25_POSGAP | 5 | 5 | 1 | 2 | 3 | 3 | 0 | 1 | 1 |

## E) Failure Mode Check

- leakage/lookahead: `proven clean` via preserved repaired chronology and `parity_clean=1` for both candidates
- ranking validity: `{'PCTL_DROP_TOP10_POSGAP': 1, 'PCTL_DROP_TOP25_POSGAP': 1}`
- sample collapse: compare retention directly in the summary table
- retention cliff: hard-gap collapse is legacy-only; both soft variants stay above the pilot retention floor
- winner deletion: both candidates remove `0` meaningful winners by reconstructed match to the frozen retention-cliff artifact
- unrealistic execution assumptions: `assumed unchanged` because entry/exit path, cost model, and stop logic were held constant; only the skip mask changed

## F) Conservative Winner Or NO_REPAIR_APPROVED

- Decision: `NO_REPAIR_APPROVED`
- Decision reason: Neither candidate is robust enough beyond point-estimate improvement.

## G) Final LINK Deployment Posture Recommendation

- Recommended posture: `NO_REPAIR_APPROVED`

## H) Exact Next Step

- If the decision is a shadow approval, patch only the LINK paper/runtime branch to run the selected soft filter in shadow mode and log both raw and filtered decisions side-by-side before any full promotion.
- If the decision is NO_REPAIR_APPROVED, stop LINK entry-gap repair and move to a different entry-quality lever.

### Route Family Used

| route_id | route_signal_count | wf_test_signal_count | first_signal_time | last_signal_time |
| --- | --- | --- | --- | --- |
| route_back_60pct | 665 | 200 | 2021-02-04 06:00:00+00:00 | 2025-12-04 17:00:00+00:00 |
| route_center_60pct | 665 | 200 | 2019-10-21 20:00:00+00:00 | 2024-09-19 20:00:00+00:00 |
| route_front_60pct | 665 | 200 | 2019-01-22 20:00:00+00:00 | 2023-10-16 21:00:00+00:00 |