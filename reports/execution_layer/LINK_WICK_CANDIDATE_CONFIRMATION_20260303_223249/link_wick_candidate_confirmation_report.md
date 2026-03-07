# LINK Wick Candidate Confirmation

- Generated UTC: `2026-03-03T22:33:14.612974+00:00`
- Artifact dir: `/root/analysis/0.87/reports/execution_layer/LINK_WICK_CANDIDATE_CONFIRMATION_20260303_223249`

## A) Discovered Code Paths / Artifacts Used

- Repaired 1h baseline dir: `/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650`
- Rebased multicoin Model A dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108`
- Winner-vs-loser forensic dir: `/root/analysis/0.87/reports/execution_layer/INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_20260303_012335`
- Latest bounded entry-repair pilot dir: `/root/analysis/0.87/reports/execution_layer/LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_20260303_013409`
- Rejected soft-gap confirmation dir: `/root/analysis/0.87/reports/execution_layer/LINK_SOFT_REPAIR_CONFIRMATION_20260303_020324`
- Non-gap screen dir: `/root/analysis/0.87/reports/execution_layer/LINK_NEW_ENTRY_LEVER_SCREEN_20260303_022124`
- LINK control trade source: `/root/analysis/0.87/reports/execution_layer/LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_20260303_013409/_trade_sources/LINKUSDT_M3_ENTRY_ONLY_FASTER_control.csv`
- Evaluation path reused from: `/root/analysis/0.87/scripts/live_coin_bounded_entry_repair_pilot.py`
- Strict confirmation helpers reused from: `/root/analysis/0.87/scripts/link_soft_repair_confirmation.py`
- Candidate reconstruction reused from: `/root/analysis/0.87/scripts/link_new_entry_lever_screen.py`

## B) Reconstruction Validity

| variant_id | reconstruction_valid | reconstruction_same_signal_ids | reconstruction_row_mask_match | acceptance_status |
| --- | --- | --- | --- | --- |
| WICK_ATR_SCORE_Q90 | 1 | 1 | 1 | 1 |
| WICK_RANGE_SCORE_Q90 | 1 | 1 | 1 | 1 |

## C) Strict Confirmation Results Table

| variant_id | acceptance_status | expectancy_delta_vs_control | cvar_delta_vs_control | maxdd_delta_vs_control | trade_count_retention_vs_control | instant_loser_delta_vs_control | fast_loser_delta_vs_control | winner_deletion_count | valid_for_ranking | robustness_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WICK_ATR_SCORE_Q90 | 1 | 1.53032e-05 | 9.32754e-05 | 0.0114482 | 0.9 | -20 | -5 | 1 | 1 | fragile |
| WICK_RANGE_SCORE_Q90 | 1 | 7.81726e-05 | 4.66377e-05 | 0.0206969 | 0.9 | -17 | -7 | 1 | 1 | fragile |

## D) Robustness Summary By Split / Route / Time Slice / Seed

| variant_id | split_confirm_count | split_total | route_confirm_count | route_total | route_family_total | route_unsupported_count | time_slice_confirm_count | time_slice_total | seed_supported | seed_confirm_count | seed_total | loss_cluster_confirm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WICK_ATR_SCORE_Q90 | 4 | 5 | 1 | 2 | 3 | 1 | 2 | 3 | 0 | 0 | 1 | 1 |
| WICK_RANGE_SCORE_Q90 | 4 | 5 | 1 | 2 | 3 | 1 | 2 | 3 | 0 | 0 | 1 | 1 |

## E) Failure Mode Check

- leakage/lookahead: `proven clean` via preserved repaired chronology and `parity_clean=1` for both candidates
- ranking validity: `{'WICK_ATR_SCORE_Q90': 1, 'WICK_RANGE_SCORE_Q90': 1}`
- sample collapse: retention values are `{'WICK_ATR_SCORE_Q90': 0.9, 'WICK_RANGE_SCORE_Q90': 0.9}`; both stay above the pilot floor
- winner deletion: `{'WICK_ATR_SCORE_Q90': 1, 'WICK_RANGE_SCORE_Q90': 1}`
- route fragility: `{'WICK_ATR_SCORE_Q90': {'route_confirm_count': 1, 'route_total': 2, 'route_family_total': 3, 'route_unsupported_count': 1}, 'WICK_RANGE_SCORE_Q90': {'route_confirm_count': 1, 'route_total': 2, 'route_family_total': 3, 'route_unsupported_count': 1}}`
- unrealistic execution assumptions: `assumed unchanged` because only the same frozen skip masks were applied; entry timing, exits, stop logic, and costs were unchanged

## F) Conservative Winner Or NO_REPAIR_APPROVED

- Decision: `NO_REPAIR_APPROVED`
- Decision reason: Both screened wick candidates remain fragile under strict confirmation.

## G) Final LINK Deployment Posture

- Final posture: `NO_REPAIR_APPROVED`

## H) Exact Recommendation

- Stop LINK local entry-repair research. Neither wick-family candidate clears strict confirmation robustly enough for shadow deployment.
