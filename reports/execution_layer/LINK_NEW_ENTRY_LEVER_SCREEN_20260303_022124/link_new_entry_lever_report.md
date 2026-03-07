# LINK New Entry Lever Screen

- Generated UTC: `2026-03-03T02:22:08.422289+00:00`
- Artifact dir: `/root/analysis/0.87/reports/execution_layer/LINK_NEW_ENTRY_LEVER_SCREEN_20260303_022124`

## A) Discovered Code Paths / Artifacts Used

- Repaired 1h baseline dir: `/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650`
- Rebased multicoin Model A dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108`
- Winner-vs-loser forensic dir: `/root/analysis/0.87/reports/execution_layer/INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_20260303_012335`
- Latest bounded entry-repair pilot dir: `/root/analysis/0.87/reports/execution_layer/LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_20260303_013409`
- Retention-cliff forensic dir: `/root/analysis/0.87/reports/execution_layer/LINK_RETENTION_CLIFF_FORENSICS_20260303_014856`
- Soft-confirmation dir (used as rejected-gap guardrail): `/root/analysis/0.87/reports/execution_layer/LINK_SOFT_REPAIR_CONFIRMATION_20260303_020324`
- LINK control trade source: `/root/analysis/0.87/reports/execution_layer/LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_20260303_013409/_trade_sources/LINKUSDT_M3_ENTRY_ONLY_FASTER_control.csv`
- Evaluation path reused from: `/root/analysis/0.87/scripts/live_coin_bounded_entry_repair_pilot.py`
- Split/route screen reused from: `/root/analysis/0.87/scripts/link_soft_repair_confirmation.py`

## B) Candidate Feature Inventory

| family_id | features | priority_score | preview_removed_instant_losers | preview_removed_fast_losers | preview_removed_meaningful_winners | preview_retention |
| --- | --- | --- | --- | --- | --- | --- |
| UPPER_WICK_TAIL_CAP | upper_wick_ratio | 13.0345 | 22 | 2 | 2 | 0.9 |
| WICK_ATR_SCORE | upper_wick_ratio|atr_percentile_1h | 12.3103 | 20 | 5 | 1 | 0.9 |
| ATR_REGIME_CAP | atr_percentile_1h | 12.2688 | 46 | 10 | 6 | 0.765517 |
| WICK_RANGE_SCORE | upper_wick_ratio|signal_range_pct | 12.1724 | 17 | 7 | 1 | 0.9 |
| SMA_STRETCH_CAP | dist_to_sma20_pct | 12.1333 | 4 | 4 | 0 | 0.948276 |

## C) Bounded Lever Families Tested

| family_id | variant_id | variant_label | threshold_value | screen_status |
| --- | --- | --- | --- | --- |
| WICK_RANGE_SCORE | WICK_RANGE_SCORE_Q90 | Reject top 10% wick+range score | 0.751897 | PROMISING_FOR_STRICT_CONFIRMATION |
| WICK_ATR_SCORE | WICK_ATR_SCORE_Q90 | Reject top 10% wick+ATR score | 0.736207 | PROMISING_FOR_STRICT_CONFIRMATION |
| SMA_STRETCH_CAP | SMA_STRETCH_POSQ90 | Reject top 10% of positive SMA stretch | 0.0215866 | REJECTED |
| UPPER_WICK_TAIL_CAP | UPPER_WICK_Q90 | Reject upper-wick top 10% | 0.548008 | REJECTED |
| ATR_REGIME_CAP | ATR_P95 | Reject ATR percentile >= 95 | 95 | REJECTED |

## D) Results Table For Each Family

| family_id | variant_id | accepted | expectancy_delta_vs_control | cvar_delta_vs_control | maxdd_delta_vs_control | trade_count_retention_vs_control | instant_loser_delta_vs_control | fast_loser_delta_vs_control | removed_meaningful_winners | valid_for_ranking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WICK_RANGE_SCORE | WICK_RANGE_SCORE_Q90 | 1 | 7.81726e-05 | 4.66377e-05 | 0.0206969 | 0.9 | -17 | -7 | 1 | 1 |
| WICK_ATR_SCORE | WICK_ATR_SCORE_Q90 | 1 | 1.53032e-05 | 9.32754e-05 | 0.0114482 | 0.9 | -20 | -5 | 1 | 1 |
| SMA_STRETCH_CAP | SMA_STRETCH_POSQ90 | 0 | 9.56761e-05 | 0 | 0.0129482 | 0.948276 | -4 | -4 | 0 | 1 |
| UPPER_WICK_TAIL_CAP | UPPER_WICK_Q90 | 0 | -5.49612e-05 | 9.32754e-05 | 0.0206969 | 0.9 | -22 | -2 | 2 | 1 |
| ATR_REGIME_CAP | ATR_P95 | 0 | -0.00094236 | 6.99565e-05 | 0.0221969 | 0.765517 | -46 | -10 | 6 | 1 |

## E) Proven Vs Assumed

- Proven: all screened levers use only causal pre-entry 1h/3m-local features already present in the existing forensic feature matrix; the rejected positive-gap family was excluded.
- Proven: acceptance metrics reuse the existing pilot logic without changing tolerances.
- Proven: parity is preserved because only skip masks were changed; entry/exit timing, stop logic, and costs stayed frozen.
- Assumed: route confirmation remains a bounded screen, not a full confirmation pass. Any promising candidate still needs the stricter confirmation workflow before deployment.

## F) Best Candidate(s) Or NO_NEW_LINK_LEVER_FOUND

- Decision: `PROMISING_NEW_LINK_LEVER_FOUND`
| family_id | variant_id | screen_status | expectancy_delta_vs_control | trade_count_retention_vs_control | instant_loser_delta_vs_control | removed_meaningful_winners |
| --- | --- | --- | --- | --- | --- | --- |
| WICK_RANGE_SCORE | WICK_RANGE_SCORE_Q90 | PROMISING_FOR_STRICT_CONFIRMATION | 7.81726e-05 | 0.9 | -17 | 1 |
| WICK_ATR_SCORE | WICK_ATR_SCORE_Q90 | PROMISING_FOR_STRICT_CONFIRMATION | 1.53032e-05 | 0.9 | -20 | 1 |

## G) Exact Recommendation On Whether LINK Entry-Repair Research Should Continue

- Continue, but only with the top 1-2 bounded non-gap candidates above and only through the same strict confirmation harness used on the rejected soft-gap family.
