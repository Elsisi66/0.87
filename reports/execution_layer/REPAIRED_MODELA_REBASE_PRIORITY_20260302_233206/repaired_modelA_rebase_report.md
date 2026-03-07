# Repaired Model A Rebase Report

- Generated UTC: 2026-03-02T23:35:38.694431+00:00
- Run dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_MODELA_REBASE_PRIORITY_20260302_233206`
- Foundation source: `/root/analysis/0.87/reports/execution_layer/UNIVERSAL_DATA_FOUNDATION_20260228_150929`
- Canonical repaired 1h baseline: `/root/analysis/0.87/reports/execution_layer/1H_CONTRACT_REPAIR_REBASELINE_20260301_140650/repaired_1h_reference_summary.csv`
- Contract: `1h owns signals`, `1h owns exits`, `3m entry only`, `no same-parent-bar 1h exits`

## Priority Symbol Results

| symbol | classification | best_candidate_id | reference_exec_expectancy_net | best_exec_expectancy_net | delta_expectancy_vs_repaired_1h | cvar_improve_ratio | maxdd_improve_ratio | best_valid_for_ranking |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AVAXUSDT | MODEL_A_NO_GO_REPAIRED | M2_ENTRY_ONLY_MORE_PASSIVE | 0.002880004866 | 0.00313810692 | 0.0002581020542 | 0.03058207172 | 0.005546061052 | 0 |
| NEARUSDT | MODEL_A_WEAK_GO_REPAIRED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.002416891056 | 0.002778441585 | 0.0003615505286 | 0.04543622084 | 0.1553284759 | 1 |
| SOLUSDT | MODEL_A_NO_GO_REPAIRED | M3_ENTRY_ONLY_FASTER | 0.001803358946 | 0.001893724986 | 9.036603984e-05 | 0.054403896 | 0.2392970862 | 0 |

## Classification

| symbol | classification | classification_reason | best_candidate_id | best_delta_expectancy_vs_repaired_1h | best_cvar_improve_ratio | best_maxdd_improve_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| AVAXUSDT | MODEL_A_NO_GO_REPAIRED | no_robust_entry_only_advantage | M2_ENTRY_ONLY_MORE_PASSIVE | 0.0002581020542 | 0.03058207172 | 0.005546061052 |
| NEARUSDT | MODEL_A_WEAK_GO_REPAIRED | non_negative_delta_with_partial_route_support | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.0003615505286 | 0.04543622084 | 0.1553284759 |
| SOLUSDT | MODEL_A_NO_GO_REPAIRED | no_robust_entry_only_advantage | M3_ENTRY_ONLY_FASTER | 9.036603984e-05 | 0.054403896 | 0.2392970862 |

## Expansion Decision

- expand_to_wider_universe: `1`
- decision: `at least one priority symbol remains a repaired-baseline go`