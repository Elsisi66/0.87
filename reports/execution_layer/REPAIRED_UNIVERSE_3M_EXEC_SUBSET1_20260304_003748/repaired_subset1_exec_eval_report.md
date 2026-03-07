# Repaired Universe 3m Execution Evaluation (Subset 1)

This is a bounded repaired-branch evaluation. It uses the frozen repaired universe as the only upstream signal/parameter source, and the existing universal foundation only as a local 3m window cache.

## Inputs Used
- Frozen repaired universe dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_1H_UNIVERSE_FREEZE_20260304_000207`
- Frozen repaired universe table: `/root/analysis/0.87/reports/execution_layer/REPAIRED_1H_UNIVERSE_FREEZE_20260304_000207/repaired_best_by_symbol.csv`
- Preflight pathology decision dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_UNIVERSE_PREEXEC_PATHOLOGY_AUDIT_20260304_001818`
- 3m cache foundation dir: `/root/analysis/0.87/reports/execution_layer/UNIVERSAL_DATA_FOUNDATION_20260228_150929`
- Approved first-wave subset: `NEARUSDT, OGUSDT, DOGEUSDT, SOLUSDT, LTCUSDT, AXSUSDT, ZECUSDT`

## Per-Symbol Summary
| priority_rank | symbol | reference_exec_expectancy_net | best_exec_expectancy_net | delta_expectancy_vs_repaired_1h | cvar_delta | maxdd_delta | trade_retention_vs_reference | classification | pathology_or_degradation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | NEARUSDT | 0.00299439 | 0.00325143 | 0.00025704 | 0.0227181 | 0.161334 | 1 | KEEP_FOR_EXEC_LAYER | none |
| 2 | OGUSDT | 0.00562868 | 0.00543212 | -0.000196558 | 0.0954161 | 0.0891219 | 1 | NO_EXEC_EDGE | negative_expectancy_delta |
| 3 | DOGEUSDT | 0.0037039 | 0.00404188 | 0.000337986 | 0.0662612 | -0.574465 | 0.987448 | NO_EXEC_EDGE | partial_3m_coverage |
| 4 | SOLUSDT | 0.00168803 | 0.00190651 | 0.000218483 | 0.0397567 | 0.267749 | 1 | KEEP_FOR_EXEC_LAYER | none |
| 5 | LTCUSDT | 9.89073e-05 | 0.000588474 | 0.000489567 | 0.0742125 | 0.23043 | 1 | KEEP_FOR_EXEC_LAYER | none |
| 6 | AXSUSDT | 0.00303937 | 0.00362818 | 0.000588811 | 0.0366985 | 0.297871 | 0.922764 | INCONCLUSIVE | best_variant_invalid_for_ranking|partial_3m_coverage |
| 7 | ZECUSDT | 0.00290577 | 0.00282323 | -8.25379e-05 | 0.0318054 | 0.306952 | 0.956341 | INCONCLUSIVE | best_variant_invalid_for_ranking|negative_expectancy_delta|partial_3m_coverage |

## Overall Outcome
- improved_count: `3`
- degraded_count: `2`
- neutral_or_inconclusive_count: `2`
- aggregate_delta_expectancy_weighted: `0.0002064396`

## Proven vs Assumed
- Proven: the subset was restricted to the seven approved first-wave repaired-universe symbols only.
- Proven: repaired 1h signals were rebuilt from the frozen repaired per-symbol params, not from the legacy universe or legacy foundation signal timeline.
- Proven: the existing `phase_v` / Model A entry-only 3m evaluation path was reused on top of those rebuilt repaired signals.
- Assumed: the latest universal foundation is acceptable as a 3m slice cache pool only; it was not used as the upstream signal source.

## Recommendation
- Final recommendation: `KEEP_3M_TO_SELECTIVE_SUBSET_ONLY`
- Selective keep list: `NEARUSDT, SOLUSDT, LTCUSDT`