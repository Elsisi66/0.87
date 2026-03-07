# Repaired Universe 3m Execution Strict Confirmation (Subset 1 Survivors)

This is a strict route-enabled repaired-branch confirmation pass. It uses the frozen repaired universe as the only upstream signal/parameter source, and the existing universal foundation only as a local 3m window cache.

## Inputs Used
- Frozen repaired universe dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_1H_UNIVERSE_FREEZE_20260304_000207`
- Frozen repaired universe table: `/root/analysis/0.87/reports/execution_layer/REPAIRED_1H_UNIVERSE_FREEZE_20260304_000207/repaired_best_by_symbol.csv`
- Preflight pathology decision dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_UNIVERSE_PREEXEC_PATHOLOGY_AUDIT_20260304_001818`
- Bounded subset decision dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_20260304_003748`
- 3m cache foundation dir: `/root/analysis/0.87/reports/execution_layer/UNIVERSAL_DATA_FOUNDATION_20260228_150929`
- Strict confirm subset: `NEARUSDT, SOLUSDT, LTCUSDT`

## Per-Symbol Summary
| priority_rank | symbol | bounded_pass_best_exec_expectancy_net | confirmed_exec_expectancy_net | delta_expectancy_vs_bounded_pass | delta_expectancy_vs_repaired_1h | cvar_delta | maxdd_delta | trade_retention_vs_reference | route_pass_rate | classification | pathology_or_degradation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | NEARUSDT | 0.00325143 | 0.00325143 | 7.11237e-17 | 0.00025704 | 0.0227181 | 0.161334 | 1 | 0.666667 | SHADOW_ONLY | route_fragility |
| 2 | SOLUSDT | 0.00190651 | 0.00190651 | 4.33681e-19 | 0.000218483 | 0.0397567 | 0.267749 | 1 | 1 | ACTIVE_3M_SURVIVOR | none |
| 3 | LTCUSDT | 0.000588474 | 0.000588474 | 4.14165e-17 | 0.000489567 | 0.0742125 | 0.23043 | 1 | 1 | ACTIVE_3M_SURVIVOR | none |

## Overall Outcome
- improved_count: `3`
- degraded_count: `0`
- neutral_or_inconclusive_count: `0`
- aggregate_delta_expectancy_weighted: `0.0002616115`

## Proven vs Assumed
- Proven: the subset was restricted to the three symbols that survived the bounded first-wave pass only.
- Proven: repaired 1h signals were rebuilt from the frozen repaired per-symbol params, not from the legacy universe or legacy foundation signal timeline.
- Proven: the existing `phase_v` / Model A entry-only 3m evaluation path was reused with the omitted route-family layer enabled.
- Assumed: the latest universal foundation is acceptable as a 3m slice cache pool only; it was not used as the upstream signal source.

## Recommendation
- Final recommendation: `SHRINK_TO_SMALLER_ACTIVE_SET`
- Active keep list: `SOLUSDT, LTCUSDT, NEARUSDT`