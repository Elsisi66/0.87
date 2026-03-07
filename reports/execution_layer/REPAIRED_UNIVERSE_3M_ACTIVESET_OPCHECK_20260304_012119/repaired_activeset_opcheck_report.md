# Repaired Universe 3m Active-Set Operational Check

This is a bounded operational/deployment check on the earned repaired-branch selective 3m subset only. It excludes discovery, optimization, and any held-back symbols.

## Inputs Used
- Frozen repaired universe dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_1H_UNIVERSE_FREEZE_20260304_000207`
- Strict confirm dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_20260304_010143`
- Prior bounded subset dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_20260304_003748`
- Foundation dir: `/root/analysis/0.87/reports/execution_layer/UNIVERSAL_DATA_FOUNDATION_20260228_150929`

## Cost Stress Definition
- base fee/slip: maker `2.0000` / taker `4.0000` / limit slip `0.5000` / market slip `2.0000`
- cost-heavy stress: maker `3.0000` / taker `6.0000` / limit slip `1.0000` / market slip `3.0000`

## Per-Symbol Summary
| symbol | strict_confirm_candidate_id | strict_confirm_expectancy | reconstructed_expectancy | reconstruction_delta_abs | cost_heavy_expectancy | cost_heavy_delta_vs_repaired_1h | coverage_outcome | classification | operational_blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SOLUSDT | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001906509291 | 0.001906509291 | 4.33680869e-19 | 0.001449412355 | 0.0003618636253 | clean_local_full_3m | ACTIVE_DEPLOYABLE_SUBSET | none |
| LTCUSDT | M2_ENTRY_ONLY_MORE_PASSIVE | 0.0005884738669 | 0.0005884738669 | 4.141652299e-17 | 0.0001340322444 | 0.0006352847364 | coverage_degraded | BLOCKED | non_local_3m_data_path |

## Proven vs Assumed
- Proven: both active names were rebuilt from the frozen repaired universe params and re-evaluated with the strict route-enabled 3m path.
- Proven: deterministic reconstruction was checked against the exact strict-confirm winner identity and headline expectancy.
- Proven: cost-heavy stress kept entry/exit mechanics unchanged and only raised fees/slippage.
- Assumed: the universal foundation remains an acceptable 3m data pool for deterministic historical reconstruction where no local full 3m parquet exists.

## Recommendation
- Final recommendation: `BLOCK_DEPLOYMENT_AND_REVIEW`
- Active deployable symbols: `SOLUSDT`
- Shadow set: `NEARUSDT`