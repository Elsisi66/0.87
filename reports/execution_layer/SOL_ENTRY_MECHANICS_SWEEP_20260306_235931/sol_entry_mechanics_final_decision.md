# SOL Entry Mechanics Final Decision

- Run dir: `/root/analysis/0.87/reports/execution_layer/SOL_ENTRY_MECHANICS_SWEEP_20260306_235931`
- Decision: `NO_CHANGE_RECOMMENDED`

## Variant Metrics (Strict Harness)

| variant_id | delta_expectancy_vs_baseline | strict_delta_expectancy_vs_1h_reference | delta_cvar_vs_baseline | maxdd_improve_ratio | retention | split_support_pass | route_pass_rate | holdout_pass | gate_g3_risk_sanity | all_required_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0_BASELINE | 0 | 0.0002184829922 | 0 | 0 | 1 | 1 | 1 | 0 | 1 | 0 |
| V2_DELAYED_REPRICE_GUARD | 0.0004503300026 | 0.0006688129948 | -8.744564571e-05 | -0.2393008005 | 1 | 0 | 0 | 0 | 0 | 0 |

## Gate Failure Counts (Promotable Variants Only)

| gate | fail_count | total_candidates | fail_variants |
| --- | --- | --- | --- |
| route_pass_rate==1.0 | 1 | 1 | V2_DELAYED_REPRICE_GUARD |
| split_support_pass==1 | 1 | 1 | V2_DELAYED_REPRICE_GUARD |
| holdout_pass==1 | 1 | 1 | V2_DELAYED_REPRICE_GUARD |
| gate_g3_risk_sanity==1 | 1 | 1 | V2_DELAYED_REPRICE_GUARD |
| all_required_pass | 1 | 1 | V2_DELAYED_REPRICE_GUARD |

## Formal Closure

- Entry-mechanics lever is formally closed for SOL under this strict harness (no promotable non-baseline variant passed all required gates).
- No additional entry-mechanics polishing is recommended in this branch.

## Pivot Recommendation

- Next lever class: **b) sizing/risk per trade**.
- Why: participation remains ~1.0, but non-baseline variants fail robustness/risk gates; this points to risk concentration control as the next bounded lever, not entry mechanics micro-tuning.