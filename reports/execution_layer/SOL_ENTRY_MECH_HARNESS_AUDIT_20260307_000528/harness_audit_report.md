# SOL Entry Mechanics Harness Audit

- Sweep run audited: `/root/analysis/0.87/reports/execution_layer/SOL_ENTRY_MECHANICS_SWEEP_20260306_235931`
- Strict reference used: `/root/analysis/0.87/reports/execution_layer/REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_20260304_010143`

## Root Cause
- Sweep route gate incorrectly used `sol_3m_lossconcentration_ga.route_confirm_for_candidate`, which compares candidate vs baseline candidate per route; baseline therefore has zero delta by construction and reported route_pass_rate=0.0.
- Fix: sweep now uses strict `phase_v.evaluate_symbol(...)` route-family outputs (delta vs 1h reference), same lineage as strict confirm.

## Baseline Invariance
- All headline checks pass: `1`
- File: `baseline_invariance_comparison.csv`

## Route Harness Sanity (Baseline)
- baseline route_pass_rate: `1.0`
- per-route pass rows: `3/3`
- File: `route_harness_sanity.csv`

## Minimal Rerun
- Variants rerun: `V0_BASELINE`, `V2_DELAYED_REPRICE_GUARD`
- File: `minimal_rerun_results.csv`

## Verdict
- `HARNESS_FIXED`