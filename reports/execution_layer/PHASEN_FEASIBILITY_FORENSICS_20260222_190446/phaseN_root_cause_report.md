# Phase N Feasibility Root-Cause Report

- Generated UTC: 2026-02-22T19:05:31.175005+00:00
- Symbol: SOLUSDT
- Frozen subset: `/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv`
- Canonical fee hash: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a` (expected `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`)
- Canonical metrics hash: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99` (expected `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`)
- Subset sha256 (observed): `7259f23f6ac3ccb6607a834d9e90fc471e03d38c00b48a7334d8b6234df65252`
- Subset sha256 (reference from prior docs): `fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d`

## Controlled Configs

- A_tight_baseline
- B_permissive_control
- C_no_killzone
- D_no_displacement
- E_no_micro_vol
- F_relaxed_limit_offset
- G_aggressive_fallback
- H_relaxed_cancel_after
- UB_market_upper_bound

## Participation Reachability

- configs reaching participation gates (entry_rate>=0.97 and trades>=200): 2
- configs reaching participation + realism (taker<=0.25, median<=45, p95<=180): 1
- participation-reaching configs:
  - B_permissive_control: entry_rate=0.9850, trades=1182, taker=0.0000, med_delay=0.00, p95=0.00
  - UB_market_upper_bound: entry_rate=0.9850, trades=1182, taker=1.0000, med_delay=0.00, p95=0.00

## Top Participation Blockers (A baseline)

- micro_vol_drop: 563
- displacement_drop: 524
- order_to_fill_drop: 102
- time_filters_drop: 0
- filled_to_valid_drop: 0

## Ablations Restoring Participation (vs A)

- D_no_displacement: delta_entry_rate=0.3883, delta_trades=466
- H_relaxed_cancel_after: delta_entry_rate=0.0233, delta_trades=28
- C_no_killzone: delta_entry_rate=0.0000, delta_trades=0
- E_no_micro_vol: delta_entry_rate=0.0000, delta_trades=0
- G_aggressive_fallback: delta_entry_rate=0.0000, delta_trades=0
- F_relaxed_limit_offset: delta_entry_rate=-0.0092, delta_trades=-11

## Regime Concentration Check (permissive control)

- vol_bucket=low: count=269, entryable_rate=0.9963
- vol_bucket=mid: count=272, entryable_rate=0.9890
- vol_bucket=high: count=193, entryable_rate=0.9741
- vol_bucket=nan: count=466, entryable_rate=0.9807
- mixed_regime_flag: 0

## Root-Cause Classification

- Class: **A**
- Interpretation: gates are reachable in-principle with realistic configs; sampler/search-space is primary bottleneck.

## Recommendation for Next Phase

- Repair sampler geometry around participation-feasible regions and penalize degenerate duplicate signatures before any larger search.
- Keep hard gates unchanged; run another small pilot with stratified sampling over entry mode / fallback / delay windows.
