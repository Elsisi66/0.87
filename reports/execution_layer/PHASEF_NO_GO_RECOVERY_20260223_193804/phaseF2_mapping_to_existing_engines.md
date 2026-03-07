# Phase F2 Mapping to Existing Engines

## Inputs and route construction
- Route data generation: `scripts.phase_d123_tail_filter.evaluate_baseline_routes()`
- Uses locked E1 execution genome and frozen representative subset.

## Feature construction
- Base feature transforms: `scripts.phase_af_ah_sizing_autorun.compute_policy_features()`
- AE-aligned risk score helper: `scripts.phase_d123_tail_filter.build_s1_risk_features()`

## Metric and gate evaluation
- Route-level weighted metrics computed with the same signal-level pnl vector convention used in AF/AG.
- Strict valid_for_ranking proxy kept unchanged from prior branch gate semantics:
  - min_delta_expectancy_vs_baseline > 0
  - min_cvar_improve_ratio >= 0
  - min_maxdd_improve_ratio > 0

## Scope guard
- No hard-gate changes.
- No execution entry/exit mechanic changes.
- No GA in this phase (direct 6-12 variant ablation only).
