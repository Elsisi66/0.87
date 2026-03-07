# Practical Decision

- Generated UTC: 2026-02-22T02:20:16.478688+00:00
- Decision: **NO_DEPLOY_PAUSE_SOL_SIGNAL_LAYER**
- Root cause classification: **signal_definition_itself**

## Contract-Locked Summary

- contract_locked: 1
- representative_subset_reused: 1
- downstream_execution_frozen: 1
- phasec_best_vs_control_nondegrade: 1

## Core Metrics

- V2R expectancy/return/maxDD/cvar5/PF/win: -0.001027 / -1.000000 / -1.000000 / -0.002200 / 0.514693 / 0.037500
- V3R expectancy/return/maxDD/cvar5/PF/win: -0.000984 / -0.999999 / -0.999999 / -0.002283 / 0.534313 / 0.041667
- V4R expectancy/return/maxDD/cvar5/PF/win: -0.000868 / -1.000000 / -1.000000 / -0.002037 / 0.539177 / 0.036667

## Top 3 Upstream Fixes

1. Add upstream regime gate to suppress down-trend/high-vol adverse buckets
   expected_impact: Reduce tail losses concentration and improve cvar5/maxDD before expectancy turn.
   required_code_changes: Signal export path: add deterministic pre-entry filter using trend_up_1h + atr_percentile_1h thresholds in 1h signal post-processing.
2. Add signal cooldown/de-clustering gate (>=4h) at 1h layer
   expected_impact: Reduce overtrading burst losses and same-direction streak drawdowns.
   required_code_changes: 1h signal post-filter: keep earliest signal in cooldown window; preserve deterministic ordering.
3. Rework entry timing at signal layer (1h delayed-open candidate set)
   expected_impact: Potentially improve fill context and reduce immediate adverse excursion.
   required_code_changes: 1h reference entry policy module: support delayed-open mode toggles (0/1/2 bars) under same contract.
