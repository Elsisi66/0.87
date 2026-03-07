# Phase L Sampler Repair Report

- Generated UTC: 2026-02-22T18:48:26.696362+00:00
- Baseline run (before patch): `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_181715`
- Smoke run (after patch): `/root/analysis/0.87/reports/execution_layer/PHASEL_SAMPLER_REPAIR_20260222_184421/smoke_ga_runs/GA_EXEC_OPT_20260222_184614`

## Scope

- Patch-only phase: no gate threshold relaxation and no full GA marathon.
- Implemented frozen canonical hash lock + feasibility-biased sampler repair + generation-time dedupe counters.
- Validation run: SOLUSDT, pop=48, gens=1, contract-locked setup.

## Freeze Hash Lock

- Before: fee_hash_match=0, metrics_hash_match=0
- After: freeze_lock_pass=1, copied_verbatim_fee=1, copied_verbatim_metrics=1
- After canonical hashes: fee=b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a, metrics=d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99

## Duplicate / Diversity Comparison

- Before duplicate rate (metric signature): 719/1830 = 0.3929
- After duplicate rate (metric signature): 25/48 = 0.5208
- After pre-eval genome-hash duplicates removed: 0
- After effective_trials_proxy: 24.0000

## Feasibility (Participation-Gate Distance)

- Entry-rate symbol slack (threshold 0.97), before median/p90/max: -0.9422 / -0.8200 / -0.2450
- Entry-rate symbol slack (threshold 0.97), after median/p90/max: -0.9700 / -0.0811 / -0.0172
- Trades slack (threshold 200), before median/p90/max: -190.0 / -146.0 / 61.0
- Trades slack (threshold 200), after median/p90/max: -200.0 / 120.0 / 143.0
- valid_for_ranking before/after: 0 / 0

## Dominant Invalid Reasons

- Before top 5:
  - SOLUSDT:entry_rate: 1830
  - overall:entry_rate: 1829
  - overall:trades<200: 1829
  - SOLUSDT:trades<50: 1599
  - SOLUSDT:taker_share: 484
- After top 5:
  - SOLUSDT:entry_rate: 48
  - overall:entry_rate: 34
  - overall:trades<200: 32
  - SOLUSDT:trades<50: 29
  - SOLUSDT:median_fill_delay: 25

## Sampler Repair Activity (after patch)

- cap_fallback_delay_min_tight: 24
- cap_cooldown_min_tight: 21
- cap_min_entry_improvement_gate_tight: 20
- cap_limit_offset_bps_tight: 11
- ensure_fill_window_for_displacement_gate: 8
- enforce_fallback_for_nonmarket: 5
- disable_skip_if_vol_gate_under_micro_filter: 4
- raise_spread_guard_floor_tight: 3
- soften_quality_gate_under_stack: 2
- cap_signal_quality_gate_tight: 1
- disable_killzone_filter_tight: 1
- sampler_reject_histogram: {} (no construction-level rejects triggered in this smoke sample)

## Outcome

- Freeze lock compliance is fixed and hash-stable.
- Pre-eval genome-hash dedupe and sampler repair telemetry are now active and observable in run artifacts.
- Metric-signature duplication remains elevated in this tiny smoke sample (25/48), so diversity is still a blocker.
- Participation gates remain binding; smoke still has zero valid-for-ranking genomes under unchanged hard gates.
- GO/NO-GO: NO-GO for full marathon until a follow-up pilot validates sustained feasibility improvement over multiple generations.
