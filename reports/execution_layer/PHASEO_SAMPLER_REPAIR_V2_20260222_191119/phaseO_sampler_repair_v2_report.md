# Phase O Sampler Repair V2 Report

- Generated UTC: 2026-02-22T19:16:22Z
- Phase dir: `/root/analysis/0.87/reports/execution_layer/PHASEO_SAMPLER_REPAIR_V2_20260222_191119`
- Smoke run dir: `/root/analysis/0.87/reports/execution_layer/PHASEO_SAMPLER_REPAIR_V2_20260222_191119/smoke_ga_runs/GA_EXEC_OPT_20260222_191119`
- Scope: sampler/search-space repair validation only (no full GA marathon)

## Frozen Lock Status

- freeze_lock_pass: `1`
- canonical_fee_model_sha256: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`
- canonical_metrics_definition_sha256: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
- copied_verbatim_fee/metrics: `1` / `1`

## Feasibility Smoke Configuration

- Symbol: SOLUSDT
- Budget: pop=180, gens=1 (cheap feasibility smoke)
- Walkforward: ON (5 splits, 70/30)
- Hard gates: unchanged
- Sampler: repaired tight-mode prior + risk-aware repair + invalid-construction rejection

## Before vs After (Phase L / Phase M -> Phase O)

| metric | Phase L | Phase M | Phase O |
| --- | ---: | ---: | ---: |
| candidates evaluated | 48 | 60 | 180 |
| valid_for_ranking_count | 0 | 0 | 18 |
| participation_pass_count | 0 | 0 | 18 |
| near_feasible_count | 1 | 3 | 94 |
| duplicate_rate (metric signature) | 0.5208 | 0.5167 | 0.1889 |
| effective_trials_proxy | 24.0 | 30.0 | 147.0 |
| entry-rate slack p50 (vs 0.97) | -0.9700 | -0.9700 | -0.0367 |
| entry-rate slack p90 (vs 0.97) | -0.0811 | -0.0667 | -0.0003 |
| entry-rate slack max (vs 0.97) | -0.0172 | -0.0339 | 0.0106 |
| trades slack p50 (vs 200) | -200.0 | -200.0 | 136.0 |
| trades slack p90 (vs 200) | 120.0 | 125.2 | 149.1 |

## Sampler Telemetry (Phase O)

- mode draws: 184
- feasible-prior share: 79.35%
- invalid-construction reject rate: 2.17%
- repaired_for_participation_risk_total: 152
- risk bucket post low share: 92.22%
- metric signature unique count: 147
- effective_trials_proxy: 147.0

## Remaining Blockers (Phase O invalid histogram)

- SOLUSDT:entry_rate: 162
- SOLUSDT:trades<50: 43
- overall:entry_rate: 43
- overall:trades<200: 43
- SOLUSDT:median_fill_delay: 34
- SOLUSDT:nan_or_inf: 34
- SOLUSDT:p95_fill_delay: 34
- SOLUSDT:taker_share: 34

## Interpretation

- Sampler quality materially improved: feasibility mass moved from mostly dead participation regions to a mix with rankable candidates.
- Participation frontier shifted upward: Phase O reached non-zero gate passers under unchanged hard gates.
- Diversity improved materially: duplicate rate dropped while effective trials rose strongly.
- Not a profitability claim yet: this smoke validates proposal feasibility, not deployability.

## Decision

- Phase M2 pilot recommendation: **GO**
- Reason: feasibility and diversity shift is large enough to justify a strict small pilot GA under the same frozen lock.
