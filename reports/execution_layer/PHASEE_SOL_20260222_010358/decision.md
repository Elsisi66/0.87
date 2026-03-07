# Phase E SOL Entry Pre-Gate Decision

- Generated UTC: 2026-02-22T01:04:19.739889+00:00
- Symbol: SOLUSDT
- Status: **PASS**

## Best Pre-Gate Config

- {"config_id": "9e16d9ae4f163245", "cooldown_h": 0, "max_signals_24h": 999, "overlap_h": 0, "session_mode": "all", "stop_distance_min": 0.001, "trend_required": 1, "vol_max_pct": 100.0, "vol_min_pct": 5.0}

## Control vs Best Deltas

- delta_expectancy_best_entry_pregate_minus_phasec_control: 0.000049
- delta_maxdd_best_entry_pregate_minus_phasec_control: 0.155162
- delta_cvar5_best_entry_pregate_minus_phasec_control: -0.000000
- delta_pnl_sum_best_entry_pregate_minus_phasec_control: 0.155162
- trades_total(control -> best): 600 -> 353
- entry_rate(control -> best): 1.0000 -> 0.5883
- split_median_expectancy_delta: 0.000213

## Gate Table

- pass_expectancy: 1
- pass_split_median: 1
- pass_maxdd_not_worse: 1
- pass_cvar_not_worse: 1
- pass_participation: 1
- pass_min_split_support: 1
- pass_data_quality: 1
- pass_reproducibility: 1
- pass_all: 1

## Recommendation

- Proceed to limited full entry optimization (Phase E2b / entry GA) on the same frozen universe, seeded with this pre-gate.
