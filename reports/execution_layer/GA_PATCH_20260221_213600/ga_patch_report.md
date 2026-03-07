# GA Patch Report (Phase B)

- Generated UTC: 2026-02-21T21:36:00.309764+00:00
- Source run: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342`
- Best genome hash: `f0c7df09af14536a559ed913`

## Anti-Cheat Results

- best_valid_for_ranking: 1
- best_overall_entries_valid: 549
- best_overall_entry_rate: 0.915000
- finite_required_metrics_pass: 1
- no_trade_genome_selected_as_best: 0
- per_symbol_entry_rate_gate_pass: 1

## Overall Metrics (Best)

- overall_baseline_expectancy_net: -0.000643
- overall_exec_expectancy_net: -0.000738
- overall_delta_expectancy_exec_minus_baseline: -0.000095
- overall_exec_cvar_5: -0.001926
- overall_baseline_cvar_5: -0.002200
- overall_exec_max_drawdown: -0.443874
- overall_baseline_max_drawdown: -0.590052
- overall_exec_taker_share: 0.041894
- overall_exec_median_fill_delay_min: 0.000000
- overall_exec_p95_fill_delay_min: 69.000000

## Invalid Reason Histogram

- overall:entry_rate: 39
- SOLUSDT:entry_rate: 34
- SOLUSDT:taker_share: 25
- overall:trades<200: 19

## Smoke Assertions

- assertions_pass: **1**
- smoke_test_result: `reports/execution_layer/GA_PATCH_20260221_213600/smoke_test_result.json`
