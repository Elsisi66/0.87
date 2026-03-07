# Phase P Full Run Report

- Generated UTC: 2026-02-22T20:04:27.599672+00:00
- Full run dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057`
- Phase dir: `/root/analysis/0.87/reports/execution_layer/PHASEP_FULL_RUN_VALIDATION_20260222_200427`

## Frozen Setup Validation

- freeze_lock_pass: 1
- canonical_fee_model_sha256: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`
- canonical_metrics_definition_sha256: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
- copied_verbatim_fee/metrics: 1/1

## A) Feasibility / Rankability

- full_valid_for_ranking_count: 2888 / 4356 (66.30%)
- full_participation_pass_count: 3045 / 4356 (69.90%)
- m2_valid_for_ranking_count: 5 / 60 (8.33%)
- m2_participation_pass_count: 5 / 60 (8.33%)
- valid_count_delta_vs_m2: 2883
- participation_count_delta_vs_m2: 3040

Top hard-gate fail reasons (full run):
- SOLUSDT:entry_rate: 1311
- SOLUSDT:trades<50: 686
- overall:entry_rate: 686
- overall:trades<200: 686
- SOLUSDT:taker_share: 385
- overall:taker_share: 357
- SOLUSDT:median_fill_delay: 227
- SOLUSDT:nan_or_inf: 227

## B) Diversity / Anti-Overfit

- duplicate_rate: 0.529385
- unique_metric_signatures: 2316
- effective_trials_uncorrelated: 2316.000000
- effective_trials_corr_adjusted: 1.237469
- avg_abs_metric_corr_proxy: 0.808018
- PSR/DSR proxies exported to `phaseP_shortlist_significance.csv`.
- Reality-check status: TODO placeholder retained (not implemented in-run).

## C) Economic Value vs Baseline Execution

- Top non-duplicate valid candidates show:
  median overall_exec_expectancy_net = 0.00005774
  median delta_expectancy_vs_baseline = 0.00089560
  median cvar_improve_ratio = 0.190848
  median maxdd_improve_ratio = 0.579504
  median taker_share = 0.081921
  median p95_fill_delay_min = 15.90
- Detailed decomposition exported in `phaseP_edge_vs_baseline_execution_decomposition.csv`.

## D) Decision Gate

- Classification: **GO_OOS_CONFIRMATION**
- Rationale: non-trivial rankable set, sane metrics, non-trivial duplicate-adjusted diversity, and at least one meaningful non-duplicate improvement vs baseline with acceptable realism.
