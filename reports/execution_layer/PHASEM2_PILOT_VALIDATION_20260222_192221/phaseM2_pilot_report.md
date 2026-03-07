# Phase M2 Pilot Report

- Generated UTC: 2026-02-22T19:25:11.135662+00:00
- Phase dir: `/root/analysis/0.87/reports/execution_layer/PHASEM2_PILOT_VALIDATION_20260222_192221`
- Pilot run dir: `/root/analysis/0.87/reports/execution_layer/PHASEM2_PILOT_VALIDATION_20260222_192221/pilot_ga_runs/GA_EXEC_OPT_20260222_192223`
- Frozen lock pass: 1

## Pilot Configuration

- Symbol: SOLUSDT
- Budget: pop=60, gens=1 (small pilot)
- Walkforward: ON (wf_splits=5, train_ratio=0.70)
- Mode: tight (Phase O sampler v2 in ga_exec_3m_opt)
- Hard gates: unchanged
- Pre-eval dedupe + effective trials + PSR/DSR proxies: enabled/reported

## Mandatory Evaluation Answers

1) valid_for_ranking candidates present? YES (count=5)
2) If yes, sane/pathology-free? YES (sane_count=5)
3) Metric diversity real or collapsed? real (duplicate_rate=0.1667, nonduplicate=51)
4) Participation-only or path metrics too? both: participation gates pass for 5, and valid-set medians are delta_expectancy_vs_baseline=-0.000104, cvar_improve=0.082863, maxdd_improve=0.236015.
5) Full run justified? YES

## Top 3 Candidates

### 1) 10d5a6fc4d8ea1417a165ec4
- valid_for_ranking: 1
- invalid_reason: 
- overall_entries_valid: 351
- overall_entry_rate: 0.975000
- overall_exec_expectancy_net: -0.00074989
- overall_delta_expectancy_exec_minus_baseline: 0.00008797
- overall_cvar_improve_ratio: 0.057338
- overall_maxdd_improve_ratio: 0.236015
- overall_exec_taker_share: 0.000000
- overall_exec_median_fill_delay_min: 0.00
- overall_exec_p95_fill_delay_min: 3.00
### 2) d806813a23b3728ef507d733
- valid_for_ranking: 1
- invalid_reason: 
- overall_entries_valid: 350
- overall_entry_rate: 0.972222
- overall_exec_expectancy_net: -0.00086251
- overall_delta_expectancy_exec_minus_baseline: -0.00002465
- overall_cvar_improve_ratio: 0.343512
- overall_maxdd_improve_ratio: 0.306024
- overall_exec_taker_share: 0.000000
- overall_exec_median_fill_delay_min: 0.00
- overall_exec_p95_fill_delay_min: 0.00
### 3) adacbbf6a82a4d31ead469a8
- valid_for_ranking: 1
- invalid_reason: 
- overall_entries_valid: 350
- overall_entry_rate: 0.972222
- overall_exec_expectancy_net: -0.00094177
- overall_delta_expectancy_exec_minus_baseline: -0.00010392
- overall_cvar_improve_ratio: 0.423142
- overall_maxdd_improve_ratio: 0.241557
- overall_exec_taker_share: 0.000000
- overall_exec_median_fill_delay_min: 0.00
- overall_exec_p95_fill_delay_min: 3.00

## Blocking Gates (Invalid Histogram)

- SOLUSDT:entry_rate: 55
- SOLUSDT:trades<50: 13
- overall:entry_rate: 13
- overall:trades<200: 13
- SOLUSDT:median_fill_delay: 10
- SOLUSDT:nan_or_inf: 10
- SOLUSDT:p95_fill_delay: 10
- SOLUSDT:taker_share: 10

## Verdict

- Pilot outcome: **GO**
- Rationale: non-zero rankable set with sane metrics and non-trivial duplicate-adjusted diversity under unchanged hard gates.
