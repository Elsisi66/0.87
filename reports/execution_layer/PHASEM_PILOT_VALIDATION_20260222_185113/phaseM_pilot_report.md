# Phase M Pilot Report

- Generated UTC: 2026-02-22T18:53:22.481754+00:00
- Phase dir: `/root/analysis/0.87/reports/execution_layer/PHASEM_PILOT_VALIDATION_20260222_185113`
- Pilot run dir: `/root/analysis/0.87/reports/execution_layer/PHASEM_PILOT_VALIDATION_20260222_185113/pilot_ga_runs/GA_EXEC_OPT_20260222_185122`
- Frozen lock pass: 1

## Pilot Configuration

- Symbol: SOLUSDT
- Budget: pop=60, gens=1 (small pilot)
- Walkforward: ON (wf_splits=5, train_ratio=0.70)
- Mode: tight
- Hard gates: unchanged
- Repaired sampler + pre-eval dedupe: enabled

## Mandatory Evaluation Answers

1) valid_for_ranking candidates present? NO (count=0)
2) If yes, sane/pathology-free? NO (sane_count=0)
3) Metric diversity real or collapsed? collapsed/weak (duplicate_rate=0.5167, nonduplicate=30)
4) Improvements source: not sufficient; best candidates improve CVaR/maxDD only under very low participation and still fail participation gates.
5) Full run justified? NO

## Blocking Gates (from invalid histogram)

- SOLUSDT:entry_rate: 60
- overall:entry_rate: 43
- overall:trades<200: 41
- SOLUSDT:trades<50: 36
- SOLUSDT:median_fill_delay: 31
- SOLUSDT:nan_or_inf: 31
- SOLUSDT:p95_fill_delay: 31
- SOLUSDT:taker_share: 31

## Top 3 Candidates

### 1) c24e434a5650d20288acbc37 
- valid_for_ranking: 0
- invalid_reason: SOLUSDT:entry_rate|SOLUSDT:trades<50|overall:entry_rate|overall:trades<200
- overall_entries_valid: 6
- overall_entry_rate: 0.016667
- overall_exec_expectancy_net: -0.00000526
- overall_cvar_improve_ratio: 0.947646
- overall_maxdd_improve_ratio: 0.995346
- overall_exec_taker_share: 0.000000
- overall_exec_median_fill_delay_min: 9.00
- psr_proxy/dsr_proxy: 0.024759 / 0.012021
### 2) 36e26de559eb5e6366de54aa 
- valid_for_ranking: 0
- invalid_reason: SOLUSDT:entry_rate|SOLUSDT:trades<50|overall:entry_rate|overall:trades<200
- overall_entries_valid: 13
- overall_entry_rate: 0.036111
- overall_exec_expectancy_net: -0.00000570
- overall_cvar_improve_ratio: 0.741344
- overall_maxdd_improve_ratio: 0.990094
- overall_exec_taker_share: 0.000000
- overall_exec_median_fill_delay_min: 21.00
- psr_proxy/dsr_proxy: 0.354378 / 0.252740
### 3) a813f19849ee5442c4aa91e7 
- valid_for_ranking: 0
- invalid_reason: SOLUSDT:entry_rate|SOLUSDT:trades<50|overall:entry_rate|overall:trades<200
- overall_entries_valid: 6
- overall_entry_rate: 0.016667
- overall_exec_expectancy_net: -0.00002051
- overall_cvar_improve_ratio: 0.762558
- overall_maxdd_improve_ratio: 0.978891
- overall_exec_taker_share: 0.000000
- overall_exec_median_fill_delay_min: 9.00
- psr_proxy/dsr_proxy: 0.068128 / 0.037355

## Verdict

- Pilot outcome: **NO_GO**
- Rationale: no rankable candidates under unchanged hard gates in this pilot budget.
