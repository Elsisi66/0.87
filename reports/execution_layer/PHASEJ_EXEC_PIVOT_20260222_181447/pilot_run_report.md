# Pilot Run Report

- Generated UTC: 2026-02-22T18:15:15.924032+00:00
- Branch executed: **A**
- Pilot GA run dir: `/root/analysis/0.87/reports/execution_layer/PHASEJ_EXEC_PIVOT_20260222_181447/pilot_ga_runs/GA_EXEC_OPT_20260222_181448`
- setup_checks_vs_expected: `{"fee_hash_match_expected": 1, "metrics_hash_match_expected": 1, "model_set_hash_match_expected": 1, "rep_hash_calc_match_manifest": 1, "rep_hash_file_match_manifest": 1, "rep_hash_match_expected": 1}`

## Pilot Outcome

- total_candidates: 48
- nonduplicate_candidates: 12
- duplicate_candidates: 36
- valid_for_ranking_candidates: 0
- sane_candidates: 0
- effective_trials_proxy: 1.593

Metric diversity among non-duplicate candidates:
- overall_exec_expectancy_net: min=-0.000401, p50=-0.000020, max=0.000004, std=0.000112
- overall_cvar_improve_ratio: min=-0.083978, p50=0.747663, max=1.000000, std=0.369363
- overall_maxdd_improve_ratio: min=0.668747, p50=0.982347, max=1.000000, std=0.092137
- overall_entry_rate: min=0.002778, p50=0.040278, max=0.486111, std=0.140925
- overall_exec_taker_share: min=0.000000, p50=0.000000, max=0.066667, std=0.018350

## Top 5 Non-Duplicate Candidates

```csv
pilot_rank,genome_hash,overall_exec_expectancy_net,overall_cvar_improve_ratio,overall_maxdd_improve_ratio,overall_entry_rate,overall_exec_taker_share,overall_exec_median_fill_delay_min,sane_candidate,valid_for_ranking,invalid_reason
1,e1b32e91fecafbd336a5b9d7,3.739327105681014e-06,1.0,1.0,0.0083333333333333,0.0,33.0,0,0,SOLUSDT:entry_rate|SOLUSDT:trades<50|overall:entry_rate|overall:trades<200
38,aae9a71b9c730c023af43a5f,-2.359644092893328e-06,0.9785440096937056,0.9980925597832716,0.0027777777777777,0.0,18.0,0,0,SOLUSDT:entry_rate|SOLUSDT:trades<50|overall:entry_rate|overall:trades<200
39,28cb6a7bff1be77215a0a780,2.18072793342665e-06,0.9679486214432552,0.9971506284451116,0.0083333333333333,0.0,36.0,0,0,SOLUSDT:entry_rate|SOLUSDT:trades<50|overall:entry_rate|overall:trades<200
40,fab8ef1c03b201c07235a375,-6.444792558923072e-06,0.9413981934450212,0.9947902920815964,0.0055555555555555,0.0,12.0,0,0,SOLUSDT:entry_rate|SOLUSDT:trades<50|overall:entry_rate|overall:trades<200
41,e4807afae2a576f5f6694772,-4.688623223238663e-06,0.9310468957405136,0.9947515696636356,0.0138888888888888,0.0,6.0,0,0,SOLUSDT:entry_rate|SOLUSDT:trades<50|overall:entry_rate|overall:trades<200
```

## Interpretation

- This pilot confirms whether execution/exit knobs produce meaningful dispersion under frozen setup.
- Primary objective in full run should be geometric fixed-size equity-step return.
- Current GA engine exports expectancy/CVaR/maxDD, so this pilot uses expectancy as primary proxy and flags that geometric export is a follow-up enhancement.
- `regime_concentration` hard gate remains TODO until per-candidate regime-bucket attribution is exported by the evaluator.

- Pilot verdict: **NO_GO** for full marathon until metric/pathology constraints are improved.
