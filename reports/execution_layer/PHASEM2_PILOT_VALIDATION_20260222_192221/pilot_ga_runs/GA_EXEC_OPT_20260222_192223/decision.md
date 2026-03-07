# GA Exec 3m Decision

- Generated UTC: 2026-02-22T19:23:06.005345+00:00
- Run dir: `/root/analysis/0.87/reports/execution_layer/PHASEM2_PILOT_VALIDATION_20260222_192221/pilot_ga_runs/GA_EXEC_OPT_20260222_192223`
- Best genome hash: `10d5a6fc4d8ea1417a165ec4`
- Repro check pass: 1

## Baseline vs Best (Overall TEST-only)

- baseline_expectancy_net: -0.000838
- exec_expectancy_net: -0.000750
- delta_expectancy_exec_minus_baseline: 0.000088
- baseline_cvar_5: -0.002200
- exec_cvar_5: -0.002073
- cvar_improve_ratio: 0.057338
- baseline_max_drawdown: -0.445347
- exec_max_drawdown: -0.340238
- maxdd_improve_ratio: 0.236015
- exec_entry_rate: 0.975000
- exec_taker_share: 0.000000
- exec_median_fill_delay_min: 0.00
- exec_p95_fill_delay_min: 3.00

## Gate Table

- expectancy >= baseline: 1
- cvar_improve >= 15%: 0
- maxdd_improve >= 15%: 1
- taker_share <= 0.25: 1
- median_fill_delay <= 45 min: 1
- p95_fill_delay <= 180 min: 1
- per-symbol entry-rate gates pass: 1
- split stability pass: 1 (min=-0.001200, median=-0.001054)

## Final

- Decision: **NO-DEPLOY**

## Artifacts

- genomes: `/root/analysis/0.87/reports/execution_layer/PHASEM2_PILOT_VALIDATION_20260222_192221/pilot_ga_runs/GA_EXEC_OPT_20260222_192223/genomes.csv`
- best genome: `/root/analysis/0.87/reports/execution_layer/PHASEM2_PILOT_VALIDATION_20260222_192221/pilot_ga_runs/GA_EXEC_OPT_20260222_192223/best_genome.json`
- top-k: `/root/analysis/0.87/reports/execution_layer/PHASEM2_PILOT_VALIDATION_20260222_192221/pilot_ga_runs/GA_EXEC_OPT_20260222_192223/top_k_genomes.json`
- pareto front: `/root/analysis/0.87/reports/execution_layer/PHASEM2_PILOT_VALIDATION_20260222_192221/pilot_ga_runs/GA_EXEC_OPT_20260222_192223/pareto_front.csv`
- split rollup: `/root/analysis/0.87/reports/execution_layer/PHASEM2_PILOT_VALIDATION_20260222_192221/pilot_ga_runs/GA_EXEC_OPT_20260222_192223/walkforward_results_by_split.csv`
- symbol risk rollup: `/root/analysis/0.87/reports/execution_layer/PHASEM2_PILOT_VALIDATION_20260222_192221/pilot_ga_runs/GA_EXEC_OPT_20260222_192223/risk_rollup_by_symbol.csv`
- overall risk rollup: `/root/analysis/0.87/reports/execution_layer/PHASEM2_PILOT_VALIDATION_20260222_192221/pilot_ga_runs/GA_EXEC_OPT_20260222_192223/risk_rollup_overall.csv`
