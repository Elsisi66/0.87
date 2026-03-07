# GA Exec 3m Decision

- Generated UTC: 2026-02-22T19:12:27.961859+00:00
- Run dir: `/root/analysis/0.87/reports/execution_layer/PHASEO_SAMPLER_REPAIR_V2_20260222_191119/smoke_ga_runs/GA_EXEC_OPT_20260222_191119`
- Best genome hash: `85961fb142b6474dd1184a6e`
- Repro check pass: 1

## Baseline vs Best (Overall TEST-only)

- baseline_expectancy_net: -0.000838
- exec_expectancy_net: -0.000674
- delta_expectancy_exec_minus_baseline: 0.000164
- baseline_cvar_5: -0.002200
- exec_cvar_5: -0.001216
- cvar_improve_ratio: 0.447182
- baseline_max_drawdown: -0.445347
- exec_max_drawdown: -0.241430
- maxdd_improve_ratio: 0.457883
- exec_entry_rate: 0.977778
- exec_taker_share: 0.000000
- exec_median_fill_delay_min: 0.00
- exec_p95_fill_delay_min: 0.00

## Gate Table

- expectancy >= baseline: 1
- cvar_improve >= 15%: 1
- maxdd_improve >= 15%: 1
- taker_share <= 0.25: 1
- median_fill_delay <= 45 min: 1
- p95_fill_delay <= 180 min: 1
- per-symbol entry-rate gates pass: 1
- split stability pass: 1 (min=-0.001170, median=-0.000637)

## Final

- Decision: **DEPLOY**

## Artifacts

- genomes: `/root/analysis/0.87/reports/execution_layer/PHASEO_SAMPLER_REPAIR_V2_20260222_191119/smoke_ga_runs/GA_EXEC_OPT_20260222_191119/genomes.csv`
- best genome: `/root/analysis/0.87/reports/execution_layer/PHASEO_SAMPLER_REPAIR_V2_20260222_191119/smoke_ga_runs/GA_EXEC_OPT_20260222_191119/best_genome.json`
- top-k: `/root/analysis/0.87/reports/execution_layer/PHASEO_SAMPLER_REPAIR_V2_20260222_191119/smoke_ga_runs/GA_EXEC_OPT_20260222_191119/top_k_genomes.json`
- pareto front: `/root/analysis/0.87/reports/execution_layer/PHASEO_SAMPLER_REPAIR_V2_20260222_191119/smoke_ga_runs/GA_EXEC_OPT_20260222_191119/pareto_front.csv`
- split rollup: `/root/analysis/0.87/reports/execution_layer/PHASEO_SAMPLER_REPAIR_V2_20260222_191119/smoke_ga_runs/GA_EXEC_OPT_20260222_191119/walkforward_results_by_split.csv`
- symbol risk rollup: `/root/analysis/0.87/reports/execution_layer/PHASEO_SAMPLER_REPAIR_V2_20260222_191119/smoke_ga_runs/GA_EXEC_OPT_20260222_191119/risk_rollup_by_symbol.csv`
- overall risk rollup: `/root/analysis/0.87/reports/execution_layer/PHASEO_SAMPLER_REPAIR_V2_20260222_191119/smoke_ga_runs/GA_EXEC_OPT_20260222_191119/risk_rollup_overall.csv`
