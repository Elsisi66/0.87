# GA Exec 3m Decision

- Generated UTC: 2026-02-22T20:00:34.165565+00:00
- Run dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057`
- Best genome hash: `7c2dd687b585669ec62ab0b9`
- Repro check pass: 1

## Baseline vs Best (Overall TEST-only)

- baseline_expectancy_net: -0.000838
- exec_expectancy_net: 0.000061
- delta_expectancy_exec_minus_baseline: 0.000898
- baseline_cvar_5: -0.002200
- exec_cvar_5: -0.001780
- cvar_improve_ratio: 0.190848
- baseline_max_drawdown: -0.445347
- exec_max_drawdown: -0.186917
- maxdd_improve_ratio: 0.580289
- exec_entry_rate: 0.983333
- exec_taker_share: 0.081921
- exec_median_fill_delay_min: 0.00
- exec_p95_fill_delay_min: 16.05

## Gate Table

- expectancy >= baseline: 1
- cvar_improve >= 15%: 1
- maxdd_improve >= 15%: 1
- taker_share <= 0.25: 1
- median_fill_delay <= 45 min: 1
- p95_fill_delay <= 180 min: 1
- per-symbol entry-rate gates pass: 1
- split stability pass: 0 (min=-0.001176, median=0.000006)

## Final

- Decision: **NO-DEPLOY**

## Artifacts

- genomes: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057/genomes.csv`
- best genome: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057/best_genome.json`
- top-k: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057/top_k_genomes.json`
- pareto front: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057/pareto_front.csv`
- split rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057/walkforward_results_by_split.csv`
- symbol risk rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057/risk_rollup_by_symbol.csv`
- overall risk rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057/risk_rollup_overall.csv`
