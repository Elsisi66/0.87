# GA Exec 3m Decision

- Generated UTC: 2026-02-20T19:45:32.248458+00:00
- Run dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_194215`
- Best genome hash: `2fdf64b8063ce39a216d4b28`
- Repro check pass: 1

## Baseline vs Best (Overall TEST-only)

- baseline_expectancy_net: -0.001298
- exec_expectancy_net: 0.000199
- delta_expectancy_exec_minus_baseline: 0.001497
- baseline_cvar_5: -0.002200
- exec_cvar_5: -0.001162
- cvar_improve_ratio: 0.471922
- baseline_max_drawdown: -0.310762
- exec_max_drawdown: -0.002613
- maxdd_improve_ratio: 0.991590
- exec_entry_rate: 0.205556
- exec_taker_share: 0.000000
- exec_median_fill_delay_min: 0.00

## Gate Table

- expectancy >= baseline: 1
- cvar_improve >= 15%: 1
- maxdd_improve >= 15%: 1
- taker_share <= 0.25: 1
- median_fill_delay <= 45 min: 1
- per-symbol entry-rate gates pass: 0
- split stability pass: 0 (min=-0.000218, median=0.000135)

## Final

- Decision: **NO-DEPLOY**

## Artifacts

- genomes: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_194215/genomes.csv`
- best genome: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_194215/best_genome.json`
- top-k: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_194215/top_k_genomes.json`
- pareto front: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_194215/pareto_front.csv`
- split rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_194215/walkforward_results_by_split.csv`
- symbol risk rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_194215/risk_rollup_by_symbol.csv`
- overall risk rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_194215/risk_rollup_overall.csv`
