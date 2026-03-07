# GA Exec 3m Decision

- Generated UTC: 2026-02-20T15:24:24.597692+00:00
- Run dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402`
- Best genome hash: `6756175bfd01c7771056f983`
- Repro check pass: 1

## Baseline vs Best (Overall TEST-only)

- baseline_expectancy_net: -0.001990
- exec_expectancy_net: 0.000000
- delta_expectancy_exec_minus_baseline: 0.001990
- baseline_cvar_5: -0.002200
- exec_cvar_5: 0.000000
- cvar_improve_ratio: 1.000000
- baseline_max_drawdown: -0.117204
- exec_max_drawdown: 0.000000
- maxdd_improve_ratio: 1.000000
- exec_entry_rate: 0.000000
- exec_taker_share: nan
- exec_median_fill_delay_min: nan

## Gate Table

- expectancy >= baseline: 1
- cvar_improve >= 15%: 1
- maxdd_improve >= 15%: 1
- taker_share <= 0.25: 0
- median_fill_delay <= 45 min: 0
- per-symbol entry-rate gates pass: 0
- split stability pass: 1 (min=0.000000, median=0.000000)

## Final

- Decision: **NO-DEPLOY**

## Artifacts

- genomes: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402/genomes.csv`
- best genome: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402/best_genome.json`
- top-k: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402/top_k_genomes.json`
- pareto front: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402/pareto_front.csv`
- split rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402/walkforward_results_by_split.csv`
- symbol risk rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402/risk_rollup_by_symbol.csv`
- overall risk rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402/risk_rollup_overall.csv`
