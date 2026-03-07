# GA Exec 3m Decision

- Generated UTC: 2026-02-21T21:35:18.941635+00:00
- Run dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342`
- Best genome hash: `f0c7df09af14536a559ed913`
- Repro check pass: 1

## Baseline vs Best (Overall TEST-only)

- baseline_expectancy_net: -0.000643
- exec_expectancy_net: -0.000738
- delta_expectancy_exec_minus_baseline: -0.000095
- baseline_cvar_5: -0.002200
- exec_cvar_5: -0.001926
- cvar_improve_ratio: 0.124129
- baseline_max_drawdown: -0.590052
- exec_max_drawdown: -0.443874
- maxdd_improve_ratio: 0.247737
- exec_entry_rate: 0.915000
- exec_taker_share: 0.041894
- exec_median_fill_delay_min: 0.00
- exec_p95_fill_delay_min: 69.00

## Gate Table

- expectancy >= baseline: 0
- cvar_improve >= 15%: 0
- maxdd_improve >= 15%: 1
- taker_share <= 0.25: 1
- median_fill_delay <= 45 min: 1
- p95_fill_delay <= 360 min: 1
- per-symbol entry-rate gates pass: 1
- split stability pass: 1 (min=-0.000962, median=-0.000721)

## Final

- Decision: **NO-DEPLOY**

## Artifacts

- genomes: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342/genomes.csv`
- best genome: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342/best_genome.json`
- top-k: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342/top_k_genomes.json`
- pareto front: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342/pareto_front.csv`
- split rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342/walkforward_results_by_split.csv`
- symbol risk rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342/risk_rollup_by_symbol.csv`
- overall risk rollup: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260221_213342/risk_rollup_overall.csv`
