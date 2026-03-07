# Phase U Harness Validation

- Generated UTC: 2026-02-22T20:49:04.585861+00:00
- Frozen representative subset CSV: `/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv`
- Canonical fee model path: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json`
- Canonical metrics definition path: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md`
- Fee hash: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a` (match=1)
- Metrics hash: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99` (match=1)
- Local freeze lock pass: `1`

## Entrypoints

- 1h signal engine: `src/bot087/optim/ga.py` (`build_entry_signal`, `run_backtest_long_only`)
- 3m execution evaluator: `src/execution/ga_exec_3m_opt.py` (`_prepare_bundles`, `_evaluate_genome`)

## Active/Base SOL Params

- Params file: `/root/analysis/0.87/data/metadata/params/SOLUSDT_C13_active_params_long.json`
- Params source: `best_by_symbol`
- best_by_symbol.csv: `/root/analysis/0.87/reports/params_scan/20260220_044949/best_by_symbol.csv`

## Execution Choice Set

- E0: baseline_execution_reference, genome_hash=`BASELINE_E0`, source=`computed_from_baseline_fields`
- E1: phaseS_primary, genome_hash=`862c940746de0da984862d95`, source=`/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057`
- E2: phaseS_backup, genome_hash=`992bd371689ba3936f3b4d09`, source=`/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057`
- E3: safe_simple_control, genome_hash=`10ba86f2f92b42bd2e0f00fdcecae2542b6c31e04a4deacd44c3bf03c518bfb2`, source=`derived_from_E1`
- E4: high_participation_realistic_control, genome_hash=`3bfd8a6da99c536be927cb97`, source=`/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057`

## Smoke

- Smoke rows: 5
- Smoke valid_for_ranking count: 0
- Smoke avg entry_rate: 0.000000
