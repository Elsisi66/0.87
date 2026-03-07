# NX0 Engine Discovery

- Generated UTC: 2026-02-27T11:53:29.637626+00:00
- Reused evaluator: `src/execution/ga_exec_3m_opt.py`
- Reused functions:
  - `_validate_and_lock_frozen_artifacts` for canonical freeze hash lock + fee/slippage parity
  - `_prepare_bundles` for representative signal -> 3m context materialization
  - `_aggregate_rows` and `_symbol_thresholds` for unchanged metrics/gate semantics
  - `_simulate_dynamic_exit_long` for mechanics-valid exit simulation
- Frozen scoring fields consumed:
  - valid_for_ranking, invalid_reason
  - overall_exec_expectancy_net, overall_delta_expectancy_exec_minus_baseline
  - overall_cvar_improve_ratio, overall_maxdd_improve_ratio
  - overall_entries_valid, overall_entry_rate
  - overall_exec_taker_share, overall_exec_median_fill_delay_min, overall_exec_p95_fill_delay_min
- Representative subset schema snapshot:
  - columns=['signal_id', 'signal_time', 'tp_mult', 'sl_mult', 'atr_percentile_1h', 'trend_up_1h']
  - rows=1200
