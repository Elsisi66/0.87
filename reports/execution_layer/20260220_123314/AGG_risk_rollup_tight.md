# AGG Risk Rollup Tight

- Source risk_dir: `/root/analysis/0.87/reports/execution_layer/20260220_123314`
- Source risk_overall_csv: `/root/analysis/0.87/reports/execution_layer/20260220_123314/risk_rollup_overall.csv`
- Source risk_by_symbol_csv: `/root/analysis/0.87/reports/execution_layer/20260220_123314/risk_rollup_by_symbol.csv`

- scope: overall
- symbols: 3
- signals_total: 1312
- baseline_mean_expectancy_net: -0.000565
- exec_mean_expectancy_net: -0.000766
- baseline_pnl_net_sum: -0.740827
- exec_pnl_net_sum: -1.004998
- baseline_pnl_std: 0.008607
- exec_pnl_std: 0.004760
- baseline_worst_decile_mean: -0.002200
- exec_worst_decile_mean: -0.002200
- baseline_cvar_5: -0.002200
- exec_cvar_5: -0.002200
- baseline_max_consecutive_losses: 84
- exec_max_consecutive_losses: 38
- baseline_SL_hit_rate_valid: 0.951220
- exec_SL_hit_rate_valid: 0.644757
- baseline_taker_share: 1.000000
- exec_taker_share: 0.130673
- baseline_median_fill_delay_min: 0.000000
- exec_median_fill_delay_min: 0.000000
- exec_median_entry_improvement_bps: 5.535745
- baseline_max_drawdown: -0.422941
- exec_max_drawdown: -0.371564
- delta_expectancy_exec_minus_baseline: -0.000201
- delta_cvar5_exec_minus_baseline: 0.000000
- delta_max_drawdown_exec_minus_baseline: 0.051377

- Rubric decision: **exec not worth it; focus on 1h edge/stops**
