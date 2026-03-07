# Phase D3 Monitoring Spec

Track these metrics separately for `paper_primary` and `shadow_backup`, plus a direct diff between them:
- daily_pnl_quote
- daily_pnl_pct_net
- trades_total
- entries_valid
- entry_rate
- avg_fill_delay_min
- p95_fill_delay_min
- taker_share
- realized_win_rate
- realized_avg_pnl_per_trade
- realized_sl_hit_rate
- realized_tp_hit_rate
- realized_drawdown_pct
- divergence_vs_expected_delta
- divergence_vs_expected_taker_share
- divergence_vs_expected_fill_delay
- primary_minus_backup_expectancy
- primary_minus_backup_drawdown

Contract-violation counters:
- hybrid_exit_mutation_detected_count
- missing_signal_count
- runtime_mismatch_count
- impossible_fill_path_count
- stale_data_count
- execution_exception_count
