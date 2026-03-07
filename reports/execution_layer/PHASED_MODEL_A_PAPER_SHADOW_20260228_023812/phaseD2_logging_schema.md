# Phase D2 Logging Schema

Every paper/shadow trade event must emit these fields:
- `candidate_role` (`paper_primary` or `shadow_backup`)
- `candidate_id`
- `symbol`
- `signal_timestamp_1h`
- `signal_bar_timestamp_1h`
- `signal_state_hash`
- `entry_wrapper_mode`
- `entry_mode`
- `entry_limit_offset_bps`
- `fallback_to_market`
- `fallback_delay_min`
- `max_fill_delay_min`
- `entry_fill_timestamp_3m`
- `entry_fill_price`
- `entry_fill_type`
- `entry_fill_delay_min`
- `entry_liquidity_type`
- `taker_share_flag`
- `exit_timestamp_1h`
- `exit_price`
- `exit_reason_1h`
- `tp_hit`
- `sl_hit`
- `realized_pnl_pct_net`
- `realized_pnl_quote`
- `contract_path`
- `model_a_purity_ok`
- `hybrid_exit_mutation_detected`
- `runtime_mismatch_flag`

Current `paper_trading/state/journal.jsonl` does not include `candidate_id`, `entry wrapper knobs`, or `hybrid_exit_mutation_detected`, so the existing paper daemon is insufficient for this schema.
