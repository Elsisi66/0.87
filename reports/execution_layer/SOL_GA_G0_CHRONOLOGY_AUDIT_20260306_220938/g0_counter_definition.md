# G0 Counter Definition

## Previous behavior (label source of false positive)

- `scripts/sol_3m_lossconcentration_ga.py` used `sum(exec_same_bar_hit)` as `same_bar_exit_count`.
- `scripts/phase_a_model_a_audit.py` forwards `same_bar_hit` from `execution_layer_3m_ict._simulate_path_long`.
- `scripts/execution_layer_3m_ict.py` sets `same_bar_hit=1` when **both** SL and TP are touched in the same evaluated bar (`hit_sl and hit_tp`), then SL wins.
- This flag is an intra-exit-candle tie marker, not an entry/exit same-parent chronology violation.

## Patched behavior (true chronology check)

- `same_bar_exit_count` now counts trades where `floor(entry_time,1h) == floor(exit_time,1h)` among filled+valid trades.
- `same_bar_touch_count` is retained as separate diagnostics.
- `entry_on_signal_count` now tracks only pre-signal entries (`entry_time < signal_time`) as chronology violations.
- `parity_clean` now requires: same_parent_exit=0, exit_before_entry=0, entry_before_signal=0, invalid geometry=0, lookahead=0.