# G0 Offending Trade Trace

Single reproduced event previously counted as `same_bar_exit_count=1`:

| candidate_id | genome_hash | signal_id | split_id | signal_time_utc | entry_time_utc | exit_time_utc | entry_parent_1h | exit_parent_1h | same_parent_bar | exit_before_entry | exec_exit_reason | exec_sl_hit | exec_tp_hit | exec_same_bar_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GA_ed12f01240bb9afc945178d5 | bbb989cf7115ff20d97207fe | SOLUSDT_2025-01-18T15:00:00+00:00 | 2 | 2025-01-18 15:00:00+00:00 | 2025-01-18 15:00:00+00:00 | 2025-01-18 16:00:00+00:00 | 2025-01-18 15:00:00+00:00 | 2025-01-18 16:00:00+00:00 | 0 | 0 | sl | 1 | 1 | 1 |

Route context for this signal:

| route_id | contains_offending_signal | valid_for_ranking | delta_expectancy_vs_baseline | cvar_improve_ratio | maxdd_improve_ratio | confirm_flag |
| --- | --- | --- | --- | --- | --- | --- |
| route_back_60pct | 1 | 1 | 0.0004158270166 | 0.08136253499 | 0.2863625287 | 1 |
| route_center_60pct | 0 | 1 | 0.0004902666525 | 0.07026764385 | 0.2860417696 | 1 |
| route_front_60pct | 0 | 1 | 0.0007312209831 | 0.08136253499 | 0.2720973141 | 1 |

- Interpretation: entry and exit occur on different 1h parent bars; this is not same-parent-bar exit.
- The `exec_same_bar_hit=1` is an SL/TP tie-in-candle marker, not chronology break.