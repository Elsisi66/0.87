# A1 Frozen Exit Semantics

- Generated UTC: 2026-02-28T01:50:20.486885+00:00
- M0 (`pure 1h reference`) keeps the original 1h signal and 1h exit engine exactly as implemented in `backtest_exec_phasec_sol._simulate_1h_reference`.
- Model A wrapper keeps those exit semantics, but reuses them after a 3m entry fill as follows:
  - 3m decides only whether and when the entry fills
  - after fill, no 3m TP/SL/time-stop/trailing/break-even/partial-take logic is allowed
  - the 1h exit path is simulated on 1h candles only
  - to avoid partial-candle lookahead, the first 1h exit-evaluation candle is the first full 1h bar with timestamp strictly greater than the realized 3m fill time
  - TP/SL levels stay owned by the 1h strategy and are re-anchored to the realized entry price using the same `tp_mult/sl_mult`
  - the exit horizon remains the fixed frozen 1h horizon from the existing contract
  - if neither TP nor SL is hit before the horizon, exit is the final 1h close inside the horizon window
- Forbidden 3m exit knobs are blocked by construction because the wrapper does not accept or route them into execution.

- Forbidden downstream exit knobs excluded: `['tp_mult', 'sl_mult', 'time_stop_min', 'break_even_enabled', 'break_even_trigger_r', 'break_even_offset_bps', 'trailing_enabled', 'trail_start_r', 'trail_step_bps', 'partial_take_enabled', 'partial_take_r', 'partial_take_pct']`
