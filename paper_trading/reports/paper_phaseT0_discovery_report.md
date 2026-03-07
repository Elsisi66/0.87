# Phase T0 Discovery Report

- Mode: paper-trading repair for forward-only hourly operation.
- Backlog replay root cause before repair:
  - `processed_bars.json` was the only forward marker.
  - `SignalRunner.rows_after(...)` returned all rows after the last marker.
  - `Reconciler.process_symbol_rows(...)` iterated every eligible row, so if the marker was old or moved back, the daemon mass-simulated historical trades in one batch.
  - `run_daemon(..., replay_bars=...)` actively rewound markers and enabled historical catch-up.
- 1h signal source today:
  - `paper_trading/app/signal_runner.py` calls `src/bot087/optim/ga.py` functions `_ensure_indicators`, `compute_cycles`, `_shift_cycles`, `build_entry_signal`.
- Processed bars tracking:
  - Per-symbol markers in `/root/analysis/0.87/paper_trading/state/processed_bars.json`.
- Repair decision:
  - introduce a forward anchor `start_from_bar_ts` in `/root/analysis/0.87/paper_trading/state/runtime_meta.json`
  - clear processed markers on reset
  - process only the newest eligible closed 1h bar per symbol
  - ignore replay requests in forward-only mode
