# Phase T1 Repair Report

- Forward-only anchor added:
  - `/root/analysis/0.87/paper_trading/state/runtime_meta.json`
  - field: `start_from_bar_ts`
- Backlog replay disabled by code changes:
  - `SignalRunner.rows_after(...)` now supports `start_from_bar_ts` and `latest_only=True`
  - `Reconciler.process_symbol_rows(...)` now processes only one newest eligible closed 1h bar
  - `run_daemon(..., replay_bars=...)` now logs and ignores replay requests in forward-only mode
  - `run_daemon(...)` bootstraps `start_from_bar_ts` automatically if missing, instead of replaying old bars
- Closed-bar-only behavior:
  - cycle loop uses `DataFeed.latest_closed_bar_ts(...)`
  - no currently forming candle is processed
- Lower timeframe status:
  - paper branch still uses only `fetch_ohlcv_1h(...)`
  - no 15m / minute execution path is active in `paper_trading/app`
- Evidence:
  - smoke validation produced zero backlog fills immediately after reset
  - see `paper_phaseT4_smoke_results.json`
