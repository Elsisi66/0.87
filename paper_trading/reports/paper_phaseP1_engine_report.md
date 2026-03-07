# Phase P1 Engine Report

- Generated UTC: `2026-02-27T11:29:30+00:00`

## Implemented Components
- Core daemon:
  - `paper_trading/app/main.py`
- Local paper ledger / execution simulation:
  - `paper_trading/app/execution_sim.py`
  - `paper_trading/app/portfolio.py`
- State persistence:
  - `paper_trading/app/state_store.py`
- Signal integration:
  - `paper_trading/app/signal_runner.py`
- Reconciliation/idempotent bar processing:
  - `paper_trading/app/reconciler.py`

## Startup Reset Behavior
- On startup reset:
  - positions reset to zero
  - orders cleared
  - processed bar markers initialized per symbol
  - reset marker persisted
- Latest reset marker:
  - `/root/analysis/0.87/paper_trading/state/startup_reset_marker.json`
- Latest reset report:
  - `/root/analysis/0.87/paper_trading/reports/startup_reset_report_20260227_112850.md`
- Result flag: `local_hard_reset_applied`

## Capital and Accounting
- Initial virtual equity configured to `320 EUR`.
- EUR accounting with quote-to-EUR conversion (fallback 1:1 when FX API unavailable).
- Tracks realized PnL, fees, slippage, trade counts, wins/losses.

## Idempotency / Persistence
- Per-symbol bar marker in `processed_bars.json` prevents duplicate fills across restarts.
- Append-only execution/signal journal in `journal.jsonl`.
- Dead-letter queue in `dead_letter_queue.jsonl` for unrecoverable events.

