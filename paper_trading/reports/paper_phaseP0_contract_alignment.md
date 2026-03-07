# Phase P0 Contract Alignment

- Generated UTC: `2026-02-27T11:29:30+00:00`

## Contract Mapping
- Backtest signal logic (`ga.py::build_entry_signal`) is reused directly by `paper_trading/app/signal_runner.py`.
- Cycle lag contract (`cycle_shift=1`, `cycle_fill=2`) is enforced in signal runner.
- Closed-bar processing is enforced in daemon cycle loop via `DataFeed.latest_closed_bar_ts(...)`.
- Reconciler only processes rows strictly after persisted symbol marker (`processed_bars.json`) and up to last closed bar.
- Entry/exit semantics in `execution_sim.py` mirror backtest ordering:
  - open at bar open
  - SL/TP priority
  - max-hold
  - RSI exit
- No lookahead:
  - All indicator inputs for signal decisions are from shifted (`t-1`) values.
  - Processing cutoff excludes forming 1h candle.

## Parity Evidence
- Structural parity report:
  - `/root/analysis/0.87/paper_trading/reports/paper_phaseP2_parity_check_report.md`
- Structural and deterministic parity result:
  - `/root/analysis/0.87/paper_trading/reports/paper_phaseP2_parity_check_results.json`
- Outcome: `overall_parity_ok=true`.

