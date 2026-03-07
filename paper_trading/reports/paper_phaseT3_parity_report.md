# Phase T3 Parity Report

- Signal candle used by backtest engine:
  - action row is candle `N`
  - signal for candle `N` is computed from candle `N-1` features because `build_entry_signal(...)` uses `shift(1)` and `cycle_shift=1`
- Action candle used by paper execution:
  - `ExecutionSimulator.process_bar(...)` opens at candle `N` `Open`
- SL/TP evaluation candle:
  - evaluated on candle `N` `High` / `Low`
  - if both TP and SL are hit in the same bar, the engine chooses `SL` first (same as backtest)
- RSI/max-hold exit timing:
  - still evaluated on the same 1h row after the open/high/low checks, matching `ga.py`
- Exact parity evidence:
  - `/root/analysis/0.87/paper_trading/reports/paper_phaseP2_parity_check_results.json` has `overall_parity_ok=true`
  - structural mismatches were zero for signal/cycle alignment on sampled symbols
