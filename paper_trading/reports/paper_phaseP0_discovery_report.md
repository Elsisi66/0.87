# Phase P0 Discovery Report

- Generated UTC: `2026-02-27T11:29:30+00:00`
- Project root: `/root/analysis/0.87`

## Signal Engine Entrypoint
- Canonical strategy/backtest implementation discovered at:
  - `src/bot087/optim/ga.py`
- Key functions used for paper contract parity:
  - `_ensure_indicators` (indicator prep)
  - `compute_cycles` + `_shift_cycles` (regime label with one-bar lag)
  - `build_entry_signal` (strict no-lookahead signal)
  - `run_backtest_long_only` (entry/exit semantics and execution contract)

## Backtest Timing Contract (Discovered)
- `cycle_shift=1` default is defined in GA config, forcing use of `t-1` regime for decisions at `t` open.
- Signal features use `shift(1)` (`RSI`, `WILLR`, `ATR`, EMA/ADX/DI derived features).
- Entry is executed at bar `t` open when `signal[t]` is true.
- Exit ordering is deterministic: SL/TP checks on bar high/low, then max-hold, then RSI exit.
- No intrabar future data is used for signal generation.

## Fee / Slippage Assumptions (Discovered)
- GA defaults: fee `7 bps`, slippage `2 bps` in deterministic backtests.
- Paper daemon settings use fee `7 bps` and randomized slippage set `{2,5,7,10,12}` per fill.
- Source config used by paper service:
  - `/root/analysis/0.87/paper_trading/config/settings.yaml`

## Universe Discovery
- Canonical source selected: `/root/analysis/0.87/reports/params_scan/20260220_044949/best_by_symbol.csv`
- Resolved universe written to:
  - `/root/analysis/0.87/paper_trading/config/resolved_universe.json`
- Resolved symbols (13):
  - `ADAUSDT, AVAXUSDT, AXSUSDT, BCHUSDT, CRVUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, NEARUSDT, SOLUSDT, TRXUSDT, XRPUSDT, ZECUSDT`

