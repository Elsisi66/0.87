# Metrics Definition

- Baseline entry definition: market at the next 3m open strictly after 1h signal timestamp (UTC).
- Baseline exit definition: strategy TP/SL geometry from signal params, evaluated sequentially on 3m bars inside bounded evaluation window.
- Fee/slippage model must be identical baseline vs candidate in each run.

## Core Formulas

- `pnl_net_pct`: net return fraction after slippage and fees, measured versus entry price.
- `expectancy_net_per_signal`: mean of per-signal pnl vector where non-filled/invalid signals contribute 0.
- `expectancy_net_per_trade`: sum(pnl_net over valid filled trades) / valid_filled_trades.
- `cvar_5_per_signal`: mean of worst 5% values of the per-signal pnl vector.
- `max_drawdown_per_signal`: max peak-to-trough drawdown on cumulative per-signal pnl curve.

## Inclusion Rules

- Valid trade: `filled=1` and no invalid stop/tp geometry flags.
- Missing slice proxy: baseline rows with `baseline_filled=0` in test universe.
