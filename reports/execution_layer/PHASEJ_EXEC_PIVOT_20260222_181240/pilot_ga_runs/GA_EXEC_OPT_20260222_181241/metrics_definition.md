# Metrics Definition

- Baseline entry: market at next 3m open after 1h signal timestamp (UTC).
- Candidate entry/exit decisions must use only information up to decision bar (no lookahead).
- `mean_expectancy_net`: mean per-signal pnl vector where non-filled/invalid rows contribute 0.
- `pnl_net_sum`: sum of per-signal net pnl values on TEST rows.
- `cvar_5`: mean of worst 5% per-signal pnl outcomes.
- `max_drawdown`: peak-to-trough drawdown on cumulative per-signal pnl curve.
- Entry-conditioned metrics (`taker_share`, `SL_hit_rate_valid`, delays) use valid filled rows only.
