# Phase H Metric Semantics Audit

- Generated UTC: 2026-02-22T14:50:18.090826+00:00
- Source formulas: `/root/analysis/0.87/scripts/backtest_exec_phasec_sol.py` and `/root/analysis/0.87/scripts/phase_g_sol_pathology_rehab.py`

## Exact Metric Definitions and Units

- `expectancy_net`: arithmetic mean of `pnl_net_pct` over valid filled trades. Unit: decimal return on position notional per trade.
- `total_return`: final compounded equity / initial equity - 1, where position size is risk-fractional (`equity * risk_per_trade / risk_pct`) each trade. Unit: decimal return on account equity.
- `max_drawdown_pct`: minimum peak-to-trough drawdown on compounded equity curve, `equity / rolling_peak - 1`. Unit: negative decimal fraction.
- `cvar_5`: mean of worst 5% of per-trade `pnl_net_pct` over valid filled trades. Unit: decimal per-trade return on position notional.
- `fatal_gate` (Phase G): 1 if `max_drawdown_pct <= -0.95` OR `total_return <= -0.95`; else 0.

## Why `expectancy_net>0` and `profit_factor>1` can coexist with ~-94% total return

The metrics are on different units/scales: expectancy/profit factor use unscaled trade `pnl_net_pct`, while total return uses equity-step returns scaled by `risk_per_trade / risk_pct` and compounded pathwise.
When `risk_pct` is very small (median around 0.001 in top uplift variants), the same `pnl_net_pct` maps to much larger equity-step moves; variance and downside clustering can make geometric growth strongly negative even if arithmetic trade expectancy is positive.

Representative paradox rows:
| variant | expectancy_net_trade_pct | profit_factor_trade | mean_risk_pct | mean_equity_step_return | reported_total_return_compounded | reported_max_drawdown_pct_compounded |
| --- | --- | --- | --- | --- | --- | --- |
| plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay1 | 0.000727 | 1.241123 | 0.002562 | -0.006581 | -0.942191 | -0.940891 |
| plus_uc_params_plus_regime_mod_plus_regime_gate | 0.000155 | 1.050695 | 0.002636 | -0.007576 | -0.996317 | -0.996235 |

## Consistency Verification (same trade sequence, fixed-size vs compounded)

- Recomputed compounded/fixed metrics from raw trade files match reported metrics within numerical tolerance.
- max_abs_total_return_error_compounded: 0.000000000000
- max_abs_total_return_error_fixed: 0.000000000001
- Detailed checks: `/root/analysis/0.87/reports/execution_layer/PHASEH_SOL_FREEZE_FORK_20260222_145017/phaseH_metric_consistency_checks.csv`

## Audit Verdict

- This is not an arithmetic bug in `total_return`.
- It is expected from path-dependent compounding under risk scaling, plus a semantics mismatch if optimization is guided by unscaled per-trade expectancy.
- Practical gate decisions should prioritize equity-based metrics (and fixed-size pre-gates) before compounding promotion.
