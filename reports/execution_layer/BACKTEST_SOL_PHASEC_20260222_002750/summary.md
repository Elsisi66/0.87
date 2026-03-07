# SOL Execution-Layer Backtest Summary

- Generated UTC: 2026-02-22T00:27:52.593451+00:00
- Run dir: `/root/analysis/0.87/reports/execution_layer/BACKTEST_SOL_PHASEC_20260222_002750`
- Phase C source: `/root/analysis/0.87/reports/execution_layer/PHASEC_SOL_20260221_231430`
- Phase A contract: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310`
- 1h reference note: Aligned 1h proxy using same frozen test signals, next-1h-open entry, same TP/SL multipliers, same 12h horizon, and identical fee/slippage contract.

## Topline

- 1H_REFERENCE_CONTROL expectancy_net / total_return / maxDD: -0.000649 / -0.996799 / -0.999175
- EXEC_3M_CONTROL expectancy_net / total_return / maxDD: -0.000643 / -0.996450 / -0.998986
- EXEC_3M_PHASEC_BEST expectancy_net / total_return / maxDD: -0.000559 / -0.993151 / -0.998117

## Required Deltas

- expectancy_net: ctrl-1h=0.000006, phasec-ctrl=0.000084, phasec-1h=0.000090
- total_return: ctrl-1h=0.000349, phasec-ctrl=0.003299, phasec-1h=0.003648
- max_drawdown_pct: ctrl-1h=0.000190, phasec-ctrl=0.000869, phasec-1h=0.001059
- cvar_5: ctrl-1h=-0.000000, phasec-ctrl=0.000250, phasec-1h=0.000250
- profit_factor: ctrl-1h=0.001884, phasec-ctrl=0.007528, phasec-1h=0.009413
- win_rate: ctrl-1h=0.003333, phasec-ctrl=-0.005000, phasec-1h=-0.001667
