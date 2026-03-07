# SOL Backtest Decision

- Generated UTC: 2026-02-22T00:27:52.594825+00:00
- Phase C cfg hash: `a285b86c4c22a26976d4a762`
- 1h reference note: Aligned 1h proxy using same frozen test signals, next-1h-open entry, same TP/SL multipliers, same 12h horizon, and identical fee/slippage contract.

## Practical Equity Conclusion

- Phase C best vs 3m control: expectancy delta = 0.000084, total_return delta = 0.003299
- Phase C best vs 1h reference: expectancy delta = 0.000090, total_return delta = 0.003648
- MaxDD phasec/control/1h: -0.998117 / -0.998986 / -0.999175

## Behavior Changes (3m control -> 3m phasec)

- Hold time median (min): 1.50 -> 0.00
- Win rate: 0.0500 -> 0.0450
- Profit factor: 0.6923 -> 0.6998
- Fees paid: 3.148718 -> 2.821171

## Recommendation

- Final: **HOLD**
- Blockers:
  - positive_practical=0
  - better_vs_3m_control=1
  - better_vs_1h_reference=1
  - maxdd_not_worse_vs_control=1
  - maxdd_not_worse_vs_1h=1

## Key Deltas Table

- expectancy_net: ctrl-1h=0.000006, phasec-ctrl=0.000084, phasec-1h=0.000090
- total_return: ctrl-1h=0.000349, phasec-ctrl=0.003299, phasec-1h=0.003648
- max_drawdown_pct: ctrl-1h=0.000190, phasec-ctrl=0.000869, phasec-1h=0.001059
- cvar_5: ctrl-1h=-0.000000, phasec-ctrl=0.000250, phasec-1h=0.000250
- profit_factor: ctrl-1h=0.001884, phasec-ctrl=0.007528, phasec-1h=0.009413
- win_rate: ctrl-1h=0.003333, phasec-ctrl=-0.005000, phasec-1h=-0.001667
- total_fees_paid: ctrl-1h=-0.279538, phasec-ctrl=-0.327547, phasec-1h=-0.607085
- median_fill_delay_min: ctrl-1h=0.000000, phasec-ctrl=0.000000, phasec-1h=0.000000
- p95_fill_delay_min: ctrl-1h=0.000000, phasec-ctrl=0.000000, phasec-1h=0.000000
