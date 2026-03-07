# Model B2 Sizing Family Spec

Model B is pure sizing only:
- No signal logic changes.
- No 3m entry logic changes.
- No 1h TP/SL/exit logic changes.
- Only `effective position size` is scaled trade by trade.

## Families
- `baseline_primary`
  - frozen Model A primary, multiplier always 1.0
- `fixed_half_on_risk`
  - constant bounded size multiplier across all trades
  - tests pure risk cut with zero state dependence
- `step_down_after_streak`
  - after two consecutive realized losses, downsize until the next win resets the streak
  - changes only exposure after confirmed adverse state
- `linear_risk_score_scale`
  - uses rolling 8-trade realized loss ratio to scale size linearly down to a bounded floor
  - no signal mutation, only size response to prior realized outcomes
- `tail_cap_size`
  - caps exposure when running cumulative PnL is below a fixed drawdown threshold
  - targets tail compression and smoother equity during losing clusters
- `regime_cap_size`
  - caps exposure only on delayed 3m fills, which are an observable entry state and materially underperform the immediate fills in the frozen primary
  - preserves all fills and exits; only resizes delayed-entry exposure
