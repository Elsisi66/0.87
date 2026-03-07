# NX1 Family Spec

- Generated UTC: 2026-02-27T11:53:59.488057+00:00
- Baseline execution genome source: E1 fallback-compatible genome, then family-specific entry mechanics wrappers.

## PASSIVE_LADDER_ADAPTIVE
- Multi-step passive limit ladder (2-3 steps) with adaptive offset from ATR-z/spread state.
- Bounded cancel/replace cadence via step_delay_min and max_fill_delay_min.
- Conditional fallback to market under toxicity cap (not unconditional taker fallback by default).

## REGIME_ROUTED_EXEC
- Regime-routed entry profile (calm/neutral/toxic) based on ATR-z + spread proxy.
- Each regime routes to distinct mode/delay/fallback behavior.
- Optional extreme-toxicity skip guard remains bounded and explicitly measurable.

## STAGED_ENTRY_RISKSHAPE
- Two-stage split entry schedule (stage1 + delayed stage2).
- Soft risk shaping via toxicity-dependent stage2 notional scaling (soft modulation, no hard starvation filter).
- Stage-wise fallback policy bounded and auditable.

All families keep: frozen subset, frozen fee/slippage model, unchanged hard gates, no hindsight fills.
