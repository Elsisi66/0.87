# Phase F2 Next Family Spec

- Generated UTC: 2026-02-23T19:37:55.234261+00:00
- Family objective: replace brittle hard skip/cooldown gating with soft, state-aware risk sizing.
- Primary evidence basis:
  - Phase E: hard-filter family failed due to subperiod expectancy fragility, not global base-route pass failure.
  - Phase AE/AF: soft risk sizing proxies improved DD/tail metrics while preserving participation/invariance.

## Proposed Control Families

1. `risk_linear`: continuous down-scaling as AE risk score rises.
2. `risk_step`: interpretable piecewise sizing on risk score bins.
3. `state_streak_cap`: apply temporary sizing cap during elevated prior loss streak state.
4. `state_tail_cap`: apply sizing cap during elevated rolling tail-count state.
5. `hybrid_stateaware`: risk-linear base + state cap + controlled recovery bonus.

## Why this family

- Avoids hard trade starvation and route/subperiod over-fragility from binary skip gates.
- Directly targets AE predictors (`prior_loss_streak_len`, `prior_rolling_tail_count_20`, `pre3m_close_to_high_dist_bps`, interaction terms).
- Preserves existing execution mechanics and hard gates; only notional scaling changes.
