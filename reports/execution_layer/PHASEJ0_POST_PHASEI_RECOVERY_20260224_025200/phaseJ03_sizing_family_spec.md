# Phase J03 Soft Sizing Family Spec

- Generated UTC: 2026-02-24T02:53:13.889260+00:00
- Objective: reduce drawdown/tail/clustering burden without participation starvation.
- Scope: sizing/risk modulation only on top of proven execution winners (E1/E2).
- Hard filters are excluded as primary lever.

## Families

A) `size_step_by_risk_score` (piecewise monotonic).
B) `size_linear_by_risk_score` (smooth monotonic).
C) `size_cap_after_loss_streak` (state throttle).
D) `size_cap_after_tail_count` (state throttle).
E) `cooldown-lite + reduced size` (approximated as state-trigger downsize, no skip).
F) `hybrid_stateaware` (risk score + state interactions).
G) Optional extreme-risk tightening reserved for future engine support (not activated here).

## Design Guards

- No entry/exit mechanics change.
- No trade filtering by default (size floor > 0).
- Mean-size normalization active.
- Same fee/slippage and hard-gate contract.
