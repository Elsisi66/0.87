# S2 Objective Family Spec

- Generated UTC: 2026-02-28T01:22:35.845482+00:00
- Objective families are implemented as execution-aware signal-ranking / bounded-pruning prototypes over the frozen representative subset.
- They do not alter execution mechanics; they alter which 1h signals survive into the fixed E1/E2 execution scorers.

## Families

### FILL_RESCUE
- target metric: positive delta expectancy vs E1/E2 with entry-rate repair
- penalties: fill_delay_risk, spread, realized_vol, weak trend
- hard exclusions: none in spec; prototype uses bounded ranked pruning only
- support minimums: {'per_route_entries_min': 200}
- route-balance requirement: front/center/back removal budgets capped to preserve all repaired routes
- center-route anti-collapse: center route gets higher fill-risk pruning budget
- tail / CVaR safeguard: do not remove more than support headroom permits
- trade density safeguard: session caps on removals

### COST_DEFENSE
- target metric: reduce cost-killed edge and taker failures
- penalties: taker_risk, spread, ATR stress, adverse selection proxy
- hard exclusions: none in spec; bounded ranked pruning
- support minimums: {'per_route_entries_min': 200}
- route-balance requirement: front route gets largest taker-risk budget
- center-route anti-collapse: center budget retained but smaller than fill-rescue
- tail / CVaR safeguard: vol-bucket caps avoid concentrating removals in one regime
- trade density safeguard: session + vol caps

### TAIL_PENALIZED
- target metric: reduce tail contribution without collapsing density
- penalties: ATR stress, large bodies, wick imbalance, impulse
- hard exclusions: none in spec; bounded ranked pruning
- support minimums: {'per_route_entries_min': 200}
- route-balance requirement: balanced removal budgets across all repaired routes
- center-route anti-collapse: center route gets moderate extra budget
- tail / CVaR safeguard: tail-focused score only; no low-signal starvation allowed
- trade density safeguard: session + vol caps

### CENTER_GUARD
- target metric: eliminate center-route universal failure as primary bottleneck
- penalties: fill_delay_risk, taker_risk, spread, ATR stress
- hard exclusions: none in spec; bounded ranked pruning
- support minimums: {'per_route_entries_min': 200}
- route-balance requirement: all routes preserved, but center route gets largest budget
- center-route anti-collapse: explicit design focus
- tail / CVaR safeguard: back-route budget remains small to avoid overfit
- trade density safeguard: session caps
