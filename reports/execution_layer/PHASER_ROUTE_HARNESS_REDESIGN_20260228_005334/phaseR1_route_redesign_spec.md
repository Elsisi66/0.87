# R1 Route Redesign Spec

- Generated UTC: 2026-02-28T00:54:05.411488+00:00
- Objective: keep route-based robustness meaningful while guaranteeing every route used in pass/fail logic is support-feasible under unchanged hard gates.
- Redesign rule:
  - Compute the smallest route size whose walkforward test windows can score at least `200` trades.
  - Under current settings, this minimum is `665` contexts (producing `200` test signals).
  - Build deterministic chronological windows of size `max(min_required, ceil(0.60 * N_subset))`.
  - Current run uses window size `720` over `N_subset=1200`.
- Route family used for pass/fail logic:
  - `route_front_60pct`: earliest support-feasible 60% window.
  - `route_center_60pct`: centered support-feasible 60% window.
  - `route_back_60pct`: latest support-feasible 60% window.
- Integrity properties:
  - deterministic on chronological order only (no label/outcome lookahead)
  - each route re-runs its own walkforward split construction
  - routes are deduplicated by exact signal-id membership
- Repair status: `ready`
- Repair blockers: `[]`

## Repaired Route Feasibility

| route_id | route_signal_count | wf_test_signal_count | hard_min_trades_overall | headroom_vs_overall_gate | route_trade_gates_reachable |
| --- | --- | --- | --- | --- | --- |
| route_back_60pct | 720 | 216 | 200 | 16 | 1 |
| route_center_60pct | 720 | 216 | 200 | 16 | 1 |
| route_front_60pct | 720 | 216 | 200 | 16 | 1 |
