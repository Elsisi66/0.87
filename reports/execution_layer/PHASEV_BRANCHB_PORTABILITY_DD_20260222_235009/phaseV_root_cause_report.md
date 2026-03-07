# Phase V Root Cause Report

- Generated UTC: 2026-02-22T23:51:41.357483+00:00
- Classification: **B**
- Reason: sol_overlay_material_dd_tail_improvement

## Portability Snapshot

- non_SOL_targets: AVAXUSDT, NEARUSDT, BCHUSDT
- non_SOL_evaluated: AVAXUSDT, NEARUSDT, BCHUSDT
- portability_GO_coins: none
- portability_feasible_but_weak: none

## SOL Drawdown Drivers

- scenario: SOL_P16_E1
- max_consecutive_losses: 45
- loss_run_ge3_count: 21
- worst_split_id/expectancy: 2 / -0.00114609
- worst_session_bucket/pnl_sum: 12_17 / -0.04960238
- worst_regime_bucket/pnl_sum: mid|up / -0.01115112
- sl_loss_share: 0.898553
- fee_drag_per_trade: 0.00088028

## Overlay Snapshot

- best_overlay: O1 (session_veto_worst:06_11) on SOL_BASE_E1
- delta_expectancy_vs_base: 0.00012263, maxdd_improve_ratio_vs_base: 0.305944, cvar_improve_ratio_vs_base: 0.054602, hard_gate_proxy_pass: 1
