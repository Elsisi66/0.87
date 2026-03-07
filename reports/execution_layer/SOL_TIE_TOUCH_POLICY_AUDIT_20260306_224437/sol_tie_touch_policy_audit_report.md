# SOL Tie-Touch Policy Audit

- Generated UTC: `2026-03-06T22:47:19.971898+00:00`
- Run dir: `/root/analysis/0.87/reports/execution_layer/SOL_TIE_TOUCH_POLICY_AUDIT_20260306_224437`
- Decision: `NO_CHANGE_RECOMMENDED`

## Policy Set

| policy_id | tie_touch_policy | conservative_for_approval | description |
| --- | --- | --- | --- |
| P1_SL_FIRST | sl_first | 1 | Conservative baseline: if SL and TP touched in same evaluated bar, resolve SL first |
| P2_TP_FIRST | tp_first | 0 | Optimistic sanity bound: resolve TP first on tie-touch |
| P3_DISTANCE_TO_ENTRY | distance_to_entry | 1 | Deterministic causal: resolve to boundary closer to entry price; equal distance falls back to SL |

## Tie-Touch Event Stats

| policy_id | tie_touch_count | tie_touch_rate | same_bar_exit_count | same_bar_touch_count | exit_before_entry_count | entry_on_signal_count |
| --- | --- | --- | --- | --- | --- | --- |
| P1_SL_FIRST | 1 | 0.0006983240223 | 0 | 1 | 0 | 0 |
| P2_TP_FIRST | 1 | 0.0006983240223 | 0 | 1 | 0 | 0 |
| P3_DISTANCE_TO_ENTRY | 1 | 0.0006983240223 | 0 | 1 | 0 | 0 |

## Tail Contribution from Tie-Touch Trades

| policy_id | worst10_tie_count | worst10_tie_share | worst25_tie_count | worst25_tie_share | tie_contrib_to_bottom_decile_share | tie_share_within_bottom_decile_losses |
| --- | --- | --- | --- | --- | --- | --- |
| P1_SL_FIRST | 0 | 0 | 0 | 0 | 0 | 0 |
| P2_TP_FIRST | 0 | 0 | 0 | 0 | 0 | 0 |
| P3_DISTANCE_TO_ENTRY | 0 | 0 | 0 | 0 | 0 | 0 |

## Policy Comparison (Splits + Routes + Holdout)

| policy_id | delta_expectancy_vs_control | delta_cvar_vs_control | maxdd_improve_ratio_vs_control | retention_vs_control | winner_retention_vs_control | instant_loser_rel_reduction | bottom_decile_rel_reduction | route_pass_rate | holdout_delta_expectancy_vs_control | holdout_delta_cvar_vs_control | gate_policy_update_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1_SL_FIRST | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| P2_TP_FIRST | 6.47295017e-05 | 0 | 0 | 1 | 1.006666667 | 0.0009784735812 | -0.0007758412405 | 0.3333333333 | 9.706036276e-05 | 0 | 0 |
| P3_DISTANCE_TO_ENTRY | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

## Proven vs Assumed

- Proven: tie-touch policy was the only changed execution variable; entries, participation mechanics, costs, and 1h signal layer were unchanged.
- Proven: chronology guard remained clean under all tested policies (no same-parent exit, no exit-before-entry, no pre-signal entry).
- Proven: split + full 3-route + holdout checks were applied policy-by-policy.
- Assumed: TP-first is an optimistic upper bound and is not conservative for promotion.

## Final Outcome

- `NO_CHANGE_RECOMMENDED`
- Selected policy row: `P1_SL_FIRST`
- Recommended next step: keep conservative SL-first tie handling and stop tie-policy branch; no robust improvement under strict harness.