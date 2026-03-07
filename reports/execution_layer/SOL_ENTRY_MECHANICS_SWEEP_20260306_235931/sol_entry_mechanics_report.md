# SOL Entry Mechanics Sweep

- Generated UTC: `2026-03-07T00:02:39.982642+00:00`
- Run dir: `/root/analysis/0.87/reports/execution_layer/SOL_ENTRY_MECHANICS_SWEEP_20260306_235931`
- Decision: `NO_CHANGE_RECOMMENDED`
- Symbol: `SOLUSDT`
- Baseline strategy lock: `M1_ENTRY_ONLY_PASSIVE_BASELINE`

## Stage 1 Screen (Train 60%)

| variant_id | screen_pass | trade_count | retention | winner_retention | delta_expectancy_vs_baseline | instant_loser_rel_reduction | fast_loser_rel_reduction | bottom_decile_rel_reduction | same_bar_exit_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0_BASELINE | 1 | 859 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| V2_DELAYED_REPRICE_GUARD | 1 | 859 | 1 | 1.052631579 | -0.0002134374149 | 0.04654895666 | -0.1565217391 | 0.05368238491 | 0 |

## Stage 2 Confirm (Splits + Routes + Holdout)

| variant_id | confirm_pass | gate_g0_chronology | gate_g1_participation | gate_g2_winner_preservation | gate_g3_risk_sanity | gate_improve_target | gate_g4_robustness | split_support_pass | route_pass_rate | holdout_pass | retention | winner_retention | delta_expectancy_vs_baseline | delta_cvar_vs_baseline | maxdd_improve_ratio | instant_loser_rel_reduction | fast_loser_rel_reduction | bottom_decile_rel_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2_DELAYED_REPRICE_GUARD | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1.16 | 0.0004503300026 | -8.744564571e-05 | -0.2393008005 | 0.04892367906 | -0.06976744186 | 0.04099570288 |
| V0_BASELINE | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

## Fill-Type and Adverse Selection Proxy

| variant_id | fills_valid | fill_limit_count | fill_market_count | fill_market_fallback_count | fill_market_guard_fallback_count | fill_limit_cap_fallback_count | fill_market_cap_fallback_count | maker_fill_share | taker_fill_share | entry_improvement_bps_mean | entry_price_vs_signal_open_bps_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V2_DELAYED_REPRICE_GUARD | 1432 | 0 | 979 | 0 | 453 | 0 | 0 | 0 | 1 | 7.54310745 | -7.54310745 |
| V0_BASELINE | 1432 | 1368 | 0 | 64 | 0 | 0 | 0 | 0.9553072626 | 0.04469273743 | -0.6426708666 | 0.6426708666 |

## Proven vs Assumed

- Proven: 1h signal layer and exits remained frozen; only entry price formation knobs were varied.
- Proven: chronology checks remained enforced (same-parent-bar exit=0, exit-before-entry=0, entry-on-signal precondition=0).
- Proven: split support + full 3-route family + holdout checks were applied per variant.
- Assumed: fallback-to-market branches model realistic fill behavior under the existing simulator cost model (no free fills).

## Final Recommendation

- `NO_CHANGE_RECOMMENDED`
- Best row by confirm ordering: `V2_DELAYED_REPRICE_GUARD`
- Next step: keep current baseline entry mechanics; this bounded sweep did not produce a robust improvement.