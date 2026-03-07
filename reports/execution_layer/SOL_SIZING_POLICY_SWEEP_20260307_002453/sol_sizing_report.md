# SOL Sizing / Risk-Per-Trade Bounded Sweep

- Generated UTC: `2026-03-07T00:27:58.581859+00:00`
- Run dir: `/root/analysis/0.87/reports/execution_layer/SOL_SIZING_POLICY_SWEEP_20260307_002453`
- Decision: `NO_CHANGE_RECOMMENDED`
- Symbol: `SOLUSDT`
- Baseline strategy lock: `M1_ENTRY_ONLY_PASSIVE_BASELINE`

## Causal Feature Set Used

- `feature_skip_signal_range_pct` (signal-bar range fraction)
- `feature_skip_upper_wick_ratio` (signal-bar upper-wick ratio)
- Both are generated in the Model A signal simulation at decision time from the signal bar only.

## Baseline Snapshot

| metric | value |
| --- | --- |
| baseline_expectancy_net | 0.001906509291 |
| baseline_cvar_5 | -0.00211207445 |
| baseline_max_drawdown | -0.1301395943 |
| baseline_trade_count | 1432 |
| baseline_bottom_decile_pnl_share | 0.1195507485 |
| baseline_same_bar_exit_count | 0 |
| baseline_route_pass_rate_strict_lineage | 1 |
| tail_cvar_abs_improve_threshold | 0.0001056037225 |

## Stage 1 Screen (Train 60%)

| variant_id | policy_family | screen_score | retention | delta_expectancy_vs_baseline | delta_cvar_vs_baseline | maxdd_improve_ratio | bottom_decile_rel_reduction | size_mult_mean | size_mult_reduced_share | parity_clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1_VOL_CAP_K0P75 | vol_cap | 22.90724883 | 1 | -0.000915067768 | 0.0005459277071 | 0.2583805522 | -0.007522934297 | 0.737477331 | 1 | 1 |
| V3_SCORE_BUCKET | score_bucket | 11.82972424 | 1 | -0.0009666775164 | 0.0002196309241 | 0.141525678 | -0.1692409474 | 0.8050058207 | 0.5518044237 | 1 |
| V1_VOL_CAP_K0P90 | vol_cap | 9.719472112 | 1 | -0.0004452345158 | 0.0002331050824 | 0.1100566626 | -0.007522934297 | 0.8849727972 | 1 | 1 |
| V2_WICK_CAP0P60_M0P75 | wick_cap | 1.701544658 | 1 | 2.105756135e-06 | 9.761374405e-05 | 0.01866215681 | 0.001794918465 | 0.9784633295 | 0.08614668219 | 1 |
| V2_WICK_CAP0P55_M0P50 | wick_cap | 0.7787865012 | 1 | -4.796504199e-05 | 0.0001138827014 | 0.009867246506 | -0.03530088256 | 0.9412107101 | 0.1175785797 | 1 |
| V0_BASELINE | constant | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |

## Stage 2 Confirm (Splits + 3 Routes + Holdout)

| variant_id | confirm_pass | gate_g0_chronology | gate_g1_participation | gate_g2_robustness | gate_g3_holdout | gate_g4_tail_objective | split_support_pass | route_pass_rate | holdout_delta_expectancy_vs_baseline | holdout_delta_cvar_vs_baseline | retention | delta_expectancy_vs_baseline | delta_cvar_vs_baseline | maxdd_improve_ratio | bottom_decile_rel_reduction | size_mult_mean | size_mult_reduced_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V0_BASELINE | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 |
| V2_WICK_CAP0P60_M0P75 | 0 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | -2.11762024e-05 | 8.015850857e-05 | 1 | -6.675252733e-06 | 7.772946286e-05 | 0.01597848287 | -0.004726042675 | 0.9769553073 | 0.09217877095 |
| V2_WICK_CAP0P55_M0P50 | 0 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | -4.934640813e-05 | 0.0001093070571 | 1 | -9.130124309e-06 | 0.0001165941943 | 0.06575943928 | -0.04229878062 | 0.9354050279 | 0.1291899441 |
| V1_VOL_CAP_K0P90 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0.6666666667 | -0.0003012476892 | 0.0002252253801 | 1 | -0.0002313514712 | 0.0002282059754 | 0.1062949006 | -0.005398063827 | 0.8886594688 | 1 |
| V3_SCORE_BUCKET | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | -0.0006520645911 | 0.00020403984 | 1 | -0.000467173707 | 0.0002088979314 | 0.01469611425 | -0.1021276133 | 0.8578910615 | 0.4441340782 |
| V1_VOL_CAP_K0P75 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 0 | -0.0006646353792 | 0.0005336276108 | 1 | -0.0005105444412 | 0.0005421840545 | 0.2552457505 | -0.005398063827 | 0.7405495573 | 1 |

## Proven vs Assumed

- Proven: signal generation, entries, exits, chronology, and costs remain unchanged from baseline; only per-trade pnl scaling is applied.
- Proven: participation and entry timing remained unchanged (retention ~1.0; no skip-mask or entry timing mutation).
- Proven: strict robustness checks were applied using split + full 3-route family + holdout.
- Assumed: feature columns are reliably present for all signals in this frozen branch (missing values fall back to neutral size=1.0).

## Final Recommendation

- `NO_CHANGE_RECOMMENDED`
- No non-baseline sizing policy passed all strict gates.
- Next step: keep current SOL baseline sizing unchanged.