# SOL 3m Skip-Mask GA

- Generated UTC: `2026-03-06T22:30:54.932821+00:00`
- Run dir: `/root/analysis/0.87/reports/execution_layer/SOL_3M_SKIPMASK_GA_20260306_222738`
- Decision: `NO_REPAIR_APPROVED`
- Symbol: `SOLUSDT`
- Baseline strategy id lock: `M1_ENTRY_ONLY_PASSIVE_BASELINE`

## A) Baseline Metrics

| scope | trade_count | expectancy_net | cvar_5 | max_drawdown | instant_loser_rate | fast_loser_rate | bottom_decile_pnl_share | same_bar_exit_count | same_bar_touch_count | exit_before_entry_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train60 | 859 | 0.003264234029 | -0.002110040831 | -0.3486083175 | 0.7252619325 | 0.1338766007 | 0.1166605633 | 0 | 0 | 0 |
| full | 1432 | 0.001906509291 | -0.00211207445 | -0.1301395943 | 0.7136871508 | 0.1201117318 | 0.1195507485 | 0 | 1 | 0 |

## B) Best Candidate Metrics + Deltas

| candidate_id | feature_skip_mask_enabled | skip_mask_logic | skip_cap_signal_range_pct | skip_cap_upper_wick_ratio | delta_expectancy_vs_baseline | delta_cvar_vs_baseline | maxdd_improve_ratio | retention | winner_retention | instant_loser_rel_reduction | bottom_decile_rel_reduction | route_pass_rate | holdout_delta_expectancy_vs_baseline | holdout_delta_cvar_vs_baseline | holdout_loss_target_pass | holdout_instant_loser_rel_reduction | holdout_bottom_decile_rel_reduction | confirm_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GA_92b8eab39dd4044acb62e178 | 1 | or | 0.1759805902 | 0.413321999 | -0.0001560783723 | 0.0001846074743 | 0.144292716 | 0.748603352 | 0.82 | 0.03407862838 | 0.02319022302 | 0 | -0.0003245053965 | 0.0001821784286 | 1 | 0.03903004638 | 0.02605730203 | 0 |

## C) Gate Checklist (G0–G4)

| candidate_id | confirm_pass | gate_g0_chronology | gate_g1_participation | gate_g2_winner_preservation | gate_g3_risk_sanity | gate_improve_target | gate_g4_robustness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GA_92b8eab39dd4044acb62e178 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 1 | 1 | 1 | 1 | 0 | 0 |
| GA_eba8fcb19b9b3d22f47638a7 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| GA_3743e00395ee88d106d42849 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |

## D) Split / Route / Holdout Evidence

| candidate_id | split_valid_for_ranking | best_min_subperiod_delta | route_confirm_count | route_total | route_pass_rate | holdout_trade_count | holdout_delta_expectancy_vs_baseline | holdout_delta_cvar_vs_baseline | holdout_pass | holdout_loss_target_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GA_92b8eab39dd4044acb62e178 | 1 | -0.0005013511478 | 0 | 3 | 0 | 955 | -0.0003245053965 | 0.0001821784286 | 0 | 1 |
| M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 9.257456019e-05 | 0 | 3 | 0 | 955 | 0 | 0 | 0 | 0 |
| GA_eba8fcb19b9b3d22f47638a7 | 1 | 0.0001119774012 | 1 | 3 | 0.3333333333 | 955 | -3.098665212e-05 | 1.457427429e-05 | 0 | 1 |
| GA_3743e00395ee88d106d42849 | 1 | 9.904217387e-05 | 1 | 3 | 0.3333333333 | 955 | -3.679734588e-05 | 1.457427429e-05 | 0 | 1 |

## E) Proven vs Assumed

- Proven: SOL-only input came from repaired posture freeze active subset and locked winner config id.
- Proven: 1h signal layer/params remained frozen from repaired universe artifacts.
- Proven: evaluation path reused `phase_a` entry-only wrapper with 1h-owned exits and chronology protections.
- Proven: route family check required 3/3 confirmations for pass.
- Proven: bounded skip-mask family used 2 causal pre-entry features `signal_range_pct` and `upper_wick_ratio` only.
- Assumed: these two selected features are sufficient leverage for this bounded family (no extra families searched).

## F) Recommended Next Step

- Stop this GA branch; either redesign the allowed entry lever family or keep current SOL baseline as active execution posture.

## Top Screened Candidates

| generation | candidate_id | skip_cap_signal_range_pct | skip_cap_upper_wick_ratio | screen_pass | screen_objective | instant_loser_rel_reduction | bottom_decile_rel_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | GA_92b8eab39dd4044acb62e178 | 0.1759805902 | 0.413321999 | 0 | 24.21545348 | 0.0469338875 | 0.01238272065 |
| 0 | GA_eba8fcb19b9b3d22f47638a7 | 0.1289105807 | 0.8323428941 | 0 | 2.270969042 | 0.002388820697 | 0.008072845765 |
| 0 | GA_3743e00395ee88d106d42849 | 0.1266441889 | 0.8732032794 | 0 | 0.8896396365 | 0.0001595749674 | -0.0001735313453 |
| 0 | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.08 | 0.8 | 0 | 0 | 0 | 0 |