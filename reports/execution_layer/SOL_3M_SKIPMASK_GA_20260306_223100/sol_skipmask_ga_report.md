# SOL 3m Skip-Mask GA

- Generated UTC: `2026-03-06T22:37:11.846918+00:00`
- Run dir: `/root/analysis/0.87/reports/execution_layer/SOL_3M_SKIPMASK_GA_20260306_223100`
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
| M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | or | 0.08 | 0.8 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## C) Gate Checklist (G0–G4)

| candidate_id | confirm_pass | gate_g0_chronology | gate_g1_participation | gate_g2_winner_preservation | gate_g3_risk_sanity | gate_improve_target | gate_g4_robustness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 1 | 1 | 1 | 1 | 0 | 0 |
| GA_d77f397fe091323f26dc3e08 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| GA_2828ac280cf9d4ac0da29784 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| GA_fe6bca06cb640a1193610f11 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| GA_6c3566f4f50c60d6ecf5c6ad | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| GA_ba169c945c01cbc77f49ea75 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| GA_78ffaa47ac88878a9592e7c0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| GA_0aadc32f5d744efc0efb6975 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| GA_076a9760cc637f020b31eed0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| GA_c264d5d9549b63399d48c77b | 0 | 1 | 0 | 0 | 0 | 1 | 0 |

## D) Split / Route / Holdout Evidence

| candidate_id | split_valid_for_ranking | best_min_subperiod_delta | route_confirm_count | route_total | route_pass_rate | holdout_trade_count | holdout_delta_expectancy_vs_baseline | holdout_delta_cvar_vs_baseline | holdout_pass | holdout_loss_target_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 9.257456019e-05 | 0 | 3 | 0 | 955 | 0 | 0 | 0 | 0 |
| GA_d77f397fe091323f26dc3e08 | 0 | -0.001968855332 | 0 | 3 | 0 | 955 | -0.00119246622 | 0.0002113269771 | 0 | 1 |
| GA_2828ac280cf9d4ac0da29784 | 0 | -0.00200059615 | 0 | 3 | 0 | 955 | -0.001103468944 | 0.0002113269771 | 0 | 1 |
| GA_fe6bca06cb640a1193610f11 | 0 | -0.00200059615 | 0 | 3 | 0 | 955 | -0.001103468944 | 0.0002113269771 | 0 | 1 |
| GA_6c3566f4f50c60d6ecf5c6ad | 0 | -0.00200059615 | 0 | 3 | 0 | 955 | -0.001103468944 | 0.0002113269771 | 0 | 1 |
| GA_ba169c945c01cbc77f49ea75 | 0 | -0.001930725637 | 0 | 3 | 0 | 955 | -0.001098039555 | 0.0002113269771 | 0 | 1 |
| GA_78ffaa47ac88878a9592e7c0 | 0 | -0.001930725637 | 0 | 3 | 0 | 955 | -0.001098039555 | 0.0002113269771 | 0 | 1 |
| GA_0aadc32f5d744efc0efb6975 | 0 | -0.001924258023 | 0 | 3 | 0 | 955 | -0.001092228861 | 0.0002113269771 | 0 | 1 |
| GA_076a9760cc637f020b31eed0 | 0 | -0.001658366384 | 0 | 3 | 0 | 955 | -0.0008925717497 | 0.00020403984 | 0 | 1 |
| GA_c264d5d9549b63399d48c77b | 0 | -0.001658366384 | 0 | 3 | 0 | 955 | -0.0008925717497 | 0.00020403984 | 0 | 1 |

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
| 5 | GA_d77f397fe091323f26dc3e08 | 0.115979647 | 0.2344560929 | 0 | 52.19906916 | 0.03850384183 | 0.004925085376 |
| 1 | GA_2828ac280cf9d4ac0da29784 | 0.115979647 | 0.2526006018 | 0 | 48.13227353 | 0.03514270496 | 0.001702834308 |
| 2 | GA_2828ac280cf9d4ac0da29784 | 0.115979647 | 0.2526006018 | 0 | 48.13227353 | 0.03514270496 | 0.001702834308 |
| 3 | GA_2828ac280cf9d4ac0da29784 | 0.115979647 | 0.2526006018 | 0 | 48.13227353 | 0.03514270496 | 0.001702834308 |
| 4 | GA_2828ac280cf9d4ac0da29784 | 0.115979647 | 0.2526006018 | 0 | 48.13227353 | 0.03514270496 | 0.001702834308 |
| 4 | GA_fe6bca06cb640a1193610f11 | 0.1094147577 | 0.2526006018 | 0 | 48.13227353 | 0.03514270496 | 0.001702834308 |
| 5 | GA_2828ac280cf9d4ac0da29784 | 0.115979647 | 0.2526006018 | 0 | 48.13227353 | 0.03514270496 | 0.001702834308 |
| 5 | GA_fe6bca06cb640a1193610f11 | 0.1094147577 | 0.2526006018 | 0 | 48.13227353 | 0.03514270496 | 0.001702834308 |
| 5 | GA_6c3566f4f50c60d6ecf5c6ad | 0.105489057 | 0.2526006018 | 0 | 48.13227353 | 0.03514270496 | 0.001702834308 |
| 5 | GA_2828ac280cf9d4ac0da29784 | 0.115979647 | 0.2526006018 | 0 | 48.13227353 | 0.03514270496 | 0.001702834308 |