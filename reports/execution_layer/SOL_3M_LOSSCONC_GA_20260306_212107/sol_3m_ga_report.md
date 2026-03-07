# SOL 3m Loss-Concentration GA

- Generated UTC: `2026-03-06T21:26:03.752520+00:00`
- Run dir: `/root/analysis/0.87/reports/execution_layer/SOL_3M_LOSSCONC_GA_20260306_212107`
- Decision: `NO_REPAIR_APPROVED`
- Symbol: `SOLUSDT`
- Baseline strategy id lock: `M1_ENTRY_ONLY_PASSIVE_BASELINE`

## A) Baseline Metrics

| scope | trade_count | expectancy_net | cvar_5 | max_drawdown | instant_loser_rate | fast_loser_rate | bottom_decile_pnl_share | same_bar_exit_count | exit_before_entry_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train60 | 859 | 0.003264234029 | -0.002110040831 | -0.3486083175 | 0.7252619325 | 0.1338766007 | 0.1166605633 | 0 | 0 |
| full | 1432 | 0.001906509291 | -0.00211207445 | -0.1301395943 | 0.7136871508 | 0.1201117318 | 0.1195507485 | 1 | 0 |

## B) Best Candidate Metrics + Deltas

| candidate_id | entry_mode | limit_offset_bps | fallback_to_market | fallback_delay_min | max_fill_delay_min | delta_expectancy_vs_baseline | delta_cvar_vs_baseline | maxdd_improve_ratio | retention | winner_retention | instant_loser_rel_reduction | bottom_decile_rel_reduction | route_pass_rate | holdout_delta_expectancy_vs_baseline | holdout_delta_cvar_vs_baseline | confirm_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_ENTRY_ONLY_PASSIVE_BASELINE | limit | 0.75 | 1 | 6 | 24 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

## C) Gate Checklist (G0–G4)

| candidate_id | confirm_pass | gate_g0_chronology | gate_g1_participation | gate_g2_winner_preservation | gate_g3_risk_sanity | gate_improve_target | gate_g4_robustness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0 | 1 | 1 | 1 | 0 | 0 |

## D) Split / Route / Holdout Evidence

| candidate_id | split_valid_for_ranking | best_min_subperiod_delta | route_confirm_count | route_total | route_pass_rate | holdout_trade_count | holdout_delta_expectancy_vs_baseline | holdout_delta_cvar_vs_baseline | holdout_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 9.257456019e-05 | 0 | 3 | 0 | 955 | 0 | 0 | 0 |

## E) Proven vs Assumed

- Proven: SOL-only input came from repaired posture freeze active subset and locked winner config id.
- Proven: 1h signal layer/params remained frozen from repaired universe artifacts.
- Proven: evaluation path reused `phase_a` entry-only wrapper with 1h-owned exits and chronology protections.
- Proven: route family check required 3/3 confirmations for pass.
- Assumed: no extra execution skip-mask features were needed in this pass; only existing 3m knob family was searched.

## F) Recommended Next Step

- Stop this GA branch; either redesign the allowed entry lever family or keep current SOL baseline as active execution posture.

## Top Screened Candidates

| generation | candidate_id | entry_mode | limit_offset_bps | fallback_to_market | fallback_delay_min | max_fill_delay_min | screen_pass | screen_objective | instant_loser_rel_reduction | bottom_decile_rel_reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 |  | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 3 |  | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 3 |  | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 4 |  | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 4 |  | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 4 |  | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 5 |  | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 5 |  | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 5 |  | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 5 |  | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |