# SOL 3m Loss-Concentration GA

- Generated UTC: `2026-03-06T21:38:37.151942+00:00`
- Run dir: `/root/analysis/0.87/reports/execution_layer/SOL_3M_LOSSCONC_GA_20260306_213204`
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
| GA_ed12f01240bb9afc945178d5 | limit | 1.352377738 | 1 | 0 | 43.98549892 | 0.0002059370252 | 6.801328e-05 | 0.02541959391 | 1 | 1.046666667 | 0.0156555773 | 0.009736013828 | 0.6666666667 | 0.0002039115392 | 5.829709714e-05 | 0 |

## C) Gate Checklist (G0–G4)

| candidate_id | confirm_pass | gate_g0_chronology | gate_g1_participation | gate_g2_winner_preservation | gate_g3_risk_sanity | gate_improve_target | gate_g4_robustness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GA_ed12f01240bb9afc945178d5 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| GA_417313a60a536a2f91f0253a | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| GA_7554cfebd6a7c207e505e026 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| GA_4c756495ee06f1b081fb4434 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| GA_887e87ada7b7b4015de8be30 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| GA_f9188dd92e2cc379aaa8fc2f | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| GA_2dd15b32b594ca0c0d7cf752 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| GA_1752210210bba8525716dc51 | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| GA_7da7e032a0d642678abfb9ea | 0 | 0 | 1 | 1 | 1 | 0 | 0 |
| GA_d605f2b8b43389b1b959c18a | 0 | 0 | 1 | 1 | 1 | 0 | 0 |

## D) Split / Route / Holdout Evidence

| candidate_id | split_valid_for_ranking | best_min_subperiod_delta | route_confirm_count | route_total | route_pass_rate | holdout_trade_count | holdout_delta_expectancy_vs_baseline | holdout_delta_cvar_vs_baseline | holdout_pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GA_ed12f01240bb9afc945178d5 | 1 | 0.0003529721643 | 2 | 3 | 0.6666666667 | 955 | 0.0002039115392 | 5.829709714e-05 | 1 |
| GA_417313a60a536a2f91f0253a | 1 | 0.0003538710574 | 2 | 3 | 0.6666666667 | 955 | 0.0002047986203 | 5.829709714e-05 | 1 |
| GA_7554cfebd6a7c207e505e026 | 1 | 0.0003537839281 | 2 | 3 | 0.6666666667 | 955 | 0.0002047126359 | 5.829709714e-05 | 1 |
| GA_4c756495ee06f1b081fb4434 | 1 | 0.0003549359278 | 2 | 3 | 0.6666666667 | 955 | 0.0002047507042 | 3.643568571e-05 | 1 |
| GA_887e87ada7b7b4015de8be30 | 1 | 0.0003549359278 | 2 | 3 | 0.6666666667 | 955 | 0.0002047507042 | 3.643568571e-05 | 1 |
| GA_f9188dd92e2cc379aaa8fc2f | 1 | 0.0003549359278 | 2 | 3 | 0.6666666667 | 955 | 0.0002047507042 | 3.643568571e-05 | 1 |
| GA_2dd15b32b594ca0c0d7cf752 | 1 | 0.0003537839281 | 2 | 3 | 0.6666666667 | 955 | 0.0002027244808 | 2.914854857e-05 | 1 |
| GA_1752210210bba8525716dc51 | 1 | 0.0003549359278 | 2 | 3 | 0.6666666667 | 955 | 0.0002027494647 | 7.287137143e-06 | 1 |
| GA_7da7e032a0d642678abfb9ea | 1 | 0.0003537839281 | 1 | 3 | 0.3333333333 | 955 | 0.0002027244808 | 2.914854857e-05 | 1 |
| GA_d605f2b8b43389b1b959c18a | 1 | 0.0003537839281 | 1 | 3 | 0.3333333333 | 955 | 0.0002027244808 | 2.914854857e-05 | 1 |

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
| 2 | GA_4c756495ee06f1b081fb4434 | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 3 | GA_4c756495ee06f1b081fb4434 | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 3 | GA_887e87ada7b7b4015de8be30 | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 4 | GA_4c756495ee06f1b081fb4434 | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 4 | GA_887e87ada7b7b4015de8be30 | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 4 | GA_4c756495ee06f1b081fb4434 | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 5 | GA_4c756495ee06f1b081fb4434 | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 5 | GA_887e87ada7b7b4015de8be30 | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 5 | GA_4c756495ee06f1b081fb4434 | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |
| 5 | GA_887e87ada7b7b4015de8be30 | limit | 1.561275024 | 1 | 0 | 43.98549892 | 0 | 1.337092918 | 0.01284109149 | 0.01751674257 |