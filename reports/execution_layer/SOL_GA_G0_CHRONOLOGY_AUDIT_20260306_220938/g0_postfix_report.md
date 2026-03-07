# G0 Postfix Report

- Generated UTC: `2026-03-06T22:11:55.211227+00:00`
- Source GA run: `/root/analysis/0.87/reports/execution_layer/SOL_3M_LOSSCONC_GA_20260306_213204`
- Final decision: `G0_CLEAN`

## Post-fix smoke check

| scenario | candidate_id | same_bar_exit_count | same_bar_touch_count | exit_before_entry_count | entry_on_signal_count | parity_clean | old_same_bar_exit_count_from_ga_confirm | confirm_pass_before_fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 1 | 0 | 0 | 1 | 1 | 0 |
| best_candidate | GA_ed12f01240bb9afc945178d5 | 0 | 1 | 0 | 0 | 1 | 1 | 0 |

## Gate impact

- Candidate route_pass_rate remains `0.6667` (unchanged).
- Candidate gate_improve_target remained `0` in original confirm row (unchanged by counter fix).
- The original G0 fail came from label misuse (`exec_same_bar_hit`) and is removed under true chronology counting.