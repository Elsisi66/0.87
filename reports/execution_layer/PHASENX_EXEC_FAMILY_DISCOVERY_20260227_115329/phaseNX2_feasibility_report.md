# NX2 Feasibility Report

- Generated UTC: 2026-02-27T11:54:17.958048+00:00
- Direct configs evaluated: `12`

## Family Verdicts

| family_id | nx2_verdict | nx2_reason | max_entry_rate | max_entries_valid | upper_bound_valid_for_ranking |
| --- | --- | --- | --- | --- | --- |
| PASSIVE_LADDER_ADAPTIVE | NX2_GO | participation appears reachable under mechanically valid settings | 1 | 360 | 1 |
| REGIME_ROUTED_EXEC | NX2_GO | participation appears reachable under mechanically valid settings | 1 | 360 | 1 |
| STAGED_ENTRY_RISKSHAPE | NX2_GO | participation appears reachable under mechanically valid settings | 1 | 360 | 0 |

## Upper-Bound Snapshot

| family_id | variant_id | valid_for_ranking | entry_rate | entries_valid_total | taker_share | median_fill_delay_min | overall_delta_expectancy_exec_minus_baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PASSIVE_LADDER_ADAPTIVE | A_upper_bound | 1 | 1 | 360 | 0.01388888889 | 0 | 0.0003430804481 |
| REGIME_ROUTED_EXEC | B_upper_bound | 1 | 1 | 360 | 0.05 | 0 | 0.000330904744 |
| STAGED_ENTRY_RISKSHAPE | C_upper_bound | 0 | 1 | 360 | 1 | 1.324922302 | -0.0006480488428 |
