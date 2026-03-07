# Phase F3 Ablation Report

- Generated UTC: 2026-02-23T19:38:43.249558+00:00
- Decision class: **STOP_NO_GO**
- Reason: no variant reached valid_for_ranking under strict multi-route gates
- Variants tested (incl baseline): `10`
- valid_for_ranking count: `0`

## Top variants

| variant_id | variant_family | valid_for_ranking | min_delta_expectancy_vs_baseline | min_cvar_improve_ratio | min_maxdd_improve_ratio | entries_valid | entry_rate | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| soft_step_mid | risk_step | 0 | 2.3218182e-05 | -0.017422257 | 0.013529351 | 70 | 0.97142857 | min_cvar_improve<0 |
| soft_step_mild | risk_step | 0 | 1.2704128e-05 | -0.016356458 | 0.0046003771 | 70 | 0.97142857 | min_cvar_improve<0 |
| soft_lin_aggr | risk_linear | 0 | 1.1592793e-05 | -0.084499693 | 0.020678807 | 70 | 0.97142857 | min_cvar_improve<0 |
| soft_lin_mid | risk_linear | 0 | 8.5410776e-06 | -0.0627013 | 0.015235267 | 70 | 0.97142857 | min_cvar_improve<0 |
| soft_lin_mild | risk_linear | 0 | 5.5952586e-06 | -0.041361328 | 0.0099806211 | 70 | 0.97142857 | min_cvar_improve<0 |
| baseline_flat | flat | 0 | 0 | 0 | 0 | 70 | 0.97142857 | min_delta_expectancy<=0;min_maxdd_improve<=0 |
| hybrid_stateaware_v2 | hybrid_stateaware | 0 | -4.5144783e-05 | -0.13170691 | 0.008332451 | 70 | 0.97142857 | min_delta_expectancy<=0;min_cvar_improve<0 |
| hybrid_stateaware_v1 | hybrid_stateaware | 0 | -5.7046318e-05 | -0.11284513 | 0.0078397089 | 70 | 0.97142857 | min_delta_expectancy<=0;min_cvar_improve<0 |
| state_streak_cap3 | state_streak_cap | 0 | -6.5721509e-05 | -0.11284513 | 0.015036247 | 70 | 0.97142857 | min_delta_expectancy<=0;min_cvar_improve<0 |
| state_tail_cap8 | state_tail_cap | 0 | -6.7804082e-05 | -0.070828741 | -0.01566153 | 70 | 0.97142857 | min_delta_expectancy<=0;min_cvar_improve<0;min_maxdd_improve<=0 |
