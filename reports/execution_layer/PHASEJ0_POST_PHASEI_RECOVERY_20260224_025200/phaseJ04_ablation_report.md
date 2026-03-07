# Phase J04 Controlled Ablation Report

- Generated UTC: 2026-02-24T02:53:18.299862+00:00
- Variants tested: `18`
- Route-choice datasets: `4`
- valid_for_ranking count: `0`

## Top variants

| variant_id | variant_family | valid_for_ranking | min_delta_expectancy_vs_baseline | min_cvar_improve_ratio | min_maxdd_improve_ratio | entries_valid | entry_rate | max_consecutive_losses | streak_ge3_count | streak_ge5_count | streak_ge10_count | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1_anchor | ae_s1_anchor | 0 | 8.7747746e-05 | -0.11699974 | -0.0069526238 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0;min_maxdd_improve<=0 |
| step_aggr | step | 0 | 4.8171887e-05 | -0.040821844 | 0.011056915 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| smooth_aggr | smooth | 0 | 4.0885338e-05 | -0.15183189 | 0.02294838 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| smooth_mid | smooth | 0 | 2.7486678e-05 | -0.10348494 | 0.015427896 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| step_mid | step | 0 | 2.2322237e-05 | -0.021839311 | 0.0029435257 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| smooth_mild | smooth | 0 | 2.0536125e-05 | -0.068897295 | 0.0088805017 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| hybrid_v4 | hybrid | 0 | 9.2042513e-06 | -0.35057626 | 0.031840993 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| step_mild | step | 0 | 8.8303772e-06 | -0.012264769 | 0.0011015559 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| hybrid_v3 | hybrid | 0 | 3.5591591e-06 | -0.22963723 | 0.020787764 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| baseline_flat | streak_control | 0 | 0 | 0 | 0 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_maxdd_improve<=0 |
| tail_cap10 | streak | 0 | -1.2398323e-05 | -0.0089495996 | 0 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0;min_maxdd_improve<=0 |
| hybrid_v1 | hybrid | 0 | -2.240063e-05 | -0.18397556 | 0.020964338 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| hybrid_v2 | hybrid | 0 | -2.7778451e-05 | -0.24368628 | 0.026836232 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| cooldown_lite_proxy1 | streak_control | 0 | -4.3705152e-05 | -0.093499126 | 0.0062689271 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| streak_cap3 | streak | 0 | -7.9092939e-05 | -0.11095285 | 0.011587743 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| cooldown_lite_proxy2 | streak_control | 0 | -7.9092939e-05 | -0.11095285 | 0.011587743 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| streak_cap4 | streak | 0 | -8.4762799e-05 | -0.13163972 | 0.013120372 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| tail_cap8 | streak | 0 | -8.542974e-05 | -0.020490423 | -0.022962185 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0;min_maxdd_improve<=0 |
