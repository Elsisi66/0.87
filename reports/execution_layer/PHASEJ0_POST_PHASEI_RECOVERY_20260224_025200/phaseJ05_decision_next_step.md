# Phase J05 Decision

- Generated UTC: 2026-02-24T02:53:18.315602+00:00
- Classification: **J0_NO_GO_ABLATION_ONLY**
- Mainline status: **STOP_NO_GO**
- Decision rationale: no soft sizing variant met acceptable risk/expectancy trade-off under strict gates.

## Explicit Answers

1. Did soft sizing improve drawdown/clustering without breaking the edge? **no**
2. Is participation preserved vs baseline? **no**
3. Is a bounded sizing GA pilot justified? **no**
4. If no, next highest-EV branch: `execution-only monitoring + feature/regime relabeling refresh`

## Top trade-off variants

| variant_id | variant_family | valid_for_ranking | frontier_score | exec_expectancy_net | delta_expectancy_vs_baseline | cvar_improve_ratio | maxdd_improve_ratio | min_split_expectancy_net | entries_valid | entry_rate | max_consecutive_losses | streak_ge3_count | streak_ge5_count | streak_ge10_count | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tail_cap10 | streak | 0 | -0.0030660708 | 0.00013247359 | -6.1972979e-06 | -0.0044747998 | 0.0027989707 | -0.0013680194 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0;min_maxdd_improve<=0 |
| step_mild | step | 0 | -0.0073263634 | 0.00014971917 | 1.1048287e-05 | -0.0078872134 | 0.003767908 | -0.0013765533 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| step_mid | step | 0 | -0.017111203 | 0.00016126825 | 2.2597364e-05 | -0.018172662 | 0.0085660055 | -0.0013583058 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| step_aggr | step | 0 | -0.03236226 | 0.0001890967 | 5.0425812e-05 | -0.034832985 | 0.016629837 | -0.0013807644 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| tail_cap8 | streak | 0 | -0.042036623 | 0.00010207824 | -3.6592649e-05 | -0.014129931 | -0.0090611149 | -0.00138536 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0;min_maxdd_improve<=0 |
| cooldown_lite_proxy1 | streak_control | 0 | -0.048536952 | 0.00010653532 | -3.2135568e-05 | -0.04668287 | 0.019860721 | -0.0013124266 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| streak_cap4 | streak | 0 | -0.05319391 | 8.7035372e-05 | -5.1635512e-05 | -0.065064474 | 0.035741793 | -0.0013073523 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| smooth_mild | smooth | 0 | -0.062777712 | 0.00016846027 | 2.9789383e-05 | -0.052162409 | 0.017153506 | -0.0013633951 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| streak_cap3 | streak | 0 | -0.063100517 | 7.1554133e-05 | -6.7116752e-05 | -0.064711526 | 0.02995398 | -0.0012964173 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| cooldown_lite_proxy2 | streak_control | 0 | -0.063100517 | 7.1554133e-05 | -6.7116752e-05 | -0.064711526 | 0.02995398 | -0.0012964173 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| smooth_mid | smooth | 0 | -0.092620097 | 0.00018250222 | 4.3831334e-05 | -0.07924493 | 0.027594515 | -0.001359422 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| smooth_aggr | smooth | 0 | -0.14435706 | 0.00020325834 | 6.4587452e-05 | -0.12112823 | 0.040642808 | -0.0013554054 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
| hybrid_v1 | hybrid | 0 | -0.16895887 | 0.00012164692 | -1.7023967e-05 | -0.12878104 | 0.034990664 | -0.0013060718 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_delta_expectancy<=0;min_cvar_improve<0 |
| S1_anchor | ae_s1_anchor | 0 | -0.17548618 | 0.00025555396 | 0.00011688307 | -0.11142098 | 0.013409178 | -0.0013474812 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0;min_maxdd_improve<=0 |
| hybrid_v3 | hybrid | 0 | -0.21524149 | 0.00014751251 | 8.8416292e-06 | -0.15525621 | 0.035638314 | -0.0012983742 | 70 | 0.97142857 | 33 | 29 | 22 | 14 | min_cvar_improve<0 |
