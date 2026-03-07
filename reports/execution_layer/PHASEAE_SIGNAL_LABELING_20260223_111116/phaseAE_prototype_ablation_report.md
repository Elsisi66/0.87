# Phase AE Prototype Ablation Report

- Generated UTC: 2026-02-23T11:12:41.827551+00:00
- Baseline E1 hash: `862c940746de0da984862d95`
- Decision: **GO**

## Baseline Metrics

| signals_total | entries_valid | entry_rate | exec_expectancy_net | exec_cvar_5 | exec_max_drawdown | max_consecutive_losses | streak_ge5_count | streak_ge10_count | taker_share | p95_fill_delay_min | min_split_expectancy_net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 360 | 355 | 0.98611111 | 5.6491907e-05 | -0.0017797459 | -0.18691674 | 33 | 20 | 12 | 0.081690141 | 15.9 | -0.0011756882 |

## Prototype Results

| variant_id | eval_type | valid_for_ranking | exec_expectancy_net | delta_exec_expectancy_vs_baseline | cvar_improve_ratio_vs_baseline | maxdd_improve_ratio_vs_baseline | max_consecutive_losses | max_consecutive_losses_reduction_ratio | streak_ge5_count | streak_ge5_reduction_ratio | streak_ge10_count | entry_rate | removed_signals_pct | taker_share | p95_fill_delay_min | min_split_expectancy_net | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1_vol_trend_guard | exact_engine_integrated | 0 | -0.00035564361 | -0.00041213552 | 0.021840985 | 0.71260119 | 24 | 0.27272727 | 6 | 0.7 | 4 | 0.99009901 | 0.71916667 | 0.07 | 12.3 | -0.0013207226 | overall:trades<200 |
| H2_spread_vol_guard | exact_engine_integrated | 0 | -0.00031095061 | -0.00036744252 | 0.02839328 | 0.75436664 | 26 | 0.21212121 | 6 | 0.7 | 4 | 0.98979592 | 0.72666667 | 0.06185567 | 13.2 | -0.0013380351 | overall:trades<200 |
| H3_impulse_vol_guard | exact_engine_integrated | 0 | -0.00033509027 | -0.00039158217 | 0.02839328 | 0.72623383 | 25 | 0.24242424 | 6 | 0.7 | 4 | 0.99 | 0.72166667 | 0.060606061 | 12.6 | -0.0013090062 | overall:trades<200 |
| baseline_E1 | exact_engine_integrated | 1 | 5.6491907e-05 | 0 | 0 | 0 | 33 | 0 | 20 | 0 | 12 | 0.98611111 | nan | 0.081690141 | 15.9 | -0.0011756882 |  |
| S1_risk_score_half_size | proxy_size_scaling | 1 | 0.00010651875 | 5.0026846e-05 | 0.010920492 | 0.11930703 | 33 | 0 | 20 | 0 | 12 | 0.98611111 | 0 | 0.081690141 | 15.9 | -0.0010716038 |  |
| S2_risk_score_quarter_size | proxy_size_scaling | 1 | 8.6686662e-05 | 3.0194755e-05 | 0 | 0.072774873 | 33 | 0 | 20 | 0 | 12 | 0.98611111 | 0 | 0.081690141 | 15.9 | -0.0011075381 |  |

## GO Criteria

- Acceptable expectancy: delta_expectancy >= -0.00002
- Risk improvement: any of {max_consecutive_losses_reduction>=20%, streak>=5_reduction>=20%, cvar_improve>=10%, maxdd_improve>=10%}
- Stability: min_split_expectancy not worse than baseline by more than 0.0002
- Participation: entry_rate >= 70% of baseline
- GO pass count: `1`
- Best GO prototype: `S1_risk_score_half_size`
