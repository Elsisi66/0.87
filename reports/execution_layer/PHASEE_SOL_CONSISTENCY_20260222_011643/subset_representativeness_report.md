# Subset Representativeness Report

- frozen_signals_total: 600
- alt_subsets: 10
- seed_base: 42
- sampling_pool_size: 1400
- regime_tvd(frozen vs broader): 0.694500

## Frozen vs Alternatives Percentile Position

| variant | metric | frozen_value | alt_mean | alt_median | frozen_percentile |
| --- | --- | --- | --- | --- | --- |
| 1h_reference | expectancy_net | -0.000643 | -0.000980 | -0.000924 | 1.000000 |
| 1h_reference | total_return | -0.996450 | -0.999046 | -0.998936 | 1.000000 |
| 1h_reference | max_drawdown_pct | -0.998986 | -0.999198 | -0.999326 | 0.600000 |
| 1h_reference | cvar_5 | -0.002200 | -0.002250 | -0.002200 | 1.000000 |
| phasec_best_exit | expectancy_net | -0.000559 | -0.000932 | -0.000878 | 1.000000 |
| phasec_best_exit | total_return | -0.999173 | -0.999868 | -0.999870 | 1.000000 |
| phasec_best_exit | max_drawdown_pct | -0.999828 | -0.999892 | -0.999905 | 0.800000 |
| phasec_best_exit | cvar_5 | -0.001950 | -0.002002 | -0.001950 | 1.000000 |

## Bias Flags

- frozen_tail_10pct_any_expectancy: 0
- frozen_tail_20pct_both_expectancy: 0
- regime_distribution_tvd_gt_0_20: 1
