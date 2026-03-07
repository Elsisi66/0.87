# Subset Representativeness Report

- frozen_signals_total: 600
- alt_subsets: 10
- seed_base: 42
- sampling_pool_size: 1400
- regime_tvd(frozen vs broader): 0.694500

## Frozen vs Alternatives Percentile Position

| variant | metric | frozen_value | alt_mean | alt_median | frozen_percentile |
| --- | --- | --- | --- | --- | --- |
| 1h_reference | expectancy_net | -0.000649 | -0.001015 | -0.001001 | 1.000000 |
| 1h_reference | total_return | -0.996799 | -0.999256 | -0.999342 | 1.000000 |
| 1h_reference | max_drawdown_pct | -0.999175 | -0.999383 | -0.999591 | 0.800000 |
| 1h_reference | cvar_5 | -0.002200 | -0.002200 | -0.002200 | 1.000000 |
| phasec_best_exit | expectancy_net | nan | nan | nan | nan |
| phasec_best_exit | total_return | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| phasec_best_exit | max_drawdown_pct | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| phasec_best_exit | cvar_5 | nan | nan | nan | nan |

## Bias Flags

- frozen_tail_10pct_any_expectancy: 0
- frozen_tail_20pct_both_expectancy: 0
- regime_distribution_tvd_gt_0_20: 1
