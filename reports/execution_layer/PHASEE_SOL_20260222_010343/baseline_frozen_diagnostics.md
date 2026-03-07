# Baseline Frozen Diagnostics

- Generated UTC: 2026-02-22T01:03:44.059676+00:00
- signals_total_test: 600
- trades_total_test: 600
- expectancy_net: -0.000559
- pnl_net_sum: -0.335322
- cvar5: -0.001950
- max_drawdown: -0.518469

## Split Metrics

| variant | split_id | signals_total | trades_total | entry_rate | expectancy_net | expectancy_net_per_signal | pnl_net_sum | cvar_5 | max_drawdown | win_rate | sl_hit_rate | tp_hit_rate | timeout_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phasec_fixed_exit_control | 0 | 120 | 120 | 1.000000 | 0.001016 | 0.001016 | 0.121930 | -0.001950 | -0.127702 | 0.066667 | 0.933333 | 0.016667 | 0.050000 |
| phasec_fixed_exit_control | 1 | 120 | 120 | 1.000000 | -0.000829 | -0.000829 | -0.099519 | -0.001950 | -0.102980 | 0.058333 | 0.941667 | 0.000000 | 0.058333 |
| phasec_fixed_exit_control | 2 | 120 | 120 | 1.000000 | -0.001159 | -0.001159 | -0.139108 | -0.001950 | -0.138660 | 0.033333 | 0.966667 | 0.000000 | 0.033333 |
| phasec_fixed_exit_control | 3 | 120 | 120 | 1.000000 | -0.001061 | -0.001061 | -0.127353 | -0.001950 | -0.144272 | 0.016667 | 0.983333 | 0.000000 | 0.016667 |
| phasec_fixed_exit_control | 4 | 120 | 120 | 1.000000 | -0.000761 | -0.000761 | -0.091271 | -0.001950 | -0.141076 | 0.050000 | 0.950000 | 0.000000 | 0.050000 |
| phasec_fixed_exit_control | overall | 600 | 600 | 1.000000 | -0.000559 | -0.000559 | -0.335322 | -0.001950 | -0.518469 | 0.045000 | 0.955000 | 0.003333 | 0.041667 |

## Losing Streak Diagnostics

- streak_count: 24
- max_losing_streak: 84
- median_losing_streak: 22.50

## Trade Clustering

- signals/day mean: 6.90, p95: 16.70, max: 21
- signals/week mean: 21.43, p95: 54.45, max: 63

## Hold-Time Distribution

- winners median hold min: 717.00
- losers median hold min: 0.00

## Regime Breakdown (top rows)

| combined_regime | trades | expectancy_net | pnl_net_sum | cvar_5 | win_rate | sl_hit_rate | tp_hit_rate | timeout_rate | median_hold_minutes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unknown|down | 201 | -0.000372 | -0.074790 | -0.001950 | 0.049751 | 0.950249 | 0.004975 | 0.044776 | 0.000000 |
| mid|up | 176 | -0.000924 | -0.162686 | -0.001950 | 0.045455 | 0.954545 | 0.000000 | 0.045455 | 0.000000 |
| low|up | 129 | 0.000285 | 0.036739 | -0.001950 | 0.054264 | 0.945736 | 0.007752 | 0.046512 | 3.000000 |
| high|up | 82 | -0.001356 | -0.111190 | -0.001950 | 0.024390 | 0.975610 | 0.000000 | 0.024390 | 0.000000 |
| mid|down | 5 | -0.001950 | -0.009748 | -0.001950 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 3.000000 |
| high|down | 4 | -0.001950 | -0.007798 | -0.001950 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 9.000000 |
| low|down | 3 | -0.001950 | -0.005849 | -0.001950 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | 3.000000 |
