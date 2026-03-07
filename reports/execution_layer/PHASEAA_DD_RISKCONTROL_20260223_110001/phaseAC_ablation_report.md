# Phase AC Controlled Ablation Report

- Generated UTC: 2026-02-23T11:00:40.895824+00:00
- Baseline genome hash: `862c940746de0da984862d95` (E1)
- Grid: baseline + each single control + top-2 control combo.
- Hard gates unchanged. No time/session veto overlays.

## Results

| variant_id | variant_type | valid_for_ranking | exec_expectancy_net | delta_exec_expectancy_vs_baseline | cvar_improve_ratio_vs_baseline | maxdd_improve_ratio_vs_baseline | max_consecutive_losses | max_consecutive_losses_reduction_ratio | streak_ge5_count | streak_ge5_reduction_ratio | streak_ge10_count | sl_loss_share | entries_valid | entry_rate | taker_share | p95_fill_delay_min | min_split_expectancy_net | go_pass | invalid_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_e_winner | baseline | 1 | 5.6491907e-05 | nan | nan | nan | 33 | nan | 20 | nan | 12 | 0.88461803 | 355 | 0.98611111 | 0.081690141 | 15.9 | -0.0011756882 | 0 |  |
| COMBO_C3_trailing_tail_guard__C1_cooldown_decluster | combo_top2 | 1 | -0.00066172618 | -0.00071821809 | 0.18124309 | -0.28617367 | 25 | 0.24242424 | 23 | -0.15 | 9 | 0.89602143 | 355 | 0.98611111 | 0.081690141 | 15.9 | -0.00089036237 | 0 |  |
| C1_cooldown_decluster | single_control | 1 | 5.6491907e-05 | 0 | 0 | 0 | 33 | 0 | 20 | 0 | 12 | 0.88461803 | 355 | 0.98611111 | 0.081690141 | 15.9 | -0.0011756882 | 0 |  |
| C2_break_even_early | single_control | 1 | -0.00039574681 | -0.00045223872 | 0.18564837 | -0.26265853 | 46 | -0.39393939 | 14 | 0.3 | 10 | 0.91085871 | 355 | 0.98611111 | 0.081690141 | 15.9 | -0.0011902539 | 0 |  |
| C3_trailing_tail_guard | single_control | 1 | -0.00066172618 | -0.00071821809 | 0.18124309 | -0.28617367 | 25 | 0.24242424 | 23 | -0.15 | 9 | 0.89602143 | 355 | 0.98611111 | 0.081690141 | 15.9 | -0.00089036237 | 0 |  |
| C4_time_stop_tighten | single_control | 1 | 5.6491907e-05 | 0 | 0 | 0 | 33 | 0 | 20 | 0 | 12 | 0.88461803 | 355 | 0.98611111 | 0.081690141 | 15.9 | -0.0011756882 | 0 |  |

## Decision Logic

- GO criteria:
  max_consecutive_losses reduction >= 20%, streak>=5 reduction >= 20%, valid_for_ranking=1, delta_expectancy >= -0.00002.
- AC decision: AC_NO_GO
- Closest variant: `C3_trailing_tail_guard` with reductions mcl=0.2424, streak5=-0.1500, delta_exp=-0.00071822, valid=1.
