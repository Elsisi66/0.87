# Aggregated Execution Report

- Generated UTC: 2026-02-20T07:49:42.564353+00:00
- Base dir: `/root/analysis/0.87/reports/execution_layer`
- Files discovered: 24
- Files used (deduped): 24

## Overall By Mode

| mode | files | signals_total | mode_entries | entry_rate_mode | avoided_losses | missed_wins | baseline_pnl_sum | mode_pnl_sum | pnl_delta | baseline_pnl_net_sum | mode_pnl_net_sum | pnl_net_delta | baseline_expectancy | mode_expectancy | baseline_expectancy_net | mode_expectancy_net | mode_taker_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 3 | 3 | 1.000000 | 0 | 0 | -0.003000 | -0.003000 | 0.000000 | -0.003000 | -0.003000 | 0.000000 | -0.001000 | -0.001000 | -0.001000 | -0.001000 | 0.000000 |
| exec_limit | 4 | 605 | 536 | 0.885950 | 69 | 0 | -0.172280 | 0.059038 | 0.231317 | -0.652203 | -0.265374 | 0.386829 | -0.000285 | 0.000110 | -0.001078 | -0.000495 | 0.070896 |
| ict_gate | 19 | 2505 | 1085 | 0.433134 | 1313 | 16 | 1.268759 | -0.718794 | -1.987552 | 1.268759 | -0.718794 | -1.987552 | 0.000517 | -0.000662 | 0.000517 | -0.000662 | 0.000000 |

## Per Symbol / Mode

| symbol | mode | signals_total | mode_entries | entry_rate_mode | avoided_losses | missed_wins | baseline_pnl_sum | mode_pnl_sum | pnl_delta | baseline_pnl_net_sum | mode_pnl_net_sum | pnl_net_delta | mode_taker_share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AVAXUSDT | ict_gate | 800 | 342 | 0.427500 | 433 | 16 | 0.850588 | -0.342000 | -1.192588 | 0.850588 | -0.342000 | -1.192588 | 0.000000 |
| NEARUSDT | ict_gate | 800 | 373 | 0.466250 | 407 | 0 | 0.680643 | -0.046287 | -0.726930 | 0.680643 | -0.046287 | -0.726930 | 0.000000 |
| SOLUSDT | baseline | 3 | 3 | 1.000000 | 0 | 0 | -0.003000 | -0.003000 | 0.000000 | -0.003000 | -0.003000 | 0.000000 | 0.000000 |
| SOLUSDT | exec_limit | 605 | 536 | 0.885950 | 69 | 0 | -0.172280 | 0.059038 | 0.231317 | -0.652203 | -0.265374 | 0.386829 | 0.070896 |
| SOLUSDT | ict_gate | 905 | 370 | 0.408840 | 473 | 0 | -0.262473 | -0.330507 | -0.068034 | -0.262473 | -0.330507 | -0.068034 | 0.000000 |

## Top Skip Reasons By Mode

- baseline:
  - none
- exec_limit:
  - volatility_gate: 43
  - after_baseline_exit: 26
- ict_gate:
  - score_below_threshold: 828
  - no_displacement: 299
  - outside_killzone: 194
  - no_sweep: 49
  - error: 30
  - no_bar_after_signal: 20

## Included Files

- `/root/analysis/0.87/reports/execution_layer/20260220_053724/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_053838/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_054000/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_054026/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_054110/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_060142/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_060207/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_060423/AVAXUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_060702/NEARUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_060917/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_060926/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_060937/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_060947/AVAXUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_060957/AVAXUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_061007/AVAXUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_061017/NEARUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_061027/NEARUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_061037/NEARUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_063200/SOLUSDT_exec_limit_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_063240/SOLUSDT_exec_limit_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_063303/SOLUSDT_exec_ict_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_063354/SOLUSDT_exec_baseline_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_065112/SOLUSDT_exec_limit_vs_baseline.csv`
- `/root/analysis/0.87/reports/execution_layer/20260220_074922/SOLUSDT_exec_limit_vs_baseline.csv`

## Walkforward Test-Only Per Symbol

| symbol | runs | test_signals | entries | entry_rate | taker_share | baseline_pnl_net_sum | pnl_net_sum | expectancy_net | sl_hit_rate | tp_hit_rate | entry_improvement_pct | max_fill_delay_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AVAXUSDT | 1 | 487 | 187 | 0.383984 | 0.128342 | -0.189535 | -0.151186 | -0.000808 | 0.631016 | 0.000000 | 0.000577 | 30.000000 |
| NEARUSDT | 1 | 414 | 209 | 0.504831 | 0.143541 | -0.406210 | -0.263242 | -0.001260 | 0.727273 | 0.000000 | 0.000789 | 30.000000 |
| SOLUSDT | 1 | 600 | 547 | 0.911667 | 0.117002 | -0.567306 | -0.447375 | -0.000818 | 0.645338 | 0.000000 | 0.000507 | 30.000000 |

## Walkforward Test-Only Overall

| row | symbol | runs | test_signals | entries | entry_rate | taker_share | baseline_pnl_net_sum | pnl_net_sum | expectancy_net | sl_hit_rate | tp_hit_rate | entry_improvement_pct | max_fill_delay_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall_test | ALL | 3 | 1501.000000 | 943.000000 | 0.628248 | 0.125133 | -1.163051 | -0.861804 | -0.000914 | 0.660657 | 0.000000 | 0.000584 | 30.000000 |

- Walkforward test summary CSV: `/root/analysis/0.87/reports/execution_layer/AGG_exec_testonly_summary.csv`
