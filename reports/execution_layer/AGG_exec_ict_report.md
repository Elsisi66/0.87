# Aggregated ICT Execution Report

- Generated UTC: 2026-02-20T06:17:46.473746+00:00
- Base dir: `/root/analysis/0.87/reports/execution_layer`
- Files included: 18
- Output CSV: `/root/analysis/0.87/reports/execution_layer/AGG_exec_ict_report.csv`

## Overall Totals

- Total signals: 2500
- Baseline entries: 2450
- ICT entries: 1085
- ICT entry rate: 0.434000
- Avoided losses (ICT skipped + baseline SL): 1308
- Missed wins (ICT skipped + baseline TP): 16
- Baseline pnl sum (%): 1.273759
- ICT pnl sum (%): -0.718794
- Baseline expectancy/trade (%): 0.000520
- ICT expectancy/trade (%): -0.000662

## Top 10 Symbols By ICT Entry Rate

| symbol | signals_total | ict_entries | entry_rate_ict |
|---|---|---|---|
| NEARUSDT | 800 | 373 | 0.466250 |
| AVAXUSDT | 800 | 342 | 0.427500 |
| SOLUSDT | 900 | 370 | 0.411111 |

## Top 10 Symbols By Avoided Losses

| symbol | avoided_losses | signals_total | entry_rate_ict |
|---|---|---|---|
| SOLUSDT | 468 | 900 | 0.411111 |
| AVAXUSDT | 433 | 800 | 0.427500 |
| NEARUSDT | 407 | 800 | 0.466250 |

## Top 10 Symbols By PnL Delta (ICT - Baseline)

| symbol | baseline_pnl_sum | ict_pnl_sum | pnl_delta |
|---|---|---|---|
| SOLUSDT | -0.257473 | -0.330507 | -0.073034 |
| NEARUSDT | 0.680643 | -0.046287 | -0.726930 |
| AVAXUSDT | 0.850588 | -0.342000 | -1.192588 |

## Global ICT Skip Reasons

- score_below_threshold: 828
- no_displacement: 296
- outside_killzone: 192
- no_sweep: 49
- error: 30
- no_bar_after_signal: 20

## Is ICT Better?

- Using realized pnl metrics: ICT total pnl delta = -1.992552 (%-sum proxy).
- Baseline expectancy/trade = 0.000520 vs ICT expectancy/trade = -0.000662.
- Interpretation: current ICT behaves as a risk filter when entry rate is low; need threshold/ablation tuning before claiming edge improvement.

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
