# Risk Rollup: Baseline vs Exec (TEST-only)

- Generated UTC: 2026-02-20T12:31:19.640987+00:00
- Max relative expectancy worsening tolerance: 0.1000

## Inputs

- SOLUSDT: `/root/analysis/0.87/reports/execution_layer/20260220_122741_walkforward_SOLUSDT/SOLUSDT_walkforward_test_signals.csv`
- AVAXUSDT: `/root/analysis/0.87/reports/execution_layer/20260220_122941_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_signals.csv`
- NEARUSDT: `/root/analysis/0.87/reports/execution_layer/20260220_123057_walkforward_NEARUSDT/NEARUSDT_walkforward_test_signals.csv`

## By Symbol (Key Deltas)

- AVAXUSDT: d_expectancy=-0.000259, d_cvar5=0.000000, d_maxDD=0.011595, exec_taker_share=0.155452, exec_median_delay=0.00
- NEARUSDT: d_expectancy=-0.000100, d_cvar5=0.000000, d_maxDD=0.050982, exec_taker_share=0.122047, exec_median_delay=0.00
- SOLUSDT: d_expectancy=-0.000203, d_cvar5=0.000000, d_maxDD=0.081058, exec_taker_share=0.116358, exec_median_delay=0.00

## Overall

- baseline_mean_expectancy_net: -0.000565
- exec_mean_expectancy_net: -0.000766
- delta_expectancy_exec_minus_baseline: -0.000201
- baseline_cvar_5: -0.002200
- exec_cvar_5: -0.002200
- baseline_max_drawdown: -0.738627
- exec_max_drawdown: -1.003148
- exec_taker_share: 0.130673
- exec_median_fill_delay_min: 0.00

## Conclusion

- Deploy-worth safety rule: CVaR improves AND max DD improves AND expectancy does not worsen beyond 10.00%.
- CVaR improves: 1
- Max DD improves: 0
- Expectancy not too worse: 0
- Safety-layer deploy-worthy: NO

- CSV by symbol: `/root/analysis/0.87/reports/execution_layer/20260220_123119/risk_rollup_by_symbol.csv`
- CSV overall: `/root/analysis/0.87/reports/execution_layer/20260220_123119/risk_rollup_overall.csv`
- Input map: `/root/analysis/0.87/reports/execution_layer/20260220_123119/risk_rollup_inputs.csv`
