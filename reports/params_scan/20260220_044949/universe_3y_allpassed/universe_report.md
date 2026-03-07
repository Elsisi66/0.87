# Long Universe Simulation

- Generated UTC: 2026-02-20T05:05:36.457272+00:00
- Scan dir: `/root/analysis/0.87/reports/params_scan/20260220_044949`
- Input best CSV: `/root/analysis/0.87/reports/params_scan/20260220_044949/best_by_symbol.csv`
- Capital (EUR): 250.00
- Allocation per coin (EUR): 19.230769
- Coins used: 13
- Window start: 2023-01-01 05:00:00+00:00
- Window end: 2025-12-31 23:00:00+00:00
- Years actual: 3.000000

## Universe Metrics

- Initial equity: 250.000000
- Final equity: 795.683520
- Net profit: 545.683520
- Return %: 218.273408
- CAGR %: 47.095748
- Max drawdown: 0.038644
- Max drawdown %: 3.864393

## Per-Coin Final Contributions

| symbol | params_file | initial_alloc_eur | final_equity | net_profit | return_pct | trades | pf | max_dd_pct |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SOLUSDT | /root/analysis/0.87/data/metadata/params/SOLUSDT_C13_active_params_long.json | 19.230769 | 184.092867 | 164.862098 | 857.282909 | 1468.0 | 1.756409 | 14.485588 |
| AVAXUSDT | /root/analysis/0.87/data/metadata/params/AVAXUSDT__UNIVERSE_LONG_active_params.json | 19.230769 | 118.916391 | 99.685621 | 518.365232 | 686.0 | 2.684720 | 10.336212 |
| CRVUSDT | /root/analysis/0.87/data/metadata/params/CRVUSDT__UNIVERSE_LONG_C1_active_params.json | 19.230769 | 80.419977 | 61.189208 | 318.183882 | 641.0 | 2.529891 | 16.409161 |
| NEARUSDT | /root/analysis/0.87/data/metadata/params/NEARUSDT_C13_active_params_long.json | 19.230769 | 64.725385 | 45.494616 | 236.572002 | 555.0 | 2.517978 | 11.391938 |
| DOGEUSDT | /root/analysis/0.87/data/metadata/params/DOGEUSDT__UNIVERSE_LONG_C1_active_params.json | 19.230769 | 57.937436 | 38.706666 | 201.274665 | 436.0 | 2.973055 | 15.384871 |
| ADAUSDT | /root/analysis/0.87/data/metadata/params/ADAUSDT_C13_active_params_long.json | 19.230769 | 56.858287 | 37.627518 | 195.663093 | 868.0 | 1.862927 | 16.325597 |
| XRPUSDT | /root/analysis/0.87/data/metadata/params/XRPUSDT__UNIVERSE_LONG_C3_active_params.json | 19.230769 | 56.122282 | 36.891512 | 191.835864 | 516.0 | 2.471026 | 24.089827 |
| ZECUSDT | /root/analysis/0.87/data/metadata/params/ZECUSDT__UNIVERSE_LONG_C1_active_params.json | 19.230769 | 38.789747 | 19.558977 | 101.706682 | 504.0 | 2.138158 | 11.487870 |
| BCHUSDT | /root/analysis/0.87/data/metadata/params/BCHUSDT__UNIVERSE_LONG_C3_active_params.json | 19.230769 | 38.219577 | 18.988808 | 98.741799 | 743.0 | 1.542150 | 12.851785 |
| LINKUSDT | /root/analysis/0.87/data/metadata/params/LINKUSDT__UNIVERSE_LONG_C1_active_params.json | 19.230769 | 31.077588 | 11.846818 | 61.603455 | 333.0 | 2.002001 | 11.398728 |
| AXSUSDT | /root/analysis/0.87/data/metadata/params/AXSUSDT__UNIVERSE_LONG_C3_active_params.json | 19.230769 | 30.163771 | 10.933002 | 56.851611 | 245.0 | 2.247150 | 10.608809 |
| LTCUSDT | /root/analysis/0.87/data/metadata/params/LTCUSDT__UNIVERSE_LONG_C3_active_params.json | 19.230769 | 20.301993 | 1.071224 | 5.570366 | 284.0 | 1.119444 | 10.913484 |
| TRXUSDT | /root/analysis/0.87/data/metadata/params/TRXUSDT__UNIVERSE_LONG_C1_active_params.json | 19.230769 | 18.058219 | -1.172550 | -6.097260 | 313.0 | 0.880400 | 17.970669 |

## Notes

- Strategy logic is unchanged; simulation reuses `run_backtest_long_only` from `src/bot087/optim/ga.py`.
- Equity curves were exported using the optional `return_equity_curve=True` hook in that same backtest function.
