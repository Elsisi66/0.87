# Long Universe Simulation

- Generated UTC: 2026-02-20T04:50:37.710582+00:00
- Scan dir: `/root/analysis/0.87/reports/params_scan/20260220_044949`
- Input best CSV: `/root/analysis/0.87/reports/params_scan/20260220_044949/best_by_symbol.csv`
- Capital (EUR): 250.00
- Allocation per coin (EUR): 62.500000
- Coins used: 4
- Window start: 2019-01-01 05:00:00+00:00
- Window end: 2025-12-31 23:00:00+00:00
- Years actual: 7.000000

## Universe Metrics

- Initial equity: 250.000000
- Final equity: 2182.110623
- Net profit: 1932.110623
- Return %: 772.844249
- CAGR %: 36.276050
- Max drawdown: 0.090581
- Max drawdown %: 9.058105

## Per-Coin Final Contributions

| symbol | params_file | initial_alloc_eur | final_equity | net_profit | return_pct | trades | pf | max_dd_pct |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ADAUSDT | /root/analysis/0.87/data/metadata/params/ADAUSDT_C13_active_params_long.json | 62.500000 | 1268.036652 | 1205.536652 | 1928.858643 | 2025.0 | 1.856250 | 16.325597 |
| XRPUSDT | /root/analysis/0.87/data/metadata/params/XRPUSDT__UNIVERSE_LONG_C3_active_params.json | 62.500000 | 549.158935 | 486.658935 | 778.654296 | 1299.0 | 2.111756 | 28.407831 |
| LTCUSDT | /root/analysis/0.87/data/metadata/params/LTCUSDT__UNIVERSE_LONG_C3_active_params.json | 62.500000 | 207.082074 | 144.582074 | 231.331318 | 674.0 | 1.841381 | 11.295497 |
| TRXUSDT | /root/analysis/0.87/data/metadata/params/TRXUSDT__UNIVERSE_LONG_C1_active_params.json | 62.500000 | 157.832962 | 95.332962 | 152.532739 | 746.0 | 1.562583 | 21.316030 |

## Dropped Symbols

- `AVAXUSDT`: insufficient_history:first=2020-09-22T06:00:00+00:00 window_start=2019-01-01T05:00:00+00:00
- `AXSUSDT`: insufficient_history:first=2020-11-04T13:00:00+00:00 window_start=2019-01-01T05:00:00+00:00
- `BCHUSDT`: insufficient_history:first=2019-11-28T10:00:00+00:00 window_start=2019-01-01T05:00:00+00:00
- `CRVUSDT`: insufficient_history:first=2020-08-15T04:00:00+00:00 window_start=2019-01-01T05:00:00+00:00
- `DOGEUSDT`: insufficient_history:first=2019-07-05T12:00:00+00:00 window_start=2019-01-01T05:00:00+00:00
- `LINKUSDT`: insufficient_history:first=2019-01-16T10:00:00+00:00 window_start=2019-01-01T05:00:00+00:00
- `NEARUSDT`: insufficient_history:first=2020-10-14T05:00:00+00:00 window_start=2019-01-01T05:00:00+00:00
- `SOLUSDT`: insufficient_history:first=2020-08-11T06:00:00+00:00 window_start=2019-01-01T05:00:00+00:00
- `ZECUSDT`: insufficient_history:first=2019-03-21T04:00:00+00:00 window_start=2019-01-01T05:00:00+00:00

## Notes

- Strategy logic is unchanged; simulation reuses `run_backtest_long_only` from `src/bot087/optim/ga.py`.
- Equity curves were exported using the optional `return_equity_curve=True` hook in that same backtest function.
