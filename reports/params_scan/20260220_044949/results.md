# Params Scan Results

- Generated UTC: 2026-02-20T04:50:24.477531+00:00
- Total runs: 123
- Successful runs: 123
- Errors: 0
- Passing runs: 40

## Score & Pass Rules

- Score formula: `score = (cagr_pct * profit_factor) / (1 + max_dd_pct)`
- Pass thresholds:
  - `net_profit > 0.0`
  - `profit_factor >= 1.15`
  - `cagr_pct >= 15.0`
  - `max_dd_pct <= 35.0`
  - `trades >= 50.0` OR `trades_per_year >= 10.0`
- `best_by_symbol.csv` side filter: `long`

## Top 30 By Score

| rank | symbol | side | pass | score | cagr_pct | pf | max_dd_pct | net_profit | trades | params_file |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | SOLUSDT | long | True | 25.380137 | 235.2946 | 1.7971 | 15.6608 | 6783048.2481 | 2551.0 | data/metadata/params/SOLUSDT_C13_active_params_long.json |
| 2 | SOLUSDT | long | True | 25.380137 | 235.2946 | 1.7971 | 15.6608 | 6783048.2481 | 2551.0 | data/metadata/params/SOLUSDT__UNIVERSE_LONG_active_params.json |
| 3 | SOLUSDT | long | True | 25.380137 | 235.2946 | 1.7971 | 15.6608 | 6783048.2481 | 2551.0 | data/metadata/params/SOLUSDT_active_params.json |
| 4 | AVAXUSDT | long | True | 18.660894 | 107.3144 | 2.6643 | 14.3220 | 457986.0638 | 1322.0 | data/metadata/params/AVAXUSDT__UNIVERSE_LONG_active_params.json |
| 5 | AVAXUSDT | long | True | 18.660894 | 107.3144 | 2.6643 | 14.3220 | 457986.0638 | 1322.0 | data/metadata/params/AVAXUSDT_active_params.json |
| 6 | AVAXUSDT | long | True | 18.136839 | 109.1310 | 2.5214 | 14.1714 | 480027.3441 | 1319.0 | data/metadata/params/AVAXUSDT__UNIVERSE_LONG_C1_active_params.json |
| 7 | AVAXUSDT | long | True | 16.588420 | 94.4594 | 2.5238 | 13.3713 | 323875.0773 | 1209.0 | data/metadata/params/AVAXUSDT__UNIVERSE_LONG_C3_active_params.json |
| 8 | AVAXUSDT | long | True | 16.452744 | 93.9205 | 2.5175 | 13.3713 | 319022.9104 | 1224.0 | data/metadata/params/AVAXUSDT_C13_active_params_long.json |
| 9 | NEARUSDT | long | True | 15.312473 | 92.2139 | 2.6120 | 14.7296 | 291936.4222 | 1058.0 | data/metadata/params/NEARUSDT_C13_active_params_long.json |
| 10 | CRVUSDT | long | True | 14.622679 | 101.5484 | 2.6192 | 17.1896 | 423859.9566 | 1160.0 | data/metadata/params/CRVUSDT__UNIVERSE_LONG_C1_active_params.json |
| 11 | CRVUSDT | long | True | 12.116950 | 89.4340 | 2.6357 | 18.4538 | 300836.2217 | 1018.0 | data/metadata/params/CRVUSDT_C13_active_params_long.json |
| 12 | CRVUSDT | long | True | 11.292022 | 72.5927 | 2.6362 | 15.9475 | 178371.8970 | 869.0 | data/metadata/params/CRVUSDT__UNIVERSE_LONG_C3_active_params.json |
| 13 | DOGEUSDT | long | True | 10.955928 | 81.4063 | 3.2071 | 22.8296 | 467916.9249 | 1025.0 | data/metadata/params/DOGEUSDT__UNIVERSE_LONG_C1_active_params.json |
| 14 | NEARUSDT | long | True | 10.228510 | 75.2790 | 2.2911 | 15.8615 | 176651.8557 | 960.0 | data/metadata/params/NEARUSDT__UNIVERSE_LONG_C1_active_params.json |
| 15 | NEARUSDT | long | True | 9.971403 | 48.2673 | 3.0758 | 13.8888 | 67980.7951 | 621.0 | data/metadata/params/NEARUSDT__UNIVERSE_LONG_C3_active_params.json |
| 16 | ZECUSDT | long | True | 8.191008 | 47.8159 | 2.3548 | 12.7463 | 131696.4204 | 1074.0 | data/metadata/params/ZECUSDT__UNIVERSE_LONG_C1_active_params.json |
| 17 | OGUSDT | long | False | 7.178837 | 109.8437 | 2.8460 | 42.5465 | 398098.6028 | 1033.0 | data/metadata/params/OGUSDT__UNIVERSE_LONG_C3_active_params.json |
| 18 | DOGEUSDT | long | True | 7.142295 | 66.7319 | 2.4104 | 21.5207 | 266376.6427 | 1135.0 | data/metadata/params/DOGEUSDT_C13_active_params_long.json |
| 19 | ADAUSDT | long | True | 6.161227 | 57.2407 | 1.8649 | 16.3256 | 317614.2282 | 2184.0 | data/metadata/params/ADAUSDT_C13_active_params_long.json |
| 20 | ADAUSDT | long | True | 6.161227 | 57.2407 | 1.8649 | 16.3256 | 317614.2282 | 2184.0 | data/metadata/params/ADAUSDT__UNIVERSE_LONG_active_params.json |
| 21 | ADAUSDT | long | True | 6.161227 | 57.2407 | 1.8649 | 16.3256 | 317614.2282 | 2184.0 | data/metadata/params/ADAUSDT_active_params.json |
| 22 | DOGEUSDT | long | True | 5.894314 | 45.3015 | 2.7457 | 20.1024 | 103128.2075 | 746.0 | data/metadata/params/DOGEUSDT__UNIVERSE_LONG_C3_active_params.json |
| 23 | LINKUSDT | long | True | 5.689143 | 32.3706 | 2.3710 | 12.4909 | 60383.2999 | 772.0 | data/metadata/params/LINKUSDT__UNIVERSE_LONG_C1_active_params.json |
| 24 | AXSUSDT | long | True | 5.354814 | 36.1391 | 2.9835 | 19.1355 | 39077.8058 | 467.0 | data/metadata/params/AXSUSDT__UNIVERSE_LONG_C3_active_params.json |
| 25 | ZECUSDT | long | True | 5.264350 | 40.4936 | 2.2643 | 16.4172 | 90386.4941 | 965.0 | data/metadata/params/ZECUSDT_C13_active_params_long.json |
| 26 | OGUSDT | long | False | 5.255605 | 87.2891 | 2.6617 | 43.2070 | 221020.5126 | 946.0 | data/metadata/params/OGUSDT_C13_active_params_long.json |
| 27 | AXSUSDT | long | True | 4.390640 | 28.6329 | 2.9756 | 18.4048 | 26633.3164 | 374.0 | data/metadata/params/AXSUSDT_C13_active_params_long.json |
| 28 | LINKUSDT | long | True | 4.379552 | 27.6343 | 2.0067 | 11.6620 | 44621.1094 | 821.0 | data/metadata/params/LINKUSDT__UNIVERSE_LONG_C3_active_params.json |
| 29 | BCHUSDT | long | True | 3.903781 | 34.0476 | 1.7052 | 13.8726 | 49623.1902 | 1315.0 | data/metadata/params/BCHUSDT__UNIVERSE_LONG_C3_active_params.json |
| 30 | AXSUSDT | long | True | 3.867879 | 31.7457 | 2.3567 | 18.3426 | 31440.2306 | 580.0 | data/metadata/params/AXSUSDT__UNIVERSE_LONG_C1_active_params.json |

## Passing Runs

- `SOLUSDT` (long): score=25.380137, cagr=235.29%, pf=1.797, dd=15.66%, params=`data/metadata/params/SOLUSDT_C13_active_params_long.json`
- `SOLUSDT` (long): score=25.380137, cagr=235.29%, pf=1.797, dd=15.66%, params=`data/metadata/params/SOLUSDT__UNIVERSE_LONG_active_params.json`
- `SOLUSDT` (long): score=25.380137, cagr=235.29%, pf=1.797, dd=15.66%, params=`data/metadata/params/SOLUSDT_active_params.json`
- `AVAXUSDT` (long): score=18.660894, cagr=107.31%, pf=2.664, dd=14.32%, params=`data/metadata/params/AVAXUSDT__UNIVERSE_LONG_active_params.json`
- `AVAXUSDT` (long): score=18.660894, cagr=107.31%, pf=2.664, dd=14.32%, params=`data/metadata/params/AVAXUSDT_active_params.json`
- `AVAXUSDT` (long): score=18.136839, cagr=109.13%, pf=2.521, dd=14.17%, params=`data/metadata/params/AVAXUSDT__UNIVERSE_LONG_C1_active_params.json`
- `AVAXUSDT` (long): score=16.588420, cagr=94.46%, pf=2.524, dd=13.37%, params=`data/metadata/params/AVAXUSDT__UNIVERSE_LONG_C3_active_params.json`
- `AVAXUSDT` (long): score=16.452744, cagr=93.92%, pf=2.518, dd=13.37%, params=`data/metadata/params/AVAXUSDT_C13_active_params_long.json`
- `NEARUSDT` (long): score=15.312473, cagr=92.21%, pf=2.612, dd=14.73%, params=`data/metadata/params/NEARUSDT_C13_active_params_long.json`
- `CRVUSDT` (long): score=14.622679, cagr=101.55%, pf=2.619, dd=17.19%, params=`data/metadata/params/CRVUSDT__UNIVERSE_LONG_C1_active_params.json`
- `CRVUSDT` (long): score=12.116950, cagr=89.43%, pf=2.636, dd=18.45%, params=`data/metadata/params/CRVUSDT_C13_active_params_long.json`
- `CRVUSDT` (long): score=11.292022, cagr=72.59%, pf=2.636, dd=15.95%, params=`data/metadata/params/CRVUSDT__UNIVERSE_LONG_C3_active_params.json`
- `DOGEUSDT` (long): score=10.955928, cagr=81.41%, pf=3.207, dd=22.83%, params=`data/metadata/params/DOGEUSDT__UNIVERSE_LONG_C1_active_params.json`
- `NEARUSDT` (long): score=10.228510, cagr=75.28%, pf=2.291, dd=15.86%, params=`data/metadata/params/NEARUSDT__UNIVERSE_LONG_C1_active_params.json`
- `NEARUSDT` (long): score=9.971403, cagr=48.27%, pf=3.076, dd=13.89%, params=`data/metadata/params/NEARUSDT__UNIVERSE_LONG_C3_active_params.json`
- `ZECUSDT` (long): score=8.191008, cagr=47.82%, pf=2.355, dd=12.75%, params=`data/metadata/params/ZECUSDT__UNIVERSE_LONG_C1_active_params.json`
- `DOGEUSDT` (long): score=7.142295, cagr=66.73%, pf=2.410, dd=21.52%, params=`data/metadata/params/DOGEUSDT_C13_active_params_long.json`
- `ADAUSDT` (long): score=6.161227, cagr=57.24%, pf=1.865, dd=16.33%, params=`data/metadata/params/ADAUSDT_C13_active_params_long.json`
- `ADAUSDT` (long): score=6.161227, cagr=57.24%, pf=1.865, dd=16.33%, params=`data/metadata/params/ADAUSDT__UNIVERSE_LONG_active_params.json`
- `ADAUSDT` (long): score=6.161227, cagr=57.24%, pf=1.865, dd=16.33%, params=`data/metadata/params/ADAUSDT_active_params.json`
- `DOGEUSDT` (long): score=5.894314, cagr=45.30%, pf=2.746, dd=20.10%, params=`data/metadata/params/DOGEUSDT__UNIVERSE_LONG_C3_active_params.json`
- `LINKUSDT` (long): score=5.689143, cagr=32.37%, pf=2.371, dd=12.49%, params=`data/metadata/params/LINKUSDT__UNIVERSE_LONG_C1_active_params.json`
- `AXSUSDT` (long): score=5.354814, cagr=36.14%, pf=2.984, dd=19.14%, params=`data/metadata/params/AXSUSDT__UNIVERSE_LONG_C3_active_params.json`
- `ZECUSDT` (long): score=5.264350, cagr=40.49%, pf=2.264, dd=16.42%, params=`data/metadata/params/ZECUSDT_C13_active_params_long.json`
- `AXSUSDT` (long): score=4.390640, cagr=28.63%, pf=2.976, dd=18.40%, params=`data/metadata/params/AXSUSDT_C13_active_params_long.json`
- `LINKUSDT` (long): score=4.379552, cagr=27.63%, pf=2.007, dd=11.66%, params=`data/metadata/params/LINKUSDT__UNIVERSE_LONG_C3_active_params.json`
- `BCHUSDT` (long): score=3.903781, cagr=34.05%, pf=1.705, dd=13.87%, params=`data/metadata/params/BCHUSDT__UNIVERSE_LONG_C3_active_params.json`
- `AXSUSDT` (long): score=3.867879, cagr=31.75%, pf=2.357, dd=18.34%, params=`data/metadata/params/AXSUSDT__UNIVERSE_LONG_C1_active_params.json`
- `ZECUSDT` (long): score=3.371758, cagr=25.19%, pf=2.143, dd=15.01%, params=`data/metadata/params/ZECUSDT__UNIVERSE_LONG_C3_active_params.json`
- `XRPUSDT` (long): score=2.390345, cagr=33.61%, pf=2.092, dd=28.41%, params=`data/metadata/params/XRPUSDT__UNIVERSE_LONG_C3_active_params.json`
- `BCHUSDT` (long): score=2.362379, cagr=21.95%, pf=1.724, dd=15.02%, params=`data/metadata/params/BCHUSDT_C13_active_params_long.json`
- `BCHUSDT` (long): score=2.325915, cagr=22.03%, pf=1.762, dd=15.68%, params=`data/metadata/params/BCHUSDT__UNIVERSE_LONG_C1_active_params.json`
- `LTCUSDT` (long): score=2.248108, cagr=15.47%, pf=1.786, dd=11.30%, params=`data/metadata/params/LTCUSDT__UNIVERSE_LONG_C3_active_params.json`
- `XRPUSDT` (long): score=1.918021, cagr=32.01%, pf=2.030, dd=32.88%, params=`data/metadata/params/XRPUSDT_C13_active_params_long.json`
- `XRPUSDT` (long): score=1.918021, cagr=32.01%, pf=2.030, dd=32.88%, params=`data/metadata/params/XRPUSDT__UNIVERSE_LONG_active_params.json`
- `XRPUSDT` (long): score=1.918021, cagr=32.01%, pf=2.030, dd=32.88%, params=`data/metadata/params/XRPUSDT_active_params.json`
- `XRPUSDT` (long): score=1.890101, cagr=31.65%, pf=2.024, dd=32.88%, params=`data/metadata/params/XRPUSDT__UNIVERSE_LONG_C1_active_params.json`
- `TRXUSDT` (long): score=1.489597, cagr=19.86%, pf=1.674, dd=21.32%, params=`data/metadata/params/TRXUSDT__UNIVERSE_LONG_C1_active_params.json`
- `LINKUSDT` (long): score=1.447653, cagr=18.67%, pf=1.634, dd=20.07%, params=`data/metadata/params/LINKUSDT_C13_active_params_long.json`
- `TRXUSDT` (long): score=0.806306, cagr=15.73%, pf=1.440, dd=27.08%, params=`data/metadata/params/TRXUSDT__UNIVERSE_LONG_C3_active_params.json`
