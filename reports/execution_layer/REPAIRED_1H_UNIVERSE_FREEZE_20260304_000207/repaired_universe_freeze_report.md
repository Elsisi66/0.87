# Repaired 1H Universe Freeze Report

- Generated UTC: `2026-03-04T00:02:07.799577+00:00`
- Fresh repaired rerun source: `/root/analysis/0.87/reports/execution_layer/REPAIRED_1H_GA_RERUN_20260303_234708`
- This rebuilt universe now replaces the old `best_by_symbol.csv`-driven long set for the repaired branch.
- Downstream 3m execution work must use this frozen repaired universe, not the legacy universe.

## Selection Rule
- Eligibility: `side == long`, `valid_for_ranking == 1`, `repaired_pass == 1`
- Exactly one canonical best candidate per symbol
- Primary sort: `repaired_score desc`
- Secondary sort: `repaired_expectancy_net desc`
- Tertiary sort: `repaired_max_dd_pct asc`
- Stable tie-breaks: `candidate_id asc`, then `param_hash asc`
- Verification: reconstructed selection matches `repaired_1h_universe_candidates.csv` exactly

## Frozen Repaired Universe Summary
- Selected symbols: `15`
- Retained from legacy long pass set: `13`
- New symbols: `2`
- Dropped symbols: `0`

| rank | symbol | candidate_id | score | expectancy_net | cvar_5 | max_dd_pct | profit_factor | trades | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SOLUSDT | R_SOLUSDT_G02_I0013 | 142.2493064 | 0.003082249057 | -0.002199520096 | 21.69447953 | 2.602936269 | 4773 | 1 |
| 2 | OGUSDT | R_OGUSDT_G01_I0009 | 51.7042549 | 0.005369397322 | -0.002199520096 | 10.8928174 | 3.767223734 | 976 | 1 |
| 3 | DOGEUSDT | R_DOGEUSDT_G00_I0001 | 41.71460345 | 0.004797196929 | -0.002199520096 | 16.12915229 | 3.514196488 | 1594 | 1 |
| 4 | NEARUSDT | R_NEARUSDT_G02_I0013 | 40.72027103 | 0.003410590329 | -0.002199520096 | 13.31252216 | 2.759139726 | 1839 | 1 |
| 5 | AXSUSDT | R_AXSUSDT_G02_I0013 | 38.74123296 | 0.005137569915 | -0.002199520096 | 9.59154976 | 3.613380411 | 819 | 1 |
| 6 | AVAXUSDT | R_AVAXUSDT_G00_I0005 | 31.50126075 | 0.003534071357 | -0.002199520096 | 16.38643901 | 2.865277415 | 1645 | 1 |
| 7 | CRVUSDT | R_CRVUSDT_G02_I0010 | 20.62460506 | 0.003493646472 | -0.002199520096 | 19.56285579 | 2.747179984 | 1540 | 1 |
| 8 | ZECUSDT | R_ZECUSDT_G00_I0002 | 18.8586762 | 0.003033087706 | -0.002199520394 | 12.44536144 | 2.601363149 | 1605 | 1 |
| 9 | BCHUSDT | R_BCHUSDT_G02_I0010 | 11.3810422 | 0.001847728735 | -0.002199520096 | 12.17458802 | 1.973488404 | 1966 | 1 |
| 10 | ADAUSDT | R_ADAUSDT_G00_I0001 | 11.08381201 | 0.003130086737 | -0.00977296811 | 32.16774783 | 2.31995983 | 2470 | 1 |
| 11 | LINKUSDT | R_LINKUSDT_G00_I0004 | 9.354704118 | 0.003022065257 | -0.002199520096 | 13.86678827 | 2.58881213 | 1039 | 1 |
| 12 | TRXUSDT | R_TRXUSDT_G02_I0011 | 4.104116938 | 0.001587467793 | -0.002199520096 | 10.55239144 | 1.856170193 | 1156 | 1 |
| 13 | XRPUSDT | R_XRPUSDT_G00_I0004 | 3.445764159 | 0.001795654497 | -0.002199520096 | 29.02257028 | 1.893402096 | 2030 | 1 |
| 14 | BTCUSDT | R_BTCUSDT_G00_I0000 | 2.218052398 | 0.00167407228 | -0.002381526199 | 13.58860709 | 1.772040723 | 905 | 1 |
| 15 | LTCUSDT | R_LTCUSDT_G00_I0000 | 1.568379599 | 0.001273251353 | -0.002199520096 | 14.96114093 | 1.658770311 | 940 | 1 |

## Legacy vs Repaired Membership Changes
- Retained symbols: `ADAUSDT, AVAXUSDT, AXSUSDT, BCHUSDT, CRVUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, NEARUSDT, SOLUSDT, TRXUSDT, XRPUSDT, ZECUSDT`
- Dropped symbols: `(none)`
- New symbols: `BTCUSDT, OGUSDT`

| symbol | legacy_pass | repaired_pass | membership_action | legacy_rank | repaired_rank | legacy_score | repaired_score | rank_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADAUSDT | 1 | 1 | STAY_PASS | 7 | 10 | 6.161226872 | 11.08381201 | 3 |
| AVAXUSDT | 1 | 1 | STAY_PASS | 2 | 6 | 18.66089358 | 31.50126075 | 4 |
| AXSUSDT | 1 | 1 | STAY_PASS | 9 | 5 | 5.354813941 | 38.74123296 | -4 |
| BCHUSDT | 1 | 1 | STAY_PASS | 10 | 9 | 3.903781135 | 11.3810422 | -1 |
| BNBUSDT | 0 | 0 | STAY_FAIL | 17 |  | 0.6410526884 |  |  |
| BTCUSDT | 0 | 1 | NEW_PASS | 15 | 14 | 1.057620764 | 2.218052398 | -1 |
| CRVUSDT | 1 | 1 | STAY_PASS | 4 | 7 | 14.62267899 | 20.62460506 | 3 |
| DOGEUSDT | 1 | 1 | STAY_PASS | 5 | 3 | 10.95592768 | 41.71460345 | -2 |
| ETHUSDT | 0 | 0 | STAY_FAIL | 16 |  | 0.9351817742 |  |  |
| LINKUSDT | 1 | 1 | STAY_PASS | 8 | 11 | 5.689143171 | 9.354704118 | 3 |
| LTCUSDT | 1 | 1 | STAY_PASS | 12 | 15 | 2.248107502 | 1.568379599 | 3 |
| NEARUSDT | 1 | 1 | STAY_PASS | 3 | 4 | 15.31247307 | 40.72027103 | 1 |
| OGUSDT | 0 | 1 | NEW_PASS | 14 | 2 | 7.178837326 | 51.7042549 | -12 |
| PAXGUSDT | 0 | 0 | STAY_FAIL | 18 |  | -0.04064739923 |  |  |
| SOLUSDT | 1 | 1 | STAY_PASS | 1 | 1 | 25.38013748 | 142.2493064 | 0 |
| TRXUSDT | 1 | 1 | STAY_PASS | 13 | 12 | 1.489596981 | 4.104116938 | -1 |
| XRPUSDT | 1 | 1 | STAY_PASS | 11 | 13 | 2.390344799 | 3.445764159 | 2 |
| ZECUSDT | 1 | 1 | STAY_PASS | 6 | 8 | 8.191007607 | 18.8586762 | 2 |

## Readiness Decision
- Ready for downstream use: `YES`
- No additional universe sanity check is required before downstream execution evaluation, because the rebuilt universe is sourced directly from the fresh repaired 1h rerun and frozen here.
