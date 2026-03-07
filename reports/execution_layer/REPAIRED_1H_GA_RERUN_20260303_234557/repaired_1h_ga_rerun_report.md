# Repaired 1H GA Rerun Report

- Generated UTC: `2026-03-03T23:47:02.663429+00:00`
- Scope: fresh 1h-only discovery rerun under the repaired chronology-valid contract.
- 3m execution layer: explicitly excluded.

## Discovered Legacy GA Code Paths
- `scripts/phase_h_execaware_1h_ga_pilot.py`: legacy bounded execution-aware 1h GA pilot and candidate evaluator source.
- `scripts/phase_i_execaware_1h_ga_expansion.py`: legacy Phase I expansion runner, population mix, and OJ2/OJ4 frontier ranking.
- `scripts/phase_u_combined_1h3m_pilot.py`: legacy helper layer for param fingerprints / signal activity mapping used by H/I.
- `src/bot087/optim/ga.py`: reusable 1h parameter mutation / crossover / normalization machinery.

## Exact Repaired-Contract Changes
- Did not reuse the contaminated Phase I shortlist as a seed shortlist or ranking base.
- Did not reuse legacy OJ2/OJ4 because those depend on execution-aware 3m metrics from the broken discovery stack.
- Reused only the parameter search mechanics from `src/bot087/optim/ga.py` (`_norm_params`, `mutate_params`, `crossover`).
- Replaced candidate evaluation with the repaired 1h-only evaluator from `scripts/repaired_frontier_contamination_audit.py`, which calls `scripts/backtest_exec_phasec_sol.py` logic with deferred next-bar exit semantics.
- Replacement objective: `repaired_1h_score = (cagr_pct * profit_factor) / (1 + max_dd_pct)` using the exact `scan_params_all_coins.py` formula and pass thresholds from `scan_meta.json`.

## Repaired Frontier Summary
- Total evaluated candidates: `36`
- Valid frontier count: `24`
- Unique symbols in valid frontier: `14`
- Top candidate: `R_SOLUSDT_G00_I0000` (SOLUSDT) score=`131.036438`

### Top Repaired Frontier

| frontier_rank | candidate_id | symbol | repaired_score | repaired_cagr_pct | repaired_profit_factor | repaired_max_dd_pct | repaired_trades | repaired_expectancy_net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | R_SOLUSDT_G00_I0000 | SOLUSDT | 131.0364376 | 1291.798033 | 2.551053285 | 24.14907821 | 5034 | 0.002962678234 |
| 2 | R_DOGEUSDT_G00_I0000 | DOGEUSDT | 38.68320862 | 189.8559996 | 3.718794412 | 17.25172874 | 1391 | 0.005300785731 |
| 3 | R_OGUSDT_G00_I0000 | OGUSDT | 29.77632806 | 180.6974785 | 3.16055923 | 18.17983582 | 1322 | 0.00423455664 |
| 4 | R_NEARUSDT_G00_I0000 | NEARUSDT | 28.96871049 | 144.4383552 | 2.843034834 | 13.17540747 | 1380 | 0.003583787267 |
| 5 | R_AVAXUSDT_G00_I0000 | AVAXUSDT | 28.87906936 | 193.9115472 | 2.779812499 | 17.66534326 | 1732 | 0.003388097752 |

## Repaired Universe Candidate Output

| repaired_rank | symbol | candidate_id | valid_for_ranking | repaired_score | legacy_score | score_delta_vs_legacy | membership_action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SOLUSDT | R_SOLUSDT_G00_I0000 | 1 | 131.0364376 | 25.38013748 | 105.6563001 | STAY_PASS |
| 2 | DOGEUSDT | R_DOGEUSDT_G00_I0000 | 1 | 38.68320862 | 10.95592768 | 27.72728094 | STAY_PASS |
| 3 | OGUSDT | R_OGUSDT_G00_I0000 | 1 | 29.77632806 | 7.178837326 | 22.59749073 | NEW_PASS |
| 4 | NEARUSDT | R_NEARUSDT_G00_I0000 | 1 | 28.96871049 | 15.31247307 | 13.65623742 | STAY_PASS |
| 5 | AVAXUSDT | R_AVAXUSDT_G00_I0000 | 1 | 28.87906936 | 18.66089358 | 10.21817579 | STAY_PASS |
| 6 | AXSUSDT | R_AXSUSDT_G00_I0000 | 1 | 18.61387516 | 5.354813941 | 13.25906122 | STAY_PASS |
| 7 | CRVUSDT | R_CRVUSDT_G00_I0000 | 1 | 17.15620957 | 14.62267899 | 2.533530573 | STAY_PASS |
| 8 | ZECUSDT | R_ZECUSDT_G00_I0000 | 1 | 12.49946056 | 8.191007607 | 4.308452954 | STAY_PASS |
| 9 | BCHUSDT | R_BCHUSDT_G00_I0000 | 1 | 9.414974137 | 3.903781135 | 5.511193002 | STAY_PASS |
| 10 | ADAUSDT | R_ADAUSDT_G00_I0000 | 1 | 8.929867538 | 6.161226872 | 2.768640666 | STAY_PASS |
| 11 | LINKUSDT | R_LINKUSDT_G00_I0000 | 1 | 8.06326459 | 5.689143171 | 2.374121419 | STAY_PASS |
| 12 | BTCUSDT | R_BTCUSDT_G00_I0000 | 1 | 2.218052398 | 1.057620764 | 1.160431634 | NEW_PASS |
| 13 | LTCUSDT | R_LTCUSDT_G00_I0001 | 1 | 1.837745319 | 2.248107502 | -0.4103621824 | STAY_PASS |
| 14 | TRXUSDT | R_TRXUSDT_G00_I0000 | 1 | 1.69580223 | 1.489596981 | 0.2062052488 | STAY_PASS |
| 15 | XRPUSDT | R_XRPUSDT_G00_I0001 | 0 | 2.642455029 | 2.390344799 | 0.2521102308 | DROP_FROM_PASS |
| 16 | ETHUSDT | R_ETHUSDT_G00_I0000 | 0 | 2.060085393 | 0.9351817742 | 1.124903619 | STAY_FAIL |
| 17 | BNBUSDT | R_BNBUSDT_G00_I0001 | 0 | 0.2343463155 | 0.6410526884 | -0.4067063729 | STAY_FAIL |
| 18 | PAXGUSDT | R_PAXGUSDT_G00_I0001 | 0 | -0.1080896336 | -0.04064739923 | -0.06744223433 | STAY_FAIL |

## Legacy Comparison
- Legacy Phase I remained SOL-seeded; this rerun is multicoin across the canonical long scan symbol set.

| diff_type | scope | legacy_topk_size | legacy_symbols | new_global_topk_size | new_global_symbols | param_hash_overlap_with_new_global | param_hash_overlap_with_new_sol | legacy_mean_oj2 | new_global_mean_score | symbol | legacy_pass | repaired_pass | legacy_score | repaired_score | score_delta | legacy_rank | repaired_rank | rank_delta | membership_action | legacy_params_file | repaired_candidate_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frontier_topk_summary | top5 | 5 | SOLUSDT | 5 | AVAXUSDT,DOGEUSDT,NEARUSDT,OGUSDT,SOLUSDT | 0 | 0 | 0.1486168527 | 51.46875083 |  |  |  |  |  |  |  |  |  |  |  |  |
| frontier_topk_summary | top10 | 10 | SOLUSDT | 10 | AVAXUSDT,AXSUSDT,CRVUSDT,DOGEUSDT,NEARUSDT,OGUSDT,SOLUSDT | 0 | 0 | 0.1447554736 | 35.38022812 |  |  |  |  |  |  |  |  |  |  |  |  |
| frontier_topk_summary | top20 | 20 | SOLUSDT | 20 | ADAUSDT,AVAXUSDT,AXSUSDT,BCHUSDT,BTCUSDT,CRVUSDT,DOGEUSDT,LINKUSDT,NEARUSDT,OGUSDT,SOLUSDT,ZECUSDT | 0 | 0 | 0.1395430877 | 21.04738309 |  |  |  |  |  |  |  |  |  |  |  |  |

## Final Recommendation
- Recommendation: `PROCEED_TO_REPAIRED_UNIVERSE_REBUILD`
- Reason: The repaired 1h rerun produced a broad enough passing long set to rebuild the universe before any new 3m work.
