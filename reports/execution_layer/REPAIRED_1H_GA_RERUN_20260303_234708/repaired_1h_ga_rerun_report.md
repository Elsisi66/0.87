# Repaired 1H GA Rerun Report

- Generated UTC: `2026-03-03T23:54:06.950842+00:00`
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
- Total evaluated candidates: `250`
- Valid frontier count: `126`
- Unique symbols in valid frontier: `15`
- Top candidate: `R_SOLUSDT_G02_I0013` (SOLUSDT) score=`142.249306`

### Top Repaired Frontier

| frontier_rank | candidate_id | symbol | repaired_score | repaired_cagr_pct | repaired_profit_factor | repaired_max_dd_pct | repaired_trades | repaired_expectancy_net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | R_SOLUSDT_G02_I0013 | SOLUSDT | 142.2493064 | 1240.243187 | 2.602936269 | 21.69447953 | 4773 | 0.003082249057 |
| 2 | R_SOLUSDT_G00_I0001 | SOLUSDT | 140.2572797 | 1474.911376 | 2.535283461 | 25.66042309 | 5337 | 0.002927686898 |
| 3 | R_SOLUSDT_G02_I0010 | SOLUSDT | 138.2690621 | 1423.278358 | 2.534761559 | 25.09167383 | 5273 | 0.00292700318 |
| 4 | R_SOLUSDT_G00_I0002 | SOLUSDT | 136.6949234 | 1162.266938 | 2.578876316 | 20.92724209 | 4768 | 0.00300918164 |
| 5 | R_SOLUSDT_G01_I0006 | SOLUSDT | 132.6899232 | 1322.992429 | 2.55083756 | 24.43327104 | 5076 | 0.002962619892 |
| 6 | R_SOLUSDT_G00_I0003 | SOLUSDT | 131.7496312 | 1340.576887 | 2.548435867 | 24.93080672 | 5107 | 0.002958082379 |
| 7 | R_SOLUSDT_G00_I0000 | SOLUSDT | 131.0364376 | 1291.798033 | 2.551053285 | 24.14907821 | 5034 | 0.002962678234 |
| 8 | R_SOLUSDT_G01_I0009 | SOLUSDT | 123.5027907 | 1618.219129 | 2.599404959 | 33.0592047 | 5178 | 0.00312507221 |
| 9 | R_SOLUSDT_G02_I0012 | SOLUSDT | 118.7662576 | 1430.401048 | 2.634928108 | 30.73463576 | 4892 | 0.003166330382 |
| 10 | R_SOLUSDT_G01_I0008 | SOLUSDT | 116.8752817 | 1341.228582 | 2.625367159 | 29.12799132 | 4781 | 0.003172734294 |
| 11 | R_SOLUSDT_G01_I0007 | SOLUSDT | 113.3104969 | 1187.703301 | 2.527753091 | 25.49552137 | 4952 | 0.00292238664 |
| 12 | R_OGUSDT_G01_I0009 | OGUSDT | 51.7042549 | 163.2261065 | 3.767223734 | 10.8928174 | 976 | 0.005369397322 |
| 13 | R_DOGEUSDT_G00_I0001 | DOGEUSDT | 41.71460345 | 203.3283562 | 3.514196488 | 16.12915229 | 1594 | 0.004797196929 |
| 14 | R_NEARUSDT_G02_I0013 | NEARUSDT | 40.72027103 | 211.228803 | 2.759139726 | 13.31252216 | 1839 | 0.003410590329 |
| 15 | R_DOGEUSDT_G01_I0009 | DOGEUSDT | 40.04921894 | 168.4391651 | 3.857284374 | 15.22298204 | 1234 | 0.005561460782 |
| 16 | R_OGUSDT_G00_I0002 | OGUSDT | 39.70129353 | 179.9274541 | 3.159766811 | 13.32015804 | 1319 | 0.004242631307 |
| 17 | R_AXSUSDT_G02_I0013 | AXSUSDT | 38.74123296 | 113.5583996 | 3.613380411 | 9.59154976 | 819 | 0.005137569915 |
| 18 | R_DOGEUSDT_G00_I0000 | DOGEUSDT | 38.68320862 | 189.8559996 | 3.718794412 | 17.25172874 | 1391 | 0.005300785731 |
| 19 | R_DOGEUSDT_G00_I0003 | DOGEUSDT | 37.5170515 | 179.7599083 | 3.6947925 | 16.70329848 | 1356 | 0.005258467596 |
| 20 | R_OGUSDT_G02_I0011 | OGUSDT | 35.43679393 | 101.9610567 | 3.443883678 | 8.908966926 | 799 | 0.004749702267 |

## Repaired Universe Candidate Output

| repaired_rank | symbol | candidate_id | valid_for_ranking | repaired_score | legacy_score | score_delta_vs_legacy | membership_action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SOLUSDT | R_SOLUSDT_G02_I0013 | 1 | 142.2493064 | 25.38013748 | 116.869169 | STAY_PASS |
| 2 | OGUSDT | R_OGUSDT_G01_I0009 | 1 | 51.7042549 | 7.178837326 | 44.52541757 | NEW_PASS |
| 3 | DOGEUSDT | R_DOGEUSDT_G00_I0001 | 1 | 41.71460345 | 10.95592768 | 30.75867577 | STAY_PASS |
| 4 | NEARUSDT | R_NEARUSDT_G02_I0013 | 1 | 40.72027103 | 15.31247307 | 25.40779796 | STAY_PASS |
| 5 | AXSUSDT | R_AXSUSDT_G02_I0013 | 1 | 38.74123296 | 5.354813941 | 33.38641902 | STAY_PASS |
| 6 | AVAXUSDT | R_AVAXUSDT_G00_I0005 | 1 | 31.50126075 | 18.66089358 | 12.84036717 | STAY_PASS |
| 7 | CRVUSDT | R_CRVUSDT_G02_I0010 | 1 | 20.62460506 | 14.62267899 | 6.001926071 | STAY_PASS |
| 8 | ZECUSDT | R_ZECUSDT_G00_I0002 | 1 | 18.8586762 | 8.191007607 | 10.66766859 | STAY_PASS |
| 9 | BCHUSDT | R_BCHUSDT_G02_I0010 | 1 | 11.3810422 | 3.903781135 | 7.477261061 | STAY_PASS |
| 10 | ADAUSDT | R_ADAUSDT_G00_I0001 | 1 | 11.08381201 | 6.161226872 | 4.92258514 | STAY_PASS |
| 11 | LINKUSDT | R_LINKUSDT_G00_I0004 | 1 | 9.354704118 | 5.689143171 | 3.665560948 | STAY_PASS |
| 12 | TRXUSDT | R_TRXUSDT_G02_I0011 | 1 | 4.104116938 | 1.489596981 | 2.614519957 | STAY_PASS |
| 13 | XRPUSDT | R_XRPUSDT_G00_I0004 | 1 | 3.445764159 | 2.390344799 | 1.05541936 | STAY_PASS |
| 14 | BTCUSDT | R_BTCUSDT_G00_I0000 | 1 | 2.218052398 | 1.057620764 | 1.160431634 | NEW_PASS |
| 15 | LTCUSDT | R_LTCUSDT_G00_I0000 | 1 | 1.568379599 | 2.248107502 | -0.6797279032 | STAY_PASS |
| 16 | ETHUSDT | R_ETHUSDT_G00_I0003 | 0 | 4.401830771 | 0.9351817742 | 3.466648996 | STAY_FAIL |
| 17 | BNBUSDT | R_BNBUSDT_G00_I0005 | 0 | 0.482796005 | 0.6410526884 | -0.1582566834 | STAY_FAIL |
| 18 | PAXGUSDT | R_PAXGUSDT_G02_I0011 | 0 | -0.1008415792 | -0.04064739923 | -0.06019418001 | STAY_FAIL |

## Legacy Comparison
- Legacy Phase I remained SOL-seeded; this rerun is multicoin across the canonical long scan symbol set.

| diff_type | scope | legacy_topk_size | legacy_symbols | new_global_topk_size | new_global_symbols | param_hash_overlap_with_new_global | param_hash_overlap_with_new_sol | legacy_mean_oj2 | new_global_mean_score | symbol | legacy_pass | repaired_pass | legacy_score | repaired_score | score_delta | legacy_rank | repaired_rank | rank_delta | membership_action | legacy_params_file | repaired_candidate_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frontier_topk_summary | top5 | 5 | SOLUSDT | 5 | SOLUSDT | 0 | 0 | 0.1486168527 | 138.0320989 |  |  |  |  |  |  |  |  |  |  |  |  |
| frontier_topk_summary | top10 | 10 | SOLUSDT | 10 | SOLUSDT | 0 | 0 | 0.1447554736 | 131.2090894 |  |  |  |  |  |  |  |  |  |  |  |  |
| frontier_topk_summary | top20 | 20 | SOLUSDT | 20 | AXSUSDT,DOGEUSDT,NEARUSDT,OGUSDT,SOLUSDT | 0 | 0 | 0.1395430877 | 89.48346596 |  |  |  |  |  |  |  |  |  |  |  |  |

## Final Recommendation
- Recommendation: `PROCEED_TO_REPAIRED_UNIVERSE_REBUILD`
- Reason: The repaired 1h rerun produced a broad enough passing long set to rebuild the universe before any new 3m work.
