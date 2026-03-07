# Multi-Coin Model A Audit

- Generated UTC: 2026-02-28T16:43:51.569055+00:00
- Foundation source: `/root/analysis/0.87/reports/execution_layer/UNIVERSAL_DATA_FOUNDATION_20260228_150929`
- Universe size: `17`
- Variants tested per coin: `5`
- Contract parity:
  - 1h signal owner: `src/bot087/optim/ga.py` via frozen universal timeline
  - 1h exit owner: `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference` semantics via `phase_a_model_a_audit.simulate_frozen_1h_exit`
  - 3m entry executor: `phase_a_model_a_audit.simulate_entry_only_fill`
  - forbidden hybrid exit knobs blocked: `['tp_mult', 'sl_mult', 'time_stop_min', 'break_even_enabled', 'break_even_trigger_r', 'break_even_offset_bps', 'trailing_enabled', 'trail_start_r', 'trail_step_bps', 'partial_take_enabled', 'partial_take_r', 'partial_take_pct']`

## Per-Coin Classification

| symbol | classification | best_candidate_id | best_delta_expectancy_vs_1h_reference | best_cvar_improve_ratio | best_maxdd_improve_ratio | best_valid_for_ranking | foundation_integrity_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ADAUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.0004607239175 | 1 | 1 | 0 | BLOCKED |
| AVAXUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.003634460567 | 0.03058207172 | 0.7172237455 | 0 | PARTIAL |
| AXSUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001171436069 | 1 | 1 | 0 | BLOCKED |
| BCHUSDT | DATA_BLOCKED | M3_ENTRY_ONLY_FASTER | 0.002343547835 | 8.169806546e-14 | 0.8667042103 | 0 | PARTIAL |
| BNBUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001190995848 | 1 | 1 | 0 | BLOCKED |
| BTCUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.0008583389264 | 1 | 1 | 0 | BLOCKED |
| CRVUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001096379696 | 1 | 1 | 0 | BLOCKED |
| DOGEUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001009309468 | 1 | 1 | 0 | BLOCKED |
| LINKUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.0008774273342 | 1 | 1 | 0 | BLOCKED |
| LTCUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001473756908 | 1 | 1 | 0 | BLOCKED |
| NEARUSDT | MODEL_A_WEAK_GO | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.003848069509 | 0.04543622084 | 0.8333147414 | 1 | PARTIAL |
| OGUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.0002255554117 | 1 | 1 | 0 | BLOCKED |
| PAXGUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.00112693572 | 1 | 1 | 0 | BLOCKED |
| SOLUSDT | MODEL_A_NO_GO | M3_ENTRY_ONLY_FASTER | 0.002537399098 | 0.054403896 | 0.8546979974 | 0 | PARTIAL |
| TRXUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001044068083 | 1 | 1 | 0 | BLOCKED |
| XRPUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.0006409943865 | 1 | 1 | 0 | BLOCKED |
| ZECUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.0001402848788 | 1 | 1 | 0 | BLOCKED |

## Top Improving Coins

| symbol | classification | best_candidate_id | delta_expectancy_vs_1h_reference | cvar_improve_ratio | maxdd_improve_ratio | best_valid_for_ranking |
| --- | --- | --- | --- | --- | --- | --- |
| NEARUSDT | MODEL_A_WEAK_GO | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.003848069509 | 0.04543622084 | 0.8333147414 | 1 |
| AVAXUSDT | MODEL_A_NO_GO | M2_ENTRY_ONLY_MORE_PASSIVE | 0.003634460567 | 0.03058207172 | 0.7172237455 | 0 |
| SOLUSDT | MODEL_A_NO_GO | M3_ENTRY_ONLY_FASTER | 0.002537399098 | 0.054403896 | 0.8546979974 | 0 |
| BCHUSDT | DATA_BLOCKED | M3_ENTRY_ONLY_FASTER | 0.002343547835 | 8.169806546e-14 | 0.8667042103 | 0 |
| LTCUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001473756908 | 1 | 1 | 0 |
| BNBUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001190995848 | 1 | 1 | 0 |
| AXSUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001171436069 | 1 | 1 | 0 |
| PAXGUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.00112693572 | 1 | 1 | 0 |
| CRVUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001096379696 | 1 | 1 | 0 |
| TRXUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0.001044068083 | 1 | 1 | 0 |

## Largest DD Improvements

| symbol | classification | best_candidate_id | maxdd_improve_ratio | delta_expectancy_vs_1h_reference | best_valid_for_ranking |
| --- | --- | --- | --- | --- | --- |
| LTCUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.001473756908 | 0 |
| BNBUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.001190995848 | 0 |
| AXSUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.001171436069 | 0 |
| PAXGUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.00112693572 | 0 |
| CRVUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.001096379696 | 0 |
| TRXUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.001044068083 | 0 |
| DOGEUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.001009309468 | 0 |
| LINKUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.0008774273342 | 0 |
| BTCUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.0008583389264 | 0 |
| XRPUSDT | DATA_BLOCKED | M1_ENTRY_ONLY_PASSIVE_BASELINE | 1 | 0.0006409943865 | 0 |

## Data-Blocked Coins

| symbol | classification_reason | foundation_missing_window_rate | foundation_signals_covered | foundation_signals_uncovered |
| --- | --- | --- | --- | --- |
| ADAUSDT | no_sliced_3m_windows | 1 | 0 | 2660 |
| AXSUSDT | no_sliced_3m_windows | 1 | 0 | 573 |
| BCHUSDT | foundation_missing_window_rate>0.0200 | 0.6539572501 | 599 | 1132 |
| BNBUSDT | no_sliced_3m_windows | 1 | 0 | 617 |
| BTCUSDT | no_sliced_3m_windows | 1 | 0 | 905 |
| CRVUSDT | no_sliced_3m_windows | 1 | 0 | 1586 |
| DOGEUSDT | no_sliced_3m_windows | 1 | 0 | 1391 |
| LINKUSDT | no_sliced_3m_windows | 1 | 0 | 968 |
| LTCUSDT | no_sliced_3m_windows | 1 | 0 | 940 |
| OGUSDT | no_sliced_3m_windows | 1 | 0 | 1322 |
| PAXGUSDT | no_sliced_3m_windows | 1 | 0 | 231 |
| TRXUSDT | no_sliced_3m_windows | 1 | 0 | 976 |
| XRPUSDT | no_sliced_3m_windows | 1 | 0 | 2034 |
| ZECUSDT | no_sliced_3m_windows | 1 | 0 | 1374 |

## Not Tradable After Execution

| symbol | classification | classification_reason | best_candidate_id | best_valid_for_ranking | best_delta_expectancy_vs_1h_reference |
| --- | --- | --- | --- | --- | --- |
| ADAUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.0004607239175 |
| AVAXUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M2_ENTRY_ONLY_MORE_PASSIVE | 0 | 0.003634460567 |
| AXSUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.001171436069 |
| BCHUSDT | DATA_BLOCKED | foundation_missing_window_rate>0.0200 | M3_ENTRY_ONLY_FASTER | 0 | 0.002343547835 |
| BNBUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.001190995848 |
| BTCUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.0008583389264 |
| CRVUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.001096379696 |
| DOGEUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.001009309468 |
| LINKUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.0008774273342 |
| LTCUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.001473756908 |
| OGUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.0002255554117 |
| PAXGUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.00112693572 |
| SOLUSDT | MODEL_A_NO_GO | no_robust_entry_only_advantage | M3_ENTRY_ONLY_FASTER | 0 | 0.002537399098 |
| TRXUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.001044068083 |
| XRPUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.0006409943865 |
| ZECUSDT | DATA_BLOCKED | no_sliced_3m_windows | M1_ENTRY_ONLY_PASSIVE_BASELINE | 0 | 0.0001402848788 |
