# Phase V SOL DD Forensics Report

- Generated UTC: 2026-02-22T23:51:41.359786+00:00
- Scenarios analyzed: SOL_BASE_E1, SOL_BASE_E2, SOL_P16_E1

## Scenario Table

scenario_id        source_tag  valid_for_ranking  entries_valid  entry_rate  exec_expectancy_net  exec_delta_expectancy_vs_baseline  exec_cvar_5  exec_max_drawdown  cvar_improve_ratio  maxdd_improve_ratio  max_consecutive_losses  loss_run_ge3_count  sl_loss_share  fee_drag_per_trade  worst_split_id worst_session_bucket worst_regime_bucket
SOL_BASE_E1 phaseQRS_survivor                  1            355    0.986111             0.000056                           0.000894    -0.001780          -0.186917            0.190848             0.580289                      33                  24       0.884618            0.000879               2                06_11              mid|up
SOL_BASE_E2 phaseQRS_survivor                  1            355    0.986111             0.000056                           0.000893    -0.001799          -0.187267            0.182012             0.579504                      33                  24       0.884708            0.000880               2                06_11              mid|up
 SOL_P16_E1        phaseU_top                  1            305    0.987055             0.000205                           0.000906    -0.001777          -0.147163            0.191953             0.568040                      45                  21       0.898553            0.000880               2                12_17              mid|up

## Overlay Micro-Benchmark

scenario_id overlay_id                      overlay_desc  entries_valid  entry_rate  exec_expectancy_net  delta_expectancy_vs_base  maxdd_improve_ratio_vs_base  cvar_improve_ratio_vs_base  removed_entries_pct  removed_loss_share_abs  hard_gate_proxy_pass
 SOL_P16_E1         O0                        no_overlay            305    0.987055             0.000205                  0.000000                     0.000000                    0.000000             0.000000                0.000000                     1
 SOL_P16_E1         O1          session_veto_worst:12_17            216    0.699029             0.000365                  0.000161                     0.218650                    0.036907             0.291803                0.299362                     0
 SOL_P16_E1         O2          daily_loss_cap:-0.003101            217    0.702265            -0.000040                 -0.000244                    -0.135827                    0.036907             0.288525                0.267814                     1
 SOL_P16_E1         O3 volatility_spike_veto_atr_pct>=85            300    0.970874             0.000229                  0.000025                     0.051620                    0.012302             0.016393                0.019737                     1
 SOL_P16_E1         O4          entry_improvement_bps>=0            305    0.987055             0.000205                  0.000000                     0.000000                    0.000000             0.000000                0.000000                     1
SOL_BASE_E1         O0                        no_overlay            355    0.986111             0.000056                  0.000000                     0.000000                    0.000000             0.000000                0.000000                     1
SOL_BASE_E1         O1          session_veto_worst:06_11            270    0.750000             0.000179                  0.000123                     0.305944                    0.054602             0.239437                0.242186                     1
SOL_BASE_E1         O2          daily_loss_cap:-0.003940            239    0.663889            -0.000173                 -0.000229                    -0.127847                    0.043682             0.326761                0.308047                     0
SOL_BASE_E1         O3 volatility_spike_veto_atr_pct>=85            347    0.963889             0.000089                  0.000033                     0.062602                    0.010920             0.022535                0.026056                     1
SOL_BASE_E1         O4          entry_improvement_bps>=0            355    0.986111             0.000056                  0.000000                     0.000000                    0.000000             0.000000                0.000000                     1
