# Phase W Overlay Benchmark Report

- Generated UTC: 2026-02-23T00:43:36.302983+00:00
- Candidate hashes: E1=862c940746de0da984862d95, E2=992bd371689ba3936f3b4d09
- Overlay rows (incl. O0): 24
- Approximate counterfactual rows: 6

Top overlays:
candidate_id overlay_id                                   overlay_desc       overlay_eval_type  hard_gate_pass  delta_expectancy_vs_base  maxdd_improve_ratio_vs_base  cvar_improve_ratio_vs_base  removed_entries_pct  loss_run_ge3_count  sl_loss_share  rank_score
          E2        O1b                             session_veto_12_17 exact_engine_integrated               1                  0.000159                     0.208248               -6.628694e-15             0.000000                  17       0.880213   45.042810
          E1        O1b                             session_veto_12_17 exact_engine_integrated               1                  0.000158                     0.206766               -1.092049e-02             0.000000                  17       0.880213   43.462930
          E1        O1c                   session_veto_subwindow_12_13 exact_engine_integrated               1                  0.000151                     0.130511                6.423819e-04             0.000000                  23       0.889010   24.717620
          E2        O1c                   session_veto_subwindow_12_13 exact_engine_integrated               1                  0.000151                     0.130267               -1.446260e-15             0.000000                  23       0.889107   24.596403
          E2        O2c loss_cluster_pause_after_3_losses_rest_session    proxy_counterfactual               1                 -0.000089                     0.023897                1.080252e-02             0.061972                  22       0.878437    2.962153
          E1        O2c loss_cluster_pause_after_3_losses_rest_session    proxy_counterfactual               1                 -0.000089                     0.023942                1.092049e-02             0.061972                  22       0.878336    2.984562
          E1        O1e        session_veto_06_11_plus_risky_vol_guard exact_engine_integrated               1                 -0.000094                     0.352468                4.200189e-03             0.000000                  20       0.905790   67.643215
          E2        O1e        session_veto_06_11_plus_risky_vol_guard exact_engine_integrated               1                 -0.000094                     0.351810               -4.338781e-15             0.000000                  20       0.905896   67.019929
