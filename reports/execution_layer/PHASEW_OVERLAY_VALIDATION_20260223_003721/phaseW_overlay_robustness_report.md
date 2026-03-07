# Phase W Overlay Robustness Report

- Generated UTC: 2026-02-23T01:05:46.445566+00:00
- Selected overlays tested: 4

candidate_id           candidate_hash overlay_id                 overlay_desc       overlay_eval_type  baseline_hard_gate_pass  baseline_delta_expectancy_vs_base  baseline_maxdd_improve_ratio_vs_base  baseline_cvar_improve_ratio_vs_base  oos_pass_rate  stress_sign_retention  stress_risk_retention  stress_gate_rate verdict
          E2 992bd371689ba3936f3b4d09        O1b           session_veto_12_17 exact_engine_integrated                        1                           0.000159                              0.208248                        -6.628694e-15            0.0                    0.6                    0.6               1.0 BRITTLE
          E1 862c940746de0da984862d95        O1b           session_veto_12_17 exact_engine_integrated                        1                           0.000158                              0.206766                        -1.092049e-02            0.0                    0.6                    0.2               1.0 BRITTLE
          E1 862c940746de0da984862d95        O1c session_veto_subwindow_12_13 exact_engine_integrated                        1                           0.000151                              0.130511                         6.423819e-04            0.0                    0.6                    1.0               1.0 BRITTLE
          E1 862c940746de0da984862d95        O1a           session_veto_06_11 exact_engine_integrated                        1                          -0.000139                              0.305944                         1.716077e-02            0.0                    0.0                    1.0               1.0 BRITTLE
