Phase: 2 (per-symbol execution configs)
Inputs: configs/execution_configs.yaml + walkforward_exec_limit.py symbol overrides
Symbol runs: reports/execution_layer/20260220_111706_walkforward_SOLUSDT/SOLUSDT_walkforward_test_summary.csv | reports/execution_layer/20260220_112651_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_summary.csv | reports/execution_layer/20260220_113038_walkforward_NEARUSDT/NEARUSDT_walkforward_test_summary.csv
AVAX test entry_rate: 0.963039
Gate: AVAX entry_rate >= 0.55 => PASS
Per-symbol constraints applied from YAML (min_entry_rate, max_taker_share, max_fill_delay_min)
Outcome: AVAX participation recovered after cache/data timestamp fix (no_3m_data cleared)
Artifacts: /root/analysis/0.87/reports/execution_layer/20260220_phase2_summary/phase2_symbol_summary.csv
Pass/Fail vs phase gate: PASS
Next: diagnose 1h bleed and apply minimal 1h risk gate(s)
