Phase: 1 (cost model realism check)
Inputs: symbols=SOLUSDT,AVAXUSDT,NEARUSDT; source walkforward test_signals under /root/analysis/0.87/reports/execution_layer
Scenarios: maker=[0.0, 2.0, 5.0], taker=[2.0, 5.0, 8.0], limit_slip=[0.0, 1.0, 3.0], market_slip=[1.0, 3.0, 6.0]
Generated rows: 243
Metric: exec beats baseline on net expectancy in 199/243 scenarios
Overall pass ratio: 0.818930
Gate threshold: 0.70
Gate result: PASS
Artifacts: cost_sensitivity_summary.csv, cost_sensitivity_report.md, cost_sensitivity_sources.csv
Next: per-symbol execution config overrides + walk-forward rerun (Phase 2)
