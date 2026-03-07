Phase: 3 (diagnose 1h bleed on test-only baseline)
Inputs: symbols=SOLUSDT,AVAXUSDT,NEARUSDT; latest walkforward test outputs under /root/analysis/0.87/reports/execution_layer
Buckets: stop_distance quantiles, ATR_1h quantiles, trend proxy, ATR percentile regime
Symbols processed: 3
Total baseline entries analyzed: 1501
Summary CSV: diagnose_1h_bleed_summary.csv
Key finding target: isolate 1-2 high-risk buckets for simple 1h risk gates
Gate status: DIAGNOSTIC (no pass/fail threshold in this phase)
Artifacts: /root/analysis/0.87/reports/execution_layer/20260220_112918
Next: enable 1h gates (off by default code path) and rerun walk-forward (Phase 4)
