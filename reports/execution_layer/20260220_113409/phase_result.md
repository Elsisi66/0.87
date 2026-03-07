Phase: 3 (diagnose 1h bleed on baseline test-only entries)
Inputs: symbols=SOLUSDT,AVAXUSDT,NEARUSDT; latest walkforward test_signals per symbol
Output dir: /root/analysis/0.87/reports/execution_layer/20260220_113409
Buckets: stop_distance quantiles, ATR_1h quantiles, trend proxy, ATR percentile regime
Recommendations: vol_regime=3, trend=0, stop_distance_min=0
Recommendation table: diagnose_1h_bleed_recommendations.csv
Gate selection policy: pick only 1-2 gates with consistent underperformance evidence
Pass/Fail: informational phase (diagnostic)
Next: enable selected 1h gates (off by default) and rerun walkforward (Phase 4)
Constraint: no 3m alpha logic added; execution layer remains execution-only
