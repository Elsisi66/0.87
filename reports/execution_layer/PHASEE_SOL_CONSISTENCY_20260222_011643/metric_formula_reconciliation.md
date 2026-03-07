# Metric Formula Reconciliation

- `expectancy_net` = mean net return per valid trade (`pnl_net_pct`).
- `total_return` = `final_equity / initial_equity - 1`.
- `max_drawdown_pct` in exec reports is negative peak-to-trough fraction.
- `max_dd_pct` in params-scan outputs is positive magnitude percentage.
- `cvar_5` = mean of worst 5% trade outcomes.

## Contract Source

- fee_model: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json` sha256=`b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`
- metrics_definition: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md` sha256=`d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
