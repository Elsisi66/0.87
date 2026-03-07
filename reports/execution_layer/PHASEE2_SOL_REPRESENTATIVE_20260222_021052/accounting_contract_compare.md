# Accounting Contract Compare

| dimension | fullscan_contract | phasec_frozen_contract | phasee2_canonical_contract |
| --- | --- | --- | --- |
| initial_equity | 10000.000000 | 1.000000 | 1.000000 |
| position_sizing | native 1h backtester sizing | fixed_fractional_risk_per_trade_compounding | fixed_fractional_risk_per_trade_compounding |
| fee_model | scan fee/slip params | phaseA fee model sha=b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a | phaseA fee model sha=b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a |
| metrics_formula | scan summary formulas | phaseA metrics sha=d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99 | phaseA metrics sha=d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99 |
| signal_universe | fullscan endogenous | frozen phasec test subset (600) | deterministic representative subset from source |
| execution_horizon_hours | native 1h backtester | 12.000000 | 12.000000 |

- Canonical contract is hard-locked for all Phase E2 variants.
