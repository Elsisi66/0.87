# Phase4 Test-Only Rollup

- Scope: latest gated walkforward runs for SOLUSDT, AVAXUSDT, NEARUSDT
- Gate: use_vol_regime_gate=1, vol_regime_max_percentile=90

## Per Symbol

- SOLUSDT: entry_rate=0.916667, expectancy_net=-0.000860, baseline_expectancy_net=-0.000643, taker_share=0.121818
- AVAXUSDT: entry_rate=0.969048, expectancy_net=-0.000604, baseline_expectancy_net=-0.000316, taker_share=0.157248
- NEARUSDT: entry_rate=0.951311, expectancy_net=-0.000837, baseline_expectancy_net=-0.000695, taker_share=0.122047

## Overall

- test_signals=1287
- entries=1211
- entry_rate=0.940948
- exec_expectancy_net=-0.000769
- baseline_expectancy_net=-0.000547
- taker_share=0.133774
- sl_hit_rate=0.642444
