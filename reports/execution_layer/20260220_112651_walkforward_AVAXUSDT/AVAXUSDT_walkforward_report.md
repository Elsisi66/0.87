# Walkforward Exec Limit Report: AVAXUSDT

- Generated UTC: 2026-02-20T11:26:51.827188+00:00
- params_file: `/root/analysis/0.87/data/metadata/params/AVAXUSDT_C13_active_params_long.json`
- execution_config: `/root/analysis/0.87/configs/execution_configs.yaml`
- symbol_config: `{"constraints": {"max_fill_delay_min": 90, "max_taker_share": 0.4, "min_entry_rate": 0.55}, "exec_mode": "exec_limit", "fallback": "market", "fallback_values": ["market"], "k_base": 0.05, "k_values": [0.0, 0.05, 0.1, 0.2], "ladder": 0, "timeout_bars": 20, "timeout_values": [10, 20, 40], "use_panic_filter": 0, "use_vol_gate": 0, "vol_gate_values": [0]}`
- total signals: 1624
- train/test: 1137/487
- train tuning CSV: `/root/analysis/0.87/reports/execution_layer/20260220_112651_walkforward_AVAXUSDT/AVAXUSDT_walkforward_train_tuning.csv`
- test signals CSV: `/root/analysis/0.87/reports/execution_layer/20260220_112651_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_signals.csv`
- test summary CSV: `/root/analysis/0.87/reports/execution_layer/20260220_112651_walkforward_AVAXUSDT/AVAXUSDT_walkforward_test_summary.csv`

## Selected Config (Train)

- k: 0.0000
- timeout_bars: 10
- fallback: market
- use_vol_gate: 0
- passes_constraints: 1

## Test Metrics

- entry_rate: 0.963039
- taker_share: 0.147122
- max_fill_delay_min: 30.00
- pnl_net_sum: -0.323921
- expectancy_net: -0.000691
- baseline_expectancy_net: -0.000516

## Top 10 Train Configs

| k | timeout_bars | fallback | use_vol_gate | entry_rate | taker_share | max_fill_delay_min | expectancy_net | pnl_net_sum | passes_constraints |
|---|---|---|---|---|---|---|---|---|---|
| 0.000000 | 10 | market | 0 | 0.917326 | 0.093960 | 60.000000 | -0.000935 | -0.975164 | 1 |
| 0.000000 | 20 | market | 0 | 0.917326 | 0.093960 | 60.000000 | -0.000935 | -0.975164 | 1 |
| 0.000000 | 40 | market | 0 | 0.917326 | 0.093960 | 60.000000 | -0.000935 | -0.975164 | 1 |
| 0.050000 | 10 | market | 0 | 0.917326 | 0.093960 | 60.000000 | -0.000935 | -0.975164 | 1 |
| 0.050000 | 20 | market | 0 | 0.917326 | 0.093960 | 60.000000 | -0.000935 | -0.975164 | 1 |
| 0.050000 | 40 | market | 0 | 0.917326 | 0.093960 | 60.000000 | -0.000935 | -0.975164 | 1 |
| 0.100000 | 10 | market | 0 | 0.917326 | 0.093960 | 60.000000 | -0.000935 | -0.975164 | 1 |
| 0.100000 | 20 | market | 0 | 0.917326 | 0.093960 | 60.000000 | -0.000935 | -0.975164 | 1 |
| 0.100000 | 40 | market | 0 | 0.917326 | 0.093960 | 60.000000 | -0.000935 | -0.975164 | 1 |
| 0.200000 | 10 | market | 0 | 0.917326 | 0.093960 | 60.000000 | -0.000935 | -0.975164 | 1 |
