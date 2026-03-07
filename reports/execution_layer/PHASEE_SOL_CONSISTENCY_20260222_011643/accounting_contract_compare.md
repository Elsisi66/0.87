# Accounting Contract Compare

| dimension | full_1h_path | frozen_phasec_path | mismatch |
| --- | --- | --- | --- |
| initial_equity | 10000.000000 | 1.000000 | 1 |
| position_sizing | native ATR-based sizing | fixed fractional risk per trade | 1 |
| compounding | yes | yes | 0 |
| fee_model | fee_bps=7.0 | phase_a_sha=b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a | 1 |
| slippage_model | slip_bps=2.0 | phase_a maker/taker slips | 1 |
| signal_universe | endogenous full-period | frozen exported test subset | 1 |
| entry_semantics | 1h backtester internal | next 3m open after signal | 1 |
| exit_semantics | 1h rules in ga.py | 3m path with fixed Phase C exit | 1 |
| bar_ambiguity_handling | 1h-candle internal behavior | 3m sequential path | 1 |
| drawdown_sign_convention | positive % in scan outputs | negative fraction in exec reports | 1 |
| expectancy_denominator | per-trade return | per-trade net return (+per-signal reported) | 0 |
