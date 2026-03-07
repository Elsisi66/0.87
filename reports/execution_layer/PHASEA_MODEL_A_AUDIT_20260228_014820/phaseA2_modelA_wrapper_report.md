# A2 Model A Wrapper Report

- Generated UTC: 2026-02-28T01:48:55.974746+00:00
- Core evaluator was not patched; a wrapper was added to isolate Model A without changing the existing hybrid engine.
- Wrapper entry stage:
  - uses the frozen 3m signal slices already prepared by `ga_exec_3m_opt`
  - supports only `entry_mode`, `limit_offset_bps`, `fallback_to_market`, `fallback_delay_min`, and `max_fill_delay_min`
- Wrapper exit stage:
  - bypasses `ga_exec_3m_opt._simulate_dynamic_exit_long` entirely
  - calls 1h-only exit simulation derived from `backtest_exec_phasec_sol._simulate_1h_reference` semantics
  - does not expose `tp_mult`, `sl_mult`, `break_even_*`, `trailing_*`, `partial_take_*`, or `time_stop_min` as candidate knobs

- Hybrid evaluator mixes exits: `1`
- Wrapper preserves 1h exit ownership: `1`
