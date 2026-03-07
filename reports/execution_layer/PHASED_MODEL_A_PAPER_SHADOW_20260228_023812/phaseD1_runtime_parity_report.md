# Phase D1 Runtime Parity Report

- Generated UTC: `2026-02-28T02:38:12.986259+00:00`
- Frozen contract lock pass: `1`
- Phase C paper authorization present: `1`
- Repaired route support-feasible metadata preserved: `0`
- Current paper runtime Model A pure: `0`

## Verified Model A Source Of Truth
- Phase C contract file: `/root/analysis/0.87/reports/execution_layer/PHASEC_MODEL_A_BOUNDED_CONFIRMATION_20260228_022501/phaseC1_contract_validation.json`
- Phase A wrapper source: `/root/analysis/0.87/scripts/phase_a_model_a_audit.py`
- Validated wrapper semantics:
  - 1h signal generation remains frozen.
  - 1h TP/SL/exit semantics remain frozen after entry fill.
  - 3m controls entry timing only via `simulate_entry_only_fill(...)`.

## Current Paper Runtime Mismatch
- `paper_trading/app/data_feed.py:59` fetches only `interval="1h"`; no `fetch_ohlcv_3m(...)` path exists.
- `paper_trading/app/signal_runner.py:86` builds signals from 1h only and exports 1h rows directly into execution.
- `paper_trading/app/main.py:432` wires the daemon to the generic `ExecutionSimulator`.
- `paper_trading/app/execution_sim.py:92` stores TP/SL on entry from params.
- `paper_trading/app/execution_sim.py:136` and `paper_trading/app/execution_sim.py:158` keep downstream exit ownership inside the runtime.
- No Model A entry-only knob wiring (`entry_mode`, `limit_offset_bps`, `fallback_to_market`, `fallback_delay_min`, `max_fill_delay_min`) exists anywhere in `paper_trading/app` or `paper_trading/config`.

## Conclusion
- The backtest contract is valid, but the current paper runtime does not implement the validated Model A wrapper.
- Starting paper/shadow on the existing daemon would not match the confirmed Phase C contract.
