# Paper Runtime Event Order Trace

## F0 Source Paths
- Active paper trade journal: `/root/analysis/0.87/paper_trading/state/journal.jsonl`
- Active paper state: `/root/analysis/0.87/paper_trading/state/portfolio_state.json`
- Dedicated Model A state journal (checked for ambiguity, no fill events present): `/root/analysis/0.87/paper_trading/state/model_a_runtime/paper_primary/journal.jsonl`
- Backtest trade-level export used for comparison: `/root/analysis/0.87/reports/execution_layer/PHASEI_SOL_POSTFIX_VALIDATION_20260222_180531/config_snapshot/trades_v2r_1h_reference_control.csv`
- Frozen Model A contract doc: `/root/analysis/0.87/reports/execution_layer/PHASEA_MODEL_A_AUDIT_20260228_014944/phaseA1_frozen_exit_semantics.md`
- Frozen Model A contract code: `/root/analysis/0.87/scripts/phase_a_model_a_audit.py`

## F1 Zero-Hold Summary
- Paper runtime total closed trades: `16`
- Paper runtime `hold_hours == 0`: `16` (100.0000%)
- Paper runtime `entry_bar_time == exit_bar_time`: `16`
- Paper runtime same-bar SL exits: `16`
- Paper runtime SL closes with inferred stop above raw open: `15`
- Backtest total filled trades: `1200`
- Backtest `hold_hours == 0`: `1052` (87.6667%)
- Backtest `entry_bar_time == exit_bar_time`: `1052`
- Backtest same-bar SL exits: `1052`

## F2 Event Order In The Active Paper Runtime
- `paper_trading/app/execution_sim.py:59-124` opens a position inside the current 1h bar when `SIGNAL` is true. The entry price is cost-adjusted immediately via `_apply_cost(bar_open, fee, slip, "buy")`.
- `paper_trading/app/execution_sim.py:126-150` then immediately re-reads the newly opened position in the same `process_bar(...)` call and checks the same bar low/high (`bar_low`, `bar_high`) against stop/target.
- `paper_trading/app/execution_sim.py:135` computes `hold = bar_index - entry_bar_index`; for a newly opened trade this is `0`, so a same-pass close is logged as `hold_hours=0`.
- `paper_trading/app/execution_sim.py:193-203` writes the close event with the same parent `bar_ts` used for the open event and copies `entry_ts` from the position, so entry/exit share the same candle timestamp when closed in the same pass.
- `paper_trading/app/reconciler.py:104-109` adds separate `event_recorded_ts` values when journaling, so the journal is not collapsing two distinct events into one timestamp; the same `bar_ts` comes from the runtime logic, not the logger.

## F3 Price Basis Findings
- Entry price in paper includes fee and slippage: `buy_px_quote = _apply_cost(bar_open, fee_bps, slip_bps, "buy")` at `paper_trading/app/execution_sim.py:62-64`.
- Stop and target are computed from that cost-inflated entry: `sl_px = entry_px_quote * sl_mult`, `tp_px = entry_px_quote * tp_mult` at `paper_trading/app/execution_sim.py:137-138`.
- Exit price also includes fee and slippage on the sell leg: `sell_px_quote = _apply_cost(exit_raw_quote, fee_bps, slip_bps, "sell")` at `paper_trading/app/execution_sim.py:165-167`.
- Fees/slippage are tracked separately in portfolio state (`fees_paid_eur`, `slippage_paid_eur`) at both entry and exit, but they are also embedded in the entry/exit executed prices. This means the stop trigger basis and PnL basis are not the same as the raw 1h reference export.
- In the active paper sample, `15 / 16` same-bar SL exits had `exit_raw_quote > entry_raw_open_quote`, which means the raw stop sat above the raw open. That makes an immediate same-bar stop mechanically likely on any bar that dips below the open.
- In the 1h reference backtest export, stop is anchored from raw entry (`entry_price = open_np[idx]`, then `sl = entry_price * sl_mult`) at `scripts/backtest_exec_phasec_sol.py:311-313`; costs are applied later in PnL via `_cost_row(...)` at `scripts/backtest_exec_phasec_sol.py:330-331`.

## F4 Model A Contract Comparison
- The frozen Model A contract explicitly says the first 1h exit-evaluation candle is the first full 1h bar strictly greater than the realized fill time (/root/analysis/0.87/reports/execution_layer/PHASEA_MODEL_A_AUDIT_20260228_014944/phaseA1_frozen_exit_semantics.md:5-10).
- The corresponding code enforces that with `np.searchsorted(..., side="right")` at `scripts/phase_a_model_a_audit.py:423`, so exit checks start after the fill hour, not on the same parent 1h candle.
- The currently observed trades are therefore not coming from the dedicated Model A runtime path; they are coming from the generic paper daemon in `paper_trading/state/journal.jsonl`.
