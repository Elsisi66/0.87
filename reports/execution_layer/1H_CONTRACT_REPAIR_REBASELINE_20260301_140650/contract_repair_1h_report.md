# 1H Contract Repair Report

- Generated UTC: `2026-03-01T14:06:50.058333+00:00`
- Signal source root: `/root/analysis/0.87/reports/execution_layer/MULTICOIN_MODELA_AUDIT_20260228_180250`
- Universe source: `/root/analysis/0.87/paper_trading/config/resolved_universe.json`

## H0 Frozen Broken Behavior
- Pre-patch 1h reference entry fill was created at the signal bar open inside `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference` (`entry_price = open_np[idx]`).
- Pre-patch same-parent-bar exit evaluation began immediately from the same bar by passing `entry_idx=int(idx)` into `scripts/execution_layer_3m_ict.py::_simulate_path_long`, whose scan loop runs `for i in range(entry_idx, ...)`.
- Pre-patch stop/target construction used raw entry open in `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference` (`sl = entry_price * sl_mult`, `tp = entry_price * tp_mult`).
- Pre-patch hold duration collapsed to zero whenever the same parent bar hit the stop because `hold_minutes = (exit_time - entry_time)` and both timestamps were the same bar.

## H1 Repaired Contract
- Patched `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference` to preserve the same entry event but start exit evaluation from the first full subsequent 1h bar (`idx + 1`) by default.
- Added explicit export fields: `entry_parent_bar_time`, `exit_eval_start_time`, `exit_eval_bar_time`, and `defer_exit_to_next_bar`.
- Signal generation input stayed unchanged; only the 1h exit chronology was rebased.

## H2/H4 Rebaseline Scope
- Symbols processed: `SOLUSDT, NEARUSDT, AVAXUSDT, ADAUSDT, AXSUSDT, BCHUSDT, CRVUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, TRXUSDT, XRPUSDT, ZECUSDT`
- Priority symbols included: `SOLUSDT`, `NEARUSDT`, `AVAXUSDT`

## H3 Signal Layer Reassessment
- Classification: `1H_BASELINE_SURVIVES_REPAIR`
- Signal layer still stands: `1`
- Rationale: `target trio remains profitable after chronology repair`

## Target Symbol Snapshot
- SOLUSDT: expectancy `0.00319752`, cvar_5 `-0.00224775`, maxdd `-0.40084286`, same_bar_rate `0.0000%`, zero_hold_rate `0.0000%`
- NEARUSDT: expectancy `0.00351154`, cvar_5 `-0.00219952`, maxdd `-0.17706259`, same_bar_rate `0.0000%`, zero_hold_rate `0.0000%`
- AVAXUSDT: expectancy `0.00334589`, cvar_5 `-0.00219952`, maxdd `-0.16986438`, same_bar_rate `0.0000%`, zero_hold_rate `0.0000%`
