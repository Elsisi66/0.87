# Analysis 0.87

`/root/analysis/0.87` is the active research-and-runtime workspace for the repaired Model A branch:

- 1h strict no-lookahead signal logic
- 3m execution-layer forensics and posture locking
- SOL-only paper runtime deployment path

This folder is not clean-room code. It is a working lab with research artifacts, runtime state, and frozen decision packs.

## Current Snapshot (as of 2026-03-08)

- Active repaired 3m posture:
  - `ACTIVE`: `SOLUSDT` (`M1_ENTRY_ONLY_PASSIVE_BASELINE`)
  - `SHADOW`: `NEARUSDT`
  - `BLOCKED`: `LTCUSDT`
  - Source: `reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126/repaired_3m_posture_report.md`
- Latest SOL paper reset/export run: `2026-03-08T01:46:06Z`
  - `rows_total=1432`, `expectancy_per_signal=0.0019065093`, `max_drawdown=-0.1301395943`
  - Source: `reports/execution_layer/SOL_PAPER_RESET_AND_HIST_EXPORT_20260308_014054/sol_paper_reset_report.md`

## How The Strategy Works

Core strategy contract is implemented in `src/bot087/optim/ga.py`.

### 1h Signal Contract (strict no-lookahead)

- Regime labels are computed per bar, then shifted by one bar (`cycle_shift=1`) so action at bar `t` uses information from `t-1`.
- Entry signal (`build_entry_signal`) uses shifted/previous-bar features:
  - RSI band filter
  - Williams %R cycle thresholds
  - trend alignment (EMA, ADX, DI, EMA slope)
  - optional cycle whitelist and hour mask
- Decision is made for bar `t`, and entry executes at bar `t` open.

### Position Sizing and Risk

- Position sizing is ATR/risk-budget constrained via `_position_size(...)`:
  - risk-per-trade budget
  - max allocation cap
  - ATR scaling (`atr_k`)
- Costs are applied with fee/slippage adjustments (`_apply_cost`).

### Exit Ordering (deterministic)

In both backtest and paper simulator path:

1. SL/TP high-low checks
2. max-hold check
3. RSI exit check

When both TP and SL are touched in the same bar, `SL` wins (deterministic tie policy).

### Paper Runtime Contract

`paper_trading/app` runs the strategy in paper mode with:

- closed-bar processing
- forward-only anchor (`start_from_bar_ts`)
- idempotent `processed_bars.json` state progression
- reset, journaling, reconciliation, and health tracking

Key launch scripts:

- `paper_trading/scripts/run_paper_daemon.sh`
- `paper_trading/scripts/hourly_signal_scan.sh`
- `paper_trading/scripts/smoke_test.py`

## Repository Layout

- `src/bot087/`: strategy, optimization, execution evaluation, datafeed
- `scripts/`: research runners, phase pipelines, diagnostics, sweeps
- `paper_trading/`: daemon app, configs, runtime state, operational reports
- `reports/execution_layer/`: decision packs, forensics, reconciliations, posture freezes
- `artifacts/`: historical GA/execution artifacts
- `data/`: processed market datasets and caches
- `tests/`: unit tests (currently minimal)

## How To Run

This repo currently has no pinned dependency manifest (`requirements.txt`, `pyproject.toml`, etc.), so the expected environment is the existing local venv.

### Common Commands

```bash
cd /root/analysis/0.87
export PYTHONPATH=/root/analysis/0.87
```

Run one paper cycle (no startup reset):

```bash
/root/analysis/0.87/.venv/bin/python -m paper_trading.app.main --once --max-cycles 1 --no-startup-reset
```

Run smoke test:

```bash
BINANCE_MODE=local_only PYTHONPATH=/root/analysis/0.87 /root/analysis/0.87/.venv/bin/python /root/analysis/0.87/paper_trading/scripts/smoke_test.py
```

Start daemon mode:

```bash
bash /root/analysis/0.87/paper_trading/scripts/run_paper_daemon.sh
```

Execution gate check (example):

```bash
/root/analysis/0.87/.venv/bin/python -m src.bot087.cli.exec_gate --symbol SOLUSDT --trades-csv <path_to_trades_csv>
```

## Tests And Validation History

This section summarizes executed tests/checks captured in prior run reports.

### Local Test Attempt In This Session

- Attempted: `/root/analysis/0.87/.venv/bin/python -m pytest -q`
- Result: not runnable in current environment (`No module named pytest`)

### Paper Runtime Validation (P/T phases)

1. `P2` parity check (2026-02-27): `overall_parity_ok=true`, structural mismatches zero on sampled symbols.
   - `paper_trading/reports/paper_phaseP2_parity_check_report.md`
2. `P3` recovery controls verified (retry/backoff, circuit breaker, quarantine, dead-letter).
   - `paper_trading/reports/paper_phaseP3_recovery_report.md`
3. `P4` smoke/idempotency: no backlog replay; restart idempotent.
   - `paper_trading/reports/paper_phaseP4_smoke_report.md`
4. `P5` readiness checklist: all items checked, including cron hourly scan install.
   - `paper_trading/reports/paper_phaseP5_readiness_checklist.md`
5. `T1` repair: forward-only anchoring + backlog replay disable confirmed.
   - `paper_trading/reports/paper_phaseT1_repair_report.md`
6. `T3` parity semantics reconfirmed (signal timing + SL-first tie behavior).
   - `paper_trading/reports/paper_phaseT3_parity_report.md`
7. `T4` repaired smoke: zero backlog fills and restart duplication.
   - `paper_trading/reports/paper_phaseT4_smoke_report.md`
8. `T5` post-reset status: 320 EUR reset, forward-only mode active.
   - `paper_trading/reports/paper_phaseT5_post_reset_status.md`
9. Launch validation: hourly scan script manual run completed.
   - `paper_trading/reports/paper_phaseP5_launch_report.md`

### Execution/Research Validation

1. `phase_result*.md` audit under `reports/execution_layer`:
   - total files: `37`
   - PASS-labeled: `13`
   - FAIL-labeled: `4`
   - COMPLETED-labeled: `7`
2. v2 final artifact freeze gate: `PASS` (snapshot complete/reproducible).
   - `reports/execution_layer/v2_final/phase_result.md`
3. Aggregate walkforward test summary (SOL/AVAX/NEAR):
   - overall test entry rate: `0.940948`
   - baseline net sum: `-0.704024`
   - exec net sum: `-0.931472` (still net negative)
   - `reports/execution_layer/AGG_exec_report.md`
4. ICT gate aggregate: filtered losses but net worse than baseline.
   - `reports/execution_layer/AGG_exec_ict_report.md`
5. 1h multi-coin GA pipeline (2026-02-09):
   - PASS: `ADAUSDT`, `AVAXUSDT`, `SOLUSDT`
   - FAIL: `BNBUSDT`, `ETHUSDT`, `XRPUSDT`
   - `artifacts/ga/pipeline_report_20260209_033952.md`
6. Repaired 1h universe freeze finalized: 15 selected symbols.
   - `reports/execution_layer/REPAIRED_1H_UNIVERSE_FREEZE_20260304_000207/repaired_universe_freeze_report.md`
7. SOL baseline trade forensics: lineage and chronology checks all clean, `overall_status=PASS`.
   - `reports/execution_layer/SOL_BASELINE_TRADE_FORENSICS_20260307_021857/sol_baseline_trade_forensics_report.md`
8. Entry mechanics sweep decision: `NO_CHANGE_RECOMMENDED`.
   - `reports/execution_layer/SOL_ENTRY_MECHANICS_SWEEP_20260306_235931/sol_entry_mechanics_report.md`
9. Cycle-state gating sweep decision: `CYCLE_GATING_TESTED_NO_REPAIR_APPROVED`.
   - `reports/execution_layer/SOL_CYCLE_STATE_GATING_SWEEP_20260307_025854/sol_cycle_gating_report.md`
10. Continuous cycle-quality sweep decision: `CYCLE_QUALITY_BRANCH_TESTED_NO_REPAIR_APPROVED`.
    - `reports/execution_layer/SOL_CYCLE_QUALITY_CONTINUOUS_SWEEP_20260307_140113/sol_cycle_quality_report.md`

### Data Quality and Reconciliation Tests

1. LTC partial-slice reconciliation changed `12 -> 9` but remained blocked.
   - `reports/execution_layer/LTC_3M_RECONCILIATION_AUDIT_20260308_013048/ltc_reconciliation_report.md`
2. LTC raw-trade reconstruction blocker run: still blocked (`9 -> 9` partial slices).
   - `reports/execution_layer/LTC_3M_RAWTRADE_RECON_BLOCKERS_20260308_014641/ltc_rawtrade_recon_report.md`
3. SOL paper reset + historical export completed all status flags (`1/1/1`).
   - `reports/execution_layer/SOL_PAPER_RESET_AND_HIST_EXPORT_20260308_014054/sol_paper_reset_report.md`

## Problems Faced (Observed) And Outcomes

1. Same-bar stop-loss pathology in historical paper runs.
   - Historical diagnosis showed `107/107` closes were instant SL in one sampled window.
   - `reports/execution_layer/HISTORICAL_PAPER_INSTANT_EXIT_DIAG_20260307_002253/historical_paper_instant_exit_report.md`
2. Mixed runtime-contract issues in generic daemon path.
   - Classified `MIXED_MULTIPLE_ISSUES`; includes stop-basis mismatch and runtime-path deviation from strict Model A semantics.
   - `paper_trading/reports/root_cause_classification.md`
3. External API instability and degraded operation.
   - DNS/API outages triggered retries/failures and degraded mode behavior.
   - `paper_trading/reports/paper_phaseP3_recovery_report.md`
   - `paper_trading/reports/daily_summary_20260307.md`
4. Execution overlays improved some micro metrics but remained net-negative on held-out aggregate tests.
   - `reports/execution_layer/AGG_exec_report.md`
5. LTC data lineage remains unresolved for full active promotion.
   - blocker persists after targeted backfill/reconstruction attempts.

## Important Notes

- This branch contains large generated artifacts and runtime state; do not assume a clean git tree.
- `paper_trading/config/.env` may contain secrets; use `.env.example` as template.
- `0.88` and `0.93` are curated snapshots derived from this branch:
  - `../0.88/README.md`
  - `../0.93/README.md`
