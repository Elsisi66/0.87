# Analysis 0.87

This repository is the working log of a fast, messy, but disciplined three-week transition:

- from a **1h signal-first strategy branch**
- into **3m execution realism and route checks**
- and finally into a **SOL-focused cycle lab + paper-runtime hardening track**

If you want the short version: we proved the core strategy contract, found where execution changed the economics, narrowed scope to SOL for cleaner control, and froze posture while we keep repairing data and runtime edges.

## Current State (March 8, 2026)

- Active 3m posture source of truth:
  - `ACTIVE`: `SOLUSDT` (`M1_ENTRY_ONLY_PASSIVE_BASELINE`)
  - `SHADOW`: `NEARUSDT`
  - `BLOCKED`: `LTCUSDT`
  - See: `reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126/repaired_3m_posture_report.md`
- Latest SOL paper reset and historical export completed on `2026-03-08`.
  - Exported trade rows: `1432`
  - Expectancy per signal: `0.0019065093`
  - Max drawdown: `-0.1301395943`
  - See: `reports/execution_layer/SOL_PAPER_RESET_AND_HIST_EXPORT_20260308_014054/sol_paper_reset_report.md`

---

## Last 3 Weeks: What Happened

### 1) Feb 20-22, 2026: We stress-tested the 1h branch against execution reality

We started from the 1h model and ran walkforward and execution overlays to answer one question: does the edge survive realistic execution behavior?

What we learned:

- entry quality and route mechanics could improve in places,
- but aggregate test economics were still net-negative in key execution summaries,
- so “strategy logic only” was not enough; execution and data contracts needed tighter control.

Useful references:

- `reports/execution_layer/AGG_exec_report.md`
- `reports/execution_layer/AGG_exec_ict_report.md`
- `reports/execution_layer/v2_final/README.md`

### 2) Feb 23-28, 2026: We hardened parity, runtime behavior, and operational safety

We then shifted from pure research to contract hardening:

- parity checks between backtest semantics and paper path,
- no-lookahead timing confirmation,
- forward-only processing and idempotent restart behavior,
- recovery controls (retry/backoff/circuit/quarantine/dead-letter),
- hourly daemon path with paper-mode safety boundaries.

This period was about turning “it backtests” into “it can run repeatedly without hidden replay or ordering bugs.”

Useful references:

- `paper_trading/reports/paper_phaseP2_parity_check_report.md`
- `paper_trading/reports/paper_phaseP3_recovery_report.md`
- `paper_trading/reports/paper_phaseP4_smoke_report.md`
- `paper_trading/reports/paper_phaseP5_launch_report.md`
- `paper_trading/reports/paper_phaseT1_repair_report.md`
- `paper_trading/reports/paper_phaseT3_parity_report.md`
- `paper_trading/reports/paper_phaseT4_smoke_report.md`

### 3) Mar 1-4, 2026: We rebuilt and froze the repaired 1h universe for downstream 3m work

After repair loops and rebaselining, we produced a repaired universe freeze that became the official base for execution-layer continuation.

- repaired 1h universe selected symbols: `15`
- this freeze replaced the older long-set dependency for repaired-branch work
- strict subset confirmation and active-set checks were used before promotion decisions

Useful references:

- `reports/execution_layer/REPAIRED_1H_UNIVERSE_FREEZE_20260304_000207/repaired_universe_freeze_report.md`
- `reports/execution_layer/REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_20260304_010143/`
- `reports/execution_layer/REPAIRED_UNIVERSE_3M_ACTIVESET_OPCHECK_20260304_012119/`

### 4) Mar 5-8, 2026: SOL-only focus, cycle lab sweeps, and paper/export closure

By this stage, the branch became deliberately narrow:

- SOL got active posture,
- NEAR stayed shadow,
- LTC remained blocked due to persistent data-quality issues.

We ran bounded cycle and entry-mechanics studies (not open-ended search), then kept only what passed robustness gates.

Key outcome: several cycle/entry branches were tested, but no robust upgrade was approved yet. Baseline stayed in place while evidence quality improved.

Useful references:

- `reports/execution_layer/SOL_ENTRY_MECHANICS_SWEEP_20260306_235931/sol_entry_mechanics_report.md`
- `reports/execution_layer/SOL_CYCLE_STATE_GATING_SWEEP_20260307_025854/sol_cycle_gating_report.md`
- `reports/execution_layer/SOL_CYCLE_QUALITY_CONTINUOUS_SWEEP_20260307_140113/sol_cycle_quality_report.md`
- `reports/execution_layer/SOL_BASELINE_TRADE_FORENSICS_20260307_021857/sol_baseline_trade_forensics_report.md`
- `reports/execution_layer/LTC_3M_RECONCILIATION_AUDIT_20260308_013048/ltc_reconciliation_report.md`
- `reports/execution_layer/LTC_3M_RAWTRADE_RECON_BLOCKERS_20260308_014641/ltc_rawtrade_recon_report.md`

---

## The Technical Arc: 1h -> 3m -> Cycle Lab

### Step A: 1h signal contract (where it started)

Core signal logic is in `src/bot087/optim/ga.py`:

- strict no-lookahead structure (`cycle_shift=1`, shifted features)
- deterministic entry/exit ordering
- ATR/risk-based position sizing

### Step B: 3m execution realism layer

We replayed signal outcomes through execution-aware paths to inspect:

- fill quality,
- taker/maker behavior,
- delay/fallback effects,
- edge decay under more realistic assumptions.

### Step C: cycle labeling and gated branch tests

Instead of broad random search, we did bounded cycle-state and cycle-quality branches with explicit causal constraints and holdout/route checks.

Result so far: tested multiple variants, no robust promotion yet.

### Step D: paper runtime with guardrails

Paper runtime (`paper_trading/app`) now emphasizes:

- forward-only operation,
- no backlog replay on reset,
- idempotent restart,
- explicit operational safety in paper mode.

---

## What Worked

- Contract clarity improved substantially: timing/parity assumptions are now explicit and tested.
- SOL baseline lineage and chronology checks came back clean in forensics runs.
- Runtime safety and repeatability (restart/recovery behavior) improved versus earlier state.

## What Did Not Work (Yet)

- Execution overlays did not consistently turn test aggregate economics positive.
- Some cycle and entry-mechanics variants looked interesting locally but failed robust promotion criteria.
- LTC data-path repair remains incomplete, so active promotion is still blocked.

## Known Open Issues

- Historical same-bar stop-loss pathologies were observed in older paper windows and required repair work.
  - `reports/execution_layer/HISTORICAL_PAPER_INSTANT_EXIT_DIAG_20260307_002253/historical_paper_instant_exit_report.md`
  - `paper_trading/reports/root_cause_classification.md`
- API/network degradations can still force degraded-mode behavior in some environments.

---

## Repo Map

- `src/bot087/`: strategy, optimization, execution evaluation
- `scripts/`: phase runners, sweeps, forensics, audits
- `paper_trading/`: runtime app, configs, state, reports
- `reports/execution_layer/`: phase outputs, freezes, decisions, diagnostics
- `artifacts/`: historical optimization and execution artifacts
- `data/`: processed datasets and caches

---

## Practical Runbook

```bash
cd /root/analysis/0.87
export PYTHONPATH=/root/analysis/0.87
```

Run one paper cycle:

```bash
/root/analysis/0.87/.venv/bin/python -m paper_trading.app.main --once --max-cycles 1 --no-startup-reset
```

Run smoke test:

```bash
BINANCE_MODE=local_only PYTHONPATH=/root/analysis/0.87 /root/analysis/0.87/.venv/bin/python /root/analysis/0.87/paper_trading/scripts/smoke_test.py
```

Start paper daemon script:

```bash
bash /root/analysis/0.87/paper_trading/scripts/run_paper_daemon.sh
```

Execution gate check (example):

```bash
/root/analysis/0.87/.venv/bin/python -m src.bot087.cli.exec_gate --symbol SOLUSDT --trades-csv <path_to_trades_csv>
```

---

## If You Are New Here

Read these in order:

1. `reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126/repaired_3m_posture_report.md`
2. `reports/execution_layer/SOL_BASELINE_TRADE_FORENSICS_20260307_021857/sol_baseline_trade_forensics_report.md`
3. `reports/execution_layer/SOL_CYCLE_QUALITY_CONTINUOUS_SWEEP_20260307_140113/sol_cycle_quality_report.md`
4. `reports/execution_layer/SOL_PAPER_RESET_AND_HIST_EXPORT_20260308_014054/sol_paper_reset_report.md`

That sequence gives you the branch posture, the baseline truth, the current cycle-lab status, and the latest operational export state.
