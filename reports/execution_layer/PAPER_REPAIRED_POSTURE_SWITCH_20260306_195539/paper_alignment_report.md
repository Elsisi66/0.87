# Paper Repaired Posture Switch + Alignment Report

## Scope
- Historical alignment window: 2026-03-04 to 2026-03-05 UTC
- Post-switch startup sanity: latest manual run from hourly scanner

## Deployment Switch Result
- Resolved universe now: SOLUSDT only via `/root/analysis/0.87/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126/repaired_active_3m_subset.csv`
- Winner enforced: `M1_ENTRY_ONLY_PASSIVE_BASELINE`
- SOL params hash: `2ebc1caabad9e0c057bb32c2ff2562e8420b456fd97eb4ea00f1829215ea5958`
- Repaired contract startup flag: `defer_exit_to_next_bar=True` (asserted)

## Historical Mismatch Evidence (Pre-Switch Window)
- Observed signal symbols: ['ADAUSDT', 'AVAXUSDT', 'AXSUSDT', 'BCHUSDT', 'CRVUSDT', 'DOGEUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT', 'SOLUSDT', 'TRXUSDT', 'XRPUSDT', 'ZECUSDT']
- Observed trade symbols: ['AVAXUSDT', 'CRVUSDT', 'DOGEUSDT', 'LINKUSDT', 'LTCUSDT', 'NEARUSDT', 'SOLUSDT', 'XRPUSDT', 'ZECUSDT']
- Non-SOL trades in window: 32
- Same-bar exits in window: 67
- Fee unit errors: 0
- Slippage set errors: 0
- Close PnL sign counts: pos=0, neg=67
- Expected SOL signal count (reconstructed repaired config): 0
- Observed SOL open count: 35

## Post-Switch Startup Sanity
- Startup assertion banner present: True
- Startup resolved universe SOL-only: True
- Startup banner captured in `paper_startup_banner.txt`

## Decision
- Outcome: `PAPER_FIXED_BUT_MISMATCH_REMAINS`
- Root cause: Historical window used legacy 13-symbol universe and non-posture params; post-switch run is posture-locked SOL-only.

## Concrete Fix Steps If Residual Mismatch Persists
- Keep cron path fixed to posture pack env exports in `hourly_signal_scan.sh`.
- Do not permit fallback to `reports/params_scan/*/best_by_symbol.csv` for paper deployment.
- If needed, reset paper state post-switch to remove legacy-position residue from historical runs.

## Remaining Mismatch Root Cause (File/Line)
- Legacy multi-symbol source path was active pre-switch via universe resolver fallback to params scan: `paper_trading/app/universe.py:284-299`.
- Runtime now posture-locked, but historical always-lose shape also reflects same-bar close eligibility in execution path:
  - open event and immediate same-bar close can happen in same call: `paper_trading/app/execution_sim.py:126-166`
  - stop/target check uses current bar high/low right after open: `paper_trading/app/execution_sim.py:143-154`
  - hold can be `0` on close event: `paper_trading/app/execution_sim.py:135` and `paper_trading/app/execution_sim.py:203`

## Concrete Next Fix (Not Applied In This Step)
- Add a chronology guard in `ExecutionSimulator.process_bar` to forbid SL/TP/RSI exit evaluation when `bar_index == entry_bar_index`.
- Re-run paper alignment after that guard with SOL-only posture lock still enabled.
