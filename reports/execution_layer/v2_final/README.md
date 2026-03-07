# Execution Layer v2 Final Snapshot

This folder is a frozen snapshot of the latest walk-forward test-only evaluation artifacts used as the starting point for the next improvement cycle.

Included run folders (copied with original timestamp names):
- `20260220_071048_walkforward_SOLUSDT`
- `20260220_074353_walkforward_AVAXUSDT`
- `20260220_074718_walkforward_NEARUSDT`

Included aggregate files:
- `AGG_exec_testonly_summary.csv`
- `AGG_exec_report.md`
- `AGG_exec_included_files.txt`

Notes:
- These files capture the state where exec_limit improved execution quality versus baseline but remained net-negative overall on test.
- Subsequent phases (cost sensitivity, per-symbol configs, 1h bleed diagnosis, and 1h risk gates) are evaluated from this baseline snapshot.

## Tight-Mode Extension Snapshot (20260220_123119)

Added after Phase A–D (risk rollup + tight constraints retune + rubric aggregation):
- `20260220_122741_walkforward_SOLUSDT`
- `20260220_122941_walkforward_AVAXUSDT`
- `20260220_123057_walkforward_NEARUSDT`
- `20260220_123119/` (risk rollups + tight aggregate outputs)

Key aggregate outputs in `20260220_123119/`:
- `AGG_exec_testonly_summary_tight.csv`
- `AGG_exec_testonly_summary_tight.md`
- `AGG_risk_rollup_tight_overall.csv`
- `AGG_risk_rollup_tight.md`
- `risk_rollup_by_symbol.csv`
- `risk_rollup_overall.csv`

Update:
- Recomputed Phase A risk overall using weighted aggregation by signal count in `20260220_123314/`.
- Use `20260220_123314/` as the authoritative tight-mode bundle.
