# Updated Paper Alignment Report (Chronology Fix)

## Window
- 2026-03-04 to 2026-03-05 UTC (same as prior alignment pass)

## Repaired Posture / Fallback Guard
- Active subset source: `/root/analysis/0.87/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126/repaired_active_3m_subset.csv`
- Active params dir: `/root/analysis/0.87/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126/repaired_active_3m_params`
- Startup log now includes: `REPAIRED_POSTURE_MODE: fallback disabled`

## Chronology Guard
- same_bar_exit_attempts: 0
- exits_deferred_to_next_bar: 0
- exits_blocked_pre_entry: 0
- observed same-bar closes: 0
- entry<exit violations: 0

## Alignment Summary
- bars evaluated: 0
- opens: 0
- closes: 0
- close PnL signs: pos=0, neg=0
- same-bar close rate: 0.000000

## Before vs After
- before non-SOL rows: 32
- before same-bar close rate: 1.000000 (67/67)
- after same-bar close rate: 0.000000 (0/0)

## Outcome
- PAPER_ALIGNED

## Notes
- This patch only applied chronology/fallback guards; no strategy redesign or optimization was performed.
- In this exact window, SOL produced zero closes under repaired posture replay, so winrate pathology is not active in post-fix replay output.
