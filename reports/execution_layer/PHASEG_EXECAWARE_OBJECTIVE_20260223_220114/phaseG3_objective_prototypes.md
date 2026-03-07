# Phase G3 Objective Prototypes

All objectives are execution-aware and use E1/E2 candidate metrics with robustness proxies.

- `OJ1 = delta_expectancy_vs_exec_baseline - 0.0003*avg_toxic_proxy - 0.0002*avg_cluster_proxy`
- `OJ2 = delta_expectancy_vs_exec_baseline + 0.40*cvar_improve_ratio + 0.30*maxdd_improve_ratio - 0.0002*avg_tail_proxy`
- `OJ3 = OJ2 - 0.08*max(0, dominant_regime_share-0.55)` (anti-concentration penalty)
- `OJ4 = OJ2 + 0.20*min_split_expectancy_net` (robust min-split emphasis)
- `OJ5 = OJ3 - 0.05*max(0, 0.85-entry_rate)` (participation floor penalty)

Scoring/ranking is prototype-only in this phase; no GA search was launched.
