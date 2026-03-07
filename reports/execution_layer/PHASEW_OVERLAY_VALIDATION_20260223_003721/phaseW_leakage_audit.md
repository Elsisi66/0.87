# Phase W Leakage Audit

- Generated UTC: 2026-02-23T00:38:22.129260+00:00
- Session overlays are clock-based and ex-ante definable from timestamp only.
- `session_veto_06_11` uses fixed clock bucket and does not depend on future PnL.
- `worst_session` style selection is train-derived; treated as in-sample learned rule and must pass OOS checks.
- Loss-control overlays (`daily_loss_cap`, `session_killswitch`, `loss_cluster_pause`) are path-dependent policy proxies in this harness.
- Proxy overlays are explicitly labeled `approximate_counterfactual=1` and are not treated as exact engine-integrated signals.
- W1 reproduction pass: `1`
