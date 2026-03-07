# Phase J01 Future Robustness Constraints Draft

- Generated UTC: 2026-02-24T02:52:00.552546+00:00
- Source: Phase I failure-forensics (lucky-point frontier collapse).

## Do-Not-Chase Signatures

1. High OJ2 + high delta candidates with route_pass_rate < 1.0.
2. Candidates with min_subperiod_delta <= 0 and/or min_subperiod_cvar < 0 despite positive aggregate metrics.
3. Candidates with bootstrap_pass_rate < 0.10 (weak perturbation confidence).
4. Candidate families with negative cvar_improve_ratio under route2/stress scenarios.
5. High-correlation duplicate clusters with near-identical metric shape (effective_trials collapse).

## Draft Robustness-First Constraints/Objectives

- Add route fragility penalty: penalty if route_pass_rate < 1.0.
- Add split stability floor: require min_subperiod_delta > 0 and min_subperiod_cvar >= 0.
- Add perturb confidence floor: bootstrap_pass_rate target >= 0.10 (pilot stage).
- Add duplicate-collapse penalty tied to correlation-adjusted effective trials.
- Downweight high in-sample gains when accompanied by negative tail-risk signs.
