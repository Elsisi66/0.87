# Phase H SOL Signal-Definition Fork Spec

- Generated UTC: 2026-02-22T14:50:18.091867+00:00
- Scope: signal-layer fork only; downstream 3m execution mechanics remain contract-locked and unchanged.

## Fork Objective

- Replace incremental tp/sl polishing with a new SOL signal-definition branch centered on entry quality and de-clustering.
- Enforce fixed-size/capped-risk practical viability before any compounding evaluation.

## Frozen Invariants

- representative_subset_sha256: `fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d`
- fee_model_sha256: `b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a`
- metrics_definition_sha256: `d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99`
- selected_model_set_sha256: `4a8cb243e7f7e6425db6726302d6326bf727fe026baca77980af0532543c2fc4`
- No lookahead; subset integrity and contract lock checks are mandatory.

## Fork Components (No tp/sl polishing in this branch)

1. Trend alignment gate
Require directional agreement with slow trend state for long entries.
2. Volatility regime gate
Permit entries only in pre-specified volatility buckets with minimum support.
3. De-clustering cooldown
Apply deterministic cooldown windows to suppress bursty correlated entries.
4. Delayed 1h entry modes
Evaluate delay modes {0,1,2} bars after signal under fixed-size first.

## Evaluation Protocol

1. Fixed-size / capped-risk stage (mandatory first)
Absolute gates: expectancy_net>0, total_return>0, maxDD>-0.35, cvar_5>-0.0015, PF>=1.05, support_ok=1.
2. Compounding stage (conditional)
Run only for fixed-size passers; reject on absolute practical gate failure.
3. Release decision
If no fixed-size candidate passes, output HOLD/NO_DEPLOY and stop.

## Research Controls to Add

| control | status | implementation_note |
| --- | --- | --- |
| multiple-testing accounting | design | Track raw trial count and effective independent trials per sweep; report adjusted confidence for top variants. |
| deflated-sharpe / PSR significance | design | Compute DSR/PSR on shortlist before promotion to compounding evaluation. |
| data-snooping reality-check benchmark | design | Add White-style reality-check style benchmark test against null strategy family. |
| purged+embargoed time-series validation | design | Apply purged walk-forward folds when expanding beyond current representative harness. |

## Next Exact Prompt

```text
SOL signal-definition FORK (contract-locked Phase H): create a new SOL-only signal layer branch that keeps the same representative subset/hash and downstream 3m execution contract unchanged. Implement only: (1) trend alignment gate, (2) volatility regime gate, (3) de-clustering cooldown, (4) delayed 1h entry modes {0,1,2}. Evaluate every candidate in fixed-size/capped-risk mode first; enforce absolute gates before compounding (expectancy_net>0, total_return>0, maxDD>-0.35, cvar_5>-0.0015, PF>=1.05, support_ok=1). Reintroduce compounding only for fixed-size passers. Include multiple-testing accounting, DSR/PSR significance, and a reality-check benchmark in shortlist decisions. If no fixed-size candidate passes absolute gates, return HOLD/NO_DEPLOY with root-cause evidence and stop.
```
