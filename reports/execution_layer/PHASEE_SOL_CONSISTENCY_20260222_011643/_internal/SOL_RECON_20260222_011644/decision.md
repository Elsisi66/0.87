# SOL Reconciliation Decision

- Generated UTC: 2026-02-22T01:16:49.330015+00:00

## Status Flags

- BUG: NO
- UNIVERSE MISMATCH: YES
- SIZING/COMPOUNDING MISMATCH: YES
- SOL deployable candidate status: HOLD

## Why

- V1 fullscan native return is 678.304825 with native contract, but frozen universe variants stay near -100% return.
- This is primarily explained by mismatch in evaluation universe and execution/sizing contract, not by a single arithmetic bug.
- On frozen universe, Phase C best beats Phase C control on expectancy by 0.000084 but remains negative in absolute terms.

## Final Recommendation

- HOLD.
- Next step: run a fair expanded SOL evaluation where fullscan and frozen pipelines share identical universe/sizing contract before additional optimization.
