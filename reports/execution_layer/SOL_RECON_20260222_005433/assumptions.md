# Assumptions

- V1 uses native `ga.py` long backtester behavior (same as params scan contract).
- V2 is an aligned fallback comparator for frozen Phase C test universe using exported signals and 3m path simulation.
- V3/V4 are loaded from Phase C trade diagnostics and re-costed under the same Phase A fee contract.
- All frozen Phase C hash checks are enforced before evaluation.
- No strategy logic modifications were made in this reconciliation run.
