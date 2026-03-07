# Phase E3 Contract Parity Tests

- Freeze lock pass: `1`
- Repaired routes remain support-feasible: `1`
- Uses 1h signal owner: `1`
- Uses 3m entry executor: `1`
- Exits owned by 1h only: `1`
- Forbidden exit controls active: `0`
- Primary/backup loaded: `1`
- Isolated books: `1`
- 1h feed ready: `1`
- 3m feed ready: `1`
- Candidate mapping exists: `1`
- Runtime launcher exists: `1`

## Exact Parity Proof
- The runtime validates the same fee/metrics hashes as the locked Model A research path.
- The 3m path is used only inside `_simulate_entry_fill(...)`.
- Exit handling is performed only inside `_maybe_close_position(...)` on 1h bars.
- Primary and backup each use a dedicated `StateStore`, proving isolated books.
