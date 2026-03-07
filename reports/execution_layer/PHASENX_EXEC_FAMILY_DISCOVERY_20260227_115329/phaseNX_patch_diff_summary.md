# NX Patch Diff Summary

- Generated UTC: 2026-02-27T11:53:59.490424+00:00
- Files changed:
  - scripts/phase_nx_exec_family_discovery.py (new)
- Rationale:
  - Implements autonomous NX0-NX7 discovery pipeline required after J0 NO_GO.
  - Reuses frozen ga_exec harness and unchanged hard-gate logic while adding three structurally distinct entry-mechanics families.
  - Adds required forensic outputs, branch-stop classifications, duplicate/effective-trials telemetry, and robustness gate artifacts.
