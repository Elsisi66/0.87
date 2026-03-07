# Patch Diff Summary

- Timestamp UTC: 2026-02-23T13:44:xxZ
- File patched: `scripts/phase_d123_tail_filter.py`
- Reason: D3 crashed with `ValueError: Invalid format specifier '.2f ' for object of type 'float'` in policy-id format string.
- Change:
  - Fixed malformed f-string IDs in `build_filter_policies()`:
    - `skip_streak{ k }_cool{ cool }m` -> `skip_streak{k}_cool{cool}m`
    - `skip_risk{ t:.2f }_streak{ k }` -> `skip_risk{t:.2f}_streak{k}`
- Scope: localized, no metric/gate logic changes.
- Rerun policy: resume from existing checkpoints in `PHASED_TAIL_BRANCH_20260223_134324`.
