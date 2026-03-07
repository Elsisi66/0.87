# Updated Paper Alignment Report (Timestamp Normalization Fix)

## Fixed Reporter Bug
- File: `scripts/regenerate_paper_chronology_alignment.py`
- Lines: `126-134` and `149-151`
- Before: `bar_ts` from events used raw string format (`YYYY-MM-DD HH:MM:SS+00:00`) against ISO-keyed signal map (`YYYY-MM-DDTHH:MM:SS+00:00`).
- After: event timestamps are normalized with `pd.to_datetime(..., utc=True).isoformat()` before signal-map lookup.

## Window
- 2026-03-04 00:00:00+00:00 to 2026-03-06 00:00:00+00:00 UTC

## Summary Table (Guarded Replay)
| metric | value |
|---|---:|
| same-bar exit count | 0 |
| entry-on-signal flag count | 0 |
| exit-before-entry count | 0 |
| total trades | 11 |

## Outcome Change vs Prior Chronology Report
- Prior entry-on-signal flags: 11
- Current entry-on-signal flags: 0
- False positives removed by normalization fix: 11

## Final Outcome
- PAPER_ALIGNED
