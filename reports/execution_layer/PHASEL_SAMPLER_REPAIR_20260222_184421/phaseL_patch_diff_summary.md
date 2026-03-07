# Phase L Patch Diff Summary

- Generated UTC: 2026-02-22T18:48:26.724698+00:00
- Files changed: `src/execution/ga_exec_3m_opt.py`
- Diff stats (this file): unavailable via git index in this workspace context (file is outside tracked index scope); changes validated via compile + smoke artifacts.

## Functional Changes

1. True freeze hash lock
   - Added canonical artifact path + expected hash CLI args.
   - Added fail-fast hash validation and fee-value consistency check.
   - Added verbatim canonical copy into run dir and `freeze_lock_validation.json`.
2. Constraint-first sampler repair
   - Added invalid-by-construction detector and optional resampling.
   - Added tight-mode feasibility repairs (fallback enforcement, offset/quality/cooldown caps, restrictive-stack softening).
3. Generation-time dedupe + counters
   - Added pre-eval genome-hash dedupe/refill step with cache-aware refill option.
   - Added dedupe/sampler counters to `gen_status.json` and run manifest.
4. Run manifest + effective-trials proxy
   - Added `run_manifest.json` with freeze, duplicate, and sampler telemetry.
   - Added metric-signature duplicate counts and `effective_trials_proxy`.

## Non-Changes (Explicit)

- No gate thresholds were changed.
- No fee/slippage values or core metrics logic were changed.
- No downstream execution mechanics were altered beyond search-space/sampler behavior.
