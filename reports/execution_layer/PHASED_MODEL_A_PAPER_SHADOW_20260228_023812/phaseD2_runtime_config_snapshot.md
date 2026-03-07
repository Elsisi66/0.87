# Phase D2 Runtime Config Snapshot

## Intended Deployment Roles
- `paper_primary`: `M3_ENTRY_ONLY_FASTER_C_WIN_02`
- `shadow_backup`: `M2_ENTRY_ONLY_MORE_PASSIVE_NOFB_C_FB_ON`
- Capital logic: isolated paper books; no combined allocation logic.

## Required Runtime Contract
- Symbol scope: `SOLUSDT` only.
- 1h signal params path: `/root/analysis/0.87/data/metadata/params/SOLUSDT_C13_active_params_long.json`
- 1h signal engine: `paper_trading/app/signal_runner.py::build_signal_frame`.
- 3m entry wrapper source: `scripts/phase_a_model_a_audit.py::simulate_entry_only_fill`.
- 1h exit source: `scripts/phase_a_model_a_audit.py::simulate_frozen_1h_exit`.

## Current Generic Daemon Snapshot
- Settings file: `/root/analysis/0.87/paper_trading/config/settings.yaml`
- Resolved universe file: `/root/analysis/0.87/paper_trading/config/resolved_universe.json`
- Current symbols tracked: `ADAUSDT,AVAXUSDT,AXSUSDT,BCHUSDT,CRVUSDT,DOGEUSDT,LINKUSDT,LTCUSDT,NEARUSDT,SOLUSDT,TRXUSDT,XRPUSDT,ZECUSDT`
- Current symbol count: `13`
- Current runtime does not load candidate-specific Model A execution knobs.
- Current runtime cannot map `paper_primary` and `shadow_backup` separately.

## Status
- Candidate mapping is defined at the deployment package level but not applied to the live paper daemon.
- This is a prepared config snapshot only; no runtime switch was applied because parity failed in D1.
