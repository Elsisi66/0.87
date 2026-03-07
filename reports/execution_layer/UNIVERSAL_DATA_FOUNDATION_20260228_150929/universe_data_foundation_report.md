# UNIVERSAL_DATA_FOUNDATION

- Generated UTC: 2026-02-28T18:02:31.470943+00:00
- Universe size: 17
- Pre-buffer hours: 6.00
- Post-buffer hours: 12.00
- Max merged window hours: 72.00
- Canonical 1h engine: `src/bot087/optim/ga.py`
- 3m bounded slicer template: `scripts/execution_layer_3m_ict.py`
- Combined harness reference: `scripts/phase_u_combined_1h3m_pilot.py`
- SOL audit reference: `scripts/phase_a_model_a_audit.py`
- Param scan source: `/root/analysis/0.87/reports/params_scan/20260220_044949/best_by_symbol.csv`
- Frozen fee model: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json`
- Frozen metrics definition: `/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md`
- SOL signal snapshot reference: `/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/config_snapshot/SOLUSDT_signals_1h.csv`

## Status

- READY: none
- PARTIAL: SOLUSDT, AVAXUSDT, BCHUSDT, CRVUSDT, NEARUSDT, ADAUSDT, AXSUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, LINKUSDT, LTCUSDT, OGUSDT, PAXGUSDT, TRXUSDT, XRPUSDT, ZECUSDT
- BLOCKED: none

## Reconstruction Summary

- SOLUSDT: signals=5034, windows_ready=393/397, status=PARTIAL, bucket=passed_1h_long, source_notes=params_from_best_by_symbol
- AVAXUSDT: signals=1732, windows_ready=362/366, status=PARTIAL, bucket=passed_1h_long, source_notes=params_from_best_by_symbol
- BCHUSDT: signals=1731, windows_ready=370/372, status=PARTIAL, bucket=passed_1h_long, source_notes=params_from_best_by_symbol
- CRVUSDT: signals=1586, windows_ready=327/328, status=PARTIAL, bucket=passed_1h_long, source_notes=params_from_best_by_symbol
- NEARUSDT: signals=1380, windows_ready=285/288, status=PARTIAL, bucket=passed_1h_long, source_notes=params_from_best_by_symbol
- ADAUSDT: signals=2660, windows_ready=492/498, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- AXSUSDT: signals=573, windows_ready=223/225, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- BNBUSDT: signals=617, windows_ready=299/302, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- BTCUSDT: signals=905, windows_ready=416/418, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- DOGEUSDT: signals=1391, windows_ready=304/309, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- LINKUSDT: signals=968, windows_ready=327/332, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- LTCUSDT: signals=940, windows_ready=342/348, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- OGUSDT: signals=1322, windows_ready=263/263, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- PAXGUSDT: signals=231, windows_ready=101/101, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- TRXUSDT: signals=976, windows_ready=386/388, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- XRPUSDT: signals=2034, windows_ready=419/424, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol
- ZECUSDT: signals=1374, windows_ready=383/386, status=PARTIAL, bucket=failed_1h_long, source_notes=params_from_best_by_symbol

## Biggest Blockers

- SOLUSDT: nan
- AVAXUSDT: nan
- BCHUSDT: nan
- NEARUSDT: nan

## Artifact Files

- `universe_signal_timeline.csv`
- `universe_signal_timeline.parquet`
- `universe_3m_window_plan.csv`
- `universe_3m_download_manifest.csv`
- `universe_3m_data_quality.csv`
- `universe_symbol_readiness.csv`
- `run_manifest.json`
