# Repro

```bash
cd /root/analysis/0.87
python3 scripts/phase_e2_sol_representative.py --symbol SOLUSDT --phase-c-dir /root/analysis/0.87/reports/execution_layer/PHASEC_SOL_20260221_231430 --phase-a-contract-dir /root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310 --signal-source-csv /root/analysis/0.87/reports/execution_layer/PHASEC_SOL_20260221_231430/config_snapshot/SOLUSDT_signals_1h.csv --outdir reports/execution_layer --representative-size 1200 --wf-splits 5 --seed 42
```
