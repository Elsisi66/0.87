# Repro

```bash
cd /root/analysis/0.87
python3 scripts/phase_e_sol_consistency.py --symbol SOLUSDT --phase-c-dir /root/analysis/0.87/reports/execution_layer/PHASEC_SOL_20260221_231430 --phase-a-contract-dir /root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310 --params-file /root/analysis/0.87/data/metadata/params/SOLUSDT_C13_active_params_long.json --best-by-symbol-csv /root/analysis/0.87/reports/params_scan/20260220_044949/best_by_symbol.csv --full-signal-csv /root/analysis/0.87/data/signals/SOLUSDT_signals_1h.csv --outdir reports/execution_layer --seed 42 --alt-subsets 10
```
