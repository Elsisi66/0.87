# Reproduction

```bash
cd /root/analysis/0.87
python3 scripts/backtest_exec_phasec_sol.py --symbol SOLUSDT --phase-c-dir /root/analysis/0.87/reports/execution_layer/PHASEC_SOL_20260221_231430 --phase-a-contract-dir /root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310 --outdir reports/execution_layer --risk-per-trade 0.01 --initial-equity 1.0 --exec-horizon-hours 12.0
```
