# Repro

- Generated UTC: 2026-02-22T20:00:34.166718+00:00
- Seed: 20260225
- Run dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057`

## Fresh Run

```bash
python3 -m src.execution.ga_exec_3m_opt \
  --symbol SOLUSDT \
  --mode tight \
  --pop 192 \
  --gens 24 \
  --workers 4 \
  --seed 20260225 \
  --export-topk 20 \
  --execution-config configs/execution_configs.yaml \
  --canonical-fee-model-path /root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json \
  --canonical-metrics-definition-path /root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md \
  --expected-fee-model-sha256 b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a \
  --expected-metrics-definition-sha256 d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99 \
  --allow-freeze-hash-mismatch 0 \
  --dedupe-avoid-cache-hashes 1 \
  --timeframe 3m \
  --pre-buffer-hours 6.0 \
  --exec-horizon-hours 12.0 \
  --max-signals 1200
```

## Resume

```bash
python3 -m src.execution.ga_exec_3m_opt --resume /root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260222_193057
```
