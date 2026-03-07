# Repro

- Generated UTC: 2026-02-20T15:24:24.599663+00:00
- Seed: 42
- Run dir: `/root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402`

## Fresh Run

```bash
python3 -m src.execution.ga_exec_3m_opt \
  --symbol SOLUSDT \
  --mode tight \
  --pop 256 \
  --gens 3 \
  --workers 1 \
  --seed 42 \
  --export-topk 20 \
  --execution-config configs/execution_configs.yaml \
  --timeframe 3m \
  --pre-buffer-hours 6.0 \
  --exec-horizon-hours 12.0 \
  --max-signals 200
```

## Resume

```bash
python3 -m src.execution.ga_exec_3m_opt --resume /root/analysis/0.87/reports/execution_layer/GA_EXEC_OPT_20260220_152402
```
