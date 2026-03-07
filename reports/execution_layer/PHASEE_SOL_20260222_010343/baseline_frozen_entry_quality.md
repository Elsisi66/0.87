# Baseline Frozen Entry Quality

- Control: Phase C best exit (cfg_hash=a285b86c4c22a26976d4a762) with original 1h signal entry stream.
- Universe: frozen test subset only (600 signals).

## Stop Distance Quantiles (signal-time, 1h)

- q0.10: 0.001000
- q0.25: 0.001000
- q0.50: 0.001000
- q0.75: 0.001000
- q0.90: 0.001000

## MAE / MFE Quantiles

- MAE q0.10: -0.003228
- MAE q0.25: -0.002269
- MAE q0.50: -0.001527
- MAE q0.75: -0.001054
- MAE q0.90: -0.000840
- MFE q0.10: 0.000046
- MFE q0.25: 0.000355
- MFE q0.50: 0.001028
- MFE q0.75: 0.002633
- MFE q0.90: 0.007374

## Immediate Adverse Excursion Proxy

- SL hit at 0 min: 0.5900
- SL hit within 15 min: 0.8400
- SL hit within 60 min: 0.9000

## Reachability Before Exit

- % reaching +0.25R before exit: 0.8150
- % reaching +0.50R before exit: 0.7433
- % reaching +1.00R before exit: 0.5967

Note: first-N-bars MAE/MFE are not available in frozen artifacts; proxies above use full-trade MAE/MFE and hold-time-window SL rates.
