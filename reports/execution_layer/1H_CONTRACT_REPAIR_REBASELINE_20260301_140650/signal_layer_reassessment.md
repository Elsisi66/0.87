# Signal Layer Reassessment

- Classification: `1H_BASELINE_SURVIVES_REPAIR`
- Signal layer still stands: `1`
- Rationale: `target trio remains profitable after chronology repair`

## Priority Symbols
- SOLUSDT: expectancy `0.00319752` (delta `0.00405383`), cvar_5 `-0.00224775`, maxdd `-0.40084286` (delta `0.58908368`), trade_count `5034`, median_hold_min `60.00`
- NEARUSDT: expectancy `0.00351154` (delta `0.00465106`), cvar_5 `-0.00219952`, maxdd `-0.17706259` (delta `0.63219942`), trade_count `1380`, median_hold_min `60.00`
- AVAXUSDT: expectancy `0.00334589` (delta `0.00406790`), cvar_5 `-0.00219952`, maxdd `-0.16986438` (delta `0.57699057`), trade_count `1732`, median_hold_min `60.00`

## Interpretation
- The signal timestamps were held constant; only the 1h exit chronology changed.
- Any drop in expectancy here is therefore attributable to removing the same-parent-bar exit path, not to a different signal family.
- The repaired same-bar and zero-hold diagnostics are in the companion CSV files.
