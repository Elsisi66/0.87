# Mismatch Checklist

- [PASS] symbol_scope: 
- [FAIL] date_range: Different windows and regime composition
- [FAIL] signal_source: Different signal universe construction
- [FAIL] entry_logic: Different entry timing granularity
- [FAIL] exit_logic: Different path-dependent exits
- [FAIL] stop_hit_granularity: Intrabar stop/TP ordering differs materially
- [FAIL] fee_slippage_model: Explicit contract mismatch
- [FAIL] position_sizing: Different leverage/notional dynamics
- [FAIL] compounding: Different equity process
- [FAIL] wf_split_scope: Frozen test-only split contract
