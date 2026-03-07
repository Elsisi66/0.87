# Root Cause Classification

## Classification
- Exact classification: `MIXED_MULTIPLE_ISSUES`

## Why This Is Not A Pure Logging Bug
- The close is generated in the same execution pass as the open (`paper_trading/app/execution_sim.py:126-167`), so the same-bar stop is a real sequencing outcome, not a journal merge artifact.
- `event_recorded_ts` is assigned separately for each event in `paper_trading/app/reconciler.py:104-109`, proving the logger is recording two distinct events.

## Why Same-Bar SL Also Exists In Backtest
- The comparison backtest export already contains same-bar stop-loss exits: `1052 / 1200` filled trades (`87.6667%`) have `entry_time == exit_time` and `exit_reason == "sl"`.
- That comes from the 1h reference implementation entering on the bar open and simulating path from the same bar index (`scripts/backtest_exec_phasec_sol.py:311-329`).

## Why The Current Paper Runtime Is Still More Pathological Than The Backtest
- The active paper runtime shows `16 / 16` zero-hold same-bar SL exits (`100%`), materially worse than the 1h reference export.
- The dominant mechanical amplifier is stop basis: paper computes stop from cost-inflated entry price (`paper_trading/app/execution_sim.py:63`, `:137-138`), while the 1h reference backtest computes stop from raw entry open (`scripts/backtest_exec_phasec_sol.py:311-313`) and adds costs only later.
- Evidence: `15 / 16` current paper SL closes had inferred `exit_raw_quote > entry_raw_open_quote`, so the raw stop sat above the raw open and could be hit by a minor dip inside the same candle.

## Why This Also Violates The Frozen Model A Contract
- The frozen Model A wrapper requires the first 1h exit-evaluation bar to be strictly after the realized fill time (`reports/execution_layer/PHASEA_MODEL_A_AUDIT_20260228_014944/phaseA1_frozen_exit_semantics.md:5-10`).
- The code implementation uses `searchsorted(..., side="right")` in `scripts/phase_a_model_a_audit.py:423-446`, which prevents same parent-bar 1h exit checks after a fill.
- The currently observed same-bar SLs are therefore not pure Model A behavior; they come from the generic daemon path, not the dedicated Model A runtime path.

## Practical Conclusion
- Same-bar SL is a real behavior in the frozen 1h reference backtest.
- The current paper runtime is not just reproducing that cleanly; it also has a stop-basis mismatch and a bar-level timestamp schema that obscures whether the close was checked on the entry bar versus a later evaluation bar.
- That makes the right label `MIXED_MULTIPLE_ISSUES`, with the main active defect being `STOP_BASIS_PRICE_BUG` inside a non-Model-A runtime path.

## Next Recommended Fix Prompt
```text
ROLE

You are in PAPER RUNTIME FIX mode for `/root/analysis/0.87`.

MISSION

Patch only the generic paper runtime so same-bar stop-loss closes no longer happen unless they are explicitly intended by the frozen reference contract being targeted.

REPAIR TARGETS

1. Keep the current signal engine unchanged.
2. Do not change strategy parameters.
3. Do not change entry signals.
4. If the target contract is pure Model A, move 1h TP/SL evaluation to the first full 1h bar strictly after the realized fill time.
5. If the target contract is 1h reference control, keep same-bar evaluation allowed, but anchor stop/target from raw entry price and apply costs only in executed fill/PnL math.
6. Add explicit journal fields: `entry_bar_ts`, `exit_bar_ts`, `exit_eval_bar_ts`, and `stop_basis_price_raw`.
7. Do not alter portfolio history except through new forward runs.

FILES TO PATCH

- `paper_trading/app/execution_sim.py`
- `paper_trading/app/reconciler.py`
- any small logging schema file if required

ACCEPTANCE TEST

- Prove whether `exit_raw_quote` for SL is above or below raw entry open after the fix.
- Prove whether a newly opened trade can still be closed in the same bar under the chosen contract.
- Produce before/after examples from one symbol.
```
