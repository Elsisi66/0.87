# Phase E1 Runtime Architecture

## A) 1h Signal / Position Owner
- `paper_trading/app/model_a_runtime.py::ModelAPaperRuntime` uses `SignalRunner` on 1h bars only.
- The 1h owner computes signal state, cycle, and TP/SL geometry from the frozen SOL params.
- Exit evaluation is performed only by the 1h owner via `_maybe_close_position(...)` using 1h bars and the locked 12h evaluation horizon.

## B) 3m Entry Executor
- `ModelAFeed` fetches 3m bars separately from the 1h feed.
- `_simulate_entry_fill(...)` handles only entry placement/fill mechanics: market vs limit, offset, bounded fallback, and fill delay.
- Once a fill is established, the 3m executor has no authority over TP, SL, or any downstream exit mutation.

## C) Paper / Shadow Coordinator
- `paper_primary` and `shadow_backup` are loaded from the locked Phase C selection files.
- Each candidate has its own isolated `StateStore` under `paper_trading/state/model_a_runtime/<role>`.
- Shared market data is used, but books, journals, and summaries are kept separate and attributable by role/candidate.
