from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.bot087.execution.execution_eval import ExecutionEvalConfig, evaluate_execution_from_trades


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--trades-csv", required=True)
    ap.add_argument("--window-sec", type=int, default=15)
    ap.add_argument("--mode", choices=["klines1s", "aggtrades"], default="klines1s")
    ap.add_argument("--market", choices=["spot", "futures"], default="spot")
    ap.add_argument("--cache-cap-gb", type=float, default=20.0)
    ap.add_argument("--cache-root", default="data/processed/execution_1s")
    ap.add_argument("--fetch-workers", type=int, default=1)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()

    run_id = args.run_id.strip() or _utc_tag()
    run_dir = Path("artifacts/execution") / args.symbol.upper() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = ExecutionEvalConfig(
        mode=args.mode,
        market=args.market,
        window_sec=int(args.window_sec),
        cache_cap_gb=float(args.cache_cap_gb),
        cap_gb=float(args.cache_cap_gb),
        cache_root=str(Path(args.cache_root).resolve()),
        fetch_workers=max(1, int(args.fetch_workers)),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        initial_equity=float(args.initial_equity),
    )

    out = evaluate_execution_from_trades(
        symbol=args.symbol.upper(),
        trades_path=args.trades_csv,
        cfg=cfg,
        run_dir=str(run_dir),
    )

    # Also write full payload for reproducibility.
    (run_dir / "result_full.json").write_text(
        json.dumps(
            {
                "symbol": out["symbol"],
                "summary": out["summary"],
                "summary_stress": out.get("summary_stress", {}),
                "time_range": out.get("time_range", {}),
                "cache_root": out.get("cache_root", ""),
                "inputs": vars(args),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    s = out["summary"]
    print("=== EXEC GATE ===")
    print(f"symbol: {out['symbol']}")
    print(f"exec_pass: {s.get('exec_pass')}")
    print(f"edge_decay: {s.get('edge_decay', 0.0):.4f}")
    print(f"pf_after: {s.get('pf_after', 0.0):.4f}")
    print(f"dd_before: {s.get('dd_before', 0.0):.4f}")
    print(f"dd_after: {s.get('dd_after', 0.0):.4f}")
    print(f"dd_delta: {s.get('dd_delta', 0.0):.4f}")
    print(f"trade_level_csv: {run_dir / 'trade_level.csv'}")
    print(f"summary_json: {run_dir / 'summary.json'}")
    print(f"fetch_log_jsonl: {run_dir / 'fetch_log.jsonl'}")


if __name__ == "__main__":
    main()
