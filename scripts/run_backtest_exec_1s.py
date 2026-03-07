#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.backtest.exec_1s import run_backtest_long_only_exec_1s, Exec1SConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--h1", required=True, help="1h parquet built FROM the 1s data")
    ap.add_argument("--sec", required=True, help="1s ohlcv parquet")
    ap.add_argument("--params", required=True, help="merged params json (cycle1+cycle3)")
    ap.add_argument("--fee_bps", type=float, default=7.0)
    ap.add_argument("--slip_bps", type=float, default=2.0)
    ap.add_argument("--equity", type=float, default=10000.0)

    # exec params (what we will GA later)
    ap.add_argument("--confirm_window", type=int, default=600)
    ap.add_argument("--confirm_bps", type=float, default=5.0)
    ap.add_argument("--abort_bps", type=float, default=30.0)
    ap.add_argument("--pullback_bps", type=float, default=6.0)
    ap.add_argument("--pullback_window", type=int, default=300)
    ap.add_argument("--no_market_fallback", action="store_true")
    ap.add_argument("--market_on_no_confirm", action="store_true")
    args = ap.parse_args()

    df1h = pd.read_parquet(args.h1)
    sec = pd.read_parquet(args.sec, columns=["Timestamp", "Open", "High", "Low", "Close"])

    with open(args.params, "r", encoding="utf-8") as f:
        p = json.load(f)
    if isinstance(p, dict) and isinstance(p.get("params"), dict):
        p = p["params"]

    exec_cfg = Exec1SConfig(
        confirm_window_sec=args.confirm_window,
        confirm_bps=args.confirm_bps,
        abort_bps=args.abort_bps,
        pullback_bps=args.pullback_bps,
        pullback_window_sec=args.pullback_window,
        market_on_no_pullback=(not args.no_market_fallback),
        market_on_no_confirm=args.market_on_no_confirm,
    )

    trades, m = run_backtest_long_only_exec_1s(
        df_1h=df1h,
        sec_df=sec,
        symbol=args.symbol,
        params=p,
        exec_cfg=exec_cfg,
        initial_equity=args.equity,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
    )

    print("\n=== EXEC 1s BACKTEST ===")
    for k in ["net_profit","trades","win_rate_pct","profit_factor","max_dd","final_equity"]:
        if k in m:
            print(f"{k}: {m[k]}")
    diag_keys = sorted([k for k in m.keys() if k.startswith("diag_")])
    if diag_keys:
        print("\n--- diagnostics ---")
        for k in diag_keys:
            print(f"{k}: {m[k]}")
    print("trades_sample:", trades[:3])

if __name__ == "__main__":
    main()
