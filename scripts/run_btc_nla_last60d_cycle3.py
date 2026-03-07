#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--market", choices=["spot", "swap"], default="spot")
    ap.add_argument("--nla-script", default="scripts/backtest_nla.py", help="Path to your NLA backtester file.")
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--initial", type=float, default=10_000.0)
    ap.add_argument("--cycle-shift", type=int, default=1)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]

    # 1) download data
    dl = [
        sys.executable, str(root / "scripts" / "download_last60d_ohlcv.py"),
        "--symbol", args.symbol,
        "--tf", args.tf,
        "--days", str(args.days),
        "--market", args.market,
    ]
    print(">>> Downloading candles...")
    subprocess.check_call(dl)

    # 2) run NLA backtester using BTC params auto-discovery + override cycle=3
    nla_path = (root / args.nla_script).resolve() if not Path(args.nla_script).is_absolute() else Path(args.nla_script)
    if not nla_path.exists():
        raise SystemExit(f"NLA script not found: {nla_path}")

    bt = [
        sys.executable, str(nla_path),
        "--symbols", args.symbol,
        "--tf", args.tf,
        "--override-cycles", "3",
        "--cycle-shift", str(args.cycle_shift),
        "--fee-bps", str(args.fee_bps),
        "--slip-bps", str(args.slip_bps),
        "--initial", str(args.initial),
    ]
    print(">>> Running NLA backtest (cycle 3)...")
    subprocess.check_call(bt)


if __name__ == "__main__":
    main()
