#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd


def sym_to_ccxt(sym: str, market: str = "spot") -> str:
    """
    BTCUSDT -> BTC/USDT (spot)
    BTCUSDT -> BTC/USDT:USDT (swap)  [Binance USD-M swap style]
    Also accepts BTC/USDT as input.
    """
    s = sym.strip().upper()
    if ":" in s:
        s = s.split(":", 1)[0]
    s = s.replace("/", "")

    quotes = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD")
    for q in quotes:
        if s.endswith(q):
            base = s[:-len(q)]
            out = f"{base}/{q}"
            if market.lower() in ("swap", "futures", "perp"):
                out = f"{out}:{q}"
            return out
    raise ValueError(f"Unsupported symbol format: {sym}")


def ensure_dirs(fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--market", choices=["spot", "swap"], default="spot")
    ap.add_argument("--out", default=None, help="Optional output parquet path. Default: data/processed/_full/<SYMBOL>_<TF>_full.parquet")
    ap.add_argument("--exchange", default="binance", help="ccxt exchange id (default: binance)")
    args = ap.parse_args()

    # project root = .../0.87
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)

    try:
        import ccxt  # type: ignore
    except Exception as e:
        print("ccxt not installed. Install it:\n  pip install ccxt", file=sys.stderr)
        raise

    sym = args.symbol.strip().upper()
    tf = args.tf.strip()
    days = int(args.days)

    out_fp = Path(args.out) if args.out else (PROJECT_ROOT / "data" / "processed" / "_full" / f"{sym}_{tf}_full.parquet")
    ensure_dirs(out_fp)

    ex_id = args.exchange.strip()
    if not hasattr(ccxt, ex_id):
        raise SystemExit(f"Unknown ccxt exchange id: {ex_id}")

    ex = getattr(ccxt, ex_id)({"enableRateLimit": True})

    # market config (binance)
    if ex_id.lower() == "binance":
        if args.market == "swap":
            ex.options = {**getattr(ex, "options", {}), "defaultType": "future"}
        else:
            ex.options = {**getattr(ex, "options", {}), "defaultType": "spot"}

    ccxt_symbol = sym_to_ccxt(sym, market=args.market)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=days)
    since_ms = int(start.timestamp() * 1000)

    all_rows = []
    limit = 1000  # Binance default max per call is usually 1000
    last_ms = None

    # Pull until we cover the range; add a safety cap
    for _ in range(50):
        batch = ex.fetch_ohlcv(ccxt_symbol, timeframe=tf, since=since_ms if last_ms is None else last_ms + 1, limit=limit)
        if not batch:
            break

        all_rows.extend(batch)
        last_ms = batch[-1][0]

        # if we've reached "now", stop
        if last_ms >= int(now.timestamp() * 1000) - 60_000:
            break

        # if batch is short, we likely hit the end
        if len(batch) < limit:
            break

    if not all_rows:
        raise SystemExit("No candles returned. Check symbol/market/timeframe.")

    df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # Trim strictly to last N days
    df = df[df["Timestamp"] >= pd.to_datetime(start, utc=True)].reset_index(drop=True)

    # Save
    df.to_parquet(out_fp, index=False)

    print(f"Saved {len(df)} candles to: {out_fp}")
    print(f"From: {df['Timestamp'].min()}  To: {df['Timestamp'].max()}")
    print(f"ccxt symbol: {ccxt_symbol} | market: {args.market} | tf: {tf}")


if __name__ == "__main__":
    main()
