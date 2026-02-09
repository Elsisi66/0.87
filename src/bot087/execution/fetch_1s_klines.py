from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from .cache import _to_utc_ts

SPOT_BASE = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"


def _request_klines(
    session: Any,
    *,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int],
    limit: int,
) -> List[list]:
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "startTime": int(start_ms),
        "limit": int(limit),
    }
    if end_ms is not None:
        params["endTime"] = int(end_ms)

    r = session.get(f"{SPOT_BASE}{KLINES_PATH}", params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Binance klines error status={r.status_code} body={r.text[:300]}")
    payload = r.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected klines payload type: {type(payload)}")
    return payload


def fetch_1s_klines(
    *,
    symbol: str,
    start_ts,
    end_ts,
    interval: str = "1s",
    limit: int = 1000,
    pause_sec: float = 0.05,
    session: Optional[Any] = None,
) -> pd.DataFrame:
    """Fetch spot klines in [start_ts, end_ts)."""
    if interval != "1s":
        # Function is generic, but pipeline expects explicit 1s support.
        interval = str(interval)

    s = _to_utc_ts(start_ts)
    e = _to_utc_ts(end_ts)
    if e <= s:
        raise ValueError("end_ts must be > start_ts")

    close_ms_excl = int(e.value // 1_000_000)
    cursor = int(s.value // 1_000_000)

    if requests is None:
        raise RuntimeError("`requests` package is required for fetch_1s_klines")

    own_session = session is None
    sess = session or requests.Session()

    rows: List[list] = []
    try:
        while cursor < close_ms_excl:
            batch = _request_klines(
                sess,
                symbol=symbol,
                interval=interval,
                start_ms=cursor,
                end_ms=close_ms_excl,
                limit=limit,
            )
            if not batch:
                break

            rows.extend(batch)
            last_open_ms = int(batch[-1][0])
            if last_open_ms <= cursor:
                break
            cursor = last_open_ms + 1
            time.sleep(max(0.0, pause_sec))
    finally:
        if own_session:
            sess.close()

    if not rows:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "QuoteVolume", "Trades"])

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_base",
            "taker_quote",
            "ignore",
        ],
    )
    out = pd.DataFrame()
    out["Timestamp"] = pd.to_datetime(pd.to_numeric(df["open_time"], errors="coerce"), unit="ms", utc=True)
    out["Open"] = pd.to_numeric(df["open"], errors="coerce")
    out["High"] = pd.to_numeric(df["high"], errors="coerce")
    out["Low"] = pd.to_numeric(df["low"], errors="coerce")
    out["Close"] = pd.to_numeric(df["close"], errors="coerce")
    out["Volume"] = pd.to_numeric(df["volume"], errors="coerce")
    out["QuoteVolume"] = pd.to_numeric(df["quote_volume"], errors="coerce")
    out["Trades"] = pd.to_numeric(df["trades"], errors="coerce")

    out = out.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).sort_values("Timestamp").drop_duplicates("Timestamp")
    out = out[(out["Timestamp"] >= s) & (out["Timestamp"] < e)].reset_index(drop=True)
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True, help="UTC timestamp/date")
    ap.add_argument("--end", required=True, help="UTC timestamp/date (exclusive)")
    ap.add_argument("--interval", default="1s")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pause", type=float, default=0.05)
    args = ap.parse_args()

    df = fetch_1s_klines(
        symbol=args.symbol.upper(),
        start_ts=args.start,
        end_ts=args.end,
        interval=args.interval,
        pause_sec=float(args.pause),
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"saved={out_path} rows={len(df)}")


if __name__ == "__main__":
    _cli()
