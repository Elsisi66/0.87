from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
try:
    import requests
except Exception:  # pragma: no cover
    requests = None

from .cache import _to_utc_ts

SPOT_BASE = "https://api.binance.com"
AGG_PATH = "/api/v3/aggTrades"


def _request_aggtrades(
    session: Any,
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> List[dict]:
    params = {
        "symbol": symbol.upper(),
        "startTime": int(start_ms),
        "endTime": int(end_ms),
        "limit": int(limit),
    }
    r = session.get(f"{SPOT_BASE}{AGG_PATH}", params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Binance aggTrades error status={r.status_code} body={r.text[:300]}")
    payload = r.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected aggTrades payload type: {type(payload)}")
    return payload


def fetch_aggtrades(
    *,
    symbol: str,
    start_ts,
    end_ts,
    limit: int = 1000,
    pause_sec: float = 0.05,
    session: Optional[Any] = None,
) -> pd.DataFrame:
    """Fetch spot aggTrades in [start_ts, end_ts)."""
    s = _to_utc_ts(start_ts)
    e = _to_utc_ts(end_ts)
    if e <= s:
        raise ValueError("end_ts must be > start_ts")

    start_ms = int(s.value // 1_000_000)
    end_ms = int(e.value // 1_000_000)

    if requests is None:
        raise RuntimeError("`requests` package is required for fetch_aggtrades")

    own_session = session is None
    sess = session or requests.Session()

    rows: List[dict] = []
    cursor = start_ms
    try:
        while cursor < end_ms:
            batch = _request_aggtrades(
                sess,
                symbol=symbol,
                start_ms=cursor,
                end_ms=end_ms,
                limit=limit,
            )
            if not batch:
                break
            rows.extend(batch)
            last_t = int(batch[-1].get("T", cursor))
            if last_t <= cursor:
                break
            cursor = last_t + 1
            time.sleep(max(0.0, pause_sec))
    finally:
        if own_session:
            sess.close()

    if not rows:
        return pd.DataFrame(columns=["Timestamp", "Price", "Qty", "AggId", "IsBuyerMaker"])

    df = pd.DataFrame(rows)
    out = pd.DataFrame()
    out["Timestamp"] = pd.to_datetime(pd.to_numeric(df.get("T"), errors="coerce"), unit="ms", utc=True)
    out["Price"] = pd.to_numeric(df.get("p"), errors="coerce")
    out["Qty"] = pd.to_numeric(df.get("q"), errors="coerce")
    out["AggId"] = pd.to_numeric(df.get("a"), errors="coerce")
    out["IsBuyerMaker"] = df.get("m", False).astype(bool)
    out = out.dropna(subset=["Timestamp", "Price", "Qty"]).sort_values("Timestamp").reset_index(drop=True)
    out = out[(out["Timestamp"] >= s) & (out["Timestamp"] < e)]
    return out


def aggtrades_to_1s_ohlcv(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "QuoteVolume", "Trades"])

    x = agg_df.copy()
    x["Timestamp"] = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
    x = x.dropna(subset=["Timestamp"])
    x["sec"] = x["Timestamp"].dt.floor("s")
    x["Quote"] = x["Price"].astype(float) * x["Qty"].astype(float)

    g = x.groupby("sec", sort=True)
    out = pd.DataFrame(
        {
            "Timestamp": g["sec"].first(),
            "Open": g["Price"].first(),
            "High": g["Price"].max(),
            "Low": g["Price"].min(),
            "Close": g["Price"].last(),
            "Volume": g["Qty"].sum(),
            "QuoteVolume": g["Quote"].sum(),
            "Trades": g["Price"].size(),
        }
    ).reset_index(drop=True)

    out = out.sort_values("Timestamp").reset_index(drop=True)
    return out


def fetch_precision_1s_from_aggtrades(
    *,
    symbol: str,
    start_ts,
    end_ts,
    limit: int = 1000,
    pause_sec: float = 0.05,
    session: Optional[Any] = None,
) -> pd.DataFrame:
    agg = fetch_aggtrades(
        symbol=symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        limit=limit,
        pause_sec=pause_sec,
        session=session,
    )
    return aggtrades_to_1s_ohlcv(agg)


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--precision-mode", action="store_true", default=True)
    args = ap.parse_args()

    if args.precision_mode:
        df = fetch_precision_1s_from_aggtrades(symbol=args.symbol.upper(), start_ts=args.start, end_ts=args.end)
    else:
        agg = fetch_aggtrades(symbol=args.symbol.upper(), start_ts=args.start, end_ts=args.end)
        df = agg

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved={out} rows={len(df)}")


if __name__ == "__main__":
    _cli()
