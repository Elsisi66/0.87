from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd

from .cache import _to_utc_ts

SPOT_BASE = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"


def _http_get_json(base: str, path: str, params: Dict[str, str], timeout: int = 30) -> object:
    qs = urllib.parse.urlencode(params)
    url = f"{base}{path}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "bot087-exec-gate/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def fetch_1s_klines(
    *,
    symbol: str,
    start_ts,
    end_ts,
    interval: str = "1s",
    max_seconds_per_request: int = 1000,
    pause_sec: float = 0.02,
    retries: int = 4,
    log_cb: Optional[Callable[[Dict], None]] = None,
) -> pd.DataFrame:
    """Fetch spot klines in [start_ts, end_ts) with <=1000 second request windows."""
    s = _to_utc_ts(start_ts)
    e = _to_utc_ts(end_ts)
    if e <= s:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "QuoteVolume", "Trades"])

    if interval != "1s":
        interval = str(interval)

    step_ms = int(max(1, int(max_seconds_per_request)) * 1000)
    cursor_ms = int(s.value // 1_000_000)
    end_ms = int(e.value // 1_000_000)

    rows: List[list] = []
    req_n = 0

    while cursor_ms < end_ms:
        chunk_end_ms = min(end_ms, cursor_ms + step_ms)
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": str(cursor_ms),
            "endTime": str(max(cursor_ms, chunk_end_ms - 1)),
            "limit": "1000",
        }

        ok = False
        last_err = None
        for attempt in range(retries + 1):
            try:
                payload = _http_get_json(SPOT_BASE, KLINES_PATH, params=params, timeout=30)
                if not isinstance(payload, list):
                    raise RuntimeError(f"Unexpected payload type: {type(payload)}")
                batch = payload
                req_n += 1
                ok = True
                if log_cb is not None:
                    log_cb(
                        {
                            "event": "fetch_klines_chunk",
                            "symbol": symbol.upper(),
                            "start_ms": cursor_ms,
                            "end_ms": chunk_end_ms,
                            "rows": len(batch),
                            "attempt": attempt,
                        }
                    )
                break
            except Exception as ex:
                last_err = ex
                sleep_s = min(10.0, 0.5 * (2 ** attempt))
                if log_cb is not None:
                    log_cb(
                        {
                            "event": "fetch_klines_retry",
                            "symbol": symbol.upper(),
                            "start_ms": cursor_ms,
                            "end_ms": chunk_end_ms,
                            "attempt": attempt,
                            "error": str(ex),
                            "sleep_sec": sleep_s,
                        }
                    )
                time.sleep(sleep_s)

        if not ok:
            raise RuntimeError(f"fetch_1s_klines failed for {symbol} {cursor_ms}-{chunk_end_ms}: {last_err}")

        if batch:
            rows.extend(batch)

        cursor_ms = chunk_end_ms
        if pause_sec > 0:
            time.sleep(pause_sec)

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

    out = out.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    out = out.sort_values("Timestamp").drop_duplicates("Timestamp")
    out = out[(out["Timestamp"] >= s) & (out["Timestamp"] < e)].reset_index(drop=True)

    if log_cb is not None:
        log_cb(
            {
                "event": "fetch_klines_done",
                "symbol": symbol.upper(),
                "rows": int(len(out)),
                "requests": int(req_n),
                "start": str(s),
                "end": str(e),
            }
        )

    return out


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", default="1s")
    args = ap.parse_args()

    df = fetch_1s_klines(symbol=args.symbol.upper(), start_ts=args.start, end_ts=args.end, interval=args.interval)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"saved={out_path} rows={len(df)}")


if __name__ == "__main__":
    _cli()
