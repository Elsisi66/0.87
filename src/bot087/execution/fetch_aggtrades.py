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
FUTURES_BASE = "https://fapi.binance.com"
AGG_PATH_SPOT = "/api/v3/aggTrades"
AGG_PATH_FUTURES = "/fapi/v1/aggTrades"


def _http_get_json(base: str, path: str, params: Dict[str, str], timeout: int = 30) -> object:
    qs = urllib.parse.urlencode(params)
    url = f"{base}{path}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "bot087-exec-gate/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def fetch_aggtrades(
    *,
    symbol: str,
    start_ts,
    end_ts,
    market: str = "spot",  # spot | futures
    limit: int = 1000,
    max_window_sec: int = 3500,
    pause_sec: float = 0.02,
    retries: int = 4,
    log_cb: Optional[Callable[[Dict], None]] = None,
) -> pd.DataFrame:
    """Fetch aggTrades in [start_ts, end_ts), chunked and paginated."""
    s = _to_utc_ts(start_ts)
    e = _to_utc_ts(end_ts)
    if e <= s:
        return pd.DataFrame(columns=["Timestamp", "Price", "Qty", "AggId", "IsBuyerMaker"])

    market = str(market).lower().strip()
    is_futures = market.startswith("fut")
    base = FUTURES_BASE if is_futures else SPOT_BASE
    path = AGG_PATH_FUTURES if is_futures else AGG_PATH_SPOT

    # Binance futures aggTrades time windows are constrained; keep <1h.
    if is_futures:
        max_window_sec = min(int(max_window_sec), 3590)

    step_ms = max(1, int(max_window_sec)) * 1000
    cursor_ms = int(s.value // 1_000_000)
    end_ms = int(e.value // 1_000_000)
    rows: List[dict] = []
    req_n = 0

    while cursor_ms < end_ms:
        chunk_end_ms = min(end_ms, cursor_ms + step_ms)
        page_cursor_ms = cursor_ms

        while page_cursor_ms < chunk_end_ms:
            params = {
                "symbol": symbol.upper(),
                "startTime": str(page_cursor_ms),
                "endTime": str(max(page_cursor_ms, chunk_end_ms - 1)),
                "limit": str(int(limit)),
            }

            ok = False
            last_err = None
            batch: List[dict] = []
            for attempt in range(retries + 1):
                try:
                    payload = _http_get_json(base, path, params=params, timeout=30)
                    if not isinstance(payload, list):
                        raise RuntimeError(f"Unexpected payload type: {type(payload)}")
                    batch = payload
                    req_n += 1
                    ok = True
                    if log_cb is not None:
                        log_cb(
                            {
                                "event": "fetch_agg_chunk",
                                "symbol": symbol.upper(),
                                "market": market,
                                "start_ms": page_cursor_ms,
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
                                "event": "fetch_agg_retry",
                                "symbol": symbol.upper(),
                                "market": market,
                                "start_ms": page_cursor_ms,
                                "end_ms": chunk_end_ms,
                                "attempt": attempt,
                                "error": str(ex),
                                "sleep_sec": sleep_s,
                            }
                        )
                    time.sleep(sleep_s)

            if not ok:
                raise RuntimeError(
                    f"fetch_aggtrades failed for {symbol} {page_cursor_ms}-{chunk_end_ms}: {last_err}"
                )

            if not batch:
                break

            rows.extend(batch)
            last_t = int(batch[-1].get("T", page_cursor_ms))
            if last_t <= page_cursor_ms:
                break
            page_cursor_ms = last_t + 1
            if pause_sec > 0:
                time.sleep(pause_sec)

        cursor_ms = chunk_end_ms
        if pause_sec > 0:
            time.sleep(pause_sec)

    if not rows:
        return pd.DataFrame(columns=["Timestamp", "Price", "Qty", "AggId", "IsBuyerMaker"])

    df = pd.DataFrame(rows)
    out = pd.DataFrame()
    out["Timestamp"] = pd.to_datetime(pd.to_numeric(df.get("T"), errors="coerce"), unit="ms", utc=True)
    out["Price"] = pd.to_numeric(df.get("p"), errors="coerce")
    out["Qty"] = pd.to_numeric(df.get("q"), errors="coerce")
    out["AggId"] = pd.to_numeric(df.get("a"), errors="coerce")
    out["IsBuyerMaker"] = df.get("m", False).astype(bool)

    out = out.dropna(subset=["Timestamp", "Price", "Qty"]).sort_values("Timestamp")
    out = out.drop_duplicates(subset=["Timestamp", "AggId"], keep="last")
    out = out[(out["Timestamp"] >= s) & (out["Timestamp"] < e)].reset_index(drop=True)

    if log_cb is not None:
        log_cb(
            {
                "event": "fetch_agg_done",
                "symbol": symbol.upper(),
                "market": market,
                "rows": int(len(out)),
                "requests": int(req_n),
                "start": str(s),
                "end": str(e),
            }
        )

    return out


def aggtrades_to_1s_ohlcv(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume", "QuoteVolume", "Trades"])

    x = agg_df.copy()
    x["Timestamp"] = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
    x = x.dropna(subset=["Timestamp"]).reset_index(drop=True)
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
    market: str = "spot",
    limit: int = 1000,
    max_window_sec: int = 3500,
    pause_sec: float = 0.02,
    retries: int = 4,
    log_cb: Optional[Callable[[Dict], None]] = None,
) -> pd.DataFrame:
    agg = fetch_aggtrades(
        symbol=symbol,
        start_ts=start_ts,
        end_ts=end_ts,
        market=market,
        limit=limit,
        max_window_sec=max_window_sec,
        pause_sec=pause_sec,
        retries=retries,
        log_cb=log_cb,
    )
    return aggtrades_to_1s_ohlcv(agg)


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--market", default="spot", choices=["spot", "futures"])
    args = ap.parse_args()

    df = fetch_precision_1s_from_aggtrades(
        symbol=args.symbol.upper(),
        start_ts=args.start,
        end_ts=args.end,
        market=args.market,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved={out} rows={len(df)}")


if __name__ == "__main__":
    _cli()
