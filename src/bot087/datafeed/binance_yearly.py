
"""
Binance downloader + indicator pipeline with yearly caching.

- RAW data:    data/raw/{symbol}_{year}.csv
- PROCESSED:   data/processed/{symbol}_{year}_proc.csv
- If PROCESSED exists and force_proc=False -> we load it and skip indicators.
"""

import os
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import requests
import ta

# -------------------------
# Config
# -------------------------
BASE_DIR = os.path.join(os.getcwd(), "data")
RAW_DIR = os.path.join(BASE_DIR, "raw")
PROC_DIR = os.path.join(BASE_DIR, "processed")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
BINANCE_MAX_LIMIT = 1000
REQUEST_TIMEOUT = 20


def _raw_path(symbol: str, year: int) -> str:
    return os.path.join(RAW_DIR, f"{symbol}_{year}.csv")


def _proc_path(symbol: str, year: int) -> str:
    return os.path.join(PROC_DIR, f"{symbol}_{year}_proc.csv")


def _to_ms(ts: datetime) -> int:
    return int(ts.replace(tzinfo=None).timestamp() * 1000)


def _request_with_retries(session: requests.Session, url: str, params: dict, retries: int = 5) -> requests.Response:
    backoff = 0.5
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 418):
                time.sleep(backoff * (2 ** (attempt - 1)))
                continue
            resp.raise_for_status()
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(backoff * (2 ** (attempt - 1)))
    raise RuntimeError("unreachable")


# -------------------------
# Download raw klines per year
# -------------------------
def download_coin_year(symbol: str, year: int, interval: str = "1h", force: bool = False) -> pd.DataFrame:
    raw_file = _raw_path(symbol, year)

    # If RAW already exists and we don't force, just load it
    if os.path.exists(raw_file) and not force:
        df = pd.read_csv(raw_file)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        else:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce")
            df.rename(columns={df.columns[0]: "Timestamp"}, inplace=True)
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
        return df

    start_dt = pd.Timestamp(f"{year}-01-01").tz_localize("UTC")
    end_dt = pd.Timestamp(f"{year + 1}-01-01").tz_localize("UTC")

    session = requests.Session()
    all_rows = []
    start_ms = _to_ms(start_dt)
    end_ms = _to_ms(end_dt)

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": BINANCE_MAX_LIMIT,
            "startTime": start_ms,
            "endTime": end_ms,
        }
        resp = _request_with_retries(session, BINANCE_KLINES, params)
        data = resp.json()
        if not data:
            break
        all_rows.extend(data)
        last_open_ms = int(data[-1][0])
        start_ms = last_open_ms + 1
        if start_ms >= end_ms:
            break
        if len(data) < BINANCE_MAX_LIMIT:
            break
        time.sleep(0.12)

    if not all_rows:
        return pd.DataFrame()

    cols = [
        "Timestamp",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "CloseTime",
        "QuoteVolume",
        "Trades",
        "TakerBase",
        "TakerQuote",
        "Ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates(subset=["Timestamp"]).dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df.to_csv(raw_file, index=False)
    print(f"✅ Saved RAW to {raw_file}")
    return df


# -------------------------
# Indicators
# -------------------------
def add_indicators(
    df: pd.DataFrame,
    compute_ema: bool = True,
    compute_rsi: bool = True,
    compute_atr: bool = True,
    compute_adx: bool = True,
    compute_macd: bool = True,
    compute_willr: bool = True,
) -> pd.DataFrame:
    out = df.copy()

    if compute_ema:
        close = out["Close"]
        out["EMA_20"] = ta.trend.EMAIndicator(close, window=20, fillna=True).ema_indicator()
        out["EMA_35"] = ta.trend.EMAIndicator(close, window=35, fillna=True).ema_indicator()
        out["EMA_50"] = ta.trend.EMAIndicator(close, window=50, fillna=True).ema_indicator()
        out["EMA_120"] = ta.trend.EMAIndicator(close, window=120, fillna=True).ema_indicator()
        out["EMA_200"] = ta.trend.EMAIndicator(close, window=200, fillna=True).ema_indicator()
        out["EMA_200_SLOPE"] = out["EMA_200"].diff()

    if compute_rsi:
        out["RSI"] = ta.momentum.RSIIndicator(out["Close"], window=14, fillna=True).rsi()

    if compute_atr:
        atr = ta.volatility.AverageTrueRange(
            out["High"], out["Low"], out["Close"], window=14, fillna=True
        )
        out["ATR"] = atr.average_true_range()

    if compute_adx:
        adx = ta.trend.ADXIndicator(
            out["High"], out["Low"], out["Close"], window=14, fillna=True
        )
        out["ADX"] = adx.adx()
        out["PLUS_DI"] = adx.adx_pos()
        out["MINUS_DI"] = adx.adx_neg()

    if compute_macd:
        macd = ta.trend.MACD(
            out["Close"], window_slow=26, window_fast=12, window_sign=9, fillna=True
        )
        out["MACD"] = macd.macd()
        out["MACD_SIGNAL"] = macd.macd_signal()

    if compute_willr:
        out["WILLR"] = ta.momentum.WilliamsRIndicator(
            out["High"], out["Low"], out["Close"], lbp=14, fillna=True
        ).williams_r()

    if "Timestamp" in out.columns:
        out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
        out["Year"] = out["Timestamp"].dt.year
    return out


# -------------------------
# High-level pipeline
# -------------------------
def download_and_process_symbol(
    symbol: str,
    interval: str,
    start: str,
    end: Optional[str] = None,
    force_raw: bool = False,
    force_proc: bool = False,
    indicator_flags: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Download + process full history for [start, end).

    If PROCESSED per-year file already exists and force_proc=False,
    we load it directly and SKIP recomputing indicators.
    """
    indicator_flags = indicator_flags or {}

    start_dt = pd.Timestamp(start)
    if start_dt.tzinfo is None:
        start_dt = start_dt.tz_localize("UTC")

    end_dt = pd.Timestamp(end) if end else None
    if end_dt is not None and end_dt.tzinfo is None:
        end_dt = end_dt.tz_localize("UTC")

    dfs = []
    last_year = end_dt.year if end_dt is not None else start_dt.year

    for year in range(start_dt.year, last_year + 1):
        # 1) RAW - ensure exists
        raw_df = download_coin_year(symbol, year, interval=interval, force=force_raw)
        if raw_df.empty:
            continue

        proc_file = _proc_path(symbol, year)

        # 2) If processed file exists and we don't force -> load and skip indicators
        if os.path.exists(proc_file) and not force_proc:
            print(f"⚡ Loading PROCESSED {proc_file}")
            df_proc = pd.read_csv(proc_file)
            df_proc["Timestamp"] = pd.to_datetime(df_proc["Timestamp"], utc=True, errors="coerce")
            df_proc = df_proc.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
        else:
            # 3) Compute indicators and save processed
            df_proc = add_indicators(raw_df, **indicator_flags)
            df_proc.to_csv(proc_file, index=False)
            print(f"✅ Saved PROCESSED {proc_file}")

        dfs.append(df_proc)

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["Timestamp"] = pd.to_datetime(df_all["Timestamp"], utc=True, errors="coerce")
    df_all = df_all.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    # Clip to [start, end)
    if end_dt is not None:
        mask = (df_all["Timestamp"] >= start_dt) & (df_all["Timestamp"] < end_dt)
    else:
        mask = df_all["Timestamp"] >= start_dt

    df_all = df_all.loc[mask].reset_index(drop=True)
    return df_all
