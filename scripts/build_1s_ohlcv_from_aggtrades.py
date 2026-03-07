#!/usr/bin/env python3
import argparse
from pathlib import Path
import zipfile
import pandas as pd
import numpy as np

AGG_COLS = [
    "agg_trade_id",
    "price",
    "qty",
    "first_trade_id",
    "last_trade_id",
    "transact_time",   # may be ms OR us depending on dataset
    "is_buyer_maker",
    "is_best_match",
]

def _find_files(raw_dir: Path):
    zips = sorted(raw_dir.rglob("*.zip"))
    if zips:
        return zips
    csvs = sorted(raw_dir.rglob("*.csv"))
    return csvs

def _read_aggtrades_any(path: Path, chunksize: int = 2_000_000):
    """
    Yields chunks as DataFrame with columns=AGG_COLS.
    Binance Vision aggTrades CSVs usually have NO header row.
    """
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise RuntimeError(f"No CSV inside zip: {path}")
            name = names[0]
            with zf.open(name) as f:
                it = pd.read_csv(
                    f,
                    header=None,
                    names=AGG_COLS,
                    chunksize=chunksize,
                )
                for chunk in it:
                    yield chunk
    else:
        it = pd.read_csv(
            path,
            header=None,
            names=AGG_COLS,
            chunksize=chunksize,
        )
        for chunk in it:
            yield chunk

def _to_ms(arr: np.ndarray) -> np.ndarray:
    """
    Detect time unit and convert to milliseconds.
    - ms  ~ 1e12 (2025)
    - us  ~ 1e15
    - ns  ~ 1e18
    """
    arr = arr.astype(np.int64, copy=False)
    # use a robust sample
    sample = arr[: min(len(arr), 100_000)]
    sample = sample[sample > 0]
    if sample.size == 0:
        return arr
    med = int(np.median(sample))

    # thresholds chosen for 2010+ epoch values
    if med > 10**17:        # ns
        return (arr // 1_000_000).astype(np.int64)
    if med > 10**14:        # us
        return (arr // 1_000).astype(np.int64)
    return arr              # already ms

def _standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Output columns: ms, price, qty (with ms in milliseconds).
    """
    # if the file *did* have a header unexpectedly, rename by position fallback
    if df.columns.tolist() != AGG_COLS and df.shape[1] >= 6:
        # positional fallback (Binance Vision is 8 cols)
        df = df.iloc[:, :8].copy()
        df.columns = AGG_COLS[: df.shape[1]]

    if "transact_time" not in df.columns or "price" not in df.columns or "qty" not in df.columns:
        raise RuntimeError(f"Bad columns after standardize: {list(df.columns)}")

    t = pd.to_numeric(df["transact_time"], errors="coerce")
    p = pd.to_numeric(df["price"], errors="coerce")
    q = pd.to_numeric(df["qty"], errors="coerce")

    x = pd.DataFrame({"t": t, "price": p, "qty": q}).dropna()
    if x.empty:
        return x.assign(ms=pd.Series(dtype=np.int64))[["ms","price","qty"]]

    ms = _to_ms(x["t"].astype(np.int64).to_numpy())
    out = pd.DataFrame({
        "ms": ms,
        "price": x["price"].to_numpy(dtype=float),
        "qty": x["qty"].to_numpy(dtype=float),
    })
    return out

def _agg_chunk_to_seconds(x: pd.DataFrame) -> pd.DataFrame:
    sec = (x["ms"].to_numpy(dtype=np.int64) // 1000).astype(np.int64)
    x = x.copy()
    x["sec"] = sec
    x["quote"] = x["price"].values * x["qty"].values

    g = x.groupby("sec", sort=False)

    out = pd.DataFrame({
        "sec": g["sec"].first(),
        "open": g["price"].first(),
        "high": g["price"].max(),
        "low": g["price"].min(),
        "close": g["price"].last(),
        "volume": g["qty"].sum(),
        "quote_volume": g["quote"].sum(),
        "trades": g["price"].size(),
        "first_ms": g["ms"].min(),
        "last_ms": g["ms"].max(),
    }).reset_index(drop=True)

    return out

def _merge_second_partials(parts: pd.DataFrame) -> pd.DataFrame:
    g = parts.groupby("sec", sort=True)

    idx_open = g["first_ms"].idxmin()
    idx_close = g["last_ms"].idxmax()

    sec_index = g.size().index.values

    out = pd.DataFrame({
        "Timestamp": pd.to_datetime(sec_index, unit="s", utc=True),
        "Open": parts.loc[idx_open, "open"].to_numpy(),
        "High": g["high"].max().to_numpy(),
        "Low": g["low"].min().to_numpy(),
        "Close": parts.loc[idx_close, "close"].to_numpy(),
        "Volume": g["volume"].sum().to_numpy(),
        "QuoteVolume": g["quote_volume"].sum().to_numpy(),
        "Trades": g["trades"].sum().to_numpy(),
    }).sort_values("Timestamp").reset_index(drop=True)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).resolve()
    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = _find_files(raw_dir)
    if not files:
        raise RuntimeError(f"No .zip or .csv files found under {raw_dir}")

    print(f"[1s] raw_dir: {raw_dir}")
    print(f"[1s] files: {len(files)} (showing first 5)")
    for p in files[:5]:
        print("  -", p.name)

    all_parts = []
    for fp in files:
        print(f"[1s] reading: {fp.name}", flush=True)
        for chunk in _read_aggtrades_any(fp, chunksize=args.chunksize):
            x = _standardize_cols(chunk)
            if x.empty:
                continue
            all_parts.append(_agg_chunk_to_seconds(x))

    if not all_parts:
        raise RuntimeError("No data aggregated. Check file contents/format.")

    parts = pd.concat(all_parts, ignore_index=True)
    out = _merge_second_partials(parts)

    out.to_parquet(out_path, index=False)
    print(f"\nSaved 1s OHLCV: {out_path}")
    print(f"rows: {len(out):,}")
    print(f"time: {out['Timestamp'].min()} -> {out['Timestamp'].max()}")

if __name__ == "__main__":
    main()
