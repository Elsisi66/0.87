#!/usr/bin/env python3
import argparse
import io
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Iterator, Optional, Tuple, List

import pandas as pd
import requests
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


BINANCE_VISION_BASE = "https://data.binance.vision"


@dataclass(frozen=True)
class VisionPath:
    market: str  # "spot" or "futures/um" etc
    freq: str    # "monthly" or "daily"
    symbol: str
    y: int
    m: int
    d: Optional[int] = None

    def url(self) -> str:
        if self.freq == "monthly":
            # e.g. https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01.zip
            return (
                f"{BINANCE_VISION_BASE}/data/{self.market}/monthly/aggTrades/{self.symbol}/"
                f"{self.symbol}-aggTrades-{self.y:04d}-{self.m:02d}.zip"
            )
        # daily
        return (
            f"{BINANCE_VISION_BASE}/data/{self.market}/daily/aggTrades/{self.symbol}/"
            f"{self.symbol}-aggTrades-{self.y:04d}-{self.m:02d}-{int(self.d):02d}.zip"
        )

    def raw_zip_name(self) -> str:
        if self.freq == "monthly":
            return f"{self.symbol}-aggTrades-{self.y:04d}-{self.m:02d}.zip"
        return f"{self.symbol}-aggTrades-{self.y:04d}-{self.m:02d}-{int(self.d):02d}.zip"


def _dt_utc(s: str) -> pd.Timestamp:
    # expects YYYY-MM-DD
    return pd.Timestamp(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc))


def _iter_months(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[Tuple[int, int]]:
    # inclusive start month, inclusive end month (end is exclusive in overall logic)
    cur_y, cur_m = start.year, start.month
    end_y, end_m = end.year, end.month
    while (cur_y, cur_m) <= (end_y, end_m):
        yield cur_y, cur_m
        if cur_m == 12:
            cur_y += 1
            cur_m = 1
        else:
            cur_m += 1


def _iter_days(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[pd.Timestamp]:
    cur = start.normalize()
    while cur < end.normalize():
        yield cur
        cur += pd.Timedelta(days=1)


def _http_head_ok(sess: requests.Session, url: str, timeout: int = 30) -> bool:
    try:
        r = sess.head(url, timeout=timeout, allow_redirects=True)
        return r.status_code == 200
    except Exception:
        return False


def _download_file(sess: requests.Session, url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    with sess.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0")) or None

        pbar = None
        if tqdm is not None:
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"DL {out_path.name}", leave=False)

        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                if pbar:
                    pbar.update(len(chunk))

        if pbar:
            pbar.close()

    os.replace(tmp_path, out_path)


def _open_zip_csv(zip_path: Path) -> io.BufferedReader:
    zf = zipfile.ZipFile(zip_path, "r")
    # usually exactly one csv inside
    members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
    if not members:
        raise RuntimeError(f"No CSV found in zip: {zip_path}")
    # return a file-like handle (caller must close zipfile via context) -> we'll manage with BytesIO instead
    # safer: read the csv bytes into memory? too big.
    # We will return (zf, handle) instead; see below.
    raise NotImplementedError


def _read_aggtrades_chunks_from_zip(zip_path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """
    Supports both header/no-header variants.

    Expected columns for aggTrades:
      agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker, was_best_match
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV found in zip: {zip_path}")
        name = members[0]
        with zf.open(name, "r") as raw:
            # pandas wants text for CSV
            bio = io.TextIOWrapper(raw, encoding="utf-8")

            # Try headered first
            try:
                it = pd.read_csv(bio, chunksize=chunksize)
                first = next(it)
                cols = [c.lower() for c in first.columns.tolist()]
                if "transact_time" not in cols and "time" not in cols:
                    raise ValueError("No transact_time column detected")
                yield first
                for ch in it:
                    yield ch
                return
            except Exception:
                pass

        # reopen and parse no-header
        with zf.open(name, "r") as raw2:
            bio2 = io.TextIOWrapper(raw2, encoding="utf-8")
            names = [
                "agg_trade_id",
                "price",
                "quantity",
                "first_trade_id",
                "last_trade_id",
                "transact_time",
                "is_buyer_maker",
                "was_best_match",
            ]
            it2 = pd.read_csv(bio2, header=None, names=names, chunksize=chunksize)
            for ch in it2:
                yield ch


def _standardize_chunk(df: pd.DataFrame) -> pd.DataFrame:
    # normalize col names
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)

    # accept some variants
    if "transact_time" not in df.columns:
        if "time" in df.columns:
            df = df.rename(columns={"time": "transact_time"})
        else:
            raise RuntimeError(f"Chunk missing transact_time/time column. cols={list(df.columns)}")

    # required
    for k in ["price", "quantity", "transact_time"]:
        if k not in df.columns:
            raise RuntimeError(f"Chunk missing required column {k}. cols={list(df.columns)}")

    out = pd.DataFrame()
    out["Timestamp"] = pd.to_datetime(pd.to_numeric(df["transact_time"], errors="coerce"), unit="ms", utc=True)
    out["Price"] = pd.to_numeric(df["price"], errors="coerce")
    out["Qty"] = pd.to_numeric(df["quantity"], errors="coerce")

    # optional
    if "agg_trade_id" in df.columns:
        out["AggTradeId"] = pd.to_numeric(df["agg_trade_id"], errors="coerce").astype("Int64")
    if "is_buyer_maker" in df.columns:
        # Binance uses True/False or 0/1 depending on dump
        v = df["is_buyer_maker"]
        if v.dtype == bool:
            out["IsBuyerMaker"] = v
        else:
            out["IsBuyerMaker"] = v.astype(str).str.lower().isin(["true", "1", "t", "yes"])

    out = out.dropna(subset=["Timestamp", "Price", "Qty"]).reset_index(drop=True)
    return out


def _write_partitioned_parquet(df: pd.DataFrame, out_root: Path, symbol: str, tag: str, part_idx: int) -> int:
    """
    Writes parquet partitioned by date (YYYY-MM-DD) under:
      out_root/<SYMBOL>/date=YYYY-MM-DD/part_<tag>_<part_idx>.parquet
    Returns next part_idx.
    """
    if df.empty:
        return part_idx

    df["date"] = df["Timestamp"].dt.strftime("%Y-%m-%d")
    for d, sub in df.groupby("date", sort=False):
        ddir = out_root / symbol / f"date={d}"
        ddir.mkdir(parents=True, exist_ok=True)
        out_path = ddir / f"part_{tag}_{part_idx:06d}.parquet"
        # keep it slim
        keep_cols = [c for c in ["Timestamp", "Price", "Qty", "AggTradeId", "IsBuyerMaker"] if c in sub.columns]
        sub = sub[keep_cols].sort_values("Timestamp").reset_index(drop=True)
        sub.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        part_idx += 1
    return part_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--market", default="spot", help='Binance Vision market path. Usually "spot".')
    ap.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD (UTC, exclusive)")
    ap.add_argument("--raw-dir", default="data/raw/binance_vision", help="where zip files will be saved")
    ap.add_argument("--processed-dir", default="data/processed/sec_trades", help="where parquet partitions are written")
    ap.add_argument("--prefer-monthly", action="store_true", help="try monthly zips first (recommended)")
    ap.add_argument("--chunksize", type=int, default=2_000_000, help="CSV chunk rows to process at once")
    ap.add_argument("--redownload", action="store_true", help="force redownload even if zip exists")
    args = ap.parse_args()

    symbol = args.symbol.upper()
    start = _dt_utc(args.start)
    end = _dt_utc(args.end)
    if end <= start:
        raise SystemExit("end must be > start")

    raw_dir = Path(args.raw_dir).resolve()
    proc_dir = Path(args.processed_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    sess = requests.Session()
    sess.headers.update({"User-Agent": "bot087-secdata/1.0"})

    # Decide file list
    candidates: List[VisionPath] = []

    if args.prefer_monthly:
        for y, m in _iter_months(start, end - pd.Timedelta(seconds=1)):
            candidates.append(VisionPath(args.market, "monthly", symbol, y, m))
    else:
        for dts in _iter_days(start, end):
            candidates.append(VisionPath(args.market, "daily", symbol, dts.year, dts.month, int(dts.day)))

    # If monthly is preferred, but a monthly zip is missing, fallback to daily for that month.
    resolved: List[VisionPath] = []
    if args.prefer_monthly:
        for y, m in _iter_months(start, end - pd.Timedelta(seconds=1)):
            vp_m = VisionPath(args.market, "monthly", symbol, y, m)
            if _http_head_ok(sess, vp_m.url()):
                resolved.append(vp_m)
            else:
                # fallback daily in that month within [start,end)
                month_start = pd.Timestamp(datetime(y, m, 1, tzinfo=timezone.utc))
                month_end = (month_start + pd.offsets.MonthBegin(1)).to_timestamp().tz_convert("UTC")
                lo = max(start, month_start)
                hi = min(end, month_end)
                for dts in _iter_days(lo, hi):
                    resolved.append(VisionPath(args.market, "daily", symbol, dts.year, dts.month, int(dts.day)))
    else:
        resolved = candidates

    if not resolved:
        raise SystemExit("No files resolved for the given range.")

    print(f"[secdata] symbol={symbol} market={args.market} range={start} -> {end}")
    print(f"[secdata] raw_dir={raw_dir}")
    print(f"[secdata] processed_dir={proc_dir}")
    print(f"[secdata] files_to_get={len(resolved)} (monthly preferred={args.prefer_monthly})")

    part_idx = 0
    total_rows_written = 0

    outer = tqdm(resolved, desc="files", unit="file") if tqdm else resolved
    for vp in outer:
        url = vp.url()
        zip_path = raw_dir / args.market / "aggTrades" / symbol / vp.raw_zip_name()

        if args.redownload or not zip_path.exists():
            # only download if file exists remotely
            if not _http_head_ok(sess, url):
                # silently skip missing files (Binance Vision coverage varies early)
                if tqdm is None:
                    print(f"[skip] missing remote: {url}")
                continue
            if tqdm is None:
                print(f"[dl] {url} -> {zip_path}")
            _download_file(sess, url, zip_path)
        else:
            if tqdm is None:
                print(f"[ok] have {zip_path.name}")

        tag = zip_path.stem.replace(symbol + "-aggTrades-", "")
        # process zip -> parquet partitions
        for ch in _read_aggtrades_chunks_from_zip(zip_path, chunksize=args.chunksize):
            std = _standardize_chunk(ch)
            # filter exact range
            std = std[(std["Timestamp"] >= start) & (std["Timestamp"] < end)]
            if std.empty:
                continue
            before = len(std)
            part_idx = _write_partitioned_parquet(std, proc_dir, symbol, tag=tag, part_idx=part_idx)
            total_rows_written += before

    print(f"[done] wrote_rows={total_rows_written:,}")
    print(f"[done] parquet_root={proc_dir / symbol}")


if __name__ == "__main__":
    main()
