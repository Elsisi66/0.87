#!/usr/bin/env python3
"""
Convert per-year processed files (e.g. ADAUSDT_2018_proc.csv/.xlsx/...) into one Parquet per symbol.

Input example:
  data/processed/_full/ADAUSDT_2018_proc.csv
  data/processed/_full/ADAUSDT_2019_proc.csv
  ...

Output:
  data/parquet/ADAUSDT.parquet
  data/parquet/ETHUSDT.parquet
  ...

The script tries read_csv first, then falls back to read_excel.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger("build_parquet")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ----------------------------
# Filename parsing
# ----------------------------
# Matches: SYMBOL_YYYY_proc(.ext)
# Example: ADAUSDT_2018_proc.csv, BTCUSDT_2017_proc.xlsx, etc.
FNAME_RE = re.compile(r"^(?P<sym>[A-Z0-9]+)_(?P<year>\d{4})_proc$", re.IGNORECASE)


def parse_symbol_year(path: Path) -> Optional[Tuple[str, int]]:
    stem = path.stem  # without extension
    m = FNAME_RE.match(stem)
    if not m:
        return None
    sym = m.group("sym").upper()
    year = int(m.group("year"))
    return sym, year


# ----------------------------
# Reading helpers
# ----------------------------
def _try_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        # common delimiters: comma, semicolon
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")
    except Exception:
        return None


def _try_read_excel(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_excel(path)
    except Exception:
        return None


def read_any_table(path: Path) -> pd.DataFrame:
    # Some Windows setups hide extensions; you might have files with no suffix.
    # We'll still attempt CSV then Excel.
    df = _try_read_csv(path)
    if df is not None and len(df) > 0:
        return df

    df = _try_read_excel(path)
    if df is not None and len(df) > 0:
        return df

    # If suffix is missing and the real file is e.g. CSV, read_csv should have worked;
    # if it didn't, the file may be empty/corrupt.
    raise ValueError(f"Could not read file as CSV or Excel: {path}")


# ----------------------------
# Normalization
# ----------------------------
TIME_CANDIDATES = [
    "timestamp", "Timestamp", "time", "Time", "date", "Date",
    "open_time", "Open time", "Open Time", "datetime", "Datetime",
]

COL_MAP = {
    # open
    "open": "open", "Open": "open", "OPEN": "open",
    # high
    "high": "high", "High": "high", "HIGH": "high",
    # low
    "low": "low", "Low": "low", "LOW": "low",
    # close
    "close": "close", "Close": "close", "CLOSE": "close",
    # volume
    "volume": "volume", "Volume": "volume", "VOL": "volume", "vol": "volume",
    "quote_volume": "volume", "Quote Volume": "volume",
}

REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]


def find_time_column(df: pd.DataFrame, override: Optional[str] = None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"--time-col '{override}' not found in columns: {list(df.columns)}")
        return override

    for c in TIME_CANDIDATES:
        if c in df.columns:
            return c

    # try fuzzy: any col containing "time" or "date"
    for c in df.columns:
        lc = str(c).lower()
        if "time" in lc or "date" in lc:
            return c

    raise ValueError(f"No obvious time column found. Columns: {list(df.columns)}")


def normalize_ohlcv(df: pd.DataFrame, time_col_override: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()

    # Standardize column names for OHLCV
    renamed = {}
    for c in df.columns:
        if c in COL_MAP:
            renamed[c] = COL_MAP[c]
    df = df.rename(columns=renamed)

    # Find timestamp
    tcol = find_time_column(df, time_col_override)
    if tcol != "timestamp":
        df = df.rename(columns={tcol: "timestamp"})

    # Parse timestamp as UTC
    # Handles: unix ms, unix s, ISO strings
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        # guess ms vs s
        # if values look like 1_600_000_000_000 -> ms
        median = float(pd.Series(ts).dropna().median())
        unit = "ms" if median > 10_000_000_000 else "s"
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit=unit, utc=True, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Keep only needed cols if present + required check
    # Some files may include extra columns; we ignore them.
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    out = df[REQUIRED].copy()

    # Coerce numeric columns
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop bad rows
    out = out.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])

    # Sort, dedup on timestamp
    out = out.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    # Index as UTC DatetimeIndex
    out = out.set_index("timestamp")
    out.index = out.index.tz_convert("UTC") if out.index.tz is not None else out.index.tz_localize("UTC")

    return out


# ----------------------------
# Main conversion
# ----------------------------
def collect_files(input_dir: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        parsed = parse_symbol_year(p)
        if not parsed:
            continue
        sym, _year = parsed
        groups.setdefault(sym, []).append(p)

    # sort by year inferred from filename
    for sym, files in groups.items():
        files.sort(key=lambda x: parse_symbol_year(x)[1])  # type: ignore
    return groups


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Requires pyarrow or fastparquet
    df.to_parquet(out_path, index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/processed/_full", help="Folder with *_YYYY_proc files")
    ap.add_argument("--output", type=str, default="data/parquet", help="Folder to write <SYMBOL>.parquet")
    ap.add_argument("--time-col", type=str, default=None, help="Override timestamp column name (if autodetect fails)")
    ap.add_argument("--symbols", type=str, default=None,
                    help="Comma-separated symbols to convert (e.g. BTCUSDT,ETHUSDT,SOLUSDT). Default: all found.")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise SystemExit(f"Input folder does not exist: {input_dir}")

    groups = collect_files(input_dir)
    if not groups:
        raise SystemExit(f"No files matched pattern SYMBOL_YYYY_proc in: {input_dir}")

    wanted = None
    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    total_syms = 0
    for sym, files in sorted(groups.items()):
        if wanted and sym not in wanted:
            continue

        log.info(f"\n=== {sym} | {len(files)} yearly files ===")
        chunks = []
        for f in files:
            try:
                raw = read_any_table(f)
                norm = normalize_ohlcv(raw, time_col_override=args.time_col)
                chunks.append(norm)
                log.info(f"  OK  {f.name} -> rows={len(norm)}  [{norm.index.min()} .. {norm.index.max()}]")
            except Exception as e:
                log.error(f"  BAD {f.name}: {e}")

        if not chunks:
            log.warning(f"SKIP {sym}: no readable chunks")
            continue

        df = pd.concat(chunks).sort_index()
        df = df[~df.index.duplicated(keep="last")]  # final dedup

        out_path = output_dir / f"{sym}.parquet"
        try:
            write_parquet(df, out_path)
        except Exception as e:
            raise SystemExit(
                f"Failed to write parquet for {sym}. You likely need pyarrow.\n"
                f"Error: {e}\n"
                f"Install: pip install pyarrow"
            )

        log.info(f"  WRITE {out_path} | rows={len(df)} | range=[{df.index.min()} .. {df.index.max()}]")
        total_syms += 1

    log.info(f"\nDone. Wrote {total_syms} symbols to {output_dir}")


if __name__ == "__main__":
    main()
