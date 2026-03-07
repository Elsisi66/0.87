#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger("build_full_parquets")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FNAME_RE = re.compile(r"^(?P<sym>[A-Z0-9]+)_(?P<year>\d{4})_proc$", re.IGNORECASE)

def parse_symbol_year(path: Path) -> Optional[Tuple[str, int]]:
    m = FNAME_RE.match(path.stem)
    if not m:
        return None
    return m.group("sym").upper(), int(m.group("year"))

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
    for sym, files in groups.items():
        files.sort(key=lambda x: parse_symbol_year(x)[1])  # type: ignore
    return groups

def read_csv_any_sep(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")

def ensure_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    if "Timestamp" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "Timestamp"})
    if "Timestamp" not in df.columns:
        raise ValueError(f"No Timestamp column. Columns={list(df.columns)[:25]}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").drop_duplicates(subset=["Timestamp"], keep="last")
    return df.reset_index(drop=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed")
    ap.add_argument("--output", default="data/processed/_full")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--symbols", default=None)
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = collect_files(input_dir)
    if not groups:
        raise SystemExit(f"No files matched pattern SYMBOL_YYYY_proc in: {input_dir}")

    wanted = None
    if args.symbols:
        wanted = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}

    wrote = 0
    for sym, files in sorted(groups.items()):
        if wanted and sym not in wanted:
            continue

        log.info(f"=== {sym} | {len(files)} yearly files ===")
        dfs = []
        for fp in files:
            df = ensure_timestamp_col(read_csv_any_sep(fp))
            dfs.append(df)

        full = ensure_timestamp_col(pd.concat(dfs, ignore_index=True))
        out_fp = output_dir / f"{sym}_{args.tf}_full.parquet"
        full.to_parquet(out_fp, index=False)

        log.info(f"WRITE {out_fp} rows={len(full)} range=[{full['Timestamp'].iloc[0]} .. {full['Timestamp'].iloc[-1]}]")
        wrote += 1

    log.info(f"Done. Wrote {wrote} full parquets into {output_dir}")

if __name__ == "__main__":
    main()
