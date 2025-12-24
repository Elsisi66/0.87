from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


def _project_root_from_file() -> Path:
    """
    Find the repo root by walking up until we see both /data and /src folders.
    Works regardless of where you run the script from.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data").is_dir() and (p / "src").is_dir():
            return p
    raise RuntimeError(f"Could not find project root starting from {here}")


PROJECT_ROOT = _project_root_from_file()
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

YEAR_RE = re.compile(r"^(?P<symbol>[A-Z0-9]+)_(?P<year>\d{4})(?:_.*)?\.csv$", re.IGNORECASE)
BASE_COLS = ["Timestamp", "Open", "High", "Low", "Close", "Volume"]


def _find_files(symbol: str) -> list[Path]:
    symbol = symbol.upper()
    files = []
    for p in PROCESSED_DIR.glob(f"{symbol}_*.csv"):
        m = YEAR_RE.match(p.name)
        if m and m.group("symbol").upper() == symbol:
            files.append(p)
    # sort by year
    return sorted(files, key=lambda x: int(YEAR_RE.match(x.name).group("year")))


def load_symbol_history(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    keep_extra_columns: bool = True,
) -> pd.DataFrame:
    files = _find_files(symbol)
    if not files:
        raise FileNotFoundError(f"No yearly CSVs found for {symbol} in {PROCESSED_DIR}")

    dfs = []
    for fp in files:
        df = pd.read_csv(fp)

        missing = [c for c in BASE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{fp.name} missing base columns: {missing}")

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if not keep_extra_columns:
            df = df[BASE_COLS].copy()

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    if start is not None:
        out = out[out["Timestamp"] >= pd.to_datetime(start, utc=True)]
    if end is not None:
        out = out[out["Timestamp"] < pd.to_datetime(end, utc=True)]

    return out
