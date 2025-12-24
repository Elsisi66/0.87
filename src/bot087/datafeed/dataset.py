from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.bot087.datafeed.loader import load_symbol_history
from src.bot087.features.build import build_features


PROJECT_ROOT = Path(__file__).resolve()
for p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
    if (p / "data").is_dir() and (p / "src").is_dir():
        PROJECT_ROOT = p
        break

FULL_DIR = PROJECT_ROOT / "data" / "processed" / "_full"
FULL_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetReport:
    symbol: str
    rows: int
    start_ts: str
    end_ts: str
    cached_path: str


def load_build_cache(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tf: str = "1h",
    cache: bool = True,
) -> Tuple[pd.DataFrame, DatasetReport]:
    """
    Load yearly *_proc.csv files, recompute indicators on the merged history,
    and optionally cache a single parquet file.
    """
    df = load_symbol_history(symbol, start=start, end=end, keep_extra_columns=True)
    df = build_features(df)

    cache_path = FULL_DIR / f"{symbol.upper()}_{tf}_full.parquet"
    if cache:
        df.to_parquet(cache_path, index=False)

    rep = DatasetReport(
        symbol=symbol.upper(),
        rows=len(df),
        start_ts=str(df["Timestamp"].iloc[0]),
        end_ts=str(df["Timestamp"].iloc[-1]),
        cached_path=str(cache_path),
    )
    return df, rep
