from __future__ import annotations

from pathlib import Path
import pandas as pd


def _project_root_from_file() -> Path:
    """
    Locate project root by walking upward until we see both /data and /src.
    Works regardless of where you run scripts from.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data").is_dir() and (p / "src").is_dir():
            return p
    raise RuntimeError(f"Could not find project root starting from {here}")


def load_full_parquet(symbol: str, tf: str = "1h") -> pd.DataFrame:
    """
    Load the cached full-history dataset created by build_btc_full.py:
    data/processed/_full/{SYMBOL}_{TF}_full.parquet
    """
    root = _project_root_from_file()
    fp = root / "data" / "processed" / "_full" / f"{symbol.upper()}_{tf}_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing cached dataset: {fp}")

    df = pd.read_parquet(fp)

    # Normalize index/time just in case
    if "Timestamp" not in df.columns:
        raise ValueError(f"'Timestamp' column not found in {fp}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df
