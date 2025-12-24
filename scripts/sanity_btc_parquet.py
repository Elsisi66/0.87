# scripts/check_willr_violations.py
# Run:  python scripts/check_willr_violations.py
#
# Prints rows where WILLR is outside [-100, 0] and shows key columns.

from pathlib import Path
import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
TF = "1h"
REL_PARQUET_PATH = Path("data/processed/_full") / f"{SYMBOL}_{TF}_full.parquet"

SHOW_COLS = [
    "Timestamp", "Open", "High", "Low", "Close", "Volume", "WILLR",
    "RSI", "ATR", "ADX", "EMA_200"
]

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def main():
    root = project_root()
    path = root / REL_PARQUET_PATH
    print(f"[PATH] {path}")

    df = pd.read_parquet(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    if "WILLR" not in df.columns:
        raise ValueError("No WILLR column found.")

    w = pd.to_numeric(df["WILLR"], errors="coerce")
    mask = (w < -100.0) | (w > 0.0) | (~np.isfinite(w.to_numpy(dtype=float)))
    bad = df.loc[mask].copy()

    print(f"[COUNT] total_rows={len(df)} willr_violations={len(bad)}")

    if bad.empty:
        print("✅ No WILLR violations.")
        return

    # Show the worst offenders first
    bad["WILLR_abs_violation"] = np.maximum(bad["WILLR"] - 0.0, -100.0 - bad["WILLR"])
    bad = bad.sort_values("WILLR_abs_violation", ascending=False)

    cols = [c for c in SHOW_COLS if c in bad.columns]
    print("\n[VIOLATIONS] showing all violating rows:\n")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 50, "display.width", 160):
        print(bad[cols].to_string(index=False))

    # Quick stats
    wbad = pd.to_numeric(bad["WILLR"], errors="coerce")
    print("\n[STATS]")
    print(f"min={float(wbad.min()):.12f} max={float(wbad.max()):.12f}")
    print(f"count_<-100={(wbad < -100).sum()} count_>0={(wbad > 0).sum()}")

if __name__ == "__main__":
    main()
