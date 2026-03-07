#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd

import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _load_df(symbol: str, tf: str) -> pd.DataFrame:
    full_dir = PROJECT_ROOT / "data" / "processed" / "_full"
    parquet_fp = full_dir / f"{symbol}_{tf}_full.parquet"
    if parquet_fp.exists():
        df = pd.read_parquet(parquet_fp)
    else:
        proc_dir = PROJECT_ROOT / "data" / "processed"
        files = sorted(proc_dir.glob(f"{symbol}_*_proc.csv"))
        if not files:
            raise FileNotFoundError(f"No parquet and no yearly CSVs for {symbol}")
        df = pd.concat([pd.read_csv(fp) for fp in files], ignore_index=True)

    if "Timestamp" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "Timestamp"})
    if "Timestamp" not in df.columns:
        raise ValueError(f"{symbol}: missing Timestamp column. cols={list(df.columns)[:30]}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df


def _unwrap_params(raw: dict) -> dict:
    if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
        return raw["params"]
    return raw


def _find_latest_params(symbol: str) -> Path:
    """
    Prefer GA outputs under output/ that contain symbol and 'best'/'param'.
    Picks the most recently modified candidate.
    """
    out_dir = PROJECT_ROOT / "output"
    if not out_dir.exists():
        raise FileNotFoundError("output/ folder not found")

    cands = []
    for fp in out_dir.rglob("*.json"):
        n = fp.name.lower()
        if symbol.lower() in n and ("best" in n or "param" in n):
            cands.append(fp)

    if not cands:
        raise FileNotFoundError(f"Could not auto-find params json for {symbol} under output/")

    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _trades_to_df(trades) -> pd.DataFrame:
    if trades is None:
        return pd.DataFrame()
    if isinstance(trades, pd.DataFrame):
        return trades.copy()
    if isinstance(trades, list):
        if len(trades) == 0:
            return pd.DataFrame()
        if isinstance(trades[0], dict):
            return pd.DataFrame(trades)
        # fallback for objects
        return pd.DataFrame([getattr(t, "__dict__", {"trade": str(t)}) for t in trades])
    return pd.DataFrame()


def _pick_col(df: pd.DataFrame, options: list[str]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None


def _yearly_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()

    exit_col = _pick_col(trades_df, ["exit_time", "exit_ts", "exit_timestamp", "ExitTime", "ExitTimestamp"])
    entry_col = _pick_col(trades_df, ["entry_time", "entry_ts", "entry_timestamp", "EntryTime", "EntryTimestamp"])
    pnl_col = _pick_col(trades_df, ["pnl", "net_pnl", "profit", "pnl_usd", "PnL", "Profit"])

    if pnl_col is None:
        return pd.DataFrame()

    # choose year from exit time if possible, else entry time
    tcol = exit_col or entry_col
    if tcol is None:
        return pd.DataFrame()

    ts = pd.to_datetime(trades_df[tcol], utc=True, errors="coerce")
    tmp = trades_df.copy()
    tmp["_year"] = ts.dt.year
    tmp["_pnl"] = pd.to_numeric(tmp[pnl_col], errors="coerce").fillna(0.0)

    def pf(x):
        wins = x[x > 0].sum()
        losses = -x[x < 0].sum()
        return float(wins / losses) if losses > 0 else float("inf") if wins > 0 else 0.0

    out = tmp.groupby("_year")["_pnl"].agg(
        trades="count",
        net="sum",
        avg="mean",
    ).reset_index().rename(columns={"_year": "year"})

    # winrate + PF
    winrate = tmp.groupby("_year")["_pnl"].apply(lambda s: float((s > 0).mean() * 100.0)).reset_index(name="winrate_pct")
    pfv = tmp.groupby("_year")["_pnl"].apply(pf).reset_index(name="pf")

    out = out.merge(winrate, left_on="year", right_on="_year", how="left").drop(columns=["_year"])
    out = out.merge(pfv, left_on="year", right_on="_year", how="left").drop(columns=["_year"])
    return out.sort_values("year")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--params", default=None, help="Path to params json. If omitted, auto-picks latest under output/")
    ap.add_argument("--cycle3", action="store_true", help="Force trade_cycles=[3] for this backtest")
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    args = ap.parse_args()

    symbol = args.symbol.strip().upper()

    # Import AFTER path is set up by running from repo root
    from src.bot087.optim.ga import _norm_params, _ensure_indicators, run_backtest_long_only

    df = _load_df(symbol, tf=args.tf)
    print(f"[DATA] {symbol} rows={len(df)} first={df['Timestamp'].iloc[0]} last={df['Timestamp'].iloc[-1]}")

    params_fp = Path(args.params) if args.params else _find_latest_params(symbol)
    raw = json.load(open(params_fp))
    p = _unwrap_params(raw)

    if args.cycle3:
        p["trade_cycles"] = [3]

    p = _norm_params(p)

    print(f"[PARAMS] file={params_fp}")
    print(f"[PARAMS] trade_cycles={p.get('trade_cycles')}")

    df2 = _ensure_indicators(df.copy(), p)

    trades, metrics = run_backtest_long_only(
        df2,
        symbol=symbol,
        p=p,
        initial_equity=args.initial_equity,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
    )

    print("\n=== SUMMARY ===")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    trades_df = _trades_to_df(trades)
    yearly = _yearly_summary(trades_df)

    run_dir = PROJECT_ROOT / "output" / "backtests" / symbol / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (run_dir / "used_params.json").write_text(json.dumps({"params": p, "_source": str(params_fp)}, indent=2, default=str))

    if not trades_df.empty:
        trades_df.to_csv(run_dir / "trades.csv", index=False)

    if yearly is not None and not yearly.empty:
        yearly.to_csv(run_dir / "yearly.csv", index=False)
        print("\n=== YEARLY ===")
        print(yearly.to_string(index=False))

    print(f"\n[SAVED] {run_dir}")

if __name__ == "__main__":
    main()
