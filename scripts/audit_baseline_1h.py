#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.execution.baseline_audit import audit_baseline
from src.bot087.optim.ga import _ensure_indicators, _norm_params, run_backtest_long_only


def _load_params(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
        raw = raw["params"]
    p = _norm_params(dict(raw))
    p["cycle_shift"] = 1
    p["two_candle_confirm"] = False
    p["require_trade_cycles"] = True
    return p


def _load_df(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing 1h parquet: {fp}")
    df = pd.read_parquet(fp)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if start:
        df = df[df["Timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df["Timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline 1h sanity audit for long-only params.")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--params", required=True, help="params json path")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--equity-cap", type=float, default=10_000_000.0)
    args = ap.parse_args()

    sym = str(args.symbol).upper()
    p = _load_params(Path(args.params).resolve())
    df = _load_df(sym, start=args.start or None, end=args.end or None)
    dfi = _ensure_indicators(df.copy(), p)

    trades, m = run_backtest_long_only(
        df=dfi,
        symbol=sym,
        p=p,
        initial_equity=float(args.initial_equity),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        collect_trades=True,
    )
    tr = pd.DataFrame(trades)
    out = PROJECT_ROOT / "artifacts" / "audit" / sym / "baseline_audit.json"
    audit = audit_baseline(
        symbol=sym,
        df=dfi,
        trades_df=tr,
        initial_equity=float(args.initial_equity),
        max_allocation=float(p.get("max_allocation", 0.7)),
        equity_cap=float(args.equity_cap),
        out_path=out,
    )
    print(json.dumps({"symbol": sym, "metrics": m, "audit_ok": audit["ok"], "audit_path": str(out)}, indent=2))


if __name__ == "__main__":
    main()

