#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

# --- bootstrap like your GA scripts ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class TestCase:
    name: str
    fee_bps: float
    slip_bps: float
    risk_per_trade: Optional[float] = None
    max_allocation: Optional[float] = None
    start: Optional[str] = None   # YYYY-MM-DD
    end: Optional[str] = None     # YYYY-MM-DD


SYMBOLS_DEFAULT = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "AVAXUSDT", "BNBUSDT", "XRPUSDT"]


def _load_df(symbol: str, tf: str = "1h") -> pd.DataFrame:
    full_dir = PROJECT_ROOT / "data" / "processed" / "_full"
    parquet_fp = full_dir / f"{symbol}_{tf}_full.parquet"

    if parquet_fp.exists():
        df = pd.read_parquet(parquet_fp)
    else:
        proc_dir = PROJECT_ROOT / "data" / "processed"
        files = sorted(proc_dir.glob(f"{symbol}_*_proc.csv"))
        if not files:
            raise FileNotFoundError(f"No parquet and no yearly processed CSVs for {symbol}")
        df = pd.concat([pd.read_csv(fp) for fp in files], ignore_index=True)

    if "Timestamp" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "Timestamp"})
    if "Timestamp" not in df.columns:
        raise ValueError(f"{symbol}: missing Timestamp column. cols={list(df.columns)[:30]}")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df


def _unwrap_params(d: dict) -> dict:
    if isinstance(d, dict) and isinstance(d.get("params"), dict):
        return d["params"]
    return d


def _load_active_params(symbol: str) -> Tuple[dict, Path]:
    fp = PROJECT_ROOT / "data" / "metadata" / "params" / f"{symbol}_active_params.json"
    if not fp.exists():
        raise FileNotFoundError(f"Missing active params: {fp}")
    raw = json.load(open(fp, "r"))
    return _unwrap_params(raw), fp


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
        return pd.DataFrame([getattr(t, "__dict__", {"trade": str(t)}) for t in trades])
    return pd.DataFrame()


def _find_pnl_col(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    cands = [c for c in df.columns if any(k in c.lower() for k in ["pnl", "profit", "net"])]
    for c in cands:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            return c
    return None


def _metrics_from_trades(trades_df: pd.DataFrame) -> Dict[str, float]:
    out = {
        "trades": 0.0,
        "wins": 0.0,
        "losses": 0.0,
        "win_rate_pct": 0.0,
        "net_profit": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0,
        "min_pnl": 0.0,
        "max_pnl": 0.0,
    }
    if trades_df.empty:
        return out

    pnl_col = _find_pnl_col(trades_df)
    if pnl_col is None:
        return out

    p = pd.to_numeric(trades_df[pnl_col], errors="coerce").fillna(0.0)
    wins = p[p > 0]
    losses = p[p < 0]

    out["trades"] = float(len(p))
    out["wins"] = float((p > 0).sum())
    out["losses"] = float((p < 0).sum())
    out["win_rate_pct"] = float((p > 0).mean() * 100.0) if len(p) else 0.0
    out["net_profit"] = float(p.sum())
    out["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    out["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    if (-losses.sum()) > 0:
        out["profit_factor"] = float(wins.sum() / (-losses.sum()))
    else:
        out["profit_factor"] = float("inf") if wins.sum() > 0 else 0.0
    out["min_pnl"] = float(p.min()) if len(p) else 0.0
    out["max_pnl"] = float(p.max()) if len(p) else 0.0
    return out


def _slice_df(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is None and end is None:
        return df
    s = pd.Timestamp(start, tz="UTC") if start else None
    e = pd.Timestamp(end, tz="UTC") if end else None
    out = df
    if s is not None:
        out = out[out["Timestamp"] >= s]
    if e is not None:
        out = out[out["Timestamp"] <= e]
    return out.reset_index(drop=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(SYMBOLS_DEFAULT))
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--force-cycle3", action="store_true", default=False)
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    # Import engine
    from src.bot087.optim.ga import _norm_params, _ensure_indicators, run_backtest_long_only

    # Define test suite
    tests: List[TestCase] = [
        TestCase("BASE", fee_bps=7.0, slip_bps=2.0),
        TestCase("STRESS", fee_bps=25.0, slip_bps=10.0),
        TestCase("EXTREME", fee_bps=50.0, slip_bps=20.0),
        TestCase("CONS_STRESS", fee_bps=25.0, slip_bps=10.0, risk_per_trade=0.005, max_allocation=0.30),

        # Subperiod splits with conservative sizing + stress costs
        TestCase("P1_2018_2020", fee_bps=25.0, slip_bps=10.0, risk_per_trade=0.005, max_allocation=0.30, start="2018-01-01", end="2020-12-31"),
        TestCase("P2_2021_2023", fee_bps=25.0, slip_bps=10.0, risk_per_trade=0.005, max_allocation=0.30, start="2021-01-01", end="2023-12-31"),
        TestCase("P3_2024_2025", fee_bps=25.0, slip_bps=10.0, risk_per_trade=0.005, max_allocation=0.30, start="2024-01-01", end="2025-12-31"),
    ]

    out_dir = PROJECT_ROOT / "output" / "stress_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for sym in symbols:
        try:
            p_active, p_path = _load_active_params(sym)
        except Exception as e:
            print(f"[FAIL] {sym}: {e}")
            continue

        # force cycle 3 only
        if args.force_cycle3:
            p_active = dict(p_active)
            p_active["trade_cycles"] = [3]

        # normalize once
        p_norm = _norm_params(p_active)

        # load data
        try:
            df = _load_df(sym, tf=args.tf)
        except Exception as e:
            print(f"[FAIL] {sym}: data load error: {e}")
            continue

        # indicators computed once (indicators don't depend on fees/slippage/risk)
        try:
            df_ind = _ensure_indicators(df.copy(), p_norm)
        except Exception as e:
            print(f"[FAIL] {sym}: indicator error: {e}")
            continue

        for tc in tests:
            # build params for this test
            p_test = dict(p_norm)
            if tc.risk_per_trade is not None:
                p_test["risk_per_trade"] = float(tc.risk_per_trade)
            if tc.max_allocation is not None:
                p_test["max_allocation"] = float(tc.max_allocation)
            p_test = _norm_params(p_test)

            df_slice = _slice_df(df_ind, tc.start, tc.end)
            if df_slice.empty:
                rows.append({
                    "symbol": sym, "test": tc.name, "trades": 0, "net_profit": 0,
                    "win_rate_pct": 0, "profit_factor": 0, "min_pnl": 0, "max_pnl": 0,
                    "note": "empty_range"
                })
                continue

            trades, metrics = run_backtest_long_only(
                df_slice,
                symbol=sym,
                p=p_test,
                initial_equity=10_000.0,
                fee_bps=tc.fee_bps,
                slippage_bps=tc.slip_bps,
            )

            tdf = _trades_to_df(trades)
            m = _metrics_from_trades(tdf)

            rows.append({
                "symbol": sym,
                "test": tc.name,
                "trade_cycles": str(p_test.get("trade_cycles")),
                "fee_bps": tc.fee_bps,
                "slip_bps": tc.slip_bps,
                "risk_per_trade": p_test.get("risk_per_trade"),
                "max_allocation": p_test.get("max_allocation"),
                "start": tc.start or "",
                "end": tc.end or "",
                "trades": int(m["trades"]),
                "wins": int(m["wins"]),
                "losses": int(m["losses"]),
                "win_rate_pct": round(m["win_rate_pct"], 2),
                "net_profit": round(m["net_profit"], 4),
                "profit_factor": round(m["profit_factor"], 4) if m["profit_factor"] != float("inf") else float("inf"),
                "min_pnl": round(m["min_pnl"], 4),
                "max_pnl": round(m["max_pnl"], 4),
                "params_file": str(p_path),
                "note": ""
            })

        print(f"[OK] {sym}: ran {len(tests)} tests using {p_path}")

    rep = pd.DataFrame(rows)
    csv_fp = out_dir / "stress_summary.csv"
    rep.to_csv(csv_fp, index=False)

    # Print compact leaderboard: for each symbol show CONS_STRESS net + PF + trades
    view = rep[rep["test"].isin(["CONS_STRESS"])].copy()
    if not view.empty:
        view = view.sort_values(["net_profit"], ascending=False)
        print("\n=== CONS_STRESS (cycle3, fee25/slip10, risk0.005/max_alloc0.30) ===")
        print(view[["symbol","trades","win_rate_pct","profit_factor","net_profit"]].to_string(index=False))

    print(f"\n[SAVED] {csv_fp}")


if __name__ == "__main__":
    main()
