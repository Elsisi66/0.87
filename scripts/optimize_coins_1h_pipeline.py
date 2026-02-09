#!/usr/bin/env python3
"""
Optimize 1h long-only strategies for multiple coins with strict no-lookahead rules.

How to run:
  scripts/venv/bin/python scripts/optimize_coins_1h_pipeline.py \
    --n-procs 3 --pop-size 48 --generations 120 --mc-splits 6

  # Include BTC as read-only baseline row at the end (never writes BTC params)
  scripts/venv/bin/python scripts/optimize_coins_1h_pipeline.py \
    --include-btc-read-only

Notes:
- Auto-discovers symbols from data/processed/_full/*_1h_full.parquet.
- Excludes BTC by default.
- Writes per-coin run folder: artifacts/ga/<SYMBOL>/<RUN_ID>/
- Appends generation history to: artifacts/ga/all_coins_history.csv
- Appends phase summary rows to: artifacts/ga/all_coins_report.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.optim.ga import (  # noqa: E402
    GAConfig,
    _ensure_indicators,
    _norm_params,
    build_entry_signal,
    run_backtest_long_only,
)

CYCLE_ARRAY_KEYS = ["willr_by_cycle", "tp_mult_by_cycle", "sl_mult_by_cycle", "exit_rsi_by_cycle"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_rows_csv(path: Path, rows: Iterable[Dict[str, Any]], ordered_columns: Optional[List[str]] = None) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    if ordered_columns is None:
        seen: List[str] = []
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.append(k)
        ordered_columns = seen

    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered_columns, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def discover_symbols(full_dir: Path, include_btc: bool = False) -> List[str]:
    out: List[str] = []
    for fp in sorted(full_dir.glob("*_1h_full.parquet")):
        stem = fp.name
        if "_" not in stem:
            continue
        symbol = stem.split("_", 1)[0].upper()
        if symbol == "BTCUSDT" and not include_btc:
            continue
        out.append(symbol)
    return sorted(set(out))


def load_df_1h(symbol: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet for {symbol}: {fp}")
    df = pd.read_parquet(fp)
    if "Timestamp" not in df.columns:
        raise ValueError(f"{symbol}: missing Timestamp column in {fp}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{symbol}: empty dataframe after Timestamp parse")
    return df


def _unwrap_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
        return dict(raw["params"])
    return dict(raw)


def load_seed_params(symbol: str) -> Dict[str, Any]:
    params_dir = PROJECT_ROOT / "data" / "metadata" / "params"
    candidates = [
        params_dir / f"{symbol}_active_params.json",
        params_dir / f"{symbol}_seed_params.json",
    ]
    for fp in candidates:
        if fp.exists():
            return _norm_params(_unwrap_params(_read_json(fp)))

    return _norm_params(
        {
            "entry_rsi_min": 52.0,
            "entry_rsi_max": 64.0,
            "entry_rsi_buffer": 2.5,
            "willr_floor": -100.0,
            "willr_by_cycle": [-78.0, -32.0, -90.0, -18.0, -50.0],
            "ema_span": 35,
            "ema_trend_long": 120,
            "ema_align": True,
            "require_ema200_slope": True,
            "adx_min": 18.0,
            "require_plus_di": True,
            "tp_mult_by_cycle": [1.035, 1.08, 1.05, 1.07, 1.05],
            "sl_mult_by_cycle": [0.985, 0.98, 0.96, 0.99, 0.97],
            "exit_rsi_by_cycle": [50.0, 56.0, 50.0, 62.0, 52.0],
            "risk_per_trade": 0.02,
            "max_allocation": 0.7,
            "atr_k": 1.0,
            "trade_cycles": [1, 3],
            "max_hold_hours": 48,
            "cycle_shift": 1,
            "cycle_fill": 2,
            "two_candle_confirm": False,
            "require_trade_cycles": True,
        }
    )


def enforce_nla_and_validate(df: pd.DataFrame, params: Dict[str, Any], symbol: str) -> pd.DataFrame:
    p = _norm_params(dict(params))
    p["cycle_shift"] = 1
    p["two_candle_confirm"] = False
    p["require_trade_cycles"] = True

    dfi = _ensure_indicators(df.copy(), p)
    required = ["RSI", "ATR", "WILLR", "ADX", "PLUS_DI", "MINUS_DI", "EMA_200", "EMA_200_SLOPE"]
    missing = [c for c in required if c not in dfi.columns]
    if missing:
        raise ValueError(f"{symbol}: missing required indicators: {missing}")

    nan_cols = [c for c in required if dfi[c].isna().any()]
    if nan_cols:
        raise ValueError(f"{symbol}: NaNs in required indicators: {nan_cols}")

    sig = build_entry_signal(dfi, p)
    if len(sig) != len(dfi):
        raise ValueError(f"{symbol}: signal length mismatch (sig={len(sig)} df={len(dfi)})")

    return dfi


@dataclass
class Paths:
    root: Path
    run_id: str
    all_history_csv: Path
    all_report_csv: Path


def build_paths(run_id: str) -> Paths:
    ga_root = PROJECT_ROOT / "artifacts" / "ga"
    ga_root.mkdir(parents=True, exist_ok=True)
    return Paths(
        root=ga_root,
        run_id=run_id,
        all_history_csv=ga_root / "all_coins_history.csv",
        all_report_csv=ga_root / "all_coins_report.csv",
    )


def init_coin_run_dir(paths: Paths, symbol: str, args: argparse.Namespace, df: pd.DataFrame) -> Path:
    run_dir = paths.root / symbol / paths.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "config_used.json",
        {
            "timestamp_utc": _utc_now_iso(),
            "symbol": symbol,
            "run_id": paths.run_id,
            "args": vars(args),
            "data": {
                "rows": int(len(df)),
                "start": str(df["Timestamp"].min()),
                "end": str(df["Timestamp"].max()),
            },
            "fees": {
                "fee_bps": float(args.fee_bps),
                "slippage_bps": float(args.slippage_bps),
            },
            "nla": {
                "cycle_shift": 1,
                "entry_rule": "t-1 features -> bar t Open",
            },
        },
    )
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="Comma-separated symbol list. Default: auto-discover from data/processed/_full")
    ap.add_argument("--include-btc-read-only", action="store_true", help="Append BTC summary row using existing params only")
    ap.add_argument("--run-id", default="", help="Optional fixed run id; default UTC timestamp")
    ap.add_argument("--n-procs", type=int, default=3)
    ap.add_argument("--pop-size", type=int, default=48)
    ap.add_argument("--generations", type=int, default=120)
    ap.add_argument("--mc-splits", type=int, default=6)
    ap.add_argument("--train-days", type=int, default=540)
    ap.add_argument("--val-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--early-stop-patience", type=int, default=40)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.pop_size > 120:
        raise SystemExit("pop-size hard limit is 120")
    if args.generations > 150:
        raise SystemExit("generations hard limit is 150")

    run_id = args.run_id.strip() or _utc_tag()
    paths = build_paths(run_id)

    full_dir = PROJECT_ROOT / "data" / "processed" / "_full"
    if args.symbols.strip():
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
        symbols = [s for s in symbols if s != "BTCUSDT"]
    else:
        symbols = discover_symbols(full_dir, include_btc=False)

    if not symbols:
        raise SystemExit("No non-BTC symbols found.")

    print(f"[pipeline] run_id={run_id}")
    print(f"[pipeline] symbols={symbols}")

    for symbol in symbols:
        df = load_df_1h(symbol)
        seed = load_seed_params(symbol)
        dfi = enforce_nla_and_validate(df, seed, symbol)
        run_dir = init_coin_run_dir(paths, symbol, args, dfi)
        print(f"[{symbol}] prepared run folder: {run_dir}")

    print("[pipeline] scaffolding complete. Next commit adds phase optimization flow.")


if __name__ == "__main__":
    main()
