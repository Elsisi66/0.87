#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _import_module_from_path(py_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--initial", type=float, default=10_000.0)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--cycle-shift", type=int, default=1)
    ap.add_argument("--only-cycle", type=int, default=3)
    ap.add_argument("--years", default="2022,2025", help="Comma-separated years, default: 2022,2025")
    ap.add_argument("--nla-file", default="scripts/backtest_nla.py", help="Path to your NLA runner python file")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    os.environ["BOT087_PROJECT_ROOT"] = str(project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    nla_path = (project_root / args.nla_file).resolve() if not Path(args.nla_file).is_absolute() else Path(args.nla_file)
    if not nla_path.exists():
        raise SystemExit(f"NLA file not found: {nla_path}")

    nla = _import_module_from_path(nla_path, "nla_runner")

    symbol = args.symbol.strip().upper()
    tf = args.tf.strip()
    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    only_cycle = int(args.only_cycle)

    # --- params (reuse your runner helpers) ---
    params_fp = nla._find_params(symbol)  # type: ignore
    p = nla._read_json(params_fp)         # type: ignore

    # normalize + indicators
    from src.bot087.optim.ga import _norm_params, _ensure_indicators
    p = _norm_params(p)
    p["trade_cycles"] = [only_cycle]
    p["require_trade_cycles"] = True

    # --- data ---
    df = nla._load_df(symbol, tf=tf)      # type: ignore
    df = nla._ensure_ohlc_cols(df)        # type: ignore
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if df.empty:
        raise SystemExit("No data loaded. Download data first into data/processed/_full/...")

    df = _ensure_indicators(df, p)

    run_tag = _utc_tag()
    base_out = project_root / "output" / "backtests_nla_year_slices" / symbol / run_tag
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== YEAR SLICES (NLA) ===")
    print(f"Symbol: {symbol}  TF: {tf}  Only cycle: {only_cycle}")
    print(f"Params: {params_fp}")
    print(f"Output: {base_out}\n")

    summary_rows = []

    for y in years:
        start = pd.Timestamp(f"{y}-01-01", tz="UTC")
        end = pd.Timestamp(f"{y}-12-31 23:59:59", tz="UTC")
        dfx = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)].reset_index(drop=True)

        if dfx.empty:
            print(f"{y}: no candles")
            continue

        trades, m = nla.run_backtest_nla(  # type: ignore
            df=dfx,
            symbol=symbol,
            p=p,
            initial_equity=float(args.initial),
            fee_bps=float(args.fee_bps),
            slip_bps=float(args.slip_bps),
            cycle_shift=int(args.cycle_shift),
        )

        out_dir = base_out / str(y)
        out_dir.mkdir(parents=True, exist_ok=True)

        tdf = pd.DataFrame(trades)
        tdf.to_csv(out_dir / "trades.csv", index=False)
        (out_dir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

        net = float(m.get("net_profit", 0.0))
        tr = int(m.get("trades", 0))
        pf = m.get("profit_factor", 0.0)
        dd = float(m.get("max_dd_pct", 0.0)) * 100.0  # NOTE: your code stores dd as fraction; if it's already pct, you'll notice (it'll be huge)
        fe = float(m.get("final_equity", args.initial))

        summary_rows.append({
            "year": y,
            "trades": tr,
            "net_profit": net,
            "final_equity": fe,
            "profit_factor": pf,
            "max_dd_pct_approx": dd,
        })

        print(f"{y}: trades={tr} net={net:.2f} final={fe:.2f} pf={pf} dd~={dd:.2f}%  saved={out_dir}")

    if summary_rows:
        print("\n--- SUMMARY ---")
        sdf = pd.DataFrame(summary_rows).sort_values("year")
        print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
