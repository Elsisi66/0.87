#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.optim.ga import (  # noqa: E402
    GAConfig,
    _ensure_indicators,
    _norm_params,
    run_backtest_long_only,
    run_ga_montecarlo,
)


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _unwrap_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
        return dict(raw["params"])
    return dict(raw)


def discover_symbols(full_dir: Path) -> List[str]:
    out: List[str] = []
    for fp in sorted(full_dir.glob("*_1h_full.parquet")):
        name = fp.name
        if "_" not in name:
            continue
        sym = name.split("_", 1)[0].upper()
        if sym:
            out.append(sym)
    return sorted(set(out))


def load_df(symbol: str, tf: str = "1h") -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_{tf}_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing parquet: {fp}")
    df = pd.read_parquet(fp)
    if "Timestamp" not in df.columns:
        raise ValueError(f"{symbol}: expected Timestamp column in {fp}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{symbol}: dataframe is empty after timestamp normalization")
    return df


def load_seed(symbol: str) -> Dict[str, Any]:
    params_dir = PROJECT_ROOT / "data" / "metadata" / "params"
    cands = [
        params_dir / f"{symbol}_active_params.json",
        params_dir / f"{symbol}_seed_params.json",
        PROJECT_ROOT / "artifacts" / "ga" / symbol / "best_params.json",
    ]
    for fp in cands:
        if fp.exists():
            return _unwrap_params(_load_json(fp))

    # Generic fallback
    return {
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
        "trade_cycles": [1, 2, 3],
        "max_hold_hours": 48,
        "cycle_shift": 1,
        "cycle_fill": 2,
        "two_candle_confirm": False,
        "require_trade_cycles": True,
    }


def slice_df(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    out = df
    if start:
        out = out[out["Timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out["Timestamp"] <= pd.Timestamp(end, tz="UTC")]
    return out.reset_index(drop=True)


def eval_backtest(df: pd.DataFrame, symbol: str, p: Dict[str, Any], fee_bps: float, slip_bps: float) -> Dict[str, float]:
    _, m = run_backtest_long_only(
        df=df,
        symbol=symbol,
        p=p,
        initial_equity=10_000.0,
        fee_bps=fee_bps,
        slippage_bps=slip_bps,
        collect_trades=False,
    )
    return {
        "net": float(m.get("net_profit", 0.0)),
        "trades": float(m.get("trades", 0.0)),
        "pf": float(m.get("profit_factor", 0.0)),
        "dd": float(m.get("max_dd", 0.0)),
        "win": float(m.get("win_rate_pct", 0.0)),
    }


def pass_fail(row: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if row["full_trades"] < args.min_full_trades:
        reasons.append(f"full_trades<{args.min_full_trades}")
    if row["full_net"] <= 0:
        reasons.append("full_net<=0")
    if row["full_pf"] < args.min_full_pf:
        reasons.append(f"full_pf<{args.min_full_pf}")
    if row["full_dd"] > args.max_full_dd:
        reasons.append(f"full_dd>{args.max_full_dd}")

    if row["oos2025_trades"] > 0 and row["oos2025_net"] < args.min_oos2025_net:
        reasons.append(f"oos2025_net<{args.min_oos2025_net}")

    if row["mc_val_net"] <= 0:
        reasons.append("mc_val_net<=0")
    if row["mc_test_net"] <= 0:
        reasons.append("mc_test_net<=0")
    if row["mc_val_pf"] < args.min_mc_val_pf:
        reasons.append(f"mc_val_pf<{args.min_mc_val_pf}")
    if row["mc_test_pf"] < args.min_mc_test_pf:
        reasons.append(f"mc_test_pf<{args.min_mc_test_pf}")

    if row["mc_val_net"] > 0 and row["mc_train_net"] > 0:
        ratio = row["mc_train_net"] / max(1e-9, row["mc_val_net"])
        if ratio > args.max_train_val_ratio:
            reasons.append(f"train_val_ratio>{args.max_train_val_ratio}")

    return len(reasons) == 0, reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="Comma-separated symbols. Default: auto from data/processed/_full")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--mc-splits", type=int, default=6)
    ap.add_argument("--train-days", type=int, default=540)
    ap.add_argument("--val-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=180)
    ap.add_argument("--pop", type=int, default=22)
    ap.add_argument("--gens", type=int, default=12)
    ap.add_argument("--btc-pop", type=int, default=30)
    ap.add_argument("--btc-gens", type=int, default=20)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # pass/fail gates
    ap.add_argument("--min-full-trades", type=float, default=60.0)
    ap.add_argument("--min-full-pf", type=float, default=1.08)
    ap.add_argument("--max-full-dd", type=float, default=0.40)
    ap.add_argument("--min-oos2025-net", type=float, default=-1500.0)
    ap.add_argument("--min-mc-val-pf", type=float, default=1.03)
    ap.add_argument("--min-mc-test-pf", type=float, default=1.00)
    ap.add_argument("--max-train-val-ratio", type=float, default=4.0)
    args = ap.parse_args()

    full_dir = PROJECT_ROOT / "data" / "processed" / "_full"
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = discover_symbols(full_dir)

    run_tag = _utc_tag()
    out_dir = PROJECT_ROOT / "artifacts" / "reports" / "campaign_1h_long" / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    print(f"[campaign] run_tag={run_tag}")
    print(f"[campaign] symbols={symbols}")

    for sym in symbols:
        print(f"\n=== OPTIMIZE {sym} ===", flush=True)
        df = load_df(sym, tf=args.tf)
        seed = _norm_params(load_seed(sym))

        is_btc = sym == "BTCUSDT"
        cfg = GAConfig(
            pop_size=(args.btc_pop if is_btc else args.pop),
            generations=(args.btc_gens if is_btc else args.gens),
            elite_k=6,
            mutation_rate=0.35,
            mutation_strength=1.0,
            n_procs=args.procs,
            mc_splits=args.mc_splits,
            train_days=args.train_days,
            val_days=args.val_days,
            test_days=args.test_days,
            seed=args.seed,
            fee_bps=args.fee_bps,
            slippage_bps=args.slip_bps,
            initial_equity=10_000.0,
            min_trades_train=40,
            min_trades_val=15,
            two_candle_confirm=False,
            require_trade_cycles=True,
            cycle_shift=1,
            cycle_fill=2,
            w_train=0.7,
            w_val=0.3,
            dd_penalty=0.45,
            trade_penalty=0.8,
            bad_val_penalty=1200.0,
            resume=args.resume,
        )

        best_p, report = run_ga_montecarlo(symbol=sym, df=df, seed_params=seed, cfg=cfg)
        best_p = _norm_params(best_p)

        # Evaluation suite
        df_ind = _ensure_indicators(df.copy(), best_p)
        full = eval_backtest(df_ind, sym, best_p, fee_bps=7.0, slip_bps=2.0)
        full_stress = eval_backtest(df_ind, sym, best_p, fee_bps=25.0, slip_bps=10.0)
        oos_2025 = eval_backtest(slice_df(df_ind, "2025-01-01", "2025-12-31"), sym, best_p, fee_bps=7.0, slip_bps=2.0)
        oos_2025_stress = eval_backtest(
            slice_df(df_ind, "2025-01-01", "2025-12-31"),
            sym,
            best_p,
            fee_bps=25.0,
            slip_bps=10.0,
        )

        mc_avg = report.get("best_overall", {}).get("avg", {})
        mc_train = mc_avg.get("train", {})
        mc_val = mc_avg.get("val", {})
        mc_test = mc_avg.get("test", {})

        row: Dict[str, Any] = {
            "symbol": sym,
            "run_id": report.get("run_id", ""),
            "cfg_pop": cfg.pop_size,
            "cfg_gens": cfg.generations,
            "cfg_mc_splits": cfg.mc_splits,
            "full_net": full["net"],
            "full_trades": full["trades"],
            "full_pf": full["pf"],
            "full_dd": full["dd"],
            "full_win": full["win"],
            "stress_net": full_stress["net"],
            "stress_trades": full_stress["trades"],
            "stress_pf": full_stress["pf"],
            "stress_dd": full_stress["dd"],
            "oos2025_net": oos_2025["net"],
            "oos2025_trades": oos_2025["trades"],
            "oos2025_pf": oos_2025["pf"],
            "oos2025_dd": oos_2025["dd"],
            "oos2025_stress_net": oos_2025_stress["net"],
            "oos2025_stress_trades": oos_2025_stress["trades"],
            "oos2025_stress_pf": oos_2025_stress["pf"],
            "oos2025_stress_dd": oos_2025_stress["dd"],
            "mc_train_net": float(mc_train.get("net", 0.0)),
            "mc_train_trades": float(mc_train.get("trades", 0.0)),
            "mc_train_pf": float(mc_train.get("pf", 0.0)),
            "mc_train_dd": float(mc_train.get("dd", 0.0)),
            "mc_val_net": float(mc_val.get("net", 0.0)),
            "mc_val_trades": float(mc_val.get("trades", 0.0)),
            "mc_val_pf": float(mc_val.get("pf", 0.0)),
            "mc_val_dd": float(mc_val.get("dd", 0.0)),
            "mc_test_net": float(mc_test.get("net", 0.0)),
            "mc_test_trades": float(mc_test.get("trades", 0.0)),
            "mc_test_pf": float(mc_test.get("pf", 0.0)),
            "mc_test_dd": float(mc_test.get("dd", 0.0)),
            "saved_active_params": str(PROJECT_ROOT / "data" / "metadata" / "params" / f"{sym}_active_params.json"),
        }

        passed, reasons = pass_fail(row, args)
        row["passed"] = int(passed)
        row["fail_reasons"] = ";".join(reasons)

        rows.append(row)

        # Write per-symbol artifact
        sym_out = out_dir / f"{sym}_campaign.json"
        payload = {
            "symbol": sym,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cfg": asdict(cfg),
            "report": report,
            "best_params": best_p,
            "summary_row": row,
        }
        sym_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print(
            f"[{sym}] PASS={passed} full_net={row['full_net']:.2f} full_pf={row['full_pf']:.2f} "
            f"oos2025_net={row['oos2025_net']:.2f} mc_val_net={row['mc_val_net']:.2f} mc_test_net={row['mc_test_net']:.2f}",
            flush=True,
        )
        if reasons:
            print(f"[{sym}] fail_reasons={reasons}", flush=True)

    rep = pd.DataFrame(rows).sort_values(["passed", "full_net"], ascending=[False, False]).reset_index(drop=True)
    rep_csv = out_dir / "summary.csv"
    rep_json = out_dir / "summary.json"
    rep.to_csv(rep_csv, index=False)
    rep_json.write_text(rep.to_json(orient="records", indent=2), encoding="utf-8")

    print("\n=== CAMPAIGN SUMMARY ===")
    if not rep.empty:
        print(rep[["symbol", "passed", "full_net", "full_pf", "full_dd", "oos2025_net", "mc_val_net", "mc_test_net", "fail_reasons"]].to_string(index=False))
    print(f"\nSaved: {rep_csv}")
    print(f"Saved: {rep_json}")


if __name__ == "__main__":
    main()
