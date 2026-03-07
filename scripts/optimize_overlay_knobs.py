#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.execution.execution_eval import ExecutionEvalConfig, run_entry_overlay_backtest_from_df
from src.bot087.optim.ga import _ensure_indicators, _norm_params, run_backtest_long_only


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_params(path: Path) -> Dict[str, Any]:
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


def _score(metrics: Dict[str, float]) -> float:
    net = float(metrics.get("net_profit", 0.0))
    pf = float(metrics.get("profit_factor", 0.0))
    dd = float(metrics.get("max_dd", 1.0))
    penalty = 0.0
    if pf < 1.2:
        penalty += (1.2 - pf) * 10_000.0
    if dd > 0.25:
        penalty += (dd - 0.25) * 10_000.0
    return float(net - penalty)


def _select_symbols(report_df: pd.DataFrame, run_id: str, include_btc: bool) -> List[str]:
    m = report_df[
        (report_df["run_id"].astype(str) == str(run_id))
        & (report_df["phase"] == "merged")
        & (report_df["PASS/FAIL"] == "PASS")
    ]
    syms = sorted(set(m["symbol"].astype(str).str.upper().tolist()))
    if include_btc and "BTCUSDT" not in syms:
        syms.insert(0, "BTCUSDT")
    return syms


def _sample_knobs(rng: random.Random) -> Dict[str, Any]:
    return {
        "overlay_mode": rng.choice(["pullback", "breakout"]),
        "overlay_window_sec": rng.randint(10, 60),
        "overlay_pullback_dip_bps": float(rng.randint(2, 30)),
        "overlay_breakout_bps": float(rng.randint(1, 10)),
        "overlay_skip_if_no_trigger": bool(rng.choice([True, False])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Overlay-only random search on validation split.")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--include-btc", action="store_true", default=True)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--val-frac", type=float, default=0.30)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fetch-workers", type=int, default=8)
    ap.add_argument("--cache-root", default="data/processed/execution_1s")
    ap.add_argument("--cache-cap-gb", type=float, default=20.0)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--btc-params", default="data/metadata/params/BTCUSDT_C13_active_params.json")
    args = ap.parse_args()

    report_path = PROJECT_ROOT / "artifacts" / "ga" / "all_coins_report.csv"
    rep = pd.read_csv(report_path)
    run_id = args.run_id.strip() if args.run_id.strip() else str(rep["run_id"].astype(str).iloc[-1])
    symbols = _select_symbols(rep, run_id, include_btc=bool(args.include_btc))
    if not symbols:
        raise SystemExit("No symbols selected for overlay optimization")

    rng = random.Random(int(args.seed))
    run_tag = _utc_tag()
    out_root = PROJECT_ROOT / "artifacts" / "execution_overlay" / "ga_overlay" / run_tag
    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for sym in symbols:
        if sym == "BTCUSDT":
            ppath = (PROJECT_ROOT / args.btc_params).resolve()
        else:
            ppath = PROJECT_ROOT / "data" / "metadata" / "params" / f"{sym}_active_params.json"
        if not ppath.exists():
            continue
        params = _load_params(ppath)
        df = _load_df(sym, start=args.start or None, end=args.end or None)
        if df.empty:
            continue
        dfi = _ensure_indicators(df.copy(), params)
        trades, _ = run_backtest_long_only(
            df=dfi,
            symbol=sym,
            p=params,
            initial_equity=float(args.initial_equity),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            collect_trades=True,
        )
        tr = pd.DataFrame(trades).sort_values(["entry_ts", "exit_ts"]).reset_index(drop=True)
        if tr.empty:
            continue
        n_val = max(1, int(len(tr) * float(args.val_frac)))
        tr_val = tr.iloc[-n_val:].reset_index(drop=True)

        best: Tuple[float, Dict[str, Any], Dict[str, Any]] | None = None
        hist = []
        for t in range(int(args.trials)):
            knobs = _sample_knobs(rng)
            cfg = ExecutionEvalConfig(
                mode="klines1s",
                market="spot",
                window_sec=15,
                cache_cap_gb=float(args.cache_cap_gb),
                cap_gb=float(args.cache_cap_gb),
                cache_root=str((PROJECT_ROOT / args.cache_root).resolve()),
                fetch_workers=max(1, int(args.fetch_workers)),
                alignment_max_gap_sec=2.0,
                alignment_open_tol_pct=0.01,
                overlay_mode=str(knobs["overlay_mode"]),
                overlay_window_sec=int(knobs["overlay_window_sec"]),
                overlay_breakout_lookback_sec=10,
                overlay_breakout_bps=float(knobs["overlay_breakout_bps"]),
                overlay_pullback_dip_bps=float(knobs["overlay_pullback_dip_bps"]),
                overlay_pullback_atr_k=1.0,
                overlay_ema_span=5,
                overlay_skip_if_no_trigger=bool(knobs["overlay_skip_if_no_trigger"]),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slip_bps),
                initial_equity=float(args.initial_equity),
            )
            out = run_entry_overlay_backtest_from_df(
                symbol=sym,
                df=dfi,
                p=params,
                cfg=cfg,
                initial_equity=float(args.initial_equity),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slip_bps),
                baseline_trades=tr_val,
                fetch_log_path=str(out_root / f"{sym}_trial_{t:03d}.fetch.jsonl"),
            )
            m = out["metrics"]
            dbg = out["debug"]
            s = _score(m)
            rec = {
                "trial": t,
                "symbol": sym,
                "score": s,
                "net": float(m.get("net_profit", 0.0)),
                "pf": float(m.get("profit_factor", 0.0)),
                "dd": float(m.get("max_dd", 0.0)),
                "trades": int(m.get("trades", 0)),
                "opportunities_count": int(dbg.get("opportunities_count", 0)),
                "overlay_trades_count": int(dbg.get("overlay_trades_count", 0)),
                **knobs,
            }
            hist.append(rec)
            if best is None or s > best[0]:
                best = (s, knobs, rec)

        if best is None:
            continue
        best_knobs = dict(best[1])
        best_payload = {
            "symbol": sym,
            "source_run_id": run_id,
            "overlay_type": "entry_only_1s",
            "best_knobs": best_knobs,
            "best_result": best[2],
            "constraints": {"pf_min": 1.2, "dd_max": 0.25},
            "search": {"trials": int(args.trials), "seed": int(args.seed), "val_frac": float(args.val_frac)},
        }

        overlay_out = PROJECT_ROOT / "data" / "metadata" / "params" / f"{sym}_overlay_1s.json"
        overlay_out.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
        pd.DataFrame(hist).to_csv(out_root / f"{sym}_overlay_trials.csv", index=False)
        rows.append(
            {
                "symbol": sym,
                "overlay_params_json": str(overlay_out),
                "best_score": float(best[0]),
                "best_net": float(best[2]["net"]),
                "best_pf": float(best[2]["pf"]),
                "best_dd": float(best[2]["dd"]),
                "best_mode": str(best_knobs["overlay_mode"]),
                "best_window_sec": int(best_knobs["overlay_window_sec"]),
                "best_skip_if_no_trigger": bool(best_knobs["overlay_skip_if_no_trigger"]),
            }
        )
        print(f"[{sym}] overlay best score={best[0]:.2f} net={best[2]['net']:.2f} pf={best[2]['pf']:.3f} dd={best[2]['dd']:.3f}")

    if rows:
        pd.DataFrame(rows).to_csv(out_root / "overlay_best_summary.csv", index=False)
        print(f"wrote {out_root / 'overlay_best_summary.csv'}")


if __name__ == "__main__":
    main()

