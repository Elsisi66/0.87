#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.execution.execution_eval import ExecutionEvalConfig, evaluate_execution_from_trades
from src.bot087.optim.ga import _ensure_indicators, _norm_params, run_backtest_long_only


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_params(path: Path) -> Dict[str, Any]:
    raw = _read_json(path)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="", help="1h pipeline run_id. Default: latest in all_coins_report.csv")
    ap.add_argument("--include-btc", action="store_true", default=True)
    ap.add_argument("--window-sec", type=int, default=15)
    ap.add_argument("--mode", choices=["klines1s", "aggtrades"], default="klines1s")
    ap.add_argument("--market", choices=["spot", "futures"], default="spot")
    ap.add_argument("--cache-cap-gb", type=float, default=20.0)
    ap.add_argument("--cache-root", default="data/processed/execution_1s")
    ap.add_argument("--fetch-workers", type=int, default=1)
    ap.add_argument("--alignment-max-gap-sec", type=float, default=2.0)
    ap.add_argument("--alignment-open-tol-pct", type=float, default=0.01)
    ap.add_argument("--overlay-mode", choices=["none", "breakout", "pullback"], default="breakout")
    ap.add_argument("--overlay-window-sec", type=int, default=30)
    ap.add_argument("--overlay-breakout-lookback-sec", type=int, default=10)
    ap.add_argument("--overlay-breakout-bps", type=float, default=3.0)
    ap.add_argument("--overlay-pullback-dip-bps", type=float, default=8.0)
    ap.add_argument("--overlay-pullback-atr-k", type=float, default=1.0)
    ap.add_argument("--overlay-ema-span", type=int, default=5)
    ap.add_argument("--overlay-partial-tp-bps", type=float, default=15.0)
    ap.add_argument("--overlay-partial-tp-frac", type=float, default=0.0)
    ap.add_argument("--overlay-partial-tp-window-sec", type=int, default=60)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument(
        "--btc-params",
        default="data/metadata/params/BTCUSDT_C13_active_params.json",
        help="BTC merged params (read-only).",
    )
    args = ap.parse_args()

    all_report_path = PROJECT_ROOT / "artifacts" / "ga" / "all_coins_report.csv"
    if not all_report_path.exists():
        raise SystemExit(f"Missing report: {all_report_path}")

    rep = pd.read_csv(all_report_path)
    if args.run_id.strip():
        run_id = args.run_id.strip()
    else:
        run_id = str(rep["run_id"].astype(str).iloc[-1])

    symbols = _select_symbols(rep, run_id, include_btc=bool(args.include_btc))
    if not symbols:
        raise SystemExit(f"No symbols selected for run_id={run_id}")

    exec_run_id = _utc_tag()
    exec_root = PROJECT_ROOT / "artifacts" / "execution"
    exec_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for sym in symbols:
        print(f"\n=== EXEC GATE {sym} ===", flush=True)
        sym_run_dir = exec_root / sym / exec_run_id
        sym_run_dir.mkdir(parents=True, exist_ok=True)

        # Params (read-only load)
        if sym == "BTCUSDT":
            ppath = (PROJECT_ROOT / args.btc_params).resolve()
        else:
            ppath = PROJECT_ROOT / "data" / "metadata" / "params" / f"{sym}_active_params.json"
        if not ppath.exists():
            print(f"[{sym}] skip: params missing {ppath}", flush=True)
            continue

        params = _load_params(ppath)

        df = _load_df(sym, start=args.start or None, end=args.end or None)
        if df.empty:
            print(f"[{sym}] skip: empty df after date filter", flush=True)
            continue

        dfi = _ensure_indicators(df.copy(), params)
        trades, base_m = run_backtest_long_only(
            df=dfi,
            symbol=sym,
            p=params,
            initial_equity=float(args.initial_equity),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            collect_trades=True,
        )

        trades_df = pd.DataFrame(trades)
        trades_csv = sym_run_dir / "trades_1h.csv"
        trades_df.to_csv(trades_csv, index=False)
        (sym_run_dir / "backtest_1h_metrics.json").write_text(json.dumps(base_m, indent=2), encoding="utf-8")

        cfg = ExecutionEvalConfig(
            mode=args.mode,
            market=args.market,
            window_sec=int(args.window_sec),
            cache_cap_gb=float(args.cache_cap_gb),
            cap_gb=float(args.cache_cap_gb),
            cache_root=str((PROJECT_ROOT / args.cache_root).resolve()),
            fetch_workers=max(1, int(args.fetch_workers)),
            alignment_max_gap_sec=float(args.alignment_max_gap_sec),
            alignment_open_tol_pct=float(args.alignment_open_tol_pct),
            overlay_mode=str(args.overlay_mode).lower(),
            overlay_window_sec=max(1, int(args.overlay_window_sec)),
            overlay_breakout_lookback_sec=max(1, int(args.overlay_breakout_lookback_sec)),
            overlay_breakout_bps=float(args.overlay_breakout_bps),
            overlay_pullback_dip_bps=float(args.overlay_pullback_dip_bps),
            overlay_pullback_atr_k=float(args.overlay_pullback_atr_k),
            overlay_ema_span=max(2, int(args.overlay_ema_span)),
            overlay_partial_tp_bps=float(args.overlay_partial_tp_bps),
            overlay_partial_tp_frac=float(args.overlay_partial_tp_frac),
            overlay_partial_tp_window_sec=max(1, int(args.overlay_partial_tp_window_sec)),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            initial_equity=float(args.initial_equity),
            merge_gap_sec=0,
        )

        out = evaluate_execution_from_trades(
            symbol=sym,
            trades_path=str(trades_csv),
            cfg=cfg,
            run_dir=str(sym_run_dir),
        )
        s = out["summary"]

        row = {
            "exec_run_id": exec_run_id,
            "source_run_id": run_id,
            "symbol": sym,
            "trades": int(s.get("trades", 0)),
            "exec_edge_decay": float(s.get("edge_decay", 0.0)),
            "exec_pf_after": float(s.get("pf_after", 0.0)),
            "exec_dd_before": float(s.get("dd_before", 0.0)),
            "exec_dd_after": float(s.get("dd_after", 0.0)),
            "exec_dd_delta": float(s.get("dd_delta", 0.0)),
            "exec_pass": bool(s.get("exec_pass", False)),
            "exec_gate_ok": bool(s.get("exec_gate_ok", s.get("exec_pass", False))),
            "overlay_ok": bool(s.get("overlay_ok", True)),
            "exec_fail_reasons": ";".join(s.get("fail_reasons", [])),
            "fallback_rate": float(s.get("fallback_rate", 0.0)),
            "alignment_fail_rate": float(s.get("alignment_fail_rate", 0.0)),
            "overlay_skip_rate": float(s.get("overlay_skip_rate", 0.0)),
            "p95_entry_slip_bps": float(s.get("p95_entry_slip_bps", 0.0)),
            "p95_exit_slip_bps": float(s.get("p95_exit_slip_bps", 0.0)),
            "summary_json": str(sym_run_dir / "summary.json"),
            "trade_level_csv": str(sym_run_dir / "trade_level.csv"),
            "fetch_log_jsonl": str(sym_run_dir / "fetch_log.jsonl"),
            "debug_summary_json": str(PROJECT_ROOT / "artifacts" / "execution_debug" / sym / exec_run_id / "debug_summary.json"),
            "params_used": str(ppath),
        }
        rows.append(row)

        print(
            f"[{sym}] exec_gate_ok={row['exec_gate_ok']} overlay_ok={row['overlay_ok']} "
            f"edge_decay={row['exec_edge_decay']:.3f} pf_after={row['exec_pf_after']:.3f} "
            f"dd_after={row['exec_dd_after']:.3f}",
            flush=True,
        )

    if not rows:
        raise SystemExit("No execution-gate outputs generated.")

    out_df = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    out_df.to_csv(exec_root / f"exec_gate_summary_{exec_run_id}.csv", index=False)
    (exec_root / f"exec_gate_summary_{exec_run_id}.json").write_text(out_df.to_json(orient="records", indent=2), encoding="utf-8")

    # Integrate into all_coins_report.csv for the corresponding merged rows.
    rep2 = rep.copy()
    for r in rows:
        mask = (
            (rep2["run_id"].astype(str) == str(run_id))
            & (rep2["symbol"].astype(str).str.upper() == str(r["symbol"]).upper())
            & (rep2["phase"].astype(str) == "merged")
        )
        if mask.any():
            rep2.loc[mask, "exec_edge_decay"] = float(r["exec_edge_decay"])
            rep2.loc[mask, "exec_pf_after"] = float(r["exec_pf_after"])
            rep2.loc[mask, "exec_dd_after"] = float(r["exec_dd_after"])
            rep2.loc[mask, "exec_pass"] = bool(r["exec_pass"])
            rep2.loc[mask, "exec_gate_ok"] = bool(r["exec_gate_ok"])
            rep2.loc[mask, "overlay_ok"] = bool(r["overlay_ok"])

    rep2.to_csv(all_report_path, index=False)

    print("\n=== EXEC GATE COMPLETE ===")
    print(f"source_run_id={run_id}")
    print(f"exec_run_id={exec_run_id}")
    print(out_df[["symbol", "exec_gate_ok", "overlay_ok", "exec_edge_decay", "exec_pf_after", "exec_dd_after"]].to_string(index=False))
    print(f"updated_report={all_report_path}")
    print(f"summary_csv={exec_root / f'exec_gate_summary_{exec_run_id}.csv'}")


if __name__ == "__main__":
    main()
