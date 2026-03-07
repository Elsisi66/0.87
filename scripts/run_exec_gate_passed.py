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

from src.bot087.execution.baseline_audit import audit_baseline
from src.bot087.execution.execution_eval import ExecutionEvalConfig, run_entry_overlay_backtest_from_df
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
    ap.add_argument("--window-sec", type=int, default=15, help="1s window for entry pricing model after trigger.")
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
    ap.add_argument("--overlay-skip-if-no-trigger", action="store_true", default=True)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--equity-cap", type=float, default=10_000_000.0)
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
    exec_root = PROJECT_ROOT / "artifacts" / "execution_overlay"
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

        # Phase 1 baseline verification: strict consistency checks.
        baseline_net = float(base_m.get("net_profit", 0.0))
        baseline_final = float(base_m.get("final_equity", 0.0))
        baseline_init = float(base_m.get("initial_equity", args.initial_equity))
        trades_net = float(trades_df["net_pnl"].sum()) if not trades_df.empty and "net_pnl" in trades_df.columns else baseline_net

        if abs((baseline_final - baseline_init) - baseline_net) > 1e-6:
            raise RuntimeError(f"[{sym}] Baseline inconsistency: final-initial != net")
        if abs(trades_net - baseline_net) > 1e-3:
            raise RuntimeError(f"[{sym}] Baseline inconsistency: sum(trade net_pnl) != net")

        audit_path = PROJECT_ROOT / "artifacts" / "audit" / sym / "baseline_audit.json"
        audit = audit_baseline(
            symbol=sym,
            df=dfi,
            trades_df=trades_df,
            initial_equity=float(args.initial_equity),
            max_allocation=float(params.get("max_allocation", 0.7)),
            equity_cap=float(args.equity_cap),
            out_path=audit_path,
        )

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
            overlay_skip_if_no_trigger=bool(args.overlay_skip_if_no_trigger),
            overlay_partial_tp_bps=float(args.overlay_partial_tp_bps),
            overlay_partial_tp_frac=float(args.overlay_partial_tp_frac),
            overlay_partial_tp_window_sec=max(1, int(args.overlay_partial_tp_window_sec)),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            initial_equity=float(args.initial_equity),
            merge_gap_sec=0,
        )

        out = run_entry_overlay_backtest_from_df(
            symbol=sym,
            df=dfi,
            p=params,
            cfg=cfg,
            initial_equity=float(args.initial_equity),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            baseline_trades=trades_df,
            fetch_log_path=str(sym_run_dir / "fetch_log.jsonl"),
        )
        tr_after = out["trades"]
        m_after = out["metrics"]
        dbg = out["debug"]

        # Build before/after summary
        before_net = float(base_m.get("net_profit", 0.0))
        after_net = float(m_after.get("net_profit", 0.0))
        if before_net > 0:
            edge_decay = float(after_net / before_net)
        elif before_net == 0:
            edge_decay = 0.0
        else:
            edge_decay = float(after_net / abs(before_net))

        pf_before = float(base_m.get("profit_factor", 0.0))
        pf_after = float(m_after.get("profit_factor", 0.0))
        dd_before = float(base_m.get("max_dd", 0.0))
        dd_after = float(m_after.get("max_dd", 0.0))
        dd_delta = float(dd_after - dd_before)
        fallback_rate = float(dbg.get("fallback_count", 0) / max(1, int(dbg.get("signals", 0))))
        alignment_fail_rate = float(dbg.get("alignment_fail_rate", 0.0))
        overlay_skip_rate = float(dbg.get("skip_rate", 0.0))
        opportunities_count = int(dbg.get("opportunities_count", int(base_m.get("trades", 0))))
        baseline_trades_count = int(dbg.get("baseline_trades_count", int(base_m.get("trades", 0))))
        overlay_trades_count = int(dbg.get("overlay_trades_count", int(m_after.get("trades", 0))))
        skipped_count = int(dbg.get("skipped_count", 0))

        if overlay_trades_count > baseline_trades_count:
            raise RuntimeError(
                f"[{sym}] BUG: overlay_trades_count={overlay_trades_count} > baseline_trades_count={baseline_trades_count}"
            )

        fail_reasons = []
        if edge_decay < 0.70:
            fail_reasons.append("edge_decay<0.70")
        if pf_after < 1.10:
            fail_reasons.append("pf_after<1.10")
        if dd_delta > 0.05:
            fail_reasons.append("dd_after-dd_before>0.05")
        exec_gate_ok = len(fail_reasons) == 0
        overlay_ok = bool(alignment_fail_rate <= 0.05 and overlay_skip_rate <= 0.90)

        # Persist per-coin artifacts.
        tr_after.to_csv(sym_run_dir / "trade_level.csv", index=False)
        summary_payload = {
            "symbol": sym,
            "time_range": {
                "start": str(dfi["Timestamp"].min()),
                "end": str(dfi["Timestamp"].max()),
            },
            "baseline_net": before_net,
            "overlay_net": after_net,
            "baseline_pf": pf_before,
            "overlay_pf": pf_after,
            "baseline_dd": dd_before,
            "overlay_dd": dd_after,
            "opportunities_count": opportunities_count,
            "baseline_trades_count": baseline_trades_count,
            "overlay_trades_count": overlay_trades_count,
            "skipped_count": skipped_count,
            "alignment_fail_rate": alignment_fail_rate,
            "fallback_rate": fallback_rate,
            "avg_entry_improvement_bps": float(dbg.get("avg_entry_improvement_bps", 0.0)),
            "before": {
                "initial_equity": float(base_m.get("initial_equity", args.initial_equity)),
                "final_equity": float(base_m.get("final_equity", 0.0)),
                "net": before_net,
                "pf": pf_before,
                "dd": dd_before,
                "trades": int(base_m.get("trades", 0)),
            },
            "after": {
                "initial_equity": float(m_after.get("initial_equity", args.initial_equity)),
                "final_equity": float(m_after.get("final_equity", 0.0)),
                "net": after_net,
                "pf": pf_after,
                "dd": dd_after,
                "trades": int(m_after.get("trades", 0)),
            },
            "edge_decay": edge_decay,
            "fallback_rate": fallback_rate,
            "alignment_fail_rate": alignment_fail_rate,
            "overlay_skip_rate": overlay_skip_rate,
            "opportunities_count": opportunities_count,
            "baseline_trades_count": baseline_trades_count,
            "overlay_trades_count": overlay_trades_count,
            "skipped_count": skipped_count,
            "avg_entry_improvement_bps": float(dbg.get("avg_entry_improvement_bps", 0.0)),
            "exec_gate_ok": exec_gate_ok,
            "overlay_ok": overlay_ok,
            "fail_reasons": fail_reasons,
            "baseline_audit": audit,
            "debug": dbg,
        }
        (sym_run_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        dbg_dir = PROJECT_ROOT / "artifacts" / "execution_debug" / sym / exec_run_id
        dbg_dir.mkdir(parents=True, exist_ok=True)
        (dbg_dir / "debug_summary.json").write_text(json.dumps(summary_payload["debug"], indent=2), encoding="utf-8")

        row = {
            "exec_run_id": exec_run_id,
            "source_run_id": run_id,
            "symbol": sym,
            "trades": int(m_after.get("trades", 0)),
            "opportunities_count": int(opportunities_count),
            "baseline_trades_count": int(baseline_trades_count),
            "overlay_trades_count": int(overlay_trades_count),
            "skipped_count": int(skipped_count),
            "exec_edge_decay": float(edge_decay),
            "exec_pf_after": float(pf_after),
            "exec_dd_before": float(dd_before),
            "exec_dd_after": float(dd_after),
            "exec_dd_delta": float(dd_delta),
            "exec_pass": bool(exec_gate_ok),
            "exec_gate_ok": bool(exec_gate_ok),
            "overlay_ok": bool(overlay_ok),
            "exec_fail_reasons": ";".join(fail_reasons),
            "fallback_rate": float(fallback_rate),
            "alignment_fail_rate": float(alignment_fail_rate),
            "overlay_skip_rate": float(overlay_skip_rate),
            "avg_entry_improvement_bps": float(dbg.get("avg_entry_improvement_bps", 0.0)),
            "baseline_initial_equity": float(base_m.get("initial_equity", args.initial_equity)),
            "baseline_final_equity": float(base_m.get("final_equity", 0.0)),
            "baseline_net": float(before_net),
            "baseline_pf": float(pf_before),
            "baseline_dd": float(dd_before),
            "baseline_trades": int(base_m.get("trades", 0)),
            "after_initial_equity": float(m_after.get("initial_equity", args.initial_equity)),
            "after_final_equity": float(m_after.get("final_equity", 0.0)),
            "after_net": float(after_net),
            "after_pf": float(pf_after),
            "after_dd": float(dd_after),
            "after_trades": int(m_after.get("trades", 0)),
            "summary_json": str(sym_run_dir / "summary.json"),
            "trade_level_csv": str(sym_run_dir / "trade_level.csv"),
            "fetch_log_jsonl": str(sym_run_dir / "fetch_log.jsonl"),
            "debug_summary_json": str(PROJECT_ROOT / "artifacts" / "execution_debug" / sym / exec_run_id / "debug_summary.json"),
            "baseline_audit_json": str(audit_path),
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
