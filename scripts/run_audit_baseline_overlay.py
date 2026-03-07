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


SYMBOLS = ["BTCUSDT", "ADAUSDT", "AVAXUSDT", "SOLUSDT"]


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
    # hard enforce spot/no-leverage sizing cap
    p["max_allocation"] = min(0.7, float(p.get("max_allocation", 0.7)))
    p["equity_sizing_cap"] = 100.0
    return p


def _load_df(symbol: str, start: str, end: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing 1h parquet: {fp}")
    df = pd.read_parquet(fp)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    df = df.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    df = df[(df["Timestamp"] >= pd.Timestamp(start, tz="UTC")) & (df["Timestamp"] <= pd.Timestamp(end, tz="UTC"))]
    if df.empty:
        raise RuntimeError(f"{symbol}: empty df after cleaning/date filter")
    return df.reset_index(drop=True)


def _edge_decay(before_net: float, after_net: float) -> float:
    if before_net > 0:
        return float(after_net / before_net)
    if before_net < 0:
        return float(after_net / abs(before_net))
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Run audit -> baseline 1h -> 1s entry overlay for BTC/ADA/AVAX/SOL.")
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--initial-equity", type=float, default=100.0)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--equity-cap", type=float, default=100_000.0)
    ap.add_argument("--overlay-mode", choices=["pullback", "breakout", "none"], default="pullback")
    ap.add_argument("--overlay-window-sec", type=int, default=30)
    ap.add_argument("--window-sec", type=int, default=15)
    ap.add_argument("--cache-cap-gb", type=float, default=20.0)
    ap.add_argument("--cache-root", default="data/processed/execution_1s")
    ap.add_argument("--fetch-workers", type=int, default=8)
    ap.add_argument("--btc-params", default="data/metadata/params/BTCUSDT_C13_active_params.json")
    args = ap.parse_args()

    if abs(float(args.initial_equity) - 100.0) > 1e-12:
        raise RuntimeError("initial_equity must be 100.0 for this correctness pipeline")

    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    run_id = _utc_tag()
    overlay_root = PROJECT_ROOT / "artifacts" / "execution_overlay"
    reports_root = PROJECT_ROOT / "artifacts" / "reports"
    reports_root.mkdir(parents=True, exist_ok=True)

    baseline_rows: List[Dict[str, Any]] = []
    overlay_rows: List[Dict[str, Any]] = []
    summary_lines: List[str] = []

    for sym in symbols:
        if sym == "BTCUSDT":
            ppath = (PROJECT_ROOT / args.btc_params).resolve()
        else:
            ppath = PROJECT_ROOT / "data" / "metadata" / "params" / f"{sym}_active_params.json"
        if not ppath.exists():
            raise FileNotFoundError(f"{sym}: params not found {ppath}")

        p = _load_params(ppath)
        df = _load_df(sym, start=args.start, end=args.end)
        dfi = _ensure_indicators(df.copy(), p)

        trades, metrics = run_backtest_long_only(
            df=dfi,
            symbol=sym,
            p=p,
            initial_equity=100.0,
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            collect_trades=True,
        )
        tr = pd.DataFrame(trades)
        audit_dir = PROJECT_ROOT / "artifacts" / "audit" / sym
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit = audit_baseline(
            symbol=sym,
            df=dfi,
            trades_df=tr,
            initial_equity=100.0,
            max_allocation=float(p.get("max_allocation", 0.7)),
            equity_cap=float(args.equity_cap),
            out_path=audit_dir / "audit.json",
        )
        if not audit.get("ok", False):
            raise RuntimeError(f"{sym}: baseline audit failed")

        baseline_rows.append(
            {
                "symbol": sym,
                "initial_equity": float(metrics.get("initial_equity", 100.0)),
                "final_equity": float(metrics.get("final_equity", 0.0)),
                "net_profit": float(metrics.get("net_profit", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 0.0)),
                "max_dd": float(metrics.get("max_dd", 0.0)),
                "trades": int(metrics.get("trades", 0)),
                "max_alloc_enforced": float(p.get("max_allocation", 0.7)),
                "audit_json": str(audit_dir / "audit.json"),
                "audit_trades_csv": str(audit_dir / "audit_trades.csv"),
            }
        )

        sym_overlay_dir = overlay_root / sym / run_id
        sym_overlay_dir.mkdir(parents=True, exist_ok=True)
        cfg = ExecutionEvalConfig(
            mode="klines1s",
            market="spot",
            window_sec=max(1, int(args.window_sec)),
            cache_cap_gb=float(args.cache_cap_gb),
            cap_gb=float(args.cache_cap_gb),
            cache_root=str((PROJECT_ROOT / args.cache_root).resolve()),
            fetch_workers=max(1, int(args.fetch_workers)),
            alignment_max_gap_sec=2.0,
            alignment_open_tol_pct=0.01,
            overlay_mode=str(args.overlay_mode),
            overlay_window_sec=max(1, int(args.overlay_window_sec)),
            overlay_breakout_lookback_sec=10,
            overlay_breakout_bps=3.0,
            overlay_pullback_dip_bps=10.0,
            overlay_pullback_atr_k=1.0,
            overlay_ema_span=5,
            overlay_skip_if_no_trigger=True,
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            initial_equity=100.0,
        )
        ov = run_entry_overlay_backtest_from_df(
            symbol=sym,
            df=dfi,
            p=p,
            cfg=cfg,
            initial_equity=100.0,
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            baseline_trades=tr,
            fetch_log_path=str(sym_overlay_dir / "fetch_log.jsonl"),
        )
        ov_tr = ov["trades"]
        ov_m = ov["metrics"]
        dbg = ov["debug"]
        ov_tr.to_csv(sym_overlay_dir / "trade_level.csv", index=False)

        baseline_net = float(metrics.get("net_profit", 0.0))
        overlay_net = float(ov_m.get("net_profit", 0.0))
        edge = _edge_decay(baseline_net, overlay_net)
        fail = edge < 0.7
        if int(dbg.get("overlay_trades_count", 0)) > int(dbg.get("baseline_trades_count", 0)):
            raise RuntimeError(f"{sym}: overlay generated extra trades")

        summary_json = {
            "symbol": sym,
            "baseline_net": baseline_net,
            "overlay_net": overlay_net,
            "baseline_pf": float(metrics.get("profit_factor", 0.0)),
            "overlay_pf": float(ov_m.get("profit_factor", 0.0)),
            "baseline_dd": float(metrics.get("max_dd", 0.0)),
            "overlay_dd": float(ov_m.get("max_dd", 0.0)),
            "opportunities_count": int(dbg.get("opportunities_count", 0)),
            "baseline_trades_count": int(dbg.get("baseline_trades_count", 0)),
            "overlay_trades_count": int(dbg.get("overlay_trades_count", 0)),
            "skipped_count": int(dbg.get("skipped_count", 0)),
            "alignment_fail_rate": float(dbg.get("alignment_fail_rate", 0.0)),
            "fallback_rate": float(dbg.get("fallback_count", 0) / max(1, int(dbg.get("signals", 0)))),
            "avg_entry_improvement_bps": float(dbg.get("avg_entry_improvement_bps", 0.0)),
            "edge_decay": float(edge),
            "edge_fail": bool(fail),
        }
        (sym_overlay_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

        overlay_rows.append(
            {
                "symbol": sym,
                "initial_equity": 100.0,
                "baseline_net": baseline_net,
                "overlay_net": overlay_net,
                "edge_decay": float(edge),
                "edge_fail": bool(fail),
                "baseline_pf": float(metrics.get("profit_factor", 0.0)),
                "overlay_pf": float(ov_m.get("profit_factor", 0.0)),
                "baseline_dd": float(metrics.get("max_dd", 0.0)),
                "overlay_dd": float(ov_m.get("max_dd", 0.0)),
                "opportunities_count": int(dbg.get("opportunities_count", 0)),
                "baseline_trades_count": int(dbg.get("baseline_trades_count", 0)),
                "overlay_trades_count": int(dbg.get("overlay_trades_count", 0)),
                "skipped_count": int(dbg.get("skipped_count", 0)),
                "alignment_fail_rate": float(dbg.get("alignment_fail_rate", 0.0)),
                "summary_json": str(sym_overlay_dir / "summary.json"),
                "trade_level_csv": str(sym_overlay_dir / "trade_level.csv"),
            }
        )

        summary_lines.append(
            f"- {sym}: baseline_net={baseline_net:.4f}, overlay_net={overlay_net:.4f}, "
            f"edge_decay={edge:.4f}, trades={int(dbg.get('baseline_trades_count',0))}->{int(dbg.get('overlay_trades_count',0))}"
        )

    baseline_df = pd.DataFrame(baseline_rows).sort_values("symbol").reset_index(drop=True)
    overlay_df = pd.DataFrame(overlay_rows).sort_values("symbol").reset_index(drop=True)
    baseline_csv = reports_root / "baseline_1h_report.csv"
    overlay_csv = reports_root / "overlay_1s_report.csv"
    baseline_df.to_csv(baseline_csv, index=False)
    overlay_df.to_csv(overlay_csv, index=False)

    final_md = reports_root / "final_summary.md"
    final_md.write_text(
        "\n".join(
            [
                "# Final Summary",
                "",
                "## Likely Explosion Source",
                "- Prior runs used high starting equity and/or unconstrained allocation (up to ~0.95), producing extreme compounded growth.",
                "- This pipeline hard-enforces initial_equity=100 and max_allocation<=0.7.",
                "",
                "## Code Changes",
                "- Added strict audit invariants and debug snapshots in `src/bot087/execution/baseline_audit.py`.",
                "- Added one-shot pipeline command `scripts/run_audit_baseline_overlay.py`.",
                "",
                "## Results",
                *summary_lines,
                "",
                "## Invariant Status",
                "- All audited symbols completed with `audit.ok = true`.",
                "- Overlay trade count never exceeded baseline opportunities.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"baseline_report={baseline_csv}")
    print(f"overlay_report={overlay_csv}")
    print(f"final_summary={final_md}")
    print(f"overlay_run_id={run_id}")


if __name__ == "__main__":
    main()
