#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.backtest.exec_1s import Exec1SConfig, run_backtest_long_only_exec_1s  # noqa: E402
from src.bot087.optim.ga import _ensure_indicators, _norm_params, run_backtest_long_only  # noqa: E402


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _unwrap_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
        return dict(raw["params"])
    return dict(raw)


def load_df_1h(symbol: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing 1h parquet: {fp}")
    df = pd.read_parquet(fp)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df


def load_active_params(symbol: str) -> Dict[str, Any]:
    fp = PROJECT_ROOT / "data" / "metadata" / "params" / f"{symbol}_active_params.json"
    if not fp.exists():
        raise FileNotFoundError(f"Missing active params: {fp}")
    return _norm_params(_unwrap_params(_load_json(fp)))


def merge_date_intervals(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not intervals:
        return []
    ints = sorted(intervals, key=lambda x: x[0])
    out = [ints[0]]
    for s, e in ints[1:]:
        ps, pe = out[-1]
        if s <= pe + pd.Timedelta(days=1):
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def build_trade_intervals(
    trades: List[Dict[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    buffer_days: int,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not trades:
        return []
    ints: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for t in trades:
        ets = pd.to_datetime(t.get("entry_ts"), utc=True, errors="coerce")
        xts = pd.to_datetime(t.get("exit_ts"), utc=True, errors="coerce")
        if pd.isna(ets) or pd.isna(xts):
            continue
        lo = min(ets, xts).floor("D") - pd.Timedelta(days=buffer_days)
        hi = max(ets, xts).floor("D") + pd.Timedelta(days=buffer_days)
        lo = max(lo, start.floor("D"))
        hi = min(hi, end.floor("D"))
        if hi >= lo:
            ints.append((lo, hi))
    return merge_date_intervals(ints)


def intervals_total_days(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> int:
    total = 0
    for s, e in intervals:
        total += int((e - s).days) + 1
    return total


def cap_to_recent_days(intervals: List[Tuple[pd.Timestamp, pd.Timestamp]], max_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if max_days <= 0:
        return intervals
    out: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    used = 0
    for s, e in sorted(intervals, key=lambda x: x[1], reverse=True):
        days = int((e - s).days) + 1
        if used >= max_days:
            break
        if used + days <= max_days:
            out.append((s, e))
            used += days
        else:
            keep = max_days - used
            ns = e - pd.Timedelta(days=keep - 1)
            out.append((ns, e))
            used = max_days
    return merge_date_intervals(sorted(out, key=lambda x: x[0]))


def run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True, help="Output summary.csv from optimize_all_symbols_1h_long.py")
    ap.add_argument("--symbols", default="", help="Optional comma list; default = passed symbols in summary")
    ap.add_argument("--from-date", default="2025-01-01")
    ap.add_argument("--to-date", default="2025-12-31")
    ap.add_argument("--buffer-days", type=int, default=1)
    ap.add_argument("--max-days-per-symbol", type=int, default=120)
    ap.add_argument("--raw-dir", default="data/raw/binance_vision_targeted")
    ap.add_argument("--tmp-proc-dir", default="data/processed/sec_trades_targeted_tmp")
    ap.add_argument("--out-sec-dir", default="data/processed/_sec")
    ap.add_argument("--cleanup", action="store_true", default=True)
    args = ap.parse_args()

    summary = pd.read_csv(args.summary_csv)
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        if "passed" in summary.columns:
            symbols = summary.loc[summary["passed"] == 1, "symbol"].astype(str).str.upper().tolist()
        else:
            symbols = summary["symbol"].astype(str).str.upper().tolist()
    symbols = sorted(set(symbols))

    if not symbols:
        raise SystemExit("No symbols selected for execution gate evaluation.")

    from_ts = pd.Timestamp(args.from_date, tz="UTC")
    to_ts = pd.Timestamp(args.to_date, tz="UTC")
    out_dir = PROJECT_ROOT / "artifacts" / "reports" / "exec_gate" / _utc_tag()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_root = (PROJECT_ROOT / args.raw_dir).resolve()
    tmp_proc_root = (PROJECT_ROOT / args.tmp_proc_dir).resolve()
    out_sec_root = (PROJECT_ROOT / args.out_sec_dir).resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    tmp_proc_root.mkdir(parents=True, exist_ok=True)
    out_sec_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for sym in symbols:
        print(f"\n=== EXEC GATE {sym} ===", flush=True)
        df_1h = load_df_1h(sym)
        params = load_active_params(sym)
        df_eval = df_1h[(df_1h["Timestamp"] >= from_ts) & (df_1h["Timestamp"] <= to_ts)].reset_index(drop=True)
        if df_eval.empty:
            print(f"[{sym}] skip: empty 1h eval range")
            continue

        # 1h trade timestamps determine targeted 1s download windows
        trades_1h, m_1h = run_backtest_long_only(
            df=_ensure_indicators(df_eval.copy(), params),
            symbol=sym,
            p=params,
            initial_equity=10_000.0,
            fee_bps=7.0,
            slippage_bps=2.0,
            collect_trades=True,
        )
        intervals = build_trade_intervals(trades_1h, from_ts, to_ts, buffer_days=args.buffer_days)
        days_before = intervals_total_days(intervals)
        intervals = cap_to_recent_days(intervals, args.max_days_per_symbol)
        days_after = intervals_total_days(intervals)
        if not intervals:
            print(f"[{sym}] skip: no trade intervals in range")
            continue

        # download targeted daily aggTrades
        for s, e in intervals:
            start_s = s.strftime("%Y-%m-%d")
            end_s = (e + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # downloader end is exclusive
            run_cmd(
                [
                    "scripts/venv/bin/python",
                    "scripts/download_sec_aggtrades_binance_vision.py",
                    "--symbol",
                    sym,
                    "--market",
                    "spot",
                    "--start",
                    start_s,
                    "--end",
                    end_s,
                    "--raw-dir",
                    str(raw_root),
                    "--processed-dir",
                    str(tmp_proc_root),
                ]
            )

        raw_symbol_dir = raw_root / "spot" / "aggTrades" / sym
        sec_out = out_sec_root / f"{sym}_1s_ohlcv_targeted.parquet"
        if not raw_symbol_dir.exists():
            print(f"[{sym}] skip: no raw zip files downloaded")
            continue

        run_cmd(
            [
                "scripts/venv/bin/python",
                "scripts/build_1s_ohlcv_from_aggtrades.py",
                "--raw_dir",
                str(raw_symbol_dir),
                "--out_path",
                str(sec_out),
            ]
        )

        sec_df = pd.read_parquet(sec_out, columns=["Timestamp", "Open", "High", "Low", "Close"])

        cfgs = {
            "strict_default": Exec1SConfig(),
            "relaxed_confirm": Exec1SConfig(
                confirm_window_sec=1800,
                confirm_bps=0.5,
                abort_bps=120.0,
                pullback_bps=2.0,
                pullback_window_sec=300,
                market_on_no_pullback=True,
                market_on_no_confirm=False,
            ),
            "market_on_no_confirm": Exec1SConfig(
                confirm_window_sec=600,
                confirm_bps=5.0,
                abort_bps=30.0,
                pullback_bps=6.0,
                pullback_window_sec=300,
                market_on_no_pullback=True,
                market_on_no_confirm=True,
            ),
        }

        for name, cfg in cfgs.items():
            _, m_exec = run_backtest_long_only_exec_1s(
                df_1h=df_eval,
                sec_df=sec_df,
                symbol=sym,
                params=params,
                exec_cfg=cfg,
                initial_equity=10_000.0,
                fee_bps=7.0,
                slippage_bps=2.0,
            )
            rows.append(
                {
                    "symbol": sym,
                    "config": name,
                    "one_h_net": float(m_1h.get("net_profit", 0.0)),
                    "one_h_trades": float(m_1h.get("trades", 0.0)),
                    "exec_net": float(m_exec.get("net_profit", 0.0)),
                    "exec_trades": float(m_exec.get("trades", 0.0)),
                    "exec_pf": float(m_exec.get("profit_factor", 0.0)),
                    "exec_dd": float(m_exec.get("max_dd", 0.0)),
                    "diag_signal_hours": float(m_exec.get("diag_signal_hours", 0.0)),
                    "diag_entry_attempts": float(m_exec.get("diag_entry_attempts", 0.0)),
                    "diag_entries_opened": float(m_exec.get("diag_entries_opened", 0.0)),
                    "diag_no_confirm_cross": float(m_exec.get("diag_entry_no_confirm_cross", 0.0)),
                    "diag_market_on_no_confirm": float(m_exec.get("diag_entry_market_on_no_confirm", 0.0)),
                    "diag_pullback_fill": float(m_exec.get("diag_entry_pullback_fill", 0.0)),
                    "diag_market_fallback": float(m_exec.get("diag_entry_market_fallback", 0.0)),
                    "trade_days_before_cap": days_before,
                    "trade_days_after_cap": days_after,
                    "sec_path": str(sec_out),
                }
            )

        if args.cleanup:
            # keep final 1s parquet only
            if raw_symbol_dir.exists():
                shutil.rmtree(raw_symbol_dir, ignore_errors=True)
            tmp_sym_dir = tmp_proc_root / sym
            if tmp_sym_dir.exists():
                shutil.rmtree(tmp_sym_dir, ignore_errors=True)

    rep = pd.DataFrame(rows)
    rep_csv = out_dir / "exec_gate_summary.csv"
    rep_json = out_dir / "exec_gate_summary.json"
    rep.to_csv(rep_csv, index=False)
    rep_json.write_text(rep.to_json(orient="records", indent=2), encoding="utf-8")

    print("\n=== EXEC GATE SUMMARY ===")
    if not rep.empty:
        print(
            rep[
                [
                    "symbol",
                    "config",
                    "one_h_net",
                    "one_h_trades",
                    "exec_net",
                    "exec_trades",
                    "exec_pf",
                    "diag_signal_hours",
                    "diag_entries_opened",
                ]
            ].to_string(index=False)
        )
    print(f"\nSaved: {rep_csv}")
    print(f"Saved: {rep_json}")


if __name__ == "__main__":
    main()
