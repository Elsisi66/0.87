#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import execution_layer_3m_ict as exec3m  # noqa: E402


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_diag_csv(run_dir: Path, symbol: str) -> Path:
    cand = run_dir / f"{symbol}_exec_limit_vs_baseline.csv"
    if cand.exists():
        return cand
    alt = sorted(run_dir.glob("*_exec_limit_vs_baseline.csv"))
    if alt:
        return alt[0]
    raise FileNotFoundError(f"Missing exec_limit diagnostics under {run_dir}")


def _metrics_from_diag(df: pd.DataFrame) -> Dict[str, float]:
    n = int(len(df))
    if n == 0:
        return {
            "signals_total": 0,
            "entry_rate": float("nan"),
            "pnl_sum": float("nan"),
            "expectancy": float("nan"),
            "sl_hit_rate": float("nan"),
            "tp_hit_rate": float("nan"),
            "median_entry_improvement_pct": float("nan"),
            "max_fill_delay_min": float("nan"),
            "entries": 0,
        }

    filled = pd.to_numeric(df.get("exec_filled", 0), errors="coerce").fillna(0).astype(int)
    sl = pd.to_numeric(df.get("exec_sl_hit", 0), errors="coerce").fillna(0).astype(int)
    tp = pd.to_numeric(df.get("exec_tp_hit", 0), errors="coerce").fillna(0).astype(int)
    pnl = pd.to_numeric(df.get("exec_pnl_pct", np.nan), errors="coerce")
    improve = pd.to_numeric(df.get("entry_improvement_pct", np.nan), errors="coerce")
    delay = pd.to_numeric(df.get("exec_fill_delay_min", df.get("exec_fill_delay_minutes", np.nan)), errors="coerce")
    valid = (
        (pd.to_numeric(df.get("exec_invalid_stop_geometry", 0), errors="coerce").fillna(0).astype(int) == 0)
        & (pd.to_numeric(df.get("exec_invalid_tp_geometry", 0), errors="coerce").fillna(0).astype(int) == 0)
    )
    mask = (filled == 1) & valid

    entries = int(mask.sum())
    sl_rate = float((sl[mask] == 1).sum() / entries) if entries > 0 else float("nan")
    tp_rate = float((tp[mask] == 1).sum() / entries) if entries > 0 else float("nan")
    pnl_sum = float(pnl[mask].sum()) if entries > 0 else float("nan")
    expectancy = float(pnl_sum / entries) if entries > 0 and np.isfinite(pnl_sum) else float("nan")
    med_imp = float(improve[mask].median()) if entries > 0 and improve[mask].notna().any() else float("nan")
    max_delay = float(delay[mask].max()) if entries > 0 and delay[mask].notna().any() else float("nan")

    return {
        "signals_total": n,
        "entry_rate": float(entries / n),
        "pnl_sum": pnl_sum,
        "expectancy": expectancy,
        "sl_hit_rate": sl_rate,
        "tp_hit_rate": tp_rate,
        "median_entry_improvement_pct": med_imp,
        "max_fill_delay_min": max_delay,
        "entries": entries,
    }


def _to_md_table(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["(none)"]
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return out


def run(args: argparse.Namespace) -> Path:
    base_parser = exec3m._build_arg_parser()
    base = base_parser.parse_args([])
    base.rank = int(args.rank)
    base.scan_dir = str(args.scan_dir)
    base.best_csv = str(args.best_csv)
    base.symbol = str(args.symbol)
    base.params_file = str(args.params_file)
    base.timeframe = str(args.timeframe)
    base.pre_buffer_hours = float(args.pre_buffer_hours)
    base.exec_horizon_hours = float(args.exec_horizon_hours)
    base.max_signals = int(args.max_signals)
    base.signal_order = str(args.signal_order)
    base.local_timezone = str(args.local_timezone)
    base.outdir = str(Path(args.outdir).resolve() / _utc_tag() / "runs")
    base.run_ablation = 0
    base.debug_ict = int(args.debug_ict)
    base.exec_mode = "exec_limit"
    base.exec_fallback = "market"

    ks = [round(0.1 * i, 1) for i in range(0, 16)]
    timeouts = [5, 10, 20, 40]
    vol_flags = [0, 1]

    rows: List[Dict[str, Any]] = []
    symbol_seen = ""
    for use_vol in vol_flags:
        for timeout_bars in timeouts:
            for k in ks:
                run_args = copy.deepcopy(base)
                run_args.exec_k = float(k)
                run_args.exec_timeout_bars = int(timeout_bars)
                run_args.use_vol_gate = int(use_vol)
                run_args.exec_use_ladder = 0
                run_args.exec_k1 = float(args.exec_k1)
                run_args.exec_k2 = float(args.exec_k2)
                run_args.vol_z_thr = float(args.vol_z_thr)
                run_args.vol_p_thr = float(args.vol_p_thr)

                run_dir = exec3m.run(run_args)
                meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
                symbol = str(meta.get("symbol", "")).strip().upper()
                if symbol:
                    symbol_seen = symbol
                diag_csv = _load_diag_csv(run_dir, symbol_seen)
                df = pd.read_csv(diag_csv)
                m = _metrics_from_diag(df)
                pass_entry = int(np.isfinite(m["entry_rate"]) and m["entry_rate"] >= float(args.min_entry_rate))
                pass_delay = int(np.isfinite(m["max_fill_delay_min"]) and m["max_fill_delay_min"] <= float(args.max_fill_delay_min))
                rows.append(
                    {
                        "symbol": symbol_seen,
                        "run_id": run_dir.name,
                        "diag_csv": str(diag_csv),
                        "k": float(k),
                        "timeout_bars": int(timeout_bars),
                        "use_vol_gate": int(use_vol),
                        "entry_rate": float(m["entry_rate"]),
                        "entries": int(m["entries"]),
                        "signals_total": int(m["signals_total"]),
                        "pnl_sum": float(m["pnl_sum"]),
                        "expectancy": float(m["expectancy"]),
                        "sl_hit_rate": float(m["sl_hit_rate"]),
                        "tp_hit_rate": float(m["tp_hit_rate"]),
                        "median_entry_improvement_pct": float(m["median_entry_improvement_pct"]),
                        "max_fill_delay_min": float(m["max_fill_delay_min"]),
                        "pass_entry_rate": int(pass_entry),
                        "pass_delay": int(pass_delay),
                        "passes_constraints": int(pass_entry and pass_delay),
                    }
                )

    out_root = Path(base.outdir).parent
    out_root.mkdir(parents=True, exist_ok=True)
    out_csv = out_root / "tuning_summary.csv"
    out_md = out_root / "tuning_report.md"

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["passes_constraints", "expectancy", "pnl_sum"], ascending=[False, False, False]).reset_index(drop=True)
    summary.to_csv(out_csv, index=False)

    lines: List[str] = []
    lines.append("# Exec Limit Tuning Report")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Symbol: `{symbol_seen}`")
    lines.append(f"- Signals max: {int(args.max_signals)}")
    lines.append(f"- Constraints: entry_rate >= {float(args.min_entry_rate):.2f}, max_fill_delay_min <= {float(args.max_fill_delay_min):.1f}")
    lines.append(f"- Output CSV: `{out_csv}`")
    lines.append("")

    passed = summary[summary["passes_constraints"] == 1].head(10)
    lines.append("## Top 10 Passing Configs")
    lines.append("")
    lines.extend(
        _to_md_table(
            passed[
                [
                    "k",
                    "timeout_bars",
                    "use_vol_gate",
                    "entry_rate",
                    "expectancy",
                    "pnl_sum",
                    "sl_hit_rate",
                    "tp_hit_rate",
                    "median_entry_improvement_pct",
                    "max_fill_delay_min",
                ]
            ]
            if not passed.empty
            else pd.DataFrame()
        )
    )
    lines.append("")
    lines.append("## Top 10 Overall")
    lines.append("")
    lines.extend(
        _to_md_table(
            summary.head(10)[
                [
                    "k",
                    "timeout_bars",
                    "use_vol_gate",
                    "entry_rate",
                    "expectancy",
                    "pnl_sum",
                    "sl_hit_rate",
                    "tp_hit_rate",
                    "median_entry_improvement_pct",
                    "max_fill_delay_min",
                    "passes_constraints",
                ]
            ]
            if not summary.empty
            else pd.DataFrame()
        )
    )
    lines.append("")
    lines.append("## Tradeoffs")
    lines.append("")
    lines.append("- Lower k increases fills but can dilute entry improvement.")
    lines.append("- Larger timeout increases participation but can extend fill delay.")
    lines.append("- Volatility gate usually lowers participation and may reduce chaotic fills.")
    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print(str(out_csv))
    print(str(out_md))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Sweep exec_limit settings and rank by expectancy.")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--scan-dir", default="")
    ap.add_argument("--best-csv", default="")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--params-file", default="")
    ap.add_argument("--timeframe", default="3m")
    ap.add_argument("--pre-buffer-hours", type=float, default=6.0)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--max-signals", type=int, default=200)
    ap.add_argument("--signal-order", choices=["latest", "oldest"], default="latest")
    ap.add_argument("--local-timezone", default="Africa/Cairo")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--min-entry-rate", type=float, default=0.50)
    ap.add_argument("--max-fill-delay-min", type=float, default=90.0)
    ap.add_argument("--vol-z-thr", type=float, default=2.5)
    ap.add_argument("--vol-p-thr", type=float, default=95.0)
    ap.add_argument("--exec-k1", type=float, default=0.3)
    ap.add_argument("--exec-k2", type=float, default=0.8)
    ap.add_argument("--debug-ict", type=int, default=0)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
