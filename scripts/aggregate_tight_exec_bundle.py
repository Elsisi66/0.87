#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _parse_list(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _improvement_ratio_abs(exec_val: float, base_val: float) -> float:
    if not np.isfinite(exec_val) or not np.isfinite(base_val):
        return float("nan")
    b = abs(float(base_val))
    if b <= 1e-12:
        return float("nan")
    e = abs(float(exec_val))
    return float((b - e) / b)


def run(args: argparse.Namespace) -> Path:
    run_dirs_raw = _parse_list(args.run_dirs)
    if not run_dirs_raw:
        raise SystemExit("run-dirs is required")

    dedup: Dict[str, Path] = {}
    for rd in run_dirs_raw:
        p = _resolve_path(rd)
        dedup[str(p.resolve())] = p.resolve()
    run_dirs = [dedup[k] for k in sorted(dedup.keys())]
    dup_removed = len(run_dirs_raw) - len(run_dirs)

    out_root = _resolve_path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for rd in run_dirs:
        files = sorted(rd.glob("*_walkforward_test_summary.csv"))
        if not files:
            continue
        df = pd.read_csv(files[0])
        if df.empty:
            continue
        r = df.iloc[0].to_dict()
        r["summary_csv"] = str(files[0].resolve())
        rows.append(r)
    if not rows:
        raise SystemExit("No walkforward summary files found in provided run dirs")

    sym = pd.DataFrame(rows)
    keep_cols = [
        "symbol",
        "run_id",
        "signals_total",
        "train_signals",
        "test_signals",
        "entries",
        "entry_rate",
        "taker_share",
        "median_fill_delay_min",
        "max_fill_delay_min",
        "pnl_net_sum",
        "expectancy_net",
        "baseline_pnl_net_sum",
        "baseline_expectancy_net",
        "sl_hit_rate",
        "tp_hit_rate",
        "median_entry_improvement_bps",
        "selected_k",
        "selected_timeout_bars",
        "selected_use_vol_gate",
        "use_vol_regime_gate",
        "vol_regime_max_percentile",
        "tight_mode",
        "constraints_min_entry_rate",
        "constraints_max_fill_delay_min",
        "constraints_max_taker_share",
        "constraints_min_median_entry_improvement_bps",
        "summary_csv",
    ]
    for c in keep_cols:
        if c not in sym.columns:
            sym[c] = np.nan
    sym = sym[keep_cols].copy()
    for c in [
        "test_signals",
        "entries",
        "entry_rate",
        "taker_share",
        "median_fill_delay_min",
        "max_fill_delay_min",
        "pnl_net_sum",
        "expectancy_net",
        "baseline_pnl_net_sum",
        "baseline_expectancy_net",
        "sl_hit_rate",
        "tp_hit_rate",
    ]:
        sym[c] = pd.to_numeric(sym[c], errors="coerce")
    sym = sym.sort_values("symbol").reset_index(drop=True)

    test_sum = float(sym["test_signals"].sum())
    entries_sum = float(sym["entries"].sum())
    overall = {
        "symbols": int(sym["symbol"].nunique()),
        "test_signals": test_sum,
        "entries": entries_sum,
        "entry_rate": float(entries_sum / test_sum) if test_sum > 0 else float("nan"),
        "taker_share": float((sym["taker_share"] * sym["entries"]).sum() / entries_sum) if entries_sum > 0 else float("nan"),
        "median_fill_delay_min_weighted": float((sym["median_fill_delay_min"] * sym["entries"]).sum() / entries_sum) if entries_sum > 0 else float("nan"),
        "max_fill_delay_min": float(sym["max_fill_delay_min"].max()) if not sym["max_fill_delay_min"].isna().all() else float("nan"),
        "exec_pnl_net_sum": float(sym["pnl_net_sum"].sum()),
        "baseline_pnl_net_sum": float(sym["baseline_pnl_net_sum"].sum()),
    }
    overall["exec_expectancy_net"] = float(overall["exec_pnl_net_sum"] / entries_sum) if entries_sum > 0 else float("nan")
    overall["baseline_expectancy_net_proxy"] = float(overall["baseline_pnl_net_sum"] / test_sum) if test_sum > 0 else float("nan")
    overall["delta_expectancy_exec_minus_baseline_proxy"] = float(overall["exec_expectancy_net"] - overall["baseline_expectancy_net_proxy"])

    # Load risk rollup outputs.
    risk_dir = _resolve_path(args.risk_dir)
    risk_sym_fp = risk_dir / "risk_rollup_by_symbol.csv"
    risk_ov_fp = risk_dir / "risk_rollup_overall.csv"
    if not risk_ov_fp.exists():
        raise FileNotFoundError(f"Missing risk overall csv: {risk_ov_fp}")
    risk_ov = pd.read_csv(risk_ov_fp)
    if risk_ov.empty:
        raise SystemExit("risk_rollup_overall.csv is empty")
    rov = risk_ov.iloc[0].to_dict()

    exec_exp = float(rov.get("exec_mean_expectancy_net", np.nan))
    base_exp = float(rov.get("baseline_mean_expectancy_net", np.nan))
    cvar_impr = _improvement_ratio_abs(float(rov.get("exec_cvar_5", np.nan)), float(rov.get("baseline_cvar_5", np.nan)))
    mdd_impr = _improvement_ratio_abs(float(rov.get("exec_max_drawdown", np.nan)), float(rov.get("baseline_max_drawdown", np.nan)))
    taker_ok = bool(np.isfinite(float(rov.get("exec_taker_share", np.nan))) and float(rov.get("exec_taker_share", np.nan)) <= float(args.rubric_max_taker_share))
    delay_ok = bool(np.isfinite(float(rov.get("exec_median_fill_delay_min", np.nan))) and float(rov.get("exec_median_fill_delay_min", np.nan)) <= float(args.rubric_max_median_delay))

    if np.isfinite(exec_exp) and np.isfinite(base_exp) and exec_exp >= base_exp:
        rubric = "exec is improving performance"
    elif (
        np.isfinite(exec_exp)
        and np.isfinite(base_exp)
        and exec_exp < base_exp
        and np.isfinite(cvar_impr)
        and np.isfinite(mdd_impr)
        and cvar_impr >= float(args.rubric_min_tail_improvement)
        and mdd_impr >= float(args.rubric_min_tail_improvement)
        and taker_ok
        and delay_ok
    ):
        rubric = "exec is acceptable as safety layer"
    else:
        rubric = "exec not worth it; focus on 1h edge/stops"

    agg_exec_csv = out_root / "AGG_exec_testonly_summary_tight.csv"
    agg_exec_md = out_root / "AGG_exec_testonly_summary_tight.md"
    agg_risk_csv = out_root / "AGG_risk_rollup_tight_overall.csv"
    agg_risk_md = out_root / "AGG_risk_rollup_tight.md"
    inputs_fp = out_root / "AGG_tight_inputs.txt"
    phase_fp = out_root / "phase_result.md"

    pd.concat([sym, pd.DataFrame([{"symbol": "ALL", **overall}])], ignore_index=True, sort=False).to_csv(agg_exec_csv, index=False)
    risk_ov.to_csv(agg_risk_csv, index=False)

    md1: List[str] = []
    md1.append("# AGG Exec Test-Only Tight")
    md1.append("")
    md1.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    md1.append(f"- Run dirs used (deduped): {len(run_dirs)}")
    if dup_removed > 0:
        md1.append(f"- Duplicates removed: {dup_removed}")
    md1.append("")
    md1.append("## Per Symbol")
    md1.append("")
    for _, r in sym.iterrows():
        md1.append(
            f"- {r['symbol']}: entry_rate={float(r['entry_rate']):.6f}, expectancy_net={float(r['expectancy_net']):.6f}, "
            f"baseline_expectancy_net={float(r['baseline_expectancy_net']):.6f}, taker_share={float(r['taker_share']):.6f}, "
            f"median_fill_delay_min={float(r['median_fill_delay_min']):.2f}"
        )
    md1.append("")
    md1.append("## Overall")
    md1.append("")
    md1.append(f"- entry_rate={float(overall['entry_rate']):.6f}")
    md1.append(f"- exec_expectancy_net={float(overall['exec_expectancy_net']):.6f}")
    md1.append(f"- baseline_expectancy_net_proxy={float(overall['baseline_expectancy_net_proxy']):.6f}")
    md1.append(f"- taker_share={float(overall['taker_share']):.6f}")
    md1.append(f"- median_fill_delay_min_weighted={float(overall['median_fill_delay_min_weighted']):.2f}")
    md1.append("")
    md1.append("## Rubric Decision")
    md1.append("")
    md1.append(f"- Decision: **{rubric}**")
    md1.append(f"- Rule thresholds: tail_improvement>={float(args.rubric_min_tail_improvement):.0%}, taker_share<={float(args.rubric_max_taker_share):.2f}, median_fill_delay<={float(args.rubric_max_median_delay):.0f}")
    md1.append(f"- CVaR improvement ratio: {cvar_impr:.6f}" if np.isfinite(cvar_impr) else "- CVaR improvement ratio: n/a")
    md1.append(f"- MaxDD improvement ratio: {mdd_impr:.6f}" if np.isfinite(mdd_impr) else "- MaxDD improvement ratio: n/a")
    md1.append(f"- Exec expectancy >= baseline expectancy: {int(np.isfinite(exec_exp) and np.isfinite(base_exp) and exec_exp >= base_exp)}")
    md1.append(f"- Taker condition met: {int(taker_ok)}")
    md1.append(f"- Delay condition met: {int(delay_ok)}")
    agg_exec_md.write_text("\n".join(md1).strip() + "\n", encoding="utf-8")

    md2: List[str] = []
    md2.append("# AGG Risk Rollup Tight")
    md2.append("")
    md2.append(f"- Source risk_dir: `{risk_dir}`")
    md2.append(f"- Source risk_overall_csv: `{risk_ov_fp}`")
    md2.append(f"- Source risk_by_symbol_csv: `{risk_sym_fp}`")
    md2.append("")
    for k, v in rov.items():
        if isinstance(v, float):
            md2.append(f"- {k}: {v:.6f}" if np.isfinite(v) else f"- {k}: n/a")
        else:
            md2.append(f"- {k}: {v}")
    md2.append("")
    md2.append(f"- Rubric decision: **{rubric}**")
    agg_risk_md.write_text("\n".join(md2).strip() + "\n", encoding="utf-8")

    inputs_lines: List[str] = []
    for rd in run_dirs:
        inputs_lines.append(str(rd.resolve()))
    inputs_fp.write_text("\n".join(inputs_lines).strip() + "\n", encoding="utf-8")

    phase_lines = [
        "Phase: C/D (tight aggregation + rubric decision)",
        f"Input run dirs: {len(run_dirs)} (duplicates removed={dup_removed})",
        f"Risk source: {risk_ov_fp}",
        f"Overall exec expectancy: {float(exec_exp):.6f}" if np.isfinite(exec_exp) else "Overall exec expectancy: n/a",
        f"Overall baseline expectancy: {float(base_exp):.6f}" if np.isfinite(base_exp) else "Overall baseline expectancy: n/a",
        f"CVaR improvement ratio: {cvar_impr:.6f}" if np.isfinite(cvar_impr) else "CVaR improvement ratio: n/a",
        f"MaxDD improvement ratio: {mdd_impr:.6f}" if np.isfinite(mdd_impr) else "MaxDD improvement ratio: n/a",
        f"Decision: {rubric}",
        f"Artifacts: {agg_exec_csv.name}, {agg_exec_md.name}, {agg_risk_csv.name}, {agg_risk_md.name}",
        "Next: copy tight bundle snapshot into reports/execution_layer/v2_final/",
    ]
    phase_fp.write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")

    print(str(out_root))
    print(str(agg_exec_csv))
    print(str(agg_exec_md))
    print(str(agg_risk_csv))
    print(str(agg_risk_md))
    print(str(phase_fp))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Aggregate tight-mode walkforward outputs + risk rollup rubric.")
    ap.add_argument("--run-dirs", required=True, help="Comma-separated walkforward run directories (tight-mode runs).")
    ap.add_argument("--risk-dir", required=True, help="Directory containing risk_rollup_overall.csv from risk_rollup_exec_layer.py")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--rubric-min-tail-improvement", type=float, default=0.15)
    ap.add_argument("--rubric-max-taker-share", type=float, default=0.25)
    ap.add_argument("--rubric-max-median-delay", type=float, default=45.0)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if not str(args.outdir).strip():
        args.outdir = str(Path("reports/execution_layer") / _utc_tag())
    run(args)


if __name__ == "__main__":
    main()
