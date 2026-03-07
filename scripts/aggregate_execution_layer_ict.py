#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _num(df: pd.DataFrame, cols: List[str], default: float = np.nan) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.full(len(df), default, dtype=float), index=df.index)


def _bool(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    x = _num(df, cols, default=np.nan)
    if x.notna().any():
        return x.fillna(0).astype(int)
    for c in cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip().str.lower()
            return s.isin({"1", "true", "t", "yes", "y"}).astype(int)
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index)


def _str(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    for c in cols:
        if c in df.columns:
            return df[c].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def _derive_mode(fp: Path, df: pd.DataFrame) -> str:
    if "exec_mode" in df.columns:
        vals = df["exec_mode"].dropna().astype(str).str.strip().str.lower()
        vals = vals[vals != ""]
        if not vals.empty:
            return str(vals.iloc[0])
    name = fp.name.lower()
    if "exec_limit_vs_baseline" in name:
        return "exec_limit"
    if "exec_baseline_vs_baseline" in name:
        return "baseline"
    if "exec_ict_vs_baseline" in name:
        return "ict_gate"
    return "unknown"


def _derive_pnl(df: pd.DataFrame, prefix: str) -> pd.Series:
    direct = _num(df, [f"{prefix}_pnl_pct"], default=np.nan)
    if direct.notna().any():
        return direct
    entry = _num(df, [f"{prefix}_entry_price"], default=np.nan)
    exit_ = _num(df, [f"{prefix}_exit_price"], default=np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pnl = (exit_ / entry) - 1.0
    pnl[(~np.isfinite(pnl)) | (entry <= 0)] = np.nan
    return pnl


def _derive_pnl_net(df: pd.DataFrame, prefix: str) -> pd.Series:
    direct = _num(df, [f"{prefix}_pnl_net_pct", f"{prefix}_pnl_pct"], default=np.nan)
    if direct.notna().any():
        return direct
    return _derive_pnl(df, prefix)


def _parse_file(fp: Path) -> Tuple[Dict[str, Any], Counter]:
    df = pd.read_csv(fp)
    n = int(len(df))

    mode = _derive_mode(fp, df)
    alt_prefix = "exec" if "exec_filled" in df.columns else ("ict" if "ict_filled" in df.columns else "exec")
    skip_col = f"{alt_prefix}_skip_reason"

    b_filled = _bool(df, ["baseline_filled"])
    e_filled = _bool(df, [f"{alt_prefix}_filled"])
    b_sl = _bool(df, ["baseline_sl_hit"])
    e_sl = _bool(df, [f"{alt_prefix}_sl_hit"])
    b_tp = _bool(df, ["baseline_tp_hit"])
    e_tp = _bool(df, [f"{alt_prefix}_tp_hit"])

    b_inv_stop = _bool(df, ["baseline_invalid_stop_geometry"])
    b_inv_tp = _bool(df, ["baseline_invalid_tp_geometry"])
    e_inv_stop = _bool(df, [f"{alt_prefix}_invalid_stop_geometry", "invalid_stop_geometry"])
    e_inv_tp = _bool(df, [f"{alt_prefix}_invalid_tp_geometry", "invalid_tp_geometry"])

    b_valid = (b_inv_stop == 0) & (b_inv_tp == 0)
    e_valid = (e_inv_stop == 0) & (e_inv_tp == 0)

    b_entry_mask = (b_filled == 1) & b_valid
    e_entry_mask = (e_filled == 1) & e_valid

    b_entries = int(b_entry_mask.sum())
    e_entries = int(e_entry_mask.sum())

    b_sl_rate = float(((b_sl == 1) & b_entry_mask).sum() / b_entries) if b_entries > 0 else float("nan")
    e_sl_rate = float(((e_sl == 1) & e_entry_mask).sum() / e_entries) if e_entries > 0 else float("nan")
    b_tp_rate = float(((b_tp == 1) & b_entry_mask).sum() / b_entries) if b_entries > 0 else float("nan")
    e_tp_rate = float(((e_tp == 1) & e_entry_mask).sum() / e_entries) if e_entries > 0 else float("nan")

    skip_mask = (e_filled == 0)
    avoided_losses = int((skip_mask & (b_sl == 1) & b_valid).sum())
    missed_wins = int((skip_mask & (b_tp == 1) & b_valid).sum())

    b_pnl = _derive_pnl(df, "baseline")
    e_pnl = _derive_pnl(df, alt_prefix)
    b_pnl_net = _derive_pnl_net(df, "baseline")
    e_pnl_net = _derive_pnl_net(df, alt_prefix)
    b_mae = _num(df, ["baseline_mae_pct"], default=np.nan)
    e_mae = _num(df, [f"{alt_prefix}_mae_pct"], default=np.nan)
    b_mfe = _num(df, ["baseline_mfe_pct"], default=np.nan)
    e_mfe = _num(df, [f"{alt_prefix}_mfe_pct"], default=np.nan)
    entry_imp = _num(df, ["entry_improvement_pct", "entry_price_delta_pct"], default=np.nan)

    fair = e_entry_mask
    b_mae_med = float(b_mae[fair].median()) if fair.any() and b_mae[fair].notna().any() else float("nan")
    e_mae_med = float(e_mae[fair].median()) if fair.any() and e_mae[fair].notna().any() else float("nan")
    b_mfe_med = float(b_mfe[fair].median()) if fair.any() and b_mfe[fair].notna().any() else float("nan")
    e_mfe_med = float(e_mfe[fair].median()) if fair.any() and e_mfe[fair].notna().any() else float("nan")

    b_pnl_sum = float(b_pnl[b_entry_mask].sum()) if b_entry_mask.any() else float("nan")
    e_pnl_sum = float(e_pnl[e_entry_mask].sum()) if e_entry_mask.any() else float("nan")
    b_pnl_net_sum = float(b_pnl_net[b_entry_mask].sum()) if b_entry_mask.any() else float("nan")
    e_pnl_net_sum = float(e_pnl_net[e_entry_mask].sum()) if e_entry_mask.any() else float("nan")
    b_pnl_med = float(b_pnl[b_entry_mask].median()) if b_entry_mask.any() and b_pnl[b_entry_mask].notna().any() else float("nan")
    e_pnl_med = float(e_pnl[e_entry_mask].median()) if e_entry_mask.any() and e_pnl[e_entry_mask].notna().any() else float("nan")
    imp_mask = e_entry_mask & entry_imp.notna()
    imp_mean = float(entry_imp[imp_mask].mean()) if imp_mask.any() else float("nan")
    imp_med = float(entry_imp[imp_mask].median()) if imp_mask.any() else float("nan")

    skip = _str(df, [skip_col]).str.strip()
    skip = skip[~skip.str.lower().isin({"", "nan", "none", "null"})]
    skip_hist = Counter(skip.tolist())
    liq = _str(df, [f"{alt_prefix}_fill_liquidity_type", "fill_liquidity_type"]).str.strip().str.lower()
    taker_share = float(((liq == "taker") & e_entry_mask).sum() / e_entries) if e_entries > 0 else float("nan")

    run_id = fp.parent.name
    symbol = fp.name.split("_")[0].upper()
    row = {
        "run_id": run_id,
        "symbol": symbol,
        "mode": mode,
        "file_path": str(fp),
        "signals_total": n,
        "baseline_entries": b_entries,
        "mode_entries": e_entries,
        "entry_rate_mode": float(e_entries / n) if n > 0 else float("nan"),
        "baseline_sl_hit_rate": b_sl_rate,
        "mode_sl_hit_rate": e_sl_rate,
        "baseline_tp_hit_rate": b_tp_rate,
        "mode_tp_hit_rate": e_tp_rate,
        "avoided_losses": avoided_losses,
        "missed_wins": missed_wins,
        "baseline_pnl_sum": b_pnl_sum,
        "mode_pnl_sum": e_pnl_sum,
        "baseline_pnl_net_sum": b_pnl_net_sum,
        "mode_pnl_net_sum": e_pnl_net_sum,
        "baseline_pnl_median": b_pnl_med,
        "mode_pnl_median": e_pnl_med,
        "mode_taker_share": taker_share,
        "baseline_mae_median": b_mae_med,
        "mode_mae_median": e_mae_med,
        "baseline_mfe_median": b_mfe_med,
        "mode_mfe_median": e_mfe_med,
        "entry_improvement_mean": imp_mean,
        "entry_improvement_median": imp_med,
        "skip_reason_top3": "|".join([f"{k}:{int(v)}" for k, v in skip_hist.most_common(3)]),
    }
    return row, skip_hist


def _to_md_table(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["(none)"]
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def run(args: argparse.Namespace) -> None:
    base = _resolve_path(args.base_dir)
    found = sorted(base.glob("**/*_vs_baseline.csv"))
    if not found:
        raise SystemExit(f"No execution comparison files found under {base}")

    # Deduplicate by resolved full path.
    uniq: Dict[str, Path] = {}
    dup_count = 0
    for fp in found:
        k = str(fp.resolve())
        if k in uniq:
            dup_count += 1
            continue
        uniq[k] = fp
    files = [uniq[k] for k in sorted(uniq.keys())]

    rows: List[Dict[str, Any]] = []
    global_skip_by_mode: Dict[str, Counter] = {}
    for fp in files:
        try:
            row, hist = _parse_file(fp)
            rows.append(row)
            mode = str(row.get("mode", "unknown"))
            global_skip_by_mode.setdefault(mode, Counter()).update(hist)
        except Exception as ex:
            rows.append(
                {
                    "run_id": fp.parent.name,
                    "symbol": fp.name.split("_")[0].upper(),
                    "mode": "parse_error",
                    "file_path": str(fp),
                    "signals_total": 0,
                    "baseline_entries": 0,
                    "mode_entries": 0,
                    "entry_rate_mode": float("nan"),
                    "baseline_sl_hit_rate": float("nan"),
                    "mode_sl_hit_rate": float("nan"),
                    "baseline_tp_hit_rate": float("nan"),
                    "mode_tp_hit_rate": float("nan"),
                    "avoided_losses": 0,
                    "missed_wins": 0,
                    "baseline_pnl_sum": float("nan"),
                    "mode_pnl_sum": float("nan"),
                    "baseline_pnl_net_sum": float("nan"),
                    "mode_pnl_net_sum": float("nan"),
                    "baseline_pnl_median": float("nan"),
                    "mode_pnl_median": float("nan"),
                    "mode_taker_share": float("nan"),
                    "baseline_mae_median": float("nan"),
                    "mode_mae_median": float("nan"),
                    "baseline_mfe_median": float("nan"),
                    "mode_mfe_median": float("nan"),
                    "entry_improvement_mean": float("nan"),
                    "entry_improvement_median": float("nan"),
                    "skip_reason_top3": f"parse_error:{type(ex).__name__}",
                }
            )

    agg = pd.DataFrame(rows).sort_values(["mode", "symbol", "run_id"]).reset_index(drop=True)

    out_csv = _resolve_path(args.out_csv)
    out_md = _resolve_path(args.out_md)
    out_files = _resolve_path(args.out_files)
    out_test_csv = _resolve_path(args.out_test_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out_csv, index=False)
    out_files.write_text("\n".join([str(x.resolve()) for x in files]) + "\n", encoding="utf-8")

    mode_rollup = (
        agg.groupby("mode", as_index=False)
        .agg(
            files=("file_path", "count"),
            signals_total=("signals_total", "sum"),
            baseline_entries=("baseline_entries", "sum"),
            mode_entries=("mode_entries", "sum"),
            avoided_losses=("avoided_losses", "sum"),
            missed_wins=("missed_wins", "sum"),
            baseline_pnl_sum=("baseline_pnl_sum", "sum"),
            mode_pnl_sum=("mode_pnl_sum", "sum"),
            baseline_pnl_net_sum=("baseline_pnl_net_sum", "sum"),
            mode_pnl_net_sum=("mode_pnl_net_sum", "sum"),
        )
    )
    mode_rollup["entry_rate_mode"] = mode_rollup["mode_entries"] / mode_rollup["signals_total"].replace(0, np.nan)
    mode_rollup["pnl_delta"] = mode_rollup["mode_pnl_sum"] - mode_rollup["baseline_pnl_sum"]
    mode_rollup["pnl_net_delta"] = mode_rollup["mode_pnl_net_sum"] - mode_rollup["baseline_pnl_net_sum"]
    mode_rollup["baseline_expectancy"] = mode_rollup["baseline_pnl_sum"] / mode_rollup["baseline_entries"].replace(0, np.nan)
    mode_rollup["mode_expectancy"] = mode_rollup["mode_pnl_sum"] / mode_rollup["mode_entries"].replace(0, np.nan)
    mode_rollup["baseline_expectancy_net"] = mode_rollup["baseline_pnl_net_sum"] / mode_rollup["baseline_entries"].replace(0, np.nan)
    mode_rollup["mode_expectancy_net"] = mode_rollup["mode_pnl_net_sum"] / mode_rollup["mode_entries"].replace(0, np.nan)
    taker_rollup = (
        agg.assign(_taker_num=agg["mode_taker_share"] * agg["mode_entries"])
        .groupby("mode", as_index=False)
        .agg(_taker_num=("_taker_num", "sum"), mode_entries=("mode_entries", "sum"))
    )
    taker_rollup["mode_taker_share"] = taker_rollup["_taker_num"] / taker_rollup["mode_entries"].replace(0, np.nan)
    mode_rollup = mode_rollup.merge(taker_rollup[["mode", "mode_taker_share"]], on="mode", how="left")

    symbol_mode_rollup = (
        agg.groupby(["symbol", "mode"], as_index=False)
        .agg(
            signals_total=("signals_total", "sum"),
            baseline_entries=("baseline_entries", "sum"),
            mode_entries=("mode_entries", "sum"),
            avoided_losses=("avoided_losses", "sum"),
            missed_wins=("missed_wins", "sum"),
            baseline_pnl_sum=("baseline_pnl_sum", "sum"),
            mode_pnl_sum=("mode_pnl_sum", "sum"),
            baseline_pnl_net_sum=("baseline_pnl_net_sum", "sum"),
            mode_pnl_net_sum=("mode_pnl_net_sum", "sum"),
        )
    )
    symbol_mode_rollup["entry_rate_mode"] = symbol_mode_rollup["mode_entries"] / symbol_mode_rollup["signals_total"].replace(0, np.nan)
    symbol_mode_rollup["pnl_delta"] = symbol_mode_rollup["mode_pnl_sum"] - symbol_mode_rollup["baseline_pnl_sum"]
    symbol_mode_rollup["pnl_net_delta"] = symbol_mode_rollup["mode_pnl_net_sum"] - symbol_mode_rollup["baseline_pnl_net_sum"]
    symbol_taker = (
        agg.assign(_taker_num=agg["mode_taker_share"] * agg["mode_entries"])
        .groupby(["symbol", "mode"], as_index=False)
        .agg(_taker_num=("_taker_num", "sum"), mode_entries=("mode_entries", "sum"))
    )
    symbol_taker["mode_taker_share"] = symbol_taker["_taker_num"] / symbol_taker["mode_entries"].replace(0, np.nan)
    symbol_mode_rollup = symbol_mode_rollup.merge(symbol_taker[["symbol", "mode", "mode_taker_share"]], on=["symbol", "mode"], how="left")

    wf_found = sorted(base.glob("**/*_walkforward_test_summary.csv"))
    wf_uniq: Dict[str, Path] = {}
    for fp in wf_found:
        wf_uniq[str(fp.resolve())] = fp
    wf_files = [wf_uniq[k] for k in sorted(wf_uniq.keys())]
    wf_summary = pd.DataFrame()
    wf_overall = pd.DataFrame()
    if wf_files:
        wf_rows = []
        for fp in wf_files:
            try:
                w = pd.read_csv(fp)
                w["_wf_file"] = str(fp.resolve())
                w["_wf_folder"] = str(fp.parent.resolve())
                wf_rows.append(w)
            except Exception:
                continue
        if wf_rows:
            wf = pd.concat(wf_rows, ignore_index=True)
            if int(getattr(args, "walkforward_latest_only", 1)) == 1 and "symbol" in wf.columns:
                wf["run_id"] = wf.get("run_id", "").astype(str)
                wf = (
                    wf.sort_values(["symbol", "run_id", "_wf_folder"], ascending=[True, True, True])
                    .groupby("symbol", as_index=False)
                    .tail(1)
                    .reset_index(drop=True)
                )
            wf["test_signals"] = pd.to_numeric(wf.get("test_signals", np.nan), errors="coerce")
            wf["entries"] = pd.to_numeric(wf.get("entries", np.nan), errors="coerce")
            wf["entry_rate"] = pd.to_numeric(wf.get("entry_rate", np.nan), errors="coerce")
            wf["taker_share"] = pd.to_numeric(wf.get("taker_share", np.nan), errors="coerce")
            wf["pnl_net_sum"] = pd.to_numeric(wf.get("pnl_net_sum", np.nan), errors="coerce")
            wf["expectancy_net"] = pd.to_numeric(wf.get("expectancy_net", np.nan), errors="coerce")
            wf["sl_hit_rate"] = pd.to_numeric(wf.get("sl_hit_rate", np.nan), errors="coerce")
            wf["tp_hit_rate"] = pd.to_numeric(wf.get("tp_hit_rate", np.nan), errors="coerce")
            wf["median_entry_improvement_pct"] = pd.to_numeric(wf.get("median_entry_improvement_pct", np.nan), errors="coerce")
            wf["max_fill_delay_min"] = pd.to_numeric(wf.get("max_fill_delay_min", np.nan), errors="coerce")
            wf["baseline_pnl_net_sum"] = pd.to_numeric(wf.get("baseline_pnl_net_sum", np.nan), errors="coerce")
            wf_summary = (
                wf.groupby("symbol", as_index=False)
                .agg(
                    runs=("run_id", "count"),
                    test_signals=("test_signals", "sum"),
                    entries=("entries", "sum"),
                    baseline_pnl_net_sum=("baseline_pnl_net_sum", "sum"),
                    pnl_net_sum=("pnl_net_sum", "sum"),
                    entry_rate=("entry_rate", "mean"),
                    taker_share=("taker_share", "mean"),
                    expectancy_net=("expectancy_net", "mean"),
                    sl_hit_rate=("sl_hit_rate", "mean"),
                    tp_hit_rate=("tp_hit_rate", "mean"),
                    entry_improvement_pct=("median_entry_improvement_pct", "mean"),
                    max_fill_delay_min=("max_fill_delay_min", "mean"),
                )
            )
            total_test_signals = float(wf_summary["test_signals"].sum()) if not wf_summary.empty else float("nan")
            total_entries = float(wf_summary["entries"].sum()) if not wf_summary.empty else float("nan")
            total_pnl_net = float(wf_summary["pnl_net_sum"].sum()) if not wf_summary.empty else float("nan")
            total_baseline_pnl_net = float(wf_summary["baseline_pnl_net_sum"].sum()) if not wf_summary.empty else float("nan")
            wf_overall = pd.DataFrame(
                [
                    {
                        "row": "overall_test",
                        "symbol": "ALL",
                        "runs": int(len(wf_summary)),
                        "test_signals": total_test_signals,
                        "entries": total_entries,
                        "entry_rate": float(total_entries / total_test_signals) if np.isfinite(total_test_signals) and total_test_signals > 0 else float("nan"),
                        "taker_share": float((wf["taker_share"] * wf["entries"]).sum() / max(1.0, wf["entries"].sum())) if not wf.empty else float("nan"),
                        "baseline_pnl_net_sum": total_baseline_pnl_net,
                        "pnl_net_sum": total_pnl_net,
                        "expectancy_net": float(total_pnl_net / total_entries) if np.isfinite(total_entries) and total_entries > 0 else float("nan"),
                        "sl_hit_rate": float((wf["sl_hit_rate"] * wf["entries"]).sum() / max(1.0, wf["entries"].sum())) if not wf.empty else float("nan"),
                        "tp_hit_rate": float((wf["tp_hit_rate"] * wf["entries"]).sum() / max(1.0, wf["entries"].sum())) if not wf.empty else float("nan"),
                        "entry_improvement_pct": float((wf["median_entry_improvement_pct"] * wf["entries"]).sum() / max(1.0, wf["entries"].sum())) if not wf.empty else float("nan"),
                        "max_fill_delay_min": float((wf["max_fill_delay_min"] * wf["entries"]).sum() / max(1.0, wf["entries"].sum())) if not wf.empty else float("nan"),
                    }
                ]
            )
            out_cols = [
                "row",
                "symbol",
                "runs",
                "test_signals",
                "entries",
                "entry_rate",
                "taker_share",
                "baseline_pnl_net_sum",
                "pnl_net_sum",
                "expectancy_net",
                "sl_hit_rate",
                "tp_hit_rate",
                "entry_improvement_pct",
                "max_fill_delay_min",
            ]
            out_test = pd.concat([wf_summary.assign(row="per_symbol"), wf_overall], ignore_index=True, sort=False)
            out_test = out_test.reindex(columns=out_cols)
            out_test.to_csv(out_test_csv, index=False)

    lines: List[str] = []
    lines.append("# Aggregated Execution Report")
    lines.append("")
    lines.append(f"- Generated UTC: {_utc_now_iso()}")
    lines.append(f"- Base dir: `{base}`")
    lines.append(f"- Files discovered: {len(found)}")
    lines.append(f"- Files used (deduped): {len(files)}")
    if dup_count > 0:
        lines.append("")
        lines.append("## WARNING")
        lines.append("")
        lines.append(f"- Duplicate file paths detected and removed: {dup_count}")
    lines.append("")
    lines.append("## Overall By Mode")
    lines.append("")
    lines.extend(
        _to_md_table(
            mode_rollup[
                [
                    "mode",
                    "files",
                    "signals_total",
                    "mode_entries",
                    "entry_rate_mode",
                    "avoided_losses",
                    "missed_wins",
                    "baseline_pnl_sum",
                    "mode_pnl_sum",
                    "pnl_delta",
                    "baseline_pnl_net_sum",
                    "mode_pnl_net_sum",
                    "pnl_net_delta",
                    "baseline_expectancy",
                    "mode_expectancy",
                    "baseline_expectancy_net",
                    "mode_expectancy_net",
                    "mode_taker_share",
                ]
            ]
            if not mode_rollup.empty
            else pd.DataFrame()
        )
    )
    lines.append("")
    lines.append("## Per Symbol / Mode")
    lines.append("")
    lines.extend(
        _to_md_table(
            symbol_mode_rollup[
                [
                    "symbol",
                    "mode",
                    "signals_total",
                    "mode_entries",
                    "entry_rate_mode",
                    "avoided_losses",
                    "missed_wins",
                    "baseline_pnl_sum",
                    "mode_pnl_sum",
                    "pnl_delta",
                    "baseline_pnl_net_sum",
                    "mode_pnl_net_sum",
                    "pnl_net_delta",
                    "mode_taker_share",
                ]
            ]
            if not symbol_mode_rollup.empty
            else pd.DataFrame()
        )
    )
    lines.append("")
    lines.append("## Top Skip Reasons By Mode")
    lines.append("")
    if global_skip_by_mode:
        for mode, hist in sorted(global_skip_by_mode.items(), key=lambda x: x[0]):
            lines.append(f"- {mode}:")
            top = hist.most_common(10)
            if not top:
                lines.append("  - none")
            else:
                for k, v in top:
                    lines.append(f"  - {k}: {int(v)}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Included Files")
    lines.append("")
    for fp in files:
        lines.append(f"- `{fp.resolve()}`")
    if not wf_summary.empty:
        lines.append("")
        lines.append("## Walkforward Test-Only Per Symbol")
        lines.append("")
        lines.extend(
            _to_md_table(
                wf_summary[
                    [
                        "symbol",
                        "runs",
                        "test_signals",
                        "entries",
                        "entry_rate",
                        "taker_share",
                        "baseline_pnl_net_sum",
                        "pnl_net_sum",
                        "expectancy_net",
                        "sl_hit_rate",
                        "tp_hit_rate",
                        "entry_improvement_pct",
                        "max_fill_delay_min",
                    ]
                ]
            )
        )
        lines.append("")
        lines.append("## Walkforward Test-Only Overall")
        lines.append("")
        lines.extend(_to_md_table(wf_overall))
        lines.append("")
        lines.append(f"- Walkforward test summary CSV: `{out_test_csv}`")

    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(str(out_csv))
    print(str(out_md))
    print(str(out_files))
    if out_test_csv.exists():
        print(str(out_test_csv))


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Aggregate execution-layer diagnostics across modes.")
    ap.add_argument("--base-dir", default="reports/execution_layer")
    ap.add_argument("--out-csv", default="reports/execution_layer/AGG_exec_report.csv")
    ap.add_argument("--out-md", default="reports/execution_layer/AGG_exec_report.md")
    ap.add_argument("--out-files", default="reports/execution_layer/AGG_exec_included_files.txt")
    ap.add_argument("--out-test-csv", default="reports/execution_layer/AGG_exec_testonly_summary.csv")
    ap.add_argument("--walkforward-latest-only", type=int, default=1)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
