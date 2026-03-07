#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _latest_run_for_symbol(base_dir: Path, symbol: str) -> Path:
    dirs = sorted(base_dir.glob(f"*_walkforward_{symbol.upper()}"), key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"No walkforward dir for symbol={symbol} under {base_dir}")
    return dirs[-1].resolve()


def _max_consecutive_losses(pnl: np.ndarray) -> int:
    cur = 0
    best = 0
    for x in pnl:
        if np.isfinite(x) and x < 0.0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _max_drawdown(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return float("nan")
    cum = np.cumsum(np.nan_to_num(pnl, nan=0.0))
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.nanmin(dd)) if dd.size else float("nan")


def _tail_mean(arr: np.ndarray, q: float) -> float:
    x = arr[np.isfinite(arr)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(float(q) * x.size)))
    xs = np.sort(x)
    return float(np.mean(xs[:k]))


def _mode_metrics(df: pd.DataFrame, mode: str) -> Dict[str, float]:
    mode = str(mode).strip().lower()
    if mode == "baseline":
        fill_col = "baseline_filled"
        inv_stop_col = "baseline_invalid_stop_geometry"
        inv_tp_col = "baseline_invalid_tp_geometry"
        pnl_col = "baseline_pnl_net_pct" if "baseline_pnl_net_pct" in df.columns else "baseline_pnl_gross_pct"
        sl_col = "baseline_sl_hit"
        liq_col = "baseline_fill_liquidity_type"
        delay_col = "baseline_fill_delay_min"
        improve_bps = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    else:
        fill_col = "exec_filled"
        inv_stop_col = "exec_invalid_stop_geometry"
        inv_tp_col = "exec_invalid_tp_geometry"
        pnl_col = "exec_pnl_net_pct" if "exec_pnl_net_pct" in df.columns else "exec_pnl_gross_pct"
        sl_col = "exec_sl_hit"
        liq_col = "exec_fill_liquidity_type"
        delay_col = "exec_fill_delay_min"
        improve_bps = pd.to_numeric(df.get("entry_improvement_pct", np.nan), errors="coerce") * 10000.0

    filled = pd.to_numeric(df.get(fill_col, 0), errors="coerce").fillna(0).astype(int)
    inv_stop = pd.to_numeric(df.get(inv_stop_col, 0), errors="coerce").fillna(0).astype(int)
    inv_tp = pd.to_numeric(df.get(inv_tp_col, 0), errors="coerce").fillna(0).astype(int)
    valid = (inv_stop == 0) & (inv_tp == 0)
    mask = (filled == 1) & valid

    pnl_raw = pd.to_numeric(df.get(pnl_col, np.nan), errors="coerce")
    pnl_sig = np.zeros(len(df), dtype=float)
    idx = mask & pnl_raw.notna()
    pnl_sig[idx.to_numpy(dtype=bool)] = pnl_raw[idx].to_numpy(dtype=float)

    entries = int(mask.sum())
    n = int(len(df))

    sl = pd.to_numeric(df.get(sl_col, 0), errors="coerce").fillna(0).astype(int)
    sl_hit_rate_valid = float(((sl == 1) & mask).sum() / entries) if entries > 0 else float("nan")

    liq = df.get(liq_col, pd.Series([""] * len(df), index=df.index)).fillna("").astype(str).str.lower()
    taker_share = float(((liq == "taker") & mask).sum() / entries) if entries > 0 else float("nan")
    delay = pd.to_numeric(df.get(delay_col, np.nan), errors="coerce")
    median_fill_delay = float(delay[mask].median()) if entries > 0 and delay[mask].notna().any() else float("nan")
    med_improve_bps = float(improve_bps[mask].median()) if entries > 0 and improve_bps[mask].notna().any() else (0.0 if mode == "baseline" else float("nan"))

    pnl_sum = float(np.sum(pnl_sig))
    mean_exp = float(np.mean(pnl_sig)) if n > 0 else float("nan")
    std = float(np.std(pnl_sig, ddof=0)) if n > 0 else float("nan")
    worst_dec = _tail_mean(pnl_sig, q=0.10)
    cvar_5 = _tail_mean(pnl_sig, q=0.05)
    max_consec_losses = _max_consecutive_losses(pnl_sig)
    mdd = _max_drawdown(pnl_sig)

    return {
        "signals_total": n,
        "entries_valid": entries,
        "mean_expectancy_net": mean_exp,
        "pnl_net_sum": pnl_sum,
        "pnl_std": std,
        "worst_decile_mean": worst_dec,
        "cvar_5": cvar_5,
        "max_consecutive_losses": max_consec_losses,
        "SL_hit_rate_valid": sl_hit_rate_valid,
        "taker_share": taker_share,
        "median_fill_delay_min": median_fill_delay,
        "median_entry_improvement_bps": med_improve_bps,
        "max_drawdown": mdd,
    }


def _expectancy_not_worse(exec_exp: float, base_exp: float, rel_limit: float) -> bool:
    if not np.isfinite(exec_exp) or not np.isfinite(base_exp):
        return False
    rel = max(0.0, float(rel_limit))
    if base_exp < 0:
        # Allow at most rel worsening in magnitude.
        return bool(exec_exp >= base_exp * (1.0 + rel))
    return bool(exec_exp >= base_exp * (1.0 - rel))


def _weighted_avg(vals: pd.Series, w: pd.Series) -> float:
    x = pd.to_numeric(vals, errors="coerce")
    wt = pd.to_numeric(w, errors="coerce").fillna(0.0)
    m = x.notna() & wt.gt(0.0)
    if not bool(m.any()):
        return float("nan")
    return float((x[m] * wt[m]).sum() / wt[m].sum())


def run(args: argparse.Namespace) -> Path:
    base_dir = _resolve_path(args.base_dir)
    out_root = _resolve_path(args.outdir) / _utc_tag()
    out_root.mkdir(parents=True, exist_ok=True)

    symbols = [s.upper() for s in _parse_list(args.symbols)]
    run_dirs_arg = _parse_list(args.run_dirs)
    run_dirs: List[Path] = []
    if run_dirs_arg:
        run_dirs = [_resolve_path(p) for p in run_dirs_arg]
    else:
        run_dirs = [_latest_run_for_symbol(base_dir, s) for s in symbols]

    by_symbol_rows: List[Dict[str, Any]] = []
    used_rows: List[Dict[str, str]] = []

    for rd in run_dirs:
        test_files = sorted(rd.glob("*_walkforward_test_signals.csv"))
        if not test_files:
            continue
        test_fp = test_files[0]
        symbol = test_fp.name.split("_", 1)[0].upper()
        used_rows.append({"symbol": symbol, "run_dir": str(rd.resolve()), "test_signals_csv": str(test_fp.resolve())})

        df = pd.read_csv(test_fp)
        df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True, errors="coerce")
        df = df.sort_values("signal_time").reset_index(drop=True)

        b = _mode_metrics(df, "baseline")
        e = _mode_metrics(df, "exec")
        row = {"symbol": symbol}
        for k, v in b.items():
            row[f"baseline_{k}"] = v
        for k, v in e.items():
            row[f"exec_{k}"] = v
        row["delta_expectancy_exec_minus_baseline"] = float(e["mean_expectancy_net"] - b["mean_expectancy_net"]) if np.isfinite(e["mean_expectancy_net"]) and np.isfinite(b["mean_expectancy_net"]) else float("nan")
        row["delta_cvar5_exec_minus_baseline"] = float(e["cvar_5"] - b["cvar_5"]) if np.isfinite(e["cvar_5"]) and np.isfinite(b["cvar_5"]) else float("nan")
        row["delta_max_drawdown_exec_minus_baseline"] = float(e["max_drawdown"] - b["max_drawdown"]) if np.isfinite(e["max_drawdown"]) and np.isfinite(b["max_drawdown"]) else float("nan")
        by_symbol_rows.append(row)
    by_symbol = pd.DataFrame(by_symbol_rows).sort_values("symbol").reset_index(drop=True)
    if by_symbol.empty:
        raise SystemExit("No valid walkforward test_signals files found")

    sig_w = pd.to_numeric(by_symbol["baseline_signals_total"], errors="coerce").fillna(0.0)
    b_ent_w = pd.to_numeric(by_symbol["baseline_entries_valid"], errors="coerce").fillna(0.0)
    e_ent_w = pd.to_numeric(by_symbol["exec_entries_valid"], errors="coerce").fillna(0.0)
    baseline_pnl_sum = float(pd.to_numeric(by_symbol["baseline_pnl_net_sum"], errors="coerce").fillna(0.0).sum())
    exec_pnl_sum = float(pd.to_numeric(by_symbol["exec_pnl_net_sum"], errors="coerce").fillna(0.0).sum())
    signals_total = int(sig_w.sum())

    overall = pd.DataFrame(
        [
            {
                "scope": "overall",
                "symbols": int(by_symbol["symbol"].nunique()),
                "signals_total": int(signals_total),
                "baseline_mean_expectancy_net": _weighted_avg(by_symbol["baseline_mean_expectancy_net"], sig_w),
                "exec_mean_expectancy_net": _weighted_avg(by_symbol["exec_mean_expectancy_net"], sig_w),
                "baseline_pnl_net_sum": baseline_pnl_sum,
                "exec_pnl_net_sum": exec_pnl_sum,
                "baseline_pnl_std": _weighted_avg(by_symbol["baseline_pnl_std"], sig_w),
                "exec_pnl_std": _weighted_avg(by_symbol["exec_pnl_std"], sig_w),
                "baseline_worst_decile_mean": _weighted_avg(by_symbol["baseline_worst_decile_mean"], sig_w),
                "exec_worst_decile_mean": _weighted_avg(by_symbol["exec_worst_decile_mean"], sig_w),
                "baseline_cvar_5": _weighted_avg(by_symbol["baseline_cvar_5"], sig_w),
                "exec_cvar_5": _weighted_avg(by_symbol["exec_cvar_5"], sig_w),
                "baseline_max_consecutive_losses": int(pd.to_numeric(by_symbol["baseline_max_consecutive_losses"], errors="coerce").fillna(0).max()),
                "exec_max_consecutive_losses": int(pd.to_numeric(by_symbol["exec_max_consecutive_losses"], errors="coerce").fillna(0).max()),
                "baseline_SL_hit_rate_valid": _weighted_avg(by_symbol["baseline_SL_hit_rate_valid"], b_ent_w),
                "exec_SL_hit_rate_valid": _weighted_avg(by_symbol["exec_SL_hit_rate_valid"], e_ent_w),
                "baseline_taker_share": _weighted_avg(by_symbol["baseline_taker_share"], b_ent_w),
                "exec_taker_share": _weighted_avg(by_symbol["exec_taker_share"], e_ent_w),
                "baseline_median_fill_delay_min": _weighted_avg(by_symbol["baseline_median_fill_delay_min"], b_ent_w),
                "exec_median_fill_delay_min": _weighted_avg(by_symbol["exec_median_fill_delay_min"], e_ent_w),
                "exec_median_entry_improvement_bps": _weighted_avg(by_symbol["exec_median_entry_improvement_bps"], e_ent_w),
                "baseline_max_drawdown": _weighted_avg(by_symbol["baseline_max_drawdown"], sig_w),
                "exec_max_drawdown": _weighted_avg(by_symbol["exec_max_drawdown"], sig_w),
            }
        ]
    )
    overall.loc[0, "delta_expectancy_exec_minus_baseline"] = float(overall.loc[0, "exec_mean_expectancy_net"] - overall.loc[0, "baseline_mean_expectancy_net"])
    overall.loc[0, "delta_cvar5_exec_minus_baseline"] = float(overall.loc[0, "exec_cvar_5"] - overall.loc[0, "baseline_cvar_5"])
    overall.loc[0, "delta_max_drawdown_exec_minus_baseline"] = float(overall.loc[0, "exec_max_drawdown"] - overall.loc[0, "baseline_max_drawdown"])

    rel_worsen_limit = float(args.max_expectancy_worsen_rel)
    exec_exp = float(overall.loc[0, "exec_mean_expectancy_net"])
    base_exp = float(overall.loc[0, "baseline_mean_expectancy_net"])
    cvar_improves = bool(np.isfinite(overall.loc[0, "delta_cvar5_exec_minus_baseline"]) and overall.loc[0, "delta_cvar5_exec_minus_baseline"] > 0.0)
    mdd_improves = bool(np.isfinite(overall.loc[0, "delta_max_drawdown_exec_minus_baseline"]) and overall.loc[0, "delta_max_drawdown_exec_minus_baseline"] > 0.0)
    exp_not_too_worse = _expectancy_not_worse(exec_exp, base_exp, rel_limit=rel_worsen_limit)
    deploy_safety = bool(cvar_improves and mdd_improves and exp_not_too_worse)

    by_symbol_fp = out_root / "risk_rollup_by_symbol.csv"
    overall_fp = out_root / "risk_rollup_overall.csv"
    used_fp = out_root / "risk_rollup_inputs.csv"
    md_fp = out_root / "risk_rollup.md"
    phase_fp = out_root / "phase_result.md"

    by_symbol.to_csv(by_symbol_fp, index=False)
    overall.to_csv(overall_fp, index=False)
    pd.DataFrame(used_rows).to_csv(used_fp, index=False)

    lines: List[str] = []
    lines.append("# Risk Rollup: Baseline vs Exec (TEST-only)")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Max relative expectancy worsening tolerance: {rel_worsen_limit:.4f}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for r in used_rows:
        lines.append(f"- {r['symbol']}: `{r['test_signals_csv']}`")
    lines.append("")
    lines.append("## By Symbol (Key Deltas)")
    lines.append("")
    for _, r in by_symbol.iterrows():
        lines.append(
            f"- {r['symbol']}: d_expectancy={float(r['delta_expectancy_exec_minus_baseline']):.6f}, "
            f"d_cvar5={float(r['delta_cvar5_exec_minus_baseline']):.6f}, d_maxDD={float(r['delta_max_drawdown_exec_minus_baseline']):.6f}, "
            f"exec_taker_share={float(r['exec_taker_share']):.6f}, exec_median_delay={float(r['exec_median_fill_delay_min']):.2f}"
        )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    lines.append("- Overall rows are weighted by signal count (and by valid entries for entry-conditioned metrics).")
    ov = overall.iloc[0]
    lines.append(f"- baseline_mean_expectancy_net: {float(ov['baseline_mean_expectancy_net']):.6f}")
    lines.append(f"- exec_mean_expectancy_net: {float(ov['exec_mean_expectancy_net']):.6f}")
    lines.append(f"- delta_expectancy_exec_minus_baseline: {float(ov['delta_expectancy_exec_minus_baseline']):.6f}")
    lines.append(f"- baseline_cvar_5: {float(ov['baseline_cvar_5']):.6f}")
    lines.append(f"- exec_cvar_5: {float(ov['exec_cvar_5']):.6f}")
    lines.append(f"- baseline_max_drawdown: {float(ov['baseline_max_drawdown']):.6f}")
    lines.append(f"- exec_max_drawdown: {float(ov['exec_max_drawdown']):.6f}")
    lines.append(f"- exec_taker_share: {float(ov['exec_taker_share']):.6f}")
    lines.append(f"- exec_median_fill_delay_min: {float(ov['exec_median_fill_delay_min']):.2f}")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        f"- Deploy-worth safety rule: CVaR improves AND max DD improves AND expectancy does not worsen beyond {rel_worsen_limit:.2%}."
    )
    lines.append(f"- CVaR improves: {int(cvar_improves)}")
    lines.append(f"- Max DD improves: {int(mdd_improves)}")
    lines.append(f"- Expectancy not too worse: {int(exp_not_too_worse)}")
    lines.append(f"- Safety-layer deploy-worthy: {'YES' if deploy_safety else 'NO'}")
    lines.append("")
    lines.append(f"- CSV by symbol: `{by_symbol_fp}`")
    lines.append(f"- CSV overall: `{overall_fp}`")
    lines.append(f"- Input map: `{used_fp}`")
    md_fp.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    phase_lines = [
        "Phase: A (risk/tail rollup on TEST-only)",
        f"Inputs: {len(used_rows)} walkforward test CSVs",
        f"Symbols: {','.join(by_symbol['symbol'].tolist())}",
        f"Overall baseline expectancy: {float(ov['baseline_mean_expectancy_net']):.6f}",
        f"Overall exec expectancy: {float(ov['exec_mean_expectancy_net']):.6f}",
        f"CVaR improves: {int(cvar_improves)}",
        f"Max DD improves: {int(mdd_improves)}",
        f"Expectancy not too worse (limit={rel_worsen_limit:.2%}): {int(exp_not_too_worse)}",
        f"Safety-layer deploy-worthy: {'YES' if deploy_safety else 'NO'}",
        f"Artifacts: {by_symbol_fp.name}, {overall_fp.name}, {md_fp.name}",
    ]
    phase_fp.write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")

    print(str(out_root))
    print(str(by_symbol_fp))
    print(str(overall_fp))
    print(str(md_fp))
    print(str(phase_fp))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Risk/tail rollup for walkforward baseline vs exec_limit test-only outputs.")
    ap.add_argument("--base-dir", default="reports/execution_layer")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--symbols", default="SOLUSDT,AVAXUSDT,NEARUSDT")
    ap.add_argument("--run-dirs", default="", help="Optional comma-separated walkforward run dirs. If empty, use latest per symbol.")
    ap.add_argument("--max-expectancy-worsen-rel", type=float, default=0.10)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
