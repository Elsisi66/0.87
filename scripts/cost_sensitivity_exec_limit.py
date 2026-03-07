#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import execution_layer_3m_ict as exec3m  # noqa: E402


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _parse_csv_tokens(raw: str, cast) -> List[Any]:
    out: List[Any] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(cast(token))
    return out


def _iter_cost_scenarios(
    maker_fee_bps: Iterable[float],
    taker_fee_bps: Iterable[float],
    limit_slip_bps: Iterable[float],
    market_slip_bps: Iterable[float],
) -> Iterable[Tuple[float, float, float, float]]:
    return itertools.product(maker_fee_bps, taker_fee_bps, limit_slip_bps, market_slip_bps)


def _latest_walkforward_signals(base_dir: Path, symbol: str) -> Path:
    glob_pat = f"*_walkforward_{symbol.upper()}/{symbol.upper()}_walkforward_test_signals.csv"
    files = sorted(base_dir.glob(glob_pat), key=lambda p: p.parent.name)
    if not files:
        raise FileNotFoundError(f"No walkforward test-signals file found for {symbol} under {base_dir}")
    return files[-1].resolve()


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _liq_type_from_row(row: pd.Series, prefix: str, fallback_type_col: str = "") -> str:
    liq_col = f"{prefix}_fill_liquidity_type"
    if liq_col in row and isinstance(row.get(liq_col, None), str):
        v = str(row.get(liq_col)).strip().lower()
        if v in {"maker", "taker"}:
            return v
    if fallback_type_col:
        et = str(row.get(fallback_type_col, "")).strip().lower()
        if et.startswith("limit"):
            return "maker"
    return "taker"


def _metrics_for_scenario(
    df: pd.DataFrame,
    maker_fee_bps: float,
    taker_fee_bps: float,
    limit_slip_bps: float,
    market_slip_bps: float,
) -> Dict[str, float]:
    b_filled = _num(df, "baseline_filled", 0.0).fillna(0).astype(int)
    e_filled = _num(df, "exec_filled", 0.0).fillna(0).astype(int)

    b_valid = (_num(df, "baseline_invalid_stop_geometry", 0.0).fillna(0).astype(int) == 0) & (
        _num(df, "baseline_invalid_tp_geometry", 0.0).fillna(0).astype(int) == 0
    )
    e_valid = (_num(df, "exec_invalid_stop_geometry", 0.0).fillna(0).astype(int) == 0) & (
        _num(df, "exec_invalid_tp_geometry", 0.0).fillna(0).astype(int) == 0
    )

    b_mask = (b_filled == 1) & b_valid
    e_mask = (e_filled == 1) & e_valid

    baseline_net: List[float] = []
    exec_net: List[float] = []
    exec_sl: List[int] = []
    exec_taker_count = 0

    for idx, row in df.iterrows():
        if bool(b_mask.loc[idx]):
            b_liq = _liq_type_from_row(row, "baseline")
            b_cost = exec3m._costed_pnl_long(
                entry_price=row.get("baseline_entry_price"),
                exit_price=row.get("baseline_exit_price"),
                entry_liquidity_type=b_liq,
                fee_bps_maker=float(maker_fee_bps),
                fee_bps_taker=float(taker_fee_bps),
                slippage_bps_limit=float(limit_slip_bps),
                slippage_bps_market=float(market_slip_bps),
            )
            if np.isfinite(b_cost["pnl_net_pct"]):
                baseline_net.append(float(b_cost["pnl_net_pct"]))

        if bool(e_mask.loc[idx]):
            e_liq = _liq_type_from_row(row, "exec", fallback_type_col="exec_entry_type")
            e_cost = exec3m._costed_pnl_long(
                entry_price=row.get("exec_entry_price"),
                exit_price=row.get("exec_exit_price"),
                entry_liquidity_type=e_liq,
                fee_bps_maker=float(maker_fee_bps),
                fee_bps_taker=float(taker_fee_bps),
                slippage_bps_limit=float(limit_slip_bps),
                slippage_bps_market=float(market_slip_bps),
            )
            if np.isfinite(e_cost["pnl_net_pct"]):
                exec_net.append(float(e_cost["pnl_net_pct"]))
            exec_sl.append(int(bool(row.get("exec_sl_hit", 0))))
            if e_liq == "taker":
                exec_taker_count += 1

    b_entries = int(b_mask.sum())
    e_entries = int(e_mask.sum())
    b_sum = float(np.nansum(np.asarray(baseline_net, dtype=float))) if baseline_net else float("nan")
    e_sum = float(np.nansum(np.asarray(exec_net, dtype=float))) if exec_net else float("nan")
    b_exp = _safe_ratio(b_sum, b_entries)
    e_exp = _safe_ratio(e_sum, e_entries)
    d_exp = float(e_exp - b_exp) if np.isfinite(e_exp) and np.isfinite(b_exp) else float("nan")
    beats = int(np.isfinite(d_exp) and d_exp > 0.0)
    sl_hit_rate = _safe_ratio(float(np.nansum(exec_sl)), float(len(exec_sl))) if exec_sl else float("nan")
    taker_share = _safe_ratio(float(exec_taker_count), float(e_entries))

    return {
        "signals_total": int(len(df)),
        "baseline_entries": b_entries,
        "exec_entries": e_entries,
        "entry_rate": _safe_ratio(float(e_entries), float(len(df))),
        "taker_share": taker_share,
        "sl_hit_rate_exec": sl_hit_rate,
        "baseline_net_pnl_sum": b_sum,
        "exec_net_pnl_sum": e_sum,
        "baseline_net_expectancy": b_exp,
        "exec_net_expectancy": e_exp,
        "expectancy_delta_exec_minus_baseline": d_exp,
        "exec_beats_baseline": beats,
    }


def _to_md_table(df: pd.DataFrame) -> List[str]:
    if df.empty:
        return ["(none)"]
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def run(args: argparse.Namespace) -> Path:
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided")

    base_dir = _resolve_path(args.base_dir)
    out_root = _resolve_path(args.outdir) / _utc_tag()
    out_root.mkdir(parents=True, exist_ok=True)

    maker_fee_vals = _parse_csv_tokens(args.maker_fee_bps, float)
    taker_fee_vals = _parse_csv_tokens(args.taker_fee_bps, float)
    limit_slip_vals = _parse_csv_tokens(args.limit_slip_bps, float)
    market_slip_vals = _parse_csv_tokens(args.market_slip_bps, float)
    scenarios = list(_iter_cost_scenarios(maker_fee_vals, taker_fee_vals, limit_slip_vals, market_slip_vals))

    rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, str]] = []

    for symbol in symbols:
        fp = _latest_walkforward_signals(base_dir=base_dir, symbol=symbol)
        source_rows.append({"symbol": symbol, "test_signals_file": str(fp)})
        df = pd.read_csv(fp)

        for maker_fee, taker_fee, limit_slip, market_slip in scenarios:
            m = _metrics_for_scenario(
                df=df,
                maker_fee_bps=float(maker_fee),
                taker_fee_bps=float(taker_fee),
                limit_slip_bps=float(limit_slip),
                market_slip_bps=float(market_slip),
            )
            rows.append(
                {
                    "symbol": symbol,
                    "maker_fee_bps": float(maker_fee),
                    "taker_fee_bps": float(taker_fee),
                    "limit_slippage_bps": float(limit_slip),
                    "market_slippage_bps": float(market_slip),
                    **m,
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise SystemExit("No cost sensitivity rows produced")

    by_symbol = (
        summary.groupby("symbol", as_index=False)
        .agg(
            scenarios=("exec_beats_baseline", "size"),
            scenarios_exec_beats_baseline=("exec_beats_baseline", "sum"),
            pass_ratio=("exec_beats_baseline", "mean"),
            avg_exec_expectancy=("exec_net_expectancy", "mean"),
            avg_baseline_expectancy=("baseline_net_expectancy", "mean"),
            avg_entry_rate=("entry_rate", "mean"),
            avg_taker_share=("taker_share", "mean"),
            avg_sl_hit_rate_exec=("sl_hit_rate_exec", "mean"),
        )
        .sort_values(["pass_ratio", "symbol"], ascending=[False, True])
        .reset_index(drop=True)
    )

    overall_scenarios = int(summary.shape[0])
    overall_beats = int(summary["exec_beats_baseline"].sum())
    overall_pass_ratio = float(overall_beats / overall_scenarios) if overall_scenarios > 0 else float("nan")
    gate_pass = bool(np.isfinite(overall_pass_ratio) and overall_pass_ratio >= float(args.pass_threshold))

    out_csv = out_root / "cost_sensitivity_summary.csv"
    out_md = out_root / "cost_sensitivity_report.md"
    out_sources = out_root / "cost_sensitivity_sources.csv"
    out_phase = out_root / "phase_result.md"

    summary.to_csv(out_csv, index=False)
    pd.DataFrame(source_rows).to_csv(out_sources, index=False)

    lines: List[str] = []
    lines.append("# Cost Sensitivity Report (exec_limit)")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Symbols: {', '.join(symbols)}")
    lines.append(f"- Base dir: `{base_dir}`")
    lines.append(f"- Scenarios per symbol: {len(scenarios)}")
    lines.append(f"- Total scenarios: {overall_scenarios}")
    lines.append(f"- Pass threshold (exec beats baseline expectancy): {float(args.pass_threshold):.2f}")
    lines.append(f"- Overall pass ratio: {overall_pass_ratio:.6f}")
    lines.append(f"- Gate result: {'PASS' if gate_pass else 'FAIL'}")
    lines.append("")
    lines.append("## Per-Symbol Gate Summary")
    lines.append("")
    lines.extend(_to_md_table(by_symbol))
    lines.append("")
    lines.append("## Source Files")
    lines.append("")
    for src in source_rows:
        lines.append(f"- `{src['symbol']}`: `{src['test_signals_file']}`")
    lines.append("")
    lines.append(f"- Detailed CSV: `{out_csv}`")
    lines.append(f"- Source list CSV: `{out_sources}`")
    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    phase_lines = [
        "Phase: 1 (cost model realism check)",
        f"Inputs: symbols={','.join(symbols)}; source walkforward test_signals under {base_dir}",
        f"Scenarios: maker={maker_fee_vals}, taker={taker_fee_vals}, limit_slip={limit_slip_vals}, market_slip={market_slip_vals}",
        f"Generated rows: {overall_scenarios}",
        f"Metric: exec beats baseline on net expectancy in {overall_beats}/{overall_scenarios} scenarios",
        f"Overall pass ratio: {overall_pass_ratio:.6f}",
        f"Gate threshold: {float(args.pass_threshold):.2f}",
        f"Gate result: {'PASS' if gate_pass else 'FAIL'}",
        f"Artifacts: {out_csv.name}, {out_md.name}, {out_sources.name}",
        "Next: per-symbol execution config overrides + walk-forward rerun (Phase 2)",
    ]
    out_phase.write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")

    print(str(out_root))
    print(str(out_csv))
    print(str(out_md))
    print(str(out_phase))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Cost-sensitivity sweep for exec_limit on walk-forward TEST-only slices.")
    ap.add_argument("--base-dir", default="reports/execution_layer")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--symbols", default="SOLUSDT,AVAXUSDT,NEARUSDT")
    ap.add_argument("--maker-fee-bps", default="0,2,5")
    ap.add_argument("--taker-fee-bps", default="2,5,8")
    ap.add_argument("--limit-slip-bps", default="0,1,3")
    ap.add_argument("--market-slip-bps", default="1,3,6")
    ap.add_argument("--pass-threshold", type=float, default=0.70)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
