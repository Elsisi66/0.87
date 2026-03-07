#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import execution_layer_3m_ict as exec3m  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _latest_walkforward_dir(base_dir: Path, symbol: str) -> Path:
    dirs = sorted(base_dir.glob(f"*_walkforward_{symbol.upper()}"), key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"No walkforward dir for {symbol} under {base_dir}")
    return dirs[-1].resolve()


def _load_run_inputs(wf_dir: Path, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    test_fp = wf_dir / f"{symbol}_walkforward_test_signals.csv"
    summary_fp = wf_dir / f"{symbol}_walkforward_test_summary.csv"
    meta_fp = wf_dir / "run_meta.json"
    if not test_fp.exists():
        raise FileNotFoundError(f"Missing file: {test_fp}")
    if not summary_fp.exists():
        raise FileNotFoundError(f"Missing file: {summary_fp}")
    test_df = pd.read_csv(test_fp)
    summary_df = pd.read_csv(summary_fp)
    summary_row = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    meta = json.loads(meta_fp.read_text(encoding="utf-8")) if meta_fp.exists() else {}
    return test_df, summary_row, meta


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


def _bucket_table(
    df: pd.DataFrame,
    bucket_col: str,
    pnl_col: str,
    sl_col: str,
    tp_col: str,
    total_n: int,
) -> pd.DataFrame:
    g = (
        df.groupby(bucket_col, dropna=False)
        .agg(
            trades=(bucket_col, "size"),
            expectancy_net=(pnl_col, "mean"),
            pnl_net_sum=(pnl_col, "sum"),
            sl_hit_rate=(sl_col, "mean"),
            tp_hit_rate=(tp_col, "mean"),
        )
        .reset_index()
    )
    g["share"] = g["trades"] / max(1, int(total_n))
    return g.sort_values("expectancy_net", ascending=True).reset_index(drop=True)


def _as_str_bucket(x: Any) -> str:
    if pd.isna(x):
        return "nan"
    return str(x)


def _quantile_bucket(series: pd.Series, q: int = 4) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return pd.Series(["nan"] * len(s), index=s.index)
    bins = min(int(q), int(valid.nunique()))
    if bins <= 1:
        return pd.Series(["all"] * len(s), index=s.index)
    out = pd.qcut(s, q=bins, duplicates="drop")
    return out.astype(str).fillna("nan")


def _gate_recommendations(entries: pd.DataFrame, overall_exp: float) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "recommend_vol_regime_gate": 0,
        "vol_regime_max_percentile": float("nan"),
        "recommend_trend_gate": 0,
        "trend_fast_col": "EMA_50",
        "trend_slow_col": "EMA_120",
        "trend_min_slope": 0.0,
        "recommend_stop_distance_min": 0,
        "stop_distance_min_pct": float("nan"),
        "notes": "",
    }
    notes: List[str] = []
    min_n = max(20, int(round(len(entries) * 0.08)))

    # Vol regime heuristic.
    vol_hi = entries[entries["vol_regime_bucket"] == "p90_100"]
    if len(vol_hi) >= min_n and np.isfinite(overall_exp):
        vol_hi_exp = float(pd.to_numeric(vol_hi["baseline_pnl_net_pct"], errors="coerce").mean())
        if np.isfinite(vol_hi_exp) and vol_hi_exp < overall_exp - 0.00015:
            rec["recommend_vol_regime_gate"] = 1
            rec["vol_regime_max_percentile"] = 90.0
            notes.append("High ATR percentile bucket (p90_100) materially underperforms.")

    # Trend heuristic.
    trend_up = entries[entries["trend_bucket"] == "trend_up"]
    trend_down = entries[entries["trend_bucket"] == "trend_down_or_flat"]
    if len(trend_up) >= min_n and len(trend_down) >= min_n:
        up_exp = float(pd.to_numeric(trend_up["baseline_pnl_net_pct"], errors="coerce").mean())
        dn_exp = float(pd.to_numeric(trend_down["baseline_pnl_net_pct"], errors="coerce").mean())
        if np.isfinite(up_exp) and np.isfinite(dn_exp) and dn_exp < up_exp - 0.00015:
            rec["recommend_trend_gate"] = 1
            notes.append("Trend-down bucket underperforms trend-up bucket.")

    # Stop-distance heuristic.
    stop_q1 = float(pd.to_numeric(entries["stop_distance_pct"], errors="coerce").quantile(0.25))
    small_stop = entries[pd.to_numeric(entries["stop_distance_pct"], errors="coerce") <= stop_q1]
    if len(small_stop) >= min_n and np.isfinite(overall_exp):
        small_exp = float(pd.to_numeric(small_stop["baseline_pnl_net_pct"], errors="coerce").mean())
        if np.isfinite(small_exp) and small_exp < overall_exp - 0.00015:
            rec["recommend_stop_distance_min"] = 1
            rec["stop_distance_min_pct"] = float(stop_q1)
            notes.append("Small stop-distance quartile underperforms.")

    rec["notes"] = " | ".join(notes)
    return rec


def run(args: argparse.Namespace) -> Path:
    base_dir = _resolve_path(args.base_dir)
    out_root = _resolve_path(args.outdir) / _utc_tag()
    out_root.mkdir(parents=True, exist_ok=True)

    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided")

    rec_rows: List[Dict[str, Any]] = []

    for symbol in symbols:
        wf_dir = _latest_walkforward_dir(base_dir=base_dir, symbol=symbol)
        test_df, summary_row, run_meta = _load_run_inputs(wf_dir=wf_dir, symbol=symbol)
        params_file = Path(str(summary_row.get("params_file", "") or run_meta.get("params_file", ""))).resolve()
        if not params_file.exists():
            raise FileNotFoundError(f"Missing params file for {symbol}: {params_file}")

        signals_total = int(float(summary_row.get("signals_total", run_meta.get("signals_total", len(test_df)))))
        test_n = int(float(summary_row.get("test_signals", run_meta.get("test_signals", len(test_df)))))
        gate_cfg = dict(run_meta.get("gate_config", {})) if isinstance(run_meta.get("gate_config", {}), dict) else {}

        payload = json.loads(params_file.read_text(encoding="utf-8"))
        p = ga_long._norm_params(exec3m._unwrap_params(payload))
        df_1h = exec3m._load_symbol_df(symbol, tf="1h")
        sigs = exec3m._build_1h_signals(
            df_1h=df_1h,
            p=p,
            max_signals=signals_total,
            order="latest",
            gate_cfg=gate_cfg,
        )
        if len(sigs) < test_n:
            test_n = len(sigs)
        sig_test = sigs[-test_n:] if test_n > 0 else sigs
        sig_map = {
            s.signal_id: {
                "signal_time": str(s.signal_time),
                "stop_distance_pct": float(s.stop_distance_pct),
                "atr_1h": float(s.atr_1h),
                "atr_percentile_1h": float(s.atr_percentile_1h),
                "trend_up_1h": int(s.trend_up_1h),
            }
            for s in sig_test
        }
        feat_df = pd.DataFrame(
            [{"signal_id": k, **v} for k, v in sig_map.items()]
        )
        merged = test_df.merge(feat_df, on="signal_id", how="left")

        b_filled = pd.to_numeric(merged.get("baseline_filled", 0), errors="coerce").fillna(0).astype(int)
        b_valid = (
            pd.to_numeric(merged.get("baseline_invalid_stop_geometry", 0), errors="coerce").fillna(0).astype(int) == 0
        ) & (
            pd.to_numeric(merged.get("baseline_invalid_tp_geometry", 0), errors="coerce").fillna(0).astype(int) == 0
        )
        entries = merged[(b_filled == 1) & b_valid].copy()
        pnl_col = "baseline_pnl_net_pct" if "baseline_pnl_net_pct" in entries.columns else "baseline_pnl_gross_pct"
        entries[pnl_col] = pd.to_numeric(entries[pnl_col], errors="coerce")
        entries["baseline_pnl_net_pct"] = entries[pnl_col]
        entries["baseline_sl_hit"] = pd.to_numeric(entries.get("baseline_sl_hit", 0), errors="coerce").fillna(0).astype(float)
        entries["baseline_tp_hit"] = pd.to_numeric(entries.get("baseline_tp_hit", 0), errors="coerce").fillna(0).astype(float)
        entries["stop_distance_pct"] = pd.to_numeric(entries.get("stop_distance_pct", np.nan), errors="coerce")
        entries["atr_1h"] = pd.to_numeric(entries.get("atr_1h", np.nan), errors="coerce")
        entries["atr_percentile_1h"] = pd.to_numeric(entries.get("atr_percentile_1h", np.nan), errors="coerce")
        entries["trend_up_1h"] = pd.to_numeric(entries.get("trend_up_1h", 0), errors="coerce").fillna(0).astype(int)

        entries["stop_distance_bucket"] = _quantile_bucket(entries["stop_distance_pct"], q=4)
        entries["atr_bucket"] = _quantile_bucket(entries["atr_1h"], q=4)
        entries["trend_bucket"] = entries["trend_up_1h"].map({1: "trend_up", 0: "trend_down_or_flat"}).fillna("trend_down_or_flat")
        bins = [-np.inf, 50.0, 80.0, 90.0, np.inf]
        labels = ["p00_50", "p50_80", "p80_90", "p90_100"]
        entries["vol_regime_bucket"] = pd.cut(entries["atr_percentile_1h"], bins=bins, labels=labels).astype(str).fillna("nan")

        total_n = int(len(entries))
        overall_exp = float(entries["baseline_pnl_net_pct"].mean()) if total_n > 0 else float("nan")
        overall_sl = float(entries["baseline_sl_hit"].mean()) if total_n > 0 else float("nan")

        stop_tbl = _bucket_table(entries, "stop_distance_bucket", "baseline_pnl_net_pct", "baseline_sl_hit", "baseline_tp_hit", total_n)
        atr_tbl = _bucket_table(entries, "atr_bucket", "baseline_pnl_net_pct", "baseline_sl_hit", "baseline_tp_hit", total_n)
        trend_tbl = _bucket_table(entries, "trend_bucket", "baseline_pnl_net_pct", "baseline_sl_hit", "baseline_tp_hit", total_n)
        vol_tbl = _bucket_table(entries, "vol_regime_bucket", "baseline_pnl_net_pct", "baseline_sl_hit", "baseline_tp_hit", total_n)

        stop_csv = out_root / f"diagnose_1h_bleed_{symbol}_stop_distance.csv"
        atr_csv = out_root / f"diagnose_1h_bleed_{symbol}_atr_quantile.csv"
        trend_csv = out_root / f"diagnose_1h_bleed_{symbol}_trend.csv"
        vol_csv = out_root / f"diagnose_1h_bleed_{symbol}_vol_regime.csv"
        stop_tbl.to_csv(stop_csv, index=False)
        atr_tbl.to_csv(atr_csv, index=False)
        trend_tbl.to_csv(trend_csv, index=False)
        vol_tbl.to_csv(vol_csv, index=False)

        rec = _gate_recommendations(entries=entries, overall_exp=overall_exp)
        rec["symbol"] = symbol
        rec["wf_dir"] = str(wf_dir)
        rec_rows.append(rec)

        md_fp = out_root / f"diagnose_1h_bleed_{symbol}.md"
        lines: List[str] = []
        lines.append(f"# Diagnose 1h Bleed: {symbol}")
        lines.append("")
        lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
        lines.append(f"- Source walkforward dir: `{wf_dir}`")
        lines.append(f"- Baseline valid test entries: {total_n}")
        lines.append(f"- Baseline net expectancy: {overall_exp:.6f}" if np.isfinite(overall_exp) else "- Baseline net expectancy: n/a")
        lines.append(f"- Baseline SL-hit rate: {overall_sl:.6f}" if np.isfinite(overall_sl) else "- Baseline SL-hit rate: n/a")
        lines.append("")
        lines.append("## Stop Distance Quantiles")
        lines.append("")
        lines.extend(_to_md_table(stop_tbl))
        lines.append("")
        lines.append("## ATR(1h) Quantiles")
        lines.append("")
        lines.extend(_to_md_table(atr_tbl))
        lines.append("")
        lines.append("## Trend Proxy")
        lines.append("")
        lines.extend(_to_md_table(trend_tbl))
        lines.append("")
        lines.append("## Vol Regime (ATR Percentile)")
        lines.append("")
        lines.extend(_to_md_table(vol_tbl))
        lines.append("")
        lines.append("## Suggested Gates")
        lines.append("")
        lines.append(f"- recommend_vol_regime_gate: {int(rec['recommend_vol_regime_gate'])}")
        lines.append(f"- vol_regime_max_percentile: {rec['vol_regime_max_percentile']}")
        lines.append(f"- recommend_trend_gate: {int(rec['recommend_trend_gate'])}")
        lines.append(f"- trend_fast_col/trend_slow_col: {rec['trend_fast_col']}/{rec['trend_slow_col']}")
        lines.append(f"- recommend_stop_distance_min: {int(rec['recommend_stop_distance_min'])}")
        lines.append(f"- stop_distance_min_pct: {rec['stop_distance_min_pct']}")
        lines.append(f"- notes: {rec['notes']}")
        lines.append("")
        lines.append("## CSV Outputs")
        lines.append("")
        lines.append(f"- `{stop_csv}`")
        lines.append(f"- `{atr_csv}`")
        lines.append(f"- `{trend_csv}`")
        lines.append(f"- `{vol_csv}`")
        md_fp.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    rec_df = pd.DataFrame(rec_rows)
    rec_csv = out_root / "diagnose_1h_bleed_recommendations.csv"
    rec_df.to_csv(rec_csv, index=False)

    phase_fp = out_root / "phase_result.md"
    vol_recs = int(rec_df["recommend_vol_regime_gate"].sum()) if not rec_df.empty else 0
    trend_recs = int(rec_df["recommend_trend_gate"].sum()) if not rec_df.empty else 0
    stop_recs = int(rec_df["recommend_stop_distance_min"].sum()) if not rec_df.empty else 0
    phase_lines = [
        "Phase: 3 (diagnose 1h bleed on baseline test-only entries)",
        f"Inputs: symbols={','.join(symbols)}; latest walkforward test_signals per symbol",
        f"Output dir: {out_root}",
        "Buckets: stop_distance quantiles, ATR_1h quantiles, trend proxy, ATR percentile regime",
        f"Recommendations: vol_regime={vol_recs}, trend={trend_recs}, stop_distance_min={stop_recs}",
        f"Recommendation table: {rec_csv.name}",
        "Gate selection policy: pick only 1-2 gates with consistent underperformance evidence",
        "Pass/Fail: informational phase (diagnostic)",
        "Next: enable selected 1h gates (off by default) and rerun walkforward (Phase 4)",
        "Constraint: no 3m alpha logic added; execution layer remains execution-only",
    ]
    phase_fp.write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")

    print(str(out_root))
    print(str(rec_csv))
    print(str(phase_fp))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Diagnose baseline 1h bleed by risk buckets on test-only walkforward outputs.")
    ap.add_argument("--base-dir", default="reports/execution_layer")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--symbols", default="SOLUSDT,AVAXUSDT,NEARUSDT")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
