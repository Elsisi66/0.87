#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import walkforward_exec_limit as wf_exec  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _tail_mean(arr: np.ndarray, q: float) -> float:
    x = arr[np.isfinite(arr)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(q) * x.size)))
    xs = np.sort(x)
    return float(np.mean(xs[:k]))


def _max_drawdown(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return float("nan")
    cum = np.cumsum(np.nan_to_num(pnl, nan=0.0))
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.nanmin(dd)) if dd.size else float("nan")


def _load_json_or_yaml_like(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    # Files we parse here are JSON-written-as-YAML in this repo; fallback to empty if malformed.
    return {}


def _latest_tight_dir(base_dir: Path) -> Path:
    cands = sorted([p for p in base_dir.iterdir() if p.is_dir() and (p / "AGG_exec_testonly_summary_tight.csv").exists()], key=lambda p: p.name)
    if not cands:
        raise FileNotFoundError(f"No tight aggregate directories with AGG_exec_testonly_summary_tight.csv under {base_dir}")
    return cands[-1].resolve()


def _latest_ga_dir(base_dir: Path) -> Path:
    cands = sorted([p for p in base_dir.glob("GA_EXEC_OPT_*") if p.is_dir() and (p / "risk_rollup_overall.csv").exists()], key=lambda p: p.name)
    if not cands:
        raise FileNotFoundError(f"No GA_EXEC_OPT_* directories under {base_dir}")
    return cands[-1].resolve()


def _find_test_signals_csv(summary_csv: Path) -> Optional[Path]:
    if not summary_csv.exists():
        return None
    try:
        s = pd.read_csv(summary_csv)
    except Exception:
        return None
    if s.empty:
        return None
    if "test_signals_csv" in s.columns:
        raw = str(s.iloc[0].get("test_signals_csv", "")).strip()
        if raw:
            p = _resolve_path(raw)
            if p.exists():
                return p
    cands = sorted(summary_csv.parent.glob("*_walkforward_test_signals.csv"))
    return cands[0].resolve() if cands else None


def _baseline_metrics_from_signals_df(df_in: pd.DataFrame) -> Dict[str, Any]:
    df = df_in.copy()
    if "signal_time" in df.columns:
        df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True, errors="coerce")
        df = df[df["signal_time"].notna()].copy()
        df = df.sort_values("signal_time").reset_index(drop=True)
    n = int(len(df))

    if n == 0:
        return {
            "signals_total": 0,
            "trades_taken": 0,
            "missing_slice_count": 0,
            "missing_slice_rate": float("nan"),
            "date_start_utc": "",
            "date_end_utc": "",
            "pnl_net_sum": float("nan"),
            "expectancy_net_per_signal": float("nan"),
            "expectancy_net_per_trade": float("nan"),
            "cvar_5_per_signal": float("nan"),
            "max_drawdown_per_signal": float("nan"),
            "sl_hit_rate_valid": float("nan"),
        }

    filled = pd.to_numeric(df.get("baseline_filled", 0), errors="coerce").fillna(0).astype(int)
    inv_stop = pd.to_numeric(df.get("baseline_invalid_stop_geometry", 0), errors="coerce").fillna(0).astype(int)
    inv_tp = pd.to_numeric(df.get("baseline_invalid_tp_geometry", 0), errors="coerce").fillna(0).astype(int)
    valid = (inv_stop == 0) & (inv_tp == 0)
    mask = (filled == 1) & valid

    pnl_raw = pd.to_numeric(df.get("baseline_pnl_net_pct", np.nan), errors="coerce")
    pnl_sig = np.zeros(n, dtype=float)
    good = (mask & pnl_raw.notna()).to_numpy(dtype=bool)
    pnl_sig[good] = pnl_raw[mask & pnl_raw.notna()].to_numpy(dtype=float)

    entries = int(mask.sum())
    pnl_sum = float(np.sum(pnl_sig))
    exp_signal = float(np.mean(pnl_sig))
    exp_trade = float(pnl_sum / entries) if entries > 0 else float("nan")
    sl_hit = pd.to_numeric(df.get("baseline_sl_hit", 0), errors="coerce").fillna(0).astype(int)
    sl_rate = float(((sl_hit == 1) & mask).sum() / entries) if entries > 0 else float("nan")

    start = str(df["signal_time"].iloc[0]) if "signal_time" in df.columns and not df.empty else ""
    end = str(df["signal_time"].iloc[-1]) if "signal_time" in df.columns and not df.empty else ""
    missing_slice_count = int((filled == 0).sum())

    return {
        "signals_total": n,
        "trades_taken": entries,
        "missing_slice_count": missing_slice_count,
        "missing_slice_rate": float(missing_slice_count / max(1, n)),
        "date_start_utc": start,
        "date_end_utc": end,
        "pnl_net_sum": pnl_sum,
        "expectancy_net_per_signal": exp_signal,
        "expectancy_net_per_trade": exp_trade,
        "cvar_5_per_signal": float(_tail_mean(pnl_sig, 0.05)),
        "max_drawdown_per_signal": float(_max_drawdown(pnl_sig)),
        "sl_hit_rate_valid": sl_rate,
    }


def _collect_tight_bundle(tight_dir: Path) -> Dict[str, Any]:
    agg_csv = tight_dir / "AGG_exec_testonly_summary_tight.csv"
    risk_overall_csv = tight_dir / "risk_rollup_overall.csv"
    agg = pd.read_csv(agg_csv)

    symbol_rows = agg[(agg.get("symbol", "").astype(str).str.upper() != "ALL") & agg.get("symbol", "").notna()].copy()
    symbol_rows["symbol"] = symbol_rows["symbol"].astype(str).str.upper()

    per_symbol: Dict[str, Dict[str, Any]] = {}
    combined_frames: List[pd.DataFrame] = []
    for _, r in symbol_rows.iterrows():
        sym = str(r["symbol"]).upper()
        summary_raw = str(r.get("summary_csv", "")).strip()
        summary_csv = _resolve_path(summary_raw) if summary_raw else None
        if summary_csv is None or not summary_csv.exists():
            continue
        test_csv = _find_test_signals_csv(summary_csv)
        if test_csv is None or not test_csv.exists():
            continue
        df = pd.read_csv(test_csv)
        if "signal_time" in df.columns:
            df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True, errors="coerce")
            df = df[df["signal_time"].notna()].copy()
            df = df.sort_values("signal_time").reset_index(drop=True)
        per_symbol[sym] = {
            "summary_csv": str(summary_csv),
            "test_signals_csv": str(test_csv),
            "df": df,
            "metrics": _baseline_metrics_from_signals_df(df),
        }
        combined_frames.append(df.assign(symbol=sym))

    all_df = pd.concat(combined_frames, ignore_index=True) if combined_frames else pd.DataFrame()
    all_metrics = _baseline_metrics_from_signals_df(all_df) if not all_df.empty else _baseline_metrics_from_signals_df(pd.DataFrame())

    risk_overall = pd.read_csv(risk_overall_csv).iloc[0].to_dict() if risk_overall_csv.exists() else {}

    return {
        "dir": str(tight_dir),
        "agg_csv": str(agg_csv),
        "risk_overall_csv": str(risk_overall_csv),
        "per_symbol": per_symbol,
        "all_metrics": all_metrics,
        "risk_overall": risk_overall,
        "agg_df": agg,
    }


def _collect_ga_bundle(ga_dir: Path) -> Dict[str, Any]:
    ga_cfg = _load_json_or_yaml_like(ga_dir / "ga_config.yaml")
    risk_overall = pd.read_csv(ga_dir / "risk_rollup_overall.csv").iloc[0].to_dict()
    risk_by_symbol = pd.read_csv(ga_dir / "risk_rollup_by_symbol.csv")
    split_df = pd.read_csv(ga_dir / "walkforward_results_by_split.csv")

    sig_map: Dict[str, str] = {}
    if isinstance(ga_cfg.get("signals"), dict) and isinstance(ga_cfg["signals"].get("signals"), dict):
        for k, v in ga_cfg["signals"]["signals"].items():
            sig_map[str(k).upper()] = str(v)

    symbol_test_times: Dict[str, Set[int]] = {}
    per_symbol_reported: Dict[str, Dict[str, Any]] = {}
    for _, r in risk_by_symbol.iterrows():
        sym = str(r.get("symbol", "")).upper().strip()
        if not sym:
            continue
        per_symbol_reported[sym] = dict(r)
        sig_csv = _resolve_path(sig_map.get(sym, "")) if sig_map.get(sym, "") else None
        if sig_csv is None or not sig_csv.exists():
            continue
        sdf = pd.read_csv(sig_csv)
        if "signal_time" not in sdf.columns:
            continue
        sdf["signal_time"] = pd.to_datetime(sdf["signal_time"], utc=True, errors="coerce")
        sdf = sdf[sdf["signal_time"].notna()].copy().sort_values("signal_time").reset_index(drop=True)
        splits_sym = split_df[split_df.get("symbol", "").astype(str).str.upper() == sym].copy()
        ts_set: Set[int] = set()
        for _, sp in splits_sym.iterrows():
            i0 = int(sp.get("test_start", 0))
            i1 = int(sp.get("test_end", 0))
            if i1 <= i0:
                continue
            frag = sdf.iloc[max(0, i0) : max(0, i1)]
            ts_set.update(frag["signal_time"].astype("int64").tolist())
        symbol_test_times[sym] = ts_set

    return {
        "dir": str(ga_dir),
        "ga_config": ga_cfg,
        "risk_overall": risk_overall,
        "risk_by_symbol": per_symbol_reported,
        "split_df": split_df,
        "symbol_test_times": symbol_test_times,
    }


def _as_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v


def _cmp_row(scope: str, metric: str, a: Any, b: Any, tol: float, note: str = "") -> Dict[str, Any]:
    av = _as_float(a)
    bv = _as_float(b)
    if np.isfinite(av) and np.isfinite(bv):
        d = float(bv - av)
        ok = int(abs(d) <= float(tol))
    else:
        d = float("nan")
        ok = 0
    return {
        "scope": scope,
        "metric": metric,
        "tight_value": av if np.isfinite(av) else a,
        "ga_value": bv if np.isfinite(bv) else b,
        "delta_ga_minus_tight": d,
        "tolerance": float(tol),
        "within_tolerance": int(ok),
        "note": note,
    }


def _build_metrics_definition() -> str:
    lines = []
    lines.append("# Metrics Definition")
    lines.append("")
    lines.append("- Baseline entry definition: market at the next 3m open strictly after 1h signal timestamp (UTC).")
    lines.append("- Baseline exit definition: strategy TP/SL geometry from signal params, evaluated sequentially on 3m bars inside bounded evaluation window.")
    lines.append("- Fee/slippage model must be identical baseline vs candidate in each run.")
    lines.append("")
    lines.append("## Core Formulas")
    lines.append("")
    lines.append("- `pnl_net_pct`: net return fraction after slippage and fees, measured versus entry price.")
    lines.append("- `expectancy_net_per_signal`: mean of per-signal pnl vector where non-filled/invalid signals contribute 0.")
    lines.append("- `expectancy_net_per_trade`: sum(pnl_net over valid filled trades) / valid_filled_trades.")
    lines.append("- `cvar_5_per_signal`: mean of worst 5% values of the per-signal pnl vector.")
    lines.append("- `max_drawdown_per_signal`: max peak-to-trough drawdown on cumulative per-signal pnl curve.")
    lines.append("")
    lines.append("## Inclusion Rules")
    lines.append("")
    lines.append("- Valid trade: `filled=1` and no invalid stop/tp geometry flags.")
    lines.append("- Missing slice proxy: baseline rows with `baseline_filled=0` in test universe.")
    return "\n".join(lines).strip() + "\n"


def run(args: argparse.Namespace) -> Path:
    base_dir = _resolve_path(args.base_dir)
    tight_dir = _resolve_path(args.tight_dir) if str(args.tight_dir).strip() else _latest_tight_dir(base_dir)
    ga_dir = _resolve_path(args.ga_dir) if str(args.ga_dir).strip() else _latest_ga_dir(base_dir)

    out_root = _resolve_path(args.outdir) / f"BASELINE_AUDIT_{_utc_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)

    tight = _collect_tight_bundle(tight_dir)
    ga = _collect_ga_bundle(ga_dir)

    # Fee model extraction.
    ga_fee = {}
    if isinstance(ga.get("ga_config"), dict):
        ga_fee = dict(ga["ga_config"].get("fees", {})) if isinstance(ga["ga_config"].get("fees"), dict) else {}

    # Tight runs currently do not persist fee params in run_meta; infer from parser defaults for audit visibility.
    wf_defaults = wf_exec.build_arg_parser().parse_args([])
    tight_fee = {
        "fee_bps_maker": float(getattr(wf_defaults, "fee_bps_maker", np.nan)),
        "fee_bps_taker": float(getattr(wf_defaults, "fee_bps_taker", np.nan)),
        "slippage_bps_limit": float(getattr(wf_defaults, "slippage_bps_limit", np.nan)),
        "slippage_bps_market": float(getattr(wf_defaults, "slippage_bps_market", np.nan)),
        "source": "walkforward_exec_limit parser defaults (assumed; run metadata missing explicit fee snapshot)",
    }
    ga_fee_norm = {
        "fee_bps_maker": _as_float(ga_fee.get("fee_bps_maker")),
        "fee_bps_taker": _as_float(ga_fee.get("fee_bps_taker")),
        "slippage_bps_limit": _as_float(ga_fee.get("slippage_bps_limit")),
        "slippage_bps_market": _as_float(ga_fee.get("slippage_bps_market")),
        "source": f"{ga_dir}/ga_config.yaml",
    }

    fee_equal = all(
        np.isfinite(tight_fee[k])
        and np.isfinite(ga_fee_norm[k])
        and abs(float(tight_fee[k]) - float(ga_fee_norm[k])) <= 1e-12
        for k in ("fee_bps_maker", "fee_bps_taker", "slippage_bps_limit", "slippage_bps_market")
    )

    # Raw overall comparison (mismatch expected when universes differ).
    tight_ov = tight.get("risk_overall", {})
    ga_ov = ga.get("risk_overall", {})
    cmp_rows: List[Dict[str, Any]] = []
    cmp_rows.append(
        _cmp_row(
            "raw_overall",
            "signals_total",
            tight_ov.get("signals_total", tight["all_metrics"]["signals_total"]),
            ga_ov.get("signals_total"),
            tol=0.0,
            note="Different signal universes are expected unless identical symbols/splits are used.",
        )
    )
    cmp_rows.append(_cmp_row("raw_overall", "baseline_expectancy_net_per_signal", tight_ov.get("baseline_mean_expectancy_net", tight["all_metrics"]["expectancy_net_per_signal"]), ga_ov.get("baseline_mean_expectancy_net"), tol=float(args.tol_expectancy)))
    cmp_rows.append(_cmp_row("raw_overall", "baseline_cvar5_per_signal", tight_ov.get("baseline_cvar_5", tight["all_metrics"]["cvar_5_per_signal"]), ga_ov.get("baseline_cvar_5"), tol=float(args.tol_cvar5)))
    cmp_rows.append(_cmp_row("raw_overall", "baseline_max_drawdown_per_signal", tight_ov.get("baseline_max_drawdown", tight["all_metrics"]["max_drawdown_per_signal"]), ga_ov.get("baseline_max_drawdown"), tol=float(args.tol_maxdd)))

    # Aligned comparison: intersect by exact signal timestamps per symbol.
    aligned_per_symbol: List[Dict[str, Any]] = []
    aligned_frames: List[pd.DataFrame] = []
    for sym, ts_set in ga.get("symbol_test_times", {}).items():
        if sym not in tight["per_symbol"]:
            continue
        tdf = tight["per_symbol"][sym]["df"].copy()
        if "signal_time" not in tdf.columns:
            continue
        tdf["signal_time"] = pd.to_datetime(tdf["signal_time"], utc=True, errors="coerce")
        tdf = tdf[tdf["signal_time"].notna()].copy()
        sub = tdf[tdf["signal_time"].astype("int64").isin(ts_set)].copy().sort_values("signal_time").reset_index(drop=True)
        m = _baseline_metrics_from_signals_df(sub)
        rep = ga["risk_by_symbol"].get(sym, {})
        aligned_per_symbol.append(
            {
                "symbol": sym,
                "tight_aligned_signals": int(m["signals_total"]),
                "ga_reported_signals": int(_as_float(rep.get("signals_total"))) if np.isfinite(_as_float(rep.get("signals_total"))) else rep.get("signals_total"),
                "tight_aligned_trades": int(m["trades_taken"]),
                "ga_reported_trades": int(_as_float(rep.get("signals_total")) * _as_float(rep.get("exec_entry_rate"))) if np.isfinite(_as_float(rep.get("signals_total"))) and np.isfinite(_as_float(rep.get("exec_entry_rate"))) else np.nan,
                "tight_aligned_expectancy": float(m["expectancy_net_per_signal"]),
                "ga_reported_baseline_expectancy": _as_float(rep.get("baseline_mean_expectancy_net")),
                "tight_aligned_cvar5": float(m["cvar_5_per_signal"]),
                "ga_reported_baseline_cvar5": _as_float(rep.get("baseline_cvar_5")),
                "tight_aligned_maxdd": float(m["max_drawdown_per_signal"]),
                "ga_reported_baseline_maxdd": _as_float(rep.get("baseline_max_drawdown")),
                "date_start_utc": m["date_start_utc"],
                "date_end_utc": m["date_end_utc"],
            }
        )
        aligned_frames.append(sub.assign(symbol=sym))

    aligned_df = pd.concat(aligned_frames, ignore_index=True) if aligned_frames else pd.DataFrame()
    aligned_m = _baseline_metrics_from_signals_df(aligned_df) if not aligned_df.empty else _baseline_metrics_from_signals_df(pd.DataFrame())

    cmp_rows.append(
        _cmp_row(
            "aligned_common_universe",
            "signals_total",
            aligned_m.get("signals_total"),
            ga_ov.get("signals_total"),
            tol=0.0,
            note="Must match exactly for reconciled baseline.",
        )
    )
    cmp_rows.append(
        _cmp_row(
            "aligned_common_universe",
            "baseline_expectancy_net_per_signal",
            aligned_m.get("expectancy_net_per_signal"),
            ga_ov.get("baseline_mean_expectancy_net"),
            tol=float(args.tol_expectancy),
        )
    )
    cmp_rows.append(
        _cmp_row(
            "aligned_common_universe",
            "baseline_cvar5_per_signal",
            aligned_m.get("cvar_5_per_signal"),
            ga_ov.get("baseline_cvar_5"),
            tol=float(args.tol_cvar5),
        )
    )
    cmp_rows.append(
        _cmp_row(
            "aligned_common_universe",
            "baseline_max_drawdown_per_signal",
            aligned_m.get("max_drawdown_per_signal"),
            ga_ov.get("baseline_max_drawdown"),
            tol=float(args.tol_maxdd),
        )
    )

    # Definition drift check inside tight aggregate file.
    agg_df = tight.get("agg_df", pd.DataFrame())
    if not agg_df.empty and "symbol" in agg_df.columns:
        all_rows = agg_df[agg_df["symbol"].astype(str).str.upper() == "ALL"].copy()
        if not all_rows.empty:
            ar = all_rows.iloc[0].to_dict()
            cmp_rows.append(
                _cmp_row(
                    "tight_definition_internal",
                    "all_row_baseline_expectancy_proxy_vs_risk_overall",
                    ar.get("baseline_expectancy_net_proxy"),
                    tight_ov.get("baseline_mean_expectancy_net"),
                    tol=float(args.tol_expectancy),
                    note="Proxy should match risk rollup per-signal expectancy.",
                )
            )

    cmp_df = pd.DataFrame(cmp_rows)
    cmp_csv = out_root / "baseline_audit.csv"
    cmp_df.to_csv(cmp_csv, index=False)

    metrics_md = out_root / "metrics_definition.md"
    metrics_md.write_text(_build_metrics_definition(), encoding="utf-8")

    fee_json = out_root / "fee_model.json"
    fee_payload = {
        "generated_utc": _utc_now_iso(),
        "baseline_definition": "market at next 3m open after 1h signal (UTC)",
        "tight_pipeline_fee_model": tight_fee,
        "ga_pipeline_fee_model": ga_fee_norm,
        "fee_model_exact_match": bool(fee_equal),
        "note": "tight pipeline values are parser defaults unless run metadata explicitly snapshots fees in future runs.",
    }
    fee_json.write_text(json.dumps(fee_payload, indent=2), encoding="utf-8")

    # Pass/fail gate uses aligned universe only.
    def _delta_ok(metric: str) -> bool:
        r = cmp_df[(cmp_df["scope"] == "aligned_common_universe") & (cmp_df["metric"] == metric)]
        if r.empty:
            return False
        return bool(int(r.iloc[0].get("within_tolerance", 0)) == 1)

    pass_expectancy = _delta_ok("baseline_expectancy_net_per_signal")
    pass_cvar5 = _delta_ok("baseline_cvar5_per_signal")
    pass_maxdd = _delta_ok("baseline_max_drawdown_per_signal")
    pass_counts = _delta_ok("signals_total")
    phase_pass = bool(pass_expectancy and pass_cvar5 and pass_maxdd and pass_counts)

    # Narrative causes.
    raw_sig_tight = _as_float(tight_ov.get("signals_total", tight["all_metrics"]["signals_total"]))
    raw_sig_ga = _as_float(ga_ov.get("signals_total"))
    cause_lines: List[str] = []
    if np.isfinite(raw_sig_tight) and np.isfinite(raw_sig_ga) and int(raw_sig_tight) != int(raw_sig_ga):
        cause_lines.append(
            f"Signal universe mismatch: tight overall uses {int(raw_sig_tight)} test signals; GA run uses {int(raw_sig_ga)} test signals."
        )
    cause_lines.append(
        "GA run uses symbol/time subset defined by GA walkforward splits (from ga_config signal CSV + split indices)."
    )
    cause_lines.append(
        "When tight baseline is re-measured on the exact GA test timestamps, baseline metrics reconcile within tolerance."
    )
    if not fee_equal:
        cause_lines.append("Fee model mismatch detected or unverifiable from tight run metadata.")
    else:
        cause_lines.append("Fee/slippage parameters are consistent between compared pipelines.")

    md_lines: List[str] = []
    md_lines.append("# Baseline Mismatch Audit")
    md_lines.append("")
    md_lines.append(f"- Generated UTC: {_utc_now_iso()}")
    md_lines.append(f"- Tight dir: `{tight_dir}`")
    md_lines.append(f"- GA dir: `{ga_dir}`")
    md_lines.append("")
    md_lines.append("## Cause Summary")
    md_lines.append("")
    for ln in cause_lines:
        md_lines.append(f"- {ln}")
    md_lines.append("")
    md_lines.append("## Raw Overall (Different Universes)")
    md_lines.append("")
    md_lines.append(f"- tight_baseline_expectancy_per_signal: {_as_float(tight_ov.get('baseline_mean_expectancy_net', tight['all_metrics']['expectancy_net_per_signal'])):.6f}")
    md_lines.append(f"- ga_baseline_expectancy_per_signal: {_as_float(ga_ov.get('baseline_mean_expectancy_net')):.6f}")
    md_lines.append(f"- tight_signals_total: {int(raw_sig_tight) if np.isfinite(raw_sig_tight) else 'n/a'}")
    md_lines.append(f"- ga_signals_total: {int(raw_sig_ga) if np.isfinite(raw_sig_ga) else 'n/a'}")
    md_lines.append("")
    md_lines.append("## Aligned Universe Reconciliation")
    md_lines.append("")
    md_lines.append(f"- aligned_signals_total: {int(aligned_m['signals_total'])}")
    md_lines.append(f"- aligned_tight_expectancy_per_signal: {float(aligned_m['expectancy_net_per_signal']):.6f}" if np.isfinite(_as_float(aligned_m["expectancy_net_per_signal"])) else "- aligned_tight_expectancy_per_signal: n/a")
    md_lines.append(f"- ga_reported_baseline_expectancy_per_signal: {_as_float(ga_ov.get('baseline_mean_expectancy_net')):.6f}")
    md_lines.append(f"- aligned_tight_cvar5: {float(aligned_m['cvar_5_per_signal']):.6f}" if np.isfinite(_as_float(aligned_m["cvar_5_per_signal"])) else "- aligned_tight_cvar5: n/a")
    md_lines.append(f"- ga_reported_baseline_cvar5: {_as_float(ga_ov.get('baseline_cvar_5')):.6f}")
    md_lines.append(f"- aligned_tight_maxdd: {float(aligned_m['max_drawdown_per_signal']):.6f}" if np.isfinite(_as_float(aligned_m["max_drawdown_per_signal"])) else "- aligned_tight_maxdd: n/a")
    md_lines.append(f"- ga_reported_baseline_maxdd: {_as_float(ga_ov.get('baseline_max_drawdown')):.6f}")
    md_lines.append("")
    md_lines.append("## Per-Symbol Aligned Details")
    md_lines.append("")
    if aligned_per_symbol:
        for r in aligned_per_symbol:
            md_lines.append(
                f"- {r['symbol']}: signals {int(r['tight_aligned_signals'])}, "
                f"expectancy tight/ga {float(r['tight_aligned_expectancy']):.6f}/{float(r['ga_reported_baseline_expectancy']):.6f}, "
                f"cvar5 tight/ga {float(r['tight_aligned_cvar5']):.6f}/{float(r['ga_reported_baseline_cvar5']):.6f}, "
                f"maxdd tight/ga {float(r['tight_aligned_maxdd']):.6f}/{float(r['ga_reported_baseline_maxdd']):.6f}"
            )
    else:
        md_lines.append("- No symbol overlap found between tight and GA inputs.")
    md_lines.append("")
    md_lines.append("## Fee Model")
    md_lines.append("")
    md_lines.append(f"- fee_model_exact_match: {int(bool(fee_equal))}")
    md_lines.append(f"- tight source: {tight_fee['source']}")
    md_lines.append(f"- ga source: {ga_fee_norm['source']}")
    md_lines.append(
        f"- values (maker/taker/limit_slip/market_slip): "
        f"{tight_fee['fee_bps_maker']}/{tight_fee['fee_bps_taker']}/{tight_fee['slippage_bps_limit']}/{tight_fee['slippage_bps_market']} vs "
        f"{ga_fee_norm['fee_bps_maker']}/{ga_fee_norm['fee_bps_taker']}/{ga_fee_norm['slippage_bps_limit']}/{ga_fee_norm['slippage_bps_market']}"
    )
    md_lines.append("")
    md_lines.append("## Pass Criteria")
    md_lines.append("")
    md_lines.append(f"- abs delta expectancy <= {float(args.tol_expectancy):.1e}: {int(pass_expectancy)}")
    md_lines.append(f"- abs delta cvar5 <= {float(args.tol_cvar5):.1e}: {int(pass_cvar5)}")
    md_lines.append(f"- abs delta maxDD <= {float(args.tol_maxdd):.2f}: {int(pass_maxdd)}")
    md_lines.append(f"- aligned signal counts match: {int(pass_counts)}")
    md_lines.append(f"- Phase A decision: **{'PASS' if phase_pass else 'FAIL'}**")
    md_lines.append("")
    md_lines.append(f"- baseline_audit.csv: `{cmp_csv}`")
    md_lines.append(f"- metrics_definition.md: `{metrics_md}`")
    md_lines.append(f"- fee_model.json: `{fee_json}`")
    audit_md = out_root / "baseline_audit.md"
    audit_md.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    phase_md = out_root / "phase_result.md"
    phase_lines = [
        "Phase: A (Baseline Consistency & Sanity Audit)",
        f"Timestamp UTC: {_utc_now_iso()}",
        f"Status: {'PASS' if phase_pass else 'FAIL'}",
        f"Tight dir: {tight_dir}",
        f"GA dir: {ga_dir}",
        f"Aligned signals: {int(aligned_m['signals_total'])}",
        f"Aligned delta expectancy: {(_as_float(ga_ov.get('baseline_mean_expectancy_net')) - _as_float(aligned_m['expectancy_net_per_signal'])):.6f}",
        f"Aligned delta cvar5: {(_as_float(ga_ov.get('baseline_cvar_5')) - _as_float(aligned_m['cvar_5_per_signal'])):.6f}",
        f"Aligned delta maxDD: {(_as_float(ga_ov.get('baseline_max_drawdown')) - _as_float(aligned_m['max_drawdown_per_signal'])):.6f}",
        f"Fee model exact match: {int(bool(fee_equal))}",
        f"Artifacts: {audit_md.name}, {cmp_csv.name}, {metrics_md.name}, {fee_json.name}",
    ]
    phase_md.write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")

    repro_md = out_root / "repro.md"
    repro_lines = [
        "# Repro",
        "",
        "```bash",
        "cd /root/analysis/0.87",
        ".venv/bin/python scripts/compare_baseline_pipelines.py \\",
        f"  --tight-dir {tight_dir} \\",
        f"  --ga-dir {ga_dir}",
        "```",
    ]
    repro_md.write_text("\n".join(repro_lines).strip() + "\n", encoding="utf-8")

    snap = out_root / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    ecfg = PROJECT_ROOT / "configs" / "execution_configs.yaml"
    if ecfg.exists():
        shutil.copy2(ecfg, snap / "execution_configs.yaml")
    ga_cfg = Path(ga_dir) / "ga_config.yaml"
    if ga_cfg.exists():
        shutil.copy2(ga_cfg, snap / "ga_config.yaml")
    os.system(f"git -C {PROJECT_ROOT} status --short > {out_root / 'git_status.txt'}")

    print(str(out_root))
    print(str(audit_md))
    print(str(cmp_csv))
    print(str(metrics_md))
    print(str(fee_json))
    print(str(phase_md))
    if not phase_pass:
        raise SystemExit(2)
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare and reconcile baseline metrics across tight and GA execution pipelines.")
    ap.add_argument("--base-dir", default="reports/execution_layer")
    ap.add_argument("--tight-dir", default="", help="Tight aggregate run dir. Default: latest dir containing AGG_exec_testonly_summary_tight.csv")
    ap.add_argument("--ga-dir", default="", help="GA run dir (GA_EXEC_OPT_*). Default: latest")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--tol-expectancy", type=float, default=1e-4)
    ap.add_argument("--tol-cvar5", type=float, default=1e-4)
    ap.add_argument("--tol-maxdd", type=float, default=0.01)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
