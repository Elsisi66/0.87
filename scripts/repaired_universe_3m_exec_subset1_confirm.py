#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402
from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import repaired_frontier_contamination_audit as audit  # noqa: E402


RUN_PREFIX = "REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM"
SUBSET_SYMBOLS = [
    "NEARUSDT",
    "SOLUSDT",
    "LTCUSDT",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_tag() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(v, utc=True))
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (set, tuple)):
            return list(v)
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def markdown_table(df: pd.DataFrame, cols: List[str], n: int = 20) -> str:
    if df.empty:
        return "_(none)_"
    use = [c for c in cols if c in df.columns]
    if not use:
        return "_(none)_"
    x = df.loc[:, use].head(n).copy()
    out = ["| " + " | ".join(x.columns.tolist()) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"]
    for row in x.itertuples(index=False):
        vals: List[str] = []
        for v in row:
            if isinstance(v, float):
                vals.append(f"{v:.6g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def find_latest_complete(root: Path, pattern: str, required: Iterable[str]) -> Optional[Path]:
    cands = sorted([p for p in root.glob(pattern) if p.is_dir()], key=lambda p: p.name)
    for cand in reversed(cands):
        if all((cand / req).exists() for req in required):
            return cand.resolve()
    return None


def discover_freeze_dir(arg_value: str) -> Path:
    if arg_value:
        freeze_dir = Path(arg_value).resolve()
    else:
        exec_root = PROJECT_ROOT / "reports" / "execution_layer"
        freeze_dir = find_latest_complete(
            exec_root,
            "REPAIRED_1H_UNIVERSE_FREEZE_*",
            [
                "repaired_best_by_symbol.csv",
                "repaired_universe_selected_params",
                "repaired_universe_freeze_manifest.json",
            ],
        )
        if freeze_dir is None:
            raise FileNotFoundError("Missing latest complete REPAIRED_1H_UNIVERSE_FREEZE_* directory")
    required = [
        freeze_dir / "repaired_best_by_symbol.csv",
        freeze_dir / "repaired_universe_selected_params",
        freeze_dir / "repaired_universe_freeze_manifest.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required freeze artifacts: " + ", ".join(missing))
    return freeze_dir


def discover_pathology_dir(arg_value: str) -> Path:
    if arg_value:
        run_dir = Path(arg_value).resolve()
    else:
        exec_root = PROJECT_ROOT / "reports" / "execution_layer"
        run_dir = find_latest_complete(
            exec_root,
            "REPAIRED_UNIVERSE_PREEXEC_PATHOLOGY_AUDIT_*",
            [
                "repaired_universe_pathology_by_symbol.csv",
                "repaired_universe_pathology_summary.csv",
                "repaired_universe_pathology_report.md",
            ],
        )
        if run_dir is None:
            raise FileNotFoundError("Missing latest complete REPAIRED_UNIVERSE_PREEXEC_PATHOLOGY_AUDIT_* directory")
    return run_dir


def discover_subset_eval_dir(arg_value: str) -> Path:
    if arg_value:
        run_dir = Path(arg_value).resolve()
    else:
        exec_root = PROJECT_ROOT / "reports" / "execution_layer"
        run_dir = find_latest_complete(
            exec_root,
            "REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_*",
            [
                "repaired_subset1_exec_eval_by_symbol.csv",
                "repaired_subset1_exec_eval_summary.csv",
                "repaired_subset1_exec_eval_report.md",
            ],
        )
        if run_dir is None:
            raise FileNotFoundError("Missing latest complete REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_* directory")
    return run_dir


def load_priority_subset(pathology_dir: Path) -> Tuple[str, List[str]]:
    summary = pd.read_csv(pathology_dir / "repaired_universe_pathology_summary.csv")
    rec_row = summary[summary["metric"].astype(str) == "recommendation"]
    subset_row = summary[summary["metric"].astype(str) == "priority_subset"]
    rec = str(rec_row.iloc[0]["value"]) if not rec_row.empty else ""
    subset_txt = str(subset_row.iloc[0]["value"]) if not subset_row.empty else ""
    subset = [x.strip().upper() for x in subset_txt.split(",") if x.strip()]
    return rec, subset


def parse_params_from_row(row: pd.Series, selected_params_dir: Path) -> Dict[str, Any]:
    symbol = str(row["symbol"]).upper()
    frozen_fp = selected_params_dir / f"{symbol}_repaired_selected_params.json"
    if frozen_fp.exists():
        payload = json.loads(frozen_fp.read_text(encoding="utf-8"))
        params = payload.get("params")
        if isinstance(params, dict):
            return dict(params)
    payload_raw = row.get("params_payload_json", "")
    if isinstance(payload_raw, str) and payload_raw.strip():
        return dict(json.loads(payload_raw))
    raise ValueError(f"Missing usable params payload for {symbol}")


def build_signal_table_for_row(
    *,
    row: pd.Series,
    params: Dict[str, Any],
    df_cache: Dict[Tuple[str, str], pd.DataFrame],
    raw_cache: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    symbol = str(row["symbol"]).upper()
    period_start = pd.to_datetime(row["period_start"], utc=True, errors="coerce")
    period_end = pd.to_datetime(row["period_end"], utc=True, errors="coerce")
    df_feat = audit.prep_full_df(symbol, params, df_cache, raw_cache)
    base = audit.build_signal_table(
        symbol=symbol,
        params_dict=params,
        df_feat=df_feat,
        period_start=period_start,
        period_end=period_end,
    )
    if base.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "signal_id",
                "signal_time_utc",
                "strategy_tp_mult",
                "strategy_sl_mult",
                "bucket_1h",
                "params_source",
                "model_source",
                "atr_percentile_1h",
            ]
        )
    out = pd.DataFrame(
        {
            "symbol": symbol,
            "signal_id": base["signal_id"].astype(str),
            "signal_time_utc": pd.to_datetime(base["signal_time"], utc=True, errors="coerce"),
            "strategy_tp_mult": pd.to_numeric(base["tp_mult"], errors="coerce"),
            "strategy_sl_mult": pd.to_numeric(base["sl_mult"], errors="coerce"),
            "bucket_1h": "REPAIRED_LONG",
            "params_source": str(row.get("params_source", "")),
            "model_source": "REPAIRED_1H_UNIVERSE_FREEZE",
            "atr_percentile_1h": np.nan,
        }
    )
    return out.dropna(subset=["signal_time_utc", "strategy_tp_mult", "strategy_sl_mult"]).sort_values(
        ["signal_time_utc", "signal_id"]
    ).reset_index(drop=True)


def local_full_3m_path(symbol: str) -> Optional[Path]:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_3m_full.parquet"
    if fp.exists():
        return fp.resolve()
    return None


def merge_foundation_windows_to_single_parquet(
    *,
    symbol: str,
    windows_df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    cache_dir: Path,
) -> Optional[Path]:
    frames: List[pd.DataFrame] = []
    paths = [str(x) for x in windows_df.get("parquet_path", pd.Series(dtype=str)).tolist() if isinstance(x, str) and x.strip()]
    seen: set[str] = set()
    for raw in paths:
        if raw in seen:
            continue
        seen.add(raw)
        fp = Path(raw)
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp)
            df = phase_v.exec3m._normalize_ohlcv_cols(df)  # pylint: disable=protected-access
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
            df = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] < end_ts)].reset_index(drop=True)
            if not df.empty:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    merged = phase_v.exec3m._normalize_ohlcv_cols(merged)  # pylint: disable=protected-access
    merged = merged[(merged["Timestamp"] >= start_ts) & (merged["Timestamp"] < end_ts)].reset_index(drop=True)
    if merged.empty:
        return None
    ensure_dir(cache_dir)
    out_fp = cache_dir / f"{symbol}_subset1_merged_3m.parquet"
    merged.to_parquet(out_fp, index=False)
    return out_fp.resolve()


def build_window_pool_for_symbol(
    *,
    symbol: str,
    signal_df: pd.DataFrame,
    foundation_state: phase_v.FoundationState,
    cache_dir: Path,
    reuse_cache_dirs: Optional[List[Path]] = None,
) -> pd.DataFrame:
    if signal_df.empty:
        return pd.DataFrame(
            columns=["symbol", "window_id", "window_start_utc", "window_end_utc", "parquet_path", "coverage_ratio", "download_source"]
        )
    sig_min = pd.to_datetime(signal_df["signal_time_utc"], utc=True).min() - pd.Timedelta(hours=48)
    sig_max = pd.to_datetime(signal_df["signal_time_utc"], utc=True).max() + pd.Timedelta(hours=48)
    local_fp = local_full_3m_path(symbol)
    if local_fp is not None:
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "window_id": f"{symbol}_LOCAL_FULL_3M",
                    "window_start_utc": sig_min,
                    "window_end_utc": sig_max,
                    "parquet_path": str(local_fp),
                    "coverage_ratio": 1.0,
                    "download_source": "local_full_3m",
                }
            ]
        )
    for prior_dir in reuse_cache_dirs or []:
        prior_fp = prior_dir / f"{symbol}_subset1_merged_3m.parquet"
        if prior_fp.exists():
            return pd.DataFrame(
                [
                    {
                        "symbol": symbol,
                        "window_id": f"{symbol}_REUSED_SUBSET1_CACHE",
                        "window_start_utc": sig_min,
                        "window_end_utc": sig_max,
                        "parquet_path": str(prior_fp.resolve()),
                        "coverage_ratio": 1.0,
                        "download_source": "prior_subset1_merged_3m",
                    }
                ]
            )
    win = foundation_state.download_manifest[
        foundation_state.download_manifest["symbol"].astype(str).str.upper() == symbol
    ].copy()
    if win.empty:
        return win
    win["window_start_utc"] = pd.to_datetime(win["window_start_utc"], utc=True, errors="coerce")
    win["window_end_utc"] = pd.to_datetime(win["window_end_utc"], utc=True, errors="coerce")
    win = win[(win["window_end_utc"] >= sig_min) & (win["window_start_utc"] <= sig_max)].copy()
    if win.empty:
        return win
    merged_fp = merge_foundation_windows_to_single_parquet(
        symbol=symbol,
        windows_df=win,
        start_ts=sig_min,
        end_ts=sig_max,
        cache_dir=cache_dir,
    )
    if merged_fp is not None:
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "window_id": f"{symbol}_MERGED_FOUNDATION_3M",
                    "window_start_utc": sig_min,
                    "window_end_utc": sig_max,
                    "parquet_path": str(merged_fp),
                    "coverage_ratio": float(pd.to_numeric(win.get("coverage_ratio", pd.Series([1.0])), errors="coerce").fillna(1.0).mean()),
                    "download_source": "foundation_merged_3m",
                }
            ]
        )
    keep = [c for c in ["symbol", "window_id", "window_start_utc", "window_end_utc", "parquet_path", "coverage_ratio", "download_source"] if c in win.columns]
    return win.loc[:, keep].sort_values(["window_start_utc", "window_id"]).reset_index(drop=True)


def quality_from_build_meta(build_meta: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    total = int(build_meta.get("signals_total", 0))
    with_data = int(build_meta.get("signals_with_3m_data", 0))
    partial = int(build_meta.get("signals_partial_3m_data", 0))
    missing = int(build_meta.get("signals_missing_3m_data", 0))
    ready = max(0, with_data - partial)
    miss_rate = float(missing / total) if total > 0 else float("nan")
    if with_data <= 0:
        status = "DATA_BLOCKED"
    elif missing > 0 or partial > 0:
        status = "PARTIAL"
    else:
        status = "READY"
    qrow = {
        "integrity_status": status,
        "windows_ready": int(ready),
        "windows_partial": int(partial),
        "windows_blocked": int(missing),
        "missing_window_rate": miss_rate,
        "signals_covered": int(with_data),
        "signals_uncovered": int(missing),
    }
    rrow = {
        "bucket_1h": "REPAIRED_LONG",
        "integrity_status": status,
        "windows_ready": int(ready),
        "windows_partial": int(partial),
        "windows_blocked": int(missing),
    }
    return qrow, rrow


def map_exec_classification(
    base_class: str,
    base_reason: str,
    best_row: pd.Series,
    qrow: Dict[str, Any],
    route_meta: Dict[str, Any],
) -> Tuple[str, str]:
    if base_class == "DATA_BLOCKED":
        return "INCONCLUSIVE", base_reason
    if best_row.empty:
        return "NO_EXEC_EDGE", "no_non_reference_variants"
    delta = float(pd.to_numeric(pd.Series([best_row.get("delta_expectancy_vs_1h_reference", np.nan)]), errors="coerce").iloc[0])
    valid = int(best_row.get("valid_for_ranking", 0)) == 1
    missing_rate = float(pd.to_numeric(pd.Series([qrow.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0])
    invalid_reason = str(best_row.get("invalid_reason", ""))
    route_supported = int(route_meta.get("route_family_supported", 0)) == 1
    route_count = int(best_row.get("route_count", 0))
    route_pass_rate = float(pd.to_numeric(pd.Series([best_row.get("route_pass_rate", np.nan)]), errors="coerce").iloc[0])

    if (not valid) and (("missing_slice_rate" in invalid_reason) or (np.isfinite(missing_rate) and missing_rate > 0.50)):
        return "INCONCLUSIVE", "coverage_limited_under_repaired_subset"
    if int(route_meta.get("route_family_supported", 0)) != 1:
        return "INCONCLUSIVE", "route_family_unavailable"
    if base_class == "MODEL_A_STRONG_GO":
        return "ACTIVE_3M_SURVIVOR", base_reason
    if base_class == "MODEL_A_WEAK_GO":
        return "SHADOW_ONLY", base_reason
    if route_supported and route_count > 0 and np.isfinite(route_pass_rate) and route_pass_rate < 0.5 and delta > 0.0:
        return "INCONCLUSIVE", "positive_delta_but_route_fragile"
    return "NO_EXEC_EDGE", base_reason


def build_overall_summary(by_symbol_df: pd.DataFrame) -> pd.DataFrame:
    improved_mask = by_symbol_df["classification"].astype(str).isin(["ACTIVE_3M_SURVIVOR", "SHADOW_ONLY"])
    degraded_mask = by_symbol_df["classification"].astype(str) == "NO_EXEC_EDGE"
    inconclusive_mask = by_symbol_df["classification"].astype(str) == "INCONCLUSIVE"

    ref_w = (
        (pd.to_numeric(by_symbol_df["reference_exec_expectancy_net"], errors="coerce") * pd.to_numeric(by_symbol_df["reference_entries_valid"], errors="coerce"))
        .sum()
        / max(1.0, pd.to_numeric(by_symbol_df["reference_entries_valid"], errors="coerce").sum())
    )
    best_w = (
        (pd.to_numeric(by_symbol_df["confirmed_exec_expectancy_net"], errors="coerce") * pd.to_numeric(by_symbol_df["best_entries_valid"], errors="coerce"))
        .sum()
        / max(1.0, pd.to_numeric(by_symbol_df["best_entries_valid"], errors="coerce").sum())
    )

    rows = [
        {"metric": "symbols_evaluated", "value": int(len(by_symbol_df))},
        {"metric": "improved_count", "value": int(improved_mask.sum())},
        {"metric": "degraded_count", "value": int(degraded_mask.sum())},
        {"metric": "neutral_or_inconclusive_count", "value": int(inconclusive_mask.sum())},
        {"metric": "aggregate_reference_expectancy_weighted", "value": float(ref_w)},
        {"metric": "aggregate_best_expectancy_weighted", "value": float(best_w)},
        {"metric": "aggregate_delta_expectancy_weighted", "value": float(best_w - ref_w)},
        {"metric": "active_3m_survivor_count", "value": int((by_symbol_df["classification"] == "ACTIVE_3M_SURVIVOR").sum())},
        {"metric": "shadow_only_count", "value": int((by_symbol_df["classification"] == "SHADOW_ONLY").sum())},
        {"metric": "no_exec_edge_count", "value": int((by_symbol_df["classification"] == "NO_EXEC_EDGE").sum())},
        {"metric": "inconclusive_count", "value": int((by_symbol_df["classification"] == "INCONCLUSIVE").sum())},
    ]
    return pd.DataFrame(rows)


def final_recommendation(by_symbol_df: pd.DataFrame, summary_df: pd.DataFrame) -> Tuple[str, List[str]]:
    keep = by_symbol_df[by_symbol_df["classification"].astype(str) == "ACTIVE_3M_SURVIVOR"]["symbol"].astype(str).tolist()
    shadow = by_symbol_df[by_symbol_df["classification"].astype(str) == "SHADOW_ONLY"]["symbol"].astype(str).tolist()
    if len(keep) == len(by_symbol_df):
        return "KEEP_SELECTIVE_3M_SUBSET", keep + shadow
    if len(keep) >= 1:
        return "SHRINK_TO_SMALLER_ACTIVE_SET", keep + shadow
    return "STOP_3M_SELECTIVE_BRANCH", keep + shadow


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Strict route-enabled repaired-universe 3m confirmation on surviving subset")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--freeze-dir", default="")
    ap.add_argument("--pathology-dir", default="")
    ap.add_argument("--subset-eval-dir", default="")
    ap.add_argument("--foundation-dir", default="")
    ap.add_argument("--seed", type=int, default=20260304)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    freeze_dir = discover_freeze_dir(args.freeze_dir)
    pathology_dir = discover_pathology_dir(args.pathology_dir)
    subset_eval_dir = discover_subset_eval_dir(args.subset_eval_dir)
    pathology_recommendation, pathology_subset = load_priority_subset(pathology_dir)

    if pathology_recommendation != "RUN_3M_ON_PRIORITY_SUBSET_FIRST":
        raise RuntimeError(
            f"Preflight pathology recommendation is not RUN_3M_ON_PRIORITY_SUBSET_FIRST: {pathology_recommendation}"
        )
    if set(SUBSET_SYMBOLS) - set(pathology_subset):
        raise RuntimeError(
            f"Approved first-wave subset mismatch vs pathology audit. Missing from pathology priority subset: {sorted(set(SUBSET_SYMBOLS) - set(pathology_subset))}"
        )
    bounded_eval_fp = subset_eval_dir / "repaired_subset1_exec_eval_by_symbol.csv"
    prior_cache_dir = subset_eval_dir / "_window_cache"
    bounded_eval_df = pd.read_csv(bounded_eval_fp)
    bounded_eval_df["symbol"] = bounded_eval_df["symbol"].astype(str).str.upper()
    bounded_keep = bounded_eval_df[
        bounded_eval_df["classification"].astype(str) == "KEEP_FOR_EXEC_LAYER"
    ]["symbol"].astype(str).str.upper().tolist()
    if sorted(bounded_keep) != sorted(SUBSET_SYMBOLS):
        raise RuntimeError(
            f"Strict confirm subset mismatch vs bounded subset survivors. Expected {sorted(SUBSET_SYMBOLS)}, got {sorted(bounded_keep)}"
        )

    foundation_dir = Path(args.foundation_dir).resolve() if str(args.foundation_dir).strip() else phase_v.find_latest_foundation_dir()
    foundation_state = phase_v.load_foundation_state(foundation_dir)

    universe_fp = freeze_dir / "repaired_best_by_symbol.csv"
    selected_params_dir = freeze_dir / "repaired_universe_selected_params"
    universe_df = pd.read_csv(universe_fp)
    if "symbol" not in universe_df.columns:
        raise KeyError("Missing required column `symbol` in repaired_best_by_symbol.csv")
    universe_df["symbol"] = universe_df["symbol"].astype(str).str.upper()
    subset_df = universe_df[universe_df["symbol"].isin(SUBSET_SYMBOLS)].copy()
    if len(subset_df) != len(SUBSET_SYMBOLS):
        missing = sorted(set(SUBSET_SYMBOLS) - set(subset_df["symbol"].tolist()))
        raise RuntimeError(f"Frozen repaired universe is missing subset symbols: {missing}")
    subset_df = subset_df.set_index("symbol").loc[SUBSET_SYMBOLS].reset_index()

    run_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    inputs_dir = ensure_dir(run_dir / "_inputs")
    window_cache_dir = ensure_dir(run_dir / "_window_cache")

    df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    raw_cache: Dict[str, pd.DataFrame] = {}
    all_signals: List[pd.DataFrame] = []
    signal_map: Dict[str, pd.DataFrame] = {}

    for row in subset_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        params = parse_params_from_row(row_s, selected_params_dir)
        sig_df = build_signal_table_for_row(row=row_s, params=params, df_cache=df_cache, raw_cache=raw_cache)
        signal_map[str(row_s["symbol"]).upper()] = sig_df
        all_signals.append(sig_df)

    combined_signals = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    combined_signals.to_csv(inputs_dir / "repaired_subset1_signal_timeline.csv", index=False)
    combined_signals.to_csv(inputs_dir / "universe_signal_timeline.csv", index=False)

    exec_args = phase_v.build_exec_args(
        foundation_state=phase_v.FoundationState(
            root=inputs_dir,
            signal_timeline=combined_signals,
            download_manifest=pd.DataFrame(),
            quality=pd.DataFrame(),
            readiness=pd.DataFrame(),
        ),
        seed=int(args.seed),
    )
    contract_validation = phase_v.build_contract_validation(exec_args=exec_args, run_dir=run_dir)
    all_variants = phase_v.sanitize_variants()
    variant_lookup = {str(v["candidate_id"]): dict(v) for v in all_variants}

    results_rows: List[pd.DataFrame] = []
    route_rows: List[pd.DataFrame] = []
    by_symbol_rows: List[Dict[str, Any]] = []
    symbol_meta: Dict[str, Any] = {}

    for row in subset_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        symbol = str(row_s["symbol"]).upper()
        sig_df = signal_map[symbol]
        params = parse_params_from_row(row_s, selected_params_dir)
        symbol_windows = build_window_pool_for_symbol(
            symbol=symbol,
            signal_df=sig_df,
            foundation_state=foundation_state,
            cache_dir=window_cache_dir,
            reuse_cache_dirs=[prior_cache_dir],
        )
        if sig_df.empty:
            by_symbol_rows.append(
                {
                    "symbol": symbol,
                    "classification": "INCONCLUSIVE",
                    "classification_reason": "no_repaired_signals_in_window",
                    "bounded_pass_best_exec_expectancy_net": float("nan"),
                    "reference_exec_expectancy_net": float("nan"),
                    "confirmed_exec_expectancy_net": float("nan"),
                    "delta_expectancy_vs_bounded_pass": float("nan"),
                    "delta_expectancy_vs_repaired_1h": float("nan"),
                    "cvar_delta": float("nan"),
                    "maxdd_delta": float("nan"),
                    "reference_entries_valid": 0,
                    "best_entries_valid": 0,
                    "best_entry_rate": float("nan"),
                    "trade_retention_vs_reference": float("nan"),
                    "best_candidate_id": "",
                    "best_valid_for_ranking": 0,
                    "best_invalid_reason": "no_repaired_signals_in_window",
                    "route_supported": 0,
                    "route_pass_rate": float("nan"),
                    "signals_total": 0,
                    "signals_with_3m_data": 0,
                    "signals_missing_3m_data": 0,
                    "missing_window_rate": float("nan"),
                    "pathology_or_degradation": "no_repaired_signals_after_freeze",
                }
            )
            continue

        bounded_row = bounded_eval_df[bounded_eval_df["symbol"] == symbol]
        if bounded_row.empty:
            raise RuntimeError(f"Missing bounded subset baseline row for {symbol}")
        bounded_row_s = bounded_row.iloc[0]
        chosen_candidate_id = str(bounded_row_s.get("best_candidate_id", "")).strip()
        if chosen_candidate_id not in variant_lookup:
            raise RuntimeError(f"Missing route-confirm candidate config for {symbol}: {chosen_candidate_id}")
        variants = [variant_lookup[chosen_candidate_id]]

        bundle, build_meta = phase_v.build_symbol_bundle(
            symbol=symbol,
            symbol_signals=sig_df,
            symbol_windows=symbol_windows,
            exec_args=exec_args,
            run_dir=run_dir,
        )
        qrow, rrow = quality_from_build_meta(build_meta)
        eval_pack = phase_v.evaluate_symbol(
            symbol=symbol,
            bundle=bundle,
            foundation_quality=qrow,
            foundation_readiness=rrow,
            exec_args=exec_args,
            variants=variants,
        )
        symbol_rows = eval_pack["results_df"].copy()
        results_rows.append(symbol_rows)
        if isinstance(eval_pack["route_df"], pd.DataFrame) and not eval_pack["route_df"].empty:
            route_rows.append(eval_pack["route_df"].copy())

        ref_row = symbol_rows[symbol_rows["candidate_id"].astype(str) == "M0_1H_REFERENCE"].iloc[0]
        best_row = phase_v.choose_best_candidate(symbol_rows)
        base_class, base_reason = phase_v.classify_symbol(best_row=best_row, foundation_quality=qrow, exec_args=exec_args)
        final_class, final_reason = map_exec_classification(
            base_class=base_class,
            base_reason=base_reason,
            best_row=best_row,
            qrow=qrow,
            route_meta=eval_pack["route_meta"],
        )

        if best_row.empty:
            best_candidate_id = ""
            best_valid = 0
            best_exp = float("nan")
            best_entries = 0
            entry_rate = float("nan")
            cvar_ratio = float("nan")
            maxdd_ratio = float("nan")
            invalid_reason = "no_non_reference_variants"
            route_supported = int(eval_pack["route_meta"].get("route_family_supported", 0))
            route_pass = 0
            route_pass_rate = float("nan")
            min_subperiod_delta = float("nan")
            route_count = int(eval_pack["route_meta"].get("route_count", 0))
        else:
            best_candidate_id = str(best_row.get("candidate_id", ""))
            best_valid = int(best_row.get("valid_for_ranking", 0))
            best_exp = float(best_row.get("exec_expectancy_net", np.nan))
            best_entries = int(best_row.get("entries_valid", 0))
            entry_rate = float(best_row.get("entry_rate", np.nan))
            cvar_ratio = float(best_row.get("cvar_improve_ratio", np.nan))
            maxdd_ratio = float(best_row.get("maxdd_improve_ratio", np.nan))
            invalid_reason = str(best_row.get("invalid_reason", ""))
            route_supported = int(best_row.get("route_supported", 0))
            route_pass = int(best_row.get("route_pass", 0))
            route_pass_rate = float(best_row.get("route_pass_rate", np.nan))
            min_subperiod_delta = float(best_row.get("min_subperiod_delta", np.nan))
            route_count = int(best_row.get("route_count", 0))

        ref_exp = float(ref_row.get("exec_expectancy_net", np.nan))
        ref_entries = int(ref_row.get("entries_valid", 0))
        delta = float(best_exp - ref_exp) if np.isfinite(best_exp) and np.isfinite(ref_exp) else float("nan")
        bounded_best_exp = float(pd.to_numeric(pd.Series([bounded_row_s.get("best_exec_expectancy_net", np.nan)]), errors="coerce").iloc[0])
        delta_vs_bounded = float(best_exp - bounded_best_exp) if np.isfinite(best_exp) and np.isfinite(bounded_best_exp) else float("nan")
        cvar_delta = cvar_ratio
        maxdd_delta = maxdd_ratio
        trade_retention = float(best_entries / ref_entries) if ref_entries > 0 else float("nan")

        pathology = []
        if best_valid != 1:
            pathology.append("best_variant_invalid_for_ranking")
        if np.isfinite(delta) and delta < 0.0:
            pathology.append("negative_expectancy_delta")
        if np.isfinite(route_pass_rate) and route_pass_rate < 1.0 and route_supported == 1:
            pathology.append("route_fragility")
        miss_rate = float(pd.to_numeric(pd.Series([qrow.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0])
        if np.isfinite(miss_rate) and miss_rate > 0.0:
            pathology.append("partial_3m_coverage")
        if np.isfinite(trade_retention) and trade_retention < 0.50:
            pathology.append("low_trade_retention")
        pathology_txt = "|".join(pathology) if pathology else "none"

        by_symbol_rows.append(
            {
                "symbol": symbol,
                "classification": final_class,
                "classification_reason": final_reason,
                "bounded_pass_best_exec_expectancy_net": bounded_best_exp,
                "reference_exec_expectancy_net": ref_exp,
                "confirmed_exec_expectancy_net": best_exp,
                "delta_expectancy_vs_bounded_pass": delta_vs_bounded,
                "delta_expectancy_vs_repaired_1h": delta,
                "cvar_delta": cvar_delta,
                "maxdd_delta": maxdd_delta,
                "reference_entries_valid": ref_entries,
                "best_entries_valid": best_entries,
                "best_entry_rate": entry_rate,
                "trade_retention_vs_reference": trade_retention,
                "best_candidate_id": best_candidate_id,
                "best_valid_for_ranking": best_valid,
                "best_invalid_reason": invalid_reason,
                "route_supported": route_supported,
                "route_pass": route_pass,
                "route_pass_rate": route_pass_rate,
                "route_count": route_count,
                "route_family_supported": int(eval_pack["route_meta"].get("route_family_supported", 0)),
                "route_family_blocker": str(eval_pack["route_meta"].get("route_family_blocker", "")),
                "best_min_subperiod_delta": min_subperiod_delta,
                "signals_total": int(build_meta.get("signals_total", 0)),
                "signals_with_3m_data": int(build_meta.get("signals_with_3m_data", 0)),
                "signals_missing_3m_data": int(build_meta.get("signals_missing_3m_data", 0)),
                "missing_window_rate": miss_rate,
                "pathology_or_degradation": pathology_txt,
            }
        )
        symbol_meta[symbol] = {
            "params_source": str(row_s.get("params_source", "")),
            "build_meta": build_meta,
            "foundation_window_rows_used": int(len(symbol_windows)),
            "foundation_window_mode": "local_full_3m" if local_full_3m_path(symbol) is not None else "foundation_cached_windows",
            "route_meta": eval_pack["route_meta"],
            "route_examples_count": int(len(eval_pack.get("route_examples_df", pd.DataFrame()))),
            "route_feasibility_count": int(len(eval_pack.get("route_feas_df", pd.DataFrame()))),
        }

    by_symbol_df = pd.DataFrame(by_symbol_rows)
    by_symbol_df = by_symbol_df.set_index("symbol").loc[SUBSET_SYMBOLS].reset_index()
    by_symbol_df["priority_rank"] = np.arange(1, len(by_symbol_df) + 1)

    summary_df = build_overall_summary(by_symbol_df)
    recommendation, keep_list = final_recommendation(by_symbol_df, summary_df)
    summary_df = pd.concat(
        [
            summary_df,
            pd.DataFrame(
                [
                    {"metric": "recommendation", "value": recommendation},
                    {"metric": "selective_keep_list", "value": ",".join(keep_list)},
                ]
            ),
        ],
        ignore_index=True,
    )

    by_symbol_fp = run_dir / "repaired_subset1_confirm_by_symbol.csv"
    summary_fp = run_dir / "repaired_subset1_confirm_summary.csv"
    report_fp = run_dir / "repaired_subset1_confirm_report.md"

    by_symbol_df.to_csv(by_symbol_fp, index=False)
    summary_df.to_csv(summary_fp, index=False)
    if results_rows:
        pd.concat(results_rows, ignore_index=True).sort_values(["symbol", "candidate_id"]).to_csv(
            run_dir / "repaired_subset1_confirm_all_candidates.csv", index=False
        )
    if route_rows:
        pd.concat(route_rows, ignore_index=True).sort_values(["symbol", "candidate_id", "route_id"]).to_csv(
            run_dir / "repaired_subset1_confirm_route_checks.csv", index=False
        )

    improved_mask = by_symbol_df["classification"].astype(str).isin(["ACTIVE_3M_SURVIVOR", "SHADOW_ONLY"])
    degraded_mask = by_symbol_df["classification"].astype(str) == "NO_EXEC_EDGE"
    inconclusive_mask = by_symbol_df["classification"].astype(str) == "INCONCLUSIVE"
    lines: List[str] = []
    lines.append("# Repaired Universe 3m Execution Strict Confirmation (Subset 1 Survivors)")
    lines.append("")
    lines.append("This is a strict route-enabled repaired-branch confirmation pass. It uses the frozen repaired universe as the only upstream signal/parameter source, and the existing universal foundation only as a local 3m window cache.")
    lines.append("")
    lines.append("## Inputs Used")
    lines.append(f"- Frozen repaired universe dir: `{freeze_dir}`")
    lines.append(f"- Frozen repaired universe table: `{universe_fp}`")
    lines.append(f"- Preflight pathology decision dir: `{pathology_dir}`")
    lines.append(f"- Bounded subset decision dir: `{subset_eval_dir}`")
    lines.append(f"- 3m cache foundation dir: `{foundation_dir}`")
    lines.append(f"- Strict confirm subset: `{', '.join(SUBSET_SYMBOLS)}`")
    lines.append("")
    lines.append("## Per-Symbol Summary")
    lines.append(
        markdown_table(
            by_symbol_df,
            [
                "priority_rank",
                "symbol",
                "bounded_pass_best_exec_expectancy_net",
                "confirmed_exec_expectancy_net",
                "delta_expectancy_vs_bounded_pass",
                "delta_expectancy_vs_repaired_1h",
                "cvar_delta",
                "maxdd_delta",
                "trade_retention_vs_reference",
                "route_pass_rate",
                "classification",
                "pathology_or_degradation",
            ],
            n=len(by_symbol_df),
        )
    )
    lines.append("")
    lines.append("## Overall Outcome")
    lines.append(f"- improved_count: `{int(improved_mask.sum())}`")
    lines.append(f"- degraded_count: `{int(degraded_mask.sum())}`")
    lines.append(f"- neutral_or_inconclusive_count: `{int(inconclusive_mask.sum())}`")
    lines.append(
        f"- aggregate_delta_expectancy_weighted: `{float(summary_df.loc[summary_df['metric'] == 'aggregate_delta_expectancy_weighted', 'value'].iloc[0]):.10f}`"
    )
    lines.append("")
    lines.append("## Proven vs Assumed")
    lines.append("- Proven: the subset was restricted to the three symbols that survived the bounded first-wave pass only.")
    lines.append("- Proven: repaired 1h signals were rebuilt from the frozen repaired per-symbol params, not from the legacy universe or legacy foundation signal timeline.")
    lines.append("- Proven: the existing `phase_v` / Model A entry-only 3m evaluation path was reused with the omitted route-family layer enabled.")
    lines.append("- Assumed: the latest universal foundation is acceptable as a 3m slice cache pool only; it was not used as the upstream signal source.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- Final recommendation: `{recommendation}`")
    lines.append(f"- Active keep list: `{', '.join(keep_list)}`")
    report_fp.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "generated_utc": utc_now().isoformat(),
        "freeze_dir": str(freeze_dir),
        "pathology_dir": str(pathology_dir),
        "foundation_dir": str(foundation_dir),
        "subset_symbols": list(SUBSET_SYMBOLS),
        "contract_validation": contract_validation,
        "reused_code_paths": [
            "scripts/phase_v_multicoin_model_a_audit.py",
            "scripts/phase_a_model_a_audit.py",
            "scripts/repaired_frontier_contamination_audit.py",
            "scripts/repaired_universe_3m_exec_subset1_confirm.py",
        ],
        "recommendation": recommendation,
        "active_keep_list": keep_list,
        "symbol_meta": symbol_meta,
        "artifacts": {
            "repaired_subset1_confirm_by_symbol": str(by_symbol_fp),
            "repaired_subset1_confirm_summary": str(summary_fp),
            "repaired_subset1_confirm_report": str(report_fp),
        },
    }
    json_dump(run_dir / "repaired_subset1_confirm_manifest.json", manifest)


if __name__ == "__main__":
    main()
