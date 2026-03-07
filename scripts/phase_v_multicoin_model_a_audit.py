#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import execution_layer_3m_ict as exec3m  # noqa: E402
from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_nx_exec_family_discovery as nx  # noqa: E402
from scripts import phase_r_route_harness_redesign as phase_r  # noqa: E402
from scripts import phase_u_universal_data_foundation as foundation  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


RUN_PREFIX = "MULTICOIN_MODELA_AUDIT"
FOUNDATION_PREFIX = "UNIVERSAL_DATA_FOUNDATION_"
FOUNDATION_REQUIRED = {
    "universe_signal_timeline.csv",
    "universe_3m_download_manifest.csv",
    "universe_3m_data_quality.csv",
    "universe_symbol_readiness.csv",
}
FORBIDDEN_EXIT_KNOBS = [
    "tp_mult",
    "sl_mult",
    "time_stop_min",
    "break_even_enabled",
    "break_even_trigger_r",
    "break_even_offset_bps",
    "trailing_enabled",
    "trail_start_r",
    "trail_step_bps",
    "partial_take_enabled",
    "partial_take_r",
    "partial_take_pct",
]
ALLOWED_MODEL_A_VARIANT_KEYS = {
    "candidate_id",
    "label",
    "entry_mode",
    "limit_offset_bps",
    "fallback_to_market",
    "fallback_delay_min",
    "max_fill_delay_min",
}


@dataclass
class FoundationState:
    root: Path
    signal_timeline: pd.DataFrame
    download_manifest: pd.DataFrame
    quality: pd.DataFrame
    readiness: pd.DataFrame


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(v, utc=True))
        if isinstance(v, Path):
            return str(v)
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def find_latest_foundation_dir() -> Path:
    cands = sorted(
        [p for p in (PROJECT_ROOT / "reports" / "execution_layer").glob(f"{FOUNDATION_PREFIX}*") if p.is_dir()],
        key=lambda p: p.name,
    )
    for p in reversed(cands):
        names = {f.name for f in p.iterdir() if f.is_file()}
        if FOUNDATION_REQUIRED.issubset(names):
            return p.resolve()
    raise FileNotFoundError("No completed UNIVERSAL_DATA_FOUNDATION run directory found")


def load_foundation_state(run_dir: Path) -> FoundationState:
    run_dir = run_dir.resolve()
    for name in FOUNDATION_REQUIRED:
        fp = run_dir / name
        if not fp.exists():
            raise FileNotFoundError(f"Missing foundation artifact: {fp}")

    signal_timeline = pd.read_csv(run_dir / "universe_signal_timeline.csv")
    signal_timeline["signal_time_utc"] = pd.to_datetime(signal_timeline["signal_time_utc"], utc=True, errors="coerce")
    signal_timeline["entry_reference_time_utc"] = pd.to_datetime(signal_timeline["entry_reference_time_utc"], utc=True, errors="coerce")

    download_manifest = pd.read_csv(run_dir / "universe_3m_download_manifest.csv")
    for c in ["window_start_utc", "window_end_utc"]:
        download_manifest[c] = pd.to_datetime(download_manifest[c], utc=True, errors="coerce")

    quality = pd.read_csv(run_dir / "universe_3m_data_quality.csv")
    readiness = pd.read_csv(run_dir / "universe_symbol_readiness.csv")
    return FoundationState(
        root=run_dir,
        signal_timeline=signal_timeline,
        download_manifest=download_manifest,
        quality=quality,
        readiness=readiness,
    )


def build_exec_args(foundation_state: FoundationState, seed: int) -> argparse.Namespace:
    args = nx.build_exec_args(signals_csv=foundation_state.root / "universe_signal_timeline.csv", seed=int(seed))
    return args


def sanitize_variants() -> List[Dict[str, Any]]:
    variants = modela.build_model_a_variants()
    clean: List[Dict[str, Any]] = []
    for cfg in variants:
        unknown = [k for k in cfg.keys() if k not in ALLOWED_MODEL_A_VARIANT_KEYS]
        if unknown:
            raise RuntimeError(f"Unexpected Model A variant keys: {unknown}")
        bad = [k for k in cfg.keys() if k in FORBIDDEN_EXIT_KNOBS]
        if bad:
            raise RuntimeError(f"Forbidden hybrid exit knobs leaked into Model A variants: {bad}")
        clean.append(dict(cfg))
    return clean


def build_contract_validation(exec_args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    lock = ga_exec._validate_and_lock_frozen_artifacts(args=exec_args, run_dir=run_dir)  # pylint: disable=protected-access
    if int(lock.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("Frozen contract validation failed")

    variants = sanitize_variants()
    variant_keys = sorted({k for cfg in variants for k in cfg.keys()})
    return {
        "generated_utc": utc_now(),
        "freeze_lock": lock,
        "wrapper_uses_1h_signal_owner": 1,
        "wrapper_uses_1h_exit_owner": 1,
        "wrapper_uses_3m_entry_only": 1,
        "hybrid_exit_mutation_enabled": 0,
        "forbidden_exit_knobs_blocked": list(FORBIDDEN_EXIT_KNOBS),
        "allowed_variant_keys": list(sorted(ALLOWED_MODEL_A_VARIANT_KEYS)),
        "observed_variant_keys": variant_keys,
        "variant_family_size": int(len(variants)),
        "variant_ids": [str(cfg["candidate_id"]) for cfg in variants],
    }


def symbol_quality_map(state: FoundationState) -> Dict[str, Dict[str, Any]]:
    return {str(r["symbol"]).upper(): dict(r) for r in state.quality.to_dict("records")}


def symbol_readiness_map(state: FoundationState) -> Dict[str, Dict[str, Any]]:
    return {str(r["symbol"]).upper(): dict(r) for r in state.readiness.to_dict("records")}


def build_signal_window_lookup(signal_time: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.to_datetime(signal_time, utc=True) - pd.Timedelta(hours=float(foundation.PRE_BUFFER_HOURS))
    end = pd.to_datetime(signal_time, utc=True) + pd.Timedelta(hours=float(foundation.POST_BUFFER_HOURS))
    return start, end


def row_or_empty(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    sub = df[df["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()
    if sub.empty:
        return {}
    return dict(sub.iloc[0].to_dict())


def resolve_symbol_constraints(symbol: str, exec_args: argparse.Namespace) -> Dict[str, float]:
    cfg_path = exec3m._resolve_path(exec_args.execution_config)
    all_cfg = ga_exec._load_execution_config(cfg_path)  # pylint: disable=protected-access
    sym_cfg = ga_exec._symbol_exec_config(all_cfg, symbol)  # pylint: disable=protected-access
    cons = sym_cfg.get("tight_constraints") if str(exec_args.mode).lower() == "tight" else sym_cfg.get("constraints")
    if not isinstance(cons, dict):
        cons = {}
    return {
        "min_entry_rate": float(cons.get("min_entry_rate", exec_args.tight_min_entry_rate_default if str(exec_args.mode).lower() == "tight" else exec_args.min_entry_rate_default)),
        "max_taker_share": float(cons.get("max_taker_share", exec_args.tight_max_taker_share_default if str(exec_args.mode).lower() == "tight" else exec_args.max_taker_share_default)),
        "max_fill_delay_min": float(cons.get("max_fill_delay_min", exec_args.tight_max_fill_delay_default if str(exec_args.mode).lower() == "tight" else exec_args.max_fill_delay_default)),
        "min_median_entry_improvement_bps": float(cons.get("min_median_entry_improvement_bps", 0.0)),
    }


def load_window_dataframe(window_row: Dict[str, Any], cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    window_id = str(window_row.get("window_id", ""))
    if not window_id:
        return pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
    if window_id in cache:
        return cache[window_id]

    raw_path = window_row.get("parquet_path", "")
    if pd.isna(raw_path) or not str(raw_path).strip():
        cache[window_id] = pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        return cache[window_id]

    fp = Path(str(raw_path))
    if not fp.exists():
        cache[window_id] = pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        return cache[window_id]

    df = pd.read_parquet(fp)
    df = exec3m._normalize_ohlcv_cols(df)
    cache[window_id] = df
    return df


def arrays_for_signal_slice(slice_df: pd.DataFrame) -> Dict[str, Any]:
    if slice_df.empty:
        return {
            "ts_ns": np.array([], dtype=np.int64),
            "open_np": np.array([], dtype=float),
            "high_np": np.array([], dtype=float),
            "low_np": np.array([], dtype=float),
            "close_np": np.array([], dtype=float),
            "atr_np": np.array([], dtype=float),
            "swing_high": np.array([], dtype=bool),
        }

    x = slice_df.copy().reset_index(drop=True)
    x["ATR14"] = exec3m._compute_atr14(x)
    ts = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
    op = pd.to_numeric(x["Open"], errors="coerce")
    hi = pd.to_numeric(x["High"], errors="coerce")
    lo = pd.to_numeric(x["Low"], errors="coerce")
    cl = pd.to_numeric(x["Close"], errors="coerce")
    atr = pd.to_numeric(x["ATR14"], errors="coerce")
    good = ts.notna() & op.notna() & hi.notna() & lo.notna() & cl.notna() & atr.notna()

    ts_ok = ts[good].tolist()
    ts_ns = np.array([int(t.value) for t in ts_ok], dtype=np.int64)
    open_np = op[good].to_numpy(dtype=float)
    high_np = hi[good].to_numpy(dtype=float)
    low_np = lo[good].to_numpy(dtype=float)
    close_np = cl[good].to_numpy(dtype=float)
    atr_np = atr[good].to_numpy(dtype=float)
    if len(ts_ns) == 0:
        swing_high = np.array([], dtype=bool)
    else:
        _, swing_high = exec3m._detect_swings(low=low_np, high=high_np, k=2)
    return {
        "ts_ns": ts_ns,
        "open_np": open_np,
        "high_np": high_np,
        "low_np": low_np,
        "close_np": close_np,
        "atr_np": atr_np,
        "swing_high": swing_high,
    }


def locate_covering_window(signal_row: pd.Series, windows: List[Dict[str, Any]], start_idx: int) -> Tuple[Optional[Dict[str, Any]], int]:
    sig_start, sig_end = build_signal_window_lookup(pd.to_datetime(signal_row["signal_time_utc"], utc=True))
    idx = int(start_idx)
    while idx < len(windows) and pd.to_datetime(windows[idx]["window_end_utc"], utc=True) < sig_start:
        idx += 1
    if idx < len(windows):
        w = windows[idx]
        if pd.to_datetime(w["window_start_utc"], utc=True) <= sig_start and pd.to_datetime(w["window_end_utc"], utc=True) >= sig_end:
            return w, idx
    return None, idx


def build_symbol_bundle(
    *,
    symbol: str,
    symbol_signals: pd.DataFrame,
    symbol_windows: pd.DataFrame,
    exec_args: argparse.Namespace,
    run_dir: Path,
) -> Tuple[ga_exec.SymbolBundle, Dict[str, Any]]:
    symbol = str(symbol).upper()
    sdf = symbol_signals.copy().sort_values(["signal_time_utc", "signal_id"]).reset_index(drop=True)
    win_df = symbol_windows.copy().sort_values(["window_start_utc", "window_id"]).reset_index(drop=True)
    windows = win_df.to_dict("records")
    window_cache: Dict[str, pd.DataFrame] = {}
    contexts: List[ga_exec.SignalContext] = []

    pointer = 0
    with_data = 0
    partial_signal_slices = 0
    missing_signal_slices = 0
    source_counts: Counter[str] = Counter()

    out_signal_dir = ensure_dir(run_dir / "_signal_inputs")
    signal_input_path = out_signal_dir / f"{symbol}_signals_1h.csv"
    keep_cols = [
        "signal_id",
        "signal_time_utc",
        "strategy_tp_mult",
        "strategy_sl_mult",
        "bucket_1h",
        "params_source",
        "model_source",
    ]
    sdf.loc[:, [c for c in keep_cols if c in sdf.columns]].rename(columns={"signal_time_utc": "signal_time"}).to_csv(signal_input_path, index=False)

    for row in sdf.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        signal_time = pd.to_datetime(row_s["signal_time_utc"], utc=True)
        sig_start, sig_end = build_signal_window_lookup(signal_time)
        cover, pointer = locate_covering_window(row_s, windows, pointer)

        slice_df = pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        slice_status = "missing"
        source_tag = "no_window"
        if cover is not None:
            source_tag = str(cover.get("download_source", ""))
            source_counts[source_tag] += 1
            df_window = load_window_dataframe(cover, window_cache)
            if not df_window.empty:
                slice_df = df_window[(df_window["Timestamp"] >= sig_start) & (df_window["Timestamp"] < sig_end)].reset_index(drop=True)
            ratio = float(pd.to_numeric(pd.Series([cover.get("coverage_ratio", np.nan)]), errors="coerce").fillna(np.nan).iloc[0])
            if not slice_df.empty:
                with_data += 1
                if np.isfinite(ratio) and ratio < 1.0:
                    slice_status = "partial"
                    partial_signal_slices += 1
                else:
                    slice_status = "ready"
            else:
                missing_signal_slices += 1
        else:
            missing_signal_slices += 1

        arr = arrays_for_signal_slice(slice_df)
        ctx = ga_exec.SignalContext(
            symbol=symbol,
            signal_id=str(row_s["signal_id"]),
            signal_time=signal_time,
            signal_ts_ns=int(signal_time.value),
            tp_mult_sig=float(row_s["strategy_tp_mult"]),
            sl_mult_sig=float(row_s["strategy_sl_mult"]),
            quality=float(pd.to_numeric(pd.Series([row_s.get("atr_percentile_1h", np.nan)]), errors="coerce").iloc[0]),
            baseline_entry_time=None,
            baseline_exit_time=None,
            baseline_exit_reason="",
            baseline_filled=0,
            baseline_valid_for_metrics=0,
            baseline_sl_hit=0,
            baseline_tp_hit=0,
            baseline_same_bar_hit=0,
            baseline_invalid_stop_geometry=0,
            baseline_invalid_tp_geometry=0,
            baseline_entry_type="",
            baseline_entry_price=float("nan"),
            baseline_exit_price=float("nan"),
            baseline_fill_liq="",
            baseline_fill_delay_min=float("nan"),
            baseline_mae_pct=float("nan"),
            baseline_mfe_pct=float("nan"),
            baseline_pnl_gross_pct=float("nan"),
            baseline_pnl_net_pct=float("nan"),
            ts_ns=arr["ts_ns"],
            open_np=arr["open_np"],
            high_np=arr["high_np"],
            low_np=arr["low_np"],
            close_np=arr["close_np"],
            atr_np=arr["atr_np"],
            swing_high=arr["swing_high"],
        )
        contexts.append(ctx)

    splits = ga_exec._build_walkforward_splits(n=len(contexts), train_ratio=float(exec_args.train_ratio), n_splits=int(exec_args.wf_splits))  # pylint: disable=protected-access
    bundle = ga_exec.SymbolBundle(
        symbol=symbol,
        signals_csv=signal_input_path,
        contexts=contexts,
        splits=splits,
        constraints=resolve_symbol_constraints(symbol, exec_args),
    )
    build_meta = {
        "signals_total": int(len(contexts)),
        "signals_with_3m_data": int(with_data),
        "signals_partial_3m_data": int(partial_signal_slices),
        "signals_missing_3m_data": int(missing_signal_slices),
        "window_source_counts": dict(sorted(source_counts.items())),
        "window_count": int(len(win_df)),
    }
    return bundle, build_meta


def evaluate_symbol(
    *,
    symbol: str,
    bundle: ga_exec.SymbolBundle,
    foundation_quality: Dict[str, Any],
    foundation_readiness: Dict[str, Any],
    exec_args: argparse.Namespace,
    variants: List[Dict[str, Any]],
) -> Dict[str, Any]:
    one_h = modela.load_1h_market(symbol)
    baseline_full = modela.build_1h_reference_rows(
        bundle=bundle,
        fee=modela.phasec_bt.FeeModel(
            fee_bps_maker=float(exec_args.fee_bps_maker),
            fee_bps_taker=float(exec_args.fee_bps_taker),
            slippage_bps_limit=float(exec_args.slippage_bps_limit),
            slippage_bps_market=float(exec_args.slippage_bps_market),
        ),
        exec_horizon_hours=float(exec_args.exec_horizon_hours),
    )
    ref_eval = modela.evaluate_reference_bundle(baseline_full)
    ref_signal_df = ref_eval["signal_rows_df"].copy()
    ref_roll = ga_exec._rollup_mode(ref_signal_df, "baseline")  # pylint: disable=protected-access

    route_blocker = ""
    try:
        route_bundles, route_examples_df, route_feas_df, route_meta = phase_r.build_support_feasible_route_family(
            base_bundle=bundle,
            args=exec_args,
            coverage_frac=0.60,
        )
        route_meta = dict(route_meta)
        route_meta["route_family_supported"] = 1
    except Exception as exc:
        route_bundles = {}
        route_examples_df = pd.DataFrame()
        route_feas_df = pd.DataFrame()
        route_blocker = f"{type(exc).__name__}:{exc}"
        route_meta = {
            "route_count": 0,
            "route_family_supported": 0,
            "route_family_blocker": route_blocker,
        }
    baseline_routes: Dict[str, pd.DataFrame] = {}
    for rid, rb in route_bundles.items():
        baseline_routes[rid] = modela.build_1h_reference_rows(
            bundle=rb,
            fee=modela.phasec_bt.FeeModel(
                fee_bps_maker=float(exec_args.fee_bps_maker),
                fee_bps_taker=float(exec_args.fee_bps_taker),
                slippage_bps_limit=float(exec_args.slippage_bps_limit),
                slippage_bps_market=float(exec_args.slippage_bps_market),
            ),
            exec_horizon_hours=float(exec_args.exec_horizon_hours),
        )

    results_rows: List[Dict[str, Any]] = [
        {
            "symbol": symbol,
            "bucket_1h": str(foundation_readiness.get("bucket_1h", "")),
            "foundation_integrity_status": str(foundation_quality.get("integrity_status", foundation_readiness.get("integrity_status", ""))),
            "candidate_id": "M0_1H_REFERENCE",
            "label": "M0 pure 1h reference",
            "valid_for_ranking": int(ref_eval["metrics"]["valid_for_ranking"]),
            "invalid_reason": str(ref_eval["metrics"]["invalid_reason"]),
            "exec_expectancy_net": float(ref_eval["metrics"]["overall_exec_expectancy_net"]),
            "delta_expectancy_vs_1h_reference": 0.0,
            "cvar_improve_ratio": 0.0,
            "maxdd_improve_ratio": 0.0,
            "entries_valid": int(ref_eval["metrics"]["overall_entries_valid"]),
            "entry_rate": float(ref_eval["metrics"]["overall_entry_rate"]),
            "taker_share": float(ref_eval["metrics"]["overall_exec_taker_share"]),
            "median_fill_delay_min": float(ref_eval["metrics"]["overall_exec_median_fill_delay_min"]),
            "p95_fill_delay_min": float(ref_eval["metrics"]["overall_exec_p95_fill_delay_min"]),
            "route_supported": int(len(route_bundles) > 0),
            "route_pass": 1,
            "route_pass_rate": 1.0,
            "min_subperiod_delta": 0.0,
            "route_count": int(len(route_bundles)),
            "foundation_missing_window_rate": float(pd.to_numeric(pd.Series([foundation_quality.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0]),
            "foundation_signals_covered": int(pd.to_numeric(pd.Series([foundation_quality.get("signals_covered", 0)]), errors="coerce").fillna(0).iloc[0]),
            "foundation_signals_uncovered": int(pd.to_numeric(pd.Series([foundation_quality.get("signals_uncovered", 0)]), errors="coerce").fillna(0).iloc[0]),
            "reference_cvar_5": float(ref_roll["cvar_5"]),
            "reference_max_drawdown": float(ref_roll["max_drawdown"]),
            "reference_pnl_net_sum": float(ref_roll["pnl_net_sum"]),
            "route_family_blocker": route_blocker,
        }
    ]
    variant_cache: Dict[str, Dict[str, Any]] = {"M0_1H_REFERENCE": ref_eval}
    route_rows: List[Dict[str, Any]] = []

    for cfg in variants:
        ev = modela.evaluate_model_a_variant(
            bundle=bundle,
            baseline_df=baseline_full,
            cfg=cfg,
            one_h=one_h,
            args=exec_args,
        )
        variant_cache[str(cfg["candidate_id"])] = ev
        m = ev["metrics"]

        route_valid_flags: Dict[str, int] = {}
        route_delta: Dict[str, float] = {}
        for rid, rb in route_bundles.items():
            rev = modela.evaluate_model_a_variant(
                bundle=rb,
                baseline_df=baseline_routes[rid],
                cfg=cfg,
                one_h=one_h,
                args=exec_args,
            )
            rm = rev["metrics"]
            route_valid_flags[rid] = int(rm["valid_for_ranking"])
            route_delta[rid] = float(rm["overall_delta_expectancy_exec_minus_baseline"])
            route_rows.append(
                {
                    "symbol": symbol,
                    "candidate_id": str(cfg["candidate_id"]),
                    "route_id": rid,
                    "valid_for_ranking": int(rm["valid_for_ranking"]),
                    "delta_expectancy_vs_1h_reference": float(rm["overall_delta_expectancy_exec_minus_baseline"]),
                    "entries_valid": int(rm["overall_entries_valid"]),
                    "entry_rate": float(rm["overall_entry_rate"]),
                    "taker_share": float(rm["overall_exec_taker_share"]),
                    "min_subperiod_delta": float(rm["min_split_delta"]),
                }
            )

        route_pass = int(
            len(route_bundles) > 0
            and all(int(v) == 1 for v in route_valid_flags.values())
            and all(np.isfinite(v) and v > 0.0 for v in route_delta.values())
        )
        route_pass_rate = (
            float(sum(1 for rid in route_valid_flags if route_valid_flags[rid] == 1 and np.isfinite(route_delta[rid]) and route_delta[rid] > 0.0))
            / float(max(1, len(route_valid_flags)))
        ) if route_valid_flags else float("nan")

        results_rows.append(
            {
                "symbol": symbol,
                "bucket_1h": str(foundation_readiness.get("bucket_1h", "")),
                "foundation_integrity_status": str(foundation_quality.get("integrity_status", foundation_readiness.get("integrity_status", ""))),
                "candidate_id": str(cfg["candidate_id"]),
                "label": str(cfg["label"]),
                "valid_for_ranking": int(m["valid_for_ranking"]),
                "invalid_reason": str(m["invalid_reason"]),
                "exec_expectancy_net": float(m["overall_exec_expectancy_net"]),
                "delta_expectancy_vs_1h_reference": float(m["overall_delta_expectancy_exec_minus_baseline"]),
                "cvar_improve_ratio": float(m["overall_cvar_improve_ratio"]),
                "maxdd_improve_ratio": float(m["overall_maxdd_improve_ratio"]),
                "entries_valid": int(m["overall_entries_valid"]),
                "entry_rate": float(m["overall_entry_rate"]),
                "taker_share": float(m["overall_exec_taker_share"]),
                "median_fill_delay_min": float(m["overall_exec_median_fill_delay_min"]),
                "p95_fill_delay_min": float(m["overall_exec_p95_fill_delay_min"]),
                "route_supported": int(len(route_bundles) > 0),
                "route_pass": int(route_pass),
                "route_pass_rate": float(route_pass_rate),
                "min_subperiod_delta": float(m["min_split_delta"]),
                "route_count": int(len(route_bundles)),
                "foundation_missing_window_rate": float(pd.to_numeric(pd.Series([foundation_quality.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0]),
                "foundation_signals_covered": int(pd.to_numeric(pd.Series([foundation_quality.get("signals_covered", 0)]), errors="coerce").fillna(0).iloc[0]),
                "foundation_signals_uncovered": int(pd.to_numeric(pd.Series([foundation_quality.get("signals_uncovered", 0)]), errors="coerce").fillna(0).iloc[0]),
                "reference_cvar_5": float(ref_roll["cvar_5"]),
                "reference_max_drawdown": float(ref_roll["max_drawdown"]),
                "reference_pnl_net_sum": float(ref_roll["pnl_net_sum"]),
                "route_family_blocker": route_blocker,
            }
        )

    results_df = pd.DataFrame(results_rows).sort_values(
        ["symbol", "candidate_id"],
        ascending=[True, True],
    ).reset_index(drop=True)
    route_df = pd.DataFrame(route_rows).sort_values(["symbol", "candidate_id", "route_id"]).reset_index(drop=True) if route_rows else pd.DataFrame()
    return {
        "results_df": results_df,
        "route_df": route_df,
        "variant_cache": variant_cache,
        "reference_rollup": ref_roll,
        "route_examples_df": route_examples_df,
        "route_feas_df": route_feas_df,
        "route_meta": route_meta,
    }


def choose_best_candidate(rows: pd.DataFrame) -> pd.Series:
    non_ref = rows[rows["candidate_id"].astype(str) != "M0_1H_REFERENCE"].copy()
    if non_ref.empty:
        return pd.Series(dtype=object)
    sort_cols = [
        "valid_for_ranking",
        "delta_expectancy_vs_1h_reference",
        "cvar_improve_ratio",
        "maxdd_improve_ratio",
        "route_pass_rate",
        "entries_valid",
    ]
    non_ref = non_ref.sort_values(sort_cols, ascending=[False, False, False, False, False, False]).reset_index(drop=True)
    return non_ref.iloc[0]


def classify_symbol(best_row: pd.Series, foundation_quality: Dict[str, Any], exec_args: argparse.Namespace) -> Tuple[str, str]:
    ready_windows = int(pd.to_numeric(pd.Series([foundation_quality.get("windows_ready", 0)]), errors="coerce").fillna(0).iloc[0])
    partial_windows = int(pd.to_numeric(pd.Series([foundation_quality.get("windows_partial", 0)]), errors="coerce").fillna(0).iloc[0])
    missing_rate = float(pd.to_numeric(pd.Series([foundation_quality.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0])

    if (ready_windows + partial_windows) <= 0:
        return "DATA_BLOCKED", "no_sliced_3m_windows"
    if np.isfinite(missing_rate) and missing_rate > float(exec_args.hard_max_missing_slice_rate):
        return "DATA_BLOCKED", f"foundation_missing_window_rate>{float(exec_args.hard_max_missing_slice_rate):.4f}"
    if best_row.empty:
        return "MODEL_A_NO_GO", "no_non_reference_variants"

    valid = int(best_row.get("valid_for_ranking", 0)) == 1
    delta = float(pd.to_numeric(pd.Series([best_row.get("delta_expectancy_vs_1h_reference", np.nan)]), errors="coerce").iloc[0])
    cvar = float(pd.to_numeric(pd.Series([best_row.get("cvar_improve_ratio", np.nan)]), errors="coerce").iloc[0])
    maxdd = float(pd.to_numeric(pd.Series([best_row.get("maxdd_improve_ratio", np.nan)]), errors="coerce").iloc[0])
    route_supported = int(best_row.get("route_supported", 0)) == 1
    route_pass = int(best_row.get("route_pass", 0)) == 1
    route_pass_rate = float(pd.to_numeric(pd.Series([best_row.get("route_pass_rate", np.nan)]), errors="coerce").iloc[0])
    min_sub = float(pd.to_numeric(pd.Series([best_row.get("min_subperiod_delta", np.nan)]), errors="coerce").iloc[0])

    route_strong_ok = (not route_supported) or route_pass
    route_weak_ok = (not route_supported) or (np.isfinite(route_pass_rate) and route_pass_rate >= (2.0 / 3.0))

    if valid and np.isfinite(delta) and delta > 0.0 and route_strong_ok and np.isfinite(min_sub) and min_sub >= 0.0 and np.isfinite(cvar) and cvar > 0.0 and np.isfinite(maxdd) and maxdd > 0.0:
        return "MODEL_A_STRONG_GO", "positive_delta_with_full_route_and_risk_improvement"
    if valid and np.isfinite(delta) and delta >= 0.0 and route_weak_ok and np.isfinite(min_sub) and min_sub >= 0.0:
        return "MODEL_A_WEAK_GO", "non_negative_delta_with_partial_route_support"
    return "MODEL_A_NO_GO", "no_robust_entry_only_advantage"


def build_reference_vs_best_row(
    *,
    symbol: str,
    symbol_rows: pd.DataFrame,
    best_row: pd.Series,
    foundation_quality: Dict[str, Any],
    classification: str,
    classification_reason: str,
) -> Dict[str, Any]:
    ref = symbol_rows[symbol_rows["candidate_id"].astype(str) == "M0_1H_REFERENCE"].iloc[0]
    out = {
        "symbol": symbol,
        "bucket_1h": str(ref.get("bucket_1h", "")),
        "foundation_integrity_status": str(ref.get("foundation_integrity_status", "")),
        "classification": classification,
        "classification_reason": classification_reason,
        "reference_exec_expectancy_net": float(ref["exec_expectancy_net"]),
        "reference_entries_valid": int(ref["entries_valid"]),
        "reference_entry_rate": float(ref["entry_rate"]),
        "reference_taker_share": float(ref["taker_share"]),
        "reference_median_fill_delay_min": float(ref["median_fill_delay_min"]),
        "reference_p95_fill_delay_min": float(ref["p95_fill_delay_min"]),
        "reference_cvar_5": float(ref["reference_cvar_5"]),
        "reference_max_drawdown": float(ref["reference_max_drawdown"]),
        "best_candidate_id": "",
        "best_candidate_label": "",
        "best_valid_for_ranking": 0,
        "best_exec_expectancy_net": float("nan"),
        "delta_expectancy_vs_1h_reference": float("nan"),
        "cvar_improve_ratio": float("nan"),
        "maxdd_improve_ratio": float("nan"),
        "best_entries_valid": 0,
        "best_entry_rate": float("nan"),
        "best_taker_share": float("nan"),
        "best_median_fill_delay_min": float("nan"),
        "best_p95_fill_delay_min": float("nan"),
        "best_route_pass": 0,
        "best_route_pass_rate": float("nan"),
        "best_min_subperiod_delta": float("nan"),
        "best_invalid_reason": "",
        "foundation_missing_window_rate": float(pd.to_numeric(pd.Series([foundation_quality.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0]),
        "foundation_signals_covered": int(pd.to_numeric(pd.Series([foundation_quality.get("signals_covered", 0)]), errors="coerce").fillna(0).iloc[0]),
        "foundation_signals_uncovered": int(pd.to_numeric(pd.Series([foundation_quality.get("signals_uncovered", 0)]), errors="coerce").fillna(0).iloc[0]),
    }
    if not best_row.empty:
        out.update(
            {
                "best_candidate_id": str(best_row.get("candidate_id", "")),
                "best_candidate_label": str(best_row.get("label", "")),
                "best_valid_for_ranking": int(best_row.get("valid_for_ranking", 0)),
                "best_exec_expectancy_net": float(best_row.get("exec_expectancy_net", np.nan)),
                "delta_expectancy_vs_1h_reference": float(best_row.get("delta_expectancy_vs_1h_reference", np.nan)),
                "cvar_improve_ratio": float(best_row.get("cvar_improve_ratio", np.nan)),
                "maxdd_improve_ratio": float(best_row.get("maxdd_improve_ratio", np.nan)),
                "best_entries_valid": int(best_row.get("entries_valid", 0)),
                "best_entry_rate": float(best_row.get("entry_rate", np.nan)),
                "best_taker_share": float(best_row.get("taker_share", np.nan)),
                "best_median_fill_delay_min": float(best_row.get("median_fill_delay_min", np.nan)),
                "best_p95_fill_delay_min": float(best_row.get("p95_fill_delay_min", np.nan)),
                "best_route_pass": int(best_row.get("route_pass", 0)),
                "best_route_pass_rate": float(best_row.get("route_pass_rate", np.nan)),
                "best_min_subperiod_delta": float(best_row.get("min_subperiod_delta", np.nan)),
                "best_invalid_reason": str(best_row.get("invalid_reason", "")),
            }
        )
    return out


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
                vals.append(f"{v:.10g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-coin Model A audit using universal sliced 3m data")
    ap.add_argument("--foundation-dir", default="", help="Path to UNIVERSAL_DATA_FOUNDATION_* run dir; defaults to latest completed run.")
    ap.add_argument("--seed", type=int, default=20260228)
    ap.add_argument("--outdir", default="reports/execution_layer")
    args_cli = ap.parse_args()

    foundation_dir = Path(args_cli.foundation_dir).resolve() if str(args_cli.foundation_dir).strip() else find_latest_foundation_dir()
    foundation_state = load_foundation_state(foundation_dir)

    run_root = (PROJECT_ROOT / args_cli.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    exec_args = build_exec_args(foundation_state=foundation_state, seed=int(args_cli.seed))
    contract_validation = build_contract_validation(exec_args=exec_args, run_dir=run_dir)
    variants = sanitize_variants()

    qual_map = symbol_quality_map(foundation_state)
    ready_map = symbol_readiness_map(foundation_state)

    all_results: List[pd.DataFrame] = []
    all_route_rows: List[pd.DataFrame] = []
    best_rows: List[Dict[str, Any]] = []
    class_rows: List[Dict[str, Any]] = []
    invalid_hist: Counter[str] = Counter()
    symbol_meta: Dict[str, Any] = {}

    for symbol in foundation.UNIVERSE:
        sig_df = foundation_state.signal_timeline[foundation_state.signal_timeline["symbol"].astype(str).str.upper() == symbol].copy()
        win_df = foundation_state.download_manifest[foundation_state.download_manifest["symbol"].astype(str).str.upper() == symbol].copy()
        qrow = qual_map.get(symbol, {})
        rrow = ready_map.get(symbol, {})

        bundle, build_meta = build_symbol_bundle(
            symbol=symbol,
            symbol_signals=sig_df,
            symbol_windows=win_df,
            exec_args=exec_args,
            run_dir=run_dir,
        )
        eval_pack = evaluate_symbol(
            symbol=symbol,
            bundle=bundle,
            foundation_quality=qrow,
            foundation_readiness=rrow,
            exec_args=exec_args,
            variants=variants,
        )
        symbol_rows = eval_pack["results_df"].copy()
        all_results.append(symbol_rows)
        if isinstance(eval_pack["route_df"], pd.DataFrame) and not eval_pack["route_df"].empty:
            all_route_rows.append(eval_pack["route_df"].copy())

        for row in symbol_rows[symbol_rows["candidate_id"].astype(str) != "M0_1H_REFERENCE"].itertuples(index=False):
            for part in [x for x in str(getattr(row, "invalid_reason", "")).split("|") if x]:
                invalid_hist[part] += 1

        best_row = choose_best_candidate(symbol_rows)
        classification, classification_reason = classify_symbol(best_row=best_row, foundation_quality=qrow, exec_args=exec_args)
        best_rows.append(
            build_reference_vs_best_row(
                symbol=symbol,
                symbol_rows=symbol_rows,
                best_row=best_row,
                foundation_quality=qrow,
                classification=classification,
                classification_reason=classification_reason,
            )
        )
        class_rows.append(
            {
                "symbol": symbol,
                "bucket_1h": str(rrow.get("bucket_1h", "")),
                "classification": classification,
                "classification_reason": classification_reason,
                "best_candidate_id": str(best_row.get("candidate_id", "")) if not best_row.empty else "",
                "best_valid_for_ranking": int(best_row.get("valid_for_ranking", 0)) if not best_row.empty else 0,
                "best_delta_expectancy_vs_1h_reference": float(best_row.get("delta_expectancy_vs_1h_reference", np.nan)) if not best_row.empty else float("nan"),
                "best_cvar_improve_ratio": float(best_row.get("cvar_improve_ratio", np.nan)) if not best_row.empty else float("nan"),
                "best_maxdd_improve_ratio": float(best_row.get("maxdd_improve_ratio", np.nan)) if not best_row.empty else float("nan"),
                "best_entry_rate": float(best_row.get("entry_rate", np.nan)) if not best_row.empty else float("nan"),
                "best_entries_valid": int(best_row.get("entries_valid", 0)) if not best_row.empty else 0,
                "best_taker_share": float(best_row.get("taker_share", np.nan)) if not best_row.empty else float("nan"),
                "best_route_pass": int(best_row.get("route_pass", 0)) if not best_row.empty else 0,
                "best_route_pass_rate": float(best_row.get("route_pass_rate", np.nan)) if not best_row.empty else float("nan"),
                "best_min_subperiod_delta": float(best_row.get("min_subperiod_delta", np.nan)) if not best_row.empty else float("nan"),
                "foundation_integrity_status": str(qrow.get("integrity_status", rrow.get("integrity_status", ""))),
                "foundation_missing_window_rate": float(pd.to_numeric(pd.Series([qrow.get("missing_window_rate", np.nan)]), errors="coerce").iloc[0]),
                "foundation_signals_covered": int(pd.to_numeric(pd.Series([qrow.get("signals_covered", 0)]), errors="coerce").fillna(0).iloc[0]),
                "foundation_signals_uncovered": int(pd.to_numeric(pd.Series([qrow.get("signals_uncovered", 0)]), errors="coerce").fillna(0).iloc[0]),
            }
        )
        symbol_meta[symbol] = {
            "bundle_build": build_meta,
            "route_meta": eval_pack["route_meta"],
            "route_examples_count": int(len(eval_pack["route_examples_df"])),
            "route_feasibility_count": int(len(eval_pack["route_feas_df"])),
        }

    results_df = pd.concat(all_results, ignore_index=True).sort_values(["symbol", "candidate_id"]).reset_index(drop=True)
    route_df = pd.concat(all_route_rows, ignore_index=True).sort_values(["symbol", "candidate_id", "route_id"]).reset_index(drop=True) if all_route_rows else pd.DataFrame()
    best_df = pd.DataFrame(best_rows).sort_values("symbol").reset_index(drop=True)
    class_df = pd.DataFrame(class_rows).sort_values("symbol").reset_index(drop=True)

    results_df.to_csv(run_dir / "multicoin_modelA_results.csv", index=False)
    best_df.to_csv(run_dir / "multicoin_modelA_reference_vs_best.csv", index=False)
    class_df.to_csv(run_dir / "multicoin_modelA_coin_classification.csv", index=False)
    json_dump(run_dir / "multicoin_modelA_invalid_reason_histogram.json", dict(sorted(invalid_hist.items(), key=lambda kv: (-kv[1], kv[0]))))
    if not route_df.empty:
        route_df.to_csv(run_dir / "multicoin_modelA_route_checks.csv", index=False)

    top_improvers = best_df.sort_values(["delta_expectancy_vs_1h_reference", "cvar_improve_ratio", "maxdd_improve_ratio"], ascending=[False, False, False]).reset_index(drop=True)
    dd_leaders = best_df.sort_values(["maxdd_improve_ratio", "delta_expectancy_vs_1h_reference"], ascending=[False, False]).reset_index(drop=True)
    blocked = class_df[class_df["classification"].astype(str) == "DATA_BLOCKED"].copy()
    no_go = class_df[class_df["classification"].astype(str) == "MODEL_A_NO_GO"].copy()
    not_tradable = class_df[class_df["classification"].astype(str).isin(["DATA_BLOCKED", "MODEL_A_NO_GO"])].copy()

    report_lines = [
        "# Multi-Coin Model A Audit",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Foundation source: `{foundation_state.root}`",
        f"- Universe size: `{len(foundation.UNIVERSE)}`",
        f"- Variants tested per coin: `{1 + len(variants)}`",
        "- Contract parity:",
        "  - 1h signal owner: `src/bot087/optim/ga.py` via frozen universal timeline",
        "  - 1h exit owner: `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference` semantics via `phase_a_model_a_audit.simulate_frozen_1h_exit`",
        "  - 3m entry executor: `phase_a_model_a_audit.simulate_entry_only_fill`",
        f"  - forbidden hybrid exit knobs blocked: `{FORBIDDEN_EXIT_KNOBS}`",
        "",
        "## Per-Coin Classification",
        "",
        markdown_table(
            class_df,
            [
                "symbol",
                "classification",
                "best_candidate_id",
                "best_delta_expectancy_vs_1h_reference",
                "best_cvar_improve_ratio",
                "best_maxdd_improve_ratio",
                "best_valid_for_ranking",
                "foundation_integrity_status",
            ],
            n=20,
        ),
        "",
        "## Top Improving Coins",
        "",
        markdown_table(
            top_improvers,
            [
                "symbol",
                "classification",
                "best_candidate_id",
                "delta_expectancy_vs_1h_reference",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "best_valid_for_ranking",
            ],
            n=10,
        ),
        "",
        "## Largest DD Improvements",
        "",
        markdown_table(
            dd_leaders,
            [
                "symbol",
                "classification",
                "best_candidate_id",
                "maxdd_improve_ratio",
                "delta_expectancy_vs_1h_reference",
                "best_valid_for_ranking",
            ],
            n=10,
        ),
        "",
        "## Data-Blocked Coins",
        "",
        markdown_table(
            blocked,
            [
                "symbol",
                "classification_reason",
                "foundation_missing_window_rate",
                "foundation_signals_covered",
                "foundation_signals_uncovered",
            ],
            n=20,
        ),
        "",
        "## Not Tradable After Execution",
        "",
        markdown_table(
            not_tradable,
            [
                "symbol",
                "classification",
                "classification_reason",
                "best_candidate_id",
                "best_valid_for_ranking",
                "best_delta_expectancy_vs_1h_reference",
            ],
            n=20,
        ),
        "",
    ]
    write_text(run_dir / "multicoin_modelA_report.md", "\n".join(report_lines))

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "foundation_dir": str(foundation_state.root),
        "git_snapshot": modela.git_snapshot(),
        "contract_validation": contract_validation,
        "symbol_universe": [{"symbol": s, "bucket_1h": foundation.USER_BUCKET[s]} for s in foundation.UNIVERSE],
        "variant_ids": [str(v["candidate_id"]) for v in variants],
        "source_scripts": {
            "model_a_wrapper": "scripts/phase_a_model_a_audit.py",
            "route_harness": "scripts/phase_r_route_harness_redesign.py",
            "foundation_builder": "scripts/phase_u_universal_data_foundation.py",
            "execution_metrics": "src/execution/ga_exec_3m_opt.py",
            "runner": "scripts/phase_v_multicoin_model_a_audit.py",
        },
        "symbol_meta": symbol_meta,
        "classification_counts": {
            "MODEL_A_STRONG_GO": int((class_df["classification"] == "MODEL_A_STRONG_GO").sum()),
            "MODEL_A_WEAK_GO": int((class_df["classification"] == "MODEL_A_WEAK_GO").sum()),
            "MODEL_A_NO_GO": int((class_df["classification"] == "MODEL_A_NO_GO").sum()),
            "DATA_BLOCKED": int((class_df["classification"] == "DATA_BLOCKED").sum()),
        },
        "outputs": {
            "multicoin_modelA_results_csv": str(run_dir / "multicoin_modelA_results.csv"),
            "multicoin_modelA_reference_vs_best_csv": str(run_dir / "multicoin_modelA_reference_vs_best.csv"),
            "multicoin_modelA_coin_classification_csv": str(run_dir / "multicoin_modelA_coin_classification.csv"),
            "multicoin_modelA_invalid_reason_histogram_json": str(run_dir / "multicoin_modelA_invalid_reason_histogram.json"),
            "multicoin_modelA_report_md": str(run_dir / "multicoin_modelA_report.md"),
            "multicoin_modelA_run_manifest_json": str(run_dir / "multicoin_modelA_run_manifest.json"),
        },
    }
    json_dump(run_dir / "multicoin_modelA_run_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
