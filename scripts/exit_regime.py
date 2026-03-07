#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import execution_layer_3m_ict as exec3m  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


@dataclass(frozen=True)
class ExitCfg:
    cfg_id: int
    tp_mult: float
    sl_mult: float
    time_stop_min: int
    break_even_enabled: int
    break_even_trigger_r: float
    break_even_offset_bps: float
    partial_take_enabled: int
    partial_take_r: float
    partial_take_pct: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _parse_symbols(raw_symbols: str, single_symbol: str) -> List[str]:
    if str(single_symbol).strip():
        return [str(single_symbol).strip().upper()]
    return [x.strip().upper() for x in str(raw_symbols).split(",") if x.strip()]


def _metrics_definition_text() -> str:
    lines = [
        "# Metrics Definition",
        "",
        "- Baseline entry definition (Phase A): market at next 3m open after 1h signal timestamp (UTC).",
        "- Phase D compares `regime_exit` vs `global_exit` using identical entry model (fixed market entry).",
        "- Phase E compares `exec_entry` vs `baseline_entry` using identical exit policy chosen from Phase D/C.",
        "- `expectancy_net`: mean per-signal net pnl (non-filled/invalid contribute 0 in per-signal vector).",
        "- `cvar_5`: mean of worst 5% per-signal pnl outcomes.",
        "- `max_drawdown`: peak-to-trough drawdown of cumulative per-signal pnl curve.",
        "- All decisions are on walkforward TEST-only slices.",
    ]
    return "\n".join(lines).strip() + "\n"


def _fee_model_payload(args: argparse.Namespace, source_script: str) -> Dict[str, Any]:
    return {
        "generated_utc": _utc_now_iso(),
        "source_script": source_script,
        "baseline_definition": "market at next 3m open after 1h signal",
        "fee_bps_maker": float(args.fee_bps_maker),
        "fee_bps_taker": float(args.fee_bps_taker),
        "slippage_bps_limit": float(args.slippage_bps_limit),
        "slippage_bps_market": float(args.slippage_bps_market),
    }


def _latest_phase_c_dir(base_dir: Path) -> Path:
    cands = sorted([p for p in base_dir.glob("EXIT_SWEEP_*") if p.is_dir() and (p / "exit_sweep_topk.csv").exists()], key=lambda p: p.name)
    if not cands:
        raise FileNotFoundError(f"No EXIT_SWEEP_* directories under {base_dir}")
    return cands[-1].resolve()


def _pick_symbol_params(best_csv: Path, symbol: str) -> Tuple[str, Path]:
    df = pd.read_csv(best_csv)
    if df.empty:
        raise RuntimeError(f"Empty best csv: {best_csv}")
    rows = df.copy()
    if "side" in rows.columns:
        rows = rows[rows["side"].astype(str).str.lower() == "long"].copy()
    if "pass" in rows.columns:
        rows = rows[rows["pass"].map(exec3m._as_bool)].copy()
    if rows.empty:
        rows = df.copy()
    rows["symbol_u"] = rows.get("symbol", "").astype(str).str.upper().str.strip()
    rows = rows[rows["symbol_u"] == str(symbol).upper().strip()].copy()
    if rows.empty:
        raise RuntimeError(f"Symbol {symbol} missing in {best_csv}")
    rows["score"] = pd.to_numeric(rows.get("score"), errors="coerce").fillna(-1e18)
    rows = rows.sort_values("score", ascending=False).reset_index(drop=True)
    params_raw = str(rows.iloc[0].get("params_file", "")).strip()
    if not params_raw:
        raise RuntimeError(f"params_file missing for {symbol} in {best_csv}")
    params_file = _resolve_path(params_raw)
    if not params_file.exists():
        raise RuntimeError(f"params_file not found for {symbol}: {params_file}")
    return str(symbol).upper().strip(), params_file


def _ensure_signal_csv(symbol: str, args: argparse.Namespace) -> Path:
    out = _resolve_path(args.signals_dir) / f"{symbol.upper()}_signals_1h.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and int(args.refresh_signals) == 0:
        return out

    scan_dir = _resolve_path(args.scan_dir) if str(args.scan_dir).strip() else exec3m._find_latest_scan_dir()
    best_csv = (scan_dir / "best_by_symbol.csv").resolve()
    _, params_file = _pick_symbol_params(best_csv=best_csv, symbol=symbol)

    payload = json.loads(params_file.read_text(encoding="utf-8"))
    p = ga_long._norm_params(exec3m._unwrap_params(payload))

    exec_cfg = ga_exec._load_execution_config(_resolve_path(args.execution_config))
    sym_cfg = ga_exec._symbol_exec_config(exec_cfg, symbol)
    gate_cfg = {
        "use_vol_regime_gate": int(sym_cfg.get("use_vol_regime_gate", args.use_vol_regime_gate)),
        "vol_regime_max_percentile": float(sym_cfg.get("vol_regime_max_percentile", args.vol_regime_max_percentile)),
        "vol_regime_lookback_bars": int(sym_cfg.get("vol_regime_lookback_bars", args.vol_regime_lookback_bars)),
        "use_trend_gate": int(sym_cfg.get("use_trend_gate", args.use_trend_gate)),
        "trend_fast_col": str(sym_cfg.get("trend_fast_col", args.trend_fast_col)),
        "trend_slow_col": str(sym_cfg.get("trend_slow_col", args.trend_slow_col)),
        "trend_min_slope": float(sym_cfg.get("trend_min_slope", args.trend_min_slope)),
        "stop_distance_min_pct": float(sym_cfg.get("stop_distance_min_pct", args.stop_distance_min_pct)),
    }

    df_1h = exec3m._load_symbol_df(symbol.upper(), tf="1h")
    signals = exec3m._build_1h_signals(
        df_1h=df_1h,
        p=p,
        max_signals=int(args.max_signals),
        order=str(args.signal_order),
        gate_cfg=gate_cfg,
    )
    rows: List[Dict[str, Any]] = []
    for s in signals:
        rows.append(
            {
                "signal_id": s.signal_id,
                "signal_time": str(s.signal_time),
                "direction": "long",
                "cycle": int(s.cycle),
                "baseline_entry_ref": "next_3m_open",
                "strategy_tp_mult": float(s.tp_mult),
                "strategy_sl_mult": float(s.sl_mult),
                "tp_mult": float(s.tp_mult),
                "sl_mult": float(s.sl_mult),
                "signal_open_1h": float(s.signal_open_1h),
                "strategy_tp_on_1h_open": float(s.signal_open_1h * s.tp_mult),
                "strategy_sl_on_1h_open": float(s.signal_open_1h * s.sl_mult),
                "stop_distance_pct": float(s.stop_distance_pct),
                "atr_1h": float(s.atr_1h),
                "atr_percentile_1h": float(s.atr_percentile_1h),
                "trend_up_1h": int(s.trend_up_1h),
            }
        )
    if not rows:
        raise RuntimeError(f"No 1h signals generated for {symbol}")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _build_eval_args(args: argparse.Namespace, symbols: Sequence[str]) -> argparse.Namespace:
    ev = ga_exec.build_arg_parser().parse_args([])
    ev.symbol = ""
    ev.symbols = ",".join([s.upper() for s in symbols])
    ev.rank = int(args.rank)
    ev.scan_dir = str(args.scan_dir)
    ev.best_csv = str(args.best_csv)
    ev.signals_dir = str(args.signals_dir)
    ev.signals_csv = ""
    ev.signal_order = str(args.signal_order)
    ev.max_signals = int(args.max_signals)
    ev.walkforward = True
    ev.train_ratio = float(args.train_ratio)
    ev.wf_splits = int(args.wf_splits)
    ev.mode = str(args.mode)
    ev.force_no_skip = 1  # no trade skipping for this stage
    ev.timeframe = str(args.timeframe)
    ev.pre_buffer_hours = float(args.pre_buffer_hours)
    ev.exec_horizon_hours = float(args.exec_horizon_hours)
    ev.cache_dir = str(args.cache_dir)
    ev.max_fetch_retries = int(args.max_fetch_retries)
    ev.retry_base_sleep = float(args.retry_base_sleep)
    ev.retry_max_sleep = float(args.retry_max_sleep)
    ev.fetch_pause_sec = float(args.fetch_pause_sec)
    ev.execution_config = str(args.execution_config)
    ev.fee_bps_maker = float(args.fee_bps_maker)
    ev.fee_bps_taker = float(args.fee_bps_taker)
    ev.slippage_bps_limit = float(args.slippage_bps_limit)
    ev.slippage_bps_market = float(args.slippage_bps_market)

    # Keep participation viable for evaluation; this phase focuses on exit logic.
    ev.hard_min_trades_overall = 0
    ev.hard_min_trade_frac_overall = 0.0
    ev.hard_min_trades_symbol = 0
    ev.hard_min_trade_frac_symbol = 0.0
    ev.hard_min_entry_rate_symbol = 0.0
    ev.hard_min_entry_rate_overall = 0.0
    ev.hard_max_missing_slice_rate = float(args.max_missing_slice_rate)
    ev.hard_max_taker_share = 1.0
    ev.hard_max_median_fill_delay_min = 1e9
    ev.hard_max_p95_fill_delay_min = 1e9
    return ev


def _entry_template_market() -> Dict[str, Any]:
    return {
        "entry_mode": "market",
        "limit_offset_bps": 0.0,
        "max_fill_delay_min": 0,
        "fallback_to_market": 1,
        "fallback_delay_min": 0,
        "max_taker_share": 1.0,
        "micro_vol_filter": 0,
        "vol_threshold": 6.0,
        "spread_guard_bps": 1e6,
        "killzone_filter": 0,
        "mss_displacement_gate": 0,
        "min_entry_improvement_bps_gate": 0.0,
        "skip_if_vol_gate": 0,
        "use_signal_quality_gate": 0,
        "min_signal_quality_gate": 0.0,
        "cooldown_min": 0,
        "trailing_enabled": 0,
        "trail_start_r": 2.0,
        "trail_step_bps": 50.0,
    }


def _entry_template_exec_tight(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "entry_mode": str(args.phase_e_exec_entry_mode),
        "limit_offset_bps": float(args.phase_e_limit_offset_bps),
        "max_fill_delay_min": int(args.phase_e_max_fill_delay_min),
        "fallback_to_market": 1,
        "fallback_delay_min": int(args.phase_e_fallback_delay_min),
        "max_taker_share": 1.0,
        "micro_vol_filter": int(args.phase_e_micro_vol_filter),
        "vol_threshold": float(args.phase_e_vol_threshold),
        "spread_guard_bps": float(args.phase_e_spread_guard_bps),
        "killzone_filter": int(args.phase_e_killzone_filter),
        "mss_displacement_gate": 0,
        "min_entry_improvement_bps_gate": 0.0,
        "skip_if_vol_gate": 0,
        "use_signal_quality_gate": 0,
        "min_signal_quality_gate": 0.0,
        "cooldown_min": 0,
        "trailing_enabled": 0,
        "trail_start_r": 2.0,
        "trail_step_bps": 50.0,
    }


def _cfg_to_genome(
    cfg: ExitCfg,
    mode: str,
    entry_template: Dict[str, Any],
) -> Dict[str, Any]:
    g = dict(entry_template)
    g.update(
        {
            "tp_mult": float(cfg.tp_mult),
            "sl_mult": float(cfg.sl_mult),
            "time_stop_min": int(cfg.time_stop_min),
            "break_even_enabled": int(cfg.break_even_enabled),
            "break_even_trigger_r": float(cfg.break_even_trigger_r),
            "break_even_offset_bps": float(cfg.break_even_offset_bps),
            "partial_take_enabled": int(cfg.partial_take_enabled),
            "partial_take_r": float(cfg.partial_take_r),
            "partial_take_pct": float(cfg.partial_take_pct),
        }
    )
    return ga_exec._repair_genome(g, mode=str(mode))


def _phase_c_candidates(phase_c_dir: Path, topk: int) -> Tuple[ExitCfg, Dict[int, ExitCfg]]:
    top_fp = phase_c_dir / "exit_sweep_topk.csv"
    if not top_fp.exists():
        raise FileNotFoundError(f"Missing {top_fp}")
    df = pd.read_csv(top_fp)
    if df.empty:
        raise RuntimeError(f"Empty {top_fp}")
    rows = df.head(max(1, int(topk))).copy()
    cfgs: Dict[int, ExitCfg] = {}
    for _, r in rows.iterrows():
        cfg = ExitCfg(
            cfg_id=int(r.get("cfg_id", len(cfgs) + 1)),
            tp_mult=float(r["tp_mult"]),
            sl_mult=float(r["sl_mult"]),
            time_stop_min=int(float(r["time_stop_min"])),
            break_even_enabled=int(float(r["break_even_enabled"])),
            break_even_trigger_r=float(r["break_even_trigger_r"]),
            break_even_offset_bps=float(r["break_even_offset_bps"]),
            partial_take_enabled=int(float(r["partial_take_enabled"])),
            partial_take_r=float(r["partial_take_r"]),
            partial_take_pct=float(r["partial_take_pct"]),
        )
        cfgs[int(cfg.cfg_id)] = cfg
    # First row in topk is Phase C best global by construction.
    best_id = int(rows.iloc[0]["cfg_id"])
    if best_id not in cfgs:
        cfgs[best_id] = ExitCfg(
            cfg_id=best_id,
            tp_mult=float(rows.iloc[0]["tp_mult"]),
            sl_mult=float(rows.iloc[0]["sl_mult"]),
            time_stop_min=int(float(rows.iloc[0]["time_stop_min"])),
            break_even_enabled=int(float(rows.iloc[0]["break_even_enabled"])),
            break_even_trigger_r=float(rows.iloc[0]["break_even_trigger_r"]),
            break_even_offset_bps=float(rows.iloc[0]["break_even_offset_bps"]),
            partial_take_enabled=int(float(rows.iloc[0]["partial_take_enabled"])),
            partial_take_r=float(rows.iloc[0]["partial_take_r"]),
            partial_take_pct=float(rows.iloc[0]["partial_take_pct"]),
        )
    return cfgs[best_id], cfgs


def _signal_features_from_csv(signal_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(signal_csv)
    if "signal_id" not in df.columns:
        df["signal_id"] = [f"sig_{i:05d}" for i in range(1, len(df) + 1)]
    if "signal_time" in df.columns:
        df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True, errors="coerce")
    atr_col = ""
    if "atr_percentile_1h" in df.columns:
        atr_col = "atr_percentile_1h"
    elif "atr_1h" in df.columns:
        atr_col = "atr_1h"
    df["atr_regime_feature"] = pd.to_numeric(df[atr_col], errors="coerce") if atr_col else np.nan
    if "trend_up_1h" in df.columns:
        df["trend_regime_feature"] = pd.to_numeric(df["trend_up_1h"], errors="coerce")
    else:
        df["trend_regime_feature"] = np.nan
    return df[["signal_id", "signal_time", "atr_regime_feature", "trend_regime_feature"]].copy()


def _atr_bucket(val: float, q1: float, q2: float) -> str:
    if not np.isfinite(val):
        return "atr_unknown"
    if not np.isfinite(q1) or not np.isfinite(q2):
        return "atr_unknown"
    if val <= q1:
        return "atr_low"
    if val <= q2:
        return "atr_med"
    return "atr_high"


def _bucket_key(atr_bucket: str, trend_val: float, include_trend: bool) -> str:
    if not include_trend or not np.isfinite(trend_val):
        return atr_bucket
    return f"{atr_bucket}|trend_{int(float(trend_val) >= 0.5)}"


def _simulate_rows(
    ctxs: Sequence[ga_exec.SignalContext],
    genome: Dict[str, Any],
    eval_cfg: Dict[str, Any],
) -> pd.DataFrame:
    out: List[Dict[str, Any]] = []
    last_entry_time: Optional[pd.Timestamp] = None
    for ctx in ctxs:
        row = ga_exec._simulate_candidate_signal(
            ctx=ctx,
            genome=genome,
            eval_cfg=eval_cfg,
            last_entry_time=last_entry_time,
        )
        if int(row.get("exec_filled", 0)) == 1:
            et = pd.to_datetime(row.get("exec_entry_time"), utc=True, errors="coerce")
            if pd.notna(et):
                last_entry_time = et
        out.append(row)
    return pd.DataFrame(out)


def _mode_roll(df_mode: pd.DataFrame) -> Dict[str, Any]:
    if df_mode.empty:
        return {"baseline": {}, "exec": {}}
    return ga_exec._aggregate_rows(df_mode)


def _weighted_avg(vals: Iterable[float], weights: Iterable[float]) -> float:
    x = np.asarray(list(vals), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0.0)
    if not np.any(m):
        return float("nan")
    return float(np.sum(x[m] * w[m]) / np.sum(w[m]))


def _stability_pass(
    exec_min: float,
    exec_med: float,
    exec_std: float,
    base_min: float,
    base_med: float,
    base_std: float,
) -> Tuple[int, int, int, int]:
    p1 = int(np.isfinite(exec_min) and np.isfinite(base_min) and exec_min >= (base_min - 2e-4))
    p2 = int(np.isfinite(exec_med) and np.isfinite(base_med) and exec_med >= (base_med + 5e-5))
    lim = max(2.5 * float(base_std), 0.0015) if np.isfinite(base_std) else 0.0015
    p3 = int(np.isfinite(exec_std) and exec_std <= lim)
    return p1, p2, p3, int(p1 and p2 and p3)


def _ensure_required_signals(symbols: Sequence[str], args: argparse.Namespace) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for s in symbols:
        out[s] = _ensure_signal_csv(s, args)
    return out


def run_phase_d(args: argparse.Namespace) -> Tuple[Path, Dict[str, Any]]:
    symbols = _parse_symbols(args.symbols, args.symbol)
    if not symbols:
        raise RuntimeError("No symbols provided")
    signal_map = _ensure_required_signals(symbols, args)

    phase_c_dir = _resolve_path(args.phase_c_dir) if str(args.phase_c_dir).strip() else _latest_phase_c_dir(_resolve_path(args.outdir))
    best_global_cfg, candidate_cfgs = _phase_c_candidates(phase_c_dir=phase_c_dir, topk=int(args.regime_topk))

    ev_args = _build_eval_args(args, symbols)
    bundles, load_meta = ga_exec._prepare_bundles(ev_args)

    feature_map: Dict[str, Dict[str, Dict[str, float]]] = {}
    for b in bundles:
        fdf = _signal_features_from_csv(signal_map[b.symbol])
        feature_map[b.symbol] = {}
        for _, r in fdf.iterrows():
            feature_map[b.symbol][str(r["signal_id"])] = {
                "atr": float(pd.to_numeric(pd.Series([r["atr_regime_feature"]]), errors="coerce").iloc[0]),
                "trend": float(pd.to_numeric(pd.Series([r["trend_regime_feature"]]), errors="coerce").iloc[0]),
            }

    eval_cfg = {
        "exec_horizon_hours": float(ev_args.exec_horizon_hours),
        "fee_bps_maker": float(ev_args.fee_bps_maker),
        "fee_bps_taker": float(ev_args.fee_bps_taker),
        "slippage_bps_limit": float(ev_args.slippage_bps_limit),
        "slippage_bps_market": float(ev_args.slippage_bps_market),
        "force_no_skip": 1,
    }

    include_trend = int(args.regime_use_trend) == 1
    entry_market = _entry_template_market()

    # Split-level regime mapping learned on TRAIN.
    bucket_stats_rows: List[Dict[str, Any]] = []
    split_bucket_cfg: Dict[Tuple[str, int, str], int] = {}
    test_bucket_counts: Dict[str, int] = {}
    test_assignments: Dict[Tuple[str, int, str], str] = {}

    for b in bundles:
        fdict = feature_map.get(b.symbol, {})
        for sp in b.splits:
            split_id = int(sp["split_id"])
            tr0 = int(sp["train_start"])
            tr1 = int(sp["train_end"])
            te0 = int(sp["test_start"])
            te1 = int(sp["test_end"])
            train_ctx = b.contexts[tr0:tr1]
            test_ctx = b.contexts[te0:te1]

            train_atr = [fdict.get(c.signal_id, {}).get("atr", np.nan) for c in train_ctx]
            train_atr_np = np.asarray(train_atr, dtype=float)
            train_atr_np = train_atr_np[np.isfinite(train_atr_np)]
            if train_atr_np.size >= 3:
                q1 = float(np.quantile(train_atr_np, 1.0 / 3.0))
                q2 = float(np.quantile(train_atr_np, 2.0 / 3.0))
            else:
                q1 = float("nan")
                q2 = float("nan")

            train_bucket_label: Dict[str, str] = {}
            for c in train_ctx:
                ft = fdict.get(c.signal_id, {})
                bkey = _bucket_key(
                    _atr_bucket(float(ft.get("atr", np.nan)), q1=q1, q2=q2),
                    trend_val=float(ft.get("trend", np.nan)),
                    include_trend=include_trend,
                )
                train_bucket_label[c.signal_id] = bkey

            test_bucket_label: Dict[str, str] = {}
            for c in test_ctx:
                ft = fdict.get(c.signal_id, {})
                bkey = _bucket_key(
                    _atr_bucket(float(ft.get("atr", np.nan)), q1=q1, q2=q2),
                    trend_val=float(ft.get("trend", np.nan)),
                    include_trend=include_trend,
                )
                test_bucket_label[c.signal_id] = bkey
                test_assignments[(b.symbol, split_id, c.signal_id)] = bkey
                # "trades overall" support count: only valid baseline opportunities in TEST.
                if int(c.baseline_filled) == 1 and int(c.baseline_valid_for_metrics) == 1:
                    test_bucket_counts[bkey] = int(test_bucket_counts.get(bkey, 0) + 1)

            train_buckets = sorted(set(train_bucket_label.values()))
            for bkey in train_buckets:
                bucket_ctx = [c for c in train_ctx if train_bucket_label.get(c.signal_id) == bkey]
                train_signals = int(len(bucket_ctx))
                chosen_cfg_id = int(best_global_cfg.cfg_id)
                chosen_exp = -1e18
                if train_signals > 0:
                    for cfg_id, cfg in candidate_cfgs.items():
                        genome = _cfg_to_genome(cfg=cfg, mode=ev_args.mode, entry_template=entry_market)
                        df_mode = _simulate_rows(bucket_ctx, genome=genome, eval_cfg=eval_cfg)
                        roll = _mode_roll(df_mode)
                        exp = float(roll.get("exec", {}).get("mean_expectancy_net", np.nan))
                        if np.isfinite(exp) and exp > chosen_exp:
                            chosen_exp = exp
                            chosen_cfg_id = int(cfg_id)
                split_bucket_cfg[(b.symbol, split_id, bkey)] = int(chosen_cfg_id)
                bucket_stats_rows.append(
                    {
                        "symbol": b.symbol,
                        "split_id": int(split_id),
                        "bucket": bkey,
                        "train_signals": int(train_signals),
                        "test_signals": int(sum(1 for c in test_ctx if test_bucket_label.get(c.signal_id) == bkey)),
                        "train_q1": q1,
                        "train_q2": q2,
                        "include_trend": int(include_trend),
                        "selected_cfg_id_train": int(chosen_cfg_id),
                        "selected_cfg_is_global": int(chosen_cfg_id == int(best_global_cfg.cfg_id)),
                    }
                )

    low_support_buckets = {k for k, v in test_bucket_counts.items() if int(v) < int(args.min_bucket_support)}

    # Evaluate global vs regime on TEST.
    split_rows: List[Dict[str, Any]] = []
    symbol_rows: List[Dict[str, Any]] = []
    global_all: List[pd.DataFrame] = []
    regime_all: List[pd.DataFrame] = []

    for b in bundles:
        symbol_global_parts: List[pd.DataFrame] = []
        symbol_regime_parts: List[pd.DataFrame] = []
        for sp in b.splits:
            split_id = int(sp["split_id"])
            te0 = int(sp["test_start"])
            te1 = int(sp["test_end"])
            test_ctx = b.contexts[te0:te1]

            g_parts: List[Dict[str, Any]] = []
            r_parts: List[Dict[str, Any]] = []
            for c in test_ctx:
                bkey = test_assignments.get((b.symbol, split_id, c.signal_id), "atr_unknown")
                regime_cfg_id = int(split_bucket_cfg.get((b.symbol, split_id, bkey), int(best_global_cfg.cfg_id)))
                use_global_fallback = int(bkey in low_support_buckets)
                if use_global_fallback == 1:
                    regime_cfg_id = int(best_global_cfg.cfg_id)
                cfg_g = best_global_cfg
                cfg_r = candidate_cfgs[regime_cfg_id]
                gen_g = _cfg_to_genome(cfg=cfg_g, mode=ev_args.mode, entry_template=entry_market)
                gen_r = _cfg_to_genome(cfg=cfg_r, mode=ev_args.mode, entry_template=entry_market)
                row_g = ga_exec._simulate_candidate_signal(ctx=c, genome=gen_g, eval_cfg=eval_cfg, last_entry_time=None)
                row_r = ga_exec._simulate_candidate_signal(ctx=c, genome=gen_r, eval_cfg=eval_cfg, last_entry_time=None)
                row_g["bucket"] = bkey
                row_r["bucket"] = bkey
                row_g["split_id"] = split_id
                row_r["split_id"] = split_id
                row_g["selected_cfg_id"] = int(cfg_g.cfg_id)
                row_r["selected_cfg_id"] = int(regime_cfg_id)
                row_r["used_global_fallback"] = int(use_global_fallback)
                g_parts.append(row_g)
                r_parts.append(row_r)

            df_g = pd.DataFrame(g_parts)
            df_r = pd.DataFrame(r_parts)
            symbol_global_parts.append(df_g)
            symbol_regime_parts.append(df_r)

            rg = _mode_roll(df_g)
            rr = _mode_roll(df_r)
            gexec = rg["exec"]
            rexec = rr["exec"]
            split_rows.append(
                {
                    "symbol": b.symbol,
                    "split_id": split_id,
                    "signals_total": int(rexec.get("signals_total", 0)),
                    "global_expectancy_net": float(gexec.get("mean_expectancy_net", np.nan)),
                    "regime_expectancy_net": float(rexec.get("mean_expectancy_net", np.nan)),
                    "delta_expectancy_regime_minus_global": float(rexec.get("mean_expectancy_net", np.nan) - gexec.get("mean_expectancy_net", np.nan)),
                    "global_cvar_5": float(gexec.get("cvar_5", np.nan)),
                    "regime_cvar_5": float(rexec.get("cvar_5", np.nan)),
                    "delta_cvar5_regime_minus_global": float(rexec.get("cvar_5", np.nan) - gexec.get("cvar_5", np.nan)),
                    "global_max_drawdown": float(gexec.get("max_drawdown", np.nan)),
                    "regime_max_drawdown": float(rexec.get("max_drawdown", np.nan)),
                    "delta_maxdd_regime_minus_global": float(rexec.get("max_drawdown", np.nan) - gexec.get("max_drawdown", np.nan)),
                    "global_entry_rate": float(gexec.get("entry_rate", np.nan)),
                    "regime_entry_rate": float(rexec.get("entry_rate", np.nan)),
                }
            )

        df_symbol_g = pd.concat(symbol_global_parts, ignore_index=True) if symbol_global_parts else pd.DataFrame()
        df_symbol_r = pd.concat(symbol_regime_parts, ignore_index=True) if symbol_regime_parts else pd.DataFrame()
        global_all.append(df_symbol_g)
        regime_all.append(df_symbol_r)

        rg = _mode_roll(df_symbol_g)
        rr = _mode_roll(df_symbol_r)
        gexec = rg["exec"]
        rexec = rr["exec"]
        symbol_rows.append(
            {
                "symbol": b.symbol,
                "signals_total": int(rexec.get("signals_total", 0)),
                "global_expectancy_net": float(gexec.get("mean_expectancy_net", np.nan)),
                "regime_expectancy_net": float(rexec.get("mean_expectancy_net", np.nan)),
                "delta_expectancy_regime_minus_global": float(rexec.get("mean_expectancy_net", np.nan) - gexec.get("mean_expectancy_net", np.nan)),
                "global_pnl_net_sum": float(gexec.get("pnl_net_sum", np.nan)),
                "regime_pnl_net_sum": float(rexec.get("pnl_net_sum", np.nan)),
                "global_cvar_5": float(gexec.get("cvar_5", np.nan)),
                "regime_cvar_5": float(rexec.get("cvar_5", np.nan)),
                "delta_cvar5_regime_minus_global": float(rexec.get("cvar_5", np.nan) - gexec.get("cvar_5", np.nan)),
                "global_max_drawdown": float(gexec.get("max_drawdown", np.nan)),
                "regime_max_drawdown": float(rexec.get("max_drawdown", np.nan)),
                "delta_maxdd_regime_minus_global": float(rexec.get("max_drawdown", np.nan) - gexec.get("max_drawdown", np.nan)),
                "global_entry_rate": float(gexec.get("entry_rate", np.nan)),
                "regime_entry_rate": float(rexec.get("entry_rate", np.nan)),
                "global_taker_share": float(gexec.get("taker_share", np.nan)),
                "regime_taker_share": float(rexec.get("taker_share", np.nan)),
                "global_median_fill_delay_min": float(gexec.get("median_fill_delay_min", np.nan)),
                "regime_median_fill_delay_min": float(rexec.get("median_fill_delay_min", np.nan)),
                "global_p95_fill_delay_min": float(gexec.get("p95_fill_delay_min", np.nan)),
                "regime_p95_fill_delay_min": float(rexec.get("p95_fill_delay_min", np.nan)),
                "global_missing_slice_rate": float(pd.to_numeric(df_symbol_g.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_symbol_g.empty else float("nan"),
                "regime_missing_slice_rate": float(pd.to_numeric(df_symbol_r.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_symbol_r.empty else float("nan"),
            }
        )

    df_global_all = pd.concat(global_all, ignore_index=True) if global_all else pd.DataFrame()
    df_regime_all = pd.concat(regime_all, ignore_index=True) if regime_all else pd.DataFrame()
    ov_g = _mode_roll(df_global_all)["exec"]
    ov_r = _mode_roll(df_regime_all)["exec"]
    split_df = pd.DataFrame(split_rows).sort_values(["symbol", "split_id"]).reset_index(drop=True)
    sym_df = pd.DataFrame(symbol_rows).sort_values("symbol").reset_index(drop=True)

    g_split = pd.to_numeric(split_df.get("global_expectancy_net", np.nan), errors="coerce")
    r_split = pd.to_numeric(split_df.get("regime_expectancy_net", np.nan), errors="coerce")
    s1, s2, s3, s_pass = _stability_pass(
        exec_min=float(r_split.min()) if not r_split.empty else float("nan"),
        exec_med=float(r_split.median()) if not r_split.empty else float("nan"),
        exec_std=float(r_split.std(ddof=0)) if not r_split.empty else float("nan"),
        base_min=float(g_split.min()) if not g_split.empty else float("nan"),
        base_med=float(g_split.median()) if not g_split.empty else float("nan"),
        base_std=float(g_split.std(ddof=0)) if not g_split.empty else float("nan"),
    )

    delta_exp = float(ov_r.get("mean_expectancy_net", np.nan) - ov_g.get("mean_expectancy_net", np.nan))
    delta_cvar = float(ov_r.get("cvar_5", np.nan) - ov_g.get("cvar_5", np.nan))
    delta_maxdd = float(ov_r.get("max_drawdown", np.nan) - ov_g.get("max_drawdown", np.nan))
    pass_exp = int(np.isfinite(delta_exp) and delta_exp >= float(args.phase_d_expectancy_min))
    pass_cvar = int(np.isfinite(delta_cvar) and delta_cvar >= -float(args.phase_d_cvar_worse_tol))
    pass_maxdd = int(np.isfinite(delta_maxdd) and delta_maxdd >= -float(args.phase_d_maxdd_worse_tol))
    pass_part = int(np.isfinite(float(ov_r.get("entry_rate", np.nan))) and float(ov_r.get("entry_rate", np.nan)) > 0.0)
    pass_data = int(
        np.isfinite(float(pd.to_numeric(df_regime_all.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean() if not df_regime_all.empty else np.nan))
        and float(pd.to_numeric(df_regime_all.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean() if not df_regime_all.empty else np.nan) <= float(args.max_missing_slice_rate)
    )
    phase_d_pass = int(pass_exp and pass_cvar and pass_maxdd and s_pass and pass_part and pass_data)

    overall_df = pd.DataFrame(
        [
            {
                "scope": "overall",
                "symbols": int(len(sym_df)),
                "signals_total": int(ov_r.get("signals_total", 0)),
                "global_expectancy_net": float(ov_g.get("mean_expectancy_net", np.nan)),
                "regime_expectancy_net": float(ov_r.get("mean_expectancy_net", np.nan)),
                "delta_expectancy_regime_minus_global": delta_exp,
                "global_pnl_net_sum": float(ov_g.get("pnl_net_sum", np.nan)),
                "regime_pnl_net_sum": float(ov_r.get("pnl_net_sum", np.nan)),
                "global_cvar_5": float(ov_g.get("cvar_5", np.nan)),
                "regime_cvar_5": float(ov_r.get("cvar_5", np.nan)),
                "delta_cvar5_regime_minus_global": delta_cvar,
                "global_max_drawdown": float(ov_g.get("max_drawdown", np.nan)),
                "regime_max_drawdown": float(ov_r.get("max_drawdown", np.nan)),
                "delta_maxdd_regime_minus_global": delta_maxdd,
                "global_entry_rate": float(ov_g.get("entry_rate", np.nan)),
                "regime_entry_rate": float(ov_r.get("entry_rate", np.nan)),
                "global_taker_share": float(ov_g.get("taker_share", np.nan)),
                "regime_taker_share": float(ov_r.get("taker_share", np.nan)),
                "global_median_fill_delay_min": float(ov_g.get("median_fill_delay_min", np.nan)),
                "regime_median_fill_delay_min": float(ov_r.get("median_fill_delay_min", np.nan)),
                "global_p95_fill_delay_min": float(ov_g.get("p95_fill_delay_min", np.nan)),
                "regime_p95_fill_delay_min": float(ov_r.get("p95_fill_delay_min", np.nan)),
                "global_missing_slice_rate": float(pd.to_numeric(df_global_all.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_global_all.empty else float("nan"),
                "regime_missing_slice_rate": float(pd.to_numeric(df_regime_all.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_regime_all.empty else float("nan"),
                "stability_min_pass": int(s1),
                "stability_median_pass": int(s2),
                "stability_std_pass": int(s3),
                "stability_pass": int(s_pass),
                "pass_expectancy": int(pass_exp),
                "pass_cvar_not_worse": int(pass_cvar),
                "pass_maxdd_not_worse": int(pass_maxdd),
                "pass_participation": int(pass_part),
                "pass_data_quality": int(pass_data),
                "phase_d_pass": int(phase_d_pass),
            }
        ]
    )

    # Enrich bucket stats with support/fallback annotations.
    bs_df = pd.DataFrame(bucket_stats_rows)
    if not bs_df.empty:
        bs_df["test_bucket_trades_overall"] = bs_df["bucket"].map(lambda b: int(test_bucket_counts.get(str(b), 0)))
        bs_df["low_confidence_bucket"] = bs_df["bucket"].map(lambda b: int(str(b) in low_support_buckets))
        bs_df["used_global_fallback_test"] = bs_df["low_confidence_bucket"].astype(int)
        bs_df["used_cfg_id_test"] = np.where(
            bs_df["used_global_fallback_test"] == 1,
            int(best_global_cfg.cfg_id),
            bs_df["selected_cfg_id_train"],
        ).astype(int)
    else:
        bs_df = pd.DataFrame(
            columns=[
                "symbol",
                "split_id",
                "bucket",
                "train_signals",
                "test_signals",
                "train_q1",
                "train_q2",
                "include_trend",
                "selected_cfg_id_train",
                "selected_cfg_is_global",
                "test_bucket_trades_overall",
                "low_confidence_bucket",
                "used_global_fallback_test",
                "used_cfg_id_test",
            ]
        )

    out_root = _resolve_path(args.outdir) / f"EXIT_REGIME_{_utc_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)
    bs_csv = out_root / "regime_bucket_stats.csv"
    split_csv = out_root / "walkforward_results_by_split.csv"
    sym_csv = out_root / "risk_rollup_by_symbol.csv"
    ov_csv = out_root / "risk_rollup_overall.csv"
    bs_df.to_csv(bs_csv, index=False)
    split_df.to_csv(split_csv, index=False)
    sym_df.to_csv(sym_csv, index=False)
    overall_df.to_csv(ov_csv, index=False)

    decision_lines: List[str] = []
    decision_lines.append("# Exit Regime Decision")
    decision_lines.append("")
    decision_lines.append(f"- Generated UTC: {_utc_now_iso()}")
    decision_lines.append(f"- Symbols: `{','.join(symbols)}`")
    decision_lines.append(f"- Phase C source: `{phase_c_dir}`")
    decision_lines.append(f"- Global reference cfg_id: {int(best_global_cfg.cfg_id)}")
    decision_lines.append(f"- Candidate configs used: {len(candidate_cfgs)}")
    decision_lines.append(f"- Low-support threshold (TEST trades): {int(args.min_bucket_support)}")
    decision_lines.append(f"- Low-support buckets fallback count: {int(len(low_support_buckets))}")
    decision_lines.append("")
    decision_lines.append("## Pass Criteria")
    decision_lines.append("")
    decision_lines.append(f"- expectancy improvement >= {float(args.phase_d_expectancy_min):.5f}: {int(pass_exp)} ({delta_exp:.6f})")
    decision_lines.append(f"- delta_maxdd >= -{float(args.phase_d_maxdd_worse_tol):.2f}: {int(pass_maxdd)} ({delta_maxdd:.6f})")
    decision_lines.append(f"- delta_cvar5 >= -{float(args.phase_d_cvar_worse_tol):.5f}: {int(pass_cvar)} ({delta_cvar:.6f})")
    decision_lines.append(f"- stability pass: {int(s_pass)}")
    decision_lines.append(f"- participation valid: {int(pass_part)}")
    decision_lines.append(f"- data quality valid: {int(pass_data)}")
    decision_lines.append("")
    decision_lines.append(f"- Decision: **{'PASS' if phase_d_pass == 1 else 'FAIL'}**")
    (out_root / "decision.md").write_text("\n".join(decision_lines).strip() + "\n", encoding="utf-8")

    selected_policy = {
        "generated_utc": _utc_now_iso(),
        "phase_d_pass": int(phase_d_pass),
        "policy_type_for_phase_e": "regime" if int(phase_d_pass) == 1 else "global",
        "global_cfg_id": int(best_global_cfg.cfg_id),
        "global_cfg": asdict(best_global_cfg),
        "candidate_cfgs": {str(k): asdict(v) for k, v in candidate_cfgs.items()},
        "low_support_buckets": sorted(list(low_support_buckets)),
        "split_bucket_cfg": {
            f"{k[0]}|{k[1]}|{k[2]}": int(v) for k, v in split_bucket_cfg.items()
        },
    }
    policy_fp = out_root / "selected_exit_policy.json"
    policy_fp.write_text(json.dumps(selected_policy, indent=2), encoding="utf-8")

    (out_root / "metrics_definition.md").write_text(_metrics_definition_text(), encoding="utf-8")
    (out_root / "fee_model.json").write_text(json.dumps(_fee_model_payload(args, "scripts/exit_regime.py"), indent=2), encoding="utf-8")

    repro_lines = [
        "# Repro",
        "",
        "```bash",
        "cd /root/analysis/0.87",
        ".venv/bin/python scripts/exit_regime.py \\",
        f"  --symbols {','.join(symbols)} \\",
        f"  --phase-c-dir {phase_c_dir}",
        "```",
    ]
    (out_root / "repro.md").write_text("\n".join(repro_lines).strip() + "\n", encoding="utf-8")

    phase_lines = [
        "Phase: D (Regime-Aware Exits)",
        f"Timestamp UTC: {_utc_now_iso()}",
        f"Status: {'PASS' if int(phase_d_pass) == 1 else 'FAIL'}",
        f"Phase C dir: {phase_c_dir}",
        f"Delta expectancy (regime-global): {delta_exp:.6f}",
        f"Delta cvar5 (regime-global): {delta_cvar:.6f}",
        f"Delta maxdd (regime-global): {delta_maxdd:.6f}",
        f"Low-support bucket fallback count: {int(len(low_support_buckets))}",
        f"Artifacts: {bs_csv.name}, {split_csv.name}, {sym_csv.name}, {ov_csv.name}, decision.md, selected_exit_policy.json",
    ]
    (out_root / "phase_result.md").write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")

    snap = out_root / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    ecfg = _resolve_path(args.execution_config)
    if ecfg.exists():
        shutil.copy2(ecfg, snap / "execution_configs.yaml")
    if (phase_c_dir / "exit_sweep_topk.csv").exists():
        shutil.copy2(phase_c_dir / "exit_sweep_topk.csv", snap / "phase_c_exit_sweep_topk.csv")
    for sym, fp in signal_map.items():
        if fp.exists():
            shutil.copy2(fp, snap / fp.name)
    os.system(f"git -C {PROJECT_ROOT} status --short > {out_root / 'git_status.txt'}")

    print(str(out_root))
    print(str(bs_csv))
    print(str(split_csv))
    print(str(sym_csv))
    print(str(ov_csv))
    print(str(out_root / "decision.md"))

    return out_root, {
        "phase_d_pass": int(phase_d_pass),
        "policy_file": str(policy_fp),
        "phase_c_dir": str(phase_c_dir),
        "symbols": symbols,
        "eval_args": ev_args,
        "bundles": bundles,
        "split_assignments": test_assignments,
        "split_bucket_cfg": split_bucket_cfg,
        "low_support_buckets": low_support_buckets,
        "candidate_cfgs": candidate_cfgs,
        "best_global_cfg": best_global_cfg,
        "signal_map": signal_map,
        "feature_map": feature_map,
    }


def _combine_baseline_exec_rows(row_b: Dict[str, Any], row_e: Dict[str, Any], split_id: int) -> Dict[str, Any]:
    return {
        "symbol": row_b.get("symbol", ""),
        "signal_id": row_b.get("signal_id", ""),
        "signal_time": row_b.get("signal_time", ""),
        "split_id": int(split_id),
        "baseline_filled": int(row_b.get("exec_filled", 0)),
        "baseline_valid_for_metrics": int(row_b.get("exec_valid_for_metrics", 0)),
        "baseline_sl_hit": int(row_b.get("exec_sl_hit", 0)),
        "baseline_tp_hit": int(row_b.get("exec_tp_hit", 0)),
        "baseline_pnl_net_pct": float(row_b.get("exec_pnl_net_pct", np.nan)),
        "baseline_pnl_gross_pct": float(row_b.get("exec_pnl_gross_pct", np.nan)),
        "baseline_fill_liquidity_type": str(row_b.get("exec_fill_liquidity_type", "")),
        "baseline_fill_delay_min": float(row_b.get("exec_fill_delay_min", np.nan)),
        "baseline_mae_pct": float(row_b.get("exec_mae_pct", np.nan)),
        "baseline_mfe_pct": float(row_b.get("exec_mfe_pct", np.nan)),
        "baseline_invalid_stop_geometry": int(row_b.get("exec_invalid_stop_geometry", 0)),
        "baseline_invalid_tp_geometry": int(row_b.get("exec_invalid_tp_geometry", 0)),
        "baseline_same_bar_hit": int(row_b.get("exec_same_bar_hit", 0)),
        "baseline_exit_reason": str(row_b.get("exec_exit_reason", "")),
        "baseline_missing_slice_flag": int(row_b.get("missing_slice_flag", 0)),
        "exec_filled": int(row_e.get("exec_filled", 0)),
        "exec_valid_for_metrics": int(row_e.get("exec_valid_for_metrics", 0)),
        "exec_sl_hit": int(row_e.get("exec_sl_hit", 0)),
        "exec_tp_hit": int(row_e.get("exec_tp_hit", 0)),
        "exec_pnl_net_pct": float(row_e.get("exec_pnl_net_pct", np.nan)),
        "exec_pnl_gross_pct": float(row_e.get("exec_pnl_gross_pct", np.nan)),
        "exec_fill_liquidity_type": str(row_e.get("exec_fill_liquidity_type", "")),
        "exec_fill_delay_min": float(row_e.get("exec_fill_delay_min", np.nan)),
        "exec_mae_pct": float(row_e.get("exec_mae_pct", np.nan)),
        "exec_mfe_pct": float(row_e.get("exec_mfe_pct", np.nan)),
        "exec_invalid_stop_geometry": int(row_e.get("exec_invalid_stop_geometry", 0)),
        "exec_invalid_tp_geometry": int(row_e.get("exec_invalid_tp_geometry", 0)),
        "exec_same_bar_hit": int(row_e.get("exec_same_bar_hit", 0)),
        "exec_exit_reason": str(row_e.get("exec_exit_reason", "")),
        "exec_missing_slice_flag": int(row_e.get("missing_slice_flag", 0)),
        "entry_improvement_bps": float(row_e.get("entry_improvement_bps", np.nan)),
    }


def run_phase_e(args: argparse.Namespace, phase_d_ctx: Dict[str, Any]) -> Path:
    symbols: List[str] = list(phase_d_ctx["symbols"])
    bundles: List[ga_exec.SymbolBundle] = list(phase_d_ctx["bundles"])
    split_assignments: Dict[Tuple[str, int, str], str] = dict(phase_d_ctx["split_assignments"])
    split_bucket_cfg: Dict[Tuple[str, int, str], int] = dict(phase_d_ctx["split_bucket_cfg"])
    low_support_buckets: set[str] = set(phase_d_ctx["low_support_buckets"])
    candidate_cfgs: Dict[int, ExitCfg] = dict(phase_d_ctx["candidate_cfgs"])
    best_global_cfg: ExitCfg = phase_d_ctx["best_global_cfg"]
    use_regime = int(phase_d_ctx["phase_d_pass"]) == 1

    ev_args: argparse.Namespace = phase_d_ctx["eval_args"]
    eval_cfg = {
        "exec_horizon_hours": float(ev_args.exec_horizon_hours),
        "fee_bps_maker": float(ev_args.fee_bps_maker),
        "fee_bps_taker": float(ev_args.fee_bps_taker),
        "slippage_bps_limit": float(ev_args.slippage_bps_limit),
        "slippage_bps_market": float(ev_args.slippage_bps_market),
        "force_no_skip": 1,
    }

    base_entry = _entry_template_market()
    exec_entry = _entry_template_exec_tight(args)

    per_split_rows: List[Dict[str, Any]] = []
    combined_rows_all: List[pd.DataFrame] = []
    symbol_rows: List[Dict[str, Any]] = []

    for b in bundles:
        symbol_parts: List[pd.DataFrame] = []
        for sp in b.splits:
            split_id = int(sp["split_id"])
            te0 = int(sp["test_start"])
            te1 = int(sp["test_end"])
            test_ctx = b.contexts[te0:te1]
            rows_split: List[Dict[str, Any]] = []
            for c in test_ctx:
                bkey = split_assignments.get((b.symbol, split_id, c.signal_id), "atr_unknown")
                if use_regime:
                    cfg_id = int(split_bucket_cfg.get((b.symbol, split_id, bkey), int(best_global_cfg.cfg_id)))
                    if bkey in low_support_buckets:
                        cfg_id = int(best_global_cfg.cfg_id)
                else:
                    cfg_id = int(best_global_cfg.cfg_id)
                cfg = candidate_cfgs.get(cfg_id, best_global_cfg)

                g_base = _cfg_to_genome(cfg=cfg, mode=ev_args.mode, entry_template=base_entry)
                g_exec = _cfg_to_genome(cfg=cfg, mode=ev_args.mode, entry_template=exec_entry)
                row_b = ga_exec._simulate_candidate_signal(ctx=c, genome=g_base, eval_cfg=eval_cfg, last_entry_time=None)
                row_e = ga_exec._simulate_candidate_signal(ctx=c, genome=g_exec, eval_cfg=eval_cfg, last_entry_time=None)
                row = _combine_baseline_exec_rows(row_b=row_b, row_e=row_e, split_id=split_id)
                row["bucket"] = bkey
                row["exit_cfg_id"] = int(cfg.cfg_id)
                row["exit_policy_used"] = "regime" if use_regime else "global"
                rows_split.append(row)

            df_split = pd.DataFrame(rows_split)
            symbol_parts.append(df_split)
            roll = ga_exec._aggregate_rows(df_split) if not df_split.empty else {"baseline": {}, "exec": {}}
            bmode = roll["baseline"]
            emode = roll["exec"]
            per_split_rows.append(
                {
                    "symbol": b.symbol,
                    "split_id": split_id,
                    "signals_total": int(emode.get("signals_total", 0)),
                    "baseline_expectancy_net": float(bmode.get("mean_expectancy_net", np.nan)),
                    "exec_expectancy_net": float(emode.get("mean_expectancy_net", np.nan)),
                    "delta_expectancy_exec_minus_baseline": float(emode.get("mean_expectancy_net", np.nan) - bmode.get("mean_expectancy_net", np.nan)),
                    "baseline_cvar_5": float(bmode.get("cvar_5", np.nan)),
                    "exec_cvar_5": float(emode.get("cvar_5", np.nan)),
                    "delta_cvar5_exec_minus_baseline": float(emode.get("cvar_5", np.nan) - bmode.get("cvar_5", np.nan)),
                    "baseline_max_drawdown": float(bmode.get("max_drawdown", np.nan)),
                    "exec_max_drawdown": float(emode.get("max_drawdown", np.nan)),
                    "delta_maxdd_exec_minus_baseline": float(emode.get("max_drawdown", np.nan) - bmode.get("max_drawdown", np.nan)),
                    "baseline_entry_rate": float(bmode.get("entry_rate", np.nan)),
                    "exec_entry_rate": float(emode.get("entry_rate", np.nan)),
                    "exec_taker_share": float(emode.get("taker_share", np.nan)),
                    "exec_median_fill_delay_min": float(emode.get("median_fill_delay_min", np.nan)),
                    "exec_p95_fill_delay_min": float(emode.get("p95_fill_delay_min", np.nan)),
                }
            )

        df_symbol = pd.concat(symbol_parts, ignore_index=True) if symbol_parts else pd.DataFrame()
        combined_rows_all.append(df_symbol)
        roll_sym = ga_exec._aggregate_rows(df_symbol) if not df_symbol.empty else {"baseline": {}, "exec": {}}
        bmode = roll_sym["baseline"]
        emode = roll_sym["exec"]
        symbol_rows.append(
            {
                "symbol": b.symbol,
                "signals_total": int(emode.get("signals_total", 0)),
                "baseline_expectancy_net": float(bmode.get("mean_expectancy_net", np.nan)),
                "exec_expectancy_net": float(emode.get("mean_expectancy_net", np.nan)),
                "delta_expectancy_exec_minus_baseline": float(emode.get("mean_expectancy_net", np.nan) - bmode.get("mean_expectancy_net", np.nan)),
                "baseline_pnl_net_sum": float(bmode.get("pnl_net_sum", np.nan)),
                "exec_pnl_net_sum": float(emode.get("pnl_net_sum", np.nan)),
                "baseline_cvar_5": float(bmode.get("cvar_5", np.nan)),
                "exec_cvar_5": float(emode.get("cvar_5", np.nan)),
                "delta_cvar5_exec_minus_baseline": float(emode.get("cvar_5", np.nan) - bmode.get("cvar_5", np.nan)),
                "baseline_max_drawdown": float(bmode.get("max_drawdown", np.nan)),
                "exec_max_drawdown": float(emode.get("max_drawdown", np.nan)),
                "delta_maxdd_exec_minus_baseline": float(emode.get("max_drawdown", np.nan) - bmode.get("max_drawdown", np.nan)),
                "baseline_entry_rate": float(bmode.get("entry_rate", np.nan)),
                "exec_entry_rate": float(emode.get("entry_rate", np.nan)),
                "baseline_taker_share": float(bmode.get("taker_share", np.nan)),
                "exec_taker_share": float(emode.get("taker_share", np.nan)),
                "baseline_median_fill_delay_min": float(bmode.get("median_fill_delay_min", np.nan)),
                "exec_median_fill_delay_min": float(emode.get("median_fill_delay_min", np.nan)),
                "exec_p95_fill_delay_min": float(emode.get("p95_fill_delay_min", np.nan)),
                "exec_median_entry_improvement_bps": float(emode.get("median_entry_improvement_bps", np.nan)),
                "baseline_sl_hit_rate_valid": float(bmode.get("SL_hit_rate_valid", np.nan)),
                "exec_sl_hit_rate_valid": float(emode.get("SL_hit_rate_valid", np.nan)),
                "baseline_missing_slice_rate": float(pd.to_numeric(df_symbol.get("baseline_missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_symbol.empty else float("nan"),
                "exec_missing_slice_rate": float(pd.to_numeric(df_symbol.get("exec_missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_symbol.empty else float("nan"),
            }
        )

    df_all = pd.concat(combined_rows_all, ignore_index=True) if combined_rows_all else pd.DataFrame()
    roll_all = ga_exec._aggregate_rows(df_all) if not df_all.empty else {"baseline": {}, "exec": {}}
    bmode = roll_all["baseline"]
    emode = roll_all["exec"]

    split_df = pd.DataFrame(per_split_rows).sort_values(["symbol", "split_id"]).reset_index(drop=True)
    sym_df = pd.DataFrame(symbol_rows).sort_values("symbol").reset_index(drop=True)

    b_split = pd.to_numeric(split_df.get("baseline_expectancy_net", np.nan), errors="coerce")
    e_split = pd.to_numeric(split_df.get("exec_expectancy_net", np.nan), errors="coerce")
    s1, s2, s3, s_pass = _stability_pass(
        exec_min=float(e_split.min()) if not e_split.empty else float("nan"),
        exec_med=float(e_split.median()) if not e_split.empty else float("nan"),
        exec_std=float(e_split.std(ddof=0)) if not e_split.empty else float("nan"),
        base_min=float(b_split.min()) if not b_split.empty else float("nan"),
        base_med=float(b_split.median()) if not b_split.empty else float("nan"),
        base_std=float(b_split.std(ddof=0)) if not b_split.empty else float("nan"),
    )

    delta_exp = float(emode.get("mean_expectancy_net", np.nan) - bmode.get("mean_expectancy_net", np.nan))
    delta_cvar = float(emode.get("cvar_5", np.nan) - bmode.get("cvar_5", np.nan))
    delta_maxdd = float(emode.get("max_drawdown", np.nan) - bmode.get("max_drawdown", np.nan))
    cvar_impr_ratio = ga_exec._improvement_ratio_abs(float(emode.get("cvar_5", np.nan)), float(bmode.get("cvar_5", np.nan)))
    maxdd_impr_ratio = ga_exec._improvement_ratio_abs(float(emode.get("max_drawdown", np.nan)), float(bmode.get("max_drawdown", np.nan)))

    p_exp = int(np.isfinite(delta_exp) and delta_exp >= float(args.phase_e_expectancy_min))
    p_cvar = int(
        (np.isfinite(cvar_impr_ratio) and cvar_impr_ratio >= float(args.phase_e_tail_improve_ratio_min))
        or (np.isfinite(delta_cvar) and delta_cvar >= float(args.phase_e_delta_cvar_min))
    )
    p_maxdd = int(
        (np.isfinite(maxdd_impr_ratio) and maxdd_impr_ratio >= float(args.phase_e_tail_improve_ratio_min))
        or (np.isfinite(delta_maxdd) and delta_maxdd >= float(args.phase_e_delta_maxdd_min))
    )
    p_realism = int(
        np.isfinite(float(emode.get("taker_share", np.nan)))
        and float(emode.get("taker_share", np.nan)) <= float(args.phase_e_max_taker_share)
        and np.isfinite(float(emode.get("median_fill_delay_min", np.nan)))
        and float(emode.get("median_fill_delay_min", np.nan)) <= float(args.phase_e_max_median_delay_min)
        and np.isfinite(float(emode.get("p95_fill_delay_min", np.nan)))
        and float(emode.get("p95_fill_delay_min", np.nan)) <= float(args.phase_e_max_p95_delay_min)
    )
    p_part_overall = int(np.isfinite(float(emode.get("entry_rate", np.nan))) and float(emode.get("entry_rate", np.nan)) >= float(args.phase_e_min_entry_rate_overall))
    p_part_symbol = int(
        not sym_df.empty
        and bool((pd.to_numeric(sym_df.get("exec_entry_rate", np.nan), errors="coerce") >= float(args.phase_e_min_entry_rate_symbol)).fillna(False).all())
    )
    p_data = int(
        np.isfinite(float(pd.to_numeric(df_all.get("exec_missing_slice_flag", 0), errors="coerce").fillna(0).mean() if not df_all.empty else np.nan))
        and float(pd.to_numeric(df_all.get("exec_missing_slice_flag", 0), errors="coerce").fillna(0).mean() if not df_all.empty else np.nan) <= float(args.max_missing_slice_rate)
    )
    phase_e_pass = int(p_exp and p_cvar and p_maxdd and p_realism and p_part_overall and p_part_symbol and p_data and s_pass)

    overall_df = pd.DataFrame(
        [
            {
                "scope": "overall",
                "symbols": int(len(sym_df)),
                "signals_total": int(emode.get("signals_total", 0)),
                "baseline_mean_expectancy_net": float(bmode.get("mean_expectancy_net", np.nan)),
                "exec_mean_expectancy_net": float(emode.get("mean_expectancy_net", np.nan)),
                "delta_expectancy_exec_minus_baseline": delta_exp,
                "baseline_pnl_net_sum": float(bmode.get("pnl_net_sum", np.nan)),
                "exec_pnl_net_sum": float(emode.get("pnl_net_sum", np.nan)),
                "baseline_cvar_5": float(bmode.get("cvar_5", np.nan)),
                "exec_cvar_5": float(emode.get("cvar_5", np.nan)),
                "cvar_improve_ratio": float(cvar_impr_ratio),
                "delta_cvar5_exec_minus_baseline": delta_cvar,
                "baseline_max_drawdown": float(bmode.get("max_drawdown", np.nan)),
                "exec_max_drawdown": float(emode.get("max_drawdown", np.nan)),
                "maxdd_improve_ratio": float(maxdd_impr_ratio),
                "delta_maxdd_exec_minus_baseline": delta_maxdd,
                "baseline_entry_rate": float(bmode.get("entry_rate", np.nan)),
                "exec_entry_rate": float(emode.get("entry_rate", np.nan)),
                "baseline_taker_share": float(bmode.get("taker_share", np.nan)),
                "exec_taker_share": float(emode.get("taker_share", np.nan)),
                "baseline_median_fill_delay_min": float(bmode.get("median_fill_delay_min", np.nan)),
                "exec_median_fill_delay_min": float(emode.get("median_fill_delay_min", np.nan)),
                "exec_p95_fill_delay_min": float(emode.get("p95_fill_delay_min", np.nan)),
                "exec_median_entry_improvement_bps": float(emode.get("median_entry_improvement_bps", np.nan)),
                "baseline_sl_hit_rate_valid": float(bmode.get("SL_hit_rate_valid", np.nan)),
                "exec_sl_hit_rate_valid": float(emode.get("SL_hit_rate_valid", np.nan)),
                "stability_min_pass": int(s1),
                "stability_median_pass": int(s2),
                "stability_std_pass": int(s3),
                "stability_pass": int(s_pass),
                "pass_expectancy": int(p_exp),
                "pass_tail_cvar": int(p_cvar),
                "pass_tail_maxdd": int(p_maxdd),
                "pass_realism": int(p_realism),
                "pass_participation_overall": int(p_part_overall),
                "pass_participation_symbol": int(p_part_symbol),
                "pass_data_quality": int(p_data),
                "phase_e_pass": int(phase_e_pass),
            }
        ]
    )

    out_root = _resolve_path(args.outdir) / f"TIGHT_REEVAL_{_utc_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)
    agg_csv = out_root / "AGG_exec_testonly_summary_tight.csv"
    agg_md = out_root / "AGG_exec_testonly_summary_tight.md"
    split_csv = out_root / "walkforward_results_by_split.csv"
    sym_csv = out_root / "risk_rollup_by_symbol.csv"
    ov_csv = out_root / "risk_rollup_overall.csv"
    dec_md = out_root / "decision.md"

    agg_rows = sym_df.copy()
    overall_for_agg = pd.DataFrame(
        [
            {
                "symbol": "ALL",
                "signals_total": int(overall_df.iloc[0]["signals_total"]),
                "baseline_expectancy_net": float(overall_df.iloc[0]["baseline_mean_expectancy_net"]),
                "exec_expectancy_net": float(overall_df.iloc[0]["exec_mean_expectancy_net"]),
                "delta_expectancy_exec_minus_baseline": float(overall_df.iloc[0]["delta_expectancy_exec_minus_baseline"]),
                "baseline_pnl_net_sum": float(overall_df.iloc[0]["baseline_pnl_net_sum"]),
                "exec_pnl_net_sum": float(overall_df.iloc[0]["exec_pnl_net_sum"]),
                "baseline_cvar_5": float(overall_df.iloc[0]["baseline_cvar_5"]),
                "exec_cvar_5": float(overall_df.iloc[0]["exec_cvar_5"]),
                "delta_cvar5_exec_minus_baseline": float(overall_df.iloc[0]["delta_cvar5_exec_minus_baseline"]),
                "baseline_max_drawdown": float(overall_df.iloc[0]["baseline_max_drawdown"]),
                "exec_max_drawdown": float(overall_df.iloc[0]["exec_max_drawdown"]),
                "delta_maxdd_exec_minus_baseline": float(overall_df.iloc[0]["delta_maxdd_exec_minus_baseline"]),
                "baseline_entry_rate": float(overall_df.iloc[0]["baseline_entry_rate"]),
                "exec_entry_rate": float(overall_df.iloc[0]["exec_entry_rate"]),
                "exec_taker_share": float(overall_df.iloc[0]["exec_taker_share"]),
                "exec_median_fill_delay_min": float(overall_df.iloc[0]["exec_median_fill_delay_min"]),
                "exec_p95_fill_delay_min": float(overall_df.iloc[0]["exec_p95_fill_delay_min"]),
            }
        ]
    )
    agg_out = pd.concat([agg_rows, overall_for_agg], ignore_index=True, sort=False)
    agg_out.to_csv(agg_csv, index=False)
    split_df.to_csv(split_csv, index=False)
    sym_df.to_csv(sym_csv, index=False)
    overall_df.to_csv(ov_csv, index=False)

    md_lines: List[str] = []
    md_lines.append("# Tight Execution Re-eval")
    md_lines.append("")
    md_lines.append(f"- Generated UTC: {_utc_now_iso()}")
    md_lines.append(f"- Symbols: `{','.join(symbols)}`")
    md_lines.append(f"- Exit policy source: {'Phase D regime' if use_regime else 'Phase C global fallback'}")
    md_lines.append(f"- Baseline entry: `market next 3m open`")
    md_lines.append(f"- Exec entry: `{args.phase_e_exec_entry_mode}` limit_offset_bps={float(args.phase_e_limit_offset_bps):.2f}, max_fill_delay_min={int(args.phase_e_max_fill_delay_min)}")
    md_lines.append("")
    md_lines.append("## Overall")
    md_lines.append("")
    md_lines.append(f"- baseline_expectancy_net: {float(overall_df.iloc[0]['baseline_mean_expectancy_net']):.6f}")
    md_lines.append(f"- exec_expectancy_net: {float(overall_df.iloc[0]['exec_mean_expectancy_net']):.6f}")
    md_lines.append(f"- delta_expectancy_exec_minus_baseline: {float(overall_df.iloc[0]['delta_expectancy_exec_minus_baseline']):.6f}")
    md_lines.append(f"- baseline_cvar_5: {float(overall_df.iloc[0]['baseline_cvar_5']):.6f}")
    md_lines.append(f"- exec_cvar_5: {float(overall_df.iloc[0]['exec_cvar_5']):.6f}")
    md_lines.append(f"- baseline_max_drawdown: {float(overall_df.iloc[0]['baseline_max_drawdown']):.6f}")
    md_lines.append(f"- exec_max_drawdown: {float(overall_df.iloc[0]['exec_max_drawdown']):.6f}")
    md_lines.append(f"- exec_entry_rate: {float(overall_df.iloc[0]['exec_entry_rate']):.6f}")
    md_lines.append(f"- exec_taker_share: {float(overall_df.iloc[0]['exec_taker_share']):.6f}")
    md_lines.append(f"- exec_median_fill_delay_min: {float(overall_df.iloc[0]['exec_median_fill_delay_min']):.2f}")
    md_lines.append(f"- exec_p95_fill_delay_min: {float(overall_df.iloc[0]['exec_p95_fill_delay_min']):.2f}")
    md_lines.append("")
    md_lines.append("## Rubric")
    md_lines.append("")
    md_lines.append(f"- expectancy >= baseline + {float(args.phase_e_expectancy_min):.5f}: {int(p_exp)}")
    md_lines.append(f"- tail cvar pass (ratio>={float(args.phase_e_tail_improve_ratio_min):.2f} or delta>={float(args.phase_e_delta_cvar_min):.5f}): {int(p_cvar)}")
    md_lines.append(f"- tail maxdd pass (ratio>={float(args.phase_e_tail_improve_ratio_min):.2f} or delta>={float(args.phase_e_delta_maxdd_min):.2f}): {int(p_maxdd)}")
    md_lines.append(f"- realism pass: {int(p_realism)}")
    md_lines.append(f"- participation pass overall/symbol: {int(p_part_overall)}/{int(p_part_symbol)}")
    md_lines.append(f"- stability pass: {int(s_pass)}")
    md_lines.append(f"- data quality pass: {int(p_data)}")
    md_lines.append(f"- Decision: **{'PASS' if int(phase_e_pass) == 1 else 'FAIL'}**")
    agg_md.write_text("\n".join(md_lines).strip() + "\n", encoding="utf-8")

    dec_lines = list(md_lines)
    dec_md.write_text("\n".join(dec_lines).strip() + "\n", encoding="utf-8")

    (out_root / "metrics_definition.md").write_text(_metrics_definition_text(), encoding="utf-8")
    (out_root / "fee_model.json").write_text(json.dumps(_fee_model_payload(args, "scripts/exit_regime.py:phase_e"), indent=2), encoding="utf-8")

    repro_lines = [
        "# Repro",
        "",
        "```bash",
        "cd /root/analysis/0.87",
        ".venv/bin/python scripts/exit_regime.py --symbols " + ",".join(symbols),
        "```",
    ]
    (out_root / "repro.md").write_text("\n".join(repro_lines).strip() + "\n", encoding="utf-8")

    phase_lines = [
        "Phase: E (Tight execution re-eval with selected exits)",
        f"Timestamp UTC: {_utc_now_iso()}",
        f"Status: {'PASS' if int(phase_e_pass) == 1 else 'FAIL'}",
        f"Exit policy source: {'Phase D regime' if use_regime else 'Phase C global fallback'}",
        f"Delta expectancy: {float(delta_exp):.6f}",
        f"Delta cvar5: {float(delta_cvar):.6f}",
        f"Delta maxdd: {float(delta_maxdd):.6f}",
        f"Artifacts: {agg_csv.name}, {agg_md.name}, {sym_csv.name}, {ov_csv.name}, {dec_md.name}",
    ]
    (out_root / "phase_result.md").write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")

    snap = out_root / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    ecfg = _resolve_path(args.execution_config)
    if ecfg.exists():
        shutil.copy2(ecfg, snap / "execution_configs.yaml")
    policy_file = _resolve_path(str(phase_d_ctx["policy_file"]))
    if policy_file.exists():
        shutil.copy2(policy_file, snap / "selected_exit_policy.json")
    os.system(f"git -C {PROJECT_ROOT} status --short > {out_root / 'git_status.txt'}")

    print(str(out_root))
    print(str(agg_csv))
    print(str(agg_md))
    print(str(sym_csv))
    print(str(ov_csv))
    print(str(dec_md))
    return out_root


def run(args: argparse.Namespace) -> Tuple[Path, Path]:
    phase_d_dir, dctx = run_phase_d(args)
    phase_e_dir = run_phase_e(args, dctx)
    return phase_d_dir, phase_e_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase D regime-aware exits + Phase E tight execution re-eval.")
    ap.add_argument("--symbols", default="SOLUSDT,AVAXUSDT,NEARUSDT")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--scan-dir", default="")
    ap.add_argument("--best-csv", default="")

    ap.add_argument("--signals-dir", default="data/signals")
    ap.add_argument("--refresh-signals", type=int, default=0)
    ap.add_argument("--signal-order", choices=["latest", "oldest"], default="latest")
    ap.add_argument("--max-signals", type=int, default=2000)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--wf-splits", type=int, default=5)
    ap.add_argument("--mode", choices=["tight", "normal"], default="normal")

    ap.add_argument("--timeframe", default="3m")
    ap.add_argument("--pre-buffer-hours", type=float, default=6.0)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--cache-dir", default="data/processed/_exec_klines_cache")
    ap.add_argument("--max-fetch-retries", type=int, default=8)
    ap.add_argument("--retry-base-sleep", type=float, default=0.5)
    ap.add_argument("--retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--fetch-pause-sec", type=float, default=0.03)
    ap.add_argument("--execution-config", default="configs/execution_configs.yaml")

    ap.add_argument("--fee-bps-maker", type=float, default=2.0)
    ap.add_argument("--fee-bps-taker", type=float, default=4.0)
    ap.add_argument("--slippage-bps-limit", type=float, default=0.5)
    ap.add_argument("--slippage-bps-market", type=float, default=2.0)

    ap.add_argument("--phase-c-dir", default="")
    ap.add_argument("--regime-topk", type=int, default=12)
    ap.add_argument("--regime-use-trend", type=int, default=1)
    ap.add_argument("--min-bucket-support", type=int, default=30)
    ap.add_argument("--max-missing-slice-rate", type=float, default=0.02)

    # Signal gate defaults used only when generating missing signal CSVs.
    ap.add_argument("--use-vol-regime-gate", type=int, default=1)
    ap.add_argument("--vol-regime-max-percentile", type=float, default=90.0)
    ap.add_argument("--vol-regime-lookback-bars", type=int, default=2160)
    ap.add_argument("--use-trend-gate", type=int, default=0)
    ap.add_argument("--trend-fast-col", default="EMA_50")
    ap.add_argument("--trend-slow-col", default="EMA_120")
    ap.add_argument("--trend-min-slope", type=float, default=0.0)
    ap.add_argument("--stop-distance-min-pct", type=float, default=0.0)

    # Phase D pass thresholds.
    ap.add_argument("--phase-d-expectancy-min", type=float, default=3e-5)
    ap.add_argument("--phase-d-maxdd-worse-tol", type=float, default=0.01)
    ap.add_argument("--phase-d-cvar-worse-tol", type=float, default=5e-5)

    # Phase E execution-entry model (tight).
    ap.add_argument("--phase-e-exec-entry-mode", choices=["market", "limit", "hybrid"], default="hybrid")
    ap.add_argument("--phase-e-limit-offset-bps", type=float, default=3.0)
    ap.add_argument("--phase-e-max-fill-delay-min", type=int, default=45)
    ap.add_argument("--phase-e-fallback-delay-min", type=int, default=15)
    ap.add_argument("--phase-e-micro-vol-filter", type=int, default=0)
    ap.add_argument("--phase-e-vol-threshold", type=float, default=3.0)
    ap.add_argument("--phase-e-spread-guard-bps", type=float, default=100.0)
    ap.add_argument("--phase-e-killzone-filter", type=int, default=0)

    # Phase E rubric thresholds.
    ap.add_argument("--phase-e-expectancy-min", type=float, default=5e-5)
    ap.add_argument("--phase-e-tail-improve-ratio-min", type=float, default=0.15)
    ap.add_argument("--phase-e-delta-cvar-min", type=float, default=1e-4)
    ap.add_argument("--phase-e-delta-maxdd-min", type=float, default=0.02)
    ap.add_argument("--phase-e-max-taker-share", type=float, default=0.25)
    ap.add_argument("--phase-e-max-median-delay-min", type=float, default=45.0)
    ap.add_argument("--phase-e-max-p95-delay-min", type=float, default=180.0)
    ap.add_argument("--phase-e-min-entry-rate-overall", type=float, default=0.70)
    ap.add_argument("--phase-e-min-entry-rate-symbol", type=float, default=0.55)

    ap.add_argument("--outdir", default="reports/execution_layer")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

