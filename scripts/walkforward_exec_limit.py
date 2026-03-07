#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _parse_floats(raw: str) -> List[float]:
    out: List[float] = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _parse_ints(raw: str) -> List[int]:
    out: List[int] = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(float(x)))
    return out


def _parse_strs(raw: str) -> List[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _parse_scalar(raw: str) -> Any:
    s = str(raw).strip()
    if not s:
        return ""
    if s.lower() in {"true", "yes"}:
        return 1
    if s.lower() in {"false", "no"}:
        return 0
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def _load_execution_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Minimal YAML-like parser for this file shape:
    # SYMBOL:
    #   key: value
    #   constraints:
    #     x: y
    out: Dict[str, Any] = {}
    cur_symbol: Optional[str] = None
    cur_section: Optional[str] = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            cur_symbol = stripped[:-1].strip()
            if cur_symbol:
                out[cur_symbol] = {}
                cur_section = None
            continue
        if cur_symbol is None:
            continue
        if indent == 2:
            if stripped.endswith(":"):
                key = stripped[:-1].strip()
                out[cur_symbol][key] = {}
                cur_section = key
            elif ":" in stripped:
                key, val = stripped.split(":", 1)
                out[cur_symbol][key.strip()] = _parse_scalar(val)
                cur_section = None
            continue
        if indent == 4 and cur_section and isinstance(out[cur_symbol].get(cur_section), dict) and ":" in stripped:
            key, val = stripped.split(":", 1)
            out[cur_symbol][cur_section][key.strip()] = _parse_scalar(val)
    return out


def _symbol_exec_config(all_cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    if not isinstance(all_cfg, dict):
        return {}
    sym_u = str(symbol).upper()
    # Accept both explicit symbol keys and nested "symbols" map.
    if sym_u in all_cfg and isinstance(all_cfg[sym_u], dict):
        return dict(all_cfg[sym_u])
    symbols_map = all_cfg.get("symbols")
    if isinstance(symbols_map, dict) and sym_u in symbols_map and isinstance(symbols_map[sym_u], dict):
        return dict(symbols_map[sym_u])
    return {}


def _pick_symbol_params(best_csv: Path, symbol: str, side: str = "long") -> Tuple[str, Path, pd.Series]:
    df = pd.read_csv(best_csv)
    if df.empty:
        raise SystemExit(f"No rows in {best_csv}")
    rows = df.copy()
    if "side" in rows.columns:
        rows = rows[rows["side"].astype(str).str.lower() == str(side).lower()].copy()
    if "pass" in rows.columns:
        rows = rows[rows["pass"].map(exec3m._as_bool)].copy()
    if rows.empty:
        rows = df.copy()
    rows["symbol_u"] = rows.get("symbol", "").astype(str).str.strip().str.upper()
    sym_u = str(symbol).strip().upper()
    rows = rows[rows["symbol_u"] == sym_u].copy()
    if rows.empty:
        raise SystemExit(f"Symbol {sym_u} not found in {best_csv}")
    rows["score"] = pd.to_numeric(rows.get("score"), errors="coerce").fillna(-1e18)
    rows = rows.sort_values(["score"], ascending=[False]).reset_index(drop=True)
    row = rows.iloc[0]
    params_raw = str(row.get("params_file", "")).strip()
    if not params_raw:
        raise SystemExit(f"Missing params_file for symbol={sym_u} in {best_csv}")
    params_file = _resolve_path(params_raw)
    if not params_file.exists():
        raise SystemExit(f"Missing params file for {sym_u}: {params_file}")
    return sym_u, params_file, row


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


def _metrics_definition_text() -> str:
    lines: List[str] = []
    lines.append("# Metrics Definition")
    lines.append("")
    lines.append("- Baseline entry: market at next 3m open after 1h signal timestamp (UTC).")
    lines.append("- Baseline exits: strategy TP/SL from signal params, simulated sequentially on 3m bars inside bounded eval window.")
    lines.append("- `expectancy_net`: mean per valid filled trade (`pnl_net_sum / entries_valid`).")
    lines.append("- `baseline_expectancy_net`: baseline mean per valid filled baseline trade.")
    lines.append("- `pnl_net_pct`: net return fraction after fees and slippage.")
    lines.append("- Invalid geometry rows are excluded from entry-conditioned metrics.")
    return "\n".join(lines).strip() + "\n"


def _build_args_like_exec(user_args: argparse.Namespace) -> argparse.Namespace:
    base = exec3m._build_arg_parser().parse_args([])
    base.timeframe = str(user_args.timeframe)
    base.pre_buffer_hours = float(user_args.pre_buffer_hours)
    base.exec_horizon_hours = float(user_args.exec_horizon_hours)
    base.local_timezone = str(user_args.local_timezone)
    base.max_fetch_retries = int(user_args.max_fetch_retries)
    base.retry_base_sleep = float(user_args.retry_base_sleep)
    base.retry_max_sleep = float(user_args.retry_max_sleep)
    base.fetch_pause_sec = float(user_args.fetch_pause_sec)
    base.exec_mode = "exec_limit"

    base.exec_use_ladder = int(user_args.exec_use_ladder)
    base.exec_k1 = float(user_args.exec_k1)
    base.exec_k2 = float(user_args.exec_k2)
    base.exec_adaptive_k = int(user_args.exec_adaptive_k)
    base.exec_two_stage = int(user_args.exec_two_stage)
    base.exec_stage1_bars = int(user_args.exec_stage1_bars)
    base.exec_stage2_bars = int(user_args.exec_stage2_bars)
    base.exec_move_away_thr = float(user_args.exec_move_away_thr)
    base.use_micro_panic = int(user_args.use_micro_panic)
    base.panic_mult = float(user_args.panic_mult)

    base.fee_bps_maker = float(user_args.fee_bps_maker)
    base.fee_bps_taker = float(user_args.fee_bps_taker)
    base.slippage_bps_limit = float(user_args.slippage_bps_limit)
    base.slippage_bps_market = float(user_args.slippage_bps_market)
    base.debug_ict = int(user_args.debug_ict)
    base.use_vol_gate = int(user_args.default_use_vol_gate)
    base.vol_z_thr = float(user_args.vol_z_thr)
    base.vol_p_thr = float(user_args.vol_p_thr)
    base.use_vol_regime_gate = int(user_args.use_vol_regime_gate)
    base.vol_regime_max_percentile = float(user_args.vol_regime_max_percentile)
    base.vol_regime_lookback_bars = int(user_args.vol_regime_lookback_bars)
    base.use_trend_gate = int(user_args.use_trend_gate)
    base.trend_fast_col = str(user_args.trend_fast_col)
    base.trend_slow_col = str(user_args.trend_slow_col)
    base.trend_min_slope = float(user_args.trend_min_slope)
    base.stop_distance_min_pct = float(user_args.stop_distance_min_pct)
    return base


def _simulate_one_signal(
    *,
    ctx: Dict[str, Any],
    sim_args: argparse.Namespace,
) -> Dict[str, Any]:
    s = ctx["signal"]
    signal_time = exec3m._to_utc_ts(s.signal_time)
    baseline = ctx["baseline"]
    df3m = ctx["df3m"]

    exec_res = exec3m._simulate_exec_limit_long(
        df3m=df3m,
        signal_time=signal_time,
        tp_mult=float(s.tp_mult),
        sl_mult=float(s.sl_mult),
        args=sim_args,
        debug_events=None,
        signal_id=s.signal_id,
        baseline_exit_time=baseline.get("exit_time"),
        eval_horizon_hours=float(sim_args.exec_horizon_hours),
    )

    baseline_liq = exec3m._liquidity_type_from_entry_type(baseline.get("entry_type", "")) if bool(baseline.get("filled", False)) else ""
    exec_liq = exec3m._liquidity_type_from_entry_type(exec_res.get("entry_type", "")) if bool(exec_res.get("filled", False)) else ""
    b_cost = exec3m._costed_pnl_long(
        entry_price=baseline.get("entry_price"),
        exit_price=baseline.get("exit_price"),
        entry_liquidity_type=baseline_liq,
        fee_bps_maker=float(sim_args.fee_bps_maker),
        fee_bps_taker=float(sim_args.fee_bps_taker),
        slippage_bps_limit=float(sim_args.slippage_bps_limit),
        slippage_bps_market=float(sim_args.slippage_bps_market),
    )
    e_cost = exec3m._costed_pnl_long(
        entry_price=exec_res.get("entry_price"),
        exit_price=exec_res.get("exit_price"),
        entry_liquidity_type=exec_liq,
        fee_bps_maker=float(sim_args.fee_bps_maker),
        fee_bps_taker=float(sim_args.fee_bps_taker),
        slippage_bps_limit=float(sim_args.slippage_bps_limit),
        slippage_bps_market=float(sim_args.slippage_bps_market),
    )

    row = {
        "signal_id": s.signal_id,
        "signal_time": str(signal_time),
        "baseline_filled": int(bool(baseline.get("filled", False))),
        "baseline_entry_price": float(baseline.get("entry_price", np.nan)),
        "baseline_exit_price": float(baseline.get("exit_price", np.nan)),
        "baseline_sl_hit": int(bool(baseline.get("sl_hit", False))),
        "baseline_tp_hit": int(bool(baseline.get("tp_hit", False))),
        "baseline_invalid_stop_geometry": int(bool(baseline.get("invalid_stop_geometry", 0))),
        "baseline_invalid_tp_geometry": int(bool(baseline.get("invalid_tp_geometry", 0))),
        "baseline_mae_pct": float(baseline.get("mae_pct", np.nan)),
        "baseline_mfe_pct": float(baseline.get("mfe_pct", np.nan)),
        "baseline_fill_delay_min": 0.0 if bool(baseline.get("filled", False)) else float("nan"),
        "baseline_fill_liquidity_type": baseline_liq,
        "baseline_pnl_gross_pct": float(b_cost["pnl_gross_pct"]),
        "baseline_pnl_net_pct": float(b_cost["pnl_net_pct"]),
        "exec_filled": int(bool(exec_res.get("filled", False))),
        "exec_entry_price": float(exec_res.get("entry_price", np.nan)),
        "exec_exit_price": float(exec_res.get("exit_price", np.nan)),
        "exec_sl_hit": int(bool(exec_res.get("sl_hit", False))),
        "exec_tp_hit": int(bool(exec_res.get("tp_hit", False))),
        "exec_invalid_stop_geometry": int(bool(exec_res.get("invalid_stop_geometry", 0))),
        "exec_invalid_tp_geometry": int(bool(exec_res.get("invalid_tp_geometry", 0))),
        "exec_mae_pct": float(exec_res.get("mae_pct", np.nan)),
        "exec_mfe_pct": float(exec_res.get("mfe_pct", np.nan)),
        "exec_entry_type": str(exec_res.get("entry_type", "")),
        "exec_fill_liquidity_type": exec_liq,
        "exec_fill_delay_min": float(exec_res.get("fill_delay_minutes", np.nan)),
        "exec_skip_reason": str(exec_res.get("skip_reason", "")),
        "exec_fallback_used": int(bool(exec_res.get("fallback_used", 0))),
        "vol_skip": int(bool(exec_res.get("vol_skip", 0))),
        "entry_improvement_pct": float((float(baseline.get("entry_price", np.nan)) - float(exec_res.get("entry_price", np.nan))) / float(baseline.get("entry_price", np.nan)))
        if np.isfinite(float(baseline.get("entry_price", np.nan)))
        and float(baseline.get("entry_price", np.nan)) > 0
        and np.isfinite(float(exec_res.get("entry_price", np.nan)))
        else float("nan"),
        "exec_pnl_gross_pct": float(e_cost["pnl_gross_pct"]),
        "exec_pnl_net_pct": float(e_cost["pnl_net_pct"]),
        "k": float(sim_args.exec_k),
        "timeout_bars": int(sim_args.exec_timeout_bars),
        "fallback": str(sim_args.exec_fallback),
        "use_vol_gate": int(sim_args.use_vol_gate),
    }
    return row


def _metrics(df: pd.DataFrame) -> Dict[str, float]:
    n = int(len(df))
    if n == 0:
        return {
            "signals_total": 0,
            "entries": 0,
            "entry_rate": float("nan"),
            "taker_share": float("nan"),
            "max_fill_delay_min": float("nan"),
            "median_fill_delay_min": float("nan"),
            "pnl_net_sum": float("nan"),
            "expectancy_net": float("nan"),
            "sl_hit_rate": float("nan"),
            "tp_hit_rate": float("nan"),
            "median_entry_improvement_pct": float("nan"),
            "median_entry_improvement_bps": float("nan"),
            "baseline_pnl_net_sum": float("nan"),
            "baseline_expectancy_net": float("nan"),
        }

    b_filled = pd.to_numeric(df["baseline_filled"], errors="coerce").fillna(0).astype(int)
    e_filled = pd.to_numeric(df["exec_filled"], errors="coerce").fillna(0).astype(int)
    b_valid = (
        (pd.to_numeric(df["baseline_invalid_stop_geometry"], errors="coerce").fillna(0).astype(int) == 0)
        & (pd.to_numeric(df["baseline_invalid_tp_geometry"], errors="coerce").fillna(0).astype(int) == 0)
    )
    e_valid = (
        (pd.to_numeric(df["exec_invalid_stop_geometry"], errors="coerce").fillna(0).astype(int) == 0)
        & (pd.to_numeric(df["exec_invalid_tp_geometry"], errors="coerce").fillna(0).astype(int) == 0)
    )
    b_mask = (b_filled == 1) & b_valid
    e_mask = (e_filled == 1) & e_valid
    entries = int(e_mask.sum())
    b_entries = int(b_mask.sum())

    exec_liq = df["exec_fill_liquidity_type"].fillna("").astype(str).str.lower()
    taker_share = float(((exec_liq == "taker") & e_mask).sum() / entries) if entries > 0 else float("nan")
    delay = pd.to_numeric(df["exec_fill_delay_min"], errors="coerce")
    pnl_net = pd.to_numeric(df["exec_pnl_net_pct"], errors="coerce")
    b_pnl_net = pd.to_numeric(df["baseline_pnl_net_pct"], errors="coerce")
    sl = pd.to_numeric(df["exec_sl_hit"], errors="coerce").fillna(0).astype(int)
    tp = pd.to_numeric(df["exec_tp_hit"], errors="coerce").fillna(0).astype(int)
    improve = pd.to_numeric(df["entry_improvement_pct"], errors="coerce")

    pnl_sum = float(pnl_net[e_mask].sum()) if entries > 0 else float("nan")
    expectancy = float(pnl_sum / entries) if entries > 0 and np.isfinite(pnl_sum) else float("nan")
    b_pnl_sum = float(b_pnl_net[b_mask].sum()) if b_entries > 0 else float("nan")
    b_expectancy = float(b_pnl_sum / b_entries) if b_entries > 0 and np.isfinite(b_pnl_sum) else float("nan")

    return {
        "signals_total": n,
        "entries": entries,
        "entry_rate": float(entries / n),
        "taker_share": taker_share,
        "max_fill_delay_min": float(delay[e_mask].max()) if entries > 0 and delay[e_mask].notna().any() else float("nan"),
        "median_fill_delay_min": float(delay[e_mask].median()) if entries > 0 and delay[e_mask].notna().any() else float("nan"),
        "pnl_net_sum": pnl_sum,
        "expectancy_net": expectancy,
        "sl_hit_rate": float((sl[e_mask] == 1).sum() / entries) if entries > 0 else float("nan"),
        "tp_hit_rate": float((tp[e_mask] == 1).sum() / entries) if entries > 0 else float("nan"),
        "median_entry_improvement_pct": float(improve[e_mask].median()) if entries > 0 and improve[e_mask].notna().any() else float("nan"),
        "median_entry_improvement_bps": float(10000.0 * improve[e_mask].median()) if entries > 0 and improve[e_mask].notna().any() else float("nan"),
        "baseline_pnl_net_sum": b_pnl_sum,
        "baseline_expectancy_net": b_expectancy,
    }


def run(args: argparse.Namespace) -> Path:
    scan_dir = _resolve_path(args.scan_dir) if str(args.scan_dir).strip() else exec3m._find_latest_scan_dir()
    best_csv = _resolve_path(args.best_csv) if str(args.best_csv).strip() else (scan_dir / "best_by_symbol.csv").resolve()
    symbol_arg = str(args.symbol).strip().upper()
    params_arg = str(args.params_file).strip()
    if symbol_arg and params_arg:
        symbol = symbol_arg
        params_file = _resolve_path(params_arg)
        selected_row = pd.Series({"symbol": symbol, "params_file": str(params_file)})
    elif symbol_arg:
        symbol, params_file, selected_row = _pick_symbol_params(best_csv=best_csv, symbol=symbol_arg, side="long")
    else:
        symbol, params_file, selected_row = exec3m._pick_symbol_from_best(best_csv=best_csv, rank=int(args.rank), side="long")

    exec_cfg_path = _resolve_path(args.execution_config)
    all_exec_cfg = _load_execution_config(exec_cfg_path) if int(args.ignore_execution_config) == 0 else {}
    symbol_cfg = _symbol_exec_config(all_exec_cfg, symbol)

    tight_mode = int(args.tight_mode) == 1
    if tight_mode:
        min_entry_rate = float(args.tight_min_entry_rate)
        max_fill_delay_min = float(args.tight_max_fill_delay_min)
        max_taker_share = float(args.tight_max_taker_share)
        min_median_entry_improvement_bps = float(args.tight_min_median_entry_improvement_bps)
        tight_cons = symbol_cfg.get("tight_constraints") if isinstance(symbol_cfg, dict) else {}
        if isinstance(tight_cons, dict):
            if "min_entry_rate" in tight_cons:
                min_entry_rate = float(tight_cons["min_entry_rate"])
            if "max_fill_delay_min" in tight_cons:
                max_fill_delay_min = float(tight_cons["max_fill_delay_min"])
            if "max_taker_share" in tight_cons:
                max_taker_share = float(tight_cons["max_taker_share"])
            if "min_median_entry_improvement_bps" in tight_cons:
                min_median_entry_improvement_bps = float(tight_cons["min_median_entry_improvement_bps"])
    else:
        min_entry_rate = float(args.min_entry_rate)
        max_fill_delay_min = float(args.max_fill_delay_min)
        max_taker_share = float(args.max_taker_share)
        min_median_entry_improvement_bps = float(args.min_median_entry_improvement_bps)
        if isinstance(symbol_cfg.get("constraints"), dict):
            cons = symbol_cfg["constraints"]
            if "min_entry_rate" in cons:
                min_entry_rate = float(cons["min_entry_rate"])
            if "max_fill_delay_min" in cons:
                max_fill_delay_min = float(cons["max_fill_delay_min"])
            if "max_taker_share" in cons:
                max_taker_share = float(cons["max_taker_share"])
            if "min_median_entry_improvement_bps" in cons:
                min_median_entry_improvement_bps = float(cons["min_median_entry_improvement_bps"])

    payload = json.loads(params_file.read_text(encoding="utf-8"))
    p = ga_long._norm_params(exec3m._unwrap_params(payload))

    gate_cfg = {
        "use_vol_regime_gate": int(args.use_vol_regime_gate),
        "vol_regime_max_percentile": float(args.vol_regime_max_percentile),
        "vol_regime_lookback_bars": int(args.vol_regime_lookback_bars),
        "use_trend_gate": int(args.use_trend_gate),
        "trend_fast_col": str(args.trend_fast_col),
        "trend_slow_col": str(args.trend_slow_col),
        "trend_min_slope": float(args.trend_min_slope),
        "stop_distance_min_pct": float(args.stop_distance_min_pct),
    }
    for k in list(gate_cfg.keys()):
        if k in symbol_cfg:
            gate_cfg[k] = symbol_cfg[k]

    df_1h = exec3m._load_symbol_df(symbol, tf="1h")
    signals = exec3m._build_1h_signals(
        df_1h=df_1h,
        p=p,
        max_signals=int(args.n_signals),
        order=str(args.signal_order),
        gate_cfg=gate_cfg,
    )
    if not signals:
        raise SystemExit("No signals found for walkforward")

    total = int(len(signals))
    train_n = max(1, int(round(total * float(args.train_ratio))))
    train_n = min(train_n, total - 1) if total > 1 else train_n
    train_signals = signals[:train_n]
    test_signals = signals[train_n:]
    if not test_signals:
        test_signals = signals[-1:]
        train_signals = signals[:-1]

    base_args = _build_args_like_exec(args)
    if "exec_mode" in symbol_cfg:
        base_args.exec_mode = str(symbol_cfg["exec_mode"])
    if "k_base" in symbol_cfg:
        base_args.exec_k = float(symbol_cfg["k_base"])
    if "timeout_bars" in symbol_cfg:
        base_args.exec_timeout_bars = int(float(symbol_cfg["timeout_bars"]))
    if "use_vol_gate" in symbol_cfg:
        base_args.use_vol_gate = int(float(symbol_cfg["use_vol_gate"]))
    if "use_panic_filter" in symbol_cfg:
        base_args.use_micro_panic = int(float(symbol_cfg["use_panic_filter"]))
    if "ladder" in symbol_cfg:
        base_args.exec_use_ladder = int(float(symbol_cfg["ladder"]))
    if "fallback" in symbol_cfg:
        base_args.exec_fallback = str(symbol_cfg["fallback"])
    if "adaptive_k" in symbol_cfg:
        base_args.exec_adaptive_k = int(float(symbol_cfg["adaptive_k"]))
    if "two_stage" in symbol_cfg:
        base_args.exec_two_stage = int(float(symbol_cfg["two_stage"]))
    if "use_vol_regime_gate" in symbol_cfg:
        base_args.use_vol_regime_gate = int(float(symbol_cfg["use_vol_regime_gate"]))
    if "vol_regime_max_percentile" in symbol_cfg:
        base_args.vol_regime_max_percentile = float(symbol_cfg["vol_regime_max_percentile"])
    if "vol_regime_lookback_bars" in symbol_cfg:
        base_args.vol_regime_lookback_bars = int(float(symbol_cfg["vol_regime_lookback_bars"]))
    if "use_trend_gate" in symbol_cfg:
        base_args.use_trend_gate = int(float(symbol_cfg["use_trend_gate"]))
    if "trend_fast_col" in symbol_cfg:
        base_args.trend_fast_col = str(symbol_cfg["trend_fast_col"])
    if "trend_slow_col" in symbol_cfg:
        base_args.trend_slow_col = str(symbol_cfg["trend_slow_col"])
    if "trend_min_slope" in symbol_cfg:
        base_args.trend_min_slope = float(symbol_cfg["trend_min_slope"])
    if "stop_distance_min_pct" in symbol_cfg:
        base_args.stop_distance_min_pct = float(symbol_cfg["stop_distance_min_pct"])
    if tight_mode:
        # Tight mode treats execution as realism layer only: market fallback, no skip policy.
        base_args.exec_fallback = "market"

    cache_root = _resolve_path(args.cache_dir)
    pre_h = float(base_args.pre_buffer_hours)
    hor_h = float(base_args.exec_horizon_hours)
    min_signal = min(exec3m._to_utc_ts(s.signal_time) for s in signals)
    max_signal = max(exec3m._to_utc_ts(s.signal_time) for s in signals)
    all_start = min_signal - pd.Timedelta(hours=pre_h)
    all_end = max_signal + pd.Timedelta(hours=hor_h)
    df3m_all = exec3m._load_or_fetch_klines(
        symbol=symbol,
        timeframe=str(base_args.timeframe),
        start_ts=all_start,
        end_ts=all_end,
        cache_root=cache_root,
        max_retries=int(base_args.max_fetch_retries),
        retry_base_sleep_sec=float(base_args.retry_base_sleep),
        retry_max_sleep_sec=float(base_args.retry_max_sleep),
        pause_sec=float(base_args.fetch_pause_sec),
    )
    df3m_all = exec3m._normalize_ohlcv_cols(df3m_all)
    contexts: List[Dict[str, Any]] = []
    for s in signals:
        signal_time = exec3m._to_utc_ts(s.signal_time)
        start_ts = signal_time - pd.Timedelta(hours=pre_h)
        end_ts = signal_time + pd.Timedelta(hours=hor_h)
        df3m = df3m_all[(df3m_all["Timestamp"] >= start_ts) & (df3m_all["Timestamp"] < end_ts)].copy()
        df3m = df3m[(df3m["Timestamp"] >= start_ts) & (df3m["Timestamp"] < end_ts)].reset_index(drop=True)
        if tight_mode and int(args.tight_require_no_missing_data) == 1 and df3m.empty:
            raise SystemExit(f"Tight mode violation: no 3m data for signal_id={s.signal_id} time={signal_time}")
        baseline = exec3m._simulate_baseline_long(
            df3m=df3m,
            signal_time=signal_time,
            tp_mult=float(s.tp_mult),
            sl_mult=float(s.sl_mult),
            eval_horizon_hours=float(base_args.exec_horizon_hours),
        )
        if tight_mode and int(args.tight_require_no_missing_data) == 1 and (not bool(baseline.get("filled", False))):
            rs = str(baseline.get("skip_reason", ""))
            if rs in {"no_3m_data", "bad_3m_data", "no_bar_after_signal"}:
                raise SystemExit(f"Tight mode violation: baseline skip_reason={rs} signal_id={s.signal_id} time={signal_time}")
        contexts.append({"signal": s, "df3m": df3m, "baseline": baseline})

    train_ids = {s.signal_id for s in train_signals}
    test_ids = {s.signal_id for s in test_signals}
    ctx_train = [c for c in contexts if c["signal"].signal_id in train_ids]
    ctx_test = [c for c in contexts if c["signal"].signal_id in test_ids]

    k_values = _parse_floats(args.k_values)
    timeout_values = _parse_ints(args.timeout_values)
    fallback_values = _parse_strs(args.fallback_values)
    vol_gate_values = [int(x) for x in _parse_ints(args.vol_gate_values)]
    if "k_values" in symbol_cfg and isinstance(symbol_cfg["k_values"], (list, tuple)):
        k_values = [float(x) for x in symbol_cfg["k_values"]]
    elif "k_base" in symbol_cfg:
        kb = float(symbol_cfg["k_base"])
        k_values = sorted({round(max(0.0, kb - 0.2), 4), round(kb, 4), round(min(1.5, kb + 0.2), 4)})
    if "timeout_values" in symbol_cfg and isinstance(symbol_cfg["timeout_values"], (list, tuple)):
        timeout_values = [int(float(x)) for x in symbol_cfg["timeout_values"]]
    elif "timeout_bars" in symbol_cfg:
        timeout_values = [int(float(symbol_cfg["timeout_bars"]))]
    if "fallback_values" in symbol_cfg and isinstance(symbol_cfg["fallback_values"], (list, tuple)):
        fallback_values = [str(x) for x in symbol_cfg["fallback_values"]]
    elif "fallback" in symbol_cfg:
        fallback_values = [str(symbol_cfg["fallback"])]
    if "vol_gate_values" in symbol_cfg and isinstance(symbol_cfg["vol_gate_values"], (list, tuple)):
        vol_gate_values = [int(float(x)) for x in symbol_cfg["vol_gate_values"]]
    elif "use_vol_gate" in symbol_cfg:
        vol_gate_values = [int(float(symbol_cfg["use_vol_gate"]))]

    if not k_values:
        k_values = [float(getattr(base_args, "exec_k", 0.5))]
    if not timeout_values:
        timeout_values = [int(getattr(base_args, "exec_timeout_bars", 20))]
    if not fallback_values:
        fallback_values = [str(getattr(base_args, "exec_fallback", "market"))]
    if not vol_gate_values:
        vol_gate_values = [int(getattr(base_args, "use_vol_gate", 1))]
    if tight_mode:
        fallback_values = ["market"]

    tune_rows: List[Dict[str, Any]] = []
    for k in k_values:
        for timeout_bars in timeout_values:
            for fallback in fallback_values:
                for use_vol_gate in vol_gate_values:
                    sim_args = argparse.Namespace(**vars(base_args))
                    sim_args.exec_k = float(k)
                    sim_args.exec_timeout_bars = int(timeout_bars)
                    sim_args.exec_fallback = str(fallback)
                    sim_args.use_vol_gate = int(use_vol_gate)
                    rows = [_simulate_one_signal(ctx=c, sim_args=sim_args) for c in ctx_train]
                    df_cfg = pd.DataFrame(rows)
                    m = _metrics(df_cfg)
                    pass_entry = int(np.isfinite(m["entry_rate"]) and m["entry_rate"] >= float(min_entry_rate))
                    pass_delay = int(np.isfinite(m["max_fill_delay_min"]) and m["max_fill_delay_min"] <= float(max_fill_delay_min))
                    pass_taker = int(np.isfinite(m["taker_share"]) and m["taker_share"] <= float(max_taker_share))
                    pass_improve = int(
                        np.isfinite(m["median_entry_improvement_bps"])
                        and (m["median_entry_improvement_bps"] >= float(min_median_entry_improvement_bps))
                    )
                    tune_rows.append(
                        {
                            "k": float(k),
                            "timeout_bars": int(timeout_bars),
                            "fallback": str(fallback),
                            "use_vol_gate": int(use_vol_gate),
                            **m,
                            "pass_entry_rate": int(pass_entry),
                            "pass_max_delay": int(pass_delay),
                            "pass_taker_share": int(pass_taker),
                            "pass_median_entry_improvement": int(pass_improve),
                            "passes_constraints": int(pass_entry and pass_delay and pass_taker and pass_improve),
                        }
                    )

    tune_df = pd.DataFrame(tune_rows)
    tune_df = tune_df.sort_values(["passes_constraints", "expectancy_net", "pnl_net_sum"], ascending=[False, False, False]).reset_index(drop=True)
    if tune_df.empty:
        raise SystemExit("Tuning grid produced no rows")
    if tight_mode and int(args.tight_require_pass) == 1 and int((tune_df["passes_constraints"] == 1).sum()) == 0:
        raise SystemExit("Tight mode violation: no configuration satisfied tight constraints on train set.")
    best = tune_df.iloc[0].to_dict()

    best_args = argparse.Namespace(**vars(base_args))
    best_args.exec_k = float(best["k"])
    best_args.exec_timeout_bars = int(best["timeout_bars"])
    best_args.exec_fallback = str(best["fallback"])
    best_args.use_vol_gate = int(best["use_vol_gate"])

    test_rows = [_simulate_one_signal(ctx=c, sim_args=best_args) for c in ctx_test]
    test_df = pd.DataFrame(test_rows)
    test_metrics = _metrics(test_df)

    run_id = _utc_tag()
    out_root = _resolve_path(args.outdir) / f"{run_id}_walkforward_{symbol}"
    out_root.mkdir(parents=True, exist_ok=True)

    train_csv = out_root / f"{symbol}_walkforward_train_tuning.csv"
    test_signals_csv = out_root / f"{symbol}_walkforward_test_signals.csv"
    test_summary_csv = out_root / f"{symbol}_walkforward_test_summary.csv"
    report_md = out_root / f"{symbol}_walkforward_report.md"
    fee_json = out_root / "fee_model.json"
    metrics_md = out_root / "metrics_definition.md"

    tune_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_signals_csv, index=False)
    summary_row = {
        "run_id": run_id,
        "symbol": symbol,
        "params_file": str(params_file),
        "signals_total": int(total),
        "train_signals": int(len(ctx_train)),
        "test_signals": int(len(ctx_test)),
        "selected_k": float(best["k"]),
        "selected_timeout_bars": int(best["timeout_bars"]),
        "selected_fallback": str(best["fallback"]),
        "selected_use_vol_gate": int(best["use_vol_gate"]),
        "selected_passes_constraints": int(best["passes_constraints"]),
        "entry_rate": float(test_metrics["entry_rate"]),
        "entries": int(test_metrics["entries"]),
        "taker_share": float(test_metrics["taker_share"]),
        "max_fill_delay_min": float(test_metrics["max_fill_delay_min"]),
        "median_fill_delay_min": float(test_metrics["median_fill_delay_min"]),
        "pnl_net_sum": float(test_metrics["pnl_net_sum"]),
        "expectancy_net": float(test_metrics["expectancy_net"]),
        "sl_hit_rate": float(test_metrics["sl_hit_rate"]),
        "tp_hit_rate": float(test_metrics["tp_hit_rate"]),
        "median_entry_improvement_pct": float(test_metrics["median_entry_improvement_pct"]),
        "median_entry_improvement_bps": float(test_metrics["median_entry_improvement_bps"]),
        "baseline_pnl_net_sum": float(test_metrics["baseline_pnl_net_sum"]),
        "baseline_expectancy_net": float(test_metrics["baseline_expectancy_net"]),
        "constraints_min_entry_rate": float(min_entry_rate),
        "constraints_max_fill_delay_min": float(max_fill_delay_min),
        "constraints_max_taker_share": float(max_taker_share),
        "constraints_min_median_entry_improvement_bps": float(min_median_entry_improvement_bps),
        "tight_mode": int(tight_mode),
        "execution_config_path": str(exec_cfg_path),
        "use_vol_regime_gate": int(gate_cfg["use_vol_regime_gate"]),
        "vol_regime_max_percentile": float(gate_cfg["vol_regime_max_percentile"]),
        "vol_regime_lookback_bars": int(gate_cfg["vol_regime_lookback_bars"]),
        "use_trend_gate": int(gate_cfg["use_trend_gate"]),
        "trend_fast_col": str(gate_cfg["trend_fast_col"]),
        "trend_slow_col": str(gate_cfg["trend_slow_col"]),
        "trend_min_slope": float(gate_cfg["trend_min_slope"]),
        "stop_distance_min_pct": float(gate_cfg["stop_distance_min_pct"]),
        "train_tuning_csv": str(train_csv),
        "test_signals_csv": str(test_signals_csv),
    }
    pd.DataFrame([summary_row]).to_csv(test_summary_csv, index=False)
    fee_model_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_script": "scripts/walkforward_exec_limit.py",
        "baseline_definition": "market at next 3m open after 1h signal",
        "fee_bps_maker": float(args.fee_bps_maker),
        "fee_bps_taker": float(args.fee_bps_taker),
        "slippage_bps_limit": float(args.slippage_bps_limit),
        "slippage_bps_market": float(args.slippage_bps_market),
    }
    fee_json.write_text(json.dumps(fee_model_payload, indent=2), encoding="utf-8")
    metrics_md.write_text(_metrics_definition_text(), encoding="utf-8")

    lines: List[str] = []
    lines.append(f"# Walkforward Exec Limit Report: {symbol}")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- params_file: `{params_file}`")
    lines.append(f"- execution_config: `{exec_cfg_path}`")
    lines.append(f"- symbol_config: `{json.dumps(symbol_cfg, sort_keys=True)}`")
    lines.append(f"- gate_config: `{json.dumps(gate_cfg, sort_keys=True)}`")
    lines.append(f"- tight_mode: {int(tight_mode)}")
    lines.append(f"- total signals: {total}")
    lines.append(f"- train/test: {len(ctx_train)}/{len(ctx_test)}")
    lines.append(f"- train tuning CSV: `{train_csv}`")
    lines.append(f"- test signals CSV: `{test_signals_csv}`")
    lines.append(f"- test summary CSV: `{test_summary_csv}`")
    lines.append("")
    lines.append("## Selected Config (Train)")
    lines.append("")
    lines.append(f"- k: {float(best['k']):.4f}")
    lines.append(f"- timeout_bars: {int(best['timeout_bars'])}")
    lines.append(f"- fallback: {best['fallback']}")
    lines.append(f"- use_vol_gate: {int(best['use_vol_gate'])}")
    lines.append(f"- passes_constraints: {int(best['passes_constraints'])}")
    lines.append("")
    lines.append("## Test Metrics")
    lines.append("")
    lines.append(f"- entry_rate: {float(test_metrics['entry_rate']):.6f}")
    lines.append(f"- taker_share: {float(test_metrics['taker_share']):.6f}" if np.isfinite(test_metrics["taker_share"]) else "- taker_share: n/a")
    lines.append(f"- max_fill_delay_min: {float(test_metrics['max_fill_delay_min']):.2f}" if np.isfinite(test_metrics["max_fill_delay_min"]) else "- max_fill_delay_min: n/a")
    lines.append(f"- median_fill_delay_min: {float(test_metrics['median_fill_delay_min']):.2f}" if np.isfinite(test_metrics["median_fill_delay_min"]) else "- median_fill_delay_min: n/a")
    lines.append(f"- pnl_net_sum: {float(test_metrics['pnl_net_sum']):.6f}" if np.isfinite(test_metrics["pnl_net_sum"]) else "- pnl_net_sum: n/a")
    lines.append(f"- expectancy_net: {float(test_metrics['expectancy_net']):.6f}" if np.isfinite(test_metrics["expectancy_net"]) else "- expectancy_net: n/a")
    lines.append(f"- baseline_expectancy_net: {float(test_metrics['baseline_expectancy_net']):.6f}" if np.isfinite(test_metrics["baseline_expectancy_net"]) else "- baseline_expectancy_net: n/a")
    lines.append(
        f"- median_entry_improvement_bps: {float(test_metrics['median_entry_improvement_bps']):.4f}"
        if np.isfinite(test_metrics["median_entry_improvement_bps"])
        else "- median_entry_improvement_bps: n/a"
    )
    lines.append("")
    lines.append("## Constraint Report")
    lines.append("")
    lines.append(f"- min_entry_rate: {float(min_entry_rate):.6f}")
    lines.append(f"- max_fill_delay_min: {float(max_fill_delay_min):.2f}")
    lines.append(f"- max_taker_share: {float(max_taker_share):.6f}")
    lines.append(f"- min_median_entry_improvement_bps: {float(min_median_entry_improvement_bps):.4f}")
    lines.append(f"- selected_passes_constraints: {int(best['passes_constraints'])}")
    if tight_mode:
        lines.append("")
        lines.append("### Tight Mode")
        lines.append("")
        lines.append(f"- tight_require_no_missing_data: {int(args.tight_require_no_missing_data)}")
        lines.append(f"- tight_require_pass: {int(args.tight_require_pass)}")
        lines.append("- fallback forced to `market`")
    lines.append("")
    lines.append("## Top 10 Train Configs")
    lines.append("")
    lines.extend(
        _to_md_table(
            tune_df.head(10)[
                [
                    "k",
                    "timeout_bars",
                    "fallback",
                    "use_vol_gate",
                    "entry_rate",
                    "taker_share",
                    "max_fill_delay_min",
                    "median_fill_delay_min",
                    "expectancy_net",
                    "median_entry_improvement_bps",
                    "pnl_net_sum",
                    "passes_constraints",
                ]
            ]
        )
    )
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    run_meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "rank": int(args.rank),
        "scan_dir": str(scan_dir),
        "best_csv": str(best_csv),
        "params_file": str(params_file),
        "execution_config_path": str(exec_cfg_path),
        "symbol_config": symbol_cfg,
        "gate_config": gate_cfg,
        "tight_mode": int(tight_mode),
        "tight_constraints": {
            "min_entry_rate": float(min_entry_rate),
            "max_fill_delay_min": float(max_fill_delay_min),
            "max_taker_share": float(max_taker_share),
            "min_median_entry_improvement_bps": float(min_median_entry_improvement_bps),
        },
        "signals_total": int(total),
        "train_signals": int(len(ctx_train)),
        "test_signals": int(len(ctx_test)),
        "selected_config": {
            "k": float(best["k"]),
            "timeout_bars": int(best["timeout_bars"]),
            "fallback": str(best["fallback"]),
            "use_vol_gate": int(best["use_vol_gate"]),
        },
        "report_md": str(report_md),
        "train_csv": str(train_csv),
        "test_signals_csv": str(test_signals_csv),
        "test_summary_csv": str(test_summary_csv),
    }
    (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    print(str(out_root))
    print(str(train_csv))
    print(str(test_summary_csv))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Walk-forward evaluation for exec_limit (train tune + test evaluate).")
    ap.add_argument("--scan-dir", default="")
    ap.add_argument("--best-csv", default="")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--symbol", default="")
    ap.add_argument("--params-file", default="")
    ap.add_argument("--n-signals", type=int, default=2000)
    ap.add_argument("--signal-order", choices=["latest", "oldest"], default="latest")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--timeframe", default="3m")
    ap.add_argument("--pre-buffer-hours", type=float, default=6.0)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--cache-dir", default="data/processed/_exec_klines_cache")
    ap.add_argument("--local-timezone", default="Africa/Cairo")
    ap.add_argument("--max-fetch-retries", type=int, default=8)
    ap.add_argument("--retry-base-sleep", type=float, default=0.5)
    ap.add_argument("--retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--fetch-pause-sec", type=float, default=0.03)

    ap.add_argument("--k-values", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5")
    ap.add_argument("--timeout-values", default="5,10,20,40")
    ap.add_argument("--fallback-values", default="market")
    ap.add_argument("--vol-gate-values", default="0,1")
    ap.add_argument("--min-entry-rate", type=float, default=0.60)
    ap.add_argument("--max-fill-delay-min", type=float, default=90.0)
    ap.add_argument("--max-taker-share", type=float, default=0.40)
    ap.add_argument("--min-median-entry-improvement-bps", type=float, default=-9999.0)
    ap.add_argument("--execution-config", default="configs/execution_configs.yaml")
    ap.add_argument("--ignore-execution-config", type=int, default=0)
    ap.add_argument("--tight-mode", type=int, default=0)
    ap.add_argument("--tight-min-entry-rate", type=float, default=0.99)
    ap.add_argument("--tight-max-fill-delay-min", type=float, default=45.0)
    ap.add_argument("--tight-max-taker-share", type=float, default=0.25)
    ap.add_argument("--tight-min-median-entry-improvement-bps", type=float, default=0.0)
    ap.add_argument("--tight-require-no-missing-data", type=int, default=1)
    ap.add_argument("--tight-require-pass", type=int, default=1)

    ap.add_argument("--default-use-vol-gate", type=int, default=1)
    ap.add_argument("--vol-z-thr", type=float, default=2.5)
    ap.add_argument("--vol-p-thr", type=float, default=95.0)
    ap.add_argument("--use-vol-regime-gate", type=int, default=0)
    ap.add_argument("--vol-regime-max-percentile", type=float, default=90.0)
    ap.add_argument("--vol-regime-lookback-bars", type=int, default=2160)
    ap.add_argument("--use-trend-gate", type=int, default=0)
    ap.add_argument("--trend-fast-col", default="EMA_50")
    ap.add_argument("--trend-slow-col", default="EMA_120")
    ap.add_argument("--trend-min-slope", type=float, default=0.0)
    ap.add_argument("--stop-distance-min-pct", type=float, default=0.0)
    ap.add_argument("--exec-use-ladder", type=int, default=0)
    ap.add_argument("--exec-k1", type=float, default=0.3)
    ap.add_argument("--exec-k2", type=float, default=0.8)
    ap.add_argument("--exec-adaptive-k", type=int, default=1)
    ap.add_argument("--exec-two-stage", type=int, default=1)
    ap.add_argument("--exec-stage1-bars", type=int, default=10)
    ap.add_argument("--exec-stage2-bars", type=int, default=10)
    ap.add_argument("--exec-move-away-thr", type=float, default=1.0)
    ap.add_argument("--use-micro-panic", type=int, default=1)
    ap.add_argument("--panic-mult", type=float, default=2.5)

    ap.add_argument("--fee-bps-maker", type=float, default=2.0)
    ap.add_argument("--fee-bps-taker", type=float, default=4.0)
    ap.add_argument("--slippage-bps-limit", type=float, default=0.5)
    ap.add_argument("--slippage-bps-market", type=float, default=2.0)

    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--debug-ict", type=int, default=0)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
