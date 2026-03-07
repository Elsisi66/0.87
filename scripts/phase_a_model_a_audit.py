#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import backtest_exec_phasec_sol as phasec_bt  # noqa: E402
from scripts import execution_layer_3m_ict as exec3m  # noqa: E402
from scripts import phase_nx_exec_family_discovery as nx  # noqa: E402
from scripts import phase_r_route_harness_redesign as phase_r  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


PHASER_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/PHASER_ROUTE_HARNESS_REDESIGN_20260228_005334"
).resolve()


@dataclass
class OneHMarket:
    ts_ns: np.ndarray
    open_np: np.ndarray
    high_np: np.ndarray
    low_np: np.ndarray
    close_np: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: int = 12) -> str:
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


def git_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["git_head"] = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        out["git_head"] = "unavailable"
    try:
        status = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "status", "--short"],
            text=True,
        )
        out["git_status_short"] = status.strip().splitlines()
    except Exception:
        out["git_status_short"] = []
    return out


def build_split_lookup(bundle: ga_exec.SymbolBundle) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for sp in bundle.splits:
        idx0 = int(sp["test_start"])
        idx1 = int(sp["test_end"])
        sid = int(sp["split_id"])
        for ctx in bundle.contexts[idx0:idx1]:
            out[str(ctx.signal_id)] = sid
    return out


def load_1h_market(symbol: str) -> OneHMarket:
    full_fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"
    if not full_fp.exists():
        raise FileNotFoundError(f"Missing 1h parquet for Model A audit: {full_fp}")
    df = pd.read_parquet(full_fp)
    df = exec3m._normalize_ohlcv_cols(df)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).sort_values("Timestamp").reset_index(drop=True)
    return OneHMarket(
        ts_ns=np.array([int(t.value) for t in pd.to_datetime(df["Timestamp"], utc=True)], dtype=np.int64),
        open_np=pd.to_numeric(df["Open"], errors="coerce").to_numpy(dtype=float),
        high_np=pd.to_numeric(df["High"], errors="coerce").to_numpy(dtype=float),
        low_np=pd.to_numeric(df["Low"], errors="coerce").to_numpy(dtype=float),
        close_np=pd.to_numeric(df["Close"], errors="coerce").to_numpy(dtype=float),
    )


def build_1h_reference_rows(
    *,
    bundle: ga_exec.SymbolBundle,
    fee: phasec_bt.FeeModel,
    exec_horizon_hours: float,
) -> pd.DataFrame:
    split_lookup = build_split_lookup(bundle)
    signals_df = pd.DataFrame(
        [
            {
                "signal_id": str(ctx.signal_id),
                "signal_time": str(pd.to_datetime(ctx.signal_time, utc=True)),
                "tp_mult": float(ctx.tp_mult_sig),
                "sl_mult": float(ctx.sl_mult_sig),
            }
            for ctx in bundle.contexts
        ]
    )
    ref = phasec_bt._simulate_1h_reference(  # pylint: disable=protected-access
        signals_df=signals_df,
        split_lookup=split_lookup,
        fee=fee,
        exec_horizon_hours=float(exec_horizon_hours),
        symbol=str(bundle.symbol),
    )
    x = ref.copy()
    x = x.rename(
        columns={
            "filled": "baseline_filled",
            "valid_for_metrics": "baseline_valid_for_metrics",
            "sl_hit": "baseline_sl_hit",
            "tp_hit": "baseline_tp_hit",
            "entry_time": "baseline_entry_time",
            "exit_time": "baseline_exit_time",
            "entry_price": "baseline_entry_price",
            "exit_price": "baseline_exit_price",
            "exit_reason": "baseline_exit_reason",
            "fill_liquidity_type": "baseline_fill_liquidity_type",
            "fill_delay_min": "baseline_fill_delay_min",
            "mae_pct": "baseline_mae_pct",
            "mfe_pct": "baseline_mfe_pct",
            "pnl_gross_pct": "baseline_pnl_gross_pct",
            "pnl_net_pct": "baseline_pnl_net_pct",
            "entry_type": "baseline_entry_type",
        }
    )
    x["baseline_same_bar_hit"] = 0
    x["baseline_invalid_stop_geometry"] = 0
    x["baseline_invalid_tp_geometry"] = 0
    keep = [
        "signal_id",
        "signal_time",
        "split_id",
        "baseline_filled",
        "baseline_valid_for_metrics",
        "baseline_sl_hit",
        "baseline_tp_hit",
        "baseline_entry_time",
        "baseline_exit_time",
        "baseline_entry_price",
        "baseline_exit_price",
        "baseline_exit_reason",
        "baseline_fill_liquidity_type",
        "baseline_fill_delay_min",
        "baseline_mae_pct",
        "baseline_mfe_pct",
        "baseline_pnl_gross_pct",
        "baseline_pnl_net_pct",
        "baseline_entry_type",
        "baseline_same_bar_hit",
        "baseline_invalid_stop_geometry",
        "baseline_invalid_tp_geometry",
    ]
    for c in keep:
        if c not in x.columns:
            x[c] = np.nan
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    for c in [
        "baseline_entry_time",
        "baseline_exit_time",
    ]:
        x[c] = pd.to_datetime(x[c], utc=True, errors="coerce")
    return x.loc[:, keep].sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def build_model_a_variants() -> List[Dict[str, Any]]:
    return [
        {
            "candidate_id": "M1_ENTRY_ONLY_PASSIVE_BASELINE",
            "label": "M1 3m entry-only passive baseline",
            "entry_mode": "limit",
            "limit_offset_bps": 0.75,
            "fallback_to_market": 1,
            "fallback_delay_min": 6.0,
            "max_fill_delay_min": 24.0,
        },
        {
            "candidate_id": "M2_ENTRY_ONLY_MORE_PASSIVE",
            "label": "M2 3m entry-only more passive",
            "entry_mode": "limit",
            "limit_offset_bps": 1.50,
            "fallback_to_market": 1,
            "fallback_delay_min": 12.0,
            "max_fill_delay_min": 45.0,
        },
        {
            "candidate_id": "M3_ENTRY_ONLY_FASTER",
            "label": "M3 3m entry-only faster",
            "entry_mode": "limit",
            "limit_offset_bps": 0.35,
            "fallback_to_market": 1,
            "fallback_delay_min": 3.0,
            "max_fill_delay_min": 9.0,
        },
        {
            "candidate_id": "M4_ENTRY_ONLY_MARKET_CONTROL",
            "label": "M4 3m entry-only market-like control",
            "entry_mode": "market",
            "limit_offset_bps": 0.0,
            "fallback_to_market": 0,
            "fallback_delay_min": 0.0,
            "max_fill_delay_min": 0.0,
        },
    ]


def route_examples_match(
    actual: pd.DataFrame,
    expected_path: Path,
) -> Tuple[int, List[str]]:
    if not expected_path.exists():
        return 0, [f"missing_phaseR_example_file:{expected_path}"]
    exp = pd.read_csv(expected_path)
    keys = [
        "route_id",
        "route_start_idx",
        "route_end_idx_exclusive",
        "route_signal_count",
        "wf_test_signal_count",
        "first_signal_id",
        "last_signal_id",
    ]
    for df in (actual, exp):
        for c in keys:
            if c not in df.columns:
                return 0, [f"missing_required_column:{c}"]
    a = actual.loc[:, keys].copy().sort_values("route_id").reset_index(drop=True)
    b = exp.loc[:, keys].copy().sort_values("route_id").reset_index(drop=True)
    if len(a) != len(b):
        return 0, [f"route_count_mismatch:{len(a)}!={len(b)}"]
    mismatches: List[str] = []
    for i in range(len(a)):
        for c in keys:
            av = a.iloc[i][c]
            bv = b.iloc[i][c]
            if str(av) != str(bv):
                mismatches.append(f"{a.iloc[i]['route_id']}:{c}:{av}!={bv}")
    return int(len(mismatches) == 0), mismatches[:12]


def simulate_entry_only_fill(ctx: ga_exec.SignalContext, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "filled": 0,
        "fill_time": pd.NaT,
        "fill_price": float("nan"),
        "fill_type": "",
        "entry_improvement_bps": float("nan"),
        "skip_reason": "",
        "missing_slice_flag": 0,
        "entry_ref_price": float("nan"),
    }
    ts_ns = ctx.ts_ns
    n = len(ts_ns)
    if n == 0:
        out["skip_reason"] = "no_3m_data"
        out["missing_slice_flag"] = 1
        return out
    sig_idx = int(np.searchsorted(ts_ns, int(ctx.signal_ts_ns), side="left"))
    if sig_idx >= n:
        out["skip_reason"] = "no_bar_after_signal"
        out["missing_slice_flag"] = 1
        return out

    open_np = ctx.open_np
    low_np = ctx.low_np
    entry_ref = float(open_np[sig_idx])
    out["entry_ref_price"] = float(entry_ref)
    if (not np.isfinite(entry_ref)) or entry_ref <= 0.0:
        out["skip_reason"] = "bad_entry_ref"
        out["missing_slice_flag"] = 1
        return out

    # Optional delayed reprice guard: wait N bars, then only continue with delayed
    # entry if adverse move from signal reference is within threshold.
    delay_bars = max(0, int(float(cfg.get("delay_bars_before_entry", 0))))
    adverse_guard_bps = float(cfg.get("adverse_move_guard_bps", float("nan")))
    guard_fallback = int(1 if int(cfg.get("adverse_guard_fallback_to_baseline_market", 1)) == 1 else 0)
    active_sig_idx = int(sig_idx)
    if delay_bars > 0:
        delayed_idx = int(sig_idx + delay_bars)
        if delayed_idx >= n:
            if guard_fallback == 1:
                fill_time = pd.to_datetime(int(ts_ns[int(sig_idx)]), utc=True)
                out.update(
                    {
                        "filled": 1,
                        "fill_time": fill_time,
                        "fill_price": float(open_np[int(sig_idx)]),
                        "fill_type": "market_guard_fallback",
                        "entry_improvement_bps": float((entry_ref - float(open_np[int(sig_idx)])) / entry_ref * 1e4),
                        "skip_reason": "",
                    }
                )
                return out
            out["skip_reason"] = "no_bar_after_delay"
            out["missing_slice_flag"] = 1
            return out
        delayed_open = float(open_np[delayed_idx])
        adverse_bps = float((delayed_open / entry_ref - 1.0) * 1e4) if np.isfinite(delayed_open) else float("nan")
        if np.isfinite(adverse_guard_bps) and np.isfinite(adverse_bps) and adverse_bps > adverse_guard_bps:
            if guard_fallback == 1:
                fill_time = pd.to_datetime(int(ts_ns[int(sig_idx)]), utc=True)
                out.update(
                    {
                        "filled": 1,
                        "fill_time": fill_time,
                        "fill_price": float(open_np[int(sig_idx)]),
                        "fill_type": "market_guard_fallback",
                        "entry_improvement_bps": float((entry_ref - float(open_np[int(sig_idx)])) / entry_ref * 1e4),
                        "skip_reason": "",
                    }
                )
                return out
            out["skip_reason"] = "adverse_guard_exceeded"
            return out
        active_sig_idx = int(delayed_idx)

    mode = str(cfg["entry_mode"]).strip().lower()
    max_fill_bars = max(0, int(np.ceil(float(cfg["max_fill_delay_min"]) / 3.0)))
    fallback_bars = max(0, int(np.ceil(float(cfg["fallback_delay_min"]) / 3.0)))
    fill_end_idx = min(n - 1, active_sig_idx + max_fill_bars)

    fill_idx: Optional[int] = None
    fill_px = float("nan")
    fill_type = ""

    if mode == "market":
        fill_idx = int(active_sig_idx)
        fill_px = float(open_np[fill_idx])
        fill_type = "market"
    else:
        limit_px = float(entry_ref * (1.0 - max(0.0, float(cfg["limit_offset_bps"])) / 1e4))
        for i in range(int(active_sig_idx), int(fill_end_idx) + 1):
            if np.isfinite(low_np[i]) and float(low_np[i]) <= limit_px:
                fill_idx = int(i)
                fill_px = float(limit_px)
                fill_type = "limit"
                break
        if fill_idx is None and int(cfg["fallback_to_market"]) == 1:
            m_idx = min(fill_end_idx, active_sig_idx + fallback_bars)
            if m_idx <= fill_end_idx:
                fill_idx = int(m_idx)
                fill_px = float(open_np[m_idx])
                fill_type = "market_fallback"

    # Optional market slippage cap: for market-like fills that are too adverse vs
    # signal reference, switch to a bounded limit+TTL reprice branch.
    market_slip_cap_bps = float(cfg.get("max_adverse_market_fill_bps", float("nan")))
    if (
        fill_idx is not None
        and str(fill_type) in {"market", "market_fallback", "market_guard_fallback", "market_cap_fallback"}
        and np.isfinite(market_slip_cap_bps)
        and market_slip_cap_bps >= 0.0
        and np.isfinite(fill_px)
        and np.isfinite(entry_ref)
        and entry_ref > 0.0
    ):
        adverse_bps = float(max(0.0, (fill_px / entry_ref - 1.0) * 1e4))
        if adverse_bps > market_slip_cap_bps:
            cap_limit_offset = float(cfg.get("cap_fallback_limit_offset_bps", 0.0))
            cap_limit_ttl_bars = max(1, int(float(cfg.get("cap_fallback_ttl_bars", 2))))
            cap_limit_px = float(entry_ref * (1.0 - max(0.0, cap_limit_offset) / 1e4))
            start_idx = int(min(max(fill_idx, 0), n - 1))
            end_idx = int(min(n - 1, start_idx + cap_limit_ttl_bars))
            limit_hit_idx: Optional[int] = None
            for i in range(start_idx, end_idx + 1):
                if np.isfinite(low_np[i]) and float(low_np[i]) <= cap_limit_px:
                    limit_hit_idx = int(i)
                    break
            if limit_hit_idx is not None:
                fill_idx = int(limit_hit_idx)
                fill_px = float(cap_limit_px)
                fill_type = "limit_cap_fallback"
            else:
                fill_idx = int(end_idx)
                fill_px = float(open_np[end_idx])
                fill_type = "market_cap_fallback"

    if fill_idx is None:
        out["skip_reason"] = "timeout_no_fill"
        return out

    fill_time = pd.to_datetime(int(ts_ns[int(fill_idx)]), utc=True)
    improve = float((entry_ref - fill_px) / entry_ref * 1e4) if np.isfinite(entry_ref) and entry_ref > 0 else float("nan")
    out.update(
        {
            "filled": 1,
            "fill_time": fill_time,
            "fill_price": float(fill_px),
            "fill_type": str(fill_type),
            "entry_improvement_bps": float(improve),
            "skip_reason": "",
        }
    )
    return out


def simulate_frozen_1h_exit(
    *,
    ctx: ga_exec.SignalContext,
    fill_time: pd.Timestamp,
    fill_price: float,
    one_h: OneHMarket,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if (not np.isfinite(fill_price)) or fill_price <= 0.0:
        return {
            "valid_for_metrics": 0,
            "exit_time": pd.NaT,
            "exit_price": float("nan"),
            "exit_reason": "invalid_fill_price",
            "sl_hit": 0,
            "tp_hit": 0,
            "same_bar_hit": 0,
            "mae_pct": float("nan"),
            "mfe_pct": float("nan"),
            "invalid_stop_geometry": 1,
            "invalid_tp_geometry": 1,
            "missing_slice_flag": 1,
        }

    fill_ts = pd.to_datetime(fill_time, utc=True, errors="coerce")
    if pd.isna(fill_ts):
        return {
            "valid_for_metrics": 0,
            "exit_time": pd.NaT,
            "exit_price": float("nan"),
            "exit_reason": "bad_fill_time",
            "sl_hit": 0,
            "tp_hit": 0,
            "same_bar_hit": 0,
            "mae_pct": float("nan"),
            "mfe_pct": float("nan"),
            "invalid_stop_geometry": 1,
            "invalid_tp_geometry": 1,
            "missing_slice_flag": 1,
        }

    start_idx = int(np.searchsorted(one_h.ts_ns, int(fill_ts.value), side="right"))
    if start_idx >= len(one_h.ts_ns):
        return {
            "valid_for_metrics": 0,
            "exit_time": fill_ts,
            "exit_price": float(fill_price),
            "exit_reason": "no_1h_bar_after_fill",
            "sl_hit": 0,
            "tp_hit": 0,
            "same_bar_hit": 0,
            "mae_pct": float("nan"),
            "mfe_pct": float("nan"),
            "invalid_stop_geometry": 0,
            "invalid_tp_geometry": 0,
            "missing_slice_flag": 1,
        }

    max_exit_ts_ns = exec3m._compute_eval_end_ns(  # pylint: disable=protected-access
        entry_ts_ns=int(fill_ts.value),
        eval_horizon_hours=float(args.exec_horizon_hours),
        baseline_exit_time=None,
    )
    tie_touch_policy = str(getattr(args, "tie_touch_policy", "sl_first")).strip().lower()
    sl_price = float(fill_price * float(ctx.sl_mult_sig))
    tp_price = float(fill_price * float(ctx.tp_mult_sig))
    sim = exec3m._simulate_path_long(  # pylint: disable=protected-access
        ts_ns=one_h.ts_ns,
        close=one_h.close_np,
        high=one_h.high_np,
        low=one_h.low_np,
        entry_idx=int(start_idx),
        entry_price=float(fill_price),
        sl_price=float(sl_price),
        tp_price=float(tp_price),
        max_exit_ts_ns=int(max_exit_ts_ns),
        tie_touch_policy=tie_touch_policy,
    )
    return {
        "valid_for_metrics": int(sim.get("valid_for_metrics", 0)),
        "exit_time": pd.to_datetime(sim.get("exit_time"), utc=True, errors="coerce"),
        "exit_price": float(sim.get("exit_price", np.nan)),
        "exit_reason": str(sim.get("exit_reason", "")),
        "sl_hit": int(bool(sim.get("sl_hit", False))),
        "tp_hit": int(bool(sim.get("tp_hit", False))),
        "same_bar_hit": int(sim.get("same_bar_hit", 0)),
        "mae_pct": float(sim.get("mae_pct", np.nan)),
        "mfe_pct": float(sim.get("mfe_pct", np.nan)),
        "invalid_stop_geometry": int(sim.get("invalid_stop_geometry", 0)),
        "invalid_tp_geometry": int(sim.get("invalid_tp_geometry", 0)),
        "missing_slice_flag": 0,
    }


def make_exec_default_row(baseline_row: pd.Series, ctx: ga_exec.SignalContext) -> Dict[str, Any]:
    return {
        "symbol": str(ctx.symbol),
        "signal_id": str(ctx.signal_id),
        "signal_time": str(pd.to_datetime(ctx.signal_time, utc=True)),
        "split_id": int(baseline_row.get("split_id", -1)),
        "baseline_filled": int(baseline_row.get("baseline_filled", 0)),
        "baseline_valid_for_metrics": int(baseline_row.get("baseline_valid_for_metrics", 0)),
        "baseline_sl_hit": int(baseline_row.get("baseline_sl_hit", 0)),
        "baseline_tp_hit": int(baseline_row.get("baseline_tp_hit", 0)),
        "baseline_pnl_net_pct": float(baseline_row.get("baseline_pnl_net_pct", np.nan)),
        "baseline_pnl_gross_pct": float(baseline_row.get("baseline_pnl_gross_pct", np.nan)),
        "baseline_fill_liquidity_type": str(baseline_row.get("baseline_fill_liquidity_type", "")),
        "baseline_fill_delay_min": float(baseline_row.get("baseline_fill_delay_min", np.nan)),
        "baseline_mae_pct": float(baseline_row.get("baseline_mae_pct", np.nan)),
        "baseline_mfe_pct": float(baseline_row.get("baseline_mfe_pct", np.nan)),
        "baseline_entry_time": str(baseline_row.get("baseline_entry_time", "")),
        "baseline_exit_time": str(baseline_row.get("baseline_exit_time", "")),
        "baseline_exit_reason": str(baseline_row.get("baseline_exit_reason", "")),
        "baseline_same_bar_hit": int(baseline_row.get("baseline_same_bar_hit", 0)),
        "baseline_invalid_stop_geometry": int(baseline_row.get("baseline_invalid_stop_geometry", 0)),
        "baseline_invalid_tp_geometry": int(baseline_row.get("baseline_invalid_tp_geometry", 0)),
        "baseline_entry_type": str(baseline_row.get("baseline_entry_type", "")),
        "baseline_entry_price": float(baseline_row.get("baseline_entry_price", np.nan)),
        "baseline_exit_price": float(baseline_row.get("baseline_exit_price", np.nan)),
        "exec_filled": 0,
        "exec_valid_for_metrics": 0,
        "exec_sl_hit": 0,
        "exec_tp_hit": 0,
        "exec_pnl_net_pct": float("nan"),
        "exec_pnl_gross_pct": float("nan"),
        "exec_fill_liquidity_type": "",
        "exec_fill_delay_min": float("nan"),
        "exec_mae_pct": float("nan"),
        "exec_mfe_pct": float("nan"),
        "entry_improvement_bps": float("nan"),
        "exec_skip_reason": "",
        "lookahead_violation": 0,
        "constraint_fail_reason": "",
        "missing_slice_flag": 0,
        "exec_exit_reason": "",
        "exec_entry_time": "",
        "exec_exit_time": "",
        "exec_entry_price": float("nan"),
        "exec_exit_price": float("nan"),
        "exec_same_bar_hit": 0,
        "exec_invalid_stop_geometry": 0,
        "exec_invalid_tp_geometry": 0,
        "exec_entry_type": "",
        "feature_skip_signal_range_pct": float("nan"),
        "feature_skip_upper_wick_ratio": float("nan"),
        "feature_skip_triggered": 0,
    }


def _signal_bar_preentry_features(
    *,
    ctx: ga_exec.SignalContext,
    one_h: OneHMarket,
) -> Dict[str, float]:
    if one_h.ts_ns.size == 0:
        return {
            "signal_range_pct": float("nan"),
            "upper_wick_ratio": float("nan"),
        }
    ts_target = int(ctx.signal_ts_ns)
    idx = int(np.searchsorted(one_h.ts_ns, ts_target, side="left"))
    if idx >= int(one_h.ts_ns.size):
        idx = int(one_h.ts_ns.size - 1)
    elif int(one_h.ts_ns[idx]) > ts_target and idx > 0:
        idx -= 1
    open_px = float(one_h.open_np[idx]) if idx < len(one_h.open_np) else float("nan")
    high_px = float(one_h.high_np[idx]) if idx < len(one_h.high_np) else float("nan")
    low_px = float(one_h.low_np[idx]) if idx < len(one_h.low_np) else float("nan")
    close_px = float(one_h.close_np[idx]) if idx < len(one_h.close_np) else float("nan")
    range_px = high_px - low_px if np.isfinite(high_px) and np.isfinite(low_px) else float("nan")
    signal_range_pct = (
        float(range_px / open_px)
        if np.isfinite(range_px) and np.isfinite(open_px) and open_px > 0.0
        else float("nan")
    )
    upper_wick_ratio = (
        float((high_px - max(open_px, close_px)) / range_px)
        if np.isfinite(high_px)
        and np.isfinite(open_px)
        and np.isfinite(close_px)
        and np.isfinite(range_px)
        and range_px > 0.0
        else float("nan")
    )
    return {
        "signal_range_pct": float(signal_range_pct),
        "upper_wick_ratio": float(upper_wick_ratio),
    }


def simulate_model_a_signal(
    *,
    ctx: ga_exec.SignalContext,
    baseline_row: pd.Series,
    cfg: Dict[str, Any],
    one_h: OneHMarket,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    row = make_exec_default_row(baseline_row=baseline_row, ctx=ctx)
    preentry_feats = _signal_bar_preentry_features(ctx=ctx, one_h=one_h)
    row["feature_skip_signal_range_pct"] = float(preentry_feats["signal_range_pct"])
    row["feature_skip_upper_wick_ratio"] = float(preentry_feats["upper_wick_ratio"])

    if int(cfg.get("feature_skip_mask_enabled", 0)) == 1:
        cap_range = float(cfg.get("skip_cap_signal_range_pct", np.nan))
        cap_wick = float(cfg.get("skip_cap_upper_wick_ratio", np.nan))
        logic = str(cfg.get("skip_mask_logic", "or")).strip().lower()
        trig_range = bool(np.isfinite(cap_range) and np.isfinite(preentry_feats["signal_range_pct"]) and preentry_feats["signal_range_pct"] > cap_range)
        trig_wick = bool(np.isfinite(cap_wick) and np.isfinite(preentry_feats["upper_wick_ratio"]) and preentry_feats["upper_wick_ratio"] > cap_wick)
        if logic == "and":
            skip_now = bool(trig_range and trig_wick)
        else:
            skip_now = bool(trig_range or trig_wick)
        if skip_now:
            row["feature_skip_triggered"] = 1
            row["exec_skip_reason"] = "feature_skip_mask"
            return row

    fill = simulate_entry_only_fill(ctx=ctx, cfg=cfg)
    row["missing_slice_flag"] = int(fill.get("missing_slice_flag", 0))
    row["entry_improvement_bps"] = float(fill.get("entry_improvement_bps", np.nan))
    if int(fill.get("filled", 0)) != 1:
        row["exec_skip_reason"] = str(fill.get("skip_reason", ""))
        return row

    fill_time = pd.to_datetime(fill.get("fill_time"), utc=True, errors="coerce")
    fill_price = float(fill.get("fill_price", np.nan))
    fill_type = str(fill.get("fill_type", ""))
    exit_res = simulate_frozen_1h_exit(
        ctx=ctx,
        fill_time=fill_time,
        fill_price=float(fill_price),
        one_h=one_h,
        args=args,
    )

    liq = "maker" if fill_type == "limit" else "taker"
    cost = exec3m._costed_pnl_long(  # pylint: disable=protected-access
        entry_price=float(fill_price),
        exit_price=float(exit_res.get("exit_price", np.nan)),
        entry_liquidity_type=str(liq),
        fee_bps_maker=float(args.fee_bps_maker),
        fee_bps_taker=float(args.fee_bps_taker),
        slippage_bps_limit=float(args.slippage_bps_limit),
        slippage_bps_market=float(args.slippage_bps_market),
    )

    row.update(
        {
            "exec_filled": 1,
            "exec_valid_for_metrics": int(exit_res.get("valid_for_metrics", 0)),
            "exec_sl_hit": int(exit_res.get("sl_hit", 0)),
            "exec_tp_hit": int(exit_res.get("tp_hit", 0)),
            "exec_pnl_net_pct": float(cost.get("pnl_net_pct", np.nan)),
            "exec_pnl_gross_pct": float(cost.get("pnl_gross_pct", np.nan)),
            "exec_fill_liquidity_type": str(liq),
            "exec_fill_delay_min": float((fill_time - pd.to_datetime(ctx.signal_time, utc=True)).total_seconds() / 60.0)
            if pd.notna(fill_time)
            else float("nan"),
            "exec_mae_pct": float(exit_res.get("mae_pct", np.nan)),
            "exec_mfe_pct": float(exit_res.get("mfe_pct", np.nan)),
            "exec_skip_reason": "",
            "exec_exit_reason": str(exit_res.get("exit_reason", "")),
            "exec_entry_time": str(fill_time) if pd.notna(fill_time) else "",
            "exec_exit_time": str(exit_res.get("exit_time", "")),
            "exec_entry_price": float(fill_price),
            "exec_exit_price": float(exit_res.get("exit_price", np.nan)),
            "exec_same_bar_hit": int(exit_res.get("same_bar_hit", 0)),
            "exec_invalid_stop_geometry": int(exit_res.get("invalid_stop_geometry", 0)),
            "exec_invalid_tp_geometry": int(exit_res.get("invalid_tp_geometry", 0)),
            "exec_entry_type": str(fill_type),
            "missing_slice_flag": int(max(int(row.get("missing_slice_flag", 0)), int(exit_res.get("missing_slice_flag", 0)))),
        }
    )
    return row


def evaluate_reference_bundle(baseline_df: pd.DataFrame) -> Dict[str, Any]:
    x = baseline_df.copy()
    if "split_id" in x.columns:
        x = x[to_num(x["split_id"]).fillna(-1).astype(int) >= 0].copy()
    if x.empty:
        x = baseline_df.copy()
    b = ga_exec._rollup_mode(x, "baseline")  # pylint: disable=protected-access
    split_rows: List[Dict[str, Any]] = []
    for split_id, g in x.groupby("split_id", dropna=False):
        r = ga_exec._rollup_mode(g, "baseline")  # pylint: disable=protected-access
        split_rows.append(
            {
                "split_id": int(split_id),
                "signals_total": int(r["signals_total"]),
                "baseline_mean_expectancy_net": float(r["mean_expectancy_net"]),
                "exec_mean_expectancy_net": float(r["mean_expectancy_net"]),
                "delta_expectancy_exec_minus_baseline": 0.0,
                "cvar_improve_ratio": 0.0,
                "maxdd_improve_ratio": 0.0,
            }
        )
    split_df = pd.DataFrame(split_rows).sort_values("split_id").reset_index(drop=True)
    return {
        "signal_rows_df": x.copy(),
        "split_rows_df": split_df,
        "metrics": {
            "valid_for_ranking": 1,
            "invalid_reason": "",
            "overall_exec_expectancy_net": float(b["mean_expectancy_net"]),
            "overall_delta_expectancy_exec_minus_baseline": 0.0,
            "overall_cvar_improve_ratio": 0.0,
            "overall_maxdd_improve_ratio": 0.0,
            "overall_entry_rate": float(b["entry_rate"]),
            "overall_entries_valid": int(b["entries_valid"]),
            "overall_exec_taker_share": float(b["taker_share"]),
            "overall_exec_median_fill_delay_min": float(b["median_fill_delay_min"]),
            "overall_exec_p95_fill_delay_min": float(b["p95_fill_delay_min"]),
            "min_split_delta": 0.0,
            "route_pass": 1,
            "center_route_delta": 0.0,
            "center_route_valid": 1,
        },
    }


def evaluate_model_a_variant(
    *,
    bundle: ga_exec.SymbolBundle,
    baseline_df: pd.DataFrame,
    cfg: Dict[str, Any],
    one_h: OneHMarket,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if float(cfg["fallback_delay_min"]) > float(cfg["max_fill_delay_min"]):
        raise RuntimeError(
            f"{cfg['candidate_id']} invalid config: fallback_delay_min>{cfg['max_fill_delay_min']}"
        )

    base_map = {str(r["signal_id"]): r for _, r in baseline_df.iterrows()}
    split_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    invalid_parts: List[str] = []

    for sp in bundle.splits:
        idx0 = int(sp["test_start"])
        idx1 = int(sp["test_end"])
        split_signal_rows: List[Dict[str, Any]] = []
        for ctx in bundle.contexts[idx0:idx1]:
            sid = str(ctx.signal_id)
            if sid not in base_map:
                raise KeyError(f"Missing baseline row for signal_id={sid}")
            row = simulate_model_a_signal(
                ctx=ctx,
                baseline_row=base_map[sid],
                cfg=cfg,
                one_h=one_h,
                args=args,
            )
            split_signal_rows.append(row)
        df_split = pd.DataFrame(split_signal_rows)
        split_roll = ga_exec._aggregate_rows(df_split)  # pylint: disable=protected-access
        split_rows.append(
            {
                "split_id": int(sp["split_id"]),
                "signals_total": int(split_roll["exec"]["signals_total"]),
                "baseline_mean_expectancy_net": float(split_roll["baseline"]["mean_expectancy_net"]),
                "exec_mean_expectancy_net": float(split_roll["exec"]["mean_expectancy_net"]),
                "delta_expectancy_exec_minus_baseline": float(split_roll["delta_expectancy_exec_minus_baseline"]),
                "cvar_improve_ratio": float(split_roll["cvar_improve_ratio"]),
                "maxdd_improve_ratio": float(split_roll["maxdd_improve_ratio"]),
                "exec_entries_valid": int(split_roll["exec"]["entries_valid"]),
                "exec_entry_rate": float(split_roll["exec"]["entry_rate"]),
            }
        )
        all_rows.extend(split_signal_rows)

    df_all = pd.DataFrame(all_rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    df_split = pd.DataFrame(split_rows).sort_values("split_id").reset_index(drop=True)
    overall = ga_exec._aggregate_rows(df_all)  # pylint: disable=protected-access
    b = overall["baseline"]
    e = overall["exec"]

    signals_total = int(e["signals_total"])
    entries_valid = int(e["entries_valid"])
    min_trades_symbol = max(
        int(args.hard_min_trades_symbol),
        int(np.ceil(float(args.hard_min_trade_frac_symbol) * max(1, signals_total))),
    )
    min_trades_overall = max(
        int(args.hard_min_trades_overall),
        int(np.ceil(float(args.hard_min_trade_frac_overall) * max(1, signals_total))),
    )

    pass_symbol_entry_rate = int(np.isfinite(e["entry_rate"]) and e["entry_rate"] >= float(args.hard_min_entry_rate_symbol))
    pass_symbol_trade_count = int(entries_valid >= min_trades_symbol)
    pass_overall_entry_rate = int(np.isfinite(e["entry_rate"]) and e["entry_rate"] >= float(args.hard_min_entry_rate_overall))
    pass_overall_trade_count = int(entries_valid >= min_trades_overall)
    pass_taker_share = int(np.isfinite(e["taker_share"]) and e["taker_share"] <= float(args.hard_max_taker_share))
    pass_median_delay = int(
        np.isfinite(e["median_fill_delay_min"]) and e["median_fill_delay_min"] <= float(args.hard_max_median_fill_delay_min)
    )
    pass_p95_delay = int(
        np.isfinite(e["p95_fill_delay_min"]) and e["p95_fill_delay_min"] <= float(args.hard_max_p95_fill_delay_min)
    )
    missing_slice_rate = float(to_num(df_all.get("missing_slice_flag", 0)).fillna(0).mean()) if not df_all.empty else float("nan")
    pass_missing = int(np.isfinite(missing_slice_rate) and missing_slice_rate <= float(args.hard_max_missing_slice_rate))
    req = [
        float(e["mean_expectancy_net"]),
        float(e["cvar_5"]),
        float(e["max_drawdown"]),
        float(e["entry_rate"]),
        float(e["taker_share"]),
        float(e["median_fill_delay_min"]),
        float(e["p95_fill_delay_min"]),
    ]
    pass_nan = int(all(np.isfinite(v) for v in req))
    split_delta = to_num(df_split.get("delta_expectancy_exec_minus_baseline", pd.Series(dtype=float)))
    pass_split = int((not df_split.empty) and split_delta.notna().all())
    min_split_delta = float(split_delta.min()) if not split_delta.empty else float("nan")

    if pass_symbol_entry_rate == 0:
        invalid_parts.append(f"{bundle.symbol}:entry_rate")
    if pass_symbol_trade_count == 0:
        invalid_parts.append(f"{bundle.symbol}:trades<{min_trades_symbol}")
    if pass_overall_entry_rate == 0:
        invalid_parts.append("overall:entry_rate")
    if pass_overall_trade_count == 0:
        invalid_parts.append(f"overall:trades<{min_trades_overall}")
    if pass_taker_share == 0:
        invalid_parts.append(f"{bundle.symbol}:taker_share")
    if pass_median_delay == 0:
        invalid_parts.append(f"{bundle.symbol}:median_fill_delay")
    if pass_p95_delay == 0:
        invalid_parts.append(f"{bundle.symbol}:p95_fill_delay")
    if pass_missing == 0:
        invalid_parts.append(f"{bundle.symbol}:missing_slice_rate>{float(args.hard_max_missing_slice_rate):.4f}")
    if pass_nan == 0:
        invalid_parts.append(f"{bundle.symbol}:nan_or_inf")
    if pass_split == 0:
        invalid_parts.append("split_metrics_missing_or_nan")

    valid_for_ranking = int(
        pass_symbol_entry_rate == 1
        and pass_symbol_trade_count == 1
        and pass_overall_entry_rate == 1
        and pass_overall_trade_count == 1
        and pass_taker_share == 1
        and pass_median_delay == 1
        and pass_p95_delay == 1
        and pass_missing == 1
        and pass_nan == 1
        and pass_split == 1
    )
    invalid_reason = "|".join(sorted(set(invalid_parts)))

    return {
        "signal_rows_df": df_all,
        "split_rows_df": df_split,
        "metrics": {
            "valid_for_ranking": int(valid_for_ranking),
            "invalid_reason": str(invalid_reason),
            "overall_exec_expectancy_net": float(e["mean_expectancy_net"]),
            "overall_delta_expectancy_exec_minus_baseline": float(overall["delta_expectancy_exec_minus_baseline"]),
            "overall_cvar_improve_ratio": float(overall["cvar_improve_ratio"]),
            "overall_maxdd_improve_ratio": float(overall["maxdd_improve_ratio"]),
            "overall_entry_rate": float(e["entry_rate"]),
            "overall_entries_valid": int(entries_valid),
            "overall_exec_taker_share": float(e["taker_share"]),
            "overall_exec_median_fill_delay_min": float(e["median_fill_delay_min"]),
            "overall_exec_p95_fill_delay_min": float(e["p95_fill_delay_min"]),
            "min_split_delta": float(min_split_delta),
            "missing_slice_rate": float(missing_slice_rate),
            "overall_signals_total": int(signals_total),
            "overall_min_trades_required": int(min_trades_overall),
            "baseline_exec_expectancy_net": float(b["mean_expectancy_net"]),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Model A audit: frozen 1h exits with 3m entry-only execution")
    ap.add_argument("--seed", type=int, default=20260228)
    ap.add_argument("--phase-r-dir", default=str(PHASER_DIR_DEFAULT))
    ap.add_argument("--outdir", default="reports/execution_layer")
    args_cli = ap.parse_args()

    run_root = (PROJECT_ROOT / args_cli.outdir).resolve()
    run_dir = run_root / f"PHASEA_MODEL_A_AUDIT_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    subset_path = Path(nx.LOCKED["representative_subset_csv"]).resolve()
    phase_r_dir = Path(args_cli.phase_r_dir).resolve()
    exec_args = nx.build_exec_args(signals_csv=subset_path, seed=int(args_cli.seed))
    lock_info = ga_exec._validate_and_lock_frozen_artifacts(args=exec_args, run_dir=run_dir)  # pylint: disable=protected-access
    if int(lock_info.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("Frozen contract validation failed")

    bundles, _load_meta = ga_exec._prepare_bundles(exec_args)  # pylint: disable=protected-access
    if not bundles:
        raise RuntimeError("No bundles prepared under locked contract")
    base_bundle = bundles[0]
    one_h = load_1h_market(base_bundle.symbol)
    fee = phasec_bt.FeeModel(
        fee_bps_maker=float(exec_args.fee_bps_maker),
        fee_bps_taker=float(exec_args.fee_bps_taker),
        slippage_bps_limit=float(exec_args.slippage_bps_limit),
        slippage_bps_market=float(exec_args.slippage_bps_market),
    )

    route_bundles, route_examples_df, route_feas_df, route_meta = phase_r.build_support_feasible_route_family(
        base_bundle=base_bundle,
        args=exec_args,
        coverage_frac=0.60,
    )
    route_match_flag, route_mismatches = route_examples_match(
        actual=route_examples_df,
        expected_path=phase_r_dir / "phaseR1_route_examples.csv",
    )

    baseline_full = build_1h_reference_rows(
        bundle=base_bundle,
        fee=fee,
        exec_horizon_hours=float(exec_args.exec_horizon_hours),
    )
    baseline_routes: Dict[str, pd.DataFrame] = {}
    for rid, bundle in route_bundles.items():
        baseline_routes[rid] = build_1h_reference_rows(
            bundle=bundle,
            fee=fee,
            exec_horizon_hours=float(exec_args.exec_horizon_hours),
        )

    phase_e2_manifest = subset_path.parent / "run_manifest.json"
    phase_e2_meta: Dict[str, Any] = {}
    if phase_e2_manifest.exists():
        try:
            phase_e2_meta = json.loads(phase_e2_manifest.read_text(encoding="utf-8"))
        except Exception:
            phase_e2_meta = {}

    hybrid_exit_mixing = 1
    forbidden_exit_knobs = [
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

    a1_lines = [
        "# A1 Model A Contract Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Frozen subset: `{subset_path}`",
        f"- Phase R route source: `{phase_r_dir}`",
        f"- Freeze lock pass: `{int(lock_info.get('freeze_lock_pass', 0))}`",
        f"- 1h signal source file (from Phase E2 if available): `{phase_e2_meta.get('source_signal_csv', 'unavailable')}`",
        "- 1h reference engine used for this audit: `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference`.",
        "- 1h reference entry semantics:",
        "  - signal candle = `signal_time` from the frozen representative subset",
        "  - action candle = first 1h bar with timestamp `>= signal_time`",
        "  - TP/SL = `signal tp_mult/sl_mult` applied to the 1h action-candle entry price",
        "  - if SL and TP are both touched in the same 1h bar, SL wins",
        "  - if neither is hit, exit at the last 1h close before the fixed horizon",
        "- Current hybrid downstream evaluator: `src/execution/ga_exec_3m_opt.py::_simulate_candidate_signal`.",
        "- Hybrid exit override evidence:",
        "  - it calls `_simulate_dynamic_exit_long` after entry fill, so exit logic is re-simulated on 3m bars",
        "  - the hybrid evaluator exposes downstream exit knobs such as `time_stop_min`, `break_even_*`, `trailing_*`, and `partial_take_*`",
        f"- Prior hybrid branch mixes execution and exits: `{hybrid_exit_mixing}`",
        f"- Repaired routes reproduced from code: `{int(route_meta.get('route_count', 0))}` routes",
        f"- Repaired route reproduction matches Phase R artifact: `{route_match_flag}`",
        f"- Route reproduction mismatches (if any): `{route_mismatches}`",
        "",
        "## Repaired Route Feasibility",
        "",
        markdown_table(
            route_feas_df,
            [
                "route_id",
                "route_signal_count",
                "wf_test_signal_count",
                "hard_min_trades_overall",
                "headroom_vs_overall_gate",
                "route_trade_gates_reachable",
            ],
            n=8,
        ),
        "",
    ]
    write_text(run_dir / "phaseA1_modelA_contract_report.md", "\n".join(a1_lines) + "\n")

    a1_exit_lines = [
        "# A1 Frozen Exit Semantics",
        "",
        f"- Generated UTC: {utc_now()}",
        "- M0 (`pure 1h reference`) keeps the original 1h signal and 1h exit engine exactly as implemented in `backtest_exec_phasec_sol._simulate_1h_reference`.",
        "- Model A wrapper keeps those exit semantics, but reuses them after a 3m entry fill as follows:",
        "  - 3m decides only whether and when the entry fills",
        "  - after fill, no 3m TP/SL/time-stop/trailing/break-even/partial-take logic is allowed",
        "  - the 1h exit path is simulated on 1h candles only",
        "  - to avoid partial-candle lookahead, the first 1h exit-evaluation candle is the first full 1h bar with timestamp strictly greater than the realized 3m fill time",
        "  - TP/SL levels stay owned by the 1h strategy and are re-anchored to the realized entry price using the same `tp_mult/sl_mult`",
        "  - the exit horizon remains the fixed frozen 1h horizon from the existing contract",
        "  - if neither TP nor SL is hit before the horizon, exit is the final 1h close inside the horizon window",
        "- Forbidden 3m exit knobs are blocked by construction because the wrapper does not accept or route them into execution.",
        "",
        f"- Forbidden downstream exit knobs excluded: `{forbidden_exit_knobs}`",
        "",
    ]
    write_text(run_dir / "phaseA1_frozen_exit_semantics.md", "\n".join(a1_exit_lines))

    wrapper_meta = {
        "generated_utc": utc_now(),
        "freeze_lock_pass": int(lock_info.get("freeze_lock_pass", 0)),
        "hybrid_exit_override_detected": int(hybrid_exit_mixing),
        "wrapper_uses_3m_entry_only": 1,
        "wrapper_uses_1h_exit_only": 1,
        "route_reproduction_match_phaseR": int(route_match_flag),
        "route_reproduction_mismatches": route_mismatches,
        "forbidden_exit_knobs_blocked": forbidden_exit_knobs,
    }
    json_dump(run_dir / "phaseA2_reproduction_check.json", wrapper_meta)

    a2_lines = [
        "# A2 Model A Wrapper Report",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Core evaluator was not patched; a wrapper was added to isolate Model A without changing the existing hybrid engine.",
        "- Wrapper entry stage:",
        "  - uses the frozen 3m signal slices already prepared by `ga_exec_3m_opt`",
        "  - supports only `entry_mode`, `limit_offset_bps`, `fallback_to_market`, `fallback_delay_min`, and `max_fill_delay_min`",
        "- Wrapper exit stage:",
        "  - bypasses `ga_exec_3m_opt._simulate_dynamic_exit_long` entirely",
        "  - calls 1h-only exit simulation derived from `backtest_exec_phasec_sol._simulate_1h_reference` semantics",
        "  - does not expose `tp_mult`, `sl_mult`, `break_even_*`, `trailing_*`, `partial_take_*`, or `time_stop_min` as candidate knobs",
        "",
        f"- Hybrid evaluator mixes exits: `{hybrid_exit_mixing}`",
        f"- Wrapper preserves 1h exit ownership: `1`",
        "",
    ]
    write_text(run_dir / "phaseA2_modelA_wrapper_report.md", "\n".join(a2_lines))

    variants = build_model_a_variants()
    invalid_hist: Counter[str] = Counter()
    results_rows: List[Dict[str, Any]] = []
    eval_cache_full: Dict[str, Dict[str, Any]] = {}

    ref_eval = evaluate_reference_bundle(baseline_full)
    ref_metrics = ref_eval["metrics"]
    results_rows.append(
        {
            "candidate_id": "M0_1H_REFERENCE",
            "label": "M0 pure original 1h reference",
            "valid_for_ranking": int(ref_metrics["valid_for_ranking"]),
            "invalid_reason": str(ref_metrics["invalid_reason"]),
            "exec_expectancy_net": float(ref_metrics["overall_exec_expectancy_net"]),
            "delta_expectancy_vs_1h_reference": 0.0,
            "cvar_improve_ratio": 0.0,
            "maxdd_improve_ratio": 0.0,
            "entry_rate": float(ref_metrics["overall_entry_rate"]),
            "entries_valid": int(ref_metrics["overall_entries_valid"]),
            "taker_share": float(ref_metrics["overall_exec_taker_share"]),
            "median_fill_delay_min": float(ref_metrics["overall_exec_median_fill_delay_min"]),
            "p95_fill_delay_min": float(ref_metrics["overall_exec_p95_fill_delay_min"]),
            "route_pass": 1,
            "min_subperiod_delta": 0.0,
            "center_route_delta": 0.0,
            "center_route_valid": 1,
            "route_pass_rate": 1.0,
        }
    )

    route_check_rows: List[Dict[str, Any]] = []

    for cfg in variants:
        ev = evaluate_model_a_variant(
            bundle=base_bundle,
            baseline_df=baseline_full,
            cfg=cfg,
            one_h=one_h,
            args=exec_args,
        )
        eval_cache_full[str(cfg["candidate_id"])] = ev
        m = ev["metrics"]

        route_valid_flags: Dict[str, int] = {}
        route_delta: Dict[str, float] = {}
        route_entries: Dict[str, int] = {}
        center_delta = float("nan")
        center_valid = 0
        for rid, rb in route_bundles.items():
            rev = evaluate_model_a_variant(
                bundle=rb,
                baseline_df=baseline_routes[rid],
                cfg=cfg,
                one_h=one_h,
                args=exec_args,
            )
            rm = rev["metrics"]
            route_valid_flags[rid] = int(rm["valid_for_ranking"])
            route_delta[rid] = float(rm["overall_delta_expectancy_exec_minus_baseline"])
            route_entries[rid] = int(rm["overall_entries_valid"])
            if rid == "route_center_60pct":
                center_delta = float(rm["overall_delta_expectancy_exec_minus_baseline"])
                center_valid = int(rm["valid_for_ranking"])
            route_check_rows.append(
                {
                    "candidate_id": str(cfg["candidate_id"]),
                    "route_id": str(rid),
                    "valid_for_ranking": int(rm["valid_for_ranking"]),
                    "delta_expectancy_vs_1h_reference": float(rm["overall_delta_expectancy_exec_minus_baseline"]),
                    "exec_expectancy_net": float(rm["overall_exec_expectancy_net"]),
                    "entries_valid": int(rm["overall_entries_valid"]),
                    "entry_rate": float(rm["overall_entry_rate"]),
                    "taker_share": float(rm["overall_exec_taker_share"]),
                    "route_test_signals": int(phase_r.bundle_test_count(rb)),
                    "min_subperiod_delta": float(rm["min_split_delta"]),
                }
            )

        route_pass = int(
            len(route_valid_flags) > 0
            and all(int(v) == 1 for v in route_valid_flags.values())
            and all(np.isfinite(v) and v > 0.0 for v in route_delta.values())
        )
        route_pass_rate = (
            float(sum(1 for rid in route_valid_flags if route_valid_flags[rid] == 1 and np.isfinite(route_delta[rid]) and route_delta[rid] > 0.0))
            / float(max(1, len(route_valid_flags)))
        )

        results_rows.append(
            {
                "candidate_id": str(cfg["candidate_id"]),
                "label": str(cfg["label"]),
                "valid_for_ranking": int(m["valid_for_ranking"]),
                "invalid_reason": str(m["invalid_reason"]),
                "exec_expectancy_net": float(m["overall_exec_expectancy_net"]),
                "delta_expectancy_vs_1h_reference": float(m["overall_delta_expectancy_exec_minus_baseline"]),
                "cvar_improve_ratio": float(m["overall_cvar_improve_ratio"]),
                "maxdd_improve_ratio": float(m["overall_maxdd_improve_ratio"]),
                "entry_rate": float(m["overall_entry_rate"]),
                "entries_valid": int(m["overall_entries_valid"]),
                "taker_share": float(m["overall_exec_taker_share"]),
                "median_fill_delay_min": float(m["overall_exec_median_fill_delay_min"]),
                "p95_fill_delay_min": float(m["overall_exec_p95_fill_delay_min"]),
                "route_pass": int(route_pass),
                "min_subperiod_delta": float(m["min_split_delta"]),
                "center_route_delta": float(center_delta),
                "center_route_valid": int(center_valid),
                "route_pass_rate": float(route_pass_rate),
            }
        )
        bad = [x for x in str(m["invalid_reason"]).split("|") if x]
        for part in bad:
            invalid_hist[str(part)] += 1

    results_df = pd.DataFrame(results_rows).sort_values(
        ["candidate_id"],
        ascending=[True],
    ).reset_index(drop=True)
    results_df.to_csv(run_dir / "phaseA3_modelA_results.csv", index=False)
    json_dump(run_dir / "phaseA3_invalid_reason_histogram.json", dict(sorted(invalid_hist.items())))

    non_ref = results_df[results_df["candidate_id"] != "M0_1H_REFERENCE"].copy()
    top_non_ref = non_ref.sort_values(
        ["delta_expectancy_vs_1h_reference", "cvar_improve_ratio", "maxdd_improve_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    s3_lines = [
        "# A3 Model A Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Candidate rows: `{len(results_df)}`",
        f"- Non-reference rows: `{len(non_ref)}`",
        "",
        "## Results",
        "",
        markdown_table(
            results_df,
            [
                "candidate_id",
                "valid_for_ranking",
                "exec_expectancy_net",
                "delta_expectancy_vs_1h_reference",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "entry_rate",
                "entries_valid",
                "taker_share",
                "route_pass",
                "center_route_delta",
                "invalid_reason",
            ],
            n=12,
        ),
        "",
    ]
    write_text(run_dir / "phaseA3_modelA_report.md", "\n".join(s3_lines))

    route_checks_df = pd.DataFrame(route_check_rows).sort_values(["candidate_id", "route_id"]).reset_index(drop=True)
    route_checks_df.to_csv(run_dir / "phaseA4_modelA_route_checks.csv", index=False)

    center_rows = route_checks_df[route_checks_df["route_id"].astype(str) == "route_center_60pct"].copy()
    a4_lines = [
        "# A4 Model A Route Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Route rows: `{len(route_checks_df)}`",
        "",
        "## Per-route Checks",
        "",
        markdown_table(
            route_checks_df,
            [
                "candidate_id",
                "route_id",
                "valid_for_ranking",
                "delta_expectancy_vs_1h_reference",
                "entries_valid",
                "entry_rate",
                "taker_share",
                "min_subperiod_delta",
            ],
            n=20,
        ),
        "",
        "## Center Route",
        "",
        markdown_table(
            center_rows,
            [
                "candidate_id",
                "valid_for_ranking",
                "delta_expectancy_vs_1h_reference",
                "entries_valid",
                "entry_rate",
                "taker_share",
                "min_subperiod_delta",
            ],
            n=12,
        ),
        "",
    ]
    write_text(run_dir / "phaseA4_modelA_route_report.md", "\n".join(a4_lines))

    strong = top_non_ref[
        (to_num(top_non_ref["valid_for_ranking"]).fillna(0).astype(int) == 1)
        & (to_num(top_non_ref["delta_expectancy_vs_1h_reference"]) > 0.0)
        & (to_num(top_non_ref["route_pass"]).fillna(0).astype(int) == 1)
        & (to_num(top_non_ref["center_route_valid"]).fillna(0).astype(int) == 1)
        & (to_num(top_non_ref["center_route_delta"]) >= 0.0)
    ].copy()
    weak = top_non_ref[
        (to_num(top_non_ref["valid_for_ranking"]).fillna(0).astype(int) == 1)
        & (to_num(top_non_ref["delta_expectancy_vs_1h_reference"]) >= 0.0)
        & (to_num(top_non_ref["center_route_valid"]).fillna(0).astype(int) == 1)
        & (to_num(top_non_ref["center_route_delta"]) >= 0.0)
    ].copy()

    if int(route_match_flag) != 1:
        classification = "MODEL_A_INFRA_BLOCKED"
        mainline_status = "MODEL_A_INFRA_BLOCKED"
        next_step = "Stop and reconcile the repaired route harness reproduction before trusting Model A conclusions."
        next_prompt = ""
    elif not strong.empty:
        best_delta = float(to_num(strong["delta_expectancy_vs_1h_reference"]).iloc[0])
        if best_delta >= 1e-4:
            classification = "MODEL_A_GO"
            mainline_status = "MODEL_A_GO"
            next_step = "Run a bounded follow-up around the surviving entry-only neighborhood with the 1h exit wrapper unchanged."
        else:
            classification = "MODEL_A_WEAK_GO"
            mainline_status = "MODEL_A_WEAK_GO"
            next_step = "Run a very small bounded confirmation around the surviving entry-only neighborhood and stop on first route regression."
        next_prompt = (
            "ROLE\n"
            "You are in bounded Model A confirmation mode for SOLUSDT.\n\n"
            "MISSION\n"
            "Re-run only the surviving 3m entry-only variants under the frozen 1h signal/exit wrapper and repaired routes. "
            "Do not add any 3m exit controls and stop on first route-center regression.\n"
        )
    elif not weak.empty:
        classification = "MODEL_A_WEAK_GO"
        mainline_status = "MODEL_A_WEAK_GO"
        next_step = "Run a very small bounded confirmation around the weakly surviving entry-only variants and stop on first route regression."
        next_prompt = (
            "ROLE\n"
            "You are in bounded Model A confirmation mode for SOLUSDT.\n\n"
            "MISSION\n"
            "Re-test only the weakly surviving 3m entry-only variants under the same frozen 1h exit wrapper and repaired routes, "
            "with no 3m exit controls and no new search dimensions.\n"
        )
    else:
        classification = "MODEL_A_NO_GO"
        mainline_status = "MODEL_A_NO_GO"
        next_step = "Stop here: once exits are frozen to 1h semantics, 3m entry-only execution does not create a robust additive edge."
        next_prompt = ""

    if next_prompt:
        write_text(run_dir / "ready_to_launch_next_prompt.txt", next_prompt)

    decision_lines = [
        "# A4 Decision",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Classification: `{classification}`",
        f"- Mainline status: `{mainline_status}`",
        f"- Strong survivors: `{len(strong)}`",
        f"- Weak survivors: `{len(weak)}`",
        "- Single best next step:",
        f"  - {next_step}",
        "",
    ]
    write_text(run_dir / "phaseA4_decision.md", "\n".join(decision_lines))

    manifest = {
        "generated_utc": utc_now(),
        "classification": classification,
        "mainline_status": mainline_status,
        "frozen_subset": str(subset_path),
        "phase_r_dir": str(phase_r_dir),
        "git_snapshot": git_snapshot(),
        "freeze_lock": lock_info,
        "route_reproduction_match_phaseR": int(route_match_flag),
        "route_reproduction_mismatches": route_mismatches,
        "variants_tested": ["M0_1H_REFERENCE"] + [str(v["candidate_id"]) for v in variants],
        "forbidden_exit_knobs_blocked": forbidden_exit_knobs,
        "hybrid_exit_override_detected": int(hybrid_exit_mixing),
    }
    json_dump(run_dir / "phaseA_run_manifest.json", manifest)

    result = {
        "furthest_phase": "A4",
        "classification": classification,
        "mainline_status": mainline_status,
        "run_dir": str(run_dir),
        "top_candidate": None if top_non_ref.empty else str(top_non_ref.iloc[0]["candidate_id"]),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
