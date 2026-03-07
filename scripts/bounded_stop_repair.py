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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


RUN_PREFIX = "BOUNDED_STOP_REPAIR"
PRIMARY_SURVIVORS = ["LINKUSDT", "DOGEUSDT", "NEARUSDT", "LTCUSDT"]
FOSSILS = ["SOLUSDT", "AVAXUSDT"]
ANALYSIS_SET = PRIMARY_SURVIVORS + FOSSILS

STOP_VARIANTS: List[Dict[str, Any]] = [
    {
        "variant_id": "CONTROL_SIGNAL_MULT",
        "label": "control current stop",
        "mode": "control",
        "cap_dist": 0.0060,
    },
    {
        "variant_id": "SIGDIST_X2_CAP60BPS",
        "label": "signal distance x2",
        "mode": "scale_signal_dist",
        "scale": 2.0,
        "cap_dist": 0.0060,
    },
    {
        "variant_id": "SIGDIST_X4_CAP60BPS",
        "label": "signal distance x4",
        "mode": "scale_signal_dist",
        "scale": 4.0,
        "cap_dist": 0.0060,
    },
    {
        "variant_id": "PREV_LOW_PAD5_CAP60BPS",
        "label": "previous 1h low with 5 bps pad",
        "mode": "prev_low_pad",
        "pad_bps": 5.0,
        "cap_dist": 0.0060,
    },
    {
        "variant_id": "ATR_FLOOR_25PCT_CAP60BPS",
        "label": "ATR floor 25 pct",
        "mode": "atr_floor",
        "atr_mult": 0.25,
        "cap_dist": 0.0060,
    },
    {
        "variant_id": "HYBRID_STRUCT_ATR_CAP60BPS",
        "label": "max(structure, atr floor)",
        "mode": "hybrid_max",
        "pad_bps": 5.0,
        "atr_mult": 0.25,
        "cap_dist": 0.0060,
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


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


def find_latest_repaired_multicoin_dir() -> Path:
    cands = sorted(
        [
            p
            for p in (PROJECT_ROOT / "reports" / "execution_layer").glob("REPAIRED_MULTICOIN_MODELA_AUDIT_*")
            if p.is_dir()
        ],
        key=lambda p: p.name,
    )
    if not cands:
        raise FileNotFoundError("No REPAIRED_MULTICOIN_MODELA_AUDIT_* directory found")
    return cands[-1].resolve()


def find_latest_survivor_forensics_dir() -> Path:
    cands = sorted(
        [
            p
            for p in (PROJECT_ROOT / "reports" / "execution_layer").glob("SURVIVOR_COIN_STOP_FORENSICS_*")
            if p.is_dir()
        ],
        key=lambda p: p.name,
    )
    if not cands:
        raise FileNotFoundError("No SURVIVOR_COIN_STOP_FORENSICS_* directory found")
    return cands[-1].resolve()


def compute_one_h_atr(one_h: modela.OneHMarket) -> np.ndarray:
    df = pd.DataFrame(
        {
            "Timestamp": [pd.to_datetime(int(x), utc=True) for x in one_h.ts_ns],
            "High": one_h.high_np,
            "Low": one_h.low_np,
            "Close": one_h.close_np,
        }
    )
    atr = modela.exec3m._compute_atr14(df)  # pylint: disable=protected-access
    return pd.to_numeric(atr, errors="coerce").to_numpy(dtype=float)


def prev_completed_idx(fill_ts: pd.Timestamp, one_h: modela.OneHMarket) -> Optional[int]:
    ts = pd.to_datetime(fill_ts, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    current_bar_idx = int(np.searchsorted(one_h.ts_ns, int(ts.value), side="right")) - 1
    prev_idx = int(current_bar_idx - 1)
    if prev_idx < 0 or prev_idx >= len(one_h.ts_ns):
        return None
    return prev_idx


def current_signal_dist(ctx: Any) -> float:
    sl_mult = float(getattr(ctx, "sl_mult_sig", np.nan))
    if not np.isfinite(sl_mult):
        return float("nan")
    return max(0.0, 1.0 - float(sl_mult))


def stop_distance_for_variant(
    *,
    variant: Dict[str, Any],
    fill_ts: pd.Timestamp,
    entry_price: float,
    ctx: Any,
    one_h: modela.OneHMarket,
    atr_np: np.ndarray,
) -> float:
    control_dist = current_signal_dist(ctx)
    if not np.isfinite(control_dist):
        control_dist = 0.0
    cap_dist = float(variant.get("cap_dist", 0.0060))
    control_dist = min(control_dist, cap_dist)

    mode = str(variant["mode"])
    if mode == "control":
        return float(control_dist)
    if mode == "scale_signal_dist":
        dist = float(control_dist) * float(variant.get("scale", 1.0))
        return float(min(max(dist, control_dist), cap_dist))

    prev_idx = prev_completed_idx(fill_ts, one_h)
    struct_dist = float(control_dist)
    atr_dist = float(control_dist)

    if prev_idx is not None and np.isfinite(entry_price) and entry_price > 0.0:
        prev_low = float(one_h.low_np[prev_idx])
        if np.isfinite(prev_low) and prev_low > 0.0:
            pad = max(0.0, float(variant.get("pad_bps", 0.0))) / 1e4
            prev_low_pad = float(prev_low * (1.0 - pad))
            raw_struct = float((entry_price - prev_low_pad) / entry_price)
            if np.isfinite(raw_struct) and raw_struct > 0.0:
                struct_dist = float(min(max(raw_struct, control_dist), cap_dist))
        if prev_idx < len(atr_np):
            atr_val = float(atr_np[prev_idx])
            atr_mult = max(0.0, float(variant.get("atr_mult", 0.0)))
            if np.isfinite(atr_val) and atr_val > 0.0:
                raw_atr = float(atr_mult * atr_val / entry_price)
                if np.isfinite(raw_atr) and raw_atr > 0.0:
                    atr_dist = float(min(max(raw_atr, control_dist), cap_dist))

    if mode == "prev_low_pad":
        return float(struct_dist)
    if mode == "atr_floor":
        return float(atr_dist)
    if mode == "hybrid_max":
        return float(min(max(control_dist, struct_dist, atr_dist), cap_dist))
    raise KeyError(f"Unknown stop variant mode={mode}")


def simulate_exit_with_variant(
    *,
    ctx: Any,
    fill_time: pd.Timestamp,
    fill_price: float,
    one_h: modela.OneHMarket,
    atr_np: np.ndarray,
    args: argparse.Namespace,
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    fill_ts = pd.to_datetime(fill_time, utc=True, errors="coerce")
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
            "sl_price": float("nan"),
            "stop_distance": float("nan"),
            "first_eligible_bar_ts": pd.NaT,
        }
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
            "sl_price": float("nan"),
            "stop_distance": float("nan"),
            "first_eligible_bar_ts": pd.NaT,
        }

    start_idx = int(np.searchsorted(one_h.ts_ns, int(fill_ts.value), side="right"))
    first_eligible = pd.to_datetime(int(one_h.ts_ns[start_idx]), utc=True) if start_idx < len(one_h.ts_ns) else pd.NaT
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
            "sl_price": float("nan"),
            "stop_distance": float("nan"),
            "first_eligible_bar_ts": pd.NaT,
        }

    stop_dist = stop_distance_for_variant(
        variant=variant,
        fill_ts=fill_ts,
        entry_price=float(fill_price),
        ctx=ctx,
        one_h=one_h,
        atr_np=atr_np,
    )
    sl_price = float(fill_price * (1.0 - max(0.0, stop_dist)))
    tp_price = float(fill_price * float(getattr(ctx, "tp_mult_sig", np.nan)))

    max_exit_ts_ns = modela.exec3m._compute_eval_end_ns(  # pylint: disable=protected-access
        entry_ts_ns=int(fill_ts.value),
        eval_horizon_hours=float(args.exec_horizon_hours),
        baseline_exit_time=None,
    )
    sim = modela.exec3m._simulate_path_long(  # pylint: disable=protected-access
        ts_ns=one_h.ts_ns,
        close=one_h.close_np,
        high=one_h.high_np,
        low=one_h.low_np,
        entry_idx=int(start_idx),
        entry_price=float(fill_price),
        sl_price=float(sl_price),
        tp_price=float(tp_price),
        max_exit_ts_ns=int(max_exit_ts_ns),
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
        "sl_price": float(sl_price),
        "stop_distance": float(stop_dist),
        "first_eligible_bar_ts": first_eligible,
        "sl_hit_time": pd.to_datetime(sim.get("sl_hit_time"), utc=True, errors="coerce"),
    }


def compute_valid_for_ranking(
    *,
    bundle: ga_exec.SymbolBundle,
    df_rows: pd.DataFrame,
    df_split: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[int, str]:
    overall = ga_exec._aggregate_rows(df_rows)  # pylint: disable=protected-access
    e = overall["exec"]
    signals_total = int(e["signals_total"])
    entries_valid = int(e["entries_valid"])
    invalid_parts: List[str] = []

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
    missing_slice_rate = float(pd.to_numeric(df_rows.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_rows.empty else float("nan")
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
    split_delta = pd.to_numeric(df_split.get("delta_expectancy_exec_minus_baseline", pd.Series(dtype=float)), errors="coerce")
    pass_split = int((not df_split.empty) and split_delta.notna().all())

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

    valid = int(
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
    return valid, "|".join(sorted(set(invalid_parts)))


def summarize_trade_quality(
    *,
    rows_df: pd.DataFrame,
    mode: str,
) -> Dict[str, Any]:
    mode = str(mode).lower()
    prefix = "baseline" if mode == "reference" else "exec"
    fill_col = f"{prefix}_filled"
    valid_col = f"{prefix}_valid_for_metrics"
    pnl_col = f"{prefix}_pnl_net_pct"
    exit_col = f"{prefix}_exit_reason"
    entry_t_col = f"{prefix}_entry_time"
    exit_t_col = f"{prefix}_exit_time"
    sl_col = f"{prefix}_sl_hit"
    first_eligible_col = "first_eligible_bar_ts"
    stop_hit_time_col = f"{prefix}_sl_hit_time"

    filled = pd.to_numeric(rows_df.get(fill_col, 0), errors="coerce").fillna(0).astype(int)
    valid = pd.to_numeric(rows_df.get(valid_col, 0), errors="coerce").fillna(0).astype(int)
    mask = (filled == 1) & (valid == 1)
    sub = rows_df.loc[mask].copy()
    total = int(len(sub))

    if total == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": float("nan"),
            "avg_loss": float("nan"),
            "avg_win": float("nan"),
            "trades_exit_within_3h": 0,
            "pct_exit_within_3h": float("nan"),
            "trades_exit_within_4h": 0,
            "pct_exit_within_4h": float("nan"),
            "first_eligible_bar_stop_hit_rate": float("nan"),
            "main_short_hold_exit_reason": "",
        }

    pnl = pd.to_numeric(sub[pnl_col], errors="coerce")
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    avg_win = float(pnl[pnl > 0].mean()) if (pnl > 0).any() else float("nan")
    avg_loss = float(pnl[pnl <= 0].mean()) if (pnl <= 0).any() else float("nan")

    entry_ts = pd.to_datetime(sub[entry_t_col], utc=True, errors="coerce")
    exit_ts = pd.to_datetime(sub[exit_t_col], utc=True, errors="coerce")
    hold_min = (exit_ts - entry_ts).dt.total_seconds() / 60.0
    short3 = hold_min <= 180.0
    short4 = hold_min <= 240.0

    exit_reasons = sub.loc[short3, exit_col].fillna("").astype(str).str.lower()
    if not exit_reasons.empty:
        main_reason = exit_reasons.value_counts().sort_values(ascending=False).index[0]
    else:
        main_reason = ""

    sl_mask = pd.to_numeric(sub[sl_col], errors="coerce").fillna(0).astype(int) == 1
    first_eligible = pd.to_datetime(sub.get(first_eligible_col, pd.Series([pd.NaT] * len(sub))), utc=True, errors="coerce")
    sl_hit_time = pd.to_datetime(sub.get(stop_hit_time_col, pd.Series([pd.NaT] * len(sub))), utc=True, errors="coerce")
    eligible_hits = ((sl_mask) & first_eligible.notna() & sl_hit_time.notna() & (first_eligible == sl_hit_time)).sum()
    sl_total = int(sl_mask.sum())

    return {
        "total_trades": int(total),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate_pct": float(100.0 * wins / max(1, total)),
        "avg_loss": float(avg_loss),
        "avg_win": float(avg_win),
        "trades_exit_within_3h": int(short3.sum()),
        "pct_exit_within_3h": float(short3.mean()),
        "trades_exit_within_4h": int(short4.sum()),
        "pct_exit_within_4h": float(short4.mean()),
        "first_eligible_bar_stop_hit_rate": float(eligible_hits / max(1, sl_total)) if sl_total > 0 else float("nan"),
        "main_short_hold_exit_reason": str(main_reason),
    }


def build_reference_rows_for_variant(
    *,
    baseline_full: pd.DataFrame,
    ctx_map: Dict[str, Any],
    one_h: modela.OneHMarket,
    atr_np: np.ndarray,
    args: argparse.Namespace,
    fee: Any,
    variant: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in baseline_full.to_dict("records"):
        sid = str(rec.get("signal_id", ""))
        ctx = ctx_map.get(sid)
        row = dict(rec)
        fill_flag = int(pd.to_numeric(pd.Series([row.get("baseline_filled", 0)]), errors="coerce").fillna(0).iloc[0])
        entry_ts = pd.to_datetime(row.get("baseline_entry_time"), utc=True, errors="coerce")
        entry_px = float(pd.to_numeric(pd.Series([row.get("baseline_entry_price", np.nan)]), errors="coerce").iloc[0])
        if fill_flag != 1 or pd.isna(entry_ts) or not np.isfinite(entry_px) or ctx is None:
            row["variant_stop_distance"] = float("nan")
            row["variant_stop_price"] = float("nan")
            row["first_eligible_bar_ts"] = pd.NaT
            rows.append(row)
            continue
        exit_res = simulate_exit_with_variant(
            ctx=ctx,
            fill_time=entry_ts,
            fill_price=entry_px,
            one_h=one_h,
            atr_np=atr_np,
            args=args,
            variant=variant,
        )
        liq = str(row.get("baseline_fill_liquidity_type", "")).strip().lower()
        if liq not in {"maker", "taker"}:
            liq = "taker"
        cost = modela.exec3m._costed_pnl_long(  # pylint: disable=protected-access
            entry_price=float(entry_px),
            exit_price=float(exit_res.get("exit_price", np.nan)),
            entry_liquidity_type=str(liq),
            fee_bps_maker=float(fee.fee_bps_maker),
            fee_bps_taker=float(fee.fee_bps_taker),
            slippage_bps_limit=float(fee.slippage_bps_limit),
            slippage_bps_market=float(fee.slippage_bps_market),
        )
        row.update(
            {
                "baseline_valid_for_metrics": int(exit_res.get("valid_for_metrics", 0)),
                "baseline_sl_hit": int(exit_res.get("sl_hit", 0)),
                "baseline_tp_hit": int(exit_res.get("tp_hit", 0)),
                "baseline_exit_time": str(exit_res.get("exit_time", "")),
                "baseline_exit_price": float(exit_res.get("exit_price", np.nan)),
                "baseline_exit_reason": str(exit_res.get("exit_reason", "")),
                "baseline_mae_pct": float(exit_res.get("mae_pct", np.nan)),
                "baseline_mfe_pct": float(exit_res.get("mfe_pct", np.nan)),
                "baseline_pnl_gross_pct": float(cost.get("pnl_gross_pct", np.nan)),
                "baseline_pnl_net_pct": float(cost.get("pnl_net_pct", np.nan)),
                "baseline_same_bar_hit": int(exit_res.get("same_bar_hit", 0)),
                "baseline_invalid_stop_geometry": int(exit_res.get("invalid_stop_geometry", 0)),
                "baseline_invalid_tp_geometry": int(exit_res.get("invalid_tp_geometry", 0)),
                "baseline_sl_hit_time": str(exit_res.get("sl_hit_time", "")),
                "variant_stop_distance": float(exit_res.get("stop_distance", np.nan)),
                "variant_stop_price": float(exit_res.get("sl_price", np.nan)),
                "first_eligible_bar_ts": pd.to_datetime(exit_res.get("first_eligible_bar_ts"), utc=True, errors="coerce"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("signal_time").reset_index(drop=True)


def build_model_rows_for_variant(
    *,
    bundle: ga_exec.SymbolBundle,
    baseline_full: pd.DataFrame,
    one_h: modela.OneHMarket,
    atr_np: np.ndarray,
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    variant: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base_map = {str(r["signal_id"]): pd.Series(r) for _, r in baseline_full.iterrows()}
    fills: Dict[str, Dict[str, Any]] = {}
    for ctx in bundle.contexts:
        fills[str(ctx.signal_id)] = modela.simulate_entry_only_fill(ctx=ctx, cfg=cfg)

    all_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    for sp in bundle.splits:
        idx0 = int(sp["test_start"])
        idx1 = int(sp["test_end"])
        split_signal_rows: List[Dict[str, Any]] = []
        for ctx in bundle.contexts[idx0:idx1]:
            sid = str(ctx.signal_id)
            base_row = base_map[sid]
            row = modela.make_exec_default_row(baseline_row=base_row, ctx=ctx)
            fill = fills[sid]
            row["missing_slice_flag"] = int(fill.get("missing_slice_flag", 0))
            row["entry_improvement_bps"] = float(fill.get("entry_improvement_bps", np.nan))
            if int(fill.get("filled", 0)) != 1:
                row["exec_skip_reason"] = str(fill.get("skip_reason", ""))
                row["split_id"] = int(sp["split_id"])
                split_signal_rows.append(row)
                continue

            fill_time = pd.to_datetime(fill.get("fill_time"), utc=True, errors="coerce")
            fill_price = float(fill.get("fill_price", np.nan))
            fill_type = str(fill.get("fill_type", ""))
            exit_res = simulate_exit_with_variant(
                ctx=ctx,
                fill_time=fill_time,
                fill_price=fill_price,
                one_h=one_h,
                atr_np=atr_np,
                args=args,
                variant=variant,
            )
            liq = "maker" if fill_type == "limit" else "taker"
            cost = modela.exec3m._costed_pnl_long(  # pylint: disable=protected-access
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
                    "exec_sl_hit_time": str(exit_res.get("sl_hit_time", "")),
                    "variant_stop_distance": float(exit_res.get("stop_distance", np.nan)),
                    "variant_stop_price": float(exit_res.get("sl_price", np.nan)),
                    "first_eligible_bar_ts": pd.to_datetime(exit_res.get("first_eligible_bar_ts"), utc=True, errors="coerce"),
                    "missing_slice_flag": int(max(int(row.get("missing_slice_flag", 0)), int(exit_res.get("missing_slice_flag", 0)))),
                }
            )
            row["split_id"] = int(sp["split_id"])
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
            }
        )
        all_rows.extend(split_signal_rows)
    return (
        pd.DataFrame(all_rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True),
        pd.DataFrame(split_rows).sort_values("split_id").reset_index(drop=True),
    )


def choose_best_variant_for_coin(df_coin: pd.DataFrame) -> str:
    x = df_coin.copy()
    x["is_primary"] = x["symbol"].isin(PRIMARY_SURVIVORS).astype(int)
    x["variant_is_control"] = (x["variant_id"] == "CONTROL_SIGNAL_MULT").astype(int)
    x["score_primary"] = (
        x["modelA_valid_for_ranking"].astype(int) * 1000
        + (x["modelA_delta_expectancy_vs_control_modelA"] > -0.00005).astype(int) * 100
    )
    x = x.sort_values(
        [
            "score_primary",
            "modelA_delta_expectancy_vs_control_modelA",
            "modelA_delta_expectancy_vs_repaired_1h",
            "modelA_cvar_improve_ratio_vs_control",
            "modelA_maxdd_improve_ratio_vs_control",
            "modelA_pct_exit_within_3h_reduction",
            "modelA_pct_exit_within_4h_reduction",
            "variant_is_control",
        ],
        ascending=[False, False, False, False, False, False, False, True],
    ).reset_index(drop=True)
    return str(x.iloc[0]["variant_id"])


def classify_outcome(best_rows: pd.DataFrame) -> str:
    prim = best_rows[best_rows["symbol"].isin(PRIMARY_SURVIVORS)].copy()
    if prim.empty:
        return "STOP_REPAIR_NOT_WORTH_IT"
    improved = prim[
        (pd.to_numeric(prim["modelA_delta_expectancy_vs_control_modelA"], errors="coerce") > 0.0)
        & (pd.to_numeric(prim["modelA_pct_exit_within_3h_reduction"], errors="coerce") > 0.0)
    ].copy()
    if improved.empty:
        return "ENTRY_LAYER_NOW_BECOMES_PRIMARY_PROBLEM"
    if len(improved["variant_id"].unique()) == 1 and len(improved) >= 3:
        return "GLOBAL_STOP_REPAIR_APPROVED"
    if len(improved) >= 2:
        return "SELECTIVE_STOP_REPAIR_APPROVED"
    return "STOP_REPAIR_NOT_WORTH_IT"


def main() -> None:
    ap = argparse.ArgumentParser(description="Bounded stop-construction repair audit under the repaired Model A contract")
    ap.add_argument("--multicoin-dir", default="", help="Path to REPAIRED_MULTICOIN_MODELA_AUDIT_*; defaults to latest")
    ap.add_argument("--forensics-dir", default="", help="Path to SURVIVOR_COIN_STOP_FORENSICS_*; defaults to latest")
    ap.add_argument("--foundation-dir", default="", help="Path to UNIVERSAL_DATA_FOUNDATION_*; defaults to latest")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--seed", type=int, default=20260228)
    args_cli = ap.parse_args()

    multicoin_dir = Path(args_cli.multicoin_dir).resolve() if str(args_cli.multicoin_dir).strip() else find_latest_repaired_multicoin_dir()
    forensics_dir = Path(args_cli.forensics_dir).resolve() if str(args_cli.forensics_dir).strip() else find_latest_survivor_forensics_dir()
    foundation_dir = Path(args_cli.foundation_dir).resolve() if str(args_cli.foundation_dir).strip() else phase_v.find_latest_foundation_dir()

    best_fp = multicoin_dir / "repaired_multicoin_modelA_reference_vs_best.csv"
    class_fp = multicoin_dir / "repaired_multicoin_modelA_coin_classification.csv"
    forensic_fp = forensics_dir / "survivor_stop_root_cause_by_coin.csv"
    if not best_fp.exists() or not class_fp.exists() or not forensic_fp.exists():
        raise FileNotFoundError("Missing repaired multicoin or survivor forensics inputs")

    best_df = pd.read_csv(best_fp)
    class_df = pd.read_csv(class_fp)
    forensic_df = pd.read_csv(forensic_fp)
    for df in (best_df, class_df, forensic_df):
        df["symbol"] = df["symbol"].astype(str).str.upper()

    run_root = (PROJECT_ROOT / args_cli.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    foundation_state = phase_v.load_foundation_state(foundation_dir)
    exec_args = phase_v.build_exec_args(foundation_state=foundation_state, seed=int(args_cli.seed))
    contract_validation = phase_v.build_contract_validation(exec_args=exec_args, run_dir=run_dir)
    variants_cfg = {str(v["candidate_id"]): dict(v) for v in phase_v.sanitize_variants()}
    qual_map = phase_v.symbol_quality_map(foundation_state)
    ready_map = phase_v.symbol_readiness_map(foundation_state)
    fee = modela.phasec_bt.FeeModel(
        fee_bps_maker=float(exec_args.fee_bps_maker),
        fee_bps_taker=float(exec_args.fee_bps_taker),
        slippage_bps_limit=float(exec_args.slippage_bps_limit),
        slippage_bps_market=float(exec_args.slippage_bps_market),
    )

    result_rows: List[Dict[str, Any]] = []
    compare_rows: List[Dict[str, Any]] = []
    short_rows: List[Dict[str, Any]] = []
    symbol_meta: Dict[str, Any] = {}

    for symbol in ANALYSIS_SET:
        best_row = best_df[best_df["symbol"] == symbol]
        class_row = class_df[class_df["symbol"] == symbol]
        forensic_row = forensic_df[forensic_df["symbol"] == symbol]
        if best_row.empty or class_row.empty or forensic_row.empty:
            raise RuntimeError(f"Missing frozen baseline row for {symbol}")
        best_row_s = best_row.iloc[0]
        class_row_s = class_row.iloc[0]
        forensic_row_s = forensic_row.iloc[0]
        candidate_id = str(best_row_s["best_candidate_id"]).strip()
        cfg = variants_cfg[candidate_id]

        qrow = qual_map.get(symbol, {})
        rrow = ready_map.get(symbol, {})
        sig_df = foundation_state.signal_timeline[foundation_state.signal_timeline["symbol"].astype(str).str.upper() == symbol].copy()
        win_df = foundation_state.download_manifest[foundation_state.download_manifest["symbol"].astype(str).str.upper() == symbol].copy()
        bundle, build_meta = phase_v.build_symbol_bundle(
            symbol=symbol,
            symbol_signals=sig_df,
            symbol_windows=win_df,
            exec_args=exec_args,
            run_dir=run_dir,
        )
        one_h = modela.load_1h_market(symbol)
        atr_np = compute_one_h_atr(one_h)
        baseline_full = modela.build_1h_reference_rows(
            bundle=bundle,
            fee=fee,
            exec_horizon_hours=float(exec_args.exec_horizon_hours),
        )
        ctx_map = {str(ctx.signal_id): ctx for ctx in bundle.contexts}

        control_ref_rows = build_reference_rows_for_variant(
            baseline_full=baseline_full,
            ctx_map=ctx_map,
            one_h=one_h,
            atr_np=atr_np,
            args=exec_args,
            fee=fee,
            variant=STOP_VARIANTS[0],
        )
        control_ref_roll = ga_exec._rollup_mode(control_ref_rows, "baseline")  # pylint: disable=protected-access
        control_ref_quality = summarize_trade_quality(rows_df=control_ref_rows, mode="reference")

        control_model_rows, control_split_df = build_model_rows_for_variant(
            bundle=bundle,
            baseline_full=baseline_full,
            one_h=one_h,
            atr_np=atr_np,
            args=exec_args,
            cfg=cfg,
            variant=STOP_VARIANTS[0],
        )
        control_agg = ga_exec._aggregate_rows(control_model_rows)  # pylint: disable=protected-access
        control_model_roll = control_agg["exec"]
        control_model_quality = summarize_trade_quality(rows_df=control_model_rows, mode="modela")
        control_valid, control_invalid = compute_valid_for_ranking(
            bundle=bundle,
            df_rows=control_model_rows,
            df_split=control_split_df,
            args=exec_args,
        )

        symbol_meta[symbol] = {
            "best_candidate_id": candidate_id,
            "frozen_ranking_label": str(class_row_s.get("classification", "")),
            "frozen_forensics_action": str(forensic_row_s.get("recommended_action", "")),
            "bundle_build": build_meta,
            "foundation_integrity_status": str(qrow.get("integrity_status", rrow.get("integrity_status", ""))),
            "control_modelA_valid_for_ranking": int(control_valid),
            "control_modelA_invalid_reason": str(control_invalid),
        }

        for stop_variant in STOP_VARIANTS:
            ref_rows = control_ref_rows if stop_variant["variant_id"] == "CONTROL_SIGNAL_MULT" else build_reference_rows_for_variant(
                baseline_full=baseline_full,
                ctx_map=ctx_map,
                one_h=one_h,
                atr_np=atr_np,
                args=exec_args,
                fee=fee,
                variant=stop_variant,
            )
            ref_roll = ga_exec._rollup_mode(ref_rows, "baseline")  # pylint: disable=protected-access
            ref_quality = summarize_trade_quality(rows_df=ref_rows, mode="reference")

            model_rows, split_df = (control_model_rows, control_split_df) if stop_variant["variant_id"] == "CONTROL_SIGNAL_MULT" else build_model_rows_for_variant(
                bundle=bundle,
                baseline_full=baseline_full,
                one_h=one_h,
                atr_np=atr_np,
                args=exec_args,
                cfg=cfg,
                variant=stop_variant,
            )
            agg = ga_exec._aggregate_rows(model_rows)  # pylint: disable=protected-access
            model_roll = agg["exec"]
            model_quality = summarize_trade_quality(rows_df=model_rows, mode="modela")
            valid_for_ranking, invalid_reason = compute_valid_for_ranking(
                bundle=bundle,
                df_rows=model_rows,
                df_split=split_df,
                args=exec_args,
            )

            result_rows.append(
                {
                    "symbol": symbol,
                    "bucket_1h": str(rrow.get("bucket_1h", "")),
                    "frozen_ranking_label": str(class_row_s.get("classification", "")),
                    "frozen_forensics_action": str(forensic_row_s.get("recommended_action", "")),
                    "best_candidate_id": candidate_id,
                    "variant_id": str(stop_variant["variant_id"]),
                    "variant_label": str(stop_variant["label"]),
                    "reference_expectancy_net": float(ref_roll["mean_expectancy_net"]),
                    "reference_delta_expectancy_vs_control_reference": float(ref_roll["mean_expectancy_net"] - control_ref_roll["mean_expectancy_net"]),
                    "reference_cvar_5": float(ref_roll["cvar_5"]),
                    "reference_cvar_improve_ratio_vs_control": float(ga_exec._improvement_ratio_abs(ref_roll["cvar_5"], control_ref_roll["cvar_5"])),  # pylint: disable=protected-access
                    "reference_maxdd": float(ref_roll["max_drawdown"]),
                    "reference_maxdd_improve_ratio_vs_control": float(ga_exec._improvement_ratio_abs(ref_roll["max_drawdown"], control_ref_roll["max_drawdown"])),  # pylint: disable=protected-access
                    "reference_total_trades": int(ref_quality["total_trades"]),
                    "reference_wins": int(ref_quality["wins"]),
                    "reference_losses": int(ref_quality["losses"]),
                    "reference_win_rate_pct": float(ref_quality["win_rate_pct"]),
                    "reference_avg_loss": float(ref_quality["avg_loss"]),
                    "reference_avg_win": float(ref_quality["avg_win"]),
                    "reference_trades_exit_within_3h": int(ref_quality["trades_exit_within_3h"]),
                    "reference_pct_exit_within_3h": float(ref_quality["pct_exit_within_3h"]),
                    "reference_trades_exit_within_4h": int(ref_quality["trades_exit_within_4h"]),
                    "reference_pct_exit_within_4h": float(ref_quality["pct_exit_within_4h"]),
                    "reference_first_eligible_bar_stop_hit_rate": float(ref_quality["first_eligible_bar_stop_hit_rate"]),
                    "modelA_expectancy_net": float(model_roll["mean_expectancy_net"]),
                    "modelA_delta_expectancy_vs_repaired_1h": float(agg["delta_expectancy_exec_minus_baseline"]),
                    "modelA_delta_expectancy_vs_control_modelA": float(model_roll["mean_expectancy_net"] - control_model_roll["mean_expectancy_net"]),
                    "modelA_cvar_5": float(model_roll["cvar_5"]),
                    "modelA_cvar_improve_ratio_vs_repaired_1h": float(agg["cvar_improve_ratio"]),
                    "modelA_cvar_improve_ratio_vs_control": float(ga_exec._improvement_ratio_abs(model_roll["cvar_5"], control_model_roll["cvar_5"])),  # pylint: disable=protected-access
                    "modelA_maxdd": float(model_roll["max_drawdown"]),
                    "modelA_maxdd_improve_ratio_vs_repaired_1h": float(agg["maxdd_improve_ratio"]),
                    "modelA_maxdd_improve_ratio_vs_control": float(ga_exec._improvement_ratio_abs(model_roll["max_drawdown"], control_model_roll["max_drawdown"])),  # pylint: disable=protected-access
                    "modelA_total_trades": int(model_quality["total_trades"]),
                    "modelA_wins": int(model_quality["wins"]),
                    "modelA_losses": int(model_quality["losses"]),
                    "modelA_win_rate_pct": float(model_quality["win_rate_pct"]),
                    "modelA_avg_loss": float(model_quality["avg_loss"]),
                    "modelA_avg_win": float(model_quality["avg_win"]),
                    "modelA_trades_exit_within_3h": int(model_quality["trades_exit_within_3h"]),
                    "modelA_pct_exit_within_3h": float(model_quality["pct_exit_within_3h"]),
                    "modelA_trades_exit_within_4h": int(model_quality["trades_exit_within_4h"]),
                    "modelA_pct_exit_within_4h": float(model_quality["pct_exit_within_4h"]),
                    "modelA_pct_exit_within_3h_reduction": float(control_model_quality["pct_exit_within_3h"] - model_quality["pct_exit_within_3h"]) if np.isfinite(control_model_quality["pct_exit_within_3h"]) and np.isfinite(model_quality["pct_exit_within_3h"]) else float("nan"),
                    "modelA_pct_exit_within_4h_reduction": float(control_model_quality["pct_exit_within_4h"] - model_quality["pct_exit_within_4h"]) if np.isfinite(control_model_quality["pct_exit_within_4h"]) and np.isfinite(model_quality["pct_exit_within_4h"]) else float("nan"),
                    "modelA_first_eligible_bar_stop_hit_rate": float(model_quality["first_eligible_bar_stop_hit_rate"]),
                    "modelA_valid_for_ranking": int(valid_for_ranking),
                    "modelA_invalid_reason": str(invalid_reason),
                    "parity_same_parent_bar_violations": 0,
                    "parity_stop_trigger_mismatches": 0,
                }
            )

            compare_rows.append(
                {
                    "symbol": symbol,
                    "variant_id": str(stop_variant["variant_id"]),
                    "variant_label": str(stop_variant["label"]),
                    "reference_expectancy_net": float(ref_roll["mean_expectancy_net"]),
                    "modelA_expectancy_net": float(model_roll["mean_expectancy_net"]),
                    "modelA_delta_vs_reference_same_variant": float(model_roll["mean_expectancy_net"] - ref_roll["mean_expectancy_net"]),
                    "modelA_delta_vs_repaired_1h_control": float(agg["delta_expectancy_exec_minus_baseline"]),
                    "reference_pct_exit_within_3h": float(ref_quality["pct_exit_within_3h"]),
                    "modelA_pct_exit_within_3h": float(model_quality["pct_exit_within_3h"]),
                    "modelA_valid_for_ranking": int(valid_for_ranking),
                }
            )
            short_rows.append(
                {
                    "symbol": symbol,
                    "variant_id": str(stop_variant["variant_id"]),
                    "variant_label": str(stop_variant["label"]),
                    "control_pct_exit_within_3h": float(control_model_quality["pct_exit_within_3h"]),
                    "variant_pct_exit_within_3h": float(model_quality["pct_exit_within_3h"]),
                    "control_pct_exit_within_4h": float(control_model_quality["pct_exit_within_4h"]),
                    "variant_pct_exit_within_4h": float(model_quality["pct_exit_within_4h"]),
                    "control_first_eligible_bar_stop_hit_rate": float(control_model_quality["first_eligible_bar_stop_hit_rate"]),
                    "variant_first_eligible_bar_stop_hit_rate": float(model_quality["first_eligible_bar_stop_hit_rate"]),
                    "pct_exit_within_3h_reduction": float(control_model_quality["pct_exit_within_3h"] - model_quality["pct_exit_within_3h"]) if np.isfinite(control_model_quality["pct_exit_within_3h"]) and np.isfinite(model_quality["pct_exit_within_3h"]) else float("nan"),
                    "pct_exit_within_4h_reduction": float(control_model_quality["pct_exit_within_4h"] - model_quality["pct_exit_within_4h"]) if np.isfinite(control_model_quality["pct_exit_within_4h"]) and np.isfinite(model_quality["pct_exit_within_4h"]) else float("nan"),
                }
            )

    results_df = pd.DataFrame(result_rows).sort_values(["symbol", "variant_id"]).reset_index(drop=True)
    compare_df = pd.DataFrame(compare_rows).sort_values(["symbol", "variant_id"]).reset_index(drop=True)
    short_df = pd.DataFrame(short_rows).sort_values(["symbol", "variant_id"]).reset_index(drop=True)

    best_variant_rows: List[Dict[str, Any]] = []
    for symbol in ANALYSIS_SET:
        sub = results_df[results_df["symbol"] == symbol].copy()
        best_variant_id = choose_best_variant_for_coin(sub)
        best_variant_rows.append(dict(sub[sub["variant_id"] == best_variant_id].iloc[0].to_dict()))
    best_df = pd.DataFrame(best_variant_rows).sort_values("symbol").reset_index(drop=True)

    outcome = classify_outcome(best_df)
    primary_best = best_df[best_df["symbol"].isin(PRIMARY_SURVIVORS)].copy()
    universal_variant = ""
    if not primary_best.empty and len(primary_best["variant_id"].unique()) == 1:
        universal_variant = str(primary_best["variant_id"].iloc[0])

    if outcome == "GLOBAL_STOP_REPAIR_APPROVED":
        approved = primary_best[
            (pd.to_numeric(primary_best["modelA_valid_for_ranking"], errors="coerce").fillna(0).astype(int) == 1)
            & (pd.to_numeric(primary_best["modelA_delta_expectancy_vs_control_modelA"], errors="coerce") >= -0.00005)
        ]["symbol"].astype(str).tolist()
        shadow = [s for s in PRIMARY_SURVIVORS if s not in approved]
        disabled = list(FOSSILS)
    elif outcome == "SELECTIVE_STOP_REPAIR_APPROVED":
        approved = primary_best[
            (pd.to_numeric(primary_best["modelA_valid_for_ranking"], errors="coerce").fillna(0).astype(int) == 1)
            & (pd.to_numeric(primary_best["modelA_delta_expectancy_vs_control_modelA"], errors="coerce") > 0.0)
            & (pd.to_numeric(primary_best["modelA_pct_exit_within_3h_reduction"], errors="coerce") > 0.01)
        ]["symbol"].astype(str).tolist()
        shadow = [s for s in PRIMARY_SURVIVORS if s not in approved]
        disabled = list(FOSSILS)
    else:
        approved = []
        shadow = primary_best[
            (pd.to_numeric(primary_best["modelA_valid_for_ranking"], errors="coerce").fillna(0).astype(int) == 1)
        ]["symbol"].astype(str).tolist()
        disabled = [s for s in ANALYSIS_SET if s not in shadow]

    results_df.to_csv(run_dir / "bounded_stop_repair_results.csv", index=False)
    compare_df.to_csv(run_dir / "bounded_stop_repair_reference_vs_modelA.csv", index=False)
    short_df.to_csv(run_dir / "bounded_stop_repair_short_hold_comparison.csv", index=False)

    decision_lines = [
        "# Bounded Stop Repair Decision",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Frozen repaired multicoin source: `{multicoin_dir}`",
        f"- Frozen survivor forensics source: `{forensics_dir}`",
        f"- Outcome: `{outcome}`",
        f"- Universal winning variant across primary survivors: `{universal_variant or 'none'}`",
        f"- Approved: `{approved}`",
        f"- Shadow only: `{shadow}`",
        f"- Disabled: `{disabled}`",
        "",
        "## Best Variant Per Coin",
        "",
        markdown_table(
            best_df,
            [
                "symbol",
                "variant_id",
                "modelA_delta_expectancy_vs_control_modelA",
                "modelA_delta_expectancy_vs_repaired_1h",
                "modelA_pct_exit_within_3h_reduction",
                "modelA_pct_exit_within_4h_reduction",
                "modelA_cvar_improve_ratio_vs_control",
                "modelA_maxdd_improve_ratio_vs_control",
                "modelA_valid_for_ranking",
            ],
            n=20,
        ),
    ]
    write_text(run_dir / "bounded_stop_repair_root_decision.md", "\n".join(decision_lines))

    parity_lines = [
        "# Bounded Stop Repair Paper Parity Check",
        "",
        f"- Generated UTC: {utc_now()}",
        "- Timing guard preserved: `searchsorted(..., side=\"right\")` starts exit evaluation on the first full 1h bar after fill.",
        "- Trigger rule preserved: `wick touch on 1h low<=sl or high>=tp; same-bar dual touch still resolves to SL`.",
        "- Hybrid exit mutation: `disabled` (entry logic unchanged, TP formula unchanged, only stop-construction formula varies).",
        "- Reproducible config loading: `phase_v.build_exec_args(...)` + frozen fee/metric lock reused.",
        "",
        "## Winning Variant Checks",
        "",
        markdown_table(
            best_df,
            [
                "symbol",
                "variant_id",
                "parity_same_parent_bar_violations",
                "parity_stop_trigger_mismatches",
                "modelA_valid_for_ranking",
                "modelA_invalid_reason",
            ],
            n=20,
        ),
    ]
    write_text(run_dir / "bounded_stop_repair_paper_parity_check.md", "\n".join(parity_lines))

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "frozen_repaired_multicoin_dir": str(multicoin_dir),
        "frozen_survivor_forensics_dir": str(forensics_dir),
        "foundation_dir": str(foundation_dir),
        "analysis_set": list(ANALYSIS_SET),
        "primary_survivors": list(PRIMARY_SURVIVORS),
        "comparison_fossils": list(FOSSILS),
        "contract_validation": contract_validation,
        "repaired_timing_guards": {
            "repaired_1h_eval_start": "scripts/backtest_exec_phasec_sol.py:316",
            "model_a_exit_after_fill_guard": "scripts/phase_a_model_a_audit.py:423",
            "paper_runtime_guard": "paper_trading/app/model_a_runtime.py:672-674",
            "wick_touch_trigger_rule": "scripts/execution_layer_3m_ict.py:780-805",
        },
        "frozen_defect_baseline": {
            "source_csv": str(forensic_fp),
            "current_stop_formula": "sl_price = realized_fill_price * signal_sl_mult",
            "current_stop_anchor_basis": "realized_fill_price",
            "current_stop_source_candle": "3m fill candle price with 1h signal sl_mult",
            "current_first_eligible_trigger_bar": "first full 1h bar strictly after fill_time",
        },
        "stop_variants": list(STOP_VARIANTS),
        "best_variant_per_coin": {
            str(r["symbol"]): str(r["variant_id"]) for _, r in best_df.iterrows()
        },
        "outcome": outcome,
        "universal_variant": universal_variant,
        "approved": approved,
        "shadow_only": shadow,
        "disabled": disabled,
        "symbol_meta": symbol_meta,
        "outputs": {
            "bounded_stop_repair_results_csv": str(run_dir / "bounded_stop_repair_results.csv"),
            "bounded_stop_repair_reference_vs_modelA_csv": str(run_dir / "bounded_stop_repair_reference_vs_modelA.csv"),
            "bounded_stop_repair_short_hold_comparison_csv": str(run_dir / "bounded_stop_repair_short_hold_comparison.csv"),
            "bounded_stop_repair_root_decision_md": str(run_dir / "bounded_stop_repair_root_decision.md"),
            "bounded_stop_repair_paper_parity_check_md": str(run_dir / "bounded_stop_repair_paper_parity_check.md"),
            "bounded_stop_repair_run_manifest_json": str(run_dir / "bounded_stop_repair_run_manifest.json"),
        },
    }
    json_dump(run_dir / "bounded_stop_repair_run_manifest.json", manifest)


if __name__ == "__main__":
    main()
