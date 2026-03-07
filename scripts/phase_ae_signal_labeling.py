#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


LOCKED = {
    "symbol": "SOLUSDT",
    "representative_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
    "primary_hash": "862c940746de0da984862d95",
    "backup_hash": "992bd371689ba3936f3b4d09",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def to_num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


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


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def session_bucket(ts: pd.Series) -> pd.Series:
    h = pd.to_datetime(ts, utc=True, errors="coerce").dt.hour
    out = pd.Series(index=h.index, dtype=object)
    out[(h >= 0) & (h < 6)] = "00_05"
    out[(h >= 6) & (h < 12)] = "06_11"
    out[(h >= 12) & (h < 18)] = "12_17"
    out[(h >= 18) & (h <= 23)] = "18_23"
    return out.fillna("unknown").astype(str)


def vol_bucket(atr_pct: pd.Series) -> pd.Series:
    x = to_num(atr_pct)
    out = pd.Series(index=x.index, dtype=object)
    out[x < 33.333333] = "low"
    out[(x >= 33.333333) & (x < 66.666667)] = "mid"
    out[x >= 66.666667] = "high"
    return out.fillna("unknown").astype(str)


def ensure_signals_schema(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x["tp_mult"] = to_num(x["tp_mult"])
    x["sl_mult"] = to_num(x["sl_mult"])
    if "atr_percentile_1h" not in x.columns:
        x["atr_percentile_1h"] = np.nan
    if "trend_up_1h" not in x.columns:
        x["trend_up_1h"] = np.nan
    x["atr_percentile_1h"] = to_num(x["atr_percentile_1h"])
    x["trend_up_1h"] = to_num(x["trend_up_1h"])
    x = x.dropna(subset=["signal_id", "signal_time", "tp_mult", "sl_mult"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    return x


def markdown_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return "_(none)_"
    x = df.loc[:, [c for c in cols if c in df.columns]].copy()
    headers = list(x.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in x.itertuples(index=False):
        vals: List[str] = []
        for v in r:
            if isinstance(v, float):
                vals.append(f"{v:.8g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_args(*, signals_csv: Path, seed: int, wf_splits: int = 5, train_ratio: float = 0.70) -> argparse.Namespace:
    parser = ga_exec.build_arg_parser()
    args = parser.parse_args([])
    args.symbol = LOCKED["symbol"]
    args.symbols = ""
    args.rank = 1
    args.signals_csv = str(signals_csv)
    args.max_signals = 1200
    args.walkforward = True
    args.wf_splits = int(wf_splits)
    args.train_ratio = float(train_ratio)
    args.mode = "tight"
    args.workers = 1
    args.seed = int(seed)
    args.pop = 1
    args.gens = 1
    args.execution_config = "configs/execution_configs.yaml"
    args.fee_bps_maker = 2.0
    args.fee_bps_taker = 4.0
    args.slippage_bps_limit = 0.5
    args.slippage_bps_market = 2.0
    args.canonical_fee_model_path = LOCKED["canonical_fee_model"]
    args.canonical_metrics_definition_path = LOCKED["canonical_metrics_definition"]
    args.expected_fee_model_sha256 = LOCKED["expected_fee_sha"]
    args.expected_metrics_definition_sha256 = LOCKED["expected_metrics_sha"]
    args.allow_freeze_hash_mismatch = 0
    return args


def load_exec_pair(exec_root: Path) -> Dict[str, Dict[str, Any]]:
    phasev_dirs = sorted([p for p in exec_root.glob("PHASEV_BRANCHB_PORTABILITY_DD_*") if p.is_dir()], key=lambda p: p.name)
    if not phasev_dirs:
        raise FileNotFoundError("No PHASEV_BRANCHB_PORTABILITY_DD_* directory found")
    for d in reversed(phasev_dirs):
        fp = d / "phaseV_exec_candidates_locked.json"
        if not fp.exists():
            continue
        obj = json.loads(fp.read_text(encoding="utf-8"))
        cands = obj.get("candidates", {})
        out: Dict[str, Dict[str, Any]] = {}
        for eid in ("E1", "E2"):
            if eid not in cands:
                continue
            out[eid] = {
                "exec_choice_id": eid,
                "description": str(cands[eid].get("description", "")),
                "genome_hash": str(cands[eid].get("genome_hash", "")),
                "genome": copy.deepcopy(cands[eid].get("genome", {})),
                "source_run": str(cands[eid].get("source_run", "")),
                "source_phasev_dir": str(d),
            }
        if out.get("E1", {}).get("genome_hash") == LOCKED["primary_hash"] and out.get("E2", {}).get("genome_hash") == LOCKED["backup_hash"]:
            return out
    raise RuntimeError("Could not load locked E1/E2 genomes from PhaseV artifacts")


def attach_signal_meta(sig_rows: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    x = sig_rows.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    feat = signals_df[["signal_id", "signal_time", "tp_mult", "sl_mult", "atr_percentile_1h", "trend_up_1h"]].copy()
    feat["signal_id"] = feat["signal_id"].astype(str)
    feat["signal_time"] = pd.to_datetime(feat["signal_time"], utc=True, errors="coerce")
    out = x.merge(feat, on=["signal_id", "signal_time"], how="left", suffixes=("", "_sig"))
    out["session_bucket"] = session_bucket(out["signal_time"])
    out["vol_bucket"] = vol_bucket(out["atr_percentile_1h"])
    out["trend_bucket"] = np.where(to_num(out["trend_up_1h"]) >= 0.5, "up", "down")
    return out


def evaluate_exact(
    *,
    run_dir: Path,
    signals_df: pd.DataFrame,
    genome: Dict[str, Any],
    seed: int,
    name: str,
    wf_splits: int = 5,
    train_ratio: float = 0.70,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, argparse.Namespace, ga_exec.SymbolBundle]:
    fp = run_dir / f"{name}.csv"
    signals_df.to_csv(fp, index=False)
    args = build_args(signals_csv=fp, seed=seed, wf_splits=wf_splits, train_ratio=train_ratio)
    bundles, _ = ga_exec._prepare_bundles(args)
    if not bundles:
        raise RuntimeError(f"No bundle built for {name}")
    bundle = bundles[0]
    met = ga_exec._evaluate_genome(genome=ga_exec._repair_genome(copy.deepcopy(genome), mode=str(args.mode), repair_hist=None), bundles=[bundle], args=args, detailed=True)
    sig = attach_signal_meta(met["signal_rows_df"], signals_df)
    split_df = met.get("split_rows_df", pd.DataFrame()).copy()
    return met, sig, split_df, args, bundle


def build_preentry_features(bundle: ga_exec.SymbolBundle, genome: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ctx in bundle.contexts:
        ts_ns = np.asarray(ctx.ts_ns, dtype=np.int64)
        o = np.asarray(ctx.open_np, dtype=float)
        h = np.asarray(ctx.high_np, dtype=float)
        l = np.asarray(ctx.low_np, dtype=float)
        c = np.asarray(ctx.close_np, dtype=float)
        atr = np.asarray(ctx.atr_np, dtype=float)

        out: Dict[str, Any] = {
            "signal_id": str(ctx.signal_id),
            "signal_time": pd.to_datetime(ctx.signal_time, utc=True),
            "has_3m_context": int(ts_ns.size > 0),
        }
        if ts_ns.size == 0:
            for k in (
                "pre3m_ret_3bars",
                "pre3m_realized_vol_12",
                "pre3m_atr_z",
                "pre3m_spread_proxy_bps",
                "pre3m_body_bps_abs",
                "pre3m_wick_ratio",
                "pre3m_impulse_atr",
                "pre3m_trend_slope_12",
                "pre3m_accel_6v6",
                "pre3m_upbar_ratio_12",
                "pre3m_close_to_high_dist_bps",
                "pre3m_close_to_low_dist_bps",
            ):
                out[k] = float("nan")
            rows.append(out)
            continue

        sig_idx = int(np.searchsorted(ts_ns, int(ctx.signal_ts_ns), side="left"))
        if sig_idx >= ts_ns.size:
            sig_idx = int(ts_ns.size - 1)
        atr_ref_idx = max(0, sig_idx - 1)
        eps = 1e-12

        # Uses bars up to atr_ref_idx (prior-only), except spread/body proxy at sig_idx mirrors execution-time guard behavior.
        c_prior = c[: atr_ref_idx + 1]
        o_sig = float(o[sig_idx]) if np.isfinite(o[sig_idx]) else float("nan")
        c_sig = float(c[sig_idx]) if np.isfinite(c[sig_idx]) else float("nan")
        h_sig = float(h[sig_idx]) if np.isfinite(h[sig_idx]) else float("nan")
        l_sig = float(l[sig_idx]) if np.isfinite(l[sig_idx]) else float("nan")
        atr_ref = float(atr[atr_ref_idx]) if atr.size else float("nan")

        if c_prior.size >= 4 and np.isfinite(c_prior[-1]) and np.isfinite(c_prior[-4]) and abs(c_prior[-4]) > eps:
            ret3 = float(c_prior[-1] / c_prior[-4] - 1.0)
        else:
            ret3 = float("nan")

        rets = np.diff(np.log(np.clip(c_prior, eps, None))) if c_prior.size >= 2 else np.array([], dtype=float)
        rv12 = float(np.nanstd(rets[-12:], ddof=0)) if rets.size >= 4 else float("nan")

        atr_hist = atr[max(0, atr_ref_idx - 7 * 24 * 20) : atr_ref_idx]
        atr_hist = atr_hist[np.isfinite(atr_hist)]
        if atr_hist.size >= 30 and np.isfinite(atr_ref):
            mu = float(np.nanmean(atr_hist))
            sd = float(np.nanstd(atr_hist))
            atr_z = float((atr_ref - mu) / sd) if sd > eps else 0.0
        else:
            atr_z = float("nan")

        spread_proxy = float((h_sig - l_sig) / max(eps, c_sig) * 1e4) if np.isfinite(h_sig) and np.isfinite(l_sig) and np.isfinite(c_sig) else float("nan")
        body_abs = float(abs(c_sig - o_sig) / max(eps, o_sig) * 1e4) if np.isfinite(c_sig) and np.isfinite(o_sig) else float("nan")
        wick_ratio = float((h_sig - l_sig) / max(eps, abs(c_sig - o_sig))) if np.isfinite(h_sig) and np.isfinite(l_sig) and np.isfinite(c_sig) and np.isfinite(o_sig) else float("nan")
        impulse_atr = float(abs(c_sig - o_sig) / max(eps, atr_ref)) if np.isfinite(c_sig) and np.isfinite(o_sig) and np.isfinite(atr_ref) else float("nan")

        if c_prior.size >= 12:
            y = c_prior[-12:]
            x = np.arange(y.size, dtype=float)
            xm = float(np.mean(x))
            ym = float(np.mean(y))
            cov = float(np.mean((x - xm) * (y - ym)))
            varx = float(np.mean((x - xm) ** 2))
            slope12 = float(cov / varx) if varx > eps else 0.0
            r = np.diff(np.log(np.clip(y, eps, None)))
            accel = float(np.mean(r[-6:]) - np.mean(r[:6])) if r.size >= 11 else float("nan")
            upbar = float(np.mean((y[1:] - y[:-1]) > 0.0)) if y.size >= 2 else float("nan")
        else:
            slope12 = float("nan")
            accel = float("nan")
            upbar = float("nan")

        close_to_high = float((h_sig - c_sig) / max(eps, c_sig) * 1e4) if np.isfinite(h_sig) and np.isfinite(c_sig) else float("nan")
        close_to_low = float((c_sig - l_sig) / max(eps, c_sig) * 1e4) if np.isfinite(l_sig) and np.isfinite(c_sig) else float("nan")

        out.update(
            {
                "pre3m_ret_3bars": ret3,
                "pre3m_realized_vol_12": rv12,
                "pre3m_atr_z": atr_z,
                "pre3m_spread_proxy_bps": spread_proxy,
                "pre3m_body_bps_abs": body_abs,
                "pre3m_wick_ratio": wick_ratio,
                "pre3m_impulse_atr": impulse_atr,
                "pre3m_trend_slope_12": slope12,
                "pre3m_accel_6v6": accel,
                "pre3m_upbar_ratio_12": upbar,
                "pre3m_close_to_high_dist_bps": close_to_high,
                "pre3m_close_to_low_dist_bps": close_to_low,
            }
        )

        # Operational proxies (available at entry decision from current config + pre-entry state).
        mode = str(genome.get("entry_mode", "market"))
        limit_off = float(genome.get("limit_offset_bps", 0.0))
        fallback = int(genome.get("fallback_to_market", 1))
        vol_thr = float(genome.get("vol_threshold", np.nan))
        spread_guard = float(genome.get("spread_guard_bps", np.nan))
        micro_vol_filter = int(genome.get("micro_vol_filter", 0))
        use_quality = int(genome.get("use_signal_quality_gate", 0))
        min_quality = float(genome.get("min_signal_quality_gate", 0.0))

        out["est_limit_distance_bps"] = float(limit_off if mode in {"limit", "hybrid"} else 0.0)
        out["est_fill_window_bars"] = float(max(0, int(math.ceil(float(genome.get("max_fill_delay_min", 0.0)) / 3.0))))
        out["est_fallback_window_bars"] = float(max(0, int(math.ceil(float(genome.get("fallback_delay_min", 0.0)) / 3.0))))
        out["cfg_micro_vol_filter"] = float(micro_vol_filter)
        out["cfg_use_signal_quality_gate"] = float(use_quality)
        out["cfg_min_signal_quality_gate"] = float(min_quality)
        out["cfg_spread_guard_bps"] = float(spread_guard)
        out["cfg_vol_threshold"] = float(vol_thr)

        taker_risk = 0
        if mode == "market":
            taker_risk = 1
        if fallback == 1 and mode in {"limit", "hybrid"} and np.isfinite(spread_proxy) and np.isfinite(spread_guard) and spread_proxy >= 0.9 * spread_guard:
            taker_risk = 1
        if micro_vol_filter == 1 and np.isfinite(atr_z) and np.isfinite(vol_thr) and atr_z >= 0.8 * vol_thr:
            taker_risk = 1
        out["est_taker_risk_proxy"] = float(taker_risk)
        out["est_fill_delay_risk_proxy"] = float(
            int(np.isfinite(spread_proxy) and np.isfinite(spread_guard) and spread_proxy >= 0.8 * spread_guard)
            + int(np.isfinite(atr_z) and atr_z >= 1.0)
        )
        rows.append(out)

    return pd.DataFrame(rows)


def build_trade_labels(df_all: pd.DataFrame) -> pd.DataFrame:
    x = df_all.copy()
    x["entry_taken"] = to_num(x.get("exec_filled", 0)).fillna(0).astype(int)
    x["exec_valid_for_metrics"] = to_num(x.get("exec_valid_for_metrics", 0)).fillna(0).astype(int)
    x["pnl_net_trade_notional_dec"] = to_num(x.get("exec_pnl_net_pct", np.nan))
    x["pnl_gross_trade_notional_dec"] = to_num(x.get("exec_pnl_gross_pct", np.nan))
    x["fee_drag_trade"] = x["pnl_gross_trade_notional_dec"] - x["pnl_net_trade_notional_dec"]
    x["entry_time_utc"] = pd.to_datetime(x.get("exec_entry_time", pd.NaT), utc=True, errors="coerce")
    x["signal_time_utc"] = pd.to_datetime(x.get("signal_time", pd.NaT), utc=True, errors="coerce")
    x["fill_delay_min"] = to_num(x.get("exec_fill_delay_min", np.nan))
    x["taker_flag"] = (x.get("exec_fill_liquidity_type", "").astype(str).str.lower() == "taker").astype(int)
    x["sl_hit_flag"] = to_num(x.get("exec_sl_hit", 0)).fillna(0).astype(int)
    x["split_id"] = to_num(x.get("split_id", np.nan))
    x["exit_reason"] = x.get("exec_exit_reason", "").astype(str)
    x["exec_mae_pct"] = to_num(x.get("exec_mae_pct", np.nan))
    x["exec_mfe_pct"] = to_num(x.get("exec_mfe_pct", np.nan))
    x["entry_for_labels"] = ((x["entry_taken"] == 1) & (x["exec_valid_for_metrics"] == 1) & x["pnl_net_trade_notional_dec"].notna()).astype(int)

    for c in (
        "y_loss",
        "y_tail_loss",
        "y_sl_loss",
        "y_cluster_loss",
        "y_toxic_trade",
        "y_good_trade",
        "y_slow_bleed",
        "y_fee_dominated",
        "prior_loss_streak_len",
        "prior_rolling_loss_rate_5",
        "prior_rolling_loss_rate_10",
        "prior_rolling_loss_rate_20",
        "prior_rolling_tail_count_5",
        "prior_rolling_tail_count_10",
        "prior_rolling_tail_count_20",
    ):
        x[c] = np.nan

    t = x[x["entry_for_labels"] == 1].copy().sort_values(["entry_time_utc", "signal_time_utc", "signal_id"]).reset_index(drop=True)
    if t.empty:
        return x

    t["y_loss"] = (t["pnl_net_trade_notional_dec"] < 0).astype(int)
    tail_cut = float(np.nanquantile(t["pnl_net_trade_notional_dec"].to_numpy(dtype=float), 0.10))
    t["y_tail_loss"] = (t["pnl_net_trade_notional_dec"] <= tail_cut).astype(int)
    t["y_sl_loss"] = ((t["sl_hit_flag"] == 1) & (t["pnl_net_trade_notional_dec"] < 0)).astype(int)
    mae_cut = float(np.nanquantile(t["exec_mae_pct"].dropna().to_numpy(dtype=float), 0.10)) if t["exec_mae_pct"].notna().any() else float("nan")
    t["y_large_ae"] = ((t["exec_mae_pct"] <= mae_cut) & t["exec_mae_pct"].notna()).astype(int) if np.isfinite(mae_cut) else 0

    # Mark cluster losses from loss streaks >=5.
    loss = t["y_loss"].to_numpy(dtype=int)
    cl = np.zeros(len(t), dtype=int)
    i = 0
    while i < len(t):
        if loss[i] == 1:
            s = i
            while i + 1 < len(t) and loss[i + 1] == 1:
                i += 1
            e = i
            if (e - s + 1) >= 5:
                cl[s : e + 1] = 1
        i += 1
    t["y_cluster_loss"] = cl
    t["y_toxic_trade"] = ((t["y_tail_loss"] == 1) | (t["y_cluster_loss"] == 1) | (t["y_large_ae"] == 1)).astype(int)
    t["y_good_trade"] = ((t["pnl_net_trade_notional_dec"] > 0) & (t["y_toxic_trade"] == 0)).astype(int)
    mfe_cut = float(np.nanquantile(t["exec_mfe_pct"].dropna().to_numpy(dtype=float), 0.40)) if t["exec_mfe_pct"].notna().any() else float("nan")
    t["y_slow_bleed"] = ((t["exit_reason"] == "time_stop") & (t["pnl_net_trade_notional_dec"] < 0) & (t["exec_mfe_pct"] <= mfe_cut)).astype(int) if np.isfinite(mfe_cut) else 0
    t["y_fee_dominated"] = ((t["pnl_gross_trade_notional_dec"] > 0) & (t["pnl_net_trade_notional_dec"] < 0)).astype(int)

    # Prior-only context labels.
    loss_hist: List[int] = []
    tail_hist: List[int] = []
    streak = 0
    prior_streak: List[int] = []
    lr5: List[float] = []
    lr10: List[float] = []
    lr20: List[float] = []
    tc5: List[float] = []
    tc10: List[float] = []
    tc20: List[float] = []

    for _, r in t.iterrows():
        prior_streak.append(float(streak))
        arr_loss = np.asarray(loss_hist, dtype=float)
        arr_tail = np.asarray(tail_hist, dtype=float)
        for n, target_loss, target_tail in ((5, lr5, tc5), (10, lr10, tc10), (20, lr20, tc20)):
            if arr_loss.size == 0:
                target_loss.append(float("nan"))
                target_tail.append(float("nan"))
            else:
                target_loss.append(float(np.nanmean(arr_loss[-n:])))
                target_tail.append(float(np.nansum(arr_tail[-n:])))

        cur_loss = int(r["y_loss"])
        cur_tail = int(r["y_tail_loss"])
        loss_hist.append(cur_loss)
        tail_hist.append(cur_tail)
        if cur_loss == 1:
            streak += 1
        else:
            streak = 0

    t["prior_loss_streak_len"] = prior_streak
    t["prior_rolling_loss_rate_5"] = lr5
    t["prior_rolling_loss_rate_10"] = lr10
    t["prior_rolling_loss_rate_20"] = lr20
    t["prior_rolling_tail_count_5"] = tc5
    t["prior_rolling_tail_count_10"] = tc10
    t["prior_rolling_tail_count_20"] = tc20

    join_cols = [
        "signal_id",
        "y_loss",
        "y_tail_loss",
        "y_sl_loss",
        "y_cluster_loss",
        "y_toxic_trade",
        "y_good_trade",
        "y_slow_bleed",
        "y_fee_dominated",
        "prior_loss_streak_len",
        "prior_rolling_loss_rate_5",
        "prior_rolling_loss_rate_10",
        "prior_rolling_loss_rate_20",
        "prior_rolling_tail_count_5",
        "prior_rolling_tail_count_10",
        "prior_rolling_tail_count_20",
    ]
    x = x.merge(t[join_cols], on="signal_id", how="left", suffixes=("", "_lab"))
    # Resolve merges preserving non-entry NaN for labels.
    for c in join_cols:
        if c == "signal_id":
            continue
        if c + "_lab" in x.columns:
            x[c] = x[c + "_lab"]
            x = x.drop(columns=[c + "_lab"])
    return x


def aggregate_metrics_from_dataset(df: pd.DataFrame, pnl_col: str = "pnl_net_trade_notional_dec") -> Dict[str, Any]:
    x = df.copy().reset_index(drop=True)
    x[pnl_col] = to_num(x[pnl_col])
    x["entry_taken"] = to_num(x.get("entry_taken", 0)).fillna(0).astype(int)
    x["entry_for_labels"] = to_num(x.get("entry_for_labels", 0)).fillna(0).astype(int)
    x["split_id"] = to_num(x.get("split_id", np.nan))
    x["taker_flag"] = to_num(x.get("taker_flag", 0)).fillna(0).astype(int)
    x["fill_delay_min"] = to_num(x.get("fill_delay_min", np.nan))

    valid = (x["entry_for_labels"] == 1) & x[pnl_col].notna()
    signals_total = int(len(x))
    entries = int(valid.sum())
    pnl_sig = np.zeros(signals_total, dtype=float)
    pnl_sig[valid.to_numpy(dtype=bool)] = x.loc[valid, pnl_col].to_numpy(dtype=float)

    exp = float(np.mean(pnl_sig)) if signals_total > 0 else float("nan")
    k5 = max(1, int(math.ceil(0.05 * len(pnl_sig)))) if len(pnl_sig) else 1
    cvar = float(np.mean(np.sort(pnl_sig)[:k5])) if len(pnl_sig) else float("nan")
    cum = np.cumsum(np.nan_to_num(pnl_sig, nan=0.0))
    peak = np.maximum.accumulate(cum) if cum.size else np.array([], dtype=float)
    dd = cum - peak if cum.size else np.array([], dtype=float)
    mdd = float(np.nanmin(dd)) if dd.size else float("nan")
    entry_rate = float(entries / max(1, signals_total))

    taker_share = float(np.mean(x.loc[valid, "taker_flag"])) if entries > 0 else float("nan")
    d = x.loc[valid, "fill_delay_min"].dropna().to_numpy(dtype=float)
    p95 = float(np.quantile(d, 0.95)) if d.size else float("nan")

    # cluster metrics from trade order.
    t = x.loc[valid].copy().sort_values(["entry_time_utc", "signal_time_utc", "signal_id"]).reset_index(drop=True)
    loss = (t[pnl_col] < 0).to_numpy(dtype=bool)
    streaks: List[int] = []
    i = 0
    while i < len(loss):
        if loss[i]:
            s = i
            while i + 1 < len(loss) and loss[i + 1]:
                i += 1
            streaks.append(i - s + 1)
        i += 1
    streak_arr = np.asarray(streaks, dtype=int) if streaks else np.array([], dtype=int)
    max_streak = int(streak_arr.max()) if streak_arr.size else 0
    ge5 = int(np.sum(streak_arr >= 5)) if streak_arr.size else 0
    ge10 = int(np.sum(streak_arr >= 10)) if streak_arr.size else 0

    min_split_exp = float("nan")
    if entries > 0 and x["split_id"].notna().any():
        z = x.copy()
        z["pnl_for_split"] = 0.0
        z.loc[valid, "pnl_for_split"] = z.loc[valid, pnl_col]
        split_exp = z.groupby("split_id", dropna=True)["pnl_for_split"].mean()
        if not split_exp.empty:
            min_split_exp = float(split_exp.min())

    return {
        "signals_total": int(signals_total),
        "entries_valid": int(entries),
        "entry_rate": float(entry_rate),
        "exec_expectancy_net": float(exp),
        "exec_cvar_5": float(cvar),
        "exec_max_drawdown": float(mdd),
        "taker_share": float(taker_share),
        "p95_fill_delay_min": float(p95),
        "max_consecutive_losses": int(max_streak),
        "streak_ge5_count": int(ge5),
        "streak_ge10_count": int(ge10),
        "min_split_expectancy_net": float(min_split_exp),
    }


def fit_bin_numeric(feature: pd.Series, y: pd.Series, n_bins: int = 5) -> Dict[str, Any]:
    v = to_num(feature)
    m = v.notna() & y.notna()
    if int(m.sum()) < 40:
        return {"ok": False}
    vv = v[m]
    yy = y[m].astype(float)
    try:
        bins = pd.qcut(vv, q=min(n_bins, max(2, int(vv.nunique()))), duplicates="drop")
    except Exception:
        return {"ok": False}
    g = pd.DataFrame({"bin": bins.astype(str), "v": vv, "y": yy}).groupby("bin", dropna=False)
    tab = g.agg(count=("y", "size"), event_rate=("y", "mean"), v_min=("v", "min"), v_max=("v", "max")).reset_index()
    if tab.empty or tab["event_rate"].nunique() <= 1:
        return {
            "ok": True,
            "tab": tab,
            "delta_event_rate": 0.0,
            "risk_ratio": 1.0,
            "direction": "flat",
            "q20": float(np.nanquantile(vv, 0.20)),
            "q80": float(np.nanquantile(vv, 0.80)),
            "high_rate": float(tab["event_rate"].max() if not tab.empty else np.nan),
            "low_rate": float(tab["event_rate"].min() if not tab.empty else np.nan),
            "high_bin_idx": -1,
            "low_bin_idx": -1,
        }
    hi_idx = int(tab["event_rate"].idxmax())
    lo_idx = int(tab["event_rate"].idxmin())
    high = float(tab.loc[hi_idx, "event_rate"])
    low = float(tab.loc[lo_idx, "event_rate"])
    direction = "high_risk_high_value" if float(tab.loc[hi_idx, "v_max"]) >= float(tab.loc[lo_idx, "v_max"]) else "high_risk_low_value"
    rr = safe_div(high, max(1e-9, low))
    return {
        "ok": True,
        "tab": tab,
        "delta_event_rate": float(high - low),
        "risk_ratio": float(rr),
        "direction": direction,
        "q20": float(np.nanquantile(vv, 0.20)),
        "q80": float(np.nanquantile(vv, 0.80)),
        "high_rate": high,
        "low_rate": low,
        "high_bin_idx": hi_idx,
        "low_bin_idx": lo_idx,
    }


def feature_stability_numeric(
    df: pd.DataFrame,
    feature: str,
    label: str,
    direction: str,
    q20: float,
    q80: float,
) -> Tuple[float, int, int]:
    signs: List[int] = []
    total = 0
    for sid, g in df.groupby("split_id", dropna=True):
        x = to_num(g[feature])
        y = to_num(g[label])
        m = x.notna() & y.notna()
        if int(m.sum()) < 20:
            continue
        total += 1
        xx = x[m]
        yy = y[m]
        if direction == "high_risk_high_value":
            hi = yy[xx >= q80]
            lo = yy[xx <= q20]
        else:
            hi = yy[xx <= q20]
            lo = yy[xx >= q80]
        if len(hi) < 5 or len(lo) < 5:
            continue
        d = float(np.nanmean(hi) - np.nanmean(lo))
        signs.append(1 if d >= 0 else -1)
    if not signs:
        return float("nan"), 0, total
    major = 1 if np.sum(np.asarray(signs) == 1) >= np.sum(np.asarray(signs) == -1) else -1
    stable_frac = float(np.mean(np.asarray(signs) == major))
    return stable_frac, len(signs), total


def run_univariate_screen(df: pd.DataFrame, feature_cols: List[str], labels: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base = df[df["entry_for_labels"] == 1].copy()
    for label in labels:
        y = to_num(base[label])
        for f in feature_cols:
            s = base[f]
            miss = float(s.isna().mean())
            support = int((~s.isna() & y.notna()).sum())
            if support < 30:
                rows.append(
                    {
                        "feature": f,
                        "label": label,
                        "feature_type": "unsupported",
                        "support": support,
                        "missing_ratio": miss,
                        "delta_event_rate": float("nan"),
                        "risk_ratio": float("nan"),
                        "direction": "none",
                        "q20": float("nan"),
                        "q80": float("nan"),
                        "stable_sign_frac": float("nan"),
                        "stable_splits_used": 0,
                        "splits_considered": 0,
                        "bin_summary": "",
                    }
                )
                continue

            numeric_candidate = to_num(s)
            is_numeric = numeric_candidate.notna().sum() >= max(20, int(0.7 * support))
            if is_numeric:
                fit = fit_bin_numeric(numeric_candidate, y, n_bins=5)
                if not fit.get("ok", False):
                    rows.append(
                        {
                            "feature": f,
                            "label": label,
                            "feature_type": "numeric_failed",
                            "support": support,
                            "missing_ratio": miss,
                            "delta_event_rate": float("nan"),
                            "risk_ratio": float("nan"),
                            "direction": "none",
                            "q20": float("nan"),
                            "q80": float("nan"),
                            "stable_sign_frac": float("nan"),
                            "stable_splits_used": 0,
                            "splits_considered": 0,
                            "bin_summary": "",
                        }
                    )
                    continue
                tab = fit["tab"]
                bsum = "; ".join([f"{r.bin}:{int(r['count'])}/{float(r['event_rate']):.4f}" for _, r in tab.iterrows()])
                stable, used, total = feature_stability_numeric(
                    base,
                    feature=f,
                    label=label,
                    direction=str(fit["direction"]),
                    q20=float(fit["q20"]),
                    q80=float(fit["q80"]),
                )
                rows.append(
                    {
                        "feature": f,
                        "label": label,
                        "feature_type": "numeric",
                        "support": support,
                        "missing_ratio": miss,
                        "delta_event_rate": float(fit["delta_event_rate"]),
                        "risk_ratio": float(fit["risk_ratio"]),
                        "direction": str(fit["direction"]),
                        "q20": float(fit["q20"]),
                        "q80": float(fit["q80"]),
                        "stable_sign_frac": float(stable),
                        "stable_splits_used": int(used),
                        "splits_considered": int(total),
                        "bin_summary": bsum,
                    }
                )
            else:
                z = pd.DataFrame({"x": s.astype(str), "y": y}).dropna()
                if z.empty:
                    continue
                grp = z.groupby("x")["y"].agg(["count", "mean"]).reset_index().sort_values(["count", "mean"], ascending=[False, False])
                grp = grp[grp["count"] >= 10].copy()
                if grp.empty:
                    continue
                high = float(grp["mean"].max())
                low = float(grp["mean"].min())
                high_cat = str(grp.loc[grp["mean"].idxmax(), "x"])
                low_cat = str(grp.loc[grp["mean"].idxmin(), "x"])
                direction = f"high_risk_category:{high_cat}"
                bsum = "; ".join([f"{r.x}:{int(r['count'])}/{float(r['mean']):.4f}" for _, r in grp.head(8).iterrows()])
                rows.append(
                    {
                        "feature": f,
                        "label": label,
                        "feature_type": "categorical",
                        "support": support,
                        "missing_ratio": miss,
                        "delta_event_rate": float(high - low),
                        "risk_ratio": float(safe_div(high, max(1e-9, low))),
                        "direction": direction,
                        "q20": float("nan"),
                        "q80": float("nan"),
                        "stable_sign_frac": float("nan"),
                        "stable_splits_used": 0,
                        "splits_considered": 0,
                        "bin_summary": bsum,
                        "high_category": high_cat,
                        "low_category": low_cat,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["label", "delta_event_rate", "support"], ascending=[True, False, False]).reset_index(drop=True)
    return out


def build_risk_flag(df: pd.DataFrame, row: pd.Series) -> pd.Series:
    f = str(row["feature"])
    ftype = str(row["feature_type"])
    if ftype == "numeric":
        x = to_num(df[f])
        q20 = float(row.get("q20", np.nan))
        q80 = float(row.get("q80", np.nan))
        direction = str(row.get("direction", ""))
        if direction == "high_risk_high_value":
            return (x >= q80).astype(int)
        if direction == "high_risk_low_value":
            return (x <= q20).astype(int)
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    if ftype == "categorical":
        cat = str(row.get("high_category", ""))
        return (df[f].astype(str) == cat).astype(int)
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index)


def run_interaction_screen(df: pd.DataFrame, uni: pd.DataFrame, labels: List[str], max_pairs: int = 20) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    base = df[df["entry_for_labels"] == 1].copy()
    for label in labels:
        u = uni[(uni["label"] == label) & (uni["support"] >= 60)].copy()
        if u.empty:
            continue
        u["abs_delta"] = to_num(u["delta_event_rate"]).abs()
        u["stability_ok"] = (to_num(u["stable_sign_frac"]).fillna(0.5) >= 0.55).astype(int)
        u = u.sort_values(["stability_ok", "abs_delta", "support"], ascending=[False, False, False]).head(12).reset_index(drop=True)
        if len(u) < 2:
            continue
        base_rate = float(np.nanmean(to_num(base[label])))
        pair_rows = []
        for i, j in itertools.combinations(range(len(u)), 2):
            r1 = u.iloc[i]
            r2 = u.iloc[j]
            f1 = build_risk_flag(base, r1)
            f2 = build_risk_flag(base, r2)
            both = (f1 == 1) & (f2 == 1)
            support = int(both.sum())
            if support < 20:
                continue
            y = to_num(base[label])
            er_both = float(np.nanmean(y[both]))
            er_other = float(np.nanmean(y[~both])) if int((~both).sum()) > 0 else float("nan")
            delta = float(er_both - er_other) if np.isfinite(er_other) else float("nan")
            lift = safe_div(er_both, max(1e-9, base_rate))

            # split stability sign
            signs = []
            for sid, g in base.groupby("split_id", dropna=True):
                yy = to_num(g[label])
                if yy.notna().sum() < 20:
                    continue
                bb = ((build_risk_flag(g, r1) == 1) & (build_risk_flag(g, r2) == 1))
                if int(bb.sum()) < 5 or int((~bb).sum()) < 5:
                    continue
                d = float(np.nanmean(yy[bb]) - np.nanmean(yy[~bb]))
                signs.append(1 if d >= 0 else -1)
            stable_frac = float(np.mean(np.asarray(signs) == (1 if delta >= 0 else -1))) if signs else float("nan")
            pair_rows.append(
                {
                    "label": label,
                    "feature_a": str(r1["feature"]),
                    "feature_b": str(r2["feature"]),
                    "support_both": support,
                    "event_rate_both": er_both,
                    "event_rate_other": er_other,
                    "delta_event_rate": delta,
                    "lift_vs_base": float(lift),
                    "stable_sign_frac": stable_frac,
                    "splits_used": int(len(signs)),
                }
            )
        p = pd.DataFrame(pair_rows)
        if p.empty:
            continue
        p["score"] = to_num(p["delta_event_rate"]).abs() * np.log1p(to_num(p["support_both"]).clip(lower=1))
        p = p.sort_values(["score", "support_both"], ascending=[False, False]).head(max_pairs)
        rows.extend(p.to_dict(orient="records"))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["label", "delta_event_rate", "support_both"], ascending=[True, False, False]).reset_index(drop=True)
    return out


def evaluate_exact_variant(
    *,
    run_dir: Path,
    variant_id: str,
    variant_desc: str,
    signals_df: pd.DataFrame,
    genome: Dict[str, Any],
    seed: int,
    baseline_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    met, sig, _split, _args, _bundle = evaluate_exact(
        run_dir=run_dir,
        signals_df=signals_df,
        genome=genome,
        seed=seed,
        name=f"ae_{variant_id}",
    )
    x = build_trade_labels(sig)
    m = aggregate_metrics_from_dataset(x, pnl_col="pnl_net_trade_notional_dec")
    row = {
        "variant_id": variant_id,
        "variant_desc": variant_desc,
        "eval_type": "exact_engine_integrated",
        "approximate_proxy": 0,
        "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
        "invalid_reason": str(met.get("invalid_reason", "")),
    }
    row.update(m)
    row["delta_exec_expectancy_vs_baseline"] = float(row["exec_expectancy_net"] - baseline_metrics["exec_expectancy_net"])
    row["cvar_improve_ratio_vs_baseline"] = safe_div(abs(baseline_metrics["exec_cvar_5"]) - abs(row["exec_cvar_5"]), abs(baseline_metrics["exec_cvar_5"]))
    row["maxdd_improve_ratio_vs_baseline"] = safe_div(abs(baseline_metrics["exec_max_drawdown"]) - abs(row["exec_max_drawdown"]), abs(baseline_metrics["exec_max_drawdown"]))
    row["max_consecutive_losses_reduction_ratio"] = safe_div(
        float(baseline_metrics["max_consecutive_losses"] - row["max_consecutive_losses"]),
        float(max(1, baseline_metrics["max_consecutive_losses"])),
    )
    row["streak_ge5_reduction_ratio"] = safe_div(
        float(baseline_metrics["streak_ge5_count"] - row["streak_ge5_count"]),
        float(max(1, baseline_metrics["streak_ge5_count"])),
    )
    row["streak_ge10_reduction_ratio"] = safe_div(
        float(baseline_metrics["streak_ge10_count"] - row["streak_ge10_count"]),
        float(max(1, baseline_metrics["streak_ge10_count"])),
    )
    return row


def evaluate_soft_proxy_variant(
    *,
    variant_id: str,
    variant_desc: str,
    base_df: pd.DataFrame,
    size_mult: pd.Series,
    baseline_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    x = base_df.copy()
    x["size_mult"] = to_num(size_mult).fillna(1.0).clip(lower=0.0)
    x["scaled_pnl_net"] = to_num(x["pnl_net_trade_notional_dec"]) * x["size_mult"]
    x["scaled_pnl_gross"] = to_num(x["pnl_gross_trade_notional_dec"]) * x["size_mult"]

    m = aggregate_metrics_from_dataset(
        x.assign(
            pnl_net_trade_notional_dec=x["scaled_pnl_net"],
            pnl_gross_trade_notional_dec=x["scaled_pnl_gross"],
        ),
        pnl_col="pnl_net_trade_notional_dec",
    )
    row = {
        "variant_id": variant_id,
        "variant_desc": variant_desc,
        "eval_type": "proxy_size_scaling",
        "approximate_proxy": 1,
        "valid_for_ranking": 1,
        "invalid_reason": "",
    }
    row.update(m)
    row["delta_exec_expectancy_vs_baseline"] = float(row["exec_expectancy_net"] - baseline_metrics["exec_expectancy_net"])
    row["cvar_improve_ratio_vs_baseline"] = safe_div(abs(baseline_metrics["exec_cvar_5"]) - abs(row["exec_cvar_5"]), abs(baseline_metrics["exec_cvar_5"]))
    row["maxdd_improve_ratio_vs_baseline"] = safe_div(abs(baseline_metrics["exec_max_drawdown"]) - abs(row["exec_max_drawdown"]), abs(baseline_metrics["exec_max_drawdown"]))
    row["max_consecutive_losses_reduction_ratio"] = safe_div(
        float(baseline_metrics["max_consecutive_losses"] - row["max_consecutive_losses"]),
        float(max(1, baseline_metrics["max_consecutive_losses"])),
    )
    row["streak_ge5_reduction_ratio"] = safe_div(
        float(baseline_metrics["streak_ge5_count"] - row["streak_ge5_count"]),
        float(max(1, baseline_metrics["streak_ge5_count"])),
    )
    row["streak_ge10_reduction_ratio"] = safe_div(
        float(baseline_metrics["streak_ge10_count"] - row["streak_ge10_count"]),
        float(max(1, baseline_metrics["streak_ge10_count"])),
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase AE signal-quality + regime-labeling pipeline (SOLUSDT, execution-aware, no GA)")
    ap.add_argument("--seed", type=int, default=20260305)
    args_cli = ap.parse_args()

    t0 = time.time()
    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_dir = exec_root / f"PHASEAE_SIGNAL_LABELING_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    rep_fp = Path(LOCKED["representative_subset_csv"]).resolve()
    fee_fp = Path(LOCKED["canonical_fee_model"]).resolve()
    metrics_fp = Path(LOCKED["canonical_metrics_definition"]).resolve()
    for fp in (rep_fp, fee_fp, metrics_fp):
        if not fp.exists():
            raise FileNotFoundError(f"Missing locked input: {fp}")

    fee_sha = sha256_file(fee_fp)
    met_sha = sha256_file(metrics_fp)
    if fee_sha != LOCKED["expected_fee_sha"]:
        raise RuntimeError(f"Fee hash mismatch: {fee_sha} != {LOCKED['expected_fee_sha']}")
    if met_sha != LOCKED["expected_metrics_sha"]:
        raise RuntimeError(f"Metrics hash mismatch: {met_sha} != {LOCKED['expected_metrics_sha']}")

    sig_in = ensure_signals_schema(pd.read_csv(rep_fp))
    exec_pair = load_exec_pair(exec_root)
    e1 = exec_pair["E1"]
    e2 = exec_pair["E2"]

    # Contract lock check via ga_exec helper.
    lock_args = build_args(signals_csv=rep_fp, seed=int(args_cli.seed))
    lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=lock_args, run_dir=run_dir)
    if int(lock_validation.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("ga_exec freeze lock validation failed")

    # AE1: reproduce baseline primary + backup.
    met1, sig1, split1, args1, bundle1 = evaluate_exact(
        run_dir=run_dir,
        signals_df=sig_in,
        genome=e1["genome"],
        seed=int(args_cli.seed),
        name="ae_baseline_e1",
    )
    met2, sig2, split2, args2, bundle2 = evaluate_exact(
        run_dir=run_dir,
        signals_df=sig_in,
        genome=e2["genome"],
        seed=int(args_cli.seed) + 1,
        name="ae_baseline_e2",
    )

    pre_feat = build_preentry_features(bundle1, e1["genome"])
    joined = sig1.merge(pre_feat, on=["signal_id", "signal_time"], how="left")
    labels_df = build_trade_labels(joined)
    labels_df["split_id"] = to_num(labels_df.get("split_id", np.nan))

    # AE1 outputs.
    labels_path: Path
    try:
        labels_path = run_dir / "phaseAE_labels_dataset.parquet"
        labels_df.to_parquet(labels_path, index=False)
    except Exception:
        labels_path = run_dir / "phaseAE_labels_dataset.csv"
        labels_df.to_csv(labels_path, index=False)

    repro_obj = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "symbol": LOCKED["symbol"],
        "frozen_contract": {
            "representative_subset_csv": str(rep_fp),
            "canonical_fee_model": str(fee_fp),
            "canonical_metrics_definition": str(metrics_fp),
            "fee_sha256": fee_sha,
            "metrics_sha256": met_sha,
            "fee_hash_match": int(fee_sha == LOCKED["expected_fee_sha"]),
            "metrics_hash_match": int(met_sha == LOCKED["expected_metrics_sha"]),
        },
        "ga_exec_freeze_lock_validation": lock_validation,
        "baseline_candidates": {
            "E1": {
                "hash": e1["genome_hash"],
                "source": e1["source_run"],
                "valid_for_ranking": int(met1.get("valid_for_ranking", 0)),
                "exec_expectancy_net": float(met1.get("overall_exec_expectancy_net", np.nan)),
                "exec_cvar_5": float(met1.get("overall_exec_cvar_5", np.nan)),
                "exec_max_drawdown": float(met1.get("overall_exec_max_drawdown", np.nan)),
                "entries_valid": int(met1.get("overall_entries_valid", 0)),
                "entry_rate": float(met1.get("overall_entry_rate", np.nan)),
            },
            "E2": {
                "hash": e2["genome_hash"],
                "source": e2["source_run"],
                "valid_for_ranking": int(met2.get("valid_for_ranking", 0)),
                "exec_expectancy_net": float(met2.get("overall_exec_expectancy_net", np.nan)),
                "exec_cvar_5": float(met2.get("overall_exec_cvar_5", np.nan)),
                "exec_max_drawdown": float(met2.get("overall_exec_max_drawdown", np.nan)),
                "entries_valid": int(met2.get("overall_entries_valid", 0)),
                "entry_rate": float(met2.get("overall_entry_rate", np.nan)),
            },
        },
        "labels_dataset_path": str(labels_path),
    }
    json_dump(run_dir / "phaseAE_reproduction_check.json", repro_obj)

    lab = labels_df[labels_df["entry_for_labels"] == 1].copy()
    balance_lines: List[str] = []
    balance_lines.append("# Phase AE Label Balance Report")
    balance_lines.append("")
    balance_lines.append(f"- Generated UTC: {utc_now()}")
    balance_lines.append(f"- Baseline E1 hash: `{e1['genome_hash']}`")
    balance_lines.append(f"- Backup E2 hash: `{e2['genome_hash']}`")
    balance_lines.append(f"- Signals total: `{len(labels_df)}`")
    balance_lines.append(f"- Labelable trades (entry_for_labels=1): `{len(lab)}`")
    balance_lines.append("")
    y_cols = ["y_loss", "y_tail_loss", "y_sl_loss", "y_cluster_loss", "y_toxic_trade", "y_good_trade", "y_slow_bleed", "y_fee_dominated"]
    rows = []
    for c in y_cols:
        v = to_num(lab[c]).dropna()
        rows.append({"label": c, "support": int(len(v)), "positive": int(v.sum()) if len(v) else 0, "rate": float(v.mean()) if len(v) else float("nan")})
    bdf = pd.DataFrame(rows)
    balance_lines.append(markdown_table(bdf, ["label", "support", "positive", "rate"]))
    write_text(run_dir / "phaseAE_label_balance_report.md", "\n".join(balance_lines))

    # AE2 feature dictionary + matrix.
    feature_groups: Dict[str, List[Tuple[str, str]]] = {
        "A_1h_signal_context": [
            ("tp_mult", "Signal TP multiplier from 1h layer"),
            ("sl_mult", "Signal SL multiplier from 1h layer"),
            ("atr_percentile_1h", "1h ATR percentile at signal time"),
            ("trend_up_1h", "1h trend direction/probability proxy"),
            ("vol_bucket", "Derived 1h volatility regime bucket"),
            ("trend_bucket", "Derived 1h trend regime bucket"),
        ],
        "B_3m_preentry_context": [
            ("pre3m_ret_3bars", "3m return over last ~3 bars before signal"),
            ("pre3m_realized_vol_12", "Realized vol over last ~12 bars (prior-only)"),
            ("pre3m_atr_z", "ATR z-score using prior history"),
            ("pre3m_spread_proxy_bps", "Spread proxy bps at decision bar"),
            ("pre3m_body_bps_abs", "Absolute body size bps at decision bar"),
            ("pre3m_wick_ratio", "Wick ratio proxy at decision bar"),
            ("pre3m_impulse_atr", "Decision-bar impulse normalized by ATR"),
            ("pre3m_trend_slope_12", "Linear slope over prior closes"),
            ("pre3m_accel_6v6", "Acceleration: last6-return mean minus prior6"),
            ("pre3m_upbar_ratio_12", "Up-bar ratio over prior window"),
            ("pre3m_close_to_high_dist_bps", "Distance of close to high"),
            ("pre3m_close_to_low_dist_bps", "Distance of close to low"),
        ],
        "C_interaction_context": [
            ("prior_loss_streak_len", "Prior-only current loss streak length"),
            ("prior_rolling_loss_rate_5", "Prior-only rolling loss rate (5)"),
            ("prior_rolling_loss_rate_10", "Prior-only rolling loss rate (10)"),
            ("prior_rolling_loss_rate_20", "Prior-only rolling loss rate (20)"),
            ("prior_rolling_tail_count_5", "Prior-only tail-loss count (5)"),
            ("prior_rolling_tail_count_10", "Prior-only tail-loss count (10)"),
            ("prior_rolling_tail_count_20", "Prior-only tail-loss count (20)"),
        ],
        "D_operational_proxies": [
            ("est_limit_distance_bps", "Configured limit distance proxy"),
            ("est_fill_window_bars", "Configured max fill window in bars"),
            ("est_fallback_window_bars", "Configured fallback delay in bars"),
            ("est_taker_risk_proxy", "Pre-entry taker-risk proxy"),
            ("est_fill_delay_risk_proxy", "Pre-entry fill-delay risk proxy"),
            ("cfg_spread_guard_bps", "Execution spread guard setting"),
            ("cfg_vol_threshold", "Execution vol-threshold setting"),
            ("cfg_micro_vol_filter", "Execution micro-vol gate enabled"),
            ("cfg_use_signal_quality_gate", "Execution signal-quality gate enabled"),
            ("cfg_min_signal_quality_gate", "Execution signal-quality threshold"),
        ],
    }

    dict_lines: List[str] = []
    dict_lines.append("# Phase AE Feature Dictionary")
    dict_lines.append("")
    dict_lines.append(f"- Generated UTC: {utc_now()}")
    dict_lines.append("- Leakage policy: features are restricted to signal-time / pre-entry context; no post-entry outcome fields are included.")
    dict_lines.append("")
    for g, items in feature_groups.items():
        dict_lines.append(f"## {g}")
        dict_lines.append("")
        for f, d in items:
            dict_lines.append(f"- `{f}`: {d}")
        dict_lines.append("")
    write_text(run_dir / "phaseAE_feature_dictionary.md", "\n".join(dict_lines))

    feature_cols = [f for items in feature_groups.values() for f, _ in items]
    feature_matrix = labels_df[["signal_id", "signal_time_utc", "split_id"] + feature_cols].copy()
    feat_path: Path
    try:
        feat_path = run_dir / "phaseAE_features_matrix.parquet"
        feature_matrix.to_parquet(feat_path, index=False)
    except Exception:
        feat_path = run_dir / "phaseAE_features_matrix.csv"
        feature_matrix.to_csv(feat_path, index=False)

    # AE3: univariate + interactions.
    labels = ["y_toxic_trade", "y_cluster_loss"]
    uni_df = run_univariate_screen(labels_df, feature_cols=feature_cols, labels=labels)
    uni_df.to_csv(run_dir / "phaseAE_univariate_screen.csv", index=False)

    int_df = run_interaction_screen(labels_df, uni_df, labels=labels, max_pairs=20)
    int_df.to_csv(run_dir / "phaseAE_interaction_screen.csv", index=False)

    # AE3 fragility/leakage checks + E2 consistency on top E1 features.
    st_lines: List[str] = []
    st_lines.append("# Phase AE Stability and Fragility Report")
    st_lines.append("")
    st_lines.append(f"- Generated UTC: {utc_now()}")
    st_lines.append("- Leakage check: excluded post-entry outcomes (`pnl`, `exit_reason`, `mae/mfe`, fill results) from feature library.")
    st_lines.append("- Split stability rule-of-thumb: `stable_sign_frac >= 0.60` considered stable.")
    st_lines.append("")
    top_uni = uni_df[(uni_df["label"] == "y_toxic_trade") & (uni_df["feature_type"] == "numeric")].copy()
    top_uni["score"] = to_num(top_uni["delta_event_rate"]).abs() * np.log1p(to_num(top_uni["support"]).clip(lower=1))
    top_uni = top_uni.sort_values(["score"], ascending=False).head(10).reset_index(drop=True)
    frag = top_uni[to_num(top_uni["stable_sign_frac"]).fillna(0.0) < 0.60]
    st_lines.append("## Top Numeric Features (y_toxic_trade)")
    st_lines.append("")
    st_lines.append(markdown_table(top_uni, ["feature", "support", "delta_event_rate", "risk_ratio", "direction", "stable_sign_frac", "bin_summary"]))
    st_lines.append("")
    st_lines.append("## Fragile Effects (split-unstable)")
    st_lines.append("")
    st_lines.append(markdown_table(frag, ["feature", "support", "delta_event_rate", "stable_sign_frac"]))
    st_lines.append("")

    # E2 direction confirmation for top 5 E1 features.
    e2_df = build_trade_labels(sig2.merge(pre_feat, on=["signal_id", "signal_time"], how="left"))
    confirm_rows = []
    top5 = top_uni.head(5)
    for _, r in top5.iterrows():
        f = str(r["feature"])
        direction = str(r["direction"])
        q20 = float(r["q20"])
        q80 = float(r["q80"])
        y = to_num(e2_df.loc[e2_df["entry_for_labels"] == 1, "y_toxic_trade"])
        x = to_num(e2_df.loc[e2_df["entry_for_labels"] == 1, f])
        m = x.notna() & y.notna()
        xx = x[m]
        yy = y[m]
        if len(xx) < 40:
            continue
        if direction == "high_risk_high_value":
            hi = yy[xx >= q80]
            lo = yy[xx <= q20]
            sign = 1
        else:
            hi = yy[xx <= q20]
            lo = yy[xx >= q80]
            sign = 1
        if len(hi) < 5 or len(lo) < 5:
            continue
        delta = float(np.nanmean(hi) - np.nanmean(lo))
        confirm_rows.append(
            {
                "feature": f,
                "e1_direction": direction,
                "e2_delta_event_rate": delta,
                "e2_direction_consistent": int(delta >= 0 if sign == 1 else delta <= 0),
                "e2_support": int(len(xx)),
            }
        )
    cdf = pd.DataFrame(confirm_rows)
    st_lines.append("## Backup Candidate (E2) Direction Confirmation")
    st_lines.append("")
    st_lines.append(markdown_table(cdf, ["feature", "e1_direction", "e2_delta_event_rate", "e2_direction_consistent", "e2_support"]))
    write_text(run_dir / "phaseAE_stability_fragility_report.md", "\n".join(st_lines))

    # AE4 prototypes.
    q = {}
    for col, pct in (
        ("pre3m_atr_z", 0.80),
        ("pre3m_realized_vol_12", 0.80),
        ("pre3m_spread_proxy_bps", 0.80),
        ("pre3m_body_bps_abs", 0.80),
        ("trend_up_1h", 0.20),
    ):
        v = to_num(labels_df[col]).dropna()
        q[col] = float(np.nanquantile(v, pct)) if len(v) else float("nan")

    prototypes = [
        {
            "prototype_id": "H1_vol_trend_guard",
            "type": "hard_filter_exact",
            "rule": "block if pre3m_atr_z>=q(pre3m_atr_z,0.8) AND trend_up_1h<=q(trend_up_1h,0.2)",
            "formula": {"all_of": [{"feature": "pre3m_atr_z", "op": ">=", "value": q["pre3m_atr_z"]}, {"feature": "trend_up_1h", "op": "<=", "value": q["trend_up_1h"]}]},
            "target": "tail + cluster risk in weak-trend high-vol conditions",
            "risk": "participation drop, possible over-filtering",
        },
        {
            "prototype_id": "H2_spread_vol_guard",
            "type": "hard_filter_exact",
            "rule": "block if pre3m_spread_proxy_bps>=q80 AND pre3m_realized_vol_12>=q80",
            "formula": {"all_of": [{"feature": "pre3m_spread_proxy_bps", "op": ">=", "value": q["pre3m_spread_proxy_bps"]}, {"feature": "pre3m_realized_vol_12", "op": ">=", "value": q["pre3m_realized_vol_12"]}]},
            "target": "fee/slippage and bad fills in stressed microstructure",
            "risk": "entry starvation if too broad",
        },
        {
            "prototype_id": "H3_impulse_vol_guard",
            "type": "hard_filter_exact",
            "rule": "block if pre3m_body_bps_abs>=q80 AND pre3m_atr_z>=q80",
            "formula": {"all_of": [{"feature": "pre3m_body_bps_abs", "op": ">=", "value": q["pre3m_body_bps_abs"]}, {"feature": "pre3m_atr_z", "op": ">=", "value": q["pre3m_atr_z"]}]},
            "target": "impulsive entries likely to mean-revert into SL",
            "risk": "removes momentum winners too",
        },
        {
            "prototype_id": "S1_risk_score_half_size",
            "type": "soft_size_proxy",
            "rule": "size=0.5 if risk_score>=2 else 1.0",
            "formula": {"risk_score_flags": ["pre3m_atr_z>=q80", "pre3m_realized_vol_12>=q80", "pre3m_spread_proxy_bps>=q80", "trend_up_1h<=q20"], "threshold": 2, "size_if_triggered": 0.5, "size_else": 1.0},
            "target": "reduce tail exposure without hard blocking",
            "risk": "proxy only (engine does not natively support per-trade sizing)",
        },
        {
            "prototype_id": "S2_risk_score_quarter_size",
            "type": "soft_size_proxy",
            "rule": "size=0.25 if risk_score>=3 else 1.0",
            "formula": {"risk_score_flags": ["pre3m_atr_z>=q80", "pre3m_realized_vol_12>=q80", "pre3m_spread_proxy_bps>=q80", "trend_up_1h<=q20"], "threshold": 3, "size_if_triggered": 0.25, "size_else": 1.0},
            "target": "aggressive tail-risk suppression under compounded risk",
            "risk": "may suppress too much gross edge",
        },
    ]

    prot_lines: List[str] = []
    prot_lines.append("# Phase AE Candidate Prototypes")
    prot_lines.append("")
    prot_lines.append(f"- Generated UTC: {utc_now()}")
    prot_lines.append("- No session/time veto rule is used.")
    prot_lines.append("- Hard filters are exact replay; soft sizing is proxy replay only.")
    prot_lines.append("")
    for p in prototypes:
        prot_lines.append(f"## {p['prototype_id']}")
        prot_lines.append("")
        prot_lines.append(f"- Type: `{p['type']}`")
        prot_lines.append(f"- Rule: {p['rule']}")
        prot_lines.append(f"- Expected impact target: {p['target']}")
        prot_lines.append(f"- Risk: {p['risk']}")
        prot_lines.append("")
    write_text(run_dir / "phaseAE_candidate_prototypes.md", "\n".join(prot_lines))

    # YAML-ish rules.
    yaml_lines: List[str] = []
    yaml_lines.append("phase: AE4")
    yaml_lines.append("symbol: SOLUSDT")
    yaml_lines.append("prototypes:")
    for p in prototypes:
        yaml_lines.append(f"  - prototype_id: {p['prototype_id']}")
        yaml_lines.append(f"    type: {p['type']}")
        yaml_lines.append(f"    rule: \"{p['rule']}\"")
        yaml_lines.append("    formula_json: |")
        formula = json.dumps(p["formula"], sort_keys=True)
        yaml_lines.append(f"      {formula}")
        yaml_lines.append(f"    target: \"{p['target']}\"")
        yaml_lines.append(f"    risk: \"{p['risk']}\"")
    write_text(run_dir / "phaseAE_prototype_rules.yaml", "\n".join(yaml_lines))

    # AE5: replay ablations.
    base_metrics = aggregate_metrics_from_dataset(labels_df, pnl_col="pnl_net_trade_notional_dec")
    base_row = {
        "variant_id": "baseline_E1",
        "variant_desc": "Baseline promoted execution winner E1",
        "eval_type": "exact_engine_integrated",
        "approximate_proxy": 0,
        "valid_for_ranking": int(met1.get("valid_for_ranking", 0)),
        "invalid_reason": str(met1.get("invalid_reason", "")),
    }
    base_row.update(base_metrics)
    base_row["delta_exec_expectancy_vs_baseline"] = 0.0
    base_row["cvar_improve_ratio_vs_baseline"] = 0.0
    base_row["maxdd_improve_ratio_vs_baseline"] = 0.0
    base_row["max_consecutive_losses_reduction_ratio"] = 0.0
    base_row["streak_ge5_reduction_ratio"] = 0.0
    base_row["streak_ge10_reduction_ratio"] = 0.0
    rows = [base_row]

    # Hard filter helpers use only pre-entry features.
    feat_idx = labels_df.set_index("signal_id")

    def hard_keep_mask(formula: Dict[str, Any]) -> pd.Series:
        m = pd.Series(True, index=feat_idx.index)
        for cond in formula.get("all_of", []):
            f = str(cond["feature"])
            op = str(cond["op"])
            v = float(cond["value"])
            x = to_num(feat_idx[f])
            if op == ">=":
                m = m & (x >= v)
            elif op == "<=":
                m = m & (x <= v)
            else:
                m = m & False
        return ~m  # keep = not blocked

    for p in prototypes:
        pid = str(p["prototype_id"])
        ptype = str(p["type"])
        if ptype == "hard_filter_exact":
            keep = hard_keep_mask(p["formula"])
            kept_ids = set(feat_idx.index[keep].astype(str).tolist())
            sig_filtered = sig_in[sig_in["signal_id"].astype(str).isin(kept_ids)].copy().reset_index(drop=True)
            r = evaluate_exact_variant(
                run_dir=run_dir,
                variant_id=pid,
                variant_desc=str(p["rule"]),
                signals_df=sig_filtered,
                genome=e1["genome"],
                seed=int(args_cli.seed) + 10 + len(rows),
                baseline_metrics=base_metrics,
            )
            r["removed_signals"] = int(len(sig_in) - len(sig_filtered))
            r["removed_signals_pct"] = float((len(sig_in) - len(sig_filtered)) / max(1, len(sig_in)))
            rows.append(r)
        else:
            x = labels_df.copy()
            f1 = (to_num(x["pre3m_atr_z"]) >= q["pre3m_atr_z"]).astype(int)
            f2 = (to_num(x["pre3m_realized_vol_12"]) >= q["pre3m_realized_vol_12"]).astype(int)
            f3 = (to_num(x["pre3m_spread_proxy_bps"]) >= q["pre3m_spread_proxy_bps"]).astype(int)
            f4 = (to_num(x["trend_up_1h"]) <= q["trend_up_1h"]).astype(int)
            risk_score = f1 + f2 + f3 + f4
            th = int(p["formula"]["threshold"])
            sz_hi = float(p["formula"]["size_if_triggered"])
            sz_lo = float(p["formula"]["size_else"])
            size_mult = pd.Series(np.where(risk_score >= th, sz_hi, sz_lo), index=x.index)
            r = evaluate_soft_proxy_variant(
                variant_id=pid,
                variant_desc=str(p["rule"]),
                base_df=x,
                size_mult=size_mult,
                baseline_metrics=base_metrics,
            )
            r["removed_signals"] = 0
            r["removed_signals_pct"] = 0.0
            rows.append(r)

    ab_df = pd.DataFrame(rows)
    ab_df.to_csv(run_dir / "phaseAE_prototype_ablation_results.csv", index=False)

    # Decision.
    cand = ab_df[ab_df["variant_id"] != "baseline_E1"].copy()
    cand["accept_expectancy"] = (to_num(cand["delta_exec_expectancy_vs_baseline"]) >= -0.00002).astype(int)
    cand["risk_improve"] = (
        (to_num(cand["max_consecutive_losses_reduction_ratio"]) >= 0.20)
        | (to_num(cand["streak_ge5_reduction_ratio"]) >= 0.20)
        | (to_num(cand["cvar_improve_ratio_vs_baseline"]) >= 0.10)
        | (to_num(cand["maxdd_improve_ratio_vs_baseline"]) >= 0.10)
    ).astype(int)
    cand["stability_ok"] = (to_num(cand["min_split_expectancy_net"]) >= (float(base_metrics["min_split_expectancy_net"]) - 0.0002)).astype(int)
    cand["participation_ok"] = (to_num(cand["entry_rate"]) >= 0.70 * float(base_metrics["entry_rate"])).astype(int)
    cand["go_flag"] = (cand["accept_expectancy"] == 1) & (cand["risk_improve"] == 1) & (cand["stability_ok"] == 1) & (cand["participation_ok"] == 1)
    go_df = cand[cand["go_flag"] == 1].copy()
    go_decision = "GO" if not go_df.empty else "NO_GO"

    rep_lines: List[str] = []
    rep_lines.append("# Phase AE Prototype Ablation Report")
    rep_lines.append("")
    rep_lines.append(f"- Generated UTC: {utc_now()}")
    rep_lines.append(f"- Baseline E1 hash: `{e1['genome_hash']}`")
    rep_lines.append(f"- Decision: **{go_decision}**")
    rep_lines.append("")
    rep_lines.append("## Baseline Metrics")
    rep_lines.append("")
    rep_lines.append(
        markdown_table(
            pd.DataFrame([base_metrics]),
            [
                "signals_total",
                "entries_valid",
                "entry_rate",
                "exec_expectancy_net",
                "exec_cvar_5",
                "exec_max_drawdown",
                "max_consecutive_losses",
                "streak_ge5_count",
                "streak_ge10_count",
                "taker_share",
                "p95_fill_delay_min",
                "min_split_expectancy_net",
            ],
        )
    )
    rep_lines.append("")
    rep_lines.append("## Prototype Results")
    rep_lines.append("")
    rep_lines.append(
        markdown_table(
            ab_df.sort_values(["eval_type", "variant_id"]),
            [
                "variant_id",
                "eval_type",
                "valid_for_ranking",
                "exec_expectancy_net",
                "delta_exec_expectancy_vs_baseline",
                "cvar_improve_ratio_vs_baseline",
                "maxdd_improve_ratio_vs_baseline",
                "max_consecutive_losses",
                "max_consecutive_losses_reduction_ratio",
                "streak_ge5_count",
                "streak_ge5_reduction_ratio",
                "streak_ge10_count",
                "entry_rate",
                "removed_signals_pct",
                "taker_share",
                "p95_fill_delay_min",
                "min_split_expectancy_net",
                "invalid_reason",
            ],
        )
    )
    rep_lines.append("")
    rep_lines.append("## GO Criteria")
    rep_lines.append("")
    rep_lines.append("- Acceptable expectancy: delta_expectancy >= -0.00002")
    rep_lines.append("- Risk improvement: any of {max_consecutive_losses_reduction>=20%, streak>=5_reduction>=20%, cvar_improve>=10%, maxdd_improve>=10%}")
    rep_lines.append("- Stability: min_split_expectancy not worse than baseline by more than 0.0002")
    rep_lines.append("- Participation: entry_rate >= 70% of baseline")
    rep_lines.append(f"- GO pass count: `{len(go_df)}`")
    if not go_df.empty:
        rep_lines.append(f"- Best GO prototype: `{go_df.sort_values('delta_exec_expectancy_vs_baseline', ascending=False).iloc[0]['variant_id']}`")
    write_text(run_dir / "phaseAE_prototype_ablation_report.md", "\n".join(rep_lines))

    dec_lines = [
        "# Phase AE Decision Next Step",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Final: **{go_decision}**",
        f"- Baseline hash: `{e1['genome_hash']}`",
        f"- Prototypes tested: `{len(cand)}`",
        f"- GO-qualified prototypes: `{len(go_df)}`",
    ]
    if go_decision == "GO":
        top = go_df.sort_values(["delta_exec_expectancy_vs_baseline", "cvar_improve_ratio_vs_baseline"], ascending=[False, False]).iloc[0].to_dict()
        dec_lines.append(f"- Selected for Phase AF seed: `{top['variant_id']}`")
    else:
        dec_lines.append("- Outcome: prototypes were either fragile, low-support, or expectancy-destructive under current replay.")
    write_text(run_dir / "phaseAE_decision_next_step.md", "\n".join(dec_lines))

    if go_decision == "GO":
        prompt = f"""Phase AF (SOLUSDT, contract-locked) controlled optimization:
Use frozen representative subset, canonical fee/metrics hashes, and unchanged hard gates.
Start from Phase AE prototype shortlist and optimize only local thresholds/weights for the GO-qualified prototype family (no full unconstrained GA).
Required:
1) keep no-leakage pre-entry features only,
2) optimize with bounded search budget (<=120 candidates),
3) compare against baseline E1 hash {e1['genome_hash']},
4) report clustering/tail risk deltas + expectancy + split stability + participation realism,
5) stop with NO_GO if gains are not stable across splits.
"""
        write_text(run_dir / "ready_to_launch_phaseAF_prompt.txt", prompt)

    # Final manifest.
    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "phase": "AE",
        "duration_sec": float(time.time() - t0),
        "symbol": LOCKED["symbol"],
        "seed": int(args_cli.seed),
        "freeze_lock_validation": lock_validation,
        "baseline_hashes": {"E1": e1["genome_hash"], "E2": e2["genome_hash"]},
        "decision": go_decision,
        "labels_dataset_path": str(labels_path),
        "features_matrix_path": str(feat_path),
        "univariate_rows": int(len(uni_df)),
        "interaction_rows": int(len(int_df)),
        "prototype_rows": int(len(ab_df)),
    }
    json_dump(run_dir / "phaseAE_run_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "decision": go_decision, "prototypes_tested": int(len(cand))}, sort_keys=True))


if __name__ == "__main__":
    main()

