#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    "preferred_hashes": ["862c940746de0da984862d95", "992bd371689ba3936f3b4d09"],
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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


def table_md(df: pd.DataFrame, cols: Sequence[str]) -> str:
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
                if np.isfinite(v):
                    vals.append(f"{v:.8g}")
                else:
                    vals.append("nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def find_latest_dir(parent: Path, prefix: str) -> Optional[Path]:
    items = sorted([p for p in parent.glob(f"{prefix}*") if p.is_dir()], key=lambda p: p.name)
    return items[-1] if items else None


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


def build_args(*, signals_csv: Path, seed: int) -> argparse.Namespace:
    parser = ga_exec.build_arg_parser()
    args = parser.parse_args([])
    args.symbol = LOCKED["symbol"]
    args.symbols = ""
    args.rank = 1
    args.signals_csv = str(signals_csv)
    args.max_signals = 1200
    args.walkforward = True
    args.wf_splits = 5
    args.train_ratio = 0.70
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


def load_best_baseline_hash(exec_root: Path) -> Tuple[str, Path, pd.DataFrame]:
    qrs_dirs = sorted([p for p in exec_root.glob("PHASEQRS_AUTORUN_*") if p.is_dir()], key=lambda p: p.name)
    if not qrs_dirs:
        raise FileNotFoundError("No PHASEQRS_AUTORUN_* directory found")

    for d in reversed(qrs_dirs):
        fp = d / "phaseS" / "phaseS_final_candidates.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if df.empty or "genome_hash" not in df.columns:
            continue
        x = df.copy()
        x["baseline_exec_expectancy"] = to_num(x.get("baseline_exec_expectancy", np.nan))
        x = x.dropna(subset=["baseline_exec_expectancy"]).sort_values("baseline_exec_expectancy", ascending=False).reset_index(drop=True)
        if x.empty:
            continue
        best_hash = str(x.iloc[0]["genome_hash"])
        return best_hash, d, x

    raise FileNotFoundError("No usable phaseS_final_candidates.csv found in PHASEQRS_AUTORUN runs")


def load_genome_from_phasev(exec_root: Path, genome_hash: str) -> Tuple[Dict[str, Any], Path, str]:
    phasev_dirs = sorted([p for p in exec_root.glob("PHASEV_BRANCHB_PORTABILITY_DD_*") if p.is_dir()], key=lambda p: p.name)
    if not phasev_dirs:
        raise FileNotFoundError("No PHASEV_BRANCHB_PORTABILITY_DD_* directory found")

    for d in reversed(phasev_dirs):
        fp = d / "phaseV_exec_candidates_locked.json"
        if not fp.exists():
            continue
        obj = json.loads(fp.read_text(encoding="utf-8"))
        cands = obj.get("candidates", {})
        for key, val in cands.items():
            h = str(val.get("genome_hash", ""))
            if h == genome_hash:
                g = copy.deepcopy(val.get("genome", {}))
                if not isinstance(g, dict) or not g:
                    raise RuntimeError(f"Genome payload empty for hash {genome_hash} in {fp}")
                return g, d, str(key)

    raise FileNotFoundError(f"Could not map genome hash {genome_hash} to phaseV_exec_candidates_locked.json")


def attach_signal_features(sig_rows: pd.DataFrame, signals_df: pd.DataFrame) -> pd.DataFrame:
    x = sig_rows.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    feat = signals_df[["signal_id", "signal_time", "atr_percentile_1h", "trend_up_1h"]].copy()
    feat["signal_id"] = feat["signal_id"].astype(str)
    feat["signal_time"] = pd.to_datetime(feat["signal_time"], utc=True, errors="coerce")
    out = x.merge(feat, on=["signal_id", "signal_time"], how="left")
    out["session_bucket"] = session_bucket(out["signal_time"])
    out["vol_bucket"] = vol_bucket(out["atr_percentile_1h"])
    out["trend_bucket"] = np.where(to_num(out["trend_up_1h"]) >= 0.5, "up", "down")
    return out


def build_trade_table(sig: pd.DataFrame) -> pd.DataFrame:
    x = sig.copy()
    x["exec_filled"] = to_num(x.get("exec_filled", 0)).fillna(0).astype(int)
    x["exec_valid_for_metrics"] = to_num(x.get("exec_valid_for_metrics", 0)).fillna(0).astype(int)
    x["exec_pnl_net_pct"] = to_num(x.get("exec_pnl_net_pct", np.nan))
    x["exec_pnl_gross_pct"] = to_num(x.get("exec_pnl_gross_pct", np.nan))
    x["exec_fill_delay_min"] = to_num(x.get("exec_fill_delay_min", np.nan))
    x["exec_entry_time"] = pd.to_datetime(x.get("exec_entry_time", pd.NaT), utc=True, errors="coerce")
    x["exec_exit_time"] = pd.to_datetime(x.get("exec_exit_time", pd.NaT), utc=True, errors="coerce")
    x["exec_entry_price"] = to_num(x.get("exec_entry_price", np.nan))
    x["exec_exit_price"] = to_num(x.get("exec_exit_price", np.nan))
    x["exec_mae_pct"] = to_num(x.get("exec_mae_pct", np.nan))
    x["exec_mfe_pct"] = to_num(x.get("exec_mfe_pct", np.nan))
    x["exec_sl_hit"] = to_num(x.get("exec_sl_hit", 0)).fillna(0).astype(int)
    x["split_id"] = to_num(x.get("split_id", np.nan))

    valid = (x["exec_filled"] == 1) & (x["exec_valid_for_metrics"] == 1) & x["exec_pnl_net_pct"].notna()
    t = x.loc[valid].copy().reset_index(drop=True)
    t = t.sort_values(["exec_entry_time", "signal_time", "signal_id"]).reset_index(drop=True)
    t["trade_id"] = [f"T{i:06d}" for i in range(1, len(t) + 1)]
    t["fee_plus_slippage_pct"] = t["exec_pnl_gross_pct"] - t["exec_pnl_net_pct"]
    t["time_stop_flag"] = (t.get("exec_exit_reason", "").astype(str) == "time_stop").astype(int)
    t["sl_hit_flag"] = t["exec_sl_hit"].astype(int)
    t["fill_delay_bucket"] = pd.cut(
        t["exec_fill_delay_min"],
        bins=[-np.inf, 3.0, 9.0, 30.0, np.inf],
        labels=["<=3m", "3-9m", "9-30m", ">30m"],
    ).astype(str)
    out_cols = [
        "trade_id",
        "signal_id",
        "signal_time",
        "exec_entry_time",
        "exec_exit_time",
        "exec_entry_price",
        "exec_exit_price",
        "exec_pnl_net_pct",
        "exec_pnl_gross_pct",
        "fee_plus_slippage_pct",
        "exec_mae_pct",
        "exec_mfe_pct",
        "sl_hit_flag",
        "time_stop_flag",
        "exec_exit_reason",
        "exec_entry_type",
        "exec_fill_liquidity_type",
        "exec_fill_delay_min",
        "fill_delay_bucket",
        "session_bucket",
        "vol_bucket",
        "trend_bucket",
        "atr_percentile_1h",
        "trend_up_1h",
        "split_id",
    ]
    return t[out_cols].copy()


def find_loss_streaks(pnl: np.ndarray) -> List[Tuple[int, int, int]]:
    out: List[Tuple[int, int, int]] = []
    n = len(pnl)
    i = 0
    while i < n:
        if np.isfinite(pnl[i]) and pnl[i] < 0:
            s = i
            while i + 1 < n and np.isfinite(pnl[i + 1]) and pnl[i + 1] < 0:
                i += 1
            e = i
            out.append((s, e, e - s + 1))
        i += 1
    return out


def clustering_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {
            "trades_total": 0,
            "max_consecutive_losses": 0,
            "streak_ge3_count": 0,
            "streak_ge5_count": 0,
            "streak_ge10_count": 0,
            "conditional_loss_rate_after_loss": float("nan"),
            "conditional_loss_rate_after_nonloss": float("nan"),
            "unconditional_loss_rate": float("nan"),
            "sl_loss_share": float("nan"),
            "loss_run_segments": [],
            "worst3_segments": [],
        }
    x = trades.copy().reset_index(drop=True)
    pnl = to_num(x["exec_pnl_net_pct"]).to_numpy(dtype=float)
    loss = np.isfinite(pnl) & (pnl < 0)
    streaks = find_loss_streaks(pnl)
    lengths = np.asarray([z[2] for z in streaks], dtype=int) if streaks else np.array([], dtype=int)

    max_streak = int(lengths.max()) if lengths.size else 0
    ge3 = int(np.sum(lengths >= 3)) if lengths.size else 0
    ge5 = int(np.sum(lengths >= 5)) if lengths.size else 0
    ge10 = int(np.sum(lengths >= 10)) if lengths.size else 0

    unconditional = float(np.mean(loss)) if loss.size else float("nan")
    cond_after_loss = float(np.mean(loss[1:][loss[:-1]])) if loss.size > 1 and np.any(loss[:-1]) else float("nan")
    cond_after_nonloss = float(np.mean(loss[1:][~loss[:-1]])) if loss.size > 1 and np.any(~loss[:-1]) else float("nan")

    neg = x[x["exec_pnl_net_pct"] < 0].copy()
    total_loss_abs = float(np.abs(neg["exec_pnl_net_pct"]).sum()) if not neg.empty else 0.0
    sl_loss_abs = float(np.abs(neg[neg["exec_exit_reason"].astype(str) == "sl"]["exec_pnl_net_pct"]).sum()) if not neg.empty else 0.0
    sl_share = float(sl_loss_abs / total_loss_abs) if total_loss_abs > 1e-12 else float("nan")

    seg_rows: List[Dict[str, Any]] = []
    for s, e, ln in streaks:
        seg = x.iloc[s : e + 1].copy()
        exit_mode = (
            seg["exec_exit_reason"].astype(str).value_counts().index[0]
            if seg["exec_exit_reason"].notna().any()
            else "unknown"
        )
        sess_mode = seg["session_bucket"].astype(str).value_counts().index[0] if seg["session_bucket"].notna().any() else "unknown"
        vol_mode = seg["vol_bucket"].astype(str).value_counts().index[0] if seg["vol_bucket"].notna().any() else "unknown"
        seg_rows.append(
            {
                "start_idx": int(s),
                "end_idx": int(e),
                "length": int(ln),
                "start_entry_time": str(pd.to_datetime(seg["exec_entry_time"].iloc[0], utc=True)),
                "end_entry_time": str(pd.to_datetime(seg["exec_entry_time"].iloc[-1], utc=True)),
                "pnl_sum": float(seg["exec_pnl_net_pct"].sum()),
                "mean_pnl": float(seg["exec_pnl_net_pct"].mean()),
                "median_fill_delay_min": float(pd.to_numeric(seg["exec_fill_delay_min"], errors="coerce").median()),
                "sl_share_in_segment": float(np.mean(seg["sl_hit_flag"] == 1)),
                "dominant_exit_reason": str(exit_mode),
                "dominant_session_bucket": str(sess_mode),
                "dominant_vol_bucket": str(vol_mode),
            }
        )
    worst3 = sorted(seg_rows, key=lambda r: (r["pnl_sum"], -r["length"]))[:3]

    return {
        "trades_total": int(len(x)),
        "max_consecutive_losses": int(max_streak),
        "streak_ge3_count": int(ge3),
        "streak_ge5_count": int(ge5),
        "streak_ge10_count": int(ge10),
        "conditional_loss_rate_after_loss": float(cond_after_loss),
        "conditional_loss_rate_after_nonloss": float(cond_after_nonloss),
        "unconditional_loss_rate": float(unconditional),
        "sl_loss_share": float(sl_share),
        "loss_run_segments": seg_rows,
        "worst3_segments": worst3,
    }


def tail_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "tail_name",
                "axis",
                "bucket",
                "trades_count",
                "pnl_sum",
                "loss_abs_sum",
                "share_of_tail_loss_abs",
                "share_of_total_loss_abs",
            ]
        )
    x = trades.copy().reset_index(drop=True)
    x["exec_pnl_net_pct"] = to_num(x["exec_pnl_net_pct"])
    x["loss_abs"] = np.where(x["exec_pnl_net_pct"] < 0, np.abs(x["exec_pnl_net_pct"]), 0.0)
    x["sl_bucket"] = np.where(x["sl_hit_flag"] == 1, "sl_hit_1", "sl_hit_0")
    total_loss_abs = float(x["loss_abs"].sum())

    n = len(x)
    idx_sorted = np.argsort(x["exec_pnl_net_pct"].to_numpy(dtype=float))
    k5 = max(1, int(math.ceil(0.05 * n)))
    k10 = max(1, int(math.ceil(0.10 * n)))
    tail_masks = {
        "cvar_5": np.isin(np.arange(n), idx_sorted[:k5]),
        "worst_decile": np.isin(np.arange(n), idx_sorted[:k10]),
    }

    axes = [
        ("sl_hit", "sl_bucket"),
        ("exit_reason", "exec_exit_reason"),
        ("session_bucket", "session_bucket"),
        ("vol_bucket", "vol_bucket"),
        ("entry_mechanic", "exec_entry_type"),
        ("fill_delay_bucket", "fill_delay_bucket"),
    ]
    rows: List[Dict[str, Any]] = []
    for tail_name, mask in tail_masks.items():
        t = x.loc[mask].copy()
        tail_loss_abs = float(t["loss_abs"].sum())
        rows.append(
            {
                "tail_name": tail_name,
                "axis": "__total__",
                "bucket": "__all__",
                "trades_count": int(len(t)),
                "pnl_sum": float(t["exec_pnl_net_pct"].sum()),
                "loss_abs_sum": float(tail_loss_abs),
                "share_of_tail_loss_abs": 1.0 if tail_loss_abs > 0 else float("nan"),
                "share_of_total_loss_abs": safe_div(tail_loss_abs, total_loss_abs),
            }
        )
        for axis, col in axes:
            grp = t.groupby(col, dropna=False)
            for bucket, g in grp:
                loss_abs = float(g["loss_abs"].sum())
                rows.append(
                    {
                        "tail_name": tail_name,
                        "axis": axis,
                        "bucket": str(bucket),
                        "trades_count": int(len(g)),
                        "pnl_sum": float(g["exec_pnl_net_pct"].sum()),
                        "loss_abs_sum": float(loss_abs),
                        "share_of_tail_loss_abs": safe_div(loss_abs, tail_loss_abs),
                        "share_of_total_loss_abs": safe_div(loss_abs, total_loss_abs),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_variant(
    *,
    variant_id: str,
    variant_type: str,
    controls_applied: str,
    genome: Dict[str, Any],
    bundle: ga_exec.SymbolBundle,
    args: argparse.Namespace,
    signals_df: pd.DataFrame,
    baseline_row: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    g = ga_exec._repair_genome(copy.deepcopy(genome), mode=str(args.mode), repair_hist=None)
    met = ga_exec._evaluate_genome(genome=g, bundles=[bundle], args=args, detailed=True)
    sig = attach_signal_features(met["signal_rows_df"], signals_df)
    trades = build_trade_table(sig)
    cl = clustering_metrics(trades)

    row: Dict[str, Any] = {
        "variant_id": variant_id,
        "variant_type": variant_type,
        "controls_applied": controls_applied,
        "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
        "hard_invalid": int(met.get("hard_invalid", 1)),
        "invalid_reason": str(met.get("invalid_reason", "")),
        "participation_pass": int(met.get("participation_pass", 0)),
        "realism_pass": int(met.get("realism_pass", 0)),
        "split_pass": int(met.get("split_pass", 0)),
        "exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
        "exec_cvar_5": float(met.get("overall_exec_cvar_5", np.nan)),
        "exec_max_drawdown": float(met.get("overall_exec_max_drawdown", np.nan)),
        "entries_valid": int(met.get("overall_entries_valid", 0)),
        "entry_rate": float(met.get("overall_entry_rate", np.nan)),
        "taker_share": float(met.get("overall_exec_taker_share", np.nan)),
        "median_fill_delay_min": float(met.get("overall_exec_median_fill_delay_min", np.nan)),
        "p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
        "min_split_expectancy_net": float(met.get("min_split_expectancy_net", np.nan)),
        "max_consecutive_losses": int(cl["max_consecutive_losses"]),
        "streak_ge3_count": int(cl["streak_ge3_count"]),
        "streak_ge5_count": int(cl["streak_ge5_count"]),
        "streak_ge10_count": int(cl["streak_ge10_count"]),
        "sl_loss_share": float(cl["sl_loss_share"]),
    }

    if baseline_row is not None:
        b_exp = float(baseline_row["exec_expectancy_net"])
        b_cvar = float(baseline_row["exec_cvar_5"])
        b_dd = float(baseline_row["exec_max_drawdown"])
        b_mcl = int(baseline_row["max_consecutive_losses"])
        b_s5 = int(baseline_row["streak_ge5_count"])
        b_s10 = int(baseline_row["streak_ge10_count"])
        row["delta_exec_expectancy_vs_baseline"] = float(row["exec_expectancy_net"] - b_exp)
        row["cvar_improve_ratio_vs_baseline"] = safe_div(abs(b_cvar) - abs(row["exec_cvar_5"]), abs(b_cvar))
        row["maxdd_improve_ratio_vs_baseline"] = safe_div(abs(b_dd) - abs(row["exec_max_drawdown"]), abs(b_dd))
        row["max_consecutive_losses_reduction_ratio"] = safe_div(float(b_mcl - int(row["max_consecutive_losses"])), float(b_mcl))
        row["streak_ge5_reduction_ratio"] = safe_div(float(b_s5 - int(row["streak_ge5_count"])), float(b_s5))
        row["streak_ge10_reduction_ratio"] = safe_div(float(b_s10 - int(row["streak_ge10_count"])), float(b_s10))
    else:
        row["delta_exec_expectancy_vs_baseline"] = float("nan")
        row["cvar_improve_ratio_vs_baseline"] = float("nan")
        row["maxdd_improve_ratio_vs_baseline"] = float("nan")
        row["max_consecutive_losses_reduction_ratio"] = float("nan")
        row["streak_ge5_reduction_ratio"] = float("nan")
        row["streak_ge10_reduction_ratio"] = float("nan")
    return row, sig, trades


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase AA/AB/AC drawdown forensics + risk-control design (SOLUSDT, contract-locked)")
    ap.add_argument("--seed", type=int, default=20260302)
    args_cli = ap.parse_args()

    t0 = time.time()
    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_dir = exec_root / f"PHASEAA_DD_RISKCONTROL_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    rep_fp = Path(LOCKED["representative_subset_csv"]).resolve()
    fee_fp = Path(LOCKED["canonical_fee_model"]).resolve()
    metrics_fp = Path(LOCKED["canonical_metrics_definition"]).resolve()
    for fp in (rep_fp, fee_fp, metrics_fp):
        if not fp.exists():
            raise FileNotFoundError(f"Missing required path: {fp}")

    fee_sha = sha256_file(fee_fp)
    metrics_sha = sha256_file(metrics_fp)
    if fee_sha != LOCKED["expected_fee_sha"]:
        raise RuntimeError(f"Fee hash mismatch: {fee_sha} != {LOCKED['expected_fee_sha']}")
    if metrics_sha != LOCKED["expected_metrics_sha"]:
        raise RuntimeError(f"Metrics hash mismatch: {metrics_sha} != {LOCKED['expected_metrics_sha']}")

    baseline_hash, qrs_dir, phaseS_df = load_best_baseline_hash(exec_root)
    baseline_genome, phasev_dir, baseline_exec_id = load_genome_from_phasev(exec_root, baseline_hash)

    sig_input = ensure_signals_schema(pd.read_csv(rep_fp))
    args = build_args(signals_csv=rep_fp, seed=int(args_cli.seed))
    lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=args, run_dir=run_dir)
    if int(lock_validation.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("ga_exec freeze lock validation failed")

    bundles, meta = ga_exec._prepare_bundles(args)
    if len(bundles) != 1:
        raise RuntimeError(f"Expected exactly 1 bundle for SOL, got {len(bundles)}")
    bundle = bundles[0]

    # Phase AA baseline reproduction and forensic tables.
    base_row, base_sig, base_trades = evaluate_variant(
        variant_id="baseline_e_winner",
        variant_type="baseline",
        controls_applied="none",
        genome=baseline_genome,
        bundle=bundle,
        args=args,
        signals_df=sig_input,
        baseline_row=None,
    )
    base_cluster = clustering_metrics(base_trades)
    tail_df = tail_attribution(base_trades)
    tail_df.to_csv(run_dir / "phaseAA_tail_attribution.csv", index=False)

    trade_table_path: Path
    try:
        trade_table_path = run_dir / "phaseAA_trade_table.parquet"
        base_trades.to_parquet(trade_table_path, index=False)
    except Exception:
        trade_table_path = run_dir / "phaseAA_trade_table.csv"
        base_trades.to_csv(trade_table_path, index=False)

    aa_lines: List[str] = []
    aa_lines.append("# Phase AA Loss Clustering Report")
    aa_lines.append("")
    aa_lines.append(f"- Generated UTC: {utc_now()}")
    aa_lines.append(f"- Baseline genome hash: `{baseline_hash}` (`{baseline_exec_id}`)")
    aa_lines.append(f"- Source PhaseS dir: `{qrs_dir}`")
    aa_lines.append(f"- Source PhaseV dir: `{phasev_dir}`")
    aa_lines.append("")
    aa_lines.append("## Baseline Reproduction")
    aa_lines.append("")
    aa_lines.append(f"- valid_for_ranking: `{int(base_row['valid_for_ranking'])}`")
    aa_lines.append(f"- exec_expectancy_net: `{float(base_row['exec_expectancy_net']):.8f}`")
    aa_lines.append(f"- exec_cvar_5: `{float(base_row['exec_cvar_5']):.8f}`")
    aa_lines.append(f"- exec_max_drawdown: `{float(base_row['exec_max_drawdown']):.8f}`")
    aa_lines.append(f"- entries_valid / entry_rate: `{int(base_row['entries_valid'])}` / `{float(base_row['entry_rate']):.6f}`")
    aa_lines.append("")
    aa_lines.append("## Loss Clustering")
    aa_lines.append("")
    aa_lines.append(f"- max_consecutive_losses: `{int(base_cluster['max_consecutive_losses'])}`")
    aa_lines.append(f"- streak>=3 count: `{int(base_cluster['streak_ge3_count'])}`")
    aa_lines.append(f"- streak>=5 count: `{int(base_cluster['streak_ge5_count'])}`")
    aa_lines.append(f"- streak>=10 count: `{int(base_cluster['streak_ge10_count'])}`")
    aa_lines.append(f"- unconditional_loss_rate: `{float(base_cluster['unconditional_loss_rate']):.6f}`")
    aa_lines.append(f"- conditional_loss_rate_after_loss: `{float(base_cluster['conditional_loss_rate_after_loss']):.6f}`")
    aa_lines.append(f"- conditional_loss_rate_after_nonloss: `{float(base_cluster['conditional_loss_rate_after_nonloss']):.6f}`")
    aa_lines.append(f"- sl_loss_share: `{float(base_cluster['sl_loss_share']):.6f}`")
    aa_lines.append("")
    aa_lines.append("## Worst 3 Loss Streak Segments")
    aa_lines.append("")
    worst_df = pd.DataFrame(base_cluster["worst3_segments"])
    aa_lines.append(
        table_md(
            worst_df,
            [
                "start_entry_time",
                "end_entry_time",
                "length",
                "pnl_sum",
                "mean_pnl",
                "sl_share_in_segment",
                "dominant_exit_reason",
                "dominant_session_bucket",
                "dominant_vol_bucket",
                "median_fill_delay_min",
            ],
        )
    )
    aa_lines.append("")
    aa_lines.append("## Tail Attribution Highlights")
    aa_lines.append("")
    if tail_df.empty:
        aa_lines.append("No tail rows available.")
    else:
        h = tail_df[
            (tail_df["tail_name"] == "cvar_5")
            & (tail_df["axis"].isin(["sl_hit", "session_bucket", "vol_bucket", "entry_mechanic"]))
        ].copy()
        h = h.sort_values(["axis", "share_of_tail_loss_abs"], ascending=[True, False]).groupby("axis", as_index=False).head(3)
        aa_lines.append(table_md(h, ["tail_name", "axis", "bucket", "trades_count", "loss_abs_sum", "share_of_tail_loss_abs"]))
    write_text(run_dir / "phaseAA_loss_clustering_report.md", "\n".join(aa_lines))

    repro = {
        "generated_utc": utc_now(),
        "phase": "AA",
        "run_dir": str(run_dir),
        "symbol": LOCKED["symbol"],
        "baseline_selection": {
            "selected_hash": baseline_hash,
            "selected_exec_id": baseline_exec_id,
            "source_phaseS_dir": str(qrs_dir),
            "source_phaseV_dir": str(phasev_dir),
            "phaseS_top_row": phaseS_df.iloc[0].to_dict() if not phaseS_df.empty else {},
        },
        "frozen_contract": {
            "representative_subset_csv": str(rep_fp),
            "canonical_fee_model": str(fee_fp),
            "canonical_metrics_definition": str(metrics_fp),
            "fee_sha256": fee_sha,
            "metrics_sha256": metrics_sha,
            "fee_hash_match": int(fee_sha == LOCKED["expected_fee_sha"]),
            "metrics_hash_match": int(metrics_sha == LOCKED["expected_metrics_sha"]),
        },
        "ga_exec_freeze_lock_validation": lock_validation,
        "baseline_metrics": {
            "valid_for_ranking": int(base_row["valid_for_ranking"]),
            "exec_expectancy_net": float(base_row["exec_expectancy_net"]),
            "exec_cvar_5": float(base_row["exec_cvar_5"]),
            "exec_max_drawdown": float(base_row["exec_max_drawdown"]),
            "entries_valid": int(base_row["entries_valid"]),
            "entry_rate": float(base_row["entry_rate"]),
            "max_consecutive_losses": int(base_cluster["max_consecutive_losses"]),
            "streak_ge5_count": int(base_cluster["streak_ge5_count"]),
            "streak_ge10_count": int(base_cluster["streak_ge10_count"]),
            "sl_loss_share": float(base_cluster["sl_loss_share"]),
        },
        "trade_table_path": str(trade_table_path),
    }
    json_dump(run_dir / "phaseAA_reproduction_check.json", repro)

    # Phase AB - structural risk controls spec (no time/session veto).
    controls: List[Dict[str, Any]] = [
        {
            "control_id": "C1_cooldown_decluster",
            "description": "Increase deterministic cooldown to reduce bursty consecutive entries.",
            "params_default": {"cooldown_min": max(45, int(baseline_genome.get("cooldown_min", 20)) + 25)},
            "bounds": {"cooldown_min": [20, 120]},
            "targets": ["max_consecutive_losses", "streak_ge5_count"],
            "risks": ["lower participation", "possible entry-rate pressure"],
        },
        {
            "control_id": "C2_break_even_early",
            "description": "Arm break-even earlier to cut SL-heavy reversals without session veto.",
            "params_default": {"break_even_enabled": 1, "break_even_trigger_r": 0.55, "break_even_offset_bps": 0.0},
            "bounds": {"break_even_trigger_r": [0.35, 0.90], "break_even_offset_bps": [0.0, 2.0]},
            "targets": ["sl_loss_share", "cvar_5", "max_drawdown"],
            "risks": ["premature stop-outs", "reduced winner tail"],
        },
        {
            "control_id": "C3_trailing_tail_guard",
            "description": "Enable earlier trailing protection to cap adverse reversals after favorable excursion.",
            "params_default": {"trailing_enabled": 1, "trail_start_r": 1.00, "trail_step_bps": 12.0},
            "bounds": {"trail_start_r": [0.80, 1.40], "trail_step_bps": [6.0, 20.0]},
            "targets": ["cvar_5", "max_drawdown", "loss_run_ge3_count"],
            "risks": ["winner truncation", "higher exit churn"],
        },
        {
            "control_id": "C4_time_stop_tighten",
            "description": "Tighten long horizon time-stop to reduce slow-bleed losses and long exposure tails.",
            "params_default": {"time_stop_min": int(max(360, min(1200, int(baseline_genome.get("time_stop_min", 2052)) // 2)))},
            "bounds": {"time_stop_min": [360, 1800]},
            "targets": ["sl_loss_share", "max_consecutive_losses", "max_drawdown"],
            "risks": ["cuts late recoveries", "can lower expectancy if too tight"],
        },
    ]

    spec_lines: List[str] = []
    spec_lines.append("# Phase AB Risk Controls Spec")
    spec_lines.append("")
    spec_lines.append(f"- Generated UTC: {utc_now()}")
    spec_lines.append("- Scope: structural controls in execution engine only; no session/killzone veto overlays.")
    spec_lines.append("- Hard gates: unchanged.")
    spec_lines.append("")
    spec_lines.append("## AA Evidence Used")
    spec_lines.append("")
    spec_lines.append(f"- max_consecutive_losses: `{int(base_cluster['max_consecutive_losses'])}`")
    spec_lines.append(f"- streak>=5 count: `{int(base_cluster['streak_ge5_count'])}`")
    spec_lines.append(f"- sl_loss_share: `{float(base_cluster['sl_loss_share']):.6f}`")
    spec_lines.append(f"- conditional_loss_rate_after_loss: `{float(base_cluster['conditional_loss_rate_after_loss']):.6f}`")
    spec_lines.append("")
    spec_lines.append("## Control Family")
    spec_lines.append("")
    for c in controls:
        spec_lines.append(f"### {c['control_id']}")
        spec_lines.append("")
        spec_lines.append(f"- Description: {c['description']}")
        spec_lines.append(f"- Default params for AC ablation: `{json.dumps(c['params_default'], sort_keys=True)}`")
        spec_lines.append(f"- Expected impact channels: `{', '.join(c['targets'])}`")
        spec_lines.append(f"- Risks: `{', '.join(c['risks'])}`")
        spec_lines.append("")
    write_text(run_dir / "phaseAB_risk_controls_spec.md", "\n".join(spec_lines))

    bounds_yaml_lines: List[str] = []
    bounds_yaml_lines.append("phase: AB")
    bounds_yaml_lines.append("symbol: SOLUSDT")
    bounds_yaml_lines.append("control_family:")
    for c in controls:
        bounds_yaml_lines.append(f"  - control_id: {c['control_id']}")
        bounds_yaml_lines.append(f"    description: \"{c['description']}\"")
        bounds_yaml_lines.append("    bounds:")
        for k, v in c["bounds"].items():
            bounds_yaml_lines.append(f"      {k}: [{v[0]}, {v[1]}]")
        bounds_yaml_lines.append("    default_for_ac:")
        for k, v in c["params_default"].items():
            if isinstance(v, float):
                bounds_yaml_lines.append(f"      {k}: {v:.10g}")
            else:
                bounds_yaml_lines.append(f"      {k}: {v}")
    write_text(run_dir / "phaseAB_param_bounds.yaml", "\n".join(bounds_yaml_lines))

    # Phase AC - controlled ablations.
    rows: List[Dict[str, Any]] = []

    rows.append(base_row)
    # Single-control variants.
    control_rows: List[Dict[str, Any]] = []
    for c in controls:
        g = copy.deepcopy(baseline_genome)
        for k, v in c["params_default"].items():
            g[k] = v
        r, _sig, _tr = evaluate_variant(
            variant_id=c["control_id"],
            variant_type="single_control",
            controls_applied=c["control_id"],
            genome=g,
            bundle=bundle,
            args=args,
            signals_df=sig_input,
            baseline_row=base_row,
        )
        rows.append(r)
        control_rows.append(r)

    # Top-2 controls by clustering improvement first, then expectancy/risk.
    ctr_df = pd.DataFrame(control_rows)
    combo_candidates: List[str] = []
    if not ctr_df.empty:
        z = ctr_df.copy()
        z["score"] = (
            to_num(z["max_consecutive_losses_reduction_ratio"]).fillna(-1e9)
            + to_num(z["streak_ge5_reduction_ratio"]).fillna(-1e9)
            + 0.25 * to_num(z["cvar_improve_ratio_vs_baseline"]).fillna(0.0)
            + 0.25 * to_num(z["maxdd_improve_ratio_vs_baseline"]).fillna(0.0)
            + 5.0 * to_num(z["delta_exec_expectancy_vs_baseline"]).fillna(-1e9)
        )
        z = z.sort_values(["score", "valid_for_ranking"], ascending=[False, False]).reset_index(drop=True)
        combo_candidates = z["variant_id"].head(2).astype(str).tolist()

    if len(combo_candidates) == 2:
        c_map = {c["control_id"]: c for c in controls}
        g = copy.deepcopy(baseline_genome)
        for cid in combo_candidates:
            for k, v in c_map[cid]["params_default"].items():
                g[k] = v
        combo_id = f"COMBO_{combo_candidates[0]}__{combo_candidates[1]}"
        r_combo, _sig_c, _tr_c = evaluate_variant(
            variant_id=combo_id,
            variant_type="combo_top2",
            controls_applied="|".join(combo_candidates),
            genome=g,
            bundle=bundle,
            args=args,
            signals_df=sig_input,
            baseline_row=base_row,
        )
        rows.append(r_combo)

    ac_df = pd.DataFrame(rows)
    if ac_df.empty:
        raise RuntimeError("AC results are empty")

    # AC GO criteria.
    crit = ac_df[ac_df["variant_id"] != "baseline_e_winner"].copy()
    crit["go_pass"] = (
        (to_num(crit["max_consecutive_losses_reduction_ratio"]) >= 0.20)
        & (to_num(crit["streak_ge5_reduction_ratio"]) >= 0.20)
        & (to_num(crit["valid_for_ranking"]) == 1)
        & (to_num(crit["delta_exec_expectancy_vs_baseline"]) >= -0.00002)
    ).astype(int)
    go_rows = crit[crit["go_pass"] == 1].copy()
    ac_go = int(len(go_rows) > 0)

    # Add criterion columns to output table.
    ac_df = ac_df.merge(crit[["variant_id", "go_pass"]], on="variant_id", how="left")
    ac_df["go_pass"] = to_num(ac_df["go_pass"]).fillna(0).astype(int)
    ac_df.to_csv(run_dir / "phaseAC_ablation_results.csv", index=False)

    # Report + decision.
    rep_lines: List[str] = []
    rep_lines.append("# Phase AC Controlled Ablation Report")
    rep_lines.append("")
    rep_lines.append(f"- Generated UTC: {utc_now()}")
    rep_lines.append(f"- Baseline genome hash: `{baseline_hash}` ({baseline_exec_id})")
    rep_lines.append("- Grid: baseline + each single control + top-2 control combo.")
    rep_lines.append("- Hard gates unchanged. No time/session veto overlays.")
    rep_lines.append("")
    rep_lines.append("## Results")
    rep_lines.append("")
    rep_lines.append(
        table_md(
            ac_df.sort_values(["variant_type", "variant_id"]),
            [
                "variant_id",
                "variant_type",
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
                "sl_loss_share",
                "entries_valid",
                "entry_rate",
                "taker_share",
                "p95_fill_delay_min",
                "min_split_expectancy_net",
                "go_pass",
                "invalid_reason",
            ],
        )
    )
    rep_lines.append("")
    rep_lines.append("## Decision Logic")
    rep_lines.append("")
    rep_lines.append("- GO criteria:")
    rep_lines.append("  max_consecutive_losses reduction >= 20%, streak>=5 reduction >= 20%, valid_for_ranking=1, delta_expectancy >= -0.00002.")
    rep_lines.append(f"- AC decision: {'AC_GO' if ac_go == 1 else 'AC_NO_GO'}")
    if ac_go == 1:
        best = go_rows.sort_values(["delta_exec_expectancy_vs_baseline", "maxdd_improve_ratio_vs_baseline"], ascending=[False, False]).iloc[0].to_dict()
        rep_lines.append(f"- Best GO variant: `{best['variant_id']}`")
    else:
        near = crit.sort_values(
            ["max_consecutive_losses_reduction_ratio", "streak_ge5_reduction_ratio", "delta_exec_expectancy_vs_baseline"],
            ascending=[False, False, False],
        ).head(1)
        if not near.empty:
            nrow = near.iloc[0].to_dict()
            rep_lines.append(
                f"- Closest variant: `{nrow['variant_id']}` with reductions "
                f"mcl={float(nrow.get('max_consecutive_losses_reduction_ratio', np.nan)):.4f}, "
                f"streak5={float(nrow.get('streak_ge5_reduction_ratio', np.nan)):.4f}, "
                f"delta_exp={float(nrow.get('delta_exec_expectancy_vs_baseline', np.nan)):.8f}, "
                f"valid={int(nrow.get('valid_for_ranking', 0))}."
            )
    write_text(run_dir / "phaseAC_ablation_report.md", "\n".join(rep_lines))

    dec_lines: List[str] = []
    dec_lines.append("# Phase AC Decision")
    dec_lines.append("")
    dec_lines.append(f"- Generated UTC: {utc_now()}")
    dec_lines.append(f"- Final: **{'AC_GO' if ac_go == 1 else 'AC_NO_GO'}**")
    dec_lines.append(f"- Baseline genome hash: `{baseline_hash}`")
    dec_lines.append(f"- Variants tested: `{len(ac_df)}`")
    dec_lines.append(f"- Variants meeting GO criteria: `{len(go_rows)}`")
    write_text(run_dir / "phaseAC_decision.md", "\n".join(dec_lines))

    if ac_go == 1:
        go_best = go_rows.sort_values(["delta_exec_expectancy_vs_baseline", "maxdd_improve_ratio_vs_baseline"], ascending=[False, False]).iloc[0].to_dict()
        prompt = f"""Execution-layer Phase AD bounded GA (SOLUSDT, contract-locked):
Use the same frozen representative subset and canonical fee/metrics hash lock, with hard gates unchanged.
Seed the search around baseline genome hash {baseline_hash} and proven risk-control variant {go_best['variant_id']}.
Search only local ranges from phaseAB_param_bounds.yaml for the active control family; keep all other execution knobs near current best values.
Run bounded GA only:
- symbol=SOLUSDT
- mode=tight
- walkforward on (wf_splits=5, train_ratio=0.70)
- pop=192
- gens=8
- workers=4
- seed=20260303
- allow_freeze_hash_mismatch=0
Required outputs: full genomes.csv, invalid_reason_histogram.json, duplicate/effective-trials summaries, and comparison vs baseline hash {baseline_hash}.
Stop with NO_GO if no candidate improves clustering materially while keeping expectancy degradation within -0.00002 and valid_for_ranking=1.
"""
        write_text(run_dir / "ready_to_launch_phaseAD_ga_prompt.txt", prompt)

    manifest = {
        "generated_utc": utc_now(),
        "phase": "AA_AB_AC",
        "run_dir": str(run_dir),
        "duration_sec": float(time.time() - t0),
        "symbol": LOCKED["symbol"],
        "seed": int(args_cli.seed),
        "source_phaseS_dir": str(qrs_dir),
        "source_phaseV_dir": str(phasev_dir),
        "baseline_hash": baseline_hash,
        "baseline_exec_id": baseline_exec_id,
        "freeze_lock_validation": lock_validation,
        "ac_decision": "AC_GO" if ac_go == 1 else "AC_NO_GO",
        "ac_variants_tested": int(len(ac_df)),
        "ac_go_count": int(len(go_rows)),
    }
    json_dump(run_dir / "phaseAA_run_manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "ac_decision": manifest["ac_decision"], "baseline_hash": baseline_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
