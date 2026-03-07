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

from scripts import backtest_exec_phasec_sol as phasec_bt  # noqa: E402
from scripts import instant_loser_vs_winner_entry_forensics as forensics  # noqa: E402
from scripts import repaired_frontier_contamination_audit as audit  # noqa: E402


RUN_PREFIX = "REPAIRED_UNIVERSE_PREEXEC_PATHOLOGY_AUDIT"


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


def load_rerun_manifest_from_freeze(freeze_dir: Path) -> Dict[str, Any]:
    freeze_manifest = json.loads((freeze_dir / "repaired_universe_freeze_manifest.json").read_text(encoding="utf-8"))
    source_truth = freeze_manifest.get("source_of_truth")
    rerun_dir: Optional[Path] = None
    if isinstance(source_truth, dict):
        rerun_dir = Path(str(source_truth.get("rerun_dir", ""))).resolve() if source_truth.get("rerun_dir") else None
    elif isinstance(source_truth, str) and source_truth.strip():
        rerun_dir = Path(source_truth).resolve()
    if rerun_dir is None:
        source_files = dict(freeze_manifest.get("source_files", {}) or {})
        rerun_manifest_path = source_files.get("rerun_manifest")
        if isinstance(rerun_manifest_path, str) and rerun_manifest_path.strip():
            cand = Path(rerun_manifest_path).resolve()
            if cand.exists():
                rerun_dir = cand.parent
    if rerun_dir and (rerun_dir / "rerun_manifest.json").exists():
        return json.loads((rerun_dir / "rerun_manifest.json").read_text(encoding="utf-8"))
    return {}


def load_fee_model(rerun_manifest: Dict[str, Any]) -> Tuple[phasec_bt.FeeModel, Path]:
    cfg = dict(rerun_manifest.get("config", {}) or {})
    fee_source = cfg.get("fee_model_source")
    fee_path = Path(str(fee_source)).resolve() if fee_source else None
    if fee_path and fee_path.exists():
        return phasec_bt._load_fee_model(fee_path), fee_path  # pylint: disable=protected-access
    repaired_signal_root = audit.rebaseline_1h._latest_multicoin_signal_root().resolve()  # pylint: disable=protected-access
    fee_path = repaired_signal_root / "fee_model.json"
    return phasec_bt._load_fee_model(fee_path), fee_path  # pylint: disable=protected-access


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


def build_signals_for_row(
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
    return audit.build_signal_table(
        symbol=symbol,
        params_dict=params,
        df_feat=df_feat,
        period_start=period_start,
        period_end=period_end,
    )


def first_eligible_path_features_from_arrays(
    symbol: str,
    entry_time: Any,
    entry_price: Any,
    market_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    fill_ts = pd.to_datetime(entry_time, utc=True, errors="coerce")
    px = audit.safe_float(entry_price)
    if pd.isna(fill_ts) or (not np.isfinite(px)) or px <= 0.0:
        return {
            "first_eligible_bar_ts": pd.NaT,
            "first1_mae_pct": float("nan"),
            "first1_mfe_pct": float("nan"),
            "first2_mae_pct": float("nan"),
            "first2_mfe_pct": float("nan"),
            "first1_close_ret_pct": float("nan"),
            "first2_close_ret_pct": float("nan"),
        }
    market = audit.load_market_arrays(symbol, market_cache)
    start_idx = int(np.searchsorted(market["ts_ns"], int(fill_ts.value), side="right"))
    if start_idx >= len(market["ts_ns"]):
        return {
            "first_eligible_bar_ts": pd.NaT,
            "first1_mae_pct": float("nan"),
            "first1_mfe_pct": float("nan"),
            "first2_mae_pct": float("nan"),
            "first2_mfe_pct": float("nan"),
            "first1_close_ret_pct": float("nan"),
            "first2_close_ret_pct": float("nan"),
        }
    hi1 = float(market["high"][start_idx])
    lo1 = float(market["low"][start_idx])
    cl1 = float(market["close"][start_idx])
    end_idx = min(len(market["ts_ns"]), start_idx + 2)
    hi2 = float(np.nanmax(market["high"][start_idx:end_idx]))
    lo2 = float(np.nanmin(market["low"][start_idx:end_idx]))
    cl2 = float(market["close"][end_idx - 1])
    return {
        "first_eligible_bar_ts": pd.to_datetime(int(market["ts_ns"][start_idx]), utc=True),
        "first1_mae_pct": float(lo1 / px - 1.0) if np.isfinite(lo1) else float("nan"),
        "first1_mfe_pct": float(hi1 / px - 1.0) if np.isfinite(hi1) else float("nan"),
        "first2_mae_pct": float(lo2 / px - 1.0) if np.isfinite(lo2) else float("nan"),
        "first2_mfe_pct": float(hi2 / px - 1.0) if np.isfinite(hi2) else float("nan"),
        "first1_close_ret_pct": float(cl1 / px - 1.0) if np.isfinite(cl1) else float("nan"),
        "first2_close_ret_pct": float(cl2 / px - 1.0) if np.isfinite(cl2) else float("nan"),
    }


def build_trade_feature_frame(
    *,
    symbol: str,
    params: Dict[str, Any],
    trades_df: pd.DataFrame,
    full_1h_df: pd.DataFrame,
    market_cache: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    trades = trades_df.copy()
    trades["filled"] = pd.to_numeric(trades.get("filled"), errors="coerce").fillna(0).astype(int)
    trades["valid_for_metrics"] = pd.to_numeric(trades.get("valid_for_metrics"), errors="coerce").fillna(0).astype(int)
    trades["hold_minutes"] = pd.to_numeric(trades.get("hold_minutes"), errors="coerce")
    trades["pnl_net_pct"] = pd.to_numeric(trades.get("pnl_net_pct"), errors="coerce")
    trades["entry_time"] = pd.to_datetime(trades.get("entry_time"), utc=True, errors="coerce")
    trades["exit_time"] = pd.to_datetime(trades.get("exit_time"), utc=True, errors="coerce")
    trades["signal_time"] = pd.to_datetime(trades.get("signal_time"), utc=True, errors="coerce")
    trades["exit_reason"] = trades.get("exit_reason", "").astype(str)
    valid = trades[(trades["filled"] == 1) & (trades["valid_for_metrics"] == 1) & trades["pnl_net_pct"].notna()].copy()
    rows: List[Dict[str, Any]] = []
    for row in valid.itertuples(index=False):
        hold_min = audit.safe_float(getattr(row, "hold_minutes", np.nan))
        pnl_net = audit.safe_float(getattr(row, "pnl_net_pct", np.nan))
        exit_reason = str(getattr(row, "exit_reason", ""))
        bucket = "neutral_small_win"
        if forensics.is_stop_exit(exit_reason) and np.isfinite(hold_min) and hold_min <= 60.0:
            bucket = "instant_loser"
        elif forensics.is_stop_exit(exit_reason) and np.isfinite(hold_min) and 60.0 < hold_min <= 240.0:
            bucket = "fast_loser"
        elif pnl_net >= float(forensics.MEANINGFUL_WINNER_PNL_THRESHOLD):
            bucket = "meaningful_winner"

        signal_time = pd.to_datetime(getattr(row, "signal_time", None), utc=True, errors="coerce")
        signal_bar = forensics.locate_bar_row(full_1h_df, signal_time)
        signal_close = audit.safe_float(getattr(signal_bar, "Close", np.nan) if signal_bar is not None else np.nan)
        entry_price = audit.safe_float(getattr(row, "entry_price", np.nan))
        action_gap_pct = (
            float(entry_price / signal_close - 1.0)
            if np.isfinite(entry_price) and np.isfinite(signal_close) and signal_close > 0.0
            else float("nan")
        )
        path_feats = first_eligible_path_features_from_arrays(
            symbol=symbol,
            entry_time=getattr(row, "entry_time", None),
            entry_price=entry_price,
            market_cache=market_cache,
        )
        rows.append(
            {
                "symbol": symbol,
                "signal_id": str(getattr(row, "signal_id", "")),
                "signal_time": signal_time,
                "entry_time": pd.to_datetime(getattr(row, "entry_time", None), utc=True, errors="coerce"),
                "exit_time": pd.to_datetime(getattr(row, "exit_time", None), utc=True, errors="coerce"),
                "bucket": bucket,
                "exit_reason": exit_reason,
                "hold_minutes": hold_min,
                "pnl_net_pct": pnl_net,
                "entry_price": entry_price,
                "exit_price": audit.safe_float(getattr(row, "exit_price", np.nan)),
                "mae_pct": audit.safe_float(getattr(row, "mae_pct", np.nan)),
                "mfe_pct": audit.safe_float(getattr(row, "mfe_pct", np.nan)),
                "signal_range_pct": audit.safe_float(getattr(signal_bar, "signal_range_pct", np.nan) if signal_bar is not None else np.nan),
                "signal_body_abs_pct": audit.safe_float(getattr(signal_bar, "signal_body_abs_pct", np.nan) if signal_bar is not None else np.nan),
                "upper_wick_ratio": audit.safe_float(getattr(signal_bar, "upper_wick_ratio", np.nan) if signal_bar is not None else np.nan),
                "lower_wick_ratio": audit.safe_float(getattr(signal_bar, "lower_wick_ratio", np.nan) if signal_bar is not None else np.nan),
                "atr14_pct": audit.safe_float(getattr(signal_bar, "atr14_pct", np.nan) if signal_bar is not None else np.nan),
                "dist_to_sma20_pct": audit.safe_float(getattr(signal_bar, "dist_to_sma20_pct", np.nan) if signal_bar is not None else np.nan),
                "breakout_dist_pct": audit.safe_float(getattr(signal_bar, "breakout_dist_pct", np.nan) if signal_bar is not None else np.nan),
                "ret_3h_pct": audit.safe_float(getattr(signal_bar, "ret_3h_pct", np.nan) if signal_bar is not None else np.nan),
                "ret_6h_pct": audit.safe_float(getattr(signal_bar, "ret_6h_pct", np.nan) if signal_bar is not None else np.nan),
                "action_gap_pct": action_gap_pct,
                "stop_distance_pct_signal": max(1e-8, 1.0 - float(getattr(row, "signal_sl_mult", params.get("stop_loss_mult", 1.0)))),
                "meaningful_winner_threshold_pct": float(forensics.MEANINGFUL_WINNER_PNL_THRESHOLD),
                **path_feats,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "symbol",
                "signal_id",
                "bucket",
                "hold_minutes",
                "pnl_net_pct",
                "first1_mae_pct",
                "first2_mae_pct",
                "first1_close_ret_pct",
                "first2_close_ret_pct",
            ]
        )
    return pd.DataFrame(rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def pct(num: int, den: int) -> float:
    return 100.0 * rate(num, den)


def median_or_nan(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce")
    x = x[np.isfinite(x)]
    return float(x.median()) if len(x) else float("nan")


def build_symbol_summary(symbol: str, feat_df: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(feat_df))
    instant = int((feat_df["bucket"] == "instant_loser").sum()) if total else 0
    fast = int((feat_df["bucket"] == "fast_loser").sum()) if total else 0
    winners = int((feat_df["bucket"] == "meaningful_winner").sum()) if total else 0
    neutral = int((feat_df["bucket"] == "neutral_small_win").sum()) if total else 0
    loser_df = feat_df[feat_df["bucket"].isin(["instant_loser", "fast_loser"])].copy()
    winner_df = feat_df[feat_df["bucket"] == "meaningful_winner"].copy()

    loser_first1_mae = median_or_nan(loser_df.get("first1_mae_pct", pd.Series(dtype=float)))
    loser_first2_mae = median_or_nan(loser_df.get("first2_mae_pct", pd.Series(dtype=float)))
    loser_first2_close = median_or_nan(loser_df.get("first2_close_ret_pct", pd.Series(dtype=float)))
    winner_first1_mae = median_or_nan(winner_df.get("first1_mae_pct", pd.Series(dtype=float)))
    winner_first2_mae = median_or_nan(winner_df.get("first2_mae_pct", pd.Series(dtype=float)))
    winner_first2_close = median_or_nan(winner_df.get("first2_close_ret_pct", pd.Series(dtype=float)))

    asymmetry_mae = (
        winner_first2_mae - loser_first2_mae
        if np.isfinite(winner_first2_mae) and np.isfinite(loser_first2_mae)
        else float("nan")
    )
    asymmetry_close = (
        winner_first2_close - loser_first2_close
        if np.isfinite(winner_first2_close) and np.isfinite(loser_first2_close)
        else float("nan")
    )
    asymmetry_strength = 0.0
    if np.isfinite(asymmetry_mae):
        asymmetry_strength += max(0.0, asymmetry_mae)
    if np.isfinite(asymmetry_close):
        asymmetry_strength += max(0.0, asymmetry_close)

    loser_burden = rate(instant + fast, total)
    winner_rate = rate(winners, total)
    trade_factor = min(1.0, float(total) / 150.0)
    asymmetry_scaled = min(1.0, asymmetry_strength / 0.01) if asymmetry_strength > 0.0 else 0.0
    priority_score = trade_factor * (0.45 * loser_burden + 0.35 * winner_rate + 0.20 * asymmetry_scaled)

    if total < 50 or winner_rate < 0.05 or (loser_burden >= 0.82 and winner_rate < 0.08 and asymmetry_scaled < 0.30):
        classification = "LOW_PRIORITY_FOR_3M"
    elif rate(instant, total) >= 0.70 and winner_rate >= 0.10 and asymmetry_scaled >= 0.35:
        classification = "HIGH_INSTANT_LOSER_BURDEN"
    elif rate(fast, total) >= 0.15 and winner_rate >= 0.10 and asymmetry_scaled >= 0.30:
        classification = "HIGH_FAST_LOSER_BURDEN"
    elif winner_rate >= 0.12 and loser_burden <= 0.75 and asymmetry_scaled >= 0.35:
        classification = "HEALTHY_WINNER_FORMATION"
    else:
        classification = "MIXED"

    if not np.isfinite(asymmetry_mae) and not np.isfinite(asymmetry_close):
        asymmetry_note = "winner/loser separation is weak because there are too few meaningful winners"
    elif (np.isfinite(asymmetry_mae) and asymmetry_mae > 0.0) and (np.isfinite(asymmetry_close) and asymmetry_close > 0.0):
        asymmetry_note = "winners absorb less early adverse excursion and close stronger by the second eligible bar"
    elif np.isfinite(asymmetry_close) and asymmetry_close > 0.0:
        asymmetry_note = "winners recover faster by the second eligible bar, but adverse excursion separation is weaker"
    elif np.isfinite(asymmetry_mae) and asymmetry_mae > 0.0:
        asymmetry_note = "winners suffer less early damage, but close behavior is not sharply separated"
    else:
        asymmetry_note = "winner and loser early-path behavior overlaps materially"

    return {
        "symbol": symbol,
        "trade_count": total,
        "instant_loser_count": instant,
        "instant_loser_rate": rate(instant, total),
        "instant_loser_rate_pct": pct(instant, total),
        "fast_loser_count": fast,
        "fast_loser_rate": rate(fast, total),
        "fast_loser_rate_pct": pct(fast, total),
        "meaningful_winner_count": winners,
        "meaningful_winner_rate": rate(winners, total),
        "meaningful_winner_rate_pct": pct(winners, total),
        "neutral_small_win_count": neutral,
        "loser_total_count": int(len(loser_df)),
        "loser_total_rate": loser_burden,
        "loser_total_rate_pct": 100.0 * loser_burden,
        "loser_first1_mae_pct_median": loser_first1_mae,
        "loser_first2_mae_pct_median": loser_first2_mae,
        "loser_first2_close_ret_pct_median": loser_first2_close,
        "winner_first1_mae_pct_median": winner_first1_mae,
        "winner_first2_mae_pct_median": winner_first2_mae,
        "winner_first2_close_ret_pct_median": winner_first2_close,
        "first2_mae_asymmetry": asymmetry_mae,
        "first2_close_asymmetry": asymmetry_close,
        "asymmetry_strength": asymmetry_strength,
        "obvious_asymmetry_note": asymmetry_note,
        "classification": classification,
        "priority_score": priority_score,
    }


def final_recommendation(summary_df: pd.DataFrame) -> Tuple[str, List[str]]:
    low_priority = summary_df[summary_df["classification"] == "LOW_PRIORITY_FOR_3M"].copy()
    strong = summary_df[summary_df["classification"].isin(["HEALTHY_WINNER_FORMATION", "HIGH_INSTANT_LOSER_BURDEN", "HIGH_FAST_LOSER_BURDEN"])].copy()
    ordered = summary_df.sort_values(
        ["priority_score", "meaningful_winner_rate", "trade_count", "symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    if len(low_priority) >= max(4, int(math.ceil(len(summary_df) * 0.4))):
        keep = ordered[ordered["classification"] != "LOW_PRIORITY_FOR_3M"]["symbol"].astype(str).tolist()
        return "TRIM_WEAK_SYMBOLS_BEFORE_3M", keep
    if len(strong) < len(summary_df):
        keep_n = max(5, min(8, len(strong) if len(strong) else len(ordered)))
        keep = ordered.head(keep_n)["symbol"].astype(str).tolist()
        return "RUN_3M_ON_PRIORITY_SUBSET_FIRST", keep
    return "RUN_3M_ON_FULL_REPAIRED_UNIVERSE", ordered["symbol"].astype(str).tolist()


def build_summary_table(summary_df: pd.DataFrame, recommendation: str, priority_symbols: List[str]) -> pd.DataFrame:
    rows = [
        {"metric": "symbol_count", "value": int(len(summary_df))},
        {"metric": "high_instant_loser_burden_count", "value": int((summary_df["classification"] == "HIGH_INSTANT_LOSER_BURDEN").sum())},
        {"metric": "high_fast_loser_burden_count", "value": int((summary_df["classification"] == "HIGH_FAST_LOSER_BURDEN").sum())},
        {"metric": "healthy_winner_formation_count", "value": int((summary_df["classification"] == "HEALTHY_WINNER_FORMATION").sum())},
        {"metric": "mixed_count", "value": int((summary_df["classification"] == "MIXED").sum())},
        {"metric": "low_priority_count", "value": int((summary_df["classification"] == "LOW_PRIORITY_FOR_3M").sum())},
        {"metric": "median_instant_loser_rate_pct", "value": float(pd.to_numeric(summary_df["instant_loser_rate_pct"], errors="coerce").median())},
        {"metric": "median_fast_loser_rate_pct", "value": float(pd.to_numeric(summary_df["fast_loser_rate_pct"], errors="coerce").median())},
        {"metric": "median_meaningful_winner_rate_pct", "value": float(pd.to_numeric(summary_df["meaningful_winner_rate_pct"], errors="coerce").median())},
        {"metric": "recommendation", "value": recommendation},
        {"metric": "priority_subset", "value": ",".join(priority_symbols)},
    ]
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Pre-execution loser/winner pathology audit on the frozen repaired universe")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--freeze-dir", default="")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    freeze_dir = discover_freeze_dir(args.freeze_dir)
    rerun_manifest = load_rerun_manifest_from_freeze(freeze_dir)
    fee_model, fee_model_path = load_fee_model(rerun_manifest)

    universe_fp = freeze_dir / "repaired_best_by_symbol.csv"
    selected_params_dir = freeze_dir / "repaired_universe_selected_params"
    universe_df = pd.read_csv(universe_fp)
    required_cols = {
        "symbol",
        "side",
        "candidate_id",
        "param_hash",
        "period_start",
        "period_end",
        "initial_equity",
    }
    missing_cols = sorted(required_cols - set(universe_df.columns))
    if missing_cols:
        raise KeyError(f"Missing required columns in repaired_best_by_symbol.csv: {missing_cols}")

    run_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = ensure_dir(run_root / f"{RUN_PREFIX}_{utc_tag()}")

    df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    raw_cache: Dict[str, pd.DataFrame] = {}
    market_cache: Dict[str, Dict[str, Any]] = {}
    per_symbol: List[Dict[str, Any]] = []
    detail_rows: List[pd.DataFrame] = []

    for row in universe_df.itertuples(index=False):
        series = pd.Series(row._asdict())
        symbol = str(series["symbol"]).upper()
        params = parse_params_from_row(series, selected_params_dir)
        signals_df = build_signals_for_row(row=series, params=params, df_cache=df_cache, raw_cache=raw_cache)
        split_lookup = {str(sid): 0 for sid in signals_df.get("signal_id", pd.Series(dtype=str)).astype(str).tolist()}
        trades_df = phasec_bt._simulate_1h_reference(  # pylint: disable=protected-access
            signals_df=signals_df,
            split_lookup=split_lookup,
            fee=fee_model,
            exec_horizon_hours=float(max(1.0, audit.safe_float(params.get("max_hold_hours")))),
            symbol=symbol,
            defer_exit_to_next_bar=True,
        )
        full_1h_df = forensics.load_full_1h_df(symbol)
        feat_df = build_trade_feature_frame(
            symbol=symbol,
            params=params,
            trades_df=trades_df,
            full_1h_df=full_1h_df,
            market_cache=market_cache,
        )
        if not feat_df.empty:
            detail_rows.append(feat_df.assign(candidate_id=str(series["candidate_id"]), param_hash=str(series["param_hash"])))
        summary = build_symbol_summary(symbol, feat_df)
        summary.update(
            {
                "side": str(series["side"]),
                "candidate_id": str(series["candidate_id"]),
                "param_hash": str(series["param_hash"]),
                "params_source": str(series.get("params_source", "")),
                "source_period_start": str(series["period_start"]),
                "source_period_end": str(series["period_end"]),
                "source_initial_equity": audit.safe_float(series["initial_equity"]),
            }
        )
        per_symbol.append(summary)

    by_symbol_df = pd.DataFrame(per_symbol)
    if by_symbol_df.empty:
        raise RuntimeError("No symbol pathology rows were produced")
    by_symbol_df = by_symbol_df.sort_values(
        ["priority_score", "meaningful_winner_rate", "trade_count", "symbol"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    by_symbol_df["priority_rank"] = np.arange(1, len(by_symbol_df) + 1)

    recommendation, priority_symbols = final_recommendation(by_symbol_df)
    summary_df = build_summary_table(by_symbol_df, recommendation, priority_symbols)

    by_symbol_fp = run_dir / "repaired_universe_pathology_by_symbol.csv"
    summary_fp = run_dir / "repaired_universe_pathology_summary.csv"
    report_fp = run_dir / "repaired_universe_pathology_report.md"

    by_symbol_df.to_csv(by_symbol_fp, index=False)
    summary_df.to_csv(summary_fp, index=False)

    lines: List[str] = []
    lines.append("# Repaired Universe Pre-Execution Pathology Audit")
    lines.append("")
    lines.append("This is a repaired 1h-only preflight diagnostic. It intentionally excludes any 3m execution overlay, local repair search, or new optimization.")
    lines.append("")
    lines.append("## Inputs Used")
    lines.append(f"- Frozen repaired universe dir: `{freeze_dir}`")
    lines.append(f"- Frozen universe table: `{universe_fp}`")
    lines.append(f"- Frozen selected params dir: `{selected_params_dir}`")
    lines.append(f"- Repaired fee model: `{fee_model_path}`")
    lines.append(f"- Reused code paths: `scripts/backtest_exec_phasec_sol.py`, `scripts/repaired_frontier_contamination_audit.py`, `scripts/instant_loser_vs_winner_entry_forensics.py`")
    lines.append("")
    lines.append("## Bucket Definitions")
    lines.append(f"- `instant_loser`: valid repaired 1h trade, stop-loss exit, hold <= 60 minutes.")
    lines.append(f"- `fast_loser`: valid repaired 1h trade, stop-loss exit, 60 < hold <= 240 minutes.")
    lines.append(f"- `meaningful_winner`: valid repaired 1h trade with net PnL >= {forensics.MEANINGFUL_WINNER_PNL_THRESHOLD:.4f}.")
    lines.append("")
    lines.append("## Per-Symbol Summary")
    lines.append(
        markdown_table(
            by_symbol_df,
            [
                "priority_rank",
                "symbol",
                "trade_count",
                "instant_loser_rate_pct",
                "fast_loser_rate_pct",
                "meaningful_winner_rate_pct",
                "classification",
                "priority_score",
            ],
            n=len(by_symbol_df),
        )
    )
    lines.append("")
    lines.append("## Observed Asymmetry")
    for row in by_symbol_df.itertuples(index=False):
        lines.append(
            f"- `{row.symbol}`: {row.obvious_asymmetry_note} "
            f"(loser first2 MAE median={row.loser_first2_mae_pct_median:.6f}, "
            f"winner first2 MAE median={row.winner_first2_mae_pct_median:.6f}, "
            f"loser first2 close median={row.loser_first2_close_ret_pct_median:.6f}, "
            f"winner first2 close median={row.winner_first2_close_ret_pct_median:.6f})."
        )
    lines.append("")
    lines.append("## Proven vs Assumed")
    lines.append("- Proven: all symbols were evaluated only from the frozen repaired universe candidates using the repaired 1h chronology-valid simulator with `defer_exit_to_next_bar=True`.")
    lines.append("- Proven: counts and rates come from the actual repaired 1h trade path, not from legacy or 3m-derived artifacts.")
    lines.append("- Assumed: downstream 3m execution work is most valuable where loser burden is material but meaningful winners still exist and early-path asymmetry is visible.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- Final recommendation: `{recommendation}`")
    lines.append(f"- Priority symbols for the next downstream 3m scope: `{', '.join(priority_symbols)}`")

    report_fp.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "generated_utc": utc_now().isoformat(),
        "freeze_dir": str(freeze_dir),
        "inputs": {
            "repaired_best_by_symbol": str(universe_fp),
            "repaired_universe_selected_params_dir": str(selected_params_dir),
            "fee_model_source": str(fee_model_path),
        },
        "reused_code_paths": [
            "scripts/backtest_exec_phasec_sol.py",
            "scripts/repaired_frontier_contamination_audit.py",
            "scripts/instant_loser_vs_winner_entry_forensics.py",
        ],
        "bucket_thresholds": {
            "instant_loser_hold_minutes_max": 60.0,
            "fast_loser_hold_minutes_max": 240.0,
            "meaningful_winner_pnl_threshold": float(forensics.MEANINGFUL_WINNER_PNL_THRESHOLD),
        },
        "recommendation": recommendation,
        "priority_symbols": priority_symbols,
        "artifacts": {
            "repaired_universe_pathology_by_symbol": str(by_symbol_fp),
            "repaired_universe_pathology_summary": str(summary_fp),
            "repaired_universe_pathology_report": str(report_fp),
        },
    }
    json_dump(run_dir / "repaired_universe_pathology_manifest.json", manifest)


if __name__ == "__main__":
    main()
