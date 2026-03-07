#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402


RUN_PREFIX = "INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS"
LIVE_COINS = ["LINKUSDT", "DOGEUSDT", "NEARUSDT", "LTCUSDT"]
LIVE_STATUS = {
    "LINKUSDT": "approved",
    "DOGEUSDT": "shadow",
    "NEARUSDT": "shadow",
    "LTCUSDT": "shadow",
}
DEFAULT_REPAIRED_MULTICOIN_DIR = Path(
    "/root/analysis/0.87/reports/execution_layer/REPAIRED_MULTICOIN_MODELA_AUDIT_20260302_234108"
).resolve()
MEANINGFUL_WINNER_PNL_THRESHOLD = 0.0020
MIN_WINNERS_FOR_DECISION = 8

ENTRY_FEATURES = [
    "signal_body_abs_pct",
    "signal_range_pct",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "atr14_pct",
    "breakout_dist_pct",
    "dist_to_sma20_pct",
    "action_gap_pct",
    "entry_improvement_bps",
    "exec_fill_delay_min",
    "taker_flag",
]
POST_ENTRY_FEATURES = [
    "first1_mae_pct",
    "first1_mfe_pct",
    "first2_mae_pct",
    "first2_mfe_pct",
    "first1_close_ret_pct",
    "first2_close_ret_pct",
]
ALL_FEATURES = ENTRY_FEATURES + POST_ENTRY_FEATURES + [
    "trend_up_1h",
    "atr_percentile_1h",
    "stop_distance_pct_signal",
]
LOWER_IS_BETTER = {
    "signal_body_abs_pct",
    "signal_range_pct",
    "upper_wick_ratio",
    "breakout_dist_pct",
    "dist_to_sma20_pct",
    "action_gap_pct",
    "exec_fill_delay_min",
    "taker_flag",
}
HIGHER_IS_BETTER = {
    "lower_wick_ratio",
    "entry_improvement_bps",
    "first1_mfe_pct",
    "first2_mfe_pct",
    "first1_close_ret_pct",
    "first2_close_ret_pct",
    "trend_up_1h",
}
FEATURE_LABELS = {
    "signal_body_abs_pct": "smaller 1h signal body",
    "signal_range_pct": "smaller 1h range expansion",
    "upper_wick_ratio": "smaller upper-wick share",
    "lower_wick_ratio": "larger lower-wick support",
    "atr14_pct": "lower local ATR percent",
    "breakout_dist_pct": "less breakout overextension",
    "dist_to_sma20_pct": "closer to 20h mean",
    "action_gap_pct": "smaller action-bar gap-up",
    "entry_improvement_bps": "better entry price improvement",
    "exec_fill_delay_min": "shorter fill delay",
    "taker_flag": "less taker-style entry",
    "first1_mae_pct": "shallower first eligible-bar adverse excursion",
    "first1_mfe_pct": "stronger first eligible-bar favorable excursion",
    "first2_mae_pct": "shallower first-2-bar adverse excursion",
    "first2_mfe_pct": "stronger first-2-bar favorable excursion",
    "first1_close_ret_pct": "better first eligible-bar close return",
    "first2_close_ret_pct": "better first-2-bar close return",
    "trend_up_1h": "better trend alignment",
    "atr_percentile_1h": "lower ATR percentile regime",
    "stop_distance_pct_signal": "larger initial stop distance",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(msg: str) -> None:
    print(msg, flush=True)


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


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: int = 20) -> str:
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


def finite_float(x: Any, default: float = float("nan")) -> float:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v):
        return float(default)
    return float(v)


def find_latest_repaired_multicoin_dir() -> Path:
    cands = sorted(
        [p for p in (PROJECT_ROOT / "reports" / "execution_layer").glob("REPAIRED_MULTICOIN_MODELA_AUDIT_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    for p in reversed(cands):
        required = {
            "repaired_multicoin_modelA_coin_classification.csv",
            "repaired_multicoin_modelA_run_manifest.json",
        }
        names = {f.name for f in p.iterdir() if f.is_file()}
        if required.issubset(names):
            return p.resolve()
    if DEFAULT_REPAIRED_MULTICOIN_DIR.exists():
        return DEFAULT_REPAIRED_MULTICOIN_DIR
    raise FileNotFoundError("No completed REPAIRED_MULTICOIN_MODELA_AUDIT directory found")


def load_best_variant_lookup(repaired_multicoin_dir: Path) -> pd.DataFrame:
    fp = repaired_multicoin_dir / "repaired_multicoin_modelA_coin_classification.csv"
    df = pd.read_csv(fp)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def hold_minutes_from_times(entry: Any, exit_: Any) -> float:
    et = pd.to_datetime(entry, utc=True, errors="coerce")
    xt = pd.to_datetime(exit_, utc=True, errors="coerce")
    if pd.isna(et) or pd.isna(xt):
        return float("nan")
    return float((xt - et).total_seconds() / 60.0)


def is_stop_exit(reason: Any) -> bool:
    r = str(reason).strip().lower()
    return ("sl" in r) or ("stop" in r)


def load_full_1h_df(symbol: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing 1h full parquet: {fp}")
    df = pd.read_parquet(fp)
    df = modela.exec3m._normalize_ohlcv_cols(df)  # pylint: disable=protected-access
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    keep = [c for c in ["Timestamp", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df.loc[:, keep].dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).sort_values("Timestamp").reset_index(drop=True)

    open_s = pd.to_numeric(df["Open"], errors="coerce")
    high_s = pd.to_numeric(df["High"], errors="coerce")
    low_s = pd.to_numeric(df["Low"], errors="coerce")
    close_s = pd.to_numeric(df["Close"], errors="coerce")
    prev_close = close_s.shift(1)
    tr = pd.concat(
        [
            (high_s - low_s),
            (high_s - prev_close).abs(),
            (low_s - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=5).mean()
    rng = (high_s - low_s)
    body = (close_s - open_s)
    denom = rng.replace(0.0, np.nan)

    df["atr14_pct"] = atr14 / open_s.replace(0.0, np.nan)
    df["signal_range_pct"] = rng / open_s.replace(0.0, np.nan)
    df["signal_body_pct"] = body / open_s.replace(0.0, np.nan)
    df["signal_body_abs_pct"] = body.abs() / open_s.replace(0.0, np.nan)
    df["upper_wick_ratio"] = (high_s - pd.concat([open_s, close_s], axis=1).max(axis=1)) / denom
    df["lower_wick_ratio"] = (pd.concat([open_s, close_s], axis=1).min(axis=1) - low_s) / denom
    df["sma20"] = close_s.rolling(20, min_periods=10).mean()
    df["dist_to_sma20_pct"] = (close_s - df["sma20"]) / df["sma20"].replace(0.0, np.nan)
    df["prior20_high"] = high_s.shift(1).rolling(20, min_periods=10).max()
    df["prior20_low"] = low_s.shift(1).rolling(20, min_periods=10).min()
    df["breakout_dist_pct"] = (close_s - df["prior20_high"]) / df["prior20_high"].replace(0.0, np.nan)
    df["dist_from_range_low_pct"] = (close_s - df["prior20_low"]) / df["prior20_low"].replace(0.0, np.nan)
    df["ret_3h_pct"] = close_s / close_s.shift(3) - 1.0
    df["ret_6h_pct"] = close_s / close_s.shift(6) - 1.0
    if "Volume" in df.columns:
        vol = pd.to_numeric(df["Volume"], errors="coerce")
        vol_mean = vol.rolling(20, min_periods=10).mean()
        vol_std = vol.rolling(20, min_periods=10).std(ddof=0)
        df["volume_z20"] = (vol - vol_mean) / vol_std.replace(0.0, np.nan)
    else:
        df["volume_z20"] = np.nan
    return df


def locate_bar_row(one_h_df: pd.DataFrame, ts: Any) -> Optional[pd.Series]:
    ts_utc = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(ts_utc):
        return None
    idx = one_h_df["Timestamp"].searchsorted(ts_utc, side="left")
    if idx < len(one_h_df) and pd.to_datetime(one_h_df.iloc[idx]["Timestamp"], utc=True) == ts_utc:
        return one_h_df.iloc[int(idx)]
    idx = one_h_df["Timestamp"].searchsorted(ts_utc, side="right") - 1
    if 0 <= idx < len(one_h_df):
        return one_h_df.iloc[int(idx)]
    return None


def first_eligible_path_features(one_h: modela.OneHMarket, fill_time: Any, entry_price: Any) -> Dict[str, Any]:
    fill_ts = pd.to_datetime(fill_time, utc=True, errors="coerce")
    px = finite_float(entry_price)
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

    start_idx = int(np.searchsorted(one_h.ts_ns, int(fill_ts.value), side="right"))
    if start_idx >= len(one_h.ts_ns):
        return {
            "first_eligible_bar_ts": pd.NaT,
            "first1_mae_pct": float("nan"),
            "first1_mfe_pct": float("nan"),
            "first2_mae_pct": float("nan"),
            "first2_mfe_pct": float("nan"),
            "first1_close_ret_pct": float("nan"),
            "first2_close_ret_pct": float("nan"),
        }

    hi1 = float(one_h.high_np[start_idx])
    lo1 = float(one_h.low_np[start_idx])
    cl1 = float(one_h.close_np[start_idx])
    end_idx = min(len(one_h.ts_ns), start_idx + 2)
    hi2 = float(np.nanmax(one_h.high_np[start_idx:end_idx]))
    lo2 = float(np.nanmin(one_h.low_np[start_idx:end_idx]))
    cl2 = float(one_h.close_np[end_idx - 1])
    return {
        "first_eligible_bar_ts": pd.to_datetime(int(one_h.ts_ns[start_idx]), utc=True),
        "first1_mae_pct": float(lo1 / px - 1.0) if np.isfinite(lo1) else float("nan"),
        "first1_mfe_pct": float(hi1 / px - 1.0) if np.isfinite(hi1) else float("nan"),
        "first2_mae_pct": float(lo2 / px - 1.0) if np.isfinite(lo2) else float("nan"),
        "first2_mfe_pct": float(hi2 / px - 1.0) if np.isfinite(hi2) else float("nan"),
        "first1_close_ret_pct": float(cl1 / px - 1.0) if np.isfinite(cl1) else float("nan"),
        "first2_close_ret_pct": float(cl2 / px - 1.0) if np.isfinite(cl2) else float("nan"),
    }


def compute_effect_size(win_s: pd.Series, lose_s: pd.Series) -> float:
    w = pd.to_numeric(win_s, errors="coerce").dropna().to_numpy(dtype=float)
    l = pd.to_numeric(lose_s, errors="coerce").dropna().to_numpy(dtype=float)
    if w.size < 2 or l.size < 2:
        return float("nan")
    mw = float(w.mean())
    ml = float(l.mean())
    vw = float(np.var(w, ddof=1))
    vl = float(np.var(l, ddof=1))
    denom = math.sqrt(max(1e-12, (vw + vl) / 2.0))
    return float((mw - ml) / denom)


def winner_better_direction(feature: str, winner_median: float, loser_median: float) -> Optional[bool]:
    if (not np.isfinite(winner_median)) or (not np.isfinite(loser_median)):
        return None
    if feature in LOWER_IS_BETTER:
        return bool(winner_median < loser_median)
    if feature in HIGHER_IS_BETTER:
        return bool(winner_median > loser_median)
    return None


def summarize_feature_separation(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    winner_df = df[df["bucket"] == "meaningful_winner"].copy()
    loser_df = df[df["bucket"].isin(["instant_loser", "fast_loser"])].copy()
    rows: List[Dict[str, Any]] = []
    if winner_df.empty or loser_df.empty:
        return pd.DataFrame(columns=[
            "feature",
            "winner_median",
            "loser_median",
            "winner_mean",
            "loser_mean",
            "effect_size",
            "winner_better",
        ]), []

    for feature in ALL_FEATURES:
        w = pd.to_numeric(winner_df.get(feature, np.nan), errors="coerce")
        l = pd.to_numeric(loser_df.get(feature, np.nan), errors="coerce")
        row = {
            "feature": feature,
            "winner_median": float(w.median()) if not w.dropna().empty else float("nan"),
            "loser_median": float(l.median()) if not l.dropna().empty else float("nan"),
            "winner_mean": float(w.mean()) if not w.dropna().empty else float("nan"),
            "loser_mean": float(l.mean()) if not l.dropna().empty else float("nan"),
            "effect_size": compute_effect_size(w, l),
        }
        row["winner_better"] = winner_better_direction(
            feature,
            float(row["winner_median"]),
            float(row["loser_median"]),
        )
        rows.append(row)

    sep_df = pd.DataFrame(rows).sort_values("effect_size", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    top_rows: List[Dict[str, Any]] = []
    for _, row in sep_df.head(5).iterrows():
        feat = str(row["feature"])
        direction = row.get("winner_better")
        if direction is True:
            call = f"winners show {FEATURE_LABELS.get(feat, feat)}"
        elif direction is False:
            call = f"winners show the opposite of {FEATURE_LABELS.get(feat, feat)}"
        else:
            call = FEATURE_LABELS.get(feat, feat)
        top_rows.append(
            {
                "feature": feat,
                "effect_size": float(row["effect_size"]),
                "winner_median": float(row["winner_median"]),
                "loser_median": float(row["loser_median"]),
                "separation_call": call,
            }
        )
    return sep_df, top_rows


def classify_coin(df: pd.DataFrame, sep_df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    trades = int(len(df))
    instant = int((df["bucket"] == "instant_loser").sum())
    fast = int((df["bucket"] == "fast_loser").sum())
    winners = int((df["bucket"] == "meaningful_winner").sum())
    loser_df = df[df["bucket"].isin(["instant_loser", "fast_loser"])].copy()
    winner_df = df[df["bucket"] == "meaningful_winner"].copy()

    top_abs_effect = float(sep_df["effect_size"].abs().max()) if not sep_df.empty else float("nan")
    strong_entry_features = 0
    for _, row in sep_df[sep_df["feature"].isin(ENTRY_FEATURES)].iterrows():
        if bool(row.get("winner_better")) and np.isfinite(row.get("effect_size")) and abs(float(row["effect_size"])) >= 0.35:
            strong_entry_features += 1

    stop_dist = float(pd.to_numeric(df.get("stop_distance_pct_signal", np.nan), errors="coerce").median())
    win_first2_mae = float(pd.to_numeric(winner_df.get("first2_mae_pct", np.nan), errors="coerce").median())
    win_first2_mfe = float(pd.to_numeric(winner_df.get("first2_mfe_pct", np.nan), errors="coerce").median())
    lose_first2_mae = float(pd.to_numeric(loser_df.get("first2_mae_pct", np.nan), errors="coerce").median())

    winners_need_room = bool(
        winners >= MIN_WINNERS_FOR_DECISION
        and np.isfinite(stop_dist)
        and stop_dist > 0.0
        and np.isfinite(win_first2_mae)
        and abs(win_first2_mae) >= max(stop_dist * 0.75, 0.0010)
        and np.isfinite(win_first2_mfe)
        and win_first2_mfe > abs(win_first2_mae) * 1.20
    )

    if winners < MIN_WINNERS_FOR_DECISION or (not np.isfinite(top_abs_effect)) or top_abs_effect < 0.20:
        decision = "NO_CLEAR_SEPARATION_DISABLE_COIN"
    elif strong_entry_features >= 2 and winners_need_room:
        decision = "FILTER_AND_GIVE_ROOM"
    elif strong_entry_features >= 2:
        decision = "FILTER_BAD_ENTRIES"
    elif winners_need_room:
        decision = "GIVE_MORE_INITIAL_ROOM"
    else:
        decision = "NO_CLEAR_SEPARATION_DISABLE_COIN"

    return decision, {
        "total_trades": trades,
        "instant_losers": instant,
        "fast_losers": fast,
        "meaningful_winners": winners,
        "top_abs_effect": top_abs_effect,
        "strong_entry_features": int(strong_entry_features),
        "winners_need_room": int(winners_need_room),
        "median_stop_distance_pct": stop_dist,
        "winner_median_first2_mae_pct": win_first2_mae,
        "winner_median_first2_mfe_pct": win_first2_mfe,
        "loser_median_first2_mae_pct": lose_first2_mae,
    }


def repair_set_for_decision(decision: str, top_rows: List[Dict[str, Any]]) -> str:
    top_feats = [str(r["feature"]) for r in top_rows[:3]]
    if decision == "FILTER_BAD_ENTRIES":
        suggestions: List[str] = []
        if "breakout_dist_pct" in top_feats or "action_gap_pct" in top_feats:
            suggestions.append("filter breakout-chase entries")
        if "signal_body_abs_pct" in top_feats or "signal_range_pct" in top_feats:
            suggestions.append("filter oversized 1h expansion candles")
        if "upper_wick_ratio" in top_feats:
            suggestions.append("filter long upper-wick closes")
        if "dist_to_sma20_pct" in top_feats and len(suggestions) < 3:
            suggestions.append("filter entries too far above 20h mean")
        return "; ".join(suggestions[:3]) or "test 1-2 entry filters on the top separating entry features"
    if decision == "GIVE_MORE_INITIAL_ROOM":
        return "test one conditional wider-initial-risk rule on high-quality entries only"
    if decision == "FILTER_AND_GIVE_ROOM":
        return "test 1-2 entry filters first, then a single conditional wider-initial-risk rule for filtered passes"
    return "disable coin unless a later entry-layer thesis appears"


def reconstruct_best_modela_trades(
    *,
    symbol: str,
    best_candidate_id: str,
    foundation_state: phase_v.FoundationState,
    exec_args: argparse.Namespace,
    run_dir: Path,
    variant_map: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    sig_df = foundation_state.signal_timeline[
        foundation_state.signal_timeline["symbol"].astype(str).str.upper() == symbol
    ].copy()
    win_df = foundation_state.download_manifest[
        foundation_state.download_manifest["symbol"].astype(str).str.upper() == symbol
    ].copy()
    bundle, build_meta = phase_v.build_symbol_bundle(
        symbol=symbol,
        symbol_signals=sig_df,
        symbol_windows=win_df,
        exec_args=exec_args,
        run_dir=run_dir,
    )
    fee = modela.phasec_bt.FeeModel(
        fee_bps_maker=float(exec_args.fee_bps_maker),
        fee_bps_taker=float(exec_args.fee_bps_taker),
        slippage_bps_limit=float(exec_args.slippage_bps_limit),
        slippage_bps_market=float(exec_args.slippage_bps_market),
    )
    base_full = modela.build_1h_reference_rows(
        bundle=bundle,
        fee=fee,
        exec_horizon_hours=float(exec_args.exec_horizon_hours),
    )
    one_h = modela.load_1h_market(symbol)
    ev = modela.evaluate_model_a_variant(
        bundle=bundle,
        baseline_df=base_full,
        cfg=variant_map[best_candidate_id],
        one_h=one_h,
        args=exec_args,
    )
    trade_df = ev["signal_rows_df"].copy().reset_index(drop=True)
    return trade_df, sig_df.reset_index(drop=True), load_full_1h_df(symbol), {
        "build_meta": build_meta,
        "metrics": ev["metrics"],
        "one_h_market": one_h,
    }


def build_trade_feature_matrix(
    *,
    symbol: str,
    trade_df: pd.DataFrame,
    signal_timeline_df: pd.DataFrame,
    one_h_df: pd.DataFrame,
    one_h_market: modela.OneHMarket,
    classification: str,
    best_candidate_id: str,
) -> pd.DataFrame:
    timeline_lookup = {
        str(r["signal_id"]): dict(r)
        for _, r in signal_timeline_df.iterrows()
    }
    rows: List[Dict[str, Any]] = []
    trades = trade_df.copy()
    trades["exec_hold_minutes_calc"] = [
        hold_minutes_from_times(e, x)
        for e, x in zip(trades.get("exec_entry_time", ""), trades.get("exec_exit_time", ""))
    ]

    for row in trades.itertuples(index=False):
        sid = str(getattr(row, "signal_id", ""))
        tl = timeline_lookup.get(sid, {})
        exec_filled = int(getattr(row, "exec_filled", 0))
        exec_valid = int(getattr(row, "exec_valid_for_metrics", 0))
        pnl_net = finite_float(getattr(row, "exec_pnl_net_pct", np.nan))
        hold_min = finite_float(getattr(row, "exec_hold_minutes_calc", np.nan))
        exit_reason = str(getattr(row, "exec_exit_reason", ""))
        if not (exec_filled == 1 and exec_valid == 1 and np.isfinite(pnl_net)):
            continue

        bucket = "neutral_small_win"
        if is_stop_exit(exit_reason) and np.isfinite(hold_min) and hold_min <= 60.0:
            bucket = "instant_loser"
        elif is_stop_exit(exit_reason) and np.isfinite(hold_min) and 60.0 < hold_min <= 240.0:
            bucket = "fast_loser"
        elif pnl_net >= float(MEANINGFUL_WINNER_PNL_THRESHOLD):
            bucket = "meaningful_winner"

        signal_time = pd.to_datetime(tl.get("signal_time_utc", getattr(row, "signal_time", None)), utc=True, errors="coerce")
        entry_ref_time = pd.to_datetime(tl.get("entry_reference_time_utc", pd.NaT), utc=True, errors="coerce")
        signal_bar = locate_bar_row(one_h_df, signal_time)
        action_bar = locate_bar_row(one_h_df, entry_ref_time)
        signal_open = finite_float(getattr(signal_bar, "Open", np.nan) if signal_bar is not None else np.nan)
        signal_high = finite_float(getattr(signal_bar, "High", np.nan) if signal_bar is not None else np.nan)
        signal_low = finite_float(getattr(signal_bar, "Low", np.nan) if signal_bar is not None else np.nan)
        signal_close = finite_float(getattr(signal_bar, "Close", np.nan) if signal_bar is not None else np.nan)
        action_open = finite_float(getattr(action_bar, "Open", np.nan) if action_bar is not None else np.nan)
        action_gap = (
            float(action_open / signal_close - 1.0)
            if np.isfinite(action_open) and np.isfinite(signal_close) and signal_close > 0.0
            else float("nan")
        )
        path_feats = first_eligible_path_features(
            one_h_market,
            getattr(row, "exec_entry_time", None),
            getattr(row, "exec_entry_price", np.nan),
        )
        stop_dist_signal = finite_float(tl.get("stop_distance_pct", np.nan))
        sl_mult = finite_float(tl.get("strategy_sl_mult", np.nan))
        if (not np.isfinite(stop_dist_signal)) and np.isfinite(sl_mult):
            stop_dist_signal = float(1.0 - sl_mult)

        feat_row: Dict[str, Any] = {
            "symbol": symbol,
            "live_status": LIVE_STATUS[symbol],
            "classification": classification,
            "best_candidate_id": best_candidate_id,
            "signal_id": sid,
            "signal_time": str(signal_time) if pd.notna(signal_time) else "",
            "entry_reference_time": str(entry_ref_time) if pd.notna(entry_ref_time) else "",
            "bucket": bucket,
            "exec_exit_reason": exit_reason,
            "exec_hold_minutes": hold_min,
            "exec_pnl_net_pct": pnl_net,
            "exec_entry_price": finite_float(getattr(row, "exec_entry_price", np.nan)),
            "exec_exit_price": finite_float(getattr(row, "exec_exit_price", np.nan)),
            "entry_improvement_bps": finite_float(getattr(row, "entry_improvement_bps", np.nan)),
            "exec_fill_delay_min": finite_float(getattr(row, "exec_fill_delay_min", np.nan)),
            "exec_fill_liquidity_type": str(getattr(row, "exec_fill_liquidity_type", "")),
            "taker_flag": int(str(getattr(row, "exec_fill_liquidity_type", "")).strip().lower() == "taker"),
            "exec_mae_pct": finite_float(getattr(row, "exec_mae_pct", np.nan)),
            "exec_mfe_pct": finite_float(getattr(row, "exec_mfe_pct", np.nan)),
            "signal_open_1h": finite_float(tl.get("signal_open_1h", signal_open)),
            "signal_open_calc": signal_open,
            "signal_high_calc": signal_high,
            "signal_low_calc": signal_low,
            "signal_close_calc": signal_close,
            "action_open_1h": action_open,
            "signal_body_abs_pct": finite_float(getattr(signal_bar, "signal_body_abs_pct", np.nan) if signal_bar is not None else np.nan),
            "signal_range_pct": finite_float(getattr(signal_bar, "signal_range_pct", np.nan) if signal_bar is not None else np.nan),
            "upper_wick_ratio": finite_float(getattr(signal_bar, "upper_wick_ratio", np.nan) if signal_bar is not None else np.nan),
            "lower_wick_ratio": finite_float(getattr(signal_bar, "lower_wick_ratio", np.nan) if signal_bar is not None else np.nan),
            "atr14_pct": finite_float(getattr(signal_bar, "atr14_pct", np.nan) if signal_bar is not None else np.nan),
            "atr_1h": finite_float(tl.get("atr_1h", np.nan)),
            "atr_percentile_1h": finite_float(tl.get("atr_percentile_1h", np.nan)),
            "trend_up_1h": finite_float(tl.get("trend_up_1h", np.nan)),
            "dist_to_sma20_pct": finite_float(getattr(signal_bar, "dist_to_sma20_pct", np.nan) if signal_bar is not None else np.nan),
            "breakout_dist_pct": finite_float(getattr(signal_bar, "breakout_dist_pct", np.nan) if signal_bar is not None else np.nan),
            "dist_from_range_low_pct": finite_float(getattr(signal_bar, "dist_from_range_low_pct", np.nan) if signal_bar is not None else np.nan),
            "ret_3h_pct": finite_float(getattr(signal_bar, "ret_3h_pct", np.nan) if signal_bar is not None else np.nan),
            "ret_6h_pct": finite_float(getattr(signal_bar, "ret_6h_pct", np.nan) if signal_bar is not None else np.nan),
            "volume_z20": finite_float(getattr(signal_bar, "volume_z20", np.nan) if signal_bar is not None else np.nan),
            "action_gap_pct": action_gap,
            "stop_distance_pct_signal": stop_dist_signal,
            "chasing_flag": int(
                np.isfinite(action_gap)
                and (
                    action_gap > 0.0
                    or finite_float(getattr(signal_bar, "breakout_dist_pct", np.nan) if signal_bar is not None else np.nan) > 0.0
                )
            ),
            "meaningful_winner_threshold_pct": float(MEANINGFUL_WINNER_PNL_THRESHOLD),
        }
        feat_row.update(path_feats)
        rows.append(feat_row)

    return pd.DataFrame(rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare instant losers vs meaningful winners on repaired-contract live coins")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--repaired-multicoin-dir", default="", help="Path to REPAIRED_MULTICOIN_MODELA_AUDIT_*; defaults to latest")
    args = ap.parse_args()

    run_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    ensure_dir(run_dir)

    repaired_multicoin_dir = Path(args.repaired_multicoin_dir).resolve() if args.repaired_multicoin_dir else find_latest_repaired_multicoin_dir()
    classification_df = load_best_variant_lookup(repaired_multicoin_dir)
    manifest_path = repaired_multicoin_dir / "repaired_multicoin_modelA_run_manifest.json"
    repaired_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    foundation_dir = Path(repaired_manifest.get("foundation_dir", phase_v.find_latest_foundation_dir())).resolve()
    foundation_state = phase_v.load_foundation_state(foundation_dir)
    exec_args = phase_v.build_exec_args(foundation_state, seed=20260303)
    variant_map = {str(cfg["candidate_id"]): dict(cfg) for cfg in phase_v.sanitize_variants()}

    log("[1/5] Reconstructing repaired best-variant trades for live coins")
    all_feature_rows: List[pd.DataFrame] = []
    bucket_rows: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    recommendation_rows: List[Dict[str, Any]] = []
    per_coin_meta: Dict[str, Any] = {}

    trade_source_dir = ensure_dir(run_dir / "_trade_sources")

    for symbol in LIVE_COINS:
        row = classification_df[classification_df["symbol"] == symbol]
        if row.empty:
            raise KeyError(f"Missing repaired multicoin classification for {symbol}")
        best_row = row.iloc[0]
        best_candidate_id = str(best_row["best_candidate_id"]).strip()
        if best_candidate_id not in variant_map:
            raise KeyError(f"Unknown best candidate for {symbol}: {best_candidate_id}")

        log(f"  - {symbol}: rebuilding {best_candidate_id}")
        trade_df, signal_timeline_df, one_h_df, rebuild_meta = reconstruct_best_modela_trades(
            symbol=symbol,
            best_candidate_id=best_candidate_id,
            foundation_state=foundation_state,
            exec_args=exec_args,
            run_dir=run_dir,
            variant_map=variant_map,
        )
        trade_path = trade_source_dir / f"{symbol}_{best_candidate_id}_trade_level.csv"
        trade_df.to_csv(trade_path, index=False)

        feat_df = build_trade_feature_matrix(
            symbol=symbol,
            trade_df=trade_df,
            signal_timeline_df=signal_timeline_df,
            one_h_df=one_h_df,
            one_h_market=rebuild_meta["one_h_market"],
            classification=str(best_row["classification"]),
            best_candidate_id=best_candidate_id,
        )
        all_feature_rows.append(feat_df)

        sep_df, top_rows = summarize_feature_separation(feat_df)
        sep_path = trade_source_dir / f"{symbol}_{best_candidate_id}_feature_separation.csv"
        sep_df.to_csv(sep_path, index=False)

        decision, diag = classify_coin(feat_df, sep_df)
        repair_set = repair_set_for_decision(decision, top_rows)

        total = int(len(feat_df))
        instant = int((feat_df["bucket"] == "instant_loser").sum())
        fast = int((feat_df["bucket"] == "fast_loser").sum())
        winners = int((feat_df["bucket"] == "meaningful_winner").sum())
        neutral = int((feat_df["bucket"] == "neutral_small_win").sum())
        bucket_rows.append(
            {
                "symbol": symbol,
                "live_status": LIVE_STATUS[symbol],
                "classification": str(best_row["classification"]),
                "best_candidate_id": best_candidate_id,
                "total_trades": total,
                "instant_losers": instant,
                "fast_losers": fast,
                "meaningful_winners": winners,
                "neutral_small_win": neutral,
                "instant_loser_pct": float(100.0 * instant / max(1, total)),
                "fast_loser_pct": float(100.0 * fast / max(1, total)),
                "meaningful_winner_pct": float(100.0 * winners / max(1, total)),
                "meaningful_winner_threshold_pct": float(MEANINGFUL_WINNER_PNL_THRESHOLD),
            }
        )

        top1 = top_rows[0] if len(top_rows) >= 1 else {}
        top2 = top_rows[1] if len(top_rows) >= 2 else {}
        top3 = top_rows[2] if len(top_rows) >= 3 else {}
        comparison_rows.append(
            {
                "symbol": symbol,
                "live_status": LIVE_STATUS[symbol],
                "classification": str(best_row["classification"]),
                "best_candidate_id": best_candidate_id,
                "total_trades": total,
                "instant_losers": instant,
                "fast_losers": fast,
                "meaningful_winners": winners,
                "meaningful_winner_threshold_pct": float(MEANINGFUL_WINNER_PNL_THRESHOLD),
                "top_feature_1": str(top1.get("feature", "")),
                "top_feature_1_call": str(top1.get("separation_call", "")),
                "top_feature_1_effect_size": finite_float(top1.get("effect_size", np.nan)),
                "top_feature_2": str(top2.get("feature", "")),
                "top_feature_2_call": str(top2.get("separation_call", "")),
                "top_feature_2_effect_size": finite_float(top2.get("effect_size", np.nan)),
                "top_feature_3": str(top3.get("feature", "")),
                "top_feature_3_call": str(top3.get("separation_call", "")),
                "top_feature_3_effect_size": finite_float(top3.get("effect_size", np.nan)),
                "winner_median_first2_mae_pct": finite_float(diag.get("winner_median_first2_mae_pct", np.nan)),
                "winner_median_first2_mfe_pct": finite_float(diag.get("winner_median_first2_mfe_pct", np.nan)),
                "loser_median_first2_mae_pct": finite_float(diag.get("loser_median_first2_mae_pct", np.nan)),
                "median_stop_distance_pct": finite_float(diag.get("median_stop_distance_pct", np.nan)),
                "strong_entry_features": int(diag.get("strong_entry_features", 0)),
                "winners_need_room": int(diag.get("winners_need_room", 0)),
                "decision": decision,
            }
        )
        recommendation_rows.append(
            {
                "symbol": symbol,
                "live_status": LIVE_STATUS[symbol],
                "classification": str(best_row["classification"]),
                "best_candidate_id": best_candidate_id,
                "decision": decision,
                "repair_scope": (
                    "disable"
                    if decision == "NO_CLEAR_SEPARATION_DISABLE_COIN"
                    else ("filter+room" if decision == "FILTER_AND_GIVE_ROOM" else ("filter" if decision == "FILTER_BAD_ENTRIES" else "room"))
                ),
                "bounded_follow_up_repair_set": repair_set,
                "top_feature_1": str(top1.get("feature", "")),
                "top_feature_2": str(top2.get("feature", "")),
                "top_feature_3": str(top3.get("feature", "")),
            }
        )
        per_coin_meta[symbol] = {
            "classification_source_row": dict(best_row.to_dict()),
            "trade_source_csv": str(trade_path),
            "feature_separation_csv": str(sep_path),
            "rebuild_meta": {
                "build_meta": rebuild_meta["build_meta"],
                "metrics": rebuild_meta["metrics"],
            },
            "top_separators": top_rows,
            "decision_diagnostics": diag,
        }

    log("[2/5] Writing required CSV artifacts")
    feature_matrix_df = pd.concat(all_feature_rows, ignore_index=True) if all_feature_rows else pd.DataFrame()
    bucket_df = pd.DataFrame(bucket_rows).sort_values("symbol").reset_index(drop=True)
    compare_df = pd.DataFrame(comparison_rows).sort_values("symbol").reset_index(drop=True)
    rec_df = pd.DataFrame(recommendation_rows).sort_values("symbol").reset_index(drop=True)

    bucket_df.to_csv(run_dir / "instant_loser_vs_winner_trade_buckets.csv", index=False)
    feature_matrix_df.to_csv(run_dir / "instant_loser_vs_winner_feature_matrix.csv", index=False)
    compare_df.to_csv(run_dir / "instant_loser_vs_winner_comparison_by_coin.csv", index=False)
    rec_df.to_csv(run_dir / "entry_repair_recommendation_by_coin.csv", index=False)

    log("[3/5] Building markdown report")
    report_lines = [
        "# Instant Loser Vs Winner Entry Forensics",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Artifact dir: `{run_dir}`",
        f"- Repaired multicoin source: `{repaired_multicoin_dir}`",
        f"- Foundation source: `{foundation_dir}`",
        "",
        "## Bucket Definitions",
        "",
        "- `instant_loser`: valid trade, stop-loss exit, hold <= 60 minutes.",
        "- `fast_loser`: valid trade, stop-loss exit, 60 < hold <= 240 minutes.",
        f"- `meaningful_winner`: valid trade, net pnl >= {MEANINGFUL_WINNER_PNL_THRESHOLD:.4f} (20 bps).",
        "- `neutral_small_win`: all other valid trades.",
        "",
        "## Live-Coin Bucket Counts",
        "",
        markdown_table(
            bucket_df,
            [
                "symbol",
                "live_status",
                "best_candidate_id",
                "total_trades",
                "instant_losers",
                "fast_losers",
                "meaningful_winners",
                "neutral_small_win",
            ],
        ),
        "",
        "## Strongest Separating Features",
        "",
        markdown_table(
            compare_df,
            [
                "symbol",
                "top_feature_1_call",
                "top_feature_2_call",
                "top_feature_3_call",
                "decision",
            ],
        ),
        "",
        "## Repair Decisions",
        "",
        markdown_table(
            rec_df,
            [
                "symbol",
                "decision",
                "bounded_follow_up_repair_set",
            ],
        ),
    ]
    (run_dir / "instant_loser_vs_winner_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    log("[4/5] Writing manifest")
    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "scope": {
            "live_coins": list(LIVE_COINS),
            "approved": ["LINKUSDT"],
            "shadow": ["DOGEUSDT", "NEARUSDT", "LTCUSDT"],
            "disabled_appendix_included": 0,
        },
        "bucket_definitions": {
            "instant_loser": "valid trade with stop-loss exit and hold <= 60 minutes",
            "fast_loser": "valid trade with stop-loss exit and 60 < hold <= 240 minutes",
            "meaningful_winner": f"valid trade with net pnl >= {MEANINGFUL_WINNER_PNL_THRESHOLD:.4f}",
            "neutral_small_win": "all other valid trades",
        },
        "contract_assumptions": {
            "repaired_contract_only": 1,
            "same_parent_bar_exit_disabled": int(
                bool(repaired_manifest.get("contract_assumptions", {}).get("same_parent_bar_exit_disabled", 1))
            ),
            "one_h_signal_owner": 1,
            "one_h_exit_owner": 1,
            "three_m_entry_only": 1,
        },
        "sources": {
            "repaired_multicoin_dir": str(repaired_multicoin_dir),
            "repaired_multicoin_classification_csv": str(repaired_multicoin_dir / "repaired_multicoin_modelA_coin_classification.csv"),
            "repaired_multicoin_manifest_json": str(manifest_path),
            "foundation_dir": str(foundation_dir),
            "foundation_signal_timeline_csv": str(foundation_dir / "universe_signal_timeline.csv"),
            "foundation_download_manifest_csv": str(foundation_dir / "universe_3m_download_manifest.csv"),
        },
        "thresholds": {
            "meaningful_winner_pnl_threshold": float(MEANINGFUL_WINNER_PNL_THRESHOLD),
            "min_winners_for_decision": int(MIN_WINNERS_FOR_DECISION),
            "strong_entry_feature_abs_effect_size": 0.35,
            "minimum_top_effect_for_any_decision": 0.20,
        },
        "per_coin_meta": per_coin_meta,
        "outputs": {
            "trade_buckets_csv": str(run_dir / "instant_loser_vs_winner_trade_buckets.csv"),
            "feature_matrix_csv": str(run_dir / "instant_loser_vs_winner_feature_matrix.csv"),
            "comparison_csv": str(run_dir / "instant_loser_vs_winner_comparison_by_coin.csv"),
            "recommendation_csv": str(run_dir / "entry_repair_recommendation_by_coin.csv"),
            "report_md": str(run_dir / "instant_loser_vs_winner_report.md"),
            "manifest_json": str(run_dir / "instant_loser_vs_winner_manifest.json"),
        },
    }
    json_dump(run_dir / "instant_loser_vs_winner_manifest.json", manifest)
    log("[5/5] Complete")
    log(str(run_dir))


if __name__ == "__main__":
    main()
