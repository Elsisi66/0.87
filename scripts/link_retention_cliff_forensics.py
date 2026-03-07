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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import instant_loser_vs_winner_entry_forensics as forensics  # noqa: E402
from scripts import live_coin_bounded_entry_repair_pilot as pilot  # noqa: E402
from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402


RUN_PREFIX = "LINK_RETENTION_CLIFF_FORENSICS"
SYMBOL = "LINKUSDT"
DEFAULT_1H_BASELINE_PATTERN = "1H_CONTRACT_REPAIR_REBASELINE_*"
DEFAULT_MULTICOIN_PATTERN = "REPAIRED_MULTICOIN_MODELA_AUDIT_*"
DEFAULT_PILOT_PATTERN = "LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_*"
DEFAULT_DIAG_PATTERN = "INSTANT_LOSER_VS_WINNER_ENTRY_FORENSICS_*"


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


def find_latest_dir(pattern: str, required_files: Sequence[str]) -> Path:
    cands = sorted(
        [p for p in (PROJECT_ROOT / "reports" / "execution_layer").glob(pattern) if p.is_dir()],
        key=lambda p: p.name,
    )
    for p in reversed(cands):
        names = {f.name for f in p.iterdir() if f.is_file()}
        if set(required_files).issubset(names):
            return p.resolve()
    raise FileNotFoundError(f"No completed run directory found for pattern {pattern}")


def discover_inputs() -> Dict[str, Any]:
    baseline_dir = find_latest_dir(DEFAULT_1H_BASELINE_PATTERN, ["repaired_1h_reference_summary.csv"])
    multicoin_dir = find_latest_dir(
        DEFAULT_MULTICOIN_PATTERN,
        [
            "repaired_multicoin_modelA_coin_classification.csv",
            "repaired_multicoin_modelA_reference_vs_best.csv",
            "repaired_multicoin_modelA_run_manifest.json",
        ],
    )
    diag_dir = find_latest_dir(
        DEFAULT_DIAG_PATTERN,
        [
            "instant_loser_vs_winner_trade_buckets.csv",
            "instant_loser_vs_winner_feature_matrix.csv",
            "entry_repair_recommendation_by_coin.csv",
        ],
    )
    pilot_dir = find_latest_dir(
        DEFAULT_PILOT_PATTERN,
        [
            "live_coin_bounded_entry_repair_results.csv",
            "live_coin_bounded_entry_repair_vs_control.csv",
            "live_coin_bounded_entry_repair_decision.csv",
        ],
    )

    baseline_summary = pd.read_csv(baseline_dir / "repaired_1h_reference_summary.csv")
    baseline_row = baseline_summary[baseline_summary["symbol"].astype(str).str.upper() == SYMBOL]
    if baseline_row.empty:
        raise KeyError(f"{SYMBOL} missing from repaired 1h baseline summary")

    multicoin_class = pd.read_csv(multicoin_dir / "repaired_multicoin_modelA_coin_classification.csv")
    multicoin_class["symbol"] = multicoin_class["symbol"].astype(str).str.upper()
    multicoin_row = multicoin_class[multicoin_class["symbol"] == SYMBOL]
    if multicoin_row.empty:
        raise KeyError(f"{SYMBOL} missing from repaired multicoin classification")

    pilot_results = pd.read_csv(pilot_dir / "live_coin_bounded_entry_repair_results.csv")
    pilot_results["symbol"] = pilot_results["symbol"].astype(str).str.upper()
    link_pilot = pilot_results[pilot_results["symbol"] == SYMBOL].copy()
    if link_pilot.empty:
        raise KeyError(f"{SYMBOL} missing from latest live coin entry repair pilot")

    hard_gap_row = link_pilot[link_pilot["variant_id"].astype(str) == "GAP_CAP_FILTER"]
    if hard_gap_row.empty:
        hard_gap_row = link_pilot[link_pilot["variant_id"].astype(str).str.contains("GAP", regex=False)]
    if hard_gap_row.empty:
        raise KeyError(f"Could not find failed hard-gap variant for {SYMBOL} in latest pilot")

    control_row = link_pilot[link_pilot["variant_id"].astype(str) == "CONTROL"]
    if control_row.empty:
        raise KeyError(f"Could not find control variant for {SYMBOL} in latest pilot")

    hard_gap_trade_csv = Path(str(hard_gap_row.iloc[0]["source_trade_csv"])).resolve()
    control_trade_csv = Path(str(control_row.iloc[0]["source_trade_csv"])).resolve()
    if not hard_gap_trade_csv.exists():
        raise FileNotFoundError(f"Missing hard-gap trade CSV: {hard_gap_trade_csv}")
    if not control_trade_csv.exists():
        raise FileNotFoundError(f"Missing control trade CSV: {control_trade_csv}")

    return {
        "baseline_dir": baseline_dir,
        "baseline_row": dict(baseline_row.iloc[0].to_dict()),
        "multicoin_dir": multicoin_dir,
        "multicoin_row": dict(multicoin_row.iloc[0].to_dict()),
        "diag_dir": diag_dir,
        "pilot_dir": pilot_dir,
        "pilot_results": pilot_results,
        "control_row": dict(control_row.iloc[0].to_dict()),
        "hard_gap_row": dict(hard_gap_row.iloc[0].to_dict()),
        "control_trade_csv": control_trade_csv,
        "hard_gap_trade_csv": hard_gap_trade_csv,
    }


def load_foundation_from_multicoin(multicoin_dir: Path) -> Tuple[phase_v.FoundationState, argparse.Namespace]:
    manifest = json.loads((multicoin_dir / "repaired_multicoin_modelA_run_manifest.json").read_text(encoding="utf-8"))
    foundation_dir = Path(str(manifest["foundation_dir"])).resolve()
    state = phase_v.load_foundation_state(foundation_dir)
    exec_args = phase_v.build_exec_args(state, seed=20260303)
    return state, exec_args


def load_trade_sources(control_trade_csv: Path, hard_gap_trade_csv: Path, diag_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    control_df = pd.read_csv(control_trade_csv)
    hard_gap_df = pd.read_csv(hard_gap_trade_csv)
    feat_df = pd.read_csv(diag_dir / "instant_loser_vs_winner_feature_matrix.csv")
    for df in (control_df, hard_gap_df):
        df["signal_id"] = df["signal_id"].astype(str)
    feat_df["symbol"] = feat_df["symbol"].astype(str).str.upper()
    feat_df = feat_df[feat_df["symbol"] == SYMBOL].copy().reset_index(drop=True)
    feat_df["signal_id"] = feat_df["signal_id"].astype(str)
    return control_df, hard_gap_df, feat_df


def build_trade_table(
    *,
    control_df: pd.DataFrame,
    hard_gap_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    control_lookup = {
        str(r["signal_id"]): dict(r)
        for _, r in control_df.iterrows()
    }
    hard_lookup = {
        str(r["signal_id"]): dict(r)
        for _, r in hard_gap_df.iterrows()
    }
    gap_vals = pd.to_numeric(feature_df["action_gap_pct"], errors="coerce")
    gap_pct_rank = gap_vals.rank(pct=True, method="average")
    deciles = pd.qcut(gap_vals.rank(method="first"), 10, labels=False, duplicates="drop") if not gap_vals.dropna().empty else pd.Series(dtype=float)

    rows: List[Dict[str, Any]] = []
    for i, feat in feature_df.reset_index(drop=True).iterrows():
        sid = str(feat["signal_id"])
        ctrl = control_lookup.get(sid, {})
        hard = hard_lookup.get(sid, {})
        control_pnl = finite_float(ctrl.get("exec_pnl_net_pct", np.nan))
        stop_dist = finite_float(feat.get("stop_distance_pct_signal", np.nan))
        rows.append(
            {
                "symbol": SYMBOL,
                "signal_id": sid,
                "signal_time": str(feat.get("signal_time", "")),
                "side": "long",
                "repaired_1h_baseline_filled": int(pd.to_numeric(pd.Series([ctrl.get("baseline_filled", 0)]), errors="coerce").fillna(0).iloc[0]),
                "repaired_1h_baseline_valid": int(pd.to_numeric(pd.Series([ctrl.get("baseline_valid_for_metrics", 0)]), errors="coerce").fillna(0).iloc[0]),
                "repaired_1h_baseline_pnl_net_pct": finite_float(ctrl.get("baseline_pnl_net_pct", np.nan)),
                "repaired_1h_baseline_exit_reason": str(ctrl.get("baseline_exit_reason", "")),
                "control_modelA_filled": int(pd.to_numeric(pd.Series([ctrl.get("exec_filled", 0)]), errors="coerce").fillna(0).iloc[0]),
                "control_modelA_valid": int(pd.to_numeric(pd.Series([ctrl.get("exec_valid_for_metrics", 0)]), errors="coerce").fillna(0).iloc[0]),
                "control_modelA_pnl_net_pct": control_pnl,
                "control_modelA_hold_minutes": pilot.hold_minutes_from_row(pd.Series(ctrl)) if ctrl else float("nan"),
                "control_modelA_exit_reason": str(ctrl.get("exec_exit_reason", "")),
                "hard_gap_keep_status": "kept" if int(pd.to_numeric(pd.Series([hard.get("repair_filter_pass", 0)]), errors="coerce").fillna(0).iloc[0]) == 1 else "dropped",
                "hard_gap_removed_flag": int(pd.to_numeric(pd.Series([hard.get("repair_filter_pass", 0)]), errors="coerce").fillna(0).iloc[0]) == 0,
                "hard_gap_exec_filled": int(pd.to_numeric(pd.Series([hard.get("exec_filled", 0)]), errors="coerce").fillna(0).iloc[0]),
                "hard_gap_exec_pnl_net_pct": finite_float(hard.get("exec_pnl_net_pct", np.nan)),
                "hard_gap_exec_skip_reason": str(hard.get("exec_skip_reason", "")),
                "action_gap_pct": finite_float(feat.get("action_gap_pct", np.nan)),
                "action_gap_pct_percentile": finite_float(gap_pct_rank.iloc[i] if i < len(gap_pct_rank) else np.nan),
                "action_gap_decile": int(deciles.iloc[i] + 1) if len(deciles) > i and pd.notna(deciles.iloc[i]) else -1,
                "first1_mae_pct": finite_float(feat.get("first1_mae_pct", np.nan)),
                "first1_mfe_pct": finite_float(feat.get("first1_mfe_pct", np.nan)),
                "first2_mae_pct": finite_float(feat.get("first2_mae_pct", np.nan)),
                "first2_mfe_pct": finite_float(feat.get("first2_mfe_pct", np.nan)),
                "first1_close_ret_pct": finite_float(feat.get("first1_close_ret_pct", np.nan)),
                "first2_close_ret_pct": finite_float(feat.get("first2_close_ret_pct", np.nan)),
                "instant_loser_label": int(str(feat.get("bucket", "")) == "instant_loser"),
                "fast_loser_label": int(str(feat.get("bucket", "")) == "fast_loser"),
                "meaningful_winner_label": int(str(feat.get("bucket", "")) == "meaningful_winner"),
                "neutral_small_win_label": int(str(feat.get("bucket", "")) == "neutral_small_win"),
                "control_r_multiple_est": float(control_pnl / stop_dist) if np.isfinite(control_pnl) and np.isfinite(stop_dist) and stop_dist > 0.0 else float("nan"),
                "repaired_1h_baseline_r_multiple_est": (
                    float(finite_float(ctrl.get("baseline_pnl_net_pct", np.nan)) / stop_dist)
                    if np.isfinite(finite_float(ctrl.get("baseline_pnl_net_pct", np.nan))) and np.isfinite(stop_dist) and stop_dist > 0.0
                    else float("nan")
                ),
                "stop_distance_pct_signal": stop_dist,
                "entry_improvement_bps": finite_float(feat.get("entry_improvement_bps", np.nan)),
                "exec_fill_delay_min": finite_float(feat.get("exec_fill_delay_min", np.nan)),
                "taker_flag": int(pd.to_numeric(pd.Series([feat.get("taker_flag", 0)]), errors="coerce").fillna(0).iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def summarize_removed_population(trade_table: pd.DataFrame) -> Dict[str, Any]:
    removed = trade_table[trade_table["hard_gap_removed_flag"] == 1].copy()
    kept = trade_table[trade_table["hard_gap_removed_flag"] == 0].copy()
    return {
        "removed_total": int(len(removed)),
        "removed_instant_losers": int(removed["instant_loser_label"].sum()),
        "removed_fast_losers": int(removed["fast_loser_label"].sum()),
        "removed_meaningful_winners": int(removed["meaningful_winner_label"].sum()),
        "removed_positive_pnl": int((pd.to_numeric(removed["control_modelA_pnl_net_pct"], errors="coerce") > 0).sum()),
        "removed_nonpositive_pnl": int((pd.to_numeric(removed["control_modelA_pnl_net_pct"], errors="coerce") <= 0).sum()),
        "kept_total": int(len(kept)),
        "kept_instant_losers": int(kept["instant_loser_label"].sum()),
        "kept_fast_losers": int(kept["fast_loser_label"].sum()),
        "kept_meaningful_winners": int(kept["meaningful_winner_label"].sum()),
    }


def decile_decomposition(trade_table: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for decile, g in trade_table[trade_table["action_gap_decile"] > 0].groupby("action_gap_decile"):
        rows.append(
            {
                "action_gap_decile": int(decile),
                "trades": int(len(g)),
                "removed_by_hard_gap": int(g["hard_gap_removed_flag"].sum()),
                "removed_share": float(g["hard_gap_removed_flag"].mean()),
                "instant_losers": int(g["instant_loser_label"].sum()),
                "fast_losers": int(g["fast_loser_label"].sum()),
                "meaningful_winners": int(g["meaningful_winner_label"].sum()),
                "median_gap_pct": float(pd.to_numeric(g["action_gap_pct"], errors="coerce").median()),
                "mean_control_pnl": float(pd.to_numeric(g["control_modelA_pnl_net_pct"], errors="coerce").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("action_gap_decile").reset_index(drop=True)


def reconstruct_control_state(multicoin_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, modela.OneHMarket, Dict[str, Any], modela.ga_exec.SymbolBundle, argparse.Namespace]:
    foundation_state, exec_args = load_foundation_from_multicoin(multicoin_dir)
    class_df = forensics.load_best_variant_lookup(multicoin_dir)
    row = class_df[class_df["symbol"] == SYMBOL]
    if row.empty:
        raise KeyError(f"{SYMBOL} missing in repaired multicoin classification")
    best_candidate_id = str(row.iloc[0]["best_candidate_id"]).strip()
    variant_map = {str(cfg["candidate_id"]): dict(cfg) for cfg in phase_v.sanitize_variants()}
    if best_candidate_id not in variant_map:
        raise KeyError(f"Unknown best candidate for {SYMBOL}: {best_candidate_id}")

    trade_df, sig_timeline_df, _one_h_df, rebuild_meta = forensics.reconstruct_best_modela_trades(
        symbol=SYMBOL,
        best_candidate_id=best_candidate_id,
        foundation_state=foundation_state,
        exec_args=exec_args,
        run_dir=multicoin_dir,
        variant_map=variant_map,
    )
    symbol_signals = sig_timeline_df[sig_timeline_df["symbol"].astype(str).str.upper() == SYMBOL].copy()
    symbol_windows = foundation_state.download_manifest[
        foundation_state.download_manifest["symbol"].astype(str).str.upper() == SYMBOL
    ].copy()
    bundle, _build_meta = phase_v.build_symbol_bundle(
        symbol=SYMBOL,
        symbol_signals=symbol_signals,
        symbol_windows=symbol_windows,
        exec_args=exec_args,
        run_dir=multicoin_dir,
    )
    return trade_df, symbol_signals, rebuild_meta["one_h_market"], row.iloc[0].to_dict(), bundle, exec_args


def build_threshold_variants(
    trade_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    bundle: modela.ga_exec.SymbolBundle,
    one_h: modela.OneHMarket,
    exec_args: argparse.Namespace,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], Dict[str, float]]:
    thresholds = pilot.compute_filter_thresholds(SYMBOL, feature_df)
    ctx_map = {str(ctx.signal_id): ctx for ctx in bundle.contexts}
    control_df = trade_df.copy().reset_index(drop=True)
    control_df["repair_filter_pass"] = 1
    control_df["repair_variant_id"] = "CONTROL"
    control_df["repair_variant_label"] = "Current repaired control"
    control_df["repair_room_floor_bps_applied"] = 0.0
    control_df["repair_effective_sl_mult"] = control_df["signal_id"].map(lambda sid: float(ctx_map[str(sid)].sl_mult_sig))
    control_df["repair_room_condition_hit"] = 0
    control_metrics = pilot.compute_variant_metrics(df=control_df, bundle=bundle, exec_args=exec_args)

    feature_lookup = {
        str(r["signal_id"]): r
        for _, r in feature_df.iterrows()
    }
    gap_series = pd.to_numeric(feature_df["action_gap_pct"], errors="coerce")
    pos_gap_mask = gap_series > 0.0
    pos_gap_q90 = float(gap_series[pos_gap_mask].quantile(0.90)) if pos_gap_mask.any() else 0.0
    pos_gap_q75 = float(gap_series[pos_gap_mask].quantile(0.75)) if pos_gap_mask.any() else 0.0
    threshold_2bps = 0.0002
    threshold_4bps = 0.0004

    variant_defs: List[Dict[str, Any]] = [
        {"variant_id": "CONTROL", "kind": "control", "label": "Current repaired control"},
        {"variant_id": "GAP_CAP_0BPS", "kind": "gap_cap", "label": "Hard gap cap <= 0 bps", "gap_cap": 0.0},
        {"variant_id": "GAP_CAP_2BPS", "kind": "gap_cap", "label": "Hard gap cap <= 2 bps", "gap_cap": threshold_2bps},
        {"variant_id": "GAP_CAP_4BPS", "kind": "gap_cap", "label": "Hard gap cap <= 4 bps", "gap_cap": threshold_4bps},
        {
            "variant_id": "PCTL_DROP_TOP10_POSGAP",
            "kind": "percentile_drop",
            "label": "Drop top 10% of positive-gap trades",
            "percentile_cut": pos_gap_q90,
        },
        {
            "variant_id": "PCTL_DROP_TOP25_POSGAP",
            "kind": "percentile_drop",
            "label": "Drop top 25% of positive-gap trades",
            "percentile_cut": pos_gap_q75,
        },
        {
            "variant_id": "SIZE_50_TOP10_POSGAP",
            "kind": "size_haircut",
            "label": "50% size haircut on top 10% positive-gap trades",
            "percentile_cut": pos_gap_q90,
            "size_mult": 0.50,
        },
        {
            "variant_id": "SIZE_75_ALL_POSGAP",
            "kind": "size_haircut",
            "label": "75% size haircut on all positive-gap trades",
            "percentile_cut": 0.0,
            "size_mult": 0.75,
        },
    ]

    rows: List[Dict[str, Any]] = []
    out_tables: List[pd.DataFrame] = []
    for spec in variant_defs:
        kind = str(spec["kind"])
        if kind == "control":
            var_df = control_df.copy().reset_index(drop=True)
        else:
            var_rows: List[Dict[str, Any]] = []
            for _, row_s in control_df.iterrows():
                sid = str(row_s["signal_id"])
                feat = feature_lookup.get(sid)
                gap = finite_float(feat.get("action_gap_pct", np.nan)) if feat is not None else float("nan")
                out = dict(row_s.to_dict())
                out["repair_variant_id"] = str(spec["variant_id"])
                out["repair_variant_label"] = str(spec["label"])
                out["repair_room_floor_bps_applied"] = 0.0
                out["repair_effective_sl_mult"] = float(ctx_map[sid].sl_mult_sig) if sid in ctx_map else float("nan")
                out["repair_room_condition_hit"] = 0

                if kind == "gap_cap":
                    keep = bool(np.isfinite(gap) and gap <= float(spec["gap_cap"]))
                    out["repair_filter_pass"] = int(keep)
                    if not keep:
                        var_rows.append(pilot.zero_exec_fields(out, skip_reason=f"retention_gap_cap_reject:{spec['variant_id']}"))
                    else:
                        var_rows.append(out)
                elif kind == "percentile_drop":
                    cut = float(spec["percentile_cut"])
                    keep = not (np.isfinite(gap) and gap > 0.0 and gap >= cut)
                    out["repair_filter_pass"] = int(keep)
                    if not keep:
                        var_rows.append(pilot.zero_exec_fields(out, skip_reason=f"retention_percentile_reject:{spec['variant_id']}"))
                    else:
                        var_rows.append(out)
                elif kind == "size_haircut":
                    cut = float(spec["percentile_cut"])
                    size_mult = float(spec["size_mult"])
                    hit = bool(np.isfinite(gap) and ((gap > 0.0 and gap >= cut) if cut > 0.0 else (gap > 0.0)))
                    out["repair_filter_pass"] = 1
                    out["repair_size_mult"] = float(size_mult) if hit else 1.0
                    if hit:
                        out["exec_pnl_net_pct"] = (
                            float(out["exec_pnl_net_pct"]) * size_mult
                            if np.isfinite(finite_float(out.get("exec_pnl_net_pct", np.nan)))
                            else out.get("exec_pnl_net_pct", np.nan)
                        )
                        out["exec_pnl_gross_pct"] = (
                            float(out["exec_pnl_gross_pct"]) * size_mult
                            if np.isfinite(finite_float(out.get("exec_pnl_gross_pct", np.nan)))
                            else out.get("exec_pnl_gross_pct", np.nan)
                        )
                    else:
                        out["repair_size_mult"] = 1.0
                    var_rows.append(out)
                else:
                    raise RuntimeError(f"Unknown variant kind: {kind}")

            var_df = pd.DataFrame(var_rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)

        metrics = pilot.compute_variant_metrics(df=var_df, bundle=bundle, exec_args=exec_args)
        accepted, checks = (
            (False, {
                "instant_losers_materially_down": 1,
                "expectancy_ok": 1,
                "cvar_ok": 1,
                "maxdd_ok": 1,
                "retention_ok": 1,
                "parity_clean": int(metrics["parity_clean"]),
                "instant_required_abs": 0,
                "instant_delta": 0,
                "expectancy_delta": 0.0,
                "cvar_delta": 0.0,
                "maxdd_delta": 0.0,
                "trade_count_retention_vs_control": 1.0,
            }) if spec["variant_id"] == "CONTROL" else pilot.accept_variant(control_metrics, metrics)
        )

        removed_mask = pd.to_numeric(var_df.get("repair_filter_pass", 1), errors="coerce").fillna(1).astype(int) == 0
        source_tbl = feature_df.set_index("signal_id")
        removed_sids = var_df.loc[removed_mask, "signal_id"].astype(str).tolist()
        removed_feat = source_tbl.loc[source_tbl.index.intersection(removed_sids)].copy() if not source_tbl.empty else pd.DataFrame()

        rows.append(
            {
                "symbol": SYMBOL,
                "variant_id": str(spec["variant_id"]),
                "variant_label": str(spec["label"]),
                "variant_kind": kind,
                "gap_cap": finite_float(spec.get("gap_cap", np.nan)),
                "percentile_cut": finite_float(spec.get("percentile_cut", np.nan)),
                "size_mult": finite_float(spec.get("size_mult", np.nan)),
                "instant_loser_count": int(metrics["instant_loser_count"]),
                "fast_loser_count": int(metrics["fast_loser_count"]),
                "meaningful_winner_count": int(metrics["meaningful_winner_count"]),
                "expectancy_net": float(metrics["expectancy_net"]),
                "cvar_5": float(metrics["cvar_5"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "win_rate": float(metrics["win_rate"]),
                "total_trades": int(metrics["total_trades"]),
                "median_hold_minutes": float(metrics["median_hold_minutes"]),
                "pct_exit_within_3h": float(metrics["pct_exit_within_3h"]),
                "pct_exit_within_4h": float(metrics["pct_exit_within_4h"]),
                "valid_for_ranking": int(metrics["valid_for_ranking"]),
                "invalid_reason": str(metrics["invalid_reason"]),
                "parity_clean": int(metrics["parity_clean"]),
                "accepted": int(accepted),
                "trade_count_retention_vs_control": float(checks["trade_count_retention_vs_control"]),
                "instant_delta_vs_control": int(checks["instant_delta"]),
                "expectancy_delta_vs_control": float(checks["expectancy_delta"]),
                "cvar_delta_vs_control": float(checks["cvar_delta"]),
                "maxdd_delta_vs_control": float(checks["maxdd_delta"]),
                "accept_instant_losers_materially_down": int(checks["instant_losers_materially_down"]),
                "accept_expectancy_ok": int(checks["expectancy_ok"]),
                "accept_cvar_ok": int(checks["cvar_ok"]),
                "accept_maxdd_ok": int(checks["maxdd_ok"]),
                "accept_retention_ok": int(checks["retention_ok"]),
                "accept_parity_clean": int(checks["parity_clean"]),
                "removed_trades": int(removed_mask.sum()),
                "removed_instant_losers": int(pd.to_numeric(removed_feat.get("bucket", pd.Series(dtype=object)).eq("instant_loser"), errors="coerce").fillna(0).sum())
                if not removed_feat.empty else 0,
                "removed_fast_losers": int(pd.to_numeric(removed_feat.get("bucket", pd.Series(dtype=object)).eq("fast_loser"), errors="coerce").fillna(0).sum())
                if not removed_feat.empty else 0,
                "removed_meaningful_winners": int(pd.to_numeric(removed_feat.get("bucket", pd.Series(dtype=object)).eq("meaningful_winner"), errors="coerce").fillna(0).sum())
                if not removed_feat.empty else 0,
                "removed_positive_gap_trades": int(((pd.to_numeric(removed_feat.get("action_gap_pct", np.nan), errors="coerce") > 0.0)).sum())
                if not removed_feat.empty else 0,
                "median_removed_gap_pct": float(pd.to_numeric(removed_feat.get("action_gap_pct", np.nan), errors="coerce").median())
                if not removed_feat.empty else float("nan"),
            }
        )
        out_tables.append(var_df.assign(_variant_id=str(spec["variant_id"])))

    combined_df = pd.concat(out_tables, ignore_index=True) if out_tables else pd.DataFrame()
    return combined_df, rows, {
        "action_gap_cap_hard_failed": 0.0,
        "action_gap_cap_soft_2bps": threshold_2bps,
        "action_gap_cap_soft_4bps": threshold_4bps,
        "positive_gap_q90": pos_gap_q90,
        "positive_gap_q75": pos_gap_q75,
    }


def choose_best_variant(summary_df: pd.DataFrame) -> Tuple[str, Optional[pd.Series], str]:
    non_control = summary_df[summary_df["variant_id"] != "CONTROL"].copy()
    accepted = non_control[non_control["accepted"] == 1].copy()
    if accepted.empty:
        # Conservative fallback: identify if a near-miss pocket exists without claiming approval.
        near = non_control[
            (non_control["parity_clean"] == 1)
            & (non_control["valid_for_ranking"] == 1)
            & (non_control["trade_count_retention_vs_control"] >= 0.90)
            & (non_control["expectancy_delta_vs_control"] >= -0.00005)
        ].copy()
        if near.empty:
            return "NO_REPAIR_APPROVED", None, "No variant met the existing acceptance logic; either retention failed or risk worsened."
        near = near.sort_values(
            ["instant_loser_count", "expectancy_delta_vs_control", "cvar_delta_vs_control", "maxdd_delta_vs_control"],
            ascending=[True, False, False, False],
        ).reset_index(drop=True)
        best = near.iloc[0]
        return "NO_REPAIR_APPROVED", best, "A narrow near-miss pocket exists, but it still fails the existing acceptance logic."

    accepted = accepted.sort_values(
        ["trade_count_retention_vs_control", "expectancy_delta_vs_control", "cvar_delta_vs_control", "maxdd_delta_vs_control"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return str(accepted.iloc[0]["variant_id"]), accepted.iloc[0], "Accepted under the existing pilot acceptance logic."


def main() -> None:
    ap = argparse.ArgumentParser(description="LINK-only retention cliff forensics around the failed hard-gap repair")
    ap.add_argument("--outdir", default="reports/execution_layer")
    args = ap.parse_args()

    run_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    ensure_dir(run_dir)

    try:
        discovered = discover_inputs()
    except Exception as exc:  # pragma: no cover
        print("BLOCKED", flush=True)
        print(str(exc), flush=True)
        raise

    log("[1/5] Loading discovered artifacts and trade sources")
    control_df, hard_gap_df, feat_df = load_trade_sources(
        control_trade_csv=discovered["control_trade_csv"],
        hard_gap_trade_csv=discovered["hard_gap_trade_csv"],
        diag_dir=discovered["diag_dir"],
    )

    log("[2/5] Building LINK trade-level forensic table")
    trade_table = build_trade_table(control_df=control_df, hard_gap_df=hard_gap_df, feature_df=feat_df)
    trade_table.to_csv(run_dir / "link_retention_cliff_trade_table.csv", index=False)

    removed_summary = summarize_removed_population(trade_table)
    decile_df = decile_decomposition(trade_table)

    log("[3/5] Reconstructing control and scanning bounded soft variants")
    _reconstructed_control, _signals, one_h, _multicoin_row, bundle, exec_args = reconstruct_control_state(discovered["multicoin_dir"])
    _combined_variant_df, summary_rows, scan_meta = build_threshold_variants(
        trade_df=control_df,
        feature_df=feat_df,
        bundle=bundle,
        one_h=one_h,
        exec_args=exec_args,
    )
    summary_df = pd.DataFrame(summary_rows).sort_values(["variant_kind", "variant_id"]).reset_index(drop=True)
    summary_df.to_csv(run_dir / "link_retention_cliff_variant_summary.csv", index=False)

    best_variant_id, best_variant_row, decision_reason = choose_best_variant(summary_df)

    log("[4/5] Writing markdown report")
    report_lines = [
        "# LINK Retention Cliff Forensics",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Artifact dir: `{run_dir}`",
        "",
        "## Discovered Inputs",
        "",
        f"- Repaired 1h baseline dir: `{discovered['baseline_dir']}`",
        f"- Rebased Model A dir: `{discovered['multicoin_dir']}`",
        f"- Winner-vs-loser diagnosis dir: `{discovered['diag_dir']}`",
        f"- Latest entry-repair pilot dir: `{discovered['pilot_dir']}`",
        f"- LINK control trade CSV: `{discovered['control_trade_csv']}`",
        f"- LINK failed hard-gap trade CSV: `{discovered['hard_gap_trade_csv']}`",
        "",
        "## Baseline Vs Failed Hard-Gap",
        "",
        f"- Repaired 1h baseline expectancy for {SYMBOL}: `{finite_float(discovered['baseline_row'].get('after_expectancy_net', np.nan)):.10f}`",
        f"- Current repaired Model A control expectancy for {SYMBOL}: `{finite_float(discovered['control_row'].get('expectancy_net', np.nan)):.10f}`",
        f"- Failed hard-gap expectancy for {SYMBOL}: `{finite_float(discovered['hard_gap_row'].get('expectancy_net', np.nan)):.10f}`",
        f"- Control trades: `{int(discovered['control_row'].get('total_trades', 0))}`",
        f"- Failed hard-gap trades: `{int(discovered['hard_gap_row'].get('total_trades', 0))}`",
        f"- Failed hard-gap retention: `{finite_float(discovered['hard_gap_row'].get('trade_count_retention_vs_control', np.nan)):.6f}`",
        "",
        "## Hard-Gap Removal Decomposition",
        "",
        f"- Removed trades: `{removed_summary['removed_total']}`",
        f"- Removed instant losers: `{removed_summary['removed_instant_losers']}`",
        f"- Removed fast losers: `{removed_summary['removed_fast_losers']}`",
        f"- Removed meaningful winners: `{removed_summary['removed_meaningful_winners']}`",
        f"- Removed positive-PnL trades: `{removed_summary['removed_positive_pnl']}`",
        f"- Removed non-positive-PnL trades: `{removed_summary['removed_nonpositive_pnl']}`",
        "",
        "## Gap Decile Decomposition",
        "",
        markdown_table(
            decile_df,
            [
                "action_gap_decile",
                "trades",
                "removed_by_hard_gap",
                "removed_share",
                "instant_losers",
                "fast_losers",
                "meaningful_winners",
                "median_gap_pct",
                "mean_control_pnl",
            ],
        ),
        "",
        "## Bounded Soft Variant Scan",
        "",
        markdown_table(
            summary_df,
            [
                "variant_id",
                "variant_kind",
                "instant_loser_count",
                "fast_loser_count",
                "expectancy_net",
                "cvar_5",
                "max_drawdown",
                "trade_count_retention_vs_control",
                "valid_for_ranking",
                "accepted",
            ],
        ),
        "",
        "## Decision",
        "",
        f"- Best surviving variant: `{best_variant_id}`",
        f"- Decision reason: {decision_reason}",
    ]
    (run_dir / "link_retention_cliff_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    log("[5/5] Writing manifest")
    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "symbol": SYMBOL,
        "discovered_inputs": {
            "baseline_dir": str(discovered["baseline_dir"]),
            "baseline_summary_csv": str(discovered["baseline_dir"] / "repaired_1h_reference_summary.csv"),
            "multicoin_dir": str(discovered["multicoin_dir"]),
            "multicoin_classification_csv": str(discovered["multicoin_dir"] / "repaired_multicoin_modelA_coin_classification.csv"),
            "multicoin_reference_vs_best_csv": str(discovered["multicoin_dir"] / "repaired_multicoin_modelA_reference_vs_best.csv"),
            "diag_dir": str(discovered["diag_dir"]),
            "diag_feature_matrix_csv": str(discovered["diag_dir"] / "instant_loser_vs_winner_feature_matrix.csv"),
            "pilot_dir": str(discovered["pilot_dir"]),
            "pilot_results_csv": str(discovered["pilot_dir"] / "live_coin_bounded_entry_repair_results.csv"),
            "pilot_vs_control_csv": str(discovered["pilot_dir"] / "live_coin_bounded_entry_repair_vs_control.csv"),
            "control_trade_csv": str(discovered["control_trade_csv"]),
            "hard_gap_trade_csv": str(discovered["hard_gap_trade_csv"]),
        },
        "gap_metric_definition": {
            "metric": "action_gap_pct",
            "formula": "action_open_1h / signal_close_1h - 1.0",
            "source_script": str((PROJECT_ROOT / "scripts" / "instant_loser_vs_winner_entry_forensics.py").resolve()),
        },
        "acceptance_logic_reused_from": str((PROJECT_ROOT / "scripts" / "live_coin_bounded_entry_repair_pilot.py").resolve()),
        "scan_meta": scan_meta,
        "removed_summary": removed_summary,
        "best_variant_id": best_variant_id,
        "best_variant_row": dict(best_variant_row.to_dict()) if best_variant_row is not None else None,
        "decision_reason": decision_reason,
        "outputs": {
            "trade_table_csv": str(run_dir / "link_retention_cliff_trade_table.csv"),
            "variant_summary_csv": str(run_dir / "link_retention_cliff_variant_summary.csv"),
            "report_md": str(run_dir / "link_retention_cliff_report.md"),
        },
    }
    json_dump(run_dir / "link_retention_cliff_manifest.json", manifest)
    log(str(run_dir))


if __name__ == "__main__":
    main()
