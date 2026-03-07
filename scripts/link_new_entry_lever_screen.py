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

from scripts import instant_loser_vs_winner_entry_forensics as forensics  # noqa: E402
from scripts import link_soft_repair_confirmation as soft_confirm  # noqa: E402
from scripts import live_coin_bounded_entry_repair_pilot as pilot  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402


SYMBOL = "LINKUSDT"
RUN_PREFIX = "LINK_NEW_ENTRY_LEVER_SCREEN"
PILOT_REQUIRED = {
    "live_coin_bounded_entry_repair_results.csv",
    "live_coin_bounded_entry_repair_vs_control.csv",
    "live_coin_bounded_entry_repair_decision.csv",
}
BASELINE_REQUIRED = {
    "repaired_1h_reference_summary.csv",
    "contract_repair_1h_report.md",
}
MULTICOIN_REQUIRED = {
    "repaired_multicoin_modelA_coin_classification.csv",
    "repaired_multicoin_modelA_run_manifest.json",
}
SOFT_CONFIRM_REQUIRED = {
    "link_soft_repair_confirmation_summary.csv",
    "link_soft_repair_confirmation_detail.csv",
}
CLIFF_REQUIRED = {
    "link_retention_cliff_trade_table.csv",
    "link_retention_cliff_variant_summary.csv",
}

FEATURE_FAMILIES: List[Dict[str, Any]] = [
    {
        "family_id": "UPPER_WICK_TAIL_CAP",
        "family_label": "Soft upper-wick tail cap",
        "lever_type": "single_feature_tail_cap",
        "features": ["upper_wick_ratio"],
        "direction": "high_bad",
        "interpretability_score": 5,
        "causal_cleanliness_score": 5,
        "variants": [
            {"variant_id": "UPPER_WICK_Q90", "label": "Reject upper-wick top 10%", "threshold_kind": "quantile", "quantile": 0.90},
            {"variant_id": "UPPER_WICK_Q85", "label": "Reject upper-wick top 15%", "threshold_kind": "quantile", "quantile": 0.85},
        ],
    },
    {
        "family_id": "ATR_REGIME_CAP",
        "family_label": "Extreme ATR percentile regime gate",
        "lever_type": "single_feature_regime_gate",
        "features": ["atr_percentile_1h"],
        "direction": "high_bad",
        "interpretability_score": 5,
        "causal_cleanliness_score": 5,
        "variants": [
            {"variant_id": "ATR_P95", "label": "Reject ATR percentile >= 95", "threshold_kind": "absolute", "threshold": 95.0},
            {"variant_id": "ATR_P90", "label": "Reject ATR percentile >= 90", "threshold_kind": "absolute", "threshold": 90.0},
        ],
    },
    {
        "family_id": "SMA_STRETCH_CAP",
        "family_label": "Positive 20h-mean stretch cap",
        "lever_type": "single_feature_positive_tail_cap",
        "features": ["dist_to_sma20_pct"],
        "direction": "high_bad",
        "interpretability_score": 5,
        "causal_cleanliness_score": 5,
        "variants": [
            {"variant_id": "SMA_STRETCH_POSQ90", "label": "Reject top 10% of positive SMA stretch", "threshold_kind": "positive_quantile", "quantile": 0.90},
            {"variant_id": "SMA_STRETCH_POSQ75", "label": "Reject top 25% of positive SMA stretch", "threshold_kind": "positive_quantile", "quantile": 0.75},
        ],
    },
    {
        "family_id": "WICK_ATR_SCORE",
        "family_label": "Compact wick+ATR exhaustion score",
        "lever_type": "two_feature_score_cap",
        "features": ["upper_wick_ratio", "atr_percentile_1h"],
        "direction": "high_bad",
        "interpretability_score": 4,
        "causal_cleanliness_score": 5,
        "variants": [
            {"variant_id": "WICK_ATR_SCORE_Q90", "label": "Reject top 10% wick+ATR score", "threshold_kind": "score_quantile", "quantile": 0.90},
            {"variant_id": "WICK_ATR_SCORE_Q85", "label": "Reject top 15% wick+ATR score", "threshold_kind": "score_quantile", "quantile": 0.85},
        ],
    },
    {
        "family_id": "WICK_RANGE_SCORE",
        "family_label": "Compact wick+range exhaustion score",
        "lever_type": "two_feature_score_cap",
        "features": ["upper_wick_ratio", "signal_range_pct"],
        "direction": "high_bad",
        "interpretability_score": 4,
        "causal_cleanliness_score": 5,
        "variants": [
            {"variant_id": "WICK_RANGE_SCORE_Q90", "label": "Reject top 10% wick+range score", "threshold_kind": "score_quantile", "quantile": 0.90},
            {"variant_id": "WICK_RANGE_SCORE_Q85", "label": "Reject top 15% wick+range score", "threshold_kind": "score_quantile", "quantile": 0.85},
        ],
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(msg: str) -> None:
    print(msg, flush=True)


def finite_float(x: Any, default: float = float("nan")) -> float:
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(v):
        return float(default)
    return float(v)


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


def find_latest_dir(prefix: str, required: Sequence[str]) -> Path:
    cands = sorted(
        [p for p in (PROJECT_ROOT / "reports" / "execution_layer").glob(f"{prefix}*") if p.is_dir()],
        key=lambda p: p.name,
    )
    need = set(required)
    for p in reversed(cands):
        names = {f.name for f in p.iterdir() if f.is_file()}
        if need.issubset(names):
            return p.resolve()
    raise FileNotFoundError(f"No completed {prefix} directory found")


def find_latest_pilot_dir() -> Path:
    cands = sorted(
        [p for p in (PROJECT_ROOT / "reports" / "execution_layer").glob("LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT_*") if p.is_dir()],
        key=lambda p: p.name,
    )
    for p in reversed(cands):
        names = {f.name for f in p.iterdir() if f.is_file()}
        if PILOT_REQUIRED.issubset(names):
            return p.resolve()
    raise FileNotFoundError("No completed LIVE_COIN_BOUNDED_ENTRY_REPAIR_PILOT directory found")


def discover_inputs() -> Dict[str, Path]:
    baseline_dir = find_latest_dir("1H_CONTRACT_REPAIR_REBASELINE_", BASELINE_REQUIRED)
    multicoin_dir = find_latest_dir("REPAIRED_MULTICOIN_MODELA_AUDIT_", MULTICOIN_REQUIRED)
    diag_dir = pilot.find_latest_diag_dir()
    pilot_dir = find_latest_pilot_dir()
    cliff_dir = find_latest_dir("LINK_RETENTION_CLIFF_FORENSICS_", CLIFF_REQUIRED)
    confirm_dir = find_latest_dir("LINK_SOFT_REPAIR_CONFIRMATION_", SOFT_CONFIRM_REQUIRED)

    comp_df = pd.read_csv(diag_dir / "instant_loser_vs_winner_comparison_by_coin.csv")
    comp_df["symbol"] = comp_df["symbol"].astype(str).str.upper()
    row = comp_df[comp_df["symbol"] == SYMBOL]
    if row.empty:
        raise KeyError(f"Missing {SYMBOL} in {diag_dir / 'instant_loser_vs_winner_comparison_by_coin.csv'}")
    best_candidate_id = str(row.iloc[0]["best_candidate_id"]).strip()

    trade_dir = pilot_dir / "_trade_sources"
    exact = trade_dir / f"{SYMBOL}_{best_candidate_id}_control.csv"
    if exact.exists():
        control_trade_csv = exact
    else:
        matches = sorted(trade_dir.glob(f"{SYMBOL}_*_control.csv"))
        if not matches:
            alt = trade_dir / f"{SYMBOL}_CONTROL.csv"
            if not alt.exists():
                raise FileNotFoundError(f"Missing LINK control trade source under {trade_dir}")
            control_trade_csv = alt
        else:
            control_trade_csv = matches[-1]

    return {
        "baseline_dir": baseline_dir,
        "multicoin_dir": multicoin_dir,
        "diag_dir": diag_dir,
        "pilot_dir": pilot_dir,
        "cliff_dir": cliff_dir,
        "confirm_dir": confirm_dir,
        "control_trade_csv": control_trade_csv.resolve(),
    }


def load_runtime_stack(multicoin_dir: Path, run_dir: Path) -> Tuple[pd.DataFrame, Any, argparse.Namespace, Path]:
    manifest_path = multicoin_dir / "repaired_multicoin_modelA_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    foundation_dir = Path(manifest.get("foundation_dir", phase_v.find_latest_foundation_dir())).resolve()
    foundation_state = phase_v.load_foundation_state(foundation_dir)
    exec_args = phase_v.build_exec_args(foundation_state, seed=20260303)

    sig_df = foundation_state.signal_timeline[
        foundation_state.signal_timeline["symbol"].astype(str).str.upper() == SYMBOL
    ].copy()
    win_df = foundation_state.download_manifest[
        foundation_state.download_manifest["symbol"].astype(str).str.upper() == SYMBOL
    ].copy()
    bundle, _ = phase_v.build_symbol_bundle(
        symbol=SYMBOL,
        symbol_signals=sig_df,
        symbol_windows=win_df,
        exec_args=exec_args,
        run_dir=run_dir,
    )
    return sig_df, bundle, exec_args, foundation_dir


def build_feature_state(diag_dir: Path) -> pd.DataFrame:
    feat = pd.read_csv(diag_dir / "instant_loser_vs_winner_feature_matrix.csv")
    feat["symbol"] = feat["symbol"].astype(str).str.upper()
    feat = feat[feat["symbol"] == SYMBOL].copy().reset_index(drop=True)
    if feat.empty:
        raise FileNotFoundError(f"No LINKUSDT rows in {diag_dir / 'instant_loser_vs_winner_feature_matrix.csv'}")
    feat["signal_id"] = feat["signal_id"].astype(str)

    feature_nums = [
        "upper_wick_ratio",
        "signal_range_pct",
        "signal_body_abs_pct",
        "atr_percentile_1h",
        "atr14_pct",
        "breakout_dist_pct",
        "dist_to_sma20_pct",
        "lower_wick_ratio",
    ]
    for col in feature_nums:
        if col in feat.columns:
            feat[col] = pd.to_numeric(feat[col], errors="coerce")

    feat["score_wick_atr"] = (
        pd.to_numeric(feat["upper_wick_ratio"], errors="coerce").rank(pct=True, method="average")
        + pd.to_numeric(feat["atr_percentile_1h"], errors="coerce").rank(pct=True, method="average")
    ) / 2.0
    feat["score_wick_range"] = (
        pd.to_numeric(feat["upper_wick_ratio"], errors="coerce").rank(pct=True, method="average")
        + pd.to_numeric(feat["signal_range_pct"], errors="coerce").rank(pct=True, method="average")
    ) / 2.0
    return feat.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def compute_threshold(feature_df: pd.DataFrame, family: Dict[str, Any], variant: Dict[str, Any]) -> float:
    kind = str(variant["threshold_kind"])
    if kind == "absolute":
        return float(variant["threshold"])

    if family["family_id"] == "WICK_ATR_SCORE":
        s = pd.to_numeric(feature_df["score_wick_atr"], errors="coerce").dropna()
    elif family["family_id"] == "WICK_RANGE_SCORE":
        s = pd.to_numeric(feature_df["score_wick_range"], errors="coerce").dropna()
    else:
        src = str(family["features"][0])
        s = pd.to_numeric(feature_df[src], errors="coerce").dropna()

    if s.empty:
        return float("nan")
    if kind == "quantile":
        return float(s.quantile(float(variant["quantile"])))
    if kind == "positive_quantile":
        pos = s[s > 0.0]
        base = pos if not pos.empty else s
        return float(base.quantile(float(variant["quantile"])))
    if kind == "score_quantile":
        return float(s.quantile(float(variant["quantile"])))
    raise KeyError(f"Unknown threshold kind: {kind}")


def candidate_value(feature_row: pd.Series, family: Dict[str, Any]) -> float:
    fid = str(family["family_id"])
    if fid == "WICK_ATR_SCORE":
        return finite_float(feature_row.get("score_wick_atr", np.nan))
    if fid == "WICK_RANGE_SCORE":
        return finite_float(feature_row.get("score_wick_range", np.nan))
    return finite_float(feature_row.get(str(family["features"][0]), np.nan))


def pass_family(feature_row: Optional[pd.Series], family: Dict[str, Any], threshold: float) -> bool:
    if feature_row is None:
        return True
    val = candidate_value(feature_row, family)
    if not np.isfinite(val) or not np.isfinite(threshold):
        return True
    direction = str(family["direction"])
    if direction == "high_bad":
        return bool(val < float(threshold))
    if direction == "low_bad":
        return bool(val > float(threshold))
    return True


def apply_filter_variant(
    control_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    family: Dict[str, Any],
    variant: Dict[str, Any],
    threshold: float,
) -> pd.DataFrame:
    feat_map = {str(r["signal_id"]): r for r in feature_df.to_dict("records")}
    rows: List[Dict[str, Any]] = []
    for _, row in control_df.iterrows():
        sid = str(row["signal_id"])
        feat = feat_map.get(sid)
        feat_row = pd.Series(feat) if feat is not None else None
        keep = pass_family(feat_row, family, threshold)
        out = dict(row.to_dict())
        out["repair_variant_id"] = str(variant["variant_id"])
        out["repair_variant_label"] = str(variant["label"])
        out["repair_family_id"] = str(family["family_id"])
        out["repair_filter_pass"] = int(keep)
        out["repair_threshold_value"] = float(threshold) if np.isfinite(threshold) else float("nan")
        out["repair_metric_value"] = candidate_value(feat_row, family) if feat_row is not None else float("nan")
        if not keep:
            out = pilot.zero_exec_fields(out, skip_reason=f"new_lever_reject:{variant['variant_id']}")
            out["repair_variant_id"] = str(variant["variant_id"])
            out["repair_variant_label"] = str(variant["label"])
            out["repair_family_id"] = str(family["family_id"])
            out["repair_filter_pass"] = 0
            out["repair_threshold_value"] = float(threshold) if np.isfinite(threshold) else float("nan")
            out["repair_metric_value"] = candidate_value(feat_row, family) if feat_row is not None else float("nan")
        rows.append(out)
    return pd.DataFrame(rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def removed_breakdown(candidate_df: pd.DataFrame, feature_df: pd.DataFrame) -> Dict[str, int]:
    removed_ids = set(candidate_df.loc[pd.to_numeric(candidate_df["repair_filter_pass"], errors="coerce").fillna(1).astype(int) == 0, "signal_id"].astype(str))
    removed = feature_df[feature_df["signal_id"].astype(str).isin(removed_ids)].copy()
    return {
        "removed_total": int(len(removed)),
        "removed_instant_losers": int((removed["bucket"] == "instant_loser").sum()),
        "removed_fast_losers": int((removed["bucket"] == "fast_loser").sum()),
        "removed_meaningful_winners": int((removed["bucket"] == "meaningful_winner").sum()),
    }


def preview_family(feature_df: pd.DataFrame, family: Dict[str, Any]) -> Dict[str, Any]:
    preview_variant = dict(family["variants"][0])
    threshold = compute_threshold(feature_df, family, preview_variant)
    tmp = pd.DataFrame(
        {
            "signal_id": feature_df["signal_id"].astype(str),
            "bucket": feature_df["bucket"].astype(str),
        }
    )
    tmp["repair_filter_pass"] = 1
    vals = [candidate_value(r, family) for _, r in feature_df.iterrows()]
    keep = [
        pass_family(pd.Series(feature_df.iloc[i].to_dict()), family, threshold)
        for i in range(len(feature_df))
    ]
    tmp["repair_filter_pass"] = pd.Series(keep, dtype=int)
    br = removed_breakdown(tmp, feature_df)
    retention = float((tmp["repair_filter_pass"] == 1).sum() / max(1, len(tmp)))
    removed_total = int(br["removed_total"])
    removed_loss = int(br["removed_instant_losers"] + br["removed_fast_losers"])
    loss_capture = float(removed_loss / max(1, removed_total))
    winner_penalty = float(br["removed_meaningful_winners"] / max(1, removed_total))
    priority_score = float(
        family["interpretability_score"]
        + family["causal_cleanliness_score"]
        + 4.0 * loss_capture
        - 4.0 * winner_penalty
        - 5.0 * max(0.0, 0.90 - retention)
    )
    return {
        "family_id": str(family["family_id"]),
        "family_label": str(family["family_label"]),
        "lever_type": str(family["lever_type"]),
        "features": "|".join(str(x) for x in family["features"]),
        "interpretability_score": int(family["interpretability_score"]),
        "causal_cleanliness_score": int(family["causal_cleanliness_score"]),
        "preview_variant_id": str(preview_variant["variant_id"]),
        "preview_threshold_value": float(threshold),
        "preview_removed_total": int(removed_total),
        "preview_removed_instant_losers": int(br["removed_instant_losers"]),
        "preview_removed_fast_losers": int(br["removed_fast_losers"]),
        "preview_removed_meaningful_winners": int(br["removed_meaningful_winners"]),
        "preview_retention": float(retention),
        "preview_loss_capture_ratio": float(loss_capture),
        "preview_winner_penalty_ratio": float(winner_penalty),
        "priority_score": float(priority_score),
        "tested": 1,
        "excludes_positive_gap_family": 1,
    }


def choose_promising(summary_df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
    if summary_df.empty:
        return "NO_NEW_LINK_LEVER_FOUND", summary_df
    x = summary_df.copy()
    x["promising_flag"] = (
        (x["accepted"] == 1)
        & (x["valid_for_ranking"] == 1)
        & (x["parity_clean"] == 1)
        & (x["removed_meaningful_winners"] <= 1)
        & (x["route_confirm_count"] >= 1)
        & (x["split_confirm_count"] >= np.maximum(3, np.floor(x["split_total"] * 0.8)).astype(int))
    ).astype(int)
    x = x.sort_values(
        [
            "promising_flag",
            "trade_count_retention_vs_control",
            "expectancy_delta_vs_control",
            "removed_meaningful_winners",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    top = x[x["promising_flag"] == 1].copy().head(2)
    if top.empty:
        return "NO_NEW_LINK_LEVER_FOUND", top
    return "PROMISING_NEW_LINK_LEVER_FOUND", top


def main() -> None:
    ap = argparse.ArgumentParser(description="Bounded LINK-only screen for new non-gap entry-repair levers")
    ap.add_argument("--outdir", default="reports/execution_layer")
    args = ap.parse_args()

    run_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    ensure_dir(run_dir)
    cand_dir = ensure_dir(run_dir / "_candidate_rows")

    log("[1/5] Discovering real artifacts and control path")
    discovered = discover_inputs()
    feature_df = build_feature_state(discovered["diag_dir"])

    log("[2/5] Rebuilding LINK evaluation bundle and loading control rows")
    _sig_df, bundle, exec_args, foundation_dir = load_runtime_stack(discovered["multicoin_dir"], run_dir)
    control_df = pd.read_csv(discovered["control_trade_csv"])
    control_df["signal_id"] = control_df["signal_id"].astype(str)
    control_df = control_df.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    control_metrics = pilot.compute_variant_metrics(df=control_df, bundle=bundle, exec_args=exec_args)

    log("[3/5] Building candidate inventory and screening bounded non-gap levers")
    inventory_rows = [preview_family(feature_df, family) for family in FEATURE_FAMILIES]
    inventory_df = pd.DataFrame(inventory_rows).sort_values(["priority_score", "preview_removed_meaningful_winners", "preview_retention"], ascending=[False, True, False]).reset_index(drop=True)
    inventory_df.to_csv(run_dir / "link_new_entry_lever_inventory.csv", index=False)

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    family_best_rows: List[Dict[str, Any]] = []

    for family in FEATURE_FAMILIES:
        best_family_row: Optional[Dict[str, Any]] = None
        for variant in family["variants"]:
            threshold = compute_threshold(feature_df, family, variant)
            cand_df = apply_filter_variant(control_df, feature_df, family, variant, threshold)
            cand_path = cand_dir / f"{variant['variant_id']}.csv"
            cand_df.to_csv(cand_path, index=False)

            metrics = pilot.compute_variant_metrics(df=cand_df, bundle=bundle, exec_args=exec_args)
            accepted, checks = pilot.accept_variant(control_metrics, metrics)
            metrics["accepted"] = int(accepted)
            metrics["trade_count_retention_vs_control"] = float(checks["trade_count_retention_vs_control"])
            metrics["expectancy_delta_vs_control"] = float(checks["expectancy_delta"])
            metrics["cvar_delta_vs_control"] = float(checks["cvar_delta"])
            metrics["maxdd_delta_vs_control"] = float(checks["maxdd_delta"])

            split_df, split_meta = soft_confirm.build_split_checks(cand_df, control_df, bundle)
            route_df, route_meta = soft_confirm.build_route_checks(
                candidate_df=cand_df,
                control_df=control_df,
                bundle=bundle,
                exec_args=exec_args,
            )
            for scope_df in [split_df, route_df]:
                if not scope_df.empty:
                    z = scope_df.copy()
                    z["family_id"] = str(family["family_id"])
                    z["variant_id"] = str(variant["variant_id"])
                    detail_rows.extend(z.to_dict(orient="records"))

            removed = removed_breakdown(cand_df, feature_df)
            row = {
                "symbol": SYMBOL,
                "family_id": str(family["family_id"]),
                "family_label": str(family["family_label"]),
                "lever_type": str(family["lever_type"]),
                "features": "|".join(str(x) for x in family["features"]),
                "variant_id": str(variant["variant_id"]),
                "variant_label": str(variant["label"]),
                "threshold_value": float(threshold) if np.isfinite(threshold) else float("nan"),
                "accepted": int(metrics["accepted"]),
                "expectancy_net": float(metrics["expectancy_net"]),
                "expectancy_delta_vs_control": float(metrics["expectancy_delta_vs_control"]),
                "cvar_5": float(metrics["cvar_5"]),
                "cvar_delta_vs_control": float(metrics["cvar_delta_vs_control"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "maxdd_delta_vs_control": float(metrics["maxdd_delta_vs_control"]),
                "trade_count_retention_vs_control": float(metrics["trade_count_retention_vs_control"]),
                "instant_loser_count": int(metrics["instant_loser_count"]),
                "instant_loser_delta_vs_control": int(metrics["instant_loser_count"] - control_metrics["instant_loser_count"]),
                "fast_loser_count": int(metrics["fast_loser_count"]),
                "fast_loser_delta_vs_control": int(metrics["fast_loser_count"] - control_metrics["fast_loser_count"]),
                "meaningful_winner_count": int(metrics["meaningful_winner_count"]),
                "meaningful_winner_delta_vs_control": int(metrics["meaningful_winner_count"] - control_metrics["meaningful_winner_count"]),
                "removed_total": int(removed["removed_total"]),
                "removed_instant_losers": int(removed["removed_instant_losers"]),
                "removed_fast_losers": int(removed["removed_fast_losers"]),
                "removed_meaningful_winners": int(removed["removed_meaningful_winners"]),
                "valid_for_ranking": int(metrics["valid_for_ranking"]),
                "invalid_reason": str(metrics["invalid_reason"]),
                "parity_clean": int(metrics["parity_clean"]),
                "split_confirm_count": int(split_meta["confirm_count"]),
                "split_total": int(split_meta["split_count"]),
                "route_confirm_count": int(route_meta["confirm_count"]),
                "route_total": int(route_meta["route_count"]),
                "route_family_total": int(route_meta["route_family_total"]),
                "route_unsupported_count": int(route_meta["route_unsupported_count"]),
                "candidate_trade_csv": str(cand_path),
            }
            row["screen_status"] = (
                "PROMISING_FOR_STRICT_CONFIRMATION"
                if (
                    int(row["accepted"]) == 1
                    and int(row["valid_for_ranking"]) == 1
                    and int(row["parity_clean"]) == 1
                    and int(row["removed_meaningful_winners"]) <= 1
                    and int(row["route_confirm_count"]) >= 1
                    and int(row["split_confirm_count"]) >= max(3, int(math.floor(int(row["split_total"]) * 0.8)))
                )
                else ("TOPLEVEL_ONLY" if int(row["accepted"]) == 1 else "REJECTED")
            )
            summary_rows.append(row)

            if best_family_row is None:
                best_family_row = row
            else:
                better = (
                    int(row["accepted"]),
                    int(row["screen_status"] == "PROMISING_FOR_STRICT_CONFIRMATION"),
                    int(row["valid_for_ranking"]),
                    float(row["trade_count_retention_vs_control"]),
                    float(row["expectancy_delta_vs_control"]),
                    -int(row["removed_meaningful_winners"]),
                )
                incumbent = (
                    int(best_family_row["accepted"]),
                    int(best_family_row["screen_status"] == "PROMISING_FOR_STRICT_CONFIRMATION"),
                    int(best_family_row["valid_for_ranking"]),
                    float(best_family_row["trade_count_retention_vs_control"]),
                    float(best_family_row["expectancy_delta_vs_control"]),
                    -int(best_family_row["removed_meaningful_winners"]),
                )
                if better > incumbent:
                    best_family_row = row
        if best_family_row is not None:
            family_best_rows.append(best_family_row)

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["screen_status", "trade_count_retention_vs_control", "expectancy_delta_vs_control"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    summary_df.to_csv(run_dir / "link_new_entry_lever_screen_summary.csv", index=False)

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["family_id", "variant_id", "scope_type", "scope_id"]).reset_index(drop=True)
        detail_df.to_csv(run_dir / "link_new_entry_lever_screen_detail.csv", index=False)

    family_best_df = pd.DataFrame(family_best_rows).sort_values(
        ["screen_status", "trade_count_retention_vs_control", "expectancy_delta_vs_control"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    decision, winners_df = choose_promising(summary_df)

    log("[4/5] Writing report and manifest")
    report_lines = [
        "# LINK New Entry Lever Screen",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Artifact dir: `{run_dir}`",
        "",
        "## A) Discovered Code Paths / Artifacts Used",
        "",
        f"- Repaired 1h baseline dir: `{discovered['baseline_dir']}`",
        f"- Rebased multicoin Model A dir: `{discovered['multicoin_dir']}`",
        f"- Winner-vs-loser forensic dir: `{discovered['diag_dir']}`",
        f"- Latest bounded entry-repair pilot dir: `{discovered['pilot_dir']}`",
        f"- Retention-cliff forensic dir: `{discovered['cliff_dir']}`",
        f"- Soft-confirmation dir (used as rejected-gap guardrail): `{discovered['confirm_dir']}`",
        f"- LINK control trade source: `{discovered['control_trade_csv']}`",
        f"- Evaluation path reused from: `{(PROJECT_ROOT / 'scripts' / 'live_coin_bounded_entry_repair_pilot.py').resolve()}`",
        f"- Split/route screen reused from: `{(PROJECT_ROOT / 'scripts' / 'link_soft_repair_confirmation.py').resolve()}`",
        "",
        "## B) Candidate Feature Inventory",
        "",
        markdown_table(
            inventory_df,
            [
                "family_id",
                "features",
                "priority_score",
                "preview_removed_instant_losers",
                "preview_removed_fast_losers",
                "preview_removed_meaningful_winners",
                "preview_retention",
            ],
        ),
        "",
        "## C) Bounded Lever Families Tested",
        "",
        markdown_table(
            family_best_df,
            [
                "family_id",
                "variant_id",
                "variant_label",
                "threshold_value",
                "screen_status",
            ],
        ),
        "",
        "## D) Results Table For Each Family",
        "",
        markdown_table(
            family_best_df,
            [
                "family_id",
                "variant_id",
                "accepted",
                "expectancy_delta_vs_control",
                "cvar_delta_vs_control",
                "maxdd_delta_vs_control",
                "trade_count_retention_vs_control",
                "instant_loser_delta_vs_control",
                "fast_loser_delta_vs_control",
                "removed_meaningful_winners",
                "valid_for_ranking",
            ],
        ),
        "",
        "## E) Proven Vs Assumed",
        "",
        "- Proven: all screened levers use only causal pre-entry 1h/3m-local features already present in the existing forensic feature matrix; the rejected positive-gap family was excluded.",
        "- Proven: acceptance metrics reuse the existing pilot logic without changing tolerances.",
        "- Proven: parity is preserved because only skip masks were changed; entry/exit timing, stop logic, and costs stayed frozen.",
        "- Assumed: route confirmation remains a bounded screen, not a full confirmation pass. Any promising candidate still needs the stricter confirmation workflow before deployment.",
        "",
        "## F) Best Candidate(s) Or NO_NEW_LINK_LEVER_FOUND",
        "",
        f"- Decision: `{decision}`",
    ]
    if winners_df.empty:
        report_lines.extend(
            [
                "- No candidate cleared the bounded screen strongly enough to justify a new strict confirmation pass.",
            ]
        )
    else:
        report_lines.extend(
            [
                markdown_table(
                    winners_df,
                    [
                        "family_id",
                        "variant_id",
                        "screen_status",
                        "expectancy_delta_vs_control",
                        "trade_count_retention_vs_control",
                        "instant_loser_delta_vs_control",
                        "removed_meaningful_winners",
                    ],
                ),
            ]
        )
    report_lines.extend(
        [
            "",
            "## G) Exact Recommendation On Whether LINK Entry-Repair Research Should Continue",
            "",
            (
                "- Continue, but only with the top 1-2 bounded non-gap candidates above and only through the same strict confirmation harness used on the rejected soft-gap family."
                if not winners_df.empty
                else "- Shift effort away from LINK local entry repair; this bounded non-gap screen did not produce a credible new lever."
            ),
        ]
    )
    (run_dir / "link_new_entry_lever_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "generated_utc": utc_now(),
        "symbol": SYMBOL,
        "discovered": {k: str(v) for k, v in discovered.items()},
        "foundation_dir": str(foundation_dir),
        "control_metrics": control_metrics,
        "feature_families_tested": FEATURE_FAMILIES,
        "positive_gap_family_excluded": 1,
        "positive_gap_exclusion_basis_dir": str(discovered["confirm_dir"]),
        "decision": decision,
        "top_winners": winners_df.to_dict(orient="records"),
    }
    json_dump(run_dir / "link_new_entry_lever_manifest.json", manifest)

    log("[5/5] Complete")
    print(str(run_dir))


if __name__ == "__main__":
    main()
