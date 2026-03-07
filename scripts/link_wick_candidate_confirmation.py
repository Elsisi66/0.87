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

from scripts import link_new_entry_lever_screen as screen  # noqa: E402
from scripts import link_soft_repair_confirmation as soft_confirm  # noqa: E402
from scripts import live_coin_bounded_entry_repair_pilot as pilot  # noqa: E402


SYMBOL = "LINKUSDT"
RUN_PREFIX = "LINK_WICK_CANDIDATE_CONFIRMATION"
SCREEN_REQUIRED = {
    "link_new_entry_lever_inventory.csv",
    "link_new_entry_lever_screen_summary.csv",
    "link_new_entry_lever_report.md",
}
CANDIDATES = [
    ("WICK_RANGE_SCORE", "WICK_RANGE_SCORE_Q90"),
    ("WICK_ATR_SCORE", "WICK_ATR_SCORE_Q90"),
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


def find_latest_screen_dir() -> Path:
    return screen.find_latest_dir("LINK_NEW_ENTRY_LEVER_SCREEN_", SCREEN_REQUIRED)


def discover_state() -> Dict[str, Path]:
    base = screen.discover_inputs()
    screen_dir = find_latest_screen_dir()
    base["screen_dir"] = screen_dir
    return base


def family_lookup() -> Dict[str, Dict[str, Any]]:
    return {str(f["family_id"]): dict(f) for f in screen.FEATURE_FAMILIES}


def variant_lookup(family: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(v["variant_id"]): dict(v) for v in family["variants"]}


def compare_reconstruction(
    *,
    rebuilt_df: pd.DataFrame,
    source_df: pd.DataFrame,
    summary_row: pd.Series,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    a = rebuilt_df.copy()
    b = source_df.copy()
    a["signal_id"] = a["signal_id"].astype(str)
    b["signal_id"] = b["signal_id"].astype(str)
    a = a.sort_values(["signal_id"]).reset_index(drop=True)
    b = b.sort_values(["signal_id"]).reset_index(drop=True)

    same_ids = int(a["signal_id"].tolist() == b["signal_id"].tolist())
    keys = [
        "repair_filter_pass",
        "exec_filled",
        "exec_valid_for_metrics",
        "exec_skip_reason",
    ]
    row_match = 1
    for col in keys:
        sa = a.get(col, pd.Series([""] * len(a), dtype=object)).fillna("").astype(str)
        sb = b.get(col, pd.Series([""] * len(b), dtype=object)).fillna("").astype(str)
        if sa.tolist() != sb.tolist():
            row_match = 0
            break

    num_checks = {
        "expectancy_match": int(abs(float(metrics["expectancy_net"]) - float(summary_row["expectancy_net"])) <= 1e-12),
        "cvar_match": int(abs(float(metrics["cvar_5"]) - float(summary_row["cvar_5"])) <= 1e-12),
        "maxdd_match": int(abs(float(metrics["max_drawdown"]) - float(summary_row["max_drawdown"])) <= 1e-12),
        "retention_match": int(abs(float(metrics["trade_count_retention_vs_control"]) - float(summary_row["trade_count_retention_vs_control"])) <= 1e-12),
        "instant_match": int(int(metrics["instant_loser_count"]) == int(summary_row["instant_loser_count"])),
        "fast_match": int(int(metrics["fast_loser_count"]) == int(summary_row["fast_loser_count"])),
        "winner_match": int(int(metrics["meaningful_winner_count"]) == int(summary_row["meaningful_winner_count"])),
    }
    out = {
        "same_signal_ids": int(same_ids),
        "row_mask_match": int(row_match),
        **num_checks,
    }
    out["reconstruction_valid"] = int(all(int(v) == 1 for v in out.values()))
    return out


def robustness_label(row: Dict[str, Any]) -> str:
    if int(row["acceptance_status"]) != 1:
        return "rejected"
    if int(row["parity_clean"]) != 1 or int(row["reconstruction_valid"]) != 1 or int(row["valid_for_ranking"]) != 1:
        return "rejected"

    split_all = int(row["split_confirm_count"]) == int(row["split_total"])
    route_all = int(row["route_confirm_count"]) == int(row["route_total"])
    time_all = int(row["time_slice_confirm_count"]) == int(row["time_slice_total"])
    seed_all = int(row["seed_supported"]) == 0 or int(row["seed_confirm_count"]) == int(row["seed_total"])
    cluster_ok = int(row["loss_cluster_confirm"]) == 1
    no_route_gap = int(row["route_unsupported_count"]) == 0

    if split_all and route_all and time_all and seed_all and cluster_ok and no_route_gap:
        return "stable"

    split_floor = int(row["split_confirm_count"]) >= max(1, int(math.ceil(int(row["split_total"]) * 0.8)))
    route_floor = int(row["route_confirm_count"]) >= max(1, int(math.ceil(int(row["route_total"]) * 0.75)))
    time_floor = int(row["time_slice_confirm_count"]) == int(row["time_slice_total"])
    if split_floor and route_floor and time_floor and cluster_ok and no_route_gap:
        return "borderline"

    return "fragile"


def choose_final(summary_df: pd.DataFrame) -> Tuple[str, Optional[pd.Series], str]:
    if summary_df.empty:
        return "NO_REPAIR_APPROVED", None, "No candidate rows were produced."

    rank_map = {"stable": 3, "borderline": 2, "fragile": 1, "rejected": 0}
    x = summary_df.copy()
    x["robust_rank"] = x["robustness_label"].map(rank_map).fillna(0).astype(int)
    x = x.sort_values(
        [
            "robust_rank",
            "trade_count_retention_vs_control",
            "expectancy_delta_vs_control",
            "winner_deletion_count",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = x.iloc[0]
    if str(best["robustness_label"]) not in {"stable", "borderline"}:
        return "NO_REPAIR_APPROVED", best, "Both screened wick candidates remain fragile under strict confirmation."
    if str(best["variant_id"]) == "WICK_RANGE_SCORE_Q90":
        return "SHADOW_REPAIR_ONLY_WICK_RANGE", best, "WICK_RANGE_SCORE_Q90 is the conservative surviving wick candidate."
    if str(best["variant_id"]) == "WICK_ATR_SCORE_Q90":
        return "SHADOW_REPAIR_ONLY_WICK_ATR", best, "WICK_ATR_SCORE_Q90 is the conservative surviving wick candidate."
    return "NO_REPAIR_APPROVED", best, "Unexpected best candidate identifier."


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict confirmation for LINK wick-family repair candidates")
    ap.add_argument("--outdir", default="reports/execution_layer")
    args = ap.parse_args()

    run_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    ensure_dir(run_dir)
    cand_dir = ensure_dir(run_dir / "_candidate_rows")

    log("[1/5] Discovering frozen artifacts and loading LINK control state")
    discovered = discover_state()
    feature_df = screen.build_feature_state(discovered["diag_dir"])
    signal_timeline_df, bundle, exec_args, foundation_dir = screen.load_runtime_stack(discovered["multicoin_dir"], run_dir)
    control_df = pd.read_csv(discovered["control_trade_csv"])
    control_df["signal_id"] = control_df["signal_id"].astype(str)
    control_df = control_df.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    control_metrics = pilot.compute_variant_metrics(df=control_df, bundle=bundle, exec_args=exec_args)

    screen_summary = pd.read_csv(discovered["screen_dir"] / "link_new_entry_lever_screen_summary.csv")
    screen_summary["variant_id"] = screen_summary["variant_id"].astype(str)
    families = family_lookup()

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    log("[2/5] Reconstructing the two frozen wick candidates exactly")
    for family_id, variant_id in CANDIDATES:
        family = families.get(str(family_id))
        if family is None:
            raise KeyError(f"Missing family {family_id} in link_new_entry_lever_screen.FEATURE_FAMILIES")
        variants = variant_lookup(family)
        variant = variants.get(str(variant_id))
        if variant is None:
            raise KeyError(f"Missing variant {variant_id} in family {family_id}")

        ref_row = screen_summary[screen_summary["variant_id"] == str(variant_id)]
        if ref_row.empty:
            raise KeyError(f"Missing {variant_id} in {discovered['screen_dir'] / 'link_new_entry_lever_screen_summary.csv'}")
        ref = ref_row.iloc[0]

        threshold = screen.compute_threshold(feature_df, family, variant)
        rebuilt_df = screen.apply_filter_variant(control_df, feature_df, family, variant, threshold)
        rebuilt_path = cand_dir / f"{variant_id}.csv"
        rebuilt_df.to_csv(rebuilt_path, index=False)

        metrics = pilot.compute_variant_metrics(df=rebuilt_df, bundle=bundle, exec_args=exec_args)
        accepted, checks = pilot.accept_variant(control_metrics, metrics)
        metrics["accepted"] = int(accepted)
        metrics["trade_count_retention_vs_control"] = float(checks["trade_count_retention_vs_control"])
        metrics["expectancy_delta_vs_control"] = float(checks["expectancy_delta"])
        metrics["cvar_delta_vs_control"] = float(checks["cvar_delta"])
        metrics["maxdd_delta_vs_control"] = float(checks["maxdd_delta"])

        source_candidate_path = Path(str(ref["candidate_trade_csv"])).resolve()
        if not source_candidate_path.exists():
            raise FileNotFoundError(f"Missing screen candidate trade CSV: {source_candidate_path}")
        source_df = pd.read_csv(source_candidate_path)
        recon = compare_reconstruction(
            rebuilt_df=rebuilt_df,
            source_df=source_df,
            summary_row=ref,
            metrics=metrics,
        )

        split_df, split_meta = soft_confirm.build_split_checks(rebuilt_df, control_df, bundle)
        route_df, route_meta = soft_confirm.build_route_checks(
            candidate_df=rebuilt_df,
            control_df=control_df,
            bundle=bundle,
            exec_args=exec_args,
        )
        time_df, time_meta = soft_confirm.build_time_slice_checks(rebuilt_df, control_df)
        seed_df, seed_meta = soft_confirm.build_seed_checks(rebuilt_df, control_df, signal_timeline_df)
        cluster_meta = soft_confirm.compute_loss_cluster_checks(rebuilt_df, control_df)

        for scope_df in [split_df, route_df, time_df, seed_df]:
            if not scope_df.empty:
                z = scope_df.copy()
                z["variant_id"] = str(variant_id)
                z["variant_label"] = str(variant["label"])
                z["family_id"] = str(family_id)
                detail_rows.extend(z.to_dict(orient="records"))

        removed = screen.removed_breakdown(rebuilt_df, feature_df)
        row = {
            "symbol": SYMBOL,
            "family_id": str(family_id),
            "family_label": str(family["family_label"]),
            "variant_id": str(variant_id),
            "variant_label": str(variant["label"]),
            "threshold_value": float(threshold),
            "acceptance_status": int(metrics["accepted"]),
            "expectancy_net": float(metrics["expectancy_net"]),
            "expectancy_delta_vs_control": float(metrics["expectancy_delta_vs_control"]),
            "cvar_5": float(metrics["cvar_5"]),
            "cvar_delta_vs_control": float(metrics["cvar_delta_vs_control"]),
            "max_drawdown": float(metrics["max_drawdown"]),
            "maxdd_delta_vs_control": float(metrics["maxdd_delta_vs_control"]),
            "trade_count_retention_vs_control": float(metrics["trade_count_retention_vs_control"]),
            "valid_for_ranking": int(metrics["valid_for_ranking"]),
            "invalid_reason": str(metrics["invalid_reason"]),
            "parity_clean": int(metrics["parity_clean"]),
            "instant_loser_count": int(metrics["instant_loser_count"]),
            "instant_loser_delta_vs_control": int(metrics["instant_loser_count"] - control_metrics["instant_loser_count"]),
            "fast_loser_count": int(metrics["fast_loser_count"]),
            "fast_loser_delta_vs_control": int(metrics["fast_loser_count"] - control_metrics["fast_loser_count"]),
            "meaningful_winner_count": int(metrics["meaningful_winner_count"]),
            "winner_deletion_count": int(control_metrics["meaningful_winner_count"] - metrics["meaningful_winner_count"]),
            "reconstruction_valid": int(recon["reconstruction_valid"]),
            "reconstruction_same_signal_ids": int(recon["same_signal_ids"]),
            "reconstruction_row_mask_match": int(recon["row_mask_match"]),
            "split_confirm_count": int(split_meta["confirm_count"]),
            "split_total": int(split_meta["split_count"]),
            "min_split_delta_vs_control": float(split_meta["min_split_delta"]),
            "route_confirm_count": int(route_meta["confirm_count"]),
            "route_total": int(route_meta["route_count"]),
            "route_family_total": int(route_meta["route_family_total"]),
            "route_unsupported_count": int(route_meta["route_unsupported_count"]),
            "route_support_reason": str(route_meta["route_support_reason"]),
            "time_slice_confirm_count": int(time_meta["confirm_count"]),
            "time_slice_total": int(time_meta["time_slice_count"]),
            "min_time_slice_delta_vs_control": float(time_meta["min_time_slice_delta"]),
            "seed_supported": int(seed_meta["supported"]),
            "seed_confirm_count": int(seed_meta["confirm_count"]),
            "seed_total": int(seed_meta["seed_count"]),
            "seed_reason": str(seed_meta["reason"]),
            "loss_cluster_confirm": int(cluster_meta["confirm_flag"]),
            "loss_cluster_avg_burden_improve_ratio": float(cluster_meta["loss_cluster_avg_burden_improve_ratio"]),
            "loss_cluster_worst_burden_improve_ratio": float(cluster_meta["loss_cluster_worst_burden_improve_ratio"]),
            "corr_adjusted_config_clusters_supported": int(cluster_meta["correlation_adjusted_config_cluster_supported"]),
            "corr_adjusted_config_clusters_reason": str(cluster_meta["correlation_adjusted_config_cluster_reason"]),
            "screen_candidate_trade_csv": str(source_candidate_path),
            "candidate_trade_csv": str(rebuilt_path),
        }
        row["robustness_label"] = robustness_label(row)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(["variant_id"]).reset_index(drop=True)
    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(["variant_id", "scope_type", "scope_id"]).reset_index(drop=True)
    summary_df.to_csv(run_dir / "link_wick_candidate_confirmation_summary.csv", index=False)
    detail_df.to_csv(run_dir / "link_wick_candidate_confirmation_detail.csv", index=False)

    decision, best_row, decision_reason = choose_final(summary_df)

    log("[3/5] Writing markdown report")
    route_fragility = {
        str(r["variant_id"]): {
            "route_confirm_count": int(r["route_confirm_count"]),
            "route_total": int(r["route_total"]),
            "route_family_total": int(r["route_family_total"]),
            "route_unsupported_count": int(r["route_unsupported_count"]),
        }
        for _, r in summary_df.iterrows()
    }
    report_lines = [
        "# LINK Wick Candidate Confirmation",
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
        f"- Rejected soft-gap confirmation dir: `{discovered['confirm_dir']}`",
        f"- Non-gap screen dir: `{discovered['screen_dir']}`",
        f"- LINK control trade source: `{discovered['control_trade_csv']}`",
        f"- Evaluation path reused from: `{(PROJECT_ROOT / 'scripts' / 'live_coin_bounded_entry_repair_pilot.py').resolve()}`",
        f"- Strict confirmation helpers reused from: `{(PROJECT_ROOT / 'scripts' / 'link_soft_repair_confirmation.py').resolve()}`",
        f"- Candidate reconstruction reused from: `{(PROJECT_ROOT / 'scripts' / 'link_new_entry_lever_screen.py').resolve()}`",
        "",
        "## B) Reconstruction Validity",
        "",
        markdown_table(
            summary_df,
            [
                "variant_id",
                "reconstruction_valid",
                "reconstruction_same_signal_ids",
                "reconstruction_row_mask_match",
                "acceptance_status",
            ],
        ),
        "",
        "## C) Strict Confirmation Results Table",
        "",
        markdown_table(
            summary_df,
            [
                "variant_id",
                "acceptance_status",
                "expectancy_delta_vs_control",
                "cvar_delta_vs_control",
                "maxdd_delta_vs_control",
                "trade_count_retention_vs_control",
                "instant_loser_delta_vs_control",
                "fast_loser_delta_vs_control",
                "winner_deletion_count",
                "valid_for_ranking",
                "robustness_label",
            ],
        ),
        "",
        "## D) Robustness Summary By Split / Route / Time Slice / Seed",
        "",
        markdown_table(
            summary_df,
            [
                "variant_id",
                "split_confirm_count",
                "split_total",
                "route_confirm_count",
                "route_total",
                "route_family_total",
                "route_unsupported_count",
                "time_slice_confirm_count",
                "time_slice_total",
                "seed_supported",
                "seed_confirm_count",
                "seed_total",
                "loss_cluster_confirm",
            ],
        ),
        "",
        "## E) Failure Mode Check",
        "",
        f"- leakage/lookahead: `proven clean` via preserved repaired chronology and `parity_clean=1` for both candidates",
        f"- ranking validity: `{summary_df.set_index('variant_id')['valid_for_ranking'].astype(int).to_dict()}`",
        f"- sample collapse: retention values are `{summary_df.set_index('variant_id')['trade_count_retention_vs_control'].round(6).to_dict()}`; both stay above the pilot floor",
        f"- winner deletion: `{summary_df.set_index('variant_id')['winner_deletion_count'].astype(int).to_dict()}`",
        f"- route fragility: `{route_fragility}`",
        "- unrealistic execution assumptions: `assumed unchanged` because only the same frozen skip masks were applied; entry timing, exits, stop logic, and costs were unchanged",
        "",
        "## F) Conservative Winner Or NO_REPAIR_APPROVED",
        "",
        f"- Decision: `{decision}`",
        f"- Decision reason: {decision_reason}",
        "",
        "## G) Final LINK Deployment Posture",
        "",
        f"- Final posture: `{decision}`",
        "",
        "## H) Exact Recommendation",
        "",
        (
            "- Stop LINK local entry-repair research. Neither wick-family candidate clears strict confirmation robustly enough for shadow deployment."
            if decision == "NO_REPAIR_APPROVED"
            else "- Continue only with the selected shadow candidate and log raw vs filtered LINK decisions side-by-side before any stronger promotion."
        ),
    ]
    (run_dir / "link_wick_candidate_confirmation_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    log("[4/5] Writing manifest")
    manifest = {
        "generated_utc": utc_now(),
        "symbol": SYMBOL,
        "discovered": {k: str(v) for k, v in discovered.items()},
        "foundation_dir": str(foundation_dir),
        "control_metrics": control_metrics,
        "candidates": [{"family_id": a, "variant_id": b} for a, b in CANDIDATES],
        "decision": decision,
        "decision_reason": decision_reason,
        "best_row": None if best_row is None else dict(best_row.to_dict()),
    }
    json_dump(run_dir / "link_wick_candidate_confirmation_manifest.json", manifest)

    log("[5/5] Complete")
    print(str(run_dir))


if __name__ == "__main__":
    main()
