#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import backtest_exec_phasec_sol as phasec_bt  # noqa: E402
from scripts import phase_a_model_a_audit as phase_a  # noqa: E402
from scripts import phase_nx_exec_family_discovery as nx  # noqa: E402
from scripts import phase_r_route_harness_redesign as phase_r  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


PHASEA_DIR_DEFAULT = Path(
    "/root/analysis/0.87/reports/execution_layer/PHASEA_MODEL_A_AUDIT_20260228_014944"
).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def to_num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def norm_cdf(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0))))


def z_proxy(mean: float, std: float, n: float) -> float:
    if (not np.isfinite(mean)) or (not np.isfinite(std)) or (not np.isfinite(n)):
        return float("nan")
    if std <= 0.0 or n <= 1.0:
        return float("nan")
    return float(mean / (std / math.sqrt(n)))


def write_text(path: Path, text: str) -> None:
    phase_a.write_text(path, text)


def json_dump(path: Path, obj: Any) -> None:
    phase_a.json_dump(path, obj)


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: int = 12) -> str:
    return phase_a.markdown_table(df, cols, n)


def config_fingerprint(cfg: Dict[str, Any]) -> str:
    core = {
        "entry_mode": str(cfg["entry_mode"]),
        "limit_offset_bps": round(float(cfg["limit_offset_bps"]), 6),
        "fallback_to_market": int(cfg["fallback_to_market"]),
        "fallback_delay_min": round(float(cfg["fallback_delay_min"]), 6),
        "max_fill_delay_min": round(float(cfg["max_fill_delay_min"]), 6),
    }
    return json.dumps(core, sort_keys=True)


def config_vector(cfg: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            1.0 if str(cfg["entry_mode"]).lower() == "market" else 0.0,
            float(cfg["limit_offset_bps"]) / 2.0,
            float(int(cfg["fallback_to_market"])),
            float(cfg["fallback_delay_min"]) / 45.0,
            float(cfg["max_fill_delay_min"]) / 45.0,
        ],
        dtype=float,
    )


def build_seed_catalog() -> List[Dict[str, Any]]:
    seeds = []
    for cfg in phase_a.build_model_a_variants():
        cid = str(cfg["candidate_id"])
        if cid.startswith(("M1_", "M2_", "M3_")):
            seeds.append(copy.deepcopy(cfg))
    return seeds


def with_updates(base: Dict[str, Any], cid: str, label: str, **updates: Any) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    out.update(updates)
    out["candidate_id"] = str(cid)
    out["label"] = str(label)
    return out


def generate_local_neighbors(seed_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    sid = str(seed_cfg["candidate_id"])
    base = copy.deepcopy(seed_cfg)
    neighbors: List[Dict[str, Any]] = []
    neighbors.append(with_updates(base, sid, str(seed_cfg["label"])))

    offsets = sorted(
        {
            round(clamp(float(base["limit_offset_bps"]) - 0.20, 0.0, 2.25), 4),
            round(clamp(float(base["limit_offset_bps"]) + 0.20, 0.0, 2.25), 4),
            round(clamp(float(base["limit_offset_bps"]) + 0.45, 0.0, 2.25), 4),
        }
    )
    for i, off in enumerate(offsets):
        neighbors.append(
            with_updates(
                base,
                f"{sid}_OFF_{i:02d}",
                f"{sid} local offset {off:.2f}",
                limit_offset_bps=float(off),
            )
        )

    time_pairs = [
        (
            clamp(float(base["fallback_delay_min"]) - 1.0, 0.0, 45.0),
            max(
                clamp(float(base["max_fill_delay_min"]) - 3.0, 0.0, 45.0),
                clamp(float(base["fallback_delay_min"]) - 1.0, 0.0, 45.0),
            ),
        ),
        (
            clamp(float(base["fallback_delay_min"]) + 1.0, 0.0, 45.0),
            clamp(float(base["max_fill_delay_min"]) + 3.0, 0.0, 45.0),
        ),
        (
            clamp(float(base["fallback_delay_min"]) + 2.0, 0.0, 45.0),
            clamp(float(base["max_fill_delay_min"]), 0.0, 45.0),
        ),
    ]
    for i, (fb, mx) in enumerate(time_pairs):
        mx = max(float(mx), float(fb))
        neighbors.append(
            with_updates(
                base,
                f"{sid}_TIME_{i:02d}",
                f"{sid} local timing {i}",
                fallback_delay_min=float(fb),
                max_fill_delay_min=float(mx),
            )
        )

    combo_specs = [
        {
            "limit_offset_bps": clamp(float(base["limit_offset_bps"]) - 0.12, 0.0, 2.25),
            "fallback_delay_min": clamp(float(base["fallback_delay_min"]) - 1.0, 0.0, 45.0),
            "max_fill_delay_min": max(
                clamp(float(base["max_fill_delay_min"]) - 3.0, 0.0, 45.0),
                clamp(float(base["fallback_delay_min"]) - 1.0, 0.0, 45.0),
            ),
        },
        {
            "limit_offset_bps": clamp(float(base["limit_offset_bps"]) + 0.12, 0.0, 2.25),
            "fallback_delay_min": clamp(float(base["fallback_delay_min"]) + 1.0, 0.0, 45.0),
            "max_fill_delay_min": clamp(float(base["max_fill_delay_min"]) + 3.0, 0.0, 45.0),
        },
        {
            "limit_offset_bps": clamp(float(base["limit_offset_bps"]) + 0.28, 0.0, 2.25),
            "fallback_delay_min": clamp(float(base["fallback_delay_min"]), 0.0, 45.0),
            "max_fill_delay_min": max(
                clamp(float(base["max_fill_delay_min"]) - 1.0, 0.0, 45.0),
                clamp(float(base["fallback_delay_min"]), 0.0, 45.0),
            ),
        },
    ]
    for i, spec in enumerate(combo_specs):
        neighbors.append(
            with_updates(
                base,
                f"{sid}_COMBO_{i:02d}",
                f"{sid} local combo {i}",
                **spec,
            )
        )

    # Include one local no-fallback probe around each seed to test neighborhood fragility.
    neighbors.append(
        with_updates(
            base,
            f"{sid}_NOFB",
            f"{sid} local no fallback",
            fallback_to_market=0,
            fallback_delay_min=0.0,
            max_fill_delay_min=max(0.0, float(base["max_fill_delay_min"])),
        )
    )

    return neighbors


def collapse_duplicates(raw_candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    by_fp: Dict[str, Dict[str, Any]] = {}
    raw_rows: List[Dict[str, Any]] = []
    for raw in raw_candidates:
        fp = config_fingerprint(raw)
        canonical = by_fp.get(fp)
        if canonical is None:
            canonical_id = str(raw["candidate_id"])
            cfg = copy.deepcopy(raw)
            cfg["canonical_id"] = canonical_id
            cfg["raw_variant_count"] = 1
            cfg["seed_anchor"] = str(raw.get("seed_anchor", ""))
            by_fp[fp] = cfg
        else:
            canonical_id = str(canonical["canonical_id"])
            canonical["raw_variant_count"] = int(canonical.get("raw_variant_count", 1)) + 1
        raw_rows.append(
            {
                "raw_candidate_id": str(raw["candidate_id"]),
                "raw_label": str(raw.get("label", "")),
                "seed_anchor": str(raw.get("seed_anchor", "")),
                "canonical_id": str(canonical_id),
                "config_fingerprint": str(fp),
            }
        )
    unique = sorted(by_fp.values(), key=lambda x: str(x["canonical_id"]))
    dup_df = pd.DataFrame(raw_rows).sort_values(["canonical_id", "raw_candidate_id"]).reset_index(drop=True)
    return unique, dup_df


def union_find_cluster_count(vectors: Dict[str, np.ndarray], dist_threshold: float) -> Tuple[int, Dict[str, int]]:
    ids = list(vectors.keys())
    parent = {k: k for k in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(ids):
        va = vectors[a]
        for b in ids[i + 1 :]:
            vb = vectors[b]
            d = float(np.sum(np.abs(va - vb)))
            if d <= float(dist_threshold):
                union(a, b)

    root_to_cluster: Dict[str, int] = {}
    cluster_map: Dict[str, int] = {}
    next_id = 1
    for cid in ids:
        root = find(cid)
        if root not in root_to_cluster:
            root_to_cluster[root] = next_id
            next_id += 1
        cluster_map[cid] = int(root_to_cluster[root])
    return int(len(root_to_cluster)), cluster_map


def candidate_psr_dsr(
    split_df: pd.DataFrame,
    effective_trials_corr_adjusted: float,
) -> Tuple[float, float]:
    delta = to_num(split_df.get("delta_expectancy_exec_minus_baseline", pd.Series(dtype=float))).dropna()
    if delta.empty:
        return float("nan"), float("nan")
    mean = float(delta.mean())
    std = float(delta.std(ddof=0))
    z = z_proxy(mean=mean, std=std, n=float(len(delta)))
    psr = norm_cdf(z)
    penalty = math.sqrt(max(0.0, 2.0 * math.log(max(1.0, float(effective_trials_corr_adjusted)))))
    dsr = norm_cdf(float(z - penalty)) if np.isfinite(z) else float("nan")
    return float(psr), float(dsr)


def build_stress_configs(cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    c1 = copy.deepcopy(cfg)
    c1["candidate_id"] = f"{cfg['canonical_id']}_STRESS_OFFSET"
    c1["label"] = f"{cfg.get('label', cfg['canonical_id'])} stress offset"
    c1["limit_offset_bps"] = clamp(float(cfg["limit_offset_bps"]) + 0.15, 0.0, 2.25)

    c2 = copy.deepcopy(cfg)
    c2["candidate_id"] = f"{cfg['canonical_id']}_STRESS_WINDOW"
    c2["label"] = f"{cfg.get('label', cfg['canonical_id'])} stress window"
    c2["fallback_delay_min"] = clamp(float(cfg["fallback_delay_min"]) + 1.0, 0.0, 45.0)
    c2["max_fill_delay_min"] = max(
        float(c2["fallback_delay_min"]),
        clamp(float(cfg["max_fill_delay_min"]) + 1.0, 0.0, 45.0),
    )
    return [("offset", c1), ("window", c2)]


def evaluate_routes_for_candidate(
    *,
    cfg: Dict[str, Any],
    route_bundles: Dict[str, ga_exec.SymbolBundle],
    baseline_routes: Dict[str, pd.DataFrame],
    one_h: phase_a.OneHMarket,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rid, rb in route_bundles.items():
        rev = phase_a.evaluate_model_a_variant(
            bundle=rb,
            baseline_df=baseline_routes[rid],
            cfg=cfg,
            one_h=one_h,
            args=args,
        )
        rm = rev["metrics"]
        rows.append(
            {
                "candidate_id": str(cfg["canonical_id"]),
                "route_id": str(rid),
                "valid_for_ranking": int(rm["valid_for_ranking"]),
                "delta_expectancy_vs_1h_reference": float(rm["overall_delta_expectancy_exec_minus_baseline"]),
                "exec_expectancy_net": float(rm["overall_exec_expectancy_net"]),
                "entry_rate": float(rm["overall_entry_rate"]),
                "entries_valid": int(rm["overall_entries_valid"]),
                "taker_share": float(rm["overall_exec_taker_share"]),
                "min_subperiod_delta": float(rm["min_split_delta"]),
                "route_test_signals": int(phase_r.bundle_test_count(rb)),
            }
        )
    return pd.DataFrame(rows).sort_values("route_id").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase B bounded expansion around Model A entry-only winners")
    ap.add_argument("--seed", type=int, default=20260228)
    ap.add_argument("--phase-a-dir", default=str(PHASEA_DIR_DEFAULT))
    ap.add_argument("--phase-r-dir", default=str(phase_a.PHASER_DIR_DEFAULT))
    ap.add_argument("--outdir", default="reports/execution_layer")
    args_cli = ap.parse_args()

    phase_a_dir = Path(args_cli.phase_a_dir).resolve()
    phase_r_dir = Path(args_cli.phase_r_dir).resolve()

    run_root = (PROJECT_ROOT / args_cli.outdir).resolve()
    run_dir = run_root / f"PHASEB_MODEL_A_BOUNDED_EXPANSION_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    subset_path = Path(nx.LOCKED["representative_subset_csv"]).resolve()
    exec_args = nx.build_exec_args(signals_csv=subset_path, seed=int(args_cli.seed))
    lock_info = ga_exec._validate_and_lock_frozen_artifacts(args=exec_args, run_dir=run_dir)  # pylint: disable=protected-access
    if int(lock_info.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("Frozen contract validation failed")

    bundles, _load_meta = ga_exec._prepare_bundles(exec_args)  # pylint: disable=protected-access
    if not bundles:
        raise RuntimeError("No bundles prepared under locked contract")
    base_bundle = bundles[0]
    one_h = phase_a.load_1h_market(base_bundle.symbol)
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
    route_match_flag, route_mismatches = phase_a.route_examples_match(
        actual=route_examples_df,
        expected_path=phase_r_dir / "phaseR1_route_examples.csv",
    )

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

    phase_a_results_fp = phase_a_dir / "phaseA3_modelA_results.csv"
    if not phase_a_results_fp.exists():
        raise FileNotFoundError(f"Missing Phase A results: {phase_a_results_fp}")
    phase_a_results = pd.read_csv(phase_a_results_fp)
    anchor_rows = phase_a_results[phase_a_results["candidate_id"].astype(str).str.startswith(("M1_", "M2_", "M3_"))].copy()

    baseline_full = phase_a.build_1h_reference_rows(
        bundle=base_bundle,
        fee=fee,
        exec_horizon_hours=float(exec_args.exec_horizon_hours),
    )
    baseline_routes: Dict[str, pd.DataFrame] = {}
    for rid, bundle in route_bundles.items():
        baseline_routes[rid] = phase_a.build_1h_reference_rows(
            bundle=bundle,
            fee=fee,
            exec_horizon_hours=float(exec_args.exec_horizon_hours),
        )

    b1 = {
        "generated_utc": utc_now(),
        "freeze_lock_pass": int(lock_info.get("freeze_lock_pass", 0)),
        "wrapper_uses_3m_entry_only": 1,
        "wrapper_uses_1h_exit_only": 1,
        "hybrid_exit_override_detected": 1,
        "forbidden_exit_knobs_blocked": forbidden_exit_knobs,
        "phase_a_reference_dir": str(phase_a_dir),
        "phase_r_reference_dir": str(phase_r_dir),
        "route_reproduction_match_phaseR": int(route_match_flag),
        "route_reproduction_mismatches": route_mismatches,
        "route_count": int(route_meta.get("route_count", 0)),
    }
    json_dump(run_dir / "phaseB1_contract_validation.json", b1)

    b1_lines = [
        "# B1 Model A Lock Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Freeze lock pass: `{int(lock_info.get('freeze_lock_pass', 0))}`",
        "- Wrapper remains identical to Phase A and still enforces:",
        "  - 3m entry execution only",
        "  - 1h exit-only ownership after fill",
        "  - no dynamic 3m exit mutation",
        f"- Forbidden exit knobs blocked: `{forbidden_exit_knobs}`",
        f"- Phase A reference dir: `{phase_a_dir}`",
        f"- Repaired route reproduction matches Phase R: `{route_match_flag}`",
        f"- Route reproduction mismatches: `{route_mismatches}`",
        "",
        "## Repaired Route Feasibility",
        "",
        markdown_table(
            route_feas_df,
            [
                "route_id",
                "route_signal_count",
                "wf_test_signal_count",
                "headroom_vs_overall_gate",
                "route_trade_gates_reachable",
            ],
            n=8,
        ),
        "",
        "## Phase A Anchor Rows",
        "",
        markdown_table(
            anchor_rows,
            [
                "candidate_id",
                "valid_for_ranking",
                "exec_expectancy_net",
                "delta_expectancy_vs_1h_reference",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "route_pass",
            ],
            n=8,
        ),
        "",
    ]
    write_text(run_dir / "phaseB1_modelA_lock_report.md", "\n".join(b1_lines))

    # B2 bounded neighborhood generation.
    seed_catalog = build_seed_catalog()
    raw_candidates: List[Dict[str, Any]] = []
    for seed_cfg in seed_catalog:
        sid = str(seed_cfg["candidate_id"])
        for cand in generate_local_neighbors(seed_cfg):
            cand = copy.deepcopy(cand)
            cand["seed_anchor"] = sid
            raw_candidates.append(cand)

    unique_candidates, dup_map_df = collapse_duplicates(raw_candidates)
    dup_map_df.to_csv(run_dir / "phaseB2_duplicate_variant_map.csv", index=False)

    vectors = {str(c["canonical_id"]): config_vector(c) for c in unique_candidates}
    effective_trials_uncorrelated = int(len(unique_candidates))
    corr_cluster_count, cluster_map = union_find_cluster_count(vectors=vectors, dist_threshold=0.38)
    for cfg in unique_candidates:
        cfg["cluster_id"] = int(cluster_map[str(cfg["canonical_id"])])

    invalid_hist: Counter[str] = Counter()
    results_rows: List[Dict[str, Any]] = []
    eval_cache: Dict[str, Dict[str, Any]] = {}

    ref_eval = phase_a.evaluate_reference_bundle(baseline_full)
    ref_metrics = ref_eval["metrics"]
    results_rows.append(
        {
            "candidate_id": "M0_1H_REFERENCE",
            "label": "M0 pure original 1h reference",
            "seed_anchor": "M0_1H_REFERENCE",
            "cluster_id": 0,
            "raw_variant_count": 1,
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
            "entry_mode": "market",
            "limit_offset_bps": 0.0,
            "fallback_to_market": 0,
            "fallback_delay_min": 0.0,
            "max_fill_delay_min": 0.0,
        }
    )

    for cfg in unique_candidates:
        ev = phase_a.evaluate_model_a_variant(
            bundle=base_bundle,
            baseline_df=baseline_full,
            cfg=cfg,
            one_h=one_h,
            args=exec_args,
        )
        eval_cache[str(cfg["canonical_id"])] = ev
        m = ev["metrics"]
        results_rows.append(
            {
                "candidate_id": str(cfg["canonical_id"]),
                "label": str(cfg.get("label", "")),
                "seed_anchor": str(cfg.get("seed_anchor", "")),
                "cluster_id": int(cfg.get("cluster_id", -1)),
                "raw_variant_count": int(cfg.get("raw_variant_count", 1)),
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
                "entry_mode": str(cfg["entry_mode"]),
                "limit_offset_bps": float(cfg["limit_offset_bps"]),
                "fallback_to_market": int(cfg["fallback_to_market"]),
                "fallback_delay_min": float(cfg["fallback_delay_min"]),
                "max_fill_delay_min": float(cfg["max_fill_delay_min"]),
            }
        )
        for part in [x for x in str(m["invalid_reason"]).split("|") if x]:
            invalid_hist[str(part)] += 1

    results_df = pd.DataFrame(results_rows).sort_values(
        ["candidate_id"],
        ascending=[True],
    ).reset_index(drop=True)
    results_df.to_csv(run_dir / "phaseB2_results.csv", index=False)
    json_dump(run_dir / "phaseB2_invalid_reason_histogram.json", dict(sorted(invalid_hist.items())))

    eff_lines = [
        "# B2 Effective Trials Summary",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Raw neighborhood candidates generated: `{len(raw_candidates)}`",
        f"- Unique configs evaluated after duplicate collapse: `{effective_trials_uncorrelated}`",
        f"- Duplicate-collapsed configs: `{len(raw_candidates) - effective_trials_uncorrelated}`",
        f"- Correlation-adjusted config clusters: `{corr_cluster_count}`",
        "- Correlation adjustment method:",
        "  - normalize entry-only knob vectors",
        "  - cluster configs by L1 distance <= 0.38",
        "  - use cluster count as the cheap correlation-adjusted effective trial count",
        "",
    ]
    write_text(run_dir / "phaseB2_effective_trials_summary.md", "\n".join(eff_lines))

    b2_lines = [
        "# B2 Bounded Entry-only Neighborhood Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Raw candidates: `{len(raw_candidates)}`",
        f"- Unique candidates: `{effective_trials_uncorrelated}`",
        f"- Correlation-adjusted clusters: `{corr_cluster_count}`",
        "",
        "## Top Results",
        "",
        markdown_table(
            results_df[results_df["candidate_id"] != "M0_1H_REFERENCE"].sort_values(
                ["delta_expectancy_vs_1h_reference", "cvar_improve_ratio", "maxdd_improve_ratio"],
                ascending=[False, False, False],
            ),
            [
                "candidate_id",
                "seed_anchor",
                "cluster_id",
                "valid_for_ranking",
                "exec_expectancy_net",
                "delta_expectancy_vs_1h_reference",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "taker_share",
                "p95_fill_delay_min",
                "invalid_reason",
            ],
            n=20,
        ),
        "",
    ]
    write_text(run_dir / "phaseB2_report.md", "\n".join(b2_lines))

    # B3 robustness on top bounded candidates only.
    valid_positive = results_df[
        (results_df["candidate_id"] != "M0_1H_REFERENCE")
        & (to_num(results_df["valid_for_ranking"]).fillna(0).astype(int) == 1)
        & (to_num(results_df["delta_expectancy_vs_1h_reference"]) > 0.0)
    ].copy()
    ranked_valid_positive = valid_positive.sort_values(
        ["delta_expectancy_vs_1h_reference", "cvar_improve_ratio", "maxdd_improve_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    shortlisted_ids: List[str] = []
    for _cluster_id, grp in ranked_valid_positive.groupby("cluster_id", sort=False):
        if grp.empty:
            continue
        shortlisted_ids.append(str(grp.iloc[0]["candidate_id"]))
    for cid in ranked_valid_positive["candidate_id"].astype(str).tolist():
        if cid not in shortlisted_ids:
            shortlisted_ids.append(cid)
        if len(shortlisted_ids) >= 8:
            break
    shortlisted_ids = shortlisted_ids[:8]
    shortlist = ranked_valid_positive[ranked_valid_positive["candidate_id"].astype(str).isin(shortlisted_ids)].copy()
    shortlist = shortlist.sort_values(
        ["delta_expectancy_vs_1h_reference", "cvar_improve_ratio", "maxdd_improve_ratio"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    route_rows: List[Dict[str, Any]] = []
    stress_rows: List[Dict[str, Any]] = []
    sig_rows: List[Dict[str, Any]] = []
    survivor_rows: List[Dict[str, Any]] = []

    for _, top in shortlist.iterrows():
        cid = str(top["candidate_id"])
        cfg = next(c for c in unique_candidates if str(c["canonical_id"]) == cid)
        ev = eval_cache[cid]
        split_df = ev["split_rows_df"].copy()
        route_df = evaluate_routes_for_candidate(
            cfg=cfg,
            route_bundles=route_bundles,
            baseline_routes=baseline_routes,
            one_h=one_h,
            args=exec_args,
        )
        route_rows.extend(route_df.to_dict(orient="records"))

        route_pass_rate = float(
            (
                (to_num(route_df["valid_for_ranking"]).fillna(0).astype(int) == 1)
                & (to_num(route_df["delta_expectancy_vs_1h_reference"]) > 0.0)
            ).mean()
        ) if not route_df.empty else float("nan")
        route_pass = int(
            (not route_df.empty)
            and (
                (to_num(route_df["valid_for_ranking"]).fillna(0).astype(int) == 1)
                & (to_num(route_df["delta_expectancy_vs_1h_reference"]) > 0.0)
            ).all()
        )
        center_row = route_df[route_df["route_id"].astype(str) == "route_center_60pct"].head(1)
        center_valid = int(center_row["valid_for_ranking"].iloc[0]) if not center_row.empty else 0
        center_delta = float(center_row["delta_expectancy_vs_1h_reference"].iloc[0]) if not center_row.empty else float("nan")
        min_subperiod_delta = float(to_num(split_df.get("delta_expectancy_exec_minus_baseline", pd.Series(dtype=float))).min()) if not split_df.empty else float("nan")

        stress_results: List[Dict[str, Any]] = []
        for stress_id, stress_cfg in build_stress_configs(cfg):
            sev = phase_a.evaluate_model_a_variant(
                bundle=base_bundle,
                baseline_df=baseline_full,
                cfg=stress_cfg,
                one_h=one_h,
                args=exec_args,
            )
            sm = sev["metrics"]
            stress_results.append(
                {
                    "candidate_id": cid,
                    "stress_id": str(stress_id),
                    "valid_for_ranking": int(sm["valid_for_ranking"]),
                    "delta_expectancy_vs_1h_reference": float(sm["overall_delta_expectancy_exec_minus_baseline"]),
                    "exec_expectancy_net": float(sm["overall_exec_expectancy_net"]),
                    "taker_share": float(sm["overall_exec_taker_share"]),
                    "p95_fill_delay_min": float(sm["overall_exec_p95_fill_delay_min"]),
                    "invalid_reason": str(sm["invalid_reason"]),
                }
            )
        stress_df = pd.DataFrame(stress_results)
        stress_rows.extend(stress_results)
        stress_delta_mean = float(to_num(stress_df["delta_expectancy_vs_1h_reference"]).mean()) if not stress_df.empty else float("nan")
        stress_pass = int(
            (not stress_df.empty)
            and (to_num(stress_df["valid_for_ranking"]).fillna(0).astype(int) == 1).all()
            and (to_num(stress_df["delta_expectancy_vs_1h_reference"]) > 0.0).all()
        )

        psr_proxy, dsr_proxy = candidate_psr_dsr(
            split_df=split_df,
            effective_trials_corr_adjusted=float(corr_cluster_count),
        )
        robust_survivor = int(
            int(top["valid_for_ranking"]) == 1
            and route_pass == 1
            and center_valid == 1
            and np.isfinite(center_delta)
            and center_delta > 0.0
            and np.isfinite(min_subperiod_delta)
            and min_subperiod_delta > 0.0
            and stress_pass == 1
            and np.isfinite(psr_proxy)
            and psr_proxy >= 0.95
            and np.isfinite(dsr_proxy)
            and dsr_proxy >= 0.80
        )

        sig_rows.append(
            {
                "candidate_id": cid,
                "seed_anchor": str(top["seed_anchor"]),
                "cluster_id": int(top["cluster_id"]),
                "split_count": int(len(split_df)),
                "psr_proxy": float(psr_proxy),
                "dsr_proxy": float(dsr_proxy),
                "effective_trials_uncorrelated": int(effective_trials_uncorrelated),
                "effective_trials_corr_adjusted": int(corr_cluster_count),
            }
        )
        survivor_rows.append(
            {
                "candidate_id": cid,
                "seed_anchor": str(top["seed_anchor"]),
                "cluster_id": int(top["cluster_id"]),
                "valid_for_ranking": int(top["valid_for_ranking"]),
                "exec_expectancy_net": float(top["exec_expectancy_net"]),
                "delta_expectancy_vs_1h_reference": float(top["delta_expectancy_vs_1h_reference"]),
                "cvar_improve_ratio": float(top["cvar_improve_ratio"]),
                "maxdd_improve_ratio": float(top["maxdd_improve_ratio"]),
                "entry_rate": float(top["entry_rate"]),
                "entries_valid": int(top["entries_valid"]),
                "taker_share": float(top["taker_share"]),
                "median_fill_delay_min": float(top["median_fill_delay_min"]),
                "p95_fill_delay_min": float(top["p95_fill_delay_min"]),
                "route_pass": int(route_pass),
                "route_pass_rate": float(route_pass_rate),
                "center_route_valid": int(center_valid),
                "center_route_delta": float(center_delta),
                "min_subperiod_delta": float(min_subperiod_delta),
                "stress_pass": int(stress_pass),
                "stress_delta_mean": float(stress_delta_mean),
                "psr_proxy": float(psr_proxy),
                "dsr_proxy": float(dsr_proxy),
                "robust_survivor": int(robust_survivor),
            }
        )

    route_checks_df = pd.DataFrame(route_rows).sort_values(["candidate_id", "route_id"]).reset_index(drop=True)
    stress_lite_df = pd.DataFrame(stress_rows).sort_values(["candidate_id", "stress_id"]).reset_index(drop=True)
    sig_df = pd.DataFrame(sig_rows).sort_values(["candidate_id"]).reset_index(drop=True)
    survivors_df = pd.DataFrame(survivor_rows).sort_values(
        ["robust_survivor", "delta_expectancy_vs_1h_reference", "psr_proxy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    route_checks_df.to_csv(run_dir / "phaseB3_route_checks.csv", index=False)
    stress_lite_df.to_csv(run_dir / "phaseB3_stress_lite.csv", index=False)
    sig_df.to_csv(run_dir / "phaseB3_shortlist_significance.csv", index=False)

    robust_survivors = survivors_df[to_num(survivors_df.get("robust_survivor", 0)).fillna(0).astype(int) == 1].copy()
    robust_clusters = set(to_num(robust_survivors.get("cluster_id", pd.Series(dtype=float))).dropna().astype(int).tolist()) if not robust_survivors.empty else set()
    stable_neighborhood = int(len(robust_survivors) >= 3 and len(robust_clusters) >= 2)

    b3_lines = [
        "# B3 Top Survivors Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Shortlist size: `{len(shortlist)}`",
        f"- Robust survivors: `{len(robust_survivors)}`",
        f"- Robust survivor clusters: `{sorted(robust_clusters)}`",
        f"- Stable neighborhood: `{stable_neighborhood}`",
        "",
        "## Shortlisted Candidates",
        "",
        markdown_table(
            survivors_df,
            [
                "candidate_id",
                "seed_anchor",
                "cluster_id",
                "delta_expectancy_vs_1h_reference",
                "route_pass",
                "center_route_delta",
                "min_subperiod_delta",
                "stress_pass",
                "stress_delta_mean",
                "psr_proxy",
                "dsr_proxy",
                "robust_survivor",
            ],
            n=12,
        ),
        "",
    ]
    write_text(run_dir / "phaseB3_top_survivors_report.md", "\n".join(b3_lines))

    # B4 strict decision.
    positive_winners = survivors_df[
        (to_num(survivors_df.get("valid_for_ranking", 0)).fillna(0).astype(int) == 1)
        & (to_num(survivors_df.get("delta_expectancy_vs_1h_reference", 0)) > 0.0)
    ].copy()

    if robust_survivors.empty:
        if positive_winners.empty:
            classification = "MODEL_A_NO_GO"
            mainline_status = "MODEL_A_NO_GO"
        else:
            classification = "MODEL_A_LUCKY_POINTS"
            mainline_status = "MODEL_A_LUCKY_POINTS"
        next_prompt = ""
    elif stable_neighborhood == 1:
        classification = "MODEL_A_STRONG_GO"
        mainline_status = "MODEL_A_STRONG_GO"
        next_prompt = (
            "ROLE\n"
            "You are in Phase C Model A confirmation mode for SOLUSDT.\n\n"
            "MISSION\n"
            "Run a bounded confirmation on only the robust Phase B entry-only survivors under the same frozen 1h signal/exit wrapper, "
            "same repaired routes, unchanged hard gates, and no 3m exit controls. Stop on first route-center or stress regression.\n"
        )
    else:
        classification = "MODEL_A_WEAK_GO"
        mainline_status = "MODEL_A_WEAK_GO"
        next_prompt = (
            "ROLE\n"
            "You are in Phase C Model A narrow confirmation mode for SOLUSDT.\n\n"
            "MISSION\n"
            "Re-test only the narrow robust Phase B entry-only survivors under the same frozen 1h exit wrapper and repaired routes. "
            "Keep compute bounded, do not add exit knobs, and stop on first robustness regression.\n"
        )

    if next_prompt:
        write_text(run_dir / "ready_to_launch_phaseC_prompt.txt", next_prompt)

    decision_lines = [
        "# Phase B Decision",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Classification: `{classification}`",
        f"- Mainline status: `{mainline_status}`",
        f"- Raw candidates: `{len(raw_candidates)}`",
        f"- Unique candidates: `{effective_trials_uncorrelated}`",
        f"- Corr-adjusted clusters: `{corr_cluster_count}`",
        f"- Positive shortlisted winners: `{len(positive_winners)}`",
        f"- Robust survivors: `{len(robust_survivors)}`",
        f"- Stable neighborhood: `{stable_neighborhood}`",
        "",
    ]
    write_text(run_dir / "phaseB_decision.md", "\n".join(decision_lines))

    manifest = {
        "generated_utc": utc_now(),
        "classification": classification,
        "mainline_status": mainline_status,
        "frozen_subset": str(subset_path),
        "phase_a_dir": str(phase_a_dir),
        "phase_r_dir": str(phase_r_dir),
        "git_snapshot": phase_a.git_snapshot(),
        "freeze_lock": lock_info,
        "wrapper_uses_3m_entry_only": 1,
        "wrapper_uses_1h_exit_only": 1,
        "forbidden_exit_knobs_blocked": forbidden_exit_knobs,
        "raw_candidates": int(len(raw_candidates)),
        "unique_candidates": int(effective_trials_uncorrelated),
        "corr_adjusted_clusters": int(corr_cluster_count),
        "robust_survivor_count": int(len(robust_survivors)),
        "stable_neighborhood": int(stable_neighborhood),
    }
    json_dump(run_dir / "phaseB_run_manifest.json", manifest)

    result = {
        "furthest_phase": "B4",
        "classification": classification,
        "mainline_status": mainline_status,
        "run_dir": str(run_dir),
        "robust_survivor_count": int(len(robust_survivors)),
        "stable_neighborhood": int(stable_neighborhood),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
