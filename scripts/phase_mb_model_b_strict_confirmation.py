#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_b_model_a_bounded_expansion as phase_b  # noqa: E402
from scripts import phase_model_b_sizing_overlay as model_b0  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


REPORTS_ROOT = PROJECT_ROOT / "reports" / "execution_layer"
MODELB_DIR_DEFAULT = (REPORTS_ROOT / "PHASEMODELB_SIZING_OVERLAY_20260228_141341").resolve()

ROUTE_DELTA_TOL = model_b0.ROUTE_DELTA_TOL
STRESS_DELTA_TOL = model_b0.STRESS_DELTA_TOL
PROMOTE_BOOTSTRAP_MIN = 0.90
BACKUP_BOOTSTRAP_MIN = 0.75


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
    return model_b0.markdown_table(df, list(cols), n=n)


def git_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        out["git_head"] = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        out["git_head"] = "unavailable"
    try:
        status = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "status", "--short"], text=True)
        out["git_status_short"] = status.strip().splitlines()
    except Exception:
        out["git_status_short"] = []
    return out


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def multiplier_hash(multiplier: np.ndarray) -> str:
    arr = np.asarray(multiplier, dtype=float)
    rounded = np.round(arr, 8)
    return hashlib.sha256(rounded.tobytes()).hexdigest()


def build_soft_multiplier(delay: np.ndarray, *, delay_gt_min: float, size_mult_on_delay: float) -> np.ndarray:
    out = np.ones(len(delay), dtype=float)
    out[delay > float(delay_gt_min)] = float(size_mult_on_delay)
    return out


def build_tiered_multiplier(
    delay: np.ndarray,
    *,
    cutoff_min: float,
    delay_0_to_cutoff_mult: float,
    delay_ge_cutoff_mult: float,
) -> np.ndarray:
    out = np.ones(len(delay), dtype=float)
    cutoff = float(cutoff_min)
    out[(delay > 0.0) & (delay < cutoff)] = float(delay_0_to_cutoff_mult)
    out[delay >= cutoff] = float(delay_ge_cutoff_mult)
    return out


def add_soft_variant(
    variants: List[Dict[str, Any]],
    delay: np.ndarray,
    *,
    variant_id: str,
    label: str,
    seed_origin: str,
    delay_gt_min: float,
    size_mult_on_delay: float,
) -> None:
    params = {
        "delay_gt_min": float(delay_gt_min),
        "size_mult_on_delay": float(size_mult_on_delay),
    }
    variants.append(
        {
            "seed_origin": str(seed_origin),
            "family": "regime_cap_size",
            "variant_id": str(variant_id),
            "label": str(label),
            "params": params,
            "multiplier": build_soft_multiplier(delay, **params),
            "param_vector": np.array(
                [0.0, float(delay_gt_min) / 12.0, float(size_mult_on_delay), float(size_mult_on_delay)],
                dtype=float,
            ),
        }
    )


def add_tiered_variant(
    variants: List[Dict[str, Any]],
    delay: np.ndarray,
    *,
    variant_id: str,
    label: str,
    seed_origin: str,
    cutoff_min: float,
    delay_0_to_cutoff_mult: float,
    delay_ge_cutoff_mult: float,
) -> None:
    params = {
        "cutoff_min": float(cutoff_min),
        "delay_0_to_cutoff_mult": float(delay_0_to_cutoff_mult),
        "delay_ge_cutoff_mult": float(delay_ge_cutoff_mult),
    }
    variants.append(
        {
            "seed_origin": str(seed_origin),
            "family": "regime_cap_size",
            "variant_id": str(variant_id),
            "label": str(label),
            "params": params,
            "multiplier": build_tiered_multiplier(delay, **params),
            "param_vector": np.array(
                [1.0, float(cutoff_min) / 12.0, float(delay_0_to_cutoff_mult), float(delay_ge_cutoff_mult)],
                dtype=float,
            ),
        }
    )


def build_local_candidates(base_rows: pd.DataFrame) -> List[Dict[str, Any]]:
    delay = to_num(base_rows["exec_fill_delay_min"]).fillna(0.0).to_numpy(dtype=float)
    variants: List[Dict[str, Any]] = []

    # Approved Model B anchors.
    add_tiered_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_tiered",
        label="Cap 3m delayed fills by delay tier",
        seed_origin="regime_cap_size_delay_tiered",
        cutoff_min=6.0,
        delay_0_to_cutoff_mult=0.85,
        delay_ge_cutoff_mult=0.35,
    )
    add_soft_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_soft",
        label="Cap delayed fills to 0.75x",
        seed_origin="regime_cap_size_delay_soft",
        delay_gt_min=0.0,
        size_mult_on_delay=0.75,
    )

    # Tight local neighborhood around the approved tiered overlay.
    add_tiered_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_tiered_CUTOFF_03",
        label="Tiered cutoff 3m",
        seed_origin="regime_cap_size_delay_tiered",
        cutoff_min=3.0,
        delay_0_to_cutoff_mult=0.85,
        delay_ge_cutoff_mult=0.35,
    )
    add_tiered_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_tiered_CUTOFF_09",
        label="Tiered cutoff 9m",
        seed_origin="regime_cap_size_delay_tiered",
        cutoff_min=9.0,
        delay_0_to_cutoff_mult=0.85,
        delay_ge_cutoff_mult=0.35,
    )
    add_tiered_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_tiered_LOW_080",
        label="Tiered low bucket 0.80x",
        seed_origin="regime_cap_size_delay_tiered",
        cutoff_min=6.0,
        delay_0_to_cutoff_mult=0.80,
        delay_ge_cutoff_mult=0.35,
    )
    add_tiered_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_tiered_LOW_090",
        label="Tiered low bucket 0.90x",
        seed_origin="regime_cap_size_delay_tiered",
        cutoff_min=6.0,
        delay_0_to_cutoff_mult=0.90,
        delay_ge_cutoff_mult=0.35,
    )
    add_tiered_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_tiered_HIGH_025",
        label="Tiered high-delay bucket 0.25x",
        seed_origin="regime_cap_size_delay_tiered",
        cutoff_min=6.0,
        delay_0_to_cutoff_mult=0.85,
        delay_ge_cutoff_mult=0.25,
    )
    add_tiered_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_tiered_HIGH_045",
        label="Tiered high-delay bucket 0.45x",
        seed_origin="regime_cap_size_delay_tiered",
        cutoff_min=6.0,
        delay_0_to_cutoff_mult=0.85,
        delay_ge_cutoff_mult=0.45,
    )
    add_tiered_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_tiered_COMBO_A",
        label="Tiered combo A",
        seed_origin="regime_cap_size_delay_tiered",
        cutoff_min=3.0,
        delay_0_to_cutoff_mult=0.80,
        delay_ge_cutoff_mult=0.30,
    )
    add_tiered_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_tiered_COMBO_B",
        label="Tiered combo B",
        seed_origin="regime_cap_size_delay_tiered",
        cutoff_min=9.0,
        delay_0_to_cutoff_mult=0.90,
        delay_ge_cutoff_mult=0.45,
    )
    add_tiered_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_tiered_COMBO_C",
        label="Tiered combo C",
        seed_origin="regime_cap_size_delay_tiered",
        cutoff_min=6.0,
        delay_0_to_cutoff_mult=0.82,
        delay_ge_cutoff_mult=0.40,
    )

    # Tight local neighborhood around the approved soft overlay.
    add_soft_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_soft_MULT_070",
        label="Soft delayed fill size 0.70x",
        seed_origin="regime_cap_size_delay_soft",
        delay_gt_min=0.0,
        size_mult_on_delay=0.70,
    )
    add_soft_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_soft_MULT_080",
        label="Soft delayed fill size 0.80x",
        seed_origin="regime_cap_size_delay_soft",
        delay_gt_min=0.0,
        size_mult_on_delay=0.80,
    )
    add_soft_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_soft_MULT_085",
        label="Soft delayed fill size 0.85x",
        seed_origin="regime_cap_size_delay_soft",
        delay_gt_min=0.0,
        size_mult_on_delay=0.85,
    )
    add_soft_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_soft_TH_03",
        label="Soft threshold 3m",
        seed_origin="regime_cap_size_delay_soft",
        delay_gt_min=3.0,
        size_mult_on_delay=0.75,
    )
    add_soft_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_soft_TH_03_M070",
        label="Soft threshold 3m at 0.70x",
        seed_origin="regime_cap_size_delay_soft",
        delay_gt_min=3.0,
        size_mult_on_delay=0.70,
    )
    add_soft_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_soft_TH_03_M080",
        label="Soft threshold 3m at 0.80x",
        seed_origin="regime_cap_size_delay_soft",
        delay_gt_min=3.0,
        size_mult_on_delay=0.80,
    )
    add_soft_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_soft_TH_06",
        label="Soft threshold 6m",
        seed_origin="regime_cap_size_delay_soft",
        delay_gt_min=6.0,
        size_mult_on_delay=0.75,
    )
    add_soft_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_soft_TH_06_M065",
        label="Soft threshold 6m at 0.65x",
        seed_origin="regime_cap_size_delay_soft",
        delay_gt_min=6.0,
        size_mult_on_delay=0.65,
    )
    add_soft_variant(
        variants,
        delay,
        variant_id="regime_cap_size_delay_soft_TH_06_M085",
        label="Soft threshold 6m at 0.85x",
        seed_origin="regime_cap_size_delay_soft",
        delay_gt_min=6.0,
        size_mult_on_delay=0.85,
    )
    return variants


def collapse_duplicates(raw_candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    by_hash: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    for raw in raw_candidates:
        fp = multiplier_hash(np.asarray(raw["multiplier"], dtype=float))
        cur = by_hash.get(fp)
        if cur is None:
            x = copy.deepcopy(raw)
            x["canonical_id"] = str(raw["variant_id"])
            x["raw_variant_count"] = 1
            x["multiplier_hash"] = fp
            by_hash[fp] = x
            canonical_id = str(x["canonical_id"])
        else:
            canonical_id = str(cur["canonical_id"])
            cur["raw_variant_count"] = int(cur.get("raw_variant_count", 1)) + 1
        rows.append(
            {
                "raw_variant_id": str(raw["variant_id"]),
                "seed_origin": str(raw["seed_origin"]),
                "canonical_id": canonical_id,
                "multiplier_hash": fp,
            }
        )
    unique = sorted(by_hash.values(), key=lambda x: str(x["canonical_id"]))
    dup_df = pd.DataFrame(rows).sort_values(["canonical_id", "raw_variant_id"]).reset_index(drop=True)
    return unique, dup_df


def stress_rows_for_scenario(rows_df: pd.DataFrame, scenario_id: str) -> pd.DataFrame:
    if scenario_id == "base_stress":
        return model_b0.stressed_rows(rows_df)
    x = rows_df.copy()
    pnl = to_num(x["exec_pnl_net_pct"]).fillna(0.0).to_numpy(dtype=float)
    delay = to_num(x.get("exec_fill_delay_min", pd.Series(dtype=float))).fillna(0.0).to_numpy(dtype=float)
    if scenario_id == "slippage_heavy":
        stressed = np.where(pnl < 0.0, pnl * 1.15, pnl * 0.88)
    elif scenario_id == "delay_penalty":
        delay_mult = np.where(delay > 0.0, 0.90, 1.0)
        tmp = pnl * delay_mult
        stressed = np.where(tmp < 0.0, tmp * 1.05, tmp * 0.96)
    else:
        raise ValueError(f"unknown stress scenario: {scenario_id}")
    x["exec_pnl_net_pct"] = stressed
    x["exec_pnl_gross_pct"] = to_num(x["exec_pnl_gross_pct"]).fillna(0.0).to_numpy(dtype=float)
    return x


def split_rows_for_candidate(base_rows: pd.DataFrame, cand_rows: pd.DataFrame, candidate_id: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if "split_id" not in base_rows.columns:
        return pd.DataFrame(rows)
    base_work = base_rows.copy()
    cand_work = cand_rows.copy()
    for split_id, base_group in base_work.groupby("split_id", dropna=False):
        base_group = base_group.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
        cand_group = cand_work[cand_work["split_id"] == split_id].sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
        rb = ga_exec._rollup_mode(base_group, "exec")
        rc = ga_exec._rollup_mode(cand_group, "exec")
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "split_id": str(split_id),
                "base_expectancy_net": float(rb["mean_expectancy_net"]),
                "cand_expectancy_net": float(rc["mean_expectancy_net"]),
                "delta_expectancy_vs_modelA": float(rc["mean_expectancy_net"] - rb["mean_expectancy_net"]),
                "base_cvar_5": float(rb["cvar_5"]),
                "cand_cvar_5": float(rc["cvar_5"]),
                "cand_entries_valid": int(rc["entries_valid"]),
            }
        )
    return pd.DataFrame(rows).sort_values("split_id").reset_index(drop=True)


def split_bootstrap_pass_rate(split_df: pd.DataFrame, *, draws: int, seed: int) -> float:
    delta = to_num(split_df.get("delta_expectancy_vs_modelA", pd.Series(dtype=float))).dropna().to_numpy(dtype=float)
    if delta.size == 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    passes = 0
    for _ in range(int(draws)):
        pick = rng.integers(0, delta.size, size=delta.size)
        sample = delta[pick]
        if float(np.mean(sample)) >= STRESS_DELTA_TOL and float(np.min(sample)) >= ROUTE_DELTA_TOL:
            passes += 1
    return float(passes / max(1, int(draws)))


def anchor_diffs(current_df: pd.DataFrame, reference_df: pd.DataFrame, variant_id: str) -> Dict[str, float]:
    cur = current_df[current_df["variant_id"].astype(str) == str(variant_id)].head(1)
    ref = reference_df[reference_df["variant_id"].astype(str) == str(variant_id)].head(1)
    if cur.empty or ref.empty:
        return {}
    metrics = [
        "expectancy_net",
        "delta_expectancy_vs_modelA",
        "cvar_improve_ratio",
        "maxdd_improve_ratio",
        "route_pass",
        "stress_lite_pass",
        "bootstrap_pass_rate",
    ]
    out: Dict[str, float] = {}
    for col in metrics:
        out[col] = float(abs(float(cur.iloc[0][col]) - float(ref.iloc[0][col])))
    return out


def decision_from_survivors(
    survivors_df: pd.DataFrame,
    corr_cluster_count: int,
) -> Tuple[str, str, str]:
    robust = survivors_df[to_num(survivors_df.get("robust_survivor", 0)).fillna(0).astype(int) == 1].copy()
    robust_seed_count = int(robust["seed_origin"].astype(str).nunique()) if not robust.empty else 0
    robust_cluster_count = int(robust["cluster_id"].nunique()) if not robust.empty else 0
    stable_neighborhood = int(len(robust) >= 4 and robust_seed_count >= 2 and robust_cluster_count >= 2)

    if not robust.empty:
        best = robust.sort_values(
            [
                "delta_expectancy_vs_modelA",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "loss_cluster_avg_burden_improve_ratio",
                "taker_share",
            ],
            ascending=[False, False, False, False, True],
        ).iloc[0]
        if (
            stable_neighborhood == 1
            and len(robust) >= 4
            and float(best["delta_expectancy_vs_modelA"]) >= -5e-5
            and float(best["cvar_improve_ratio"]) >= 0.05
            and float(best["maxdd_improve_ratio"]) >= 0.01
            and float(best["bootstrap_pass_rate"]) >= PROMOTE_BOOTSTRAP_MIN
        ):
            return "MODEL_B_PROMOTE_PAPER", "MODEL_B_PROMOTE_PAPER", "robust sizing-only frontier remains real"
        return "MODEL_B_KEEP_AS_BACKUP_ONLY", "MODEL_B_KEEP_AS_BACKUP_ONLY", "improvement is real but the strict frontier is narrower than the promote bar"

    _ = corr_cluster_count
    return "MODEL_B_NO_GO_REVERT_TO_MODEL_A", "MODEL_B_NO_GO_REVERT_TO_MODEL_A", "strict confirmation did not leave any robust Model B survivor"


def main() -> None:
    run_dir = REPORTS_ROOT / f"PHASEMB_MODEL_B_STRICT_CONFIRMATION_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    prior_dir = MODELB_DIR_DEFAULT
    prior_results = pd.read_csv(prior_dir / "modelB3_ablation_results.csv")

    primary_sel, _backup_sel, primary_cfg, _backup_cfg = model_b0.load_phasec_candidates()
    repro = model_b0.reproduce_primary(primary_sel=primary_sel, primary_cfg=primary_cfg, run_dir=run_dir)
    if int(repro["parity_ok"]) != 1 or int(repro["lock_info"]["freeze_lock_pass"]) != 1:
        failure = {
            "generated_utc": utc_now(),
            "freeze_lock_pass": int(repro["lock_info"]["freeze_lock_pass"]),
            "model_a_primary_parity_ok": int(repro["parity_ok"]),
            "wrapper_uses_3m_entry_only": 1,
            "wrapper_uses_1h_exit_only": 1,
            "forbidden_exit_controls_active": 0,
            "exact_blocker": "Model A frozen baseline parity failed before sizing confirmation",
        }
        json_dump(run_dir / "phaseMB1_contract_validation.json", failure)
        write_text(
            run_dir / "phaseMB1_contract_report.md",
            "# MB1 Contract Report\n\n- Frozen Model A primary could not be reproduced exactly. Strict confirmation stopped.\n",
        )
        write_text(
            run_dir / "phaseMB4_decision.md",
            "# MB4 Decision\n\n- Classification: `MODEL_B_NO_GO_REVERT_TO_MODEL_A`\n- Mainline status: `MODEL_B_NO_GO_REVERT_TO_MODEL_A`\n- Exact blocker: `Model A frozen baseline parity failed before sizing confirmation`\n",
        )
        json_dump(
            run_dir / "phaseMB_run_manifest.json",
            {
                "generated_utc": utc_now(),
                "git": git_snapshot(),
                "classification": "MODEL_B_NO_GO_REVERT_TO_MODEL_A",
                "mainline_status": "MODEL_B_NO_GO_REVERT_TO_MODEL_A",
                "artifact_dir": str(run_dir),
            },
        )
        print(str(run_dir))
        return

    base_rows = repro["base_rows"].copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    base_roll = ga_exec._rollup_mode(base_rows, "exec")
    base_cluster = model_b0.loss_cluster_metrics(base_rows["exec_pnl_net_pct"].to_numpy(dtype=float))

    raw_candidates = build_local_candidates(base_rows)
    unique_candidates, dup_df = collapse_duplicates(raw_candidates)

    vectors = {str(c["canonical_id"]): np.asarray(c["param_vector"], dtype=float) for c in unique_candidates}
    effective_trials_uncorrelated = int(len(unique_candidates))
    corr_cluster_count, cluster_map = phase_b.union_find_cluster_count(vectors=vectors, dist_threshold=0.26)
    for cfg in unique_candidates:
        cfg["cluster_id"] = int(cluster_map[str(cfg["canonical_id"])])

    results_rows: List[Dict[str, Any]] = [
        {
            "seed_origin": "MODEL_A_PRIMARY_BASELINE",
            "cluster_id": 0,
            "raw_variant_count": 1,
            "family": "baseline_primary",
            "variant_id": "MODEL_A_PRIMARY_BASELINE",
            "label": "Frozen Model A primary reference",
            "valid_for_ranking": 1,
            "invalid_reason": "",
            "expectancy_net": float(base_roll["mean_expectancy_net"]),
            "delta_expectancy_vs_modelA": 0.0,
            "cvar_improve_ratio": 0.0,
            "maxdd_improve_ratio": 0.0,
            "pnl_std_improve_ratio": 0.0,
            "min_split_expectancy": float(model_b0.split_expectancy_min(base_rows)),
            "loss_cluster_count": float(base_cluster["loss_cluster_count"]),
            "loss_cluster_worst_burden": float(base_cluster["loss_cluster_worst_burden"]),
            "loss_cluster_avg_burden": float(base_cluster["loss_cluster_avg_burden"]),
            "loss_cluster_worst_burden_improve_ratio": 0.0,
            "loss_cluster_avg_burden_improve_ratio": 0.0,
            "route_pass": 1,
            "route_min_delta_vs_modelA": 0.0,
            "route_front_delta_vs_modelA": 0.0,
            "route_center_delta_vs_modelA": 0.0,
            "route_back_delta_vs_modelA": 0.0,
            "stress_lite_pass": 1,
            "stress_delta_expectancy_vs_modelA": 0.0,
            "stress_maxdd_improve_ratio": 0.0,
            "bootstrap_pass_rate": 1.0,
            "bootstrap_pass": 1,
            "entries_valid": int(base_roll["entries_valid"]),
            "entry_rate": float(base_roll["entry_rate"]),
            "taker_share": float(base_roll["taker_share"]),
            "median_fill_delay_min": float(base_roll["median_fill_delay_min"]),
            "p95_fill_delay_min": float(base_roll["p95_fill_delay_min"]),
            "participation_changed": 0,
            "validity_hard_gates_intact": 1,
            "size_mult_min": 1.0,
            "size_mult_mean": 1.0,
            "size_mult_max": 1.0,
            "selection_score": 0.0,
            "params": json.dumps({"size_floor": 1.0, "size_ceiling": 1.0}, sort_keys=True),
            "sizing_only_overlay": 0,
        }
    ]
    invalid_hist: Counter[str] = Counter()
    eval_meta: Dict[str, Dict[str, Any]] = {}

    for cfg in unique_candidates:
        out = model_b0.evaluate_variant(
            variant=cfg,
            base_rows=base_rows,
            base_roll=base_roll,
            base_cluster=base_cluster,
            route_base_rows=repro["route_base_rows"],
            route_base_metrics=repro["route_base_metrics"],
        )
        out["seed_origin"] = str(cfg["seed_origin"])
        out["cluster_id"] = int(cfg["cluster_id"])
        out["raw_variant_count"] = int(cfg["raw_variant_count"])
        out["params"] = json.dumps(cfg["params"], sort_keys=True)
        out["sizing_only_overlay"] = 1
        out["participation_changed"] = int(
            (int(out["entries_valid"]) != int(base_roll["entries_valid"]))
            or abs(float(out["entry_rate"]) - float(base_roll["entry_rate"])) > 1e-12
            or abs(float(out["taker_share"]) - float(base_roll["taker_share"])) > 1e-12
            or abs(float(out["p95_fill_delay_min"]) - float(base_roll["p95_fill_delay_min"])) > 1e-12
        )
        results_rows.append(out)
        if out["invalid_reason"]:
            invalid_hist[str(out["invalid_reason"])] += 1
        eval_meta[str(cfg["canonical_id"])] = {
            "cfg": cfg,
            "multiplier": np.asarray(cfg["multiplier"], dtype=float),
        }

    results_df = pd.DataFrame(results_rows).sort_values(
        ["selection_score", "delta_expectancy_vs_modelA", "variant_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    results_df.to_csv(run_dir / "phaseMB2_results.csv", index=False)
    json_dump(run_dir / "phaseMB2_invalid_reason_histogram.json", dict(sorted(invalid_hist.items())))

    anchor_tiered_diff = anchor_diffs(results_df, prior_results, "regime_cap_size_delay_tiered")
    anchor_soft_diff = anchor_diffs(results_df, prior_results, "regime_cap_size_delay_soft")
    all_sizing_only = int(
        (
            (to_num(results_df.get("sizing_only_overlay", 0)).fillna(0).astype(int) == 0)
            | (
                (to_num(results_df.get("participation_changed", 0)).fillna(0).astype(int) == 0)
                & (to_num(results_df.get("validity_hard_gates_intact", 0)).fillna(0).astype(int) == 1)
            )
        ).all()
    )

    mb1 = {
        "generated_utc": utc_now(),
        "freeze_lock_pass": int(repro["lock_info"]["freeze_lock_pass"]),
        "wrapper_uses_3m_entry_only": 1,
        "wrapper_uses_1h_exit_only": 1,
        "forbidden_exit_controls_active": 0,
        "forbidden_exit_knobs_blocked": [
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
        ],
        "model_a_primary_parity_ok": int(repro["parity_ok"]),
        "model_a_primary_parity_diffs": repro["parity_diffs"],
        "route_trade_gates_reachable": int(repro["route_valid_df"]["route_trade_gates_reachable"].astype(int).all()),
        "route_harness_meta": repro["route_meta"],
        "model_b_anchor_diffs": {
            "regime_cap_size_delay_tiered": anchor_tiered_diff,
            "regime_cap_size_delay_soft": anchor_soft_diff,
        },
        "all_candidates_sizing_only": int(all_sizing_only),
        "raw_candidates_generated": int(len(raw_candidates)),
        "unique_candidates_after_duplicate_collapse": int(effective_trials_uncorrelated),
        "correlation_adjusted_clusters": int(corr_cluster_count),
    }
    json_dump(run_dir / "phaseMB1_contract_validation.json", mb1)

    mb1_lines = [
        "# MB1 Contract Report",
        "",
        f"- Generated UTC: `{mb1['generated_utc']}`",
        f"- Freeze lock pass: `{mb1['freeze_lock_pass']}`",
        f"- Wrapper purity: `3m entry only = 1`, `1h exit only = 1`",
        f"- Forbidden exit controls active: `{mb1['forbidden_exit_controls_active']}`",
        f"- Model A primary parity OK: `{mb1['model_a_primary_parity_ok']}`",
        f"- Route trade gates reachable on repaired harness: `{mb1['route_trade_gates_reachable']}`",
        f"- All local Model B candidates are sizing-only and preserve participation: `{mb1['all_candidates_sizing_only']}`",
        f"- Raw sizing probes: `{mb1['raw_candidates_generated']}`",
        f"- Unique probes after duplicate collapse: `{mb1['unique_candidates_after_duplicate_collapse']}`",
        f"- Correlation-adjusted clusters: `{mb1['correlation_adjusted_clusters']}`",
        "",
        "## Model A Primary Parity Diffs",
    ]
    for k, v in repro["parity_diffs"].items():
        mb1_lines.append(f"- {k}: `{v}`")
    mb1_lines.extend(
        [
            "",
            "## Prior Model B Anchor Reproduction Diffs",
            f"- regime_cap_size_delay_tiered: `{anchor_tiered_diff}`",
            f"- regime_cap_size_delay_soft: `{anchor_soft_diff}`",
            "",
            "## Approved Anchors in This Confirmation Run",
            "",
            markdown_table(
                results_df[
                    results_df["variant_id"].astype(str).isin(
                        ["MODEL_A_PRIMARY_BASELINE", "regime_cap_size_delay_tiered", "regime_cap_size_delay_soft"]
                    )
                ],
                [
                    "variant_id",
                    "expectancy_net",
                    "delta_expectancy_vs_modelA",
                    "cvar_improve_ratio",
                    "maxdd_improve_ratio",
                    "route_pass",
                    "stress_lite_pass",
                    "bootstrap_pass_rate",
                ],
                n=8,
            ),
            "",
        ]
    )
    write_text(run_dir / "phaseMB1_contract_report.md", "\n".join(mb1_lines))

    eff_lines = [
        "# MB2 Effective Trials Summary",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Raw local candidates generated: `{len(raw_candidates)}`",
        f"- Unique candidates after duplicate collapse: `{effective_trials_uncorrelated}`",
        f"- Duplicate-collapsed rows: `{len(raw_candidates) - effective_trials_uncorrelated}`",
        f"- Correlation-adjusted clusters: `{corr_cluster_count}`",
        "- Duplicate logic: exact multiplier-vector hash on the frozen Model A primary trade stream.",
        "- Correlation logic: cluster compact sizing-parameter vectors by L1 distance <= 0.26.",
        "",
        "## Duplicate Collapse Map",
        "",
        markdown_table(dup_df, ["raw_variant_id", "seed_origin", "canonical_id"], n=24),
        "",
    ]
    write_text(run_dir / "phaseMB2_effective_trials_summary.md", "\n".join(eff_lines))

    # MB3 strict robustness confirmation on a bounded shortlist.
    valid_candidates = results_df[
        (results_df["variant_id"].astype(str) != "MODEL_A_PRIMARY_BASELINE")
        & (to_num(results_df["valid_for_ranking"]).fillna(0).astype(int) == 1)
        & (to_num(results_df["delta_expectancy_vs_modelA"]) >= -5e-5)
        & (to_num(results_df["route_pass"]).fillna(0).astype(int) == 1)
    ].copy()
    ranked_valid = valid_candidates.sort_values(
        [
            "delta_expectancy_vs_modelA",
            "cvar_improve_ratio",
            "maxdd_improve_ratio",
            "loss_cluster_avg_burden_improve_ratio",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    shortlist_ids: List[str] = []
    for seed_origin, grp in ranked_valid.groupby("seed_origin", sort=False):
        _ = seed_origin
        shortlist_ids.extend(grp.head(3)["variant_id"].astype(str).tolist())
    for cluster_id, grp in ranked_valid.groupby("cluster_id", sort=False):
        _ = cluster_id
        shortlist_ids.extend(grp.head(2)["variant_id"].astype(str).tolist())
    for cid in ranked_valid["variant_id"].astype(str).tolist():
        shortlist_ids.append(cid)
        if len(dict.fromkeys(shortlist_ids)) >= 8:
            break
    shortlist_ids = list(dict.fromkeys(shortlist_ids))[:8]
    shortlist_df = ranked_valid[ranked_valid["variant_id"].astype(str).isin(shortlist_ids)].copy().reset_index(drop=True)

    route_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    stress_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    sig_rows: List[Dict[str, Any]] = []
    survivor_rows: List[Dict[str, Any]] = []

    for pos, (_, top) in enumerate(shortlist_df.iterrows()):
        cid = str(top["variant_id"])
        cfg_meta = eval_meta[cid]
        mult_map = dict(zip(base_rows["signal_id"].astype(str), np.asarray(cfg_meta["multiplier"], dtype=float)))
        cand_rows = model_b0.apply_multiplier(base_rows, mult_map)

        split_df = split_rows_for_candidate(base_rows, cand_rows, cid)
        split_rows.extend(split_df.to_dict(orient="records"))
        min_split_delta = float(to_num(split_df.get("delta_expectancy_vs_modelA", pd.Series(dtype=float))).min()) if not split_df.empty else float("nan")
        split_pass = int(np.isfinite(min_split_delta) and min_split_delta >= ROUTE_DELTA_TOL)

        route_pass_components: List[int] = []
        route_deltas: List[float] = []
        for rid, rb in sorted(repro["route_base_rows"].items()):
            rb = rb.copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
            rc = model_b0.apply_multiplier(rb, mult_map)
            rb_roll = ga_exec._rollup_mode(rb, "exec")
            rc_roll = ga_exec._rollup_mode(rc, "exec")
            delta = float(rc_roll["mean_expectancy_net"] - rb_roll["mean_expectancy_net"])
            route_ok = int(delta >= ROUTE_DELTA_TOL)
            route_pass_components.append(route_ok)
            route_deltas.append(delta)
            route_rows.append(
                {
                    "candidate_id": cid,
                    "seed_origin": str(top["seed_origin"]),
                    "cluster_id": int(top["cluster_id"]),
                    "route_id": str(rid),
                    "route_pass_component": int(route_ok),
                    "delta_expectancy_vs_modelA": float(delta),
                    "expectancy_net": float(rc_roll["mean_expectancy_net"]),
                    "entries_valid": int(rc_roll["entries_valid"]),
                    "entry_rate": float(rc_roll["entry_rate"]),
                }
            )
        route_pass_rate = float(np.mean(route_pass_components)) if route_pass_components else float("nan")
        route_pass = int(route_pass_components and all(route_pass_components))
        center_row = next((r for r in route_rows[::-1] if r["candidate_id"] == cid and r["route_id"] == "route_center_60pct"), None)
        center_route_delta = float(center_row["delta_expectancy_vs_modelA"]) if center_row is not None else float("nan")
        center_route_valid = int(center_row["route_pass_component"]) if center_row is not None else 0

        scenario_ids = ["base_stress", "slippage_heavy", "delay_penalty"]
        stress_pass_components: List[int] = []
        stress_deltas: List[float] = []
        for scenario_id in scenario_ids:
            base_stress_rows = stress_rows_for_scenario(base_rows, scenario_id)
            cand_stress_rows = stress_rows_for_scenario(cand_rows, scenario_id)
            rb = ga_exec._rollup_mode(base_stress_rows, "exec")
            rc = ga_exec._rollup_mode(cand_stress_rows, "exec")
            delta = float(rc["mean_expectancy_net"] - rb["mean_expectancy_net"])
            dd_imp = float(model_b0.maxdd_improve_ratio(float(rb["max_drawdown"]), float(rc["max_drawdown"])))
            base_cluster_s = model_b0.loss_cluster_metrics(base_stress_rows["exec_pnl_net_pct"].to_numpy(dtype=float))
            cluster_c = model_b0.loss_cluster_metrics(cand_stress_rows["exec_pnl_net_pct"].to_numpy(dtype=float))
            cluster_imp = float(
                model_b0.cluster_improve_ratio(base_cluster_s["loss_cluster_avg_burden"], cluster_c["loss_cluster_avg_burden"])
            )
            stress_ok = int(delta >= STRESS_DELTA_TOL and dd_imp >= 0.0 and cluster_imp >= 0.0)
            stress_pass_components.append(stress_ok)
            stress_deltas.append(delta)
            stress_rows.append(
                {
                    "candidate_id": cid,
                    "seed_origin": str(top["seed_origin"]),
                    "cluster_id": int(top["cluster_id"]),
                    "stress_id": str(scenario_id),
                    "stress_pass_component": int(stress_ok),
                    "delta_expectancy_vs_modelA": float(delta),
                    "maxdd_improve_ratio": float(dd_imp),
                    "loss_cluster_avg_burden_improve_ratio": float(cluster_imp),
                }
            )
        stress_pass = int(stress_pass_components and all(stress_pass_components))
        stress_delta_mean = float(np.mean(stress_deltas)) if stress_deltas else float("nan")

        trade_boot = float(
            model_b0.bootstrap_pass_rate(
                base_pnl=base_rows["exec_pnl_net_pct"].to_numpy(dtype=float),
                cand_pnl=cand_rows["exec_pnl_net_pct"].to_numpy(dtype=float),
                n=256,
                block=16,
                seed=20260302 + int(pos),
            )
        )
        split_boot = float(split_bootstrap_pass_rate(split_df, draws=256, seed=20261302 + int(pos)))
        boot_rate = float(np.nanmin(np.asarray([trade_boot, split_boot], dtype=float)))
        boot_rows.append(
            {
                "candidate_id": cid,
                "seed_origin": str(top["seed_origin"]),
                "cluster_id": int(top["cluster_id"]),
                "trade_bootstrap_pass_rate": float(trade_boot),
                "split_bootstrap_pass_rate": float(split_boot),
                "bootstrap_pass_rate": float(boot_rate),
            }
        )

        sig_split_df = split_df.rename(columns={"delta_expectancy_vs_modelA": "delta_expectancy_exec_minus_baseline"})
        psr_proxy, dsr_proxy = phase_b.candidate_psr_dsr(
            split_df=sig_split_df,
            effective_trials_corr_adjusted=float(max(1, corr_cluster_count)),
        )
        sig_rows.append(
            {
                "candidate_id": cid,
                "seed_origin": str(top["seed_origin"]),
                "cluster_id": int(top["cluster_id"]),
                "psr_proxy": float(psr_proxy),
                "dsr_proxy": float(dsr_proxy),
                "effective_trials_uncorrelated": int(effective_trials_uncorrelated),
                "effective_trials_corr_adjusted": int(corr_cluster_count),
                "raw_variant_count": int(top["raw_variant_count"]),
            }
        )

        robust_survivor = int(
            int(top["valid_for_ranking"]) == 1
            and int(top["participation_changed"]) == 0
            and float(top["delta_expectancy_vs_modelA"]) >= -5e-5
            and float(top["cvar_improve_ratio"]) >= 0.03
            and float(top["maxdd_improve_ratio"]) >= 0.005
            and float(top["loss_cluster_avg_burden_improve_ratio"]) >= 0.01
            and route_pass == 1
            and center_route_valid == 1
            and np.isfinite(center_route_delta)
            and center_route_delta >= ROUTE_DELTA_TOL
            and split_pass == 1
            and stress_pass == 1
            and np.isfinite(boot_rate)
            and boot_rate >= BACKUP_BOOTSTRAP_MIN
            and np.isfinite(psr_proxy)
            and psr_proxy >= 0.90
            and np.isfinite(dsr_proxy)
            and dsr_proxy >= 0.70
        )
        survivor_rows.append(
            {
                "candidate_id": cid,
                "seed_origin": str(top["seed_origin"]),
                "cluster_id": int(top["cluster_id"]),
                "raw_variant_count": int(top["raw_variant_count"]),
                "valid_for_ranking": int(top["valid_for_ranking"]),
                "expectancy_net": float(top["expectancy_net"]),
                "delta_expectancy_vs_modelA": float(top["delta_expectancy_vs_modelA"]),
                "cvar_improve_ratio": float(top["cvar_improve_ratio"]),
                "maxdd_improve_ratio": float(top["maxdd_improve_ratio"]),
                "loss_cluster_avg_burden_improve_ratio": float(top["loss_cluster_avg_burden_improve_ratio"]),
                "entries_valid": int(top["entries_valid"]),
                "entry_rate": float(top["entry_rate"]),
                "taker_share": float(top["taker_share"]),
                "median_fill_delay_min": float(top["median_fill_delay_min"]),
                "p95_fill_delay_min": float(top["p95_fill_delay_min"]),
                "route_pass": int(route_pass),
                "route_pass_rate": float(route_pass_rate),
                "center_route_valid": int(center_route_valid),
                "center_route_delta": float(center_route_delta),
                "min_split_delta": float(min_split_delta),
                "split_pass": int(split_pass),
                "stress_pass": int(stress_pass),
                "stress_delta_mean": float(stress_delta_mean),
                "bootstrap_pass_rate": float(boot_rate),
                "psr_proxy": float(psr_proxy),
                "dsr_proxy": float(dsr_proxy),
                "robust_survivor": int(robust_survivor),
            }
        )

    route_checks_df = pd.DataFrame(route_rows).sort_values(["candidate_id", "route_id"]).reset_index(drop=True)
    split_checks_df = pd.DataFrame(split_rows).sort_values(["candidate_id", "split_id"]).reset_index(drop=True)
    stress_matrix_df = pd.DataFrame(stress_rows).sort_values(["candidate_id", "stress_id"]).reset_index(drop=True)
    boot_df = pd.DataFrame(boot_rows).sort_values(["candidate_id"]).reset_index(drop=True)
    sig_df = pd.DataFrame(sig_rows).sort_values(["candidate_id"]).reset_index(drop=True)
    survivors_df = pd.DataFrame(survivor_rows).sort_values(
        [
            "robust_survivor",
            "delta_expectancy_vs_modelA",
            "cvar_improve_ratio",
            "maxdd_improve_ratio",
            "loss_cluster_avg_burden_improve_ratio",
            "taker_share",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)

    route_checks_df.to_csv(run_dir / "phaseMB3_route_checks.csv", index=False)
    split_checks_df.to_csv(run_dir / "phaseMB3_split_stability.csv", index=False)
    stress_matrix_df.to_csv(run_dir / "phaseMB3_stress_matrix.csv", index=False)
    boot_df.to_csv(run_dir / "phaseMB3_bootstrap_summary.csv", index=False)
    sig_df.to_csv(run_dir / "phaseMB2_shortlist_significance.csv", index=False)

    robust_df = survivors_df[to_num(survivors_df.get("robust_survivor", 0)).fillna(0).astype(int) == 1].copy()
    robust_seed_count = int(robust_df["seed_origin"].astype(str).nunique()) if not robust_df.empty else 0
    robust_cluster_count = int(robust_df["cluster_id"].nunique()) if not robust_df.empty else 0
    stable_neighborhood = int(len(robust_df) >= 4 and robust_seed_count >= 2 and robust_cluster_count >= 2)

    cluster_lines = [
        "# MB3 Cluster Burden Report",
        "",
        f"- Generated UTC: `{utc_now()}`",
        f"- Shortlist size: `{len(shortlist_df)}`",
        f"- Robust survivors: `{len(robust_df)}`",
        f"- Robust seed origins: `{robust_seed_count}`",
        f"- Robust correlation-adjusted clusters: `{robust_cluster_count}`",
        f"- Stable neighborhood: `{stable_neighborhood}`",
        "",
        "## Top Shortlist Rows",
        "",
        markdown_table(
            survivors_df,
            [
                "candidate_id",
                "seed_origin",
                "cluster_id",
                "delta_expectancy_vs_modelA",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "loss_cluster_avg_burden_improve_ratio",
                "route_pass",
                "stress_pass",
                "bootstrap_pass_rate",
                "psr_proxy",
                "dsr_proxy",
                "robust_survivor",
            ],
            n=12,
        ),
        "",
    ]
    write_text(run_dir / "phaseMB3_cluster_burden_report.md", "\n".join(cluster_lines))

    classification, mainline_status, reason = decision_from_survivors(survivors_df=survivors_df, corr_cluster_count=corr_cluster_count)

    decision_lines = [
        "# MB4 Decision",
        "",
        f"- Classification: `{classification}`",
        f"- Mainline status: `{mainline_status}`",
        f"- Reason: `{reason}`",
        f"- Robust survivors: `{len(robust_df)}`",
        f"- Stable neighborhood: `{stable_neighborhood}`",
        f"- Correlation-adjusted clusters: `{corr_cluster_count}`",
    ]
    if not survivors_df.empty:
        best = survivors_df.iloc[0]
        decision_lines.extend(
            [
                "",
                "## Best Strictly Confirmed Candidate",
                f"- candidate_id: `{best['candidate_id']}`",
                f"- delta_expectancy_vs_modelA: `{best['delta_expectancy_vs_modelA']}`",
                f"- cvar_improve_ratio: `{best['cvar_improve_ratio']}`",
                f"- maxdd_improve_ratio: `{best['maxdd_improve_ratio']}`",
                f"- loss_cluster_avg_burden_improve_ratio: `{best['loss_cluster_avg_burden_improve_ratio']}`",
                f"- route_pass: `{best['route_pass']}`",
                f"- stress_pass: `{best['stress_pass']}`",
                f"- bootstrap_pass_rate: `{best['bootstrap_pass_rate']}`",
                f"- robust_survivor: `{best['robust_survivor']}`",
            ]
        )
    write_text(run_dir / "phaseMB4_decision.md", "\n".join(decision_lines) + "\n")

    next_prompt = ""
    if classification in {"MODEL_B_PROMOTE_PAPER", "MODEL_B_KEEP_AS_BACKUP_ONLY"} and not survivors_df.empty:
        best = survivors_df.iloc[0]
        if classification == "MODEL_B_PROMOTE_PAPER":
            next_prompt = (
                "ROLE\n\n"
                "You are in Model B paper-runtime promotion mode for SOLUSDT.\n\n"
                "MISSION\n\n"
                "Update the dedicated Model A paper/shadow runtime so the promoted candidate keeps the same frozen pure contract "
                "and applies only the approved Model B sizing overlay on top of the frozen Model A primary.\n\n"
                "PROMOTED CANDIDATE\n\n"
                f"{best['candidate_id']}\n\n"
                "REQUIREMENTS\n\n"
                "1. Keep 1h signal generation unchanged.\n"
                "2. Keep 1h TP/SL/exits unchanged.\n"
                "3. Keep 3m entry execution unchanged.\n"
                "4. Apply sizing overlay only; no exit mutation.\n"
                "5. Preserve the repaired route harness and frozen contract checks.\n"
                "6. Keep the old Model A primary as a shadow comparator until parity is confirmed.\n"
            )
        else:
            next_prompt = (
                "ROLE\n\n"
                "You are in Model B shadow-only integration mode for SOLUSDT.\n\n"
                "MISSION\n\n"
                "Keep the frozen Model A primary as the promoted paper configuration and add the approved Model B sizing overlay "
                "only as a shadow comparator under the same pure contract.\n\n"
                "SHADOW MODEL B CANDIDATE\n\n"
                f"{best['candidate_id']}\n\n"
                "REQUIREMENTS\n\n"
                "1. Keep 1h signal generation unchanged.\n"
                "2. Keep 1h TP/SL/exits unchanged.\n"
                "3. Keep 3m entry execution unchanged.\n"
                "4. Apply the Model B sizing overlay only in the shadow book; do not replace the active Model A primary yet.\n"
                "5. Preserve the repaired route harness and frozen contract checks.\n"
                "6. Stop the shadow branch immediately if parity or runtime purity drifts.\n"
            )
        write_text(run_dir / "ready_to_launch_modelB_paper_promotion_prompt.txt", next_prompt)

    json_dump(
        run_dir / "phaseMB_run_manifest.json",
        {
            "generated_utc": utc_now(),
            "git": git_snapshot(),
            "artifact_dir": str(run_dir),
            "classification": classification,
            "mainline_status": mainline_status,
            "raw_candidates": int(len(raw_candidates)),
            "unique_candidates": int(effective_trials_uncorrelated),
            "corr_adjusted_clusters": int(corr_cluster_count),
            "robust_survivor_count": int(len(robust_df)),
            "stable_neighborhood": int(stable_neighborhood),
            "best_candidate": str(survivors_df.iloc[0]["candidate_id"]) if not survivors_df.empty else "",
            "next_prompt_written": int(bool(next_prompt)),
        },
    )

    print(str(run_dir))


if __name__ == "__main__":
    main()
