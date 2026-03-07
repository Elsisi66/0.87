#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import backtest_exec_phasec_sol as phasec_bt  # noqa: E402
from scripts import phase_a_model_a_audit as phase_a  # noqa: E402
from scripts import phase_nx_exec_family_discovery as nx  # noqa: E402
from scripts import phase_r_route_harness_redesign as phase_r  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


REPORTS_ROOT = PROJECT_ROOT / "reports" / "execution_layer"
PHASEC_DIR = (REPORTS_ROOT / "PHASEC_MODEL_A_BOUNDED_CONFIRMATION_20260228_022501").resolve()
PHASER_DIR = (REPORTS_ROOT / "PHASER_ROUTE_HARNESS_REDESIGN_20260228_005334").resolve()

ROUTE_DELTA_TOL = -5e-5
STRESS_DELTA_TOL = -5e-5
BOOTSTRAP_PASS_MIN_GO = 0.60
BOOTSTRAP_PASS_MIN_WEAK = 0.50


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


def markdown_table(df: pd.DataFrame, cols: List[str], n: int = 12) -> str:
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


def load_phasec_candidates() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    selection = pd.read_csv(PHASEC_DIR / "phaseC4_primary_backup.csv")
    results = pd.read_csv(PHASEC_DIR / "phaseC2_results.csv")

    primary_sel = selection[selection["selection_role"].astype(str).str.lower() == "primary"].iloc[0]
    backup_sel = selection[selection["selection_role"].astype(str).str.lower() == "backup"].iloc[0]

    primary_cfg = results[results["candidate_id"].astype(str) == str(primary_sel["candidate_id"])].iloc[0]
    backup_cfg = results[results["candidate_id"].astype(str) == str(backup_sel["candidate_id"])].iloc[0]
    return primary_sel, backup_sel, primary_cfg, backup_cfg


def model_a_cfg_from_row(row: pd.Series) -> Dict[str, Any]:
    return {
        "candidate_id": str(row["candidate_id"]),
        "label": str(row["candidate_id"]),
        "entry_mode": str(row["entry_mode"]),
        "limit_offset_bps": float(row["limit_offset_bps"]),
        "fallback_to_market": int(row["fallback_to_market"]),
        "fallback_delay_min": float(row["fallback_delay_min"]),
        "max_fill_delay_min": float(row["max_fill_delay_min"]),
    }


def reproduce_primary(primary_sel: pd.Series, primary_cfg: pd.Series, run_dir: Path) -> Dict[str, Any]:
    subset_path = Path(nx.LOCKED["representative_subset_csv"]).resolve()
    args = nx.build_exec_args(signals_csv=subset_path, seed=20260228)
    lock_info = ga_exec._validate_and_lock_frozen_artifacts(args=args, run_dir=run_dir)  # pylint: disable=protected-access
    if int(lock_info.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("frozen contract validation failed")

    bundles, _load_meta = ga_exec._prepare_bundles(args)  # pylint: disable=protected-access
    if not bundles:
        raise RuntimeError("no bundles prepared")

    base_bundle = bundles[0]
    one_h = phase_a.load_1h_market(base_bundle.symbol)
    fee = phasec_bt.FeeModel(
        fee_bps_maker=float(args.fee_bps_maker),
        fee_bps_taker=float(args.fee_bps_taker),
        slippage_bps_limit=float(args.slippage_bps_limit),
        slippage_bps_market=float(args.slippage_bps_market),
    )
    base_ref = phase_a.build_1h_reference_rows(
        bundle=base_bundle,
        fee=fee,
        exec_horizon_hours=float(args.exec_horizon_hours),
    )
    cfg = model_a_cfg_from_row(primary_cfg)
    base_eval = phase_a.evaluate_model_a_variant(
        bundle=base_bundle,
        baseline_df=base_ref,
        cfg=cfg,
        one_h=one_h,
        args=args,
    )

    route_bundles, route_examples_df, route_valid_df, route_meta = phase_r.build_support_feasible_route_family(
        base_bundle=base_bundle,
        args=args,
        coverage_frac=0.60,
    )
    route_base_rows: Dict[str, pd.DataFrame] = {}
    route_base_metrics: Dict[str, Dict[str, float]] = {}
    for rid, rb in route_bundles.items():
        route_ref = phase_a.build_1h_reference_rows(
            bundle=rb,
            fee=fee,
            exec_horizon_hours=float(args.exec_horizon_hours),
        )
        rev = phase_a.evaluate_model_a_variant(
            bundle=rb,
            baseline_df=route_ref,
            cfg=cfg,
            one_h=one_h,
            args=args,
        )
        route_base_rows[rid] = rev["signal_rows_df"].copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
        route_roll = ga_exec._rollup_mode(route_base_rows[rid], "exec")
        route_base_metrics[rid] = {
            "expectancy_net": float(route_roll["mean_expectancy_net"]),
            "entry_rate": float(route_roll["entry_rate"]),
            "entries_valid": int(route_roll["entries_valid"]),
        }

    # Exact parity check against the approved Phase C primary row.
    exp_map = {
        "overall_exec_expectancy_net": float(primary_sel["exec_expectancy_net"]),
        "overall_delta_expectancy_exec_minus_baseline": float(primary_sel["delta_expectancy_vs_1h_reference"]),
        "overall_cvar_improve_ratio": float(primary_sel["cvar_improve_ratio"]),
        "overall_maxdd_improve_ratio": float(primary_sel["maxdd_improve_ratio"]),
        "overall_entry_rate": float(primary_sel["entry_rate"]),
        "overall_entries_valid": int(primary_sel["entries_valid"]),
        "overall_exec_taker_share": float(primary_sel["taker_share"]),
        "overall_exec_p95_fill_delay_min": float(primary_sel["p95_fill_delay_min"]),
        "min_split_delta": float(primary_sel["min_subperiod_delta"]),
    }
    obs = base_eval["metrics"]
    diffs = {}
    parity_ok = 1
    tol = 1e-12
    for k, vexp in exp_map.items():
        vobs = obs[k]
        diff = abs(float(vobs) - float(vexp))
        diffs[k] = float(diff)
        if diff > tol:
            parity_ok = 0

    return {
        "args": args,
        "lock_info": lock_info,
        "bundle": base_bundle,
        "one_h": one_h,
        "fee": fee,
        "base_eval": base_eval,
        "base_rows": base_eval["signal_rows_df"].copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True),
        "base_metrics": copy.deepcopy(base_eval["metrics"]),
        "route_bundles": route_bundles,
        "route_examples_df": route_examples_df,
        "route_valid_df": route_valid_df,
        "route_meta": route_meta,
        "route_base_rows": route_base_rows,
        "route_base_metrics": route_base_metrics,
        "primary_cfg": cfg,
        "parity_ok": int(parity_ok),
        "parity_diffs": diffs,
    }


def build_sizing_variants(base_rows: pd.DataFrame) -> List[Dict[str, Any]]:
    x = base_rows.copy().reset_index(drop=True)
    delay = to_num(x["exec_fill_delay_min"]).fillna(0.0).to_numpy(dtype=float)
    pnl = to_num(x["exec_pnl_net_pct"]).fillna(0.0).to_numpy(dtype=float)

    variants: List[Dict[str, Any]] = []

    def add_variant(
        family: str,
        variant_id: str,
        label: str,
        multiplier: np.ndarray,
        params: Dict[str, Any],
    ) -> None:
        variants.append(
            {
                "family": family,
                "variant_id": variant_id,
                "label": label,
                "params": params,
                "multiplier": multiplier.astype(float),
            }
        )

    add_variant(
        "baseline_primary",
        "MODEL_A_PRIMARY_BASELINE",
        "Frozen Model A primary reference",
        np.ones(len(x), dtype=float),
        {"size_floor": 1.0, "size_ceiling": 1.0},
    )

    add_variant(
        "fixed_half_on_risk",
        "fixed_half_on_risk_50",
        "Constant 0.50x size",
        np.full(len(x), 0.50, dtype=float),
        {"constant_size_mult": 0.50},
    )
    add_variant(
        "fixed_half_on_risk",
        "fixed_half_on_risk_70",
        "Constant 0.70x size",
        np.full(len(x), 0.70, dtype=float),
        {"constant_size_mult": 0.70},
    )

    for variant_id, down_mult in [("step_down_after_streak_soft", 0.75), ("step_down_after_streak_hard", 0.50)]:
        mult = []
        loss_streak = 0
        for raw_pnl in pnl:
            cur_mult = down_mult if loss_streak >= 2 else 1.0
            mult.append(cur_mult)
            if raw_pnl < 0:
                loss_streak += 1
            elif raw_pnl > 0:
                loss_streak = 0
        add_variant(
            "step_down_after_streak",
            variant_id,
            f"Size down to {down_mult:.2f}x after 2 losses",
            np.asarray(mult, dtype=float),
            {"trigger_loss_streak": 2, "down_mult": down_mult, "recovery": "next_win_resets"},
        )

    for variant_id, floor_mult, coef in [
        ("linear_risk_score_scale_soft", 0.75, 0.25),
        ("linear_risk_score_scale_hard", 0.60, 0.40),
    ]:
        mult = []
        scaled_hist: List[float] = []
        for raw_pnl in pnl:
            recent = scaled_hist[-8:]
            loss_ratio = (sum(1 for z in recent if z < 0.0) / len(recent)) if recent else 0.0
            cur_mult = max(floor_mult, 1.0 - coef * loss_ratio)
            mult.append(cur_mult)
            scaled_hist.append(float(raw_pnl * cur_mult))
        add_variant(
            "linear_risk_score_scale",
            variant_id,
            f"Linear scale from 1.0x to floor {floor_mult:.2f}x from rolling loss ratio",
            np.asarray(mult, dtype=float),
            {"rolling_window": 8, "floor_mult": floor_mult, "loss_ratio_coef": coef},
        )

    for variant_id, dd_trigger, cap_mult in [
        ("tail_cap_size_soft", -0.010, 0.80),
        ("tail_cap_size_hard", -0.020, 0.60),
    ]:
        mult = []
        eq = 0.0
        peak = 0.0
        for raw_pnl in pnl:
            drawdown = eq - peak
            cur_mult = cap_mult if drawdown < dd_trigger else 1.0
            mult.append(cur_mult)
            eq += float(raw_pnl * cur_mult)
            peak = max(peak, eq)
        add_variant(
            "tail_cap_size",
            variant_id,
            f"Cap size to {cap_mult:.2f}x when rolling drawdown < {dd_trigger:.3f}",
            np.asarray(mult, dtype=float),
            {"drawdown_trigger": dd_trigger, "cap_mult": cap_mult},
        )

    soft_mult = np.ones(len(x), dtype=float)
    soft_mult[delay > 0.0] = 0.75
    add_variant(
        "regime_cap_size",
        "regime_cap_size_delay_soft",
        "Cap delayed fills to 0.75x",
        soft_mult,
        {"delay_gt_min": 0.0, "size_mult_on_delay": 0.75},
    )

    hard_mult = np.ones(len(x), dtype=float)
    hard_mult[delay >= 6.0] = 0.35
    hard_mult[(delay > 0.0) & (delay < 6.0)] = 0.85
    add_variant(
        "regime_cap_size",
        "regime_cap_size_delay_tiered",
        "Cap 3m delayed fills by delay tier",
        hard_mult,
        {"delay_0_to_6_mult": 0.85, "delay_ge_6_mult": 0.35},
    )

    return variants


def write_sizing_specs(variants: List[Dict[str, Any]], run_dir: Path) -> None:
    fam_lines = [
        "# Model B2 Sizing Family Spec",
        "",
        "Model B is pure sizing only:",
        "- No signal logic changes.",
        "- No 3m entry logic changes.",
        "- No 1h TP/SL/exit logic changes.",
        "- Only `effective position size` is scaled trade by trade.",
        "",
        "## Families",
    ]
    seen = set()
    bounds: Dict[str, Any] = {}
    for v in variants:
        if v["family"] in seen:
            continue
        seen.add(v["family"])
        fam = v["family"]
        if fam == "baseline_primary":
            fam_lines.extend(
                [
                    "- `baseline_primary`",
                    "  - frozen Model A primary, multiplier always 1.0",
                ]
            )
            bounds[fam] = {"size_floor": 1.0, "size_ceiling": 1.0}
        elif fam == "fixed_half_on_risk":
            fam_lines.extend(
                [
                    "- `fixed_half_on_risk`",
                    "  - constant bounded size multiplier across all trades",
                    "  - tests pure risk cut with zero state dependence",
                ]
            )
            bounds[fam] = {"constant_size_mult": [0.50, 0.70]}
        elif fam == "step_down_after_streak":
            fam_lines.extend(
                [
                    "- `step_down_after_streak`",
                    "  - after two consecutive realized losses, downsize until the next win resets the streak",
                    "  - changes only exposure after confirmed adverse state",
                ]
            )
            bounds[fam] = {"trigger_loss_streak": [2], "down_mult": [0.50, 0.75]}
        elif fam == "linear_risk_score_scale":
            fam_lines.extend(
                [
                    "- `linear_risk_score_scale`",
                    "  - uses rolling 8-trade realized loss ratio to scale size linearly down to a bounded floor",
                    "  - no signal mutation, only size response to prior realized outcomes",
                ]
            )
            bounds[fam] = {"rolling_window": [8], "floor_mult": [0.60, 0.75], "loss_ratio_coef": [0.25, 0.40]}
        elif fam == "tail_cap_size":
            fam_lines.extend(
                [
                    "- `tail_cap_size`",
                    "  - caps exposure when running cumulative PnL is below a fixed drawdown threshold",
                    "  - targets tail compression and smoother equity during losing clusters",
                ]
            )
            bounds[fam] = {"drawdown_trigger": [-0.020, -0.010], "cap_mult": [0.60, 0.80]}
        elif fam == "regime_cap_size":
            fam_lines.extend(
                [
                    "- `regime_cap_size`",
                    "  - caps exposure only on delayed 3m fills, which are an observable entry state and materially underperform the immediate fills in the frozen primary",
                    "  - preserves all fills and exits; only resizes delayed-entry exposure",
                ]
            )
            bounds[fam] = {
                "delay_gt_min": [0.0],
                "size_mult_on_delay": [0.75],
                "delay_0_to_6_mult": [0.85],
                "delay_ge_6_mult": [0.35],
            }

    write_text(run_dir / "modelB2_sizing_family_spec.md", "\n".join(fam_lines) + "\n")
    # JSON is valid YAML 1.2 and keeps deps minimal.
    write_text(run_dir / "modelB2_param_bounds.yaml", json.dumps(bounds, indent=2, sort_keys=True) + "\n")


def apply_multiplier(rows_df: pd.DataFrame, mult_map: Dict[str, float]) -> pd.DataFrame:
    x = rows_df.copy()
    mult = x["signal_id"].astype(str).map(mult_map).fillna(1.0).astype(float)
    x["modelb_size_mult"] = mult
    x["exec_pnl_net_pct"] = to_num(x["exec_pnl_net_pct"]).fillna(0.0) * mult
    x["exec_pnl_gross_pct"] = to_num(x["exec_pnl_gross_pct"]).fillna(0.0) * mult
    return x


def split_expectancy_min(rows_df: pd.DataFrame) -> float:
    if rows_df.empty or "split_id" not in rows_df.columns:
        return float("nan")
    vals = []
    for _, g in rows_df.groupby("split_id", dropna=False):
        r = ga_exec._rollup_mode(g, "exec")
        vals.append(float(r["mean_expectancy_net"]))
    if not vals:
        return float("nan")
    return float(np.nanmin(np.asarray(vals, dtype=float)))


def loss_cluster_metrics(pnl: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(pnl, dtype=float)
    clusters: List[float] = []
    cur = 0.0
    for x in arr:
        if np.isfinite(x) and x < 0.0:
            cur += float(x)
        else:
            if cur < 0.0:
                clusters.append(float(cur))
            cur = 0.0
    if cur < 0.0:
        clusters.append(float(cur))
    if not clusters:
        return {
            "loss_cluster_count": 0.0,
            "loss_cluster_worst_burden": 0.0,
            "loss_cluster_avg_burden": 0.0,
            "loss_cluster_median_burden": 0.0,
        }
    c = np.asarray(clusters, dtype=float)
    return {
        "loss_cluster_count": float(len(clusters)),
        "loss_cluster_worst_burden": float(np.nanmin(c)),
        "loss_cluster_avg_burden": float(np.nanmean(c)),
        "loss_cluster_median_burden": float(np.nanmedian(c)),
    }


def cvar_improve_ratio(base_cvar: float, cand_cvar: float) -> float:
    denom = abs(float(base_cvar))
    if denom <= 1e-12:
        return 0.0
    return float((float(cand_cvar) - float(base_cvar)) / denom)


def maxdd_improve_ratio(base_dd: float, cand_dd: float) -> float:
    denom = abs(float(base_dd))
    if denom <= 1e-12:
        return 0.0
    return float((float(cand_dd) - float(base_dd)) / denom)


def cluster_improve_ratio(base_val: float, cand_val: float) -> float:
    denom = abs(float(base_val))
    if denom <= 1e-12:
        return 0.0
    return float((float(cand_val) - float(base_val)) / denom)


def stressed_rows(rows_df: pd.DataFrame) -> pd.DataFrame:
    x = rows_df.copy()
    pnl = to_num(x["exec_pnl_net_pct"]).fillna(0.0).to_numpy(dtype=float)
    x["exec_pnl_net_pct"] = np.where(pnl < 0.0, pnl * 1.10, pnl * 0.92)
    x["exec_pnl_gross_pct"] = to_num(x["exec_pnl_gross_pct"]).fillna(0.0).to_numpy(dtype=float)
    return x


def bootstrap_pass_rate(base_pnl: np.ndarray, cand_pnl: np.ndarray, *, n: int = 100, block: int = 16, seed: int = 20260301) -> float:
    rng = np.random.default_rng(int(seed))
    base = np.asarray(base_pnl, dtype=float)
    cand = np.asarray(cand_pnl, dtype=float)
    m = int(len(base))
    if m == 0 or len(cand) != m:
        return float("nan")
    passes = 0
    for _ in range(int(n)):
        idx: List[int] = []
        while len(idx) < m:
            start = int(rng.integers(0, m))
            idx.extend([(start + j) % m for j in range(int(block))])
        pick = np.asarray(idx[:m], dtype=int)
        base_s = base[pick]
        cand_s = cand[pick]
        base_exp = float(np.mean(base_s))
        cand_exp = float(np.mean(cand_s))
        base_dd = float(ga_exec._max_drawdown(base_s))  # pylint: disable=protected-access
        cand_dd = float(ga_exec._max_drawdown(cand_s))  # pylint: disable=protected-access
        dd_imp = maxdd_improve_ratio(base_dd, cand_dd)
        if (cand_exp - base_exp) >= STRESS_DELTA_TOL and dd_imp >= 0.0:
            passes += 1
    return float(passes / max(1, int(n)))


def evaluate_variant(
    variant: Dict[str, Any],
    base_rows: pd.DataFrame,
    base_roll: Dict[str, float],
    base_cluster: Dict[str, float],
    route_base_rows: Dict[str, pd.DataFrame],
    route_base_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    mult = np.asarray(variant["multiplier"], dtype=float)
    mult_map = dict(zip(base_rows["signal_id"].astype(str), mult))
    cand_rows = apply_multiplier(base_rows, mult_map)
    cand_roll = ga_exec._rollup_mode(cand_rows, "exec")
    cand_cluster = loss_cluster_metrics(cand_rows["exec_pnl_net_pct"].to_numpy(dtype=float))

    base_stress_rows = stressed_rows(base_rows)
    cand_stress_rows = stressed_rows(cand_rows)
    base_stress = ga_exec._rollup_mode(base_stress_rows, "exec")
    cand_stress = ga_exec._rollup_mode(cand_stress_rows, "exec")

    route_deltas: Dict[str, float] = {}
    route_pass = 1
    min_route_delta = float("nan")
    for rid, route_rows in route_base_rows.items():
        rb = route_rows.copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
        rc = apply_multiplier(rb, mult_map)
        rb_roll = ga_exec._rollup_mode(rb, "exec")
        rc_roll = ga_exec._rollup_mode(rc, "exec")
        delta = float(rc_roll["mean_expectancy_net"] - rb_roll["mean_expectancy_net"])
        route_deltas[rid] = delta
        if not np.isfinite(min_route_delta) or delta < min_route_delta:
            min_route_delta = delta
        if delta < ROUTE_DELTA_TOL:
            route_pass = 0

    stress_delta = float(cand_stress["mean_expectancy_net"] - base_stress["mean_expectancy_net"])
    stress_maxdd_imp = float(maxdd_improve_ratio(float(base_stress["max_drawdown"]), float(cand_stress["max_drawdown"])))
    stress_lite_pass = int((stress_delta >= STRESS_DELTA_TOL) and (stress_maxdd_imp >= 0.0))

    bootstrap_rate = float(
        bootstrap_pass_rate(
            base_pnl=base_rows["exec_pnl_net_pct"].to_numpy(dtype=float),
            cand_pnl=cand_rows["exec_pnl_net_pct"].to_numpy(dtype=float),
        )
    )
    bootstrap_pass = int(np.isfinite(bootstrap_rate) and bootstrap_rate >= BOOTSTRAP_PASS_MIN_GO)

    delta_expectancy = float(cand_roll["mean_expectancy_net"] - base_roll["mean_expectancy_net"])
    cvar_imp = float(cvar_improve_ratio(float(base_roll["cvar_5"]), float(cand_roll["cvar_5"])))
    maxdd_imp = float(maxdd_improve_ratio(float(base_roll["max_drawdown"]), float(cand_roll["max_drawdown"])))
    std_imp = float(cluster_improve_ratio(float(base_roll["pnl_std"]), float(cand_roll["pnl_std"])) * -1.0)
    cluster_worst_imp = float(
        cluster_improve_ratio(base_cluster["loss_cluster_worst_burden"], cand_cluster["loss_cluster_worst_burden"])
    )
    cluster_avg_imp = float(
        cluster_improve_ratio(base_cluster["loss_cluster_avg_burden"], cand_cluster["loss_cluster_avg_burden"])
    )

    valid_for_ranking = int(
        np.all(np.isfinite(mult))
        and float(np.min(mult)) >= 0.0
        and float(np.max(mult)) <= 1.0
        and int(base_rows["exec_valid_for_metrics"].sum()) == int(cand_rows["exec_valid_for_metrics"].sum())
        and int(base_rows["exec_filled"].sum()) == int(cand_rows["exec_filled"].sum())
    )
    invalid_reason = ""
    if valid_for_ranking == 0:
        invalid_reason = "invalid_size_vector"

    out = {
        "family": str(variant["family"]),
        "variant_id": str(variant["variant_id"]),
        "label": str(variant["label"]),
        "valid_for_ranking": int(valid_for_ranking),
        "invalid_reason": str(invalid_reason),
        "expectancy_net": float(cand_roll["mean_expectancy_net"]),
        "delta_expectancy_vs_modelA": float(delta_expectancy),
        "cvar_improve_ratio": float(cvar_imp),
        "maxdd_improve_ratio": float(maxdd_imp),
        "pnl_std_improve_ratio": float(std_imp),
        "min_split_expectancy": float(split_expectancy_min(cand_rows)),
        "loss_cluster_count": float(cand_cluster["loss_cluster_count"]),
        "loss_cluster_worst_burden": float(cand_cluster["loss_cluster_worst_burden"]),
        "loss_cluster_avg_burden": float(cand_cluster["loss_cluster_avg_burden"]),
        "loss_cluster_worst_burden_improve_ratio": float(cluster_worst_imp),
        "loss_cluster_avg_burden_improve_ratio": float(cluster_avg_imp),
        "route_pass": int(route_pass),
        "route_min_delta_vs_modelA": float(min_route_delta),
        "route_front_delta_vs_modelA": float(route_deltas.get("route_front_60pct", np.nan)),
        "route_center_delta_vs_modelA": float(route_deltas.get("route_center_60pct", np.nan)),
        "route_back_delta_vs_modelA": float(route_deltas.get("route_back_60pct", np.nan)),
        "stress_lite_pass": int(stress_lite_pass),
        "stress_delta_expectancy_vs_modelA": float(stress_delta),
        "stress_maxdd_improve_ratio": float(stress_maxdd_imp),
        "bootstrap_pass_rate": float(bootstrap_rate),
        "bootstrap_pass": int(bootstrap_pass),
        "entries_valid": int(cand_roll["entries_valid"]),
        "entry_rate": float(cand_roll["entry_rate"]),
        "taker_share": float(cand_roll["taker_share"]),
        "median_fill_delay_min": float(cand_roll["median_fill_delay_min"]),
        "p95_fill_delay_min": float(cand_roll["p95_fill_delay_min"]),
        "participation_changed": int(False),
        "validity_hard_gates_intact": int(valid_for_ranking),
        "size_mult_min": float(np.min(mult)),
        "size_mult_mean": float(np.mean(mult)),
        "size_mult_max": float(np.max(mult)),
        "selection_score": float(cvar_imp + maxdd_imp + cluster_avg_imp + (delta_expectancy * 10000.0)),
    }
    return out


def classify(df: pd.DataFrame) -> tuple[str, str, str, pd.DataFrame]:
    candidates = df[df["variant_id"] != "MODEL_A_PRIMARY_BASELINE"].copy()
    if candidates.empty:
        return "MODEL_B_NO_GO", "MODEL_B_NO_GO", "no overlay candidates", candidates

    strong = candidates[
        (candidates["valid_for_ranking"] == 1)
        & (candidates["delta_expectancy_vs_modelA"] >= -5e-5)
        & (candidates["cvar_improve_ratio"] >= 0.05)
        & (candidates["maxdd_improve_ratio"] >= 0.01)
        & (candidates["route_pass"] == 1)
        & (candidates["stress_lite_pass"] == 1)
        & (candidates["bootstrap_pass_rate"] >= BOOTSTRAP_PASS_MIN_GO)
    ].copy()
    if not strong.empty:
        strong = strong.sort_values(
            ["selection_score", "delta_expectancy_vs_modelA", "cvar_improve_ratio", "maxdd_improve_ratio"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        return "MODEL_B_GO", "MODEL_B_GO", "material risk-shape improvement with no meaningful expectancy damage", strong

    weak = candidates[
        (candidates["valid_for_ranking"] == 1)
        & (candidates["delta_expectancy_vs_modelA"] >= -1.5e-4)
        & (candidates["cvar_improve_ratio"] >= 0.02)
        & (candidates["maxdd_improve_ratio"] >= 0.005)
        & (candidates["route_pass"] == 1)
        & (candidates["stress_lite_pass"] == 1)
        & (candidates["bootstrap_pass_rate"] >= BOOTSTRAP_PASS_MIN_WEAK)
    ].copy()
    if not weak.empty:
        weak = weak.sort_values(
            ["selection_score", "delta_expectancy_vs_modelA", "cvar_improve_ratio", "maxdd_improve_ratio"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
        return "MODEL_B_WEAK_GO", "MODEL_B_WEAK_GO", "limited but defensible risk-shape improvement", weak

    candidates = candidates.sort_values(
        ["selection_score", "delta_expectancy_vs_modelA", "cvar_improve_ratio", "maxdd_improve_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return "MODEL_B_NO_GO", "MODEL_B_NO_GO", "sizing did not clear the bounded risk-shape bar", candidates


def main() -> None:
    run_dir = REPORTS_ROOT / f"PHASEMODELB_SIZING_OVERLAY_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    primary_sel, backup_sel, primary_cfg, backup_cfg = load_phasec_candidates()
    repro = reproduce_primary(primary_sel=primary_sel, primary_cfg=primary_cfg, run_dir=run_dir)

    modelb1_contract = {
        "generated_utc": utc_now(),
        "phase_c_primary_candidate_id": str(primary_sel["candidate_id"]),
        "phase_c_backup_candidate_id": str(backup_sel["candidate_id"]),
        "freeze_lock_pass": int(repro["lock_info"]["freeze_lock_pass"]),
        "parity_ok": int(repro["parity_ok"]),
        "parity_diffs": repro["parity_diffs"],
        "primary_cfg": repro["primary_cfg"],
        "backup_reference_cfg": model_a_cfg_from_row(backup_cfg),
        "route_harness_meta": repro["route_meta"],
        "route_trade_gates_reachable": int(repro["route_valid_df"]["route_trade_gates_reachable"].astype(int).all()),
    }
    json_dump(run_dir / "modelB1_contract_validation.json", modelb1_contract)

    repro_lines = [
        "# Model B1 Reproduction Report",
        "",
        f"- Generated UTC: `{modelb1_contract['generated_utc']}`",
        f"- Frozen lock pass: `{modelb1_contract['freeze_lock_pass']}`",
        f"- Primary parity OK vs approved Phase C row: `{modelb1_contract['parity_ok']}`",
        f"- Primary candidate: `{primary_sel['candidate_id']}`",
        f"- Backup reference (for context only): `{backup_sel['candidate_id']}`",
        "",
        "## Reproduced Primary Metrics",
        f"- expectancy_net: `{repro['base_metrics']['overall_exec_expectancy_net']}`",
        f"- delta_expectancy_vs_1h_reference: `{repro['base_metrics']['overall_delta_expectancy_exec_minus_baseline']}`",
        f"- cvar_improve_ratio: `{repro['base_metrics']['overall_cvar_improve_ratio']}`",
        f"- maxdd_improve_ratio: `{repro['base_metrics']['overall_maxdd_improve_ratio']}`",
        f"- entry_rate: `{repro['base_metrics']['overall_entry_rate']}`",
        f"- entries_valid: `{repro['base_metrics']['overall_entries_valid']}`",
        f"- taker_share: `{repro['base_metrics']['overall_exec_taker_share']}`",
        f"- p95_fill_delay_min: `{repro['base_metrics']['overall_exec_p95_fill_delay_min']}`",
        "",
        "## Parity Diffs vs Approved Phase C Primary",
    ]
    for k, v in repro["parity_diffs"].items():
        repro_lines.append(f"- {k}: `{v}`")
    repro_lines.extend(
        [
            "",
            "## Route Baseline",
            markdown_table(
                pd.DataFrame(
                    [
                        {"route_id": rid, **vals}
                        for rid, vals in sorted(repro["route_base_metrics"].items())
                    ]
                ),
                ["route_id", "expectancy_net", "entry_rate", "entries_valid"],
                n=8,
            ),
        ]
    )
    write_text(run_dir / "modelB1_reproduction_report.md", "\n".join(repro_lines) + "\n")

    if int(repro["parity_ok"]) != 1:
        decision = [
            "# Model B4 Decision",
            "",
            "- Classification: `MODEL_B_INFRA_FAIL`",
            "- Mainline status: `MODEL_B_INFRA_FAIL`",
            "- Exact blocker: primary Model A baseline could not be reproduced under the frozen lock.",
        ]
        write_text(run_dir / "modelB4_decision.md", "\n".join(decision) + "\n")
        json_dump(
            run_dir / "modelB_run_manifest.json",
            {
                "generated_utc": utc_now(),
                "git": git_snapshot(),
                "classification": "MODEL_B_INFRA_FAIL",
                "mainline_status": "MODEL_B_INFRA_FAIL",
                "artifact_dir": str(run_dir),
            },
        )
        print(str(run_dir))
        return

    variants = build_sizing_variants(repro["base_rows"])
    write_sizing_specs(variants, run_dir)

    base_roll = ga_exec._rollup_mode(repro["base_rows"], "exec")
    base_cluster = loss_cluster_metrics(repro["base_rows"]["exec_pnl_net_pct"].to_numpy(dtype=float))

    rows: List[Dict[str, Any]] = []
    invalid_hist = Counter()
    for variant in variants:
        out = evaluate_variant(
            variant=variant,
            base_rows=repro["base_rows"],
            base_roll=base_roll,
            base_cluster=base_cluster,
            route_base_rows=repro["route_base_rows"],
            route_base_metrics=repro["route_base_metrics"],
        )
        out["params"] = json.dumps(variant["params"], sort_keys=True)
        rows.append(out)
        if out["invalid_reason"]:
            invalid_hist[out["invalid_reason"]] += 1

    df = pd.DataFrame(rows).sort_values(
        ["selection_score", "variant_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    df.to_csv(run_dir / "modelB3_ablation_results.csv", index=False)
    json_dump(run_dir / "modelB3_invalid_reason_histogram.json", dict(invalid_hist))

    classification, mainline_status, reason_text, ranked = classify(df)
    top_table = markdown_table(
        ranked.head(6),
        [
            "variant_id",
            "family",
            "expectancy_net",
            "delta_expectancy_vs_modelA",
            "cvar_improve_ratio",
            "maxdd_improve_ratio",
            "loss_cluster_avg_burden_improve_ratio",
            "route_pass",
            "stress_lite_pass",
            "bootstrap_pass_rate",
        ],
        n=6,
    )

    ablation_lines = [
        "# Model B3 Ablation Report",
        "",
        "## Baseline Reference",
        f"- Frozen Model A primary expectancy_net: `{base_roll['mean_expectancy_net']}`",
        f"- Frozen Model A primary cvar_5: `{base_roll['cvar_5']}`",
        f"- Frozen Model A primary max_drawdown: `{base_roll['max_drawdown']}`",
        f"- Frozen Model A primary max_consecutive_losses: `{base_roll['max_consecutive_losses']}`",
        f"- Frozen Model A primary loss_cluster_worst_burden: `{base_cluster['loss_cluster_worst_burden']}`",
        f"- Frozen Model A primary loss_cluster_avg_burden: `{base_cluster['loss_cluster_avg_burden']}`",
        "",
        "## Top Model B Variants",
        top_table,
        "",
        f"## Interim Readout",
        f"- Preliminary classification: `{classification}`",
        f"- Reason: `{reason_text}`",
    ]
    write_text(run_dir / "modelB3_ablation_report.md", "\n".join(ablation_lines) + "\n")

    decision_lines = [
        "# Model B4 Decision",
        "",
        f"- Classification: `{classification}`",
        f"- Mainline status: `{mainline_status}`",
        f"- Reason: `{reason_text}`",
    ]
    if not ranked.empty:
        best = ranked.iloc[0]
        decision_lines.extend(
            [
                "",
                "## Best Variant",
                f"- variant_id: `{best['variant_id']}`",
                f"- expectancy_net: `{best['expectancy_net']}`",
                f"- delta_expectancy_vs_modelA: `{best['delta_expectancy_vs_modelA']}`",
                f"- cvar_improve_ratio: `{best['cvar_improve_ratio']}`",
                f"- maxdd_improve_ratio: `{best['maxdd_improve_ratio']}`",
                f"- loss_cluster_avg_burden_improve_ratio: `{best['loss_cluster_avg_burden_improve_ratio']}`",
                f"- route_pass: `{best['route_pass']}`",
                f"- stress_lite_pass: `{best['stress_lite_pass']}`",
                f"- bootstrap_pass_rate: `{best['bootstrap_pass_rate']}`",
            ]
        )

    next_prompt = ""
    if classification in {"MODEL_B_GO", "MODEL_B_WEAK_GO"} and not ranked.empty:
        best = ranked.iloc[0]
        next_prompt = (
            "ROLE\n\n"
            "You are in Model B bounded follow-up mode.\n\n"
            "MISSION\n\n"
            "Expand only around the best Model B sizing overlay on top of the frozen Model A primary. "
            "Do not change signal logic, 1h exits, or 3m entry execution. Keep sizing-only and bounded.\n\n"
            f"ANCHOR VARIANT\n\n{best['variant_id']}\n\n"
            "REQUIREMENTS\n\n"
            "1. Same frozen Model A contract.\n"
            "2. Same repaired route harness.\n"
            "3. No exit mutation.\n"
            "4. No large GA; local neighborhood only.\n"
            "5. Stop on first expectancy damage beyond -5e-5 versus frozen Model A primary.\n"
        )
        write_text(run_dir / "ready_to_launch_modelB_next_prompt.txt", next_prompt)

    write_text(run_dir / "modelB4_decision.md", "\n".join(decision_lines) + "\n")

    json_dump(
        run_dir / "modelB_run_manifest.json",
        {
            "generated_utc": utc_now(),
            "git": git_snapshot(),
            "artifact_dir": str(run_dir),
            "classification": classification,
            "mainline_status": mainline_status,
            "primary_candidate": str(primary_sel["candidate_id"]),
            "backup_reference": str(backup_sel["candidate_id"]),
            "best_variant": (str(ranked.iloc[0]["variant_id"]) if not ranked.empty else ""),
        },
    )

    print(str(run_dir))


if __name__ == "__main__":
    main()
