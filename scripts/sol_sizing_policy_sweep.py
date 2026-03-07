#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_a_model_a_audit as modela  # noqa: E402
from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402
from scripts import repaired_universe_3m_exec_subset1 as subset1  # noqa: E402
from scripts import sol_3m_lossconcentration_ga as ga  # noqa: E402


RUN_PREFIX = "SOL_SIZING_POLICY_SWEEP"
SYMBOL = "SOLUSDT"
BASELINE_STRATEGY_ID = "M1_ENTRY_ONLY_PASSIVE_BASELINE"

FEATURE_RANGE = "feature_skip_signal_range_pct"
FEATURE_WICK = "feature_skip_upper_wick_ratio"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def utc_tag() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


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
                vals.append(f"{v:.10g}" if np.isfinite(v) else "nan")
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


def quality_from_build_meta(build_meta: Dict[str, Any]) -> Dict[str, Any]:
    total = int(build_meta.get("signals_total", 0))
    with_data = int(build_meta.get("signals_with_3m_data", 0))
    partial = int(build_meta.get("signals_partial_3m_data", 0))
    missing = int(build_meta.get("signals_missing_3m_data", 0))
    ready = max(0, with_data - partial)
    miss_rate = float(missing / total) if total > 0 else float("nan")
    if with_data <= 0:
        status = "DATA_BLOCKED"
    elif missing > 0 or partial > 0:
        status = "PARTIAL"
    else:
        status = "READY"
    return {
        "integrity_status": status,
        "windows_ready": int(ready),
        "windows_partial": int(partial),
        "windows_blocked": int(missing),
        "missing_window_rate": miss_rate,
        "signals_covered": int(with_data),
        "signals_uncovered": int(missing),
    }


def build_variants() -> List[Dict[str, Any]]:
    return [
        {
            "variant_id": "V0_BASELINE",
            "label": "Constant 1.00 size",
            "policy_family": "constant",
            "const_mult": 1.00,
        },
        {
            "variant_id": "V1_VOL_CAP_K0P90",
            "label": "Vol cap k=0.90, clamp[0.50,1.00] on signal_range_pct",
            "policy_family": "vol_cap",
            "k": 0.90,
            "min_mult": 0.50,
            "max_mult": 1.00,
        },
        {
            "variant_id": "V1_VOL_CAP_K0P75",
            "label": "Vol cap k=0.75, clamp[0.50,1.00] on signal_range_pct",
            "policy_family": "vol_cap",
            "k": 0.75,
            "min_mult": 0.50,
            "max_mult": 1.00,
        },
        {
            "variant_id": "V2_WICK_CAP0P60_M0P75",
            "label": "Wick stress cap 0.60 -> size 0.75",
            "policy_family": "wick_cap",
            "wick_cap": 0.60,
            "stress_mult": 0.75,
            "base_mult": 1.00,
        },
        {
            "variant_id": "V2_WICK_CAP0P55_M0P50",
            "label": "Wick stress cap 0.55 -> size 0.50",
            "policy_family": "wick_cap",
            "wick_cap": 0.55,
            "stress_mult": 0.50,
            "base_mult": 1.00,
        },
        {
            "variant_id": "V3_SCORE_BUCKET",
            "label": "2-feature score bucket (range+wick): {1.00,0.75,0.50}",
            "policy_family": "score_bucket",
            "a": 25.0,  # scales range_pct (~0.005-0.03) into wick-comparable magnitude
            "b": 1.00,
            "t1": 0.60,
            "t2": 0.90,
            "mid_mult": 0.75,
            "high_mult": 0.50,
        },
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SOL-only bounded sizing-policy sweep (causal, no trade selection changes)")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--posture-dir", default="")
    ap.add_argument("--strict-confirm-dir", default="")
    ap.add_argument("--seed", type=int, default=20260307)
    ap.add_argument("--retention-floor", type=float, default=0.99)
    ap.add_argument("--tail_bottom_decile_rel", type=float, default=0.10)
    ap.add_argument("--tail_maxdd_improve_ratio", type=float, default=0.05)
    ap.add_argument("--tail_cvar_abs_min", type=float, default=0.00005)
    ap.add_argument("--variants", default="")  # optional comma-separated subset
    return ap


def subset_by_signal_ids(df: pd.DataFrame, signal_ids: Sequence[str]) -> pd.DataFrame:
    keep = set(str(x) for x in signal_ids)
    return df[df["signal_id"].astype(str).isin(keep)].copy().reset_index(drop=True)


def compute_size_mult(
    rows_df: pd.DataFrame,
    variant: Dict[str, Any],
) -> pd.Series:
    n = len(rows_df)
    out = pd.Series(np.ones(n, dtype=float), index=rows_df.index, dtype=float)
    fam = str(variant.get("policy_family", "constant")).strip().lower()
    range_feat = to_num(rows_df.get(FEATURE_RANGE, np.nan))
    wick_feat = to_num(rows_df.get(FEATURE_WICK, np.nan))

    if fam == "constant":
        out[:] = float(variant.get("const_mult", 1.0))
    elif fam == "vol_cap":
        k = float(variant.get("k", 1.0))
        min_mult = float(variant.get("min_mult", 0.50))
        max_mult = float(variant.get("max_mult", 1.00))
        vol = range_feat.copy()
        vol = vol.where(np.isfinite(vol), 0.0)
        vol = vol.clip(lower=0.0)
        out = (k / (1.0 + vol)).clip(lower=min_mult, upper=max_mult).astype(float)
    elif fam == "wick_cap":
        cap = float(variant.get("wick_cap", 0.60))
        stress_mult = float(variant.get("stress_mult", 0.75))
        base_mult = float(variant.get("base_mult", 1.00))
        trig = (wick_feat > cap) & wick_feat.notna()
        out[:] = base_mult
        out.loc[trig] = stress_mult
    elif fam == "score_bucket":
        a = float(variant.get("a", 25.0))
        b = float(variant.get("b", 1.0))
        t1 = float(variant.get("t1", 0.60))
        t2 = float(variant.get("t2", 0.90))
        mid_mult = float(variant.get("mid_mult", 0.75))
        high_mult = float(variant.get("high_mult", 0.50))
        rr = range_feat.where(np.isfinite(range_feat), 0.0)
        ww = wick_feat.where(np.isfinite(wick_feat), 0.0)
        score = a * rr + b * ww
        out[:] = 1.0
        out.loc[(score > t1) & (score <= t2)] = mid_mult
        out.loc[score > t2] = high_mult
    else:
        raise RuntimeError(f"Unknown policy_family: {fam}")

    out = out.where(np.isfinite(out), 1.0).clip(lower=0.0, upper=1.0)
    return out.astype(float)


def apply_sizing_policy(
    base_rows_df: pd.DataFrame,
    variant: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    x = base_rows_df.copy()
    filled = to_num(x.get("exec_filled", 0)).fillna(0).astype(int) == 1
    valid = to_num(x.get("exec_valid_for_metrics", 0)).fillna(0).astype(int) == 1
    pnl_net = to_num(x.get("exec_pnl_net_pct", np.nan))
    pnl_gross = to_num(x.get("exec_pnl_gross_pct", np.nan))
    scale_mask = filled & valid & pnl_net.notna()

    mult = compute_size_mult(x, variant)
    mult_used = pd.Series(np.ones(len(x), dtype=float), index=x.index, dtype=float)
    mult_used.loc[scale_mask] = mult.loc[scale_mask].astype(float)

    x["size_mult"] = mult_used.astype(float)
    x["exec_pnl_net_pct_raw"] = pnl_net.astype(float)
    x["exec_pnl_gross_pct_raw"] = pnl_gross.astype(float)
    x.loc[scale_mask, "exec_pnl_net_pct"] = (pnl_net.loc[scale_mask] * mult_used.loc[scale_mask]).astype(float)
    x.loc[scale_mask, "exec_pnl_gross_pct"] = (pnl_gross.loc[scale_mask] * mult_used.loc[scale_mask]).astype(float)

    used = mult_used.loc[scale_mask].astype(float)
    stats = {
        "size_mult_mean": float(np.nanmean(used)) if len(used) > 0 else float("nan"),
        "size_mult_median": float(np.nanmedian(used)) if len(used) > 0 else float("nan"),
        "size_mult_p10": float(np.nanpercentile(used, 10)) if len(used) > 0 else float("nan"),
        "size_mult_p90": float(np.nanpercentile(used, 90)) if len(used) > 0 else float("nan"),
        "size_mult_min": float(np.nanmin(used)) if len(used) > 0 else float("nan"),
        "size_mult_max": float(np.nanmax(used)) if len(used) > 0 else float("nan"),
        "size_mult_reduced_share": float((used < 1.0 - 1e-12).mean()) if len(used) > 0 else float("nan"),
        "size_mult_le_0p75_share": float((used <= 0.75 + 1e-12).mean()) if len(used) > 0 else float("nan"),
        "size_mult_le_0p50_share": float((used <= 0.50 + 1e-12).mean()) if len(used) > 0 else float("nan"),
    }
    return x, stats


def split_detail_and_gate(candidate_rows: pd.DataFrame) -> Tuple[pd.DataFrame, int, float]:
    if "split_id" not in candidate_rows.columns:
        return pd.DataFrame(), 0, float("nan")
    details: List[Dict[str, Any]] = []
    for split_id, g in candidate_rows.groupby(to_num(candidate_rows["split_id"]).fillna(-1).astype(int)):
        if int(split_id) < 0:
            continue
        agg = modela.ga_exec._aggregate_rows(g)  # pylint: disable=protected-access
        e = agg["exec"]
        details.append(
            {
                "detail_type": "split",
                "split_id": int(split_id),
                "signals_total": int(e["signals_total"]),
                "entries_valid": int(e["entries_valid"]),
                "entry_rate": float(e["entry_rate"]),
                "delta_expectancy_vs_1h_reference": float(agg["delta_expectancy_exec_minus_baseline"]),
                "cvar_improve_ratio_vs_1h_reference": float(agg["cvar_improve_ratio"]),
                "maxdd_improve_ratio_vs_1h_reference": float(agg["maxdd_improve_ratio"]),
            }
        )
    split_df = pd.DataFrame(details).sort_values("split_id").reset_index(drop=True) if details else pd.DataFrame()
    if split_df.empty:
        return split_df, 0, float("nan")
    deltas = to_num(split_df["delta_expectancy_vs_1h_reference"])
    valid_flags = to_num(split_df["entries_valid"]).fillna(0).astype(int) > 0
    split_support_pass = int(bool(valid_flags.all()) and bool(np.isfinite(deltas).all()) and bool((deltas > 0.0).all()))
    min_split_delta = float(deltas.min()) if len(deltas) > 0 else float("nan")
    return split_df, split_support_pass, min_split_delta


def route_detail_and_gate(
    *,
    variant: Dict[str, Any],
    route_baseline_rows: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, int, float, int]:
    details: List[Dict[str, Any]] = []
    route_pass_count = 0
    for route_id, base_rows in route_baseline_rows.items():
        cand_rows, _stats = apply_sizing_policy(base_rows, variant)
        agg = modela.ga_exec._aggregate_rows(cand_rows)  # pylint: disable=protected-access
        e = agg["exec"]
        delta_1h = float(agg["delta_expectancy_exec_minus_baseline"])
        valid = int(np.isfinite(delta_1h) and int(e["entries_valid"]) > 0)
        pass_flag = int(valid == 1 and delta_1h > 0.0)
        route_pass_count += int(pass_flag)
        details.append(
            {
                "detail_type": "route",
                "route_id": str(route_id),
                "valid_for_ranking": int(valid),
                "entries_valid": int(e["entries_valid"]),
                "entry_rate": float(e["entry_rate"]),
                "taker_share": float(e["taker_share"]),
                "delta_expectancy_vs_1h_reference": float(delta_1h),
                "cvar_improve_ratio_vs_1h_reference": float(agg["cvar_improve_ratio"]),
                "maxdd_improve_ratio_vs_1h_reference": float(agg["maxdd_improve_ratio"]),
                "route_pass_flag": int(pass_flag),
            }
        )
    route_df = pd.DataFrame(details).sort_values("route_id").reset_index(drop=True) if details else pd.DataFrame()
    total = int(len(details))
    route_pass_rate = float(route_pass_count / max(1, total))
    route_support_pass = int(total > 0 and route_pass_count == total)
    return route_df, route_support_pass, route_pass_rate, total


def load_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    posture_dir = (
        Path(args.posture_dir).resolve()
        if str(args.posture_dir).strip()
        else find_latest_complete(
            exec_root,
            "REPAIRED_BRANCH_3M_POSTURE_FREEZE_*",
            [
                "repaired_active_3m_subset.csv",
                "repaired_active_3m_params",
                "repaired_3m_posture_table.csv",
                "repaired_3m_posture_manifest.json",
            ],
        )
    )
    if posture_dir is None:
        raise FileNotFoundError("Missing completed REPAIRED_BRANCH_3M_POSTURE_FREEZE_* directory")

    strict_confirm_dir = (
        Path(args.strict_confirm_dir).resolve()
        if str(args.strict_confirm_dir).strip()
        else find_latest_complete(
            exec_root,
            "REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_*",
            [
                "repaired_subset1_confirm_by_symbol.csv",
                "repaired_subset1_confirm_manifest.json",
            ],
        )
    )
    if strict_confirm_dir is None:
        raise FileNotFoundError("Missing completed REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_* directory")

    posture_manifest = json.loads((posture_dir / "repaired_3m_posture_manifest.json").read_text(encoding="utf-8"))
    strict_manifest = json.loads((strict_confirm_dir / "repaired_subset1_confirm_manifest.json").read_text(encoding="utf-8"))

    freeze_dir = Path(
        posture_manifest.get("source_artifacts", {}).get("frozen_repaired_universe_dir", strict_manifest.get("freeze_dir", ""))
    ).resolve()
    if not freeze_dir.exists():
        raise FileNotFoundError(f"Freeze directory does not exist: {freeze_dir}")
    foundation_dir = Path(strict_manifest.get("foundation_dir", "")).resolve()
    if not foundation_dir.exists():
        raise FileNotFoundError(f"Foundation directory does not exist: {foundation_dir}")

    posture_active_df = pd.read_csv(posture_dir / "repaired_active_3m_subset.csv")
    posture_active_df["symbol"] = posture_active_df["symbol"].astype(str).str.upper()
    sol_active = posture_active_df[posture_active_df["symbol"] == SYMBOL].copy()
    if sol_active.empty:
        raise RuntimeError("SOLUSDT is not present in repaired active subset")
    if int(len(sol_active)) != 1:
        raise RuntimeError(f"Expected one SOLUSDT active row, found {len(sol_active)}")
    winner_id = str(sol_active.iloc[0]["winner_config_id"]).strip()
    if winner_id != BASELINE_STRATEGY_ID:
        raise RuntimeError(f"Baseline strategy mismatch: expected {BASELINE_STRATEGY_ID}, got {winner_id}")

    freeze_df = pd.read_csv(freeze_dir / "repaired_best_by_symbol.csv")
    freeze_df["symbol"] = freeze_df["symbol"].astype(str).str.upper()
    sol_row_df = freeze_df[(freeze_df["symbol"] == SYMBOL) & (freeze_df["side"].astype(str).str.lower() == "long")].copy()
    if sol_row_df.empty:
        raise RuntimeError("SOLUSDT long row missing in repaired_best_by_symbol.csv")
    sol_row = sol_row_df.iloc[0]
    selected_params_dir = freeze_dir / "repaired_universe_selected_params"
    params_payload = subset1.parse_params_from_row(sol_row, selected_params_dir)

    return {
        "posture_dir": posture_dir,
        "strict_confirm_dir": strict_confirm_dir,
        "freeze_dir": freeze_dir,
        "foundation_dir": foundation_dir,
        "sol_row": sol_row,
        "params_payload": params_payload,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    inputs = load_inputs(args)

    run_dir = ensure_dir((PROJECT_ROOT / args.outdir).resolve() / f"{RUN_PREFIX}_{utc_tag()}")
    inputs_dir = ensure_dir(run_dir / "_inputs")
    cache_dir = ensure_dir(run_dir / "_window_cache")

    foundation_state = phase_v.load_foundation_state(inputs["foundation_dir"])
    df_cache: Dict[Any, pd.DataFrame] = {}
    raw_cache: Dict[str, pd.DataFrame] = {}
    signal_df = subset1.build_signal_table_for_row(
        row=inputs["sol_row"],
        params=inputs["params_payload"],
        df_cache=df_cache,
        raw_cache=raw_cache,
    )
    if signal_df.empty:
        raise RuntimeError("No rebuilt repaired 1h signals for SOLUSDT")
    signal_df.to_csv(inputs_dir / "sol_signal_timeline.csv", index=False)
    signal_df.to_csv(inputs_dir / "universe_signal_timeline.csv", index=False)

    symbol_windows = subset1.build_window_pool_for_symbol(
        symbol=SYMBOL,
        signal_df=signal_df,
        foundation_state=foundation_state,
        cache_dir=cache_dir,
    )
    if symbol_windows.empty:
        raise RuntimeError("No usable 3m windows for SOLUSDT")

    exec_args = phase_v.build_exec_args(
        foundation_state=phase_v.FoundationState(
            root=inputs_dir,
            signal_timeline=signal_df.copy(),
            download_manifest=pd.DataFrame(),
            quality=pd.DataFrame(),
            readiness=pd.DataFrame(),
        ),
        seed=int(args.seed),
    )
    setattr(exec_args, "tie_touch_policy", "sl_first")
    contract_validation = phase_v.build_contract_validation(exec_args=exec_args, run_dir=run_dir)

    bundle, build_meta = phase_v.build_symbol_bundle(
        symbol=SYMBOL,
        symbol_signals=signal_df.copy(),
        symbol_windows=symbol_windows,
        exec_args=exec_args,
        run_dir=run_dir,
    )
    n_total = int(len(bundle.contexts))
    if n_total < 500:
        raise RuntimeError(f"Insufficient contexts for sizing sweep ({n_total})")
    train_end = int(math.floor(0.60 * n_total))
    holdout_start = int(math.floor(0.80 * n_total))
    train_bundle = ga.build_subbundle(bundle, 0, train_end, exec_args)
    holdout_signal_ids = [str(ctx.signal_id) for ctx in bundle.contexts[holdout_start:]]

    fee = modela.phasec_bt.FeeModel(
        fee_bps_maker=float(exec_args.fee_bps_maker),
        fee_bps_taker=float(exec_args.fee_bps_taker),
        slippage_bps_limit=float(exec_args.slippage_bps_limit),
        slippage_bps_market=float(exec_args.slippage_bps_market),
    )
    one_h = modela.load_1h_market(SYMBOL)

    baseline_1h_full = modela.build_1h_reference_rows(bundle=bundle, fee=fee, exec_horizon_hours=float(exec_args.exec_horizon_hours))
    baseline_1h_train = modela.build_1h_reference_rows(bundle=train_bundle, fee=fee, exec_horizon_hours=float(exec_args.exec_horizon_hours))

    variants = phase_v.sanitize_variants()
    variant_map = {str(v["candidate_id"]): dict(v) for v in variants}
    if BASELINE_STRATEGY_ID not in variant_map:
        raise RuntimeError(f"Baseline variant {BASELINE_STRATEGY_ID} missing from sanitized variants")
    baseline_cfg = ga.normalize_cfg(dict(variant_map[BASELINE_STRATEGY_ID]))

    baseline_train_eval = ga.evaluate_cfg_on_bundle(
        bundle=train_bundle,
        baseline_1h_df=baseline_1h_train,
        cfg=baseline_cfg,
        one_h=one_h,
        exec_args=exec_args,
    )
    baseline_full_eval = ga.evaluate_cfg_on_bundle(
        bundle=bundle,
        baseline_1h_df=baseline_1h_full,
        cfg=baseline_cfg,
        one_h=one_h,
        exec_args=exec_args,
    )
    baseline_train_rows = baseline_train_eval["signal_rows_df"].copy().reset_index(drop=True)
    baseline_full_rows = baseline_full_eval["signal_rows_df"].copy().reset_index(drop=True)
    baseline_train_loss = ga.compute_loss_metrics(baseline_train_rows)
    baseline_full_loss = ga.compute_loss_metrics(baseline_full_rows)
    baseline_train_chrono = ga.compute_chronology_stats(baseline_train_rows)
    baseline_full_chrono = ga.compute_chronology_stats(baseline_full_rows)
    baseline_full_agg = modela.ga_exec._aggregate_rows(baseline_full_rows)  # pylint: disable=protected-access
    baseline_full_roll = modela.ga_exec._rollup_mode(baseline_full_rows, "exec")  # pylint: disable=protected-access

    # Strict harness check: baseline must remain full-route confirmed under phase_v lineage.
    qrow = quality_from_build_meta(build_meta)
    rrow = {
        "bucket_1h": "REPAIRED_LONG",
        "integrity_status": str(qrow.get("integrity_status", "")),
        "windows_ready": int(qrow.get("windows_ready", 0)),
        "windows_partial": int(qrow.get("windows_partial", 0)),
        "windows_blocked": int(qrow.get("windows_blocked", 0)),
    }
    strict_pack = phase_v.evaluate_symbol(
        symbol=SYMBOL,
        bundle=bundle,
        foundation_quality=qrow,
        foundation_readiness=rrow,
        exec_args=exec_args,
        variants=[baseline_cfg],
    )
    strict_rows = strict_pack["results_df"].copy()
    strict_row = strict_rows[strict_rows["candidate_id"].astype(str) == BASELINE_STRATEGY_ID]
    if strict_row.empty:
        raise RuntimeError(f"Strict harness missing baseline row for {BASELINE_STRATEGY_ID}")
    strict_route_pass = float(to_num(pd.Series([strict_row.iloc[0].get("route_pass_rate", np.nan)])).iloc[0])
    if not (np.isfinite(strict_route_pass) and abs(strict_route_pass - 1.0) <= 1e-12):
        raise RuntimeError(f"Strict harness baseline route pass is not 1.0: {strict_route_pass}")

    route_bundles, route_baseline_1h, route_meta = ga.build_route_family(
        base_bundle=bundle,
        exec_args=exec_args,
        fee=fee,
    )
    route_baseline_rows: Dict[str, pd.DataFrame] = {}
    for rid, rb in route_bundles.items():
        rev = ga.evaluate_cfg_on_bundle(
            bundle=rb,
            baseline_1h_df=route_baseline_1h[rid],
            cfg=baseline_cfg,
            one_h=one_h,
            exec_args=exec_args,
        )
        route_baseline_rows[rid] = rev["signal_rows_df"].copy().reset_index(drop=True)

    all_policies = build_variants()
    req_ids = [x.strip() for x in str(args.variants).split(",") if x.strip()]
    if req_ids:
        allowed = {v["variant_id"] for v in all_policies}
        unknown = [x for x in req_ids if x not in allowed]
        if unknown:
            raise RuntimeError(f"Unknown --variants values: {unknown}")
        policies = [copy.deepcopy(v) for v in all_policies if v["variant_id"] in req_ids]
    else:
        policies = [copy.deepcopy(v) for v in all_policies]

    if "V0_BASELINE" not in [p["variant_id"] for p in policies]:
        raise RuntimeError("V0_BASELINE must be included for invariant sizing sweep")

    # Tail improvement absolute threshold in cvar units.
    baseline_cvar_abs = abs(float(baseline_full_roll["cvar_5"])) if np.isfinite(float(baseline_full_roll["cvar_5"])) else 0.0
    cvar_improve_abs_threshold = max(float(args.tail_cvar_abs_min), 0.05 * baseline_cvar_abs)

    screen_rows: List[Dict[str, Any]] = []
    confirm_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for pol in policies:
        pol = copy.deepcopy(pol)
        variant_id = str(pol["variant_id"])

        # Stage 1: Train (60%) screen.
        train_rows, train_size_stats = apply_sizing_policy(baseline_train_rows, pol)
        tr_cmp = ga.compare_candidate_vs_baseline(train_rows, baseline_train_rows)
        tr_loss = ga.compute_loss_metrics(train_rows)
        tr_chrono = ga.compute_chronology_stats(train_rows)
        tr_ret = float(tr_loss.get("trade_count", 0) / max(1, baseline_train_loss.get("trade_count", 0)))
        tr_tail_rel = ga.relative_reduction(
            float(baseline_train_loss.get("bottom_decile_pnl_share", np.nan)),
            float(tr_loss.get("bottom_decile_pnl_share", np.nan)),
        )
        tr_fast_rel = ga.relative_reduction(
            float(baseline_train_loss.get("fast_loser_rate", np.nan)),
            float(tr_loss.get("fast_loser_rate", np.nan)),
        )
        tr_score = (
            450.0 * float(tr_cmp.get("delta_expectancy", np.nan) if np.isfinite(tr_cmp.get("delta_expectancy", np.nan)) else -1e9)
            + 160.0 * float(tr_cmp.get("delta_cvar_5", np.nan) if np.isfinite(tr_cmp.get("delta_cvar_5", np.nan)) else -1e9)
            + 90.0 * float(tr_cmp.get("maxdd_improve_ratio", np.nan) if np.isfinite(tr_cmp.get("maxdd_improve_ratio", np.nan)) else -1e9)
            + 3.0 * float(tr_tail_rel if np.isfinite(tr_tail_rel) else -1e9)
        )
        screen_rows.append(
            {
                "variant_id": variant_id,
                "label": str(pol.get("label", "")),
                "policy_family": str(pol.get("policy_family", "")),
                "screen_score": float(tr_score),
                "trade_count": int(tr_loss.get("trade_count", 0)),
                "retention": float(tr_ret),
                "delta_expectancy_vs_baseline": float(tr_cmp.get("delta_expectancy", np.nan)),
                "delta_cvar_vs_baseline": float(tr_cmp.get("delta_cvar_5", np.nan)),
                "maxdd_improve_ratio": float(tr_cmp.get("maxdd_improve_ratio", np.nan)),
                "fast_loser_rel_reduction": float(tr_fast_rel),
                "bottom_decile_rel_reduction": float(tr_tail_rel),
                "parity_clean": int(tr_chrono.get("parity_clean", 0)),
                "same_bar_exit_count": int(tr_chrono.get("same_bar_exit_count", 0)),
                "exit_before_entry_count": int(tr_chrono.get("exit_before_entry_count", 0)),
                "entry_on_signal_count": int(tr_chrono.get("entry_on_signal_count", 0)),
                **train_size_stats,
            }
        )

        # Stage 2: strict confirm on full + routes + holdout.
        full_rows, full_size_stats = apply_sizing_policy(baseline_full_rows, pol)
        full_cmp = ga.compare_candidate_vs_baseline(full_rows, baseline_full_rows)
        full_loss = ga.compute_loss_metrics(full_rows)
        full_chrono = ga.compute_chronology_stats(full_rows)
        full_agg = modela.ga_exec._aggregate_rows(full_rows)  # pylint: disable=protected-access
        full_roll = modela.ga_exec._rollup_mode(full_rows, "exec")  # pylint: disable=protected-access

        split_df, split_support_pass, min_split_delta = split_detail_and_gate(full_rows)
        if not split_df.empty:
            split_df = split_df.copy()
            split_df["variant_id"] = variant_id
            detail_rows.extend(split_df.to_dict("records"))

        route_df, route_support_pass, route_pass_rate, route_total = route_detail_and_gate(
            variant=pol,
            route_baseline_rows=route_baseline_rows,
        )
        if not route_df.empty:
            route_df = route_df.copy()
            route_df["variant_id"] = variant_id
            detail_rows.extend(route_df.to_dict("records"))

        holdout_cand = subset_by_signal_ids(full_rows, holdout_signal_ids)
        holdout_base = subset_by_signal_ids(baseline_full_rows, holdout_signal_ids)
        holdout_cmp = ga.compare_candidate_vs_baseline(holdout_cand, holdout_base) if (not holdout_cand.empty and not holdout_base.empty) else {
            "delta_expectancy": float("nan"),
            "delta_cvar_5": float("nan"),
            "maxdd_improve_ratio": float("nan"),
        }
        holdout_loss = ga.compute_loss_metrics(holdout_cand) if not holdout_cand.empty else {
            "fast_loser_rate": float("nan"),
            "bottom_decile_pnl_share": float("nan"),
        }
        holdout_base_loss = ga.compute_loss_metrics(holdout_base) if not holdout_base.empty else {
            "fast_loser_rate": float("nan"),
            "bottom_decile_pnl_share": float("nan"),
        }
        holdout_fast_rel = ga.relative_reduction(
            float(holdout_base_loss.get("fast_loser_rate", np.nan)),
            float(holdout_loss.get("fast_loser_rate", np.nan)),
        )
        holdout_tail_rel = ga.relative_reduction(
            float(holdout_base_loss.get("bottom_decile_pnl_share", np.nan)),
            float(holdout_loss.get("bottom_decile_pnl_share", np.nan)),
        )

        # Hard gates.
        g0 = int(
            int(full_chrono.get("parity_clean", 0)) == 1
            and int(full_chrono.get("same_bar_exit_count", 0)) == 0
            and int(full_chrono.get("exit_before_entry_count", 0)) == 0
            and int(full_chrono.get("entry_on_signal_count", 0)) == 0
        )
        retention = float(full_loss.get("trade_count", 0) / max(1, baseline_full_loss.get("trade_count", 0)))
        g1 = int(retention >= float(args.retention_floor))
        g2 = int(int(split_support_pass) == 1 and np.isfinite(route_pass_rate) and abs(float(route_pass_rate) - 1.0) <= 1e-12)
        g3 = int(
            np.isfinite(float(holdout_cmp.get("delta_expectancy", np.nan)))
            and float(holdout_cmp["delta_expectancy"]) >= 0.0
            and np.isfinite(float(holdout_cmp.get("delta_cvar_5", np.nan)))
            and float(holdout_cmp["delta_cvar_5"]) >= 0.0
        )
        full_bottom_decile_rel = ga.relative_reduction(
            float(baseline_full_loss.get("bottom_decile_pnl_share", np.nan)),
            float(full_loss.get("bottom_decile_pnl_share", np.nan)),
        )
        g4 = int(
            (np.isfinite(float(full_cmp.get("delta_cvar_5", np.nan))) and float(full_cmp["delta_cvar_5"]) >= float(cvar_improve_abs_threshold))
            or (np.isfinite(float(full_cmp.get("maxdd_improve_ratio", np.nan))) and float(full_cmp["maxdd_improve_ratio"]) >= float(args.tail_maxdd_improve_ratio))
            or (np.isfinite(float(full_bottom_decile_rel)) and float(full_bottom_decile_rel) >= float(args.tail_bottom_decile_rel))
        )
        confirm_pass = int(g0 == 1 and g1 == 1 and g2 == 1 and g3 == 1 and g4 == 1)

        confirm_rows.append(
            {
                "variant_id": variant_id,
                "label": str(pol.get("label", "")),
                "policy_family": str(pol.get("policy_family", "")),
                "policy_json": json.dumps(pol, sort_keys=True),
                "confirm_pass": int(confirm_pass),
                "gate_g0_chronology": int(g0),
                "gate_g1_participation": int(g1),
                "gate_g2_robustness": int(g2),
                "gate_g3_holdout": int(g3),
                "gate_g4_tail_objective": int(g4),
                "split_support_pass": int(split_support_pass),
                "best_min_subperiod_delta": float(min_split_delta),
                "route_total": int(route_total),
                "route_pass_rate": float(route_pass_rate),
                "route_support_pass": int(route_support_pass),
                "holdout_trade_count": int(len(holdout_cand)),
                "holdout_delta_expectancy_vs_baseline": float(holdout_cmp.get("delta_expectancy", np.nan)),
                "holdout_delta_cvar_vs_baseline": float(holdout_cmp.get("delta_cvar_5", np.nan)),
                "holdout_maxdd_improve_ratio": float(holdout_cmp.get("maxdd_improve_ratio", np.nan)),
                "holdout_fast_loser_rel_reduction": float(holdout_fast_rel),
                "holdout_bottom_decile_rel_reduction": float(holdout_tail_rel),
                "trade_count": int(full_loss.get("trade_count", 0)),
                "retention": float(retention),
                "expectancy_net": float(full_cmp.get("candidate_expectancy_net", np.nan)),
                "delta_expectancy_vs_baseline": float(full_cmp.get("delta_expectancy", np.nan)),
                "delta_expectancy_vs_1h_reference": float(full_agg.get("delta_expectancy_exec_minus_baseline", np.nan)),
                "cvar_5": float(full_cmp.get("candidate_cvar_5", np.nan)),
                "delta_cvar_vs_baseline": float(full_cmp.get("delta_cvar_5", np.nan)),
                "cvar_improve_ratio_vs_1h_reference": float(full_agg.get("cvar_improve_ratio", np.nan)),
                "max_drawdown": float(full_cmp.get("candidate_max_drawdown", np.nan)),
                "maxdd_improve_ratio": float(full_cmp.get("maxdd_improve_ratio", np.nan)),
                "maxdd_improve_ratio_vs_1h_reference": float(full_agg.get("maxdd_improve_ratio", np.nan)),
                "instant_loser_rate": float(full_loss.get("instant_loser_rate", np.nan)),
                "fast_loser_rate": float(full_loss.get("fast_loser_rate", np.nan)),
                "bottom_decile_pnl_share": float(full_loss.get("bottom_decile_pnl_share", np.nan)),
                "worst_10_trades_sum": float(full_loss.get("worst_10_trades_sum", np.nan)),
                "worst_25_trades_sum": float(full_loss.get("worst_25_trades_sum", np.nan)),
                "instant_loser_rel_reduction": float(
                    ga.relative_reduction(float(baseline_full_loss.get("instant_loser_rate", np.nan)), float(full_loss.get("instant_loser_rate", np.nan)))
                ),
                "fast_loser_rel_reduction": float(
                    ga.relative_reduction(float(baseline_full_loss.get("fast_loser_rate", np.nan)), float(full_loss.get("fast_loser_rate", np.nan)))
                ),
                "bottom_decile_rel_reduction": float(full_bottom_decile_rel),
                "parity_clean": int(full_chrono.get("parity_clean", 0)),
                "same_bar_exit_count": int(full_chrono.get("same_bar_exit_count", 0)),
                "same_bar_touch_count": int(full_chrono.get("same_bar_touch_count", 0)),
                "exit_before_entry_count": int(full_chrono.get("exit_before_entry_count", 0)),
                "entry_on_signal_count": int(full_chrono.get("entry_on_signal_count", 0)),
                "signals_total": int(full_roll.get("signals_total", 0)),
                "entries_valid": int(full_roll.get("entries_valid", 0)),
                "entry_rate": float(full_roll.get("entry_rate", np.nan)),
                "taker_share": float(full_roll.get("taker_share", np.nan)),
                "median_fill_delay_min": float(full_roll.get("median_fill_delay_min", np.nan)),
                "p95_fill_delay_min": float(full_roll.get("p95_fill_delay_min", np.nan)),
                **full_size_stats,
            }
        )

    screen_df = pd.DataFrame(screen_rows).sort_values(
        ["screen_score", "delta_expectancy_vs_baseline"],
        ascending=[False, False],
    ).reset_index(drop=True)
    confirm_df = pd.DataFrame(confirm_rows).sort_values(
        ["confirm_pass", "delta_expectancy_vs_baseline", "delta_cvar_vs_baseline", "maxdd_improve_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    detail_df = pd.DataFrame(detail_rows)

    confirm_df.to_csv(run_dir / "sol_sizing_by_variant.csv", index=False)
    detail_df.to_csv(run_dir / "sol_sizing_route_split_detail.csv", index=False)

    approved_df = confirm_df[(confirm_df["confirm_pass"] == 1) & (confirm_df["variant_id"].astype(str) != "V0_BASELINE")].copy()
    decision = "APPROVED_SIZING_POLICY" if not approved_df.empty else "NO_CHANGE_RECOMMENDED"
    best_row = approved_df.iloc[0] if not approved_df.empty else confirm_df.iloc[0]

    baseline_table = pd.DataFrame(
        [
            {
                "metric": "baseline_expectancy_net",
                "value": float(baseline_full_roll["mean_expectancy_net"]),
            },
            {
                "metric": "baseline_cvar_5",
                "value": float(baseline_full_roll["cvar_5"]),
            },
            {
                "metric": "baseline_max_drawdown",
                "value": float(baseline_full_roll["max_drawdown"]),
            },
            {
                "metric": "baseline_trade_count",
                "value": int(baseline_full_loss["trade_count"]),
            },
            {
                "metric": "baseline_bottom_decile_pnl_share",
                "value": float(baseline_full_loss["bottom_decile_pnl_share"]),
            },
            {
                "metric": "baseline_same_bar_exit_count",
                "value": int(baseline_full_chrono["same_bar_exit_count"]),
            },
            {
                "metric": "baseline_route_pass_rate_strict_lineage",
                "value": float(strict_route_pass),
            },
            {
                "metric": "tail_cvar_abs_improve_threshold",
                "value": float(cvar_improve_abs_threshold),
            },
        ]
    )
    baseline_table.to_csv(run_dir / "sol_sizing_baseline_metrics.csv", index=False)

    report_lines: List[str] = [
        "# SOL Sizing / Risk-Per-Trade Bounded Sweep",
        "",
        f"- Generated UTC: `{utc_now_iso()}`",
        f"- Run dir: `{run_dir}`",
        f"- Decision: `{decision}`",
        f"- Symbol: `{SYMBOL}`",
        f"- Baseline strategy lock: `{BASELINE_STRATEGY_ID}`",
        "",
        "## Causal Feature Set Used",
        "",
        f"- `{FEATURE_RANGE}` (signal-bar range fraction)",
        f"- `{FEATURE_WICK}` (signal-bar upper-wick ratio)",
        "- Both are generated in the Model A signal simulation at decision time from the signal bar only.",
        "",
        "## Baseline Snapshot",
        "",
        markdown_table(
            baseline_table,
            ["metric", "value"],
            n=20,
        ),
        "",
        "## Stage 1 Screen (Train 60%)",
        "",
        markdown_table(
            screen_df,
            [
                "variant_id",
                "policy_family",
                "screen_score",
                "retention",
                "delta_expectancy_vs_baseline",
                "delta_cvar_vs_baseline",
                "maxdd_improve_ratio",
                "bottom_decile_rel_reduction",
                "size_mult_mean",
                "size_mult_reduced_share",
                "parity_clean",
            ],
            n=20,
        ),
        "",
        "## Stage 2 Confirm (Splits + 3 Routes + Holdout)",
        "",
        markdown_table(
            confirm_df,
            [
                "variant_id",
                "confirm_pass",
                "gate_g0_chronology",
                "gate_g1_participation",
                "gate_g2_robustness",
                "gate_g3_holdout",
                "gate_g4_tail_objective",
                "split_support_pass",
                "route_pass_rate",
                "holdout_delta_expectancy_vs_baseline",
                "holdout_delta_cvar_vs_baseline",
                "retention",
                "delta_expectancy_vs_baseline",
                "delta_cvar_vs_baseline",
                "maxdd_improve_ratio",
                "bottom_decile_rel_reduction",
                "size_mult_mean",
                "size_mult_reduced_share",
            ],
            n=30,
        ),
        "",
        "## Proven vs Assumed",
        "",
        "- Proven: signal generation, entries, exits, chronology, and costs remain unchanged from baseline; only per-trade pnl scaling is applied.",
        "- Proven: participation and entry timing remained unchanged (retention ~1.0; no skip-mask or entry timing mutation).",
        "- Proven: strict robustness checks were applied using split + full 3-route family + holdout.",
        "- Assumed: feature columns are reliably present for all signals in this frozen branch (missing values fall back to neutral size=1.0).",
        "",
        "## Final Recommendation",
        "",
        f"- `{decision}`",
    ]

    if decision == "APPROVED_SIZING_POLICY":
        report_lines.append(f"- Approved variant: `{best_row.get('variant_id', '')}`")
        report_lines.append("- Next step: run op-check and paper SHADOW with the approved sizing policy before active promotion.")
    else:
        report_lines.append("- No non-baseline sizing policy passed all strict gates.")
        report_lines.append("- Next step: keep current SOL baseline sizing unchanged.")

    (run_dir / "sol_sizing_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "generated_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "decision": decision,
        "symbol": SYMBOL,
        "baseline_strategy_id": BASELINE_STRATEGY_ID,
        "policies_tested": policies,
        "features_used": [FEATURE_RANGE, FEATURE_WICK],
        "gates": {
            "g0_chronology": "parity_clean==1 and same_bar_exit_count==0 and exit_before_entry_count==0 and entry_on_signal_count==0",
            "g1_participation": f"retention>={float(args.retention_floor):.4f}",
            "g2_robustness": "split_support_pass==1 and route_pass_rate==1.0",
            "g3_holdout": "holdout_delta_expectancy_vs_baseline>=0 and holdout_delta_cvar_vs_baseline>=0",
            "g4_tail_objective": (
                f"delta_cvar_vs_baseline>={float(cvar_improve_abs_threshold):.8f} "
                f"OR maxdd_improve_ratio>={float(args.tail_maxdd_improve_ratio):.4f} "
                f"OR bottom_decile_rel_reduction>={float(args.tail_bottom_decile_rel):.4f}"
            ),
        },
        "input_paths": {
            "posture_dir": str(inputs["posture_dir"]),
            "strict_confirm_dir": str(inputs["strict_confirm_dir"]),
            "freeze_dir": str(inputs["freeze_dir"]),
            "foundation_dir": str(inputs["foundation_dir"]),
        },
        "contract_validation": contract_validation,
        "build_meta": build_meta,
        "route_meta": route_meta,
        "strict_baseline_route_pass_rate": float(strict_route_pass),
        "outputs": {
            "by_variant_csv": str(run_dir / "sol_sizing_by_variant.csv"),
            "route_split_detail_csv": str(run_dir / "sol_sizing_route_split_detail.csv"),
            "report_md": str(run_dir / "sol_sizing_report.md"),
        },
    }
    json_dump(run_dir / "sol_sizing_manifest.json", manifest)

    if decision == "APPROVED_SIZING_POLICY":
        best_payload = {
            "generated_utc": utc_now_iso(),
            "decision": decision,
            "symbol": SYMBOL,
            "best_variant": best_row.to_dict(),
            "run_dir": str(run_dir),
        }
        json_dump(run_dir / "sol_sizing_best_policy.json", best_payload)

    print(json.dumps({"decision": decision, "run_dir": str(run_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
