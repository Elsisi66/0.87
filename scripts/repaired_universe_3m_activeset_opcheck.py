#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_v_multicoin_model_a_audit as phase_v  # noqa: E402
from scripts import repaired_universe_3m_exec_subset1_confirm as confirm  # noqa: E402


RUN_PREFIX = "REPAIRED_UNIVERSE_3M_ACTIVESET_OPCHECK"
ACTIVE_SYMBOLS = ["SOLUSDT", "LTCUSDT"]
SHADOW_SYMBOL = "NEARUSDT"
RECON_TOL = 1e-12


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_tag() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


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
                vals.append(f"{v:.10g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def find_latest_complete(root: Path, pattern: str, required: List[str]) -> Optional[Path]:
    cands = sorted([p for p in root.glob(pattern) if p.is_dir()], key=lambda p: p.name)
    for cand in reversed(cands):
        if all((cand / req).exists() for req in required):
            return cand.resolve()
    return None


def discover_confirm_dir(arg_value: str) -> Path:
    if arg_value:
        run_dir = Path(arg_value).resolve()
    else:
        exec_root = PROJECT_ROOT / "reports" / "execution_layer"
        run_dir = find_latest_complete(
            exec_root,
            "REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_*",
            [
                "repaired_subset1_confirm_by_symbol.csv",
                "repaired_subset1_confirm_summary.csv",
                "repaired_subset1_confirm_report.md",
                "repaired_subset1_confirm_manifest.json",
            ],
        )
        if run_dir is None:
            raise FileNotFoundError("Missing latest complete REPAIRED_UNIVERSE_3M_EXEC_SUBSET1_CONFIRM_* directory")
    return run_dir


def discover_subset_eval_dir(arg_value: str) -> Path:
    return confirm.discover_subset_eval_dir(arg_value)


def build_cost_heavy_args(exec_args: argparse.Namespace) -> argparse.Namespace:
    x = copy.deepcopy(exec_args)
    x.fee_bps_maker = float(x.fee_bps_maker) + 1.0
    x.fee_bps_taker = float(x.fee_bps_taker) + 2.0
    x.slippage_bps_limit = float(x.slippage_bps_limit) * 2.0
    x.slippage_bps_market = float(x.slippage_bps_market) * 1.5
    return x


def safe_float(x: Any) -> float:
    return float(pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0])


def eval_single_variant(
    *,
    symbol: str,
    bundle: Any,
    qrow: Dict[str, Any],
    rrow: Dict[str, Any],
    exec_args: argparse.Namespace,
    variant: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    pack = phase_v.evaluate_symbol(
        symbol=symbol,
        bundle=bundle,
        foundation_quality=qrow,
        foundation_readiness=rrow,
        exec_args=exec_args,
        variants=[variant],
    )
    rows = pack["results_df"].copy()
    best = phase_v.choose_best_candidate(rows)
    return rows, best, pack


def classify_symbol(
    *,
    deterministic_ok: bool,
    cost_stress_ok: bool,
    strict_row: pd.Series,
    build_meta: Dict[str, Any],
    source_mode: str,
) -> Tuple[str, str]:
    if not deterministic_ok:
        return "BLOCKED", "reconstruction_mismatch_vs_strict_confirm"
    if int(build_meta.get("signals_missing_3m_data", 0)) > 0 or int(build_meta.get("signals_partial_3m_data", 0)) > 0:
        return "BLOCKED", "coverage_not_fully_clean"
    if not cost_stress_ok:
        return "ACTIVE_BUT_SHADOW_FIRST", "winner_fails_cost_heavy_stress"
    if source_mode == "foundation_cached_windows":
        return "ACTIVE_BUT_SHADOW_FIRST", "depends_on_merged_foundation_cache"
    return "ACTIVE_DEPLOYABLE_SUBSET", "deterministic_and_cost_resilient"


def final_recommendation(by_symbol_df: pd.DataFrame) -> str:
    classes = by_symbol_df["classification"].astype(str).tolist()
    if all(c == "ACTIVE_DEPLOYABLE_SUBSET" for c in classes):
        return "FREEZE_ACTIVE_2_SYMBOL_SUBSET"
    if any(c == "BLOCKED" for c in classes):
        return "BLOCK_DEPLOYMENT_AND_REVIEW"
    return "FREEZE_SHADOW_FIRST_ONLY"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Operational / deployment check for repaired 3m active subset")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--freeze-dir", default="")
    ap.add_argument("--confirm-dir", default="")
    ap.add_argument("--subset-eval-dir", default="")
    ap.add_argument("--foundation-dir", default="")
    ap.add_argument("--seed", type=int, default=20260304)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    freeze_dir = confirm.discover_freeze_dir(args.freeze_dir)
    confirm_dir = discover_confirm_dir(args.confirm_dir)
    subset_eval_dir = discover_subset_eval_dir(args.subset_eval_dir)

    confirm_by_symbol_fp = confirm_dir / "repaired_subset1_confirm_by_symbol.csv"
    confirm_summary_fp = confirm_dir / "repaired_subset1_confirm_summary.csv"
    confirm_manifest_fp = confirm_dir / "repaired_subset1_confirm_manifest.json"
    confirm_route_fp = confirm_dir / "repaired_subset1_confirm_route_checks.csv"

    strict_df = pd.read_csv(confirm_by_symbol_fp)
    strict_df["symbol"] = strict_df["symbol"].astype(str).str.upper()
    strict_active = strict_df[strict_df["classification"].astype(str) == "ACTIVE_3M_SURVIVOR"]["symbol"].tolist()
    if sorted(strict_active) != sorted(ACTIVE_SYMBOLS):
        raise RuntimeError(f"Strict-confirm active set mismatch. Expected {sorted(ACTIVE_SYMBOLS)}, got {sorted(strict_active)}")
    near_row = strict_df[strict_df["symbol"] == SHADOW_SYMBOL]
    if near_row.empty or str(near_row.iloc[0]["classification"]) != "SHADOW_ONLY":
        raise RuntimeError("NEARUSDT is not preserved as SHADOW_ONLY in strict confirm artifacts")

    confirm_manifest = json.loads(confirm_manifest_fp.read_text(encoding="utf-8"))
    foundation_dir = Path(args.foundation_dir).resolve() if str(args.foundation_dir).strip() else Path(str(confirm_manifest["foundation_dir"])).resolve()
    foundation_state = phase_v.load_foundation_state(foundation_dir)

    universe_fp = freeze_dir / "repaired_best_by_symbol.csv"
    selected_params_dir = freeze_dir / "repaired_universe_selected_params"
    universe_df = pd.read_csv(universe_fp)
    universe_df["symbol"] = universe_df["symbol"].astype(str).str.upper()
    active_df = universe_df[universe_df["symbol"].isin(ACTIVE_SYMBOLS)].copy()
    if len(active_df) != len(ACTIVE_SYMBOLS):
        missing = sorted(set(ACTIVE_SYMBOLS) - set(active_df["symbol"].tolist()))
        raise RuntimeError(f"Frozen repaired universe missing active symbols: {missing}")
    active_df = active_df.set_index("symbol").loc[ACTIVE_SYMBOLS].reset_index()

    run_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = run_root / f"{RUN_PREFIX}_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    inputs_dir = confirm.ensure_dir(run_dir / "_inputs")
    cache_dir = confirm.ensure_dir(run_dir / "_window_cache")
    active_params_dir = confirm.ensure_dir(run_dir / "repaired_active_3m_params")

    df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    raw_cache: Dict[str, pd.DataFrame] = {}
    signal_map: Dict[str, pd.DataFrame] = {}
    all_signals: List[pd.DataFrame] = []

    for row in active_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        params = confirm.parse_params_from_row(row_s, selected_params_dir)
        sig_df = confirm.build_signal_table_for_row(row=row_s, params=params, df_cache=df_cache, raw_cache=raw_cache)
        symbol = str(row_s["symbol"]).upper()
        signal_map[symbol] = sig_df
        all_signals.append(sig_df)

    combined_signals = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    combined_signals.to_csv(inputs_dir / "repaired_activeset_signal_timeline.csv", index=False)
    combined_signals.to_csv(inputs_dir / "universe_signal_timeline.csv", index=False)

    exec_args = phase_v.build_exec_args(
        foundation_state=phase_v.FoundationState(
            root=inputs_dir,
            signal_timeline=combined_signals,
            download_manifest=pd.DataFrame(),
            quality=pd.DataFrame(),
            readiness=pd.DataFrame(),
        ),
        seed=int(args.seed),
    )
    contract_validation = phase_v.build_contract_validation(exec_args=exec_args, run_dir=run_dir)
    stress_args = build_cost_heavy_args(exec_args)
    variant_lookup = {str(v["candidate_id"]): dict(v) for v in phase_v.sanitize_variants()}

    by_rows: List[Dict[str, Any]] = []
    symbol_meta: Dict[str, Any] = {}

    for row in active_df.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        symbol = str(row_s["symbol"]).upper()
        strict_row = strict_df[strict_df["symbol"] == symbol].iloc[0]
        chosen_id = str(strict_row["best_candidate_id"]).strip()
        if chosen_id not in variant_lookup:
            raise RuntimeError(f"Missing winner config in variant family: {chosen_id}")
        variant = dict(variant_lookup[chosen_id])

        params = confirm.parse_params_from_row(row_s, selected_params_dir)
        sig_df = signal_map[symbol]
        symbol_windows = confirm.build_window_pool_for_symbol(
            symbol=symbol,
            signal_df=sig_df,
            foundation_state=foundation_state,
            cache_dir=cache_dir,
            reuse_cache_dirs=[],
        )
        bundle, build_meta = phase_v.build_symbol_bundle(
            symbol=symbol,
            symbol_signals=sig_df,
            symbol_windows=symbol_windows,
            exec_args=exec_args,
            run_dir=run_dir,
        )
        qrow, rrow = confirm.quality_from_build_meta(build_meta)

        base_rows, base_best, base_pack = eval_single_variant(
            symbol=symbol,
            bundle=bundle,
            qrow=qrow,
            rrow=rrow,
            exec_args=exec_args,
            variant=variant,
        )
        stress_rows, stress_best, stress_pack = eval_single_variant(
            symbol=symbol,
            bundle=bundle,
            qrow=qrow,
            rrow=rrow,
            exec_args=stress_args,
            variant=variant,
        )

        strict_exp = safe_float(strict_row["confirmed_exec_expectancy_net"])
        recon_exp = safe_float(base_best.get("exec_expectancy_net", np.nan)) if not base_best.empty else float("nan")
        recon_delta_abs = abs(recon_exp - strict_exp) if np.isfinite(recon_exp) and np.isfinite(strict_exp) else float("nan")
        candidate_match = int(str(base_best.get("candidate_id", "")) == chosen_id) if not base_best.empty else 0
        route_match = int(int(base_best.get("route_pass", 0)) == int(strict_row["route_pass"])) if not base_best.empty else 0
        route_count_match = int(int(base_best.get("route_count", 0)) == int(strict_row["route_count"])) if not base_best.empty else 0
        deterministic_ok = bool(
            candidate_match == 1
            and route_match == 1
            and route_count_match == 1
            and np.isfinite(recon_delta_abs)
            and recon_delta_abs <= RECON_TOL
        )

        base_delta_vs_ref = safe_float(base_best.get("delta_expectancy_vs_1h_reference", np.nan)) if not base_best.empty else float("nan")
        base_route_pass = int(base_best.get("route_pass", 0)) if not base_best.empty else 0
        base_valid = int(base_best.get("valid_for_ranking", 0)) if not base_best.empty else 0
        stress_exp = safe_float(stress_best.get("exec_expectancy_net", np.nan)) if not stress_best.empty else float("nan")
        stress_delta_vs_ref = safe_float(stress_best.get("delta_expectancy_vs_1h_reference", np.nan)) if not stress_best.empty else float("nan")
        stress_route_pass = int(stress_best.get("route_pass", 0)) if not stress_best.empty else 0
        stress_valid = int(stress_best.get("valid_for_ranking", 0)) if not stress_best.empty else 0
        stress_entry_rate = safe_float(stress_best.get("entry_rate", np.nan)) if not stress_best.empty else float("nan")
        cost_stress_ok = bool(
            stress_valid == 1
            and stress_route_pass == 1
            and np.isfinite(stress_delta_vs_ref)
            and stress_delta_vs_ref > 0.0
        )
        cost_sensitivity_outcome = (
            "passes_cost_heavy_positive_route_clean" if cost_stress_ok else "fails_cost_heavy_positive_delta_or_route"
        )
        coverage_outcome = "clean"
        if int(build_meta.get("signals_missing_3m_data", 0)) > 0 or int(build_meta.get("signals_partial_3m_data", 0)) > 0:
            coverage_outcome = "coverage_degraded"
        elif str(symbol_meta.get(symbol, {}).get("foundation_window_mode", "")) == "foundation_cached_windows":
            coverage_outcome = "clean_merged_foundation_cache"

        source_mode = "local_full_3m" if confirm.local_full_3m_path(symbol) is not None else "foundation_cached_windows"
        if coverage_outcome == "clean":
            coverage_outcome = "clean_local_full_3m" if source_mode == "local_full_3m" else "clean_merged_foundation_cache"

        classification, reason = classify_symbol(
            deterministic_ok=deterministic_ok,
            cost_stress_ok=cost_stress_ok,
            strict_row=strict_row,
            build_meta=build_meta,
            source_mode=source_mode,
        )

        blockers: List[str] = []
        if not deterministic_ok:
            blockers.append("reconstruction_mismatch")
        if not cost_stress_ok:
            blockers.append("cost_heavy_fragility")
        if coverage_outcome != "clean_local_full_3m":
            blockers.append("non_local_3m_data_path")

        by_rows.append(
            {
                "symbol": symbol,
                "strict_confirm_candidate_id": chosen_id,
                "classification": classification,
                "classification_reason": reason,
                "strict_confirm_expectancy": strict_exp,
                "reconstructed_expectancy": recon_exp,
                "reconstruction_delta_abs": recon_delta_abs,
                "reconstruction_tolerance_ok": int(deterministic_ok),
                "base_delta_vs_repaired_1h": base_delta_vs_ref,
                "base_route_pass": base_route_pass,
                "base_valid_for_ranking": base_valid,
                "cost_heavy_expectancy": stress_exp,
                "cost_heavy_delta_vs_repaired_1h": stress_delta_vs_ref,
                "cost_heavy_route_pass": stress_route_pass,
                "cost_heavy_valid_for_ranking": stress_valid,
                "cost_heavy_entry_rate": stress_entry_rate,
                "cost_sensitivity_outcome": cost_sensitivity_outcome,
                "coverage_outcome": coverage_outcome,
                "signals_total": int(build_meta.get("signals_total", 0)),
                "signals_with_3m_data": int(build_meta.get("signals_with_3m_data", 0)),
                "signals_missing_3m_data": int(build_meta.get("signals_missing_3m_data", 0)),
                "signals_partial_3m_data": int(build_meta.get("signals_partial_3m_data", 0)),
                "window_source_mode": source_mode,
                "window_source_counts_json": json.dumps(build_meta.get("window_source_counts", {}), sort_keys=True),
                "route_family_supported": int(base_pack["route_meta"].get("route_family_supported", 0)),
                "route_count": int(base_pack["route_meta"].get("route_count", 0)),
                "operational_blockers": "|".join(blockers) if blockers else "none",
            }
        )

        active_payload = {
            "symbol": symbol,
            "strict_confirm_candidate_id": chosen_id,
            "model_a_variant": variant,
            "repaired_1h_params_source": str(row_s.get("params_source", "")),
            "repaired_1h_params": params,
            "strict_confirm_artifact": str(confirm_by_symbol_fp),
            "opcheck_classification": classification,
            "opcheck_reason": reason,
        }
        confirm.json_dump(active_params_dir / f"{symbol}_active_3m_pack.json", active_payload)

        symbol_meta[symbol] = {
            "params_source": str(row_s.get("params_source", "")),
            "strict_confirm_candidate_id": chosen_id,
            "base_route_meta": base_pack["route_meta"],
            "stress_route_meta": stress_pack["route_meta"],
            "build_meta": build_meta,
            "source_mode": source_mode,
            "variant": variant,
        }

    by_symbol_df = pd.DataFrame(by_rows).set_index("symbol").loc[ACTIVE_SYMBOLS].reset_index()
    recommendation = final_recommendation(by_symbol_df)
    active_keep = by_symbol_df[by_symbol_df["classification"] == "ACTIVE_DEPLOYABLE_SUBSET"].copy()
    shadow_extra = by_symbol_df[by_symbol_df["classification"].isin(["ACTIVE_BUT_SHADOW_FIRST", "SHADOW_ONLY"])].copy()
    shadow_row = near_row.copy()
    shadow_row = shadow_row.assign(
        strict_status="SHADOW_ONLY_FROM_STRICT_CONFIRM",
        freeze_role="shadow_only_reference",
    )

    summary_rows = [
        {"metric": "symbols_evaluated", "value": int(len(by_symbol_df))},
        {"metric": "active_deployable_count", "value": int((by_symbol_df["classification"] == "ACTIVE_DEPLOYABLE_SUBSET").sum())},
        {"metric": "active_but_shadow_first_count", "value": int((by_symbol_df["classification"] == "ACTIVE_BUT_SHADOW_FIRST").sum())},
        {"metric": "blocked_count", "value": int((by_symbol_df["classification"] == "BLOCKED").sum())},
        {"metric": "recommendation", "value": recommendation},
        {"metric": "active_symbols", "value": ",".join(active_keep["symbol"].astype(str).tolist())},
        {"metric": "shadow_symbols", "value": ",".join(([SHADOW_SYMBOL] + shadow_extra["symbol"].astype(str).tolist()))},
    ]
    summary_df = pd.DataFrame(summary_rows)

    by_fp = run_dir / "repaired_activeset_opcheck_by_symbol.csv"
    summary_fp = run_dir / "repaired_activeset_opcheck_summary.csv"
    report_fp = run_dir / "repaired_activeset_opcheck_report.md"
    active_fp = run_dir / "repaired_active_3m_subset.csv"
    shadow_fp = run_dir / "repaired_shadow_3m_subset.csv"

    by_symbol_df.to_csv(by_fp, index=False)
    summary_df.to_csv(summary_fp, index=False)

    if not active_keep.empty:
        active_keep.loc[:, [
            "symbol",
            "strict_confirm_candidate_id",
            "strict_confirm_expectancy",
            "reconstructed_expectancy",
            "cost_heavy_expectancy",
            "classification",
            "classification_reason",
        ]].rename(columns={"strict_confirm_candidate_id": "candidate_id"}).to_csv(active_fp, index=False)
    else:
        pd.DataFrame(columns=["symbol", "candidate_id", "strict_confirm_expectancy", "reconstructed_expectancy", "cost_heavy_expectancy", "classification", "classification_reason"]).to_csv(active_fp, index=False)

    shadow_parts = [shadow_row]
    if not shadow_extra.empty:
        shadow_parts.append(
            shadow_extra.loc[:, ["symbol", "strict_confirm_candidate_id", "classification", "classification_reason"]].rename(
                columns={"strict_confirm_candidate_id": "candidate_id"}
            )
        )
    pd.concat(shadow_parts, ignore_index=True).to_csv(shadow_fp, index=False)

    lines: List[str] = []
    lines.append("# Repaired Universe 3m Active-Set Operational Check")
    lines.append("")
    lines.append("This is a bounded operational/deployment check on the earned repaired-branch selective 3m subset only. It excludes discovery, optimization, and any held-back symbols.")
    lines.append("")
    lines.append("## Inputs Used")
    lines.append(f"- Frozen repaired universe dir: `{freeze_dir}`")
    lines.append(f"- Strict confirm dir: `{confirm_dir}`")
    lines.append(f"- Prior bounded subset dir: `{subset_eval_dir}`")
    lines.append(f"- Foundation dir: `{foundation_dir}`")
    lines.append("")
    lines.append("## Cost Stress Definition")
    lines.append(f"- base fee/slip: maker `{float(exec_args.fee_bps_maker):.4f}` / taker `{float(exec_args.fee_bps_taker):.4f}` / limit slip `{float(exec_args.slippage_bps_limit):.4f}` / market slip `{float(exec_args.slippage_bps_market):.4f}`")
    lines.append(f"- cost-heavy stress: maker `{float(stress_args.fee_bps_maker):.4f}` / taker `{float(stress_args.fee_bps_taker):.4f}` / limit slip `{float(stress_args.slippage_bps_limit):.4f}` / market slip `{float(stress_args.slippage_bps_market):.4f}`")
    lines.append("")
    lines.append("## Per-Symbol Summary")
    lines.append(
        markdown_table(
            by_symbol_df,
            [
                "symbol",
                "strict_confirm_candidate_id",
                "strict_confirm_expectancy",
                "reconstructed_expectancy",
                "reconstruction_delta_abs",
                "cost_heavy_expectancy",
                "cost_heavy_delta_vs_repaired_1h",
                "coverage_outcome",
                "classification",
                "operational_blockers",
            ],
            n=len(by_symbol_df),
        )
    )
    lines.append("")
    lines.append("## Proven vs Assumed")
    lines.append("- Proven: both active names were rebuilt from the frozen repaired universe params and re-evaluated with the strict route-enabled 3m path.")
    lines.append("- Proven: deterministic reconstruction was checked against the exact strict-confirm winner identity and headline expectancy.")
    lines.append("- Proven: cost-heavy stress kept entry/exit mechanics unchanged and only raised fees/slippage.")
    lines.append("- Assumed: the universal foundation remains an acceptable 3m data pool for deterministic historical reconstruction where no local full 3m parquet exists.")
    lines.append("")
    lines.append("## Recommendation")
    lines.append(f"- Final recommendation: `{recommendation}`")
    lines.append(f"- Active deployable symbols: `{', '.join(active_keep['symbol'].astype(str).tolist()) if not active_keep.empty else ''}`")
    lines.append(f"- Shadow set: `{SHADOW_SYMBOL}{(',' + ','.join(shadow_extra['symbol'].astype(str).tolist())) if not shadow_extra.empty else ''}`")
    report_fp.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "generated_utc": utc_now().isoformat(),
        "freeze_dir": str(freeze_dir),
        "confirm_dir": str(confirm_dir),
        "subset_eval_dir": str(subset_eval_dir),
        "foundation_dir": str(foundation_dir),
        "active_symbols_requested": list(ACTIVE_SYMBOLS),
        "shadow_symbol_reference": SHADOW_SYMBOL,
        "contract_validation": contract_validation,
        "cost_stress": {
            "fee_bps_maker_base": float(exec_args.fee_bps_maker),
            "fee_bps_taker_base": float(exec_args.fee_bps_taker),
            "slippage_bps_limit_base": float(exec_args.slippage_bps_limit),
            "slippage_bps_market_base": float(exec_args.slippage_bps_market),
            "fee_bps_maker_stress": float(stress_args.fee_bps_maker),
            "fee_bps_taker_stress": float(stress_args.fee_bps_taker),
            "slippage_bps_limit_stress": float(stress_args.slippage_bps_limit),
            "slippage_bps_market_stress": float(stress_args.slippage_bps_market),
        },
        "recommendation": recommendation,
        "symbol_meta": symbol_meta,
        "artifacts": {
            "repaired_activeset_opcheck_by_symbol": str(by_fp),
            "repaired_activeset_opcheck_summary": str(summary_fp),
            "repaired_activeset_opcheck_report": str(report_fp),
            "repaired_active_3m_subset": str(active_fp),
            "repaired_shadow_3m_subset": str(shadow_fp),
            "repaired_active_3m_params": str(active_params_dir),
        },
    }
    confirm.json_dump(run_dir / "repaired_activeset_manifest.json", manifest)


if __name__ == "__main__":
    main()
