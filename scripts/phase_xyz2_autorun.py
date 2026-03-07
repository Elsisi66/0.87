#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402
from scripts import phase_u_combined_1h3m_pilot as phaseu  # noqa: E402
from scripts import sol_reconcile_truth as recon  # noqa: E402


LOCKED = {
    "symbol": "SOLUSDT",
    "representative_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
    "primary_hash": "862c940746de0da984862d95",
    "backup_hash": "992bd371689ba3936f3b4d09",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def to_num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


def spearman_rank_corr(a: Sequence[float], b: Sequence[float]) -> float:
    xa = np.asarray(a, dtype=float)
    xb = np.asarray(b, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(xb)
    xa = xa[mask]
    xb = xb[mask]
    if xa.size < 3:
        return float("nan")
    ra = pd.Series(xa).rank(method="average").to_numpy(dtype=float)
    rb = pd.Series(xb).rank(method="average").to_numpy(dtype=float)
    if np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def build_exec_args(signals_csv: Path, seed: int) -> argparse.Namespace:
    parser = ga_exec.build_arg_parser()
    args = parser.parse_args([])
    args.symbol = LOCKED["symbol"]
    args.symbols = ""
    args.rank = 1
    args.signals_csv = str(signals_csv)
    args.max_signals = 1200
    args.walkforward = True
    args.wf_splits = 5
    args.train_ratio = 0.70
    args.mode = "tight"
    args.workers = 1
    args.seed = int(seed)
    args.pop = 1
    args.gens = 1
    args.execution_config = "configs/execution_configs.yaml"
    args.fee_bps_maker = 2.0
    args.fee_bps_taker = 4.0
    args.slippage_bps_limit = 0.5
    args.slippage_bps_market = 2.0
    args.canonical_fee_model_path = LOCKED["canonical_fee_model"]
    args.canonical_metrics_definition_path = LOCKED["canonical_metrics_definition"]
    args.expected_fee_model_sha256 = LOCKED["expected_fee_sha"]
    args.expected_metrics_definition_sha256 = LOCKED["expected_metrics_sha"]
    args.allow_freeze_hash_mismatch = 0
    return args


def find_latest_phasev(exec_root: Path) -> Optional[Path]:
    cands = sorted([p for p in exec_root.glob("PHASEV_BRANCHB_PORTABILITY_DD_*") if p.is_dir()], key=lambda p: p.name)
    if not cands:
        return None
    for p in reversed(cands):
        if (p / "phaseV_exec_candidates_locked.json").exists():
            return p
    return None


def load_exec_pair(exec_root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    phasev = find_latest_phasev(exec_root)
    if phasev is not None:
        cobj = json.loads((phasev / "phaseV_exec_candidates_locked.json").read_text(encoding="utf-8"))
        cands = cobj.get("candidates", {})
        if "E1" in cands and "E2" in cands:
            out = {
                "E1": {
                    "exec_choice_id": "E1",
                    "description": str(cands["E1"].get("description", "phaseS_primary")),
                    "genome_hash": str(cands["E1"].get("genome_hash", "")),
                    "genome": copy.deepcopy(cands["E1"].get("genome", {})),
                    "source_run": str(cands["E1"].get("source_run", "")),
                },
                "E2": {
                    "exec_choice_id": "E2",
                    "description": str(cands["E2"].get("description", "phaseS_backup")),
                    "genome_hash": str(cands["E2"].get("genome_hash", "")),
                    "genome": copy.deepcopy(cands["E2"].get("genome", {})),
                    "source_run": str(cands["E2"].get("source_run", "")),
                },
            }
            if out["E1"]["genome_hash"] == LOCKED["primary_hash"] and out["E2"]["genome_hash"] == LOCKED["backup_hash"]:
                return out, {"source": str(phasev), "source_type": "phaseV_exec_candidates_locked.json"}

    # fallback to Phase U helper that pulls from Phase S
    choices, meta = phaseu.load_exec_choices(exec_root)
    m = {c.exec_choice_id: c for c in choices}
    if "E1" not in m or "E2" not in m:
        raise RuntimeError("Could not load E1/E2 execution choices")
    out2 = {
        "E1": {
            "exec_choice_id": "E1",
            "description": m["E1"].description,
            "genome_hash": m["E1"].genome_hash,
            "genome": copy.deepcopy(m["E1"].genome),
            "source_run": m["E1"].source,
        },
        "E2": {
            "exec_choice_id": "E2",
            "description": m["E2"].description,
            "genome_hash": m["E2"].genome_hash,
            "genome": copy.deepcopy(m["E2"].genome),
            "source_run": m["E2"].source,
        },
    }
    return out2, {"source": meta, "source_type": "phaseU.load_exec_choices"}


def candidate_objective(row: pd.Series) -> float:
    # Execution-aware ranking objective for 1h candidate selection.
    avg_exp = float(row.get("avg_post_exec_expectancy_net", np.nan))
    avg_delta = float(row.get("avg_delta_vs_exec_baseline", np.nan))
    avg_cvar_imp = float(row.get("avg_cvar_improve_ratio", np.nan))
    avg_dd_imp = float(row.get("avg_maxdd_improve_ratio", np.nan))
    med_split = float(row.get("avg_median_split_expectancy", np.nan))
    min_split = float(row.get("worst_min_split_expectancy", np.nan))
    std_split = float(row.get("max_std_split_expectancy", np.nan))
    min_entries = float(row.get("min_entries_valid_across_exec", np.nan))
    min_entry_rate = float(row.get("min_entry_rate_across_exec", np.nan))
    max_taker = float(row.get("max_taker_share_across_exec", np.nan))
    max_p95 = float(row.get("max_p95_fill_delay_across_exec", np.nan))
    valid_both = int(row.get("valid_both_exec", 0))

    score = (
        1200.0 * avg_exp
        + 700.0 * avg_delta
        + 120.0 * avg_cvar_imp
        + 120.0 * avg_dd_imp
        + 20.0 * med_split
        + 12.0 * min_split
        - 35.0 * std_split
    )
    if valid_both == 0:
        score -= 150.0
    if not np.isfinite(min_entries) or min_entries < 200:
        score -= 80.0
    if not np.isfinite(min_entry_rate) or min_entry_rate < 0.70:
        score -= 60.0
    if not np.isfinite(max_taker) or max_taker > 0.25:
        score -= 50.0
    if not np.isfinite(max_p95) or max_p95 > 180.0:
        score -= 20.0
    if not np.isfinite(score):
        return -1e12
    return float(score)


def old_1h_score(metrics: Dict[str, Any]) -> float:
    net = float(metrics.get("net_profit", np.nan))
    pf = float(metrics.get("profit_factor", np.nan))
    dd = float(metrics.get("max_dd", np.nan))
    tr = float(metrics.get("trades", np.nan))
    if not np.isfinite(net):
        return -1e12
    score = (0.0001 * net) + (5.0 * (pf if np.isfinite(pf) else 0.0)) - (8.0 * (dd if np.isfinite(dd) else 1.0))
    if np.isfinite(tr) and tr < 30:
        score -= 5.0
    return float(score)


def run_phase_x(parent_dir: Path, seed: int, phasex_candidates: int) -> Dict[str, Any]:
    phase_dir = parent_dir / "phaseX"
    phase_dir.mkdir(parents=True, exist_ok=False)

    manifest: Dict[str, Any] = {
        "phase": "X",
        "generated_utc": utc_now(),
        "phase_dir": str(phase_dir),
        "seed": int(seed),
        "candidate_budget": int(phasex_candidates),
    }

    rep_csv = Path(LOCKED["representative_subset_csv"]).resolve()
    fee_path = Path(LOCKED["canonical_fee_model"]).resolve()
    metrics_path = Path(LOCKED["canonical_metrics_definition"]).resolve()
    missing = [str(p) for p in (rep_csv, fee_path, metrics_path) if not p.exists()]
    fee_sha = sha256_file(fee_path) if fee_path.exists() else ""
    metrics_sha = sha256_file(metrics_path) if metrics_path.exists() else ""

    args_lock = build_exec_args(signals_csv=rep_csv, seed=seed)
    lock_validation = {}
    lock_err = ""
    if not missing:
        try:
            lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=args_lock, run_dir=phase_dir)
        except Exception as exc:
            lock_err = f"{type(exc).__name__}: {exc}"
    lock_pass = int(
        (len(missing) == 0)
        and (fee_sha == LOCKED["expected_fee_sha"])
        and (metrics_sha == LOCKED["expected_metrics_sha"])
        and (int(lock_validation.get("freeze_lock_pass", 0)) == 1)
        and (not lock_err)
    )

    contract = {
        "generated_utc": utc_now(),
        "symbol": LOCKED["symbol"],
        "representative_subset_csv": str(rep_csv),
        "canonical_fee_model": str(fee_path),
        "canonical_metrics_definition": str(metrics_path),
        "expected_fee_sha256": LOCKED["expected_fee_sha"],
        "expected_metrics_sha256": LOCKED["expected_metrics_sha"],
        "actual_fee_sha256": fee_sha,
        "actual_metrics_sha256": metrics_sha,
        "missing_files": missing,
        "ga_exec_freeze_lock_validation": lock_validation,
        "lock_error": lock_err,
        "lock_pass": int(lock_pass),
    }
    json_dump(phase_dir / "phaseX_contract_check.json", contract)
    manifest["contract_check"] = contract

    objective_lines = [
        "objective_name: phaseX_execution_aware_1h_ranking",
        "scope: SOLUSDT 1h candidates scored through fixed 3m execution pair (E1/E2)",
        "primary:",
        "  - avg_post_exec_expectancy_net",
        "  - avg_delta_vs_exec_baseline",
        "  - split_stability: [avg_median_split_expectancy, worst_min_split_expectancy, max_std_split_expectancy]",
        "secondary:",
        "  - avg_cvar_improve_ratio",
        "  - avg_maxdd_improve_ratio",
        "  - support: [min_entries_valid_across_exec, min_entry_rate_across_exec]",
        "tertiary:",
        "  - fill_realism: [max_taker_share_across_exec, max_p95_fill_delay_across_exec]",
        "penalties:",
        "  - invalid_or_unrankable: strong_penalty",
        "  - weak_support: entries<200 or entry_rate<0.70",
        "  - realism_breach: taker_share>0.25 or p95_fill_delay>180",
        "  - NaN/pathology saturation",
        "notes:",
        "  - raw 1h backtest pnl is not primary objective",
        "  - hard execution gates unchanged",
        "  - canonical freeze hash lock required",
    ]
    write_text(phase_dir / "phaseX_1h_objective_spec.yaml", "\n".join(objective_lines))

    if lock_pass != 1:
        cls = "E"
        write_text(
            phase_dir / "phaseX_decision.md",
            "\n".join(
                [
                    "# Phase X Decision",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Classification: **{cls} (INFRA_FAIL)**",
                    "- Reason: contract lock/hash validation failed.",
                    "- Mainline: STOP",
                ]
            ),
        )
        write_text(
            phase_dir / "phaseX_infra_fail_report.md",
            "\n".join(
                [
                    "# Phase X Infra Fail Report",
                    "",
                    f"- Missing files: {', '.join(missing) if missing else '(none)'}",
                    f"- Fee hash match: {int(fee_sha == LOCKED['expected_fee_sha'])}",
                    f"- Metrics hash match: {int(metrics_sha == LOCKED['expected_metrics_sha'])}",
                    f"- ga_exec lock pass: {int(lock_validation.get('freeze_lock_pass', 0))}",
                    f"- lock_error: {lock_err or '(none)'}",
                ]
            ),
        )
        write_text(
            phase_dir / "ready_to_launch_phaseX_retry_prompt.txt",
            "Phase X retry: fix canonical lock path/hash mismatch first, then rerun execution-aware 1h pilot with identical candidate budget and unchanged hard gates.",
        )
        return {
            "classification": cls,
            "mainline_status": "STOP_INFRA_FAIL",
            "phase_dir": str(phase_dir),
            "top_candidates": pd.DataFrame(),
            "candidate_results": pd.DataFrame(),
            "ranking_cmp": pd.DataFrame(),
            "next_prompt_path": str(phase_dir / "ready_to_launch_phaseX_retry_prompt.txt"),
        }

    t_eval = time.time()
    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    exec_pair, exec_pair_meta = load_exec_pair(exec_root)
    manifest["exec_pair_source"] = exec_pair_meta

    params_path, base_params_raw, params_meta = phaseu.load_active_sol_params()
    base_params = ga_long._norm_params(copy.deepcopy(base_params_raw))
    candidates = phaseu.generate_1h_candidates(base_params=base_params, n_total=int(phasex_candidates), seed=int(seed))
    manifest["active_params_path"] = str(params_path)
    manifest["active_params_meta"] = params_meta
    manifest["candidate_generated"] = int(len(candidates))

    rep_subset = pd.read_csv(rep_csv)
    rep_subset["signal_time"] = pd.to_datetime(rep_subset["signal_time"], utc=True, errors="coerce")
    rep_subset = rep_subset.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rep_subset["signal_id"] = rep_subset["signal_id"].astype(str)

    df1h = recon._load_symbol_df("SOLUSDT", tf="1h")
    df_feat = ga_long._ensure_indicators(df1h.copy(), base_params)
    df_feat = ga_long._prepare_signal_df(df_feat, assume_prepared=False)
    rep_idx = phaseu.build_rep_subset_with_idx(rep_subset=rep_subset, df_feat=df_feat)

    bundles, _load_meta = ga_exec._prepare_bundles(args_lock)
    if not bundles:
        raise RuntimeError("Phase X: no execution bundles prepared")
    base_bundle = bundles[0]

    rows: List[Dict[str, Any]] = []
    signature_owner: Dict[str, str] = {}
    unique_cache: Dict[str, Dict[str, Any]] = {}

    for cand in candidates:
        cid = str(cand["signal_candidate_id"])
        active_ids, sig_diag = phaseu.active_signal_ids_for_params(df_feat=df_feat, params=cand["params"], rep_idx=rep_idx)
        sig_signature = sha256_text("|".join(sorted(active_ids)))
        duplicate_of = signature_owner.get(sig_signature, "")
        if not duplicate_of:
            signature_owner[sig_signature] = cid

        if duplicate_of and duplicate_of in unique_cache:
            cached = copy.deepcopy(unique_cache[duplicate_of])
            cached.update(
                {
                    "signal_candidate_id": cid,
                    "signal_candidate_name": str(cand.get("name", "")),
                    "signal_candidate_kind": str(cand.get("kind", "")),
                    "signal_param_hash": str(cand.get("param_hash", "")),
                    "duplicate_of": duplicate_of,
                    "is_unique_signal": 0,
                }
            )
            rows.append(cached)
            continue

        # Old 1h ranking proxy from classical 1h backtest metrics.
        trades_old, old_metrics = ga_long.run_backtest_long_only(
            df=df_feat,
            symbol="SOLUSDT",
            p=cand["params"],
            initial_equity=10_000.0,
            fee_bps=7.0,
            slippage_bps=2.0,
            collect_trades=False,
            assume_prepared=True,
        )
        _ = trades_old  # keep explicit to show call intent
        old_score = old_1h_score(old_metrics)

        c_bundle = phaseu.build_candidate_bundle(base_bundle=base_bundle, active_ids=active_ids, args=args_lock)
        n_ctx = int(len(c_bundle.contexts))

        if n_ctx == 0:
            e1 = {
                "valid_for_ranking": 0,
                "overall_entries_valid": 0.0,
                "overall_entry_rate": 0.0,
                "overall_exec_expectancy_net": float("nan"),
                "overall_delta_expectancy_exec_minus_baseline": float("nan"),
                "overall_cvar_improve_ratio": float("nan"),
                "overall_maxdd_improve_ratio": float("nan"),
                "overall_exec_taker_share": float("nan"),
                "overall_exec_p95_fill_delay_min": float("nan"),
                "min_split_expectancy_net": float("nan"),
                "median_split_expectancy_net": float("nan"),
                "std_split_expectancy_net": float("nan"),
                "invalid_reason": "no_signals",
                "hard_invalid": 1,
            }
            e2 = copy.deepcopy(e1)
            e0 = phaseu.baseline_metrics_from_bundle(bundle=c_bundle, args=args_lock)
        else:
            e1 = ga_exec._evaluate_genome(genome=exec_pair["E1"]["genome"], bundles=[c_bundle], args=args_lock, detailed=False)
            e2 = ga_exec._evaluate_genome(genome=exec_pair["E2"]["genome"], bundles=[c_bundle], args=args_lock, detailed=False)
            e0 = phaseu.baseline_metrics_from_bundle(bundle=c_bundle, args=args_lock)

        avg_exp = float(np.nanmean([float(e1.get("overall_exec_expectancy_net", np.nan)), float(e2.get("overall_exec_expectancy_net", np.nan))]))
        avg_delta = float(np.nanmean([float(e1.get("overall_delta_expectancy_exec_minus_baseline", np.nan)), float(e2.get("overall_delta_expectancy_exec_minus_baseline", np.nan))]))
        avg_cvar = float(np.nanmean([float(e1.get("overall_cvar_improve_ratio", np.nan)), float(e2.get("overall_cvar_improve_ratio", np.nan))]))
        avg_dd = float(np.nanmean([float(e1.get("overall_maxdd_improve_ratio", np.nan)), float(e2.get("overall_maxdd_improve_ratio", np.nan))]))
        min_entries = float(np.nanmin([float(e1.get("overall_entries_valid", np.nan)), float(e2.get("overall_entries_valid", np.nan))]))
        min_entry_rate = float(np.nanmin([float(e1.get("overall_entry_rate", np.nan)), float(e2.get("overall_entry_rate", np.nan))]))
        max_taker = float(np.nanmax([float(e1.get("overall_exec_taker_share", np.nan)), float(e2.get("overall_exec_taker_share", np.nan))]))
        max_p95 = float(np.nanmax([float(e1.get("overall_exec_p95_fill_delay_min", np.nan)), float(e2.get("overall_exec_p95_fill_delay_min", np.nan))]))
        avg_med_split = float(np.nanmean([float(e1.get("median_split_expectancy_net", np.nan)), float(e2.get("median_split_expectancy_net", np.nan))]))
        worst_min_split = float(np.nanmin([float(e1.get("min_split_expectancy_net", np.nan)), float(e2.get("min_split_expectancy_net", np.nan))]))
        max_std_split = float(np.nanmax([float(e1.get("std_split_expectancy_net", np.nan)), float(e2.get("std_split_expectancy_net", np.nan))]))
        valid_both = int(int(e1.get("valid_for_ranking", 0)) == 1 and int(e2.get("valid_for_ranking", 0)) == 1)

        row = {
            "signal_candidate_id": cid,
            "signal_candidate_name": str(cand.get("name", "")),
            "signal_candidate_kind": str(cand.get("kind", "")),
            "signal_param_hash": str(cand.get("param_hash", "")),
            "signal_signature": sig_signature,
            "duplicate_of": "",
            "is_unique_signal": 1,
            "signals_active": int(len(active_ids)),
            "mapped_to_1h_index": int(sig_diag.get("mapped_to_1h_index", 0)),
            "active_rate_vs_rep": float(sig_diag.get("active_rate_vs_rep", np.nan)),
            "old_1h_net_profit": float(old_metrics.get("net_profit", np.nan)),
            "old_1h_profit_factor": float(old_metrics.get("profit_factor", np.nan)),
            "old_1h_max_dd": float(old_metrics.get("max_dd", np.nan)),
            "old_1h_trades": float(old_metrics.get("trades", np.nan)),
            "old_1h_score": float(old_score),
            "e0_expectancy": float(e0.get("overall_exec_expectancy_net", np.nan)),
            "e0_entries_valid": float(e0.get("overall_entries_valid", np.nan)),
            "E1_valid_for_ranking": int(e1.get("valid_for_ranking", 0)),
            "E1_entries_valid": float(e1.get("overall_entries_valid", np.nan)),
            "E1_entry_rate": float(e1.get("overall_entry_rate", np.nan)),
            "E1_expectancy": float(e1.get("overall_exec_expectancy_net", np.nan)),
            "E1_delta_vs_e0": float(e1.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
            "E1_cvar_improve": float(e1.get("overall_cvar_improve_ratio", np.nan)),
            "E1_maxdd_improve": float(e1.get("overall_maxdd_improve_ratio", np.nan)),
            "E1_taker_share": float(e1.get("overall_exec_taker_share", np.nan)),
            "E1_p95_fill_delay": float(e1.get("overall_exec_p95_fill_delay_min", np.nan)),
            "E1_min_split": float(e1.get("min_split_expectancy_net", np.nan)),
            "E1_median_split": float(e1.get("median_split_expectancy_net", np.nan)),
            "E1_std_split": float(e1.get("std_split_expectancy_net", np.nan)),
            "E1_invalid_reason": str(e1.get("invalid_reason", "")),
            "E2_valid_for_ranking": int(e2.get("valid_for_ranking", 0)),
            "E2_entries_valid": float(e2.get("overall_entries_valid", np.nan)),
            "E2_entry_rate": float(e2.get("overall_entry_rate", np.nan)),
            "E2_expectancy": float(e2.get("overall_exec_expectancy_net", np.nan)),
            "E2_delta_vs_e0": float(e2.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
            "E2_cvar_improve": float(e2.get("overall_cvar_improve_ratio", np.nan)),
            "E2_maxdd_improve": float(e2.get("overall_maxdd_improve_ratio", np.nan)),
            "E2_taker_share": float(e2.get("overall_exec_taker_share", np.nan)),
            "E2_p95_fill_delay": float(e2.get("overall_exec_p95_fill_delay_min", np.nan)),
            "E2_min_split": float(e2.get("min_split_expectancy_net", np.nan)),
            "E2_median_split": float(e2.get("median_split_expectancy_net", np.nan)),
            "E2_std_split": float(e2.get("std_split_expectancy_net", np.nan)),
            "E2_invalid_reason": str(e2.get("invalid_reason", "")),
            "valid_both_exec": int(valid_both),
            "avg_post_exec_expectancy_net": float(avg_exp),
            "avg_delta_vs_exec_baseline": float(avg_delta),
            "avg_cvar_improve_ratio": float(avg_cvar),
            "avg_maxdd_improve_ratio": float(avg_dd),
            "min_entries_valid_across_exec": float(min_entries),
            "min_entry_rate_across_exec": float(min_entry_rate),
            "max_taker_share_across_exec": float(max_taker),
            "max_p95_fill_delay_across_exec": float(max_p95),
            "avg_median_split_expectancy": float(avg_med_split),
            "worst_min_split_expectancy": float(worst_min_split),
            "max_std_split_expectancy": float(max_std_split),
        }
        row["execaware_score"] = candidate_objective(pd.Series(row))
        rows.append(row)
        unique_cache[cid] = copy.deepcopy(row)

    df = pd.DataFrame(rows)
    df["old_1h_rank"] = (
        df.sort_values(["old_1h_score", "signal_candidate_id"], ascending=[False, True]).reset_index(drop=True).index + 1
    ).astype(float)
    df["execaware_rank"] = (
        df.sort_values(["execaware_score", "signal_candidate_id"], ascending=[False, True]).reset_index(drop=True).index + 1
    ).astype(float)

    # stable rank assignment back by candidate id
    old_rank_map = (
        df.sort_values(["old_1h_score", "signal_candidate_id"], ascending=[False, True])[["signal_candidate_id"]]
        .reset_index(drop=True)
        .reset_index()
    )
    old_rank_map["old_1h_rank"] = old_rank_map["index"] + 1
    old_rank_map = old_rank_map.set_index("signal_candidate_id")["old_1h_rank"].to_dict()
    new_rank_map = (
        df.sort_values(["execaware_score", "signal_candidate_id"], ascending=[False, True])[["signal_candidate_id"]]
        .reset_index(drop=True)
        .reset_index()
    )
    new_rank_map["execaware_rank"] = new_rank_map["index"] + 1
    new_rank_map = new_rank_map.set_index("signal_candidate_id")["execaware_rank"].to_dict()
    df["old_1h_rank"] = df["signal_candidate_id"].map(old_rank_map).astype(float)
    df["execaware_rank"] = df["signal_candidate_id"].map(new_rank_map).astype(float)

    df.to_csv(phase_dir / "phaseX_1h_candidate_results.csv", index=False)

    unique_df = df[df["is_unique_signal"] == 1].copy().reset_index(drop=True)
    topk = 5
    old_top = unique_df.sort_values(["old_1h_rank"], ascending=[True])["signal_candidate_id"].head(topk).tolist()
    new_top = unique_df.sort_values(["execaware_rank"], ascending=[True])["signal_candidate_id"].head(topk).tolist()
    overlap = sorted(set(old_top).intersection(set(new_top)))
    spear = spearman_rank_corr(unique_df["old_1h_rank"].to_numpy(dtype=float), unique_df["execaware_rank"].to_numpy(dtype=float))

    cmp_df = unique_df[
        [
            "signal_candidate_id",
            "signal_candidate_name",
            "signal_candidate_kind",
            "signals_active",
            "old_1h_score",
            "old_1h_rank",
            "execaware_score",
            "execaware_rank",
            "avg_post_exec_expectancy_net",
            "avg_delta_vs_exec_baseline",
            "avg_cvar_improve_ratio",
            "avg_maxdd_improve_ratio",
            "valid_both_exec",
            "min_entries_valid_across_exec",
            "min_entry_rate_across_exec",
        ]
    ].copy()
    cmp_df["rank_shift_new_minus_old"] = cmp_df["execaware_rank"] - cmp_df["old_1h_rank"]
    cmp_df.to_csv(phase_dir / "phaseX_execaware_ranking_comparison.csv", index=False)

    # top candidates
    top_candidates = unique_df.sort_values(
        ["execaware_score", "avg_post_exec_expectancy_net", "avg_delta_vs_exec_baseline", "valid_both_exec"],
        ascending=[False, False, False, False],
    ).head(10)
    top_candidates.to_csv(phase_dir / "phaseX_top_candidates.csv", index=False)

    # Decomposition (active candidate vs top execution-aware)
    active_row = unique_df[unique_df["signal_candidate_name"].astype(str) == "base_active"].copy()
    if active_row.empty:
        active_row = unique_df.sort_values(["old_1h_rank"], ascending=[True]).head(1).copy()
    active = active_row.iloc[0]
    top = top_candidates.iloc[0]

    sig_gain_e1 = float(top["E1_expectancy"] - active["E1_expectancy"])
    sig_gain_e2 = float(top["E2_expectancy"] - active["E2_expectancy"])
    exec_gain_active_e1 = float(active["E1_expectancy"] - active["e0_expectancy"])
    exec_gain_active_e2 = float(active["E2_expectancy"] - active["e0_expectancy"])
    exec_gain_top_e1 = float(top["E1_expectancy"] - top["e0_expectancy"])
    exec_gain_top_e2 = float(top["E2_expectancy"] - top["e0_expectancy"])
    avg_signal_gain = float(np.nanmean([sig_gain_e1, sig_gain_e2]))
    avg_exec_gain_active = float(np.nanmean([exec_gain_active_e1, exec_gain_active_e2]))
    avg_exec_gain_top = float(np.nanmean([exec_gain_top_e1, exec_gain_top_e2]))

    decomp_lines = []
    decomp_lines.append("# Phase X Signal vs Execution Decomposition")
    decomp_lines.append("")
    decomp_lines.append(f"- Generated UTC: {utc_now()}")
    decomp_lines.append(
        f"- Active/reference 1h candidate: {active['signal_candidate_id']} ({active['signal_candidate_name']})"
    )
    decomp_lines.append(
        f"- Top execution-aware 1h candidate: {top['signal_candidate_id']} ({top['signal_candidate_name']})"
    )
    decomp_lines.append("")
    decomp_lines.append("## Expectancy decomposition")
    decomp_lines.append("")
    decomp_lines.append(f"- signal_gain_E1 = {sig_gain_e1:.8f}")
    decomp_lines.append(f"- signal_gain_E2 = {sig_gain_e2:.8f}")
    decomp_lines.append(f"- avg_signal_gain = {avg_signal_gain:.8f}")
    decomp_lines.append(f"- active_exec_gain_vs_E0 (avg E1/E2) = {avg_exec_gain_active:.8f}")
    decomp_lines.append(f"- top_exec_gain_vs_E0 (avg E1/E2) = {avg_exec_gain_top:.8f}")
    if np.isfinite(avg_signal_gain) and np.isfinite(avg_exec_gain_active):
        if abs(avg_signal_gain) < 0.5 * abs(avg_exec_gain_active):
            decomp_lines.append("- attribution_result: execution-dominant (signal gain materially smaller than execution uplift).")
        elif avg_signal_gain > 0:
            decomp_lines.append("- attribution_result: mixed (signal contributes but execution still critical).")
        else:
            decomp_lines.append("- attribution_result: negative signal contribution vs active reference.")
    write_text(phase_dir / "phaseX_signal_vs_execution_decomposition.md", "\n".join(decomp_lines))

    # decision
    material_rank_change = int((np.isfinite(spear) and spear <= 0.80) or (len(overlap) <= 3))
    robust_signal_gain = int(np.isfinite(sig_gain_e1) and np.isfinite(sig_gain_e2) and sig_gain_e1 > 0 and sig_gain_e2 > 0)
    support_ok = int(int(top["valid_both_exec"]) == 1 and float(top["min_entries_valid_across_exec"]) >= 200 and float(top["min_entry_rate_across_exec"]) >= 0.70)
    meaningful_gain = int(np.isfinite(avg_signal_gain) and avg_signal_gain >= 0.00002)

    if material_rank_change and robust_signal_gain and support_ok and meaningful_gain:
        if avg_signal_gain >= 0.00008 and float(top["avg_maxdd_improve_ratio"]) > float(active["avg_maxdd_improve_ratio"]):
            classification = "A"
            reason = "1h_redesign_material_and_robust_gain"
        else:
            classification = "B"
            reason = "1h_redesign_gain_present_but_moderate"
    else:
        if (not meaningful_gain) or (not material_rank_change):
            classification = "D"
            reason = "execution_only_still_best_under_execaware_scoring"
        else:
            classification = "C"
            reason = "1h_redesign_no_material_robust_gain"

    report_lines = []
    report_lines.append("# Phase X Pilot Report")
    report_lines.append("")
    report_lines.append(f"- Generated UTC: {utc_now()}")
    report_lines.append(f"- Candidate budget: {len(candidates)}")
    report_lines.append(f"- Unique signal candidates: {int(unique_df['is_unique_signal'].sum())}")
    report_lines.append(f"- Duplicate signal candidates: {int((df['is_unique_signal'] == 0).sum())}")
    report_lines.append(f"- Spearman(old_rank,new_rank): {spear:.6f}" if np.isfinite(spear) else "- Spearman(old_rank,new_rank): nan")
    report_lines.append(f"- Top-{topk} overlap count: {len(overlap)} ({', '.join(overlap) if overlap else 'none'})")
    report_lines.append(f"- Active candidate: {active['signal_candidate_id']} ({active['signal_candidate_name']})")
    report_lines.append(f"- Top execution-aware candidate: {top['signal_candidate_id']} ({top['signal_candidate_name']})")
    report_lines.append(f"- avg_signal_gain(E1/E2): {avg_signal_gain:.8f}")
    report_lines.append(f"- avg_exec_gain_active(E1/E2 vs E0): {avg_exec_gain_active:.8f}")
    report_lines.append(f"- avg_exec_gain_top(E1/E2 vs E0): {avg_exec_gain_top:.8f}")
    report_lines.append(f"- material_rank_change={material_rank_change}, robust_signal_gain={robust_signal_gain}, support_ok={support_ok}, meaningful_gain={meaningful_gain}")
    report_lines.append(f"- classification: **{classification}** ({reason})")
    write_text(phase_dir / "phaseX_pilot_report.md", "\n".join(report_lines))

    decision_lines = []
    decision_lines.append("# Phase X Decision")
    decision_lines.append("")
    decision_lines.append(f"- Generated UTC: {utc_now()}")
    decision_lines.append(f"- Classification: **{classification}**")
    decision_lines.append(f"- Reason: {reason}")
    if classification in {"A", "B"}:
        decision_lines.append("- Mainline: CONTINUE (Phase Y justified).")
    else:
        decision_lines.append("- Mainline: STOP (Phase X no-go branch).")
    decision_lines.append(f"- Spearman rank correlation: {spear:.6f}" if np.isfinite(spear) else "- Spearman rank correlation: nan")
    decision_lines.append(f"- Top-{topk} overlap: {len(overlap)}")
    decision_lines.append(f"- avg_signal_gain(E1/E2): {avg_signal_gain:.8f}")
    decision_lines.append(f"- avg_exec_gain_active(E1/E2 vs E0): {avg_exec_gain_active:.8f}")
    write_text(phase_dir / "phaseX_decision.md", "\n".join(decision_lines))

    # fallback package for X no-go
    next_prompt_path = None
    if classification in {"C", "D"}:
        ng = phase_dir / "phaseX_no_go_package"
        ng.mkdir(parents=True, exist_ok=False)
        write_text(
            ng / "phaseX_no_go_reasoning.md",
            "\n".join(
                [
                    "# Phase X No-Go Reasoning",
                    "",
                    f"- classification: {classification}",
                    f"- reason: {reason}",
                    f"- spearman_rank_corr: {spear:.6f}" if np.isfinite(spear) else "- spearman_rank_corr: nan",
                    f"- top5_overlap: {len(overlap)}",
                    f"- avg_signal_gain(E1/E2): {avg_signal_gain:.8f}",
                    f"- avg_exec_gain_active(E1/E2 vs E0): {avg_exec_gain_active:.8f}",
                    "- conclusion: execution remains the dominant uplift source; current 1h redesign neighborhood does not justify immediate 1h GA expansion.",
                ]
            ),
        )
        write_text(
            ng / "phaseX_execution_only_recommendation.md",
            "\n".join(
                [
                    "# Phase X Execution-Only Recommendation",
                    "",
                    "- Continue SOL execution-layer paper/shadow route using promoted execution pair (E1/E2).",
                    "- Keep hard gates, canonical lock checks, and realism monitoring unchanged.",
                    "- Do not launch larger 1h GA until new upstream signal labels/features are prepared.",
                ]
            ),
        )
        write_text(
            ng / "phaseX_1h_forensic_backlog.md",
            "\n".join(
                [
                    "# Phase X 1H Forensic Backlog",
                    "",
                    "Required before retrying 1h objective redesign:",
                    "1) Build richer ex-ante labels tied to downstream execution outcomes (split-aware).",
                    "2) Add regime/session-conditioned 1h diagnostics for loss clustering and stop concentration.",
                    "3) Add explicit signal quality features that correlate with 3m fill realism and adverse runs.",
                    "4) Re-test no-op/low-sensitivity 1h knobs to prune dead dimensions before next GA.",
                ]
            ),
        )
        next_prompt_path = ng / "ready_to_launch_phaseX_alt_prompt.txt"
        write_text(
            next_prompt_path,
            "Phase X-alt (no new GA): keep execution winners fixed (E1/E2) and run a 1h forensic data-labeling sprint to build execution-aware signal quality labels (split/session/regime loss-cluster diagnostics). Produce a pruned 1h knob set and only then rerun a small execution-aware 1h pilot.",
        )

    manifest.update(
        {
            "classification": classification,
            "classification_reason": reason,
            "evaluation_runtime_sec": float(time.time() - t_eval),
            "phase_runtime_sec": float(time.time() - t_eval),
            "candidate_rows": int(len(df)),
            "unique_candidates": int((df["is_unique_signal"] == 1).sum()),
            "duplicate_candidates": int((df["is_unique_signal"] == 0).sum()),
            "spearman_old_vs_execaware": float(spear) if np.isfinite(spear) else None,
            "top5_overlap": int(len(overlap)),
            "avg_signal_gain": float(avg_signal_gain) if np.isfinite(avg_signal_gain) else None,
            "avg_exec_gain_active": float(avg_exec_gain_active) if np.isfinite(avg_exec_gain_active) else None,
            "next_prompt_path": str(next_prompt_path) if next_prompt_path else "",
        }
    )

    return {
        "classification": classification,
        "mainline_status": "CONTINUE_JUSTIFIED" if classification in {"A", "B"} else "STOP_NO_GO",
        "phase_dir": str(phase_dir),
        "top_candidates": top_candidates.copy(),
        "candidate_results": df.copy(),
        "ranking_cmp": cmp_df.copy(),
        "manifest": manifest,
        "next_prompt_path": str(next_prompt_path) if next_prompt_path else "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Autonomous X->... controller for SOL 1h+3m stack")
    ap.add_argument("--seed", type=int, default=20260223)
    ap.add_argument("--phasex-candidates", type=int, default=32, help="Phase X pilot candidate budget (20-40 recommended)")
    args = ap.parse_args()

    t0 = time.time()
    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_dir = exec_root / f"PHASEXYZ2_AUTORUN_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    run_manifest: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "phase_sequence_target": ["X", "Y", "Y2", "Z", "Z2"],
        "seed": int(args.seed),
        "phasex_candidates": int(args.phasex_candidates),
        "code_modified": "YES (new script phase_xyz2_autorun.py)",
        "commands": [{"cmd": "python scripts/phase_xyz2_autorun.py", "utc": utc_now()}],
    }

    # Phase X
    x = run_phase_x(parent_dir=run_dir, seed=int(args.seed), phasex_candidates=int(args.phasex_candidates))
    run_manifest["phaseX"] = {
        "classification": x["classification"],
        "phase_dir": x["phase_dir"],
        "mainline_status": x["mainline_status"],
        "next_prompt_path": x.get("next_prompt_path", ""),
    }

    furthest = "X"
    classification = str(x["classification"])
    mainline_status = str(x["mainline_status"])
    stop_reason = ""

    if classification in {"A", "B"}:
        # This autonomous run intentionally gates at X unless strong signal evidence is observed.
        # If GO appears, we still stop and request explicit Phase Y launch prompt artifact.
        stop_reason = "Phase X GO observed but bounded autonomous run configured to stop after Phase X for controlled review."
        mainline_status = "CONTINUE_JUSTIFIED"
        write_text(
            run_dir / "ready_to_launch_phaseY_prompt.txt",
            "Phase Y launch: run medium 1h redesign GA (96x4 or 128x4) scored by fixed execution templates E1/E2 under canonical lock checks; include duplicate/effective-trials controls and stop if no robust signal-driven gain.",
        )
    else:
        stop_reason = "Phase X no-go/infra classification triggered branch stop by policy."

    run_manifest.update(
        {
            "furthest_phase_reached": furthest,
            "classification_at_furthest": classification,
            "mainline_status": mainline_status,
            "stop_reason": stop_reason,
            "duration_sec": float(time.time() - t0),
        }
    )
    json_dump(run_dir / "pipeline_run_manifest.json", run_manifest)

    print(json.dumps({"run_dir": str(run_dir), "furthest_phase": furthest, "classification": classification, "mainline_status": mainline_status}))


if __name__ == "__main__":
    main()

