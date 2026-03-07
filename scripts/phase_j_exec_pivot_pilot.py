#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports" / "execution_layer"

EXPECTED_REP_HASH = "fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d"
EXPECTED_FEE_HASH = "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a"
EXPECTED_METRICS_HASH = "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99"
EXPECTED_MODEL_SET_HASH = "4a8cb243e7f7e6425db6726302d6326bf727fe026baca77980af0532543c2fc4"


@dataclass
class PhaseInputs:
    phaseh_dir: Path
    phaseh_manifest: Dict[str, Any]
    phaseg_dir: Path
    phaseg_manifest: Dict[str, Any]
    postfix_dir: Path
    postfix_manifest: Dict[str, Any]
    forensic_dir: Path
    fixed_size_passers: int
    bug_fix_confirmed: int
    branch: str
    branch_reason: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_tag() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = (PROJECT_ROOT / pp).resolve()
    return pp


def _latest_dir(prefix: str) -> Path:
    cands = sorted([p for p in REPORTS_ROOT.glob(f"{prefix}_*") if p.is_dir()], key=lambda x: x.name)
    if not cands:
        raise FileNotFoundError(f"No run dirs matching {prefix}_* under {REPORTS_ROOT}")
    return cands[-1]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_rep_subset(df: pd.DataFrame) -> str:
    x = df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rows = [f"{str(r.signal_id)}|{pd.to_datetime(r.signal_time, utc=True).isoformat()}" for r in x.itertuples(index=False)]
    return _sha256_text("\n".join(rows))


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_fixed_size_passers(postfix_report: Path) -> int:
    txt = postfix_report.read_text(encoding="utf-8")
    m = re.search(r"fixed_size_passers_nonduplicate:\s*([0-9]+)", txt)
    if not m:
        return -1
    return int(m.group(1))


def _detect_bug_fix_confirmed(forensic_dir: Path) -> int:
    patch_fp = forensic_dir / "patch_summary.md"
    tests_fp = forensic_dir / "metric_unit_tests_output.txt"
    if not patch_fp.exists() or not tests_fp.exists():
        return 0
    t = tests_fp.read_text(encoding="utf-8")
    has_pass = ("SUMMARY pass=" in t) and ("fail=0" in t)
    return int(has_pass)


def _classify_branch(fixed_size_passers: int, bug_fix_confirmed: int) -> Tuple[str, str]:
    if bug_fix_confirmed == 1 and fixed_size_passers == 0:
        return (
            "A",
            "Implementation patches/tests are present, but fixed-size survivors remain zero; signal fork still non-deployable.",
        )
    if bug_fix_confirmed == 1 and fixed_size_passers > 0:
        return (
            "B",
            "Implementation patches/tests are present and at least one fixed-size survivor exists; signal fork shows residual promise.",
        )
    return (
        "C",
        "No conclusive material bug-fix confirmation in artifacts and signal fork remains non-viable.",
    )


def _load_phase_inputs() -> PhaseInputs:
    phaseh_dir = _latest_dir("PHASEH_SOL_FREEZE_FORK")
    phaseh_manifest = _json(phaseh_dir / "phaseH_run_manifest.json")

    phaseg_dir = _resolve(str(phaseh_manifest.get("phaseg_source_dir", "")))
    if not phaseg_dir.exists():
        phaseg_dir = _latest_dir("PHASEG_SOL_PATHOLOGY_REHAB")
    phaseg_manifest = _json(phaseg_dir / "run_manifest.json")

    postfix_dir = _latest_dir("PHASEI_SOL_POSTFIX_VALIDATION")
    postfix_manifest = _json(postfix_dir / "phaseI_postfix_run_manifest.json")
    fixed_size_passers = _extract_fixed_size_passers(postfix_dir / "phaseI_sol_postfix_validation_report.md")

    forensic_dir = _latest_dir("PHASEI_FORENSIC_DEBUG")
    bug_fix_confirmed = _detect_bug_fix_confirmed(forensic_dir)

    branch, branch_reason = _classify_branch(fixed_size_passers=fixed_size_passers, bug_fix_confirmed=bug_fix_confirmed)
    return PhaseInputs(
        phaseh_dir=phaseh_dir,
        phaseh_manifest=phaseh_manifest,
        phaseg_dir=phaseg_dir,
        phaseg_manifest=phaseg_manifest,
        postfix_dir=postfix_dir,
        postfix_manifest=postfix_manifest,
        forensic_dir=forensic_dir,
        fixed_size_passers=fixed_size_passers,
        bug_fix_confirmed=bug_fix_confirmed,
        branch=branch,
        branch_reason=branch_reason,
    )


def _setup_checks(ctx: PhaseInputs) -> Dict[str, int]:
    rep_hash = str(ctx.phaseh_manifest.get("representative_subset_sha256", "")).strip()
    fee_hash = str(ctx.phaseh_manifest.get("fee_model_sha256", "")).strip()
    metrics_hash = str(ctx.phaseh_manifest.get("metrics_definition_sha256", "")).strip()
    model_hash = str(ctx.phaseh_manifest.get("selected_model_set_sha256", "")).strip()

    rep_subset_path = _resolve(str(ctx.phaseg_manifest.get("representative_subset_path", "")))
    rep_hash_file = _resolve(str(ctx.phaseg_manifest.get("e2_dir", ""))) / "representative_subset_hash.txt"
    rep_hash_from_file = rep_hash_file.read_text(encoding="utf-8").strip() if rep_hash_file.exists() else ""
    rep_hash_calc = ""
    if rep_subset_path.exists():
        rep_df = pd.read_csv(rep_subset_path)
        if {"signal_id", "signal_time"}.issubset(rep_df.columns):
            rep_hash_calc = _hash_rep_subset(rep_df[["signal_id", "signal_time"]].copy())

    return {
        "rep_hash_match_expected": int(rep_hash == EXPECTED_REP_HASH),
        "rep_hash_file_match_manifest": int(rep_hash_from_file == rep_hash) if rep_hash_from_file else 0,
        "rep_hash_calc_match_manifest": int(rep_hash_calc == rep_hash) if rep_hash_calc else 0,
        "fee_hash_match_expected": int(fee_hash == EXPECTED_FEE_HASH),
        "metrics_hash_match_expected": int(metrics_hash == EXPECTED_METRICS_HASH),
        "model_set_hash_match_expected": int(model_hash == EXPECTED_MODEL_SET_HASH),
    }


def _write_search_space_yaml(path: Path) -> None:
    txt = """schema_version: 1
experiment_name: phase_j_sol_execution_exit_pivot
symbol: SOLUSDT
mode: tight
frozen_contract:
  representative_subset_sha256: fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d
  fee_model_sha256: b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a
  metrics_definition_sha256: d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99
  selected_model_set_sha256: 4a8cb243e7f7e6425db6726302d6326bf727fe026baca77980af0532543c2fc4
entry_knobs:
  entry_mode: [market, limit, marketable_limit]
  limit_offset_bps: [0.0, 30.0]
  max_fill_delay_min: [0, 45]
  fallback_to_market: [0, 1]
  fallback_delay_min: [0, max_fill_delay_min]
  max_taker_share: [0.0, 0.25]
  mss_displacement_gate: [0, 1]
  min_entry_improvement_bps_gate: [0.0, 20.0]
  micro_vol_filter: [0, 1]
  vol_threshold: [0.5, 6.0]
  spread_guard_bps: [0.0, 100.0]
  killzone_filter: [0, 1]
  skip_if_vol_gate: [0, 1]
  cooldown_min: [0, 240]
exit_knobs:
  tp_mult: [0.5, 3.0]
  sl_mult: [0.3, 2.0]
  time_stop_min: [0, 4320]
  trailing_enabled: [0, 1]
  trail_start_r: [0.5, 2.0]
  trail_step_bps: [1.0, 50.0]
  break_even_enabled: [0, 1]
  break_even_trigger_r: [0.25, 1.5]
  break_even_offset_bps: [0.0, 10.0]
  partial_take_enabled: [0, 1]
  partial_take_r: [0.3, 1.5]
  partial_take_pct: [0.1, 0.9]
objectives:
  primary:
    - fixed_size_geometric_equity_step_return
  secondary:
    - max_drawdown_floor
    - cvar_floor
    - split_support_stability
  tertiary:
    - expectancy_net
    - fill_quality
penalties:
  - overtrading
  - excessive_taker_share
  - regime_concentration
hard_constraints:
  - min_split_trades
  - support_ok
  - nan_pathology_forbidden
  - single_regime_dominance_cap
anti_overfit_controls:
  - duplicate_candidate_collapse
  - effective_trials_estimate
  - psr_dsr_reporting
  - reality_check_bootstrap_placeholder
pilot:
  candidate_budget: 48
  generator: constrained_random
  stop_if_no_viable: true
"""
    path.write_text(txt, encoding="utf-8")


def _write_plan_md(path: Path, ctx: PhaseInputs, checks: Dict[str, int], ga_dir_rel: str) -> None:
    rep_path = str(ctx.phaseg_manifest.get("representative_subset_path", ""))
    lines: List[str] = []
    lines.append("# Next Experiment Plan")
    lines.append("")
    lines.append(f"- Generated UTC: {_utc_now().isoformat()}")
    lines.append(f"- Chosen branch: **{ctx.branch}**")
    lines.append(f"- Branch reason: {ctx.branch_reason}")
    lines.append("")
    lines.append("## Evidence Basis")
    lines.append("")
    lines.append(f"- Phase I postfix source: `{ctx.postfix_dir}`")
    lines.append(f"- Phase I forensic source: `{ctx.forensic_dir}`")
    lines.append(f"- fixed_size_passers_nonduplicate: {ctx.fixed_size_passers}")
    lines.append(f"- bug_fix_confirmed_from_forensic_tests: {ctx.bug_fix_confirmed}")
    lines.append(f"- postfix recommendation: {ctx.postfix_manifest.get('recommendation', 'n/a')}")
    lines.append("")
    lines.append("## Frozen Setup Confirmation")
    lines.append("")
    lines.append(f"- representative_subset_path: `{rep_path}`")
    lines.append(f"- representative_subset_sha256: `{ctx.phaseh_manifest.get('representative_subset_sha256', '')}`")
    lines.append(f"- fee_model_sha256: `{ctx.phaseh_manifest.get('fee_model_sha256', '')}`")
    lines.append(f"- metrics_definition_sha256: `{ctx.phaseh_manifest.get('metrics_definition_sha256', '')}`")
    lines.append(f"- selected_model_set_sha256: `{ctx.phaseh_manifest.get('selected_model_set_sha256', '')}`")
    lines.append(f"- setup_checks_vs_expected: `{json.dumps(checks, sort_keys=True)}`")
    lines.append("")
    lines.append("## Execution/Exit Pivot Design")
    lines.append("")
    lines.append("Entry search family:")
    lines.append("- delay and fill control (`max_fill_delay_min`, `fallback_to_market`, `fallback_delay_min`)")
    lines.append("- entry price discipline (`limit_offset_bps`, `min_entry_improvement_bps_gate`, `spread_guard_bps`)")
    lines.append("- entry quality gating (`mss_displacement_gate`, `micro_vol_filter`, `skip_if_vol_gate`, `killzone_filter`)")
    lines.append("- flow pacing (`cooldown_min`, taker-share limits)")
    lines.append("")
    lines.append("Exit search family:")
    lines.append("- target/stop geometry (`tp_mult`, `sl_mult`)")
    lines.append("- time-based exits (`time_stop_min`)")
    lines.append("- adaptive protection (`break_even_*`, `trailing_*`)")
    lines.append("- staged exits (`partial_take_*`)")
    lines.append("")
    lines.append("Objective hierarchy for full run:")
    lines.append("- Primary: fixed-size geometric equity-step return (when exported by runner)")
    lines.append("- Secondary: drawdown and CVaR floors plus split stability/support")
    lines.append("- Tertiary: expectancy and fill-quality")
    lines.append("- Penalties: overtrading, taker-share excess, regime concentration")
    lines.append("")
    lines.append("Hard constraints:")
    lines.append("- minimum split trade support")
    lines.append("- support/validity gates")
    lines.append("- no NaN/metric pathology")
    lines.append("- no single-regime dominance beyond threshold (requires regime audit column in full run)")
    lines.append("")
    lines.append("Anti-overfit controls:")
    lines.append("- duplicate candidate collapse")
    lines.append("- effective trial count after pruning")
    lines.append("- PSR/DSR proxy reporting for shortlist")
    lines.append("- reality-check bootstrap placeholder (explicit TODO)")
    lines.append("")
    lines.append("## Pilot Spec")
    lines.append("")
    lines.append("- Scope: SOLUSDT only")
    lines.append("- Budget: 48 candidates (single generation)")
    lines.append("- Engine: `src/execution/ga_exec_3m_opt.py`")
    lines.append("- Run root: `" + ga_dir_rel + "`")
    lines.append("- Stop condition: if zero viable candidates after hard constraints, mark NO_GO for full marathon")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- No downstream contract, fee model, or subset edits were introduced in this phase.")
    lines.append("- This pilot is a diversity/viability probe, not a full optimization campaign.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _norm_cdf(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0))))


def _effective_trials_from_corr(mat: np.ndarray) -> float:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
        return float("nan")
    vals = np.linalg.eigvalsh(mat)
    vals = np.clip(vals, 0.0, None)
    s1 = float(np.sum(vals))
    s2 = float(np.sum(vals**2))
    if s2 <= 0.0:
        return float("nan")
    return float((s1 * s1) / s2)


def _build_signature(df: pd.DataFrame) -> pd.Series:
    cols = [
        "overall_exec_expectancy_net",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "overall_entry_rate",
        "overall_exec_taker_share",
        "overall_exec_median_fill_delay_min",
        "overall_entries_valid",
        "hard_invalid",
        "valid_for_ranking",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    p = df[cols].copy()
    for c in cols:
        p[c] = pd.to_numeric(p[c], errors="coerce").round(12)
    payload = p.apply(lambda r: "|".join([str(x) for x in r.tolist()]), axis=1)
    return payload.map(lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()[:24])


def _run_pilot_ga(run_dir: Path, ctx: PhaseInputs, args: argparse.Namespace) -> Path:
    ga_root = run_dir / "pilot_ga_runs"
    ga_root.mkdir(parents=True, exist_ok=True)

    rep_subset_csv = _resolve(str(ctx.phaseg_manifest.get("representative_subset_path", "")))
    if not rep_subset_csv.exists():
        raise FileNotFoundError(f"Missing representative subset csv: {rep_subset_csv}")
    rep_subset_df = pd.read_csv(rep_subset_csv)
    if not {"signal_id", "signal_time"}.issubset(rep_subset_df.columns):
        raise RuntimeError(f"Representative subset missing required columns in {rep_subset_csv}")
    rep_hash = _hash_rep_subset(rep_subset_df[["signal_id", "signal_time"]].copy())
    rep_hash_expected = str(ctx.phaseh_manifest.get("representative_subset_sha256", "")).strip()
    if rep_hash != rep_hash_expected:
        raise RuntimeError(f"Subset hash mismatch before pilot: calc={rep_hash} expected={rep_hash_expected}")

    rep_hash_file = _resolve(str(ctx.phaseg_manifest.get("e2_dir", ""))) / "representative_subset_hash.txt"
    if rep_hash_file.exists():
        rep_hash_ref = rep_hash_file.read_text(encoding="utf-8").strip()
        if rep_hash_ref != rep_hash:
            raise RuntimeError(f"Subset hash mismatch vs representative_subset_hash.txt: file={rep_hash_ref} calc={rep_hash}")

    cmd = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        "-m",
        "src.execution.ga_exec_3m_opt",
        "--symbol",
        "SOLUSDT",
        "--signals-csv",
        str(rep_subset_csv),
        "--max-signals",
        "1200",
        "--walkforward",
        "--wf-splits",
        "5",
        "--train-ratio",
        "0.70",
        "--mode",
        "tight",
        "--pop",
        str(int(args.pilot_candidates)),
        "--gens",
        "1",
        "--workers",
        str(int(args.workers)),
        "--seed",
        str(int(args.seed)),
        "--fee-bps-maker",
        "2.0",
        "--fee-bps-taker",
        "4.0",
        "--slippage-bps-limit",
        "0.5",
        "--slippage-bps-market",
        "2.0",
        "--execution-config",
        "configs/execution_configs.yaml",
        "--outdir",
        str(ga_root),
    ]

    cmd_txt = " ".join(cmd)
    (run_dir / "pilot_command.sh").write_text(cmd_txt + "\n", encoding="utf-8")

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (run_dir / "pilot_command_output.log").write_text(proc.stdout or "", encoding="utf-8")
    ga_runs = sorted([p for p in ga_root.glob("GA_EXEC_OPT_*") if p.is_dir()], key=lambda x: x.name)
    if not ga_runs:
        raise RuntimeError(
            f"Pilot GA run failed (code={proc.returncode}) and no run dir produced under {ga_root}. "
            f"See {run_dir / 'pilot_command_output.log'}"
        )
    ga_last = ga_runs[-1]
    genomes_fp = ga_last / "genomes.csv"
    known_no_viable = "No valid genomes passed hard constraints" in (proc.stdout or "")
    if proc.returncode != 0 and not (known_no_viable and genomes_fp.exists()):
        raise RuntimeError(
            f"Pilot GA run failed (code={proc.returncode}). See {run_dir / 'pilot_command_output.log'}"
        )
    return ga_last


def _build_pilot_outputs(run_dir: Path, ga_run_dir: Path, ctx: PhaseInputs, branch: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    genomes_fp = ga_run_dir / "genomes.csv"
    if not genomes_fp.exists():
        raise FileNotFoundError(f"Missing genomes.csv in pilot run: {genomes_fp}")
    g = pd.read_csv(genomes_fp)
    if g.empty:
        raise RuntimeError("Pilot genomes.csv is empty.")

    if "genome_hash" not in g.columns:
        g["genome_hash"] = [f"row_{i:04d}" for i in range(len(g))]

    g["metric_signature"] = _build_signature(g)
    dup_counts = g["metric_signature"].value_counts()
    g["is_metric_duplicate"] = g["metric_signature"].map(lambda x: int(dup_counts.get(x, 0) > 1))

    for c in [
        "valid_for_ranking",
        "hard_invalid",
        "constraint_pass",
        "participation_pass",
        "realism_pass",
        "nan_pass",
        "data_quality_pass",
        "split_pass",
    ]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0).astype(int)
        else:
            g[c] = 0

    for c in [
        "overall_exec_expectancy_net",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "overall_entry_rate",
        "overall_exec_taker_share",
        "overall_exec_median_fill_delay_min",
        "overall_exec_p95_fill_delay_min",
        "overall_entries_valid",
        "overall_signals_total",
        "median_split_expectancy_net",
        "std_split_expectancy_net",
        "split_count",
    ]:
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors="coerce")
        else:
            g[c] = np.nan

    # Conservative "sane" definition for pilot viability.
    g["sane_candidate"] = (
        (g["valid_for_ranking"] == 1)
        & (g["nan_pass"] == 1)
        & np.isfinite(g["overall_exec_expectancy_net"])
        & np.isfinite(g["overall_cvar_improve_ratio"])
        & np.isfinite(g["overall_maxdd_improve_ratio"])
    ).astype(int)

    # Objective hierarchy proxy for this pilot runner:
    # primary proxy uses net expectancy because geometric step return is not exported in ga_exec_3m_opt.
    g["primary_equity_growth_proxy"] = g["overall_exec_expectancy_net"]
    g["secondary_path_score"] = (
        0.5 * g["overall_cvar_improve_ratio"].fillna(-1e9)
        + 0.5 * g["overall_maxdd_improve_ratio"].fillna(-1e9)
    )
    g["penalty_overtrading"] = np.where(g["overall_entry_rate"] > 0.97, 0.02, 0.0)
    g["penalty_taker"] = np.where(g["overall_exec_taker_share"] > 0.25, 0.03, 0.0)
    g["penalty_regime_concentration"] = np.nan  # Requires regime-bucket export from evaluator.
    g["composite_rank_score"] = (
        g["primary_equity_growth_proxy"].fillna(-1e9)
        + 0.25 * g["secondary_path_score"].fillna(-1e9)
        - g["penalty_overtrading"].fillna(0.0)
        - g["penalty_taker"].fillna(0.0)
    )

    g = g.sort_values(
        ["sane_candidate", "composite_rank_score", "overall_exec_expectancy_net", "overall_cvar_improve_ratio"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    g["pilot_rank"] = np.arange(1, len(g) + 1)

    # PSR/DSR proxy on split expectancy stats.
    split_n = pd.to_numeric(g["split_count"], errors="coerce").fillna(0).clip(lower=1)
    sr_proxy = pd.to_numeric(g["median_split_expectancy_net"], errors="coerce") / (
        pd.to_numeric(g["std_split_expectancy_net"], errors="coerce").replace(0.0, np.nan)
    )
    z_proxy = sr_proxy * np.sqrt(split_n)
    g["psr_proxy"] = z_proxy.map(_norm_cdf)

    nondup = g[g["is_metric_duplicate"] == 0].copy()
    m_for_corr = nondup[
        [
            "overall_exec_expectancy_net",
            "overall_cvar_improve_ratio",
            "overall_maxdd_improve_ratio",
            "overall_entry_rate",
            "overall_exec_taker_share",
        ]
    ].copy()
    for c in m_for_corr.columns:
        m_for_corr[c] = pd.to_numeric(m_for_corr[c], errors="coerce")
    m_for_corr = m_for_corr.dropna()
    if len(m_for_corr) >= 3:
        corr = np.corrcoef(m_for_corr.values, rowvar=False)
        eff_trials = _effective_trials_from_corr(corr)
    else:
        eff_trials = float(len(nondup))
    if not np.isfinite(eff_trials) or eff_trials <= 0:
        eff_trials = float(max(1, len(nondup)))
    g["effective_trials_proxy"] = float(eff_trials)
    g["dsr_proxy"] = np.power(pd.to_numeric(g["psr_proxy"], errors="coerce"), float(eff_trials))

    # Duplicate map.
    dup_rows = g[["genome_hash", "pilot_rank", "metric_signature", "is_metric_duplicate"]].copy()
    dup_rows["duplicate_group_size"] = dup_rows["metric_signature"].map(dup_counts)
    dup_rows.to_csv(run_dir / "duplicate_variant_map.csv", index=False)

    keep_cols = [
        "pilot_rank",
        "genome_hash",
        "is_metric_duplicate",
        "duplicate_group_size",
        "valid_for_ranking",
        "hard_invalid",
        "constraint_pass",
        "participation_pass",
        "realism_pass",
        "nan_pass",
        "data_quality_pass",
        "split_pass",
        "sane_candidate",
        "overall_signals_total",
        "overall_entries_valid",
        "overall_entry_rate",
        "overall_exec_expectancy_net",
        "overall_baseline_expectancy_net",
        "overall_delta_expectancy_exec_minus_baseline",
        "overall_exec_cvar_5",
        "overall_baseline_cvar_5",
        "overall_cvar_improve_ratio",
        "overall_exec_max_drawdown",
        "overall_baseline_max_drawdown",
        "overall_maxdd_improve_ratio",
        "overall_exec_taker_share",
        "overall_exec_median_fill_delay_min",
        "overall_exec_p95_fill_delay_min",
        "min_split_expectancy_net",
        "median_split_expectancy_net",
        "std_split_expectancy_net",
        "tail_gate_pass_cvar",
        "tail_gate_pass_maxdd",
        "primary_equity_growth_proxy",
        "secondary_path_score",
        "penalty_overtrading",
        "penalty_taker",
        "composite_rank_score",
        "psr_proxy",
        "dsr_proxy",
        "g_entry_mode",
        "g_limit_offset_bps",
        "g_max_fill_delay_min",
        "g_fallback_to_market",
        "g_fallback_delay_min",
        "g_max_taker_share",
        "g_micro_vol_filter",
        "g_vol_threshold",
        "g_spread_guard_bps",
        "g_killzone_filter",
        "g_mss_displacement_gate",
        "g_min_entry_improvement_bps_gate",
        "g_tp_mult",
        "g_sl_mult",
        "g_time_stop_min",
        "g_break_even_enabled",
        "g_break_even_trigger_r",
        "g_break_even_offset_bps",
        "g_trailing_enabled",
        "g_trail_start_r",
        "g_trail_step_bps",
        "g_partial_take_enabled",
        "g_partial_take_r",
        "g_partial_take_pct",
        "g_skip_if_vol_gate",
        "g_use_signal_quality_gate",
        "g_min_signal_quality_gate",
        "g_cooldown_min",
        "invalid_reason",
    ]
    for c in keep_cols:
        if c not in g.columns:
            g[c] = np.nan

    out_df = g[keep_cols].copy()
    out_df.to_csv(run_dir / "pilot_run_results.csv", index=False)

    top5 = out_df[(out_df["is_metric_duplicate"] == 0)].head(5).copy()
    top5.to_csv(run_dir / "pilot_top5_nonduplicate.csv", index=False)

    summary: Dict[str, Any] = {
        "branch": branch,
        "total_candidates": int(len(g)),
        "nonduplicate_candidates": int((g["is_metric_duplicate"] == 0).sum()),
        "duplicate_candidates": int((g["is_metric_duplicate"] == 1).sum()),
        "sane_candidates": int((g["sane_candidate"] == 1).sum()),
        "valid_for_ranking_candidates": int((g["valid_for_ranking"] == 1).sum()),
        "effective_trials_proxy": float(eff_trials),
        "best_nondup_expectancy": float(top5["overall_exec_expectancy_net"].iloc[0]) if not top5.empty else float("nan"),
        "best_nondup_cvar_improve": float(top5["overall_cvar_improve_ratio"].iloc[0]) if not top5.empty else float("nan"),
        "best_nondup_maxdd_improve": float(top5["overall_maxdd_improve_ratio"].iloc[0]) if not top5.empty else float("nan"),
    }
    return out_df, summary


def _write_pilot_report(
    path: Path,
    ctx: PhaseInputs,
    checks: Dict[str, int],
    ga_run_dir: Path,
    summary: Dict[str, Any],
    results: pd.DataFrame,
) -> None:
    nondup = results[results["is_metric_duplicate"] == 0].copy()
    top5 = nondup.head(5).copy()

    def _fmt(v: Any, n: int = 6) -> str:
        try:
            fv = float(v)
            if np.isfinite(fv):
                return f"{fv:.{n}f}"
        except Exception:
            pass
        return "nan"

    diversity_lines: List[str] = []
    for c in [
        "overall_exec_expectancy_net",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "overall_entry_rate",
        "overall_exec_taker_share",
    ]:
        vals = pd.to_numeric(nondup.get(c, pd.Series(dtype=float)), errors="coerce")
        vals = vals[np.isfinite(vals)]
        if vals.empty:
            diversity_lines.append(f"- {c}: n/a")
        else:
            diversity_lines.append(
                f"- {c}: min={vals.min():.6f}, p50={vals.median():.6f}, max={vals.max():.6f}, std={vals.std(ddof=0):.6f}"
            )

    lines: List[str] = []
    lines.append("# Pilot Run Report")
    lines.append("")
    lines.append(f"- Generated UTC: {_utc_now().isoformat()}")
    lines.append(f"- Branch executed: **{ctx.branch}**")
    lines.append(f"- Pilot GA run dir: `{ga_run_dir}`")
    lines.append(f"- setup_checks_vs_expected: `{json.dumps(checks, sort_keys=True)}`")
    lines.append("")
    lines.append("## Pilot Outcome")
    lines.append("")
    lines.append(f"- total_candidates: {summary['total_candidates']}")
    lines.append(f"- nonduplicate_candidates: {summary['nonduplicate_candidates']}")
    lines.append(f"- duplicate_candidates: {summary['duplicate_candidates']}")
    lines.append(f"- valid_for_ranking_candidates: {summary['valid_for_ranking_candidates']}")
    lines.append(f"- sane_candidates: {summary['sane_candidates']}")
    lines.append(f"- effective_trials_proxy: {_fmt(summary['effective_trials_proxy'], 3)}")
    lines.append("")
    lines.append("Metric diversity among non-duplicate candidates:")
    lines.extend(diversity_lines)
    lines.append("")
    lines.append("## Top 5 Non-Duplicate Candidates")
    lines.append("")
    if top5.empty:
        lines.append("- none")
    else:
        show = top5[
            [
                "pilot_rank",
                "genome_hash",
                "overall_exec_expectancy_net",
                "overall_cvar_improve_ratio",
                "overall_maxdd_improve_ratio",
                "overall_entry_rate",
                "overall_exec_taker_share",
                "overall_exec_median_fill_delay_min",
                "sane_candidate",
                "valid_for_ranking",
                "invalid_reason",
            ]
        ].copy()
        lines.append("```csv")
        lines.append(show.to_csv(index=False).strip())
        lines.append("```")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This pilot confirms whether execution/exit knobs produce meaningful dispersion under frozen setup.")
    lines.append("- Primary objective in full run should be geometric fixed-size equity-step return.")
    lines.append("- Current GA engine exports expectancy/CVaR/maxDD, so this pilot uses expectancy as primary proxy and flags that geometric export is a follow-up enhancement.")
    lines.append("- `regime_concentration` hard gate remains TODO until per-candidate regime-bucket attribution is exported by the evaluator.")
    lines.append("")
    if int(summary["sane_candidates"]) == 0:
        lines.append("- Pilot verdict: **NO_GO** for full marathon until metric/pathology constraints are improved.")
    else:
        lines.append("- Pilot verdict: **GO** for a larger constrained execution/exit search, with hard constraints and anti-overfit controls retained.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_launch_prompt(path: Path, run_dir: Path, ga_run_dir: Path, ctx: PhaseInputs, summary: Dict[str, Any]) -> None:
    prompt = f"""Execution-layer GA full run (SOLUSDT, contract-locked pivot after Phase I):

Use frozen setup only:
- representative_subset_sha256={ctx.phaseh_manifest.get('representative_subset_sha256', '')}
- fee_model_sha256={ctx.phaseh_manifest.get('fee_model_sha256', '')}
- metrics_definition_sha256={ctx.phaseh_manifest.get('metrics_definition_sha256', '')}
- selected_model_set_sha256={ctx.phaseh_manifest.get('selected_model_set_sha256', '')}
- representative subset CSV: {ctx.phaseg_manifest.get('representative_subset_path', '')}

Run a constrained execution/exit optimization on SOLUSDT only (no signal-definition edits):
1) Entry knobs: delay/fallback/entry mode/limit offset/micro-vol filter/killzone/displacement/entry-improvement/cooldown.
2) Exit knobs: tp/sl, time stop, trailing, break-even, partial take.
3) Keep hard constraints active: min trades, entry-rate floors, taker-share cap, fill-delay caps, NaN/pathology rejection.
4) Add duplicate collapse and effective-trial reporting post-run; include PSR/DSR proxy and reality-check TODO note.
5) Rank with objective hierarchy:
   - Primary: fixed-size geometric equity-step return (or explicit proxy if still not exported)
   - Secondary: maxDD/CVaR improvement + split stability/support
   - Tertiary: expectancy/fill quality with penalties for overtrading and taker share.

Suggested command baseline:
`.venv/bin/python -m src.execution.ga_exec_3m_opt --symbol SOLUSDT --signals-csv {ctx.phaseg_manifest.get('representative_subset_path', '')} --max-signals 1200 --walkforward --wf-splits 5 --train-ratio 0.70 --mode tight --pop 192 --gens 24 --workers 4 --seed 20260223 --fee-bps-maker 2.0 --fee-bps-taker 4.0 --slippage-bps-limit 0.5 --slippage-bps-market 2.0 --execution-config configs/execution_configs.yaml --outdir reports/execution_layer`

Reference pilot:
- Phase J run dir: {run_dir}
- Pilot GA run dir: {ga_run_dir}
- Pilot sane candidates: {summary.get('sane_candidates', 'n/a')} / {summary.get('total_candidates', 'n/a')}
"""
    path.write_text(prompt, encoding="utf-8")


def _write_run_manifest(path: Path, ctx: PhaseInputs, checks: Dict[str, int], ga_run_dir: Path, summary: Dict[str, Any]) -> None:
    payload = {
        "generated_utc": _utc_now().isoformat(),
        "branch": ctx.branch,
        "branch_reason": ctx.branch_reason,
        "phaseh_source_dir": str(ctx.phaseh_dir),
        "phaseg_source_dir": str(ctx.phaseg_dir),
        "phasei_postfix_source_dir": str(ctx.postfix_dir),
        "phasei_forensic_source_dir": str(ctx.forensic_dir),
        "representative_subset_sha256": str(ctx.phaseh_manifest.get("representative_subset_sha256", "")),
        "fee_model_sha256": str(ctx.phaseh_manifest.get("fee_model_sha256", "")),
        "metrics_definition_sha256": str(ctx.phaseh_manifest.get("metrics_definition_sha256", "")),
        "selected_model_set_sha256": str(ctx.phaseh_manifest.get("selected_model_set_sha256", "")),
        "setup_checks_vs_expected": checks,
        "pilot_ga_run_dir": str(ga_run_dir),
        "pilot_summary": summary,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase J SOL execution-layer pivot planning + small pilot run.")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--pilot-candidates", type=int, default=48)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2))))
    ap.add_argument("--seed", type=int, default=20260222)
    ap.add_argument("--skip-pilot", action="store_true")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if int(args.pilot_candidates) < 30 or int(args.pilot_candidates) > 60:
        raise SystemExit("--pilot-candidates must be in [30, 60] for this phase.")

    ctx = _load_phase_inputs()
    checks = _setup_checks(ctx)

    out_root = _resolve(args.outdir)
    run_dir = out_root / f"PHASEJ_EXEC_PIVOT_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Always emit plan + search-space artifacts first.
    ga_dir_rel = str((run_dir / "pilot_ga_runs").resolve())
    _write_search_space_yaml(run_dir / "ga_search_space.yaml")
    _write_plan_md(run_dir / "next_experiment_plan.md", ctx=ctx, checks=checks, ga_dir_rel=ga_dir_rel)

    # Branch B still writes artifacts but avoids forcing execution pivot pilot.
    if ctx.branch == "B":
        empty = pd.DataFrame(
            columns=[
                "pilot_rank",
                "genome_hash",
                "is_metric_duplicate",
                "valid_for_ranking",
                "overall_exec_expectancy_net",
                "overall_cvar_improve_ratio",
                "overall_maxdd_improve_ratio",
                "invalid_reason",
            ]
        )
        empty.to_csv(run_dir / "pilot_run_results.csv", index=False)
        (run_dir / "duplicate_variant_map.csv").write_text("genome_hash,pilot_rank,metric_signature,is_metric_duplicate,duplicate_group_size\n", encoding="utf-8")
        summary = {
            "branch": ctx.branch,
            "total_candidates": 0,
            "nonduplicate_candidates": 0,
            "duplicate_candidates": 0,
            "sane_candidates": 0,
            "valid_for_ranking_candidates": 0,
            "effective_trials_proxy": float("nan"),
        }
        _write_pilot_report(
            run_dir / "pilot_run_report.md",
            ctx=ctx,
            checks=checks,
            ga_run_dir=Path("not_run_branch_B"),
            summary=summary,
            results=empty,
        )
        _write_launch_prompt(
            run_dir / "ready_to_launch_full_run_prompt.txt",
            run_dir=run_dir,
            ga_run_dir=Path("not_run_branch_B"),
            ctx=ctx,
            summary=summary,
        )
        _write_run_manifest(run_dir / "phaseJ_run_manifest.json", ctx=ctx, checks=checks, ga_run_dir=Path("not_run_branch_B"), summary=summary)
        print(str(run_dir))
        return

    if args.skip_pilot:
        raise SystemExit("--skip-pilot is not allowed for branch A/C in this phase.")

    ga_run_dir = _run_pilot_ga(run_dir=run_dir, ctx=ctx, args=args)
    results, summary = _build_pilot_outputs(run_dir=run_dir, ga_run_dir=ga_run_dir, ctx=ctx, branch=ctx.branch)

    _write_pilot_report(
        run_dir / "pilot_run_report.md",
        ctx=ctx,
        checks=checks,
        ga_run_dir=ga_run_dir,
        summary=summary,
        results=results,
    )
    _write_launch_prompt(
        run_dir / "ready_to_launch_full_run_prompt.txt",
        run_dir=run_dir,
        ga_run_dir=ga_run_dir,
        ctx=ctx,
        summary=summary,
    )
    _write_run_manifest(run_dir / "phaseJ_run_manifest.json", ctx=ctx, checks=checks, ga_run_dir=ga_run_dir, summary=summary)
    print(str(run_dir))


if __name__ == "__main__":
    random.seed(20260222)
    np.random.seed(20260222)
    main()
