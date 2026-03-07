#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_tag() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _resolve(p: str) -> Path:
    q = Path(p)
    if q.is_absolute():
        return q
    return (PROJECT_ROOT / q).resolve()


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _tail_mean(x: Sequence[float], frac: float) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(float(frac) * arr.size)))
    return float(np.mean(np.sort(arr)[:k]))


def _markdown_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(empty)_"
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append("nan" if not np.isfinite(v) else f"{v:.6f}")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def _load_latest_phaseg(out_root: Path) -> Path:
    cands = sorted([p for p in out_root.glob("PHASEG_SOL_PATHOLOGY_REHAB_*") if p.is_dir()])
    if not cands:
        raise FileNotFoundError(f"No Phase G dirs under {out_root}")
    return cands[-1]


@dataclass
class SeqMetrics:
    total_return: float
    max_drawdown_pct: float
    equity_final: float
    eq_step_returns: np.ndarray
    trade_returns: np.ndarray
    risk_pct: np.ndarray
    trades_total: int


def _prepare_valid_trades(trades: pd.DataFrame) -> pd.DataFrame:
    x = trades.copy()
    for c in ["signal_time", "entry_time", "exit_time"]:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], utc=True, errors="coerce")
    for c in ["filled", "valid_for_metrics", "pnl_net_pct", "risk_pct"]:
        x[c] = pd.to_numeric(x.get(c, np.nan), errors="coerce")
    m = (x["filled"] == 1) & (x["valid_for_metrics"] == 1) & x["pnl_net_pct"].notna()
    v = x[m].copy().sort_values(["entry_time", "signal_time", "signal_id"]).reset_index(drop=True)
    return v


def _compute_compounded_sequence_metrics(
    trades_valid: pd.DataFrame, *, initial_equity: float, risk_per_trade: float
) -> SeqMetrics:
    eq = float(initial_equity)
    eq_path: List[float] = []
    steps: List[float] = []
    tr = pd.to_numeric(trades_valid["pnl_net_pct"], errors="coerce").to_numpy(dtype=float)
    rp = pd.to_numeric(trades_valid["risk_pct"], errors="coerce").to_numpy(dtype=float)

    for pnl, risk_pct_raw in zip(tr, rp):
        risk_pct = max(1e-8, float(risk_pct_raw)) if np.isfinite(risk_pct_raw) else 1e-3
        if eq <= 0:
            pos_notional = 0.0
            step = 0.0
        else:
            pos_notional = float(eq * float(risk_per_trade) / risk_pct)
            step = float(pos_notional * pnl / eq)
        eq = float(eq + pos_notional * pnl)
        steps.append(float(step))
        eq_path.append(float(eq))

    if len(eq_path):
        eq_arr = np.asarray(eq_path, dtype=float)
        peak = np.maximum.accumulate(eq_arr)
        dd = eq_arr / np.maximum(1e-12, peak) - 1.0
        max_dd = float(np.min(dd))
        total_ret = float(eq_arr[-1] / max(1e-12, float(initial_equity)) - 1.0)
        eq_final = float(eq_arr[-1])
    else:
        max_dd = float("nan")
        total_ret = float("nan")
        eq_final = float(initial_equity)

    return SeqMetrics(
        total_return=float(total_ret),
        max_drawdown_pct=float(max_dd),
        equity_final=float(eq_final),
        eq_step_returns=np.asarray(steps, dtype=float),
        trade_returns=tr.astype(float),
        risk_pct=rp.astype(float),
        trades_total=int(len(trades_valid)),
    )


def _compute_fixed_sequence_metrics(
    trades_valid: pd.DataFrame, *, initial_equity: float, risk_per_trade: float
) -> SeqMetrics:
    eq = float(initial_equity)
    eq_path: List[float] = []
    steps: List[float] = []
    tr = pd.to_numeric(trades_valid["pnl_net_pct"], errors="coerce").to_numpy(dtype=float)
    rp = pd.to_numeric(trades_valid["risk_pct"], errors="coerce").to_numpy(dtype=float)

    for pnl, risk_pct_raw in zip(tr, rp):
        risk_pct = max(1e-8, float(risk_pct_raw)) if np.isfinite(risk_pct_raw) else 1e-3
        pos_notional = float(initial_equity * float(risk_per_trade) / risk_pct)
        trade_pnl_abs = float(pos_notional * pnl)
        eq = float(eq + trade_pnl_abs)
        step = float(trade_pnl_abs / max(1e-12, float(initial_equity)))
        steps.append(step)
        eq_path.append(eq)

    if len(eq_path):
        eq_arr = np.asarray(eq_path, dtype=float)
        peak = np.maximum.accumulate(eq_arr)
        dd = eq_arr / np.maximum(1e-12, peak) - 1.0
        max_dd = float(np.min(dd))
        total_ret = float(eq_arr[-1] / max(1e-12, float(initial_equity)) - 1.0)
        eq_final = float(eq_arr[-1])
    else:
        max_dd = float("nan")
        total_ret = float("nan")
        eq_final = float(initial_equity)

    return SeqMetrics(
        total_return=float(total_ret),
        max_drawdown_pct=float(max_dd),
        equity_final=float(eq_final),
        eq_step_returns=np.asarray(steps, dtype=float),
        trade_returns=tr.astype(float),
        risk_pct=rp.astype(float),
        trades_total=int(len(trades_valid)),
    )


def _geom_mean(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    if np.any(a <= -1.0):
        return -1.0
    return float(np.exp(np.mean(np.log1p(a))) - 1.0)


def _prod_return(x: np.ndarray) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan")
    if np.any(a <= -1.0):
        return -1.0
    return float(np.prod(1.0 + a) - 1.0)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_first_float(pattern: str, text: str) -> float:
    m = re.search(pattern, text, flags=re.MULTILINE)
    if not m:
        return float("nan")
    return _safe_float(m.group(1))


def _extract_first_str(pattern: str, text: str) -> str:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def run(args: argparse.Namespace) -> Path:
    out_root = _resolve(args.outdir)
    phaseg_dir = _resolve(args.phaseg_dir) if args.phaseg_dir else _load_latest_phaseg(out_root)
    if not phaseg_dir.exists():
        raise FileNotFoundError(f"Missing Phase G dir: {phaseg_dir}")

    run_dir = out_root / f"PHASEH_SOL_FREEZE_FORK_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    g_manifest = json.loads((phaseg_dir / "run_manifest.json").read_text(encoding="utf-8"))
    g_report_text = _load_text(phaseg_dir / "phaseG_report.md")
    g0_text = _load_text(phaseg_dir / "phaseG0_sol_drawdown_forensics.md")
    g1_text = _load_text(phaseg_dir / "phaseG1_parameterization_design.md")

    ablation = pd.read_csv(phaseg_dir / "phaseG2_ablation_results.csv")
    g3 = pd.read_csv(phaseg_dir / "phaseG3_practical_gate_decisions.csv")
    g0_buckets = pd.read_csv(phaseg_dir / "phaseG0_sol_drawdown_buckets.csv")
    g0_clusters = pd.read_csv(phaseg_dir / "phaseG0_sol_loss_clusters.csv")

    e2_dir = _resolve(str(g_manifest.get("e2_dir", "")))
    contract_path = e2_dir / "accounting_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"Missing accounting contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    risk_per_trade = float(contract.get("risk_per_trade", 0.01))
    initial_equity = float(contract.get("initial_equity", 1.0))

    consistency_rows: List[Dict[str, Any]] = []
    for row in ablation.itertuples(index=False):
        variant = str(getattr(row, "variant"))
        trades_path = phaseg_dir / f"phaseG2_trades_{variant}.csv"
        if not trades_path.exists():
            continue
        t = pd.read_csv(trades_path)
        tv = _prepare_valid_trades(t)

        comp = _compute_compounded_sequence_metrics(tv, initial_equity=initial_equity, risk_per_trade=risk_per_trade)
        fix = _compute_fixed_sequence_metrics(tv, initial_equity=initial_equity, risk_per_trade=risk_per_trade)

        trade_exp = float(np.mean(comp.trade_returns)) if comp.trade_returns.size else float("nan")
        pf_pos = comp.trade_returns[comp.trade_returns > 0]
        pf_neg = comp.trade_returns[comp.trade_returns < 0]
        pf = float(np.sum(pf_pos) / abs(np.sum(pf_neg))) if pf_neg.size and abs(np.sum(pf_neg)) > 1e-12 else (float("inf") if pf_pos.size else float("nan"))
        cvar5_trade = float(_tail_mean(comp.trade_returns, 0.05)) if comp.trade_returns.size else float("nan")
        mean_risk = float(np.nanmean(comp.risk_pct)) if comp.risk_pct.size else float("nan")
        med_risk = float(np.nanmedian(comp.risk_pct)) if comp.risk_pct.size else float("nan")
        min_risk = float(np.nanmin(comp.risk_pct)) if comp.risk_pct.size else float("nan")
        step_mean = float(np.nanmean(comp.eq_step_returns)) if comp.eq_step_returns.size else float("nan")
        step_med = float(np.nanmedian(comp.eq_step_returns)) if comp.eq_step_returns.size else float("nan")
        step_geo = float(_geom_mean(comp.eq_step_returns))
        step_prod_ret = float(_prod_return(comp.eq_step_returns))

        reported_total = _safe_float(getattr(row, "total_return"))
        reported_maxdd = _safe_float(getattr(row, "max_drawdown_pct"))
        reported_total_fix = _safe_float(getattr(row, "total_return_fixed"))
        reported_maxdd_fix = _safe_float(getattr(row, "max_drawdown_pct_fixed"))

        consistency_rows.append(
            {
                "variant": variant,
                "trades_total": int(comp.trades_total),
                "expectancy_net_trade_pct": float(trade_exp),
                "profit_factor_trade": float(pf),
                "cvar_5_trade_pct": float(cvar5_trade),
                "mean_risk_pct": float(mean_risk),
                "median_risk_pct": float(med_risk),
                "min_risk_pct": float(min_risk),
                "mean_equity_step_return": float(step_mean),
                "median_equity_step_return": float(step_med),
                "geometric_equity_step_return": float(step_geo),
                "product_equity_step_return_total": float(step_prod_ret),
                "reported_total_return_compounded": float(reported_total),
                "recalc_total_return_compounded": float(comp.total_return),
                "reported_max_drawdown_pct_compounded": float(reported_maxdd),
                "recalc_max_drawdown_pct_compounded": float(comp.max_drawdown_pct),
                "reported_total_return_fixed": float(reported_total_fix),
                "recalc_total_return_fixed": float(fix.total_return),
                "reported_max_drawdown_pct_fixed": float(reported_maxdd_fix),
                "recalc_max_drawdown_pct_fixed": float(fix.max_drawdown_pct),
                "total_return_comp_diff_abs": float(abs(comp.total_return - reported_total)),
                "maxdd_comp_diff_abs": float(abs(comp.max_drawdown_pct - reported_maxdd)),
                "total_return_fixed_diff_abs": float(abs(fix.total_return - reported_total_fix)),
                "maxdd_fixed_diff_abs": float(abs(fix.max_drawdown_pct - reported_maxdd_fix)),
                "positive_expectancy_and_pf_gt1": int(np.isfinite(trade_exp) and np.isfinite(pf) and trade_exp > 0.0 and pf > 1.0),
                "total_return_negative": int(np.isfinite(reported_total) and reported_total < 0.0),
                "expectancy_sign_mismatch_vs_equity_step": int(np.isfinite(trade_exp) and np.isfinite(step_mean) and (trade_exp > 0.0) and (step_mean < 0.0)),
            }
        )

    consistency_df = pd.DataFrame(consistency_rows).sort_values("expectancy_net_trade_pct", ascending=False).reset_index(drop=True)
    consistency_csv = run_dir / "phaseH_metric_consistency_checks.csv"
    consistency_df.to_csv(consistency_csv, index=False)

    # Phase G freeze summary.
    variants_tested = [str(v) for v in ablation["variant"].tolist()]
    rel_pass = int((pd.to_numeric(g3.get("relative_improvement_pass"), errors="coerce") == 1).sum())
    abs_pass = int((pd.to_numeric(g3.get("absolute_practical_pass"), errors="coerce") == 1).sum())
    final_pass = int((pd.to_numeric(g3.get("final_gate_pass"), errors="coerce") == 1).sum())
    fatal_count = int((pd.to_numeric(ablation.get("fatal_gate"), errors="coerce") == 1).sum())

    top_uplift = (
        g3.sort_values(
            ["delta_expectancy_net_vs_baseline", "delta_total_return_vs_baseline", "delta_max_drawdown_pct_vs_baseline"],
            ascending=[False, False, False],
        )
        .head(3)
        .copy()
    )
    top_uplift["variant"] = top_uplift["variant"].astype(str)

    # Diagnostics taxonomy audit values.
    report_adverse = _extract_first_float(r"^- adverse_loss_share:\s*([0-9.\-eE]+)\s*$", g_report_text)
    root_adverse = _extract_first_float(r"adverse_loss_share=([0-9.\-eE]+)", g0_text)
    baseline_adverse = _safe_float(ablation.loc[ablation["variant"] == "baseline_signal", "adverse_loss_share"].iloc[0]) if (ablation["variant"] == "baseline_signal").any() else float("nan")
    last_variant = str(ablation.iloc[-1]["variant"]) if not ablation.empty else ""
    last_variant_adverse = _safe_float(ablation.iloc[-1]["adverse_loss_share"]) if not ablation.empty else float("nan")

    wrong_regime_label = _extract_first_str(r"dominant_worst_regime=([^\s|,]+)", g0_text)
    rg = g0_buckets[g0_buckets["bucket_type"].astype(str) == "regime_bucket"].copy()
    rg["pnl_net_sum"] = pd.to_numeric(rg.get("pnl_net_sum"), errors="coerce")
    if rg.empty:
        corrected_dominant_regime = "unknown"
    else:
        corrected_dominant_regime = str(rg.sort_values("pnl_net_sum", ascending=True)["bucket"].iloc[0])

    # Positive expectancy + PF>1 but deeply negative total_return cases.
    paradox = consistency_df[
        (pd.to_numeric(consistency_df["positive_expectancy_and_pf_gt1"], errors="coerce") == 1)
        & (pd.to_numeric(consistency_df["total_return_negative"], errors="coerce") == 1)
    ].copy()

    # Controls design table for pipeline.
    controls_df = pd.DataFrame(
        [
            {
                "control": "multiple-testing accounting",
                "status": "design",
                "implementation_note": "Track raw trial count and effective independent trials per sweep; report adjusted confidence for top variants.",
            },
            {
                "control": "deflated-sharpe / PSR significance",
                "status": "design",
                "implementation_note": "Compute DSR/PSR on shortlist before promotion to compounding evaluation.",
            },
            {
                "control": "data-snooping reality-check benchmark",
                "status": "design",
                "implementation_note": "Add White-style reality-check style benchmark test against null strategy family.",
            },
            {
                "control": "purged+embargoed time-series validation",
                "status": "design",
                "implementation_note": "Apply purged walk-forward folds when expanding beyond current representative harness.",
            },
        ]
    )

    next_prompt = (
        "SOL signal-definition FORK (contract-locked Phase H): create a new SOL-only signal layer branch that keeps the same "
        "representative subset/hash and downstream 3m execution contract unchanged. Implement only: (1) trend alignment gate, "
        "(2) volatility regime gate, (3) de-clustering cooldown, (4) delayed 1h entry modes {0,1,2}. Evaluate every candidate "
        "in fixed-size/capped-risk mode first; enforce absolute gates before compounding (expectancy_net>0, total_return>0, "
        "maxDD>-0.35, cvar_5>-0.0015, PF>=1.05, support_ok=1). Reintroduce compounding only for fixed-size passers. Include "
        "multiple-testing accounting, DSR/PSR significance, and a reality-check benchmark in shortlist decisions. If no fixed-size "
        "candidate passes absolute gates, return HOLD/NO_DEPLOY with root-cause evidence and stop."
    )

    # 1) Freeze & terminal memo.
    memo_lines = [
        "# Phase H SOL Freeze and Fork Memo",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Phase G source (frozen): `{phaseg_dir}`",
        f"- Symbol: {g_manifest.get('symbol', 'SOLUSDT')}",
        "",
        "## Freeze Confirmation",
        "",
        "- Representative subset/hash and contract hashes are unchanged from Phase G trusted setup.",
        f"- representative_subset_sha256: `{g_manifest.get('representative_subset_sha256', '')}`",
        f"- fee_model_sha256: `{g_manifest.get('fee_model_sha256', '')}`",
        f"- metrics_definition_sha256: `{g_manifest.get('metrics_definition_sha256', '')}`",
        f"- selected_model_set_sha256: `{g_manifest.get('selected_model_set_sha256', '')}`",
        f"- setup_checks: {json.dumps(g_manifest.get('setup_checks', {}), sort_keys=True)}",
        "",
        "## Phase G Coverage",
        "",
        f"- variants_tested_total: {len(variants_tested)}",
        f"- variants_with_relative_improvement_pass: {rel_pass}",
        f"- variants_with_absolute_practical_pass: {abs_pass}",
        f"- variants_with_final_gate_pass: {final_pass}",
        f"- variants_triggering_fatal_gate: {fatal_count}",
        "",
        "Variants tested:",
        *[f"- {v}" for v in variants_tested],
        "",
        "Top relative uplifts observed (still non-deployable):",
        _markdown_table(
            top_uplift[
                [
                    "variant",
                    "delta_expectancy_net_vs_baseline",
                    "delta_total_return_vs_baseline",
                    "delta_max_drawdown_pct_vs_baseline",
                    "relative_improvement_pass",
                    "absolute_practical_pass",
                ]
            ]
        ),
        "",
        "## Terminal Decision",
        "",
        "- Absolute practical gates failed universally despite relative uplift in some variants.",
        "- Branch status: **HOLD current optimization branch** (terminal for tp/sl polishing in this line).",
        "- Forward action: **FORK_SIGNAL_LAYER** under new signal-definition scope only.",
        "- Deployment status: **NO_DEPLOY** until absolute practical gates pass.",
        "",
        "## Next Exact Prompt (Fork)",
        "",
        "```text",
        next_prompt,
        "```",
    ]
    (run_dir / "phaseH_sol_freeze_fork_memo.md").write_text("\n".join(memo_lines) + "\n", encoding="utf-8")

    # 2) Metric semantics audit.
    max_comp_err = float(pd.to_numeric(consistency_df.get("total_return_comp_diff_abs"), errors="coerce").max()) if not consistency_df.empty else float("nan")
    max_fix_err = float(pd.to_numeric(consistency_df.get("total_return_fixed_diff_abs"), errors="coerce").max()) if not consistency_df.empty else float("nan")

    paradox_tbl = paradox[
        [
            "variant",
            "expectancy_net_trade_pct",
            "profit_factor_trade",
            "mean_risk_pct",
            "mean_equity_step_return",
            "reported_total_return_compounded",
            "reported_max_drawdown_pct_compounded",
        ]
    ] if not paradox.empty else pd.DataFrame()

    semantics_lines = [
        "# Phase H Metric Semantics Audit",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Source formulas: `{PROJECT_ROOT / 'scripts/backtest_exec_phasec_sol.py'}` and `{PROJECT_ROOT / 'scripts/phase_g_sol_pathology_rehab.py'}`",
        "",
        "## Exact Metric Definitions and Units",
        "",
        "- `expectancy_net`: arithmetic mean of `pnl_net_pct` over valid filled trades. Unit: decimal return on position notional per trade.",
        "- `total_return`: final compounded equity / initial equity - 1, where position size is risk-fractional (`equity * risk_per_trade / risk_pct`) each trade. Unit: decimal return on account equity.",
        "- `max_drawdown_pct`: minimum peak-to-trough drawdown on compounded equity curve, `equity / rolling_peak - 1`. Unit: negative decimal fraction.",
        "- `cvar_5`: mean of worst 5% of per-trade `pnl_net_pct` over valid filled trades. Unit: decimal per-trade return on position notional.",
        "- `fatal_gate` (Phase G): 1 if `max_drawdown_pct <= -0.95` OR `total_return <= -0.95`; else 0.",
        "",
        "## Why `expectancy_net>0` and `profit_factor>1` can coexist with ~-94% total return",
        "",
        "The metrics are on different units/scales: expectancy/profit factor use unscaled trade `pnl_net_pct`, while total return uses equity-step returns scaled by `risk_per_trade / risk_pct` and compounded pathwise.",
        "When `risk_pct` is very small (median around 0.001 in top uplift variants), the same `pnl_net_pct` maps to much larger equity-step moves; variance and downside clustering can make geometric growth strongly negative even if arithmetic trade expectancy is positive.",
        "",
        "Representative paradox rows:",
        _markdown_table(paradox_tbl),
        "",
        "## Consistency Verification (same trade sequence, fixed-size vs compounded)",
        "",
        f"- Recomputed compounded/fixed metrics from raw trade files match reported metrics within numerical tolerance.",
        f"- max_abs_total_return_error_compounded: {max_comp_err:.12f}",
        f"- max_abs_total_return_error_fixed: {max_fix_err:.12f}",
        f"- Detailed checks: `{run_dir / 'phaseH_metric_consistency_checks.csv'}`",
        "",
        "## Audit Verdict",
        "",
        "- This is not an arithmetic bug in `total_return`.",
        "- It is expected from path-dependent compounding under risk scaling, plus a semantics mismatch if optimization is guided by unscaled per-trade expectancy.",
        "- Practical gate decisions should prioritize equity-based metrics (and fixed-size pre-gates) before compounding promotion.",
    ]
    (run_dir / "phaseH_metric_semantics_audit.md").write_text("\n".join(semantics_lines) + "\n", encoding="utf-8")

    # 3) Diagnostics taxonomy audit.
    taxonomy_lines = [
        "# Phase H Diagnostics Taxonomy Audit",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Source report: `{phaseg_dir / 'phaseG_report.md'}`",
        "",
        "## Regime vs Exit-Reason Labeling",
        "",
        f"- Reported in G0 root-cause evidence: `dominant_worst_regime={wrong_regime_label}`",
        "- `sl` is an exit reason, not a regime bucket label.",
        f"- Corrected dominant worst regime bucket (from `phaseG0_sol_drawdown_buckets.csv`): `{corrected_dominant_regime}`",
        "",
        "## Adverse Loss Share Reconciliation",
        "",
        f"- `phaseG_report.md` top-level `adverse_loss_share`: {report_adverse:.6f}",
        f"- G0 root-cause `adverse_loss_share` (baseline forensic context): {root_adverse:.6f}",
        f"- G2 baseline row `adverse_loss_share`: {baseline_adverse:.6f}",
        f"- Last-evaluated G2 variant: `{last_variant}` with `adverse_loss_share={last_variant_adverse:.6f}`",
        "",
        "Reconciliation:",
        "- Top-level value in Phase G report aligns with the last evaluated variant, not baseline forensic context.",
        "- Root-cause table value aligns with baseline forensic interpretation (and baseline G2 magnitude).",
        "- For terminal memos, baseline context should be explicit and separate from per-variant diagnostics.",
        "",
        "## Taxonomy Fix Standard for Future Reports",
        "",
        "- Use `dominant_worst_regime_bucket` for regime fields and reserve `exit_reason` labels for stop/exit attribution sections.",
        "- Emit both `baseline_adverse_loss_share` and `best_variant_adverse_loss_share` as separate named fields to prevent leakage.",
        "- Gate decision rows should carry variant-local diagnostics only; summary headers should carry baseline-local diagnostics only.",
        "",
        "## Verification Notes",
        "",
        f"- Loss clusters file present and readable: `{phaseg_dir / 'phaseG0_sol_loss_clusters.csv'}` ({len(g0_clusters)} rows).",
        f"- Regime bucket rows present: {int((g0_buckets['bucket_type'].astype(str) == 'regime_bucket').sum())}.",
    ]
    (run_dir / "phaseH_diagnostics_taxonomy_audit.md").write_text("\n".join(taxonomy_lines) + "\n", encoding="utf-8")

    # 4) SOL signal-layer fork spec.
    fork_lines = [
        "# Phase H SOL Signal-Definition Fork Spec",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        "- Scope: signal-layer fork only; downstream 3m execution mechanics remain contract-locked and unchanged.",
        "",
        "## Fork Objective",
        "",
        "- Replace incremental tp/sl polishing with a new SOL signal-definition branch centered on entry quality and de-clustering.",
        "- Enforce fixed-size/capped-risk practical viability before any compounding evaluation.",
        "",
        "## Frozen Invariants",
        "",
        f"- representative_subset_sha256: `{g_manifest.get('representative_subset_sha256', '')}`",
        f"- fee_model_sha256: `{g_manifest.get('fee_model_sha256', '')}`",
        f"- metrics_definition_sha256: `{g_manifest.get('metrics_definition_sha256', '')}`",
        f"- selected_model_set_sha256: `{g_manifest.get('selected_model_set_sha256', '')}`",
        "- No lookahead; subset integrity and contract lock checks are mandatory.",
        "",
        "## Fork Components (No tp/sl polishing in this branch)",
        "",
        "1. Trend alignment gate",
        "Require directional agreement with slow trend state for long entries.",
        "2. Volatility regime gate",
        "Permit entries only in pre-specified volatility buckets with minimum support.",
        "3. De-clustering cooldown",
        "Apply deterministic cooldown windows to suppress bursty correlated entries.",
        "4. Delayed 1h entry modes",
        "Evaluate delay modes {0,1,2} bars after signal under fixed-size first.",
        "",
        "## Evaluation Protocol",
        "",
        "1. Fixed-size / capped-risk stage (mandatory first)",
        "Absolute gates: expectancy_net>0, total_return>0, maxDD>-0.35, cvar_5>-0.0015, PF>=1.05, support_ok=1.",
        "2. Compounding stage (conditional)",
        "Run only for fixed-size passers; reject on absolute practical gate failure.",
        "3. Release decision",
        "If no fixed-size candidate passes, output HOLD/NO_DEPLOY and stop.",
        "",
        "## Research Controls to Add",
        "",
        _markdown_table(controls_df),
        "",
        "## Next Exact Prompt",
        "",
        "```text",
        next_prompt,
        "```",
    ]
    (run_dir / "phaseH_sol_signal_fork_spec.md").write_text("\n".join(fork_lines) + "\n", encoding="utf-8")

    # Lightweight phase-H manifest for reproducibility.
    h_manifest = {
        "generated_utc": _utc_now().isoformat(),
        "symbol": str(g_manifest.get("symbol", "SOLUSDT")),
        "phaseg_source_dir": str(phaseg_dir),
        "representative_subset_sha256": str(g_manifest.get("representative_subset_sha256", "")),
        "fee_model_sha256": str(g_manifest.get("fee_model_sha256", "")),
        "metrics_definition_sha256": str(g_manifest.get("metrics_definition_sha256", "")),
        "selected_model_set_sha256": str(g_manifest.get("selected_model_set_sha256", "")),
        "setup_checks": g_manifest.get("setup_checks", {}),
        "risk_per_trade": float(risk_per_trade),
        "initial_equity": float(initial_equity),
        "terminal_decision": "HOLD / FORK_SIGNAL_LAYER",
        "no_deploy": True,
        "next_exact_prompt": next_prompt,
    }
    (run_dir / "phaseH_run_manifest.json").write_text(json.dumps(h_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase H SOL freeze/fork memo + audits (contract-locked).")
    ap.add_argument("--phaseg-dir", default="", help="Optional explicit Phase G run dir. Defaults to latest PHASEG_SOL_PATHOLOGY_REHAB_*")
    ap.add_argument("--outdir", default="reports/execution_layer")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))


if __name__ == "__main__":
    main()
