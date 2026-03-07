#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_ae_signal_labeling as ae  # noqa: E402
from scripts import phase_af_ah_sizing_autorun as af  # noqa: E402
from scripts import phase_e_paper_confirm_autorun as econf  # noqa: E402
from scripts import phase_g_execaware_objective_redesign as gmod  # noqa: E402
from scripts import phase_u_combined_1h3m_pilot as pu  # noqa: E402
from scripts import sol_reconcile_truth as recon  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


LOCKED = {
    "phase_g_dir": "/root/analysis/0.87/reports/execution_layer/PHASEG_EXECAWARE_OBJECTIVE_20260223_220114",
    "frozen_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model_path": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition_path": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha256": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha256": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
    "allow_freeze_hash_mismatch": 0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(v, utc=True))
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def markdown_table(df: pd.DataFrame, cols: Sequence[str], n: int = 12) -> str:
    if df.empty:
        return "| (empty) |\n| --- |\n| (no rows) |"
    use = [c for c in cols if c in df.columns]
    d = df[use].head(n).copy()
    hdr = "| " + " | ".join(use) + " |"
    sep = "| " + " | ".join(["---"] * len(use)) + " |"
    rows = [hdr, sep]
    for _, r in d.iterrows():
        vals: List[str] = []
        for c in use:
            v = r[c]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("")
                else:
                    vals.append(f"{v:.10g}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def safe_div(a: float, b: float) -> float:
    if (not np.isfinite(a)) or (not np.isfinite(b)) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


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


def seq_sample(rng: random.Random, seq: Sequence[Any]) -> Any:
    if not seq:
        raise RuntimeError("empty sequence sample")
    return seq[rng.randrange(0, len(seq))]


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


@dataclass
class EvalCtx:
    rep_subset: pd.DataFrame
    rep_idx: pd.DataFrame
    signal_feat: pd.DataFrame
    df_feat: pd.DataFrame
    base_bundle: ga_exec.SymbolBundle
    exec_args: argparse.Namespace
    eval_choices: List[pu.ExecChoice]
    base_by_choice: Dict[str, Dict[str, Any]]
    proxy_idx: pd.DataFrame
    signal_meta: pd.DataFrame
    g1_valid: pd.DataFrame
    route_sets: Dict[str, pd.DataFrame]
    route_id_sets: Dict[str, set[str]]
    route_base_bundles: Dict[str, ga_exec.SymbolBundle]


def clone_args(args: argparse.Namespace) -> argparse.Namespace:
    out = argparse.Namespace()
    for k, v in vars(args).items():
        setattr(out, k, copy.deepcopy(v))
    return out


def build_exec_args(
    *,
    signals_csv: Path,
    seed: int,
    fee_mult: float = 1.0,
    extra_slip_bps: float = 0.0,
) -> argparse.Namespace:
    args = pu.build_exec_args(signals_csv=signals_csv, mode="tight", seed=seed)
    args.canonical_fee_model_path = LOCKED["canonical_fee_model_path"]
    args.canonical_metrics_definition_path = LOCKED["canonical_metrics_definition_path"]
    args.expected_fee_model_sha256 = LOCKED["expected_fee_sha256"]
    args.expected_metrics_definition_sha256 = LOCKED["expected_metrics_sha256"]
    args.allow_freeze_hash_mismatch = int(LOCKED["allow_freeze_hash_mismatch"])
    args.fee_bps_maker = float(args.fee_bps_maker) * float(fee_mult)
    args.fee_bps_taker = float(args.fee_bps_taker) * float(fee_mult)
    args.slippage_bps_limit = float(args.slippage_bps_limit) + float(extra_slip_bps)
    args.slippage_bps_market = float(args.slippage_bps_market) + float(extra_slip_bps)
    return args


def validate_phase_h1(
    *,
    run_dir: Path,
    seed: int,
) -> Tuple[str, Dict[str, Any], Optional[EvalCtx], Optional[Dict[str, Any]]]:
    t0 = time.time()
    lock_obj: Dict[str, Any] = {"generated_utc": utc_now()}

    phase_g_dir = Path(LOCKED["phase_g_dir"])
    subset_path = Path(LOCKED["frozen_subset_csv"])
    fee_path = Path(LOCKED["canonical_fee_model_path"])
    metrics_path = Path(LOCKED["canonical_metrics_definition_path"])

    missing = [str(p) for p in [phase_g_dir, subset_path, fee_path, metrics_path] if not p.exists()]
    if missing:
        reason = f"missing required paths: {missing}"
        write_text(run_dir / "phaseH_decision_next_step.md", f"# Phase H Decision\n\n- Classification: **STOP_INFRA**\n- Reason: {reason}")
        return "STOP_INFRA", {"reason": reason}, None, None

    fee_sha = sha256_file(fee_path)
    metrics_sha = sha256_file(metrics_path)
    lock_obj["fee_sha256"] = fee_sha
    lock_obj["metrics_sha256"] = metrics_sha
    lock_obj["expected_fee_sha256"] = LOCKED["expected_fee_sha256"]
    lock_obj["expected_metrics_sha256"] = LOCKED["expected_metrics_sha256"]
    lock_obj["fee_hash_match_expected"] = int(fee_sha == LOCKED["expected_fee_sha256"])
    lock_obj["metrics_hash_match_expected"] = int(metrics_sha == LOCKED["expected_metrics_sha256"])

    try:
        rep_subset = ae.ensure_signals_schema(pd.read_csv(subset_path))
    except Exception as e:
        reason = f"subset load/schema failure: {e}"
        write_text(run_dir / "phaseH_decision_next_step.md", f"# Phase H Decision\n\n- Classification: **STOP_INFRA**\n- Reason: {reason}")
        return "STOP_INFRA", {"reason": reason}, None, None

    lock_obj["subset_rows"] = int(len(rep_subset))
    lock_obj["subset_columns"] = list(rep_subset.columns)
    lock_obj["subset_non_empty"] = int(len(rep_subset) > 0)

    required_phaseg_files = [
        "phaseG1_execaware_signal_table.parquet",
        "phaseG3_candidate_results.csv",
        "phaseG3_objective_prototypes.md",
    ]
    lock_obj["phase_g_dir"] = str(phase_g_dir)
    lock_obj["phase_g_files_present"] = {f: int((phase_g_dir / f).exists()) for f in required_phaseg_files}

    lock_pass = (
        lock_obj["fee_hash_match_expected"] == 1
        and lock_obj["metrics_hash_match_expected"] == 1
        and lock_obj["subset_non_empty"] == 1
        and all(v == 1 for v in lock_obj["phase_g_files_present"].values())
    )

    # Build base context and smoke check.
    try:
        out_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
        exec_choices, exec_meta = pu.load_exec_choices(out_root)
        ch_map = {c.exec_choice_id: c for c in exec_choices}
        if "E1" not in ch_map or "E2" not in ch_map:
            raise RuntimeError("E1/E2 execution choices not available")
        eval_choices = [ch_map["E1"], ch_map["E2"]]

        params_path, base_params_raw, params_meta = pu.load_active_sol_params()
        base_params = ga_long._norm_params(copy.deepcopy(base_params_raw))
        df1h = recon._load_symbol_df("SOLUSDT", tf="1h")
        df_feat = ga_long._ensure_indicators(df1h.copy(), base_params)
        df_feat = ga_long._prepare_signal_df(df_feat, assume_prepared=False)

        rep_idx = pu.build_rep_subset_with_idx(rep_subset=rep_subset, df_feat=df_feat)
        signal_feat = gmod.build_signal_feature_frame(rep_subset=rep_subset, df_feat=df_feat)

        exec_args = build_exec_args(signals_csv=subset_path, seed=seed)
        lock_info = ga_exec._validate_and_lock_frozen_artifacts(args=exec_args, run_dir=run_dir)
        lock_obj["freeze_lock_validation"] = lock_info

        bundles, _ = ga_exec._prepare_bundles(exec_args)
        if not bundles:
            raise RuntimeError("no bundles prepared for smoke eval")
        base_bundle = bundles[0]

        # P00 smoke via active params.
        base_ids, base_diag = pu.active_signal_ids_for_params(df_feat=df_feat, params=base_params, rep_idx=rep_idx)
        p00_bundle = pu.build_candidate_bundle(base_bundle=base_bundle, active_ids=base_ids, args=exec_args)
        smoke_rows: List[Dict[str, Any]] = []
        base_by_choice: Dict[str, Dict[str, Any]] = {}
        for ch in eval_choices:
            met = pu.eval_exec_choice(bundle=p00_bundle, args=exec_args, choice=ch)
            base_by_choice[ch.exec_choice_id] = met
            smoke_rows.append(
                {
                    "exec_choice_id": ch.exec_choice_id,
                    "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
                    "overall_exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
                    "overall_exec_cvar_5": float(met.get("overall_exec_cvar_5", np.nan)),
                    "overall_exec_max_drawdown": float(met.get("overall_exec_max_drawdown", np.nan)),
                    "overall_entries_valid": int(met.get("overall_entries_valid", 0)),
                    "overall_entry_rate": float(met.get("overall_entry_rate", np.nan)),
                    "overall_exec_taker_share": float(met.get("overall_exec_taker_share", np.nan)),
                    "overall_exec_p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
                    "invalid_reason": str(met.get("invalid_reason", "")),
                }
            )
        smoke_df = pd.DataFrame(smoke_rows)
        smoke_ok = int(
            len(smoke_df) == 2
            and (to_num(smoke_df["valid_for_ranking"]) == 1).all()
            and np.isfinite(to_num(smoke_df["overall_exec_expectancy_net"]).to_numpy(dtype=float)).all()
        )

        g1 = pd.read_parquet(phase_g_dir / "phaseG1_execaware_signal_table.parquet")
        g1v = g1[g1["entry_for_labels"] == 1].copy()
        proxy_map = (
            g1v.groupby("signal_id", dropna=False)
            .agg(
                toxic_rate=("y_toxic_trade", "mean"),
                cluster_rate=("y_cluster_loss", "mean"),
                tail_rate=("y_tail_loss", "mean"),
            )
            .reset_index()
        )
        proxy_map["signal_id"] = proxy_map["signal_id"].astype(str)
        proxy_idx = proxy_map.set_index("signal_id")

        signal_meta = signal_feat.copy()
        signal_meta["signal_id"] = signal_meta["signal_id"].astype(str)
        signal_meta["regime_key"] = signal_meta["vol_bucket"].astype(str) + "|" + signal_meta["trend_bucket"].astype(str)

        route_sets = af.route_signal_sets(rep_subset)
        route_id_sets = {rid: set(df["signal_id"].astype(str).tolist()) for rid, df in route_sets.items()}
        route_base_bundles: Dict[str, ga_exec.SymbolBundle] = {}
        for rid, ids in route_id_sets.items():
            route_ctx = [ctx for ctx in base_bundle.contexts if str(ctx.signal_id) in ids]
            route_ctx = sorted(route_ctx, key=lambda r: (pd.to_datetime(r.signal_time, utc=True), str(r.signal_id)))
            splits = ga_exec._build_walkforward_splits(
                n=len(route_ctx),
                train_ratio=float(exec_args.train_ratio),
                n_splits=int(exec_args.wf_splits),
            )
            route_base_bundles[rid] = ga_exec.SymbolBundle(
                symbol=base_bundle.symbol,
                signals_csv=base_bundle.signals_csv,
                contexts=route_ctx,
                splits=splits,
                constraints=dict(base_bundle.constraints),
            )

        h1_manifest = {
            "generated_utc": utc_now(),
            "duration_sec": float(time.time() - t0),
            "phase": "H1",
            "lock_pass": int(lock_pass),
            "smoke_ok": int(smoke_ok),
            "phase_g_dir": str(phase_g_dir),
            "exec_choices": [{"id": c.exec_choice_id, "genome_hash": c.genome_hash} for c in eval_choices],
            "hard_gate_snapshot": {
                "hard_min_entry_rate_overall": float(exec_args.hard_min_entry_rate_overall),
                "hard_min_trade_frac_overall": float(exec_args.hard_min_trade_frac_overall),
                "hard_min_trades_overall": int(exec_args.hard_min_trades_overall),
                "hard_max_taker_share": float(exec_args.hard_max_taker_share),
                "hard_max_median_fill_delay_min": float(exec_args.hard_max_median_fill_delay_min),
                "hard_max_p95_fill_delay_min": float(exec_args.hard_max_p95_fill_delay_min),
                "allow_freeze_hash_mismatch": int(exec_args.allow_freeze_hash_mismatch),
            },
            "active_params_path": str(params_path),
            "active_params_meta": params_meta,
            "base_diag": base_diag,
            "exec_meta": exec_meta,
        }
        json_dump(run_dir / "phaseH1_run_manifest.json", h1_manifest)
        json_dump(run_dir / "phaseH1_contract_validation.json", lock_obj)
        json_dump(run_dir / "phaseH1_smoke_eval_check.json", {"generated_utc": utc_now(), "smoke_ok": int(smoke_ok), "rows": smoke_rows})

        if not lock_pass:
            reason = "contract/freeze lock validation failed"
            write_text(run_dir / "phaseH_decision_next_step.md", f"# Phase H Decision\n\n- Classification: **STOP_INFRA**\n- Reason: {reason}")
            return "STOP_INFRA", {"reason": reason}, None, None
        if smoke_ok != 1:
            reason = "smoke eval failed for P00/E1/E2"
            write_text(run_dir / "phaseH_decision_next_step.md", f"# Phase H Decision\n\n- Classification: **STOP_INFRA**\n- Reason: {reason}")
            return "STOP_INFRA", {"reason": reason}, None, None

        ctx = EvalCtx(
            rep_subset=rep_subset,
            rep_idx=rep_idx,
            signal_feat=signal_feat,
            df_feat=df_feat,
            base_bundle=base_bundle,
            exec_args=exec_args,
            eval_choices=eval_choices,
            base_by_choice=base_by_choice,
            proxy_idx=proxy_idx,
            signal_meta=signal_meta,
            g1_valid=g1v,
            route_sets=route_sets,
            route_id_sets=route_id_sets,
            route_base_bundles=route_base_bundles,
        )
        return "PASS", {"reason": "contract+smoke pass"}, ctx, h1_manifest
    except Exception as e:
        reason = f"H1 runtime failure: {e}"
        write_text(run_dir / "phaseH_decision_next_step.md", f"# Phase H Decision\n\n- Classification: **STOP_INFRA**\n- Reason: {reason}")
        return "STOP_INFRA", {"reason": reason}, None, None


def norm_params(p: Dict[str, Any]) -> Dict[str, Any]:
    q = ga_long._norm_params(copy.deepcopy(p))
    q["entry_rsi_min"] = clamp(float(q["entry_rsi_min"]), 5.0, 80.0)
    q["entry_rsi_max"] = clamp(float(q["entry_rsi_max"]), 20.0, 95.0)
    if float(q["entry_rsi_max"]) <= float(q["entry_rsi_min"]) + 1.0:
        q["entry_rsi_max"] = min(95.0, float(q["entry_rsi_min"]) + 5.0)
    q["adx_min"] = clamp(float(q.get("adx_min", 18.0)), 8.0, 40.0)
    q["cycle1_adx_boost"] = clamp(float(q.get("cycle1_adx_boost", 8.0)), 0.0, 30.0)
    q["cycle1_ema_sep_atr"] = clamp(float(q.get("cycle1_ema_sep_atr", 0.35)), 0.0, 2.0)
    q["willr_by_cycle"] = [clamp(float(x), -100.0, -1.0) for x in list(q.get("willr_by_cycle", [-60, -60, -60, -60, -60]))]
    tc = sorted({int(x) for x in list(q.get("trade_cycles", [1, 2, 3])) if int(x) in {1, 2, 3}})
    q["trade_cycles"] = tc if tc else [1, 2, 3]
    return q


def mutate_params(p: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    q = copy.deepcopy(p)
    ops = ["rsi", "adx", "boost", "sep", "willr", "cycles"]
    k = rng.randint(1, 3)
    for op in rng.sample(ops, k=k):
        if op == "rsi":
            d = seq_sample(rng, [-5, -3, -2, -1, 1, 2, 3, 5])
            q["entry_rsi_min"] = float(q.get("entry_rsi_min", 35.0)) + float(d)
            q["entry_rsi_max"] = float(q.get("entry_rsi_max", 70.0)) + float(d) + float(seq_sample(rng, [-1, 0, 1]))
        elif op == "adx":
            q["adx_min"] = float(q.get("adx_min", 18.0)) + float(seq_sample(rng, [-6, -4, -2, 2, 4, 6]))
        elif op == "boost":
            q["cycle1_adx_boost"] = float(q.get("cycle1_adx_boost", 8.0)) + float(seq_sample(rng, [-6, -3, -2, 2, 3, 6]))
        elif op == "sep":
            q["cycle1_ema_sep_atr"] = float(q.get("cycle1_ema_sep_atr", 0.35)) + float(seq_sample(rng, [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20]))
        elif op == "willr":
            w = list(q.get("willr_by_cycle", [-60, -60, -60, -60, -60]))
            if rng.random() < 0.5:
                d = float(seq_sample(rng, [-10, -8, -4, 4, 8, 10]))
                w = [float(x + d) for x in w]
            else:
                idx = int(seq_sample(rng, [1, 2, 3]))
                w[idx] = float(w[idx] + float(seq_sample(rng, [-12, -8, -4, 4, 8, 12])))
            q["willr_by_cycle"] = w
        elif op == "cycles":
            q["trade_cycles"] = copy.deepcopy(seq_sample(rng, [[1, 2], [1, 3], [2, 3], [1, 2, 3]]))
    return norm_params(q)


def crossover_params(a: Dict[str, Any], b: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    q = {}
    num_fields = ["entry_rsi_min", "entry_rsi_max", "adx_min", "cycle1_adx_boost", "cycle1_ema_sep_atr"]
    for f in num_fields:
        av = float(a.get(f, np.nan))
        bv = float(b.get(f, np.nan))
        if rng.random() < 0.5:
            q[f] = av
        else:
            q[f] = bv
        if rng.random() < 0.25 and np.isfinite(av) and np.isfinite(bv):
            q[f] = 0.5 * (av + bv)
    aw = list(a.get("willr_by_cycle", [-60, -60, -60, -60, -60]))
    bw = list(b.get("willr_by_cycle", [-60, -60, -60, -60, -60]))
    w = []
    for i in range(min(len(aw), len(bw))):
        w.append(float(aw[i] if rng.random() < 0.5 else bw[i]))
    while len(w) < 5:
        w.append(-60.0)
    q["willr_by_cycle"] = w
    q["trade_cycles"] = copy.deepcopy(a.get("trade_cycles", [1, 2, 3]) if rng.random() < 0.5 else b.get("trade_cycles", [1, 2, 3]))
    return norm_params(q)


def params_from_phaseg_seeds(ctx: EvalCtx, seed: int) -> Dict[str, Dict[str, Any]]:
    # Reconstruct exact candidate params from Phase G deterministic generator.
    params_path, base_params_raw, _ = pu.load_active_sol_params()
    _ = params_path
    base = norm_params(base_params_raw)
    cands = pu.generate_1h_candidates(base_params=base, n_total=24, seed=seed)
    out: Dict[str, Dict[str, Any]] = {}
    for c in cands:
        out[str(c["signal_candidate_id"])] = norm_params(c["params"])
    return out


def compute_proxy_features(active_ids: set[str], ctx: EvalCtx) -> Dict[str, float]:
    if not active_ids:
        return {
            "avg_toxic_proxy": float("nan"),
            "avg_cluster_proxy": float("nan"),
            "avg_tail_proxy": float("nan"),
            "dominant_session_share": float("nan"),
            "dominant_regime_share": float("nan"),
            "min_subperiod_expectancy_proxy": float("nan"),
            "route_subperiod_proxy_pass": 0,
        }
    sid = pd.Index(sorted(active_ids))
    px = ctx.proxy_idx.reindex(sid)
    avg_toxic = float(np.nanmean(to_num(px["toxic_rate"])))
    avg_cluster = float(np.nanmean(to_num(px["cluster_rate"])))
    avg_tail = float(np.nanmean(to_num(px["tail_rate"])))

    sm = ctx.signal_meta[ctx.signal_meta["signal_id"].isin(active_ids)].copy()
    if sm.empty:
        dom_session = float("nan")
        dom_regime = float("nan")
    else:
        dom_session = float(sm["session_bucket"].value_counts(normalize=True, dropna=False).iloc[0])
        dom_regime = float(sm["regime_key"].value_counts(normalize=True, dropna=False).iloc[0])

    gp = ctx.g1_valid[ctx.g1_valid["signal_id"].astype(str).isin(active_ids)].copy()
    if gp.empty:
        min_sub = float("nan")
        sub_pass = 0
    else:
        sub_agg = gp.groupby(["exec_choice_id", "route_id", "subperiod_id"], dropna=False)["pnl_net_trade_notional_dec"].mean().reset_index()
        min_sub = float(to_num(sub_agg["pnl_net_trade_notional_dec"]).min()) if not sub_agg.empty else float("nan")
        sub_pass = int(np.isfinite(min_sub) and min_sub > 0.0)
    return {
        "avg_toxic_proxy": avg_toxic,
        "avg_cluster_proxy": avg_cluster,
        "avg_tail_proxy": avg_tail,
        "dominant_session_share": dom_session,
        "dominant_regime_share": dom_regime,
        "min_subperiod_expectancy_proxy": min_sub,
        "route_subperiod_proxy_pass": sub_pass,
    }


def evaluate_candidate(
    *,
    cand_id: str,
    params: Dict[str, Any],
    origin: str,
    generation: int,
    ctx: EvalCtx,
    sig_cache: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[str]]:
    p = norm_params(params)
    p_hash = pu.param_fingerprint(p)
    active_ids, diag = pu.active_signal_ids_for_params(df_feat=ctx.df_feat, params=p, rep_idx=ctx.rep_idx)
    sig_hash = sha256_text("|".join(sorted(active_ids)))
    dup_of: Optional[str] = None
    if sig_hash in sig_cache:
        dup_of = str(sig_cache[sig_hash]["candidate_id"])

    px = compute_proxy_features(active_ids, ctx)
    bundle = pu.build_candidate_bundle(base_bundle=ctx.base_bundle, active_ids=active_ids, args=ctx.exec_args)
    row: Dict[str, Any] = {
        "candidate_id": str(cand_id),
        "generation": int(generation),
        "seed_origin": str(origin),
        "param_hash": str(p_hash),
        "signal_signature": str(sig_hash),
        "signals_active": int(len(active_ids)),
        "active_rate_vs_rep": float(diag.get("active_rate_vs_rep", np.nan)),
        "duplicate_of_candidate_id": dup_of if dup_of else "",
        "valid_for_ranking": 0,
        "exec_expectancy_net": float("nan"),
        "delta_expectancy_vs_exec_baseline": float("nan"),
        "cvar_improve_ratio": float("nan"),
        "maxdd_improve_ratio": float("nan"),
        "min_split_expectancy_net": float("nan"),
        "entries_valid": 0,
        "entry_rate": 0.0,
        "taker_share": float("nan"),
        "p95_fill_delay_min": float("nan"),
        "overall_exec_pnl_std": float("nan"),
        "split_count": int(ctx.exec_args.wf_splits),
        "support_flag": 0,
        "invalid_reason": "",
        "route_subperiod_proxy_pass": int(px["route_subperiod_proxy_pass"]),
        "min_subperiod_expectancy_proxy": float(px["min_subperiod_expectancy_proxy"]),
        "avg_toxic_proxy": float(px["avg_toxic_proxy"]),
        "avg_cluster_proxy": float(px["avg_cluster_proxy"]),
        "avg_tail_proxy": float(px["avg_tail_proxy"]),
        "dominant_session_share": float(px["dominant_session_share"]),
        "dominant_regime_share": float(px["dominant_regime_share"]),
        "candidate_hash": sha256_text(json.dumps(p, sort_keys=True)),
        "params_json": json.dumps(p, sort_keys=True),
    }

    if len(bundle.contexts) == 0:
        row["invalid_reason"] = "no_signals"
        row.update(gmod.candidate_objective_scores(pd.Series(row)))
        row["rank_key"] = json.dumps([float("-inf"), float("-inf"), float("-inf")])
        return row, [], dup_of

    choice_rows: List[Dict[str, Any]] = []
    invalid: List[str] = []
    for ch in ctx.eval_choices:
        met = pu.eval_exec_choice(bundle=bundle, args=ctx.exec_args, choice=ch)
        b = ctx.base_by_choice[ch.exec_choice_id]
        de = float(met.get("overall_exec_expectancy_net", np.nan) - b.get("overall_exec_expectancy_net", np.nan))
        cvar_imp = safe_div(
            abs(float(b.get("overall_exec_cvar_5", np.nan))) - abs(float(met.get("overall_exec_cvar_5", np.nan))),
            abs(float(b.get("overall_exec_cvar_5", np.nan))),
        )
        dd_imp = safe_div(
            abs(float(b.get("overall_exec_max_drawdown", np.nan))) - abs(float(met.get("overall_exec_max_drawdown", np.nan))),
            abs(float(b.get("overall_exec_max_drawdown", np.nan))),
        )
        cm = {
            "candidate_id": str(cand_id),
            "generation": int(generation),
            "seed_origin": str(origin),
            "exec_choice_id": str(ch.exec_choice_id),
            "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
            "exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
            "delta_expectancy_vs_exec_baseline": de,
            "cvar_improve_ratio": float(cvar_imp),
            "maxdd_improve_ratio": float(dd_imp),
            "overall_exec_pnl_std": float(met.get("overall_exec_pnl_std", np.nan)),
            "min_split_expectancy_net": float(met.get("min_split_expectancy_net", np.nan)),
            "entries_valid": int(met.get("overall_entries_valid", 0)),
            "entry_rate": float(met.get("overall_entry_rate", np.nan)),
            "taker_share": float(met.get("overall_exec_taker_share", np.nan)),
            "p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
            "invalid_reason": str(met.get("invalid_reason", "")),
        }
        if int(cm["valid_for_ranking"]) != 1 and cm["invalid_reason"]:
            invalid.append(cm["invalid_reason"])
        choice_rows.append(cm)

    cm_df = pd.DataFrame(choice_rows)
    row["valid_for_ranking"] = int((to_num(cm_df["valid_for_ranking"]) == 1).all())
    row["exec_expectancy_net"] = float(to_num(cm_df["exec_expectancy_net"]).mean())
    row["delta_expectancy_vs_exec_baseline"] = float(to_num(cm_df["delta_expectancy_vs_exec_baseline"]).mean())
    row["cvar_improve_ratio"] = float(to_num(cm_df["cvar_improve_ratio"]).mean())
    row["maxdd_improve_ratio"] = float(to_num(cm_df["maxdd_improve_ratio"]).mean())
    row["overall_exec_pnl_std"] = float(to_num(cm_df["overall_exec_pnl_std"]).mean())
    row["min_split_expectancy_net"] = float(to_num(cm_df["min_split_expectancy_net"]).min())
    row["entries_valid"] = int(to_num(cm_df["entries_valid"]).min())
    row["entry_rate"] = float(to_num(cm_df["entry_rate"]).min())
    row["taker_share"] = float(to_num(cm_df["taker_share"]).max())
    row["p95_fill_delay_min"] = float(to_num(cm_df["p95_fill_delay_min"]).max())
    row["support_flag"] = int(row["entries_valid"] >= 50 and row["entry_rate"] >= 0.10)
    row["invalid_reason"] = "|".join(sorted(set([x for x in invalid if x])))
    row.update(gmod.candidate_objective_scores(pd.Series(row)))
    oj2 = float(row.get("OJ2", np.nan))
    oj4 = float(row.get("OJ4", np.nan))
    de = float(row.get("delta_expectancy_vs_exec_baseline", np.nan))
    rank_vals = [oj2 if np.isfinite(oj2) else float("-inf"), oj4 if np.isfinite(oj4) else float("-inf"), de if np.isfinite(de) else float("-inf")]
    row["rank_key"] = json.dumps(rank_vals)
    return row, choice_rows, dup_of


def fitness(row: Dict[str, Any]) -> float:
    if int(row.get("valid_for_ranking", 0)) != 1:
        return float("-inf")
    v = float(row.get("OJ2", np.nan))
    return v if np.isfinite(v) else float("-inf")


def phase_h2_ga(
    *,
    run_dir: Path,
    ctx: EvalCtx,
    seed: int,
    pop: int,
    gens: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    t0 = time.time()
    rng = random.Random(seed)
    phaseg_seed_map = params_from_phaseg_seeds(ctx, seed=20260223)
    must_ids = ["P00", "P08", "P15", "P16", "P17"]
    seed_pool: List[Dict[str, Any]] = []
    for sid in must_ids:
        if sid in phaseg_seed_map:
            seed_pool.append({"params": phaseg_seed_map[sid], "origin": f"phaseg_seed_{sid}"})
    if not seed_pool:
        params_path, base_params, _ = pu.load_active_sol_params()
        _ = params_path
        seed_pool.append({"params": norm_params(base_params), "origin": "fallback_base"})

    population: List[Dict[str, Any]] = []
    # exact seeds
    for s in seed_pool:
        population.append({"params": norm_params(s["params"]), "origin": s["origin"]})

    # seed neighborhoods
    for s in seed_pool:
        for _ in range(8):
            population.append({"params": mutate_params(s["params"], rng), "origin": f"mut_seed_{s['origin']}"})

    # exploration mass
    _, base_params_raw, _ = pu.load_active_sol_params()
    expl = pu.generate_1h_candidates(base_params=norm_params(base_params_raw), n_total=max(64, pop * 2), seed=seed + 77)
    for c in expl:
        population.append({"params": norm_params(c["params"]), "origin": "explore"})
    rng.shuffle(population)
    if len(population) < pop:
        while len(population) < pop:
            population.append({"params": mutate_params(seed_pool[0]["params"], rng), "origin": "fill_mut"})
    population = population[:pop]

    evaluated_rows: List[Dict[str, Any]] = []
    choice_rows_all: List[Dict[str, Any]] = []
    eval_cache_by_param: Dict[str, Dict[str, Any]] = {}
    sig_cache: Dict[str, Dict[str, Any]] = {}
    dup_map_rows: List[Dict[str, Any]] = []

    telemetry: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "pop": int(pop),
        "gens": int(gens),
        "budget_target": int(pop * gens),
        "objective_primary": "OJ2",
        "objective_secondary": "OJ4",
        "seed_candidates": must_ids,
        "generation_stats": [],
        "origin_counts": {},
        "duplicate_param_reuse": 0,
        "duplicate_signal_signature": 0,
    }

    next_id = 0
    for gen in range(gens):
        gen_rows: List[Dict[str, Any]] = []
        for indiv in population:
            p = norm_params(indiv["params"])
            p_hash = pu.param_fingerprint(p)
            if p_hash in eval_cache_by_param:
                row = copy.deepcopy(eval_cache_by_param[p_hash])
                row["generation"] = int(gen)
                row["seed_origin"] = str(indiv["origin"])
                telemetry["duplicate_param_reuse"] = int(telemetry["duplicate_param_reuse"] + 1)
            else:
                cid = f"H{next_id:04d}"
                next_id += 1
                row, choice_rows, dup_of = evaluate_candidate(
                    cand_id=cid,
                    params=p,
                    origin=str(indiv["origin"]),
                    generation=gen,
                    ctx=ctx,
                    sig_cache=sig_cache,
                )
                eval_cache_by_param[p_hash] = copy.deepcopy(row)
                if dup_of:
                    telemetry["duplicate_signal_signature"] = int(telemetry["duplicate_signal_signature"] + 1)
                    dup_map_rows.append(
                        {
                            "candidate_id": str(row["candidate_id"]),
                            "duplicate_of_candidate_id": str(dup_of),
                            "signal_signature": str(row["signal_signature"]),
                            "param_hash": str(row["param_hash"]),
                            "seed_origin": str(row["seed_origin"]),
                        }
                    )
                else:
                    sig_cache[str(row["signal_signature"])] = {"candidate_id": str(row["candidate_id"])}
                evaluated_rows.append(copy.deepcopy(row))
                choice_rows_all.extend(choice_rows)
            gen_rows.append(row)
            telemetry["origin_counts"][str(indiv["origin"])] = int(telemetry["origin_counts"].get(str(indiv["origin"]), 0) + 1)

        gen_df = pd.DataFrame(gen_rows)
        if gen_df.empty:
            break
        valid_n = int((to_num(gen_df["valid_for_ranking"]) == 1).sum())
        best = gen_df.sort_values(["valid_for_ranking", "OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False, False]).head(1)
        best_obj = float(best.iloc[0]["OJ2"]) if not best.empty else float("nan")
        telemetry["generation_stats"].append(
            {
                "generation": int(gen),
                "population_size": int(len(gen_df)),
                "valid_for_ranking_count": int(valid_n),
                "best_OJ2": best_obj,
                "best_candidate_id": str(best.iloc[0]["candidate_id"]) if not best.empty else "",
            }
        )

        # Selection for next generation.
        gen_df["_fit"] = gen_df.apply(lambda r: fitness(r.to_dict()), axis=1)
        pool = gen_df.sort_values(["_fit", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)
        elites = pool.head(max(8, pop // 8))
        parents = pool.head(max(12, int(pop * 0.40)))
        parent_params: List[Dict[str, Any]] = []
        for _, r in parents.iterrows():
            try:
                parent_params.append(json.loads(str(r["params_json"])))
            except Exception:
                continue
        if not parent_params:
            parent_params = [norm_params(seed_pool[0]["params"])]

        new_pop: List[Dict[str, Any]] = []
        for _, r in elites.iterrows():
            try:
                p = json.loads(str(r["params_json"]))
            except Exception:
                continue
            new_pop.append({"params": norm_params(p), "origin": "elite"})

        while len(new_pop) < pop:
            u = rng.random()
            if u < 0.40 and len(parent_params) >= 2:
                pa = seq_sample(rng, parent_params)
                pb = seq_sample(rng, parent_params)
                child = crossover_params(pa, pb, rng)
                new_pop.append({"params": child, "origin": "crossover"})
            elif u < 0.80:
                pa = seq_sample(rng, parent_params)
                child = mutate_params(pa, rng)
                new_pop.append({"params": child, "origin": "mutation"})
            else:
                ex = seq_sample(rng, seed_pool)["params"]
                child = mutate_params(ex, rng)
                new_pop.append({"params": child, "origin": "explore"})
        population = new_pop[:pop]

    res = pd.DataFrame(evaluated_rows).drop_duplicates(subset=["candidate_hash"]).copy()
    if not res.empty:
        res = res.sort_values(["valid_for_ranking", "OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False, False]).reset_index(drop=True)
    choice_df = pd.DataFrame(choice_rows_all)
    dup_df = pd.DataFrame(dup_map_rows)

    # invalid histogram
    invalid_hist: Dict[str, int] = {}
    for _, r in res.iterrows():
        if int(r.get("valid_for_ranking", 0)) == 1:
            continue
        rs = str(r.get("invalid_reason", "")).strip()
        if not rs:
            invalid_hist["unknown_invalid"] = int(invalid_hist.get("unknown_invalid", 0) + 1)
            continue
        for part in rs.split("|"):
            part = part.strip()
            if not part:
                continue
            invalid_hist[part] = int(invalid_hist.get(part, 0) + 1)

    # effective trials and significance
    valid = res[to_num(res.get("valid_for_ranking", 0)) == 1].copy()
    metric_cols = [
        "delta_expectancy_vs_exec_baseline",
        "cvar_improve_ratio",
        "maxdd_improve_ratio",
        "min_split_expectancy_net",
        "entry_rate",
    ]
    for c in metric_cols:
        if c not in valid.columns:
            valid[c] = np.nan
    if not valid.empty:
        sig = pu.metric_signature(valid.rename(columns={
            "exec_expectancy_net": "overall_exec_expectancy_net",
            "entries_valid": "overall_entries_valid",
            "entry_rate": "overall_entry_rate",
            "taker_share": "overall_exec_taker_share",
            "p95_fill_delay_min": "overall_exec_p95_fill_delay_min",
            "candidate_hash": "signal_signature",
            "candidate_id": "exec_choice_id",
        }))
        valid["metric_signature"] = sig
        unique_metric = int(valid["metric_signature"].nunique())
        mat = valid[metric_cols].to_numpy(dtype=float)
        n_eff, avg_abs_corr = pu.effective_trials_from_corr(mat)
    else:
        unique_metric = 0
        n_eff = 0.0
        avg_abs_corr = 0.0

    shortlist = valid.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).head(12).copy()
    if not shortlist.empty:
        shortlist["psr_proxy"] = shortlist.apply(
            lambda r: norm_cdf(
                z_proxy(
                    float(r.get("exec_expectancy_net", np.nan)),
                    max(1e-12, float(r.get("overall_exec_pnl_std", np.nan))),
                    max(2.0, float(r.get("entries_valid", np.nan))),
                )
            ),
            axis=1,
        )
        dsr_denom = max(1.0, math.sqrt(max(1.0, float(n_eff))))
        shortlist["dsr_proxy"] = to_num(shortlist["psr_proxy"]) / dsr_denom
    else:
        shortlist["psr_proxy"] = []
        shortlist["dsr_proxy"] = []

    res.to_csv(run_dir / "phaseH2_pilot_results.csv", index=False)
    json_dump(run_dir / "phaseH2_invalid_reason_histogram.json", invalid_hist)
    json_dump(run_dir / "phaseH2_sampler_telemetry.json", telemetry)
    dup_df.to_csv(run_dir / "phaseH2_duplicate_variant_map.csv", index=False)
    shortlist.to_csv(run_dir / "phaseH2_shortlist_significance.csv", index=False)
    write_text(
        run_dir / "phaseH2_effective_trials_summary.md",
        "\n".join(
            [
                "# Phase H2 Effective Trials Summary",
                "",
                f"- Generated UTC: {utc_now()}",
                f"- Evaluated unique candidates: `{len(res)}`",
                f"- Valid candidates: `{int((to_num(res.get('valid_for_ranking', 0)) == 1).sum())}`",
                f"- Unique metric signatures (valid): `{unique_metric}`",
                f"- Duplicate-adjusted effective trials (corr-adjusted): `{float(n_eff):.4f}`",
                f"- Average absolute cross-metric correlation proxy: `{float(avg_abs_corr):.6f}`",
                "",
                "PSR/DSR are screening proxies only.",
            ]
        ),
    )
    rep_lines = [
        "# Phase H2 Pilot Report",
        "",
        f"- Generated UTC: {utc_now()}",
        f"- Duration sec: `{float(time.time() - t0):.3f}`",
        f"- Population: `{pop}`",
        f"- Generations: `{gens}`",
        f"- Evaluated unique candidates: `{len(res)}`",
        f"- Valid for ranking: `{int((to_num(res.get('valid_for_ranking', 0)) == 1).sum())}`",
        "",
        "## Top by OJ2",
        "",
        markdown_table(
            res.sort_values(["valid_for_ranking", "OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False, False]),
            [
                "candidate_id",
                "seed_origin",
                "valid_for_ranking",
                "OJ2",
                "exec_expectancy_net",
                "delta_expectancy_vs_exec_baseline",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "min_split_expectancy_net",
                "entries_valid",
                "entry_rate",
                "invalid_reason",
            ],
            n=15,
        ),
    ]
    write_text(run_dir / "phaseH2_pilot_report.md", "\n".join(rep_lines))
    return res, choice_df, {"invalid_hist": invalid_hist, "telemetry": telemetry, "n_eff": n_eff, "avg_abs_corr": avg_abs_corr}


def candidate_route_trade_tables(
    *,
    run_dir: Path,
    ctx: EvalCtx,
    params: Dict[str, Any],
    candidate_id: str,
    seed: int,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    active_ids, _ = pu.active_signal_ids_for_params(df_feat=ctx.df_feat, params=params, rep_idx=ctx.rep_idx)
    eval_art = run_dir / "phaseH3_eval_artifacts"
    eval_art.mkdir(parents=True, exist_ok=True)

    for rid, rdf in ctx.route_sets.items():
        route_ids = ctx.route_id_sets[rid]
        ids = sorted(active_ids.intersection(route_ids))
        if not ids:
            for ch in ctx.eval_choices:
                out[(rid, ch.exec_choice_id)] = pd.DataFrame(
                    columns=[
                        "signal_id",
                        "signal_time_utc",
                        "entry_for_labels",
                        "pnl_net_trade_notional_dec",
                        "fee_drag_trade",
                        "taker_flag",
                        "fill_delay_min",
                        "split_id",
                    ]
                )
            continue
        sig = rdf[rdf["signal_id"].astype(str).isin(ids)].copy().reset_index(drop=True)
        sig = ae.ensure_signals_schema(sig)
        for ch in ctx.eval_choices:
            met, sig_rows, _spl, _args, bundle = ae.evaluate_exact(
                run_dir=eval_art,
                signals_df=sig,
                genome=copy.deepcopy(ch.genome),
                seed=int(seed),
                name=f"{candidate_id}_{rid}_{ch.exec_choice_id}_{seed}",
            )
            _ = met
            pre = ae.build_preentry_features(bundle, ch.genome)
            lbl = ae.build_trade_labels(sig_rows.merge(pre, on=["signal_id", "signal_time"], how="left"))
            lbl["signal_time_utc"] = pd.to_datetime(lbl["signal_time_utc"], utc=True, errors="coerce")
            lbl["route_id"] = rid
            lbl["exec_choice_id"] = ch.exec_choice_id
            out[(rid, ch.exec_choice_id)] = lbl
    return out


def eval_route_choice_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "exec_expectancy_net": float("nan"),
            "exec_cvar_5": float("nan"),
            "exec_max_drawdown": float("nan"),
            "entries_valid": 0,
            "entry_rate": float("nan"),
            "taker_share": float("nan"),
            "p95_fill_delay_min": float("nan"),
            "min_split_expectancy_net": float("nan"),
        }
    x = df.copy()
    if "pnl_scenario" not in x.columns:
        x["pnl_scenario"] = to_num(x.get("pnl_net_trade_notional_dec", np.nan))
    if "fill_delay_scenario" not in x.columns:
        x["fill_delay_scenario"] = to_num(x.get("fill_delay_min", np.nan))
    keep = pd.Series(np.ones(len(df), dtype=bool), index=df.index)
    m = econf.eval_policy_metrics_on_scenario(x, keep_mask=keep)
    return {
        "exec_expectancy_net": float(m["exec_expectancy_net"]),
        "exec_cvar_5": float(m["exec_cvar_5"]),
        "exec_max_drawdown": float(m["exec_max_drawdown"]),
        "entries_valid": int(m["entries_valid"]),
        "entry_rate": float(m["entry_rate"]),
        "taker_share": float(m["taker_share"]),
        "p95_fill_delay_min": float(m["p95_fill_delay_min"]),
        "min_split_expectancy_net": float(m["min_split_expectancy_net"]),
    }


def subperiod_delta_cvar(df_c: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[float, float]:
    if df_c.empty or df_b.empty:
        return float("nan"), float("nan")

    def _label(df: pd.DataFrame) -> pd.DataFrame:
        x = df.sort_values(["signal_time_utc", "signal_id"]).reset_index(drop=True).copy()
        n = len(x)
        if n == 0:
            x["subperiod_id"] = np.nan
        else:
            x["subperiod_id"] = np.clip(np.floor(np.linspace(0, 3, n, endpoint=False)).astype(int), 0, 2)
        return x

    xc = _label(df_c)
    xb = _label(df_b)
    mins_delta: List[float] = []
    mins_cvar: List[float] = []
    for sid in [0, 1, 2]:
        cseg = xc[xc["subperiod_id"] == sid].copy()
        bseg = xb[xb["subperiod_id"] == sid].copy()
        if cseg.empty or bseg.empty:
            continue
        mc = eval_route_choice_metrics(cseg)
        mb = eval_route_choice_metrics(bseg)
        delta = float(mc["exec_expectancy_net"] - mb["exec_expectancy_net"])
        cvar_imp = safe_div(abs(float(mb["exec_cvar_5"])) - abs(float(mc["exec_cvar_5"])), abs(float(mb["exec_cvar_5"])))
        if np.isfinite(delta):
            mins_delta.append(delta)
        if np.isfinite(cvar_imp):
            mins_cvar.append(float(cvar_imp))
    return (
        float(min(mins_delta)) if mins_delta else float("nan"),
        float(min(mins_cvar)) if mins_cvar else float("nan"),
    )


def bootstrap_pass_rate(
    cand_tables: Dict[Tuple[str, str], pd.DataFrame],
    base_tables: Dict[Tuple[str, str], pd.DataFrame],
    n_boot: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(int(seed))
    keys = sorted([k for k in cand_tables.keys() if k in base_tables])
    if not keys:
        return float("nan")
    pass_hits = 0
    n_tot = 0
    for _ in range(n_boot):
        mins = {"de": [], "cvar": [], "dd": []}
        valid_boot = True
        for k in keys:
            cd = cand_tables[k]
            bd = base_tables[k]
            if cd.empty or bd.empty:
                valid_boot = False
                break
            ic = rng.integers(0, len(cd), size=len(cd))
            ib = rng.integers(0, len(bd), size=len(bd))
            cbs = cd.iloc[ic].reset_index(drop=True)
            bbs = bd.iloc[ib].reset_index(drop=True)
            mc = eval_route_choice_metrics(cbs)
            mb = eval_route_choice_metrics(bbs)
            de = float(mc["exec_expectancy_net"] - mb["exec_expectancy_net"])
            cvar_imp = safe_div(abs(float(mb["exec_cvar_5"])) - abs(float(mc["exec_cvar_5"])), abs(float(mb["exec_cvar_5"])))
            dd_imp = safe_div(abs(float(mb["exec_max_drawdown"])) - abs(float(mc["exec_max_drawdown"])), abs(float(mb["exec_max_drawdown"])))
            if (not np.isfinite(de)) or (not np.isfinite(cvar_imp)) or (not np.isfinite(dd_imp)):
                valid_boot = False
                break
            mins["de"].append(de)
            mins["cvar"].append(cvar_imp)
            mins["dd"].append(dd_imp)
        if not valid_boot or not mins["de"]:
            continue
        n_tot += 1
        if (min(mins["de"]) > 0.0) and (min(mins["cvar"]) >= 0.0) and (min(mins["dd"]) > 0.0):
            pass_hits += 1
    if n_tot == 0:
        return float("nan")
    return float(pass_hits / n_tot)


def phase_h3_robustness(
    *,
    run_dir: Path,
    ctx: EvalCtx,
    h2_res: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Survivors: top non-duplicate valid + references.
    z = h2_res.copy()
    z["valid_for_ranking"] = to_num(z["valid_for_ranking"]).fillna(0).astype(int)
    z = z[z["valid_for_ranking"] == 1].copy()
    z = z.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)
    z = z[z["duplicate_of_candidate_id"].fillna("").astype(str).str.strip() == ""].copy()
    surv = z.head(5).copy()

    # include baseline + phaseG P16 reference if present.
    refs = h2_res[h2_res["seed_origin"].astype(str).isin(["phaseg_seed_P00", "phaseg_seed_P16"])].copy()
    eval_df = pd.concat([surv, refs], axis=0, ignore_index=True).drop_duplicates(subset=["candidate_id"]).copy()
    eval_df = eval_df.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)

    if eval_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # build trade tables once for each candidate.
    cand_trade: Dict[str, Dict[Tuple[str, str], pd.DataFrame]] = {}
    for i, r in eval_df.iterrows():
        params = json.loads(str(r["params_json"]))
        cid = str(r["candidate_id"])
        cand_trade[cid] = candidate_route_trade_tables(run_dir=run_dir, ctx=ctx, params=params, candidate_id=cid, seed=seed + i * 97 + 11)

    # baseline reference from P00 (if missing, first row fallback).
    base_row = eval_df[eval_df["seed_origin"].astype(str).str.contains("phaseg_seed_P00", regex=True, na=False)].head(1)
    if base_row.empty:
        base_row = eval_df.head(1)
    base_id = str(base_row.iloc[0]["candidate_id"])
    base_tables = cand_trade[base_id]

    route_rows: List[Dict[str, Any]] = []
    for _, r in eval_df.iterrows():
        cid = str(r["candidate_id"])
        for rid in sorted(ctx.route_sets.keys()):
            de_list: List[float] = []
            cvar_list: List[float] = []
            dd_list: List[float] = []
            sub_de: List[float] = []
            sub_cv: List[float] = []
            entries: List[int] = []
            rates: List[float] = []
            for ch in ctx.eval_choices:
                key = (rid, ch.exec_choice_id)
                cd = cand_trade[cid].get(key, pd.DataFrame())
                bd = base_tables.get(key, pd.DataFrame())
                mc = eval_route_choice_metrics(cd)
                mb = eval_route_choice_metrics(bd)
                de = float(mc["exec_expectancy_net"] - mb["exec_expectancy_net"])
                cvar_imp = safe_div(abs(float(mb["exec_cvar_5"])) - abs(float(mc["exec_cvar_5"])), abs(float(mb["exec_cvar_5"])))
                dd_imp = safe_div(abs(float(mb["exec_max_drawdown"])) - abs(float(mc["exec_max_drawdown"])), abs(float(mb["exec_max_drawdown"])))
                sde, scv = subperiod_delta_cvar(cd, bd)
                if np.isfinite(de):
                    de_list.append(de)
                if np.isfinite(cvar_imp):
                    cvar_list.append(cvar_imp)
                if np.isfinite(dd_imp):
                    dd_list.append(dd_imp)
                if np.isfinite(sde):
                    sub_de.append(sde)
                if np.isfinite(scv):
                    sub_cv.append(scv)
                entries.append(int(mc["entries_valid"]))
                rates.append(float(mc["entry_rate"]))
            route_rows.append(
                {
                    "candidate_id": cid,
                    "route_id": rid,
                    "min_delta_expectancy_vs_base": float(min(de_list)) if de_list else float("nan"),
                    "min_cvar_improve_ratio_vs_base": float(min(cvar_list)) if cvar_list else float("nan"),
                    "min_maxdd_improve_ratio_vs_base": float(min(dd_list)) if dd_list else float("nan"),
                    "min_subperiod_delta": float(min(sub_de)) if sub_de else float("nan"),
                    "min_subperiod_cvar_improve": float(min(sub_cv)) if sub_cv else float("nan"),
                    "min_entries_valid": int(min(entries)) if entries else 0,
                    "min_entry_rate": float(min(rates)) if rates else float("nan"),
                    "route_pass": int(
                        (de_list and min(de_list) > 0.0)
                        and (cvar_list and min(cvar_list) >= 0.0)
                        and (dd_list and min(dd_list) > 0.0)
                        and (sub_de and min(sub_de) > 0.0)
                        and (sub_cv and min(sub_cv) >= 0.0)
                    ),
                }
            )
    route_df = pd.DataFrame(route_rows)

    scenarios = [
        {"scenario_id": "S00_base"},
        {"scenario_id": "S01_cost125", "cost_multiplier": 1.25},
        {"scenario_id": "S02_slip_p1", "extra_slippage_bps": 1.0},
        {"scenario_id": "S03_lat_entry1", "entry_delay_bars": 1, "latency_penalty_bps_per_bar": 1.0},
        {"scenario_id": "S04_spread15", "spread_multiplier": 1.5},
        {"scenario_id": "S05_cost125_slip1", "cost_multiplier": 1.25, "extra_slippage_bps": 1.0},
    ]

    stress_rows: List[Dict[str, Any]] = []
    for _, r in eval_df.iterrows():
        cid = str(r["candidate_id"])
        for sc in scenarios:
            sid = str(sc["scenario_id"])
            de_list: List[float] = []
            cv_list: List[float] = []
            dd_list: List[float] = []
            kept_rate: List[float] = []
            pathology = 0
            for rid in sorted(ctx.route_sets.keys()):
                for ch in ctx.eval_choices:
                    key = (rid, ch.exec_choice_id)
                    cd0 = cand_trade[cid].get(key, pd.DataFrame())
                    bd0 = base_tables.get(key, pd.DataFrame())
                    cd = econf.apply_scenario_to_route_df(cd0, sc)
                    bd = econf.apply_scenario_to_route_df(bd0, sc)
                    mc = eval_route_choice_metrics(cd)
                    mb = eval_route_choice_metrics(bd)
                    de = float(mc["exec_expectancy_net"] - mb["exec_expectancy_net"])
                    cv = safe_div(abs(float(mb["exec_cvar_5"])) - abs(float(mc["exec_cvar_5"])), abs(float(mb["exec_cvar_5"])))
                    dd = safe_div(abs(float(mb["exec_max_drawdown"])) - abs(float(mc["exec_max_drawdown"])), abs(float(mb["exec_max_drawdown"])))
                    if (not np.isfinite(de)) or (not np.isfinite(cv)) or (not np.isfinite(dd)):
                        pathology = 1
                    else:
                        de_list.append(de)
                        cv_list.append(cv)
                        dd_list.append(dd)
                    kept = safe_div(float(mc["entries_valid"]), max(1.0, float(mb["entries_valid"])))
                    kept_rate.append(float(kept) if np.isfinite(kept) else float("nan"))
            pass_flag = int(
                pathology == 0
                and de_list
                and cv_list
                and dd_list
                and min(de_list) > 0.0
                and min(cv_list) >= 0.0
                and min(dd_list) > 0.0
            )
            stress_rows.append(
                {
                    "candidate_id": cid,
                    "scenario_id": sid,
                    "min_delta_expectancy_vs_base": float(min(de_list)) if de_list else float("nan"),
                    "min_cvar_improve_ratio_vs_base": float(min(cv_list)) if cv_list else float("nan"),
                    "min_maxdd_improve_ratio_vs_base": float(min(dd_list)) if dd_list else float("nan"),
                    "min_filter_kept_entries_pct": float(np.nanmin(np.asarray(kept_rate, dtype=float))) if kept_rate else float("nan"),
                    "no_pathology": int(pathology == 0),
                    "scenario_pass": int(pass_flag),
                }
            )

        # bootstrap on base scenario for route perturbation/resampling.
        br = bootstrap_pass_rate(cand_trade[cid], base_tables, n_boot=200, seed=seed + 701 + hash(cid) % 100000)
        stress_rows.append(
            {
                "candidate_id": cid,
                "scenario_id": "BOOTSTRAP",
                "min_delta_expectancy_vs_base": float("nan"),
                "min_cvar_improve_ratio_vs_base": float("nan"),
                "min_maxdd_improve_ratio_vs_base": float("nan"),
                "min_filter_kept_entries_pct": float("nan"),
                "no_pathology": 1,
                "scenario_pass": int(np.isfinite(br) and br >= 0.60),
                "bootstrap_pass_rate": float(br),
            }
        )

    stress_df = pd.DataFrame(stress_rows)
    if not stress_df.empty and "bootstrap_pass_rate" not in stress_df.columns:
        stress_df["bootstrap_pass_rate"] = np.nan
    return route_df, stress_df


def classify_h4(
    *,
    run_dir: Path,
    h2_res: pd.DataFrame,
    route_df: pd.DataFrame,
    stress_df: pd.DataFrame,
) -> Tuple[str, str, Optional[str], pd.DataFrame]:
    valid = h2_res[to_num(h2_res["valid_for_ranking"]) == 1].copy()
    valid = valid[valid["duplicate_of_candidate_id"].fillna("").astype(str).str.strip() == ""].copy()
    valid = valid.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)

    # references
    p00 = h2_res[h2_res["seed_origin"].astype(str).str.contains("phaseg_seed_P00", regex=True, na=False)].head(1)
    p16 = h2_res[h2_res["seed_origin"].astype(str).str.contains("phaseg_seed_P16", regex=True, na=False)].head(1)

    p00_oj2 = float(p00.iloc[0]["OJ2"]) if not p00.empty else float("nan")
    p16_oj2 = float(p16.iloc[0]["OJ2"]) if not p16.empty else float("nan")

    agg_rows: List[Dict[str, Any]] = []
    for _, r in valid.iterrows():
        cid = str(r["candidate_id"])
        rr = route_df[route_df["candidate_id"] == cid].copy()
        ss = stress_df[(stress_df["candidate_id"] == cid) & (stress_df["scenario_id"] != "BOOTSTRAP")].copy()
        bs = stress_df[(stress_df["candidate_id"] == cid) & (stress_df["scenario_id"] == "BOOTSTRAP")].copy()
        route_pass_rate = float(np.mean(to_num(rr["route_pass"]) == 1)) if not rr.empty else 0.0
        scen_pass_rate = float(np.mean(to_num(ss["scenario_pass"]) == 1)) if not ss.empty else 0.0
        min_sub_delta = float(to_num(rr["min_subperiod_delta"]).min()) if not rr.empty else float("nan")
        min_sub_cvar = float(to_num(rr["min_subperiod_cvar_improve"]).min()) if not rr.empty else float("nan")
        bootstrap_rate = float(to_num(bs["bootstrap_pass_rate"]).iloc[0]) if (not bs.empty and np.isfinite(to_num(bs["bootstrap_pass_rate"]).iloc[0])) else float("nan")
        agg_rows.append(
            {
                "candidate_id": cid,
                "seed_origin": str(r["seed_origin"]),
                "OJ2": float(r["OJ2"]),
                "delta_expectancy_vs_exec_baseline": float(r["delta_expectancy_vs_exec_baseline"]),
                "cvar_improve_ratio": float(r["cvar_improve_ratio"]),
                "maxdd_improve_ratio": float(r["maxdd_improve_ratio"]),
                "entries_valid": int(r["entries_valid"]),
                "entry_rate": float(r["entry_rate"]),
                "route_pass_rate": route_pass_rate,
                "stress_pass_rate": scen_pass_rate,
                "bootstrap_pass_rate": bootstrap_rate,
                "min_subperiod_delta": min_sub_delta,
                "min_subperiod_cvar_improve": min_sub_cvar,
                "beats_p00_oj2": int(np.isfinite(p00_oj2) and float(r["OJ2"]) > p00_oj2 + 1e-6),
                "beats_p16_oj2": int(np.isfinite(p16_oj2) and float(r["OJ2"]) > p16_oj2 + 1e-6),
            }
        )
    agg_cols = [
        "candidate_id",
        "seed_origin",
        "OJ2",
        "delta_expectancy_vs_exec_baseline",
        "cvar_improve_ratio",
        "maxdd_improve_ratio",
        "entries_valid",
        "entry_rate",
        "route_pass_rate",
        "stress_pass_rate",
        "bootstrap_pass_rate",
        "min_subperiod_delta",
        "min_subperiod_cvar_improve",
        "beats_p00_oj2",
        "beats_p16_oj2",
    ]
    if agg_rows:
        agg = pd.DataFrame(agg_rows).sort_values(["stress_pass_rate", "route_pass_rate", "OJ2"], ascending=[False, False, False]).reset_index(drop=True)
    else:
        agg = pd.DataFrame(columns=agg_cols)

    if agg.empty:
        cls = "NO_GO_EXECUTION_DOMINANT_AGAIN"
        reason = "no non-duplicate valid candidates after H2"
        prompt = None
    else:
        robust = agg[
            (to_num(agg["delta_expectancy_vs_exec_baseline"]) >= 5e-5)
            & (to_num(agg["cvar_improve_ratio"]) >= 0.0)
            & (to_num(agg["maxdd_improve_ratio"]) >= 0.0)
            & (to_num(agg["route_pass_rate"]) >= 1.0)
            & (to_num(agg["stress_pass_rate"]) >= 0.60)
            & (to_num(agg["min_subperiod_delta"]) > 0.0)
            & (to_num(agg["min_subperiod_cvar_improve"]) >= 0.0)
            & (to_num(agg["entry_rate"]) >= 0.75 * float(to_num(h2_res["entry_rate"]).max()))
        ].copy()
        weak = agg[
            (to_num(agg["delta_expectancy_vs_exec_baseline"]) > 0.0)
            & (to_num(agg["cvar_improve_ratio"]) >= 0.0)
            & (to_num(agg["maxdd_improve_ratio"]) >= 0.0)
            & (to_num(agg["stress_pass_rate"]) >= 0.40)
        ].copy()

        if len(robust) >= 2 and int(robust["beats_p00_oj2"].max()) == 1 and int(robust["beats_p16_oj2"].max()) == 1:
            cls = "GO_STRONG"
            reason = "multiple non-duplicate candidates retained additive gains with route/split/stress-lite robustness"
            prompt = (
                "ROLE\n"
                "You are in Phase I execution-aware 1h GA expansion mode.\n\n"
                "MISSION\n"
                "Run a larger bounded execution-aware 1h GA using top Phase H survivors as seeds, with mandatory E1/E2 scoring and full route/split robustness checks.\n\n"
                "RULES\n"
                "1) Hard gates unchanged.\n"
                "2) Frozen contract lock enforced (allow_freeze_hash_mismatch=0).\n"
                "3) Increase compute moderately (e.g., 192x8), not marathon.\n"
                "4) Keep duplicate-adjusted significance, PSR/DSR proxies, and robustness matrix mandatory.\n"
                "5) Stop NO_GO if gains collapse under robustness."
            )
        elif len(weak) >= 1 and int(weak["beats_p00_oj2"].max()) == 1:
            cls = "GO_WEAK_PILOT_EXTEND"
            reason = "at least one candidate shows additive gain but robustness evidence is mixed"
            prompt = (
                "ROLE\n"
                "You are in Phase H2 bounded refinement mode.\n\n"
                "MISSION\n"
                "Run one more bounded execution-aware 1h pilot focused on robustness around the top Phase H weak survivors before any larger expansion.\n\n"
                "RULES\n"
                "1) Hard gates unchanged.\n"
                "2) Keep E1/E2 scoring and frozen lock mandatory.\n"
                "3) Tight candidate neighborhood around weak survivors + anti-overfit diversity mass.\n"
                "4) Require stronger route/split/stress-lite pass thresholds than prior H run.\n"
                "5) Stop NO_GO if improvements remain fragile."
            )
        else:
            # Distinguish execution-dominant vs robustness collapse.
            had_h2_gain = int((to_num(valid["delta_expectancy_vs_exec_baseline"]) > 5e-5).any())
            if had_h2_gain == 1:
                cls = "NO_GO_ROBUSTNESS_COLLAPSE"
                reason = "H2 additive gains did not survive route/split/stress-lite checks"
            else:
                cls = "NO_GO_EXECUTION_DOMINANT_AGAIN"
                reason = "GA pilot did not produce material additive signal-side gains beyond seeds/base"
            prompt = None

    lines: List[str] = []
    lines.append("# Phase H Decision")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append(f"- Classification: **{cls}**")
    lines.append(f"- Reason: {reason}")
    lines.append(f"- Non-duplicate valid candidates: `{len(valid)}`")
    lines.append("")
    lines.append("## Candidate Robustness Aggregate")
    lines.append("")
    lines.append(
        markdown_table(
            agg,
            [
                "candidate_id",
                "seed_origin",
                "OJ2",
                "delta_expectancy_vs_exec_baseline",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "route_pass_rate",
                "stress_pass_rate",
                "bootstrap_pass_rate",
                "min_subperiod_delta",
                "min_subperiod_cvar_improve",
                "beats_p00_oj2",
                "beats_p16_oj2",
            ],
            n=15,
        )
    )
    write_text(run_dir / "phaseH_decision_next_step.md", "\n".join(lines))

    if prompt is not None:
        write_text(run_dir / "ready_to_launch_phaseH2_or_I_prompt.txt", prompt)
    return cls, reason, prompt, agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase H execution-aware 1h GA pilot (bounded)")
    ap.add_argument("--seed", type=int, default=20260223)
    ap.add_argument("--pop", type=int, default=96)
    ap.add_argument("--gens", type=int, default=4)
    args = ap.parse_args()

    out_root = PROJECT_ROOT / "reports" / "execution_layer"
    run_dir = out_root / f"PHASEH_EXECAWARE_1H_GA_PILOT_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    # H1
    h1_cls, h1_meta, ctx, h1_manifest = validate_phase_h1(run_dir=run_dir, seed=int(args.seed))
    if h1_cls != "PASS" or ctx is None:
        manifest = {
            "generated_utc": utc_now(),
            "run_dir": str(run_dir),
            "classification": "STOP_INFRA",
            "mainline_status": "STOP_INFRA",
            "reason": str(h1_meta.get("reason", "H1 failed")),
            "duration_sec": float(time.time() - t0),
        }
        json_dump(run_dir / "phaseH_run_manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True))
        return

    # H2
    h2_res, h2_choice, h2_meta = phase_h2_ga(
        run_dir=run_dir,
        ctx=ctx,
        seed=int(args.seed) + 11,
        pop=int(args.pop),
        gens=int(args.gens),
    )
    _ = h2_choice
    if h2_res.empty:
        cls = "NO_GO_EXECUTION_DOMINANT_AGAIN"
        reason = "H2 produced empty candidate set"
        write_text(run_dir / "phaseH_decision_next_step.md", f"# Phase H Decision\n\n- Classification: **{cls}**\n- Reason: {reason}")
        manifest = {
            "generated_utc": utc_now(),
            "run_dir": str(run_dir),
            "classification": cls,
            "mainline_status": "STOP_NO_GO",
            "reason": reason,
            "duration_sec": float(time.time() - t0),
        }
        json_dump(run_dir / "phaseH_run_manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True))
        return

    # H3
    route_df, stress_df = phase_h3_robustness(
        run_dir=run_dir,
        ctx=ctx,
        h2_res=h2_res,
        seed=int(args.seed) + 101,
    )
    route_df.to_csv(run_dir / "phaseH3_route_split_checks.csv", index=False)
    stress_df.to_csv(run_dir / "phaseH3_robustness_matrix.csv", index=False)
    rep = []
    rep.append("# Phase H3 Top Survivors Report")
    rep.append("")
    rep.append(f"- Generated UTC: {utc_now()}")
    rep.append(f"- Route check rows: `{len(route_df)}`")
    rep.append(f"- Stress rows: `{len(stress_df)}`")
    rep.append("")
    rep.append("## Route/Split Checks")
    rep.append("")
    rep.append(
        markdown_table(
            route_df.sort_values(["route_pass", "min_delta_expectancy_vs_base"], ascending=[False, False]),
            [
                "candidate_id",
                "route_id",
                "min_delta_expectancy_vs_base",
                "min_cvar_improve_ratio_vs_base",
                "min_maxdd_improve_ratio_vs_base",
                "min_subperiod_delta",
                "min_subperiod_cvar_improve",
                "min_entries_valid",
                "min_entry_rate",
                "route_pass",
            ],
            n=30,
        )
    )
    rep.append("")
    rep.append("## Stress-Lite Matrix")
    rep.append("")
    rep.append(
        markdown_table(
            stress_df.sort_values(["scenario_pass", "min_delta_expectancy_vs_base"], ascending=[False, False]),
            [
                "candidate_id",
                "scenario_id",
                "min_delta_expectancy_vs_base",
                "min_cvar_improve_ratio_vs_base",
                "min_maxdd_improve_ratio_vs_base",
                "scenario_pass",
                "bootstrap_pass_rate",
            ],
            n=40,
        )
    )
    write_text(run_dir / "phaseH3_top_survivors_report.md", "\n".join(rep))

    # H4
    cls, reason, prompt, agg = classify_h4(run_dir=run_dir, h2_res=h2_res, route_df=route_df, stress_df=stress_df)
    mainline_status = "CONTINUE_READY_FOR_PHASEH2_OR_I" if cls in {"GO_STRONG", "GO_WEAK_PILOT_EXTEND"} else "STOP_NO_GO"
    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "classification": cls,
        "mainline_status": mainline_status,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "phaseH1_run_manifest": h1_manifest,
        "phaseH2_sampler_telemetry": h2_meta.get("telemetry", {}),
        "phaseH4_candidate_aggregate_rows": int(len(agg)),
        "prompt_generated": int(prompt is not None),
    }
    json_dump(run_dir / "phaseH_run_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
