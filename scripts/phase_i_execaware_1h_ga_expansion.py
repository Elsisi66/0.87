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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_ae_signal_labeling as ae  # noqa: E402
from scripts import phase_af_ah_sizing_autorun as af  # noqa: E402
from scripts import phase_e_paper_confirm_autorun as econf  # noqa: E402
from scripts import phase_h_execaware_1h_ga_pilot as ph  # noqa: E402
from scripts import phase_u_combined_1h3m_pilot as pu  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402


LOCKED = {
    "phase_h_dir": "/root/analysis/0.87/reports/execution_layer/PHASEH_EXECAWARE_1H_GA_PILOT_20260223_224855",
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
                vals.append("" if np.isnan(v) else f"{v:.10g}")
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


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


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


def seq_sample(rng: random.Random, seq: Sequence[Any]) -> Any:
    if not seq:
        raise RuntimeError("empty sequence sample")
    return seq[rng.randrange(0, len(seq))]


def mutate_params(p: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    q = copy.deepcopy(p)
    ops = ["rsi", "adx", "boost", "sep", "willr", "cycles"]
    for op in rng.sample(ops, k=rng.randint(1, 3)):
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
    q: Dict[str, Any] = {}
    for f in ["entry_rsi_min", "entry_rsi_max", "adx_min", "cycle1_adx_boost", "cycle1_ema_sep_atr"]:
        av = float(a.get(f, np.nan))
        bv = float(b.get(f, np.nan))
        q[f] = av if rng.random() < 0.5 else bv
        if rng.random() < 0.25 and np.isfinite(av) and np.isfinite(bv):
            q[f] = 0.5 * (av + bv)
    aw = list(a.get("willr_by_cycle", [-60, -60, -60, -60, -60]))
    bw = list(b.get("willr_by_cycle", [-60, -60, -60, -60, -60]))
    q["willr_by_cycle"] = [float(aw[i] if rng.random() < 0.5 else bw[i]) for i in range(5)]
    q["trade_cycles"] = copy.deepcopy(a.get("trade_cycles", [1, 2, 3]) if rng.random() < 0.5 else b.get("trade_cycles", [1, 2, 3]))
    return norm_params(q)


def phase_i1_contract_and_repro(
    *,
    run_dir: Path,
    seed: int,
) -> Tuple[str, Dict[str, Any], Optional[ph.EvalCtx], Dict[str, Any], Dict[str, Any]]:
    t0 = time.time()
    lock = {
        "generated_utc": utc_now(),
        "paths": {
            "phase_h_dir": LOCKED["phase_h_dir"],
            "subset_csv": LOCKED["frozen_subset_csv"],
            "fee_model": LOCKED["canonical_fee_model_path"],
            "metrics_definition": LOCKED["canonical_metrics_definition_path"],
        },
    }
    for k in ["phase_h_dir", "frozen_subset_csv", "canonical_fee_model_path", "canonical_metrics_definition_path"]:
        p = Path(LOCKED[k])
        lock[f"{k}_exists"] = int(p.exists())
        if not p.exists():
            reason = f"missing required path: {p}"
            return "STOP_INFRA", {"reason": reason}, None, lock, {}

    fee_sha = sha256_file(Path(LOCKED["canonical_fee_model_path"]))
    met_sha = sha256_file(Path(LOCKED["canonical_metrics_definition_path"]))
    lock["fee_sha256"] = fee_sha
    lock["metrics_sha256"] = met_sha
    lock["expected_fee_sha256"] = LOCKED["expected_fee_sha256"]
    lock["expected_metrics_sha256"] = LOCKED["expected_metrics_sha256"]
    lock["fee_hash_match_expected"] = int(fee_sha == LOCKED["expected_fee_sha256"])
    lock["metrics_hash_match_expected"] = int(met_sha == LOCKED["expected_metrics_sha256"])
    if lock["fee_hash_match_expected"] != 1 or lock["metrics_hash_match_expected"] != 1:
        reason = "canonical hash lock mismatch"
        return "STOP_INFRA", {"reason": reason}, None, lock, {}

    # Reuse validated H1 context builder to ensure identical evaluator path.
    h1_cls, h1_meta, ctx, h1_manifest = ph.validate_phase_h1(run_dir=run_dir, seed=seed)
    if h1_cls != "PASS" or ctx is None:
        return "STOP_INFRA", {"reason": f"H1 precheck failed: {h1_meta}"}, None, lock, {}

    # Seed reproduction check against Phase H survivors.
    h_dir = Path(LOCKED["phase_h_dir"])
    h2 = pd.read_csv(h_dir / "phaseH2_pilot_results.csv")
    must = ["H0313", "H0020"]
    tol_rows: List[Dict[str, Any]] = []
    repro_ok = 1
    for cid in must:
        z = h2[h2["candidate_id"].astype(str) == cid].copy()
        if z.empty:
            tol_rows.append({"candidate_id": cid, "exists_in_phaseH2": 0, "repro_pass": 0, "reason": "missing"})
            repro_ok = 0
            continue
        r = z.iloc[0]
        params = norm_params(json.loads(str(r["params_json"])))
        row, _choice_rows, _dup = ph.evaluate_candidate(
            cand_id=f"repro_{cid}",
            params=params,
            origin="phaseH_repro",
            generation=0,
            ctx=ctx,
            sig_cache={},
        )
        prior_delta = float(r.get("delta_expectancy_vs_exec_baseline", np.nan))
        new_delta = float(row.get("delta_expectancy_vs_exec_baseline", np.nan))
        prior_exp = float(r.get("exec_expectancy_net", np.nan))
        new_exp = float(row.get("exec_expectancy_net", np.nan))
        prior_cvar = float(r.get("cvar_improve_ratio", np.nan))
        new_cvar = float(row.get("cvar_improve_ratio", np.nan))
        prior_dd = float(r.get("maxdd_improve_ratio", np.nan))
        new_dd = float(row.get("maxdd_improve_ratio", np.nan))

        tol_delta = max(5e-5, 0.20 * abs(prior_delta))
        tol_exp = max(5e-5, 0.20 * abs(prior_exp))
        d_delta = abs(new_delta - prior_delta)
        d_exp = abs(new_exp - prior_exp)
        pass_row = int(
            np.isfinite(new_delta)
            and np.isfinite(new_exp)
            and (d_delta <= tol_delta)
            and (d_exp <= tol_exp)
            and ((new_cvar >= -1e-8) == (prior_cvar >= -1e-8))
            and ((new_dd > 0.0) == (prior_dd > 0.0))
            and int(row.get("valid_for_ranking", 0)) == int(r.get("valid_for_ranking", 0))
        )
        if pass_row != 1:
            repro_ok = 0
        tol_rows.append(
            {
                "candidate_id": cid,
                "exists_in_phaseH2": 1,
                "repro_pass": pass_row,
                "prior_exec_expectancy_net": prior_exp,
                "repro_exec_expectancy_net": new_exp,
                "prior_delta_expectancy_vs_exec_baseline": prior_delta,
                "repro_delta_expectancy_vs_exec_baseline": new_delta,
                "prior_cvar_improve_ratio": prior_cvar,
                "repro_cvar_improve_ratio": new_cvar,
                "prior_maxdd_improve_ratio": prior_dd,
                "repro_maxdd_improve_ratio": new_dd,
                "abs_diff_delta": d_delta,
                "delta_tolerance": tol_delta,
                "abs_diff_exec_expectancy": d_exp,
                "exec_expectancy_tolerance": tol_exp,
            }
        )

    repro = {
        "generated_utc": utc_now(),
        "phase_h_dir": str(h_dir),
        "reproduction_pass": int(repro_ok),
        "rows": tol_rows,
    }
    i1_manifest = {
        "generated_utc": utc_now(),
        "phase": "I1",
        "duration_sec": float(time.time() - t0),
        "lock_pass": int(lock["fee_hash_match_expected"] == 1 and lock["metrics_hash_match_expected"] == 1),
        "seed_reproduction_pass": int(repro_ok),
        "h1_manifest_reused": h1_manifest,
        "code_snapshot": {
            "phase_i_script": str((PROJECT_ROOT / "scripts" / "phase_i_execaware_1h_ga_expansion.py").resolve()),
            "phase_h_script": str((PROJECT_ROOT / "scripts" / "phase_h_execaware_1h_ga_pilot.py").resolve()),
            "ga_exec_script": str((PROJECT_ROOT / "src" / "execution" / "ga_exec_3m_opt.py").resolve()),
        },
    }
    return "PASS" if repro_ok == 1 else "STOP_INFRA", {"reason": "seed reproduction mismatch" if repro_ok != 1 else "ok"}, ctx, lock, {"seed_repro": repro, "manifest": i1_manifest}


def phase_i_seed_pool(h_dir: Path) -> Tuple[List[Tuple[str, Dict[str, Any]]], Dict[str, Any]]:
    h2 = pd.read_csv(h_dir / "phaseH2_pilot_results.csv")
    h3_route = pd.read_csv(h_dir / "phaseH3_route_split_checks.csv")

    route_rate = (
        h3_route.groupby("candidate_id", dropna=False)["route_pass"]
        .mean()
        .reset_index()
        .rename(columns={"route_pass": "route_pass_rate"})
    )
    h2x = h2.merge(route_rate, on="candidate_id", how="left")
    h2x["route_pass_rate"] = to_num(h2x["route_pass_rate"]).fillna(0.0)

    seeds: List[Tuple[str, Dict[str, Any]]] = []
    must = ["H0313", "H0020"]
    for cid in must:
        z = h2x[h2x["candidate_id"].astype(str) == cid].head(1)
        if z.empty:
            continue
        p = norm_params(json.loads(str(z.iloc[0]["params_json"])))
        seeds.append((f"phaseH_seed_{cid}", p))

    # Additional robust-ish seeds: route pass == 1, valid, non-dup.
    extra = h2x[
        (to_num(h2x["valid_for_ranking"]) == 1)
        & (h2x["duplicate_of_candidate_id"].fillna("").astype(str).str.strip() == "")
        & (to_num(h2x["route_pass_rate"]) >= 1.0)
    ].copy()
    extra = extra.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).head(8)
    for _, r in extra.iterrows():
        cid = str(r["candidate_id"])
        if cid in must:
            continue
        p = norm_params(json.loads(str(r["params_json"])))
        seeds.append((f"phaseH_routepass_{cid}", p))

    if not seeds:
        raise RuntimeError("No seed params loaded from Phase H")

    meta = {
        "phase_h_dir": str(h_dir),
        "mandatory_seed_ids": must,
        "seed_count": int(len(seeds)),
        "seed_ids": [s[0] for s in seeds],
    }
    return seeds, meta


def candidate_mix_population(
    *,
    pop: int,
    seed_pairs: List[Tuple[str, Dict[str, Any]]],
    rng: random.Random,
    explore_pool: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Mandatory mapping:
    # A) 30% near H0313, B) 25% near H0020, C) 20% near other route-pass seeds, D) 25% exploration.
    h0313 = next((p for n, p in seed_pairs if "H0313" in n), seed_pairs[0][1])
    h0020 = next((p for n, p in seed_pairs if "H0020" in n), seed_pairs[min(1, len(seed_pairs) - 1)][1])
    c_pool = [p for n, p in seed_pairs if ("H0313" not in n and "H0020" not in n)]
    if not c_pool:
        c_pool = [h0313, h0020]

    nA = int(round(pop * 0.30))
    nB = int(round(pop * 0.25))
    nC = int(round(pop * 0.20))
    nD = pop - nA - nB - nC
    out: List[Dict[str, Any]] = []

    # exact seeds first
    out.append({"params": copy.deepcopy(h0313), "seed_origin": "phaseH_seed_H0313"})
    out.append({"params": copy.deepcopy(h0020), "seed_origin": "phaseH_seed_H0020"})

    for _ in range(max(0, nA - 1)):
        out.append({"params": mutate_params(h0313, rng), "seed_origin": "near_H0313"})
    for _ in range(max(0, nB - 1)):
        out.append({"params": mutate_params(h0020, rng), "seed_origin": "near_H0020"})
    for _ in range(max(0, nC)):
        base = seq_sample(rng, c_pool)
        out.append({"params": mutate_params(base, rng), "seed_origin": "near_routepass_other"})
    for _ in range(max(0, nD)):
        if explore_pool:
            c = seq_sample(rng, explore_pool)
            out.append({"params": norm_params(c), "seed_origin": "exploration"})
        else:
            out.append({"params": mutate_params(h0313, rng), "seed_origin": "exploration_fallback"})
    rng.shuffle(out)
    if len(out) < pop:
        while len(out) < pop:
            out.append({"params": mutate_params(h0313, rng), "seed_origin": "fill_mut"})
    return out[:pop]


def phase_i2_ga(
    *,
    run_dir: Path,
    ctx: ph.EvalCtx,
    seed: int,
    pop: int,
    gens: int,
    h_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    t0 = time.time()
    rng = random.Random(seed)

    seed_pairs, seed_meta = phase_i_seed_pool(h_dir)
    _, base_params, _ = pu.load_active_sol_params()
    explore_cands = pu.generate_1h_candidates(base_params=norm_params(base_params), n_total=max(200, pop * 2), seed=seed + 97)
    explore_pool = [norm_params(c["params"]) for c in explore_cands]

    population = candidate_mix_population(pop=pop, seed_pairs=seed_pairs, rng=rng, explore_pool=explore_pool)
    initial_population_count = len(population)

    eval_cache: Dict[str, Dict[str, Any]] = {}
    sig_cache: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, Any]] = []
    choice_rows: List[Dict[str, Any]] = []
    dup_rows: List[Dict[str, Any]] = []

    telemetry: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "objective_primary": "OJ2",
        "objective_secondary": "OJ4",
        "pop": int(pop),
        "gens": int(gens),
        "budget_target": int(pop * gens),
        "seed_mix_meta": seed_meta,
        "initial_population_count": int(initial_population_count),
        "duplicate_param_reuse": 0,
        "duplicate_signal_signature": 0,
        "generation_stats": [],
        "origin_counts": {},
    }

    next_id = 0
    for gen in range(gens):
        gen_rows: List[Dict[str, Any]] = []
        for indiv in population:
            params = norm_params(indiv["params"])
            p_hash = pu.param_fingerprint(params)
            if p_hash in eval_cache:
                r = copy.deepcopy(eval_cache[p_hash])
                r["generation"] = int(gen)
                r["seed_origin"] = str(indiv["seed_origin"])
                telemetry["duplicate_param_reuse"] = int(telemetry["duplicate_param_reuse"] + 1)
            else:
                cid = f"I{next_id:05d}"
                next_id += 1
                r, cr, dup_of = ph.evaluate_candidate(
                    cand_id=cid,
                    params=params,
                    origin=str(indiv["seed_origin"]),
                    generation=int(gen),
                    ctx=ctx,
                    sig_cache=sig_cache,
                )
                eval_cache[p_hash] = copy.deepcopy(r)
                rows.append(copy.deepcopy(r))
                choice_rows.extend(cr)
                if dup_of:
                    telemetry["duplicate_signal_signature"] = int(telemetry["duplicate_signal_signature"] + 1)
                    dup_rows.append(
                        {
                            "candidate_id": str(r["candidate_id"]),
                            "duplicate_of_candidate_id": str(dup_of),
                            "signal_signature": str(r["signal_signature"]),
                            "param_hash": str(r["param_hash"]),
                            "seed_origin": str(r["seed_origin"]),
                        }
                    )
                else:
                    sig_cache[str(r["signal_signature"])] = {"candidate_id": str(r["candidate_id"])}
            gen_rows.append(r)
            telemetry["origin_counts"][str(indiv["seed_origin"])] = int(telemetry["origin_counts"].get(str(indiv["seed_origin"]), 0) + 1)

        gdf = pd.DataFrame(gen_rows)
        gdf["_fit"] = to_num(gdf.get("OJ2", np.nan))
        gdf.loc[to_num(gdf.get("valid_for_ranking", 0)) != 1, "_fit"] = float("-inf")
        gdf = gdf.sort_values(["_fit", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)
        valid_n = int((to_num(gdf["valid_for_ranking"]) == 1).sum())
        telemetry["generation_stats"].append(
            {
                "generation": int(gen),
                "population_size": int(len(gdf)),
                "valid_for_ranking_count": int(valid_n),
                "best_OJ2": float(gdf.iloc[0]["OJ2"]) if not gdf.empty else float("nan"),
                "best_candidate_id": str(gdf.iloc[0]["candidate_id"]) if not gdf.empty else "",
            }
        )

        # evolve next generation
        elites = gdf.head(max(12, pop // 10))
        parent_pool = gdf.head(max(24, int(pop * 0.45)))
        parents: List[Dict[str, Any]] = []
        for _, r in parent_pool.iterrows():
            try:
                parents.append(norm_params(json.loads(str(r["params_json"]))))
            except Exception:
                continue
        if not parents:
            parents = [seed_pairs[0][1]]

        new_pop: List[Dict[str, Any]] = []
        for _, r in elites.iterrows():
            try:
                p = norm_params(json.loads(str(r["params_json"])))
                new_pop.append({"params": p, "seed_origin": "elite"})
            except Exception:
                continue
        while len(new_pop) < pop:
            u = rng.random()
            if u < 0.45 and len(parents) >= 2:
                pa = seq_sample(rng, parents)
                pb = seq_sample(rng, parents)
                new_pop.append({"params": crossover_params(pa, pb, rng), "seed_origin": "crossover"})
            elif u < 0.80:
                pa = seq_sample(rng, parents)
                new_pop.append({"params": mutate_params(pa, rng), "seed_origin": "mutation"})
            else:
                if explore_pool:
                    ex = seq_sample(rng, explore_pool)
                else:
                    ex = seed_pairs[0][1]
                new_pop.append({"params": mutate_params(ex, rng), "seed_origin": "exploration"})
        population = new_pop[:pop]

    res = pd.DataFrame(rows).drop_duplicates(subset=["candidate_hash"]).copy()
    if res.empty:
        return res, pd.DataFrame(), telemetry
    res = res.sort_values(["valid_for_ranking", "OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False, False]).reset_index(drop=True)
    cr = pd.DataFrame(choice_rows)
    if not cr.empty:
        # attach per-choice component metrics
        piv = (
            cr.pivot_table(
                index="candidate_id",
                columns="exec_choice_id",
                values=["exec_expectancy_net", "delta_expectancy_vs_exec_baseline", "cvar_improve_ratio", "maxdd_improve_ratio"],
                aggfunc="first",
            )
            .sort_index(axis=1)
        )
        piv.columns = [f"{a}_{b}".lower() for a, b in piv.columns.to_flat_index()]
        piv = piv.reset_index()
        res = res.merge(piv, on="candidate_id", how="left")

    dup_df = pd.DataFrame(dup_rows)
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
            if part:
                invalid_hist[part] = int(invalid_hist.get(part, 0) + 1)

    # Duplicate stats
    valid = res[to_num(res["valid_for_ranking"]) == 1].copy()
    valid_nondup = valid[valid["duplicate_of_candidate_id"].fillna("").astype(str).str.strip() == ""].copy()
    metric_cols = ["delta_expectancy_vs_exec_baseline", "cvar_improve_ratio", "maxdd_improve_ratio", "min_split_expectancy_net", "entry_rate"]
    mat = valid_nondup[metric_cols].to_numpy(dtype=float) if not valid_nondup.empty else np.zeros((0, len(metric_cols)))
    n_eff, avg_abs_corr = pu.effective_trials_from_corr(mat)
    dup_stats = pd.DataFrame(
        [
            {
                "evaluated_unique_candidates": int(len(res)),
                "valid_for_ranking_count": int(len(valid)),
                "valid_nonduplicate_count": int(len(valid_nondup)),
                "duplicate_ratio_among_valid": float(1.0 - safe_div(float(len(valid_nondup)), float(max(1, len(valid))))),
                "effective_trials_corr_adjusted": float(n_eff),
                "avg_abs_metric_corr": float(avg_abs_corr),
                "duration_sec": float(time.time() - t0),
            }
        ]
    )

    shortlist = valid_nondup.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).head(20).copy()
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

    res.to_csv(run_dir / "phaseI2_ga_results.csv", index=False)
    json_dump(run_dir / "phaseI2_invalid_reason_histogram.json", invalid_hist)
    json_dump(run_dir / "phaseI2_sampler_telemetry.json", telemetry)
    dup_df.to_csv(run_dir / "phaseI2_duplicate_variant_map.csv", index=False)
    dup_stats.to_csv(run_dir / "phaseI2_duplicate_stats.csv", index=False)
    shortlist.to_csv(run_dir / "phaseI2_shortlist_significance.csv", index=False)
    write_text(
        run_dir / "phaseI2_effective_trials_summary.md",
        "\n".join(
            [
                "# Phase I2 Effective Trials Summary",
                "",
                f"- Generated UTC: {utc_now()}",
                f"- Evaluated unique candidates: `{len(res)}`",
                f"- Valid candidates: `{len(valid)}`",
                f"- Valid non-duplicate candidates: `{len(valid_nondup)}`",
                f"- Effective trials (corr-adjusted): `{float(n_eff):.4f}`",
                f"- Avg absolute metric correlation: `{float(avg_abs_corr):.6f}`",
                "- PSR/DSR are screening proxies only.",
            ]
        ),
    )
    rep = []
    rep.append("# Phase I2 GA Report")
    rep.append("")
    rep.append(f"- Generated UTC: {utc_now()}")
    rep.append(f"- Population: `{pop}`")
    rep.append(f"- Generations: `{gens}`")
    rep.append(f"- Evaluated unique candidates: `{len(res)}`")
    rep.append(f"- Valid for ranking: `{len(valid)}`")
    rep.append(f"- Valid non-duplicate: `{len(valid_nondup)}`")
    rep.append("")
    rep.append("## Top Non-duplicate Valid Candidates")
    rep.append("")
    rep.append(
        markdown_table(
            shortlist,
            [
                "candidate_id",
                "seed_origin",
                "OJ2",
                "exec_expectancy_net",
                "delta_expectancy_vs_exec_baseline",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "min_split_expectancy_net",
                "entries_valid",
                "entry_rate",
                "taker_share",
                "p95_fill_delay_min",
                "psr_proxy",
                "dsr_proxy",
            ],
            n=20,
        )
    )
    write_text(run_dir / "phaseI2_ga_report.md", "\n".join(rep))
    return res, shortlist, {"telemetry": telemetry, "dup_stats": dup_stats.iloc[0].to_dict()}


def phase_i3_robustness(
    *,
    run_dir: Path,
    ctx: ph.EvalCtx,
    i2_res: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # top non-dup valid 8 + mandatory H0313/H0020 if present.
    valid = i2_res[(to_num(i2_res["valid_for_ranking"]) == 1) & (i2_res["duplicate_of_candidate_id"].fillna("").astype(str).str.strip() == "")].copy()
    valid = valid.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)
    surv = valid.head(8).copy()
    refs = i2_res[i2_res["seed_origin"].astype(str).isin(["phaseH_seed_H0313", "phaseH_seed_H0020"])].copy()
    eval_df = pd.concat([surv, refs], axis=0, ignore_index=True).drop_duplicates(subset=["candidate_id"]).copy()
    eval_df = eval_df.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)
    if eval_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # base active from P00 seed in I2 if present else first row.
    base_row = i2_res[i2_res["seed_origin"].astype(str).str.contains("P00", regex=False, na=False)].head(1)
    if base_row.empty:
        base_row = i2_res.head(1)
    base_id = str(base_row.iloc[0]["candidate_id"])

    cand_trade: Dict[str, Dict[Tuple[str, str], pd.DataFrame]] = {}
    for i, r in eval_df.iterrows():
        cid = str(r["candidate_id"])
        params = norm_params(json.loads(str(r["params_json"])))
        cand_trade[cid] = ph.candidate_route_trade_tables(
            run_dir=run_dir,
            ctx=ctx,
            params=params,
            candidate_id=cid,
            seed=seed + i * 97 + 13,
        )
    base_tables = cand_trade[base_id]

    route_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    for _, r in eval_df.iterrows():
        cid = str(r["candidate_id"])
        for rid in sorted(ctx.route_sets.keys()):
            de_list: List[float] = []
            cv_list: List[float] = []
            dd_list: List[float] = []
            sde_list: List[float] = []
            scv_list: List[float] = []
            entries: List[int] = []
            rates: List[float] = []
            for ch in ctx.eval_choices:
                key = (rid, ch.exec_choice_id)
                cd = cand_trade[cid].get(key, pd.DataFrame())
                bd = base_tables.get(key, pd.DataFrame())
                mc = ph.eval_route_choice_metrics(cd)
                mb = ph.eval_route_choice_metrics(bd)
                de = float(mc["exec_expectancy_net"] - mb["exec_expectancy_net"])
                cv = safe_div(abs(float(mb["exec_cvar_5"])) - abs(float(mc["exec_cvar_5"])), abs(float(mb["exec_cvar_5"])))
                dd = safe_div(abs(float(mb["exec_max_drawdown"])) - abs(float(mc["exec_max_drawdown"])), abs(float(mb["exec_max_drawdown"])))
                sde, scv = ph.subperiod_delta_cvar(cd, bd)
                if np.isfinite(de):
                    de_list.append(de)
                if np.isfinite(cv):
                    cv_list.append(cv)
                if np.isfinite(dd):
                    dd_list.append(dd)
                if np.isfinite(sde):
                    sde_list.append(sde)
                if np.isfinite(scv):
                    scv_list.append(scv)
                entries.append(int(mc["entries_valid"]))
                rates.append(float(mc["entry_rate"]))
            route_pass = int(
                de_list
                and cv_list
                and dd_list
                and (min(de_list) > 0.0)
                and (min(cv_list) >= 0.0)
                and (min(dd_list) > 0.0)
                and sde_list
                and scv_list
                and (min(sde_list) > 0.0)
                and (min(scv_list) >= 0.0)
            )
            route_rows.append(
                {
                    "candidate_id": cid,
                    "route_id": rid,
                    "min_delta_expectancy_vs_base": float(min(de_list)) if de_list else float("nan"),
                    "min_cvar_improve_ratio_vs_base": float(min(cv_list)) if cv_list else float("nan"),
                    "min_maxdd_improve_ratio_vs_base": float(min(dd_list)) if dd_list else float("nan"),
                    "min_entries_valid": int(min(entries)) if entries else 0,
                    "min_entry_rate": float(min(rates)) if rates else float("nan"),
                    "route_pass": int(route_pass),
                }
            )
            split_rows.append(
                {
                    "candidate_id": cid,
                    "route_id": rid,
                    "min_subperiod_delta": float(min(sde_list)) if sde_list else float("nan"),
                    "min_subperiod_cvar_improve": float(min(scv_list)) if scv_list else float("nan"),
                    "split_stability_pass": int(sde_list and scv_list and min(sde_list) > 0.0 and min(scv_list) >= 0.0),
                }
            )

    route_df = pd.DataFrame(route_rows)
    split_df = pd.DataFrame(split_rows)

    scenarios = [
        {"scenario_id": "S00_base"},
        {"scenario_id": "S01_cost125", "cost_multiplier": 1.25},
        {"scenario_id": "S02_cost150", "cost_multiplier": 1.50},
        {"scenario_id": "S03_slip_p1", "extra_slippage_bps": 1.0},
        {"scenario_id": "S04_slip_p2", "extra_slippage_bps": 2.0},
        {"scenario_id": "S05_lat_entry1", "entry_delay_bars": 1, "latency_penalty_bps_per_bar": 1.0},
        {"scenario_id": "S06_spread15", "spread_multiplier": 1.5},
        {"scenario_id": "S07_cost125_slip1", "cost_multiplier": 1.25, "extra_slippage_bps": 1.0},
        {"scenario_id": "S08_trim_head10", "trim_head_pct": 0.10},
        {"scenario_id": "S09_trim_tail10", "trim_tail_pct": 0.10},
    ]

    stress_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    for _, r in eval_df.iterrows():
        cid = str(r["candidate_id"])
        for sc in scenarios:
            sid = str(sc["scenario_id"])
            de_list: List[float] = []
            cv_list: List[float] = []
            dd_list: List[float] = []
            kept_list: List[float] = []
            pathology = 0
            for rid in sorted(ctx.route_sets.keys()):
                for ch in ctx.eval_choices:
                    key = (rid, ch.exec_choice_id)
                    cd0 = cand_trade[cid].get(key, pd.DataFrame())
                    bd0 = base_tables.get(key, pd.DataFrame())
                    cd = econf.apply_scenario_to_route_df(cd0, sc)
                    bd = econf.apply_scenario_to_route_df(bd0, sc)
                    mc = ph.eval_route_choice_metrics(cd)
                    mb = ph.eval_route_choice_metrics(bd)
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
                    kept_list.append(float(kept) if np.isfinite(kept) else float("nan"))
            pass_flag = int(
                pathology == 0
                and de_list
                and cv_list
                and dd_list
                and (min(de_list) > 0.0)
                and (min(cv_list) >= 0.0)
                and (min(dd_list) > 0.0)
                and (np.nanmin(np.asarray(kept_list, dtype=float)) >= 0.60)
            )
            stress_rows.append(
                {
                    "candidate_id": cid,
                    "scenario_id": sid,
                    "min_delta_expectancy_vs_base": float(min(de_list)) if de_list else float("nan"),
                    "min_cvar_improve_ratio_vs_base": float(min(cv_list)) if cv_list else float("nan"),
                    "min_maxdd_improve_ratio_vs_base": float(min(dd_list)) if dd_list else float("nan"),
                    "min_filter_kept_entries_pct": float(np.nanmin(np.asarray(kept_list, dtype=float))) if kept_list else float("nan"),
                    "no_pathology": int(pathology == 0),
                    "scenario_pass": int(pass_flag),
                }
            )
        br = ph.bootstrap_pass_rate(cand_trade[cid], base_tables, n_boot=300, seed=seed + 900 + hash(cid) % 100000)
        boot_rows.append(
            {
                "candidate_id": cid,
                "bootstrap_pass_rate": float(br),
                "bootstrap_n": 300,
                "bootstrap_pass_flag_0p14": int(np.isfinite(br) and br >= 0.14),
                "bootstrap_pass_flag_0p10": int(np.isfinite(br) and br >= 0.10),
                "bootstrap_pass_flag_0p05": int(np.isfinite(br) and br >= 0.05),
            }
        )

    stress_df = pd.DataFrame(stress_rows)
    boot_df = pd.DataFrame(boot_rows)
    return route_df, split_df, stress_df.merge(boot_df[["candidate_id", "bootstrap_pass_rate"]], on="candidate_id", how="left")


def phase_i4_decision(
    *,
    run_dir: Path,
    i2_res: pd.DataFrame,
    route_df: pd.DataFrame,
    split_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    h_dir: Path,
) -> Tuple[str, str, Optional[str], pd.DataFrame]:
    # Build candidate-level robustness summary.
    valid = i2_res[(to_num(i2_res["valid_for_ranking"]) == 1) & (i2_res["duplicate_of_candidate_id"].fillna("").astype(str).str.strip() == "")].copy()
    valid = valid.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)
    if valid.empty:
        return "I_NO_GO_EXECUTION_DOMINANT_AGAIN", "no valid non-duplicate candidates after I2", None, pd.DataFrame()

    route_sum = (
        route_df.groupby("candidate_id", dropna=False)
        .agg(
            route_pass_rate=("route_pass", "mean"),
            min_route_delta=("min_delta_expectancy_vs_base", "min"),
            min_route_cvar=("min_cvar_improve_ratio_vs_base", "min"),
            min_route_dd=("min_maxdd_improve_ratio_vs_base", "min"),
            min_route_entries=("min_entries_valid", "min"),
            min_route_entry_rate=("min_entry_rate", "min"),
        )
        .reset_index()
    )
    split_sum = (
        split_df.groupby("candidate_id", dropna=False)
        .agg(
            min_subperiod_delta=("min_subperiod_delta", "min"),
            min_subperiod_cvar_improve=("min_subperiod_cvar_improve", "min"),
            split_stability_pass_rate=("split_stability_pass", "mean"),
        )
        .reset_index()
    )
    s_nonboot = stress_df.copy()
    s_nonboot = s_nonboot[s_nonboot["scenario_id"] != "BOOTSTRAP"].copy()
    stress_sum = (
        s_nonboot.groupby("candidate_id", dropna=False)
        .agg(
            stress_pass_rate=("scenario_pass", "mean"),
            min_stress_delta=("min_delta_expectancy_vs_base", "min"),
            min_stress_cvar=("min_cvar_improve_ratio_vs_base", "min"),
            min_stress_dd=("min_maxdd_improve_ratio_vs_base", "min"),
            min_stress_kept=("min_filter_kept_entries_pct", "min"),
        )
        .reset_index()
    )
    boot_sum = (
        stress_df.groupby("candidate_id", dropna=False)
        .agg(bootstrap_pass_rate=("bootstrap_pass_rate", "max"))
        .reset_index()
    )

    agg = valid.merge(route_sum, on="candidate_id", how="left").merge(split_sum, on="candidate_id", how="left").merge(stress_sum, on="candidate_id", how="left").merge(boot_sum, on="candidate_id", how="left")

    # Phase H refs.
    h2 = pd.read_csv(h_dir / "phaseH2_pilot_results.csv")
    h0313 = h2[h2["candidate_id"].astype(str) == "H0313"].head(1)
    h0020 = h2[h2["candidate_id"].astype(str) == "H0020"].head(1)
    h0313_oj2 = float(h0313.iloc[0]["OJ2"]) if not h0313.empty else float("nan")
    h0313_delta = float(h0313.iloc[0]["delta_expectancy_vs_exec_baseline"]) if not h0313.empty else float("nan")
    h0020_oj2 = float(h0020.iloc[0]["OJ2"]) if not h0020.empty else float("nan")

    agg["beats_H0313_OJ2"] = ((to_num(agg["OJ2"]) > h0313_oj2 + 1e-6)).astype(int)
    agg["beats_H0313_delta"] = ((to_num(agg["delta_expectancy_vs_exec_baseline"]) > h0313_delta + 1e-6)).astype(int)
    agg["beats_H0020_OJ2"] = ((to_num(agg["OJ2"]) > h0020_oj2 + 1e-6)).astype(int)

    # stable neighborhood around winner (param distance).
    winner = agg.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).head(1)
    stable_neighborhood = 0
    near_count = 0
    if not winner.empty:
        wp = json.loads(str(winner.iloc[0]["params_json"]))

        def dist(p: Dict[str, Any], q: Dict[str, Any]) -> float:
            p = norm_params(p)
            q = norm_params(q)
            d = 0.0
            d += abs(float(p["entry_rsi_min"]) - float(q["entry_rsi_min"])) / 20.0
            d += abs(float(p["entry_rsi_max"]) - float(q["entry_rsi_max"])) / 20.0
            d += abs(float(p.get("adx_min", 18.0)) - float(q.get("adx_min", 18.0))) / 12.0
            d += abs(float(p.get("cycle1_adx_boost", 8.0)) - float(q.get("cycle1_adx_boost", 8.0))) / 12.0
            d += abs(float(p.get("cycle1_ema_sep_atr", 0.35)) - float(q.get("cycle1_ema_sep_atr", 0.35))) / 0.4
            wpv = list(p.get("willr_by_cycle", [-60, -60, -60, -60, -60]))
            wqv = list(q.get("willr_by_cycle", [-60, -60, -60, -60, -60]))
            d += float(np.mean([abs(float(a) - float(b)) for a, b in zip(wpv, wqv)])) / 20.0
            if sorted(p.get("trade_cycles", [])) != sorted(q.get("trade_cycles", [])):
                d += 1.0
            return d

        near_flags = []
        for _, r in agg.iterrows():
            p = json.loads(str(r["params_json"]))
            near_flags.append(dist(p, wp) <= 1.20)
        agg["near_winner_neighborhood"] = near_flags
        near = agg[agg["near_winner_neighborhood"] == True].copy()  # noqa: E712
        near_count = int(len(near))
        # robust-near count
        near_rob = near[
            (to_num(near["route_pass_rate"]) >= 1.0)
            & (to_num(near["stress_pass_rate"]) >= 0.60)
            & (to_num(near["min_subperiod_delta"]) > 0.0)
            & (to_num(near["min_subperiod_cvar_improve"]) >= 0.0)
        ].copy()
        stable_neighborhood = int(len(near_rob) >= 2)
    else:
        agg["near_winner_neighborhood"] = False

    # Survivor gate
    survivors = agg[
        (to_num(agg["route_pass_rate"]) >= 1.0)
        & (to_num(agg["stress_pass_rate"]) >= 0.60)
        & (to_num(agg["min_subperiod_delta"]) > 0.0)
        & (to_num(agg["min_subperiod_cvar_improve"]) >= 0.0)
        & (to_num(agg["min_route_delta"]) > 0.0)
        & (to_num(agg["min_route_cvar"]) >= 0.0)
        & (to_num(agg["min_route_dd"]) > 0.0)
        & (to_num(agg["min_route_entries"]) >= 50)
        & (to_num(agg["min_route_entry_rate"]) >= 0.70)
        & (to_num(agg["min_stress_kept"]) >= 0.60)
    ].copy()

    # Frontier vs H file.
    frontier = agg.copy()
    frontier["phase"] = "I"
    href = pd.DataFrame(
        [
            {
                "candidate_id": "H0313_ref",
                "phase": "H",
                "OJ2": h0313_oj2,
                "delta_expectancy_vs_exec_baseline": h0313_delta,
            },
            {
                "candidate_id": "H0020_ref",
                "phase": "H",
                "OJ2": h0020_oj2,
                "delta_expectancy_vs_exec_baseline": float(h0020.iloc[0]["delta_expectancy_vs_exec_baseline"]) if not h0020.empty else float("nan"),
            },
        ]
    )
    frontier_cmp = pd.concat([frontier, href], axis=0, ignore_index=True, sort=False)
    frontier_cmp.to_csv(run_dir / "phaseI_frontier_comparison_vs_H.csv", index=False)

    # classification
    robust_count = int(len(survivors))
    beats_h0313 = int((to_num(survivors.get("beats_H0313_OJ2", 0)) == 1).any())
    best_boot = float(to_num(survivors.get("bootstrap_pass_rate", np.nan)).max()) if robust_count else float("nan")

    if robust_count >= 3 and beats_h0313 == 1 and np.isfinite(best_boot) and best_boot > 0.14 and stable_neighborhood == 1:
        cls = "I_GO_STRONG_STABLE_FRONTIER"
        reason = "multiple robust survivors beat H frontier with improved bootstrap and neighborhood stability"
    elif robust_count >= 2 and beats_h0313 == 1:
        cls = "I_GO_WEAK_FRONTIER_NEEDS_CONFIRM"
        reason = "robust survivors exist and at least one beats H0313, but bootstrap/stability remains mixed"
    else:
        # Distinguish C vs D.
        top_i = agg.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).head(1)
        top_oj2 = float(top_i.iloc[0]["OJ2"]) if not top_i.empty else float("nan")
        if (not np.isfinite(top_oj2)) or (np.isfinite(h0313_oj2) and top_oj2 <= h0313_oj2 + 1e-6):
            cls = "I_NO_GO_EXECUTION_DOMINANT_AGAIN"
            reason = "expanded GA did not materially exceed H frontier after robustness"
        else:
            cls = "I_NO_GO_FRAGILE_LUCKY_POINTS"
            reason = "in-sample gains exist but robust survivor set is too thin/fragile"

    prompt = None
    if cls in {"I_GO_STRONG_STABLE_FRONTIER", "I_GO_WEAK_FRONTIER_NEEDS_CONFIRM"}:
        prompt = (
            "ROLE\n"
            "You are in Phase J confirmation mode for execution-aware 1h+3m frontier survivors.\n\n"
            "MISSION\n"
            "Run strict OOS/route/split/stress confirmation on top Phase I non-duplicate survivors, then produce paper/shadow eligibility decision package.\n\n"
            "RULES\n"
            "1) Hard gates unchanged, frozen lock mandatory.\n"
            "2) No new GA in Phase J confirmation.\n"
            "3) Include bootstrap/resampling confidence and rollback-trigger recommendations.\n"
            "4) Stop NO_GO if Phase I gains fail confirmation."
        )
        write_text(run_dir / "ready_to_launch_phaseJ_confirmation_prompt.txt", prompt)

    # decision report
    rep = []
    rep.append("# Phase I Decision")
    rep.append("")
    rep.append(f"- Generated UTC: {utc_now()}")
    rep.append(f"- Classification: **{cls}**")
    rep.append(f"- Reason: {reason}")
    rep.append(f"- Non-duplicate valid candidates: `{len(valid)}`")
    rep.append(f"- Robust survivors: `{robust_count}`")
    rep.append(f"- Best robust bootstrap pass rate: `{best_boot}`")
    rep.append(f"- Stable neighborhood around winner: `{stable_neighborhood}`")
    rep.append("")
    rep.append("## Candidate Robustness Aggregate")
    rep.append("")
    rep.append(
        markdown_table(
            agg.sort_values(["stress_pass_rate", "route_pass_rate", "OJ2"], ascending=[False, False, False]),
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
                "beats_H0313_OJ2",
                "beats_H0020_OJ2",
                "near_winner_neighborhood",
            ],
            n=20,
        )
    )
    write_text(run_dir / "phaseI_decision_next_step.md", "\n".join(rep))
    return cls, reason, prompt, agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase I execution-aware 1h GA expansion (bounded)")
    ap.add_argument("--seed", type=int, default=20260224)
    ap.add_argument("--pop", type=int, default=192)
    ap.add_argument("--gens", type=int, default=8)
    args = ap.parse_args()

    run_dir = PROJECT_ROOT / "reports" / "execution_layer" / f"PHASEI_EXECAWARE_1H_GA_EXPANSION_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    # I1
    i1_cls, i1_meta, ctx, i1_lock, i1_extra = phase_i1_contract_and_repro(run_dir=run_dir, seed=int(args.seed))
    json_dump(run_dir / "phaseI1_contract_validation.json", i1_lock)
    json_dump(run_dir / "phaseI1_seed_reproduction_check.json", i1_extra.get("seed_repro", {}))
    json_dump(run_dir / "phaseI1_run_manifest.json", i1_extra.get("manifest", {}))
    if i1_cls != "PASS" or ctx is None:
        cls = "I_STOP_INFRA"
        reason = str(i1_meta.get("reason", "I1 failed"))
        write_text(run_dir / "phaseI_decision_next_step.md", f"# Phase I Decision\n\n- Classification: **{cls}**\n- Reason: {reason}")
        manifest = {
            "generated_utc": utc_now(),
            "run_dir": str(run_dir),
            "classification": cls,
            "mainline_status": "STOP_INFRA",
            "reason": reason,
            "duration_sec": float(time.time() - t0),
        }
        json_dump(run_dir / "phaseI_run_manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True))
        return

    # I2
    h_dir = Path(LOCKED["phase_h_dir"])
    i2_res, i2_short, i2_meta = phase_i2_ga(
        run_dir=run_dir,
        ctx=ctx,
        seed=int(args.seed) + 11,
        pop=int(args.pop),
        gens=int(args.gens),
        h_dir=h_dir,
    )
    if i2_res.empty:
        cls = "I_NO_GO_EXECUTION_DOMINANT_AGAIN"
        reason = "I2 produced empty frontier"
        write_text(run_dir / "phaseI_decision_next_step.md", f"# Phase I Decision\n\n- Classification: **{cls}**\n- Reason: {reason}")
        manifest = {
            "generated_utc": utc_now(),
            "run_dir": str(run_dir),
            "classification": cls,
            "mainline_status": "STOP_NO_GO",
            "reason": reason,
            "duration_sec": float(time.time() - t0),
            "i2_meta": i2_meta,
        }
        json_dump(run_dir / "phaseI_run_manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True))
        return

    # I3
    route_df, split_df, stress_df = phase_i3_robustness(
        run_dir=run_dir,
        ctx=ctx,
        i2_res=i2_res,
        seed=int(args.seed) + 211,
    )
    route_df.to_csv(run_dir / "phaseI3_route_checks.csv", index=False)
    split_df.to_csv(run_dir / "phaseI3_split_stability.csv", index=False)
    stress_df.to_csv(run_dir / "phaseI3_stress_matrix.csv", index=False)
    boot_df = (
        stress_df.groupby("candidate_id", dropna=False)
        .agg(bootstrap_pass_rate=("bootstrap_pass_rate", "max"))
        .reset_index()
    )
    boot_df["bootstrap_n"] = 300
    boot_df.to_csv(run_dir / "phaseI3_bootstrap_summary.csv", index=False)

    rep = []
    rep.append("# Phase I3 Top Survivors Report")
    rep.append("")
    rep.append(f"- Generated UTC: {utc_now()}")
    rep.append(f"- Route rows: `{len(route_df)}`")
    rep.append(f"- Split rows: `{len(split_df)}`")
    rep.append(f"- Stress rows: `{len(stress_df)}`")
    rep.append("")
    rep.append("## Route Checks")
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
                "min_entries_valid",
                "min_entry_rate",
                "route_pass",
            ],
            n=40,
        )
    )
    rep.append("")
    rep.append("## Split Stability")
    rep.append("")
    rep.append(
        markdown_table(
            split_df.sort_values(["split_stability_pass", "min_subperiod_delta"], ascending=[False, False]),
            ["candidate_id", "route_id", "min_subperiod_delta", "min_subperiod_cvar_improve", "split_stability_pass"],
            n=40,
        )
    )
    rep.append("")
    rep.append("## Stress Matrix")
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
            n=80,
        )
    )
    write_text(run_dir / "phaseI3_top_survivors_report.md", "\n".join(rep))

    # I4
    cls, reason, prompt, agg = phase_i4_decision(
        run_dir=run_dir,
        i2_res=i2_res,
        route_df=route_df,
        split_df=split_df,
        stress_df=stress_df,
        h_dir=h_dir,
    )
    mainline_status = "CONTINUE_READY_FOR_PHASEJ" if cls in {"I_GO_STRONG_STABLE_FRONTIER", "I_GO_WEAK_FRONTIER_NEEDS_CONFIRM"} else "STOP_NO_GO"
    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "classification": cls,
        "mainline_status": mainline_status,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "i1_meta": i1_extra.get("manifest", {}),
        "i2_meta": i2_meta,
        "i3_rows": {
            "route": int(len(route_df)),
            "split": int(len(split_df)),
            "stress": int(len(stress_df)),
            "agg_candidates": int(len(agg)),
        },
        "prompt_generated": int(prompt is not None),
        "phase_h_source_dir": str(h_dir),
    }
    json_dump(run_dir / "phaseI_run_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

