#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import sys

os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import phase_u_combined_1h3m_pilot as phase_u  # noqa: E402
from scripts import repaired_frontier_contamination_audit as audit  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402


@dataclass
class RunConfig:
    pop_size: int
    generations: int
    elite_k: int
    mutation_rate: float
    mutation_strength: float
    seed: int
    shortlist_n: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_tag() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def markdown_table(df: pd.DataFrame, cols: List[str], n: int = 20) -> str:
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
                vals.append("" if not math.isfinite(v) else f"{v:.10g}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Fresh repaired 1h-only GA/discovery rerun")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--phase-i-dir", default="")
    ap.add_argument("--params-scan-dir", default="")
    ap.add_argument("--repaired-baseline-dir", default="")
    ap.add_argument("--contamination-audit-dir", default="")
    ap.add_argument("--pop-size", type=int, default=6)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--elite-k", type=int, default=2)
    ap.add_argument("--mutation-rate", type=float, default=0.65)
    ap.add_argument("--mutation-strength", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20260304)
    ap.add_argument("--shortlist-n", type=int, default=20)
    return ap


def discover_required_dir(base: Path, pattern: str, required: List[str]) -> Path:
    cands = sorted([p for p in base.glob(pattern) if p.is_dir()], key=lambda p: p.name)
    for cand in reversed(cands):
        if all((cand / req).exists() for req in required):
            return cand.resolve()
    raise FileNotFoundError(f"missing required directory for pattern={pattern} under {base}")


def discover_artifacts(args: argparse.Namespace) -> Dict[str, Path]:
    exec_root = PROJECT_ROOT / "reports" / "execution_layer"
    params_root = PROJECT_ROOT / "reports" / "params_scan"
    out: Dict[str, Path] = {}
    out["phase_i_dir"] = (
        Path(args.phase_i_dir).resolve()
        if args.phase_i_dir
        else discover_required_dir(exec_root, "PHASEI_EXECAWARE_1H_GA_EXPANSION_*", ["phaseI2_ga_results.csv", "phaseI_run_manifest.json"])
    )
    out["params_scan_dir"] = (
        Path(args.params_scan_dir).resolve()
        if args.params_scan_dir
        else discover_required_dir(params_root, "*", ["best_by_symbol.csv", "scan_meta.json"])
    )
    out["repaired_baseline_dir"] = (
        Path(args.repaired_baseline_dir).resolve()
        if args.repaired_baseline_dir
        else discover_required_dir(exec_root, "1H_CONTRACT_REPAIR_REBASELINE_*", ["repaired_1h_reference_summary.csv"])
    )
    out["contamination_audit_dir"] = (
        Path(args.contamination_audit_dir).resolve()
        if args.contamination_audit_dir
        else discover_required_dir(exec_root, "REPAIRED_FRONTIER_CONTAMINATION_AUDIT_*", ["repaired_frontier_contamination_manifest.json"])
    )
    return out


def mutate_params(base: Dict[str, Any], cfg: RunConfig) -> Dict[str, Any]:
    return ga_long.mutate_params(copy.deepcopy(base), cfg.mutation_strength, rate=cfg.mutation_rate)


def crossover_params(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return ga_long.crossover(copy.deepcopy(a), copy.deepcopy(b))


def normalize_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    return ga_long._norm_params(copy.deepcopy(raw))  # pylint: disable=protected-access


def candidate_fitness(row: Dict[str, Any]) -> float:
    score = float(row.get("repaired_score", -1e18))
    expectancy = float(row.get("repaired_expectancy_net", float("nan")))
    trades = float(row.get("repaired_trades", 0.0))
    if int(row.get("repaired_pass", 0)) != 1:
        penalty = -1e12
        if math.isfinite(score):
            penalty += score
        if math.isfinite(expectancy):
            penalty += expectancy * 1e6
        return penalty
    extra = 0.0
    if math.isfinite(expectancy):
        extra += expectancy * 1e6
    extra += min(float(trades), 5000.0)
    return score * 1e6 + extra


def evaluate_symbol_candidate(
    *,
    symbol: str,
    params: Dict[str, Any],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    initial_equity: float,
    thresholds: audit.scan.Thresholds,
    fee_model: audit.phasec_bt.FeeModel,
    df_cache: Dict[Tuple[str, str], pd.DataFrame],
    raw_cache: Dict[str, pd.DataFrame],
    market_cache: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    repaired = audit.evaluate_repaired_1h_candidate(
        symbol=symbol,
        params_dict=params,
        period_start=period_start,
        period_end=period_end,
        initial_equity=initial_equity,
        fee_model=fee_model,
        df_cache=df_cache,
        raw_cache=raw_cache,
        market_cache=market_cache,
        thresholds=thresholds,
    )
    detail = repaired["repaired_pass_detail"]
    return {
        "repaired_score": float(repaired["repaired_1h_score"]),
        "repaired_pass": int(bool(repaired["repaired_pass"])),
        "repaired_trades": int(detail["trades"]),
        "repaired_cagr_pct": float(detail["cagr_pct"]),
        "repaired_profit_factor": float(detail["profit_factor"]),
        "repaired_max_dd_pct": float(detail["max_dd_pct"]),
        "repaired_final_equity": float(detail["final_equity"]),
        "repaired_net_profit": float(detail["net_profit"]),
        "repaired_expectancy_net": float(repaired["repaired_expectancy_net"]),
        "repaired_cvar_5": float(repaired["repaired_cvar_5"]),
        "repaired_win_rate_pct": float(detail["trades"] and ((float(detail["trades"]) - float(detail.get("losses", 0.0))) / float(detail["trades"]) * 100.0) or 0.0),
    }


def evaluate_symbol_ga(
    *,
    symbol_row: pd.Series,
    cfg: RunConfig,
    thresholds: audit.scan.Thresholds,
    fee_model: audit.phasec_bt.FeeModel,
    df_cache: Dict[Tuple[str, str], pd.DataFrame],
    raw_cache: Dict[str, pd.DataFrame],
    market_cache: Dict[str, Dict[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    symbol = str(symbol_row["symbol"]).upper()
    params_path = (PROJECT_ROOT / str(symbol_row["params_file"])).resolve()
    payload = audit.scan.load_json(params_path)
    seed_params = normalize_params(audit.scan.unwrap_params(payload))
    period_start = pd.to_datetime(symbol_row["period_start"], utc=True)
    period_end = pd.to_datetime(symbol_row["period_end"], utc=True)
    initial_equity = float(symbol_row["initial_equity"])

    random.seed(cfg.seed + abs(hash(symbol)) % 100000)

    eval_cache: Dict[str, Dict[str, Any]] = {}
    population: List[Dict[str, Any]] = [{"params": copy.deepcopy(seed_params), "origin": "seed_active_params"}]
    while len(population) < cfg.pop_size:
        population.append({"params": mutate_params(seed_params, cfg), "origin": "seed_mutation"})

    all_rows: List[Dict[str, Any]] = []
    gen_stats: List[Dict[str, Any]] = []
    next_id = 0
    best_row: Optional[Dict[str, Any]] = None

    for gen in range(cfg.generations):
        gen_rows: List[Dict[str, Any]] = []
        for pidx, indiv in enumerate(population):
            params = normalize_params(indiv["params"])
            param_hash = phase_u.param_fingerprint(params)
            if param_hash in eval_cache:
                row = copy.deepcopy(eval_cache[param_hash])
                row["generation"] = int(gen)
                row["population_index"] = int(pidx)
                row["seed_origin"] = str(indiv["origin"])
                row["cache_hit"] = 1
            else:
                cid = f"R_{symbol}_G{gen:02d}_I{next_id:04d}"
                next_id += 1
                metrics = evaluate_symbol_candidate(
                    symbol=symbol,
                    params=params,
                    period_start=period_start,
                    period_end=period_end,
                    initial_equity=initial_equity,
                    thresholds=thresholds,
                    fee_model=fee_model,
                    df_cache=df_cache,
                    raw_cache=raw_cache,
                    market_cache=market_cache,
                )
                row = {
                    "candidate_id": cid,
                    "symbol": symbol,
                    "side": "long",
                    "generation": int(gen),
                    "population_index": int(pidx),
                    "seed_origin": str(indiv["origin"]),
                    "cache_hit": 0,
                    "params_file_source": str(symbol_row["params_file"]),
                    "period_start": str(period_start),
                    "period_end": str(period_end),
                    "initial_equity": float(initial_equity),
                    "param_hash": param_hash,
                    "valid_for_ranking": int(metrics["repaired_pass"]),
                    **metrics,
                    "params_json": json.dumps(params, sort_keys=True),
                }
                row["fitness"] = candidate_fitness(row)
                eval_cache[param_hash] = copy.deepcopy(row)
                all_rows.append(copy.deepcopy(row))
            if "fitness" not in row:
                row["fitness"] = candidate_fitness(row)
            gen_rows.append(row)
        gdf = pd.DataFrame(gen_rows).sort_values(["fitness", "repaired_score", "repaired_expectancy_net"], ascending=[False, False, False]).reset_index(drop=True)
        top = gdf.iloc[0].to_dict()
        if best_row is None or float(top["fitness"]) > float(best_row["fitness"]):
            best_row = copy.deepcopy(top)
        gen_stats.append(
            {
                "symbol": symbol,
                "generation": int(gen),
                "population_size": int(len(gdf)),
                "valid_for_ranking_count": int((pd.to_numeric(gdf["valid_for_ranking"], errors="coerce").fillna(0).astype(int) == 1).sum()),
                "best_candidate_id": str(top["candidate_id"]),
                "best_score": float(top["repaired_score"]),
                "best_fitness": float(top["fitness"]),
            }
        )

        elites = gdf.head(max(1, cfg.elite_k))
        parent_pool = gdf.head(max(2, min(len(gdf), cfg.pop_size)))
        parent_params = [normalize_params(json.loads(str(x))) for x in parent_pool["params_json"].astype(str).tolist()]
        elite_params = [normalize_params(json.loads(str(x))) for x in elites["params_json"].astype(str).tolist()]
        new_pop: List[Dict[str, Any]] = [{"params": p, "origin": "elite"} for p in elite_params]
        while len(new_pop) < cfg.pop_size:
            u = random.random()
            if u < 0.45 and len(parent_params) >= 2:
                pa, pb = random.sample(parent_params, 2)
                new_pop.append({"params": crossover_params(pa, pb), "origin": "crossover"})
            elif u < 0.85:
                pa = random.choice(parent_params)
                new_pop.append({"params": mutate_params(pa, cfg), "origin": "mutation"})
            else:
                new_pop.append({"params": mutate_params(seed_params, cfg), "origin": "seed_refresh"})
        population = new_pop[: cfg.pop_size]

    result_df = pd.DataFrame(all_rows).drop_duplicates(subset=["param_hash"]).copy()
    if result_df.empty:
        return result_df, {}, gen_stats
    result_df = result_df.sort_values(["valid_for_ranking", "repaired_score", "repaired_expectancy_net"], ascending=[False, False, False]).reset_index(drop=True)
    if best_row is None:
        best_row = result_df.iloc[0].to_dict()
    return result_df, best_row, gen_stats


def classify_recommendation(universe_df: pd.DataFrame, frontier_df: pd.DataFrame) -> Tuple[str, str]:
    valid = universe_df[universe_df["valid_for_ranking"] == 1].copy()
    if valid.empty:
        return "REPAIRED_FRONTIER_TOO_WEAK_STOP_AND_RETHINK", "No symbols passed the repaired 1h discovery gates."
    strong = valid[
        (pd.to_numeric(valid["repaired_score"], errors="coerce") > 5.0)
        & (pd.to_numeric(valid["repaired_profit_factor"], errors="coerce") >= 1.2)
    ].copy()
    if len(valid) >= 6 and len(strong) >= 3:
        return "PROCEED_TO_REPAIRED_UNIVERSE_REBUILD", "The repaired 1h rerun produced a broad enough passing long set to rebuild the universe before any new 3m work."
    if len(valid) >= 3:
        return "PROCEED_TO_REPAIRED_UNIVERSE_REBUILD", "The repaired 1h rerun produced a usable but narrower long set; rebuild the universe from these repaired results first."
    if len(frontier_df) >= 1 and len(valid) >= 1:
        return "PROCEED_TO_3M_EXECUTION_EVALUATION_ON_NEW_FRONTIER", "The repaired 1h rerun is narrow but still yields a small valid frontier; 3m work should only use this new repaired frontier."
    return "REPAIRED_FRONTIER_TOO_WEAK_STOP_AND_RETHINK", "The repaired 1h rerun did not produce a stable enough passing set."


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = RunConfig(
        pop_size=int(args.pop_size),
        generations=int(args.generations),
        elite_k=int(args.elite_k),
        mutation_rate=float(args.mutation_rate),
        mutation_strength=float(args.mutation_strength),
        seed=int(args.seed),
        shortlist_n=int(args.shortlist_n),
    )

    try:
        paths = discover_artifacts(args)
    except Exception as exc:
        print(str(exc))
        print("ls -R /root/analysis/0.87/reports/execution_layer /root/analysis/0.87/reports/params_scan")
        print("/root/analysis/0.87/reports/execution_layer, /root/analysis/0.87/reports/params_scan")
        return

    run_dir = (PROJECT_ROOT / args.outdir).resolve() / f"REPAIRED_1H_GA_RERUN_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    phase_i_df = audit.canonical_phase_i_rows(pd.read_csv(paths["phase_i_dir"] / "phaseI2_ga_results.csv"))
    best_by_symbol = pd.read_csv(paths["params_scan_dir"] / "best_by_symbol.csv")
    best_by_symbol = best_by_symbol[best_by_symbol["side"].astype(str).str.lower() == "long"].copy().reset_index(drop=True)
    scan_meta = load_json(paths["params_scan_dir"] / "scan_meta.json")
    thresholds = audit.load_thresholds(scan_meta)
    repaired_signal_root = audit.rebaseline_1h._latest_multicoin_signal_root()  # pylint: disable=protected-access
    fee_model = audit.phasec_bt._load_fee_model(repaired_signal_root / "fee_model.json")  # pylint: disable=protected-access

    df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    raw_cache: Dict[str, pd.DataFrame] = {}
    market_cache: Dict[str, Dict[str, Any]] = {}

    all_result_frames: List[pd.DataFrame] = []
    per_symbol_best_rows: List[Dict[str, Any]] = []
    generation_stats: List[Dict[str, Any]] = []

    for _, row in best_by_symbol.iterrows():
        res_df, best_row, gen_stats = evaluate_symbol_ga(
            symbol_row=row,
            cfg=cfg,
            thresholds=thresholds,
            fee_model=fee_model,
            df_cache=df_cache,
            raw_cache=raw_cache,
            market_cache=market_cache,
        )
        if not res_df.empty:
            all_result_frames.append(res_df)
            if best_row:
                per_symbol_best_rows.append(best_row)
        generation_stats.extend(gen_stats)

    if not all_result_frames:
        print("missing script / artifact / code path or the exact repaired-contract incompatibility: repaired 1h GA rerun evaluated zero candidates")
        print("/root/analysis/0.87/.venv/bin/python /root/analysis/0.87/scripts/repaired_1h_ga_rerun.py")
        print(str(paths["params_scan_dir"]))
        return

    all_results = pd.concat(all_result_frames, ignore_index=True)
    all_results = all_results.sort_values(["valid_for_ranking", "repaired_score", "repaired_expectancy_net"], ascending=[False, False, False]).reset_index(drop=True)
    all_results["global_rank"] = np.arange(1, len(all_results) + 1)
    frontier = all_results[all_results["valid_for_ranking"] == 1].copy().reset_index(drop=True)
    frontier["frontier_rank"] = np.arange(1, len(frontier) + 1)
    shortlist = frontier.head(cfg.shortlist_n).copy()
    universe_candidates = pd.DataFrame(per_symbol_best_rows).copy()
    if not universe_candidates.empty:
        universe_candidates = universe_candidates.sort_values(["valid_for_ranking", "repaired_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        universe_candidates["repaired_rank"] = np.arange(1, len(universe_candidates) + 1)
        actions: List[str] = []
        legacy_rank_by_symbol = {}
        legacy_pass_by_symbol = {}
        legacy_score_by_symbol = {}
        legacy_params_by_symbol = {}
        for rank, (_, row) in enumerate(
            best_by_symbol.sort_values(
                by=[
                    best_by_symbol["pass"].map(lambda x: int(str(x).strip().lower() in {"1", "true", "yes"})).name if False else "symbol"
                ]
            ).iterrows(),
            start=1,
        ):
            pass
        best_sorted = best_by_symbol.assign(
            _pass=best_by_symbol["pass"].map(lambda x: int(str(x).strip().lower() in {"1", "true", "yes"})),
            _score=pd.to_numeric(best_by_symbol["score"], errors="coerce"),
        ).sort_values(["_pass", "_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
        best_sorted["legacy_rank"] = np.arange(1, len(best_sorted) + 1)
        for _, r in best_sorted.iterrows():
            sym = str(r["symbol"]).upper()
            legacy_rank_by_symbol[sym] = int(r["legacy_rank"])
            legacy_pass_by_symbol[sym] = int(str(r["pass"]).strip().lower() in {"1", "true", "yes"})
            legacy_score_by_symbol[sym] = float(r["score"])
            legacy_params_by_symbol[sym] = str(r["params_file"])
        universe_candidates["legacy_pass"] = universe_candidates["symbol"].map(legacy_pass_by_symbol).fillna(0).astype(int)
        universe_candidates["legacy_score"] = pd.to_numeric(universe_candidates["symbol"].map(legacy_score_by_symbol), errors="coerce")
        universe_candidates["legacy_rank"] = pd.to_numeric(universe_candidates["symbol"].map(legacy_rank_by_symbol), errors="coerce").fillna(-1).astype(int)
        universe_candidates["legacy_params_file"] = universe_candidates["symbol"].map(legacy_params_by_symbol).fillna("")
        universe_candidates["score_delta_vs_legacy"] = pd.to_numeric(universe_candidates["repaired_score"], errors="coerce") - pd.to_numeric(universe_candidates["legacy_score"], errors="coerce")
        universe_candidates["rank_delta_vs_legacy"] = pd.to_numeric(universe_candidates["repaired_rank"], errors="coerce") - pd.to_numeric(universe_candidates["legacy_rank"], errors="coerce")
        for _, r in universe_candidates.iterrows():
            lp = int(r["legacy_pass"])
            rp = int(r["valid_for_ranking"])
            if lp == 1 and rp == 1:
                actions.append("STAY_PASS")
            elif lp == 1 and rp == 0:
                actions.append("DROP_FROM_PASS")
            elif lp == 0 and rp == 1:
                actions.append("NEW_PASS")
            else:
                actions.append("STAY_FAIL")
        universe_candidates["membership_action"] = actions

    legacy_top = phase_i_df.head(20).copy()
    diff_rows: List[Dict[str, Any]] = []
    new_global_hashes = set(all_results["param_hash"].astype(str))
    new_sol_df = all_results[all_results["symbol"] == "SOLUSDT"].copy().sort_values(["valid_for_ranking", "repaired_score"], ascending=[False, False]).reset_index(drop=True)
    new_sol_hashes = set(new_sol_df["param_hash"].astype(str))
    for k in [5, 10, 20]:
        old_top = phase_i_df.head(min(k, len(phase_i_df)))
        old_hashes = set(old_top["param_hash"].astype(str))
        old_symbols = sorted(set(["SOLUSDT"]))
        new_global_top = frontier.head(min(k, len(frontier)))
        new_global_symbols = sorted(set(new_global_top["symbol"].astype(str))) if not new_global_top.empty else []
        diff_rows.append(
            {
                "diff_type": "frontier_topk_summary",
                "scope": f"top{k}",
                "legacy_topk_size": int(len(old_top)),
                "legacy_symbols": ",".join(old_symbols),
                "new_global_topk_size": int(len(new_global_top)),
                "new_global_symbols": ",".join(new_global_symbols),
                "param_hash_overlap_with_new_global": int(len(old_hashes & new_global_hashes)),
                "param_hash_overlap_with_new_sol": int(len(old_hashes & new_sol_hashes)),
                "legacy_mean_oj2": float(pd.to_numeric(old_top["OJ2"], errors="coerce").mean()) if not old_top.empty else float("nan"),
                "new_global_mean_score": float(pd.to_numeric(new_global_top["repaired_score"], errors="coerce").mean()) if not new_global_top.empty else float("nan"),
            }
        )
    if not universe_candidates.empty:
        for _, r in universe_candidates.iterrows():
            diff_rows.append(
                {
                    "diff_type": "symbol_universe_delta",
                    "scope": "per_symbol",
                    "symbol": str(r["symbol"]),
                    "legacy_pass": int(r["legacy_pass"]),
                    "repaired_pass": int(r["valid_for_ranking"]),
                    "legacy_score": float(r["legacy_score"]),
                    "repaired_score": float(r["repaired_score"]),
                    "score_delta": float(r["score_delta_vs_legacy"]),
                    "legacy_rank": int(r["legacy_rank"]),
                    "repaired_rank": int(r["repaired_rank"]),
                    "rank_delta": int(r["rank_delta_vs_legacy"]),
                    "membership_action": str(r["membership_action"]),
                    "legacy_params_file": str(r["legacy_params_file"]),
                    "repaired_candidate_id": str(r["candidate_id"]),
                }
            )
    diff_df = pd.DataFrame(diff_rows)

    frontier_summary = {
        "total_candidates_evaluated": int(len(all_results)),
        "valid_frontier_count": int(len(frontier)),
        "unique_symbols_in_frontier": int(frontier["symbol"].nunique()) if not frontier.empty else 0,
        "top_frontier_symbol_mix": frontier.head(10)["symbol"].value_counts().to_dict() if not frontier.empty else {},
        "top_score": float(frontier.iloc[0]["repaired_score"]) if not frontier.empty else float("nan"),
        "top_candidate_id": str(frontier.iloc[0]["candidate_id"]) if not frontier.empty else "",
    }
    recommendation, rec_reason = classify_recommendation(universe_candidates, frontier)

    all_results.to_csv(run_dir / "repaired_1h_ga_results.csv", index=False)
    frontier.to_csv(run_dir / "repaired_1h_frontier.csv", index=False)
    shortlist.to_csv(run_dir / "repaired_1h_shortlist.csv", index=False)
    diff_df.to_csv(run_dir / "repaired_1h_vs_legacy_frontier_diff.csv", index=False)
    if not universe_candidates.empty:
        universe_candidates.to_csv(run_dir / "repaired_1h_universe_candidates.csv", index=False)
    pd.DataFrame(generation_stats).to_csv(run_dir / "repaired_1h_generation_stats.csv", index=False)

    rerun_manifest = {
        "generated_utc": utc_now().isoformat(),
        "run_dir": str(run_dir),
        "discovered_paths": {k: str(v) for k, v in paths.items()},
        "legacy_code_paths": [
            "scripts/phase_h_execaware_1h_ga_pilot.py",
            "scripts/phase_i_execaware_1h_ga_expansion.py",
            "scripts/phase_u_combined_1h3m_pilot.py",
            "src/bot087/optim/ga.py",
        ],
        "repaired_code_paths": [
            "scripts/repaired_1h_ga_rerun.py",
            "scripts/repaired_frontier_contamination_audit.py",
            "scripts/backtest_exec_phasec_sol.py",
        ],
        "objective_change": {
            "legacy_invalid_objective": "OJ2 execution-aware 3m objective from Phase I/H",
            "replacement_objective": "repaired_1h_score = (cagr_pct * profit_factor) / (1 + max_dd_pct)",
            "replacement_pass_logic": "scan_params_all_coins thresholds from scan_meta.json",
            "why": "Old H/I objective depends on execution-aware 3m metrics and cannot be used in a pure 1h-only rerun.",
        },
        "config": {
            "pop_size": cfg.pop_size,
            "generations": cfg.generations,
            "elite_k": cfg.elite_k,
            "mutation_rate": cfg.mutation_rate,
            "mutation_strength": cfg.mutation_strength,
            "seed": cfg.seed,
            "shortlist_n": cfg.shortlist_n,
            "thresholds": scan_meta.get("thresholds", {}),
            "fee_model_source": str(repaired_signal_root / "fee_model.json"),
        },
        "frontier_summary": frontier_summary,
        "recommendation": recommendation,
        "recommendation_reason": rec_reason,
    }
    json_dump(run_dir / "rerun_manifest.json", rerun_manifest)

    report_lines: List[str] = []
    report_lines.append("# Repaired 1H GA Rerun Report")
    report_lines.append("")
    report_lines.append(f"- Generated UTC: `{utc_now().isoformat()}`")
    report_lines.append("- Scope: fresh 1h-only discovery rerun under the repaired chronology-valid contract.")
    report_lines.append("- 3m execution layer: explicitly excluded.")
    report_lines.append("")
    report_lines.append("## Discovered Legacy GA Code Paths")
    report_lines.append("- `scripts/phase_h_execaware_1h_ga_pilot.py`: legacy bounded execution-aware 1h GA pilot and candidate evaluator source.")
    report_lines.append("- `scripts/phase_i_execaware_1h_ga_expansion.py`: legacy Phase I expansion runner, population mix, and OJ2/OJ4 frontier ranking.")
    report_lines.append("- `scripts/phase_u_combined_1h3m_pilot.py`: legacy helper layer for param fingerprints / signal activity mapping used by H/I.")
    report_lines.append("- `src/bot087/optim/ga.py`: reusable 1h parameter mutation / crossover / normalization machinery.")
    report_lines.append("")
    report_lines.append("## Exact Repaired-Contract Changes")
    report_lines.append("- Did not reuse the contaminated Phase I shortlist as a seed shortlist or ranking base.")
    report_lines.append("- Did not reuse legacy OJ2/OJ4 because those depend on execution-aware 3m metrics from the broken discovery stack.")
    report_lines.append("- Reused only the parameter search mechanics from `src/bot087/optim/ga.py` (`_norm_params`, `mutate_params`, `crossover`).")
    report_lines.append("- Replaced candidate evaluation with the repaired 1h-only evaluator from `scripts/repaired_frontier_contamination_audit.py`, which calls `scripts/backtest_exec_phasec_sol.py` logic with deferred next-bar exit semantics.")
    report_lines.append("- Replacement objective: `repaired_1h_score = (cagr_pct * profit_factor) / (1 + max_dd_pct)` using the exact `scan_params_all_coins.py` formula and pass thresholds from `scan_meta.json`.")
    report_lines.append("")
    report_lines.append("## Repaired Frontier Summary")
    report_lines.append(f"- Total evaluated candidates: `{len(all_results)}`")
    report_lines.append(f"- Valid frontier count: `{len(frontier)}`")
    report_lines.append(f"- Unique symbols in valid frontier: `{frontier['symbol'].nunique() if not frontier.empty else 0}`")
    if not frontier.empty:
        report_lines.append(f"- Top candidate: `{frontier.iloc[0]['candidate_id']}` ({frontier.iloc[0]['symbol']}) score=`{float(frontier.iloc[0]['repaired_score']):.6f}`")
    report_lines.append("")
    report_lines.append("### Top Repaired Frontier")
    report_lines.append("")
    report_lines.append(
        markdown_table(
            frontier,
            [
                "frontier_rank",
                "candidate_id",
                "symbol",
                "repaired_score",
                "repaired_cagr_pct",
                "repaired_profit_factor",
                "repaired_max_dd_pct",
                "repaired_trades",
                "repaired_expectancy_net",
            ],
            n=min(cfg.shortlist_n, 20),
        )
    )
    report_lines.append("")
    if not universe_candidates.empty:
        report_lines.append("## Repaired Universe Candidate Output")
        report_lines.append("")
        report_lines.append(
            markdown_table(
                universe_candidates,
                [
                    "repaired_rank",
                    "symbol",
                    "candidate_id",
                    "valid_for_ranking",
                    "repaired_score",
                    "legacy_score",
                    "score_delta_vs_legacy",
                    "membership_action",
                ],
                n=len(universe_candidates),
            )
        )
        report_lines.append("")
    report_lines.append("## Legacy Comparison")
    report_lines.append("- Legacy Phase I remained SOL-seeded; this rerun is multicoin across the canonical long scan symbol set.")
    topk_summary = diff_df[diff_df["diff_type"] == "frontier_topk_summary"].copy()
    if not topk_summary.empty:
        report_lines.append("")
        report_lines.append(markdown_table(topk_summary, list(topk_summary.columns), n=len(topk_summary)))
        report_lines.append("")
    report_lines.append("## Final Recommendation")
    report_lines.append(f"- Recommendation: `{recommendation}`")
    report_lines.append(f"- Reason: {rec_reason}")
    (run_dir / "repaired_1h_ga_rerun_report.md").write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")

    print(json.dumps({"run_dir": str(run_dir), "recommendation": recommendation, "valid_frontier_count": int(len(frontier)), "universe_pass_count": int((universe_candidates['valid_for_ranking'] == 1).sum()) if not universe_candidates.empty else 0}, sort_keys=True))


if __name__ == "__main__":
    main()
