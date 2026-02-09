from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.bot087.execution.execution_eval import ExecutionEvalConfig, evaluate_with_sec_data


@dataclass(frozen=True)
class ExecutionGAConfig:
    pop_size: int = 32
    generations: int = 60
    elite_k: int = 6
    mutation_rate: float = 0.4
    seed: int = 42
    early_stop_patience: int = 20

    min_window_sec: int = 5
    max_window_sec: int = 120
    max_delay_sec: int = 20


def _rand_ind(cfg: ExecutionGAConfig) -> Dict[str, int]:
    return {
        "window_sec": random.randint(cfg.min_window_sec, cfg.max_window_sec),
        "entry_delay_sec": random.randint(0, cfg.max_delay_sec),
        "exit_delay_sec": random.randint(0, cfg.max_delay_sec),
    }


def _mutate(ind: Dict[str, int], cfg: ExecutionGAConfig) -> Dict[str, int]:
    out = dict(ind)
    if random.random() < cfg.mutation_rate:
        out["window_sec"] = int(max(cfg.min_window_sec, min(cfg.max_window_sec, out["window_sec"] + random.randint(-8, 8))))
    if random.random() < cfg.mutation_rate:
        out["entry_delay_sec"] = int(max(0, min(cfg.max_delay_sec, out["entry_delay_sec"] + random.randint(-3, 3))))
    if random.random() < cfg.mutation_rate:
        out["exit_delay_sec"] = int(max(0, min(cfg.max_delay_sec, out["exit_delay_sec"] + random.randint(-3, 3))))
    return out


def _cross(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {
        "window_sec": a["window_sec"] if random.random() < 0.5 else b["window_sec"],
        "entry_delay_sec": a["entry_delay_sec"] if random.random() < 0.5 else b["entry_delay_sec"],
        "exit_delay_sec": a["exit_delay_sec"] if random.random() < 0.5 else b["exit_delay_sec"],
    }


def _fitness(summary: Dict[str, Any]) -> float:
    # Prefer positive net delta with broad per-trade improvement.
    delta = float(summary.get("delta_net", 0.0))
    improved_ratio = float(summary.get("improved_ratio", 0.0))
    return float(delta + 100.0 * improved_ratio)


def run_execution_ga(
    *,
    symbol: str,
    trades: pd.DataFrame,
    sec_df: pd.DataFrame,
    base_eval_cfg: ExecutionEvalConfig,
    ga_cfg: ExecutionGAConfig,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    if ga_cfg.pop_size < 2:
        raise ValueError("pop_size must be >= 2")

    random.seed(ga_cfg.seed)
    population = [_rand_ind(ga_cfg) for _ in range(ga_cfg.pop_size)]

    best_overall: Dict[str, Any] | None = None
    stale = 0
    history: List[Dict[str, Any]] = []

    for gen in range(ga_cfg.generations):
        scored: List[Dict[str, Any]] = []
        for ind in population:
            cfg = ExecutionEvalConfig(
                mode=base_eval_cfg.mode,
                window_sec=int(ind["window_sec"]),
                entry_delay_sec=int(ind["entry_delay_sec"]),
                exit_delay_sec=int(ind["exit_delay_sec"]),
                cap_gb=base_eval_cfg.cap_gb,
                fee_bps=base_eval_cfg.fee_bps,
                slippage_bps=base_eval_cfg.slippage_bps,
                pause_sec=base_eval_cfg.pause_sec,
                cache_root=base_eval_cfg.cache_root,
            )
            eval_out = evaluate_with_sec_data(symbol=symbol, trades=trades, sec_df=sec_df, cfg=cfg)
            score = _fitness(eval_out["summary"])
            scored.append({"ind": ind, "score": score, "eval": eval_out})

        scored.sort(key=lambda x: x["score"], reverse=True)
        best = scored[0]
        history.append(
            {
                "gen": gen,
                "score": float(best["score"]),
                "window_sec": int(best["ind"]["window_sec"]),
                "entry_delay_sec": int(best["ind"]["entry_delay_sec"]),
                "exit_delay_sec": int(best["ind"]["exit_delay_sec"]),
                "delta_net": float(best["eval"]["summary"].get("delta_net", 0.0)),
                "improved_ratio": float(best["eval"]["summary"].get("improved_ratio", 0.0)),
            }
        )

        if best_overall is None or best["score"] > best_overall["score"]:
            best_overall = best
            stale = 0
        else:
            stale += 1

        if ga_cfg.early_stop_patience > 0 and stale >= ga_cfg.early_stop_patience:
            break

        elites = [x["ind"] for x in scored[: max(2, ga_cfg.elite_k)]]
        new_pop = list(elites)
        while len(new_pop) < ga_cfg.pop_size:
            p1, p2 = random.sample(elites, 2)
            child = _cross(p1, p2)
            child = _mutate(child, ga_cfg)
            new_pop.append(child)
        population = new_pop

    if best_overall is None:
        raise RuntimeError("Execution GA produced no result")

    report = {
        "symbol": symbol,
        "best": best_overall,
        "history": history,
        "ga_cfg": ga_cfg.__dict__,
    }
    return dict(best_overall["ind"]), report
