#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _parse_floats(raw: str) -> List[float]:
    out: List[float] = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(float(x))
    return out


def _parse_ints(raw: str) -> List[int]:
    out: List[int] = []
    for x in str(raw).split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(float(x)))
    return out


def _parse_bools(raw: str) -> List[int]:
    out: List[int] = []
    for x in str(raw).split(","):
        s = x.strip().lower()
        if not s:
            continue
        out.append(1 if s in {"1", "true", "yes", "y"} else 0)
    return out


@dataclass
class ExitCfg:
    tp_mult: float
    sl_mult: float
    time_stop_min: int
    break_even_enabled: int
    break_even_trigger_r: float
    break_even_offset_bps: float
    partial_take_enabled: int
    partial_take_r: float
    partial_take_pct: float


def _cfg_to_genome(cfg: ExitCfg, mode: str) -> Dict[str, Any]:
    g = {
        # Entry fixed to baseline market-at-next-open behavior.
        "entry_mode": "market",
        "limit_offset_bps": 0.0,
        "max_fill_delay_min": 0,
        "fallback_to_market": 1,
        "fallback_delay_min": 0,
        "max_taker_share": 1.0,
        "micro_vol_filter": 0,
        "vol_threshold": 6.0,
        "spread_guard_bps": 1e6,
        "killzone_filter": 0,
        "mss_displacement_gate": 0,
        "min_entry_improvement_bps_gate": 0.0,
        # Exit sweep knobs.
        "tp_mult": float(cfg.tp_mult),
        "sl_mult": float(cfg.sl_mult),
        "time_stop_min": int(cfg.time_stop_min),
        "break_even_enabled": int(cfg.break_even_enabled),
        "break_even_trigger_r": float(cfg.break_even_trigger_r),
        "break_even_offset_bps": float(cfg.break_even_offset_bps),
        "trailing_enabled": 0,
        "trail_start_r": 2.0,
        "trail_step_bps": 50.0,
        "partial_take_enabled": int(cfg.partial_take_enabled),
        "partial_take_r": float(cfg.partial_take_r),
        "partial_take_pct": float(cfg.partial_take_pct),
        "skip_if_vol_gate": 0,
        "use_signal_quality_gate": 0,
        "min_signal_quality_gate": 0.0,
        "cooldown_min": 0,
    }
    return ga_exec._repair_genome(g, mode=mode)


def _build_eval_args(user_args: argparse.Namespace) -> argparse.Namespace:
    args = ga_exec.build_arg_parser().parse_args([])
    args.symbol = str(user_args.symbol).strip().upper()
    args.symbols = str(user_args.symbols).strip().upper()
    args.rank = int(user_args.rank)
    args.signals_dir = str(user_args.signals_dir)
    args.signals_csv = str(user_args.signals_csv)
    args.signal_order = str(user_args.signal_order)
    args.max_signals = int(user_args.max_signals)
    args.train_ratio = float(user_args.train_ratio)
    args.wf_splits = int(user_args.wf_splits)
    args.mode = str(user_args.mode)
    args.force_no_skip = 1  # keep entry participation high for exit-only sweep
    args.timeframe = str(user_args.timeframe)
    args.pre_buffer_hours = float(user_args.pre_buffer_hours)
    args.exec_horizon_hours = float(user_args.exec_horizon_hours)
    args.cache_dir = str(user_args.cache_dir)
    args.max_fetch_retries = int(user_args.max_fetch_retries)
    args.retry_base_sleep = float(user_args.retry_base_sleep)
    args.retry_max_sleep = float(user_args.retry_max_sleep)
    args.fetch_pause_sec = float(user_args.fetch_pause_sec)
    args.execution_config = str(user_args.execution_config)
    args.fee_bps_maker = float(user_args.fee_bps_maker)
    args.fee_bps_taker = float(user_args.fee_bps_taker)
    args.slippage_bps_limit = float(user_args.slippage_bps_limit)
    args.slippage_bps_market = float(user_args.slippage_bps_market)
    # Relax anti-cheat realism gates for this phase: entry is intentionally fixed baseline market.
    args.hard_min_trades_overall = 0
    args.hard_min_trade_frac_overall = 0.0
    args.hard_min_trades_symbol = 0
    args.hard_min_trade_frac_symbol = 0.0
    args.hard_min_entry_rate_symbol = 0.0
    args.hard_min_entry_rate_overall = 0.0
    args.hard_max_missing_slice_rate = float(user_args.max_missing_slice_rate)
    args.hard_max_taker_share = 1.0
    args.hard_max_median_fill_delay_min = 1e9
    args.hard_max_p95_fill_delay_min = 1e9
    return args


def _config_grid(args: argparse.Namespace) -> List[ExitCfg]:
    tp = _parse_floats(args.tp_mults)
    sl = _parse_floats(args.sl_mults)
    ts = _parse_ints(args.time_stops_min)
    be = _parse_bools(args.break_even_enabled)
    be_trig = _parse_floats(args.break_even_trigger_r)
    be_off = _parse_floats(args.break_even_offset_bps)
    pt_en = _parse_bools(args.partial_take_enabled)
    pt_r = _parse_floats(args.partial_take_r)
    pt_pct = _parse_floats(args.partial_take_pct)

    grid: List[ExitCfg] = []
    for x in itertools.product(tp, sl, ts, be, be_trig, be_off, pt_en, pt_r, pt_pct):
        cfg = ExitCfg(
            tp_mult=float(x[0]),
            sl_mult=float(x[1]),
            time_stop_min=int(x[2]),
            break_even_enabled=int(x[3]),
            break_even_trigger_r=float(x[4]),
            break_even_offset_bps=float(x[5]),
            partial_take_enabled=int(x[6]),
            partial_take_r=float(x[7]),
            partial_take_pct=float(x[8]),
        )
        if cfg.partial_take_enabled == 1 and cfg.partial_take_r >= cfg.tp_mult:
            continue
        grid.append(cfg)

    max_cfg = int(args.max_configs)
    if max_cfg > 0 and len(grid) > max_cfg:
        # Deterministic uniform subsample.
        idx = np.linspace(0, len(grid) - 1, num=max_cfg, dtype=int)
        grid = [grid[i] for i in sorted(set(idx.tolist()))]
    return grid


def _stability_flags(
    exec_min: float,
    exec_med: float,
    exec_std: float,
    base_min: float,
    base_med: float,
    base_std: float,
) -> Tuple[int, int, int, int]:
    min_ok = int(np.isfinite(exec_min) and np.isfinite(base_min) and exec_min >= (base_min - 2e-4))
    med_ok = int(np.isfinite(exec_med) and np.isfinite(base_med) and exec_med >= (base_med + 5e-5))
    std_lim = max(2.5 * float(base_std), 0.0015) if np.isfinite(base_std) else 0.0015
    std_ok = int(np.isfinite(exec_std) and exec_std <= std_lim)
    return min_ok, med_ok, std_ok, int(min_ok and med_ok and std_ok)


def _overall_row(best_eval: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scope": "overall",
        "symbols": int(best_eval.get("split_rows_df", pd.DataFrame()).get("symbol", pd.Series(dtype=str)).nunique() if isinstance(best_eval.get("split_rows_df"), pd.DataFrame) else 0),
        "signals_total": int(best_eval.get("overall_signals_total", 0)),
        "baseline_mean_expectancy_net": float(best_eval.get("overall_baseline_expectancy_net", np.nan)),
        "exec_mean_expectancy_net": float(best_eval.get("overall_exec_expectancy_net", np.nan)),
        "delta_expectancy_exec_minus_baseline": float(best_eval.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
        "baseline_pnl_net_sum": float(best_eval.get("overall_baseline_pnl_net_sum", np.nan)),
        "exec_pnl_net_sum": float(best_eval.get("overall_exec_pnl_net_sum", np.nan)),
        "baseline_cvar_5": float(best_eval.get("overall_baseline_cvar_5", np.nan)),
        "exec_cvar_5": float(best_eval.get("overall_exec_cvar_5", np.nan)),
        "delta_cvar5_exec_minus_baseline": float(best_eval.get("overall_exec_cvar_5", np.nan) - best_eval.get("overall_baseline_cvar_5", np.nan)),
        "baseline_max_drawdown": float(best_eval.get("overall_baseline_max_drawdown", np.nan)),
        "exec_max_drawdown": float(best_eval.get("overall_exec_max_drawdown", np.nan)),
        "delta_maxdd_exec_minus_baseline": float(best_eval.get("overall_exec_max_drawdown", np.nan) - best_eval.get("overall_baseline_max_drawdown", np.nan)),
        "exec_entry_rate": float(best_eval.get("overall_entry_rate", np.nan)),
        "exec_taker_share": float(best_eval.get("overall_exec_taker_share", np.nan)),
        "exec_median_fill_delay_min": float(best_eval.get("overall_exec_median_fill_delay_min", np.nan)),
        "exec_p95_fill_delay_min": float(best_eval.get("overall_exec_p95_fill_delay_min", np.nan)),
        "overall_missing_slice_rate": float(best_eval.get("overall_missing_slice_rate", np.nan)),
        "min_split_expectancy_net": float(best_eval.get("min_split_expectancy_net", np.nan)),
        "median_split_expectancy_net": float(best_eval.get("median_split_expectancy_net", np.nan)),
        "std_split_expectancy_net": float(best_eval.get("std_split_expectancy_net", np.nan)),
    }


def run(args: argparse.Namespace) -> Path:
    eval_args = _build_eval_args(args)
    bundles, load_meta = ga_exec._prepare_bundles(eval_args)
    grid = _config_grid(args)
    if not grid:
        raise SystemExit("No exit configurations generated")

    out_root = _resolve_path(args.outdir) / f"EXIT_SWEEP_{_utc_tag()}"
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    base_split_stats: Optional[Tuple[float, float, float]] = None
    best_idx = -1
    best_key: Tuple[int, float, float] = (-1, -1e9, -1e9)
    best_cfg: Optional[ExitCfg] = None

    for i, cfg in enumerate(grid, start=1):
        genome = _cfg_to_genome(cfg, mode=str(eval_args.mode))
        detailed = (base_split_stats is None)
        ev = ga_exec._evaluate_genome(genome=genome, bundles=bundles, args=eval_args, detailed=detailed)

        if detailed and isinstance(ev.get("split_rows_df"), pd.DataFrame):
            sdf = ev["split_rows_df"]
            b_exp = pd.to_numeric(sdf.get("baseline_mean_expectancy_net", np.nan), errors="coerce")
            base_split_stats = (
                float(b_exp.min()) if not b_exp.empty else float("nan"),
                float(b_exp.median()) if not b_exp.empty else float("nan"),
                float(b_exp.std(ddof=0)) if not b_exp.empty else float("nan"),
            )
        if base_split_stats is None:
            base_split_stats = (float("nan"), float("nan"), float("nan"))

        d_exp = float(ev.get("overall_delta_expectancy_exec_minus_baseline", np.nan))
        d_cvar = float(ev.get("overall_exec_cvar_5", np.nan) - ev.get("overall_baseline_cvar_5", np.nan))
        d_maxdd = float(ev.get("overall_exec_max_drawdown", np.nan) - ev.get("overall_baseline_max_drawdown", np.nan))
        min_ok, med_ok, std_ok, stab_ok = _stability_flags(
            exec_min=float(ev.get("min_split_expectancy_net", np.nan)),
            exec_med=float(ev.get("median_split_expectancy_net", np.nan)),
            exec_std=float(ev.get("std_split_expectancy_net", np.nan)),
            base_min=float(base_split_stats[0]),
            base_med=float(base_split_stats[1]),
            base_std=float(base_split_stats[2]),
        )

        pass_expectancy = int(np.isfinite(d_exp) and d_exp >= float(args.expectancy_epsilon))
        pass_cvar = int(np.isfinite(d_cvar) and d_cvar >= -float(args.cvar_worse_tol))
        pass_maxdd = int(np.isfinite(d_maxdd) and d_maxdd >= -float(args.maxdd_worse_tol))
        pass_data = int(np.isfinite(float(ev.get("overall_missing_slice_rate", np.nan))) and float(ev.get("overall_missing_slice_rate", np.nan)) <= float(args.max_missing_slice_rate))
        pass_participation = int(np.isfinite(float(ev.get("overall_entry_rate", np.nan))) and float(ev.get("overall_entry_rate", np.nan)) > 0.0)
        pass_all = int(pass_expectancy and pass_cvar and pass_maxdd and stab_ok and pass_data and pass_participation)

        row = {
            "cfg_id": int(i),
            "tp_mult": float(cfg.tp_mult),
            "sl_mult": float(cfg.sl_mult),
            "time_stop_min": int(cfg.time_stop_min),
            "break_even_enabled": int(cfg.break_even_enabled),
            "break_even_trigger_r": float(cfg.break_even_trigger_r),
            "break_even_offset_bps": float(cfg.break_even_offset_bps),
            "partial_take_enabled": int(cfg.partial_take_enabled),
            "partial_take_r": float(cfg.partial_take_r),
            "partial_take_pct": float(cfg.partial_take_pct),
            "signals_total": int(ev.get("overall_signals_total", 0)),
            "entries_valid": int(ev.get("overall_entries_valid", 0)),
            "baseline_expectancy_net": float(ev.get("overall_baseline_expectancy_net", np.nan)),
            "exec_expectancy_net": float(ev.get("overall_exec_expectancy_net", np.nan)),
            "delta_expectancy_exec_minus_baseline": d_exp,
            "baseline_cvar_5": float(ev.get("overall_baseline_cvar_5", np.nan)),
            "exec_cvar_5": float(ev.get("overall_exec_cvar_5", np.nan)),
            "delta_cvar5_exec_minus_baseline": d_cvar,
            "baseline_max_drawdown": float(ev.get("overall_baseline_max_drawdown", np.nan)),
            "exec_max_drawdown": float(ev.get("overall_exec_max_drawdown", np.nan)),
            "delta_maxdd_exec_minus_baseline": d_maxdd,
            "exec_entry_rate": float(ev.get("overall_entry_rate", np.nan)),
            "exec_taker_share": float(ev.get("overall_exec_taker_share", np.nan)),
            "exec_median_fill_delay_min": float(ev.get("overall_exec_median_fill_delay_min", np.nan)),
            "exec_p95_fill_delay_min": float(ev.get("overall_exec_p95_fill_delay_min", np.nan)),
            "overall_missing_slice_rate": float(ev.get("overall_missing_slice_rate", np.nan)),
            "split_min_expectancy_exec": float(ev.get("min_split_expectancy_net", np.nan)),
            "split_median_expectancy_exec": float(ev.get("median_split_expectancy_net", np.nan)),
            "split_std_expectancy_exec": float(ev.get("std_split_expectancy_net", np.nan)),
            "split_min_expectancy_baseline": float(base_split_stats[0]),
            "split_median_expectancy_baseline": float(base_split_stats[1]),
            "split_std_expectancy_baseline": float(base_split_stats[2]),
            "pass_expectancy": int(pass_expectancy),
            "pass_cvar_not_worse": int(pass_cvar),
            "pass_maxdd_not_worse": int(pass_maxdd),
            "pass_stability_min": int(min_ok),
            "pass_stability_median": int(med_ok),
            "pass_stability_std": int(std_ok),
            "pass_stability": int(stab_ok),
            "pass_data_quality": int(pass_data),
            "pass_participation": int(pass_participation),
            "pass_all": int(pass_all),
        }
        rows.append(row)

        key = (int(pass_all), float(ev.get("overall_exec_expectancy_net", -1e9)), float(d_exp if np.isfinite(d_exp) else -1e9))
        if key > best_key:
            best_key = key
            best_idx = i - 1
            best_cfg = cfg

    res_df = pd.DataFrame(rows).sort_values(
        ["pass_all", "exec_expectancy_net", "delta_expectancy_exec_minus_baseline", "exec_max_drawdown"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    res_csv = out_root / "exit_sweep_results.csv"
    res_df.to_csv(res_csv, index=False)

    topk = res_df.head(int(args.topk)).copy()
    topk_csv = out_root / "exit_sweep_topk.csv"
    topk.to_csv(topk_csv, index=False)

    if best_cfg is None:
        raise SystemExit("No best configuration selected")
    best_genome = _cfg_to_genome(best_cfg, mode=str(eval_args.mode))
    best_eval = ga_exec._evaluate_genome(genome=best_genome, bundles=bundles, args=eval_args, detailed=True)

    split_df = best_eval["split_rows_df"] if isinstance(best_eval.get("split_rows_df"), pd.DataFrame) else pd.DataFrame()
    sym_df = best_eval["symbol_rows_df"] if isinstance(best_eval.get("symbol_rows_df"), pd.DataFrame) else pd.DataFrame()
    ov_df = pd.DataFrame([_overall_row(best_eval)])

    split_csv = out_root / "walkforward_results_by_split.csv"
    sym_csv = out_root / "risk_rollup_by_symbol.csv"
    ov_csv = out_root / "risk_rollup_overall.csv"
    split_df.to_csv(split_csv, index=False)
    sym_df.to_csv(sym_csv, index=False)
    ov_df.to_csv(ov_csv, index=False)

    best_row = res_df.iloc[0].to_dict() if not res_df.empty else {}
    decision_pass = int(best_row.get("pass_all", 0)) if best_row else 0

    dec_lines: List[str] = []
    dec_lines.append("# Exit Sweep Decision")
    dec_lines.append("")
    dec_lines.append(f"- Generated UTC: {_utc_now_iso()}")
    dec_lines.append(f"- symbols: `{','.join(load_meta.get('symbols', []))}`")
    dec_lines.append(f"- configs evaluated: {len(res_df)}")
    dec_lines.append("")
    dec_lines.append("## Best Config")
    dec_lines.append("")
    for k in [
        "tp_mult",
        "sl_mult",
        "time_stop_min",
        "break_even_enabled",
        "break_even_trigger_r",
        "break_even_offset_bps",
        "partial_take_enabled",
        "partial_take_r",
        "partial_take_pct",
    ]:
        if k in best_row:
            dec_lines.append(f"- {k}: {best_row[k]}")
    dec_lines.append("")
    dec_lines.append("## Rubric")
    dec_lines.append("")
    dec_lines.append(f"- pass_expectancy (delta >= {float(args.expectancy_epsilon):.6f}): {int(best_row.get('pass_expectancy', 0))}")
    dec_lines.append(f"- pass_maxdd_not_worse (delta >= -{float(args.maxdd_worse_tol):.3f}): {int(best_row.get('pass_maxdd_not_worse', 0))}")
    dec_lines.append(f"- pass_cvar_not_worse (delta >= -{float(args.cvar_worse_tol):.6f}): {int(best_row.get('pass_cvar_not_worse', 0))}")
    dec_lines.append(f"- pass_stability: {int(best_row.get('pass_stability', 0))}")
    dec_lines.append(f"- pass_data_quality: {int(best_row.get('pass_data_quality', 0))}")
    dec_lines.append(f"- pass_participation: {int(best_row.get('pass_participation', 0))}")
    dec_lines.append(f"- Decision: **{'PASS' if decision_pass == 1 else 'FAIL'}**")
    dec_lines.append("")
    dec_lines.append("## Outputs")
    dec_lines.append("")
    dec_lines.append(f"- results: `{res_csv}`")
    dec_lines.append(f"- topk: `{topk_csv}`")
    dec_lines.append(f"- split rollup: `{split_csv}`")
    dec_lines.append(f"- symbol rollup: `{sym_csv}`")
    dec_lines.append(f"- overall rollup: `{ov_csv}`")
    (out_root / "decision.md").write_text("\n".join(dec_lines).strip() + "\n", encoding="utf-8")

    repro_lines = [
        "# Repro",
        "",
        "```bash",
        "python3 scripts/exit_sweep.py \\",
        f"  --symbols {args.symbols if str(args.symbols).strip() else args.symbol} \\",
        f"  --mode {args.mode} \\",
        f"  --max-signals {int(args.max_signals)} \\",
        f"  --wf-splits {int(args.wf_splits)}",
        "```",
    ]
    (out_root / "repro.md").write_text("\n".join(repro_lines).strip() + "\n", encoding="utf-8")

    phase_lines = [
        "Phase: C (Exit Sweep baseline entry)",
        f"Timestamp UTC: {_utc_now_iso()}",
        f"Status: {'PASS' if decision_pass == 1 else 'FAIL'}",
        f"Configs evaluated: {len(res_df)}",
        f"Best delta expectancy: {float(best_row.get('delta_expectancy_exec_minus_baseline', np.nan)):.6f}",
        f"Best delta cvar5: {float(best_row.get('delta_cvar5_exec_minus_baseline', np.nan)):.6f}",
        f"Best delta maxdd: {float(best_row.get('delta_maxdd_exec_minus_baseline', np.nan)):.6f}",
        f"Artifacts: {res_csv.name}, {topk_csv.name}, {split_csv.name}, {sym_csv.name}, {ov_csv.name}, decision.md",
    ]
    (out_root / "phase_result.md").write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")

    meta = {
        "generated_utc": _utc_now_iso(),
        "load_meta": load_meta,
        "args": vars(args),
        "eval_args": {k: v for k, v in vars(eval_args).items() if k in {"mode", "force_no_skip", "fee_bps_maker", "fee_bps_taker", "slippage_bps_limit", "slippage_bps_market", "max_signals", "train_ratio", "wf_splits"}},
        "grid_size": int(len(grid)),
    }
    (out_root / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    snap = out_root / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    ecfg = PROJECT_ROOT / "configs" / "execution_configs.yaml"
    if ecfg.exists():
        shutil.copy2(ecfg, snap / "execution_configs.yaml")
    if str(args.signals_csv).strip():
        sig = _resolve_path(args.signals_csv)
        if sig.exists():
            shutil.copy2(sig, snap / sig.name)
    os.system(f"git -C {PROJECT_ROOT} status --short > {out_root / 'git_status.txt'}")

    print(str(out_root))
    print(str(res_csv))
    print(str(topk_csv))
    print(str(split_csv))
    print(str(sym_csv))
    print(str(ov_csv))
    print(str(out_root / "decision.md"))
    return out_root


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Walkforward TEST-only exit sweep with fixed baseline market entry.")
    ap.add_argument("--symbols", default="SOLUSDT,AVAXUSDT,NEARUSDT")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--signals-dir", default="data/signals")
    ap.add_argument("--signals-csv", default="")
    ap.add_argument("--signal-order", choices=["latest", "oldest"], default="latest")
    ap.add_argument("--max-signals", type=int, default=2000)
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--wf-splits", type=int, default=5)
    ap.add_argument("--mode", choices=["tight", "normal"], default="normal")

    ap.add_argument("--timeframe", default="3m")
    ap.add_argument("--pre-buffer-hours", type=float, default=6.0)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--cache-dir", default="data/processed/_exec_klines_cache")
    ap.add_argument("--max-fetch-retries", type=int, default=8)
    ap.add_argument("--retry-base-sleep", type=float, default=0.5)
    ap.add_argument("--retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--fetch-pause-sec", type=float, default=0.03)
    ap.add_argument("--execution-config", default="configs/execution_configs.yaml")

    ap.add_argument("--fee-bps-maker", type=float, default=2.0)
    ap.add_argument("--fee-bps-taker", type=float, default=4.0)
    ap.add_argument("--slippage-bps-limit", type=float, default=0.5)
    ap.add_argument("--slippage-bps-market", type=float, default=2.0)

    ap.add_argument("--tp-mults", default="0.7,0.8,1.0,1.2,1.5,2.0")
    ap.add_argument("--sl-mults", default="0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--time-stops-min", default="360,720,1440,2160,2880")
    ap.add_argument("--break-even-enabled", default="0,1")
    ap.add_argument("--break-even-trigger-r", default="0.5,0.75,1.0")
    ap.add_argument("--break-even-offset-bps", default="0,2,5")
    ap.add_argument("--partial-take-enabled", default="0,1")
    ap.add_argument("--partial-take-r", default="0.6,0.8,1.0")
    ap.add_argument("--partial-take-pct", default="0.25,0.5")
    ap.add_argument("--max-configs", type=int, default=0, help="Optional deterministic cap on evaluated configs (0=all).")
    ap.add_argument("--topk", type=int, default=20)

    ap.add_argument("--expectancy-epsilon", type=float, default=5e-5)
    ap.add_argument("--maxdd-worse-tol", type=float, default=0.02)
    ap.add_argument("--cvar-worse-tol", type=float, default=1e-4)
    ap.add_argument("--max-missing-slice-rate", type=float, default=0.02)

    ap.add_argument("--outdir", default="reports/execution_layer")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
