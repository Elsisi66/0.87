#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import exit_sweep  # noqa: E402
from scripts.phase_d_regime_features import build_regime_labels  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


REQ_PHASEC_CONTROL = {
    "tp_mult": 1.0,
    "sl_mult": 0.75,
    "time_stop_min": 720,
    "break_even_enabled": 0,
    "break_even_trigger_r": 0.5,
    "break_even_offset_bps": 0.0,
    "partial_take_enabled": 0,
    "partial_take_r": 0.8,
    "partial_take_pct": 0.25,
    "cfg_hash": "a285b86c4c22a26976d4a762",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _json_dump(path: Path, obj: Any) -> None:
    def _default(x: Any) -> Any:
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
        return str(x)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def _write_git_status(path: Path) -> None:
    try:
        p = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "status", "--short"],
            capture_output=True,
            text=True,
            check=False,
        )
        path.write_text(str(p.stdout), encoding="utf-8")
    except Exception as e:
        path.write_text(f"git status failed: {e}\n", encoding="utf-8")


def _cfg_hash(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _parse_fee_model(phase_a_fee_json: Path) -> Dict[str, float]:
    obj = json.loads(phase_a_fee_json.read_text(encoding="utf-8"))
    for key in ["fee_bps_maker", "fee_bps_taker", "slippage_bps_limit", "slippage_bps_market"]:
        if key in obj:
            return {
                "fee_bps_maker": float(obj["fee_bps_maker"]),
                "fee_bps_taker": float(obj["fee_bps_taker"]),
                "slippage_bps_limit": float(obj["slippage_bps_limit"]),
                "slippage_bps_market": float(obj["slippage_bps_market"]),
            }
    for parent in ["tight_pipeline_fee_model", "ga_pipeline_fee_model"]:
        cand = obj.get(parent)
        if isinstance(cand, dict) and all(k in cand for k in ["fee_bps_maker", "fee_bps_taker", "slippage_bps_limit", "slippage_bps_market"]):
            return {
                "fee_bps_maker": float(cand["fee_bps_maker"]),
                "fee_bps_taker": float(cand["fee_bps_taker"]),
                "slippage_bps_limit": float(cand["slippage_bps_limit"]),
                "slippage_bps_market": float(cand["slippage_bps_market"]),
            }
    raise RuntimeError(f"Could not parse fee model from {phase_a_fee_json}")


def _load_phasec_contract(phase_c_dir: Path) -> Dict[str, Any]:
    req = [
        phase_c_dir / "run_manifest.json",
        phase_c_dir / "signal_subset.csv",
        phase_c_dir / "signal_subset_hash.txt",
        phase_c_dir / "wf_split_definition.json",
        phase_c_dir / "refine" / "exit_sweep_results_refine.csv",
        phase_c_dir / "metrics_definition.md",
        phase_c_dir / "fee_model.json",
    ]
    missing = [str(p) for p in req if not p.exists()]
    if missing:
        raise FileNotFoundError("Phase C dir incomplete:\n" + "\n".join(missing))

    man = json.loads((phase_c_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if str(man.get("symbol", "")).upper() != "SOLUSDT":
        raise RuntimeError("Phase C manifest symbol is not SOLUSDT")

    phase_a = man.get("phase_a_contract", {})
    m_path = Path(str(phase_a.get("metrics_definition_path", ""))).resolve()
    f_path = Path(str(phase_a.get("fee_model_path", ""))).resolve()
    if not m_path.exists() or not f_path.exists():
        raise FileNotFoundError("Phase A contract paths from Phase C manifest are missing")
    m_hash = _sha256_file(m_path)
    f_hash = _sha256_file(f_path)
    if str(phase_a.get("metrics_definition_sha256", "")) and m_hash != str(phase_a["metrics_definition_sha256"]):
        raise RuntimeError("Phase A metrics_definition hash mismatch")
    if str(phase_a.get("fee_model_sha256", "")) and f_hash != str(phase_a["fee_model_sha256"]):
        raise RuntimeError("Phase A fee_model hash mismatch")

    subset_csv = phase_c_dir / "signal_subset.csv"
    subset_hash = hashlib.sha256(
        "\n".join(
            [
                f"{r['signal_id']}|{pd.to_datetime(r['signal_time'], utc=True, errors='coerce').isoformat()}"
                for _, r in pd.read_csv(subset_csv).sort_values("signal_time").reset_index(drop=True).iterrows()
            ]
        ).encode("utf-8")
    ).hexdigest()
    subset_expected = (phase_c_dir / "signal_subset_hash.txt").read_text(encoding="utf-8").strip()
    if subset_hash != subset_expected:
        raise RuntimeError("Phase C signal subset hash mismatch")
    if str(man.get("signal_subset_hash", "")) and subset_hash != str(man["signal_subset_hash"]):
        raise RuntimeError("Phase C manifest signal_subset_hash mismatch")

    split_json = phase_c_dir / "wf_split_definition.json"
    split_hash = _sha256_file(split_json)
    if str(man.get("split_definition_sha256", "")) and split_hash != str(man["split_definition_sha256"]):
        raise RuntimeError("Phase C split_definition hash mismatch")

    ctrl = man.get("final_selected_cfg", {})
    ctrl_hash = str(man.get("final_selected_cfg_hash", "")).strip()
    for k, v in REQ_PHASEC_CONTROL.items():
        if k == "cfg_hash":
            continue
        if float(ctrl.get(k, np.nan)) != float(v):
            raise RuntimeError(f"Phase C control mismatch on {k}: got {ctrl.get(k)} expected {v}")
    if ctrl_hash != REQ_PHASEC_CONTROL["cfg_hash"]:
        raise RuntimeError(f"Phase C control hash mismatch: got {ctrl_hash} expected {REQ_PHASEC_CONTROL['cfg_hash']}")

    return {
        "manifest": man,
        "phase_a_metrics_path": m_path,
        "phase_a_fee_path": f_path,
        "phase_a_metrics_hash": m_hash,
        "phase_a_fee_hash": f_hash,
        "subset_hash": subset_hash,
        "split_hash": split_hash,
        "control_cfg": ctrl,
        "control_cfg_hash": ctrl_hash,
    }


def _load_split_definition(path: Path) -> List[Dict[str, int]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    splits = obj.get("splits", [])
    out: List[Dict[str, int]] = []
    for s in splits:
        out.append(
            {
                "split_id": int(s["split_id"]),
                "train_start": int(s["train_start"]),
                "train_end": int(s["train_end"]),
                "test_start": int(s["test_start"]),
                "test_end": int(s["test_end"]),
            }
        )
    out = sorted(out, key=lambda x: x["split_id"])
    return out


def _build_eval_args(
    *,
    symbol: str,
    signal_subset_csv: Path,
    mode: str,
    split_count: int,
    fee: Dict[str, float],
    args: argparse.Namespace,
) -> argparse.Namespace:
    ev = ga_exec.build_arg_parser().parse_args([])
    ev.symbol = str(symbol).upper()
    ev.symbols = str(symbol).upper()
    ev.signals_csv = str(signal_subset_csv)
    ev.signals_dir = str(signal_subset_csv.parent)
    ev.signal_order = "latest"
    ev.max_signals = 10**9
    ev.walkforward = True
    ev.train_ratio = float(args.train_ratio)
    ev.wf_splits = int(split_count)
    ev.mode = str(mode)
    ev.force_no_skip = 1
    ev.timeframe = str(args.timeframe)
    ev.pre_buffer_hours = float(args.pre_buffer_hours)
    ev.exec_horizon_hours = float(args.exec_horizon_hours)
    ev.cache_dir = str(args.cache_dir)
    ev.max_fetch_retries = int(args.max_fetch_retries)
    ev.retry_base_sleep = float(args.retry_base_sleep)
    ev.retry_max_sleep = float(args.retry_max_sleep)
    ev.fetch_pause_sec = float(args.fetch_pause_sec)
    ev.execution_config = str(args.execution_config)
    ev.fee_bps_maker = float(fee["fee_bps_maker"])
    ev.fee_bps_taker = float(fee["fee_bps_taker"])
    ev.slippage_bps_limit = float(fee["slippage_bps_limit"])
    ev.slippage_bps_market = float(fee["slippage_bps_market"])
    ev.workers = int(args.workers)
    # No anti-cheat entry gates in this phase; exit-only comparison.
    ev.hard_min_trades_overall = 0
    ev.hard_min_trade_frac_overall = 0.0
    ev.hard_min_trades_symbol = 0
    ev.hard_min_trade_frac_symbol = 0.0
    ev.hard_min_entry_rate_symbol = 0.0
    ev.hard_min_entry_rate_overall = 0.0
    ev.hard_max_taker_share = 1.0
    ev.hard_max_median_fill_delay_min = 1e9
    ev.hard_max_p95_fill_delay_min = 1e9
    ev.hard_max_missing_slice_rate = float(args.max_missing_slice_rate)
    return ev


def _control_cfg_from_manifest(ctrl: Dict[str, Any]) -> exit_sweep.ExitCfg:
    return exit_sweep.ExitCfg(
        tp_mult=float(ctrl["tp_mult"]),
        sl_mult=float(ctrl["sl_mult"]),
        time_stop_min=int(ctrl["time_stop_min"]),
        break_even_enabled=int(ctrl["break_even_enabled"]),
        break_even_trigger_r=float(ctrl["break_even_trigger_r"]),
        break_even_offset_bps=float(ctrl["break_even_offset_bps"]),
        partial_take_enabled=int(ctrl["partial_take_enabled"]),
        partial_take_r=float(ctrl["partial_take_r"]),
        partial_take_pct=float(ctrl["partial_take_pct"]),
    )


def _candidate_pool(
    *,
    refine_csv: Path,
    control_cfg: exit_sweep.ExitCfg,
    control_hash: str,
    topn: int,
    max_pool: int,
) -> pd.DataFrame:
    df = pd.read_csv(refine_csv)
    if df.empty:
        raise RuntimeError(f"Empty refine CSV: {refine_csv}")
    # Gate-aware ranking from Phase C.
    order_cols = [
        "pass_all",
        "pass_stability",
        "delta_expectancy_best_exit_minus_baseline_exit",
        "delta_maxdd_best_exit_minus_baseline_exit",
        "delta_cvar5_best_exit_minus_baseline_exit",
    ]
    for c in order_cols:
        if c not in df.columns:
            df[c] = 0
    x = df.sort_values(order_cols, ascending=[False, False, False, False, False], kind="mergesort").copy()
    keep = x.head(int(topn)).copy()

    # Local neighbors around control on tp/sl/time, keep other knobs fixed as control.
    tp_vals = sorted(pd.to_numeric(df["tp_mult"], errors="coerce").dropna().unique().tolist())
    sl_vals = sorted(pd.to_numeric(df["sl_mult"], errors="coerce").dropna().unique().tolist())
    ts_vals = sorted(pd.to_numeric(df["time_stop_min"], errors="coerce").dropna().astype(int).unique().tolist())

    def _neighbors(v: float, vals: List[float]) -> List[float]:
        if not vals:
            return [float(v)]
        d = sorted(vals)
        try:
            i = d.index(float(v))
        except ValueError:
            i = int(np.argmin([abs(float(z) - float(v)) for z in d]))
        idx = [max(0, i - 1), i, min(len(d) - 1, i + 1)]
        return sorted({float(d[j]) for j in idx})

    tp_n = _neighbors(float(control_cfg.tp_mult), tp_vals)
    sl_n = _neighbors(float(control_cfg.sl_mult), sl_vals)
    ts_n = [int(x) for x in _neighbors(float(control_cfg.time_stop_min), [float(v) for v in ts_vals])]
    rows_extra: List[Dict[str, Any]] = []
    for tp in tp_n:
        for sl in sl_n:
            for ts in ts_n:
                rows_extra.append(
                    {
                        "tp_mult": float(tp),
                        "sl_mult": float(sl),
                        "time_stop_min": int(ts),
                        "break_even_enabled": int(control_cfg.break_even_enabled),
                        "break_even_trigger_r": float(control_cfg.break_even_trigger_r),
                        "break_even_offset_bps": float(control_cfg.break_even_offset_bps),
                        "partial_take_enabled": int(control_cfg.partial_take_enabled),
                        "partial_take_r": float(control_cfg.partial_take_r),
                        "partial_take_pct": float(control_cfg.partial_take_pct),
                        "source": "neighbor_control",
                        "pass_all": 0,
                        "pass_stability": 0,
                        "delta_expectancy_best_exit_minus_baseline_exit": np.nan,
                        "delta_maxdd_best_exit_minus_baseline_exit": np.nan,
                        "delta_cvar5_best_exit_minus_baseline_exit": np.nan,
                    }
                )
    extra = pd.DataFrame(rows_extra)
    keep["source"] = "phasec_refine_top"
    pool = pd.concat([keep, extra], ignore_index=True, sort=False)

    # Ensure control exists exactly.
    control_row = pd.DataFrame(
        [
            {
                "tp_mult": float(control_cfg.tp_mult),
                "sl_mult": float(control_cfg.sl_mult),
                "time_stop_min": int(control_cfg.time_stop_min),
                "break_even_enabled": int(control_cfg.break_even_enabled),
                "break_even_trigger_r": float(control_cfg.break_even_trigger_r),
                "break_even_offset_bps": float(control_cfg.break_even_offset_bps),
                "partial_take_enabled": int(control_cfg.partial_take_enabled),
                "partial_take_r": float(control_cfg.partial_take_r),
                "partial_take_pct": float(control_cfg.partial_take_pct),
                "source": "phasec_control",
            }
        ]
    )
    pool = pd.concat([pool, control_row], ignore_index=True, sort=False)

    key_cols = [
        "tp_mult",
        "sl_mult",
        "time_stop_min",
        "break_even_enabled",
        "break_even_trigger_r",
        "break_even_offset_bps",
        "partial_take_enabled",
        "partial_take_r",
        "partial_take_pct",
    ]
    pool = pool.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
    if int(max_pool) > 0 and len(pool) > int(max_pool):
        # Deterministic cap: keep control, then gate-ranked, then lexicographic.
        pool["is_control"] = (
            (pd.to_numeric(pool["tp_mult"], errors="coerce") == float(control_cfg.tp_mult))
            & (pd.to_numeric(pool["sl_mult"], errors="coerce") == float(control_cfg.sl_mult))
            & (pd.to_numeric(pool["time_stop_min"], errors="coerce") == int(control_cfg.time_stop_min))
            & (pd.to_numeric(pool["break_even_enabled"], errors="coerce") == int(control_cfg.break_even_enabled))
            & (pd.to_numeric(pool["break_even_trigger_r"], errors="coerce") == float(control_cfg.break_even_trigger_r))
            & (pd.to_numeric(pool["break_even_offset_bps"], errors="coerce") == float(control_cfg.break_even_offset_bps))
            & (pd.to_numeric(pool["partial_take_enabled"], errors="coerce") == int(control_cfg.partial_take_enabled))
            & (pd.to_numeric(pool["partial_take_r"], errors="coerce") == float(control_cfg.partial_take_r))
            & (pd.to_numeric(pool["partial_take_pct"], errors="coerce") == float(control_cfg.partial_take_pct))
        ).astype(int)
        pool = pool.sort_values(
            [
                "is_control",
                "pass_all",
                "pass_stability",
                "delta_expectancy_best_exit_minus_baseline_exit",
                "delta_maxdd_best_exit_minus_baseline_exit",
                "delta_cvar5_best_exit_minus_baseline_exit",
                "tp_mult",
                "sl_mult",
                "time_stop_min",
            ],
            ascending=[False, False, False, False, False, False, True, True, True],
            kind="mergesort",
        ).head(int(max_pool))
        pool = pool.drop(columns=["is_control"])

    pool = pool.reset_index(drop=True)
    pool["cfg_hash"] = [
        _cfg_hash(
            {
                "tp_mult": float(r["tp_mult"]),
                "sl_mult": float(r["sl_mult"]),
                "time_stop_min": int(r["time_stop_min"]),
                "break_even_enabled": int(r["break_even_enabled"]),
                "break_even_trigger_r": float(r["break_even_trigger_r"]),
                "break_even_offset_bps": float(r["break_even_offset_bps"]),
                "partial_take_enabled": int(r["partial_take_enabled"]),
                "partial_take_r": float(r["partial_take_r"]),
                "partial_take_pct": float(r["partial_take_pct"]),
            }
        )
        for _, r in pool.iterrows()
    ]
    pool["cfg_id"] = np.arange(1, len(pool) + 1, dtype=int)
    if str(control_hash):
        hit = pool[pool["cfg_hash"] == str(control_hash)]
        if hit.empty:
            raise RuntimeError("Control cfg hash missing from candidate pool")
    return pool


def _cfg_row_to_exitcfg(r: pd.Series) -> exit_sweep.ExitCfg:
    return exit_sweep.ExitCfg(
        tp_mult=float(r["tp_mult"]),
        sl_mult=float(r["sl_mult"]),
        time_stop_min=int(r["time_stop_min"]),
        break_even_enabled=int(r["break_even_enabled"]),
        break_even_trigger_r=float(r["break_even_trigger_r"]),
        break_even_offset_bps=float(r["break_even_offset_bps"]),
        partial_take_enabled=int(r["partial_take_enabled"]),
        partial_take_r=float(r["partial_take_r"]),
        partial_take_pct=float(r["partial_take_pct"]),
    )


def _simulate_candidate_rows(
    *,
    bundle: ga_exec.SymbolBundle,
    pool_df: pd.DataFrame,
    mode: str,
    eval_cfg: Dict[str, Any],
) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    for _, r in pool_df.iterrows():
        cfg_id = int(r["cfg_id"])
        cfg = _cfg_row_to_exitcfg(r)
        genome = exit_sweep._cfg_to_genome(cfg, mode=mode)
        rows: List[Dict[str, Any]] = []
        for ctx in bundle.contexts:
            row = ga_exec._simulate_candidate_signal(
                ctx=ctx,
                genome=genome,
                eval_cfg=eval_cfg,
                last_entry_time=None,
            )
            row["signal_id"] = ctx.signal_id
            row["symbol"] = ctx.symbol
            row["signal_time"] = str(ctx.signal_time)
            rows.append(row)
        df = pd.DataFrame(rows).drop_duplicates(subset=["signal_id"], keep="last").set_index("signal_id", drop=False)
        out[cfg_id] = df
    return out


def _expectancy_from_rows(df: pd.DataFrame) -> float:
    if df.empty:
        return float("-inf")
    filled = pd.to_numeric(df.get("exec_filled", 0), errors="coerce").fillna(0).astype(int)
    valid = pd.to_numeric(df.get("exec_valid_for_metrics", 0), errors="coerce").fillna(0).astype(int)
    pnl = pd.to_numeric(df.get("exec_pnl_net_pct", np.nan), errors="coerce")
    n = int(len(df))
    vec = np.zeros(n, dtype=float)
    m = (filled == 1) & (valid == 1) & pnl.notna()
    if m.any():
        vec[m.to_numpy(dtype=bool)] = pnl[m].to_numpy(dtype=float)
    return float(np.mean(vec))


def _trades_from_rows(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    filled = pd.to_numeric(df.get("exec_filled", 0), errors="coerce").fillna(0).astype(int)
    valid = pd.to_numeric(df.get("exec_valid_for_metrics", 0), errors="coerce").fillna(0).astype(int)
    return int(((filled == 1) & (valid == 1)).sum())


def _apply_stability(split_df: pd.DataFrame) -> Tuple[int, int, int, int]:
    g = pd.to_numeric(split_df.get("phasec_global_expectancy_net", np.nan), errors="coerce")
    r = pd.to_numeric(split_df.get("regime_expectancy_net", np.nan), errors="coerce")
    s1, s2, s3, s_all = exit_sweep._stability_flags(
        exec_min=float(r.min()) if r.notna().any() else float("nan"),
        exec_med=float(r.median()) if r.notna().any() else float("nan"),
        exec_std=float(r.std(ddof=0)) if r.notna().any() else float("nan"),
        base_min=float(g.min()) if g.notna().any() else float("nan"),
        base_med=float(g.median()) if g.notna().any() else float("nan"),
        base_std=float(g.std(ddof=0)) if g.notna().any() else float("nan"),
    )
    return int(s1), int(s2), int(s3), int(s_all)


def run(args: argparse.Namespace) -> Path:
    if str(args.symbol).strip().upper() != "SOLUSDT":
        raise SystemExit("Scope lock: --symbol must be SOLUSDT")
    phase_c_dir = _resolve(args.phase_c_dir)
    contract = _load_phasec_contract(phase_c_dir)
    manifest = contract["manifest"]

    run_dir = _resolve(args.outdir) / f"PHASED_SOL_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    snap = run_dir / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    # Freeze + validate inputs from Phase C.
    subset_csv = (phase_c_dir / "signal_subset.csv").resolve()
    split_json = (phase_c_dir / "wf_split_definition.json").resolve()
    split_def = _load_split_definition(split_json)
    signal_df = pd.read_csv(subset_csv)
    signal_df["signal_time"] = pd.to_datetime(signal_df["signal_time"], utc=True, errors="coerce")
    signal_df = signal_df.sort_values("signal_time").reset_index(drop=True)
    if "signal_id" not in signal_df.columns:
        signal_df["signal_id"] = [f"sig_{i:05d}" for i in range(1, len(signal_df) + 1)]

    # Copy contract files exactly (same definitions as Phase A).
    shutil.copy2(contract["phase_a_metrics_path"], run_dir / "metrics_definition.md")
    shutil.copy2(contract["phase_a_fee_path"], run_dir / "fee_model.json")
    fee = _parse_fee_model(run_dir / "fee_model.json")

    # Build regimes from frozen subset (no lookahead features).
    labels_df, cov_df, diag_text = build_regime_labels(
        signal_df=signal_df,
        min_hist_bars=int(args.min_hist_bars),
        use_trend=int(args.use_trend),
        use_session=int(args.use_session),
    )
    labels_path = run_dir / "regime_labels.csv"
    cov_path = run_dir / "regime_coverage.csv"
    diag_path = run_dir / "regime_diagnostics.md"
    labels_df.to_csv(labels_path, index=False)
    cov_df.to_csv(cov_path, index=False)
    diag_path.write_text(diag_text, encoding="utf-8")

    # Build evaluator bundle from frozen subset and frozen splits.
    eval_args = _build_eval_args(
        symbol="SOLUSDT",
        signal_subset_csv=subset_csv,
        mode=str(args.mode),
        split_count=len(split_def),
        fee=fee,
        args=args,
    )
    bundles, _ = ga_exec._prepare_bundles(eval_args)
    if len(bundles) != 1:
        raise RuntimeError(f"Expected 1 bundle, got {len(bundles)}")
    bundle = bundles[0]
    bundle.splits = [dict(s) for s in split_def]
    if len(bundle.contexts) != len(signal_df):
        raise RuntimeError(f"Context/signal length mismatch: {len(bundle.contexts)} vs {len(signal_df)}")

    # Candidate pool.
    control_cfg = _control_cfg_from_manifest(contract["control_cfg"])
    pool_df = _candidate_pool(
        refine_csv=phase_c_dir / "refine" / "exit_sweep_results_refine.csv",
        control_cfg=control_cfg,
        control_hash=contract["control_cfg_hash"],
        topn=int(args.candidate_topn),
        max_pool=int(args.candidate_max_pool),
    )
    pool_csv = run_dir / "phase_d_candidate_pool.csv"
    pool_df.to_csv(pool_csv, index=False)
    pool_manifest = {
        "generated_utc": _utc_now_iso(),
        "source_refine_csv": str((phase_c_dir / "refine" / "exit_sweep_results_refine.csv").resolve()),
        "candidate_topn": int(args.candidate_topn),
        "candidate_max_pool": int(args.candidate_max_pool),
        "pool_size": int(len(pool_df)),
        "control_cfg_hash": str(contract["control_cfg_hash"]),
    }
    _json_dump(run_dir / "candidate_pool_manifest.json", pool_manifest)

    # Pre-sim all candidates once.
    eval_cfg = {
        "exec_horizon_hours": float(eval_args.exec_horizon_hours),
        "fee_bps_maker": float(eval_args.fee_bps_maker),
        "fee_bps_taker": float(eval_args.fee_bps_taker),
        "slippage_bps_limit": float(eval_args.slippage_bps_limit),
        "slippage_bps_market": float(eval_args.slippage_bps_market),
        "force_no_skip": 1,
    }
    rows_by_cfg = _simulate_candidate_rows(
        bundle=bundle,
        pool_df=pool_df,
        mode=str(eval_args.mode),
        eval_cfg=eval_cfg,
    )

    signal_to_label = dict(zip(labels_df["signal_id"].astype(str), labels_df["combined_regime"].astype(str)))
    signal_to_idx = {ctx.signal_id: i for i, ctx in enumerate(bundle.contexts)}

    # Determine test support from control rows and merge low-support regimes into "other".
    control_cfg_id = int(pool_df[pool_df["cfg_hash"] == contract["control_cfg_hash"]]["cfg_id"].iloc[0])
    control_rows = rows_by_cfg[control_cfg_id]
    regime_test_trade_count: Dict[str, int] = {}
    split_of_signal: Dict[str, int] = {}
    for sp in split_def:
        sid = int(sp["split_id"])
        for ctx in bundle.contexts[int(sp["test_start"]) : int(sp["test_end"])]:
            split_of_signal[ctx.signal_id] = sid
            rr = control_rows.loc[ctx.signal_id]
            if int(rr.get("exec_filled", 0)) == 1 and int(rr.get("exec_valid_for_metrics", 0)) == 1:
                lb = signal_to_label.get(ctx.signal_id, "other")
                regime_test_trade_count[lb] = int(regime_test_trade_count.get(lb, 0) + 1)

    # Cardinality + support pruning.
    support_sorted = sorted(regime_test_trade_count.items(), key=lambda kv: (-kv[1], kv[0]))
    keep_regimes = {k for k, v in support_sorted if int(v) >= int(args.min_regime_support)}
    if len(keep_regimes) > int(args.max_regimes):
        keep_regimes = {k for k, _ in support_sorted[: int(args.max_regimes)]}
    merged_labels: Dict[str, str] = {}
    for sid, lb in signal_to_label.items():
        merged_labels[sid] = lb if lb in keep_regimes else "other"

    # Save merged coverage.
    merged_cov_rows: Dict[str, int] = {}
    for sid, lb in merged_labels.items():
        merged_cov_rows[lb] = int(merged_cov_rows.get(lb, 0) + 1)
    merged_cov = pd.DataFrame(
        [{"combined_regime_merged": k, "signals": int(v), "test_trades_control": int(regime_test_trade_count.get(k, 0))} for k, v in merged_cov_rows.items()]
    ).sort_values(["signals", "combined_regime_merged"], ascending=[False, True])
    merged_cov.to_csv(cov_path, index=False)
    labels_df["combined_regime_merged"] = labels_df["signal_id"].astype(str).map(lambda x: merged_labels.get(x, "other"))
    labels_df.to_csv(labels_path, index=False)

    # Split-wise train selection + test eval.
    map_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    signal_rows_global: List[Dict[str, Any]] = []
    signal_rows_regime: List[Dict[str, Any]] = []
    split_mapping: Dict[str, Dict[str, int]] = {}

    pool_info = pool_df.set_index("cfg_id").to_dict(orient="index")
    cfg_ids = [int(x) for x in pool_df["cfg_id"].tolist()]
    for sp in split_def:
        split_id = int(sp["split_id"])
        tr_idx = range(int(sp["train_start"]), int(sp["train_end"]))
        te_idx = range(int(sp["test_start"]), int(sp["test_end"]))
        tr_ids = [bundle.contexts[i].signal_id for i in tr_idx]
        te_ids = [bundle.contexts[i].signal_id for i in te_idx]
        tr_labels = {sid: merged_labels.get(sid, "other") for sid in tr_ids}
        te_labels = {sid: merged_labels.get(sid, "other") for sid in te_ids}
        regimes = sorted(set(tr_labels.values()) | set(te_labels.values()))
        mapping: Dict[str, int] = {}

        for rg in regimes:
            train_ids_rg = [sid for sid in tr_ids if tr_labels.get(sid) == rg]
            test_ids_rg = [sid for sid in te_ids if te_labels.get(sid) == rg]
            reason = ""
            if rg == "other":
                mapping[rg] = int(control_cfg_id)
                reason = "merged_or_low_support"
            elif len(train_ids_rg) < int(args.min_train_regime_signals):
                mapping[rg] = int(control_cfg_id)
                reason = "train_sparse_fallback_control"
            else:
                best_cfg = int(control_cfg_id)
                best_key = (-1e18, -1e18, -1e18, 0)
                for cfg_id in cfg_ids:
                    d = rows_by_cfg[cfg_id].loc[train_ids_rg]
                    exp = _expectancy_from_rows(d)
                    roll = ga_exec._aggregate_rows(d) if not d.empty else {"exec": {}}
                    cvar = float(roll["exec"].get("cvar_5", np.nan))
                    maxdd = float(roll["exec"].get("max_drawdown", np.nan))
                    key = (
                        float(exp) if np.isfinite(exp) else -1e18,
                        float(cvar) if np.isfinite(cvar) else -1e18,
                        float(maxdd) if np.isfinite(maxdd) else -1e18,
                        -int(cfg_id),
                    )
                    if key > best_key:
                        best_key = key
                        best_cfg = int(cfg_id)
                mapping[rg] = int(best_cfg)
                reason = "train_selected"

            cfg_meta = pool_info[int(mapping[rg])]
            map_rows.append(
                {
                    "split_id": int(split_id),
                    "regime": str(rg),
                    "train_signals": int(len(train_ids_rg)),
                    "test_signals": int(len(test_ids_rg)),
                    "selected_cfg_id_train": int(mapping[rg]),
                    "selected_cfg_hash": str(cfg_meta["cfg_hash"]),
                    "selected_tp_mult": float(cfg_meta["tp_mult"]),
                    "selected_sl_mult": float(cfg_meta["sl_mult"]),
                    "selected_time_stop_min": int(cfg_meta["time_stop_min"]),
                    "selected_break_even_enabled": int(cfg_meta["break_even_enabled"]),
                    "selected_partial_take_enabled": int(cfg_meta["partial_take_enabled"]),
                    "selection_reason": reason,
                }
            )
        split_mapping[str(split_id)] = {str(k): int(v) for k, v in mapping.items()}

        # Evaluate test rows for split.
        split_global_df = rows_by_cfg[control_cfg_id].loc[te_ids].copy()
        split_regime_rows: List[Dict[str, Any]] = []
        for sid in te_ids:
            rg = te_labels.get(sid, "other")
            cfg_id = int(mapping.get(rg, control_cfg_id))
            rr = rows_by_cfg[cfg_id].loc[sid].to_dict()
            rr["applied_cfg_id"] = int(cfg_id)
            rr["regime_label"] = str(rg)
            split_regime_rows.append(rr)
        split_regime_df = pd.DataFrame(split_regime_rows)

        split_global_df["regime_label"] = [te_labels.get(sid, "other") for sid in te_ids]
        split_global_df["applied_cfg_id"] = int(control_cfg_id)
        split_global_df["split_id"] = int(split_id)
        split_regime_df["split_id"] = int(split_id)

        signal_rows_global.append(split_global_df)
        signal_rows_regime.append(split_regime_df)

        g_roll = ga_exec._aggregate_rows(split_global_df)
        r_roll = ga_exec._aggregate_rows(split_regime_df)
        g = g_roll["exec"]
        r = r_roll["exec"]
        split_rows.append(
            {
                "symbol": "SOLUSDT",
                "split_id": int(split_id),
                "signals_total": int(r.get("signals_total", 0)),
                "trades_total_phasec_global": int(g.get("entries_valid", 0)),
                "trades_total_regime_exit": int(r.get("entries_valid", 0)),
                "phasec_global_expectancy_net": float(g.get("mean_expectancy_net", np.nan)),
                "regime_expectancy_net": float(r.get("mean_expectancy_net", np.nan)),
                "delta_expectancy_regime_exit_minus_phasec_global_exit": float(r.get("mean_expectancy_net", np.nan) - g.get("mean_expectancy_net", np.nan)),
                "phasec_global_cvar_5": float(g.get("cvar_5", np.nan)),
                "regime_cvar_5": float(r.get("cvar_5", np.nan)),
                "delta_cvar5_regime_exit_minus_phasec_global_exit": float(r.get("cvar_5", np.nan) - g.get("cvar_5", np.nan)),
                "phasec_global_max_drawdown": float(g.get("max_drawdown", np.nan)),
                "regime_max_drawdown": float(r.get("max_drawdown", np.nan)),
                "delta_maxdd_regime_exit_minus_phasec_global_exit": float(r.get("max_drawdown", np.nan) - g.get("max_drawdown", np.nan)),
                "phasec_global_entry_rate": float(g.get("entry_rate", np.nan)),
                "regime_entry_rate": float(r.get("entry_rate", np.nan)),
                "phasec_global_taker_share": float(g.get("taker_share", np.nan)),
                "regime_taker_share": float(r.get("taker_share", np.nan)),
                "phasec_global_median_fill_delay_min": float(g.get("median_fill_delay_min", np.nan)),
                "regime_median_fill_delay_min": float(r.get("median_fill_delay_min", np.nan)),
                "phasec_global_p95_fill_delay_min": float(g.get("p95_fill_delay_min", np.nan)),
                "regime_p95_fill_delay_min": float(r.get("p95_fill_delay_min", np.nan)),
                "regime_missing_slice_rate": float(pd.to_numeric(split_regime_df.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean())
                if not split_regime_df.empty
                else float("nan"),
            }
        )

    split_df = pd.DataFrame(split_rows).sort_values("split_id").reset_index(drop=True)
    map_df = pd.DataFrame(map_rows).sort_values(["split_id", "regime"]).reset_index(drop=True)
    regime_mapping_csv = run_dir / "regime_mapping_by_split.csv"
    map_df.to_csv(regime_mapping_csv, index=False)
    split_csv = run_dir / "walkforward_results_by_split.csv"
    split_df.to_csv(split_csv, index=False)

    all_g = pd.concat(signal_rows_global, ignore_index=True) if signal_rows_global else pd.DataFrame()
    all_r = pd.concat(signal_rows_regime, ignore_index=True) if signal_rows_regime else pd.DataFrame()
    g_roll_all = ga_exec._aggregate_rows(all_g) if not all_g.empty else {"exec": {}}
    r_roll_all = ga_exec._aggregate_rows(all_r) if not all_r.empty else {"exec": {}}
    g = g_roll_all["exec"]
    r = r_roll_all["exec"]

    d_exp = float(r.get("mean_expectancy_net", np.nan) - g.get("mean_expectancy_net", np.nan))
    d_cvar = float(r.get("cvar_5", np.nan) - g.get("cvar_5", np.nan))
    d_maxdd = float(r.get("max_drawdown", np.nan) - g.get("max_drawdown", np.nan))
    s1, s2, s3, s_pass = _apply_stability(split_df)
    p_exp = int(np.isfinite(d_exp) and d_exp >= float(args.expectancy_epsilon))
    p_maxdd = int(np.isfinite(d_maxdd) and d_maxdd >= -float(args.maxdd_worse_tol))
    p_cvar = int(np.isfinite(d_cvar) and d_cvar >= -float(args.cvar_worse_tol))
    p_part = int(np.isfinite(float(r.get("entry_rate", np.nan))) and float(r.get("entry_rate", np.nan)) > 0.0)
    miss_rate = float(pd.to_numeric(all_r.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not all_r.empty else float("nan")
    p_data = int(np.isfinite(miss_rate) and miss_rate <= float(args.max_missing_slice_rate))
    non_other = merged_cov[merged_cov["combined_regime_merged"] != "other"].copy()
    regime_support_ok = int(
        (non_other.empty or (pd.to_numeric(non_other["test_trades_control"], errors="coerce").fillna(0) >= int(args.min_regime_support)).all())
        and (
            non_other.empty
            or float(pd.to_numeric(non_other["test_trades_control"], errors="coerce").max() / max(1.0, pd.to_numeric(non_other["test_trades_control"], errors="coerce").sum()))
            <= float(args.max_regime_share)
        )
    )
    pass_all = int(p_exp and p_maxdd and p_cvar and s_pass and p_part and p_data and regime_support_ok)

    by_symbol = pd.DataFrame(
        [
            {
                "symbol": "SOLUSDT",
                "signals_total": int(r.get("signals_total", 0)),
                "trades_total_phasec_global": int(g.get("entries_valid", 0)),
                "trades_total_regime_exit": int(r.get("entries_valid", 0)),
                "phasec_global_expectancy_net": float(g.get("mean_expectancy_net", np.nan)),
                "regime_expectancy_net": float(r.get("mean_expectancy_net", np.nan)),
                "delta_expectancy_regime_exit_minus_phasec_global_exit": d_exp,
                "phasec_global_pnl_net_sum": float(g.get("pnl_net_sum", np.nan)),
                "regime_pnl_net_sum": float(r.get("pnl_net_sum", np.nan)),
                "phasec_global_cvar_5": float(g.get("cvar_5", np.nan)),
                "regime_cvar_5": float(r.get("cvar_5", np.nan)),
                "delta_cvar5_regime_exit_minus_phasec_global_exit": d_cvar,
                "phasec_global_max_drawdown": float(g.get("max_drawdown", np.nan)),
                "regime_max_drawdown": float(r.get("max_drawdown", np.nan)),
                "delta_maxdd_regime_exit_minus_phasec_global_exit": d_maxdd,
                "phasec_global_entry_rate": float(g.get("entry_rate", np.nan)),
                "regime_entry_rate": float(r.get("entry_rate", np.nan)),
                "phasec_global_taker_share": float(g.get("taker_share", np.nan)),
                "regime_taker_share": float(r.get("taker_share", np.nan)),
                "phasec_global_median_fill_delay_min": float(g.get("median_fill_delay_min", np.nan)),
                "regime_median_fill_delay_min": float(r.get("median_fill_delay_min", np.nan)),
                "phasec_global_p95_fill_delay_min": float(g.get("p95_fill_delay_min", np.nan)),
                "regime_p95_fill_delay_min": float(r.get("p95_fill_delay_min", np.nan)),
                "regime_missing_slice_rate": miss_rate,
                "stability_pass": int(s_pass),
                "regime_support_ok": int(regime_support_ok),
            }
        ]
    )
    by_symbol_csv = run_dir / "risk_rollup_by_symbol.csv"
    by_symbol.to_csv(by_symbol_csv, index=False)

    overall = by_symbol.copy()
    overall.insert(0, "scope", "overall")
    overall.insert(1, "symbols", 1)
    overall["pass_expectancy"] = int(p_exp)
    overall["pass_maxdd_not_worse"] = int(p_maxdd)
    overall["pass_cvar_not_worse"] = int(p_cvar)
    overall["pass_stability_min"] = int(s1)
    overall["pass_stability_median"] = int(s2)
    overall["pass_stability_std"] = int(s3)
    overall["pass_stability"] = int(s_pass)
    overall["pass_data_quality"] = int(p_data)
    overall["pass_participation"] = int(p_part)
    overall["pass_regime_support_ok"] = int(regime_support_ok)
    overall["pass_all"] = int(pass_all)
    overall_csv = run_dir / "risk_rollup_overall.csv"
    overall.to_csv(overall_csv, index=False)

    cmp_csv = run_dir / "comparison_vs_phasec_global_exit.csv"
    comparison = pd.DataFrame(
        [
            {
                "symbol": "SOLUSDT",
                "phasec_control_cfg_hash": str(contract["control_cfg_hash"]),
                "phasec_control_cfg": json.dumps(contract["control_cfg"], sort_keys=True),
                "phase_d_pass": int(pass_all),
                "delta_expectancy_regime_exit_minus_phasec_global_exit": d_exp,
                "delta_maxdd_regime_exit_minus_phasec_global_exit": d_maxdd,
                "delta_cvar5_regime_exit_minus_phasec_global_exit": d_cvar,
                "pass_expectancy": int(p_exp),
                "pass_maxdd_not_worse": int(p_maxdd),
                "pass_cvar_not_worse": int(p_cvar),
                "pass_stability": int(s_pass),
                "pass_data_quality": int(p_data),
                "pass_participation": int(p_part),
                "pass_regime_support_ok": int(regime_support_ok),
                "regime_support_ok": int(regime_support_ok),
            }
        ]
    )
    comparison.to_csv(cmp_csv, index=False)

    # Final mapping (consensus across splits by test_signals weight).
    final_map: Dict[str, int] = {}
    for regime in sorted(map_df["regime"].astype(str).unique().tolist()):
        t = map_df[map_df["regime"].astype(str) == regime].copy()
        if t.empty:
            continue
        t["w"] = pd.to_numeric(t["test_signals"], errors="coerce").fillna(0.0)
        agg = t.groupby("selected_cfg_id_train", as_index=False)["w"].sum().sort_values(["w", "selected_cfg_id_train"], ascending=[False, True])
        final_map[regime] = int(agg.iloc[0]["selected_cfg_id_train"])

    final_map_json = run_dir / "final_regime_mapping.json"
    _json_dump(
        final_map_json,
        {
            "generated_utc": _utc_now_iso(),
            "symbol": "SOLUSDT",
            "control_cfg_hash": str(contract["control_cfg_hash"]),
            "split_mappings": split_mapping,
            "consensus_mapping": final_map,
            "low_support_merged_to_other": sorted([k for k in regime_test_trade_count.keys() if k not in keep_regimes]),
            "keep_regimes": sorted(list(keep_regimes)),
        },
    )

    # Decision docs.
    decision_lines = [
        "# Phase D SOL Decision",
        "",
        f"- Generated UTC: {_utc_now_iso()}",
        "- Symbol: SOLUSDT",
        f"- Phase C control dir: `{phase_c_dir}`",
        f"- Phase C control cfg_hash: `{contract['control_cfg_hash']}`",
        "",
        "## Deltas vs Phase C Global Exit",
        "",
        f"- delta_expectancy_regime_exit_minus_phasec_global_exit: {d_exp:.6f}",
        f"- delta_maxdd_regime_exit_minus_phasec_global_exit: {d_maxdd:.6f}",
        f"- delta_cvar5_regime_exit_minus_phasec_global_exit: {d_cvar:.6f}",
        "",
        "## Gates",
        "",
        f"- pass_expectancy (>= {float(args.expectancy_epsilon):.6f}): {int(p_exp)}",
        f"- pass_maxdd_not_worse (>= -{float(args.maxdd_worse_tol):.3f}): {int(p_maxdd)}",
        f"- pass_cvar_not_worse (>= -{float(args.cvar_worse_tol):.6f}): {int(p_cvar)}",
        f"- pass_stability: {int(s_pass)}",
        f"- pass_data_quality: {int(p_data)}",
        f"- pass_participation: {int(p_part)}",
        f"- regime_support_ok: {int(regime_support_ok)}",
        "",
        f"- Final Decision: **{'PASS' if int(pass_all) == 1 else 'FAIL'}**",
    ]
    (run_dir / "decision.md").write_text("\n".join(decision_lines).strip() + "\n", encoding="utf-8")

    cmp_md_lines = [
        "# Comparison vs Phase C Global Exit",
        "",
        f"- Phase C control: `{contract['control_cfg_hash']}`",
        f"- delta_expectancy_regime_exit_minus_phasec_global_exit: {d_exp:.6f}",
        f"- delta_maxdd_regime_exit_minus_phasec_global_exit: {d_maxdd:.6f}",
        f"- delta_cvar5_regime_exit_minus_phasec_global_exit: {d_cvar:.6f}",
        f"- Decision: **{'PASS' if int(pass_all) == 1 else 'FAIL'}**",
    ]
    (run_dir / "comparison_vs_phasec_global_exit.md").write_text("\n".join(cmp_md_lines).strip() + "\n", encoding="utf-8")

    (run_dir / "repro.md").write_text(
        "\n".join(
            [
                "# Repro",
                "",
                "```bash",
                "cd /root/analysis/0.87",
                ".venv/bin/python scripts/phase_d_sol_runner.py \\",
                f"  --phase-c-dir {phase_c_dir} \\",
                "  --symbol SOLUSDT \\",
                f"  --mode {args.mode}",
                "```",
            ]
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    (run_dir / "phase_result.md").write_text(
        "\n".join(
            [
                "Phase: D (SOL regime-aware exits)",
                f"Timestamp UTC: {_utc_now_iso()}",
                f"Status: {'PASS' if int(pass_all) == 1 else 'FAIL'}",
                f"Delta expectancy: {d_exp:.6f}",
                f"Delta maxDD: {d_maxdd:.6f}",
                f"Delta CVaR5: {d_cvar:.6f}",
                f"regime_support_ok: {int(regime_support_ok)}",
                "No multi-symbol execution.",
            ]
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    _write_git_status(run_dir / "git_status.txt")
    for p in [
        phase_c_dir / "run_manifest.json",
        phase_c_dir / "decision.md",
        phase_c_dir / "signal_subset.csv",
        phase_c_dir / "signal_subset_hash.txt",
        phase_c_dir / "wf_split_definition.json",
        contract["phase_a_metrics_path"],
        contract["phase_a_fee_path"],
        _resolve(args.execution_config),
    ]:
        if Path(p).exists():
            shutil.copy2(Path(p), snap / Path(p).name)

    run_manifest = {
        "generated_utc": _utc_now_iso(),
        "symbol": "SOLUSDT",
        "phase_c_dir": str(phase_c_dir),
        "phase_c_manifest_path": str((phase_c_dir / "run_manifest.json").resolve()),
        "phase_c_manifest_sha256": _sha256_file(phase_c_dir / "run_manifest.json"),
        "phase_c_control_cfg_hash": str(contract["control_cfg_hash"]),
        "phase_a_contract": {
            "metrics_definition_path": str(contract["phase_a_metrics_path"]),
            "metrics_definition_sha256": str(contract["phase_a_metrics_hash"]),
            "fee_model_path": str(contract["phase_a_fee_path"]),
            "fee_model_sha256": str(contract["phase_a_fee_hash"]),
        },
        "frozen_inputs": {
            "signal_subset_path": str(subset_csv),
            "signal_subset_hash": str(contract["subset_hash"]),
            "wf_split_definition_path": str(split_json),
            "wf_split_definition_sha256": str(contract["split_hash"]),
        },
        "seeds": {
            "candidate_pool_seed": int(args.candidate_seed),
            "mapping_seed": int(args.mapping_seed),
        },
        "caps": {
            "candidate_topn": int(args.candidate_topn),
            "candidate_max_pool": int(args.candidate_max_pool),
            "max_regimes": int(args.max_regimes),
            "min_regime_support": int(args.min_regime_support),
        },
        "final_mapping_path": str(final_map_json),
        "comparison_csv": str(cmp_csv),
        "decision_pass": int(pass_all),
    }
    _json_dump(run_dir / "run_manifest.json", run_manifest)

    print(str(run_dir))
    print(str(run_dir / "decision.md"))
    print(str(cmp_csv))
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase D regime-aware exits on SOL only vs Phase C global control.")
    ap.add_argument("--phase-c-dir", default="reports/execution_layer/PHASEC_SOL_20260221_231430")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--mode", choices=["tight", "normal"], default="tight")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--execution-config", default="configs/execution_configs.yaml")
    ap.add_argument("--workers", type=int, default=3)

    ap.add_argument("--timeframe", default="3m")
    ap.add_argument("--pre-buffer-hours", type=float, default=6.0)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--cache-dir", default="data/processed/_exec_klines_cache")
    ap.add_argument("--max-fetch-retries", type=int, default=8)
    ap.add_argument("--retry-base-sleep", type=float, default=0.5)
    ap.add_argument("--retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--fetch-pause-sec", type=float, default=0.03)
    ap.add_argument("--train-ratio", type=float, default=0.7)

    ap.add_argument("--use-trend", type=int, default=1)
    ap.add_argument("--use-session", type=int, default=0)
    ap.add_argument("--min-hist-bars", type=int, default=50)
    ap.add_argument("--min-regime-support", type=int, default=30)
    ap.add_argument("--max-regimes", type=int, default=12)
    ap.add_argument("--max-regime-share", type=float, default=0.85)
    ap.add_argument("--min-train-regime-signals", type=int, default=20)

    ap.add_argument("--candidate-topn", type=int, default=40)
    ap.add_argument("--candidate-max-pool", type=int, default=80)
    ap.add_argument("--candidate-seed", type=int, default=42)
    ap.add_argument("--mapping-seed", type=int, default=123)

    ap.add_argument("--expectancy-epsilon", type=float, default=5e-5)
    ap.add_argument("--maxdd-worse-tol", type=float, default=0.02)
    ap.add_argument("--cvar-worse-tol", type=float, default=1e-4)
    ap.add_argument("--max-missing-slice-rate", type=float, default=0.02)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
