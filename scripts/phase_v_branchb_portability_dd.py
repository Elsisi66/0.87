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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402
from scripts import sol_reconcile_truth as recon  # noqa: E402
from scripts import phase_u_combined_1h3m_pilot as phaseu  # noqa: E402


LOCKED = {
    "representative_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
}

PORTABILITY_CANDIDATES = ["AVAXUSDT", "NEARUSDT", "BCHUSDT"]


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


def df_to_plain_table(df: pd.DataFrame, cols: Optional[Sequence[str]] = None) -> str:
    x = df
    if cols is not None:
        keep = [c for c in cols if c in x.columns]
        if keep:
            x = x.loc[:, keep]
    if x.empty:
        return "(none)"
    return x.to_string(index=False)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


def tail_mean(arr: np.ndarray, q: float) -> float:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(q) * x.size)))
    return float(np.mean(np.sort(x)[:k]))


def max_drawdown_from_pnl(pnl_sig: np.ndarray) -> float:
    if pnl_sig.size == 0:
        return float("nan")
    cum = np.cumsum(np.nan_to_num(pnl_sig, nan=0.0))
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.nanmin(dd)) if dd.size else float("nan")


def max_consecutive_losses(pnl_sig: np.ndarray) -> int:
    best = 0
    cur = 0
    for x in pnl_sig:
        if np.isfinite(x) and x < 0.0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def norm_cdf(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0))))


def z_proxy(mean: float, std: float, n: float) -> float:
    if not (np.isfinite(mean) and np.isfinite(std) and np.isfinite(n)):
        return float("nan")
    if std <= 0.0 or n <= 1.0:
        return float("nan")
    return float(mean / (std / math.sqrt(n)))


def effective_trials_from_corr(mat: np.ndarray) -> Tuple[float, float]:
    if mat.size == 0 or mat.shape[0] <= 1:
        return float(mat.shape[0]), 0.0
    x = mat.copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = np.isfinite(col)
        fill = float(np.nanmedian(col[finite])) if finite.any() else 0.0
        col[~finite] = fill
        x[:, j] = col
    cc = np.corrcoef(x)
    if np.ndim(cc) == 0:
        return float(x.shape[0]), 0.0
    iu = np.triu_indices_from(cc, k=1)
    vals = np.abs(cc[iu])
    vals = vals[np.isfinite(vals)]
    avg_abs = float(vals.mean()) if vals.size else 0.0
    n = float(x.shape[0])
    n_eff = float(n / max(1e-9, (1.0 + (n - 1.0) * avg_abs)))
    return n_eff, avg_abs


def parse_rank_key(v: Any) -> Tuple[float, ...]:
    if isinstance(v, list):
        return tuple(float(x) for x in v)
    s = str(v).strip()
    if not s:
        return tuple()
    try:
        arr = json.loads(s)
        if isinstance(arr, list):
            return tuple(float(x) for x in arr)
    except Exception:
        pass
    try:
        arr = eval(s, {"__builtins__": {}}, {})  # noqa: S307
        if isinstance(arr, (list, tuple)):
            return tuple(float(x) for x in arr)
    except Exception:
        pass
    return tuple()


def build_exec_args(symbol: str, signals_csv: Path, max_signals: int, seed: int) -> argparse.Namespace:
    parser = ga_exec.build_arg_parser()
    args = parser.parse_args([])
    args.symbol = str(symbol).upper()
    args.symbols = ""
    args.rank = 1
    args.signals_csv = str(signals_csv)
    args.max_signals = int(max_signals)
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


def extract_genome(row: pd.Series) -> Dict[str, Any]:
    g: Dict[str, Any] = {}
    for c in row.index:
        if not c.startswith("g_"):
            continue
        k = c[2:]
        v = row[c]
        if pd.isna(v):
            continue
        if isinstance(v, (np.integer, int)):
            g[k] = int(v)
        elif isinstance(v, (np.floating, float)):
            fv = float(v)
            if abs(fv - round(fv)) < 1e-12 and k in {
                "max_fill_delay_min",
                "fallback_to_market",
                "fallback_delay_min",
                "micro_vol_filter",
                "killzone_filter",
                "mss_displacement_gate",
                "time_stop_min",
                "break_even_enabled",
                "trailing_enabled",
                "partial_take_enabled",
                "skip_if_vol_gate",
                "use_signal_quality_gate",
                "cooldown_min",
            }:
                g[k] = int(round(fv))
            else:
                g[k] = fv
        else:
            g[k] = v
    return ga_exec._repair_genome(g, mode="tight", repair_hist=None)


def baseline_metrics_from_bundle(bundle: ga_exec.SymbolBundle, args: argparse.Namespace) -> Dict[str, Any]:
    n = int(len(bundle.contexts))
    if n == 0:
        return {
            "overall_signals_total": 0,
            "overall_entries_valid": 0,
            "overall_entry_rate": 0.0,
            "overall_exec_expectancy_net": float("nan"),
            "overall_exec_pnl_std": float("nan"),
            "overall_exec_cvar_5": float("nan"),
            "overall_exec_max_drawdown": float("nan"),
            "overall_exec_taker_share": float("nan"),
            "overall_exec_median_fill_delay_min": float("nan"),
            "overall_exec_p95_fill_delay_min": float("nan"),
            "overall_delta_expectancy_exec_minus_baseline": 0.0,
            "overall_cvar_improve_ratio": 0.0,
            "overall_maxdd_improve_ratio": 0.0,
            "min_split_expectancy_net": float("nan"),
            "median_split_expectancy_net": float("nan"),
            "std_split_expectancy_net": float("nan"),
            "constraint_pass": 1,
            "participation_pass": 0,
            "realism_pass": 0,
            "nan_pass": 0,
            "split_pass": 0,
            "hard_invalid": 1,
            "valid_for_ranking": 0,
            "invalid_reason": "no_signals",
            "participation_fail_reason": "overall:trades<1|overall:entry_rate",
            "realism_fail_reason": "",
            "split_fail_reason": "split_count_mismatch",
        }

    filled = np.array([int(getattr(c, "baseline_filled", 0)) for c in bundle.contexts], dtype=int)
    valid = np.array([int(getattr(c, "baseline_valid_for_metrics", 0)) for c in bundle.contexts], dtype=int)
    pnl_raw = np.array([float(getattr(c, "baseline_pnl_net_pct", np.nan)) for c in bundle.contexts], dtype=float)
    liq = np.array([str(getattr(c, "baseline_fill_liq", "")).strip().lower() for c in bundle.contexts], dtype=object)
    delay = np.array([float(getattr(c, "baseline_fill_delay_min", np.nan)) for c in bundle.contexts], dtype=float)

    mask = (filled == 1) & (valid == 1) & np.isfinite(pnl_raw)
    entries = int(mask.sum())
    pnl_sig = np.zeros(n, dtype=float)
    pnl_sig[mask] = pnl_raw[mask]

    entry_rate = float(entries / max(1, n))
    exp = float(np.mean(pnl_sig))
    std = float(np.std(pnl_sig, ddof=0))
    cvar5 = float(tail_mean(pnl_sig, 0.05))
    maxdd = float(max_drawdown_from_pnl(pnl_sig))

    if entries > 0:
        taker_share = float(((liq == "taker") & mask).sum() / entries)
        d = delay[mask & np.isfinite(delay)]
        med_delay = float(np.median(d)) if d.size else float("nan")
        p95_delay = float(np.quantile(d, 0.95)) if d.size else float("nan")
    else:
        taker_share = float("nan")
        med_delay = float("nan")
        p95_delay = float("nan")

    split_exp: List[float] = []
    split_count = 0
    for sp in bundle.splits:
        lo = int(sp["test_start"])
        hi = int(sp["test_end"])
        if hi <= lo:
            continue
        split_count += 1
        seg = pnl_sig[lo:hi]
        split_exp.append(float(np.mean(seg)) if seg.size else float("nan"))

    min_split = float(np.nanmin(split_exp)) if split_exp else float("nan")
    med_split = float(np.nanmedian(split_exp)) if split_exp else float("nan")
    std_split = float(np.nanstd(split_exp, ddof=0)) if split_exp else float("nan")

    min_trades_overall = max(int(args.hard_min_trades_overall), int(math.ceil(float(args.hard_min_trade_frac_overall) * max(1, n))))
    participation_pass = int((entries >= min_trades_overall) and np.isfinite(entry_rate) and entry_rate >= float(args.hard_min_entry_rate_overall))
    realism_pass = int(
        np.isfinite(taker_share)
        and np.isfinite(med_delay)
        and np.isfinite(p95_delay)
        and taker_share <= float(args.hard_max_taker_share)
        and med_delay <= float(args.hard_max_median_fill_delay_min)
        and p95_delay <= float(args.hard_max_p95_fill_delay_min)
    )
    nan_pass = int(np.isfinite(exp) and np.isfinite(cvar5) and np.isfinite(maxdd) and np.isfinite(std))
    split_pass = int(split_count == len(bundle.splits) and split_count > 0 and np.isfinite(min_split) and np.isfinite(med_split) and np.isfinite(std_split))

    invalid_reasons: List[str] = []
    part_fail: List[str] = []
    real_fail: List[str] = []
    split_fail: List[str] = []
    if participation_pass == 0:
        if entries < min_trades_overall:
            part_fail.append(f"overall:trades<{min_trades_overall}")
        if (not np.isfinite(entry_rate)) or entry_rate < float(args.hard_min_entry_rate_overall):
            part_fail.append("overall:entry_rate")
    if realism_pass == 0:
        if (not np.isfinite(taker_share)) or taker_share > float(args.hard_max_taker_share):
            real_fail.append("overall:taker_share")
        if (not np.isfinite(med_delay)) or med_delay > float(args.hard_max_median_fill_delay_min):
            real_fail.append("overall:median_fill_delay")
        if (not np.isfinite(p95_delay)) or p95_delay > float(args.hard_max_p95_fill_delay_min):
            real_fail.append("overall:p95_fill_delay")
    if split_pass == 0:
        split_fail.append("split_count_mismatch_or_nan")
    invalid_reasons.extend(part_fail)
    invalid_reasons.extend(real_fail)
    invalid_reasons.extend(split_fail)
    if nan_pass == 0:
        invalid_reasons.append("nan:baseline_metric")

    hard_invalid = int((participation_pass == 0) or (realism_pass == 0) or (nan_pass == 0) or (split_pass == 0))

    return {
        "overall_signals_total": int(n),
        "overall_entries_valid": int(entries),
        "overall_min_trades_required": int(min_trades_overall),
        "overall_entry_rate": float(entry_rate),
        "overall_exec_expectancy_net": float(exp),
        "overall_exec_pnl_std": float(std),
        "overall_exec_cvar_5": float(cvar5),
        "overall_exec_max_drawdown": float(maxdd),
        "overall_exec_taker_share": float(taker_share),
        "overall_exec_median_fill_delay_min": float(med_delay),
        "overall_exec_p95_fill_delay_min": float(p95_delay),
        "overall_delta_expectancy_exec_minus_baseline": 0.0,
        "overall_cvar_improve_ratio": 0.0,
        "overall_maxdd_improve_ratio": 0.0,
        "min_split_expectancy_net": float(min_split),
        "median_split_expectancy_net": float(med_split),
        "std_split_expectancy_net": float(std_split),
        "constraint_pass": 1,
        "participation_pass": int(participation_pass),
        "realism_pass": int(realism_pass),
        "nan_pass": int(nan_pass),
        "split_pass": int(split_pass),
        "hard_invalid": int(hard_invalid),
        "valid_for_ranking": int(hard_invalid == 0),
        "invalid_reason": "|".join(sorted(set(invalid_reasons))),
        "participation_fail_reason": "|".join(sorted(set(part_fail))),
        "realism_fail_reason": "|".join(sorted(set(real_fail))),
        "split_fail_reason": "|".join(sorted(set(split_fail))),
    }


def latest_phase_u_dir(exec_root: Path) -> Optional[Path]:
    cands = sorted([p for p in exec_root.glob("PHASEU_COMBINED_1H3M_PILOT_*") if p.is_dir()], key=lambda p: p.name)
    if not cands:
        return None
    for p in reversed(cands):
        man = p / "phaseU_run_manifest.json"
        if man.exists():
            return p
    return None


def load_execution_candidates(exec_root: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    phase_u_dir = latest_phase_u_dir(exec_root)
    if phase_u_dir is None:
        raise FileNotFoundError("No PHASEU_COMBINED_1H3M_PILOT_* run found")
    man = json.loads((phase_u_dir / "phaseU_run_manifest.json").read_text(encoding="utf-8"))

    choices = {x["exec_choice_id"]: x for x in man.get("exec_choices", [])}
    for req in ("E1", "E2", "E4"):
        if req not in choices:
            raise RuntimeError(f"Missing {req} in phase U exec choices")

    src_run = Path(str(man.get("source_phase_s", {}).get("source_run_dir", ""))).resolve()
    if not src_run.exists():
        src_run = Path(str(choices["E1"].get("source", ""))).resolve()
    if not src_run.exists():
        raise FileNotFoundError(f"Cannot locate source GA run for execution candidates: {src_run}")

    gdf = pd.read_csv(src_run / "genomes.csv")

    out: Dict[str, Dict[str, Any]] = {}
    for key in ("E1", "E2", "E4"):
        h = str(choices[key].get("genome_hash", "")).strip()
        r = gdf[gdf["genome_hash"].astype(str) == h]
        if r.empty:
            raise RuntimeError(f"Genome hash not found in source run for {key}: {h}")
        genome = extract_genome(r.iloc[0])
        out[key] = {
            "exec_choice_id": key,
            "description": str(choices[key].get("description", "")),
            "genome_hash": h,
            "genome": genome,
            "source_run": str(src_run),
        }

    meta = {
        "phase_u_dir": str(phase_u_dir),
        "phase_u_classification": str(man.get("classification", "")),
        "source_run": str(src_run),
    }
    return out, meta


def find_best_params_for_symbol(symbol: str, scan_csv: Path) -> Optional[Path]:
    if not scan_csv.exists():
        return None
    df = pd.read_csv(scan_csv)
    if df.empty:
        return None
    x = df.copy()
    x["symbol"] = x.get("symbol", "").astype(str)
    x["side"] = x.get("side", "").astype(str).str.lower()
    if "pass" in x.columns:
        x["pass"] = x["pass"].astype(str).str.lower().isin({"1", "true", "yes"})
        x = x[x["pass"]]
    x = x[(x["symbol"] == symbol) & (x["side"] == "long")]
    if x.empty:
        return None
    x["score"] = to_num(x.get("score", np.nan))
    x = x.sort_values("score", ascending=False).reset_index(drop=True)
    p = PROJECT_ROOT / str(x.iloc[0].get("params_file", "")).strip()
    return p if p.exists() else None


def generate_signals_from_params(symbol: str, params_path: Path, max_signals: int) -> pd.DataFrame:
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    p = ga_long._norm_params(phaseu.unwrap_params(payload))
    df = recon._load_symbol_df(symbol=symbol, tf="1h")
    df_feat = ga_long._ensure_indicators(df.copy(), p)
    df_feat = ga_long._prepare_signal_df(df_feat, assume_prepared=False)

    sig = np.asarray(ga_long.build_entry_signal(df_feat, p, assume_prepared=True), dtype=bool)
    cycles_raw = ga_long.compute_cycles(df_feat, p)
    cycles = ga_long._shift_cycles(cycles_raw, shift=int(p.get("cycle_shift", 1)), fill=int(p.get("cycle_fill", 2)))

    tp_vec = [float(x) for x in p.get("tp_mult_by_cycle", [1.1] * 5)]
    sl_vec = [float(x) for x in p.get("sl_mult_by_cycle", [0.999] * 5)]

    ts = pd.to_datetime(df_feat["Timestamp"], utc=True, errors="coerce")
    rows: List[Dict[str, Any]] = []
    for i in np.where(sig)[0].tolist():
        c = int(cycles[i]) if np.isfinite(cycles[i]) else 2
        c = min(max(c, 0), 4)
        rows.append(
            {
                "signal_id": f"{symbol.lower()}_sig_{i:06d}",
                "signal_time": pd.to_datetime(ts.iloc[i], utc=True),
                "tp_mult": float(tp_vec[c]),
                "sl_mult": float(sl_vec[c]),
                "cycle": int(c),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    if int(max_signals) > 0 and len(out) > int(max_signals):
        out = out.iloc[-int(max_signals) :].copy().reset_index(drop=True)
    return out


def subset_hash(df: pd.DataFrame) -> str:
    x = df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x["tp_mult"] = pd.to_numeric(x["tp_mult"], errors="coerce")
    x["sl_mult"] = pd.to_numeric(x["sl_mult"], errors="coerce")
    x = x.dropna(subset=["signal_id", "signal_time", "tp_mult", "sl_mult"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rows = [f"{r.signal_id}|{pd.to_datetime(r.signal_time, utc=True).isoformat()}|{float(r.tp_mult):.10f}|{float(r.sl_mult):.10f}" for r in x.itertuples(index=False)]
    return sha256_text("\n".join(rows))


def session_bucket(ts: pd.Series) -> pd.Series:
    h = pd.to_datetime(ts, utc=True, errors="coerce").dt.hour
    out = pd.Series(index=h.index, dtype=object)
    out[(h >= 0) & (h < 6)] = "00_05"
    out[(h >= 6) & (h < 12)] = "06_11"
    out[(h >= 12) & (h < 18)] = "12_17"
    out[(h >= 18) & (h <= 23)] = "18_23"
    return out.fillna("unknown").astype(str)


def vol_bucket(atr_pct: pd.Series) -> pd.Series:
    x = pd.to_numeric(atr_pct, errors="coerce")
    out = pd.Series(index=x.index, dtype=object)
    out[x <= 33.3333333333] = "low"
    out[(x > 33.3333333333) & (x <= 66.6666666667)] = "mid"
    out[x > 66.6666666667] = "high"
    return out.fillna("unknown").astype(str)


def eval_portability_coin(
    symbol: str,
    signals_csv: Path,
    subset_info: Dict[str, Any],
    exec_candidates: Dict[str, Dict[str, Any]],
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any], ga_exec.SymbolBundle, argparse.Namespace]:
    args = build_exec_args(symbol=symbol, signals_csv=signals_csv, max_signals=10_000, seed=seed)
    bundles, load_meta = ga_exec._prepare_bundles(args)
    if not bundles:
        raise RuntimeError(f"No bundles prepared for {symbol}")
    bundle = bundles[0]

    rows: List[Dict[str, Any]] = []

    base = baseline_metrics_from_bundle(bundle=bundle, args=args)
    rows.append(
        {
            "symbol": symbol,
            "exec_choice_id": "E0",
            "exec_choice_desc": "baseline_execution_reference",
            "exec_genome_hash": "BASELINE_E0",
            "subset_hash": subset_info.get("subset_hash", ""),
            "subset_method": subset_info.get("subset_method", ""),
            "subset_path": subset_info.get("subset_path", ""),
            "subset_source": subset_info.get("subset_source", ""),
            "params_path": subset_info.get("params_path", ""),
            "signals_total": int(base.get("overall_signals_total", 0)),
            "entries_valid": float(base.get("overall_entries_valid", np.nan)),
            "entry_rate": float(base.get("overall_entry_rate", np.nan)),
            "exec_expectancy_net": float(base.get("overall_exec_expectancy_net", np.nan)),
            "delta_expectancy_exec_minus_baseline": 0.0,
            "cvar_improve_ratio": 0.0,
            "maxdd_improve_ratio": 0.0,
            "exec_cvar_5": float(base.get("overall_exec_cvar_5", np.nan)),
            "exec_max_drawdown": float(base.get("overall_exec_max_drawdown", np.nan)),
            "taker_share": float(base.get("overall_exec_taker_share", np.nan)),
            "median_fill_delay_min": float(base.get("overall_exec_median_fill_delay_min", np.nan)),
            "p95_fill_delay_min": float(base.get("overall_exec_p95_fill_delay_min", np.nan)),
            "min_split_expectancy_net": float(base.get("min_split_expectancy_net", np.nan)),
            "median_split_expectancy_net": float(base.get("median_split_expectancy_net", np.nan)),
            "std_split_expectancy_net": float(base.get("std_split_expectancy_net", np.nan)),
            "valid_for_ranking": int(base.get("valid_for_ranking", 0)),
            "hard_invalid": int(base.get("hard_invalid", 1)),
            "constraint_pass": int(base.get("constraint_pass", 1)),
            "participation_pass": int(base.get("participation_pass", 0)),
            "realism_pass": int(base.get("realism_pass", 0)),
            "nan_pass": int(base.get("nan_pass", 0)),
            "split_pass": int(base.get("split_pass", 0)),
            "invalid_reason": str(base.get("invalid_reason", "")),
            "participation_fail_reason": str(base.get("participation_fail_reason", "")),
            "realism_fail_reason": str(base.get("realism_fail_reason", "")),
            "split_fail_reason": str(base.get("split_fail_reason", "")),
            "missing_slice_rate": float("nan"),
            "data_quality_pass": int(base.get("split_pass", 0)),
            "eval_time_sec": float(base.get("eval_time_sec", 0.0)),
        }
    )

    for k in ("E1", "E2", "E4"):
        met = ga_exec._evaluate_genome(genome=exec_candidates[k]["genome"], bundles=[bundle], args=args, detailed=False)
        rows.append(
            {
                "symbol": symbol,
                "exec_choice_id": k,
                "exec_choice_desc": exec_candidates[k]["description"],
                "exec_genome_hash": exec_candidates[k]["genome_hash"],
                "subset_hash": subset_info.get("subset_hash", ""),
                "subset_method": subset_info.get("subset_method", ""),
                "subset_path": subset_info.get("subset_path", ""),
                "subset_source": subset_info.get("subset_source", ""),
                "params_path": subset_info.get("params_path", ""),
                "signals_total": int(met.get("overall_signals_total", 0)),
                "entries_valid": float(met.get("overall_entries_valid", np.nan)),
                "entry_rate": float(met.get("overall_entry_rate", np.nan)),
                "exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
                "delta_expectancy_exec_minus_baseline": float(met.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
                "cvar_improve_ratio": float(met.get("overall_cvar_improve_ratio", np.nan)),
                "maxdd_improve_ratio": float(met.get("overall_maxdd_improve_ratio", np.nan)),
                "exec_cvar_5": float(met.get("overall_exec_cvar_5", np.nan)),
                "exec_max_drawdown": float(met.get("overall_exec_max_drawdown", np.nan)),
                "taker_share": float(met.get("overall_exec_taker_share", np.nan)),
                "median_fill_delay_min": float(met.get("overall_exec_median_fill_delay_min", np.nan)),
                "p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
                "min_split_expectancy_net": float(met.get("min_split_expectancy_net", np.nan)),
                "median_split_expectancy_net": float(met.get("median_split_expectancy_net", np.nan)),
                "std_split_expectancy_net": float(met.get("std_split_expectancy_net", np.nan)),
                "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
                "hard_invalid": int(met.get("hard_invalid", 1)),
                "constraint_pass": int(met.get("constraint_pass", 1)),
                "participation_pass": int(met.get("participation_pass", 0)),
                "realism_pass": int(met.get("realism_pass", 0)),
                "nan_pass": int(met.get("nan_pass", 0)),
                "split_pass": int(met.get("split_pass", 0)),
                "invalid_reason": str(met.get("invalid_reason", "")),
                "participation_fail_reason": str(met.get("participation_fail_reason", "")),
                "realism_fail_reason": str(met.get("realism_fail_reason", "")),
                "split_fail_reason": str(met.get("split_fail_reason", "")),
                "missing_slice_rate": float(met.get("overall_missing_slice_rate", np.nan)),
                "data_quality_pass": int(met.get("data_quality_pass", 1)),
                "eval_time_sec": float(met.get("eval_time_sec", np.nan)),
            }
        )

    pdf = pd.DataFrame(rows)
    return pdf, load_meta, bundle, args


def detailed_eval_scenario(
    scenario_id: str,
    source_tag: str,
    bundle: ga_exec.SymbolBundle,
    args: argparse.Namespace,
    genome: Dict[str, Any],
    genome_hash: str,
    rep_feat_map: pd.DataFrame,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    met = ga_exec._evaluate_genome(genome=genome, bundles=[bundle], args=args, detailed=True)
    sig = met["signal_rows_df"].copy()
    spl = met["split_rows_df"].copy()

    sig["signal_id"] = sig["signal_id"].astype(str)
    sig["signal_time"] = pd.to_datetime(sig["signal_time"], utc=True, errors="coerce")
    sig["exec_pnl_net_pct"] = pd.to_numeric(sig.get("exec_pnl_net_pct"), errors="coerce")
    sig["exec_pnl_gross_pct"] = pd.to_numeric(sig.get("exec_pnl_gross_pct"), errors="coerce")
    sig["exec_filled"] = pd.to_numeric(sig.get("exec_filled"), errors="coerce").fillna(0).astype(int)
    sig["exec_valid_for_metrics"] = pd.to_numeric(sig.get("exec_valid_for_metrics"), errors="coerce").fillna(0).astype(int)
    sig["exec_fill_delay_min"] = pd.to_numeric(sig.get("exec_fill_delay_min"), errors="coerce")
    sig["exec_mae_pct"] = pd.to_numeric(sig.get("exec_mae_pct"), errors="coerce")
    sig["entry_improvement_bps"] = pd.to_numeric(sig.get("entry_improvement_bps"), errors="coerce")

    sig = sig.merge(rep_feat_map, on="signal_id", how="left")
    sig["session_bucket"] = session_bucket(sig["signal_time"])
    if "vol_bucket" not in sig.columns:
        sig["vol_bucket"] = vol_bucket(sig.get("atr_percentile_1h"))
    if "trend_bucket" not in sig.columns:
        sig["trend_bucket"] = np.where(pd.to_numeric(sig.get("trend_up_1h"), errors="coerce") >= 0.5, "up", "down")
    sig["regime_bucket"] = sig["vol_bucket"].astype(str) + "|" + sig["trend_bucket"].astype(str)

    vmask = (sig["exec_filled"] == 1) & (sig["exec_valid_for_metrics"] == 1) & sig["exec_pnl_net_pct"].notna()
    v = sig[vmask].copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)

    pnl_sig = np.zeros(len(sig), dtype=float)
    idx = np.where(vmask.to_numpy(dtype=bool))[0]
    pnl_sig[idx] = sig.loc[vmask, "exec_pnl_net_pct"].to_numpy(dtype=float)

    neg = v[v["exec_pnl_net_pct"] < 0].copy()
    total_loss_abs = float(np.abs(neg["exec_pnl_net_pct"]).sum()) if not neg.empty else 0.0

    split_worst = None
    if not spl.empty and "exec_mean_expectancy_net" in spl.columns:
        t = spl.copy()
        t["exec_mean_expectancy_net"] = pd.to_numeric(t["exec_mean_expectancy_net"], errors="coerce")
        t = t.sort_values("exec_mean_expectancy_net", ascending=True)
        if not t.empty:
            split_worst = t.iloc[0]

    session_worst = None
    regime_worst = None
    if not v.empty:
        g1 = v.groupby("session_bucket", dropna=False)["exec_pnl_net_pct"].sum().reset_index().sort_values("exec_pnl_net_pct")
        if not g1.empty:
            session_worst = g1.iloc[0]
        g2 = v.groupby("regime_bucket", dropna=False)["exec_pnl_net_pct"].sum().reset_index().sort_values("exec_pnl_net_pct")
        if not g2.empty:
            regime_worst = g2.iloc[0]

    # loss run stats
    runs_ge3 = 0
    cur = 0
    for x in v["exec_pnl_net_pct"].to_numpy(dtype=float):
        if np.isfinite(x) and x < 0:
            cur += 1
        else:
            if cur >= 3:
                runs_ge3 += 1
            cur = 0
    if cur >= 3:
        runs_ge3 += 1

    sl_loss_share = float("nan")
    if total_loss_abs > 1e-12:
        sl_loss = np.abs(neg[neg["exec_exit_reason"].astype(str).str.lower() == "sl"]["exec_pnl_net_pct"]).sum()
        sl_loss_share = float(sl_loss / total_loss_abs)

    fee_drag_total = float((v["exec_pnl_gross_pct"] - v["exec_pnl_net_pct"]).sum()) if not v.empty else float("nan")
    fee_drag_per_trade = float((v["exec_pnl_gross_pct"] - v["exec_pnl_net_pct"]).mean()) if not v.empty else float("nan")
    gross_sum = float(v["exec_pnl_gross_pct"].sum()) if not v.empty else float("nan")
    fee_drag_to_gross_abs_ratio = safe_div(fee_drag_total, abs(gross_sum)) if np.isfinite(gross_sum) else float("nan")

    out = {
        "scenario_id": scenario_id,
        "source_tag": source_tag,
        "genome_hash": genome_hash,
        "signals_total": int(met.get("overall_signals_total", len(sig))),
        "entries_valid": int(met.get("overall_entries_valid", 0)),
        "entry_rate": float(met.get("overall_entry_rate", np.nan)),
        "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
        "exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
        "exec_delta_expectancy_vs_baseline": float(met.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
        "exec_cvar_5": float(met.get("overall_exec_cvar_5", np.nan)),
        "exec_max_drawdown": float(met.get("overall_exec_max_drawdown", np.nan)),
        "cvar_improve_ratio": float(met.get("overall_cvar_improve_ratio", np.nan)),
        "maxdd_improve_ratio": float(met.get("overall_maxdd_improve_ratio", np.nan)),
        "taker_share": float(met.get("overall_exec_taker_share", np.nan)),
        "median_fill_delay_min": float(met.get("overall_exec_median_fill_delay_min", np.nan)),
        "p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
        "min_split_expectancy_net": float(met.get("min_split_expectancy_net", np.nan)),
        "median_split_expectancy_net": float(met.get("median_split_expectancy_net", np.nan)),
        "std_split_expectancy_net": float(met.get("std_split_expectancy_net", np.nan)),
        "worst_split_id": int(split_worst["split_id"]) if split_worst is not None and np.isfinite(split_worst.get("split_id", np.nan)) else -1,
        "worst_split_expectancy": float(split_worst["exec_mean_expectancy_net"]) if split_worst is not None else float("nan"),
        "worst_session_bucket": str(session_worst["session_bucket"]) if session_worst is not None else "",
        "worst_session_pnl_sum": float(session_worst["exec_pnl_net_pct"]) if session_worst is not None else float("nan"),
        "worst_regime_bucket": str(regime_worst["regime_bucket"]) if regime_worst is not None else "",
        "worst_regime_pnl_sum": float(regime_worst["exec_pnl_net_pct"]) if regime_worst is not None else float("nan"),
        "max_consecutive_losses": int(max_consecutive_losses(v["exec_pnl_net_pct"].to_numpy(dtype=float))) if not v.empty else 0,
        "loss_run_ge3_count": int(runs_ge3),
        "sl_loss_share": float(sl_loss_share),
        "fee_drag_total": float(fee_drag_total),
        "fee_drag_per_trade": float(fee_drag_per_trade),
        "fee_drag_to_gross_abs_ratio": float(fee_drag_to_gross_abs_ratio),
        "gross_expectancy_valid": float(v["exec_pnl_gross_pct"].mean()) if not v.empty else float("nan"),
        "net_expectancy_valid": float(v["exec_pnl_net_pct"].mean()) if not v.empty else float("nan"),
        "mae_mean_loss": float(neg["exec_mae_pct"].mean()) if not neg.empty else float("nan"),
        "mae_p95_loss": float(neg["exec_mae_pct"].quantile(0.95)) if not neg.empty else float("nan"),
        "hard_invalid": int(met.get("hard_invalid", 1)),
        "invalid_reason": str(met.get("invalid_reason", "")),
        "participation_fail_reason": str(met.get("participation_fail_reason", "")),
        "realism_fail_reason": str(met.get("realism_fail_reason", "")),
    }
    return out, sig, spl


def overlay_metrics_from_signal_df(
    sig: pd.DataFrame,
    args: argparse.Namespace,
    overlay_id: str,
    overlay_desc: str,
    keep_mask: np.ndarray,
    base_row: Dict[str, Any],
) -> Dict[str, Any]:
    x = sig.copy().reset_index(drop=True)
    valid_entry = (
        pd.to_numeric(x.get("exec_filled"), errors="coerce").fillna(0).astype(int) == 1
    ) & (
        pd.to_numeric(x.get("exec_valid_for_metrics"), errors="coerce").fillna(0).astype(int) == 1
    ) & pd.to_numeric(x.get("exec_pnl_net_pct"), errors="coerce").notna()

    keep = np.asarray(keep_mask, dtype=bool)
    if keep.size != len(x):
        keep = np.ones(len(x), dtype=bool)

    eff_mask = valid_entry.to_numpy(dtype=bool) & keep
    pnl = np.zeros(len(x), dtype=float)
    pnl[eff_mask] = pd.to_numeric(x.loc[eff_mask, "exec_pnl_net_pct"], errors="coerce").to_numpy(dtype=float)

    signals_total = int(len(x))
    entries = int(eff_mask.sum())
    entry_rate = float(entries / max(1, signals_total))
    exp = float(np.mean(pnl))
    cvar5 = float(tail_mean(pnl, 0.05))
    maxdd = float(max_drawdown_from_pnl(pnl))

    taker = pd.to_numeric((x.get("exec_fill_liquidity_type", "").astype(str).str.lower() == "taker").astype(int), errors="coerce").fillna(0)
    taker_share = float(taker[eff_mask].mean()) if entries > 0 else float("nan")

    delay = pd.to_numeric(x.get("exec_fill_delay_min"), errors="coerce")
    d = delay[eff_mask & delay.notna().to_numpy(dtype=bool)]
    med_delay = float(d.median()) if len(d) else float("nan")
    p95_delay = float(d.quantile(0.95)) if len(d) else float("nan")

    min_trades_overall = max(int(args.hard_min_trades_overall), int(math.ceil(float(args.hard_min_trade_frac_overall) * max(1, signals_total))))
    participation_pass = int(entries >= min_trades_overall and np.isfinite(entry_rate) and entry_rate >= float(args.hard_min_entry_rate_overall))
    realism_pass = int(
        np.isfinite(taker_share)
        and np.isfinite(med_delay)
        and np.isfinite(p95_delay)
        and taker_share <= float(args.hard_max_taker_share)
        and med_delay <= float(args.hard_max_median_fill_delay_min)
        and p95_delay <= float(args.hard_max_p95_fill_delay_min)
    )

    removed_entries = int(valid_entry.sum() - entries)
    removed_pct = float(removed_entries / max(1, int(valid_entry.sum())))

    neg_orig = pd.to_numeric(x.loc[valid_entry, "exec_pnl_net_pct"], errors="coerce")
    neg_orig = neg_orig[neg_orig < 0]
    total_loss_abs = float(np.abs(neg_orig).sum()) if len(neg_orig) else 0.0
    removed_loss_abs = 0.0
    if removed_entries > 0 and total_loss_abs > 1e-12:
        rneg = pd.to_numeric(x.loc[valid_entry & (~keep), "exec_pnl_net_pct"], errors="coerce")
        rneg = rneg[rneg < 0]
        removed_loss_abs = float(np.abs(rneg).sum())
    removed_loss_share = float(removed_loss_abs / total_loss_abs) if total_loss_abs > 1e-12 else float("nan")

    base_exp = float(base_row.get("exec_expectancy_net", np.nan))
    base_cvar = float(base_row.get("exec_cvar_5", np.nan))
    base_dd = float(base_row.get("exec_max_drawdown", np.nan))

    out = {
        "scenario_id": str(base_row.get("scenario_id", "")),
        "overlay_id": overlay_id,
        "overlay_desc": overlay_desc,
        "approximate_counterfactual": 1,
        "signals_total": signals_total,
        "entries_valid": entries,
        "entry_rate": entry_rate,
        "exec_expectancy_net": exp,
        "exec_cvar_5": cvar5,
        "exec_max_drawdown": maxdd,
        "taker_share": taker_share,
        "median_fill_delay_min": med_delay,
        "p95_fill_delay_min": p95_delay,
        "removed_entries_count": removed_entries,
        "removed_entries_pct": removed_pct,
        "removed_loss_share_abs": removed_loss_share,
        "delta_expectancy_vs_base": float(exp - base_exp) if np.isfinite(exp) and np.isfinite(base_exp) else float("nan"),
        "delta_cvar_vs_base": float(cvar5 - base_cvar) if np.isfinite(cvar5) and np.isfinite(base_cvar) else float("nan"),
        "delta_maxdd_vs_base": float(maxdd - base_dd) if np.isfinite(maxdd) and np.isfinite(base_dd) else float("nan"),
        "cvar_improve_ratio_vs_base": safe_div(abs(base_cvar) - abs(cvar5), abs(base_cvar)) if np.isfinite(base_cvar) else float("nan"),
        "maxdd_improve_ratio_vs_base": safe_div(abs(base_dd) - abs(maxdd), abs(base_dd)) if np.isfinite(base_dd) else float("nan"),
        "participation_pass_proxy": participation_pass,
        "realism_pass_proxy": realism_pass,
        "hard_gate_proxy_pass": int(participation_pass == 1 and realism_pass == 1),
    }
    return out


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    exec_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_dir = exec_root / f"PHASEV_BRANCHB_PORTABILITY_DD_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    subset_dir = run_dir / "subsets"
    subset_dir.mkdir(exist_ok=True)

    manifest: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "phase": "V",
        "run_dir": str(run_dir),
        "project_root": str(PROJECT_ROOT),
        "commands": [{"cmd": "python -m scripts.phase_v_branchb_portability_dd", "utc": utc_now()}],
        "code_modified": "YES (new script phase_v_branchb_portability_dd.py)",
    }

    invalid_hist: Counter[str] = Counter()

    # V1 lock checks + candidate extraction
    fee_path = Path(LOCKED["canonical_fee_model"]).resolve()
    metrics_path = Path(LOCKED["canonical_metrics_definition"]).resolve()
    rep_sol_path = Path(LOCKED["representative_subset_csv"]).resolve()

    for fp in (fee_path, metrics_path, rep_sol_path):
        if not fp.exists():
            raise FileNotFoundError(f"Missing required locked file: {fp}")

    fee_sha = sha256_file(fee_path)
    metrics_sha = sha256_file(metrics_path)
    freeze_hash_pass = int(fee_sha == LOCKED["expected_fee_sha"] and metrics_sha == LOCKED["expected_metrics_sha"])

    exec_candidates, exec_meta = load_execution_candidates(exec_root)
    exec_locked_out = {
        "generated_utc": utc_now(),
        "source_phase_u": exec_meta,
        "candidates": {
            k: {
                "exec_choice_id": v["exec_choice_id"],
                "description": v["description"],
                "genome_hash": v["genome_hash"],
                "source_run": v["source_run"],
                "genome": v["genome"],
            }
            for k, v in exec_candidates.items()
        },
    }
    json_dump(run_dir / "phaseV_exec_candidates_locked.json", exec_locked_out)

    # validate lock via ga_exec contract validator too
    args_lock = build_exec_args(symbol="SOLUSDT", signals_csv=rep_sol_path, max_signals=1200, seed=args.seed)
    lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=args_lock, run_dir=run_dir)

    contract_md: List[str] = []
    contract_md.append("# Phase V Contract Validation")
    contract_md.append("")
    contract_md.append(f"- Generated UTC: {utc_now()}")
    contract_md.append(f"- canonical fee path: `{fee_path}`")
    contract_md.append(f"- canonical metrics path: `{metrics_path}`")
    contract_md.append(f"- fee sha256: `{fee_sha}` (match={int(fee_sha == LOCKED['expected_fee_sha'])})")
    contract_md.append(f"- metrics sha256: `{metrics_sha}` (match={int(metrics_sha == LOCKED['expected_metrics_sha'])})")
    contract_md.append(f"- ga_exec freeze lock pass: `{int(lock_validation.get('freeze_lock_pass', 0))}`")
    contract_md.append("")
    contract_md.append("## Execution Candidates")
    contract_md.append("")
    for k in ("E1", "E2", "E4"):
        contract_md.append(
            f"- {k}: hash=`{exec_candidates[k]['genome_hash']}`, source=`{exec_candidates[k]['source_run']}`, desc=`{exec_candidates[k]['description']}`"
        )
    write_text(run_dir / "phaseV_contract_validation.md", "\n".join(contract_md))

    # Identify best params files
    best_scan_csv = PROJECT_ROOT / "reports" / "params_scan" / "20260220_044949" / "best_by_symbol.csv"
    params_by_symbol: Dict[str, Optional[Path]] = {}
    for sym in ["SOLUSDT"] + PORTABILITY_CANDIDATES:
        params_by_symbol[sym] = find_best_params_for_symbol(sym, best_scan_csv)

    # V2 portability matrix (SOL + other coins)
    portability_rows: List[Dict[str, Any]] = []
    portability_meta: Dict[str, Any] = {"coins": {}, "selected_coins": []}
    eval_context: Dict[str, Any] = {}

    target_coins: List[str] = ["SOLUSDT"]
    for sym in PORTABILITY_CANDIDATES:
        if params_by_symbol.get(sym) is not None:
            target_coins.append(sym)
    target_coins = target_coins[:4]  # SOL + up to 3 coins

    sol_bundle: Optional[ga_exec.SymbolBundle] = None
    sol_args: Optional[argparse.Namespace] = None

    for sym in target_coins:
        subset_info: Dict[str, Any]
        if sym == "SOLUSDT":
            sdf = pd.read_csv(rep_sol_path)
            sdf["signal_time"] = pd.to_datetime(sdf["signal_time"], utc=True, errors="coerce")
            sdf = sdf.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
            sdf["signal_id"] = sdf["signal_id"].astype(str)
            sdf["tp_mult"] = pd.to_numeric(sdf.get("tp_mult"), errors="coerce")
            sdf["sl_mult"] = pd.to_numeric(sdf.get("sl_mult"), errors="coerce")
            sol_subset_fp = subset_dir / "SOLUSDT_subset_signals.csv"
            sdf[["signal_id", "signal_time", "tp_mult", "sl_mult", "atr_percentile_1h", "trend_up_1h"]].to_csv(sol_subset_fp, index=False)
            subset_info = {
                "subset_method": "frozen_representative_subset",
                "subset_path": str(sol_subset_fp),
                "subset_source": str(rep_sol_path),
                "subset_hash": subset_hash(sdf[["signal_id", "signal_time", "tp_mult", "sl_mult"]].copy()),
                "params_path": str(params_by_symbol.get(sym) or ""),
                "signals_count": int(len(sdf)),
            }
            args_coin = build_exec_args(symbol=sym, signals_csv=sol_subset_fp, max_signals=1200, seed=args.seed)
        else:
            ppath = params_by_symbol.get(sym)
            if ppath is None:
                portability_meta["coins"][sym] = {"status": "skip_no_params"}
                continue
            try:
                sdf = generate_signals_from_params(sym, ppath, max_signals=int(args.portability_max_signals))
            except Exception as exc:
                portability_meta["coins"][sym] = {"status": "skip_signal_gen_error", "error": f"{type(exc).__name__}: {exc}"}
                continue
            if sdf.empty:
                portability_meta["coins"][sym] = {"status": "skip_empty_signals"}
                continue
            fp = subset_dir / f"{sym}_subset_signals.csv"
            sdf[["signal_id", "signal_time", "tp_mult", "sl_mult", "cycle"]].to_csv(fp, index=False)
            subset_info = {
                "subset_method": "generated_from_best_params_latestN",
                "subset_path": str(fp),
                "subset_source": f"best_params:{ppath}",
                "subset_hash": subset_hash(sdf[["signal_id", "signal_time", "tp_mult", "sl_mult"]].copy()),
                "params_path": str(ppath),
                "signals_count": int(len(sdf)),
            }
            args_coin = build_exec_args(symbol=sym, signals_csv=fp, max_signals=int(args.portability_max_signals), seed=args.seed)

        try:
            pdf, load_meta, bundle, args_used = eval_portability_coin(
                symbol=sym,
                signals_csv=Path(subset_info["subset_path"]),
                subset_info=subset_info,
                exec_candidates=exec_candidates,
                seed=args.seed,
            )
        except Exception as exc:
            portability_meta["coins"][sym] = {
                "status": "eval_error",
                "subset": subset_info,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue

        portability_rows.extend(pdf.to_dict(orient="records"))
        portability_meta["coins"][sym] = {
            "status": "ok",
            "subset": subset_info,
            "signals_loaded": int(load_meta.get("bundle_sizes", {}).get(sym, {}).get("signals", len(bundle.contexts))),
            "splits": int(len(bundle.splits)),
        }
        portability_meta["selected_coins"].append(sym)

        eval_context[sym] = {"bundle": bundle, "args": args_used, "subset_info": subset_info}
        if sym == "SOLUSDT":
            sol_bundle = bundle
            sol_args = args_used

    portability_df = pd.DataFrame(portability_rows)
    portability_df.to_csv(run_dir / "phaseV_portability_matrix.csv", index=False)

    # per-coin best + runner-up
    summary_rows: List[Dict[str, Any]] = []
    if not portability_df.empty:
        for sym, grp in portability_df.groupby("symbol", dropna=False):
            z = grp.copy()
            z["valid_for_ranking"] = to_num(z["valid_for_ranking"]).fillna(0)
            z["exec_expectancy_net"] = to_num(z["exec_expectancy_net"])
            z["delta_expectancy_exec_minus_baseline"] = to_num(z["delta_expectancy_exec_minus_baseline"])
            z["cvar_improve_ratio"] = to_num(z["cvar_improve_ratio"])
            z["maxdd_improve_ratio"] = to_num(z["maxdd_improve_ratio"])
            z = z.sort_values(
                ["valid_for_ranking", "delta_expectancy_exec_minus_baseline", "exec_expectancy_net", "cvar_improve_ratio", "maxdd_improve_ratio"],
                ascending=[False, False, False, False, False],
            ).reset_index(drop=True)
            if not z.empty:
                top = z.iloc[0]
                summary_rows.append(
                    {
                        "symbol": sym,
                        "rank": 1,
                        "exec_choice_id": str(top["exec_choice_id"]),
                        "valid_for_ranking": int(top["valid_for_ranking"]),
                        "exec_expectancy_net": float(top["exec_expectancy_net"]),
                        "delta_vs_e0": float(top["delta_expectancy_exec_minus_baseline"]),
                        "cvar_improve_ratio": float(top["cvar_improve_ratio"]),
                        "maxdd_improve_ratio": float(top["maxdd_improve_ratio"]),
                        "entry_rate": float(top["entry_rate"]),
                        "entries_valid": float(top["entries_valid"]),
                        "taker_share": float(top["taker_share"]),
                        "p95_fill_delay_min": float(top["p95_fill_delay_min"]),
                        "invalid_reason": str(top["invalid_reason"]),
                    }
                )
            if len(z) > 1:
                top2 = z.iloc[1]
                summary_rows.append(
                    {
                        "symbol": sym,
                        "rank": 2,
                        "exec_choice_id": str(top2["exec_choice_id"]),
                        "valid_for_ranking": int(top2["valid_for_ranking"]),
                        "exec_expectancy_net": float(top2["exec_expectancy_net"]),
                        "delta_vs_e0": float(top2["delta_expectancy_exec_minus_baseline"]),
                        "cvar_improve_ratio": float(top2["cvar_improve_ratio"]),
                        "maxdd_improve_ratio": float(top2["maxdd_improve_ratio"]),
                        "entry_rate": float(top2["entry_rate"]),
                        "entries_valid": float(top2["entries_valid"]),
                        "taker_share": float(top2["taker_share"]),
                        "p95_fill_delay_min": float(top2["p95_fill_delay_min"]),
                        "invalid_reason": str(top2["invalid_reason"]),
                    }
                )

    portability_summary_df = pd.DataFrame(summary_rows)

    pmd: List[str] = []
    pmd.append("# Phase V Portability Summary")
    pmd.append("")
    pmd.append(f"- Generated UTC: {utc_now()}")
    pmd.append(f"- Coins evaluated: {', '.join(portability_meta.get('selected_coins', [])) if portability_meta.get('selected_coins') else 'none'}")
    pmd.append(f"- Matrix rows: {len(portability_df)}")
    pmd.append("")

    if not portability_df.empty:
        x = portability_df.copy()
        x["valid_for_ranking"] = to_num(x["valid_for_ranking"]).fillna(0).astype(int)
        x["delta_expectancy_exec_minus_baseline"] = to_num(x["delta_expectancy_exec_minus_baseline"])
        x["cvar_improve_ratio"] = to_num(x["cvar_improve_ratio"])
        x["maxdd_improve_ratio"] = to_num(x["maxdd_improve_ratio"])

        pmd.append("## Retention vs Baseline")
        pmd.append("")
        for sym, grp in x.groupby("symbol", dropna=False):
            non_e0 = grp[grp["exec_choice_id"] != "E0"].copy()
            if non_e0.empty:
                pmd.append(f"- {sym}: no non-baseline execution choices evaluated.")
                continue
            best = non_e0.sort_values(
                ["valid_for_ranking", "delta_expectancy_exec_minus_baseline", "cvar_improve_ratio", "maxdd_improve_ratio"],
                ascending=[False, False, False, False],
            ).iloc[0]
            pmd.append(
                f"- {sym}: best={best['exec_choice_id']} valid={int(best['valid_for_ranking'])} delta_vs_E0={float(best['delta_expectancy_exec_minus_baseline']):.8f} cvar_improve={float(best['cvar_improve_ratio']):.6f} maxdd_improve={float(best['maxdd_improve_ratio']):.6f}"
            )

        pmd.append("")
        pmd.append("## Best/Runner-Up Per Coin")
        pmd.append("")
        if not portability_summary_df.empty:
            pmd.append(df_to_plain_table(portability_summary_df))
        else:
            pmd.append("_(none)_")
    else:
        pmd.append("No portability rows were produced.")

    write_text(run_dir / "phaseV_portability_summary.md", "\n".join(pmd))

    # V3 SOL DD forensics + overlays
    if sol_bundle is None or sol_args is None:
        raise RuntimeError("SOL bundle unavailable for V3")

    rep_sol = pd.read_csv(rep_sol_path)
    rep_sol["signal_id"] = rep_sol["signal_id"].astype(str)
    rep_sol["atr_percentile_1h"] = pd.to_numeric(rep_sol.get("atr_percentile_1h"), errors="coerce")
    rep_sol["trend_up_1h"] = pd.to_numeric(rep_sol.get("trend_up_1h"), errors="coerce")
    rep_sol["vol_bucket"] = vol_bucket(rep_sol["atr_percentile_1h"])
    rep_sol["trend_bucket"] = np.where(rep_sol["trend_up_1h"] >= 0.5, "up", "down")
    rep_feat_map = rep_sol[["signal_id", "atr_percentile_1h", "trend_up_1h", "vol_bucket", "trend_bucket"]].drop_duplicates("signal_id")

    scenarios: List[Tuple[str, str, ga_exec.SymbolBundle, Dict[str, Any], str]] = []
    scenarios.append(("SOL_BASE_E1", "phaseQRS_survivor", sol_bundle, exec_candidates["E1"]["genome"], exec_candidates["E1"]["genome_hash"]))
    scenarios.append(("SOL_BASE_E2", "phaseQRS_survivor", sol_bundle, exec_candidates["E2"]["genome"], exec_candidates["E2"]["genome_hash"]))

    # Add Phase U top signal candidate if different from base
    phase_u_dir = Path(exec_meta["phase_u_dir"])
    pu_short = phase_u_dir / "phaseU_shortlist_significance.csv"
    if pu_short.exists():
        su = pd.read_csv(pu_short)
        if not su.empty:
            su["valid_for_ranking"] = to_num(su.get("valid_for_ranking", 0)).fillna(0)
            su = su[(su["valid_for_ranking"] == 1) & (su["exec_choice_id"].astype(str) != "E0")].copy()
            su = su.sort_values(
                ["valid_for_ranking", "equity_growth_proxy", "overall_exec_expectancy_net"],
                ascending=[False, False, False],
            )
            if not su.empty:
                top = su.iloc[0]
                sid = str(top.get("signal_candidate_id", ""))
                exid = str(top.get("exec_choice_id", "E1"))
                if sid and sid != "P00" and exid in exec_candidates:
                    try:
                        ppath, base_params_raw, _ = phaseu.load_active_sol_params()
                        base_params = ga_long._norm_params(copy.deepcopy(base_params_raw))
                        ccount = int(load_json_safe(phase_u_dir / "phaseU_run_manifest.json", {}).get("candidate_budget", {}).get("signal_candidates", 25))
                        seed_u = 20260226
                        cands = phaseu.generate_1h_candidates(base_params=base_params, n_total=ccount, seed=seed_u)
                        cand_map = {str(c["signal_candidate_id"]): c for c in cands}
                        if sid in cand_map:
                            df1h = recon._load_symbol_df("SOLUSDT", tf="1h")
                            feat = ga_long._ensure_indicators(df1h.copy(), base_params)
                            feat = ga_long._prepare_signal_df(feat, assume_prepared=False)
                            rep_idx = phaseu.build_rep_subset_with_idx(rep_subset=rep_sol.copy(), df_feat=feat)
                            active_ids, _diag = phaseu.active_signal_ids_for_params(df_feat=feat, params=cand_map[sid]["params"], rep_idx=rep_idx)
                            bun = phaseu.build_candidate_bundle(base_bundle=sol_bundle, active_ids=active_ids, args=sol_args)
                            scenarios.append((f"SOL_{sid}_{exid}", "phaseU_top", bun, exec_candidates[exid]["genome"], exec_candidates[exid]["genome_hash"]))
                    except Exception:
                        pass

    dd_rows: List[Dict[str, Any]] = []
    signal_frames: Dict[str, pd.DataFrame] = {}
    split_frames: Dict[str, pd.DataFrame] = {}

    for sid, stg, bun, genome, ghash in scenarios:
        row, sig_df, split_df = detailed_eval_scenario(
            scenario_id=sid,
            source_tag=stg,
            bundle=bun,
            args=sol_args,
            genome=genome,
            genome_hash=ghash,
            rep_feat_map=rep_feat_map,
        )
        dd_rows.append(row)
        signal_frames[sid] = sig_df
        split_frames[sid] = split_df

    dd_df = pd.DataFrame(dd_rows)
    dd_df.to_csv(run_dir / "phaseV_sol_dd_forensics.csv", index=False)

    # Overlay micro-benchmark on top 2 scenario rows by validity/expectancy
    over_rows: List[Dict[str, Any]] = []
    if not dd_df.empty:
        dz = dd_df.copy()
        dz["valid_for_ranking"] = to_num(dz["valid_for_ranking"]).fillna(0)
        dz["exec_expectancy_net"] = to_num(dz["exec_expectancy_net"])
        dz = dz.sort_values(["valid_for_ranking", "exec_expectancy_net"], ascending=[False, False]).reset_index(drop=True)
        overlay_targets = dz["scenario_id"].head(min(2, len(dz))).tolist()

        for sid in overlay_targets:
            sig = signal_frames[sid].copy().reset_index(drop=True)
            vmask = (
                (to_num(sig.get("exec_filled", 0)).fillna(0).astype(int) == 1)
                & (to_num(sig.get("exec_valid_for_metrics", 0)).fillna(0).astype(int) == 1)
                & to_num(sig.get("exec_pnl_net_pct", np.nan)).notna()
            )
            base_row = dd_df[dd_df["scenario_id"] == sid].iloc[0].to_dict()

            # O0 baseline (no overlay)
            keep0 = np.ones(len(sig), dtype=bool)
            over_rows.append(overlay_metrics_from_signal_df(sig=sig, args=sol_args, overlay_id="O0", overlay_desc="no_overlay", keep_mask=keep0, base_row=base_row))

            # O1 worst session veto
            sess = session_bucket(sig["signal_time"])
            if vmask.any():
                ssum = pd.DataFrame({"session": sess[vmask], "pnl": to_num(sig.loc[vmask, "exec_pnl_net_pct"])}).groupby("session")["pnl"].sum()
                worst_sess = str(ssum.sort_values(ascending=True).index[0]) if not ssum.empty else ""
            else:
                worst_sess = ""
            keep1 = np.ones(len(sig), dtype=bool)
            if worst_sess:
                keep1 = ~(sess.astype(str).to_numpy() == worst_sess)
            over_rows.append(
                overlay_metrics_from_signal_df(
                    sig=sig,
                    args=sol_args,
                    overlay_id="O1",
                    overlay_desc=f"session_veto_worst:{worst_sess or 'none'}",
                    keep_mask=keep1,
                    base_row=base_row,
                )
            )

            # O2 daily loss cap
            keep2 = np.ones(len(sig), dtype=bool)
            y = sig.copy().reset_index(drop=True)
            y["signal_time"] = pd.to_datetime(y["signal_time"], utc=True, errors="coerce")
            y["day"] = y["signal_time"].dt.floor("D")
            y["pnl"] = to_num(y.get("exec_pnl_net_pct", np.nan))
            neg_day = y[vmask].groupby("day")["pnl"].sum()
            neg_day = neg_day[neg_day < 0]
            cap = float(neg_day.quantile(0.5)) if len(neg_day) else -0.002
            cap = min(-0.0005, float(cap))
            for d, grp in y.groupby("day", dropna=False):
                if pd.isna(d):
                    continue
                cum = 0.0
                stop = False
                for i in grp.index.tolist():
                    if not bool(vmask.iloc[i]):
                        continue
                    if stop:
                        keep2[i] = False
                        continue
                    pi = float(y.loc[i, "pnl"]) if np.isfinite(y.loc[i, "pnl"]) else 0.0
                    cum += pi
                    if cum <= cap:
                        stop = True
            over_rows.append(
                overlay_metrics_from_signal_df(
                    sig=sig,
                    args=sol_args,
                    overlay_id="O2",
                    overlay_desc=f"daily_loss_cap:{cap:.6f}",
                    keep_mask=keep2,
                    base_row=base_row,
                )
            )

            # O3 volatility-spike veto
            atr = to_num(sig.get("atr_percentile_1h", np.nan))
            keep3 = np.ones(len(sig), dtype=bool)
            keep3[np.where(atr >= 85.0)[0]] = False
            over_rows.append(
                overlay_metrics_from_signal_df(
                    sig=sig,
                    args=sol_args,
                    overlay_id="O3",
                    overlay_desc="volatility_spike_veto_atr_pct>=85",
                    keep_mask=keep3,
                    base_row=base_row,
                )
            )

            # O4 entry-improvement filter
            imp = to_num(sig.get("entry_improvement_bps", np.nan)).fillna(-9999.0)
            keep4 = (imp >= 0.0).to_numpy(dtype=bool)
            over_rows.append(
                overlay_metrics_from_signal_df(
                    sig=sig,
                    args=sol_args,
                    overlay_id="O4",
                    overlay_desc="entry_improvement_bps>=0",
                    keep_mask=keep4,
                    base_row=base_row,
                )
            )

    overlay_df = pd.DataFrame(over_rows)
    overlay_df.to_csv(run_dir / "phaseV_risk_overlay_benchmark.csv", index=False)

    # invalid reason histogram
    if not portability_df.empty:
        for s in portability_df["invalid_reason"].fillna("").astype(str):
            if not s.strip():
                continue
            for part in [p for p in s.split("|") if p]:
                invalid_hist[part] += 1
    if not dd_df.empty:
        for s in dd_df["invalid_reason"].fillna("").astype(str):
            if not s.strip():
                continue
            for part in [p for p in s.split("|") if p]:
                invalid_hist[part] += 1
    json_dump(run_dir / "phaseV_invalid_reason_histogram.json", dict(sorted(invalid_hist.items(), key=lambda kv: (-kv[1], kv[0]))))

    # V4 decision
    # Portability score on non-SOL coins
    portability_go_coins: List[str] = []
    portability_feasible_weak: List[str] = []
    if not portability_df.empty:
        px = portability_df.copy()
        px["valid_for_ranking"] = to_num(px["valid_for_ranking"]).fillna(0)
        px["delta_expectancy_exec_minus_baseline"] = to_num(px["delta_expectancy_exec_minus_baseline"])
        px["cvar_improve_ratio"] = to_num(px["cvar_improve_ratio"])
        px["maxdd_improve_ratio"] = to_num(px["maxdd_improve_ratio"])
        for sym, grp in px.groupby("symbol", dropna=False):
            if sym == "SOLUSDT":
                continue
            g = grp[grp["exec_choice_id"] != "E0"].copy()
            if g.empty:
                continue
            g = g.sort_values(
                ["valid_for_ranking", "delta_expectancy_exec_minus_baseline", "cvar_improve_ratio", "maxdd_improve_ratio"],
                ascending=[False, False, False, False],
            )
            top = g.iloc[0]
            if int(top["valid_for_ranking"]) == 1 and float(top["delta_expectancy_exec_minus_baseline"]) > 0 and (
                float(top["cvar_improve_ratio"]) > 0 or float(top["maxdd_improve_ratio"]) > 0
            ):
                portability_go_coins.append(str(sym))
            elif int(top["valid_for_ranking"]) == 1:
                portability_feasible_weak.append(str(sym))

    # Overlay materiality
    overlay_material = False
    best_overlay_row: Optional[pd.Series] = None
    if not overlay_df.empty:
        oz = overlay_df[overlay_df["overlay_id"] != "O0"].copy()
        if not oz.empty:
            oz["hard_gate_proxy_pass"] = to_num(oz["hard_gate_proxy_pass"]).fillna(0)
            oz["maxdd_improve_ratio_vs_base"] = to_num(oz["maxdd_improve_ratio_vs_base"])
            oz["cvar_improve_ratio_vs_base"] = to_num(oz["cvar_improve_ratio_vs_base"])
            oz["delta_expectancy_vs_base"] = to_num(oz["delta_expectancy_vs_base"])
            oz = oz.sort_values(
                ["hard_gate_proxy_pass", "maxdd_improve_ratio_vs_base", "cvar_improve_ratio_vs_base", "delta_expectancy_vs_base"],
                ascending=[False, False, False, False],
            )
            best_overlay_row = oz.iloc[0]
            if int(best_overlay_row.get("hard_gate_proxy_pass", 0)) == 1:
                if float(best_overlay_row.get("maxdd_improve_ratio_vs_base", np.nan)) >= 0.10 and float(best_overlay_row.get("cvar_improve_ratio_vs_base", np.nan)) >= 0.05:
                    if float(best_overlay_row.get("delta_expectancy_vs_base", np.nan)) >= -0.0002:
                        overlay_material = True

    # Infra status
    non_sol_target = [c for c in target_coins if c != "SOLUSDT"]
    non_sol_ok = [c for c in portability_meta.get("selected_coins", []) if c != "SOLUSDT"]
    infra_fail = len(non_sol_ok) == 0

    # Determine class
    if infra_fail:
        cls = "E"
        cls_reason = "no_nonSOL_portability_eval_completed"
    elif len(portability_go_coins) >= 1:
        cls = "A"
        cls_reason = f"portability_positive_on:{','.join(portability_go_coins)}"
    elif overlay_material:
        cls = "B"
        cls_reason = "sol_overlay_material_dd_tail_improvement"
    else:
        # if SOL execution still improves vs baseline but portability weak
        sol_best_exec = None
        if not portability_df.empty:
            sol_non = portability_df[(portability_df["symbol"] == "SOLUSDT") & (portability_df["exec_choice_id"] != "E0")].copy()
            if not sol_non.empty:
                sol_non["valid_for_ranking"] = to_num(sol_non["valid_for_ranking"]).fillna(0)
                sol_non["delta_expectancy_exec_minus_baseline"] = to_num(sol_non["delta_expectancy_exec_minus_baseline"])
                sol_non = sol_non.sort_values(["valid_for_ranking", "delta_expectancy_exec_minus_baseline"], ascending=[False, False])
                sol_best_exec = sol_non.iloc[0]
        if sol_best_exec is not None and int(sol_best_exec.get("valid_for_ranking", 0)) == 1 and float(sol_best_exec.get("delta_expectancy_exec_minus_baseline", 0.0)) > 0:
            cls = "C"
            cls_reason = "sol_execution_edge_persists_portability_weak_overlay_not_material"
        else:
            cls = "D"
            cls_reason = "absolute_ceiling_likely_upstream_signal_objective"

    # Root cause report
    root_lines: List[str] = []
    root_lines.append("# Phase V Root Cause Report")
    root_lines.append("")
    root_lines.append(f"- Generated UTC: {utc_now()}")
    root_lines.append(f"- Classification: **{cls}**")
    root_lines.append(f"- Reason: {cls_reason}")
    root_lines.append("")
    root_lines.append("## Portability Snapshot")
    root_lines.append("")
    root_lines.append(f"- non_SOL_targets: {', '.join(non_sol_target) if non_sol_target else 'none'}")
    root_lines.append(f"- non_SOL_evaluated: {', '.join(non_sol_ok) if non_sol_ok else 'none'}")
    root_lines.append(f"- portability_GO_coins: {', '.join(portability_go_coins) if portability_go_coins else 'none'}")
    root_lines.append(f"- portability_feasible_but_weak: {', '.join(portability_feasible_weak) if portability_feasible_weak else 'none'}")
    root_lines.append("")
    root_lines.append("## SOL Drawdown Drivers")
    root_lines.append("")
    if not dd_df.empty:
        d = dd_df.sort_values(["valid_for_ranking", "exec_expectancy_net"], ascending=[False, False]).iloc[0]
        root_lines.append(f"- scenario: {d['scenario_id']}")
        root_lines.append(f"- max_consecutive_losses: {int(d['max_consecutive_losses'])}")
        root_lines.append(f"- loss_run_ge3_count: {int(d['loss_run_ge3_count'])}")
        root_lines.append(f"- worst_split_id/expectancy: {int(d['worst_split_id'])} / {float(d['worst_split_expectancy']):.8f}")
        root_lines.append(f"- worst_session_bucket/pnl_sum: {d['worst_session_bucket']} / {float(d['worst_session_pnl_sum']):.8f}")
        root_lines.append(f"- worst_regime_bucket/pnl_sum: {d['worst_regime_bucket']} / {float(d['worst_regime_pnl_sum']):.8f}")
        root_lines.append(f"- sl_loss_share: {float(d['sl_loss_share']):.6f}")
        root_lines.append(f"- fee_drag_per_trade: {float(d['fee_drag_per_trade']):.8f}")
    else:
        root_lines.append("- no SOL forensic rows")

    root_lines.append("")
    root_lines.append("## Overlay Snapshot")
    root_lines.append("")
    if best_overlay_row is not None:
        root_lines.append(
            f"- best_overlay: {best_overlay_row['overlay_id']} ({best_overlay_row['overlay_desc']}) on {best_overlay_row['scenario_id']}"
        )
        root_lines.append(
            f"- delta_expectancy_vs_base: {float(best_overlay_row['delta_expectancy_vs_base']):.8f}, maxdd_improve_ratio_vs_base: {float(best_overlay_row['maxdd_improve_ratio_vs_base']):.6f}, cvar_improve_ratio_vs_base: {float(best_overlay_row['cvar_improve_ratio_vs_base']):.6f}, hard_gate_proxy_pass: {int(best_overlay_row['hard_gate_proxy_pass'])}"
        )
    else:
        root_lines.append("- no overlay rows")

    write_text(run_dir / "phaseV_root_cause_report.md", "\n".join(root_lines))

    # SOL forensic report
    f_lines: List[str] = []
    f_lines.append("# Phase V SOL DD Forensics Report")
    f_lines.append("")
    f_lines.append(f"- Generated UTC: {utc_now()}")
    f_lines.append(f"- Scenarios analyzed: {', '.join(dd_df['scenario_id'].tolist()) if not dd_df.empty else 'none'}")
    f_lines.append("")
    if not dd_df.empty:
        f_lines.append("## Scenario Table")
        f_lines.append("")
        cols = [
            "scenario_id",
            "source_tag",
            "valid_for_ranking",
            "entries_valid",
            "entry_rate",
            "exec_expectancy_net",
            "exec_delta_expectancy_vs_baseline",
            "exec_cvar_5",
            "exec_max_drawdown",
            "cvar_improve_ratio",
            "maxdd_improve_ratio",
            "max_consecutive_losses",
            "loss_run_ge3_count",
            "sl_loss_share",
            "fee_drag_per_trade",
            "worst_split_id",
            "worst_session_bucket",
            "worst_regime_bucket",
        ]
        f_lines.append(df_to_plain_table(dd_df, cols=cols))
        f_lines.append("")
    if not overlay_df.empty:
        f_lines.append("## Overlay Micro-Benchmark")
        f_lines.append("")
        ocols = [
            "scenario_id",
            "overlay_id",
            "overlay_desc",
            "entries_valid",
            "entry_rate",
            "exec_expectancy_net",
            "delta_expectancy_vs_base",
            "maxdd_improve_ratio_vs_base",
            "cvar_improve_ratio_vs_base",
            "removed_entries_pct",
            "removed_loss_share_abs",
            "hard_gate_proxy_pass",
        ]
        f_lines.append(df_to_plain_table(overlay_df, cols=ocols))
    write_text(run_dir / "phaseV_sol_dd_forensics_report.md", "\n".join(f_lines))

    # Decision memo + prompts
    decision_lines: List[str] = []
    decision_lines.append("# Decision Next Step")
    decision_lines.append("")
    decision_lines.append(f"- Generated UTC: {utc_now()}")
    decision_lines.append(f"- Phase V classification: **{cls}**")
    decision_lines.append(f"- Reason: {cls_reason}")
    decision_lines.append(f"- Recommended next 3h focus: {('portability expansion' if cls=='A' else 'SOL risk overlays' if cls=='B' else 'execution SOL caution (no scaling)' if cls=='C' else '1h objective redesign' if cls=='D' else 'infra portability repair')}")
    decision_lines.append(f"- More execution GA compute justified now: {('yes' if cls=='A' else 'no')}")
    decision_lines.append("- Stop-doing note: avoid full joint 1h×3m GA marathons until portability/overlay gates are evidenced.")
    write_text(run_dir / "decision_next_step.md", "\n".join(decision_lines))

    if cls == "A":
        next_prompt = """Phase W portability expansion (contract-locked execution): keep hard gates and canonical fee/metrics lock unchanged, scale execution candidate comparison E0/E1/E2/E4 to 3-5 additional coins with fixed best 1h params and documented per-coin subsets. Prioritize coins with existing 3m infra and adequate signal support. Include duplicate-adjusted controls, per-coin OOS split stability, and promote only coins with positive delta vs E0 plus risk improvement."""
        fail_prompt = """Failure branch (A): if expanded portability shows flat/negative economics across added coins, stop rollout and switch to signal-objective redesign before more execution GA compute."""
    elif cls == "B":
        next_prompt = """Phase W SOL overlay-focused pilot (contract-locked): keep execution candidate fixed to best current (E1 or E2), keep hard gates unchanged, and run a small overlay search over validated lightweight controls (session veto, daily loss cap, volatility veto, entry-improvement filter). Optimize for maxDD/CVaR improvement with bounded expectancy sacrifice and enforce participation realism. No new signal family and no full GA marathon."""
        fail_prompt = """Failure branch (B): if overlay pilot improves tails only by collapsing participation or breaking hard-gate proxies, stop overlay optimization and move to 1h objective redesign."""
    elif cls == "C":
        next_prompt = """Phase W execution robustness-only maintenance (SOL): freeze coin scaling and avoid new execution GA marathons. Run periodic paper/shadow robustness checks on SOL E1/E2 with stress micro-matrix, while preparing a separate upstream 1h objective redesign track."""
        fail_prompt = """Failure branch (C): if SOL robustness drifts or economics weaken in paper checks, de-prioritize the branch and halt new execution compute."""
    elif cls == "D":
        next_prompt = """Phase W 1h objective redesign kickoff: rebuild SOL 1h optimization objective to score through fixed realistic execution proxy (E1) rather than standalone 1h metrics. Keep execution hard gates/locks unchanged, run only a small validation batch, and verify rank consistency before any larger search."""
        fail_prompt = """Failure branch (D): if redesigned 1h objective still fails to improve execution-scored outcomes, stop SOL branch and reallocate research budget."""
    else:
        next_prompt = """NO_GO_INFRA_PORTABILITY re-entry: repair portability harness/data mismatches first (subset generation, symbol data availability, and execution bundle construction), then rerun V1/V2 only with strict lock checks before any forensics/overlay compute."""
        fail_prompt = """Failure branch (E): do not run V3/V4 mainline until non-SOL portability evaluations complete successfully with comparable subsets and lock validation."""

    write_text(run_dir / "ready_to_launch_next_prompt.txt", next_prompt)
    write_text(run_dir / "ready_to_launch_failure_branch_prompt.txt", fail_prompt)

    # finalize manifest
    manifest.update(
        {
            "completed_utc": utc_now(),
            "duration_sec": float(time.time() - t0),
            "classification": cls,
            "classification_reason": cls_reason,
            "freeze": {
                "canonical_fee_model": str(fee_path),
                "canonical_metrics_definition": str(metrics_path),
                "canonical_fee_sha256": fee_sha,
                "canonical_metrics_sha256": metrics_sha,
                "expected_fee_sha256": LOCKED["expected_fee_sha"],
                "expected_metrics_sha256": LOCKED["expected_metrics_sha"],
                "hash_match": int(freeze_hash_pass),
                "ga_exec_lock_validation": lock_validation,
            },
            "phase_u_source": exec_meta,
            "target_coins": target_coins,
            "portability_meta": portability_meta,
            "portability_rows": int(len(portability_df)),
            "sol_forensic_rows": int(len(dd_df)),
            "overlay_rows": int(len(overlay_df)),
            "invalid_reason_histogram": dict(sorted(invalid_hist.items(), key=lambda kv: (-kv[1], kv[0]))),
            "best_overlay": best_overlay_row.to_dict() if best_overlay_row is not None else None,
        }
    )
    json_dump(run_dir / "phaseV_run_manifest.json", manifest)

    return manifest


def load_json_safe(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase V Branch-B portability + SOL DD forensics + overlays")
    ap.add_argument("--seed", type=int, default=20260227)
    ap.add_argument("--portability-max-signals", type=int, default=600)
    args = ap.parse_args()

    manifest = run(args)
    print(json.dumps({"run_dir": manifest.get("run_dir", ""), "classification": manifest.get("classification", "")}, sort_keys=True))


if __name__ == "__main__":
    main()
