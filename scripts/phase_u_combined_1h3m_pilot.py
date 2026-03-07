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

from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402
from scripts import sol_reconcile_truth as recon  # noqa: E402


LOCKED = {
    "representative_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
}


@dataclass
class ExecChoice:
    exec_choice_id: str
    description: str
    genome_hash: str
    genome: Optional[Dict[str, Any]]
    source: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


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


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("params"), dict):
        return dict(payload["params"])
    return dict(payload)


def build_exec_args(signals_csv: Path, mode: str = "tight", seed: int = 20260226) -> argparse.Namespace:
    parser = ga_exec.build_arg_parser()
    args = parser.parse_args([])
    args.symbol = "SOLUSDT"
    args.symbols = ""
    args.rank = 1
    args.signals_csv = str(signals_csv)
    args.max_signals = 1200
    args.walkforward = True
    args.wf_splits = 5
    args.train_ratio = 0.70
    args.mode = mode
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


def latest_phase_s_dir(root: Path) -> Path:
    cands = sorted([p for p in root.glob("PHASEQRS_AUTORUN_*") if p.is_dir()], key=lambda p: p.name)
    if not cands:
        raise FileNotFoundError("No PHASEQRS_AUTORUN_* directories found")
    for p in reversed(cands):
        if (p / "phaseS" / "phaseS_final_candidates.csv").exists() and (p / "phaseS" / "phaseS_release_manifest.json").exists():
            return p
    raise FileNotFoundError("No PHASEQRS_AUTORUN dir with phaseS artifacts found")


def load_exec_choices(exec_reports_root: Path) -> Tuple[List[ExecChoice], Dict[str, Any]]:
    phase_s_parent = latest_phase_s_dir(exec_reports_root)
    phase_s_dir = phase_s_parent / "phaseS"
    final_df = pd.read_csv(phase_s_dir / "phaseS_final_candidates.csv")
    rel = load_json(phase_s_dir / "phaseS_release_manifest.json")
    src_run = Path(str(rel.get("source_run_dir", ""))).resolve()
    if not src_run.exists():
        raise FileNotFoundError(f"Source GA run from phase S not found: {src_run}")
    genomes_df = pd.read_csv(src_run / "genomes.csv")

    role_to_hash: Dict[str, str] = {}
    for _, r in final_df.iterrows():
        role_to_hash[str(r["role"]).strip().lower()] = str(r["genome_hash"]).strip()
    if "primary" not in role_to_hash or "backup" not in role_to_hash:
        raise RuntimeError("Phase S final candidates missing primary/backup")

    def genome_by_hash(h: str) -> Dict[str, Any]:
        z = genomes_df[genomes_df["genome_hash"].astype(str) == str(h)].copy()
        if z.empty:
            raise RuntimeError(f"Genome hash not found in source run: {h}")
        return extract_genome(z.iloc[0])

    g1_hash = role_to_hash["primary"]
    g2_hash = role_to_hash["backup"]
    g1 = genome_by_hash(g1_hash)
    g2 = genome_by_hash(g2_hash)

    # E4: high-participation but realistic control from same full run.
    z = genomes_df.copy()
    z["valid_for_ranking"] = to_num(z.get("valid_for_ranking", 0)).fillna(0).astype(int)
    z = z[z["valid_for_ranking"] == 1].copy()
    z["overall_entry_rate"] = to_num(z.get("overall_entry_rate", np.nan))
    z["overall_exec_taker_share"] = to_num(z.get("overall_exec_taker_share", np.nan))
    z["overall_exec_p95_fill_delay_min"] = to_num(z.get("overall_exec_p95_fill_delay_min", np.nan))
    z = z[(z["overall_exec_taker_share"] <= 0.25) & (z["overall_exec_p95_fill_delay_min"] <= 180.0)].copy()
    z = z[~z["genome_hash"].astype(str).isin({g1_hash, g2_hash})].copy()
    if z.empty:
        z = genomes_df[~genomes_df["genome_hash"].astype(str).isin({g1_hash, g2_hash})].copy()
    z = z.sort_values(["overall_entry_rate", "overall_exec_expectancy_net"], ascending=[False, False]).reset_index(drop=True)
    e4_hash = str(z.iloc[0]["genome_hash"]) if not z.empty else g1_hash
    e4 = genome_by_hash(e4_hash)

    # E3: simple safe genome derived from primary with fewer moving parts.
    e3 = copy.deepcopy(g1)
    e3.update(
        {
            "entry_mode": "market",
            "fallback_to_market": 1,
            "fallback_delay_min": 3,
            "max_fill_delay_min": 12,
            "limit_offset_bps": 0.5,
            "micro_vol_filter": 0,
            "killzone_filter": 0,
            "mss_displacement_gate": 0,
            "use_signal_quality_gate": 0,
            "skip_if_vol_gate": 0,
            "cooldown_min": 0,
            "time_stop_min": 180,
            "break_even_enabled": 0,
            "trailing_enabled": 0,
            "partial_take_enabled": 0,
            "tp_mult": 1.0,
            "sl_mult": 1.0,
            "min_entry_improvement_bps_gate": -5.0,
            "max_taker_share": 0.25,
        }
    )
    e3 = ga_exec._repair_genome(e3, mode="tight", repair_hist=None)

    choices = [
        ExecChoice("E0", "baseline_execution_reference", "BASELINE_E0", None, "computed_from_baseline_fields"),
        ExecChoice("E1", "phaseS_primary", g1_hash, g1, str(src_run)),
        ExecChoice("E2", "phaseS_backup", g2_hash, g2, str(src_run)),
        ExecChoice("E3", "safe_simple_control", sha256_text(json.dumps(e3, sort_keys=True)), e3, "derived_from_E1"),
        ExecChoice("E4", "high_participation_realistic_control", e4_hash, e4, str(src_run)),
    ]
    meta = {
        "phase_s_parent": str(phase_s_parent),
        "phase_s_dir": str(phase_s_dir),
        "source_run_dir": str(src_run),
        "primary_hash": g1_hash,
        "backup_hash": g2_hash,
        "high_participation_hash": e4_hash,
    }
    return choices, meta


def load_active_sol_params() -> Tuple[Path, Dict[str, Any], Dict[str, Any]]:
    scan_dirs = sorted((PROJECT_ROOT / "reports" / "params_scan").glob("*"), key=lambda p: p.name)
    chosen_csv: Optional[Path] = None
    best_row: Optional[pd.Series] = None
    for d in reversed(scan_dirs):
        fp = d / "best_by_symbol.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        if df.empty:
            continue
        x = df.copy()
        x["symbol"] = x.get("symbol", "").astype(str)
        x["side"] = x.get("side", "").astype(str).str.lower()
        if "pass" in x.columns:
            x["pass"] = x["pass"].astype(str).str.lower().isin({"1", "true", "yes"})
            x = x[x["pass"]]
        x = x[(x["symbol"] == "SOLUSDT") & (x["side"] == "long")]
        if x.empty:
            continue
        x["score"] = to_num(x.get("score", np.nan))
        x = x.sort_values("score", ascending=False).reset_index(drop=True)
        chosen_csv = fp
        best_row = x.iloc[0]
        break

    if best_row is None:
        p = PROJECT_ROOT / "data" / "metadata" / "params" / "SOLUSDT_C13_active_params_long.json"
        payload = load_json(p)
        return p, unwrap_params(payload), {"source": "fallback_active_params", "best_csv": ""}

    params_path = PROJECT_ROOT / str(best_row.get("params_file", "")).strip()
    if not params_path.exists():
        raise FileNotFoundError(f"Params file from best_by_symbol missing: {params_path}")
    payload = load_json(params_path)
    return params_path, unwrap_params(payload), {
        "source": "best_by_symbol",
        "best_csv": str(chosen_csv),
        "score": float(best_row.get("score", np.nan)),
        "row": {k: (v.item() if isinstance(v, (np.integer, np.floating)) else str(v)) for k, v in best_row.to_dict().items()},
    }


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def param_fingerprint(p: Dict[str, Any]) -> str:
    keep = {
        "entry_rsi_min": float(p.get("entry_rsi_min", np.nan)),
        "entry_rsi_max": float(p.get("entry_rsi_max", np.nan)),
        "adx_min": float(p.get("adx_min", np.nan)),
        "cycle1_adx_boost": float(p.get("cycle1_adx_boost", np.nan)),
        "cycle1_ema_sep_atr": float(p.get("cycle1_ema_sep_atr", np.nan)),
        "willr_by_cycle": [float(x) for x in p.get("willr_by_cycle", [])],
        "trade_cycles": [int(x) for x in p.get("trade_cycles", [])],
    }
    return sha256_text(json.dumps(keep, sort_keys=True))


def generate_1h_candidates(base_params: Dict[str, Any], n_total: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(int(seed))
    base = ga_long._norm_params(copy.deepcopy(base_params))

    cands: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add_one(name: str, p: Dict[str, Any], kind: str) -> None:
        q = ga_long._norm_params(copy.deepcopy(p))
        q["entry_rsi_min"] = clamp(q["entry_rsi_min"], 5.0, 80.0)
        q["entry_rsi_max"] = clamp(q["entry_rsi_max"], 20.0, 95.0)
        if q["entry_rsi_max"] <= q["entry_rsi_min"] + 1.0:
            q["entry_rsi_max"] = min(95.0, q["entry_rsi_min"] + 5.0)
        q["adx_min"] = clamp(q.get("adx_min", 18.0), 8.0, 40.0)
        q["cycle1_adx_boost"] = clamp(q.get("cycle1_adx_boost", 8.0), 0.0, 30.0)
        q["cycle1_ema_sep_atr"] = clamp(q.get("cycle1_ema_sep_atr", 0.35), 0.0, 2.0)
        w = [clamp(float(x), -100.0, -1.0) for x in list(q.get("willr_by_cycle", [-60, -60, -60, -60, -60]))]
        q["willr_by_cycle"] = w
        fp = param_fingerprint(q)
        if fp in seen:
            return
        seen.add(fp)
        cands.append({"signal_candidate_id": f"P{len(cands):02d}", "name": name, "kind": kind, "params": q, "param_hash": fp})

    add_one("base_active", base, "base")

    deterministic_mods: List[Tuple[str, Dict[str, Any]]] = []

    for d in (-4.0, -2.0, 2.0, 4.0):
        p = copy.deepcopy(base)
        p["entry_rsi_min"] = float(base["entry_rsi_min"] + d)
        p["entry_rsi_max"] = float(base["entry_rsi_max"] + d)
        deterministic_mods.append((f"rsi_shift_{d:+.0f}", p))

    for d in (-6.0, -3.0, 3.0, 6.0):
        p = copy.deepcopy(base)
        p["adx_min"] = float(base.get("adx_min", 18.0) + d)
        deterministic_mods.append((f"adx_min_{d:+.0f}", p))

    for d in (-6.0, -3.0, 3.0, 6.0):
        p = copy.deepcopy(base)
        p["cycle1_adx_boost"] = float(base.get("cycle1_adx_boost", 8.0) + d)
        deterministic_mods.append((f"cycle1_adx_boost_{d:+.0f}", p))

    for d in (-0.20, -0.10, 0.10, 0.20):
        p = copy.deepcopy(base)
        p["cycle1_ema_sep_atr"] = float(base.get("cycle1_ema_sep_atr", 0.35) + d)
        deterministic_mods.append((f"cycle1_ema_sep_{d:+.2f}", p))

    for d in (-8.0, -4.0, 4.0, 8.0):
        p = copy.deepcopy(base)
        p["willr_by_cycle"] = [float(x + d) for x in list(base.get("willr_by_cycle", [-60, -60, -60, -60, -60]))]
        deterministic_mods.append((f"willr_all_{d:+.0f}", p))

    cycle_sets = [[1, 2, 3], [1, 3], [2, 3]]
    for cs in cycle_sets:
        p = copy.deepcopy(base)
        p["trade_cycles"] = cs
        deterministic_mods.append((f"trade_cycles_{'_'.join(str(x) for x in cs)}", p))

    for nm, p in deterministic_mods:
        add_one(nm, p, "local")
        if len(cands) >= n_total:
            break

    deltas_rsi = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
    deltas_adx = [-8.0, -5.0, -2.0, 2.0, 5.0, 8.0]
    deltas_boost = [-8.0, -4.0, -2.0, 2.0, 4.0, 8.0]
    deltas_sep = [-0.30, -0.15, -0.05, 0.05, 0.15, 0.30]
    deltas_willr = [-12.0, -8.0, -4.0, 4.0, 8.0, 12.0]

    attempts = 0
    while len(cands) < n_total and attempts < n_total * 40:
        attempts += 1
        p = copy.deepcopy(base)
        moves = rng.sample(["rsi", "adx", "boost", "sep", "willr", "cycles"], k=rng.randint(2, 4))
        if "rsi" in moves:
            d = rng.choice(deltas_rsi)
            p["entry_rsi_min"] = float(base["entry_rsi_min"] + d)
            p["entry_rsi_max"] = float(base["entry_rsi_max"] + d + rng.choice([-1.0, 0.0, 1.0]))
        if "adx" in moves:
            p["adx_min"] = float(base.get("adx_min", 18.0) + rng.choice(deltas_adx))
        if "boost" in moves:
            p["cycle1_adx_boost"] = float(base.get("cycle1_adx_boost", 8.0) + rng.choice(deltas_boost))
        if "sep" in moves:
            p["cycle1_ema_sep_atr"] = float(base.get("cycle1_ema_sep_atr", 0.35) + rng.choice(deltas_sep))
        if "willr" in moves:
            d = rng.choice(deltas_willr)
            w = list(base.get("willr_by_cycle", [-60, -60, -60, -60, -60]))
            if rng.random() < 0.5:
                w = [float(x + d) for x in w]
            else:
                idx = rng.choice([1, 2, 3])
                w[idx] = float(w[idx] + d)
            p["willr_by_cycle"] = w
        if "cycles" in moves:
            p["trade_cycles"] = rng.choice([[1, 2], [1, 2, 3], [1, 3], [2, 3]])
        kind = "explore" if abs(float(p.get("adx_min", base.get("adx_min", 18.0))) - float(base.get("adx_min", 18.0))) >= 5.0 else "local"
        add_one(f"rand_{attempts:03d}", p, kind)

    return cands[:n_total]


def build_rep_subset_with_idx(rep_subset: pd.DataFrame, df_feat: pd.DataFrame) -> pd.DataFrame:
    rep = rep_subset.copy()
    rep["signal_time"] = pd.to_datetime(rep["signal_time"], utc=True, errors="coerce")
    rep = rep.dropna(subset=["signal_time"]).copy()
    rep["signal_id"] = rep["signal_id"].astype(str)
    # Use Timestamp.value to force nanosecond epoch consistently across datetime dtypes.
    rep["_ts_ns"] = rep["signal_time"].map(lambda x: int(pd.to_datetime(x, utc=True).value)).astype(np.int64)

    ts_full = pd.to_datetime(df_feat["Timestamp"], utc=True, errors="coerce")
    idx_by_ns: Dict[int, int] = {int(pd.to_datetime(v, utc=True).value): int(i) for i, v in enumerate(ts_full.tolist())}
    rep["_full_idx"] = rep["_ts_ns"].map(lambda x: idx_by_ns.get(int(x), -1)).astype(int)
    rep = rep.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    return rep


def active_signal_ids_for_params(df_feat: pd.DataFrame, params: Dict[str, Any], rep_idx: pd.DataFrame) -> Tuple[set[str], Dict[str, Any]]:
    sig = ga_long.build_entry_signal(df_feat, params, assume_prepared=True)
    sig = np.asarray(sig, dtype=bool)

    full_idx = rep_idx["_full_idx"].to_numpy(dtype=int)
    keep_mask = np.zeros(len(rep_idx), dtype=bool)
    valid_idx_mask = (full_idx >= 0) & (full_idx < len(sig))
    keep_mask[valid_idx_mask] = sig[full_idx[valid_idx_mask]]

    active_ids = set(rep_idx.loc[keep_mask, "signal_id"].astype(str).tolist())
    diag = {
        "rep_signals_total": int(len(rep_idx)),
        "mapped_to_1h_index": int(valid_idx_mask.sum()),
        "active_signals": int(len(active_ids)),
        "active_rate_vs_rep": float(len(active_ids) / max(1, len(rep_idx))),
    }
    return active_ids, diag


def build_candidate_bundle(base_bundle: ga_exec.SymbolBundle, active_ids: set[str], args: argparse.Namespace) -> ga_exec.SymbolBundle:
    contexts = [ctx for ctx in base_bundle.contexts if str(ctx.signal_id) in active_ids]
    contexts = sorted(contexts, key=lambda r: (pd.to_datetime(r.signal_time, utc=True), str(r.signal_id)))
    splits = ga_exec._build_walkforward_splits(n=len(contexts), train_ratio=float(args.train_ratio), n_splits=int(args.wf_splits))
    return ga_exec.SymbolBundle(
        symbol=base_bundle.symbol,
        signals_csv=base_bundle.signals_csv,
        contexts=contexts,
        splits=splits,
        constraints=dict(base_bundle.constraints),
    )


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


def eval_exec_choice(bundle: ga_exec.SymbolBundle, args: argparse.Namespace, choice: ExecChoice) -> Dict[str, Any]:
    if choice.exec_choice_id == "E0":
        out = baseline_metrics_from_bundle(bundle=bundle, args=args)
        out["eval_time_sec"] = 0.0
        return out
    if choice.genome is None:
        raise RuntimeError(f"Genome missing for choice {choice.exec_choice_id}")
    return ga_exec._evaluate_genome(genome=choice.genome, bundles=[bundle], args=args, detailed=False)


def metric_signature(df: pd.DataFrame) -> pd.Series:
    cols = [
        "signal_signature",
        "exec_choice_id",
        "overall_signals_total",
        "overall_entries_valid",
        "overall_entry_rate",
        "overall_exec_expectancy_net",
        "overall_exec_cvar_5",
        "overall_exec_max_drawdown",
        "overall_exec_taker_share",
        "overall_exec_p95_fill_delay_min",
    ]
    x = df.copy()
    for c in cols:
        if c not in x.columns:
            x[c] = np.nan
    return (
        x[cols]
        .round(12)
        .apply(lambda r: "|".join("nan" if pd.isna(v) else str(v) for v in r.tolist()), axis=1)
        .map(lambda s: sha256_text(str(s)))
    )


def choose_best_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    z = df.copy()
    z["valid_for_ranking"] = to_num(z.get("valid_for_ranking", 0)).fillna(0)
    z["equity_growth_proxy"] = to_num(z.get("equity_growth_proxy", np.nan))
    z["overall_cvar_improve_ratio"] = to_num(z.get("overall_cvar_improve_ratio", np.nan))
    z["overall_maxdd_improve_ratio"] = to_num(z.get("overall_maxdd_improve_ratio", np.nan))
    z["std_split_expectancy_net"] = to_num(z.get("std_split_expectancy_net", np.nan))
    z["overall_exec_expectancy_net"] = to_num(z.get("overall_exec_expectancy_net", np.nan))
    z = z.sort_values(
        [
            "valid_for_ranking",
            "equity_growth_proxy",
            "overall_cvar_improve_ratio",
            "overall_maxdd_improve_ratio",
            "overall_exec_expectancy_net",
            "std_split_expectancy_net",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    return z.iloc[0]


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t_start = time.time()

    out_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_dir = out_root / f"PHASEU_COMBINED_1H3M_PILOT_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "phase": "U",
        "project_root": str(PROJECT_ROOT),
        "run_dir": str(run_dir),
        "code_modified": "YES (new script phase_u_combined_1h3m_pilot.py)",
        "commands": [{"cmd": "python -m scripts.phase_u_combined_1h3m_pilot", "utc": utc_now()}],
    }

    try:
        # U1: lock validation + harness assembly
        rep_csv = Path(LOCKED["representative_subset_csv"]).resolve()
        fee_path = Path(LOCKED["canonical_fee_model"]).resolve()
        metrics_path = Path(LOCKED["canonical_metrics_definition"]).resolve()
        for fp in (rep_csv, fee_path, metrics_path):
            if not fp.exists():
                raise FileNotFoundError(f"Locked file missing: {fp}")

        fee_sha = sha256_file(fee_path)
        metrics_sha = sha256_file(metrics_path)
        freeze_ok = int(fee_sha == LOCKED["expected_fee_sha"] and metrics_sha == LOCKED["expected_metrics_sha"])

        exec_choices, exec_meta = load_exec_choices(out_root)
        params_path, base_params_raw, params_meta = load_active_sol_params()
        base_params = ga_long._norm_params(copy.deepcopy(base_params_raw))

        rep_subset = pd.read_csv(rep_csv)
        rep_subset["signal_time"] = pd.to_datetime(rep_subset["signal_time"], utc=True, errors="coerce")
        rep_subset = rep_subset.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
        rep_subset["signal_id"] = rep_subset["signal_id"].astype(str)

        df1h = recon._load_symbol_df("SOLUSDT", tf="1h")
        df_feat = ga_long._ensure_indicators(df1h.copy(), base_params)
        df_feat = ga_long._prepare_signal_df(df_feat, assume_prepared=False)

        rep_idx = build_rep_subset_with_idx(rep_subset=rep_subset, df_feat=df_feat)

        exec_args = build_exec_args(signals_csv=rep_csv, mode="tight", seed=args.seed)
        freeze_lock_local = ga_exec._validate_and_lock_frozen_artifacts(args=exec_args, run_dir=run_dir)
        if int(freeze_lock_local.get("freeze_lock_pass", 0)) != 1:
            raise RuntimeError("Execution freeze lock validation failed")

        bundles, load_meta = ga_exec._prepare_bundles(exec_args)
        if not bundles:
            raise RuntimeError("No bundles prepared")
        base_bundle = bundles[0]
        if str(base_bundle.symbol).upper() != "SOLUSDT":
            raise RuntimeError(f"Unexpected bundle symbol: {base_bundle.symbol}")

        # Smoke tests (3-5 candidates)
        candidates_all = generate_1h_candidates(base_params=base_params, n_total=int(args.signal_candidate_count), seed=int(args.seed))
        smoke_rows: List[Dict[str, Any]] = []
        for cand in candidates_all[: min(5, len(candidates_all))]:
            active_ids, diag = active_signal_ids_for_params(df_feat=df_feat, params=cand["params"], rep_idx=rep_idx)
            b = build_candidate_bundle(base_bundle=base_bundle, active_ids=active_ids, args=exec_args)
            if len(b.contexts) == 0:
                met = {
                    "valid_for_ranking": 0,
                    "overall_entries_valid": 0.0,
                    "overall_entry_rate": 0.0,
                    "overall_exec_expectancy_net": float("nan"),
                    "overall_exec_cvar_5": float("nan"),
                    "overall_exec_max_drawdown": float("nan"),
                    "nan_pass": 0,
                    "participation_pass": 0,
                    "realism_pass": 0,
                    "invalid_reason": "no_signals",
                }
            else:
                met = eval_exec_choice(bundle=b, args=exec_args, choice=next(c for c in exec_choices if c.exec_choice_id == "E1"))
            smoke_rows.append(
                {
                    "signal_candidate_id": cand["signal_candidate_id"],
                    "candidate_name": cand["name"],
                    "signals_active": int(len(active_ids)),
                    "mapped_to_1h_index": int(diag["mapped_to_1h_index"]),
                    "exec_choice_id": "E1",
                    "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
                    "overall_entries_valid": float(met.get("overall_entries_valid", np.nan)),
                    "overall_entry_rate": float(met.get("overall_entry_rate", np.nan)),
                    "overall_exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
                    "overall_exec_cvar_5": float(met.get("overall_exec_cvar_5", np.nan)),
                    "overall_exec_max_drawdown": float(met.get("overall_exec_max_drawdown", np.nan)),
                    "nan_pass": int(met.get("nan_pass", 0)),
                    "participation_pass": int(met.get("participation_pass", 0)),
                    "realism_pass": int(met.get("realism_pass", 0)),
                    "invalid_reason": str(met.get("invalid_reason", "")),
                }
            )
        smoke_df = pd.DataFrame(smoke_rows)
        smoke_df.to_csv(run_dir / "phaseU_smoke_results.csv", index=False)

        harness_lines: List[str] = []
        harness_lines.append("# Phase U Harness Validation")
        harness_lines.append("")
        harness_lines.append(f"- Generated UTC: {utc_now()}")
        harness_lines.append(f"- Frozen representative subset CSV: `{rep_csv}`")
        harness_lines.append(f"- Canonical fee model path: `{fee_path}`")
        harness_lines.append(f"- Canonical metrics definition path: `{metrics_path}`")
        harness_lines.append(f"- Fee hash: `{fee_sha}` (match={int(fee_sha == LOCKED['expected_fee_sha'])})")
        harness_lines.append(f"- Metrics hash: `{metrics_sha}` (match={int(metrics_sha == LOCKED['expected_metrics_sha'])})")
        harness_lines.append(f"- Local freeze lock pass: `{int(freeze_lock_local.get('freeze_lock_pass', 0))}`")
        harness_lines.append("")
        harness_lines.append("## Entrypoints")
        harness_lines.append("")
        harness_lines.append("- 1h signal engine: `src/bot087/optim/ga.py` (`build_entry_signal`, `run_backtest_long_only`)")
        harness_lines.append("- 3m execution evaluator: `src/execution/ga_exec_3m_opt.py` (`_prepare_bundles`, `_evaluate_genome`)")
        harness_lines.append("")
        harness_lines.append("## Active/Base SOL Params")
        harness_lines.append("")
        harness_lines.append(f"- Params file: `{params_path}`")
        harness_lines.append(f"- Params source: `{params_meta.get('source', '')}`")
        if str(params_meta.get("best_csv", "")):
            harness_lines.append(f"- best_by_symbol.csv: `{params_meta.get('best_csv')}`")
        harness_lines.append("")
        harness_lines.append("## Execution Choice Set")
        harness_lines.append("")
        for c in exec_choices:
            harness_lines.append(f"- {c.exec_choice_id}: {c.description}, genome_hash=`{c.genome_hash}`, source=`{c.source}`")
        harness_lines.append("")
        harness_lines.append("## Smoke")
        harness_lines.append("")
        harness_lines.append(f"- Smoke rows: {len(smoke_df)}")
        if not smoke_df.empty:
            harness_lines.append(f"- Smoke valid_for_ranking count: {int(to_num(smoke_df['valid_for_ranking']).fillna(0).sum())}")
            harness_lines.append(f"- Smoke avg entry_rate: {float(to_num(smoke_df['overall_entry_rate']).mean()):.6f}")
        write_text(run_dir / "phaseU_harness_validation.md", "\n".join(harness_lines))

        # U2: combined pilot search (reduced local 1h + discrete exec gene)
        pilot_rows: List[Dict[str, Any]] = []
        u2_start = time.time()

        for cand in candidates_all:
            active_ids, sig_diag = active_signal_ids_for_params(df_feat=df_feat, params=cand["params"], rep_idx=rep_idx)
            signal_sig = sha256_text("|".join(sorted(active_ids)))
            bundle = build_candidate_bundle(base_bundle=base_bundle, active_ids=active_ids, args=exec_args)

            if len(bundle.contexts) == 0:
                # record empty quickly for all exec choices
                for choice in exec_choices:
                    pilot_rows.append(
                        {
                            "signal_candidate_id": cand["signal_candidate_id"],
                            "signal_candidate_name": cand["name"],
                            "signal_candidate_kind": cand["kind"],
                            "signal_param_hash": cand["param_hash"],
                            "exec_choice_id": choice.exec_choice_id,
                            "exec_choice_desc": choice.description,
                            "exec_genome_hash": choice.genome_hash,
                            "signal_signature": signal_sig,
                            "signals_total": 0,
                            "mapped_to_1h_index": int(sig_diag["mapped_to_1h_index"]),
                            "entries_valid": 0,
                            "entry_rate": 0.0,
                            "overall_exec_expectancy_net": float("nan"),
                            "overall_delta_expectancy_exec_minus_baseline": float("nan"),
                            "overall_exec_cvar_5": float("nan"),
                            "overall_exec_max_drawdown": float("nan"),
                            "overall_cvar_improve_ratio": float("nan"),
                            "overall_maxdd_improve_ratio": float("nan"),
                            "overall_exec_taker_share": float("nan"),
                            "overall_exec_median_fill_delay_min": float("nan"),
                            "overall_exec_p95_fill_delay_min": float("nan"),
                            "overall_exec_pnl_std": float("nan"),
                            "min_split_expectancy_net": float("nan"),
                            "median_split_expectancy_net": float("nan"),
                            "std_split_expectancy_net": float("nan"),
                            "support_ok": 0,
                            "participation_pass": 0,
                            "realism_pass": 0,
                            "nan_pass": 0,
                            "split_pass": 0,
                            "constraint_pass": 1,
                            "valid_for_ranking": 0,
                            "hard_invalid": 1,
                            "invalid_reason": "no_signals",
                            "participation_fail_reason": "overall:trades<1|overall:entry_rate",
                            "realism_fail_reason": "",
                            "split_fail_reason": "split_count_mismatch",
                            "eval_time_sec": 0.0,
                        }
                    )
                continue

            # evaluate all execution choices
            for choice in exec_choices:
                t0 = time.time()
                met = eval_exec_choice(bundle=bundle, args=exec_args, choice=choice)
                t1 = time.time()
                pilot_rows.append(
                    {
                        "signal_candidate_id": cand["signal_candidate_id"],
                        "signal_candidate_name": cand["name"],
                        "signal_candidate_kind": cand["kind"],
                        "signal_param_hash": cand["param_hash"],
                        "exec_choice_id": choice.exec_choice_id,
                        "exec_choice_desc": choice.description,
                        "exec_genome_hash": choice.genome_hash,
                        "signal_signature": signal_sig,
                        "signals_total": int(met.get("overall_signals_total", len(bundle.contexts))),
                        "mapped_to_1h_index": int(sig_diag["mapped_to_1h_index"]),
                        "entries_valid": float(met.get("overall_entries_valid", np.nan)),
                        "entry_rate": float(met.get("overall_entry_rate", np.nan)),
                        "overall_exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
                        "overall_delta_expectancy_exec_minus_baseline": float(met.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
                        "overall_exec_cvar_5": float(met.get("overall_exec_cvar_5", np.nan)),
                        "overall_exec_max_drawdown": float(met.get("overall_exec_max_drawdown", np.nan)),
                        "overall_cvar_improve_ratio": float(met.get("overall_cvar_improve_ratio", np.nan)),
                        "overall_maxdd_improve_ratio": float(met.get("overall_maxdd_improve_ratio", np.nan)),
                        "overall_exec_taker_share": float(met.get("overall_exec_taker_share", np.nan)),
                        "overall_exec_median_fill_delay_min": float(met.get("overall_exec_median_fill_delay_min", np.nan)),
                        "overall_exec_p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
                        "overall_exec_pnl_std": float(met.get("overall_exec_pnl_std", np.nan)),
                        "min_split_expectancy_net": float(met.get("min_split_expectancy_net", np.nan)),
                        "median_split_expectancy_net": float(met.get("median_split_expectancy_net", np.nan)),
                        "std_split_expectancy_net": float(met.get("std_split_expectancy_net", np.nan)),
                        "support_ok": int(met.get("split_pass", 0)),
                        "participation_pass": int(met.get("participation_pass", 0)),
                        "realism_pass": int(met.get("realism_pass", 0)),
                        "nan_pass": int(met.get("nan_pass", 0)),
                        "split_pass": int(met.get("split_pass", 0)),
                        "constraint_pass": int(met.get("constraint_pass", 1)),
                        "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
                        "hard_invalid": int(met.get("hard_invalid", 1)),
                        "invalid_reason": str(met.get("invalid_reason", "")),
                        "participation_fail_reason": str(met.get("participation_fail_reason", "")),
                        "realism_fail_reason": str(met.get("realism_fail_reason", "")),
                        "split_fail_reason": str(met.get("split_fail_reason", "")),
                        "eval_time_sec": float(met.get("eval_time_sec", t1 - t0)),
                    }
                )

            # optional stop if time budget exceeded
            elapsed_min = (time.time() - u2_start) / 60.0
            if elapsed_min >= float(args.max_u2_minutes):
                break

        pilot_df = pd.DataFrame(pilot_rows)
        if pilot_df.empty:
            raise RuntimeError("No pilot rows generated")

        # delta vs E0 same signal candidate
        base_by_signal = (
            pilot_df[pilot_df["exec_choice_id"] == "E0"][["signal_candidate_id", "overall_exec_expectancy_net", "overall_exec_cvar_5", "overall_exec_max_drawdown"]]
            .rename(
                columns={
                    "overall_exec_expectancy_net": "e0_expectancy",
                    "overall_exec_cvar_5": "e0_cvar",
                    "overall_exec_max_drawdown": "e0_maxdd",
                }
            )
            .drop_duplicates("signal_candidate_id")
        )
        pilot_df = pilot_df.merge(base_by_signal, on="signal_candidate_id", how="left")
        pilot_df["delta_expectancy_vs_e0"] = to_num(pilot_df["overall_exec_expectancy_net"]) - to_num(pilot_df["e0_expectancy"])
        pilot_df["delta_cvar_vs_e0"] = to_num(pilot_df["overall_exec_cvar_5"]) - to_num(pilot_df["e0_cvar"])
        pilot_df["delta_maxdd_vs_e0"] = to_num(pilot_df["overall_exec_max_drawdown"]) - to_num(pilot_df["e0_maxdd"])

        # ranking proxy (equity growth aware proxy)
        pilot_df["equity_growth_proxy"] = to_num(pilot_df["overall_exec_expectancy_net"]) * to_num(pilot_df["entry_rate"]).fillna(0.0)

        # duplicate map
        pilot_df["metric_signature"] = metric_signature(pilot_df)
        pilot_df["joint_signature"] = pilot_df.apply(
            lambda r: sha256_text(
                "|".join(
                    [
                        str(r.get("signal_signature", "")),
                        str(r.get("exec_choice_id", "")),
                        str(r.get("metric_signature", "")),
                    ]
                )
            ),
            axis=1,
        )
        pilot_df["joint_dup_count"] = pilot_df.groupby("joint_signature")["joint_signature"].transform("count").astype(int)
        pilot_df["is_joint_duplicate"] = (pilot_df["joint_dup_count"] > 1).astype(int)

        dup_rows: List[Dict[str, Any]] = []
        for sig, grp in pilot_df.groupby("joint_signature", dropna=False):
            if int(len(grp)) <= 1:
                continue
            rep_idx = grp.index[0]
            for i, (_, r) in enumerate(grp.iterrows()):
                dup_rows.append(
                    {
                        "joint_signature": str(sig),
                        "group_size": int(len(grp)),
                        "representative_index": int(rep_idx),
                        "row_index": int(r.name),
                        "is_representative": int(i == 0),
                        "signal_candidate_id": str(r["signal_candidate_id"]),
                        "exec_choice_id": str(r["exec_choice_id"]),
                        "signal_param_hash": str(r["signal_param_hash"]),
                        "exec_genome_hash": str(r["exec_genome_hash"]),
                    }
                )
        dup_df = pd.DataFrame(dup_rows)
        dup_df.to_csv(run_dir / "phaseU_duplicate_variant_map.csv", index=False)

        # non-duplicate view
        nond = pilot_df.drop_duplicates("joint_signature", keep="first").copy().reset_index(drop=True)

        # effective trials proxy + significance
        feat_cols = [
            "equity_growth_proxy",
            "overall_exec_expectancy_net",
            "overall_cvar_improve_ratio",
            "overall_maxdd_improve_ratio",
            "entry_rate",
            "overall_exec_taker_share",
            "overall_exec_p95_fill_delay_min",
            "delta_expectancy_vs_e0",
        ]
        X = nond[feat_cols].copy()
        for c in feat_cols:
            X[c] = to_num(X[c])
        mat = X.to_numpy(dtype=float)
        n_eff_corr, avg_abs_corr = effective_trials_from_corr(mat)
        n_uncorr = float(nond.shape[0])
        penalty = math.sqrt(2.0 * math.log(max(2.0, n_eff_corr if np.isfinite(n_eff_corr) else n_uncorr)))

        nond["z_expectancy_proxy"] = [
            z_proxy(m, s, n)
            for m, s, n in zip(
                to_num(nond["overall_exec_expectancy_net"]),
                np.maximum(1e-12, to_num(nond["overall_exec_pnl_std"])),
                np.maximum(2.0, to_num(nond["entries_valid"])),
            )
        ]
        nond["psr_proxy"] = nond["z_expectancy_proxy"].map(norm_cdf)
        nond["dsr_proxy"] = nond["z_expectancy_proxy"].map(lambda z: norm_cdf(float(z) - penalty) if np.isfinite(z) else float("nan"))

        shortlist = nond.copy()
        shortlist["valid_for_ranking"] = to_num(shortlist["valid_for_ranking"]).fillna(0)
        shortlist = shortlist.sort_values(
            [
                "valid_for_ranking",
                "equity_growth_proxy",
                "overall_cvar_improve_ratio",
                "overall_maxdd_improve_ratio",
                "overall_exec_expectancy_net",
                "std_split_expectancy_net",
            ],
            ascending=[False, False, False, False, False, True],
        ).reset_index(drop=True)
        shortlist["shortlist_rank"] = np.arange(1, len(shortlist) + 1)
        shortlist_top = shortlist.head(min(50, len(shortlist))).copy()
        shortlist_top.to_csv(run_dir / "phaseU_shortlist_significance.csv", index=False)

        # invalid histogram
        invalid_hist: Dict[str, int] = {}
        for s in pilot_df.loc[to_num(pilot_df["valid_for_ranking"]).fillna(0).astype(int) == 0, "invalid_reason"].fillna("").astype(str):
            if not s.strip():
                invalid_hist["none"] = invalid_hist.get("none", 0) + 1
                continue
            parts = [p for p in s.split("|") if p]
            if not parts:
                invalid_hist["none"] = invalid_hist.get("none", 0) + 1
            for p in parts:
                invalid_hist[p] = invalid_hist.get(p, 0) + 1
        json_dump(run_dir / "phaseU_invalid_reason_histogram.json", invalid_hist)

        # additional controls summary
        et_lines: List[str] = []
        et_lines.append("# Phase U Effective Trials Summary")
        et_lines.append("")
        et_lines.append(f"- Generated UTC: {utc_now()}")
        et_lines.append(f"- pilot_rows_total: {int(len(pilot_df))}")
        et_lines.append(f"- nonduplicate_joint_signatures: {int(len(nond))}")
        et_lines.append(f"- duplicate_rate_raw: {float(to_num(pilot_df['is_joint_duplicate']).mean()):.6f}")
        et_lines.append(f"- effective_trials_uncorrelated_proxy: {n_uncorr:.6f}")
        et_lines.append(f"- avg_abs_corr_proxy: {avg_abs_corr:.6f}")
        et_lines.append(f"- effective_trials_corr_adjusted_proxy: {n_eff_corr:.6f}")
        et_lines.append("")
        et_lines.append("Reality-check control:")
        et_lines.append("- TODO: bootstrap reality-check benchmark remains a placeholder in this pilot.")
        write_text(run_dir / "phaseU_effective_trials_summary.md", "\n".join(et_lines))

        # U3: decomposition / route comparison
        def row_for(cand_id: str, ex: str) -> Optional[pd.Series]:
            z = shortlist[(shortlist["signal_candidate_id"] == cand_id) & (shortlist["exec_choice_id"] == ex)]
            if z.empty:
                return None
            return z.iloc[0]

        base_id = "P00"
        # actual base id is first candidate id generated.
        if candidates_all:
            base_id = str(candidates_all[0]["signal_candidate_id"])

        r0 = row_for(base_id, "E0")
        r1 = row_for(base_id, "E1")

        valid_non_e0 = shortlist[(shortlist["exec_choice_id"] != "E0") & (to_num(shortlist["valid_for_ranking"]).fillna(0) == 1)].copy()
        best_non_e0 = choose_best_row(valid_non_e0)
        if best_non_e0 is None:
            best_non_e0 = choose_best_row(shortlist[shortlist["exec_choice_id"] != "E0"].copy())

        if best_non_e0 is not None:
            r2 = best_non_e0
            r3 = row_for(str(r2["signal_candidate_id"]), "E0")
        else:
            r2 = None
            r3 = None

        e0_only = shortlist[shortlist["exec_choice_id"] == "E0"].copy()
        best_1h_e0 = choose_best_row(e0_only)
        r4 = None
        if best_1h_e0 is not None:
            r4 = row_for(str(best_1h_e0["signal_candidate_id"]), "E1")

        route_rows: List[Dict[str, Any]] = []

        def emit_route(route_id: str, label: str, row: Optional[pd.Series]) -> None:
            route_rows.append(
                {
                    "route_id": route_id,
                    "label": label,
                    "signal_candidate_id": str(row["signal_candidate_id"]) if row is not None else "",
                    "exec_choice_id": str(row["exec_choice_id"]) if row is not None else "",
                    "valid_for_ranking": int(row["valid_for_ranking"]) if row is not None and np.isfinite(float(row["valid_for_ranking"])) else 0,
                    "signals_total": float(row["signals_total"]) if row is not None else np.nan,
                    "entries_valid": float(row["entries_valid"]) if row is not None else np.nan,
                    "entry_rate": float(row["entry_rate"]) if row is not None else np.nan,
                    "overall_exec_expectancy_net": float(row["overall_exec_expectancy_net"]) if row is not None else np.nan,
                    "delta_expectancy_vs_e0": float(row["delta_expectancy_vs_e0"]) if row is not None else np.nan,
                    "overall_cvar_improve_ratio": float(row["overall_cvar_improve_ratio"]) if row is not None else np.nan,
                    "overall_maxdd_improve_ratio": float(row["overall_maxdd_improve_ratio"]) if row is not None else np.nan,
                    "overall_exec_taker_share": float(row["overall_exec_taker_share"]) if row is not None else np.nan,
                    "overall_exec_p95_fill_delay_min": float(row["overall_exec_p95_fill_delay_min"]) if row is not None else np.nan,
                    "min_split_expectancy_net": float(row["min_split_expectancy_net"]) if row is not None else np.nan,
                    "median_split_expectancy_net": float(row["median_split_expectancy_net"]) if row is not None else np.nan,
                    "std_split_expectancy_net": float(row["std_split_expectancy_net"]) if row is not None else np.nan,
                }
            )

        emit_route("R0", "current_1h_best + baseline_exec_E0", r0)
        emit_route("R1", "current_1h_best + phaseS_primary_exec_E1", r1)
        emit_route("R2", "best_combined_pilot_candidate", r2)
        emit_route("R3", "best_combined_signal_under_E0", r3)
        emit_route("R4", "best_1h_only_under_E1", r4)

        route_df = pd.DataFrame(route_rows)
        route_df.to_csv(run_dir / "phaseU_route_comparison.csv", index=False)

        def row_val(row: Optional[pd.Series], key: str) -> float:
            if row is None:
                return float("nan")
            try:
                return float(row.get(key, np.nan))
            except Exception:
                return float("nan")

        signal_gain = row_val(r3, "overall_exec_expectancy_net") - row_val(r0, "overall_exec_expectancy_net")
        exec_gain_base = row_val(r1, "overall_exec_expectancy_net") - row_val(r0, "overall_exec_expectancy_net")
        exec_gain_on_best_signal = row_val(r2, "overall_exec_expectancy_net") - row_val(r3, "overall_exec_expectancy_net")
        total_gain = row_val(r2, "overall_exec_expectancy_net") - row_val(r0, "overall_exec_expectancy_net")

        decomp_lines: List[str] = []
        decomp_lines.append("# Phase U Improvement Decomposition")
        decomp_lines.append("")
        decomp_lines.append(f"- Generated UTC: {utc_now()}")
        decomp_lines.append(f"- signal_gain (R3-R0 expectancy): {signal_gain:.8f}" if np.isfinite(signal_gain) else "- signal_gain (R3-R0 expectancy): nan")
        decomp_lines.append(f"- execution_gain_base (R1-R0 expectancy): {exec_gain_base:.8f}" if np.isfinite(exec_gain_base) else "- execution_gain_base (R1-R0 expectancy): nan")
        decomp_lines.append(
            f"- execution_gain_on_best_signal (R2-R3 expectancy): {exec_gain_on_best_signal:.8f}"
            if np.isfinite(exec_gain_on_best_signal)
            else "- execution_gain_on_best_signal (R2-R3 expectancy): nan"
        )
        decomp_lines.append(f"- total_gain (R2-R0 expectancy): {total_gain:.8f}" if np.isfinite(total_gain) else "- total_gain (R2-R0 expectancy): nan")
        decomp_lines.append("")
        decomp_lines.append("## Attribution")
        decomp_lines.append("")
        if np.isfinite(signal_gain) and np.isfinite(exec_gain_on_best_signal):
            if abs(exec_gain_on_best_signal) > abs(signal_gain) * 1.5:
                decomp_lines.append("- Dominant source: execution routing effect dominates over 1h signal variation.")
            elif abs(signal_gain) > abs(exec_gain_on_best_signal) * 1.5:
                decomp_lines.append("- Dominant source: 1h signal variation dominates execution routing effect.")
            else:
                decomp_lines.append("- Dominant source: mixed signal + execution contribution.")
        else:
            decomp_lines.append("- Dominant source: insufficient comparable routes to attribute robustly.")
        decomp_lines.append("")
        decomp_lines.append("## Drawdown / Participation Notes")
        decomp_lines.append("")
        if r2 is not None:
            decomp_lines.append(
                f"- Best combined candidate cvar/maxdd improve ratios: {row_val(r2, 'overall_cvar_improve_ratio'):.6f} / {row_val(r2, 'overall_maxdd_improve_ratio'):.6f}"
            )
            decomp_lines.append(
                f"- Best combined participation: entries={row_val(r2, 'entries_valid'):.0f}, entry_rate={row_val(r2, 'entry_rate'):.6f}, taker_share={row_val(r2, 'overall_exec_taker_share'):.6f}, p95_delay={row_val(r2, 'overall_exec_p95_fill_delay_min'):.2f}"
            )
        write_text(run_dir / "phaseU_improvement_decomposition.md", "\n".join(decomp_lines))

        # Ranking hold-up test (1h-only rank vs E1 rank)
        e0_rank_df = shortlist[shortlist["exec_choice_id"] == "E0"][["signal_candidate_id", "overall_exec_expectancy_net"]].rename(
            columns={"overall_exec_expectancy_net": "e0_expectancy"}
        )
        e1_rank_df = shortlist[shortlist["exec_choice_id"] == "E1"][["signal_candidate_id", "overall_exec_expectancy_net"]].rename(
            columns={"overall_exec_expectancy_net": "e1_expectancy"}
        )
        rank_cmp = e0_rank_df.merge(e1_rank_df, on="signal_candidate_id", how="inner")
        rank_corr = spearman_rank_corr(rank_cmp["e0_expectancy"].to_numpy(dtype=float), rank_cmp["e1_expectancy"].to_numpy(dtype=float))
        if np.isfinite(rank_corr) and rank_corr >= 0.60:
            ranking_hold = "yes"
        elif np.isfinite(rank_corr) and rank_corr <= 0.30:
            ranking_hold = "no"
        else:
            ranking_hold = "partially"

        # classify
        valid_non_e0_count = int(to_num(pilot_df[pilot_df["exec_choice_id"] != "E0"]["valid_for_ranking"]).fillna(0).sum())
        eval_count = int(len(pilot_df))
        nondup_count = int(len(nond))

        best_nondup = choose_best_row(valid_non_e0 if not valid_non_e0.empty else shortlist[shortlist["exec_choice_id"] != "E0"])
        best_expect = row_val(best_nondup, "overall_exec_expectancy_net") if best_nondup is not None else float("nan")
        best_vs_r0 = best_expect - row_val(r0, "overall_exec_expectancy_net")
        best_vs_r1 = best_expect - row_val(r1, "overall_exec_expectancy_net")

        diversity_ok = bool(nondup_count >= max(20, int(0.25 * eval_count)) and np.isfinite(n_eff_corr) and n_eff_corr >= 8.0)
        stability_ok = bool(r2 is not None and np.isfinite(row_val(r2, "min_split_expectancy_net")))

        if freeze_ok == 0:
            classification = "E"
            class_reason = "freeze_hash_lock_failed"
        elif valid_non_e0_count == 0:
            classification = "D"
            class_reason = "zero_valid_for_ranking_combined_candidates"
        else:
            # execution-only dominance
            if np.isfinite(exec_gain_base) and exec_gain_base > 0 and np.isfinite(signal_gain) and abs(signal_gain) <= abs(exec_gain_base) * 0.2:
                classification = "B"
                class_reason = "execution_routing_drives_gain_signal_variation_small"
            elif ranking_hold == "no":
                classification = "C"
                class_reason = "1h_rank_inversion_under_real_3m_execution"
            elif (
                np.isfinite(best_vs_r0)
                and np.isfinite(best_vs_r1)
                and best_vs_r0 > 0.0
                and best_vs_r1 >= -1e-9
                and diversity_ok
                and stability_ok
            ):
                classification = "A"
                class_reason = "combined_candidates_show_material_improvement_with_diversity"
            elif np.isfinite(exec_gain_base) and exec_gain_base > 0:
                classification = "B"
                class_reason = "execution_improves_but_combined_signal_search_not_superior"
            elif ranking_hold == "no":
                classification = "C"
                class_reason = "signal_objective_misaligned_to_execution_scored_outcome"
            else:
                classification = "D"
                class_reason = "no_meaningful_combined_progress"

        # root cause report
        root_lines: List[str] = []
        root_lines.append("# Phase U Root Cause Report")
        root_lines.append("")
        root_lines.append(f"- Generated UTC: {utc_now()}")
        root_lines.append(f"- Classification: **{classification}**")
        root_lines.append(f"- Reason: {class_reason}")
        root_lines.append("")
        root_lines.append("## Summary Stats")
        root_lines.append("")
        root_lines.append(f"- combined_evaluations_total: {eval_count}")
        root_lines.append(f"- valid_for_ranking_count_nonE0: {valid_non_e0_count}")
        root_lines.append(f"- nonduplicate_joint_count: {nondup_count}")
        root_lines.append(f"- effective_trials_corr_adjusted: {n_eff_corr:.6f}")
        root_lines.append(f"- ranking_hold_spearman_corr_e0_vs_e1: {rank_corr:.6f}" if np.isfinite(rank_corr) else "- ranking_hold_spearman_corr_e0_vs_e1: nan")
        root_lines.append(f"- ranking_hold_status: {ranking_hold}")
        root_lines.append(f"- best_expectancy_vs_R0: {best_vs_r0:.8f}" if np.isfinite(best_vs_r0) else "- best_expectancy_vs_R0: nan")
        root_lines.append(f"- best_expectancy_vs_R1: {best_vs_r1:.8f}" if np.isfinite(best_vs_r1) else "- best_expectancy_vs_R1: nan")
        root_lines.append("")
        root_lines.append("## Failure Clusters")
        root_lines.append("")
        if invalid_hist:
            top_items = sorted(invalid_hist.items(), key=lambda kv: kv[1], reverse=True)[:10]
            for k, v in top_items:
                root_lines.append(f"- {k}: {int(v)}")
        else:
            root_lines.append("- none")
        write_text(run_dir / "phaseU_root_cause_report.md", "\n".join(root_lines))

        # decision and prompts
        decision_lines: List[str] = []
        decision_lines.append("# Decision Next Step")
        decision_lines.append("")
        decision_lines.append(f"- Generated UTC: {utc_now()}")
        decision_lines.append(f"- Phase U classification: **{classification}**")
        decision_lines.append(f"- Reason: {class_reason}")
        decision_lines.append(f"- valid_for_ranking_nonE0: {valid_non_e0_count}")
        decision_lines.append(f"- ranking_hold_under_3m: {ranking_hold}")
        decision_lines.append("- Status: NO_DEPLOY (research validation phase only).")
        write_text(run_dir / "decision_next_step.md", "\n".join(decision_lines))

        if classification == "A":
            next_prompt = """Combined Phase U+ expansion (SOLUSDT, contract-locked): keep frozen representative subset + canonical fee/metrics hash lock unchanged. Run a larger combined search with outer 1h signal local neighborhood expansion (2x current candidate count) and discrete execution gene constrained to E1/E2 plus one nearby execution micro-subspace; keep hard execution gates unchanged. Maintain duplicate collapse, effective-trials accounting, and PSR/DSR shortlist screening. Stop early if valid_for_ranking density drops below pilot baseline. Deliver route decomposition R0-R4 again and classify GO_OOS_COMBINED_CONFIRMATION vs ITERATE_OBJECTIVE."""
            failure_prompt = """Failure branch not applicable: Phase U classified COMBINED_GO_LOCAL. If a rerun fails lock checks, stop immediately and emit NO_GO_INFRA with exact mismatch diagnostics."""
        elif classification == "B":
            next_prompt = """Execution-only continuation (SOLUSDT, contract-locked): freeze current 1h signal definition and run execution-layer objective/sampler refinement around E1/E2 only. Keep hard gates unchanged and prioritize robustness (split stability + cvar/maxdd retention) over nominal expectancy. Include duplicate-adjusted effective trials and stress micro-matrix before any larger GA budget."""
            failure_prompt = """Failure branch (B): do not run full combined GA. If execution-only refinement cannot improve R1 robustness with unchanged gates, classify NO_GO_COMPUTE_BURN and stop this branch."""
        elif classification == "C":
            next_prompt = """Signal-objective redesign (SOLUSDT, contract-locked): rebuild 1h GA objective so candidate ranking is scored through fixed execution proxy E1 (not standalone 1h expectancy). Keep execution hard gates/locks unchanged. Run a small objective-validation batch first, then reassess ranking consistency vs E0/E1 before any expanded combined search."""
            failure_prompt = """Failure branch (C): if redesigned 1h objective still shows rank inversion under E1 scoring, halt combined search and freeze SOL branch as NO_GO until signal family is redefined."""
        elif classification == "E":
            next_prompt = """NO_GO_INFRA re-entry: fix contract-lock/harness mismatch first. Verify canonical fee/metrics hashes and representative subset path wiring with allow_freeze_hash_mismatch=0, then rerun only U1 smoke until freeze_lock_pass=1 and no integration errors."""
            failure_prompt = """Failure branch (E): do not run U2/U3. Deliver minimal reproducible command + stack trace + patch list, then stop mainline progression."""
        else:
            next_prompt = """NO_GO_COMPUTE_BURN branch: keep frozen locks and stop large combined compute. Run a targeted forensic batch on top 10 pilot candidates to isolate why economics stay weak (cost drag vs signal timing vs split instability), then decide between objective redesign and branch kill."""
            failure_prompt = """Failure branch (D): if forensic batch confirms weak economics with unchanged gates, terminate combined SOL branch for now and allocate research budget to alternative symbols/families."""

        write_text(run_dir / "ready_to_launch_next_prompt.txt", next_prompt)
        write_text(run_dir / "ready_to_launch_failure_branch_prompt.txt", failure_prompt)

        # final outputs
        pilot_df = pilot_df.sort_values(["signal_candidate_id", "exec_choice_id"]).reset_index(drop=True)
        pilot_df.to_csv(run_dir / "phaseU_combined_pilot_results.csv", index=False)

        manifest.update(
            {
                "completed_utc": utc_now(),
                "duration_sec": float(time.time() - t_start),
                "classification": classification,
                "classification_reason": class_reason,
                "freeze_global": {
                    "representative_subset_csv": str(rep_csv),
                    "canonical_fee_model": str(fee_path),
                    "canonical_metrics_definition": str(metrics_path),
                    "canonical_fee_sha256": fee_sha,
                    "canonical_metrics_sha256": metrics_sha,
                    "expected_fee_sha256": LOCKED["expected_fee_sha"],
                    "expected_metrics_sha256": LOCKED["expected_metrics_sha"],
                    "freeze_hash_match": int(freeze_ok),
                    "freeze_lock_local": freeze_lock_local,
                },
                "entrypoints": {
                    "signal_1h": "src/bot087/optim/ga.py::build_entry_signal",
                    "execution_3m": "src/execution/ga_exec_3m_opt.py::_evaluate_genome",
                },
                "active_params": {
                    "path": str(params_path),
                    "path_sha256": sha256_file(params_path),
                    "meta": params_meta,
                },
                "exec_choices": [
                    {
                        "exec_choice_id": c.exec_choice_id,
                        "description": c.description,
                        "genome_hash": c.genome_hash,
                        "source": c.source,
                    }
                    for c in exec_choices
                ],
                "source_phase_s": exec_meta,
                "candidate_budget": {
                    "signal_candidates": int(len(candidates_all)),
                    "execution_choices": int(len(exec_choices)),
                    "combined_evaluations": int(len(pilot_df)),
                    "nonduplicate_joint": int(len(nond)),
                },
                "stats": {
                    "valid_for_ranking_non_e0": int(valid_non_e0_count),
                    "effective_trials_uncorrelated": float(n_uncorr),
                    "effective_trials_corr_adjusted": float(n_eff_corr),
                    "avg_abs_corr": float(avg_abs_corr),
                    "ranking_hold_status": ranking_hold,
                    "ranking_spearman_corr_e0_vs_e1": float(rank_corr) if np.isfinite(rank_corr) else None,
                },
                "files": {
                    "phaseU_harness_validation": str(run_dir / "phaseU_harness_validation.md"),
                    "phaseU_smoke_results": str(run_dir / "phaseU_smoke_results.csv"),
                    "phaseU_combined_pilot_results": str(run_dir / "phaseU_combined_pilot_results.csv"),
                    "phaseU_invalid_reason_histogram": str(run_dir / "phaseU_invalid_reason_histogram.json"),
                    "phaseU_duplicate_variant_map": str(run_dir / "phaseU_duplicate_variant_map.csv"),
                    "phaseU_effective_trials_summary": str(run_dir / "phaseU_effective_trials_summary.md"),
                    "phaseU_shortlist_significance": str(run_dir / "phaseU_shortlist_significance.csv"),
                    "phaseU_route_comparison": str(run_dir / "phaseU_route_comparison.csv"),
                    "phaseU_improvement_decomposition": str(run_dir / "phaseU_improvement_decomposition.md"),
                    "phaseU_root_cause_report": str(run_dir / "phaseU_root_cause_report.md"),
                    "decision_next_step": str(run_dir / "decision_next_step.md"),
                    "ready_to_launch_next_prompt": str(run_dir / "ready_to_launch_next_prompt.txt"),
                    "ready_to_launch_failure_branch_prompt": str(run_dir / "ready_to_launch_failure_branch_prompt.txt"),
                },
            }
        )
        json_dump(run_dir / "phaseU_run_manifest.json", manifest)
        return manifest

    except Exception as exc:
        # mandatory NO_GO_INFRA branch
        err_msg = f"{type(exc).__name__}: {exc}"
        manifest.update(
            {
                "completed_utc": utc_now(),
                "duration_sec": float(time.time() - t_start),
                "classification": "E",
                "classification_reason": "infra_exception",
                "error": err_msg,
            }
        )
        write_text(
            run_dir / "phaseU_harness_validation.md",
            "\n".join(
                [
                    "# Phase U Harness Validation",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    "- Result: FAILED",
                    f"- Error: {err_msg}",
                    "- Classification forced: E (NO_GO_INFRA)",
                ]
            ),
        )
        pd.DataFrame(
            [
                {
                    "signal_candidate_id": "",
                    "candidate_name": "",
                    "signals_active": 0,
                    "exec_choice_id": "",
                    "valid_for_ranking": 0,
                    "invalid_reason": "infra_exception",
                }
            ]
        ).to_csv(run_dir / "phaseU_smoke_results.csv", index=False)
        pd.DataFrame().to_csv(run_dir / "phaseU_combined_pilot_results.csv", index=False)
        json_dump(run_dir / "phaseU_invalid_reason_histogram.json", {"infra_exception": 1})
        pd.DataFrame().to_csv(run_dir / "phaseU_duplicate_variant_map.csv", index=False)
        write_text(
            run_dir / "phaseU_effective_trials_summary.md",
            "# Phase U Effective Trials Summary\n\n- Not available due to NO_GO_INFRA.",
        )
        pd.DataFrame().to_csv(run_dir / "phaseU_shortlist_significance.csv", index=False)
        pd.DataFrame().to_csv(run_dir / "phaseU_route_comparison.csv", index=False)
        write_text(
            run_dir / "phaseU_improvement_decomposition.md",
            "# Phase U Improvement Decomposition\n\n- Not available due to NO_GO_INFRA.",
        )
        write_text(
            run_dir / "phaseU_root_cause_report.md",
            "\n".join(
                [
                    "# Phase U Root Cause Report",
                    "",
                    "- Classification: **E**",
                    "- Reason: infrastructure/harness mismatch blocked valid conclusion.",
                    f"- Error: {err_msg}",
                ]
            ),
        )
        write_text(
            run_dir / "decision_next_step.md",
            "\n".join(
                [
                    "# Decision Next Step",
                    "",
                    "- Phase U classification: **E (NO_GO_INFRA)**",
                    f"- Error: {err_msg}",
                    "- Status: stop U2/U3 and fix harness integration first.",
                ]
            ),
        )
        write_text(
            run_dir / "ready_to_launch_next_prompt.txt",
            "NO_GO_INFRA re-entry: fix canonical lock or harness integration issue exactly as reported in phaseU_harness_validation.md, rerun U1 smoke only, and stop if freeze lock still fails.",
        )
        write_text(
            run_dir / "ready_to_launch_failure_branch_prompt.txt",
            "Failure branch (E): do not continue to U2/U3. Provide minimal reproducible command, mismatch details, and patch list; then stop.",
        )
        json_dump(run_dir / "phaseU_run_manifest.json", manifest)
        return manifest


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase U combined 1h+3m contract-locked pilot for SOLUSDT")
    ap.add_argument("--seed", type=int, default=20260226)
    ap.add_argument("--signal-candidate-count", type=int, default=25, help="Number of reduced 1h candidates (combined evals = candidates * 5 exec choices)")
    ap.add_argument("--max-u2-minutes", type=float, default=75.0, help="Timebox for U2 evaluation loop")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    manifest = run(args)
    print(json.dumps({"run_dir": manifest.get("run_dir", ""), "classification": manifest.get("classification", "")}, sort_keys=True))


if __name__ == "__main__":
    main()
