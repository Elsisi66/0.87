#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports" / "execution_layer"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_af_ah_sizing_autorun as af  # noqa: E402
from scripts import phase_u_combined_1h3m_pilot as pu  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402

LOCKED = {
    "representative_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
    "allow_freeze_hash_mismatch": 0,
}

FAMILY_A = "PASSIVE_LADDER_ADAPTIVE"
FAMILY_B = "REGIME_ROUTED_EXEC"
FAMILY_C = "STAGED_ENTRY_RISKSHAPE"


@dataclass
class FamilyVariant:
    family_id: str
    variant_id: str
    profile: str
    params: Dict[str, Any]
    base_genome: Dict[str, Any]
    is_upper_bound: int = 0
    seed_origin: str = "manual"


@dataclass
class EvalArtifacts:
    metrics: Dict[str, Any]
    split_df: pd.DataFrame
    symbol_df: pd.DataFrame
    signal_df: pd.DataFrame


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def safe_div(a: float, b: float) -> float:
    if (not np.isfinite(a)) or (not np.isfinite(b)) or abs(float(b)) <= 1e-12:
        return float("nan")
    return float(a / b)


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
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(v, utc=True))
        if isinstance(v, Path):
            return str(v)
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


def git_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        head = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
        out["git_head"] = head
    except Exception:
        out["git_head"] = "unavailable"
    try:
        status = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "status", "--short"], text=True)
        out["git_status_short"] = status.strip().splitlines()
    except Exception:
        out["git_status_short"] = []
    return out


def build_exec_args(signals_csv: Path, seed: int) -> argparse.Namespace:
    ap = ga_exec.build_arg_parser()
    args = ap.parse_args([])
    args.symbol = "SOLUSDT"
    args.symbols = ""
    args.rank = 1
    args.signals_csv = str(signals_csv)
    args.max_signals = 100000
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
    args.allow_freeze_hash_mismatch = int(LOCKED["allow_freeze_hash_mismatch"])
    return args


def extract_base_genome() -> Dict[str, Any]:
    try:
        choices, _meta = pu.load_exec_choices(REPORTS_ROOT)
        ch = {c.exec_choice_id: c for c in choices}
        if "E1" in ch and ch["E1"].genome:
            return ga_exec._repair_genome(copy.deepcopy(ch["E1"].genome), mode="tight", repair_hist=None)
    except Exception:
        pass
    fallback = {
        "entry_mode": "hybrid",
        "limit_offset_bps": 1.0,
        "max_fill_delay_min": 24,
        "fallback_to_market": 1,
        "fallback_delay_min": 2,
        "max_taker_share": 0.25,
        "micro_vol_filter": 0,
        "vol_threshold": 3.0,
        "spread_guard_bps": 80.0,
        "killzone_filter": 0,
        "mss_displacement_gate": 0,
        "min_entry_improvement_bps_gate": 0.0,
        "tp_mult": 1.0,
        "sl_mult": 1.0,
        "time_stop_min": 240,
        "break_even_enabled": 1,
        "break_even_trigger_r": 0.75,
        "break_even_offset_bps": 0.0,
        "trailing_enabled": 1,
        "trail_start_r": 1.0,
        "trail_step_bps": 12.0,
        "partial_take_enabled": 1,
        "partial_take_r": 0.8,
        "partial_take_pct": 0.5,
        "skip_if_vol_gate": 0,
        "use_signal_quality_gate": 0,
        "min_signal_quality_gate": 0.0,
        "cooldown_min": 0,
    }
    return ga_exec._repair_genome(fallback, mode="tight", repair_hist=None)


def family_param_bounds() -> Dict[str, Any]:
    return {
        FAMILY_A: {
            "ladder_steps": [2, 3],
            "base_limit_offset_bps": [0.1, 3.0],
            "ladder_step_bps": [0.1, 2.5],
            "step_delay_min": [3, 12],
            "max_fill_delay_min": [12, 45],
            "fallback_policy": ["always", "conditional", "never"],
            "fallback_delay_min": [1, 8],
            "vol_adapt_strength": [0.0, 0.8],
            "spread_adapt_strength": [0.0, 0.8],
            "fallback_toxicity_max": [0.30, 0.95],
            "micro_vol_z_max": [2.5, 6.0],
            "skip_if_micro_vol": [0, 1],
            "max_taker_share": [0.10, 0.25],
        },
        FAMILY_B: {
            "vol_z_low": [0.2, 1.2],
            "vol_z_high": [1.5, 3.5],
            "spread_low_bps": [20.0, 45.0],
            "spread_high_bps": [45.0, 90.0],
            "calm_mode": ["limit", "hybrid"],
            "calm_limit_offset_bps": [0.1, 2.0],
            "calm_max_fill_delay_min": [18, 45],
            "calm_fallback_delay_min": [1, 6],
            "neutral_mode": ["hybrid", "limit"],
            "neutral_limit_offset_bps": [0.2, 4.0],
            "neutral_max_fill_delay_min": [12, 30],
            "neutral_fallback_delay_min": [1, 6],
            "toxic_mode": ["market", "hybrid", "limit"],
            "toxic_limit_offset_bps": [0.0, 2.5],
            "toxic_max_fill_delay_min": [3, 18],
            "toxic_fallback_delay_min": [0, 4],
            "toxic_skip_extreme": [0, 1],
            "extreme_vol_z": [3.5, 6.0],
            "extreme_spread_bps": [70.0, 140.0],
            "max_taker_share": [0.10, 0.25],
        },
        FAMILY_C: {
            "stage1_mode": ["limit", "hybrid", "market"],
            "stage1_frac": [0.35, 0.80],
            "stage1_limit_offset_bps": [0.0, 2.5],
            "stage1_max_fill_delay_min": [3, 18],
            "stage2_mode": ["limit", "hybrid", "market"],
            "stage2_delay_min": [3, 18],
            "stage2_limit_offset_bps": [0.0, 4.0],
            "stage2_max_fill_delay_min": [6, 36],
            "fallback_policy": ["always", "conditional", "never"],
            "fallback_delay_min": [1, 8],
            "fallback_toxicity_max": [0.30, 0.95],
            "soft_risk_scale": [0.0, 1.0],
            "soft_risk_floor_frac": [0.15, 0.95],
            "micro_vol_z_max": [2.5, 6.0],
            "skip_if_micro_vol": [0, 1],
            "max_taker_share": [0.10, 0.25],
        },
    }


def variant_hash(v: FamilyVariant) -> str:
    payload = {
        "family_id": v.family_id,
        "params": v.params,
        "base_genome": v.base_genome,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))[:24]


def _pick_from_range(rng: random.Random, lo: float, hi: float, mode: str) -> float:
    if mode == "conservative":
        q0, q1 = 0.15, 0.45
    elif mode == "aggressive":
        q0, q1 = 0.55, 0.95
    else:
        q0, q1 = 0.35, 0.65
    q = rng.uniform(q0, q1)
    return float(lo + q * (hi - lo))


def _pick_int(rng: random.Random, lo: int, hi: int, mode: str) -> int:
    return int(round(_pick_from_range(rng, float(lo), float(hi), mode)))


def _sample_family_params(family_id: str, rng: random.Random, profile: str) -> Dict[str, Any]:
    if family_id == FAMILY_A:
        steps = 2 if rng.random() < (0.75 if profile != "aggressive" else 0.40) else 3
        p = {
            "ladder_steps": int(steps),
            "base_limit_offset_bps": _pick_from_range(rng, 0.1, 3.0, profile),
            "ladder_step_bps": _pick_from_range(rng, 0.1, 2.5, profile),
            "step_delay_min": _pick_int(rng, 3, 12, profile),
            "max_fill_delay_min": _pick_int(rng, 12, 45, "aggressive" if profile == "aggressive" else "mid"),
            "fallback_policy": "always" if profile == "conservative" else ("conditional" if rng.random() < 0.7 else "always"),
            "fallback_delay_min": _pick_int(rng, 1, 8, profile),
            "vol_adapt_strength": _pick_from_range(rng, 0.1, 0.8, profile),
            "spread_adapt_strength": _pick_from_range(rng, 0.1, 0.8, profile),
            "fallback_toxicity_max": _pick_from_range(rng, 0.45, 0.95, profile),
            "micro_vol_z_max": _pick_from_range(rng, 2.5, 6.0, profile),
            "skip_if_micro_vol": 0 if profile != "aggressive" else int(rng.random() < 0.30),
            "max_taker_share": _pick_from_range(rng, 0.10, 0.25, profile),
            "cooldown_min": _pick_int(rng, 0, 20, profile),
        }
        p["fallback_delay_min"] = min(int(p["fallback_delay_min"]), int(p["max_fill_delay_min"]))
        return p

    if family_id == FAMILY_B:
        p = {
            "vol_z_low": _pick_from_range(rng, 0.2, 1.2, profile),
            "vol_z_high": _pick_from_range(rng, 1.5, 3.5, profile),
            "spread_low_bps": _pick_from_range(rng, 20.0, 45.0, profile),
            "spread_high_bps": _pick_from_range(rng, 45.0, 90.0, profile),
            "calm_mode": "limit" if rng.random() < 0.5 else "hybrid",
            "calm_limit_offset_bps": _pick_from_range(rng, 0.1, 2.0, profile),
            "calm_max_fill_delay_min": _pick_int(rng, 18, 45, profile),
            "calm_fallback_delay_min": _pick_int(rng, 1, 6, profile),
            "neutral_mode": "hybrid" if rng.random() < 0.8 else "limit",
            "neutral_limit_offset_bps": _pick_from_range(rng, 0.2, 4.0, profile),
            "neutral_max_fill_delay_min": _pick_int(rng, 12, 30, profile),
            "neutral_fallback_delay_min": _pick_int(rng, 1, 6, profile),
            "toxic_mode": "market" if profile != "aggressive" else random.choice(["market", "hybrid"]),
            "toxic_limit_offset_bps": _pick_from_range(rng, 0.0, 2.5, profile),
            "toxic_max_fill_delay_min": _pick_int(rng, 3, 18, profile),
            "toxic_fallback_delay_min": _pick_int(rng, 0, 4, profile),
            "toxic_skip_extreme": int(rng.random() < (0.05 if profile == "conservative" else 0.20)),
            "extreme_vol_z": _pick_from_range(rng, 3.5, 6.0, profile),
            "extreme_spread_bps": _pick_from_range(rng, 70.0, 140.0, profile),
            "max_taker_share": _pick_from_range(rng, 0.10, 0.25, profile),
            "cooldown_min": _pick_int(rng, 0, 20, profile),
        }
        if p["vol_z_low"] >= p["vol_z_high"]:
            p["vol_z_high"] = float(p["vol_z_low"] + 0.7)
        if p["spread_low_bps"] >= p["spread_high_bps"]:
            p["spread_high_bps"] = float(p["spread_low_bps"] + 20.0)
        return p

    if family_id == FAMILY_C:
        s1 = _pick_from_range(rng, 0.35, 0.80, profile)
        p = {
            "stage1_mode": random.choice(["limit", "hybrid"]) if profile != "aggressive" else random.choice(["limit", "hybrid", "market"]),
            "stage1_frac": float(s1),
            "stage1_limit_offset_bps": _pick_from_range(rng, 0.0, 2.5, profile),
            "stage1_max_fill_delay_min": _pick_int(rng, 3, 18, profile),
            "stage2_mode": random.choice(["limit", "hybrid", "market"]),
            "stage2_delay_min": _pick_int(rng, 3, 18, profile),
            "stage2_limit_offset_bps": _pick_from_range(rng, 0.0, 4.0, profile),
            "stage2_max_fill_delay_min": _pick_int(rng, 6, 36, profile),
            "fallback_policy": "always" if profile == "conservative" else ("conditional" if rng.random() < 0.7 else "always"),
            "fallback_delay_min": _pick_int(rng, 1, 8, profile),
            "fallback_toxicity_max": _pick_from_range(rng, 0.45, 0.95, profile),
            "soft_risk_scale": _pick_from_range(rng, 0.0, 1.0, profile),
            "soft_risk_floor_frac": _pick_from_range(rng, 0.15, 0.95, profile),
            "micro_vol_z_max": _pick_from_range(rng, 2.5, 6.0, profile),
            "skip_if_micro_vol": 0 if profile != "aggressive" else int(rng.random() < 0.25),
            "max_taker_share": _pick_from_range(rng, 0.10, 0.25, profile),
            "cooldown_min": _pick_int(rng, 0, 20, profile),
        }
        return p

    raise ValueError(f"Unknown family_id={family_id}")


def build_nx2_variants(base_genome: Dict[str, Any]) -> List[FamilyVariant]:
    out: List[FamilyVariant] = []

    out.extend(
        [
            FamilyVariant(
                family_id=FAMILY_A,
                variant_id="A_conservative",
                profile="conservative",
                base_genome=copy.deepcopy(base_genome),
                params={
                    "ladder_steps": 2,
                    "base_limit_offset_bps": 0.8,
                    "ladder_step_bps": 0.5,
                    "step_delay_min": 6,
                    "max_fill_delay_min": 24,
                    "fallback_policy": "always",
                    "fallback_delay_min": 2,
                    "vol_adapt_strength": 0.35,
                    "spread_adapt_strength": 0.35,
                    "fallback_toxicity_max": 0.90,
                    "micro_vol_z_max": 5.5,
                    "skip_if_micro_vol": 0,
                    "max_taker_share": 0.20,
                    "cooldown_min": 4,
                },
            ),
            FamilyVariant(
                family_id=FAMILY_A,
                variant_id="A_mid",
                profile="mid",
                base_genome=copy.deepcopy(base_genome),
                params={
                    "ladder_steps": 3,
                    "base_limit_offset_bps": 1.2,
                    "ladder_step_bps": 0.8,
                    "step_delay_min": 6,
                    "max_fill_delay_min": 30,
                    "fallback_policy": "conditional",
                    "fallback_delay_min": 3,
                    "vol_adapt_strength": 0.50,
                    "spread_adapt_strength": 0.55,
                    "fallback_toxicity_max": 0.72,
                    "micro_vol_z_max": 4.2,
                    "skip_if_micro_vol": 0,
                    "max_taker_share": 0.22,
                    "cooldown_min": 8,
                },
            ),
            FamilyVariant(
                family_id=FAMILY_A,
                variant_id="A_aggressive",
                profile="aggressive",
                base_genome=copy.deepcopy(base_genome),
                params={
                    "ladder_steps": 3,
                    "base_limit_offset_bps": 2.0,
                    "ladder_step_bps": 1.4,
                    "step_delay_min": 9,
                    "max_fill_delay_min": 33,
                    "fallback_policy": "conditional",
                    "fallback_delay_min": 5,
                    "vol_adapt_strength": 0.75,
                    "spread_adapt_strength": 0.70,
                    "fallback_toxicity_max": 0.55,
                    "micro_vol_z_max": 3.5,
                    "skip_if_micro_vol": 1,
                    "max_taker_share": 0.17,
                    "cooldown_min": 12,
                },
            ),
            FamilyVariant(
                family_id=FAMILY_A,
                variant_id="A_upper_bound",
                profile="upper_bound",
                base_genome=copy.deepcopy(base_genome),
                is_upper_bound=1,
                params={
                    "ladder_steps": 2,
                    "base_limit_offset_bps": 0.1,
                    "ladder_step_bps": 0.2,
                    "step_delay_min": 3,
                    "max_fill_delay_min": 45,
                    "fallback_policy": "always",
                    "fallback_delay_min": 1,
                    "vol_adapt_strength": 0.15,
                    "spread_adapt_strength": 0.10,
                    "fallback_toxicity_max": 0.95,
                    "micro_vol_z_max": 6.0,
                    "skip_if_micro_vol": 0,
                    "max_taker_share": 0.25,
                    "cooldown_min": 0,
                },
            ),
        ]
    )

    out.extend(
        [
            FamilyVariant(
                family_id=FAMILY_B,
                variant_id="B_conservative",
                profile="conservative",
                base_genome=copy.deepcopy(base_genome),
                params={
                    "vol_z_low": 0.5,
                    "vol_z_high": 2.2,
                    "spread_low_bps": 30.0,
                    "spread_high_bps": 65.0,
                    "calm_mode": "limit",
                    "calm_limit_offset_bps": 0.7,
                    "calm_max_fill_delay_min": 30,
                    "calm_fallback_delay_min": 2,
                    "neutral_mode": "hybrid",
                    "neutral_limit_offset_bps": 1.0,
                    "neutral_max_fill_delay_min": 21,
                    "neutral_fallback_delay_min": 2,
                    "toxic_mode": "market",
                    "toxic_limit_offset_bps": 0.0,
                    "toxic_max_fill_delay_min": 3,
                    "toxic_fallback_delay_min": 0,
                    "toxic_skip_extreme": 0,
                    "extreme_vol_z": 5.0,
                    "extreme_spread_bps": 120.0,
                    "max_taker_share": 0.25,
                    "cooldown_min": 2,
                },
            ),
            FamilyVariant(
                family_id=FAMILY_B,
                variant_id="B_mid",
                profile="mid",
                base_genome=copy.deepcopy(base_genome),
                params={
                    "vol_z_low": 0.6,
                    "vol_z_high": 2.0,
                    "spread_low_bps": 28.0,
                    "spread_high_bps": 60.0,
                    "calm_mode": "limit",
                    "calm_limit_offset_bps": 1.0,
                    "calm_max_fill_delay_min": 27,
                    "calm_fallback_delay_min": 3,
                    "neutral_mode": "hybrid",
                    "neutral_limit_offset_bps": 1.3,
                    "neutral_max_fill_delay_min": 18,
                    "neutral_fallback_delay_min": 2,
                    "toxic_mode": "hybrid",
                    "toxic_limit_offset_bps": 0.4,
                    "toxic_max_fill_delay_min": 9,
                    "toxic_fallback_delay_min": 1,
                    "toxic_skip_extreme": 0,
                    "extreme_vol_z": 4.5,
                    "extreme_spread_bps": 100.0,
                    "max_taker_share": 0.23,
                    "cooldown_min": 6,
                },
            ),
            FamilyVariant(
                family_id=FAMILY_B,
                variant_id="B_aggressive",
                profile="aggressive",
                base_genome=copy.deepcopy(base_genome),
                params={
                    "vol_z_low": 0.8,
                    "vol_z_high": 1.8,
                    "spread_low_bps": 24.0,
                    "spread_high_bps": 55.0,
                    "calm_mode": "hybrid",
                    "calm_limit_offset_bps": 1.6,
                    "calm_max_fill_delay_min": 22,
                    "calm_fallback_delay_min": 3,
                    "neutral_mode": "hybrid",
                    "neutral_limit_offset_bps": 1.8,
                    "neutral_max_fill_delay_min": 15,
                    "neutral_fallback_delay_min": 3,
                    "toxic_mode": "limit",
                    "toxic_limit_offset_bps": 1.0,
                    "toxic_max_fill_delay_min": 12,
                    "toxic_fallback_delay_min": 3,
                    "toxic_skip_extreme": 1,
                    "extreme_vol_z": 4.0,
                    "extreme_spread_bps": 90.0,
                    "max_taker_share": 0.18,
                    "cooldown_min": 12,
                },
            ),
            FamilyVariant(
                family_id=FAMILY_B,
                variant_id="B_upper_bound",
                profile="upper_bound",
                base_genome=copy.deepcopy(base_genome),
                is_upper_bound=1,
                params={
                    "vol_z_low": 0.3,
                    "vol_z_high": 2.8,
                    "spread_low_bps": 35.0,
                    "spread_high_bps": 85.0,
                    "calm_mode": "limit",
                    "calm_limit_offset_bps": 0.2,
                    "calm_max_fill_delay_min": 45,
                    "calm_fallback_delay_min": 1,
                    "neutral_mode": "hybrid",
                    "neutral_limit_offset_bps": 0.4,
                    "neutral_max_fill_delay_min": 30,
                    "neutral_fallback_delay_min": 1,
                    "toxic_mode": "market",
                    "toxic_limit_offset_bps": 0.0,
                    "toxic_max_fill_delay_min": 3,
                    "toxic_fallback_delay_min": 0,
                    "toxic_skip_extreme": 0,
                    "extreme_vol_z": 6.0,
                    "extreme_spread_bps": 140.0,
                    "max_taker_share": 0.25,
                    "cooldown_min": 0,
                },
            ),
        ]
    )

    out.extend(
        [
            FamilyVariant(
                family_id=FAMILY_C,
                variant_id="C_conservative",
                profile="conservative",
                base_genome=copy.deepcopy(base_genome),
                params={
                    "stage1_mode": "limit",
                    "stage1_frac": 0.65,
                    "stage1_limit_offset_bps": 0.8,
                    "stage1_max_fill_delay_min": 12,
                    "stage2_mode": "hybrid",
                    "stage2_delay_min": 6,
                    "stage2_limit_offset_bps": 1.2,
                    "stage2_max_fill_delay_min": 18,
                    "fallback_policy": "always",
                    "fallback_delay_min": 2,
                    "fallback_toxicity_max": 0.90,
                    "soft_risk_scale": 0.35,
                    "soft_risk_floor_frac": 0.55,
                    "micro_vol_z_max": 5.5,
                    "skip_if_micro_vol": 0,
                    "max_taker_share": 0.22,
                    "cooldown_min": 4,
                },
            ),
            FamilyVariant(
                family_id=FAMILY_C,
                variant_id="C_mid",
                profile="mid",
                base_genome=copy.deepcopy(base_genome),
                params={
                    "stage1_mode": "limit",
                    "stage1_frac": 0.52,
                    "stage1_limit_offset_bps": 1.0,
                    "stage1_max_fill_delay_min": 9,
                    "stage2_mode": "market",
                    "stage2_delay_min": 6,
                    "stage2_limit_offset_bps": 0.0,
                    "stage2_max_fill_delay_min": 6,
                    "fallback_policy": "conditional",
                    "fallback_delay_min": 3,
                    "fallback_toxicity_max": 0.70,
                    "soft_risk_scale": 0.60,
                    "soft_risk_floor_frac": 0.40,
                    "micro_vol_z_max": 4.5,
                    "skip_if_micro_vol": 0,
                    "max_taker_share": 0.23,
                    "cooldown_min": 8,
                },
            ),
            FamilyVariant(
                family_id=FAMILY_C,
                variant_id="C_aggressive",
                profile="aggressive",
                base_genome=copy.deepcopy(base_genome),
                params={
                    "stage1_mode": "hybrid",
                    "stage1_frac": 0.42,
                    "stage1_limit_offset_bps": 1.5,
                    "stage1_max_fill_delay_min": 9,
                    "stage2_mode": "limit",
                    "stage2_delay_min": 9,
                    "stage2_limit_offset_bps": 2.0,
                    "stage2_max_fill_delay_min": 21,
                    "fallback_policy": "conditional",
                    "fallback_delay_min": 5,
                    "fallback_toxicity_max": 0.55,
                    "soft_risk_scale": 0.90,
                    "soft_risk_floor_frac": 0.25,
                    "micro_vol_z_max": 3.5,
                    "skip_if_micro_vol": 1,
                    "max_taker_share": 0.17,
                    "cooldown_min": 12,
                },
            ),
            FamilyVariant(
                family_id=FAMILY_C,
                variant_id="C_upper_bound",
                profile="upper_bound",
                base_genome=copy.deepcopy(base_genome),
                is_upper_bound=1,
                params={
                    "stage1_mode": "hybrid",
                    "stage1_frac": 0.55,
                    "stage1_limit_offset_bps": 0.2,
                    "stage1_max_fill_delay_min": 15,
                    "stage2_mode": "market",
                    "stage2_delay_min": 3,
                    "stage2_limit_offset_bps": 0.0,
                    "stage2_max_fill_delay_min": 6,
                    "fallback_policy": "always",
                    "fallback_delay_min": 1,
                    "fallback_toxicity_max": 0.95,
                    "soft_risk_scale": 0.20,
                    "soft_risk_floor_frac": 0.70,
                    "micro_vol_z_max": 6.0,
                    "skip_if_micro_vol": 0,
                    "max_taker_share": 0.25,
                    "cooldown_min": 0,
                },
            ),
        ]
    )

    return out


def build_ablation_variants(base_genome: Dict[str, Any], seed: int, n_per_family: int) -> List[FamilyVariant]:
    rng = random.Random(seed)
    out: List[FamilyVariant] = []
    for fam in [FAMILY_A, FAMILY_B, FAMILY_C]:
        n_cons = max(3, int(math.floor(n_per_family / 3)))
        n_mid = max(3, int(math.floor(n_per_family / 3)))
        n_agg = max(3, int(n_per_family - n_cons - n_mid))
        for profile, n in [("conservative", n_cons), ("mid", n_mid), ("aggressive", n_agg)]:
            for i in range(n):
                p = _sample_family_params(fam, rng=rng, profile=profile)
                out.append(
                    FamilyVariant(
                        family_id=fam,
                        variant_id=f"{fam}_{profile}_{i:02d}",
                        profile=profile,
                        params=p,
                        base_genome=copy.deepcopy(base_genome),
                        is_upper_bound=0,
                        seed_origin="sampled_ablation",
                    )
                )
    return out


def build_route_bundles(base_bundle: ga_exec.SymbolBundle, rep_subset: pd.DataFrame, args: argparse.Namespace) -> Dict[str, ga_exec.SymbolBundle]:
    route_sets = af.route_signal_sets(rep_subset)
    ctx_by_id = {str(c.signal_id): c for c in base_bundle.contexts}
    out: Dict[str, ga_exec.SymbolBundle] = {}
    for rid, rdf in route_sets.items():
        ids = rdf["signal_id"].astype(str).tolist() if "signal_id" in rdf.columns else []
        route_ctx = [ctx_by_id[sid] for sid in ids if sid in ctx_by_id]
        route_ctx = sorted(route_ctx, key=lambda z: (pd.to_datetime(z.signal_time, utc=True), str(z.signal_id)))
        splits = ga_exec._build_walkforward_splits(n=len(route_ctx), train_ratio=float(args.train_ratio), n_splits=int(args.wf_splits))
        out[rid] = ga_exec.SymbolBundle(
            symbol=base_bundle.symbol,
            signals_csv=base_bundle.signals_csv,
            contexts=route_ctx,
            splits=splits,
            constraints=dict(base_bundle.constraints),
        )
    return out


def _state_from_ctx(ctx: ga_exec.SignalContext) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": 0,
        "reason": "",
        "sig_idx": -1,
        "entry_ref": float("nan"),
        "atr_z": 0.0,
        "spread_bps": float("nan"),
        "toxicity": 0.5,
    }

    n = len(ctx.ts_ns)
    if n == 0:
        out["reason"] = "no_3m_data"
        return out

    sig_idx = int(np.searchsorted(ctx.ts_ns, int(ctx.signal_ts_ns), side="left"))
    if sig_idx >= n:
        out["reason"] = "no_bar_after_signal"
        return out

    entry_ref = float(ctx.open_np[sig_idx])
    if (not np.isfinite(entry_ref)) or entry_ref <= 0.0:
        out["reason"] = "bad_entry_ref"
        return out

    atr_ref_idx = max(0, sig_idx - 1)
    atr_ref = float(ctx.atr_np[atr_ref_idx]) if ctx.atr_np.size else float("nan")
    hist_start = max(0, atr_ref_idx - 7 * 24 * 20)
    hist = np.asarray(ctx.atr_np[hist_start:atr_ref_idx], dtype=float)
    hist = hist[np.isfinite(hist)]
    atr_z = 0.0
    if hist.size >= 30 and np.isfinite(atr_ref):
        mu = float(np.nanmean(hist))
        sd = float(np.nanstd(hist))
        atr_z = float((atr_ref - mu) / sd) if sd > 1e-12 else 0.0

    cl = float(ctx.close_np[sig_idx]) if sig_idx < len(ctx.close_np) else float("nan")
    hi = float(ctx.high_np[sig_idx]) if sig_idx < len(ctx.high_np) else float("nan")
    lo = float(ctx.low_np[sig_idx]) if sig_idx < len(ctx.low_np) else float("nan")
    spread = float((hi - lo) / max(1e-12, cl) * 1e4) if np.isfinite(cl) else float("nan")

    vol_term = clamp(max(0.0, float(atr_z)) / 4.0, 0.0, 1.0)
    spr_term = clamp(max(0.0, float(spread)) / 120.0, 0.0, 1.0) if np.isfinite(spread) else 0.5
    toxicity = float(clamp(0.55 * vol_term + 0.45 * spr_term, 0.0, 1.0))

    out.update(
        {
            "ok": 1,
            "reason": "",
            "sig_idx": int(sig_idx),
            "entry_ref": float(entry_ref),
            "atr_z": float(atr_z),
            "spread_bps": float(spread),
            "toxicity": float(toxicity),
        }
    )
    return out


def _first_limit_fill(low_np: np.ndarray, start_idx: int, end_idx: int, limit_px: float) -> Optional[int]:
    if start_idx > end_idx:
        return None
    s = max(0, int(start_idx))
    e = min(len(low_np) - 1, int(end_idx))
    for i in range(s, e + 1):
        lo = float(low_np[i])
        if np.isfinite(lo) and lo <= float(limit_px):
            return int(i)
    return None


def _to_delay_min(ctx: ga_exec.SignalContext, idx: int) -> float:
    if idx < 0 or idx >= len(ctx.ts_ns):
        return float("nan")
    t = pd.to_datetime(int(ctx.ts_ns[idx]), utc=True)
    return float((t - pd.to_datetime(ctx.signal_time, utc=True)).total_seconds() / 60.0)


def _costed_from_legs_mixed_entry(
    *,
    entry_price: float,
    legs: List[Tuple[float, float]],
    maker_frac: float,
    fee_bps_maker: float,
    fee_bps_taker: float,
    slippage_bps_limit: float,
    slippage_bps_market: float,
) -> Dict[str, float]:
    e = float(entry_price)
    if (not np.isfinite(e)) or e <= 0.0 or (not legs):
        return {
            "pnl_gross_pct": float("nan"),
            "pnl_net_pct": float("nan"),
            "entry_fee_bps": float("nan"),
            "exit_fee_bps": float("nan"),
            "entry_slippage_bps": float("nan"),
            "exit_slippage_bps": float("nan"),
            "total_cost_bps": float("nan"),
        }

    total_frac = sum(max(0.0, float(f)) for f, _ in legs)
    if total_frac <= 1e-12:
        return {
            "pnl_gross_pct": float("nan"),
            "pnl_net_pct": float("nan"),
            "entry_fee_bps": float("nan"),
            "exit_fee_bps": float("nan"),
            "entry_slippage_bps": float("nan"),
            "exit_slippage_bps": float("nan"),
            "total_cost_bps": float("nan"),
        }

    norm_legs = [(float(f) / total_frac, float(px)) for f, px in legs]
    gross = float(sum(f * ((px / e) - 1.0) for f, px in norm_legs if np.isfinite(px)))

    mf = clamp(float(maker_frac), 0.0, 1.0)
    tf = 1.0 - mf
    entry_fee_bps = float(mf * fee_bps_maker + tf * fee_bps_taker)
    entry_slip_bps = float(mf * slippage_bps_limit + tf * slippage_bps_market)

    exit_fee_bps = float(fee_bps_taker)
    exit_slip_bps = float(slippage_bps_market)

    entry_eff = float(e * (1.0 + entry_slip_bps / 1e4))
    exit_eff = 0.0
    for f, px in norm_legs:
        if np.isfinite(px):
            exit_eff += float(f) * float(px * (1.0 - exit_slip_bps / 1e4))

    net = float((exit_eff / entry_eff) - 1.0 - (entry_fee_bps + exit_fee_bps) / 1e4)
    return {
        "pnl_gross_pct": float(gross),
        "pnl_net_pct": float(net),
        "entry_fee_bps": float(entry_fee_bps),
        "exit_fee_bps": float(exit_fee_bps),
        "entry_slippage_bps": float(entry_slip_bps),
        "exit_slippage_bps": float(exit_slip_bps),
        "total_cost_bps": float((gross - net) * 1e4),
    }


def _family_threshold_genome(v: FamilyVariant) -> Dict[str, Any]:
    g = copy.deepcopy(v.base_genome)
    g["max_taker_share"] = float(v.params.get("max_taker_share", g.get("max_taker_share", 0.25)))
    g["min_entry_improvement_bps_gate"] = float(v.params.get("min_entry_improvement_bps_gate", g.get("min_entry_improvement_bps_gate", 0.0)))
    return ga_exec._repair_genome(g, mode="tight", repair_hist=None)


def _struct_fail_reasons(v: FamilyVariant) -> List[str]:
    p = v.params
    rs: List[str] = []
    if v.family_id == FAMILY_A:
        if int(p.get("ladder_steps", 2)) not in {2, 3}:
            rs.append("ladder_steps_out_of_range")
        if float(p.get("fallback_delay_min", 0.0)) > float(p.get("max_fill_delay_min", 0.0)):
            rs.append("fallback_delay_gt_max_fill")
    elif v.family_id == FAMILY_B:
        if float(p.get("vol_z_low", 0.0)) >= float(p.get("vol_z_high", 0.0)):
            rs.append("vol_threshold_order_invalid")
        if float(p.get("spread_low_bps", 0.0)) >= float(p.get("spread_high_bps", 0.0)):
            rs.append("spread_threshold_order_invalid")
    elif v.family_id == FAMILY_C:
        if float(p.get("stage1_frac", 0.5)) <= 0.0 or float(p.get("stage1_frac", 0.5)) >= 1.0:
            rs.append("stage1_frac_invalid")
        if float(p.get("fallback_delay_min", 0.0)) > max(float(p.get("stage1_max_fill_delay_min", 0.0)), float(p.get("stage2_max_fill_delay_min", 0.0))):
            rs.append("fallback_delay_gt_stage_fill")
    return rs


def _simulate_family_fill(
    *,
    ctx: ga_exec.SignalContext,
    state: Dict[str, Any],
    v: FamilyVariant,
) -> Dict[str, Any]:
    p = v.params
    sig_idx = int(state["sig_idx"])
    entry_ref = float(state["entry_ref"])
    atr_z = float(state["atr_z"])
    spread_bps = float(state["spread_bps"])
    tox = float(state["toxicity"])

    n = len(ctx.ts_ns)

    out = {
        "filled": False,
        "fill_idx": -1,
        "entry_price": float("nan"),
        "fill_type": "",
        "maker_frac": 0.0,
        "filled_frac": 0.0,
        "weighted_delay_min": float("nan"),
        "limit_placed": 0,
        "limit_filled": 0,
        "fallback_triggered": 0,
        "market_filled": 0,
        "state_skip_reason": "",
        "route_bucket": "",
    }

    # Shared micro-vol gate.
    mv_max = float(p.get("micro_vol_z_max", 9.0))
    if int(p.get("skip_if_micro_vol", 0)) == 1 and np.isfinite(atr_z) and atr_z > mv_max:
        out["state_skip_reason"] = "micro_vol_gate"
        return out

    if v.family_id == FAMILY_A:
        steps = int(p.get("ladder_steps", 2))
        base_off = float(p.get("base_limit_offset_bps", 0.5))
        step_off = float(p.get("ladder_step_bps", 0.5))
        step_delay_bars = max(1, int(math.ceil(float(p.get("step_delay_min", 6)) / 3.0)))
        max_fill_bars = max(0, int(math.ceil(float(p.get("max_fill_delay_min", 24)) / 3.0)))
        deadline = min(n - 1, sig_idx + max_fill_bars)

        vol_mult = 1.0 + float(p.get("vol_adapt_strength", 0.0)) * clamp(max(0.0, atr_z) / 3.0, 0.0, 1.0)
        spr_mult = 1.0 + float(p.get("spread_adapt_strength", 0.0)) * clamp(max(0.0, spread_bps) / 80.0, 0.0, 1.0)
        adapt_mult = clamp(vol_mult * spr_mult, 0.75, 2.0)

        best_idx: Optional[int] = None
        best_px = float("nan")
        for s in range(steps):
            st = sig_idx + s * step_delay_bars
            if st > deadline:
                break
            en = deadline if s == steps - 1 else min(deadline, st + step_delay_bars - 1)
            off = float((base_off + s * step_off) * adapt_mult)
            limit_px = float(entry_ref * (1.0 - off / 1e4))
            out["limit_placed"] += 1
            idx = _first_limit_fill(ctx.low_np, st, en, limit_px)
            if idx is not None:
                best_idx = int(idx)
                best_px = float(limit_px)
                break

        if best_idx is None:
            pol = str(p.get("fallback_policy", "conditional")).lower()
            do_fb = pol == "always"
            if pol == "conditional":
                do_fb = bool(tox <= float(p.get("fallback_toxicity_max", 0.75)))
            if do_fb:
                fb_bars = max(0, int(math.ceil(float(p.get("fallback_delay_min", 2)) / 3.0)))
                fb_idx = min(n - 1, sig_idx + fb_bars)
                best_idx = int(fb_idx)
                best_px = float(ctx.open_np[best_idx])
                out["fallback_triggered"] = 1
                out["market_filled"] = 1
                out["fill_type"] = "market_fallback"
            else:
                out["state_skip_reason"] = "timeout_no_fill"
                return out
        else:
            out["limit_filled"] = 1
            out["fill_type"] = "limit_ladder"

        out["filled"] = True
        out["fill_idx"] = int(best_idx)
        out["entry_price"] = float(best_px)
        out["maker_frac"] = 1.0 if out["market_filled"] == 0 else 0.0
        out["filled_frac"] = 1.0
        out["weighted_delay_min"] = _to_delay_min(ctx, int(best_idx))
        out["route_bucket"] = "ladder"
        return out

    if v.family_id == FAMILY_B:
        vol_lo = float(p.get("vol_z_low", 0.5))
        vol_hi = float(p.get("vol_z_high", 2.0))
        spr_lo = float(p.get("spread_low_bps", 30.0))
        spr_hi = float(p.get("spread_high_bps", 65.0))

        if (np.isfinite(atr_z) and atr_z >= vol_hi) or (np.isfinite(spread_bps) and spread_bps >= spr_hi):
            regime = "toxic"
        elif (np.isfinite(atr_z) and atr_z <= vol_lo) and (np.isfinite(spread_bps) and spread_bps <= spr_lo):
            regime = "calm"
        else:
            regime = "neutral"

        out["route_bucket"] = regime

        if int(p.get("toxic_skip_extreme", 0)) == 1 and regime == "toxic":
            if (np.isfinite(atr_z) and atr_z >= float(p.get("extreme_vol_z", 4.5))) and (
                np.isfinite(spread_bps) and spread_bps >= float(p.get("extreme_spread_bps", 100.0))
            ):
                out["state_skip_reason"] = "extreme_toxic_skip"
                return out

        mode = str(p.get(f"{regime}_mode", "hybrid")).lower()
        limit_off = float(p.get(f"{regime}_limit_offset_bps", 1.0))
        max_fill_bars = max(0, int(math.ceil(float(p.get(f"{regime}_max_fill_delay_min", 18)) / 3.0)))
        fallback_bars = max(0, int(math.ceil(float(p.get(f"{regime}_fallback_delay_min", 2)) / 3.0)))
        deadline = min(n - 1, sig_idx + max_fill_bars)

        if mode == "market":
            out["filled"] = True
            out["fill_idx"] = int(sig_idx)
            out["entry_price"] = float(ctx.open_np[sig_idx])
            out["fill_type"] = "market_regime"
            out["maker_frac"] = 0.0
            out["filled_frac"] = 1.0
            out["weighted_delay_min"] = 0.0
            out["market_filled"] = 1
            return out

        out["limit_placed"] = 1
        limit_px = float(entry_ref * (1.0 - limit_off / 1e4))
        idx = _first_limit_fill(ctx.low_np, sig_idx, deadline, limit_px)
        if idx is not None:
            out["filled"] = True
            out["fill_idx"] = int(idx)
            out["entry_price"] = float(limit_px)
            out["fill_type"] = "limit_regime"
            out["maker_frac"] = 1.0
            out["filled_frac"] = 1.0
            out["weighted_delay_min"] = _to_delay_min(ctx, int(idx))
            out["limit_filled"] = 1
            return out

        # fallback for limit/hybrid profile
        fb_idx = min(n - 1, sig_idx + fallback_bars)
        out["filled"] = True
        out["fill_idx"] = int(fb_idx)
        out["entry_price"] = float(ctx.open_np[fb_idx])
        out["fill_type"] = "market_fallback_regime"
        out["maker_frac"] = 0.0
        out["filled_frac"] = 1.0
        out["weighted_delay_min"] = _to_delay_min(ctx, int(fb_idx))
        out["fallback_triggered"] = 1
        out["market_filled"] = 1
        return out

    if v.family_id == FAMILY_C:
        stage1_frac = clamp(float(p.get("stage1_frac", 0.55)), 0.05, 0.95)
        soft_scale = clamp(float(p.get("soft_risk_scale", 0.5)), 0.0, 2.0)
        floor_frac = clamp(float(p.get("soft_risk_floor_frac", 0.35)), 0.0, 1.0)
        stage2_base = max(0.0, 1.0 - stage1_frac)
        stage2_scale = max(floor_frac, 1.0 - soft_scale * tox)
        stage2_frac = clamp(stage2_base * stage2_scale, 0.0, 1.0)
        stage1_alloc = clamp(1.0 - stage2_frac, 0.0, 1.0)

        def _simulate_stage(start_idx: int, mode: str, off_bps: float, max_delay_min: float, fallback_ok: bool) -> Dict[str, Any]:
            z = {
                "filled": False,
                "idx": -1,
                "px": float("nan"),
                "maker": 0.0,
                "limit_placed": 0,
                "limit_filled": 0,
                "fallback": 0,
                "market": 0,
            }
            st = min(max(0, int(start_idx)), n - 1)
            m = str(mode).lower()
            if m == "market":
                z.update({"filled": True, "idx": int(st), "px": float(ctx.open_np[st]), "maker": 0.0, "market": 1})
                return z
            max_bars = max(0, int(math.ceil(float(max_delay_min) / 3.0)))
            en = min(n - 1, st + max_bars)
            z["limit_placed"] = 1
            limit_px = float(entry_ref * (1.0 - float(off_bps) / 1e4))
            idx = _first_limit_fill(ctx.low_np, st, en, limit_px)
            if idx is not None:
                z.update({"filled": True, "idx": int(idx), "px": float(limit_px), "maker": 1.0, "limit_filled": 1})
                return z
            if fallback_ok:
                fb_bars = max(0, int(math.ceil(float(p.get("fallback_delay_min", 2)) / 3.0)))
                fb_idx = min(n - 1, st + fb_bars)
                z.update({"filled": True, "idx": int(fb_idx), "px": float(ctx.open_np[fb_idx]), "maker": 0.0, "fallback": 1, "market": 1})
            return z

        pol = str(p.get("fallback_policy", "conditional")).lower()
        fallback_allowed = (pol == "always") or (pol == "conditional" and tox <= float(p.get("fallback_toxicity_max", 0.75)))

        fills: List[Tuple[float, int, float, float]] = []
        limit_placed = 0
        limit_filled = 0
        fallback_trg = 0
        market_fill = 0

        if stage1_alloc > 1e-6:
            s1 = _simulate_stage(
                start_idx=sig_idx,
                mode=str(p.get("stage1_mode", "limit")),
                off_bps=float(p.get("stage1_limit_offset_bps", 0.8)),
                max_delay_min=float(p.get("stage1_max_fill_delay_min", 12)),
                fallback_ok=fallback_allowed,
            )
            limit_placed += int(s1["limit_placed"])
            limit_filled += int(s1["limit_filled"])
            fallback_trg += int(s1["fallback"])
            market_fill += int(s1["market"])
            if bool(s1["filled"]):
                fills.append((float(stage1_alloc), int(s1["idx"]), float(s1["px"]), float(s1["maker"])))

        if stage2_frac > 1e-6:
            st2 = min(n - 1, sig_idx + max(0, int(math.ceil(float(p.get("stage2_delay_min", 6)) / 3.0))))
            s2 = _simulate_stage(
                start_idx=st2,
                mode=str(p.get("stage2_mode", "market")),
                off_bps=float(p.get("stage2_limit_offset_bps", 1.2)),
                max_delay_min=float(p.get("stage2_max_fill_delay_min", 18)),
                fallback_ok=fallback_allowed,
            )
            limit_placed += int(s2["limit_placed"])
            limit_filled += int(s2["limit_filled"])
            fallback_trg += int(s2["fallback"])
            market_fill += int(s2["market"])
            if bool(s2["filled"]):
                fills.append((float(stage2_frac), int(s2["idx"]), float(s2["px"]), float(s2["maker"])))

        if not fills:
            out["state_skip_reason"] = "timeout_no_fill"
            return out

        tot = sum(max(0.0, f[0]) for f in fills)
        if tot <= 1e-12:
            out["state_skip_reason"] = "timeout_no_fill"
            return out

        norm = [(float(f[0]) / tot, int(f[1]), float(f[2]), float(f[3])) for f in fills]
        entry_px = float(sum(w * px for w, _idx, px, _m in norm))
        maker_frac = float(sum(w * m for w, _idx, _px, m in norm))
        entry_idx = int(min(idx for _w, idx, _px, _m in norm))
        weighted_delay = float(sum(w * _to_delay_min(ctx, idx) for w, idx, _px, _m in norm))

        out.update(
            {
                "filled": True,
                "fill_idx": int(entry_idx),
                "entry_price": float(entry_px),
                "fill_type": "staged_entry",
                "maker_frac": float(maker_frac),
                "filled_frac": float(min(1.0, tot)),
                "weighted_delay_min": float(weighted_delay),
                "limit_placed": int(limit_placed),
                "limit_filled": int(limit_filled),
                "fallback_triggered": int(fallback_trg > 0),
                "market_filled": int(market_fill > 0),
                "route_bucket": "staged",
            }
        )
        return out

    out["state_skip_reason"] = "unknown_family"
    return out


def simulate_family_signal(
    *,
    ctx: ga_exec.SignalContext,
    v: FamilyVariant,
    eval_cfg: Dict[str, Any],
    last_entry_time: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "symbol": ctx.symbol,
        "signal_id": ctx.signal_id,
        "signal_time": str(ctx.signal_time),
        "signal_tp_mult": float(ctx.tp_mult_sig),
        "signal_sl_mult": float(ctx.sl_mult_sig),
        "baseline_filled": int(ctx.baseline_filled),
        "baseline_valid_for_metrics": int(ctx.baseline_valid_for_metrics),
        "baseline_sl_hit": int(ctx.baseline_sl_hit),
        "baseline_tp_hit": int(ctx.baseline_tp_hit),
        "baseline_pnl_net_pct": float(ctx.baseline_pnl_net_pct),
        "baseline_pnl_gross_pct": float(ctx.baseline_pnl_gross_pct),
        "baseline_fill_liquidity_type": str(ctx.baseline_fill_liq),
        "baseline_fill_delay_min": float(ctx.baseline_fill_delay_min),
        "baseline_mae_pct": float(ctx.baseline_mae_pct),
        "baseline_mfe_pct": float(ctx.baseline_mfe_pct),
        "baseline_entry_time": str(ctx.baseline_entry_time) if ctx.baseline_entry_time is not None else "",
        "baseline_exit_time": str(ctx.baseline_exit_time) if ctx.baseline_exit_time is not None else "",
        "baseline_exit_reason": str(ctx.baseline_exit_reason),
        "baseline_same_bar_hit": int(ctx.baseline_same_bar_hit),
        "baseline_invalid_stop_geometry": int(ctx.baseline_invalid_stop_geometry),
        "baseline_invalid_tp_geometry": int(ctx.baseline_invalid_tp_geometry),
        "baseline_entry_type": str(ctx.baseline_entry_type),
        "baseline_entry_price": float(ctx.baseline_entry_price),
        "baseline_exit_price": float(ctx.baseline_exit_price),
        "exec_filled": 0,
        "exec_valid_for_metrics": 0,
        "exec_sl_hit": 0,
        "exec_tp_hit": 0,
        "exec_pnl_net_pct": float("nan"),
        "exec_pnl_gross_pct": float("nan"),
        "exec_fill_liquidity_type": "",
        "exec_fill_delay_min": float("nan"),
        "exec_mae_pct": float("nan"),
        "exec_mfe_pct": float("nan"),
        "entry_improvement_bps": float("nan"),
        "exec_skip_reason": "",
        "lookahead_violation": 0,
        "constraint_fail_reason": "",
        "missing_slice_flag": 0,
        "funnel_time_ok": 0,
        "funnel_micro_vol_ok": 0,
        "funnel_state_ok": 0,
        "funnel_limit_placed": 0,
        "funnel_limit_filled": 0,
        "funnel_fallback_triggered": 0,
        "funnel_market_filled": 0,
        "funnel_route_bucket": "",
        "funnel_filled_frac": 0.0,
    }

    cooldown_min = int(v.params.get("cooldown_min", 0))
    if last_entry_time is not None and cooldown_min > 0:
        delta_min = float((pd.to_datetime(ctx.signal_time, utc=True) - pd.to_datetime(last_entry_time, utc=True)).total_seconds() / 60.0)
        if delta_min < float(cooldown_min):
            out["exec_skip_reason"] = "cooldown"
            return out

    state = _state_from_ctx(ctx)
    if int(state["ok"]) != 1:
        out["exec_skip_reason"] = str(state["reason"])
        out["missing_slice_flag"] = 1 if str(state["reason"]) in {"no_3m_data", "no_bar_after_signal", "bad_entry_ref"} else 0
        return out

    out["funnel_time_ok"] = 1
    out["funnel_micro_vol_ok"] = 1

    fill = _simulate_family_fill(ctx=ctx, state=state, v=v)
    out["funnel_route_bucket"] = str(fill.get("route_bucket", ""))
    out["funnel_limit_placed"] = int(fill.get("limit_placed", 0))
    out["funnel_limit_filled"] = int(fill.get("limit_filled", 0))
    out["funnel_fallback_triggered"] = int(fill.get("fallback_triggered", 0))
    out["funnel_market_filled"] = int(fill.get("market_filled", 0))

    if str(fill.get("state_skip_reason", "")) == "micro_vol_gate":
        out["funnel_micro_vol_ok"] = 0
        out["exec_skip_reason"] = "micro_vol_gate"
        return out

    if str(fill.get("state_skip_reason", "")):
        out["funnel_state_ok"] = 0
        out["exec_skip_reason"] = str(fill.get("state_skip_reason", "state_gate"))
        return out

    out["funnel_state_ok"] = 1

    if not bool(fill.get("filled", False)):
        out["exec_skip_reason"] = "timeout_no_fill"
        return out

    fill_idx = int(fill["fill_idx"])
    fill_px = float(fill["entry_price"])
    fill_time = pd.to_datetime(int(ctx.ts_ns[fill_idx]), utc=True)

    if ctx.baseline_exit_time is not None and fill_time > pd.to_datetime(ctx.baseline_exit_time, utc=True):
        out["exec_skip_reason"] = "after_baseline_exit"
        return out

    entry_ref = float(state["entry_ref"])
    improve_bps = float((entry_ref - fill_px) / entry_ref * 1e4) if entry_ref > 0 and np.isfinite(entry_ref) else float("nan")

    max_exit_ts_ns = ga_exec.exec3m._compute_eval_end_ns(
        entry_ts_ns=int(ctx.ts_ns[fill_idx]),
        eval_horizon_hours=float(eval_cfg["exec_horizon_hours"]),
        baseline_exit_time=ctx.baseline_exit_time,
    )

    genome = copy.deepcopy(v.base_genome)
    exit_res = ga_exec._simulate_dynamic_exit_long(
        ts_ns=ctx.ts_ns,
        close_np=ctx.close_np,
        high_np=ctx.high_np,
        low_np=ctx.low_np,
        entry_idx=int(fill_idx),
        entry_price=float(fill_px),
        tp_mult_sig=float(ctx.tp_mult_sig),
        sl_mult_sig=float(ctx.sl_mult_sig),
        genome=genome,
        max_exit_ts_ns=int(max_exit_ts_ns),
    )

    legs = list(exit_res.get("legs", [])) if bool(genome.get("partial_take_enabled", 0)) else [(1.0, float(exit_res.get("exit_price", np.nan)))]

    maker_frac = float(fill.get("maker_frac", 0.0))
    filled_frac = float(fill.get("filled_frac", 1.0))

    cost = _costed_from_legs_mixed_entry(
        entry_price=float(fill_px),
        legs=legs,
        maker_frac=float(maker_frac),
        fee_bps_maker=float(eval_cfg["fee_bps_maker"]),
        fee_bps_taker=float(eval_cfg["fee_bps_taker"]),
        slippage_bps_limit=float(eval_cfg["slippage_bps_limit"]),
        slippage_bps_market=float(eval_cfg["slippage_bps_market"]),
    )

    out.update(
        {
            "exec_filled": 1,
            "exec_valid_for_metrics": int(exit_res.get("valid_for_metrics", 0)),
            "exec_sl_hit": int(bool(exit_res.get("sl_hit", False))),
            "exec_tp_hit": int(bool(exit_res.get("tp_hit", False))),
            "exec_pnl_net_pct": float(cost["pnl_net_pct"]) * float(filled_frac),
            "exec_pnl_gross_pct": float(cost["pnl_gross_pct"]) * float(filled_frac),
            "exec_fill_liquidity_type": "maker" if maker_frac >= 0.999 else "taker",
            "exec_fill_delay_min": float(fill.get("weighted_delay_min", _to_delay_min(ctx, fill_idx))),
            "exec_mae_pct": float(exit_res.get("mae_pct", np.nan)),
            "exec_mfe_pct": float(exit_res.get("mfe_pct", np.nan)),
            "entry_improvement_bps": float(improve_bps),
            "exec_skip_reason": "",
            "exec_exit_reason": str(exit_res.get("exit_reason", "")),
            "exec_entry_time": str(fill_time),
            "exec_exit_time": str(exit_res.get("exit_time", "")),
            "exec_entry_price": float(fill_px),
            "exec_exit_price": float(exit_res.get("exit_price", np.nan)),
            "exec_same_bar_hit": int(exit_res.get("same_bar_hit", 0)),
            "exec_invalid_stop_geometry": int(exit_res.get("invalid_stop_geometry", 0)),
            "exec_invalid_tp_geometry": int(exit_res.get("invalid_tp_geometry", 0)),
            "exec_entry_type": str(fill.get("fill_type", "")),
            "funnel_filled_frac": float(filled_frac),
        }
    )
    return out


def evaluate_family_variant(
    *,
    variant: FamilyVariant,
    bundles: List[ga_exec.SymbolBundle],
    args: argparse.Namespace,
    detailed: bool,
    scenario: Optional[Dict[str, Any]] = None,
) -> EvalArtifacts:
    t0 = time.time()
    mode = str(args.mode).lower()

    struct_fail_reasons = _struct_fail_reasons(variant)

    split_rows: List[Dict[str, Any]] = []
    symbol_rows: List[Dict[str, Any]] = []
    all_signal_rows: List[Dict[str, Any]] = []

    participation_fail: List[str] = []
    realism_fail: List[str] = []
    nan_fail: List[str] = []
    data_quality_fail: List[str] = []
    split_fail: List[str] = []

    lookahead_violations = 0
    expected_split_count = int(sum(len(b.splits) for b in bundles))

    fee_mult = float(scenario.get("fee_mult", 1.0)) if scenario else 1.0
    slip_add = float(scenario.get("slip_add", 0.0)) if scenario else 0.0
    eval_cfg = {
        "exec_horizon_hours": float(args.exec_horizon_hours),
        "fee_bps_maker": float(args.fee_bps_maker) * fee_mult,
        "fee_bps_taker": float(args.fee_bps_taker) * fee_mult,
        "slippage_bps_limit": float(args.slippage_bps_limit) + slip_add,
        "slippage_bps_market": float(args.slippage_bps_market) + slip_add,
    }

    fee_model_identical = int(
        np.isfinite(eval_cfg["fee_bps_maker"])
        and np.isfinite(eval_cfg["fee_bps_taker"])
        and np.isfinite(eval_cfg["slippage_bps_limit"])
        and np.isfinite(eval_cfg["slippage_bps_market"])
    )
    if fee_model_identical == 0:
        struct_fail_reasons.append("fee_model_invalid")

    th_genome = _family_threshold_genome(variant)

    for bundle in bundles:
        symbol_all_rows: List[Dict[str, Any]] = []
        thresholds = ga_exec._symbol_thresholds(bundle=bundle, genome=th_genome, mode=mode, args=args)

        for sp in bundle.splits:
            idx0 = int(sp["test_start"])
            idx1 = int(sp["test_end"])
            split_signal_rows: List[Dict[str, Any]] = []
            last_entry_time: Optional[pd.Timestamp] = None

            for ctx in bundle.contexts[idx0:idx1]:
                row = simulate_family_signal(ctx=ctx, v=variant, eval_cfg=eval_cfg, last_entry_time=last_entry_time)
                row["split_id"] = int(sp["split_id"])
                row["split_test_start"] = int(idx0)
                row["split_test_end"] = int(idx1)
                if int(row.get("exec_filled", 0)) == 1:
                    et = pd.to_datetime(row.get("exec_entry_time"), utc=True, errors="coerce")
                    if pd.notna(et):
                        last_entry_time = et
                lookahead_violations += int(row.get("lookahead_violation", 0))
                split_signal_rows.append(row)

            df_split = pd.DataFrame(split_signal_rows)
            split_roll = ga_exec._aggregate_rows(df_split)
            b = split_roll["baseline"]
            e = split_roll["exec"]

            split_rows.append(
                {
                    "family_id": variant.family_id,
                    "variant_id": variant.variant_id,
                    "symbol": bundle.symbol,
                    "split_id": int(sp["split_id"]),
                    "test_start": int(idx0),
                    "test_end": int(idx1),
                    "signals_total": int(e["signals_total"]),
                    "baseline_entries_valid": int(b["entries_valid"]),
                    "exec_entries_valid": int(e["entries_valid"]),
                    "baseline_mean_expectancy_net": float(b["mean_expectancy_net"]),
                    "exec_mean_expectancy_net": float(e["mean_expectancy_net"]),
                    "delta_expectancy_exec_minus_baseline": float(split_roll["delta_expectancy_exec_minus_baseline"]),
                    "baseline_cvar_5": float(b["cvar_5"]),
                    "exec_cvar_5": float(e["cvar_5"]),
                    "cvar_improve_ratio": float(split_roll["cvar_improve_ratio"]),
                    "baseline_max_drawdown": float(b["max_drawdown"]),
                    "exec_max_drawdown": float(e["max_drawdown"]),
                    "maxdd_improve_ratio": float(split_roll["maxdd_improve_ratio"]),
                    "exec_entry_rate": float(e["entry_rate"]),
                    "exec_taker_share": float(e["taker_share"]),
                    "exec_median_fill_delay_min": float(e["median_fill_delay_min"]),
                    "exec_p95_fill_delay_min": float(e["p95_fill_delay_min"]),
                    "exec_median_entry_improvement_bps": float(e["median_entry_improvement_bps"]),
                }
            )
            symbol_all_rows.extend(split_signal_rows)

        df_symbol = pd.DataFrame(symbol_all_rows).sort_values("signal_time").reset_index(drop=True)
        symbol_roll = ga_exec._aggregate_rows(df_symbol)
        b = symbol_roll["baseline"]
        e = symbol_roll["exec"]

        signals_sym = int(e["signals_total"])
        entries_sym = int(e["entries_valid"])
        min_trades_symbol = max(int(args.hard_min_trades_symbol), int(math.ceil(float(args.hard_min_trade_frac_symbol) * max(1, signals_sym))))
        min_entry_rate_symbol = max(float(args.hard_min_entry_rate_symbol), float(thresholds["min_entry_rate"]))

        s_entry_pass = int(np.isfinite(e["entry_rate"]) and e["entry_rate"] >= min_entry_rate_symbol)
        s_trade_count_pass = int(entries_sym >= int(min_trades_symbol))
        if s_entry_pass == 0:
            participation_fail.append(f"{bundle.symbol}:entry_rate")
        if s_trade_count_pass == 0:
            participation_fail.append(f"{bundle.symbol}:trades<{min_trades_symbol}")

        max_taker_symbol = min(float(args.hard_max_taker_share), float(thresholds["max_taker_share"]))
        max_delay_symbol = min(float(args.hard_max_median_fill_delay_min), float(thresholds["max_fill_delay_min"]))
        s_taker_pass = int(np.isfinite(e["taker_share"]) and e["taker_share"] <= max_taker_symbol)
        s_delay_pass = int(np.isfinite(e["median_fill_delay_min"]) and e["median_fill_delay_min"] <= max_delay_symbol)
        s_p95_pass = int(np.isfinite(e["p95_fill_delay_min"]) and e["p95_fill_delay_min"] <= float(args.hard_max_p95_fill_delay_min))
        s_improve_pass = int(np.isfinite(e["median_entry_improvement_bps"]) and e["median_entry_improvement_bps"] >= float(thresholds["min_median_entry_improvement_bps"]))
        if s_taker_pass == 0:
            realism_fail.append(f"{bundle.symbol}:taker_share")
        if s_delay_pass == 0:
            realism_fail.append(f"{bundle.symbol}:median_fill_delay")
        if s_p95_pass == 0:
            realism_fail.append(f"{bundle.symbol}:p95_fill_delay")

        miss_rate_sym = float(pd.to_numeric(df_symbol.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_symbol.empty else 0.0
        s_data_pass = int(miss_rate_sym <= float(args.hard_max_missing_slice_rate))
        if s_data_pass == 0:
            data_quality_fail.append(f"{bundle.symbol}:missing_slice_rate>{float(args.hard_max_missing_slice_rate):.4f}")

        req_symbol = [
            float(e["mean_expectancy_net"]),
            float(e["cvar_5"]),
            float(e["max_drawdown"]),
            float(e["entry_rate"]),
            float(e["taker_share"]),
            float(e["median_fill_delay_min"]),
            float(e["p95_fill_delay_min"]),
        ]
        s_nan_pass = int(all(np.isfinite(v) for v in req_symbol))
        if s_nan_pass == 0:
            nan_fail.append(f"{bundle.symbol}:nan_or_inf")

        symbol_rows.append(
            {
                "family_id": variant.family_id,
                "variant_id": variant.variant_id,
                "symbol": bundle.symbol,
                "signals_total": int(signals_sym),
                "exec_entries_valid": int(entries_sym),
                "baseline_mean_expectancy_net": float(b["mean_expectancy_net"]),
                "exec_mean_expectancy_net": float(e["mean_expectancy_net"]),
                "delta_expectancy_exec_minus_baseline": float(symbol_roll["delta_expectancy_exec_minus_baseline"]),
                "baseline_pnl_net_sum": float(b["pnl_net_sum"]),
                "exec_pnl_net_sum": float(e["pnl_net_sum"]),
                "baseline_cvar_5": float(b["cvar_5"]),
                "exec_cvar_5": float(e["cvar_5"]),
                "cvar_improve_ratio": float(symbol_roll["cvar_improve_ratio"]),
                "baseline_max_drawdown": float(b["max_drawdown"]),
                "exec_max_drawdown": float(e["max_drawdown"]),
                "maxdd_improve_ratio": float(symbol_roll["maxdd_improve_ratio"]),
                "baseline_SL_hit_rate_valid": float(b["SL_hit_rate_valid"]),
                "exec_SL_hit_rate_valid": float(e["SL_hit_rate_valid"]),
                "exec_entry_rate": float(e["entry_rate"]),
                "exec_taker_share": float(e["taker_share"]),
                "exec_median_fill_delay_min": float(e["median_fill_delay_min"]),
                "exec_p95_fill_delay_min": float(e["p95_fill_delay_min"]),
                "exec_median_entry_improvement_bps": float(e["median_entry_improvement_bps"]),
                "missing_slice_rate": float(miss_rate_sym),
                "threshold_min_entry_rate": float(min_entry_rate_symbol),
                "threshold_min_trades": int(min_trades_symbol),
                "threshold_max_taker_share": float(max_taker_symbol),
                "threshold_max_fill_delay_min": float(max_delay_symbol),
                "threshold_max_p95_fill_delay_min": float(args.hard_max_p95_fill_delay_min),
                "threshold_min_median_entry_improvement_bps": float(thresholds["min_median_entry_improvement_bps"]),
                "pass_entry_rate": int(s_entry_pass),
                "pass_trade_count": int(s_trade_count_pass),
                "pass_taker_share": int(s_taker_pass),
                "pass_fill_delay": int(s_delay_pass),
                "pass_p95_fill_delay": int(s_p95_pass),
                "pass_entry_improvement": int(s_improve_pass),
                "pass_data_quality": int(s_data_pass),
                "pass_nan_finite": int(s_nan_pass),
            }
        )
        all_signal_rows.extend(symbol_all_rows)

    df_all = pd.DataFrame(all_signal_rows).sort_values("signal_time").reset_index(drop=True)
    overall_roll = ga_exec._aggregate_rows(df_all)
    b = overall_roll["baseline"]
    e = overall_roll["exec"]

    df_split_all = pd.DataFrame(split_rows)
    split_expectancy = pd.to_numeric(df_split_all.get("exec_mean_expectancy_net", np.nan), errors="coerce")
    min_split_expectancy = float(split_expectancy.min()) if not split_expectancy.empty else float("nan")
    med_split_expectancy = float(split_expectancy.median()) if not split_expectancy.empty else float("nan")
    std_split_expectancy = float(split_expectancy.std(ddof=0)) if not split_expectancy.empty else float("nan")

    if int(len(split_rows)) != int(expected_split_count):
        split_fail.append(f"split_count:{len(split_rows)}!={expected_split_count}")
    if split_expectancy.empty or split_expectancy.isna().any():
        split_fail.append("split_metrics_missing_or_nan")

    overall_signals = int(e["signals_total"])
    overall_entries = int(e["entries_valid"])
    min_trades_overall = max(int(args.hard_min_trades_overall), int(math.ceil(float(args.hard_min_trade_frac_overall) * max(1, overall_signals))))
    overall_entry_rate_pass = int(np.isfinite(e["entry_rate"]) and e["entry_rate"] >= float(args.hard_min_entry_rate_overall))
    overall_trade_count_pass = int(overall_entries >= int(min_trades_overall))
    if overall_entry_rate_pass == 0:
        participation_fail.append("overall:entry_rate")
    if overall_trade_count_pass == 0:
        participation_fail.append(f"overall:trades<{min_trades_overall}")

    overall_taker_pass = int(np.isfinite(e["taker_share"]) and e["taker_share"] <= float(args.hard_max_taker_share))
    overall_median_delay_pass = int(np.isfinite(e["median_fill_delay_min"]) and e["median_fill_delay_min"] <= float(args.hard_max_median_fill_delay_min))
    overall_p95_delay_pass = int(np.isfinite(e["p95_fill_delay_min"]) and e["p95_fill_delay_min"] <= float(args.hard_max_p95_fill_delay_min))
    if overall_taker_pass == 0:
        realism_fail.append("overall:taker_share")
    if overall_median_delay_pass == 0:
        realism_fail.append("overall:median_fill_delay")
    if overall_p95_delay_pass == 0:
        realism_fail.append("overall:p95_fill_delay")

    missing_slice_rate = float(pd.to_numeric(df_all.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_all.empty else float("nan")
    data_quality_pass = int(np.isfinite(missing_slice_rate) and missing_slice_rate <= float(args.hard_max_missing_slice_rate))
    if data_quality_pass == 0:
        data_quality_fail.append(f"overall:missing_slice_rate>{float(args.hard_max_missing_slice_rate):.4f}")

    req_overall = [
        float(e["mean_expectancy_net"]),
        float(e["cvar_5"]),
        float(e["max_drawdown"]),
        float(e["entry_rate"]),
        float(e["taker_share"]),
        float(e["median_fill_delay_min"]),
        float(e["p95_fill_delay_min"]),
    ]
    nan_pass = int(all(np.isfinite(v) for v in req_overall))
    if nan_pass == 0:
        nan_fail.append("overall:nan_or_inf")

    split_pass = int(len(split_fail) == 0)
    if lookahead_violations > 0:
        struct_fail_reasons.append("lookahead_violation")

    constraint_pass = int(len(struct_fail_reasons) == 0 and fee_model_identical == 1 and split_pass == 1)
    participation_pass = int(len(participation_fail) == 0)
    realism_pass = int(len(realism_fail) == 0)
    viability_pass = int(participation_pass == 1 and realism_pass == 1)

    invalid_reasons: List[str] = []
    invalid_reasons.extend(struct_fail_reasons)
    invalid_reasons.extend(sorted(set(participation_fail)))
    invalid_reasons.extend(sorted(set(realism_fail)))
    invalid_reasons.extend(sorted(set(nan_fail)))
    invalid_reasons.extend(sorted(set(data_quality_fail)))
    invalid_reasons.extend(sorted(set(split_fail)))

    hard_invalid = int(
        (constraint_pass == 0)
        or (participation_pass == 0)
        or (realism_pass == 0)
        or (nan_pass == 0)
        or (data_quality_pass == 0)
        or (split_pass == 0)
    )
    valid_for_ranking = int(hard_invalid == 0)

    cvar_gate_pass = int(np.isfinite(overall_roll["cvar_improve_ratio"]) and overall_roll["cvar_improve_ratio"] >= float(args.gate_cvar_improve_min))
    maxdd_gate_pass = int(np.isfinite(overall_roll["maxdd_improve_ratio"]) and overall_roll["maxdd_improve_ratio"] >= float(args.gate_maxdd_improve_min))

    rank_key = (
        int(valid_for_ranking),
        int(constraint_pass),
        int(participation_pass),
        int(realism_pass),
        float(e["mean_expectancy_net"]) if np.isfinite(e["mean_expectancy_net"]) else -1e9,
        float(overall_roll["cvar_improve_ratio"]) if np.isfinite(overall_roll["cvar_improve_ratio"]) else -1e9,
        float(overall_roll["maxdd_improve_ratio"]) if np.isfinite(overall_roll["maxdd_improve_ratio"]) else -1e9,
        float(med_split_expectancy) if np.isfinite(med_split_expectancy) else -1e9,
        -float(std_split_expectancy) if np.isfinite(std_split_expectancy) else -1e9,
    )

    met: Dict[str, Any] = {
        "family_id": variant.family_id,
        "variant_id": variant.variant_id,
        "profile": variant.profile,
        "variant_hash": variant_hash(variant),
        "constraint_pass": int(constraint_pass),
        "constraint_fail_reason": "|".join(sorted(set(struct_fail_reasons))),
        "participation_pass": int(participation_pass),
        "participation_fail_reason": "|".join(sorted(set(participation_fail))),
        "realism_pass": int(realism_pass),
        "realism_fail_reason": "|".join(sorted(set(realism_fail))),
        "nan_pass": int(nan_pass),
        "nan_fail_reason": "|".join(sorted(set(nan_fail))),
        "data_quality_pass": int(data_quality_pass),
        "data_quality_fail_reason": "|".join(sorted(set(data_quality_fail))),
        "split_pass": int(split_pass),
        "split_fail_reason": "|".join(sorted(set(split_fail))),
        "hard_invalid": int(hard_invalid),
        "valid_for_ranking": int(valid_for_ranking),
        "invalid_reason": "|".join(sorted(set(invalid_reasons))),
        "viability_pass": int(viability_pass),
        "viability_fail_reason": "|".join(sorted(set(participation_fail + realism_fail))),
        "fee_model_identical": int(fee_model_identical),
        "lookahead_violations": int(lookahead_violations),
        "split_count": int(len(split_rows)),
        "expected_split_count": int(expected_split_count),
        "overall_signals_total": int(overall_signals),
        "overall_entries_valid": int(overall_entries),
        "overall_min_trades_required": int(min_trades_overall),
        "overall_entry_rate": float(e["entry_rate"]),
        "overall_exec_expectancy_net": float(e["mean_expectancy_net"]),
        "overall_baseline_expectancy_net": float(b["mean_expectancy_net"]),
        "overall_delta_expectancy_exec_minus_baseline": float(overall_roll["delta_expectancy_exec_minus_baseline"]),
        "overall_exec_pnl_net_sum": float(e["pnl_net_sum"]),
        "overall_baseline_pnl_net_sum": float(b["pnl_net_sum"]),
        "overall_exec_cvar_5": float(e["cvar_5"]),
        "overall_baseline_cvar_5": float(b["cvar_5"]),
        "overall_cvar_improve_ratio": float(overall_roll["cvar_improve_ratio"]),
        "overall_exec_max_drawdown": float(e["max_drawdown"]),
        "overall_baseline_max_drawdown": float(b["max_drawdown"]),
        "overall_maxdd_improve_ratio": float(overall_roll["maxdd_improve_ratio"]),
        "overall_exec_taker_share": float(e["taker_share"]),
        "overall_exec_median_fill_delay_min": float(e["median_fill_delay_min"]),
        "overall_exec_p95_fill_delay_min": float(e["p95_fill_delay_min"]),
        "overall_exec_median_entry_improvement_bps": float(e["median_entry_improvement_bps"]),
        "overall_missing_slice_rate": float(missing_slice_rate),
        "overall_exec_sl_hit_rate_valid": float(e["SL_hit_rate_valid"]),
        "overall_baseline_sl_hit_rate_valid": float(b["SL_hit_rate_valid"]),
        "overall_exec_worst_decile_mean": float(e["worst_decile_mean"]),
        "overall_baseline_worst_decile_mean": float(b["worst_decile_mean"]),
        "overall_exec_pnl_std": float(e["pnl_std"]),
        "overall_baseline_pnl_std": float(b["pnl_std"]),
        "min_split_expectancy_net": float(min_split_expectancy),
        "median_split_expectancy_net": float(med_split_expectancy),
        "std_split_expectancy_net": float(std_split_expectancy),
        "tail_gate_pass_cvar": int(cvar_gate_pass),
        "tail_gate_pass_maxdd": int(maxdd_gate_pass),
        "rank_key": [float(x) for x in rank_key],
        "eval_time_sec": float(time.time() - t0),
    }

    return EvalArtifacts(metrics=met, split_df=df_split_all if detailed else pd.DataFrame(), symbol_df=pd.DataFrame(symbol_rows) if detailed else pd.DataFrame(), signal_df=df_all if detailed else pd.DataFrame())


def funnel_from_signal_df(df: pd.DataFrame, met: Dict[str, Any]) -> Dict[str, Any]:
    x = df.copy()
    if x.empty:
        return {
            "signals_total": 0,
            "signals_eligible_after_time_filters": 0,
            "signals_eligible_after_micro_vol": 0,
            "signals_eligible_after_state_gates": 0,
            "signals_with_limit_placed": 0,
            "signals_limit_filled": 0,
            "signals_fallback_triggered": 0,
            "signals_market_filled": 0,
            "entries_valid_total": 0,
            "trades_total": 0,
            "entry_rate": float("nan"),
            "median_fill_delay_min": float("nan"),
            "p95_fill_delay_min": float("nan"),
            "taker_share": float("nan"),
        }
    return {
        "signals_total": int(len(x)),
        "signals_eligible_after_time_filters": int(to_num(x.get("funnel_time_ok", 0)).fillna(0).astype(int).sum()),
        "signals_eligible_after_micro_vol": int(to_num(x.get("funnel_micro_vol_ok", 0)).fillna(0).astype(int).sum()),
        "signals_eligible_after_state_gates": int(to_num(x.get("funnel_state_ok", 0)).fillna(0).astype(int).sum()),
        "signals_with_limit_placed": int(to_num(x.get("funnel_limit_placed", 0)).fillna(0).astype(int).sum()),
        "signals_limit_filled": int(to_num(x.get("funnel_limit_filled", 0)).fillna(0).astype(int).sum()),
        "signals_fallback_triggered": int(to_num(x.get("funnel_fallback_triggered", 0)).fillna(0).astype(int).sum()),
        "signals_market_filled": int(to_num(x.get("funnel_market_filled", 0)).fillna(0).astype(int).sum()),
        "entries_valid_total": int(met.get("overall_entries_valid", 0)),
        "trades_total": int(met.get("overall_entries_valid", 0)),
        "entry_rate": float(met.get("overall_entry_rate", np.nan)),
        "median_fill_delay_min": float(met.get("overall_exec_median_fill_delay_min", np.nan)),
        "p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
        "taker_share": float(met.get("overall_exec_taker_share", np.nan)),
    }


def nx2_family_verdict(df: pd.DataFrame, args: argparse.Namespace) -> Tuple[str, str]:
    if df.empty:
        return "NX2_NO_GO", "no evaluated configs"
    max_entry = float(to_num(df["entry_rate"]).max())
    max_entries = int(to_num(df["entries_valid_total"]).max())
    min_taker_upper = float(to_num(df["taker_share"]).min())
    realistic = int(np.isfinite(min_taker_upper) and min_taker_upper <= float(args.hard_max_taker_share) + 0.15)
    reachable = int(
        np.isfinite(max_entry)
        and max_entry >= max(0.55, float(args.hard_min_entry_rate_overall) * 0.85)
        and max_entries >= int(max(120, float(args.hard_min_trades_overall) * 0.70))
    )
    if reachable == 1 and realistic == 1:
        return "NX2_GO", "participation appears reachable under mechanically valid settings"
    return "NX2_NO_GO", "participation/realism envelope not convincingly reachable"


def family_nx3_verdict(df: pd.DataFrame) -> Tuple[str, str]:
    if df.empty:
        return "FAMILY_NO_GO", "no ablation rows"
    valid = df[to_num(df["valid_for_ranking"]).fillna(0).astype(int) == 1].copy()
    if valid.empty:
        return "FAMILY_NO_GO", "zero valid_for_ranking variants"

    sane = valid[
        np.isfinite(to_num(valid["overall_exec_expectancy_net"]))
        & np.isfinite(to_num(valid["overall_cvar_improve_ratio"]))
        & np.isfinite(to_num(valid["overall_maxdd_improve_ratio"]))
    ].copy()
    if sane.empty:
        return "FAMILY_NO_GO", "valid rows had metric pathologies"

    path_supported = sane[
        (to_num(sane["overall_delta_expectancy_exec_minus_baseline"]) > 0.0)
        & (
            (to_num(sane["overall_cvar_improve_ratio"]) > 0.0)
            | (to_num(sane["overall_maxdd_improve_ratio"]) > 0.0)
        )
    ].copy()
    if path_supported.empty:
        return "FAMILY_NO_GO", "improvements are participation-only or absent"

    robust_lite = path_supported[
        (to_num(path_supported["route_pass_lite"]).fillna(0).astype(int) == 1)
        & (to_num(path_supported["stress_lite_pass"]).fillna(0).astype(int) == 1)
    ].copy()
    if robust_lite.empty:
        return "FAMILY_NO_GO", "all path-supported variants collapsed under route/stress-lite"

    return "FAMILY_GO_TO_GA", "at least one non-pathological, path-supported, stress-lite surviving variant"


def evaluate_route_lite(
    *,
    variant: FamilyVariant,
    route_bundles: Dict[str, ga_exec.SymbolBundle],
    args: argparse.Namespace,
) -> Tuple[int, float]:
    rows = []
    for rid, b in route_bundles.items():
        ev = evaluate_family_variant(variant=variant, bundles=[b], args=args, detailed=False)
        m = ev.metrics
        rows.append(
            {
                "route_id": rid,
                "valid_for_ranking": int(m.get("valid_for_ranking", 0)),
                "delta": float(m.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
            }
        )
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return 0, float("nan")
    pass_flag = int((to_num(rdf["valid_for_ranking"]) == 1).all())
    min_delta = float(to_num(rdf["delta"]).min()) if "delta" in rdf.columns else float("nan")
    return pass_flag, min_delta


def evaluate_stress_lite(variant: FamilyVariant, bundles: List[ga_exec.SymbolBundle], args: argparse.Namespace) -> Tuple[int, Dict[str, Any]]:
    sc = {"fee_mult": 1.15, "slip_add": 0.5}
    ev = evaluate_family_variant(variant=variant, bundles=bundles, args=args, detailed=False, scenario=sc)
    m = ev.metrics
    pass_flag = int(
        int(m.get("valid_for_ranking", 0)) == 1
        and float(m.get("overall_delta_expectancy_exec_minus_baseline", np.nan)) > 0.0
        and float(m.get("overall_cvar_improve_ratio", np.nan)) >= 0.0
        and float(m.get("overall_maxdd_improve_ratio", np.nan)) >= 0.0
    )
    return pass_flag, {
        "stress_valid_for_ranking": int(m.get("valid_for_ranking", 0)),
        "stress_delta_expectancy": float(m.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
        "stress_cvar_improve_ratio": float(m.get("overall_cvar_improve_ratio", np.nan)),
        "stress_maxdd_improve_ratio": float(m.get("overall_maxdd_improve_ratio", np.nan)),
    }


def ga_fitness(m: Dict[str, Any]) -> float:
    if int(m.get("valid_for_ranking", 0)) != 1:
        return float("-inf")
    de = float(m.get("overall_delta_expectancy_exec_minus_baseline", np.nan))
    cv = float(m.get("overall_cvar_improve_ratio", np.nan))
    dd = float(m.get("overall_maxdd_improve_ratio", np.nan))
    taker = float(m.get("overall_exec_taker_share", np.nan))
    delay = float(m.get("overall_exec_median_fill_delay_min", np.nan))
    if not (np.isfinite(de) and np.isfinite(cv) and np.isfinite(dd)):
        return float("-inf")
    penalty = max(0.0, taker - 0.25) * 0.50 + max(0.0, delay - 45.0) / 200.0
    return float(de + 0.35 * cv + 0.35 * dd - penalty)


def mutate_family_params(family_id: str, params: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    p = copy.deepcopy(params)
    # mutation by resampling around profile buckets.
    mode = random.choice(["conservative", "mid", "aggressive"])
    fresh = _sample_family_params(family_id, rng=rng, profile=mode)
    keys = list(p.keys())
    k = max(1, min(len(keys), rng.randint(1, 4)))
    for kk in rng.sample(keys, k=k):
        if kk in fresh:
            p[kk] = fresh[kk]
    return p


def crossover_family_params(a: Dict[str, Any], b: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    keys = sorted(set(a.keys()).intersection(set(b.keys())))
    out = {}
    for k in keys:
        out[k] = a[k] if rng.random() < 0.5 else b[k]
    return out


def parse_invalid_hist(df: pd.DataFrame) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    if df.empty:
        return hist
    for s in df["invalid_reason"].fillna("").astype(str):
        txt = s.strip()
        if not txt:
            continue
        for p in [x.strip() for x in txt.split("|") if x.strip()]:
            hist[p] = int(hist.get(p, 0) + 1)
    return dict(sorted(hist.items(), key=lambda kv: kv[0]))


def metric_signature(df: pd.DataFrame) -> pd.Series:
    cols = [
        "overall_delta_expectancy_exec_minus_baseline",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "overall_entry_rate",
        "overall_exec_taker_share",
        "overall_exec_median_fill_delay_min",
        "overall_entries_valid",
        "valid_for_ranking",
    ]
    z = df.copy()
    for c in cols:
        if c not in z.columns:
            z[c] = np.nan
    p = z[cols].copy()
    for c in cols:
        p[c] = to_num(p[c]).round(12)
    txt = p.apply(lambda r: "|".join(str(x) for x in r.tolist()), axis=1)
    return txt.map(lambda t: sha256_text(t)[:24])


def bootstrap_pass_rate(signal_df: pd.DataFrame, n_boot: int, seed: int) -> float:
    if signal_df.empty:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    n = len(signal_df)
    hit = 0
    tot = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bs = signal_df.iloc[idx].reset_index(drop=True)
        roll = ga_exec._aggregate_rows(bs)
        de = float(roll["delta_expectancy_exec_minus_baseline"])
        cv = float(roll["cvar_improve_ratio"])
        dd = float(roll["maxdd_improve_ratio"])
        if not (np.isfinite(de) and np.isfinite(cv) and np.isfinite(dd)):
            continue
        tot += 1
        if de > 0.0 and cv >= 0.0 and dd >= 0.0:
            hit += 1
    if tot == 0:
        return float("nan")
    return float(hit / tot)


def neighborhood_stability(
    *,
    best: FamilyVariant,
    bundles: List[ga_exec.SymbolBundle],
    args: argparse.Namespace,
    seed: int,
) -> Tuple[int, int, int]:
    rng = random.Random(seed)
    robust = 0
    total = 0
    for _ in range(6):
        p = mutate_family_params(best.family_id, best.params, rng=rng)
        v = FamilyVariant(
            family_id=best.family_id,
            variant_id=f"neigh_{total:02d}",
            profile="neighbor",
            params=p,
            base_genome=copy.deepcopy(best.base_genome),
            seed_origin="neighborhood",
        )
        m = evaluate_family_variant(variant=v, bundles=bundles, args=args, detailed=False).metrics
        total += 1
        ok = int(
            int(m.get("valid_for_ranking", 0)) == 1
            and float(m.get("overall_delta_expectancy_exec_minus_baseline", np.nan)) > 0.0
            and float(m.get("overall_cvar_improve_ratio", np.nan)) >= 0.0
            and float(m.get("overall_maxdd_improve_ratio", np.nan)) >= 0.0
        )
        robust += ok
    stable = int(robust >= 3)
    return stable, robust, total


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase NX execution-family discovery (3m execution layer, contract-locked)")
    ap.add_argument("--seed", type=int, default=20260225)
    ap.add_argument("--nx3-per-family", type=int, default=12)
    ap.add_argument("--ga-pop", type=int, default=96)
    ap.add_argument("--ga-gens", type=int, default=3)
    ap.add_argument("--ga-workers", type=int, default=1)
    args = ap.parse_args()

    run_dir = REPORTS_ROOT / f"PHASENX_EXEC_FAMILY_DISCOVERY_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    t_start = time.time()

    phases_executed: List[str] = []
    branch_decisions: Dict[str, Any] = {}
    compute_budgets: Dict[str, Any] = {
        "nx2_direct_configs": 0,
        "nx3_ablation_variants": 0,
        "nx5_ga_pop": int(args.ga_pop),
        "nx5_ga_gens": int(args.ga_gens),
    }

    git_meta = git_snapshot()
    locked_hashes = {
        "representative_subset_csv_sha256": sha256_file(Path(LOCKED["representative_subset_csv"])) if Path(LOCKED["representative_subset_csv"]).exists() else "",
        "canonical_fee_model_sha256": sha256_file(Path(LOCKED["canonical_fee_model"])) if Path(LOCKED["canonical_fee_model"]).exists() else "",
        "canonical_metrics_definition_sha256": sha256_file(Path(LOCKED["canonical_metrics_definition"])) if Path(LOCKED["canonical_metrics_definition"]).exists() else "",
    }

    final_classification = ""
    mainline_status = ""
    furthest_phase = "NX0"

    # NX0: Contract validation + engine discovery.
    subset_path = Path(LOCKED["representative_subset_csv"])
    fee_path = Path(LOCKED["canonical_fee_model"])
    metrics_path = Path(LOCKED["canonical_metrics_definition"])

    contract_obj: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "paths": {
            "representative_subset_csv": str(subset_path),
            "canonical_fee_model": str(fee_path),
            "canonical_metrics_definition": str(metrics_path),
        },
        "expected_hashes": {
            "fee_sha256": LOCKED["expected_fee_sha"],
            "metrics_sha256": LOCKED["expected_metrics_sha"],
        },
        "observed_hashes": locked_hashes,
        "allow_freeze_hash_mismatch": int(LOCKED["allow_freeze_hash_mismatch"]),
    }

    if not subset_path.exists() or not fee_path.exists() or not metrics_path.exists():
        contract_obj["contract_lock_pass"] = 0
        contract_obj["mismatch_fields"] = [
            k
            for k, p in {
                "representative_subset_csv": subset_path,
                "canonical_fee_model": fee_path,
                "canonical_metrics_definition": metrics_path,
            }.items()
            if not p.exists()
        ]
        json_dump(run_dir / "phaseNX0_contract_validation.json", contract_obj)
        write_text(
            run_dir / "phaseNX0_engine_discovery.md",
            "# NX0 Engine Discovery\n\n- Contract files missing; engine discovery skipped due fail-fast lock behavior.\n",
        )
        phases_executed.append("NX0")
        furthest_phase = "NX0"
        final_classification = "STOP_NO_GO_CONTRACT"
        mainline_status = "STOP_NO_GO_CONTRACT"
        branch_decisions["NX0"] = {"classification": final_classification, "reason": "required contract files missing"}
    else:
        fee_ok = int(locked_hashes["canonical_fee_model_sha256"] == LOCKED["expected_fee_sha"])
        met_ok = int(locked_hashes["canonical_metrics_definition_sha256"] == LOCKED["expected_metrics_sha"])

        rep_df = pd.read_csv(subset_path)
        contract_obj["subset_rows"] = int(len(rep_df))
        contract_obj["subset_columns"] = list(rep_df.columns)
        contract_obj["fee_hash_match_expected"] = int(fee_ok)
        contract_obj["metrics_hash_match_expected"] = int(met_ok)

        exec_args = build_exec_args(signals_csv=subset_path, seed=int(args.seed))
        try:
            lock_info = ga_exec._validate_and_lock_frozen_artifacts(args=exec_args, run_dir=run_dir)
            contract_obj["ga_exec_freeze_lock_validation"] = lock_info
            freeze_lock_pass = int(lock_info.get("freeze_lock_pass", 0))
        except Exception as e:
            contract_obj["ga_exec_freeze_lock_validation_error"] = str(e)
            freeze_lock_pass = 0

        contract_obj["contract_lock_pass"] = int(fee_ok == 1 and met_ok == 1 and freeze_lock_pass == 1)
        contract_obj["mismatch_fields"] = []
        if fee_ok != 1:
            contract_obj["mismatch_fields"].append("fee_model_sha256")
        if met_ok != 1:
            contract_obj["mismatch_fields"].append("metrics_definition_sha256")
        if freeze_lock_pass != 1:
            contract_obj["mismatch_fields"].append("ga_exec_freeze_lock_validation")

        json_dump(run_dir / "phaseNX0_contract_validation.json", contract_obj)

        eng_lines = [
            "# NX0 Engine Discovery",
            "",
            f"- Generated UTC: {utc_now()}",
            "- Reused evaluator: `src/execution/ga_exec_3m_opt.py`",
            "- Reused functions:",
            "  - `_validate_and_lock_frozen_artifacts` for canonical freeze hash lock + fee/slippage parity",
            "  - `_prepare_bundles` for representative signal -> 3m context materialization",
            "  - `_aggregate_rows` and `_symbol_thresholds` for unchanged metrics/gate semantics",
            "  - `_simulate_dynamic_exit_long` for mechanics-valid exit simulation",
            "- Frozen scoring fields consumed:",
            "  - valid_for_ranking, invalid_reason",
            "  - overall_exec_expectancy_net, overall_delta_expectancy_exec_minus_baseline",
            "  - overall_cvar_improve_ratio, overall_maxdd_improve_ratio",
            "  - overall_entries_valid, overall_entry_rate",
            "  - overall_exec_taker_share, overall_exec_median_fill_delay_min, overall_exec_p95_fill_delay_min",
            "- Representative subset schema snapshot:",
            f"  - columns={list(rep_df.columns)}",
            f"  - rows={len(rep_df)}",
        ]
        write_text(run_dir / "phaseNX0_engine_discovery.md", "\n".join(eng_lines) + "\n")

        phases_executed.append("NX0")

        if int(contract_obj["contract_lock_pass"]) != 1:
            furthest_phase = "NX0"
            final_classification = "STOP_NO_GO_CONTRACT"
            mainline_status = "STOP_NO_GO_CONTRACT"
            branch_decisions["NX0"] = {
                "classification": final_classification,
                "reason": "contract hash lock and/or ga_exec freeze lock failed",
                "mismatch_fields": contract_obj["mismatch_fields"],
            }
        else:
            # Build core context for remaining phases.
            bundles, load_meta = ga_exec._prepare_bundles(exec_args)
            _ = load_meta
            if not bundles:
                furthest_phase = "NX0"
                final_classification = "STOP_NO_GO_CONTRACT"
                mainline_status = "STOP_NO_GO_CONTRACT"
                branch_decisions["NX0"] = {
                    "classification": final_classification,
                    "reason": "no bundles prepared under frozen harness",
                }
            else:
                base_bundle = bundles[0]
                base_genome = extract_base_genome()
                route_bundles = build_route_bundles(base_bundle=base_bundle, rep_subset=rep_df, args=exec_args)

                # NX1: family spec + mapping.
                furthest_phase = "NX1"
                bounds = family_param_bounds()
                write_text(run_dir / "phaseNX1_param_bounds.yaml", json.dumps(bounds, indent=2, sort_keys=True) + "\n")

                spec_lines = [
                    "# NX1 Family Spec",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    "- Baseline execution genome source: E1 fallback-compatible genome, then family-specific entry mechanics wrappers.",
                    "",
                    f"## {FAMILY_A}",
                    "- Multi-step passive limit ladder (2-3 steps) with adaptive offset from ATR-z/spread state.",
                    "- Bounded cancel/replace cadence via step_delay_min and max_fill_delay_min.",
                    "- Conditional fallback to market under toxicity cap (not unconditional taker fallback by default).",
                    "",
                    f"## {FAMILY_B}",
                    "- Regime-routed entry profile (calm/neutral/toxic) based on ATR-z + spread proxy.",
                    "- Each regime routes to distinct mode/delay/fallback behavior.",
                    "- Optional extreme-toxicity skip guard remains bounded and explicitly measurable.",
                    "",
                    f"## {FAMILY_C}",
                    "- Two-stage split entry schedule (stage1 + delayed stage2).",
                    "- Soft risk shaping via toxicity-dependent stage2 notional scaling (soft modulation, no hard starvation filter).",
                    "- Stage-wise fallback policy bounded and auditable.",
                    "",
                    "All families keep: frozen subset, frozen fee/slippage model, unchanged hard gates, no hindsight fills.",
                ]
                write_text(run_dir / "phaseNX1_family_spec.md", "\n".join(spec_lines) + "\n")

                map_lines = [
                    "# NX1 Mapping To Existing Engines",
                    "",
                    "- Core scorer/gates reused from `src/execution/ga_exec_3m_opt.py`:",
                    "  - unchanged aggregation and hard-gate validity logic",
                    "  - unchanged CVaR/maxDD/expectancy definitions",
                    "- New family mechanics implemented as wrapper-level entry simulators feeding equivalent per-signal rows into existing scorer.",
                    "- Exit path remains `ga_exec._simulate_dynamic_exit_long` for all families.",
                    "- Contract lock remains `ga_exec._validate_and_lock_frozen_artifacts` with `allow_freeze_hash_mismatch=0`.",
                    "",
                    "Telemetry additions:",
                    "- Funnel counters: time/micro-vol/state eligibility, limit placement/fill, fallback/market fill counts.",
                    "- Route bucket traces for regime-routed family.",
                    "- Duplicate/effective-trial accounting at ablation + GA pilot stages.",
                ]
                write_text(run_dir / "phaseNX1_mapping_to_existing_engines.md", "\n".join(map_lines) + "\n")

                patch_lines = [
                    "# NX Patch Diff Summary",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    "- Files changed:",
                    "  - scripts/phase_nx_exec_family_discovery.py (new)",
                    "- Rationale:",
                    "  - Implements autonomous NX0-NX7 discovery pipeline required after J0 NO_GO.",
                    "  - Reuses frozen ga_exec harness and unchanged hard-gate logic while adding three structurally distinct entry-mechanics families.",
                    "  - Adds required forensic outputs, branch-stop classifications, duplicate/effective-trials telemetry, and robustness gate artifacts.",
                ]
                write_text(run_dir / "phaseNX_patch_diff_summary.md", "\n".join(patch_lines) + "\n")

                phases_executed.append("NX1")

                # NX2: Feasibility funnel + upper-bound.
                furthest_phase = "NX2"
                nx2_variants = build_nx2_variants(base_genome=base_genome)
                compute_budgets["nx2_direct_configs"] = int(len(nx2_variants))

                nx2_rows: List[Dict[str, Any]] = []
                for vv in nx2_variants:
                    ev = evaluate_family_variant(variant=vv, bundles=bundles, args=exec_args, detailed=True)
                    met = ev.metrics
                    fn = funnel_from_signal_df(ev.signal_df, met)
                    row = {
                        "family_id": vv.family_id,
                        "variant_id": vv.variant_id,
                        "profile": vv.profile,
                        "is_upper_bound": int(vv.is_upper_bound),
                        "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
                        "invalid_reason": str(met.get("invalid_reason", "")),
                        "overall_exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
                        "overall_delta_expectancy_exec_minus_baseline": float(met.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
                        "overall_cvar_improve_ratio": float(met.get("overall_cvar_improve_ratio", np.nan)),
                        "overall_maxdd_improve_ratio": float(met.get("overall_maxdd_improve_ratio", np.nan)),
                        "overall_entries_valid": int(met.get("overall_entries_valid", 0)),
                        "overall_entry_rate": float(met.get("overall_entry_rate", np.nan)),
                        "overall_exec_taker_share": float(met.get("overall_exec_taker_share", np.nan)),
                        "overall_exec_median_fill_delay_min": float(met.get("overall_exec_median_fill_delay_min", np.nan)),
                        "overall_exec_p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
                    }
                    row.update(fn)
                    nx2_rows.append(row)

                nx2_df = pd.DataFrame(nx2_rows)
                nx2_df.to_csv(run_dir / "phaseNX2_feasibility_funnel.csv", index=False)
                nx2_ub = nx2_df[nx2_df["is_upper_bound"].astype(int) == 1].copy().reset_index(drop=True)
                nx2_ub.to_csv(run_dir / "phaseNX2_family_upper_bound_tests.csv", index=False)

                nx2_family_summary_rows: List[Dict[str, Any]] = []
                for fam in [FAMILY_A, FAMILY_B, FAMILY_C]:
                    fam_df = nx2_df[nx2_df["family_id"] == fam].copy()
                    verdict, reason = nx2_family_verdict(fam_df, args=exec_args)
                    nx2_family_summary_rows.append(
                        {
                            "family_id": fam,
                            "nx2_verdict": verdict,
                            "nx2_reason": reason,
                            "max_entry_rate": float(to_num(fam_df.get("entry_rate", pd.Series(dtype=float))).max()) if not fam_df.empty else float("nan"),
                            "max_entries_valid": int(to_num(fam_df.get("entries_valid_total", pd.Series(dtype=float))).max()) if not fam_df.empty else 0,
                            "upper_bound_valid_for_ranking": int(to_num(fam_df[fam_df["is_upper_bound"] == 1].get("valid_for_ranking", pd.Series(dtype=float))).fillna(0).astype(int).max()) if not fam_df.empty else 0,
                        }
                    )
                nx2_sum = pd.DataFrame(nx2_family_summary_rows)

                nx2_rep = [
                    "# NX2 Feasibility Report",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Direct configs evaluated: `{len(nx2_df)}`",
                    "",
                    "## Family Verdicts",
                    "",
                    markdown_table(nx2_sum, ["family_id", "nx2_verdict", "nx2_reason", "max_entry_rate", "max_entries_valid", "upper_bound_valid_for_ranking"], n=10),
                    "",
                    "## Upper-Bound Snapshot",
                    "",
                    markdown_table(
                        nx2_ub,
                        [
                            "family_id",
                            "variant_id",
                            "valid_for_ranking",
                            "entry_rate",
                            "entries_valid_total",
                            "taker_share",
                            "median_fill_delay_min",
                            "overall_delta_expectancy_exec_minus_baseline",
                        ],
                        n=10,
                    ),
                ]
                write_text(run_dir / "phaseNX2_feasibility_report.md", "\n".join(nx2_rep) + "\n")
                phases_executed.append("NX2")

                # NX3: controlled ablation bench (no GA).
                furthest_phase = "NX3"
                nx2_go_fams = set(nx2_sum[nx2_sum["nx2_verdict"] == "NX2_GO"]["family_id"].tolist())

                ab_all = build_ablation_variants(base_genome=base_genome, seed=int(args.seed) + 17, n_per_family=int(args.nx3_per_family))
                ab_vars = [v for v in ab_all if v.family_id in nx2_go_fams]
                compute_budgets["nx3_ablation_variants"] = int(len(ab_vars))

                ab_rows: List[Dict[str, Any]] = []
                for vv in ab_vars:
                    ev = evaluate_family_variant(variant=vv, bundles=bundles, args=exec_args, detailed=False)
                    met = ev.metrics
                    route_pass, route_min_delta = evaluate_route_lite(variant=vv, route_bundles=route_bundles, args=exec_args)
                    stress_pass, stress_obj = evaluate_stress_lite(variant=vv, bundles=bundles, args=exec_args)
                    row = {
                        "family_id": vv.family_id,
                        "variant_id": vv.variant_id,
                        "profile": vv.profile,
                        "variant_hash": variant_hash(vv),
                        "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
                        "invalid_reason": str(met.get("invalid_reason", "")),
                        "overall_exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
                        "overall_delta_expectancy_exec_minus_baseline": float(met.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
                        "overall_cvar_improve_ratio": float(met.get("overall_cvar_improve_ratio", np.nan)),
                        "overall_maxdd_improve_ratio": float(met.get("overall_maxdd_improve_ratio", np.nan)),
                        "overall_entries_valid": int(met.get("overall_entries_valid", 0)),
                        "overall_entry_rate": float(met.get("overall_entry_rate", np.nan)),
                        "overall_exec_taker_share": float(met.get("overall_exec_taker_share", np.nan)),
                        "overall_exec_median_fill_delay_min": float(met.get("overall_exec_median_fill_delay_min", np.nan)),
                        "overall_exec_p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
                        "min_split_expectancy_net": float(met.get("min_split_expectancy_net", np.nan)),
                        "route_pass_lite": int(route_pass),
                        "route_min_delta_lite": float(route_min_delta),
                        "stress_lite_pass": int(stress_pass),
                    }
                    row.update(stress_obj)
                    ab_rows.append(row)

                ab_df = pd.DataFrame(ab_rows)
                ab_df.to_csv(run_dir / "phaseNX3_ablation_results.csv", index=False)

                inv_hist = parse_invalid_hist(ab_df)
                json_dump(run_dir / "phaseNX3_invalid_reason_histogram.json", inv_hist)

                dup_stats = pd.DataFrame()
                if not ab_df.empty:
                    ab_df["metric_signature"] = metric_signature(ab_df)
                    grp = ab_df.groupby(["family_id", "metric_signature"], dropna=False).size().reset_index(name="count")
                    dup_stats = grp[grp["count"] > 1].copy().sort_values(["count", "family_id"], ascending=[False, True]).reset_index(drop=True)
                    if not dup_stats.empty:
                        dup_stats.to_csv(run_dir / "phaseNX3_duplicate_stats.csv", index=False)

                nx3_family_rows: List[Dict[str, Any]] = []
                for fam in [FAMILY_A, FAMILY_B, FAMILY_C]:
                    fam_df = ab_df[ab_df["family_id"] == fam].copy()
                    if fam not in nx2_go_fams:
                        nx3_family_rows.append(
                            {
                                "family_id": fam,
                                "nx2_verdict": "NX2_NO_GO",
                                "nx3_verdict": "FAMILY_NO_GO",
                                "nx3_reason": "excluded after NX2",
                                "valid_for_ranking_count": 0,
                                "best_delta": float("nan"),
                                "best_cvar": float("nan"),
                                "best_maxdd": float("nan"),
                                "route_stress_lite_survivors": 0,
                            }
                        )
                        continue

                    verdict, reason = family_nx3_verdict(fam_df)
                    valid_df = fam_df[to_num(fam_df["valid_for_ranking"]).fillna(0).astype(int) == 1].copy()
                    if not valid_df.empty:
                        best = valid_df.sort_values(
                            ["overall_delta_expectancy_exec_minus_baseline", "overall_cvar_improve_ratio", "overall_maxdd_improve_ratio"],
                            ascending=[False, False, False],
                        ).iloc[0]
                        best_delta = float(best["overall_delta_expectancy_exec_minus_baseline"])
                        best_cv = float(best["overall_cvar_improve_ratio"])
                        best_dd = float(best["overall_maxdd_improve_ratio"])
                    else:
                        best_delta = float("nan")
                        best_cv = float("nan")
                        best_dd = float("nan")

                    rs_lite = int(
                        (
                            (to_num(valid_df.get("route_pass_lite", 0)).fillna(0).astype(int) == 1)
                            & (to_num(valid_df.get("stress_lite_pass", 0)).fillna(0).astype(int) == 1)
                        ).sum()
                    )

                    nx3_family_rows.append(
                        {
                            "family_id": fam,
                            "nx2_verdict": "NX2_GO",
                            "nx3_verdict": verdict,
                            "nx3_reason": reason,
                            "valid_for_ranking_count": int(len(valid_df)),
                            "best_delta": best_delta,
                            "best_cvar": best_cv,
                            "best_maxdd": best_dd,
                            "route_stress_lite_survivors": int(rs_lite),
                        }
                    )

                nx3_sum = pd.DataFrame(nx3_family_rows)
                nx3_rep_lines = [
                    "# NX3 Ablation Report",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Families entering NX3: `{sorted(nx2_go_fams)}`",
                    f"- Variants evaluated: `{len(ab_df)}`",
                    "",
                    "## Family-Level Verdicts",
                    "",
                    markdown_table(nx3_sum, ["family_id", "nx3_verdict", "nx3_reason", "valid_for_ranking_count", "best_delta", "best_cvar", "best_maxdd", "route_stress_lite_survivors"], n=10),
                    "",
                    "## Top Ablations",
                    "",
                    markdown_table(
                        ab_df.sort_values(["valid_for_ranking", "overall_delta_expectancy_exec_minus_baseline", "overall_cvar_improve_ratio", "overall_maxdd_improve_ratio"], ascending=[False, False, False, False]),
                        [
                            "family_id",
                            "variant_id",
                            "profile",
                            "valid_for_ranking",
                            "overall_exec_expectancy_net",
                            "overall_delta_expectancy_exec_minus_baseline",
                            "overall_cvar_improve_ratio",
                            "overall_maxdd_improve_ratio",
                            "overall_entry_rate",
                            "overall_exec_taker_share",
                            "route_pass_lite",
                            "stress_lite_pass",
                            "invalid_reason",
                        ],
                        n=25,
                    ),
                ]
                write_text(run_dir / "phaseNX3_ablation_report.md", "\n".join(nx3_rep_lines) + "\n")
                phases_executed.append("NX3")

                # NX4: family selection.
                furthest_phase = "NX4"
                nx4_rows: List[Dict[str, Any]] = []
                selected_family: Optional[str] = None
                selected_reason = ""

                for _, r in nx3_sum.iterrows():
                    fam = str(r["family_id"])
                    fam_df = ab_df[ab_df["family_id"] == fam].copy()
                    valid = fam_df[to_num(fam_df["valid_for_ranking"]).fillna(0).astype(int) == 1].copy()
                    path = valid[
                        (to_num(valid["overall_delta_expectancy_exec_minus_baseline"]) > 0.0)
                        & ((to_num(valid["overall_cvar_improve_ratio"]) > 0.0) | (to_num(valid["overall_maxdd_improve_ratio"]) > 0.0))
                    ].copy()
                    rs = path[
                        (to_num(path.get("route_pass_lite", 0)).fillna(0).astype(int) == 1)
                        & (to_num(path.get("stress_lite_pass", 0)).fillna(0).astype(int) == 1)
                    ].copy()
                    best_score = float("-inf")
                    if not rs.empty:
                        rs = rs.copy()
                        rs["frontier_score"] = (
                            to_num(rs["overall_delta_expectancy_exec_minus_baseline"]).fillna(-1e9)
                            + 0.35 * to_num(rs["overall_cvar_improve_ratio"]).fillna(-1e9)
                            + 0.35 * to_num(rs["overall_maxdd_improve_ratio"]).fillna(-1e9)
                        )
                        best_score = float(to_num(rs["frontier_score"]).max())
                    nx4_rows.append(
                        {
                            "family_id": fam,
                            "nx3_verdict": str(r["nx3_verdict"]),
                            "valid_for_ranking_count": int(len(valid)),
                            "path_supported_count": int(len(path)),
                            "route_stress_lite_count": int(len(rs)),
                            "best_frontier_score": best_score,
                        }
                    )

                nx4_df = pd.DataFrame(nx4_rows).sort_values(["best_frontier_score", "route_stress_lite_count", "valid_for_ranking_count"], ascending=[False, False, False]).reset_index(drop=True)
                nx4_df.to_csv(run_dir / "phaseNX4_family_comparison.csv", index=False)

                go_df = nx4_df[(nx4_df["nx3_verdict"] == "FAMILY_GO_TO_GA") & (np.isfinite(to_num(nx4_df["best_frontier_score"])))]
                if not go_df.empty:
                    selected_family = str(go_df.iloc[0]["family_id"])
                    selected_reason = "highest frontier score among NX3_GO families with route/stress-lite survivors"
                    final_classification = "NX4_GO_TO_NX5"
                    mainline_status = "CONTINUE"
                    branch_decisions["NX4"] = {
                        "selected_family": selected_family,
                        "reason": selected_reason,
                    }
                else:
                    selected_family = None
                    final_classification = "STOP_NO_GO_NO_FAMILY_FRONTIER"
                    mainline_status = "STOP_NO_GO_NO_FAMILY_FRONTIER"
                    branch_decisions["NX4"] = {
                        "selected_family": None,
                        "reason": "no family met NX3 GO criteria",
                    }

                nx4_rep_lines = [
                    "# NX4 Selection Report",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Selected family: `{selected_family if selected_family else 'none'}`",
                    f"- Reason: {selected_reason if selected_reason else 'no qualifying family'}",
                    "",
                    "## Comparison",
                    "",
                    markdown_table(nx4_df, ["family_id", "nx3_verdict", "valid_for_ranking_count", "path_supported_count", "route_stress_lite_count", "best_frontier_score"], n=10),
                ]
                write_text(run_dir / "phaseNX4_selection_report.md", "\n".join(nx4_rep_lines) + "\n")

                nx4_decision_lines = [
                    "# NX4 Decision",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Decision: `{final_classification}`",
                    f"- Mainline status: `{mainline_status}`",
                    f"- Selected family: `{selected_family if selected_family else 'none'}`",
                ]
                write_text(run_dir / "phaseNX4_decision_next_step.md", "\n".join(nx4_decision_lines) + "\n")
                phases_executed.append("NX4")

                # Stop at NX4 no-family branch.
                if selected_family is None:
                    furthest_phase = "NX4"
                else:
                    # NX5: bounded GA pilot on selected family.
                    furthest_phase = "NX5"
                    rng = random.Random(int(args.seed) + 101)
                    fam_ab_seed = [v for v in ab_vars if v.family_id == selected_family]
                    if fam_ab_seed:
                        fam_ab_seed = sorted(
                            fam_ab_seed,
                            key=lambda z: z.variant_id,
                        )
                    seed_pool = fam_ab_seed[: min(8, len(fam_ab_seed))]
                    if not seed_pool:
                        seed_pool = [
                            FamilyVariant(
                                family_id=selected_family,
                                variant_id="seed_mid",
                                profile="mid",
                                params=_sample_family_params(selected_family, rng=rng, profile="mid"),
                                base_genome=copy.deepcopy(base_genome),
                                seed_origin="fallback_seed",
                            )
                        ]

                    pop_size = int(args.ga_pop)
                    gens = int(args.ga_gens)
                    population: List[FamilyVariant] = []
                    for sv in seed_pool:
                        population.append(copy.deepcopy(sv))
                    while len(population) < pop_size:
                        sp = seed_pool[rng.randrange(0, len(seed_pool))]
                        profile = random.choice(["conservative", "mid", "aggressive"])
                        population.append(
                            FamilyVariant(
                                family_id=selected_family,
                                variant_id=f"ga_seed_{len(population):04d}",
                                profile=profile,
                                params=mutate_family_params(selected_family, sp.params, rng=rng),
                                base_genome=copy.deepcopy(base_genome),
                                seed_origin="seed_mutation",
                            )
                        )
                    population = population[:pop_size]

                    eval_cache: Dict[str, Dict[str, Any]] = {}
                    telemetry: Dict[str, Any] = {
                        "generated_utc": utc_now(),
                        "family_id": selected_family,
                        "pop": pop_size,
                        "gens": gens,
                        "generation_stats": [],
                        "duplicate_param_reuse": 0,
                        "origin_counts": {},
                    }
                    all_rows: List[Dict[str, Any]] = []

                    next_id = 0
                    for gen in range(gens):
                        gen_rows: List[Dict[str, Any]] = []
                        for indiv in population:
                            h = variant_hash(indiv)
                            if h in eval_cache:
                                m = copy.deepcopy(eval_cache[h])
                                telemetry["duplicate_param_reuse"] = int(telemetry["duplicate_param_reuse"] + 1)
                            else:
                                ev = evaluate_family_variant(variant=indiv, bundles=bundles, args=exec_args, detailed=False)
                                m = copy.deepcopy(ev.metrics)
                                eval_cache[h] = m
                            row = copy.deepcopy(m)
                            row["candidate_id"] = f"NX5_{next_id:04d}"
                            row["candidate_hash"] = h
                            row["generation"] = int(gen)
                            row["seed_origin"] = str(indiv.seed_origin)
                            row["profile"] = str(indiv.profile)
                            row["params_json"] = json.dumps(indiv.params, sort_keys=True)
                            row["fitness"] = float(ga_fitness(row))
                            gen_rows.append(row)
                            all_rows.append(copy.deepcopy(row))
                            telemetry["origin_counts"][str(indiv.seed_origin)] = int(telemetry["origin_counts"].get(str(indiv.seed_origin), 0) + 1)
                            next_id += 1

                        gdf = pd.DataFrame(gen_rows)
                        if gdf.empty:
                            break
                        valid_n = int((to_num(gdf["valid_for_ranking"]).fillna(0).astype(int) == 1).sum())
                        best = gdf.sort_values(["fitness", "overall_delta_expectancy_exec_minus_baseline"], ascending=[False, False]).head(1)
                        telemetry["generation_stats"].append(
                            {
                                "generation": int(gen),
                                "population_size": int(len(gdf)),
                                "valid_for_ranking_count": int(valid_n),
                                "best_fitness": float(best.iloc[0]["fitness"]) if not best.empty else float("nan"),
                                "best_candidate_hash": str(best.iloc[0]["candidate_hash"]) if not best.empty else "",
                            }
                        )

                        # breed next generation
                        pool = gdf.sort_values(["fitness", "overall_delta_expectancy_exec_minus_baseline"], ascending=[False, False]).reset_index(drop=True)
                        elites = pool.head(max(8, pop_size // 10)).copy()
                        parent_pool = pool.head(max(12, pop_size // 3)).copy()

                        parent_vars: List[Dict[str, Any]] = []
                        for _, rr in parent_pool.iterrows():
                            try:
                                parent_vars.append(json.loads(str(rr["params_json"])))
                            except Exception:
                                continue
                        if not parent_vars:
                            parent_vars = [copy.deepcopy(seed_pool[0].params)]

                        new_pop: List[FamilyVariant] = []
                        for _, rr in elites.iterrows():
                            try:
                                pp = json.loads(str(rr["params_json"]))
                            except Exception:
                                continue
                            new_pop.append(
                                FamilyVariant(
                                    family_id=selected_family,
                                    variant_id=f"elite_{len(new_pop):04d}",
                                    profile="elite",
                                    params=pp,
                                    base_genome=copy.deepcopy(base_genome),
                                    seed_origin="elite",
                                )
                            )

                        while len(new_pop) < pop_size:
                            u = rng.random()
                            if u < 0.35 and len(parent_vars) >= 2:
                                pa = parent_vars[rng.randrange(0, len(parent_vars))]
                                pb = parent_vars[rng.randrange(0, len(parent_vars))]
                                ch = crossover_family_params(pa, pb, rng=rng)
                                new_pop.append(
                                    FamilyVariant(
                                        family_id=selected_family,
                                        variant_id=f"child_{len(new_pop):04d}",
                                        profile="crossover",
                                        params=ch,
                                        base_genome=copy.deepcopy(base_genome),
                                        seed_origin="crossover",
                                    )
                                )
                            elif u < 0.85:
                                pa = parent_vars[rng.randrange(0, len(parent_vars))]
                                ch = mutate_family_params(selected_family, pa, rng=rng)
                                new_pop.append(
                                    FamilyVariant(
                                        family_id=selected_family,
                                        variant_id=f"mut_{len(new_pop):04d}",
                                        profile="mutation",
                                        params=ch,
                                        base_genome=copy.deepcopy(base_genome),
                                        seed_origin="mutation",
                                    )
                                )
                            else:
                                pr = random.choice(["conservative", "mid", "aggressive"])
                                ch = _sample_family_params(selected_family, rng=rng, profile=pr)
                                new_pop.append(
                                    FamilyVariant(
                                        family_id=selected_family,
                                        variant_id=f"exp_{len(new_pop):04d}",
                                        profile=pr,
                                        params=ch,
                                        base_genome=copy.deepcopy(base_genome),
                                        seed_origin="explore",
                                    )
                                )
                        population = new_pop[:pop_size]

                    pilot_df = pd.DataFrame(all_rows).drop_duplicates(subset=["candidate_hash"]).copy()
                    if pilot_df.empty:
                        final_classification = "STOP_NO_GO_GA_PILOT"
                        mainline_status = final_classification
                        branch_decisions["NX5"] = {"classification": final_classification, "reason": "empty GA pilot result set"}
                        write_text(run_dir / "phaseNX5_pilot_report.md", "# NX5 Pilot Report\n\n- No GA pilot rows were produced.\n")
                        pd.DataFrame().to_csv(run_dir / "phaseNX5_ga_pilot_results.csv", index=False)
                        json_dump(run_dir / "phaseNX5_invalid_reason_histogram.json", {})
                        pd.DataFrame().to_csv(run_dir / "phaseNX5_duplicate_variant_map.csv", index=False)
                        write_text(run_dir / "phaseNX5_effective_trials_summary.md", "# NX5 Effective Trials Summary\n\n- n/a\n")
                        pd.DataFrame().to_csv(run_dir / "phaseNX5_shortlist_significance.csv", index=False)
                        json_dump(run_dir / "phaseNX5_sampler_telemetry.json", telemetry)
                        json_dump(run_dir / "phaseNX5_run_manifest.json", {"classification": final_classification})
                        phases_executed.append("NX5")
                    else:
                        pilot_df = pilot_df.sort_values(["fitness", "overall_delta_expectancy_exec_minus_baseline"], ascending=[False, False]).reset_index(drop=True)
                        pilot_df.to_csv(run_dir / "phaseNX5_ga_pilot_results.csv", index=False)

                        invalid_hist = parse_invalid_hist(pilot_df)
                        json_dump(run_dir / "phaseNX5_invalid_reason_histogram.json", invalid_hist)

                        pilot_df["metric_signature"] = metric_signature(pilot_df)
                        sig_to_first: Dict[str, str] = {}
                        dup_rows: List[Dict[str, Any]] = []
                        for _, r in pilot_df.iterrows():
                            sig = str(r["metric_signature"])
                            cid = str(r["candidate_id"])
                            if sig in sig_to_first:
                                dup_of = sig_to_first[sig]
                            else:
                                dup_of = ""
                                sig_to_first[sig] = cid
                            dup_rows.append(
                                {
                                    "candidate_id": cid,
                                    "duplicate_of_candidate_id": dup_of,
                                    "candidate_hash": str(r["candidate_hash"]),
                                    "metric_signature": sig,
                                }
                            )
                        dup_df = pd.DataFrame(dup_rows)
                        dup_df.to_csv(run_dir / "phaseNX5_duplicate_variant_map.csv", index=False)

                        valid = pilot_df[to_num(pilot_df["valid_for_ranking"]).fillna(0).astype(int) == 1].copy()
                        valid_nondup = valid.merge(dup_df, on="candidate_id", how="left")
                        valid_nondup = valid_nondup[valid_nondup["duplicate_of_candidate_id"].fillna("").astype(str).str.strip() == ""].copy()

                        metric_cols = [
                            "overall_delta_expectancy_exec_minus_baseline",
                            "overall_cvar_improve_ratio",
                            "overall_maxdd_improve_ratio",
                            "overall_entry_rate",
                            "overall_exec_taker_share",
                            "overall_exec_median_fill_delay_min",
                        ]
                        mat = valid_nondup[metric_cols].to_numpy(dtype=float) if not valid_nondup.empty else np.zeros((0, len(metric_cols)), dtype=float)
                        n_eff, avg_abs = pu.effective_trials_from_corr(mat)
                        uncorr_trials = int(valid_nondup["metric_signature"].nunique()) if not valid_nondup.empty else 0

                        et_lines = [
                            "# NX5 Effective Trials Summary",
                            "",
                            f"- Generated UTC: {utc_now()}",
                            f"- Valid for ranking (raw): `{int(len(valid))}`",
                            f"- Valid non-duplicate metric signatures: `{int(uncorr_trials)}`",
                            f"- Duplicate-adjusted effective trials (corr-adjusted): `{float(n_eff):.6f}`",
                            f"- Average absolute correlation proxy: `{float(avg_abs):.6f}`",
                            "- Reality-check TODO: run SPA/White-style multiple-testing correction before any promotion decision.",
                        ]
                        write_text(run_dir / "phaseNX5_effective_trials_summary.md", "\n".join(et_lines) + "\n")

                        shortlist = valid_nondup.sort_values(["fitness", "overall_delta_expectancy_exec_minus_baseline"], ascending=[False, False]).head(12).copy()
                        if not shortlist.empty:
                            shortlist["psr_proxy"] = shortlist.apply(
                                lambda r: norm_cdf(
                                    z_proxy(
                                        float(r.get("overall_exec_expectancy_net", np.nan)),
                                        max(1e-12, float(r.get("overall_exec_pnl_std", np.nan))),
                                        max(2.0, float(r.get("overall_entries_valid", np.nan))),
                                    )
                                ),
                                axis=1,
                            )
                            dsr_denom = max(1.0, math.sqrt(max(1.0, float(n_eff))))
                            shortlist["dsr_proxy"] = to_num(shortlist["psr_proxy"]).fillna(0.0) / dsr_denom
                        else:
                            shortlist["psr_proxy"] = []
                            shortlist["dsr_proxy"] = []
                        shortlist.to_csv(run_dir / "phaseNX5_shortlist_significance.csv", index=False)

                        json_dump(run_dir / "phaseNX5_sampler_telemetry.json", telemetry)

                        top3 = shortlist.head(3).copy()
                        sane_top = int(
                            (not top3.empty)
                            and np.isfinite(to_num(top3["overall_delta_expectancy_exec_minus_baseline"]).to_numpy(dtype=float)).all()
                            and np.isfinite(to_num(top3["overall_cvar_improve_ratio"]).to_numpy(dtype=float)).all()
                            and np.isfinite(to_num(top3["overall_maxdd_improve_ratio"]).to_numpy(dtype=float)).all()
                        )
                        nontrivial_div = int((uncorr_trials >= 2) and (float(n_eff) >= 1.5))
                        not_lucky_corner = int(
                            (not top3.empty)
                            and int((to_num(top3["overall_delta_expectancy_exec_minus_baseline"]) > 0.0).sum()) >= 2
                            and int(((to_num(top3["overall_cvar_improve_ratio"]) > 0.0) | (to_num(top3["overall_maxdd_improve_ratio"]) > 0.0)).sum()) >= 2
                        )

                        go_nx6 = int(
                            int(len(valid)) > 0
                            and sane_top == 1
                            and nontrivial_div == 1
                            and not_lucky_corner == 1
                        )

                        nx5_manifest = {
                            "generated_utc": utc_now(),
                            "family_id": selected_family,
                            "pop": pop_size,
                            "gens": gens,
                            "valid_for_ranking_count": int(len(valid)),
                            "valid_nonduplicate_count": int(len(valid_nondup)),
                            "effective_trials_uncorrelated": int(uncorr_trials),
                            "effective_trials_corr_adjusted": float(n_eff),
                            "avg_abs_corr_proxy": float(avg_abs),
                            "decision_go_nx6": int(go_nx6),
                            "reason_flags": {
                                "sane_top": sane_top,
                                "nontrivial_diversity": nontrivial_div,
                                "not_lucky_corner": not_lucky_corner,
                            },
                        }
                        json_dump(run_dir / "phaseNX5_run_manifest.json", nx5_manifest)

                        rep_lines = [
                            "# NX5 Pilot Report",
                            "",
                            f"- Generated UTC: {utc_now()}",
                            f"- Selected family: `{selected_family}`",
                            f"- Population x generations: `{pop_size} x {gens}`",
                            f"- Evaluated unique candidates: `{len(pilot_df)}`",
                            f"- Valid for ranking: `{len(valid)}`",
                            f"- Valid non-duplicate: `{len(valid_nondup)}`",
                            f"- Effective trials (uncorrelated): `{uncorr_trials}`",
                            f"- Effective trials (corr-adjusted): `{float(n_eff):.6f}`",
                            "",
                            "## Top Candidates",
                            "",
                            markdown_table(
                                shortlist,
                                [
                                    "candidate_id",
                                    "fitness",
                                    "valid_for_ranking",
                                    "overall_exec_expectancy_net",
                                    "overall_delta_expectancy_exec_minus_baseline",
                                    "overall_cvar_improve_ratio",
                                    "overall_maxdd_improve_ratio",
                                    "overall_entry_rate",
                                    "overall_exec_taker_share",
                                    "overall_exec_median_fill_delay_min",
                                    "psr_proxy",
                                    "dsr_proxy",
                                ],
                                n=12,
                            ),
                        ]
                        write_text(run_dir / "phaseNX5_pilot_report.md", "\n".join(rep_lines) + "\n")
                        phases_executed.append("NX5")

                        if go_nx6 != 1:
                            furthest_phase = "NX5"
                            final_classification = "STOP_NO_GO_GA_PILOT"
                            mainline_status = final_classification
                            branch_decisions["NX5"] = {
                                "classification": final_classification,
                                "valid_for_ranking_count": int(len(valid)),
                                "effective_trials_uncorrelated": int(uncorr_trials),
                                "effective_trials_corr_adjusted": float(n_eff),
                                "top_invalid_reasons": dict(list(invalid_hist.items())[:8]),
                                "gains_participation_only": int(not_lucky_corner == 0),
                            }
                        else:
                            # NX6 robustness gate.
                            furthest_phase = "NX6"
                            survivors = shortlist.head(3).copy()

                            route_rows: List[Dict[str, Any]] = []
                            split_rows: List[Dict[str, Any]] = []
                            stress_rows: List[Dict[str, Any]] = []
                            boot_rows: List[Dict[str, Any]] = []
                            fail_rows: List[Dict[str, Any]] = []

                            survivor_variants: Dict[str, FamilyVariant] = {}
                            for _, rr in survivors.iterrows():
                                try:
                                    pp = json.loads(str(rr["params_json"]))
                                except Exception:
                                    continue
                                cid = str(rr["candidate_id"])
                                vv = FamilyVariant(
                                    family_id=selected_family,
                                    variant_id=cid,
                                    profile="nx6_survivor",
                                    params=pp,
                                    base_genome=copy.deepcopy(base_genome),
                                    seed_origin="nx5_survivor",
                                )
                                survivor_variants[cid] = vv

                            for cid, vv in survivor_variants.items():
                                ev_full = evaluate_family_variant(variant=vv, bundles=bundles, args=exec_args, detailed=True)
                                mfull = ev_full.metrics

                                for rid, rb in route_bundles.items():
                                    evr = evaluate_family_variant(variant=vv, bundles=[rb], args=exec_args, detailed=False)
                                    mr = evr.metrics
                                    route_rows.append(
                                        {
                                            "candidate_id": cid,
                                            "route_id": rid,
                                            "valid_for_ranking": int(mr.get("valid_for_ranking", 0)),
                                            "delta_expectancy": float(mr.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
                                            "cvar_improve_ratio": float(mr.get("overall_cvar_improve_ratio", np.nan)),
                                            "maxdd_improve_ratio": float(mr.get("overall_maxdd_improve_ratio", np.nan)),
                                            "entry_rate": float(mr.get("overall_entry_rate", np.nan)),
                                        }
                                    )

                                split_df = ev_full.split_df.copy()
                                split_min = float(to_num(split_df.get("exec_mean_expectancy_net", pd.Series(dtype=float))).min()) if not split_df.empty else float("nan")
                                split_med = float(to_num(split_df.get("exec_mean_expectancy_net", pd.Series(dtype=float))).median()) if not split_df.empty else float("nan")
                                split_pass = int(np.isfinite(split_min) and np.isfinite(split_med) and split_min >= (split_med - abs(split_med) * float(exec_args.stability_drawdown_mult)))
                                split_rows.append(
                                    {
                                        "candidate_id": cid,
                                        "split_count": int(len(split_df)),
                                        "min_split_expectancy_net": split_min,
                                        "median_split_expectancy_net": split_med,
                                        "std_split_expectancy_net": float(mfull.get("std_split_expectancy_net", np.nan)),
                                        "split_stability_pass": int(split_pass),
                                    }
                                )

                                scenarios = [
                                    ("S00_base", {"fee_mult": 1.0, "slip_add": 0.0}),
                                    ("S01_cost125", {"fee_mult": 1.25, "slip_add": 0.0}),
                                    ("S02_slip1", {"fee_mult": 1.0, "slip_add": 1.0}),
                                    ("S03_cost125_slip1", {"fee_mult": 1.25, "slip_add": 1.0}),
                                    ("S04_cost150_slip2", {"fee_mult": 1.50, "slip_add": 2.0}),
                                ]
                                scen_pass = 0
                                for sid, sc in scenarios:
                                    evs = evaluate_family_variant(variant=vv, bundles=bundles, args=exec_args, detailed=False, scenario=sc)
                                    ms = evs.metrics
                                    pflag = int(
                                        int(ms.get("valid_for_ranking", 0)) == 1
                                        and float(ms.get("overall_delta_expectancy_exec_minus_baseline", np.nan)) > 0.0
                                        and float(ms.get("overall_cvar_improve_ratio", np.nan)) >= 0.0
                                        and float(ms.get("overall_maxdd_improve_ratio", np.nan)) >= 0.0
                                    )
                                    scen_pass += pflag
                                    stress_rows.append(
                                        {
                                            "candidate_id": cid,
                                            "scenario_id": sid,
                                            "valid_for_ranking": int(ms.get("valid_for_ranking", 0)),
                                            "delta_expectancy": float(ms.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
                                            "cvar_improve_ratio": float(ms.get("overall_cvar_improve_ratio", np.nan)),
                                            "maxdd_improve_ratio": float(ms.get("overall_maxdd_improve_ratio", np.nan)),
                                            "scenario_pass": int(pflag),
                                        }
                                    )

                                br = bootstrap_pass_rate(ev_full.signal_df, n_boot=250, seed=int(args.seed) + hash(cid) % 100000)
                                boot_rows.append(
                                    {
                                        "candidate_id": cid,
                                        "bootstrap_pass_rate": float(br),
                                        "bootstrap_pass": int(np.isfinite(br) and br >= 0.60),
                                        "n_boot": 250,
                                    }
                                )

                            route_df = pd.DataFrame(route_rows)
                            split_stab_df = pd.DataFrame(split_rows)
                            stress_df = pd.DataFrame(stress_rows)
                            boot_df = pd.DataFrame(boot_rows)

                            route_df.to_csv(run_dir / "phaseNX6_route_checks.csv", index=False)
                            split_stab_df.to_csv(run_dir / "phaseNX6_split_stability.csv", index=False)
                            stress_df.to_csv(run_dir / "phaseNX6_stress_matrix.csv", index=False)
                            boot_df.to_csv(run_dir / "phaseNX6_bootstrap_summary.csv", index=False)

                            robust_rows: List[Dict[str, Any]] = []
                            for cid in sorted(set(route_df.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist())):
                                rr = route_df[route_df["candidate_id"] == cid].copy()
                                ss = stress_df[stress_df["candidate_id"] == cid].copy()
                                bb = boot_df[boot_df["candidate_id"] == cid].copy()
                                sp = split_stab_df[split_stab_df["candidate_id"] == cid].copy()

                                route_pass_rate = float(np.mean((to_num(rr["valid_for_ranking"]) == 1) & (to_num(rr["delta_expectancy"]) > 0.0))) if not rr.empty else 0.0
                                stress_pass_rate = float(np.mean(to_num(ss["scenario_pass"]) == 1)) if not ss.empty else 0.0
                                boot_rate = float(to_num(bb["bootstrap_pass_rate"]).iloc[0]) if not bb.empty else float("nan")
                                split_pass = int(to_num(sp["split_stability_pass"]).iloc[0]) if not sp.empty else 0

                                robust = int(
                                    route_pass_rate >= 1.0
                                    and stress_pass_rate >= 0.60
                                    and split_pass == 1
                                    and np.isfinite(boot_rate)
                                    and boot_rate >= 0.60
                                )
                                robust_rows.append(
                                    {
                                        "candidate_id": cid,
                                        "route_pass_rate": route_pass_rate,
                                        "stress_pass_rate": stress_pass_rate,
                                        "bootstrap_pass_rate": boot_rate,
                                        "split_stability_pass": split_pass,
                                        "robust_survivor": robust,
                                    }
                                )

                            robust_df = pd.DataFrame(robust_rows).sort_values(["robust_survivor", "stress_pass_rate", "bootstrap_pass_rate"], ascending=[False, False, False]).reset_index(drop=True)

                            # failure mode breakdown
                            for _, rr in robust_df.iterrows():
                                cid = str(rr["candidate_id"])
                                rsub = route_df[route_df["candidate_id"] == cid].copy()
                                ssub = stress_df[stress_df["candidate_id"] == cid].copy()
                                bsub = boot_df[boot_df["candidate_id"] == cid].copy()
                                psub = split_stab_df[split_stab_df["candidate_id"] == cid].copy()
                                fail_rows.append(
                                    {
                                        "candidate_id": cid,
                                        "route_fail_count": int((~((to_num(rsub["valid_for_ranking"]) == 1) & (to_num(rsub["delta_expectancy"]) > 0.0))).sum()) if not rsub.empty else 0,
                                        "stress_fail_count": int((to_num(ssub["scenario_pass"]).fillna(0).astype(int) == 0).sum()) if not ssub.empty else 0,
                                        "bootstrap_fail": int((not bsub.empty) and (to_num(bsub["bootstrap_pass"]).iloc[0] != 1)),
                                        "split_fail": int((not psub.empty) and (to_num(psub["split_stability_pass"]).iloc[0] != 1)),
                                    }
                                )

                            fail_df = pd.DataFrame(fail_rows)
                            fail_df.to_csv(run_dir / "phaseNX6_failure_mode_breakdown.csv", index=False)

                            robust_survivors = robust_df[robust_df["robust_survivor"] == 1].copy()
                            stable_neighborhood = 0
                            neigh_robust = 0
                            neigh_total = 0
                            if not robust_survivors.empty:
                                best_cid = str(robust_survivors.iloc[0]["candidate_id"])
                                if best_cid in survivor_variants:
                                    stable_neighborhood, neigh_robust, neigh_total = neighborhood_stability(
                                        best=survivor_variants[best_cid], bundles=bundles, args=exec_args, seed=int(args.seed) + 303
                                    )

                            nx6_go = int((not robust_survivors.empty) and stable_neighborhood == 1)
                            if nx6_go == 1:
                                final_classification = "CONTINUE_READY_FOR_CONFIRMATION"
                                mainline_status = "CONTINUE_READY_FOR_CONFIRMATION"
                                branch_decisions["NX6"] = {
                                    "classification": final_classification,
                                    "robust_survivor_count": int(len(robust_survivors)),
                                    "stable_neighborhood": int(stable_neighborhood),
                                    "neighborhood_robust": int(neigh_robust),
                                    "neighborhood_total": int(neigh_total),
                                }

                                prompt = (
                                    "ROLE\n"
                                    "You are in Phase NY confirmation mode for execution-family validation under frozen contract.\n\n"
                                    "MISSION\n"
                                    "Run OOS/paper-style confirmation only for surviving NX winners. Keep hard gates and contract lock unchanged.\n\n"
                                    "RULES\n"
                                    "1) Keep representative subset/fee/metrics lock with allow_freeze_hash_mismatch=0.\n"
                                    "2) Use winner set from NX6 robust survivors only; no new family expansion.\n"
                                    "3) Keep duplicate-adjusted effective trials, PSR/DSR proxy reporting, and robustness matrix mandatory.\n"
                                    "4) Stop NO_GO if route/split/stress/bootstrap deteriorate versus NX6 medians.\n"
                                    "5) Produce explicit forensic at first contract or robustness failure.\n"
                                )
                                write_text(run_dir / "ready_to_launch_phaseNY_confirmation_prompt.txt", prompt)
                                phases_executed.append("NX7")
                                furthest_phase = "NX7"
                            else:
                                final_classification = "STOP_NO_GO_FRAGILE"
                                mainline_status = final_classification
                                branch_decisions["NX6"] = {
                                    "classification": final_classification,
                                    "robust_survivor_count": int(len(robust_survivors)),
                                    "stable_neighborhood": int(stable_neighborhood),
                                    "neighborhood_robust": int(neigh_robust),
                                    "neighborhood_total": int(neigh_total),
                                }

                            rep_lines = [
                                "# NX6 Top Survivors Report",
                                "",
                                f"- Generated UTC: {utc_now()}",
                                f"- Robust survivors: `{int(len(robust_survivors))}`",
                                f"- Stable neighborhood: `{int(stable_neighborhood)}` ({neigh_robust}/{neigh_total})",
                                "",
                                "## Robustness Aggregate",
                                "",
                                markdown_table(robust_df, ["candidate_id", "route_pass_rate", "stress_pass_rate", "bootstrap_pass_rate", "split_stability_pass", "robust_survivor"], n=10),
                            ]
                            write_text(run_dir / "phaseNX6_top_survivors_report.md", "\n".join(rep_lines) + "\n")

                            dec_lines = [
                                "# NX6 Decision",
                                "",
                                f"- Generated UTC: {utc_now()}",
                                f"- Classification: `{final_classification}`",
                                f"- Mainline status: `{mainline_status}`",
                                f"- Robust survivor count: `{int(len(robust_survivors))}`",
                                f"- Stable neighborhood: `{int(stable_neighborhood)}` ({neigh_robust}/{neigh_total})",
                            ]
                            write_text(run_dir / "phaseNX6_decision_next_step.md", "\n".join(dec_lines) + "\n")
                            phases_executed.append("NX6")

    # Final run manifest.
    if not final_classification:
        final_classification = "STOP_NO_GO_CONTRACT"
        mainline_status = "STOP_NO_GO_CONTRACT"

    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "git_and_file_hashes": {
            "git": git_meta,
            "locked_files": locked_hashes,
            "script_sha256": sha256_file(Path(__file__)) if Path(__file__).exists() else "",
        },
        "frozen_contract_validation": contract_obj,
        "phases_executed": phases_executed,
        "branch_decisions": branch_decisions,
        "compute_budgets_used": compute_budgets,
        "final_classification": final_classification,
        "mainline_status": mainline_status,
        "furthest_phase": furthest_phase,
        "duration_sec": float(time.time() - t_start),
    }
    json_dump(run_dir / "run_manifest.json", manifest)

    print(json.dumps({"run_dir": str(run_dir), "furthest_phase": furthest_phase, "classification": final_classification, "mainline_status": mainline_status}, sort_keys=True))


if __name__ == "__main__":
    main()
