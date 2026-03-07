#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports" / "execution_layer"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_tag() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _resolve(p: str | Path) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = (PROJECT_ROOT / pp).resolve()
    return pp


def _latest_run_dir(prefix: str) -> Path:
    cands = sorted([p for p in REPORTS_ROOT.glob(f"{prefix}_*") if p.is_dir()], key=lambda x: x.name)
    if not cands:
        raise FileNotFoundError(f"No dirs matching {prefix}_* in {REPORTS_ROOT}")
    return cands[-1]


def _json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(float(b)) <= 1e-12:
        return float("nan")
    return float(a / b)


def _spearman_no_scipy(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return float("nan")
    xr = x[m].rank(method="average")
    yr = y[m].rank(method="average")
    return float(xr.corr(yr, method="pearson"))


def _sanitize_token(tok: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "_", str(tok).strip())
    x = re.sub(r"_+", "_", x).strip("_")
    return x.lower() or "empty"


@dataclass
class Thresholds:
    symbol: str
    mode: str
    hard_min_entry_rate_symbol: float
    hard_min_entry_rate_overall: float
    hard_min_trades_symbol: int
    hard_min_trade_frac_symbol: float
    hard_min_trades_overall: int
    hard_min_trade_frac_overall: float
    hard_max_taker_share: float
    hard_max_median_fill_delay_min: float
    hard_max_p95_fill_delay_min: float
    symbol_min_entry_rate: float
    symbol_max_fill_delay_min: float
    symbol_max_taker_share_cap: float
    execution_config_path: str


def _load_thresholds_for_run(run_dir: Path) -> Thresholds:
    # Pull defaults directly from current GA parser to avoid hardcoding.
    from src.execution import ga_exec_3m_opt as ga

    ap = ga.build_arg_parser()
    d = ap.parse_args([])

    ga_cfg_path = run_dir / "ga_config.yaml"
    ga_cfg: Dict[str, Any] = {}
    if ga_cfg_path.exists():
        txt = ga_cfg_path.read_text(encoding="utf-8")
        try:
            ga_cfg = json.loads(txt)
        except Exception:
            ga_cfg = {}
    mode = str(ga_cfg.get("mode", "tight")).lower()
    symbols = ga_cfg.get("symbols", [])
    symbol = str(symbols[0]).upper() if isinstance(symbols, list) and symbols else "SOLUSDT"

    exec_cfg_path = ""
    signals_blob = ga_cfg.get("signals", {}) if isinstance(ga_cfg, dict) else {}
    if isinstance(signals_blob, dict):
        exec_cfg_path = str(signals_blob.get("execution_config_path", "")).strip()
    if not exec_cfg_path:
        exec_cfg_path = str(PROJECT_ROOT / "configs" / "execution_configs.yaml")

    all_exec_cfg = ga._load_execution_config(_resolve(exec_cfg_path))
    sym_cfg = ga._symbol_exec_config(all_exec_cfg, symbol)
    if mode == "tight":
        cons = dict(sym_cfg.get("tight_constraints", sym_cfg.get("constraints", {})))
        sym_min_entry = float(cons.get("min_entry_rate", d.tight_min_entry_rate_default))
        sym_max_delay = float(cons.get("max_fill_delay_min", d.tight_max_fill_delay_default))
        sym_max_taker = float(cons.get("max_taker_share", d.tight_max_taker_share_default))
    else:
        cons = dict(sym_cfg.get("constraints", {}))
        sym_min_entry = float(cons.get("min_entry_rate", d.min_entry_rate_default))
        sym_max_delay = float(cons.get("max_fill_delay_min", d.max_fill_delay_default))
        sym_max_taker = float(cons.get("max_taker_share", d.max_taker_share_default))

    return Thresholds(
        symbol=symbol,
        mode=mode,
        hard_min_entry_rate_symbol=float(d.hard_min_entry_rate_symbol),
        hard_min_entry_rate_overall=float(d.hard_min_entry_rate_overall),
        hard_min_trades_symbol=int(d.hard_min_trades_symbol),
        hard_min_trade_frac_symbol=float(d.hard_min_trade_frac_symbol),
        hard_min_trades_overall=int(d.hard_min_trades_overall),
        hard_min_trade_frac_overall=float(d.hard_min_trade_frac_overall),
        hard_max_taker_share=float(d.hard_max_taker_share),
        hard_max_median_fill_delay_min=float(d.hard_max_median_fill_delay_min),
        hard_max_p95_fill_delay_min=float(d.hard_max_p95_fill_delay_min),
        symbol_min_entry_rate=float(sym_min_entry),
        symbol_max_fill_delay_min=float(sym_max_delay),
        symbol_max_taker_share_cap=float(sym_max_taker),
        execution_config_path=str(_resolve(exec_cfg_path)),
    )


def _ensure_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[col]


def _ensure_int_flag(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df[col]


def _parse_invalid_reason_flags(df: pd.DataFrame) -> List[str]:
    if "invalid_reason" not in df.columns:
        df["invalid_reason"] = ""
    df["invalid_reason"] = df["invalid_reason"].astype(str).fillna("")

    token_set: set[str] = set()
    for s in df["invalid_reason"].tolist():
        for tok in str(s).split("|"):
            t = str(tok).strip()
            if t:
                token_set.add(t)
    tokens = sorted(token_set)
    for tok in tokens:
        cname = f"fail_reason__{_sanitize_token(tok)}"
        df[cname] = df["invalid_reason"].map(lambda x, t=tok: int(t in str(x).split("|"))).astype(int)
    return tokens


def _compute_row_thresholds(df: pd.DataFrame, th: Thresholds) -> pd.DataFrame:
    sig = _ensure_numeric(df, "overall_signals_total").fillna(0)
    over_from_frac = np.ceil(sig * float(th.hard_min_trade_frac_overall))
    sym_from_frac = np.ceil(sig * float(th.hard_min_trade_frac_symbol))
    df["th_overall_min_trades_base"] = np.maximum(float(th.hard_min_trades_overall), over_from_frac)
    df["th_symbol_min_trades_base"] = np.maximum(float(th.hard_min_trades_symbol), sym_from_frac)
    # Prefer explicit per-row threshold if present in run output.
    if "overall_min_trades_required" in df.columns:
        x = pd.to_numeric(df["overall_min_trades_required"], errors="coerce")
        df.loc[x.notna(), "th_overall_min_trades_base"] = x[x.notna()]

    df["th_overall_entry_rate_base"] = float(th.hard_min_entry_rate_overall)
    # Symbol-specific entry floor comes from execution config for this mode (tight/normal).
    df["th_symbol_entry_rate_base"] = float(th.symbol_min_entry_rate)

    # Overall hard realism gates.
    df["th_overall_max_taker_share_base"] = float(th.hard_max_taker_share)
    df["th_overall_max_median_delay_base"] = float(th.hard_max_median_fill_delay_min)
    df["th_overall_max_p95_delay_base"] = float(th.hard_max_p95_fill_delay_min)

    # Symbol realism gates (max_taker can be further reduced by genome max_taker_share).
    df["th_symbol_max_taker_cap_base"] = float(th.symbol_max_taker_share_cap)
    g_taker = pd.to_numeric(df.get("g_max_taker_share", np.nan), errors="coerce")
    df["th_symbol_max_taker_share_row"] = np.where(
        np.isfinite(g_taker),
        np.minimum(df["th_symbol_max_taker_cap_base"], g_taker),
        df["th_symbol_max_taker_cap_base"],
    )
    df["th_symbol_max_median_delay_base"] = float(th.symbol_max_fill_delay_min)
    df["th_symbol_max_p95_delay_base"] = float(th.hard_max_p95_fill_delay_min)
    return df


def _compute_gate_flags_and_slacks(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Standard numeric fields used for gate calculations.
    _ensure_numeric(df, "overall_entry_rate")
    _ensure_numeric(df, "overall_entries_valid")
    _ensure_numeric(df, "overall_exec_taker_share")
    _ensure_numeric(df, "overall_exec_median_fill_delay_min")
    _ensure_numeric(df, "overall_exec_p95_fill_delay_min")
    _ensure_numeric(df, "overall_exec_expectancy_net")
    _ensure_numeric(df, "overall_cvar_improve_ratio")
    _ensure_numeric(df, "overall_maxdd_improve_ratio")

    _ensure_int_flag(df, "valid_for_ranking")
    _ensure_int_flag(df, "hard_invalid")
    _ensure_int_flag(df, "constraint_pass")
    _ensure_int_flag(df, "realism_pass")
    _ensure_int_flag(df, "nan_pass")
    _ensure_int_flag(df, "data_quality_pass")
    _ensure_int_flag(df, "split_pass")

    # Participation + realism slacks.
    df["slack_overall_entry_rate"] = df["overall_entry_rate"] - df["th_overall_entry_rate_base"]
    df["slack_overall_min_trades"] = df["overall_entries_valid"] - df["th_overall_min_trades_base"]
    df["slack_symbol_entry_rate"] = df["overall_entry_rate"] - df["th_symbol_entry_rate_base"]
    df["slack_symbol_min_trades"] = df["overall_entries_valid"] - df["th_symbol_min_trades_base"]
    df["slack_overall_taker_share"] = df["th_overall_max_taker_share_base"] - df["overall_exec_taker_share"]
    df["slack_overall_median_fill_delay"] = df["th_overall_max_median_delay_base"] - df["overall_exec_median_fill_delay_min"]
    df["slack_overall_p95_fill_delay"] = df["th_overall_max_p95_delay_base"] - df["overall_exec_p95_fill_delay_min"]

    df["slack_symbol_taker_share"] = df["th_symbol_max_taker_share_row"] - df["overall_exec_taker_share"]
    df["slack_symbol_median_fill_delay"] = df["th_symbol_max_median_delay_base"] - df["overall_exec_median_fill_delay_min"]
    df["slack_symbol_p95_fill_delay"] = df["th_symbol_max_p95_delay_base"] - df["overall_exec_p95_fill_delay_min"]

    # Gate fail flags.
    df["fail_overall_entry_rate"] = (
        (~np.isfinite(df["overall_entry_rate"])) | (df["overall_entry_rate"] < df["th_overall_entry_rate_base"])
    ).astype(int)
    df["fail_overall_min_trades"] = (
        (~np.isfinite(df["overall_entries_valid"])) | (df["overall_entries_valid"] < df["th_overall_min_trades_base"])
    ).astype(int)
    df["fail_symbol_entry_rate"] = (
        (~np.isfinite(df["overall_entry_rate"])) | (df["overall_entry_rate"] < df["th_symbol_entry_rate_base"])
    ).astype(int)
    df["fail_symbol_min_trades"] = (
        (~np.isfinite(df["overall_entries_valid"])) | (df["overall_entries_valid"] < df["th_symbol_min_trades_base"])
    ).astype(int)
    df["fail_overall_taker_share"] = (
        (~np.isfinite(df["overall_exec_taker_share"])) | (df["overall_exec_taker_share"] > df["th_overall_max_taker_share_base"])
    ).astype(int)
    df["fail_overall_median_fill_delay"] = (
        (~np.isfinite(df["overall_exec_median_fill_delay_min"]))
        | (df["overall_exec_median_fill_delay_min"] > df["th_overall_max_median_delay_base"])
    ).astype(int)
    df["fail_overall_p95_fill_delay"] = (
        (~np.isfinite(df["overall_exec_p95_fill_delay_min"]))
        | (df["overall_exec_p95_fill_delay_min"] > df["th_overall_max_p95_delay_base"])
    ).astype(int)

    df["fail_symbol_taker_share"] = (
        (~np.isfinite(df["overall_exec_taker_share"])) | (df["overall_exec_taker_share"] > df["th_symbol_max_taker_share_row"])
    ).astype(int)
    df["fail_symbol_median_fill_delay"] = (
        (~np.isfinite(df["overall_exec_median_fill_delay_min"]))
        | (df["overall_exec_median_fill_delay_min"] > df["th_symbol_max_median_delay_base"])
    ).astype(int)
    df["fail_symbol_p95_fill_delay"] = (
        (~np.isfinite(df["overall_exec_p95_fill_delay_min"]))
        | (df["overall_exec_p95_fill_delay_min"] > df["th_symbol_max_p95_delay_base"])
    ).astype(int)

    df["fail_nan"] = (df["nan_pass"] != 1).astype(int)
    df["fail_data_quality"] = (df["data_quality_pass"] != 1).astype(int)
    df["fail_split"] = (df["split_pass"] != 1).astype(int)
    df["fail_constraint"] = (df["constraint_pass"] != 1).astype(int)

    gate_cols = [
        "fail_overall_entry_rate",
        "fail_overall_min_trades",
        "fail_symbol_entry_rate",
        "fail_symbol_min_trades",
        "fail_symbol_taker_share",
        "fail_symbol_median_fill_delay",
        "fail_symbol_p95_fill_delay",
        "fail_nan",
        "fail_data_quality",
        "fail_split",
        "fail_constraint",
    ]
    slack_cols = [
        "slack_overall_entry_rate",
        "slack_overall_min_trades",
        "slack_symbol_entry_rate",
        "slack_symbol_min_trades",
        "slack_symbol_taker_share",
        "slack_symbol_median_fill_delay",
        "slack_symbol_p95_fill_delay",
        "slack_overall_taker_share",
        "slack_overall_median_fill_delay",
        "slack_overall_p95_fill_delay",
    ]
    for c in gate_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return gate_cols, slack_cols


def _gate_breakdown(
    df: pd.DataFrame,
    gate_cols: List[str],
    reason_tokens: List[str],
) -> pd.DataFrame:
    n = max(1, len(df))
    # For earliest-binding estimate.
    fail_count = df[gate_cols].sum(axis=1)

    threshold_expr = {
        "fail_overall_entry_rate": "overall_entry_rate >= th_overall_entry_rate_base",
        "fail_overall_min_trades": "overall_entries_valid >= th_overall_min_trades_base",
        "fail_symbol_entry_rate": "overall_entry_rate >= th_symbol_entry_rate_base",
        "fail_symbol_min_trades": "overall_entries_valid >= th_symbol_min_trades_base",
        "fail_symbol_taker_share": "overall_exec_taker_share <= th_symbol_max_taker_share_row",
        "fail_symbol_median_fill_delay": "overall_exec_median_fill_delay_min <= th_symbol_max_median_delay_base",
        "fail_symbol_p95_fill_delay": "overall_exec_p95_fill_delay_min <= th_symbol_max_p95_delay_base",
        "fail_nan": "nan_pass == 1",
        "fail_data_quality": "data_quality_pass == 1",
        "fail_split": "split_pass == 1",
        "fail_constraint": "constraint_pass == 1",
    }
    metric_map = {
        "fail_overall_entry_rate": "slack_overall_entry_rate",
        "fail_overall_min_trades": "slack_overall_min_trades",
        "fail_symbol_entry_rate": "slack_symbol_entry_rate",
        "fail_symbol_min_trades": "slack_symbol_min_trades",
        "fail_symbol_taker_share": "slack_symbol_taker_share",
        "fail_symbol_median_fill_delay": "slack_symbol_median_fill_delay",
        "fail_symbol_p95_fill_delay": "slack_symbol_p95_fill_delay",
    }

    rows: List[Dict[str, Any]] = []
    for g in gate_cols:
        failed = int(df[g].sum())
        only_fail = int(((df[g] == 1) & (fail_count == 1)).sum())
        near_fail = int(((df[g] == 1) & (fail_count <= 2)).sum())
        row: Dict[str, Any] = {
            "gate": g,
            "source": "computed_gate",
            "threshold_expr": threshold_expr.get(g, ""),
            "failed_count": failed,
            "failed_pct": float(failed / n),
            "pass_count": int(n - failed),
            "earliest_binding_only_fail_count": only_fail,
            "near_binding_fail_count_le2": near_fail,
        }
        if g in metric_map:
            s = pd.to_numeric(df[metric_map[g]], errors="coerce")
            row["slack_p05"] = float(s.quantile(0.05)) if s.notna().any() else float("nan")
            row["slack_p50"] = float(s.quantile(0.50)) if s.notna().any() else float("nan")
            row["slack_p95"] = float(s.quantile(0.95)) if s.notna().any() else float("nan")
            row["slack_mean_failed_only"] = float(s[df[g] == 1].mean()) if (df[g] == 1).any() else float("nan")
        else:
            row["slack_p05"] = float("nan")
            row["slack_p50"] = float("nan")
            row["slack_p95"] = float("nan")
            row["slack_mean_failed_only"] = float("nan")
        rows.append(row)

    # Token-derived failure incidence.
    for tok in reason_tokens:
        c = f"fail_reason__{_sanitize_token(tok)}"
        if c not in df.columns:
            continue
        failed = int(df[c].sum())
        rows.append(
            {
                "gate": tok,
                "source": "invalid_reason_token",
                "threshold_expr": "",
                "failed_count": failed,
                "failed_pct": float(failed / n),
                "pass_count": int(n - failed),
                "earliest_binding_only_fail_count": int(((df[c] == 1) & (fail_count == 1)).sum()),
                "near_binding_fail_count_le2": int(((df[c] == 1) & (fail_count <= 2)).sum()),
                "slack_p05": float("nan"),
                "slack_p50": float("nan"),
                "slack_p95": float("nan"),
                "slack_mean_failed_only": float("nan"),
            }
        )

    out = pd.DataFrame(rows).sort_values(["source", "failed_count", "gate"], ascending=[True, False, True]).reset_index(drop=True)
    return out


def _cofailure_matrix(df: pd.DataFrame, gate_cols: List[str]) -> pd.DataFrame:
    mat = pd.DataFrame(index=gate_cols, columns=gate_cols, dtype=int)
    for a in gate_cols:
        for b in gate_cols:
            mat.loc[a, b] = int(((df[a] == 1) & (df[b] == 1)).sum())
    mat.index.name = "gate_a"
    return mat


def _build_near_feasible_frontier(df: pd.DataFrame, gate_cols: List[str]) -> pd.DataFrame:
    x = df.copy()

    # Participation-focused normalized deficits.
    x["def_overall_entry"] = np.maximum(0.0, -x["slack_overall_entry_rate"] / np.maximum(1e-9, x["th_overall_entry_rate_base"]))
    x["def_overall_trades"] = np.maximum(0.0, -x["slack_overall_min_trades"] / np.maximum(1e-9, x["th_overall_min_trades_base"]))
    x["def_symbol_entry"] = np.maximum(0.0, -x["slack_symbol_entry_rate"] / np.maximum(1e-9, x["th_symbol_entry_rate_base"]))
    x["def_symbol_trades"] = np.maximum(0.0, -x["slack_symbol_min_trades"] / np.maximum(1e-9, x["th_symbol_min_trades_base"]))
    x["def_taker"] = np.maximum(0.0, -x["slack_symbol_taker_share"] / np.maximum(1e-9, x["th_symbol_max_taker_share_row"]))
    x["def_median_delay"] = np.maximum(0.0, -x["slack_symbol_median_fill_delay"] / np.maximum(1.0, x["th_symbol_max_median_delay_base"]))
    x["def_p95_delay"] = np.maximum(0.0, -x["slack_symbol_p95_fill_delay"] / np.maximum(1.0, x["th_symbol_max_p95_delay_base"]))
    x["def_nan"] = x["fail_nan"].astype(float)
    x["def_data_quality"] = x["fail_data_quality"].astype(float)
    x["def_split"] = x["fail_split"].astype(float)
    x["def_constraint"] = x["fail_constraint"].astype(float)
    x["frontier_norm_deficit_total"] = (
        x["def_overall_entry"]
        + x["def_overall_trades"]
        + x["def_symbol_entry"]
        + x["def_symbol_trades"]
        + x["def_taker"]
        + x["def_median_delay"]
        + x["def_p95_delay"]
        + x["def_nan"]
        + x["def_data_quality"]
        + x["def_split"]
        + x["def_constraint"]
    )

    x["fail_count_total"] = x[gate_cols].sum(axis=1).astype(int)
    x["fail_count_participation"] = x[
        ["fail_overall_entry_rate", "fail_overall_min_trades", "fail_symbol_entry_rate", "fail_symbol_min_trades"]
    ].sum(axis=1).astype(int)

    # Objective proxy for invalid candidate sorting.
    o1 = pd.to_numeric(x["overall_exec_expectancy_net"], errors="coerce").fillna(-1e9)
    o2 = pd.to_numeric(x["overall_cvar_improve_ratio"], errors="coerce").fillna(-1e9)
    o3 = pd.to_numeric(x["overall_maxdd_improve_ratio"], errors="coerce").fillna(-1e9)
    x["objective_proxy"] = o1 + 0.25 * (o2 + o3)

    for c in ["genome_hash"]:
        if c not in x.columns:
            x[c] = [f"row_{i:06d}" for i in range(len(x))]

    invalid = x[x["hard_invalid"] == 1].copy()
    near_12 = invalid[(invalid["fail_count_total"] >= 1) & (invalid["fail_count_total"] <= 2)].copy()
    near_12 = near_12.sort_values(["fail_count_total", "frontier_norm_deficit_total", "objective_proxy"], ascending=[True, True, False]).head(300)
    near_12["near_source"] = "fail_count_1_to_2"

    top_obj = invalid.sort_values(["objective_proxy", "frontier_norm_deficit_total"], ascending=[False, True]).head(80).copy()
    top_obj["near_source"] = "top_objective_invalid"

    top_frontier = invalid.sort_values(["frontier_norm_deficit_total", "fail_count_total", "objective_proxy"], ascending=[True, True, False]).head(120).copy()
    top_frontier["near_source"] = "closest_frontier_invalid"

    near = pd.concat([near_12, top_obj, top_frontier], axis=0, ignore_index=True)
    near["near_source"] = near.groupby("genome_hash")["near_source"].transform(lambda s: "|".join(sorted(set(s.astype(str)))))
    near = near.drop_duplicates(subset=["genome_hash"], keep="first").copy()

    def _failed_gate_list(r: pd.Series) -> str:
        fs = [g for g in gate_cols if int(r.get(g, 0)) == 1]
        return "|".join(fs)

    near["failed_gates"] = near.apply(_failed_gate_list, axis=1)
    keep_cols = [
        "genome_hash",
        "near_source",
        "hard_invalid",
        "valid_for_ranking",
        "fail_count_total",
        "fail_count_participation",
        "failed_gates",
        "frontier_norm_deficit_total",
        "overall_exec_expectancy_net",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "objective_proxy",
        "overall_entry_rate",
        "overall_entries_valid",
        "overall_exec_taker_share",
        "overall_exec_median_fill_delay_min",
        "overall_exec_p95_fill_delay_min",
        "th_overall_entry_rate_base",
        "th_overall_min_trades_base",
        "th_symbol_entry_rate_base",
        "th_symbol_min_trades_base",
        "th_symbol_max_taker_share_row",
        "th_symbol_max_median_delay_base",
        "th_symbol_max_p95_delay_base",
        "slack_overall_entry_rate",
        "slack_overall_min_trades",
        "slack_symbol_entry_rate",
        "slack_symbol_min_trades",
        "slack_symbol_taker_share",
        "slack_symbol_median_fill_delay",
        "slack_symbol_p95_fill_delay",
        "fail_overall_entry_rate",
        "fail_overall_min_trades",
        "fail_symbol_entry_rate",
        "fail_symbol_min_trades",
        "fail_symbol_taker_share",
        "fail_symbol_median_fill_delay",
        "fail_symbol_p95_fill_delay",
        "fail_nan",
        "fail_data_quality",
        "fail_split",
        "fail_constraint",
        "invalid_reason",
    ]
    for c in keep_cols:
        if c not in near.columns:
            near[c] = np.nan
    near = near[keep_cols].sort_values(
        ["fail_count_total", "frontier_norm_deficit_total", "objective_proxy"],
        ascending=[True, True, False],
    )
    return near.reset_index(drop=True)


def _counterfactual_passcounts(df: pd.DataFrame, deltas: List[float]) -> pd.DataFrame:
    # Keep non-participation gates fixed exactly as observed.
    other_fixed_pass = (
        (df["fail_symbol_taker_share"] == 0)
        & (df["fail_symbol_median_fill_delay"] == 0)
        & (df["fail_symbol_p95_fill_delay"] == 0)
        & (df["fail_nan"] == 0)
        & (df["fail_data_quality"] == 0)
        & (df["fail_split"] == 0)
        & (df["fail_constraint"] == 0)
    )

    rows: List[Dict[str, Any]] = []
    for d_oe, d_ot, d_se, d_st in itertools.product(deltas, deltas, deltas, deltas):
        th_oe = df["th_overall_entry_rate_base"] * (1.0 + float(d_oe))
        th_ot = np.ceil(df["th_overall_min_trades_base"] * (1.0 + float(d_ot)))
        th_se = df["th_symbol_entry_rate_base"] * (1.0 + float(d_se))
        th_st = np.ceil(df["th_symbol_min_trades_base"] * (1.0 + float(d_st)))

        pass_participation = (
            np.isfinite(df["overall_entry_rate"])
            & np.isfinite(df["overall_entries_valid"])
            & (df["overall_entry_rate"] >= th_oe)
            & (df["overall_entries_valid"] >= th_ot)
            & (df["overall_entry_rate"] >= th_se)
            & (df["overall_entries_valid"] >= th_st)
        )
        pass_cf = pass_participation & other_fixed_pass

        rows.append(
            {
                "delta_overall_entry_floor": float(d_oe),
                "delta_overall_min_trades": float(d_ot),
                "delta_symbol_entry_floor": float(d_se),
                "delta_symbol_min_trades": float(d_st),
                "counterfactual_pass_count": int(pass_cf.sum()),
                "counterfactual_pass_pct": float(pass_cf.mean()),
                "counterfactual_participation_pass_count": int(pass_participation.sum()),
                "counterfactual_other_fixed_pass_count": int(other_fixed_pass.sum()),
                "counterfactual_new_valid_from_invalid_count": int(((pass_cf) & (df["hard_invalid"] == 1)).sum()),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        [
            "counterfactual_pass_count",
            "delta_overall_entry_floor",
            "delta_overall_min_trades",
            "delta_symbol_entry_floor",
            "delta_symbol_min_trades",
        ],
        ascending=[False, True, True, True, True],
    )
    return out.reset_index(drop=True)


def _parameter_failure_association(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    param_cols = [c for c in x.columns if c.startswith("g_")]
    if not param_cols:
        return pd.DataFrame()

    x["fail_participation_any"] = (
        (x["fail_overall_entry_rate"] == 1)
        | (x["fail_overall_min_trades"] == 1)
        | (x["fail_symbol_entry_rate"] == 1)
        | (x["fail_symbol_min_trades"] == 1)
    ).astype(int)

    er = pd.to_numeric(x["overall_entry_rate"], errors="coerce")
    tr = pd.to_numeric(x["overall_entries_valid"], errors="coerce")

    rows: List[Dict[str, Any]] = []
    for p in param_cols:
        s_raw = x[p]
        n = len(s_raw)
        nunique = int(s_raw.dropna().nunique())
        # Try numeric view.
        s_num = pd.to_numeric(s_raw, errors="coerce")
        numeric_share = float(s_num.notna().mean())
        is_numeric = numeric_share >= 0.80

        corr_er = _spearman_no_scipy(s_num, er) if is_numeric and s_num.notna().sum() >= 10 and er.notna().sum() >= 10 else float("nan")
        corr_tr = _spearman_no_scipy(s_num, tr) if is_numeric and s_num.notna().sum() >= 10 and tr.notna().sum() >= 10 else float("nan")

        fail_delta = float("nan")
        entry_delta = float("nan")
        trades_delta = float("nan")
        group_entry_range = float("nan")
        group_trades_range = float("nan")
        group_fail_range = float("nan")
        low_bucket = ""
        high_bucket = ""

        if is_numeric and nunique >= 6:
            q1 = float(s_num.quantile(0.25))
            q4 = float(s_num.quantile(0.75))
            g_lo = x[s_num <= q1]
            g_hi = x[s_num >= q4]
            if len(g_lo) and len(g_hi):
                fail_lo = float(g_lo["fail_participation_any"].mean())
                fail_hi = float(g_hi["fail_participation_any"].mean())
                entry_lo = float(pd.to_numeric(g_lo["overall_entry_rate"], errors="coerce").mean())
                entry_hi = float(pd.to_numeric(g_hi["overall_entry_rate"], errors="coerce").mean())
                trades_lo = float(pd.to_numeric(g_lo["overall_entries_valid"], errors="coerce").mean())
                trades_hi = float(pd.to_numeric(g_hi["overall_entries_valid"], errors="coerce").mean())
                fail_delta = float(fail_hi - fail_lo)
                entry_delta = float(entry_hi - entry_lo)
                trades_delta = float(trades_hi - trades_lo)
                low_bucket = f"<=q25({q1:.6g})"
                high_bucket = f">=q75({q4:.6g})"
        else:
            # Categorical or sparse numeric.
            grp = (
                x.assign(_p=s_raw.astype(str))
                .groupby("_p", dropna=False)
                .agg(
                    n=("fail_participation_any", "size"),
                    fail_rate=("fail_participation_any", "mean"),
                    entry_rate=("overall_entry_rate", "mean"),
                    trades=("overall_entries_valid", "mean"),
                )
                .sort_values("n", ascending=False)
            )
            if not grp.empty:
                group_entry_range = float(grp["entry_rate"].max() - grp["entry_rate"].min())
                group_trades_range = float(grp["trades"].max() - grp["trades"].min())
                group_fail_range = float(grp["fail_rate"].max() - grp["fail_rate"].min())
                lo = grp.sort_values("entry_rate", ascending=True).head(1)
                hi = grp.sort_values("entry_rate", ascending=False).head(1)
                if not lo.empty:
                    low_bucket = str(lo.index[0])
                if not hi.empty:
                    high_bucket = str(hi.index[0])

        assoc_strength = float(
            np.nansum(
                [
                    abs(corr_er) if np.isfinite(corr_er) else 0.0,
                    abs(corr_tr) if np.isfinite(corr_tr) else 0.0,
                    abs(fail_delta) if np.isfinite(fail_delta) else 0.0,
                    abs(entry_delta) / max(1e-9, float(np.nanstd(er))) if np.isfinite(entry_delta) and np.nanstd(er) > 0 else 0.0,
                    abs(trades_delta) / max(1e-9, float(np.nanstd(tr))) if np.isfinite(trades_delta) and np.nanstd(tr) > 0 else 0.0,
                    abs(group_fail_range) if np.isfinite(group_fail_range) else 0.0,
                ]
            )
        )
        dead_knob = int((nunique <= 1) or (assoc_strength < 0.05))
        collapse_flag = int(
            (
                (np.isfinite(corr_er) and corr_er < -0.15)
                or (np.isfinite(corr_tr) and corr_tr < -0.15)
                or (np.isfinite(fail_delta) and fail_delta > 0.15)
                or (np.isfinite(group_fail_range) and group_fail_range > 0.20)
            )
        )

        rows.append(
            {
                "parameter": p,
                "n_rows": int(n),
                "unique_count": int(nunique),
                "numeric_share": float(numeric_share),
                "is_numeric": int(is_numeric),
                "spearman_corr_entry_rate": float(corr_er),
                "spearman_corr_entries_valid": float(corr_tr),
                "fail_rate_delta_high_minus_low": float(fail_delta),
                "entry_rate_delta_high_minus_low": float(entry_delta),
                "entries_valid_delta_high_minus_low": float(trades_delta),
                "group_entry_rate_range": float(group_entry_range),
                "group_entries_valid_range": float(group_trades_range),
                "group_fail_rate_range": float(group_fail_range),
                "low_bucket": str(low_bucket),
                "high_bucket": str(high_bucket),
                "association_strength": float(assoc_strength),
                "collapses_participation_flag": int(collapse_flag),
                "dead_knob_flag": int(dead_knob),
            }
        )

    out = pd.DataFrame(rows).sort_values(
        ["collapses_participation_flag", "dead_knob_flag", "association_strength", "parameter"],
        ascending=[False, True, False, True],
    )
    return out.reset_index(drop=True)


def _choose_root_cause_class(
    df: pd.DataFrame,
    gate_breakdown: pd.DataFrame,
    cf: pd.DataFrame,
    assoc: pd.DataFrame,
) -> Tuple[str, str]:
    n = max(1, len(df))
    # Participation dominance evidence.
    g = gate_breakdown[gate_breakdown["source"] == "computed_gate"].copy()
    f_over_entry = float(g.loc[g["gate"] == "fail_overall_entry_rate", "failed_pct"].iloc[0]) if (g["gate"] == "fail_overall_entry_rate").any() else 0.0
    f_over_trades = float(g.loc[g["gate"] == "fail_overall_min_trades", "failed_pct"].iloc[0]) if (g["gate"] == "fail_overall_min_trades").any() else 0.0
    participation_dominant = (f_over_entry >= 0.90) and (f_over_trades >= 0.90)

    # Counterfactual feasibility under ±20%.
    max_cf_pass = int(cf["counterfactual_pass_count"].max()) if not cf.empty else 0
    has_reasonable_shift_feasible = max_cf_pass > 0

    # Sampler/search-space issue evidence.
    med_entry = float(pd.to_numeric(df["overall_entry_rate"], errors="coerce").median())
    med_trades = float(pd.to_numeric(df["overall_entries_valid"], errors="coerce").median())
    collapsers = int(assoc["collapses_participation_flag"].sum()) if not assoc.empty else 0
    sampler_issue = (med_entry < 0.20) and (med_trades < 60.0) and (collapsers >= 3)

    if (not has_reasonable_shift_feasible) and sampler_issue:
        return (
            "D",
            "No valid candidates appear even under ±20% participation-gate counterfactuals while sampled behavior remains overwhelmingly low-participation.",
        )
    if participation_dominant and sampler_issue:
        return (
            "C",
            "Both gate strictness and sampler/space geometry contribute: participation gates dominate failures and sampled regions are heavily low-participation.",
        )
    if participation_dominant:
        return (
            "A",
            "Participation constraints dominate and counterfactual shifts indicate gate strictness is the primary bottleneck.",
        )
    if sampler_issue:
        return (
            "B",
            "Search-space/sampler behavior is mostly infeasible for participation even before other constraints bind.",
        )
    return ("D", "No credible evidence that feasibility is reachable under reasonable threshold perturbations.")


def _write_report(
    out_path: Path,
    run_dir: Path,
    n: int,
    th: Thresholds,
    gate_breakdown: pd.DataFrame,
    cofail: pd.DataFrame,
    near: pd.DataFrame,
    cf: pd.DataFrame,
    assoc: pd.DataFrame,
    root_class: str,
    root_reason: str,
) -> None:
    g = gate_breakdown[gate_breakdown["source"] == "computed_gate"].copy()
    g_top = g.sort_values("failed_count", ascending=False).head(8)

    co = cofail.copy()
    co_long = (
        co.reset_index()
        .melt(id_vars=["gate_a"], var_name="gate_b", value_name="cofail_count")
        .query("gate_a != gate_b")
        .sort_values("cofail_count", ascending=False)
        .head(12)
    )
    # Keep unique unordered pairs.
    seen: set[Tuple[str, str]] = set()
    rows_pair: List[Dict[str, Any]] = []
    for r in co_long.itertuples(index=False):
        a = str(r.gate_a)
        b = str(r.gate_b)
        k = tuple(sorted((a, b)))
        if k in seen:
            continue
        seen.add(k)
        rows_pair.append({"gate_a": a, "gate_b": b, "cofail_count": int(r.cofail_count)})
        if len(rows_pair) >= 8:
            break
    co_show = pd.DataFrame(rows_pair)

    near_show = near.head(12).copy()
    cf_best = cf.head(12).copy()
    cf_single = cf[
        (cf["delta_overall_entry_floor"] == 0.0)
        & (cf["delta_overall_min_trades"] == 0.0)
        & (cf["delta_symbol_entry_floor"] == 0.0)
        & (cf["delta_symbol_min_trades"] == 0.0)
    ].copy()
    cf_baseline_pass = int(cf_single["counterfactual_pass_count"].iloc[0]) if not cf_single.empty else 0
    cf_max_pass = int(cf["counterfactual_pass_count"].max()) if not cf.empty else 0

    min_shift_note = "none"
    cf_pos = cf[cf["counterfactual_pass_count"] > 0].copy()
    if not cf_pos.empty:
        cf_pos["loosen_magnitude_sum"] = (
            np.maximum(0.0, -cf_pos["delta_overall_entry_floor"])
            + np.maximum(0.0, -cf_pos["delta_overall_min_trades"])
            + np.maximum(0.0, -cf_pos["delta_symbol_entry_floor"])
            + np.maximum(0.0, -cf_pos["delta_symbol_min_trades"])
        )
        best_shift = cf_pos.sort_values(["loosen_magnitude_sum", "counterfactual_pass_count"], ascending=[True, False]).head(1)
        if not best_shift.empty:
            b = best_shift.iloc[0]
            min_shift_note = (
                f"overall_entry={b['delta_overall_entry_floor']:+.0%}, overall_trades={b['delta_overall_min_trades']:+.0%}, "
                f"symbol_entry={b['delta_symbol_entry_floor']:+.0%}, symbol_trades={b['delta_symbol_min_trades']:+.0%} -> "
                f"pass_count={int(b['counterfactual_pass_count'])}"
            )

    assoc_top = assoc.sort_values(["collapses_participation_flag", "association_strength"], ascending=[False, False]).head(12)
    dead_knobs = assoc[assoc["dead_knob_flag"] == 1]["parameter"].tolist()

    lines: List[str] = []
    lines.append("# Phase K Viability Forensics Report")
    lines.append("")
    lines.append(f"- Generated UTC: {_utc_now().isoformat()}")
    lines.append(f"- Analyzed run: `{run_dir}`")
    lines.append(f"- Candidates analyzed: {n}")
    lines.append("")
    lines.append("## Root Cause Class")
    lines.append("")
    lines.append(f"- Classification: **{root_class}**")
    lines.append(f"- Reason: {root_reason}")
    lines.append("")
    lines.append("## Gate Thresholds (Diagnostic Baseline)")
    lines.append("")
    lines.append(f"- symbol: {th.symbol} (mode={th.mode})")
    lines.append(f"- overall entry floor: {th.hard_min_entry_rate_overall:.4f}")
    lines.append(f"- symbol entry floor (from execution config): {th.symbol_min_entry_rate:.4f}")
    lines.append(f"- overall min trades floor: max({th.hard_min_trades_overall}, ceil({th.hard_min_trade_frac_overall:.2f} * signals))")
    lines.append(f"- symbol min trades floor: max({th.hard_min_trades_symbol}, ceil({th.hard_min_trade_frac_symbol:.2f} * signals))")
    lines.append(f"- symbol max taker share cap (before genome cap): {th.symbol_max_taker_share_cap:.4f}")
    lines.append(f"- symbol max median fill delay min: {th.symbol_max_fill_delay_min:.2f}")
    lines.append(f"- symbol max p95 fill delay min: {th.hard_max_p95_fill_delay_min:.2f}")
    lines.append("")
    lines.append("## Failure Incidence")
    lines.append("")
    lines.append("```csv")
    lines.append(g_top.to_csv(index=False).strip())
    lines.append("```")
    lines.append("")
    lines.append("## Co-Failure Highlights")
    lines.append("")
    if co_show.empty:
        lines.append("- none")
    else:
        lines.append("```csv")
        lines.append(co_show.to_csv(index=False).strip())
        lines.append("```")
    lines.append("")
    lines.append("## Earliest-Binding Gate Estimate")
    lines.append("")
    eb = g.sort_values(["earliest_binding_only_fail_count", "near_binding_fail_count_le2"], ascending=[False, False]).head(8)
    lines.append("```csv")
    lines.append(eb[["gate", "failed_count", "failed_pct", "earliest_binding_only_fail_count", "near_binding_fail_count_le2", "slack_p50"]].to_csv(index=False).strip())
    lines.append("```")
    lines.append("")
    lines.append("## Near-Feasible Frontier (Invalid Candidates)")
    lines.append("")
    lines.append(f"- near_feasible_rows: {len(near)}")
    lines.append("```csv")
    lines.append(
        near_show[
            [
                "genome_hash",
                "near_source",
                "fail_count_total",
                "failed_gates",
                "frontier_norm_deficit_total",
                "overall_exec_expectancy_net",
                "overall_cvar_improve_ratio",
                "overall_maxdd_improve_ratio",
                "slack_overall_entry_rate",
                "slack_overall_min_trades",
                "slack_symbol_taker_share",
            ]
        ].to_csv(index=False).strip()
    )
    lines.append("```")
    lines.append("")
    lines.append("## Participation-Gate Counterfactuals (DIAGNOSTIC ONLY)")
    lines.append("")
    lines.append(f"- baseline pass_count (all deltas 0): {cf_baseline_pass}")
    lines.append(f"- max pass_count over tested perturbations: {cf_max_pass}")
    lines.append(f"- minimum shift that yields >0 pass (if any): {min_shift_note}")
    lines.append("```csv")
    lines.append(cf_best.to_csv(index=False).strip())
    lines.append("```")
    lines.append("")
    lines.append("## Sampler/Search-Space Feasibility Check")
    lines.append("")
    lines.append(f"- parameters_analyzed: {len(assoc)}")
    lines.append(f"- participation-collapse knobs flagged: {int(assoc['collapses_participation_flag'].sum()) if not assoc.empty else 0}")
    lines.append(f"- dead knobs flagged: {len(dead_knobs)}")
    if dead_knobs:
        lines.append(f"- dead knob list (first 20): {', '.join(dead_knobs[:20])}")
    lines.append("```csv")
    lines.append(
        assoc_top[
            [
                "parameter",
                "unique_count",
                "spearman_corr_entry_rate",
                "spearman_corr_entries_valid",
                "fail_rate_delta_high_minus_low",
                "entry_rate_delta_high_minus_low",
                "entries_valid_delta_high_minus_low",
                "association_strength",
                "collapses_participation_flag",
                "dead_knob_flag",
            ]
        ].to_csv(index=False).strip()
    )
    lines.append("```")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- Final class: **{root_class}**")
    if root_class == "A":
        lines.append("- Recommendation: gate calibration study first, then rerun constrained GA.")
    elif root_class == "B":
        lines.append("- Recommendation: repair search-space/sampler geometry before any gate adjustment.")
    elif root_class == "C":
        lines.append("- Recommendation: do both in sequence: sampler repair first, then tightly-controlled gate calibration diagnostics.")
    else:
        lines.append("- Recommendation: treat current branch as likely dead unless major model/execution definition changes are introduced.")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    run_dir = _resolve(args.run_dir) if str(args.run_dir).strip() else _latest_run_dir("GA_EXEC_OPT")
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run dir: {run_dir}")

    genomes_fp = run_dir / "genomes.csv"
    if not genomes_fp.exists():
        raise FileNotFoundError(f"Missing genomes.csv in {run_dir}")

    out_root = _resolve(args.outdir)
    out_dir = out_root / f"PHASEK_EXEC_VIABILITY_FORENSICS_{_utc_tag()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(genomes_fp)
    n_total = len(df)
    if n_total == 0:
        raise RuntimeError("genomes.csv is empty")

    reason_tokens = _parse_invalid_reason_flags(df)
    th = _load_thresholds_for_run(run_dir)
    df = _compute_row_thresholds(df, th=th)
    gate_cols, _slack_cols = _compute_gate_flags_and_slacks(df)

    # Failure anatomy.
    gate_breakdown = _gate_breakdown(df, gate_cols=gate_cols, reason_tokens=reason_tokens)
    gate_breakdown.to_csv(out_dir / "gate_failure_breakdown.csv", index=False)

    cofail = _cofailure_matrix(df, gate_cols=gate_cols)
    cofail.to_csv(out_dir / "gate_cofailure_matrix.csv")

    near = _build_near_feasible_frontier(df, gate_cols=gate_cols)
    near.to_csv(out_dir / "near_feasible_frontier.csv", index=False)

    deltas = [-0.20, -0.10, 0.0, 0.10, 0.20]
    cf = _counterfactual_passcounts(df, deltas=deltas)
    cf.to_csv(out_dir / "gate_counterfactual_passcounts.csv", index=False)

    assoc = _parameter_failure_association(df)
    assoc.to_csv(out_dir / "parameter_failure_association.csv", index=False)

    root_class, root_reason = _choose_root_cause_class(df, gate_breakdown, cf, assoc)
    _write_report(
        out_path=out_dir / "phaseK_viability_forensics_report.md",
        run_dir=run_dir,
        n=n_total,
        th=th,
        gate_breakdown=gate_breakdown,
        cofail=cofail,
        near=near,
        cf=cf,
        assoc=assoc,
        root_class=root_class,
        root_reason=root_reason,
    )

    # Manifest.
    gen_status = _json(run_dir / "gen_status.json") if (run_dir / "gen_status.json").exists() else {}
    invalid_hist = _json(run_dir / "invalid_reason_histogram.json") if (run_dir / "invalid_reason_histogram.json").exists() else {}
    manifest = {
        "generated_utc": _utc_now().isoformat(),
        "phase": "K",
        "mode": "diagnostics_only",
        "analyzed_run_dir": str(run_dir.resolve()),
        "source_files": {
            "genomes_csv": str(genomes_fp.resolve()),
            "gen_status_json": str((run_dir / "gen_status.json").resolve()) if (run_dir / "gen_status.json").exists() else "",
            "invalid_reason_histogram_json": str((run_dir / "invalid_reason_histogram.json").resolve())
            if (run_dir / "invalid_reason_histogram.json").exists()
            else "",
        },
        "candidate_count": int(n_total),
        "gate_columns_considered": gate_cols,
        "invalid_reason_tokens_detected": reason_tokens,
        "thresholds_used": {
            "symbol": th.symbol,
            "mode": th.mode,
            "execution_config_path": th.execution_config_path,
            "hard_min_entry_rate_symbol": th.hard_min_entry_rate_symbol,
            "hard_min_entry_rate_overall": th.hard_min_entry_rate_overall,
            "hard_min_trades_symbol": th.hard_min_trades_symbol,
            "hard_min_trade_frac_symbol": th.hard_min_trade_frac_symbol,
            "hard_min_trades_overall": th.hard_min_trades_overall,
            "hard_min_trade_frac_overall": th.hard_min_trade_frac_overall,
            "hard_max_taker_share": th.hard_max_taker_share,
            "hard_max_median_fill_delay_min": th.hard_max_median_fill_delay_min,
            "hard_max_p95_fill_delay_min": th.hard_max_p95_fill_delay_min,
            "symbol_min_entry_rate": th.symbol_min_entry_rate,
            "symbol_max_fill_delay_min": th.symbol_max_fill_delay_min,
            "symbol_max_taker_share_cap": th.symbol_max_taker_share_cap,
        },
        "counterfactual_deltas_tested": deltas,
        "classification": {"root_cause_class": root_class, "reason": root_reason},
        "gen_status_snapshot": gen_status,
        "invalid_reason_histogram_snapshot": invalid_hist,
    }
    (out_dir / "phaseK_run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase K viability/gate-frontier forensics for latest GA_EXEC_OPT run (diagnostics only).")
    ap.add_argument("--run-dir", default="", help="Optional explicit GA_EXEC_OPT run dir to analyze.")
    ap.add_argument("--outdir", default="reports/execution_layer", help="Output root directory.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))


if __name__ == "__main__":
    main()
