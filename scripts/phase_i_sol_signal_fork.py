#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import backtest_exec_phasec_sol as phasec_bt  # noqa: E402
from scripts import phase_e2_sol_representative as e2  # noqa: E402
from scripts import phase_g_sol_pathology_rehab as g  # noqa: E402


EXPECTED_SELECTED_MODEL_SET_SHA256 = "4a8cb243e7f7e6425db6726302d6326bf727fe026baca77980af0532543c2fc4"
EXPECTED_REPRESENTATIVE_SUBSET_SHA256 = "fdc34c3dcab18e8f8577857d7f879f92af822fc24bf3e0ec90a346a2a4cc372d"
EXPECTED_CONTRACT_ID = "SOL_PHASEE2_CANONICAL_V1"


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    is_baseline: bool
    apply_trend_gate: bool
    trend_min: float
    apply_vol_gate: bool
    allowed_vol_buckets: Tuple[str, ...]
    cooldown_hours: int
    delay_bars: int


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_tag() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _resolve(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        x = float(v)
        if not np.isfinite(x):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _tail_mean(arr: np.ndarray, frac: float) -> float:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(frac) * x.size)))
    return float(np.mean(np.sort(x)[:k]))


def _hash_float_vector(arr: np.ndarray, decimals: int = 12) -> str:
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return hashlib.sha256(b"").hexdigest()
    fmt = f"{{:.{int(decimals)}f}}"
    parts: List[str] = []
    for v in x:
        if np.isfinite(v):
            parts.append(fmt.format(float(v)))
        else:
            parts.append("nan")
    blob = "|".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _compute_cvar5_trade_notional(arr: np.ndarray) -> Tuple[float, int, str]:
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    h = _hash_float_vector(x)
    if x.size == 0:
        return float("nan"), 0, h
    k = max(1, int(math.ceil(0.05 * x.size)))
    cvar = float(np.mean(np.sort(x)[:k]))
    return cvar, int(k), h


def _norm_cdf(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0))))


def _sharpe_moments(x: np.ndarray) -> Tuple[float, float, float, float]:
    r = np.asarray(x, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mu = float(np.mean(r))
    sd = float(np.std(r, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return float("nan"), float("nan"), float("nan"), float("nan")
    z = (r - mu) / sd
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    sr = float(mu / sd)
    return sr, skew, kurt, float(n)


def _psr(sr_hat: float, sr_star: float, skew: float, kurt: float, n: float) -> float:
    if not (np.isfinite(sr_hat) and np.isfinite(sr_star) and np.isfinite(skew) and np.isfinite(kurt) and np.isfinite(n) and n > 1):
        return float("nan")
    denom = float(1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat**2))
    if denom <= 1e-12:
        return float("nan")
    z = float((sr_hat - sr_star) * math.sqrt(max(1e-12, n - 1.0)) / math.sqrt(denom))
    return _norm_cdf(z)


def _geom_mean_return(step_ret: np.ndarray) -> float:
    x = np.asarray(step_ret, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if np.any(x <= -1.0):
        return -1.0
    return float(np.exp(np.mean(np.log1p(x))) - 1.0)


def _geom_mean_return_with_ruin_flag(step_ret: np.ndarray) -> Tuple[float, int]:
    x = np.asarray(step_ret, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), 0
    ruin = int(np.any(x <= -1.0))
    if ruin == 1:
        return float("nan"), 1
    return float(np.exp(np.mean(np.log1p(x))) - 1.0), 0


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "_(empty)_"
    x = df.head(max_rows).copy()
    cols = [str(c) for c in x.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in x.iterrows():
        vals: List[str] = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append("nan" if not np.isfinite(v) else f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _latest_dir(root: Path, prefix: str) -> Path:
    cands = sorted([p for p in root.glob(f"{prefix}_*") if p.is_dir()])
    if not cands:
        raise FileNotFoundError(f"No dirs found for prefix `{prefix}` under {root}")
    return cands[-1]


def _trend_bucket(v: float) -> str:
    if not np.isfinite(v):
        return "unknown"
    return "up" if float(v) >= 0.5 else "down"


def _prepare_rep_context(
    *,
    phaseg_dir: Path,
    symbol: str,
    expected_rep_hash: str,
    expected_selected_model_hash: str,
) -> Dict[str, Any]:
    g_manifest = json.loads((phaseg_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if str(g_manifest.get("symbol", "")).upper() != symbol:
        raise RuntimeError(f"Phase G symbol mismatch: {g_manifest.get('symbol')}")

    selected_model_hash = str(g_manifest.get("selected_model_set_sha256", ""))
    if selected_model_hash != expected_selected_model_hash:
        raise RuntimeError(f"selected_model_set_sha256 mismatch: {selected_model_hash}")

    e2_dir = _resolve(str(g_manifest.get("e2_dir")))
    if not e2_dir.exists():
        raise FileNotFoundError(f"Missing E2 dir: {e2_dir}")

    e2_manifest = json.loads((e2_dir / "run_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((e2_dir / "accounting_contract.json").read_text(encoding="utf-8"))
    if str(contract.get("contract_id", "")) != EXPECTED_CONTRACT_ID:
        raise RuntimeError(f"Contract id mismatch: {contract.get('contract_id')}")
    if str(contract.get("symbol", "")).upper() != symbol:
        raise RuntimeError(f"Contract symbol mismatch: {contract.get('symbol')}")

    fee_model_path = _resolve(str(contract["fee_model_path"]))
    metrics_def_path = _resolve(str(contract["metrics_definition_path"]))
    fee_hash = g._sha256_file(fee_model_path)
    metrics_hash = g._sha256_file(metrics_def_path)
    if fee_hash != g.EXPECTED_PHASEA_FEE_HASH:
        raise RuntimeError(f"Fee hash mismatch: {fee_hash}")
    if metrics_hash != g.EXPECTED_PHASEA_METRICS_HASH:
        raise RuntimeError(f"Metrics hash mismatch: {metrics_hash}")

    rep_subset_path = e2_dir / "representative_subset_signals.csv"
    rep_subset_hash_ref = (e2_dir / "representative_subset_hash.txt").read_text(encoding="utf-8").strip()
    rep_subset = pd.read_csv(rep_subset_path)
    rep_subset["signal_id"] = rep_subset["signal_id"].astype(str)
    rep_subset["signal_time"] = pd.to_datetime(rep_subset["signal_time"], utc=True, errors="coerce")
    rep_subset = rep_subset.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rep_subset_hash_calc = g._hash_rep_subset(rep_subset[["signal_id", "signal_time"]].copy())
    if rep_subset_hash_calc != rep_subset_hash_ref:
        raise RuntimeError("Representative subset hash mismatch vs hash file.")
    if rep_subset_hash_calc != str(e2_manifest.get("representative_subset_sha256", "")):
        raise RuntimeError("Representative subset hash mismatch vs E2 manifest.")
    if rep_subset_hash_calc != str(expected_rep_hash):
        raise RuntimeError(f"Representative subset hash mismatch vs expected: {rep_subset_hash_calc}")

    split_definition = list(e2_manifest.get("split_definition", []))
    if not split_definition:
        raise RuntimeError("Missing split definition in E2 manifest.")
    split_lookup = e2._split_lookup(rep_subset, split_definition)

    signal_source_path = _resolve(str(e2_manifest.get("signal_source_csv")))
    source_signals = pd.read_csv(signal_source_path)
    source_signals["signal_id"] = source_signals["signal_id"].astype(str)
    source_signals["signal_time"] = pd.to_datetime(source_signals["signal_time"], utc=True, errors="coerce")
    source_signals["cycle"] = pd.to_numeric(source_signals.get("cycle"), errors="coerce").fillna(0).astype(int)
    source_signals["atr_percentile_1h"] = pd.to_numeric(source_signals.get("atr_percentile_1h"), errors="coerce")
    source_signals["trend_up_1h"] = pd.to_numeric(source_signals.get("trend_up_1h"), errors="coerce")

    rep_feat = rep_subset.merge(
        source_signals[["signal_id", "cycle", "atr_percentile_1h", "trend_up_1h"]],
        on="signal_id",
        how="left",
        suffixes=("", "_src"),
    )
    rep_feat["atr_percentile_1h"] = pd.to_numeric(rep_feat["atr_percentile_1h"], errors="coerce").fillna(
        pd.to_numeric(rep_feat.get("atr_percentile_1h_src"), errors="coerce")
    )
    rep_feat["trend_up_1h"] = pd.to_numeric(rep_feat["trend_up_1h"], errors="coerce").fillna(
        pd.to_numeric(rep_feat.get("trend_up_1h_src"), errors="coerce")
    )
    rep_feat["vol_bucket"] = g._vol_bucket(rep_feat["atr_percentile_1h"])
    rep_feat["trend_bucket"] = rep_feat["trend_up_1h"].map(_trend_bucket)
    rep_feat["regime_bucket"] = rep_feat["vol_bucket"].astype(str) + "|" + rep_feat["trend_bucket"].astype(str)

    t_v3 = pd.read_csv(e2_dir / "trades_v3r_exec_3m_phasec_control.csv")
    t_v3["signal_id"] = t_v3["signal_id"].astype(str)
    for c in ["signal_time", "entry_time", "exit_time"]:
        if c in t_v3.columns:
            t_v3[c] = pd.to_datetime(t_v3[c], utc=True, errors="coerce")
    for c in ["signal_tp_mult", "signal_sl_mult", "entry_price", "exit_price", "pnl_net_pct", "risk_pct"]:
        if c in t_v3.columns:
            t_v3[c] = pd.to_numeric(t_v3[c], errors="coerce")

    entries_base = g._entry_table_from_v3(t_v3, split_lookup)
    entries_base = entries_base.merge(
        rep_feat[["signal_id", "signal_time", "cycle", "atr_percentile_1h", "trend_up_1h", "vol_bucket", "trend_bucket", "regime_bucket"]],
        on=["signal_id", "signal_time"],
        how="left",
    )

    return {
        "phaseg_manifest": g_manifest,
        "e2_dir": e2_dir,
        "e2_manifest": e2_manifest,
        "contract": contract,
        "fee_hash": fee_hash,
        "metrics_hash": metrics_hash,
        "rep_subset_hash": rep_subset_hash_calc,
        "selected_model_set_hash": selected_model_hash,
        "rep_subset": rep_subset,
        "rep_feat": rep_feat,
        "split_definition": split_definition,
        "entries_base": entries_base,
        "signal_source_path": signal_source_path,
    }


def _parse_float_list(raw: str) -> List[float]:
    out: List[float] = []
    for t in str(raw).split(","):
        s = str(t).strip()
        if not s:
            continue
        out.append(float(s))
    return out


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for t in str(raw).split(","):
        s = str(t).strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _parse_str_list(raw: str) -> List[str]:
    out: List[str] = []
    for t in str(raw).split(","):
        s = str(t).strip()
        if not s:
            continue
        out.append(s)
    return out


def _supported_vol_modes(rep_feat: pd.DataFrame, min_bucket_support: int) -> Dict[str, Tuple[str, ...]]:
    counts = rep_feat["vol_bucket"].astype(str).value_counts(dropna=False).to_dict()
    supported = sorted([b for b, n in counts.items() if b != "unknown" and int(n) >= int(min_bucket_support)])
    if not supported:
        supported = sorted([b for b, n in counts.items() if b != "unknown" and int(n) > 0])
    if not supported:
        supported = ("low", "mid", "high")
    sset: Set[str] = set(supported)

    modes: Dict[str, Tuple[str, ...]] = {"all_supported": tuple(sorted(sset))}
    if "mid" in sset:
        modes["mid_only"] = ("mid",)
    if {"low", "mid"}.issubset(sset):
        modes["low_mid"] = ("low", "mid")
    if {"mid", "high"}.issubset(sset):
        modes["mid_high"] = ("mid", "high")
    if "high" in sset:
        modes["high_only"] = ("high",)
    # deduplicate by tuple value while preserving first key.
    seen: Set[Tuple[str, ...]] = set()
    uniq: Dict[str, Tuple[str, ...]] = {}
    for k, v in modes.items():
        vv = tuple(sorted(v))
        if vv in seen:
            continue
        seen.add(vv)
        uniq[k] = vv
    return uniq


def _build_candidates(
    *,
    rep_feat: pd.DataFrame,
    trend_thresholds: Sequence[float],
    cooldown_hours: Sequence[int],
    delay_bars: Sequence[int],
    min_bucket_support: int,
    vol_mode_filter: Sequence[str],
) -> List[CandidateConfig]:
    cands: List[CandidateConfig] = [
        CandidateConfig(
            name="baseline_contract_locked",
            is_baseline=True,
            apply_trend_gate=False,
            trend_min=0.5,
            apply_vol_gate=False,
            allowed_vol_buckets=tuple(),
            cooldown_hours=0,
            delay_bars=0,
        )
    ]

    vol_modes = _supported_vol_modes(rep_feat, min_bucket_support=min_bucket_support)
    filters = [str(v).strip() for v in vol_mode_filter if str(v).strip()]
    if filters:
        missing = [k for k in filters if k not in vol_modes]
        if missing:
            raise RuntimeError(f"Unknown vol mode filter(s): {missing}; available={sorted(vol_modes.keys())}")
        vol_modes = {k: vol_modes[k] for k in filters}
    for tmin in trend_thresholds:
        for mode_name, buckets in vol_modes.items():
            for cd in cooldown_hours:
                for d in delay_bars:
                    nm = f"fork_trend{tmin:.2f}_{mode_name}_cd{int(cd)}h_d{int(d)}"
                    cands.append(
                        CandidateConfig(
                            name=nm,
                            is_baseline=False,
                            apply_trend_gate=True,
                            trend_min=float(tmin),
                            apply_vol_gate=True,
                            allowed_vol_buckets=tuple(sorted(str(x) for x in buckets)),
                            cooldown_hours=int(cd),
                            delay_bars=int(d),
                        )
                    )
    return cands


def _apply_candidate_with_trace(entries_base: pd.DataFrame, rep_feat: pd.DataFrame, cfg: CandidateConfig, symbol: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    trace: Dict[str, int] = {}
    x = entries_base[
        [
            "signal_id",
            "signal_time",
            "split_id",
            "signal_tp_mult",
            "signal_sl_mult",
            "entry_time",
            "entry_price",
            "cycle",
            "atr_percentile_1h",
            "trend_up_1h",
            "vol_bucket",
            "trend_bucket",
            "regime_bucket",
        ]
    ].copy()
    trace["signals_before_all"] = int(len(x))

    if cfg.apply_trend_gate:
        trace["signals_before_trend_gate"] = int(len(x))
        tr = pd.to_numeric(x["trend_up_1h"], errors="coerce")
        x = x[tr >= float(cfg.trend_min)].copy()
        trace["signals_after_trend_gate"] = int(len(x))
        trace["removed_by_trend_gate"] = int(trace["signals_before_trend_gate"] - trace["signals_after_trend_gate"])
    else:
        trace["signals_before_trend_gate"] = int(len(x))
        trace["signals_after_trend_gate"] = int(len(x))
        trace["removed_by_trend_gate"] = 0

    if cfg.apply_vol_gate:
        trace["signals_before_vol_gate"] = int(len(x))
        x = x[x["vol_bucket"].astype(str).isin(set(cfg.allowed_vol_buckets))].copy()
        trace["signals_after_vol_gate"] = int(len(x))
        trace["removed_by_vol_gate"] = int(trace["signals_before_vol_gate"] - trace["signals_after_vol_gate"])
    else:
        trace["signals_before_vol_gate"] = int(len(x))
        trace["signals_after_vol_gate"] = int(len(x))
        trace["removed_by_vol_gate"] = 0

    x = x.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    trace["signals_before_cooldown"] = int(len(x))
    if cfg.cooldown_hours > 0:
        x = g._apply_cooldown(x, cooldown_hours=int(cfg.cooldown_hours))
    trace["signals_after_cooldown"] = int(len(x))
    trace["removed_by_cooldown"] = int(trace["signals_before_cooldown"] - trace["signals_after_cooldown"])

    trace["signals_before_delay"] = int(len(x))
    before_delay_ids = set(x["signal_id"].astype(str).tolist())
    if cfg.delay_bars in (0, 1, 2):
        x = g._delay_entries_using_1h_open(
            x[["signal_id", "signal_time", "split_id", "signal_tp_mult", "signal_sl_mult", "entry_time", "entry_price"]].copy(),
            symbol=symbol,
            offset_hours_after_signal=int(1 + cfg.delay_bars),
        )
        x = x.merge(
            rep_feat[["signal_id", "signal_time", "cycle", "atr_percentile_1h", "trend_up_1h", "vol_bucket", "trend_bucket", "regime_bucket"]],
            on=["signal_id", "signal_time"],
            how="left",
        )
    trace["signals_after_delay"] = int(len(x))
    after_delay_ids = set(x["signal_id"].astype(str).tolist())
    trace["removed_by_delay_or_rebuild"] = int(len(before_delay_ids - after_delay_ids))

    trace["signals_before_dropna"] = int(len(x))
    x = x.dropna(subset=["signal_time", "entry_time", "entry_price", "signal_tp_mult", "signal_sl_mult"]).copy()
    x = x.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    trace["signals_after_dropna"] = int(len(x))
    trace["removed_by_dropna"] = int(trace["signals_before_dropna"] - trace["signals_after_dropna"])
    trace["signals_after_all"] = int(len(x))
    return x, trace


def _valid_trades(trades: pd.DataFrame) -> pd.DataFrame:
    x = trades.copy()
    for c in ["filled", "valid_for_metrics", "pnl_net_pct", "risk_pct"]:
        x[c] = pd.to_numeric(x.get(c, np.nan), errors="coerce")
    for c in ["signal_time", "entry_time", "exit_time"]:
        x[c] = pd.to_datetime(x.get(c), utc=True, errors="coerce")
    v = x[(x["filled"] == 1) & (x["valid_for_metrics"] == 1) & x["pnl_net_pct"].notna()].copy()
    return v.sort_values(["entry_time", "signal_time", "signal_id"]).reset_index(drop=True)


def _support_stats(valid: pd.DataFrame, rep_subset: pd.DataFrame, split_definition: Sequence[Dict[str, int]], min_split_trades_req: int) -> Tuple[int, int, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    min_split_trades = int(10**9)
    valid_ids = valid["signal_id"].astype(str)
    ss = rep_subset.reset_index(drop=True)

    for sp in split_definition:
        sid = int(sp["split_id"])
        lo, hi = int(sp["test_start"]), int(sp["test_end"])
        seg = ss.iloc[lo:hi].copy()
        ids = set(seg["signal_id"].astype(str).tolist())
        n_trades = int(valid_ids.isin(ids).sum())
        min_split_trades = min(min_split_trades, n_trades)
        rows.append({"split_id": sid, "signals_total": int(len(seg)), "trades_total": int(n_trades)})

    if min_split_trades == int(10**9):
        min_split_trades = 0
    support_ok = int(min_split_trades >= int(min_split_trades_req))
    return support_ok, int(min_split_trades), rows


def _dominant_worst_regime_bucket(valid: pd.DataFrame) -> str:
    if valid.empty:
        return "none"
    x = valid.copy()
    x["pnl_net_pct"] = pd.to_numeric(x["pnl_net_pct"], errors="coerce")
    gsum = x.groupby("regime_bucket", dropna=False)["pnl_net_pct"].sum().reset_index()
    if gsum.empty:
        return "unknown"
    gsum = gsum.sort_values("pnl_net_pct", ascending=True).reset_index(drop=True)
    return str(gsum.iloc[0]["regime_bucket"])


def _adverse_loss_share(valid: pd.DataFrame) -> float:
    if valid.empty:
        return float("nan")
    x = valid[pd.to_numeric(valid["pnl_net_pct"], errors="coerce") < 0].copy()
    if x.empty:
        return 0.0
    t = pd.to_numeric(x["trend_up_1h"], errors="coerce")
    v = pd.to_numeric(x["atr_percentile_1h"], errors="coerce")
    return float(((t < 0.5) | (v >= 66.6666666667)).mean())


def _fixed_equity_step_returns(eq_fix: pd.DataFrame, initial_equity: float) -> np.ndarray:
    x = eq_fix.copy()
    x["equity_fixed"] = pd.to_numeric(x.get("equity_fixed"), errors="coerce")
    x = x[x["equity_fixed"].notna()].copy().reset_index(drop=True)
    if x.empty:
        return np.asarray([], dtype=float)
    prev = np.concatenate(([float(initial_equity)], x["equity_fixed"].to_numpy(dtype=float)[:-1]))
    cur = x["equity_fixed"].to_numpy(dtype=float)
    step = cur / np.maximum(1e-12, prev) - 1.0
    return step.astype(float)


def _per_signal_fixed_step(eq_fix: pd.DataFrame, rep_signal_ids: Sequence[str], initial_equity: float) -> np.ndarray:
    ids = [str(x) for x in rep_signal_ids]
    x = eq_fix.copy()
    x["signal_id"] = x.get("signal_id", "").astype(str)
    x["trade_pnl_abs_fixed"] = pd.to_numeric(x.get("trade_pnl_abs_fixed"), errors="coerce")
    if x.empty:
        return np.zeros(len(ids), dtype=float)
    agg = x.groupby("signal_id", dropna=False)["trade_pnl_abs_fixed"].sum()
    out = np.zeros(len(ids), dtype=float)
    for i, sid in enumerate(ids):
        v = _safe_float(agg.get(sid, 0.0), default=0.0)
        out[i] = float(v / max(1e-12, float(initial_equity)))
    return out


def _estimate_effective_trials(signal_step_mat: np.ndarray) -> Tuple[float, float]:
    # Returns: n_eff, avg_pairwise_corr.
    m = np.asarray(signal_step_mat, dtype=float)
    if m.ndim != 2:
        return float("nan"), float("nan")
    n_trials = int(m.shape[1])
    if n_trials <= 1:
        return 1.0, float("nan")
    corr = np.corrcoef(m, rowvar=False)
    if corr.shape[0] != n_trials:
        return float(n_trials), float("nan")
    vals: List[float] = []
    for i in range(n_trials):
        for j in range(i + 1, n_trials):
            c = _safe_float(corr[i, j])
            if np.isfinite(c):
                vals.append(c)
    if not vals:
        return float(n_trials), float("nan")
    rbar = float(np.mean(vals))
    neff = 1.0 + (float(n_trials) - 1.0) * (1.0 - rbar)
    neff = float(max(1.0, min(float(n_trials), neff)))
    return neff, rbar


def run(args: argparse.Namespace) -> Path:
    symbol = str(args.symbol).strip().upper()
    if symbol != "SOLUSDT":
        raise RuntimeError("Phase I fork runner is scoped to SOLUSDT.")

    out_root = _resolve(args.outdir)
    phaseh_dir = _resolve(args.phaseh_dir) if args.phaseh_dir else _latest_dir(out_root, "PHASEH_SOL_FREEZE_FORK")
    if not phaseh_dir.exists():
        raise FileNotFoundError(f"Missing Phase H dir: {phaseh_dir}")
    phaseh_manifest_path = phaseh_dir / "phaseH_run_manifest.json"
    if not phaseh_manifest_path.exists():
        raise FileNotFoundError(f"Missing Phase H manifest: {phaseh_manifest_path}")
    phaseh_manifest = json.loads(phaseh_manifest_path.read_text(encoding="utf-8"))

    phaseg_dir = _resolve(str(phaseh_manifest.get("phaseg_source_dir", "")))
    if args.phaseg_dir:
        phaseg_dir = _resolve(args.phaseg_dir)

    rep_hash_expected = str(args.expected_rep_subset_hash or phaseh_manifest.get("representative_subset_sha256", EXPECTED_REPRESENTATIVE_SUBSET_SHA256))
    model_hash_expected = str(args.expected_selected_model_set_hash or phaseh_manifest.get("selected_model_set_sha256", EXPECTED_SELECTED_MODEL_SET_SHA256))

    ctx = _prepare_rep_context(
        phaseg_dir=phaseg_dir,
        symbol=symbol,
        expected_rep_hash=rep_hash_expected,
        expected_selected_model_hash=model_hash_expected,
    )

    run_dir = out_root / f"PHASEI_SOL_SIGNAL_FORK_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snap = run_dir / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    # Snapshot core files for reproducibility.
    for fp in [
        ctx["e2_dir"] / "run_manifest.json",
        ctx["e2_dir"] / "accounting_contract.json",
        ctx["e2_dir"] / "pass_fail_gates.json",
        ctx["e2_dir"] / "representative_subset_signals.csv",
        ctx["e2_dir"] / "trades_v2r_1h_reference_control.csv",
        ctx["e2_dir"] / "trades_v3r_exec_3m_phasec_control.csv",
        ctx["e2_dir"] / "trades_v4r_exec_3m_phasec_best.csv",
        _resolve(str(ctx["contract"]["fee_model_path"])),
        _resolve(str(ctx["contract"]["metrics_definition_path"])),
        _resolve(str(ctx["signal_source_path"])),
        phaseg_dir / "phaseG2_ablation_results.csv",
        phaseg_dir / "phaseG3_practical_gate_decisions.csv",
    ]:
        if Path(fp).exists():
            shutil.copy2(fp, snap / Path(fp).name)

    fee = phasec_bt._load_fee_model(_resolve(str(ctx["contract"]["fee_model_path"])))
    initial_equity = float(ctx["contract"].get("initial_equity", args.initial_equity))
    risk_per_trade = float(ctx["contract"].get("risk_per_trade", args.risk_per_trade))

    rep_subset = ctx["rep_subset"].copy()
    rep_feat = ctx["rep_feat"].copy()
    split_definition = ctx["split_definition"]
    entries_base = ctx["entries_base"].copy()

    trend_thresholds = _parse_float_list(args.trend_thresholds)
    cooldown_hours = _parse_int_list(args.cooldown_hours)
    delay_bars = _parse_int_list(args.delay_bars)
    vol_mode_filter = _parse_str_list(args.vol_mode_filter)
    if not delay_bars:
        delay_bars = [0, 1, 2]
    delay_bars = [int(x) for x in delay_bars if int(x) in (0, 1, 2)]
    if not delay_bars:
        delay_bars = [0, 1, 2]

    cands = _build_candidates(
        rep_feat=rep_feat,
        trend_thresholds=trend_thresholds,
        cooldown_hours=cooldown_hours,
        delay_bars=delay_bars,
        min_bucket_support=int(args.min_bucket_support),
        vol_mode_filter=vol_mode_filter,
    )

    rep_signal_ids = rep_subset["signal_id"].astype(str).tolist()

    rows: List[Dict[str, Any]] = []
    candidate_trades: Dict[str, pd.DataFrame] = {}
    candidate_signals_total: Dict[str, int] = {}
    candidate_fixed_step_vec: Dict[str, np.ndarray] = {}
    split_rows_all: List[Dict[str, Any]] = []

    for cfg in cands:
        ent, stage_trace = _apply_candidate_with_trace(entries_base, rep_feat, cfg, symbol=symbol)
        trades = e2._simulate_1h_from_entries(
            entries_df=ent[["signal_id", "signal_time", "split_id", "signal_tp_mult", "signal_sl_mult", "entry_time", "entry_price"]].copy(),
            symbol=symbol,
            fee=fee,
            exec_horizon_hours=float(args.exec_horizon_hours),
        )
        trades = trades.merge(
            rep_feat[["signal_id", "signal_time", "cycle", "atr_percentile_1h", "trend_up_1h", "vol_bucket", "trend_bucket", "regime_bucket"]],
            on=["signal_id", "signal_time"],
            how="left",
        )
        valid = _valid_trades(trades)
        eq_fix, m_fix = g._compute_fixed_size_equity_curve(
            trades,
            signals_total=int(len(ent)),
            initial_equity=initial_equity,
            risk_per_trade=risk_per_trade,
        )
        step_ret = _fixed_equity_step_returns(eq_fix, initial_equity=initial_equity)
        geom_step_legacy = _geom_mean_return(step_ret)
        geom_step_clean, ruin_event_fixed = _geom_mean_return_with_ruin_flag(step_ret)

        ret = pd.to_numeric(valid["pnl_net_pct"], errors="coerce").dropna().to_numpy(dtype=float)
        expectancy_trade = float(np.mean(ret)) if ret.size else float("nan")
        cvar5_trade, cvar_tail_n, trade_vec_hash = _compute_cvar5_trade_notional(ret)
        pos = ret[ret > 0]
        neg = ret[ret < 0]
        pf_trade = float(np.sum(pos) / abs(np.sum(neg))) if neg.size and abs(np.sum(neg)) > 1e-12 else (float("inf") if pos.size else float("nan"))
        win_rate_trade = float((ret > 0).sum() / max(1, ret.size))

        support_ok, min_split_trades, split_rows = _support_stats(
            valid,
            rep_subset,
            split_definition,
            min_split_trades_req=int(args.min_split_trades),
        )
        for sr in split_rows:
            split_rows_all.append(
                {
                    "variant": cfg.name,
                    "split_id": int(sr["split_id"]),
                    "signals_total": int(sr["signals_total"]),
                    "trades_total": int(sr["trades_total"]),
                }
            )

        variant_adverse = _adverse_loss_share(valid)
        variant_worst_regime = _dominant_worst_regime_bucket(valid)

        total_return_fixed = _safe_float(m_fix.get("total_return_fixed"))
        maxdd_fixed = _safe_float(m_fix.get("max_drawdown_pct_fixed"))
        fatal_gate_fixed = int(
            (np.isfinite(total_return_fixed) and total_return_fixed <= float(args.fatal_total_return))
            or (np.isfinite(maxdd_fixed) and maxdd_fixed <= float(args.fatal_max_dd))
        )

        gate_expectancy = int(np.isfinite(expectancy_trade) and expectancy_trade > 0.0)
        gate_total_return_fixed = int(np.isfinite(total_return_fixed) and total_return_fixed > 0.0)
        gate_maxdd_fixed = int(np.isfinite(maxdd_fixed) and maxdd_fixed > float(args.deploy_max_dd_floor))
        gate_cvar5 = int(np.isfinite(cvar5_trade) and cvar5_trade > float(args.deploy_cvar5_floor))
        gate_pf = int(np.isfinite(pf_trade) and pf_trade >= float(args.deploy_pf_floor))
        gate_support = int(support_ok == 1)

        fixed_absolute_pass = int(
            gate_expectancy == 1
            and gate_total_return_fixed == 1
            and gate_maxdd_fixed == 1
            and gate_cvar5 == 1
            and gate_pf == 1
            and gate_support == 1
        )

        rows.append(
            {
                "variant": cfg.name,
                "is_baseline": int(cfg.is_baseline),
                "trend_alignment_gate_on": int(cfg.apply_trend_gate),
                "trend_min_threshold": float(cfg.trend_min),
                "trend_threshold_effective_on_signal_count": int(stage_trace.get("removed_by_trend_gate", 0) > 0),
                "volatility_gate_on": int(cfg.apply_vol_gate),
                "allowed_vol_buckets": ",".join(cfg.allowed_vol_buckets),
                "cooldown_hours": int(cfg.cooldown_hours),
                "delay_1h_bars": int(cfg.delay_bars),
                "signals_total": int(len(ent)),
                "trades_total": int(len(valid)),
                "signals_before_all": int(stage_trace.get("signals_before_all", 0)),
                "signals_after_trend_gate": int(stage_trace.get("signals_after_trend_gate", 0)),
                "signals_after_vol_gate": int(stage_trace.get("signals_after_vol_gate", 0)),
                "signals_after_cooldown": int(stage_trace.get("signals_after_cooldown", 0)),
                "signals_after_delay": int(stage_trace.get("signals_after_delay", 0)),
                "signals_after_dropna": int(stage_trace.get("signals_after_dropna", 0)),
                "removed_by_trend_gate": int(stage_trace.get("removed_by_trend_gate", 0)),
                "removed_by_vol_gate": int(stage_trace.get("removed_by_vol_gate", 0)),
                "removed_by_cooldown": int(stage_trace.get("removed_by_cooldown", 0)),
                "removed_by_delay_or_rebuild": int(stage_trace.get("removed_by_delay_or_rebuild", 0)),
                "removed_by_dropna": int(stage_trace.get("removed_by_dropna", 0)),
                # Trade-level notional metrics (unit: decimal return per trade on position notional).
                "expectancy_net_trade_notional_dec": float(expectancy_trade),
                "cvar_5_trade_notional_dec": float(cvar5_trade),
                "profit_factor_trade": float(pf_trade),
                "win_rate_trade": float(win_rate_trade),
                "trade_return_vector_sha256": str(trade_vec_hash),
                "trade_return_count": int(ret.size),
                "cvar_tail_count": int(cvar_tail_n),
                # Fixed-size equity-path metrics (unit: decimal return on account equity path).
                "total_return_fixed_equity_dec": float(total_return_fixed),
                "max_drawdown_pct_fixed_equity_dec": float(maxdd_fixed),
                "geometric_equity_step_return_fixed": float(geom_step_legacy),
                "geometric_equity_step_return_fixed_clean": float(geom_step_clean),
                "ruin_event_fixed": int(ruin_event_fixed),
                "min_split_trades": int(min_split_trades),
                "support_ok": int(support_ok),
                "fatal_gate_fixed": int(fatal_gate_fixed),
                "fixed_absolute_practical_pass": int(fixed_absolute_pass),
                # Diagnostics taxonomy-safe fields (variant-local only).
                "variant_adverse_loss_share": float(variant_adverse),
                "dominant_worst_regime_bucket": str(variant_worst_regime),
                # Gate bits.
                "gate_expectancy_trade_gt_0": int(gate_expectancy),
                "gate_total_return_fixed_gt_0": int(gate_total_return_fixed),
                "gate_maxdd_fixed_gt_floor": int(gate_maxdd_fixed),
                "gate_cvar5_trade_gt_floor": int(gate_cvar5),
                "gate_pf_trade_ge_floor": int(gate_pf),
                "gate_support_ok": int(gate_support),
            }
        )

        candidate_trades[cfg.name] = trades
        candidate_signals_total[cfg.name] = int(len(ent))
        candidate_fixed_step_vec[cfg.name] = _per_signal_fixed_step(eq_fix, rep_signal_ids=rep_signal_ids, initial_equity=initial_equity)

    if not rows:
        raise RuntimeError("No candidates were evaluated in Phase I.")

    results = pd.DataFrame(rows)
    baseline_row = results[results["is_baseline"] == 1]
    if baseline_row.empty:
        raise RuntimeError("Missing baseline candidate in Phase I results.")
    baseline_adverse = _safe_float(baseline_row["variant_adverse_loss_share"].iloc[0])
    baseline_worst_regime = str(baseline_row["dominant_worst_regime_bucket"].iloc[0])

    # Ranking among fixed-size survivors only.
    results["fixed_rank"] = np.nan
    survivors = results[results["fixed_absolute_practical_pass"] == 1].copy()
    if not survivors.empty:
        survivors = survivors.sort_values(
            [
                "ruin_event_fixed",
                "geometric_equity_step_return_fixed_clean",
                "max_drawdown_pct_fixed_equity_dec",
                "cvar_5_trade_notional_dec",
                "min_split_trades",
                "expectancy_net_trade_notional_dec",
                "profit_factor_trade",
            ],
            ascending=[True, False, False, False, False, False, False],
        ).reset_index(drop=True)
        survivors["fixed_rank"] = np.arange(1, len(survivors) + 1, dtype=int)
        rank_map = {str(r.variant): int(r.fixed_rank) for r in survivors.itertuples(index=False)}
        results["fixed_rank"] = results["variant"].astype(str).map(rank_map)

    # Baseline-local vs best-variant-local diagnostics (explicit separation).
    if not survivors.empty:
        best_variant_name = str(survivors.iloc[0]["variant"])
    else:
        # No fixed-size passers: select best by ranking objective for diagnostics only.
        tmp = results.sort_values(
            ["ruin_event_fixed", "geometric_equity_step_return_fixed_clean", "max_drawdown_pct_fixed_equity_dec", "cvar_5_trade_notional_dec"],
            ascending=[True, False, False, False],
        )
        best_variant_name = str(tmp.iloc[0]["variant"])
    best_variant_row = results[results["variant"] == best_variant_name].iloc[0]
    best_variant_adverse = _safe_float(best_variant_row["variant_adverse_loss_share"])

    results["baseline_adverse_loss_share"] = float(baseline_adverse)
    results["baseline_dominant_worst_regime_bucket"] = str(baseline_worst_regime)
    results["best_variant_name"] = str(best_variant_name)
    results["best_variant_adverse_loss_share"] = float(best_variant_adverse)

    # Fixed-size gate table.
    fixed_gate = results[
        [
            "variant",
            "is_baseline",
            "signals_total",
            "trades_total",
            "expectancy_net_trade_notional_dec",
            "total_return_fixed_equity_dec",
            "max_drawdown_pct_fixed_equity_dec",
            "cvar_5_trade_notional_dec",
            "profit_factor_trade",
            "geometric_equity_step_return_fixed",
            "geometric_equity_step_return_fixed_clean",
            "ruin_event_fixed",
            "support_ok",
            "min_split_trades",
            "fatal_gate_fixed",
            "fixed_absolute_practical_pass",
            "fixed_rank",
            "gate_expectancy_trade_gt_0",
            "gate_total_return_fixed_gt_0",
            "gate_maxdd_fixed_gt_floor",
            "gate_cvar5_trade_gt_floor",
            "gate_pf_trade_ge_floor",
            "gate_support_ok",
        ]
    ].copy()
    fixed_gate = fixed_gate.sort_values(
        ["fixed_absolute_practical_pass", "fixed_rank", "ruin_event_fixed", "geometric_equity_step_return_fixed_clean"],
        ascending=[False, True, True, False],
        na_position="last",
    )
    fixed_gate.to_csv(run_dir / "phaseI_sol_fixed_size_gate_table.csv", index=False)

    # Multiple-testing accounting.
    fork_only = results[results["is_baseline"] == 0].copy()
    trial_variants = fork_only["variant"].astype(str).tolist()
    if trial_variants:
        mat = np.column_stack([candidate_fixed_step_vec[v] for v in trial_variants])
    else:
        mat = np.zeros((len(rep_signal_ids), 1), dtype=float)
    n_eff, avg_corr = _estimate_effective_trials(mat)
    raw_trials = int(len(trial_variants))

    # Shortlist significance (PSR/DSR).
    shortlist_src = survivors.copy() if not survivors.empty else results.sort_values(
        ["ruin_event_fixed", "geometric_equity_step_return_fixed_clean", "max_drawdown_pct_fixed_equity_dec", "cvar_5_trade_notional_dec"],
        ascending=[True, False, False, False],
    ).copy()
    shortlist = shortlist_src.head(int(args.shortlist_top_k)).copy()
    sig_rows: List[Dict[str, Any]] = []
    for r in shortlist.itertuples(index=False):
        vname = str(getattr(r, "variant"))
        step = candidate_fixed_step_vec.get(vname, np.asarray([], dtype=float))
        sr_hat, skew, kurt, n = _sharpe_moments(step)
        psr0 = _psr(sr_hat, 0.0, skew, kurt, n)
        sr_star = float(math.sqrt(max(0.0, 2.0 * math.log(max(1.0, float(n_eff)))) / max(1.0, float(n - 1.0)))) if np.isfinite(n) else float("nan")
        dsr_neff = _psr(sr_hat, sr_star, skew, kurt, n)
        sig_rows.append(
            {
                "variant": vname,
                "fixed_rank": _safe_float(getattr(r, "fixed_rank")),
                "promoted_to_compounding": int(_safe_float(getattr(r, "fixed_absolute_practical_pass"), default=0.0) == 1.0),
                "n_obs_equity_steps": int(n) if np.isfinite(n) else 0,
                "sharpe_step": float(sr_hat),
                "skew_step": float(skew),
                "kurtosis_step": float(kurt),
                "psr_vs_sr0": float(psr0),
                "dsr_neff": float(dsr_neff),
                "n_eff_used": float(n_eff),
                "raw_trials_used": int(raw_trials),
            }
        )
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(run_dir / "phaseI_sol_shortlist_significance.csv", index=False)

    # Compounding follow-up only for fixed-size passers.
    comp_columns = [
        "variant",
        "fixed_rank",
        "signals_total",
        "trades_total",
        "expectancy_net_trade_notional_dec",
        "cvar_5_trade_notional_dec",
        "profit_factor_trade",
        "total_return_compounded_equity_dec",
        "max_drawdown_pct_compounded_equity_dec",
        "support_ok",
        "fatal_gate_compounded",
        "compounded_absolute_practical_pass",
        "baseline_adverse_loss_share",
        "best_variant_adverse_loss_share",
        "baseline_dominant_worst_regime_bucket",
        "dominant_worst_regime_bucket",
    ]
    comp_rows: List[Dict[str, Any]] = []
    if not survivors.empty:
        for r in survivors.itertuples(index=False):
            vname = str(getattr(r, "variant"))
            trades = candidate_trades[vname]
            sig_total = int(candidate_signals_total[vname])
            _, m_comp, _ = g._compute_metrics(
                trades,
                signals_total=sig_total,
                initial_equity=initial_equity,
                risk_per_trade=risk_per_trade,
            )
            # Re-apply practical gates under compounding.
            exp_trade = _safe_float(getattr(r, "expectancy_net_trade_notional_dec"))
            cvar_trade = _safe_float(getattr(r, "cvar_5_trade_notional_dec"))
            pf_trade = _safe_float(getattr(r, "profit_factor_trade"))
            support_ok = int(_safe_float(getattr(r, "support_ok"), default=0.0))
            tr_comp = _safe_float(m_comp.get("total_return"))
            dd_comp = _safe_float(m_comp.get("max_drawdown_pct"))
            fatal_comp = int(
                (np.isfinite(tr_comp) and tr_comp <= float(args.fatal_total_return))
                or (np.isfinite(dd_comp) and dd_comp <= float(args.fatal_max_dd))
            )
            comp_abs_pass = int(
                np.isfinite(exp_trade)
                and exp_trade > 0.0
                and np.isfinite(tr_comp)
                and tr_comp > 0.0
                and np.isfinite(dd_comp)
                and dd_comp > float(args.deploy_max_dd_floor)
                and np.isfinite(cvar_trade)
                and cvar_trade > float(args.deploy_cvar5_floor)
                and np.isfinite(pf_trade)
                and pf_trade >= float(args.deploy_pf_floor)
                and support_ok == 1
            )
            comp_rows.append(
                {
                    "variant": vname,
                    "fixed_rank": int(_safe_float(getattr(r, "fixed_rank"), default=999999)),
                    "signals_total": int(sig_total),
                    "trades_total": int(_safe_float(m_comp.get("trades_total"), default=0.0)),
                    "expectancy_net_trade_notional_dec": float(exp_trade),
                    "cvar_5_trade_notional_dec": float(cvar_trade),
                    "profit_factor_trade": float(pf_trade),
                    "total_return_compounded_equity_dec": float(tr_comp),
                    "max_drawdown_pct_compounded_equity_dec": float(dd_comp),
                    "support_ok": int(support_ok),
                    "fatal_gate_compounded": int(fatal_comp),
                    "compounded_absolute_practical_pass": int(comp_abs_pass),
                    "baseline_adverse_loss_share": float(baseline_adverse),
                    "best_variant_adverse_loss_share": float(best_variant_adverse),
                    "baseline_dominant_worst_regime_bucket": str(baseline_worst_regime),
                    "dominant_worst_regime_bucket": str(getattr(r, "dominant_worst_regime_bucket")),
                }
            )
    comp_df = pd.DataFrame(comp_rows, columns=comp_columns)
    if not comp_df.empty:
        comp_df = comp_df.sort_values(["compounded_absolute_practical_pass", "fixed_rank"], ascending=[False, True]).reset_index(drop=True)
    # Create file regardless for deterministic artifacts; may be empty when no fixed-size passers.
    comp_df.to_csv(run_dir / "phaseI_sol_compounding_followup_table.csv", index=False)

    # Final decision + root-cause classification.
    any_fixed_pass = int((results["fixed_absolute_practical_pass"] == 1).any())
    any_comp_pass = int((comp_df.get("compounded_absolute_practical_pass", pd.Series(dtype=int)) == 1).any()) if not comp_df.empty else 0
    if any_comp_pass == 1:
        final_decision = "PROCEED_TO_PAPER"
    elif any_fixed_pass == 1:
        final_decision = "HOLD"
    else:
        final_decision = "NO_DEPLOY"

    root_causes: List[str] = []
    if final_decision != "PROCEED_TO_PAPER":
        if any_fixed_pass == 0:
            fail_counts = {
                "signal_quality_gates_fail": int(
                    ((results["gate_expectancy_trade_gt_0"] == 0) | (results["gate_pf_trade_ge_floor"] == 0) | (results["gate_cvar5_trade_gt_floor"] == 0)).sum()
                ),
                "equity_path_gates_fail_fixed": int(
                    ((results["gate_total_return_fixed_gt_0"] == 0) | (results["gate_maxdd_fixed_gt_floor"] == 0)).sum()
                ),
                "support_stability_fail": int((results["gate_support_ok"] == 0).sum()),
            }
            avg_adverse = float(pd.to_numeric(results["variant_adverse_loss_share"], errors="coerce").mean())
            if fail_counts["signal_quality_gates_fail"] >= max(fail_counts.values()):
                root_causes.append("signal quality issue")
            if fail_counts["equity_path_gates_fail_fixed"] >= max(fail_counts.values()):
                root_causes.append("regime concentration / path-quality issue")
            if fail_counts["support_stability_fail"] > 0:
                root_causes.append("support/stability issue")
            if np.isfinite(avg_adverse) and avg_adverse >= 0.55:
                root_causes.append("regime concentration issue")
        else:
            root_causes.append("sizing/compounding amplification issue")
            root_causes.append("interaction issue")
    root_causes = sorted(set(root_causes))

    # Persist core tables.
    results = results.sort_values(
        ["fixed_absolute_practical_pass", "fixed_rank", "ruin_event_fixed", "geometric_equity_step_return_fixed_clean", "max_drawdown_pct_fixed_equity_dec"],
        ascending=[False, True, True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    results.to_csv(run_dir / "phaseI_sol_signal_fork_results.csv", index=False)
    pd.DataFrame(split_rows_all).to_csv(run_dir / "phaseI_split_trade_support.csv", index=False)

    # Multiple testing summary markdown.
    baseline_name = str(baseline_row.iloc[0]["variant"])
    benchmark_rows: List[Dict[str, Any]] = []
    baseline_geom = _safe_float(baseline_row.iloc[0]["geometric_equity_step_return_fixed_clean"])
    baseline_ret_fix = _safe_float(baseline_row.iloc[0]["total_return_fixed_equity_dec"])
    for _, rr in shortlist.iterrows():
        benchmark_rows.append(
            {
                "variant": str(rr["variant"]),
                "delta_geom_step_vs_baseline": _safe_float(rr["geometric_equity_step_return_fixed_clean"]) - baseline_geom,
                "delta_total_return_fixed_vs_baseline": _safe_float(rr["total_return_fixed_equity_dec"]) - baseline_ret_fix,
            }
        )
    benchmark_df = pd.DataFrame(benchmark_rows)

    mt_lines = [
        "# Phase I Multiple-Testing Summary",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Baseline variant: `{baseline_name}`",
        f"- Raw fork trials evaluated: {raw_trials}",
        f"- Effective trials estimate (correlation-adjusted): {float(n_eff):.6f}",
        f"- Average pairwise correlation across fork trial step-returns: {float(avg_corr):.6f}" if np.isfinite(avg_corr) else "- Average pairwise correlation: n/a",
        f"- Shortlist size: {int(len(sig_df))}",
        "",
        "## Reality-Check Benchmark Note",
        "",
        "Shortlisted variants are compared against the contract-locked baseline for directional uplift context.",
        "A full White reality-check bootstrap is not implemented in this run; this remains a required control before any deployment escalation.",
        "",
        _markdown_table(benchmark_df),
    ]
    (run_dir / "phaseI_sol_multiple_testing_summary.md").write_text("\n".join(mt_lines) + "\n", encoding="utf-8")

    # Final report.
    setup_checks = {
        "symbol_match": int(str(ctx["contract"].get("symbol", "")).upper() == symbol),
        "contract_id_match": int(str(ctx["contract"].get("contract_id", "")) == EXPECTED_CONTRACT_ID),
        "fee_hash_match": int(str(ctx["fee_hash"]) == g.EXPECTED_PHASEA_FEE_HASH),
        "metrics_hash_match": int(str(ctx["metrics_hash"]) == g.EXPECTED_PHASEA_METRICS_HASH),
        "subset_hash_match": int(str(ctx["rep_subset_hash"]) == str(rep_hash_expected)),
        "selected_model_set_hash_match": int(str(ctx["selected_model_set_hash"]) == str(model_hash_expected)),
        "split_integrity": int(len(split_definition) >= 1),
    }

    fixed_pass_count = int((results["fixed_absolute_practical_pass"] == 1).sum())
    comp_pass_count = int((comp_df["compounded_absolute_practical_pass"] == 1).sum()) if not comp_df.empty else 0

    report_lines = [
        "# Phase I SOL Signal-Definition Fork Report",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Symbol: {symbol}",
        f"- Source Phase H dir: `{phaseh_dir}`",
        f"- Source Phase G dir: `{phaseg_dir}`",
        "",
        "## Frozen Setup Confirmation",
        "",
        f"- representative_subset_sha256: `{ctx['rep_subset_hash']}`",
        f"- fee_model_sha256: `{ctx['fee_hash']}`",
        f"- metrics_definition_sha256: `{ctx['metrics_hash']}`",
        f"- selected_model_set_sha256: `{ctx['selected_model_set_hash']}`",
        f"- setup_checks: {json.dumps(setup_checks, sort_keys=True)}",
        "",
        "## Fork Components Implemented",
        "",
        "- trend alignment gate (slow trend threshold filter for long entries)",
        "- volatility regime gate (supported bucket filtering)",
        "- de-clustering cooldown windows",
        "- delayed 1h entry modes {0,1,2}",
        "",
        "## Fixed-Size Stage (Mandatory First)",
        "",
        "Metric units:",
        "- trade-level notional metrics: decimal per-trade returns on position notional (`expectancy_net_trade_notional_dec`, `cvar_5_trade_notional_dec`, `profit_factor_trade`)",
        "- equity-path metrics: decimal account equity outcomes (`total_return_fixed_equity_dec`, `max_drawdown_pct_fixed_equity_dec`, `geometric_equity_step_return_fixed_clean`)",
        "",
        f"- candidates_total: {int(len(results))}",
        f"- fixed_size_passers: {fixed_pass_count}",
        f"- baseline_adverse_loss_share: {baseline_adverse:.6f}",
        f"- best_variant_adverse_loss_share: {best_variant_adverse:.6f}",
        f"- baseline_dominant_worst_regime_bucket: `{baseline_worst_regime}`",
        f"- best_variant: `{best_variant_name}`",
        f"- best_variant_dominant_worst_regime_bucket: `{str(best_variant_row['dominant_worst_regime_bucket'])}`",
        "",
        _markdown_table(
            results[
                [
                    "variant",
                    "signals_total",
                    "trades_total",
                    "expectancy_net_trade_notional_dec",
                    "cvar_5_trade_notional_dec",
                    "profit_factor_trade",
                    "geometric_equity_step_return_fixed",
                    "geometric_equity_step_return_fixed_clean",
                    "ruin_event_fixed",
                    "total_return_fixed_equity_dec",
                    "max_drawdown_pct_fixed_equity_dec",
                    "support_ok",
                    "fixed_absolute_practical_pass",
                    "fixed_rank",
                ]
            ]
        ),
        "",
        "Ranking policy used (fixed-size survivors only):",
        "1. primary: `geometric_equity_step_return_fixed_clean` (with `ruin_event_fixed`=0 required for meaningful ranking)",
        "2. secondary: `max_drawdown_pct_fixed_equity_dec`, `cvar_5_trade_notional_dec`, support/stability",
        "3. tertiary: `expectancy_net_trade_notional_dec`, `profit_factor_trade`",
        "",
        "## Compounding Follow-up (Conditional)",
        "",
        f"- fixed-size survivors promoted: {fixed_pass_count}",
        f"- compounded passers: {comp_pass_count}",
        _markdown_table(
            comp_df[
                [
                    "variant",
                    "fixed_rank",
                    "expectancy_net_trade_notional_dec",
                    "total_return_compounded_equity_dec",
                    "max_drawdown_pct_compounded_equity_dec",
                    "cvar_5_trade_notional_dec",
                    "profit_factor_trade",
                    "support_ok",
                    "compounded_absolute_practical_pass",
                ]
            ]
            if not comp_df.empty
            else pd.DataFrame(
                columns=[
                    "variant",
                    "fixed_rank",
                    "expectancy_net_trade_notional_dec",
                    "total_return_compounded_equity_dec",
                    "max_drawdown_pct_compounded_equity_dec",
                    "cvar_5_trade_notional_dec",
                    "profit_factor_trade",
                    "support_ok",
                    "compounded_absolute_practical_pass",
                ]
            )
        ),
        "",
        "## Multiple-Testing and Significance Controls",
        "",
        f"- raw_trials: {raw_trials}",
        f"- effective_trials_estimate: {float(n_eff):.6f}",
        f"- shortlist_rows: {int(len(sig_df))}",
        f"- shortlist_significance_file: `{run_dir / 'phaseI_sol_shortlist_significance.csv'}`",
        f"- multiple_testing_summary_file: `{run_dir / 'phaseI_sol_multiple_testing_summary.md'}`",
        "",
        "## Final Decision",
        "",
        f"- final_decision: **{final_decision}**",
        f"- deployment_status: **{'NO_DEPLOY' if final_decision != 'PROCEED_TO_PAPER' else 'PAPER_ELIGIBLE'}**",
        f"- root_cause_classification: {', '.join(root_causes) if root_causes else 'none'}",
        "",
        "Stop condition:",
        "- If no fixed-size candidate passes absolute gates, optimization stops in this branch with HOLD/NO_DEPLOY.",
    ]
    (run_dir / "phaseI_sol_signal_fork_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    run_manifest = {
        "generated_utc": _utc_now().isoformat(),
        "symbol": symbol,
        "phaseh_source_dir": str(phaseh_dir),
        "phaseg_source_dir": str(phaseg_dir),
        "contract_id": str(ctx["contract"].get("contract_id")),
        "representative_subset_sha256": str(ctx["rep_subset_hash"]),
        "fee_model_sha256": str(ctx["fee_hash"]),
        "metrics_definition_sha256": str(ctx["metrics_hash"]),
        "selected_model_set_sha256": str(ctx["selected_model_set_hash"]),
        "setup_checks": setup_checks,
        "raw_trials": int(raw_trials),
        "effective_trials_estimate": float(n_eff),
        "fixed_size_passers": int(fixed_pass_count),
        "compounded_passers": int(comp_pass_count),
        "final_decision": str(final_decision),
        "root_cause_classification": root_causes,
    }
    (run_dir / "phaseI_run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase I SOL signal-definition fork implementation (contract-locked).")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--phaseh-dir", default="", help="Optional explicit Phase H dir. Defaults to latest PHASEH_SOL_FREEZE_FORK_*")
    ap.add_argument("--phaseg-dir", default="", help="Optional explicit Phase G dir override")

    ap.add_argument("--expected-rep-subset-hash", default=EXPECTED_REPRESENTATIVE_SUBSET_SHA256)
    ap.add_argument("--expected-selected-model-set-hash", default=EXPECTED_SELECTED_MODEL_SET_SHA256)

    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--initial-equity", type=float, default=1.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)

    ap.add_argument("--trend-thresholds", default="0.50,0.60")
    ap.add_argument("--vol-mode-filter", default="", help="Optional comma-separated keys from supported vol modes (e.g. all_supported,mid_only). Empty means all modes.")
    ap.add_argument("--cooldown-hours", default="0,2,4,6")
    ap.add_argument("--delay-bars", default="0,1,2")

    ap.add_argument("--min-split-trades", type=int, default=40)
    ap.add_argument("--min-bucket-support", type=int, default=30)

    ap.add_argument("--fatal-max-dd", type=float, default=-0.95)
    ap.add_argument("--fatal-total-return", type=float, default=-0.95)

    ap.add_argument("--deploy-max-dd-floor", type=float, default=-0.35)
    ap.add_argument("--deploy-cvar5-floor", type=float, default=-0.0015)
    ap.add_argument("--deploy-pf-floor", type=float, default=1.05)

    ap.add_argument("--shortlist-top-k", type=int, default=5)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))


if __name__ == "__main__":
    main()
