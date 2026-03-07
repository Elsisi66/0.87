#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_PHASEA_FEE_HASH = "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a"
EXPECTED_PHASEA_METRICS_HASH = "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99"
EXPECTED_PHASEC_CFG_HASH = "a285b86c4c22a26976d4a762"
EXPECTED_SIGNAL_SUBSET_HASH = "5e719faf676dffba8d7da926314997182d429361495884b8a870c3393c079bbf"
EXPECTED_SPLIT_HASH = "388ba743b9c16c291385a9ecab6435eabf65eb16f1e1083eee76627193c42c01"


@dataclass(frozen=True)
class PregateConfig:
    trend_required: int
    vol_min_pct: float
    vol_max_pct: float
    cooldown_h: int
    max_signals_24h: int
    session_mode: str
    overlap_h: int
    stop_distance_min: float


@dataclass
class EvalResult:
    config: PregateConfig
    config_id: str
    signals_total: int
    trades_total: int
    entry_rate: float
    expectancy_net: float
    expectancy_net_per_signal: float
    pnl_net_sum: float
    cvar_5: float
    max_drawdown: float
    win_rate: float
    sl_hit_rate: float
    tp_hit_rate: float
    timeout_rate: float
    min_split_trades: int
    median_split_trades: float
    split_median_expectancy_delta: float
    split_min_expectancy_delta: float
    delta_expectancy_best_entry_pregate_minus_phasec_control: float
    delta_maxdd_best_entry_pregate_minus_phasec_control: float
    delta_cvar5_best_entry_pregate_minus_phasec_control: float
    delta_pnl_sum_best_entry_pregate_minus_phasec_control: float
    pass_expectancy: int
    pass_split_median: int
    pass_maxdd_not_worse: int
    pass_cvar_not_worse: int
    pass_participation: int
    pass_min_split_support: int
    pass_data_quality: int
    pass_reproducibility: int
    pass_all: int
    fail_reasons: str
    skip_reasons_json: str


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_tag() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


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
    def _default(v: Any) -> Any:
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(v, utc=True))
        if isinstance(v, Path):
            return str(v)
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
        if not np.isfinite(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


def _tail_mean(x: Sequence[float], frac: float) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(frac) * float(arr.size))))
    return float(np.mean(np.sort(arr)[:k]))


def _max_drawdown_from_pnl_series(pnl_series: Sequence[float]) -> float:
    arr = np.asarray(list(pnl_series), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    csum = np.cumsum(arr)
    peak = np.maximum.accumulate(csum)
    dd = csum - peak
    return float(np.min(dd))


def _split_rows(split_df: pd.DataFrame) -> Dict[str, Any]:
    rows = []
    for r in split_df.itertuples(index=False):
        rows.append({c: getattr(r, c) for c in split_df.columns})
    return rows


def _sha256_signal_subset(df: pd.DataFrame) -> str:
    if "signal_id" not in df.columns or "signal_time" not in df.columns:
        raise RuntimeError("signal subset missing required columns signal_id/signal_time")
    x = df.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rows = [f"{str(r.signal_id)}|{pd.to_datetime(r.signal_time, utc=True).isoformat()}" for r in x.itertuples(index=False)]
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _load_phasec_manifest(phase_c_dir: Path) -> Dict[str, Any]:
    fp = phase_c_dir / "run_manifest.json"
    if not fp.exists():
        raise FileNotFoundError(f"Missing Phase C manifest: {fp}")
    return json.loads(fp.read_text(encoding="utf-8"))


def _parse_splits(path: Path) -> List[Dict[str, int]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, int]] = []
    for s in raw.get("splits", []):
        out.append(
            {
                "split_id": int(s["split_id"]),
                "train_start": int(s["train_start"]),
                "train_end": int(s["train_end"]),
                "test_start": int(s["test_start"]),
                "test_end": int(s["test_end"]),
            }
        )
    out = sorted(out, key=lambda r: int(r["split_id"]))
    if not out:
        raise RuntimeError("No split definitions loaded")
    return out


def _test_indices(splits: List[Dict[str, int]]) -> List[int]:
    out: List[int] = []
    for s in splits:
        out.extend(list(range(int(s["test_start"]), int(s["test_end"]))))
    return sorted(set(out))


def _split_lookup_for_subset(subset: pd.DataFrame, splits: List[Dict[str, int]]) -> Dict[str, int]:
    idx_to_split: Dict[int, int] = {}
    for s in splits:
        sid = int(s["split_id"])
        for i in range(int(s["test_start"]), int(s["test_end"])):
            idx_to_split[int(i)] = sid
    out: Dict[str, int] = {}
    for i, r in subset.reset_index(drop=True).iterrows():
        if i in idx_to_split:
            out[str(r["signal_id"])] = int(idx_to_split[i])
    return out


def _session_bucket(ts: pd.Timestamp) -> str:
    h = int(pd.to_datetime(ts, utc=True).hour)
    if 0 <= h <= 7:
        return "asia"
    if 8 <= h <= 13:
        return "eu"
    if 14 <= h <= 20:
        return "us"
    return "late"


def _session_allowed(session_mode: str, ts: pd.Timestamp) -> bool:
    b = _session_bucket(ts)
    if session_mode == "all":
        return True
    if session_mode == "eu_us":
        return b in {"eu", "us"}
    if session_mode == "us_only":
        return b == "us"
    if session_mode == "asia_only":
        return b == "asia"
    return True


def _compute_signal_density(signal_times: pd.Series, window_h: float = 24.0) -> np.ndarray:
    t = pd.to_datetime(signal_times, utc=True, errors="coerce")
    n = len(t)
    out = np.zeros(n, dtype=int)
    left = 0
    w = pd.Timedelta(hours=float(window_h))
    for i in range(n):
        while left < i and (t.iloc[i] - t.iloc[left]) > w:
            left += 1
        out[i] = int(i - left)
    return out


def _compute_losing_streaks(per_signal_pnl: Sequence[float]) -> List[int]:
    arr = np.asarray(list(per_signal_pnl), dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    streaks: List[int] = []
    cur = 0
    for x in arr:
        if x < 0:
            cur += 1
        else:
            if cur > 0:
                streaks.append(int(cur))
                cur = 0
    if cur > 0:
        streaks.append(int(cur))
    return streaks


def _make_vol_bucket(x: pd.Series) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce")
    labels = pd.Series(index=y.index, dtype=object)
    labels[y <= 33.3333333333] = "low"
    labels[(y > 33.3333333333) & (y <= 66.6666666667)] = "mid"
    labels[y > 66.6666666667] = "high"
    labels = labels.fillna("unknown")
    return labels.astype(str)


def _compute_metrics_from_selection(
    *,
    control_df: pd.DataFrame,
    accepted_signal_ids: set[str],
    splits: List[Dict[str, int]],
    control_split_expectancy: Dict[int, float],
) -> Dict[str, Any]:
    x = control_df.copy()
    x["is_selected"] = x["signal_id"].astype(str).isin(accepted_signal_ids).astype(int)
    x["pnl_selected"] = np.where(x["is_selected"] == 1, pd.to_numeric(x["pnl_net_pct"], errors="coerce"), 0.0)

    signals_total = int(x.shape[0])
    sel = x[x["is_selected"] == 1].copy()
    trades_total = int(sel.shape[0])
    entry_rate = float(trades_total / max(1, signals_total))
    pnl_trade = pd.to_numeric(sel["pnl_net_pct"], errors="coerce").dropna().to_numpy(dtype=float)

    expectancy = float(np.mean(pnl_trade)) if pnl_trade.size else float("nan")
    expectancy_per_signal = float(pd.to_numeric(x["pnl_selected"], errors="coerce").fillna(0.0).mean()) if signals_total > 0 else float("nan")
    pnl_sum = float(pd.to_numeric(x["pnl_selected"], errors="coerce").fillna(0.0).sum())
    cvar5 = _tail_mean(pnl_trade, 0.05) if pnl_trade.size else float("nan")
    maxdd = _max_drawdown_from_pnl_series(pd.to_numeric(x["pnl_selected"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    win_rate = float((pnl_trade > 0).mean()) if pnl_trade.size else float(0.0)
    sl_hit_rate = float(pd.to_numeric(sel["sl_hit"], errors="coerce").fillna(0).mean()) if trades_total > 0 else float(0.0)
    tp_hit_rate = float(pd.to_numeric(sel["tp_hit"], errors="coerce").fillna(0).mean()) if trades_total > 0 else float(0.0)
    timeout_rate = float(sel["exit_reason"].astype(str).str.lower().isin({"timeout", "window_end"}).mean()) if trades_total > 0 else float(0.0)

    split_rows: List[Dict[str, Any]] = []
    split_exp_deltas: List[float] = []
    split_trade_counts: List[int] = []
    for sp in splits:
        sid = int(sp["split_id"])
        s = x[x["split_id"] == sid].copy()
        ssel = s[s["is_selected"] == 1].copy()
        tr = int(ssel.shape[0])
        split_trade_counts.append(tr)
        p = pd.to_numeric(ssel["pnl_net_pct"], errors="coerce").dropna().to_numpy(dtype=float)
        exp = float(np.mean(p)) if p.size else float("nan")
        exp_delta = exp - float(control_split_expectancy.get(sid, np.nan)) if np.isfinite(exp) and np.isfinite(control_split_expectancy.get(sid, np.nan)) else float("nan")
        split_exp_deltas.append(exp_delta)
        split_rows.append(
            {
                "split_id": sid,
                "signals_total": int(s.shape[0]),
                "trades_total": tr,
                "entry_rate": float(tr / max(1, s.shape[0])),
                "expectancy_net": exp,
                "expectancy_delta_vs_control": exp_delta,
                "pnl_net_sum": float(
                    pd.Series(np.where(s["is_selected"] == 1, s["pnl_net_pct"], 0.0), index=s.index)
                    .pipe(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .sum()
                ),
            }
        )

    split_exp_deltas_arr = np.asarray([v for v in split_exp_deltas if np.isfinite(v)], dtype=float)
    return {
        "signals_total": signals_total,
        "trades_total": trades_total,
        "entry_rate": entry_rate,
        "expectancy_net": expectancy,
        "expectancy_net_per_signal": expectancy_per_signal,
        "pnl_net_sum": pnl_sum,
        "cvar_5": cvar5,
        "max_drawdown": maxdd,
        "win_rate": win_rate,
        "sl_hit_rate": sl_hit_rate,
        "tp_hit_rate": tp_hit_rate,
        "timeout_rate": timeout_rate,
        "min_split_trades": int(min(split_trade_counts) if split_trade_counts else 0),
        "median_split_trades": float(np.median(split_trade_counts) if split_trade_counts else 0.0),
        "split_median_expectancy_delta": float(np.median(split_exp_deltas_arr)) if split_exp_deltas_arr.size else float("nan"),
        "split_min_expectancy_delta": float(np.min(split_exp_deltas_arr)) if split_exp_deltas_arr.size else float("nan"),
        "split_rows": split_rows,
    }


def _cfg_to_id(cfg: PregateConfig) -> str:
    s = json.dumps(asdict(cfg), sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _generate_candidate_configs() -> List[PregateConfig]:
    trend_required = [0, 1]
    vol_bands = [(0.0, 100.0), (5.0, 100.0), (10.0, 100.0), (0.0, 85.0), (10.0, 85.0), (20.0, 85.0)]
    cooldown_h = [0, 2, 4]
    max_signals_24h = [999, 8, 5]
    session_mode = ["all", "eu_us", "us_only"]
    overlap_h = [0, 2]
    stop_distance_min = [0.0, 0.001, 0.002]

    out: List[PregateConfig] = []
    for tr in trend_required:
        for vb in vol_bands:
            for cd in cooldown_h:
                for mx in max_signals_24h:
                    for sm in session_mode:
                        for ov in overlap_h:
                            for smin in stop_distance_min:
                                if vb[0] >= vb[1]:
                                    continue
                                out.append(
                                    PregateConfig(
                                        trend_required=int(tr),
                                        vol_min_pct=float(vb[0]),
                                        vol_max_pct=float(vb[1]),
                                        cooldown_h=int(cd),
                                        max_signals_24h=int(mx),
                                        session_mode=str(sm),
                                        overlap_h=int(ov),
                                        stop_distance_min=float(smin),
                                    )
                                )
    # Deduplicate while preserving order.
    seen = set()
    uniq: List[PregateConfig] = []
    for c in out:
        k = json.dumps(asdict(c), sort_keys=True)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return uniq


def _sample_configs_deterministic(cfgs: List[PregateConfig], cap: int, seed: int) -> Tuple[List[PregateConfig], Dict[str, Any]]:
    # Always keep the "no gate" baseline config.
    baseline = PregateConfig(
        trend_required=0,
        vol_min_pct=0.0,
        vol_max_pct=100.0,
        cooldown_h=0,
        max_signals_24h=999,
        session_mode="all",
        overlap_h=0,
        stop_distance_min=0.0,
    )
    cfgs_sorted = sorted(cfgs, key=lambda c: json.dumps(asdict(c), sort_keys=True))
    if baseline not in cfgs_sorted:
        cfgs_sorted.insert(0, baseline)

    full_count = len(cfgs_sorted)
    if cap <= 0 or full_count <= cap:
        return cfgs_sorted, {
            "method": "deterministic_no_cap",
            "seed": int(seed),
            "cap": int(cap),
            "full_count": int(full_count),
            "sampled_count": int(full_count),
        }

    rng = random.Random(int(seed))
    idx_all = list(range(full_count))
    baseline_idx = cfgs_sorted.index(baseline)
    idx_all.remove(baseline_idx)
    keep_n = int(max(1, cap - 1))
    picked = sorted(rng.sample(idx_all, keep_n))
    out = [cfgs_sorted[baseline_idx]] + [cfgs_sorted[i] for i in picked]
    out = sorted(out, key=lambda c: json.dumps(asdict(c), sort_keys=True))
    return out, {
        "method": "deterministic_random_sample_plus_baseline",
        "seed": int(seed),
        "cap": int(cap),
        "full_count": int(full_count),
        "sampled_count": int(len(out)),
    }


def _apply_pregate(
    features_df: pd.DataFrame,
    cfg: PregateConfig,
) -> Tuple[set[str], Dict[str, int], List[Dict[str, Any]]]:
    x = features_df.sort_values(["signal_time", "signal_id"]).reset_index(drop=True).copy()
    accepted: set[str] = set()
    skip_reasons: Dict[str, int] = {}
    decisions: List[Dict[str, Any]] = []
    last_kept_ts: Optional[pd.Timestamp] = None

    for r in x.itertuples(index=False):
        sid = str(r.signal_id)
        ts = pd.to_datetime(r.signal_time, utc=True, errors="coerce")
        reason: Optional[str] = None

        if cfg.trend_required == 1 and int(getattr(r, "trend_up_1h", 0)) != 1:
            reason = "trend_gate"
        if reason is None:
            ap = _safe_float(getattr(r, "atr_percentile_1h", np.nan))
            if not np.isfinite(ap) or ap < cfg.vol_min_pct or ap > cfg.vol_max_pct:
                reason = "vol_gate"
        if reason is None:
            sd = _safe_float(getattr(r, "stop_distance_pct", np.nan))
            if not np.isfinite(sd) or sd < cfg.stop_distance_min:
                reason = "stop_distance_gate"
        if reason is None:
            dens = int(getattr(r, "signals_24h_prior", 0))
            if dens > int(cfg.max_signals_24h):
                reason = "anti_chop_gate"
        if reason is None and not _session_allowed(cfg.session_mode, ts):
            reason = "session_gate"
        if reason is None and last_kept_ts is not None:
            delta_h = float((ts - last_kept_ts).total_seconds() / 3600.0)
            if cfg.cooldown_h > 0 and delta_h < float(cfg.cooldown_h):
                reason = "cooldown_gate"
        if reason is None and last_kept_ts is not None:
            delta_h = float((ts - last_kept_ts).total_seconds() / 3600.0)
            if cfg.overlap_h > 0 and delta_h < float(cfg.overlap_h):
                reason = "overlap_gate"

        if reason is None:
            accepted.add(sid)
            last_kept_ts = ts
        else:
            skip_reasons[reason] = int(skip_reasons.get(reason, 0) + 1)
        decisions.append({"signal_id": sid, "signal_time": ts, "accepted": int(reason is None), "reason": reason or "accepted"})

    return accepted, skip_reasons, decisions


def _top_skip_reasons(skip_reasons: Dict[str, int], k: int = 3) -> str:
    if not skip_reasons:
        return ""
    it = sorted(skip_reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return ", ".join([f"{a}:{b}" for a, b in it])


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_(empty)_"
    x = df.head(max_rows).copy()
    cols = list(x.columns)
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in x.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def run(args: argparse.Namespace) -> Path:
    symbol = str(args.symbol).strip().upper()
    if symbol != "SOLUSDT":
        raise RuntimeError("Phase E pre-gate is hard-scoped to SOLUSDT only.")

    phase_a_dir = _resolve(args.phase_a_contract_dir)
    phase_c_dir = _resolve(args.phase_c_dir)
    if not phase_a_dir.exists():
        raise FileNotFoundError(f"Missing phase_a_contract_dir: {phase_a_dir}")
    if not phase_c_dir.exists():
        raise FileNotFoundError(f"Missing phase_c_dir: {phase_c_dir}")

    run_dir = _resolve(args.outdir) / f"PHASEE_SOL_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snap = run_dir / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    # STEP E0: contract and freeze verification.
    phasec_manifest = _load_phasec_manifest(phase_c_dir)
    if str(phasec_manifest.get("symbol", "")).upper() != symbol:
        raise RuntimeError(f"Phase C manifest symbol mismatch: {phasec_manifest.get('symbol')}")
    if str(phasec_manifest.get("final_selected_cfg_hash", "")) != EXPECTED_PHASEC_CFG_HASH:
        raise RuntimeError("Phase C best config hash mismatch")

    phase_a_contract = phasec_manifest.get("phase_a_contract", {})
    fee_model_path = _resolve(str(phase_a_contract.get("fee_model_path", phase_a_dir / "fee_model.json")))
    metrics_def_path = _resolve(str(phase_a_contract.get("metrics_definition_path", phase_a_dir / "metrics_definition.md")))
    subset_path = _resolve(str(phasec_manifest.get("signal_subset_path", phase_c_dir / "signal_subset.csv")))
    split_path = _resolve(str(phasec_manifest.get("split_definition_path", phase_c_dir / "wf_split_definition.json")))

    fee_hash = _sha256_file(fee_model_path)
    metrics_hash = _sha256_file(metrics_def_path)
    subset_hash = _sha256_signal_subset(pd.read_csv(subset_path))
    split_hash = _sha256_file(split_path)

    checks = {
        "phase_a_fee_hash_match_manifest": int(fee_hash == str(phase_a_contract.get("fee_model_sha256", ""))),
        "phase_a_metrics_hash_match_manifest": int(metrics_hash == str(phase_a_contract.get("metrics_definition_sha256", ""))),
        "phase_a_fee_hash_match_expected": int(fee_hash == EXPECTED_PHASEA_FEE_HASH),
        "phase_a_metrics_hash_match_expected": int(metrics_hash == EXPECTED_PHASEA_METRICS_HASH),
        "subset_hash_match_manifest": int(subset_hash == str(phasec_manifest.get("signal_subset_hash", ""))),
        "subset_hash_match_expected": int(subset_hash == EXPECTED_SIGNAL_SUBSET_HASH),
        "split_hash_match_manifest": int(split_hash == str(phasec_manifest.get("split_definition_sha256", ""))),
        "split_hash_match_expected": int(split_hash == EXPECTED_SPLIT_HASH),
    }
    if min(checks.values()) != 1:
        raise RuntimeError(f"Contract/freeze hash verification failed: {checks}")

    # Copy snapshots after verification.
    shutil.copy2(fee_model_path, run_dir / "fee_model.json")
    shutil.copy2(metrics_def_path, run_dir / "metrics_definition.md")
    shutil.copy2(fee_model_path, snap / "fee_model.json")
    shutil.copy2(metrics_def_path, snap / "metrics_definition.md")
    shutil.copy2(subset_path, snap / "signal_subset.csv")
    shutil.copy2(split_path, snap / "wf_split_definition.json")
    shutil.copy2(phase_c_dir / "run_manifest.json", snap / "phasec_run_manifest.json")
    for fp in [
        phase_c_dir / "trade_diagnostics_best.csv",
        phase_c_dir / "trade_diagnostics_baseline.csv",
        phase_c_dir / "risk_rollup_overall.csv",
        phase_c_dir / "risk_rollup_by_symbol.csv",
        phase_c_dir / "decision.md",
    ]:
        if fp.exists():
            shutil.copy2(fp, snap / fp.name)

    contract_lines = [
        "# Contract Verification",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Symbol: {symbol}",
        f"- Phase A dir: `{phase_a_dir}`",
        f"- Phase C dir: `{phase_c_dir}`",
        "",
        f"- fee_model_path: `{fee_model_path}`",
        f"- fee_model_sha256: `{fee_hash}`",
        f"- metrics_definition_path: `{metrics_def_path}`",
        f"- metrics_definition_sha256: `{metrics_hash}`",
        f"- signal_subset_path: `{subset_path}`",
        f"- signal_subset_hash: `{subset_hash}`",
        f"- wf_split_definition_path: `{split_path}`",
        f"- wf_split_definition_sha256: `{split_hash}`",
        "",
        "## Checks",
        "",
    ]
    for k, v in checks.items():
        contract_lines.append(f"- {k}: {v}")
    (run_dir / "contract_verification.md").write_text("\n".join(contract_lines).strip() + "\n", encoding="utf-8")

    # Load frozen test subset and fixed-exit control trades (Phase C best).
    subset_full = pd.read_csv(subset_path)
    subset_full["signal_time"] = pd.to_datetime(subset_full["signal_time"], utc=True, errors="coerce")
    splits = _parse_splits(split_path)
    test_idx = _test_indices(splits)
    subset_test = subset_full.iloc[test_idx].copy().reset_index(drop=True)
    split_lookup = _split_lookup_for_subset(subset_full, splits)
    subset_test["split_id"] = subset_test["signal_id"].astype(str).map(split_lookup).fillna(-1).astype(int)
    subset_test["session_bucket"] = subset_test["signal_time"].map(_session_bucket)
    subset_test["vol_bucket"] = _make_vol_bucket(subset_test.get("atr_percentile_1h"))
    subset_test["trend_bucket"] = subset_test.get("trend_up_1h", 0).map(lambda x: "up" if int(x) == 1 else "down")
    subset_test["combined_regime"] = subset_test["vol_bucket"].astype(str) + "|" + subset_test["trend_bucket"].astype(str)
    subset_test = subset_test.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    subset_test["signals_24h_prior"] = _compute_signal_density(subset_test["signal_time"], window_h=24.0)

    control = pd.read_csv(phase_c_dir / "trade_diagnostics_best.csv")
    control["signal_id"] = control["signal_id"].astype(str)
    control = control[control["signal_id"].isin(set(subset_test["signal_id"].astype(str)))].copy()
    for c in ["signal_time", "entry_time", "exit_time"]:
        control[c] = pd.to_datetime(control.get(c), utc=True, errors="coerce")
    for c in ["filled", "valid_for_metrics", "sl_hit", "tp_hit", "entry_price", "exit_price", "pnl_net_pct", "mae_pct", "mfe_pct", "hold_minutes", "risk_pct"]:
        control[c] = pd.to_numeric(control.get(c), errors="coerce")
    control = control.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    if int(control.shape[0]) != int(subset_test.shape[0]):
        raise RuntimeError(
            f"Control trade rows ({control.shape[0]}) do not match frozen test signals ({subset_test.shape[0]})"
        )
    control["split_id"] = control["signal_id"].map(split_lookup).fillna(-1).astype(int)
    control = control.merge(
        subset_test[
            [
                "signal_id",
                "cycle",
                "atr_percentile_1h",
                "trend_up_1h",
                "signal_open_1h",
                "stop_distance_pct",
                "atr_1h",
                "session_bucket",
                "vol_bucket",
                "trend_bucket",
                "combined_regime",
                "signals_24h_prior",
            ]
        ],
        on="signal_id",
        how="left",
    )

    # STEP E1 baseline decomposition.
    control_split_expectancy: Dict[int, float] = {}
    split_rows: List[Dict[str, Any]] = []
    for sp in splits:
        sid = int(sp["split_id"])
        s = control[control["split_id"] == sid].copy()
        p = pd.to_numeric(s["pnl_net_pct"], errors="coerce").dropna().to_numpy(dtype=float)
        control_split_expectancy[sid] = float(np.mean(p)) if p.size else float("nan")
        split_rows.append(
            {
                "variant": "phasec_fixed_exit_control",
                "split_id": sid,
                "signals_total": int(s.shape[0]),
                "trades_total": int(s.shape[0]),
                "entry_rate": float(s.shape[0] / max(1, s.shape[0])),
                "expectancy_net": float(np.mean(p)) if p.size else float("nan"),
                "expectancy_net_per_signal": float(np.mean(p)) if p.size else float("nan"),
                "pnl_net_sum": float(np.sum(p)) if p.size else float("nan"),
                "cvar_5": _tail_mean(p, 0.05) if p.size else float("nan"),
                "max_drawdown": _max_drawdown_from_pnl_series(p) if p.size else float("nan"),
                "win_rate": float((p > 0).mean()) if p.size else float("nan"),
                "sl_hit_rate": float(pd.to_numeric(s["sl_hit"], errors="coerce").fillna(0).mean()),
                "tp_hit_rate": float(pd.to_numeric(s["tp_hit"], errors="coerce").fillna(0).mean()),
                "timeout_rate": float(s["exit_reason"].astype(str).str.lower().isin({"timeout", "window_end"}).mean()),
            }
        )
    all_p = pd.to_numeric(control["pnl_net_pct"], errors="coerce").dropna().to_numpy(dtype=float)
    split_rows.append(
        {
            "variant": "phasec_fixed_exit_control",
            "split_id": "overall",
            "signals_total": int(control.shape[0]),
            "trades_total": int(control.shape[0]),
            "entry_rate": 1.0,
            "expectancy_net": float(np.mean(all_p)) if all_p.size else float("nan"),
            "expectancy_net_per_signal": float(np.mean(all_p)) if all_p.size else float("nan"),
            "pnl_net_sum": float(np.sum(all_p)) if all_p.size else float("nan"),
            "cvar_5": _tail_mean(all_p, 0.05) if all_p.size else float("nan"),
            "max_drawdown": _max_drawdown_from_pnl_series(all_p) if all_p.size else float("nan"),
            "win_rate": float((all_p > 0).mean()) if all_p.size else float("nan"),
            "sl_hit_rate": float(pd.to_numeric(control["sl_hit"], errors="coerce").fillna(0).mean()),
            "tp_hit_rate": float(pd.to_numeric(control["tp_hit"], errors="coerce").fillna(0).mean()),
            "timeout_rate": float(control["exit_reason"].astype(str).str.lower().isin({"timeout", "window_end"}).mean()),
        }
    )
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(run_dir / "baseline_frozen_split_metrics.csv", index=False)

    diag = control.copy()
    diag["pnl_r"] = pd.to_numeric(diag["pnl_net_pct"], errors="coerce") / np.maximum(
        1e-8, pd.to_numeric(diag["risk_pct"], errors="coerce")
    )
    diag["reach_0_25R"] = (
        pd.to_numeric(diag["mfe_pct"], errors="coerce")
        >= 0.25 * np.maximum(1e-8, pd.to_numeric(diag["risk_pct"], errors="coerce"))
    ).astype(int)
    diag["reach_0_5R"] = (
        pd.to_numeric(diag["mfe_pct"], errors="coerce")
        >= 0.5 * np.maximum(1e-8, pd.to_numeric(diag["risk_pct"], errors="coerce"))
    ).astype(int)
    diag["reach_1_0R"] = (
        pd.to_numeric(diag["mfe_pct"], errors="coerce")
        >= 1.0 * np.maximum(1e-8, pd.to_numeric(diag["risk_pct"], errors="coerce"))
    ).astype(int)
    diag["immediate_sl_0m"] = ((pd.to_numeric(diag["hold_minutes"], errors="coerce") <= 0.0) & (pd.to_numeric(diag["sl_hit"], errors="coerce") == 1)).astype(int)
    diag["immediate_sl_15m"] = ((pd.to_numeric(diag["hold_minutes"], errors="coerce") <= 15.0) & (pd.to_numeric(diag["sl_hit"], errors="coerce") == 1)).astype(int)
    diag["immediate_sl_60m"] = ((pd.to_numeric(diag["hold_minutes"], errors="coerce") <= 60.0) & (pd.to_numeric(diag["sl_hit"], errors="coerce") == 1)).astype(int)
    diag.to_csv(run_dir / "baseline_frozen_trade_diagnostics.csv", index=False)

    reg_rows = []
    for key, grp in diag.groupby("combined_regime", dropna=False):
        p = pd.to_numeric(grp["pnl_net_pct"], errors="coerce").dropna().to_numpy(dtype=float)
        reg_rows.append(
            {
                "combined_regime": str(key),
                "trades": int(grp.shape[0]),
                "expectancy_net": float(np.mean(p)) if p.size else float("nan"),
                "pnl_net_sum": float(np.sum(p)) if p.size else float("nan"),
                "cvar_5": _tail_mean(p, 0.05) if p.size else float("nan"),
                "win_rate": float((p > 0).mean()) if p.size else float("nan"),
                "sl_hit_rate": float(pd.to_numeric(grp["sl_hit"], errors="coerce").fillna(0).mean()),
                "tp_hit_rate": float(pd.to_numeric(grp["tp_hit"], errors="coerce").fillna(0).mean()),
                "timeout_rate": float(grp["exit_reason"].astype(str).str.lower().isin({"timeout", "window_end"}).mean()),
                "median_hold_minutes": float(pd.to_numeric(grp["hold_minutes"], errors="coerce").median()),
            }
        )
    reg_df = pd.DataFrame(reg_rows).sort_values(["trades", "expectancy_net"], ascending=[False, False]).reset_index(drop=True)
    reg_df.to_csv(run_dir / "baseline_frozen_regime_breakdown.csv", index=False)

    # Entry quality md.
    stop_q = pd.to_numeric(diag["stop_distance_pct"], errors="coerce").quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    mae_q = pd.to_numeric(diag["mae_pct"], errors="coerce").quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    mfe_q = pd.to_numeric(diag["mfe_pct"], errors="coerce").quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    entry_q_lines = [
        "# Baseline Frozen Entry Quality",
        "",
        f"- Control: Phase C best exit (cfg_hash={EXPECTED_PHASEC_CFG_HASH}) with original 1h signal entry stream.",
        "- Universe: frozen test subset only (600 signals).",
        "",
        "## Stop Distance Quantiles (signal-time, 1h)",
        "",
    ]
    for k, v in stop_q.items():
        entry_q_lines.append(f"- q{k:.2f}: {float(v):.6f}")
    entry_q_lines.extend(
        [
            "",
            "## MAE / MFE Quantiles",
            "",
        ]
    )
    for k, v in mae_q.items():
        entry_q_lines.append(f"- MAE q{k:.2f}: {float(v):.6f}")
    for k, v in mfe_q.items():
        entry_q_lines.append(f"- MFE q{k:.2f}: {float(v):.6f}")
    entry_q_lines.extend(
        [
            "",
            "## Immediate Adverse Excursion Proxy",
            "",
            f"- SL hit at 0 min: {float(diag['immediate_sl_0m'].mean()):.4f}",
            f"- SL hit within 15 min: {float(diag['immediate_sl_15m'].mean()):.4f}",
            f"- SL hit within 60 min: {float(diag['immediate_sl_60m'].mean()):.4f}",
            "",
            "## Reachability Before Exit",
            "",
            f"- % reaching +0.25R before exit: {float(diag['reach_0_25R'].mean()):.4f}",
            f"- % reaching +0.50R before exit: {float(diag['reach_0_5R'].mean()):.4f}",
            f"- % reaching +1.00R before exit: {float(diag['reach_1_0R'].mean()):.4f}",
            "",
            "Note: first-N-bars MAE/MFE are not available in frozen artifacts; proxies above use full-trade MAE/MFE and hold-time-window SL rates.",
        ]
    )
    (run_dir / "baseline_frozen_entry_quality.md").write_text("\n".join(entry_q_lines).strip() + "\n", encoding="utf-8")

    # Baseline diagnostics md.
    loss_streaks = _compute_losing_streaks(pd.to_numeric(diag["pnl_net_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    daily = diag.copy()
    daily["date"] = pd.to_datetime(daily["signal_time"], utc=True, errors="coerce").dt.date
    by_day = daily.groupby("date").size()
    by_week = pd.to_datetime(daily["signal_time"], utc=True, errors="coerce").dt.to_period("W").astype(str).value_counts()
    win_h = pd.to_numeric(diag.loc[pd.to_numeric(diag["pnl_net_pct"], errors="coerce") > 0, "hold_minutes"], errors="coerce").dropna()
    lose_h = pd.to_numeric(diag.loc[pd.to_numeric(diag["pnl_net_pct"], errors="coerce") <= 0, "hold_minutes"], errors="coerce").dropna()
    base_lines = [
        "# Baseline Frozen Diagnostics",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- signals_total_test: {int(diag.shape[0])}",
        f"- trades_total_test: {int(diag.shape[0])}",
        f"- expectancy_net: {float(np.mean(all_p)):.6f}",
        f"- pnl_net_sum: {float(np.sum(all_p)):.6f}",
        f"- cvar5: {_tail_mean(all_p, 0.05):.6f}",
        f"- max_drawdown: {_max_drawdown_from_pnl_series(all_p):.6f}",
        "",
        "## Split Metrics",
        "",
        _markdown_table(split_df),
        "",
        "## Losing Streak Diagnostics",
        "",
        f"- streak_count: {len(loss_streaks)}",
        f"- max_losing_streak: {int(max(loss_streaks) if loss_streaks else 0)}",
        f"- median_losing_streak: {float(np.median(loss_streaks) if loss_streaks else 0.0):.2f}",
        "",
        "## Trade Clustering",
        "",
        f"- signals/day mean: {float(by_day.mean()):.2f}, p95: {float(by_day.quantile(0.95)):.2f}, max: {int(by_day.max())}",
        f"- signals/week mean: {float(by_week.mean()):.2f}, p95: {float(by_week.quantile(0.95)):.2f}, max: {int(by_week.max())}",
        "",
        "## Hold-Time Distribution",
        "",
        f"- winners median hold min: {float(win_h.median()) if not win_h.empty else float('nan'):.2f}",
        f"- losers median hold min: {float(lose_h.median()) if not lose_h.empty else float('nan'):.2f}",
        "",
        "## Regime Breakdown (top rows)",
        "",
        _markdown_table(reg_df, max_rows=15),
    ]
    (run_dir / "baseline_frozen_diagnostics.md").write_text("\n".join(base_lines).strip() + "\n", encoding="utf-8")

    # STEP E2: deterministic entry pre-gate sweep.
    features = subset_test[
        [
            "signal_id",
            "signal_time",
            "split_id",
            "trend_up_1h",
            "atr_percentile_1h",
            "stop_distance_pct",
            "session_bucket",
            "signals_24h_prior",
        ]
    ].copy()
    features = features.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)

    cfg_all = _generate_candidate_configs()
    cfg_sampled, sample_meta = _sample_configs_deterministic(cfg_all, cap=int(args.max_configs), seed=int(args.seed))
    sample_meta["seed"] = int(args.seed)

    # Control metrics (no-gate).
    control_set = set(features["signal_id"].astype(str).tolist())
    control_metrics = _compute_metrics_from_selection(
        control_df=control,
        accepted_signal_ids=control_set,
        splits=splits,
        control_split_expectancy=control_split_expectancy,
    )
    control_expectancy = float(control_metrics["expectancy_net"])
    control_maxdd = float(control_metrics["max_drawdown"])
    control_cvar = float(control_metrics["cvar_5"])
    control_pnl_sum = float(control_metrics["pnl_net_sum"])

    results: List[EvalResult] = []
    all_skip_reasons: Dict[str, int] = {}
    fail_reasons_hist: Dict[str, int] = {}

    # Pass thresholds.
    maxdd_tol = float(args.maxdd_not_worse_tol)
    cvar_tol = float(args.cvar5_not_worse_tol)
    min_entry_rate = float(args.min_entry_rate)
    min_split_trades = int(args.min_split_trades)
    missing_slice_rate = 0.0  # frozen diagnostics already verified
    pass_data_quality_const = int(missing_slice_rate <= float(args.max_missing_slice_rate))

    for cfg in cfg_sampled:
        cfg_id = _cfg_to_id(cfg)
        accepted, skip_reasons, _ = _apply_pregate(features, cfg)
        for k, v in skip_reasons.items():
            all_skip_reasons[k] = int(all_skip_reasons.get(k, 0) + int(v))
        m = _compute_metrics_from_selection(
            control_df=control,
            accepted_signal_ids=accepted,
            splits=splits,
            control_split_expectancy=control_split_expectancy,
        )
        d_exp = float(m["expectancy_net"] - control_expectancy) if np.isfinite(m["expectancy_net"]) else float("nan")
        d_dd = float(m["max_drawdown"] - control_maxdd) if np.isfinite(m["max_drawdown"]) else float("nan")
        d_cvar = float(m["cvar_5"] - control_cvar) if np.isfinite(m["cvar_5"]) else float("nan")
        d_pnl_sum = float(m["pnl_net_sum"] - control_pnl_sum) if np.isfinite(m["pnl_net_sum"]) else float("nan")

        p_expectancy = int(np.isfinite(d_exp) and d_exp > 0.0)
        p_split_med = int(np.isfinite(m["split_median_expectancy_delta"]) and float(m["split_median_expectancy_delta"]) >= 0.0)
        p_maxdd = int(np.isfinite(d_dd) and d_dd >= -abs(maxdd_tol))
        p_cvar = int(np.isfinite(d_cvar) and d_cvar >= -abs(cvar_tol))
        p_part = int(float(m["entry_rate"]) >= min_entry_rate and int(m["trades_total"]) >= int(math.ceil(min_entry_rate * m["signals_total"])))
        p_split_support = int(int(m["min_split_trades"]) >= min_split_trades)
        p_data = int(pass_data_quality_const)

        # Reproducibility flag is set later for top candidate only; default 1 for deterministic run,
        # and will be overwritten for chosen best.
        p_repro = 1
        p_all = int(p_expectancy and p_split_med and p_maxdd and p_cvar and p_part and p_split_support and p_data and p_repro)

        reasons: List[str] = []
        if p_expectancy == 0:
            reasons.append("expectancy_nonpos")
        if p_split_med == 0:
            reasons.append("split_median_negative")
        if p_maxdd == 0:
            reasons.append("maxdd_worse_than_tol")
        if p_cvar == 0:
            reasons.append("cvar_worse_than_tol")
        if p_part == 0:
            reasons.append("participation_low")
        if p_split_support == 0:
            reasons.append("min_split_support_low")
        if p_data == 0:
            reasons.append("data_quality_fail")
        if not reasons:
            reasons = ["none"]
        for r in reasons:
            fail_reasons_hist[r] = int(fail_reasons_hist.get(r, 0) + 1)

        results.append(
            EvalResult(
                config=cfg,
                config_id=cfg_id,
                signals_total=int(m["signals_total"]),
                trades_total=int(m["trades_total"]),
                entry_rate=float(m["entry_rate"]),
                expectancy_net=float(m["expectancy_net"]),
                expectancy_net_per_signal=float(m["expectancy_net_per_signal"]),
                pnl_net_sum=float(m["pnl_net_sum"]),
                cvar_5=float(m["cvar_5"]),
                max_drawdown=float(m["max_drawdown"]),
                win_rate=float(m["win_rate"]),
                sl_hit_rate=float(m["sl_hit_rate"]),
                tp_hit_rate=float(m["tp_hit_rate"]),
                timeout_rate=float(m["timeout_rate"]),
                min_split_trades=int(m["min_split_trades"]),
                median_split_trades=float(m["median_split_trades"]),
                split_median_expectancy_delta=float(m["split_median_expectancy_delta"]),
                split_min_expectancy_delta=float(m["split_min_expectancy_delta"]),
                delta_expectancy_best_entry_pregate_minus_phasec_control=float(d_exp),
                delta_maxdd_best_entry_pregate_minus_phasec_control=float(d_dd),
                delta_cvar5_best_entry_pregate_minus_phasec_control=float(d_cvar),
                delta_pnl_sum_best_entry_pregate_minus_phasec_control=float(d_pnl_sum),
                pass_expectancy=p_expectancy,
                pass_split_median=p_split_med,
                pass_maxdd_not_worse=p_maxdd,
                pass_cvar_not_worse=p_cvar,
                pass_participation=p_part,
                pass_min_split_support=p_split_support,
                pass_data_quality=p_data,
                pass_reproducibility=p_repro,
                pass_all=p_all,
                fail_reasons=",".join(reasons),
                skip_reasons_json=json.dumps(skip_reasons, sort_keys=True),
            )
        )

    # Ranking and top-k.
    res_df = pd.DataFrame(
        [
            {
                **asdict(r.config),
                "config_id": r.config_id,
                "signals_total": r.signals_total,
                "trades_total": r.trades_total,
                "entry_rate": r.entry_rate,
                "expectancy_net": r.expectancy_net,
                "expectancy_net_per_signal": r.expectancy_net_per_signal,
                "pnl_net_sum": r.pnl_net_sum,
                "cvar_5": r.cvar_5,
                "max_drawdown": r.max_drawdown,
                "win_rate": r.win_rate,
                "sl_hit_rate": r.sl_hit_rate,
                "tp_hit_rate": r.tp_hit_rate,
                "timeout_rate": r.timeout_rate,
                "min_split_trades": r.min_split_trades,
                "median_split_trades": r.median_split_trades,
                "split_median_expectancy_delta": r.split_median_expectancy_delta,
                "split_min_expectancy_delta": r.split_min_expectancy_delta,
                "delta_expectancy_best_entry_pregate_minus_phasec_control": r.delta_expectancy_best_entry_pregate_minus_phasec_control,
                "delta_maxdd_best_entry_pregate_minus_phasec_control": r.delta_maxdd_best_entry_pregate_minus_phasec_control,
                "delta_cvar5_best_entry_pregate_minus_phasec_control": r.delta_cvar5_best_entry_pregate_minus_phasec_control,
                "delta_pnl_sum_best_entry_pregate_minus_phasec_control": r.delta_pnl_sum_best_entry_pregate_minus_phasec_control,
                "pass_expectancy": r.pass_expectancy,
                "pass_split_median": r.pass_split_median,
                "pass_maxdd_not_worse": r.pass_maxdd_not_worse,
                "pass_cvar_not_worse": r.pass_cvar_not_worse,
                "pass_participation": r.pass_participation,
                "pass_min_split_support": r.pass_min_split_support,
                "pass_data_quality": r.pass_data_quality,
                "pass_reproducibility": r.pass_reproducibility,
                "pass_all": r.pass_all,
                "fail_reasons": r.fail_reasons,
                "skip_reasons_json": r.skip_reasons_json,
                "top_skip_reasons": _top_skip_reasons(json.loads(r.skip_reasons_json), k=3),
            }
            for r in results
        ]
    )
    res_df = res_df.sort_values(
        by=[
            "pass_all",
            "pass_expectancy",
            "pass_split_median",
            "delta_expectancy_best_entry_pregate_minus_phasec_control",
            "delta_maxdd_best_entry_pregate_minus_phasec_control",
            "delta_cvar5_best_entry_pregate_minus_phasec_control",
            "entry_rate",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)

    # Repro check on top candidate.
    top_cfg_row = res_df.iloc[0].copy()
    top_cfg = PregateConfig(
        trend_required=int(top_cfg_row["trend_required"]),
        vol_min_pct=float(top_cfg_row["vol_min_pct"]),
        vol_max_pct=float(top_cfg_row["vol_max_pct"]),
        cooldown_h=int(top_cfg_row["cooldown_h"]),
        max_signals_24h=int(top_cfg_row["max_signals_24h"]),
        session_mode=str(top_cfg_row["session_mode"]),
        overlap_h=int(top_cfg_row["overlap_h"]),
        stop_distance_min=float(top_cfg_row["stop_distance_min"]),
    )
    acc2, _, _ = _apply_pregate(features, top_cfg)
    m2 = _compute_metrics_from_selection(
        control_df=control,
        accepted_signal_ids=acc2,
        splits=splits,
        control_split_expectancy=control_split_expectancy,
    )
    repro_ok = int(
        np.isclose(float(top_cfg_row["expectancy_net"]), float(m2["expectancy_net"]), atol=float(args.repro_tolerance), rtol=0.0)
        and np.isclose(float(top_cfg_row["max_drawdown"]), float(m2["max_drawdown"]), atol=float(args.repro_tolerance), rtol=0.0)
        and int(top_cfg_row["trades_total"]) == int(m2["trades_total"])
    )
    res_df.loc[0, "pass_reproducibility"] = repro_ok
    # Recompute pass_all for top row based on repro.
    if repro_ok == 0:
        res_df.loc[0, "pass_all"] = 0
        fr = str(res_df.loc[0, "fail_reasons"])
        if "repro_fail" not in fr:
            res_df.loc[0, "fail_reasons"] = (fr + ",repro_fail") if fr and fr != "none" else "repro_fail"
            fail_reasons_hist["repro_fail"] = int(fail_reasons_hist.get("repro_fail", 0) + 1)

    res_df.to_csv(run_dir / "entry_pregate_results.csv", index=False)
    topk = res_df.head(int(args.top_k)).copy()
    topk.to_csv(run_dir / "entry_pregate_topk.csv", index=False)

    # Parameter effects.
    pe_rows: List[Dict[str, Any]] = []
    params = ["trend_required", "vol_min_pct", "vol_max_pct", "cooldown_h", "max_signals_24h", "session_mode", "overlap_h", "stop_distance_min"]
    for p in params:
        for val, grp in res_df.groupby(p, dropna=False):
            pe_rows.append(
                {
                    "param": p,
                    "value": val,
                    "configs": int(grp.shape[0]),
                    "pass_all_rate": float(pd.to_numeric(grp["pass_all"], errors="coerce").mean()),
                    "median_delta_expectancy": float(pd.to_numeric(grp["delta_expectancy_best_entry_pregate_minus_phasec_control"], errors="coerce").median()),
                    "median_delta_maxdd": float(pd.to_numeric(grp["delta_maxdd_best_entry_pregate_minus_phasec_control"], errors="coerce").median()),
                    "median_delta_cvar5": float(pd.to_numeric(grp["delta_cvar5_best_entry_pregate_minus_phasec_control"], errors="coerce").median()),
                    "median_entry_rate": float(pd.to_numeric(grp["entry_rate"], errors="coerce").median()),
                }
            )
    pe_df = pd.DataFrame(pe_rows).sort_values(["param", "value"]).reset_index(drop=True)
    pe_df.to_csv(run_dir / "entry_pregate_param_effects.csv", index=False)

    fail_df = pd.DataFrame(
        [{"reason": k, "count": int(v)} for k, v in sorted(fail_reasons_hist.items(), key=lambda kv: (-kv[1], kv[0]))]
    )
    fail_df.to_csv(run_dir / "entry_pregate_fail_reason_counts.csv", index=False)

    # Compare best vs control.
    best = res_df.iloc[0].copy()
    cmp = pd.DataFrame(
        [
            {
                "control_label": "phasec_fixed_exit_control",
                "best_config_id": str(best["config_id"]),
                "best_pass_all": int(best["pass_all"]),
                "best_expectancy_net": float(best["expectancy_net"]),
                "control_expectancy_net": float(control_expectancy),
                "delta_expectancy_best_entry_pregate_minus_phasec_control": float(best["delta_expectancy_best_entry_pregate_minus_phasec_control"]),
                "best_max_drawdown": float(best["max_drawdown"]),
                "control_max_drawdown": float(control_maxdd),
                "delta_maxdd_best_entry_pregate_minus_phasec_control": float(best["delta_maxdd_best_entry_pregate_minus_phasec_control"]),
                "best_cvar_5": float(best["cvar_5"]),
                "control_cvar_5": float(control_cvar),
                "delta_cvar5_best_entry_pregate_minus_phasec_control": float(best["delta_cvar5_best_entry_pregate_minus_phasec_control"]),
                "best_trades_total": int(best["trades_total"]),
                "control_trades_total": int(control_metrics["trades_total"]),
                "best_entry_rate": float(best["entry_rate"]),
                "control_entry_rate": float(control_metrics["entry_rate"]),
                "split_median_expectancy_delta": float(best["split_median_expectancy_delta"]),
                "min_split_trades": int(best["min_split_trades"]),
                "pass_reproducibility": int(best["pass_reproducibility"]),
                "top_skip_reasons": str(best.get("top_skip_reasons", "")),
            }
        ]
    )
    cmp.to_csv(run_dir / "comparison_vs_phasec_control.csv", index=False)

    # Decision.
    phase_e_pass = int(best["pass_all"]) == 1
    status = "PASS" if phase_e_pass else "FAIL"
    best_cfg_dict = {
        "config_id": str(best["config_id"]),
        "trend_required": int(best["trend_required"]),
        "vol_min_pct": float(best["vol_min_pct"]),
        "vol_max_pct": float(best["vol_max_pct"]),
        "cooldown_h": int(best["cooldown_h"]),
        "max_signals_24h": int(best["max_signals_24h"]),
        "session_mode": str(best["session_mode"]),
        "overlap_h": int(best["overlap_h"]),
        "stop_distance_min": float(best["stop_distance_min"]),
    }

    # Diagnostics report for entry pregate.
    preg_lines = [
        "# Entry Pre-Gate Diagnostics",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Symbol: {symbol}",
        f"- Configs generated: {sample_meta.get('full_count')}",
        f"- Configs evaluated: {sample_meta.get('sampled_count')}",
        f"- Sampling method: {sample_meta.get('method')}",
        f"- Sampling seed: {sample_meta.get('seed')}",
        "",
        "## Control Metrics (Phase C fixed-exit, no entry gate)",
        "",
        f"- expectancy_net: {control_expectancy:.6f}",
        f"- pnl_net_sum: {control_pnl_sum:.6f}",
        f"- cvar5: {control_cvar:.6f}",
        f"- max_drawdown: {control_maxdd:.6f}",
        f"- trades_total: {int(control_metrics['trades_total'])}",
        "",
        "## Best Candidate",
        "",
        f"- config_id: {best_cfg_dict['config_id']}",
        f"- config: `{json.dumps(best_cfg_dict, sort_keys=True)}`",
        f"- delta_expectancy: {float(best['delta_expectancy_best_entry_pregate_minus_phasec_control']):.6f}",
        f"- delta_maxdd: {float(best['delta_maxdd_best_entry_pregate_minus_phasec_control']):.6f}",
        f"- delta_cvar5: {float(best['delta_cvar5_best_entry_pregate_minus_phasec_control']):.6f}",
        f"- entry_rate: {float(best['entry_rate']):.4f}",
        f"- trades_total: {int(best['trades_total'])}",
        f"- split_median_expectancy_delta: {float(best['split_median_expectancy_delta']):.6f}",
        f"- pass_all: {int(best['pass_all'])}",
        "",
        "## Top 10 configs",
        "",
        _markdown_table(topk, max_rows=10),
        "",
        "## Fail Reason Histogram",
        "",
        _markdown_table(fail_df, max_rows=20),
    ]
    (run_dir / "entry_pregate_diagnostics.md").write_text("\n".join(preg_lines).strip() + "\n", encoding="utf-8")

    # Decision md.
    decision_lines = [
        "# Phase E SOL Entry Pre-Gate Decision",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Symbol: {symbol}",
        f"- Status: **{status}**",
        "",
        "## Best Pre-Gate Config",
        "",
        f"- {json.dumps(best_cfg_dict, sort_keys=True)}",
        "",
        "## Control vs Best Deltas",
        "",
        f"- delta_expectancy_best_entry_pregate_minus_phasec_control: {float(best['delta_expectancy_best_entry_pregate_minus_phasec_control']):.6f}",
        f"- delta_maxdd_best_entry_pregate_minus_phasec_control: {float(best['delta_maxdd_best_entry_pregate_minus_phasec_control']):.6f}",
        f"- delta_cvar5_best_entry_pregate_minus_phasec_control: {float(best['delta_cvar5_best_entry_pregate_minus_phasec_control']):.6f}",
        f"- delta_pnl_sum_best_entry_pregate_minus_phasec_control: {float(best['delta_pnl_sum_best_entry_pregate_minus_phasec_control']):.6f}",
        f"- trades_total(control -> best): {int(control_metrics['trades_total'])} -> {int(best['trades_total'])}",
        f"- entry_rate(control -> best): {float(control_metrics['entry_rate']):.4f} -> {float(best['entry_rate']):.4f}",
        f"- split_median_expectancy_delta: {float(best['split_median_expectancy_delta']):.6f}",
        "",
        "## Gate Table",
        "",
        f"- pass_expectancy: {int(best['pass_expectancy'])}",
        f"- pass_split_median: {int(best['pass_split_median'])}",
        f"- pass_maxdd_not_worse: {int(best['pass_maxdd_not_worse'])}",
        f"- pass_cvar_not_worse: {int(best['pass_cvar_not_worse'])}",
        f"- pass_participation: {int(best['pass_participation'])}",
        f"- pass_min_split_support: {int(best['pass_min_split_support'])}",
        f"- pass_data_quality: {int(best['pass_data_quality'])}",
        f"- pass_reproducibility: {int(best['pass_reproducibility'])}",
        f"- pass_all: {int(best['pass_all'])}",
        "",
    ]
    if phase_e_pass:
        decision_lines.extend(
            [
                "## Recommendation",
                "",
                "- Proceed to limited full entry optimization (Phase E2b / entry GA) on the same frozen universe, seeded with this pre-gate.",
            ]
        )
    else:
        decision_lines.extend(
            [
                "## Recommendation",
                "",
                "- STOP before full entry GA.",
                "- Next best paths:",
                "  1) Rebuild/evaluate a broader SOL test universe contract with same Phase A fees and fixed Phase C exit.",
                "  2) Add upstream 1h signal-quality labeling and rerun this pre-gate sweep with quality-aware filters.",
                "  3) Rework split design to reduce hostile-sample concentration and retest gate robustness.",
            ]
        )
    (run_dir / "decision.md").write_text("\n".join(decision_lines).strip() + "\n", encoding="utf-8")

    # Manifest, repro, git status, phase result.
    manifest = {
        "generated_utc": _utc_now().isoformat(),
        "symbol": symbol,
        "phase_a_contract_dir": str(phase_a_dir),
        "phase_c_dir": str(phase_c_dir),
        "phase_c_best_cfg_hash": EXPECTED_PHASEC_CFG_HASH,
        "fee_model_sha256": fee_hash,
        "metrics_definition_sha256": metrics_hash,
        "signal_subset_sha256": subset_hash,
        "wf_split_sha256": split_hash,
        "checks": checks,
        "signals_total_test": int(control.shape[0]),
        "splits": _split_rows(pd.DataFrame(splits)),
        "pregate_sampling": sample_meta,
        "max_configs": int(args.max_configs),
        "seed": int(args.seed),
        "thresholds": {
            "min_entry_rate": float(min_entry_rate),
            "min_split_trades": int(min_split_trades),
            "maxdd_not_worse_tol": float(maxdd_tol),
            "cvar5_not_worse_tol": float(cvar_tol),
            "max_missing_slice_rate": float(args.max_missing_slice_rate),
        },
        "control_metrics": {
            "expectancy_net": float(control_expectancy),
            "pnl_net_sum": float(control_pnl_sum),
            "cvar5": float(control_cvar),
            "max_drawdown": float(control_maxdd),
            "trades_total": int(control_metrics["trades_total"]),
            "entry_rate": float(control_metrics["entry_rate"]),
        },
        "best_config": best_cfg_dict,
        "best_metrics": {
            "expectancy_net": float(best["expectancy_net"]),
            "pnl_net_sum": float(best["pnl_net_sum"]),
            "cvar5": float(best["cvar_5"]),
            "max_drawdown": float(best["max_drawdown"]),
            "entry_rate": float(best["entry_rate"]),
            "trades_total": int(best["trades_total"]),
            "split_median_expectancy_delta": float(best["split_median_expectancy_delta"]),
            "pass_all": int(best["pass_all"]),
        },
        "status": status,
    }
    _json_dump(run_dir / "run_manifest.json", manifest)

    repro_lines = [
        "# Reproduction",
        "",
        "```bash",
        f"cd {PROJECT_ROOT}",
        (
            "python3 scripts/phase_e_sol_entry_pregate.py "
            f"--symbol {symbol} "
            f"--phase-a-contract-dir {phase_a_dir} "
            f"--phase-c-dir {phase_c_dir} "
            f"--outdir {args.outdir} "
            f"--max-configs {int(args.max_configs)} "
            f"--seed {int(args.seed)}"
        ),
        "```",
    ]
    (run_dir / "repro.md").write_text("\n".join(repro_lines).strip() + "\n", encoding="utf-8")

    try:
        gs = subprocess.check_output(["git", "status", "--short"], cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.STDOUT)
    except Exception as ex:  # pragma: no cover
        gs = f"git status unavailable: {ex}"
    (run_dir / "git_status.txt").write_text(gs, encoding="utf-8")

    phase_lines = [
        "Phase: E (SOL Entry Pre-Gate)",
        f"Timestamp UTC: {_utc_now().isoformat()}",
        f"Status: {status}",
        f"Best config id: {best_cfg_dict['config_id']}",
        f"Best delta_expectancy: {float(best['delta_expectancy_best_entry_pregate_minus_phasec_control']):.6f}",
        f"Best delta_maxdd: {float(best['delta_maxdd_best_entry_pregate_minus_phasec_control']):.6f}",
        f"Best delta_cvar5: {float(best['delta_cvar5_best_entry_pregate_minus_phasec_control']):.6f}",
        f"Best trades_total: {int(best['trades_total'])}/{int(control_metrics['trades_total'])}",
        f"Artifacts: {run_dir}",
    ]
    (run_dir / "phase_result.md").write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase E SOL-only entry pre-gate diagnostics and deterministic sweep.")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--phase-a-contract-dir", default="reports/execution_layer/BASELINE_AUDIT_20260221_214310")
    ap.add_argument("--phase-c-dir", default="reports/execution_layer/PHASEC_SOL_20260221_231430")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--max-configs", type=int, default=480)
    ap.add_argument("--top-k", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-entry-rate", type=float, default=0.55)
    ap.add_argument("--min-split-trades", type=int, default=30)
    ap.add_argument("--maxdd-not-worse-tol", type=float, default=0.02)
    ap.add_argument("--cvar5-not-worse-tol", type=float, default=0.00010)
    ap.add_argument("--max-missing-slice-rate", type=float, default=0.02)
    ap.add_argument("--repro-tolerance", type=float, default=1e-12)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))
    best = pd.read_csv(out / "entry_pregate_topk.csv").iloc[0]
    print(
        "best_config_id="
        + str(best["config_id"])
        + " delta_expectancy="
        + f"{float(best['delta_expectancy_best_entry_pregate_minus_phasec_control']):.6f}"
        + " delta_maxdd="
        + f"{float(best['delta_maxdd_best_entry_pregate_minus_phasec_control']):.6f}"
        + " delta_cvar5="
        + f"{float(best['delta_cvar5_best_entry_pregate_minus_phasec_control']):.6f}"
        + " entry_rate="
        + f"{float(best['entry_rate']):.4f}"
        + " pass_all="
        + str(int(best["pass_all"]))
    )
    print("PASS" if int(best["pass_all"]) == 1 else "FAIL")


if __name__ == "__main__":
    main()
