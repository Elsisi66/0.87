#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import backtest_exec_phasec_sol as phasec_bt  # noqa: E402
from scripts import phase_e2_sol_representative as e2  # noqa: E402


EXPECTED_PHASEA_FEE_HASH = "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a"
EXPECTED_PHASEA_METRICS_HASH = "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99"
EXPECTED_PHASEC_CFG_HASH = "a285b86c4c22a26976d4a762"


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


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, (datetime, pd.Timestamp)):
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


def _markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_(empty)_"
    x = df.head(max_rows).copy()
    cols = list(x.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in x.iterrows():
        vals: List[str] = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _hash_rep_subset(df: pd.DataFrame) -> str:
    x = df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rows = [f"{str(r.signal_id)}|{pd.to_datetime(r.signal_time, utc=True).isoformat()}" for r in x.itertuples(index=False)]
    return _sha256_text("\n".join(rows))


def _vol_bucket(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=y.index, dtype=object)
    out[y <= 33.3333333333] = "low"
    out[(y > 33.3333333333) & (y <= 66.6666666667)] = "mid"
    out[y > 66.6666666667] = "high"
    return out.fillna("unknown").astype(str)


def _hour_bucket(hour_series: pd.Series) -> pd.Series:
    h = pd.to_numeric(hour_series, errors="coerce")
    out = pd.Series(index=h.index, dtype=object)
    out[(h >= 0) & (h < 6)] = "00_05"
    out[(h >= 6) & (h < 12)] = "06_11"
    out[(h >= 12) & (h < 18)] = "12_17"
    out[(h >= 18) & (h <= 23)] = "18_23"
    return out.fillna("unknown").astype(str)


def _compute_metrics_for_ids(
    trades_df: pd.DataFrame,
    signal_ids: set[str],
    *,
    signals_total: int,
    initial_equity: float,
    risk_per_trade: float,
) -> Dict[str, Any]:
    x = trades_df[trades_df["signal_id"].astype(str).isin(signal_ids)].copy()
    eq, m, _ = phasec_bt._compute_equity_curve(
        x,
        signals_total=int(signals_total),
        initial_equity=float(initial_equity),
        risk_per_trade=float(risk_per_trade),
    )
    m = dict(m)
    m["equity_end"] = float(eq["equity"].iloc[-1]) if not eq.empty else float("nan")
    return m


def _split_metrics(
    trades_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    splits: List[Dict[str, int]],
    *,
    initial_equity: float,
    risk_per_trade: float,
    variant: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ss = subset_df.reset_index(drop=True)
    for sp in splits:
        sid = int(sp["split_id"])
        lo, hi = int(sp["test_start"]), int(sp["test_end"])
        s = ss.iloc[lo:hi].copy()
        sig_ids = set(s["signal_id"].astype(str).tolist())
        m = _compute_metrics_for_ids(
            trades_df,
            sig_ids,
            signals_total=len(s),
            initial_equity=float(initial_equity),
            risk_per_trade=float(risk_per_trade),
        )
        rows.append(
            {
                "variant": variant,
                "split_id": sid,
                "signals_total": int(len(s)),
                "trades_total": int(m.get("trades_total", 0)),
                "expectancy_net": float(m.get("expectancy_net", np.nan)),
                "total_return": float(m.get("total_return", np.nan)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(m.get("cvar_5", np.nan)),
                "profit_factor": float(m.get("profit_factor", np.nan)),
                "win_rate": float(m.get("win_rate", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _apply_cooldown(df: pd.DataFrame, cooldown_hours: int) -> pd.DataFrame:
    if cooldown_hours <= 0 or df.empty:
        return df.copy()
    x = df.sort_values("signal_time").copy()
    keep: List[int] = []
    last_t: Optional[pd.Timestamp] = None
    for i, r in x.iterrows():
        t = pd.to_datetime(r["signal_time"], utc=True, errors="coerce")
        if pd.isna(t):
            continue
        if last_t is None or (t - last_t).total_seconds() >= cooldown_hours * 3600.0:
            keep.append(i)
            last_t = t
    return x.loc[keep].copy().sort_values("signal_time").reset_index(drop=True)


def _entry_table_from_v3(v3: pd.DataFrame, split_lookup: Dict[str, int]) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "signal_id": v3["signal_id"].astype(str),
            "signal_time": pd.to_datetime(v3["signal_time"], utc=True, errors="coerce"),
            "split_id": v3["signal_id"].astype(str).map(split_lookup).fillna(-1).astype(int),
            "signal_tp_mult": pd.to_numeric(v3["signal_tp_mult"], errors="coerce"),
            "signal_sl_mult": pd.to_numeric(v3["signal_sl_mult"], errors="coerce"),
            "entry_time": pd.to_datetime(v3["entry_time"], utc=True, errors="coerce"),
            "entry_price": pd.to_numeric(v3["entry_price"], errors="coerce"),
        }
    )
    return out.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def _delay_entries_using_1h_open(
    entries: pd.DataFrame,
    symbol: str,
    offset_hours_after_signal: int,
) -> pd.DataFrame:
    k1h = e2.recon._load_symbol_df(symbol=symbol, tf="1h").copy()
    k1h["Timestamp"] = pd.to_datetime(k1h["Timestamp"], utc=True, errors="coerce")
    k1h = k1h.dropna(subset=["Timestamp", "Open"]).sort_values("Timestamp").reset_index(drop=True)
    ts = pd.to_datetime(k1h["Timestamp"], utc=True)
    ts_ns = np.array([int(t.value) for t in ts], dtype=np.int64)
    op = pd.to_numeric(k1h["Open"], errors="coerce").to_numpy(dtype=float)

    out = entries.copy()
    new_t: List[pd.Timestamp] = []
    new_px: List[float] = []
    for r in out.itertuples(index=False):
        st = pd.to_datetime(getattr(r, "signal_time"), utc=True, errors="coerce")
        if pd.isna(st):
            new_t.append(pd.NaT)
            new_px.append(float("nan"))
            continue
        target = int((st + pd.Timedelta(hours=int(offset_hours_after_signal))).value)
        idx = int(np.searchsorted(ts_ns, target, side="left"))
        if idx >= len(ts_ns):
            new_t.append(pd.NaT)
            new_px.append(float("nan"))
        else:
            new_t.append(pd.to_datetime(int(ts_ns[idx]), utc=True))
            new_px.append(float(op[idx]) if np.isfinite(op[idx]) else float("nan"))
    out["entry_time"] = new_t
    out["entry_price"] = new_px
    return out


def run(args: argparse.Namespace) -> Path:
    symbol = str(args.symbol).strip().upper()
    if symbol != "SOLUSDT":
        raise RuntimeError("Phase F is hard-scoped to SOLUSDT only.")

    e2_dir = _resolve(args.e2_dir)
    if not e2_dir.exists():
        raise FileNotFoundError(f"Missing E2 dir: {e2_dir}")
    out_root = _resolve(args.outdir)
    run_dir = out_root / f"PHASEF_SOL_SIGNAL_REHAB_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snap = run_dir / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    e2_manifest = json.loads((e2_dir / "run_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((e2_dir / "accounting_contract.json").read_text(encoding="utf-8"))
    phase_c_dir = _resolve(str(e2_manifest["phase_c_dir"]))

    if str(e2_manifest.get("symbol", "")).upper() != symbol:
        raise RuntimeError(f"E2 manifest symbol mismatch: {e2_manifest.get('symbol')}")
    if str(contract.get("symbol", "")).upper() != symbol:
        raise RuntimeError(f"E2 contract symbol mismatch: {contract.get('symbol')}")
    if str(contract.get("fee_model_sha256", "")) != EXPECTED_PHASEA_FEE_HASH:
        raise RuntimeError("E2 fee contract hash mismatch.")
    if str(contract.get("metrics_definition_sha256", "")) != EXPECTED_PHASEA_METRICS_HASH:
        raise RuntimeError("E2 metrics contract hash mismatch.")

    fee_model_path = _resolve(str(contract["fee_model_path"]))
    metrics_def_path = _resolve(str(contract["metrics_definition_path"]))
    fee_hash = _sha256_file(fee_model_path)
    metrics_hash = _sha256_file(metrics_def_path)
    if fee_hash != EXPECTED_PHASEA_FEE_HASH or metrics_hash != EXPECTED_PHASEA_METRICS_HASH:
        raise RuntimeError("Phase A contract hashes changed from expected constants.")

    rep_subset_path = e2_dir / "representative_subset_signals.csv"
    rep_subset_hash_file = e2_dir / "representative_subset_hash.txt"
    rep_subset = pd.read_csv(rep_subset_path)
    rep_subset["signal_id"] = rep_subset["signal_id"].astype(str)
    rep_subset["signal_time"] = pd.to_datetime(rep_subset["signal_time"], utc=True, errors="coerce")
    rep_subset = rep_subset.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rep_hash_calc = _hash_rep_subset(rep_subset[["signal_id", "signal_time"]].copy())
    rep_hash_ref = rep_subset_hash_file.read_text(encoding="utf-8").strip()
    if rep_hash_calc != rep_hash_ref or rep_hash_calc != str(e2_manifest.get("representative_subset_sha256", "")):
        raise RuntimeError("Representative subset hash mismatch with E2 manifest.")

    split_definition = list(e2_manifest.get("split_definition", []))
    if not split_definition:
        raise RuntimeError("E2 split definition missing.")
    split_lookup = e2._split_lookup(rep_subset, split_definition)

    # Copy snapshots.
    for fp in [
        fee_model_path,
        metrics_def_path,
        e2_dir / "accounting_contract.json",
        e2_dir / "run_manifest.json",
        rep_subset_path,
        e2_dir / "representative_subset_manifest.json",
        e2_dir / "pass_fail_gates.json",
        e2_dir / "trades_v2r_1h_reference_control.csv",
        e2_dir / "trades_v3r_exec_3m_phasec_control.csv",
        e2_dir / "trades_v4r_exec_3m_phasec_best.csv",
        phase_c_dir / "run_manifest.json",
    ]:
        if fp.exists():
            shutil.copy2(fp, snap / fp.name)

    # Load frozen downstream trade tables.
    v2 = pd.read_csv(e2_dir / "trades_v2r_1h_reference_control.csv")
    v3 = pd.read_csv(e2_dir / "trades_v3r_exec_3m_phasec_control.csv")
    v4 = pd.read_csv(e2_dir / "trades_v4r_exec_3m_phasec_best.csv")
    for df in [v2, v3, v4]:
        df["signal_id"] = df["signal_id"].astype(str)
        for c in ["signal_time", "entry_time", "exit_time"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        for c in ["pnl_net_pct", "mae_pct", "mfe_pct", "risk_pct", "entry_price", "exit_price", "signal_tp_mult", "signal_sl_mult"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    # Enrich representative subset with signal metadata.
    signal_source_csv = _resolve(str(e2_manifest["signal_source_csv"]))
    src = pd.read_csv(signal_source_csv)
    src["signal_id"] = src["signal_id"].astype(str)
    src["signal_time"] = pd.to_datetime(src["signal_time"], utc=True, errors="coerce")
    features = src[
        [
            "signal_id",
            "direction",
            "cycle",
            "baseline_entry_ref",
            "strategy_tp_mult",
            "strategy_sl_mult",
            "stop_distance_pct",
            "atr_1h",
            "atr_percentile_1h",
            "trend_up_1h",
        ]
    ].copy()
    rep = rep_subset.merge(features, on="signal_id", how="left", suffixes=("", "_src"))
    rep["signal_time"] = pd.to_datetime(rep["signal_time"], utc=True, errors="coerce")
    rep["atr_percentile_1h"] = pd.to_numeric(rep["atr_percentile_1h"], errors="coerce")
    rep["trend_up_1h"] = pd.to_numeric(rep["trend_up_1h"], errors="coerce")
    rep["stop_distance_pct"] = pd.to_numeric(rep["stop_distance_pct"], errors="coerce")
    rep["cycle"] = pd.to_numeric(rep["cycle"], errors="coerce")
    rep["direction"] = rep["direction"].fillna("long").astype(str)
    rep["vol_bucket"] = _vol_bucket(rep["atr_percentile_1h"])
    rep["trend_bucket"] = rep["trend_up_1h"].map(lambda v: "up" if _safe_float(v) >= 0.5 else "down").fillna("unknown")
    rep["regime_bucket"] = rep["vol_bucket"].astype(str) + "|" + rep["trend_bucket"].astype(str)
    rep["dow"] = pd.to_datetime(rep["signal_time"], utc=True).dt.day_name().fillna("unknown")
    rep["hour"] = pd.to_datetime(rep["signal_time"], utc=True).dt.hour
    rep["hour_bucket"] = _hour_bucket(rep["hour"])
    rep = rep.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)

    fee = phasec_bt._load_fee_model(fee_model_path)
    init_eq = float(contract["initial_equity"])
    risk_pt = float(contract["risk_per_trade"])

    # Base metrics for fixed variants.
    rep_ids = set(rep["signal_id"].astype(str).tolist())
    m_v2 = _compute_metrics_for_ids(v2, rep_ids, signals_total=len(rep), initial_equity=init_eq, risk_per_trade=risk_pt)
    m_v3 = _compute_metrics_for_ids(v3, rep_ids, signals_total=len(rep), initial_equity=init_eq, risk_per_trade=risk_pt)
    m_v4 = _compute_metrics_for_ids(v4, rep_ids, signals_total=len(rep), initial_equity=init_eq, risk_per_trade=risk_pt)

    # Regime edge breakdown on 1h reference layer.
    regime_rows: List[Dict[str, Any]] = []
    for rg, g in rep.groupby("regime_bucket", dropna=False):
        ids = set(g["signal_id"].astype(str).tolist())
        m = _compute_metrics_for_ids(v2, ids, signals_total=len(g), initial_equity=init_eq, risk_per_trade=risk_pt)
        regime_rows.append(
            {
                "regime_bucket": str(rg),
                "signals_total": int(len(g)),
                "trades_total": int(m.get("trades_total", 0)),
                "expectancy_net": float(m.get("expectancy_net", np.nan)),
                "win_rate": float(m.get("win_rate", np.nan)),
                "profit_factor": float(m.get("profit_factor", np.nan)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(m.get("cvar_5", np.nan)),
            }
        )
    regime_df = pd.DataFrame(regime_rows).sort_values("signals_total", ascending=False).reset_index(drop=True)
    regime_df.to_csv(run_dir / "regime_edge_breakdown.csv", index=False)

    # Signal clustering diagnostics.
    xs = rep.sort_values("signal_time").copy()
    xs["gap_min"] = xs["signal_time"].diff().dt.total_seconds().div(60.0)
    xs["same_dir_prev"] = (xs["direction"] == xs["direction"].shift(1)).astype(int)
    streak = []
    s = 0
    prev = None
    for d in xs["direction"].astype(str):
        if d == prev:
            s += 1
        else:
            s = 1
        streak.append(s)
        prev = d
    xs["direction_streak"] = streak
    xs["adverse_regime"] = ((xs["trend_bucket"] == "down") | (xs["vol_bucket"] == "high")).astype(int)
    adverse_ids = set(xs.loc[xs["adverse_regime"] == 1, "signal_id"].astype(str))
    benign_ids = set(xs.loc[xs["adverse_regime"] == 0, "signal_id"].astype(str))
    m_adv = _compute_metrics_for_ids(v2, adverse_ids, signals_total=max(1, len(adverse_ids)), initial_equity=init_eq, risk_per_trade=risk_pt)
    m_ben = _compute_metrics_for_ids(v2, benign_ids, signals_total=max(1, len(benign_ids)), initial_equity=init_eq, risk_per_trade=risk_pt)
    cluster_rows = pd.DataFrame(
        [
            {"metric": "signals_total", "value": float(len(xs)), "notes": ""},
            {"metric": "median_gap_min", "value": float(pd.to_numeric(xs["gap_min"], errors="coerce").median()), "notes": "inter-signal spacing"},
            {"metric": "p95_gap_min", "value": float(pd.to_numeric(xs["gap_min"], errors="coerce").quantile(0.95)), "notes": "inter-signal spacing"},
            {"metric": "signals_per_day_median", "value": float(xs.groupby(xs["signal_time"].dt.floor("D"))["signal_id"].count().median()), "notes": "burst density"},
            {"metric": "signals_per_day_p95", "value": float(xs.groupby(xs["signal_time"].dt.floor("D"))["signal_id"].count().quantile(0.95)), "notes": "burst density"},
            {"metric": "max_direction_streak", "value": float(pd.to_numeric(xs["direction_streak"], errors="coerce").max()), "notes": "same-direction streak"},
            {"metric": "adverse_regime_share", "value": float(xs["adverse_regime"].mean()), "notes": "share of signals in adverse regime"},
            {"metric": "expectancy_adverse", "value": float(m_adv.get("expectancy_net", np.nan)), "notes": "1h ref"},
            {"metric": "expectancy_non_adverse", "value": float(m_ben.get("expectancy_net", np.nan)), "notes": "1h ref"},
        ]
    )
    cluster_rows.to_csv(run_dir / "signal_clustering_report.csv", index=False)

    # Entry timing sensitivity on 1h layer.
    entries_base = _entry_table_from_v3(v3, split_lookup)
    delay_variants: List[Tuple[str, pd.DataFrame]] = [("current_next_3m_open", entries_base.copy())]
    for h in [1, 2, 3]:
        delay_variants.append((f"next_1h_open_plus_{h-1}bar_delay", _delay_entries_using_1h_open(entries_base, symbol=symbol, offset_hours_after_signal=h)))
    delay_rows: List[Dict[str, Any]] = []
    for name, ent in delay_variants:
        t = e2._simulate_1h_from_entries(
            entries_df=ent,
            symbol=symbol,
            fee=fee,
            exec_horizon_hours=float(args.exec_horizon_hours),
        )
        ids = set(ent["signal_id"].astype(str).tolist())
        m = _compute_metrics_for_ids(t, ids, signals_total=len(rep), initial_equity=init_eq, risk_per_trade=risk_pt)
        delay_rows.append(
            {
                "variant": name,
                "signals_total": int(len(rep)),
                "trades_total": int(m.get("trades_total", 0)),
                "expectancy_net": float(m.get("expectancy_net", np.nan)),
                "total_return": float(m.get("total_return", np.nan)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(m.get("cvar_5", np.nan)),
                "profit_factor": float(m.get("profit_factor", np.nan)),
                "win_rate": float(m.get("win_rate", np.nan)),
            }
        )
    entry_delay_df = pd.DataFrame(delay_rows)
    entry_delay_df.to_csv(run_dir / "entry_delay_sensitivity.csv", index=False)

    # Hold horizon sensitivity on 1h layer (fixed entries).
    horizon_rows: List[Dict[str, Any]] = []
    for h in [6, 12, 18, 24, 36, 48]:
        t = e2._simulate_1h_from_entries(
            entries_df=entries_base,
            symbol=symbol,
            fee=fee,
            exec_horizon_hours=float(h),
        )
        m = _compute_metrics_for_ids(t, rep_ids, signals_total=len(rep), initial_equity=init_eq, risk_per_trade=risk_pt)
        horizon_rows.append(
            {
                "horizon_hours": int(h),
                "signals_total": int(len(rep)),
                "trades_total": int(m.get("trades_total", 0)),
                "expectancy_net": float(m.get("expectancy_net", np.nan)),
                "total_return": float(m.get("total_return", np.nan)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(m.get("cvar_5", np.nan)),
                "profit_factor": float(m.get("profit_factor", np.nan)),
                "win_rate": float(m.get("win_rate", np.nan)),
            }
        )
    horizon_df = pd.DataFrame(horizon_rows)
    horizon_df.to_csv(run_dir / "hold_horizon_sensitivity.csv", index=False)

    # Ablation variants (signal-layer only; no execution logic changes).
    full_df = rep.copy()
    cycle_only_df = rep[rep["cycle"].fillna(0).astype(int) >= 1].copy()
    minf_df = rep[
        (rep["cycle"].fillna(0).astype(int) >= 1)
        & (rep["trend_up_1h"].fillna(0) >= 1)
        & (rep["atr_percentile_1h"].fillna(-1) >= 10)
        & (rep["atr_percentile_1h"].fillna(999) <= 90)
    ].copy()
    minf_cd_df = _apply_cooldown(minf_df, cooldown_hours=4)

    abl_defs = [
        ("cycle_detection_only", cycle_only_df),
        ("cycle_plus_min_filters", minf_df),
        ("cycle_plus_min_filters_cooldown4h", minf_cd_df),
        ("full_signal_params", full_df),
    ]
    abl_rows: List[Dict[str, Any]] = []
    for name, sdf in abl_defs:
        ids = set(sdf["signal_id"].astype(str).tolist())
        m = _compute_metrics_for_ids(v2, ids, signals_total=len(sdf), initial_equity=init_eq, risk_per_trade=risk_pt)
        abl_rows.append(
            {
                "ablation": name,
                "signals_total": int(len(sdf)),
                "signals_share_of_rep": float(len(sdf) / max(1, len(rep))),
                "trades_total": int(m.get("trades_total", 0)),
                "expectancy_net": float(m.get("expectancy_net", np.nan)),
                "total_return": float(m.get("total_return", np.nan)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(m.get("cvar_5", np.nan)),
                "profit_factor": float(m.get("profit_factor", np.nan)),
                "win_rate": float(m.get("win_rate", np.nan)),
            }
        )
    ablation_df = pd.DataFrame(abl_rows)
    ablation_df.to_csv(run_dir / "ablation_results.csv", index=False)

    # Signal-quality report: overall and split-level across fixed variants.
    sq_rows: List[Dict[str, Any]] = []
    for name, m in [("V2R_1H_REFERENCE_CONTROL", m_v2), ("V3R_EXEC_3M_CONTROL", m_v3), ("V4R_EXEC_3M_PHASEC_BEST", m_v4)]:
        sq_rows.append(
            {
                "scope": "overall",
                "variant": name,
                "split_id": -1,
                "signals_total": int(len(rep)),
                "trades_total": int(m.get("trades_total", 0)),
                "expectancy_net": float(m.get("expectancy_net", np.nan)),
                "total_return": float(m.get("total_return", np.nan)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(m.get("cvar_5", np.nan)),
                "profit_factor": float(m.get("profit_factor", np.nan)),
                "win_rate": float(m.get("win_rate", np.nan)),
            }
        )
    sq_split = pd.concat(
        [
            _split_metrics(v2, rep, split_definition, initial_equity=init_eq, risk_per_trade=risk_pt, variant="V2R_1H_REFERENCE_CONTROL"),
            _split_metrics(v3, rep, split_definition, initial_equity=init_eq, risk_per_trade=risk_pt, variant="V3R_EXEC_3M_CONTROL"),
            _split_metrics(v4, rep, split_definition, initial_equity=init_eq, risk_per_trade=risk_pt, variant="V4R_EXEC_3M_PHASEC_BEST"),
        ],
        ignore_index=True,
    )
    sq_split.insert(0, "scope", "split")
    sq = pd.concat([pd.DataFrame(sq_rows), sq_split], ignore_index=True, sort=False)
    sq.to_csv(run_dir / "signal_quality_report.csv", index=False)

    # Tail diagnostics.
    v2t = v2[v2["signal_id"].astype(str).isin(rep_ids)].copy()
    v2t = v2t[(pd.to_numeric(v2t["filled"], errors="coerce") == 1) & (pd.to_numeric(v2t["valid_for_metrics"], errors="coerce") == 1)].copy()
    v2t = v2t.merge(rep[["signal_id", "regime_bucket", "cycle", "vol_bucket", "trend_bucket", "hour_bucket", "dow"]], on="signal_id", how="left")
    v2t["pnl_net_pct"] = pd.to_numeric(v2t["pnl_net_pct"], errors="coerce")
    v2t["risk_pct"] = pd.to_numeric(v2t["risk_pct"], errors="coerce")
    v2t["mfe_pct"] = pd.to_numeric(v2t["mfe_pct"], errors="coerce")
    v2t["mae_pct"] = pd.to_numeric(v2t["mae_pct"], errors="coerce")
    tail_cut = float(v2t["pnl_net_pct"].quantile(0.10)) if not v2t.empty else float("nan")
    v2t_tail = v2t[v2t["pnl_net_pct"] <= tail_cut].copy() if np.isfinite(tail_cut) else v2t.iloc[0:0].copy()
    top_loss_regime = (
        v2t_tail.groupby("regime_bucket", dropna=False)["signal_id"]
        .count()
        .reset_index(name="tail_loss_count")
        .sort_values("tail_loss_count", ascending=False)
        .head(5)
    )
    # R-reach distributions.
    r = pd.to_numeric(v2t["risk_pct"], errors="coerce")
    r = r.where(r > 0)
    r_reach_05 = float(((pd.to_numeric(v2t["mfe_pct"], errors="coerce") / r) >= 0.5).mean()) if not v2t.empty else float("nan")
    r_reach_10 = float(((pd.to_numeric(v2t["mfe_pct"], errors="coerce") / r) >= 1.0).mean()) if not v2t.empty else float("nan")
    r_reach_15 = float(((pd.to_numeric(v2t["mfe_pct"], errors="coerce") / r) >= 1.5).mean()) if not v2t.empty else float("nan")

    # Build diagnostics markdown.
    best_reg = regime_df.sort_values("expectancy_net", ascending=False).head(3)
    worst_reg = regime_df.sort_values("expectancy_net", ascending=True).head(3)
    best_abl = ablation_df.sort_values("expectancy_net", ascending=False).head(1)
    signal_diag_lines = [
        "# Signal Layer Diagnostics",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- E2 standard dir: `{e2_dir}`",
        f"- Contract hash fee/metrics: `{fee_hash}` / `{metrics_hash}`",
        f"- Representative subset size/hash: {len(rep)} / `{rep_hash_calc}`",
        "",
        "## Fixed Variant Headline Metrics",
        "",
        f"- V2R expectancy/return/maxDD/cvar5/PF: {_safe_float(m_v2['expectancy_net']):.6f} / {_safe_float(m_v2['total_return']):.6f} / {_safe_float(m_v2['max_drawdown_pct']):.6f} / {_safe_float(m_v2['cvar_5']):.6f} / {_safe_float(m_v2['profit_factor']):.6f}",
        f"- V3R expectancy/return/maxDD/cvar5/PF: {_safe_float(m_v3['expectancy_net']):.6f} / {_safe_float(m_v3['total_return']):.6f} / {_safe_float(m_v3['max_drawdown_pct']):.6f} / {_safe_float(m_v3['cvar_5']):.6f} / {_safe_float(m_v3['profit_factor']):.6f}",
        f"- V4R expectancy/return/maxDD/cvar5/PF: {_safe_float(m_v4['expectancy_net']):.6f} / {_safe_float(m_v4['total_return']):.6f} / {_safe_float(m_v4['max_drawdown_pct']):.6f} / {_safe_float(m_v4['cvar_5']):.6f} / {_safe_float(m_v4['profit_factor']):.6f}",
        "",
        "## Regime Edge (1h reference)",
        "",
        "Best regimes:",
        _markdown_table(best_reg),
        "",
        "Worst regimes:",
        _markdown_table(worst_reg),
        "",
        "## Signal Clustering",
        "",
        _markdown_table(cluster_rows),
        "",
        "## Entry Delay Sensitivity",
        "",
        _markdown_table(entry_delay_df),
        "",
        "## Hold Horizon Sensitivity",
        "",
        _markdown_table(horizon_df),
        "",
        "## Ablations",
        "",
        _markdown_table(ablation_df),
        "",
        "## Tail / Distribution Diagnostics",
        "",
        f"- Tail cutoff (bottom decile pnl_net_pct): {tail_cut:.6f}",
        f"- R reach >=0.5 / >=1.0 / >=1.5: {r_reach_05:.4f} / {r_reach_10:.4f} / {r_reach_15:.4f}",
        "",
        "Top tail-loss regimes:",
        _markdown_table(top_loss_regime),
    ]
    (run_dir / "signal_layer_diagnostics.md").write_text("\n".join(signal_diag_lines) + "\n", encoding="utf-8")

    # Gates + practical decision.
    no_exec_drift = int(
        _sha256_file(e2_dir / "trades_v3r_exec_3m_phasec_control.csv") == _sha256_file(snap / "trades_v3r_exec_3m_phasec_control.csv")
        and _sha256_file(e2_dir / "trades_v4r_exec_3m_phasec_best.csv") == _sha256_file(snap / "trades_v4r_exec_3m_phasec_best.csv")
    )
    best_abl_row = best_abl.iloc[0].to_dict() if not best_abl.empty else {}
    best_abl_improves = int(
        np.isfinite(_safe_float(best_abl_row.get("expectancy_net")))
        and _safe_float(best_abl_row.get("expectancy_net")) >= (_safe_float(m_v2.get("expectancy_net")) + float(args.min_ablation_expectancy_delta))
    )
    absolute_deployable = int(
        _safe_float(m_v4.get("expectancy_net")) > 0.0
        and _safe_float(m_v4.get("total_return")) > 0.0
        and _safe_float(m_v4.get("max_drawdown_pct")) > float(args.max_dd_deploy_floor)
        and _safe_float(m_v4.get("cvar_5")) > float(args.cvar5_deploy_floor)
    )
    likely_root = "signal_definition_itself"
    if best_abl_improves == 1 and absolute_deployable == 0:
        likely_root = "regime_exposure_and_overtrading"
    if float(cluster_rows.loc[cluster_rows["metric"] == "signals_per_day_median", "value"].iloc[0]) > 10:
        likely_root = "overtrading_signal_clustering"

    gates = {
        "contract_locked": int(fee_hash == EXPECTED_PHASEA_FEE_HASH and metrics_hash == EXPECTED_PHASEA_METRICS_HASH),
        "representative_subset_reused": int(rep_hash_calc == rep_hash_ref),
        "downstream_execution_frozen": int(no_exec_drift),
        "phasec_best_vs_control_nondegrade": int(_safe_float(m_v4.get("expectancy_net")) >= _safe_float(m_v3.get("expectancy_net"))),
        "absolute_deployable": int(absolute_deployable),
        "best_ablation_improves_v2": int(best_abl_improves),
        "likely_root_cause": likely_root,
    }
    _json_dump(run_dir / "pass_fail_gates.json", gates)

    top_fixes = [
        {
            "rank": 1,
            "fix": "Add upstream regime gate to suppress down-trend/high-vol adverse buckets",
            "expected_impact": "Reduce tail losses concentration and improve cvar5/maxDD before expectancy turn.",
            "required_code_changes": "Signal export path: add deterministic pre-entry filter using trend_up_1h + atr_percentile_1h thresholds in 1h signal post-processing.",
        },
        {
            "rank": 2,
            "fix": "Add signal cooldown/de-clustering gate (>=4h) at 1h layer",
            "expected_impact": "Reduce overtrading burst losses and same-direction streak drawdowns.",
            "required_code_changes": "1h signal post-filter: keep earliest signal in cooldown window; preserve deterministic ordering.",
        },
        {
            "rank": 3,
            "fix": "Rework entry timing at signal layer (1h delayed-open candidate set)",
            "expected_impact": "Potentially improve fill context and reduce immediate adverse excursion.",
            "required_code_changes": "1h reference entry policy module: support delayed-open mode toggles (0/1/2 bars) under same contract.",
        },
    ]
    decision_class = "NO_DEPLOY_PAUSE_SOL_SIGNAL_LAYER" if gates["absolute_deployable"] == 0 else "DEPLOY_CANDIDATE"
    practical_lines = [
        "# Practical Decision",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Decision: **{decision_class}**",
        f"- Root cause classification: **{likely_root}**",
        "",
        "## Contract-Locked Summary",
        "",
        f"- contract_locked: {gates['contract_locked']}",
        f"- representative_subset_reused: {gates['representative_subset_reused']}",
        f"- downstream_execution_frozen: {gates['downstream_execution_frozen']}",
        f"- phasec_best_vs_control_nondegrade: {gates['phasec_best_vs_control_nondegrade']}",
        "",
        "## Core Metrics",
        "",
        f"- V2R expectancy/return/maxDD/cvar5/PF/win: {_safe_float(m_v2['expectancy_net']):.6f} / {_safe_float(m_v2['total_return']):.6f} / {_safe_float(m_v2['max_drawdown_pct']):.6f} / {_safe_float(m_v2['cvar_5']):.6f} / {_safe_float(m_v2['profit_factor']):.6f} / {_safe_float(m_v2['win_rate']):.6f}",
        f"- V3R expectancy/return/maxDD/cvar5/PF/win: {_safe_float(m_v3['expectancy_net']):.6f} / {_safe_float(m_v3['total_return']):.6f} / {_safe_float(m_v3['max_drawdown_pct']):.6f} / {_safe_float(m_v3['cvar_5']):.6f} / {_safe_float(m_v3['profit_factor']):.6f} / {_safe_float(m_v3['win_rate']):.6f}",
        f"- V4R expectancy/return/maxDD/cvar5/PF/win: {_safe_float(m_v4['expectancy_net']):.6f} / {_safe_float(m_v4['total_return']):.6f} / {_safe_float(m_v4['max_drawdown_pct']):.6f} / {_safe_float(m_v4['cvar_5']):.6f} / {_safe_float(m_v4['profit_factor']):.6f} / {_safe_float(m_v4['win_rate']):.6f}",
        "",
        "## Top 3 Upstream Fixes",
        "",
    ]
    for fx in top_fixes:
        practical_lines.extend(
            [
                f"{fx['rank']}. {fx['fix']}",
                f"   expected_impact: {fx['expected_impact']}",
                f"   required_code_changes: {fx['required_code_changes']}",
            ]
        )
    (run_dir / "practical_decision.md").write_text("\n".join(practical_lines) + "\n", encoding="utf-8")

    # Run manifest.
    _json_dump(
        run_dir / "run_manifest.json",
        {
            "generated_utc": _utc_now().isoformat(),
            "symbol": symbol,
            "e2_dir": str(e2_dir),
            "phase_c_dir": str(phase_c_dir),
            "phasec_cfg_hash": EXPECTED_PHASEC_CFG_HASH,
            "fee_model_path": str(fee_model_path),
            "fee_model_sha256": fee_hash,
            "metrics_definition_path": str(metrics_def_path),
            "metrics_definition_sha256": metrics_hash,
            "representative_subset_path": str(rep_subset_path),
            "representative_subset_sha256": rep_hash_calc,
            "split_definition": split_definition,
            "contract_id": str(contract.get("contract_id")),
            "exec_horizon_hours": float(args.exec_horizon_hours),
            "initial_equity": init_eq,
            "risk_per_trade": risk_pt,
            "decision_class": decision_class,
            "likely_root_cause": likely_root,
            "gates": gates,
        },
    )

    # Repro + git status.
    repro_lines = [
        "# Repro",
        "",
        "```bash",
        f"cd {PROJECT_ROOT}",
        "python3 scripts/phase_f_sol_signal_rehab.py "
        f"--symbol {symbol} "
        f"--e2-dir {e2_dir} "
        f"--outdir {args.outdir} "
        f"--exec-horizon-hours {float(args.exec_horizon_hours)}",
        "```",
    ]
    (run_dir / "repro.md").write_text("\n".join(repro_lines) + "\n", encoding="utf-8")
    try:
        gs = subprocess.check_output(["git", "status", "--short"], cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.STDOUT)
    except Exception as ex:
        gs = f"git status unavailable: {ex}"
    (run_dir / "git_status.txt").write_text(gs, encoding="utf-8")

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase F SOL signal-layer rehab diagnostics on E2 contract-locked representative harness.")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--e2-dir", default="reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--min-ablation-expectancy-delta", type=float, default=0.00005)
    ap.add_argument("--max-dd-deploy-floor", type=float, default=-0.35)
    ap.add_argument("--cvar5-deploy-floor", type=float, default=-0.0015)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))
    gates = json.loads((out / "pass_fail_gates.json").read_text(encoding="utf-8"))
    print(str(gates.get("likely_root_cause", "unknown")))


if __name__ == "__main__":
    main()
