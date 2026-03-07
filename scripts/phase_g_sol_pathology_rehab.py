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

from scripts import backtest_exec_phasec_sol as phasec_bt  # noqa: E402
from scripts import phase_e2_sol_representative as e2  # noqa: E402


EXPECTED_PHASEA_FEE_HASH = "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a"
EXPECTED_PHASEA_METRICS_HASH = "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99"


@dataclass(frozen=True)
class VariantConfig:
    name: str
    regime_gate: bool
    cooldown_hours: int
    delay_bars: int
    use_uc_params: bool
    use_regime_modifiers: bool


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


def _markdown_table(df: pd.DataFrame, max_rows: int = 100) -> str:
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


def _vol_bucket(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    out = pd.Series(index=y.index, dtype=object)
    out[y <= 33.3333333333] = "low"
    out[(y > 33.3333333333) & (y <= 66.6666666667)] = "mid"
    out[y > 66.6666666667] = "high"
    return out.fillna("unknown").astype(str)


def _gap_bucket_minutes(gap_min: pd.Series) -> pd.Series:
    y = pd.to_numeric(gap_min, errors="coerce")
    out = pd.Series(index=y.index, dtype=object)
    out[y <= 60.0] = "gap_le_1h"
    out[(y > 60.0) & (y <= 240.0)] = "gap_1h_4h"
    out[(y > 240.0) & (y <= 720.0)] = "gap_4h_12h"
    out[y > 720.0] = "gap_gt_12h"
    return out.fillna("gap_unknown").astype(str)


def _hash_rep_subset(df: pd.DataFrame) -> str:
    x = df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rows = [f"{str(r.signal_id)}|{pd.to_datetime(r.signal_time, utc=True).isoformat()}" for r in x.itertuples(index=False)]
    return _sha256_text("\n".join(rows))


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


def _delay_entries_using_1h_open(entries: pd.DataFrame, symbol: str, offset_hours_after_signal: int) -> pd.DataFrame:
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


def _compute_metrics(
    trades: pd.DataFrame,
    *,
    signals_total: int,
    initial_equity: float,
    risk_per_trade: float,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    return phasec_bt._compute_equity_curve(
        trades,
        signals_total=int(signals_total),
        initial_equity=float(initial_equity),
        risk_per_trade=float(risk_per_trade),
    )


def _compute_fixed_size_equity_curve(
    trades: pd.DataFrame,
    *,
    signals_total: int,
    initial_equity: float,
    risk_per_trade: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    x = trades.copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    for c in ["signal_time", "entry_time", "exit_time"]:
        x[c] = pd.to_datetime(x.get(c), utc=True, errors="coerce")
    for c in ["filled", "valid_for_metrics", "pnl_net_pct", "risk_pct"]:
        x[c] = pd.to_numeric(x.get(c, np.nan), errors="coerce")

    valid = x[(x["filled"] == 1) & (x["valid_for_metrics"] == 1) & x["pnl_net_pct"].notna()].copy()
    valid = valid.sort_values(["entry_time", "signal_time"]).reset_index(drop=True)

    eq = float(initial_equity)
    rows: List[Dict[str, Any]] = []
    for r in valid.itertuples(index=False):
        risk_pct = max(1e-8, _safe_float(getattr(r, "risk_pct", 0.001)))
        pos_notional = float(initial_equity) * float(risk_per_trade) / risk_pct
        pnl = _safe_float(getattr(r, "pnl_net_pct"))
        trade_pnl_abs = pos_notional * pnl
        eq += trade_pnl_abs
        ts = pd.to_datetime(getattr(r, "exit_time"), utc=True, errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(getattr(r, "signal_time"), utc=True, errors="coerce")
        rows.append(
            {
                "timestamp": ts,
                "signal_id": str(getattr(r, "signal_id")),
                "equity_fixed": float(eq),
                "trade_pnl_abs_fixed": float(trade_pnl_abs),
                "trade_pnl_pct": float(pnl),
            }
        )

    if not rows:
        eq_df = pd.DataFrame(
            [{"timestamp": pd.NaT, "signal_id": "", "equity_fixed": float(initial_equity), "trade_pnl_abs_fixed": 0.0, "trade_pnl_pct": 0.0}]
        )
    else:
        eq_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    peak = eq_df["equity_fixed"].cummax()
    eq_df["drawdown_pct_fixed"] = eq_df["equity_fixed"] / np.maximum(1e-12, peak) - 1.0

    ret = float(eq_df["equity_fixed"].iloc[-1] / max(1e-12, float(initial_equity)) - 1.0)
    metrics = {
        "signals_total": int(signals_total),
        "trades_total": int(len(valid)),
        "total_return_fixed": float(ret),
        "max_drawdown_pct_fixed": float(eq_df["drawdown_pct_fixed"].min()),
        "equity_end_fixed": float(eq_df["equity_fixed"].iloc[-1]),
    }
    return eq_df, metrics


def _extract_cycle_vector(params: Dict[str, Any], base_key: str) -> List[float]:
    k = f"{base_key}_by_cycle"
    if k in params and isinstance(params[k], list):
        vals = [float(v) for v in params[k]]
        if len(vals) >= 5:
            return vals[:5]
    vals: List[float] = []
    ok = True
    for i in range(5):
        k2 = f"{base_key}_cycle{i}"
        if k2 not in params:
            ok = False
            break
        vals.append(float(params[k2]))
    if ok and len(vals) == 5:
        return vals
    raise RuntimeError(f"Missing {base_key} cycle vector in params keys={list(params.keys())}")


def _load_universe_pass_priors(summary_json: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    obj = json.loads(summary_json.read_text(encoding="utf-8"))
    rows = [r for r in obj.get("rows", []) if str(r.get("side", "")).lower() == "long" and str(r.get("PASS/FAIL", "")).upper() == "PASS"]
    if not rows:
        raise RuntimeError("No passed long-side rows found in universe summary.")

    model_rows: List[Dict[str, Any]] = []
    for r in rows:
        sym = str(r["symbol"]).upper()
        ppath = Path(str(r.get("param_path", "")))
        if not ppath.exists():
            raise FileNotFoundError(f"Missing param file for {sym}: {ppath}")
        pobj = json.loads(ppath.read_text(encoding="utf-8"))
        params = pobj.get("params", {})
        tp = _extract_cycle_vector(params, "tp_mult")
        sl = _extract_cycle_vector(params, "sl_mult")
        model_rows.append(
            {
                "symbol": sym,
                "param_path": str(ppath),
                "param_sha256": _sha256_file(ppath),
                "tp_vec": tp,
                "sl_vec": sl,
                "risk_per_trade": _safe_float(params.get("risk_per_trade")),
                "max_allocation": _safe_float(params.get("max_allocation")),
            }
        )

    df = pd.DataFrame(model_rows).sort_values("symbol").reset_index(drop=True)
    tp_mat = np.vstack(df["tp_vec"].to_list())
    sl_mat = np.vstack(df["sl_vec"].to_list())
    prior = {
        "symbols": df["symbol"].tolist(),
        "tp_prior_median": np.median(tp_mat, axis=0).tolist(),
        "sl_prior_median": np.median(sl_mat, axis=0).tolist(),
        "tp_prior_mean": np.mean(tp_mat, axis=0).tolist(),
        "sl_prior_mean": np.mean(sl_mat, axis=0).tolist(),
        "risk_per_trade_median": float(np.nanmedian(pd.to_numeric(df["risk_per_trade"], errors="coerce"))),
        "max_allocation_median": float(np.nanmedian(pd.to_numeric(df["max_allocation"], errors="coerce"))),
    }

    model_hash_lines = [f"{r.symbol}|{r.param_path}|{r.param_sha256}" for r in df.itertuples(index=False)]
    prior["selected_model_set_sha256"] = _sha256_text("\n".join(model_hash_lines))
    return df, prior


def _build_uc_vectors(
    *,
    prior_tp: Sequence[float],
    prior_sl: Sequence[float],
    sol_tp: Sequence[float],
    sol_sl: Sequence[float],
    offset_scale: float,
    tp_offset_cap: float,
    sl_offset_cap: float,
) -> Dict[str, Any]:
    p_tp = np.array(prior_tp, dtype=float)
    p_sl = np.array(prior_sl, dtype=float)
    s_tp = np.array(sol_tp, dtype=float)
    s_sl = np.array(sol_sl, dtype=float)

    raw_tp_off = s_tp - p_tp
    raw_sl_off = s_sl - p_sl

    scaled_tp_off = raw_tp_off * float(offset_scale)
    scaled_sl_off = raw_sl_off * float(offset_scale)

    clip_tp_off = np.clip(scaled_tp_off, -abs(float(tp_offset_cap)), abs(float(tp_offset_cap)))
    clip_sl_off = np.clip(scaled_sl_off, -abs(float(sl_offset_cap)), abs(float(sl_offset_cap)))

    uc_tp = p_tp + clip_tp_off
    uc_sl = p_sl + clip_sl_off

    return {
        "offset_scale": float(offset_scale),
        "tp_vector": uc_tp.tolist(),
        "sl_vector": uc_sl.tolist(),
        "tp_offset_raw": raw_tp_off.tolist(),
        "sl_offset_raw": raw_sl_off.tolist(),
        "tp_offset_applied": clip_tp_off.tolist(),
        "sl_offset_applied": clip_sl_off.tolist(),
        "tp_offset_cap": float(tp_offset_cap),
        "sl_offset_cap": float(sl_offset_cap),
    }


def _apply_uc_params(entries: pd.DataFrame, feats: pd.DataFrame, uc_cfg: Dict[str, Any], *, use_regime_modifiers: bool, regime_tp_delta: float, regime_sl_delta: float) -> pd.DataFrame:
    x = entries.copy()
    needed = ["cycle", "atr_percentile_1h", "trend_up_1h"]
    missing = [c for c in needed if c not in x.columns]
    if missing:
        x = x.merge(feats[["signal_id", *missing]], on="signal_id", how="left")
    tp_vec = [float(v) for v in uc_cfg["tp_vector"]]
    sl_vec = [float(v) for v in uc_cfg["sl_vector"]]

    def _pick_cycle(v: Any) -> int:
        c = int(_safe_float(v))
        if c < 0:
            c = 0
        if c > 4:
            c = 4
        return c

    cyc = x["cycle"].map(_pick_cycle)
    x["signal_tp_mult"] = [tp_vec[c] for c in cyc]
    x["signal_sl_mult"] = [sl_vec[c] for c in cyc]

    if use_regime_modifiers:
        vol = pd.to_numeric(x["atr_percentile_1h"], errors="coerce")
        trend = pd.to_numeric(x["trend_up_1h"], errors="coerce")
        adverse = ((trend < 0.5) | (vol >= 66.6666666667)).fillna(False)
        favorable = ((trend >= 0.5) & (vol <= 33.3333333333)).fillna(False)

        tp = pd.to_numeric(x["signal_tp_mult"], errors="coerce")
        sl = pd.to_numeric(x["signal_sl_mult"], errors="coerce")

        tp = tp.where(~adverse, tp - abs(float(regime_tp_delta)))
        sl = sl.where(~adverse, sl + abs(float(regime_sl_delta)))

        tp = tp.where(~favorable, tp + abs(float(regime_tp_delta)) * 0.5)
        sl = sl.where(~favorable, sl - abs(float(regime_sl_delta)) * 0.5)

        x["signal_tp_mult"] = tp.clip(lower=1.001, upper=1.40)
        x["signal_sl_mult"] = sl.clip(lower=0.80, upper=0.999)

    return x.drop(columns=[c for c in ["cycle", "atr_percentile_1h", "trend_up_1h"] if c in x.columns], errors="ignore")


def _regime_gate_filter(df: pd.DataFrame, *, vol_min: float, vol_max: float, require_trend_up: bool) -> pd.DataFrame:
    x = df.copy()
    atrp = pd.to_numeric(x["atr_percentile_1h"], errors="coerce")
    trend = pd.to_numeric(x["trend_up_1h"], errors="coerce")

    m = atrp.between(float(vol_min), float(vol_max), inclusive="both")
    if require_trend_up:
        m = m & (trend >= 0.5)
    return x[m.fillna(False)].copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def _compute_split_metrics_single(
    trades: pd.DataFrame,
    subset_df: pd.DataFrame,
    splits: Sequence[Dict[str, int]],
    *,
    initial_equity: float,
    risk_per_trade: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ss = subset_df.reset_index(drop=True)
    for sp in splits:
        sid = int(sp["split_id"])
        lo, hi = int(sp["test_start"]), int(sp["test_end"])
        s = ss.iloc[lo:hi].copy()
        ids = set(s["signal_id"].astype(str).tolist())
        t = trades[trades["signal_id"].astype(str).isin(ids)].copy()
        _, m, _ = _compute_metrics(t, signals_total=len(s), initial_equity=initial_equity, risk_per_trade=risk_per_trade)
        rows.append(
            {
                "split_id": sid,
                "signals_total": int(len(s)),
                "trades_total": int(m.get("trades_total", 0)),
                "expectancy_net": float(m.get("expectancy_net", np.nan)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(m.get("cvar_5", np.nan)),
                "profit_factor": float(m.get("profit_factor", np.nan)),
                "win_rate": float(m.get("win_rate", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def _loss_clusters(trades: pd.DataFrame) -> pd.DataFrame:
    x = trades.copy().sort_values(["entry_time", "signal_time"]).reset_index(drop=True)
    x["pnl_net_pct"] = pd.to_numeric(x["pnl_net_pct"], errors="coerce")
    x["entry_time"] = pd.to_datetime(x["entry_time"], utc=True, errors="coerce")
    x["exit_time"] = pd.to_datetime(x["exit_time"], utc=True, errors="coerce")

    # Concurrent open positions at each entry.
    open_ends: List[pd.Timestamp] = []
    conc: List[int] = []
    for r in x.itertuples(index=False):
        et = pd.to_datetime(getattr(r, "entry_time"), utc=True, errors="coerce")
        xt = pd.to_datetime(getattr(r, "exit_time"), utc=True, errors="coerce")
        open_ends = [t for t in open_ends if pd.notna(et) and pd.notna(t) and t > et]
        open_ends.append(xt if pd.notna(xt) else et)
        conc.append(len(open_ends))
    x["concurrent_open_positions"] = conc

    x["is_loss"] = (x["pnl_net_pct"] < 0).astype(int)
    x["gap_min"] = x["entry_time"].diff().dt.total_seconds().div(60.0)

    rows: List[Dict[str, Any]] = []
    cid = 0
    i = 0
    while i < len(x):
        if int(x.at[i, "is_loss"]) != 1:
            i += 1
            continue
        j = i
        while j < len(x) and int(x.at[j, "is_loss"]) == 1:
            j += 1
        seg = x.iloc[i:j].copy()
        cid += 1
        rows.append(
            {
                "cluster_id": int(cid),
                "start_time": pd.to_datetime(seg["entry_time"].min(), utc=True),
                "end_time": pd.to_datetime(seg["entry_time"].max(), utc=True),
                "loss_count": int(len(seg)),
                "pnl_net_pct_sum": float(pd.to_numeric(seg["pnl_net_pct"], errors="coerce").sum()),
                "pnl_net_pct_min": float(pd.to_numeric(seg["pnl_net_pct"], errors="coerce").min()),
                "median_gap_min": float(pd.to_numeric(seg["gap_min"], errors="coerce").median()),
                "max_concurrent_open_positions": int(pd.to_numeric(seg["concurrent_open_positions"], errors="coerce").max()),
            }
        )
        i = j
    return pd.DataFrame(rows)


def _drawdown_threshold_events(eq_df: pd.DataFrame, trades_enriched: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    if eq_df.empty:
        return pd.DataFrame(columns=["dd_threshold", "first_breach_time", "dominant_regime_bucket", "dominant_cycle", "trend_down_share", "high_vol_share", "signals_after_threshold"])

    x = eq_df.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    x["drawdown_pct"] = pd.to_numeric(x.get("drawdown_pct"), errors="coerce")

    trows: List[Dict[str, Any]] = []
    te = trades_enriched.copy()
    te["signal_time"] = pd.to_datetime(te["signal_time"], utc=True, errors="coerce")

    for th in thresholds:
        breach = x[x["drawdown_pct"] <= -abs(float(th))].copy()
        if breach.empty:
            trows.append(
                {
                    "dd_threshold": float(th),
                    "first_breach_time": pd.NaT,
                    "dominant_regime_bucket": "none",
                    "dominant_cycle": "none",
                    "trend_down_share": np.nan,
                    "high_vol_share": np.nan,
                    "signals_after_threshold": 0,
                }
            )
            continue
        t0 = pd.to_datetime(breach["timestamp"].iloc[0], utc=True)
        aft = te[te["signal_time"] >= t0].copy()
        dom_regime = "unknown"
        dom_cycle = "unknown"
        if not aft.empty:
            dom_regime = str(aft["regime_bucket"].astype(str).value_counts().idxmax())
            dom_cycle = str(int(pd.to_numeric(aft["cycle"], errors="coerce").fillna(-1).astype(int).value_counts().idxmax()))
        trend_down_share = float((pd.to_numeric(aft["trend_up_1h"], errors="coerce") < 0.5).mean()) if not aft.empty else np.nan
        high_vol_share = float((pd.to_numeric(aft["atr_percentile_1h"], errors="coerce") >= 66.6666666667).mean()) if not aft.empty else np.nan
        trows.append(
            {
                "dd_threshold": float(th),
                "first_breach_time": t0,
                "dominant_regime_bucket": dom_regime,
                "dominant_cycle": dom_cycle,
                "trend_down_share": trend_down_share,
                "high_vol_share": high_vol_share,
                "signals_after_threshold": int(len(aft)),
            }
        )
    return pd.DataFrame(trows)


def _classify_variant_reason(
    *,
    metrics_comp: Dict[str, Any],
    metrics_fixed: Dict[str, Any],
    adverse_loss_share: float,
    baseline_expectancy: float,
) -> str:
    exp = _safe_float(metrics_comp.get("expectancy_net"))
    dd = _safe_float(metrics_comp.get("max_drawdown_pct"))
    ret = _safe_float(metrics_comp.get("total_return"))
    fixed_ret = _safe_float(metrics_fixed.get("total_return_fixed"))

    if np.isfinite(ret) and np.isfinite(fixed_ret):
        if (ret <= -0.95) and (fixed_ret > -0.60):
            return "sizing/compounding amplification issue"
    if np.isfinite(adverse_loss_share) and adverse_loss_share >= 0.60:
        return "regime concentration issue"
    if np.isfinite(exp) and np.isfinite(dd) and (exp < 0.0) and (dd <= -0.90):
        return "signal quality issue"
    if np.isfinite(exp) and np.isfinite(baseline_expectancy) and exp >= baseline_expectancy:
        return "interaction issue"
    return "signal quality issue"


def run(args: argparse.Namespace) -> Path:
    symbol = str(args.symbol).strip().upper()
    if symbol != "SOLUSDT":
        raise RuntimeError("Phase G workflow is hard-scoped to SOLUSDT.")

    e2_dir = _resolve(args.e2_dir)
    if not e2_dir.exists():
        raise FileNotFoundError(f"Missing E2 dir: {e2_dir}")
    out_root = _resolve(args.outdir)
    run_dir = out_root / f"PHASEG_SOL_PATHOLOGY_REHAB_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snap = run_dir / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    # Contract lock context.
    e2_manifest = json.loads((e2_dir / "run_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((e2_dir / "accounting_contract.json").read_text(encoding="utf-8"))
    phase_c_dir = _resolve(str(e2_manifest.get("phase_c_dir")))
    phase_a_dir = _resolve(str(e2_manifest.get("phase_a_contract_dir", "reports/execution_layer/BASELINE_AUDIT_20260221_214310")))

    if str(e2_manifest.get("symbol", "")).upper() != symbol:
        raise RuntimeError(f"E2 manifest symbol mismatch: {e2_manifest.get('symbol')}")
    if str(contract.get("symbol", "")).upper() != symbol:
        raise RuntimeError(f"E2 contract symbol mismatch: {contract.get('symbol')}")

    fee_model_path = _resolve(str(contract["fee_model_path"]))
    metrics_def_path = _resolve(str(contract["metrics_definition_path"]))
    fee_hash = _sha256_file(fee_model_path)
    metrics_hash = _sha256_file(metrics_def_path)
    if fee_hash != EXPECTED_PHASEA_FEE_HASH:
        raise RuntimeError(f"Fee model hash mismatch: {fee_hash}")
    if metrics_hash != EXPECTED_PHASEA_METRICS_HASH:
        raise RuntimeError(f"Metrics definition hash mismatch: {metrics_hash}")

    rep_subset_path = e2_dir / "representative_subset_signals.csv"
    rep_subset_hash_ref = (e2_dir / "representative_subset_hash.txt").read_text(encoding="utf-8").strip()
    rep_subset = pd.read_csv(rep_subset_path)
    rep_subset["signal_id"] = rep_subset["signal_id"].astype(str)
    rep_subset["signal_time"] = pd.to_datetime(rep_subset["signal_time"], utc=True, errors="coerce")
    rep_subset = rep_subset.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rep_subset_hash_calc = _hash_rep_subset(rep_subset[["signal_id", "signal_time"]].copy())
    if rep_subset_hash_calc != rep_subset_hash_ref:
        raise RuntimeError("Representative subset hash mismatch.")
    if rep_subset_hash_calc != str(e2_manifest.get("representative_subset_sha256", "")):
        raise RuntimeError("Representative subset hash mismatch vs E2 manifest.")

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
    rep_feat["vol_bucket"] = _vol_bucket(rep_feat["atr_percentile_1h"])
    rep_feat["trend_bucket"] = rep_feat["trend_up_1h"].map(lambda v: "up" if _safe_float(v) >= 0.5 else "down").fillna("unknown")
    rep_feat["regime_bucket"] = rep_feat["vol_bucket"].astype(str) + "|" + rep_feat["trend_bucket"].astype(str)

    # Copy snapshots.
    for fp in [
        fee_model_path,
        metrics_def_path,
        rep_subset_path,
        e2_dir / "run_manifest.json",
        e2_dir / "accounting_contract.json",
        e2_dir / "pass_fail_gates.json",
        e2_dir / "trades_v2r_1h_reference_control.csv",
        e2_dir / "trades_v3r_exec_3m_phasec_control.csv",
        e2_dir / "trades_v4r_exec_3m_phasec_best.csv",
        signal_source_path,
        phase_c_dir / "run_manifest.json",
    ]:
        if fp.exists():
            shutil.copy2(fp, snap / fp.name)

    # Load frozen trade tables.
    t_v2 = pd.read_csv(e2_dir / "trades_v2r_1h_reference_control.csv")
    t_v3 = pd.read_csv(e2_dir / "trades_v3r_exec_3m_phasec_control.csv")
    t_v4 = pd.read_csv(e2_dir / "trades_v4r_exec_3m_phasec_best.csv")
    for df in [t_v2, t_v3, t_v4]:
        df["signal_id"] = df["signal_id"].astype(str)
        for c in ["signal_time", "entry_time", "exit_time"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        for c in ["pnl_net_pct", "risk_pct", "mae_pct", "mfe_pct", "signal_tp_mult", "signal_sl_mult", "entry_price", "exit_price"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    init_eq = float(contract.get("initial_equity", args.initial_equity))
    risk_pt = float(contract.get("risk_per_trade", args.risk_per_trade))
    fee = phasec_bt._load_fee_model(fee_model_path)

    # Build baseline entries from frozen V3 timestamps/prices to ensure contract-locked entry semantics.
    entries_base = _entry_table_from_v3(t_v3, split_lookup)
    entries_base = entries_base.merge(rep_feat[["signal_id", "signal_time", "cycle", "atr_percentile_1h", "trend_up_1h"]], on=["signal_id", "signal_time"], how="left")

    # --------------------
    # G0 Drawdown Forensics
    # --------------------
    # Use V4 as forensic baseline: downstream best yet still catastrophic.
    forensic_trades = t_v4.copy()
    forensic_trades = forensic_trades[forensic_trades["signal_id"].astype(str).isin(set(rep_feat["signal_id"].astype(str)))].copy()
    forensic_trades = forensic_trades.merge(rep_feat[["signal_id", "signal_time", "cycle", "atr_percentile_1h", "trend_up_1h", "regime_bucket", "vol_bucket", "trend_bucket"]], on=["signal_id", "signal_time"], how="left")
    forensic_trades = forensic_trades.sort_values(["entry_time", "signal_time"]).reset_index(drop=True)

    eq_comp, m_comp, monthly_comp = _compute_metrics(
        forensic_trades,
        signals_total=len(rep_feat),
        initial_equity=init_eq,
        risk_per_trade=risk_pt,
    )
    eq_fix, m_fix = _compute_fixed_size_equity_curve(
        forensic_trades,
        signals_total=len(rep_feat),
        initial_equity=init_eq,
        risk_per_trade=risk_pt,
    )

    ft = forensic_trades.copy().sort_values(["entry_time", "signal_time"]).reset_index(drop=True)
    ft["month"] = pd.to_datetime(ft["signal_time"], utc=True).dt.to_period("M").astype(str)
    ft["quarter"] = pd.to_datetime(ft["signal_time"], utc=True).dt.to_period("Q").astype(str)
    ft["signal_gap_min"] = pd.to_datetime(ft["signal_time"], utc=True).diff().dt.total_seconds().div(60.0)
    ft["cooldown_bucket"] = _gap_bucket_minutes(ft["signal_gap_min"])

    # Attribution buckets.
    bucket_rows: List[Dict[str, Any]] = []

    def _add_bucket(df: pd.DataFrame, btype: str, col: str) -> None:
        if df.empty:
            return
        for k, g in df.groupby(col, dropna=False):
            vals = pd.to_numeric(g["pnl_net_pct"], errors="coerce")
            losses = (vals < 0).sum()
            bucket_rows.append(
                {
                    "bucket_type": btype,
                    "bucket": str(k),
                    "signals_total": int(g["signal_id"].nunique()),
                    "trades_total": int(len(g)),
                    "pnl_net_sum": float(vals.sum()),
                    "expectancy_net": float(vals.mean()),
                    "loss_rate": float(losses / max(1, len(g))),
                    "sl_hit_rate": float(pd.to_numeric(g["sl_hit"], errors="coerce").fillna(0).mean()),
                    "tp_hit_rate": float(pd.to_numeric(g["tp_hit"], errors="coerce").fillna(0).mean()),
                }
            )

    _add_bucket(ft, "month", "month")
    _add_bucket(ft, "quarter", "quarter")
    _add_bucket(ft, "vol_bucket", "vol_bucket")
    _add_bucket(ft, "trend_bucket", "trend_bucket")
    _add_bucket(ft, "regime_bucket", "regime_bucket")
    _add_bucket(ft, "cooldown_bucket", "cooldown_bucket")

    # Entry timing mode attribution (if available): baseline + delayed modes under same contract.
    delay_rows: List[Dict[str, Any]] = []
    for name, off_h in [
        ("next_3m_open_baseline", 0),
        ("next_1h_open_plus_0bar", 1),
        ("next_1h_open_plus_1bar", 2),
        ("next_1h_open_plus_2bar", 3),
    ]:
        if off_h == 0:
            ent = entries_base[["signal_id", "signal_time", "split_id", "signal_tp_mult", "signal_sl_mult", "entry_time", "entry_price"]].copy()
        else:
            ent = _delay_entries_using_1h_open(
                entries_base[["signal_id", "signal_time", "split_id", "signal_tp_mult", "signal_sl_mult", "entry_time", "entry_price"]].copy(),
                symbol=symbol,
                offset_hours_after_signal=off_h,
            )
        t = e2._simulate_1h_from_entries(entries_df=ent, symbol=symbol, fee=fee, exec_horizon_hours=float(args.exec_horizon_hours))
        _, mm, _ = _compute_metrics(t, signals_total=len(ent), initial_equity=init_eq, risk_per_trade=risk_pt)
        delay_rows.append(
            {
                "mode": name,
                "signals_total": int(len(ent)),
                "trades_total": int(mm.get("trades_total", 0)),
                "expectancy_net": float(mm.get("expectancy_net", np.nan)),
                "total_return": float(mm.get("total_return", np.nan)),
                "max_drawdown_pct": float(mm.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(mm.get("cvar_5", np.nan)),
            }
        )
        bucket_rows.append(
            {
                "bucket_type": "entry_timing_mode",
                "bucket": name,
                "signals_total": int(len(ent)),
                "trades_total": int(mm.get("trades_total", 0)),
                "pnl_net_sum": float(mm.get("expectancy_net", np.nan) * max(1, int(mm.get("trades_total", 0)))),
                "expectancy_net": float(mm.get("expectancy_net", np.nan)),
                "loss_rate": np.nan,
                "sl_hit_rate": float(mm.get("sl_hit_rate", np.nan)),
                "tp_hit_rate": float(mm.get("tp_hit_rate", np.nan)),
            }
        )
    delay_df = pd.DataFrame(delay_rows)

    # Loss clusters + overlapping risk.
    clusters_df = _loss_clusters(ft)
    clusters_df.to_csv(run_dir / "phaseG0_sol_loss_clusters.csv", index=False)

    # Stop/exit attribution for losing trades.
    lose = ft[pd.to_numeric(ft["pnl_net_pct"], errors="coerce") < 0].copy()
    stop_attr = (
        lose.groupby("exit_reason", dropna=False)["signal_id"].count().reset_index(name="loss_count").sort_values("loss_count", ascending=False)
    )
    if not stop_attr.empty:
        stop_attr["loss_share"] = stop_attr["loss_count"] / max(1, int(stop_attr["loss_count"].sum()))

    # Drawdown thresholds and post-threshold dominant conditions.
    dd_thresh_df = _drawdown_threshold_events(eq_comp, ft, thresholds=[0.20, 0.40, 0.60, 0.80])
    for r in dd_thresh_df.itertuples(index=False):
        bucket_rows.append(
            {
                "bucket_type": "dd_threshold_post",
                "bucket": f"dd_{int(float(r.dd_threshold)*100)}pct",
                "signals_total": int(getattr(r, "signals_after_threshold", 0)),
                "trades_total": int(getattr(r, "signals_after_threshold", 0)),
                "pnl_net_sum": np.nan,
                "expectancy_net": np.nan,
                "loss_rate": np.nan,
                "sl_hit_rate": np.nan,
                "tp_hit_rate": np.nan,
            }
        )

    buckets_df = pd.DataFrame(bucket_rows)
    buckets_df.to_csv(run_dir / "phaseG0_sol_drawdown_buckets.csv", index=False)

    # Collapse start: first DD breach at 20% (or earliest non-zero DD if never breaches).
    collapse_start = pd.NaT
    if not eq_comp.empty and "drawdown_pct" in eq_comp.columns:
        c = eq_comp[pd.to_numeric(eq_comp["drawdown_pct"], errors="coerce") <= -0.20]
        if not c.empty:
            collapse_start = pd.to_datetime(c["timestamp"].iloc[0], utc=True)
        else:
            c2 = eq_comp[pd.to_numeric(eq_comp["drawdown_pct"], errors="coerce") < 0.0]
            if not c2.empty:
                collapse_start = pd.to_datetime(c2["timestamp"].iloc[0], utc=True)

    # Root-cause ranking scores.
    adverse_loss_share = float(
        (
            (pd.to_numeric(lose["trend_up_1h"], errors="coerce") < 0.5)
            | (pd.to_numeric(lose["atr_percentile_1h"], errors="coerce") >= 66.6666666667)
        ).mean()
    ) if not lose.empty else float("nan")
    sl_loss_share = float((lose["exit_reason"].astype(str) == "sl").mean()) if not lose.empty else float("nan")
    amp_ratio = np.nan
    if np.isfinite(_safe_float(m_comp.get("total_return"))) and np.isfinite(_safe_float(m_fix.get("total_return_fixed"))):
        comp_mag = abs(_safe_float(m_comp.get("total_return")))
        fix_mag = abs(_safe_float(m_fix.get("total_return_fixed")))
        amp_ratio = float(comp_mag / max(1e-9, fix_mag))

    root_rank = pd.DataFrame(
        [
            {
                "rank": 1,
                "root_cause": "signal quality issue",
                "evidence": f"expectancy_net={_safe_float(m_comp.get('expectancy_net')):.6f}, sl_loss_share={sl_loss_share:.4f}, win_rate={_safe_float(m_comp.get('win_rate')):.4f}",
            },
            {
                "rank": 2,
                "root_cause": "regime concentration issue",
                "evidence": f"adverse_loss_share={adverse_loss_share:.4f}, dominant_worst_regime={str(stop_attr['exit_reason'].iloc[0]) if not stop_attr.empty else 'n/a'}",
            },
            {
                "rank": 3,
                "root_cause": "sizing/compounding amplification issue",
                "evidence": f"comp_total_return={_safe_float(m_comp.get('total_return')):.6f}, fixed_total_return={_safe_float(m_fix.get('total_return_fixed')):.6f}, amplification_ratio={amp_ratio:.4f}",
            },
        ]
    )

    g0_lines = [
        "# Phase G0 SOL Drawdown Forensics",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Baseline forensic variant: V4R_EXEC_3M_PHASEC_BEST (from {e2_dir.name})",
        f"- Representative subset hash: `{rep_subset_hash_calc}`",
        f"- Contract lock hashes fee/metrics: `{fee_hash}` / `{metrics_hash}`",
        "",
        "## Equity Path Decomposition",
        "",
        f"- collapse_start_time_utc: {collapse_start}",
        f"- compounding_total_return: {_safe_float(m_comp.get('total_return')):.6f}",
        f"- compounding_max_drawdown_pct: {_safe_float(m_comp.get('max_drawdown_pct')):.6f}",
        f"- fixed_size_total_return: {_safe_float(m_fix.get('total_return_fixed')):.6f}",
        f"- fixed_size_max_drawdown_pct: {_safe_float(m_fix.get('max_drawdown_pct_fixed')):.6f}",
        "",
        "## Fatal Drawdown Thresholds",
        "",
        _markdown_table(dd_thresh_df),
        "",
        "## Root-Cause Ranking",
        "",
        _markdown_table(root_rank),
        "",
        "## PnL Attribution Highlights",
        "",
        "Monthly (worst 5):",
        _markdown_table(
            buckets_df[buckets_df["bucket_type"] == "month"].sort_values("pnl_net_sum", ascending=True).head(5)
        ),
        "",
        "Regime buckets (worst 5):",
        _markdown_table(
            buckets_df[buckets_df["bucket_type"] == "regime_bucket"].sort_values("pnl_net_sum", ascending=True).head(5)
        ),
        "",
        "Cooldown buckets:",
        _markdown_table(
            buckets_df[buckets_df["bucket_type"] == "cooldown_bucket"].sort_values("bucket")
        ),
        "",
        "Entry timing sensitivity:",
        _markdown_table(delay_df),
        "",
        "## Loss Cluster Analysis",
        "",
        _markdown_table(clusters_df.head(20)),
        "",
        "## Stop/Exit Attribution (Losses)",
        "",
        _markdown_table(stop_attr),
    ]
    (run_dir / "phaseG0_sol_drawdown_forensics.md").write_text("\n".join(g0_lines) + "\n", encoding="utf-8")

    # --------------------
    # G1 Universe-Conditioned Parameterization Design
    # --------------------
    universe_summary = _resolve(args.universe_summary_json)
    prior_df, prior = _load_universe_pass_priors(universe_summary)

    # SOL anchor vector from selected SOL parameter file (fallback to source median by cycle if needed).
    sol_param_path = _resolve(args.sol_param_path)
    sol_param_obj = json.loads(sol_param_path.read_text(encoding="utf-8"))
    sol_params = sol_param_obj.get("params", {})
    sol_tp = _extract_cycle_vector(sol_params, "tp_mult")
    sol_sl = _extract_cycle_vector(sol_params, "sl_mult")

    # G1 candidate scan over bounded offset scales with fail-fast.
    offset_scales = [float(x.strip()) for x in str(args.uc_offset_scales).split(",") if str(x).strip()]
    g1_rows: List[Dict[str, Any]] = []
    fatal_streak = 0
    chosen_uc: Optional[Dict[str, Any]] = None

    for sc in offset_scales:
        uc_cfg = _build_uc_vectors(
            prior_tp=prior["tp_prior_median"],
            prior_sl=prior["sl_prior_median"],
            sol_tp=sol_tp,
            sol_sl=sol_sl,
            offset_scale=float(sc),
            tp_offset_cap=float(args.tp_offset_cap),
            sl_offset_cap=float(args.sl_offset_cap),
        )
        ent_uc = _apply_uc_params(
            entries_base[["signal_id", "signal_time", "split_id", "signal_tp_mult", "signal_sl_mult", "entry_time", "entry_price"]].copy(),
            rep_feat,
            uc_cfg,
            use_regime_modifiers=False,
            regime_tp_delta=float(args.regime_tp_delta),
            regime_sl_delta=float(args.regime_sl_delta),
        )
        t_uc = e2._simulate_1h_from_entries(entries_df=ent_uc, symbol=symbol, fee=fee, exec_horizon_hours=float(args.exec_horizon_hours))
        _, m_uc, _ = _compute_metrics(t_uc, signals_total=len(ent_uc), initial_equity=init_eq, risk_per_trade=risk_pt)
        split_uc = _compute_split_metrics_single(t_uc, rep_feat, split_definition, initial_equity=init_eq, risk_per_trade=risk_pt)

        min_split_trades = int(pd.to_numeric(split_uc["trades_total"], errors="coerce").min()) if not split_uc.empty else 0
        support_ok = int(min_split_trades >= int(args.min_split_trades))
        fatal = int(
            (_safe_float(m_uc.get("max_drawdown_pct")) <= float(args.fatal_max_dd))
            or (_safe_float(m_uc.get("total_return")) <= float(args.fatal_total_return))
        )

        g1_rows.append(
            {
                "offset_scale": float(sc),
                "signals_total": int(len(ent_uc)),
                "trades_total": int(m_uc.get("trades_total", 0)),
                "expectancy_net": float(m_uc.get("expectancy_net", np.nan)),
                "total_return": float(m_uc.get("total_return", np.nan)),
                "max_drawdown_pct": float(m_uc.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(m_uc.get("cvar_5", np.nan)),
                "profit_factor": float(m_uc.get("profit_factor", np.nan)),
                "min_split_trades": int(min_split_trades),
                "support_ok": int(support_ok),
                "fatal": int(fatal),
            }
        )

        if fatal == 1:
            fatal_streak += 1
        else:
            fatal_streak = 0

        if chosen_uc is None:
            chosen_uc = uc_cfg
        else:
            prev = _safe_float(g1_rows[-2]["expectancy_net"])
            cur = _safe_float(g1_rows[-1]["expectancy_net"])
            best_so_far = _safe_float(max(pd.to_numeric(pd.DataFrame(g1_rows)["expectancy_net"], errors="coerce")))
            if np.isfinite(cur) and np.isfinite(best_so_far) and cur >= best_so_far:
                chosen_uc = uc_cfg

        # Fail-fast for candidate search only.
        if fatal_streak >= int(args.fail_fast_streak):
            break

    g1_scan_df = pd.DataFrame(g1_rows)
    g1_scan_df.to_csv(run_dir / "phaseG1_uc_candidate_scan.csv", index=False)

    # Select best non-fatal/support-ok candidate; else best expectancy.
    if g1_scan_df.empty:
        raise RuntimeError("G1 candidate scan produced no rows.")
    viable = g1_scan_df[(g1_scan_df["support_ok"] == 1) & (g1_scan_df["fatal"] == 0)].copy()
    if not viable.empty:
        best_row = viable.sort_values("expectancy_net", ascending=False).iloc[0]
    else:
        best_row = g1_scan_df.sort_values("expectancy_net", ascending=False).iloc[0]

    chosen_scale = float(best_row["offset_scale"])
    uc_cfg_final = _build_uc_vectors(
        prior_tp=prior["tp_prior_median"],
        prior_sl=prior["sl_prior_median"],
        sol_tp=sol_tp,
        sol_sl=sol_sl,
        offset_scale=chosen_scale,
        tp_offset_cap=float(args.tp_offset_cap),
        sl_offset_cap=float(args.sl_offset_cap),
    )

    g1_design = {
        "global_prior_symbols": prior["symbols"],
        "selected_model_set_sha256": prior["selected_model_set_sha256"],
        "tp_prior_median": prior["tp_prior_median"],
        "sl_prior_median": prior["sl_prior_median"],
        "sol_anchor_tp": sol_tp,
        "sol_anchor_sl": sol_sl,
        "offset_scale_selected": chosen_scale,
        "tp_offset_cap": float(args.tp_offset_cap),
        "sl_offset_cap": float(args.sl_offset_cap),
        "uc_tp_vector": uc_cfg_final["tp_vector"],
        "uc_sl_vector": uc_cfg_final["sl_vector"],
        "uc_tp_offsets_applied": uc_cfg_final["tp_offset_applied"],
        "uc_sl_offsets_applied": uc_cfg_final["sl_offset_applied"],
        "regime_modifiers": {
            "enabled_in_variants": True,
            "adverse_tp_delta": -abs(float(args.regime_tp_delta)),
            "adverse_sl_delta": abs(float(args.regime_sl_delta)),
            "favorable_tp_delta": abs(float(args.regime_tp_delta)) * 0.5,
            "favorable_sl_delta": -abs(float(args.regime_sl_delta)) * 0.5,
            "sparse": True,
            "bounded": True,
        },
        "stability_constraints": {
            "min_trades": int(args.min_trades),
            "min_split_trades": int(args.min_split_trades),
            "min_bucket_support": int(args.min_bucket_support),
            "parameter_drift_limit": float(args.param_drift_limit),
            "drift_measured": 0.0,
        },
        "practical_risk_constraints": {
            "fatal_max_dd": float(args.fatal_max_dd),
            "fatal_total_return": float(args.fatal_total_return),
            "deploy_max_dd_floor": float(args.deploy_max_dd_floor),
            "deploy_cvar5_floor": float(args.deploy_cvar5_floor),
            "require_positive_expectancy": True,
            "require_positive_total_return": True,
            "no_accept_if_absolute_profile_trash": True,
        },
    }
    _json_dump(run_dir / "phaseG1_parameterization_design.json", g1_design)

    g1_lines = [
        "# Phase G1 Universe-Conditioned Parameterization",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Universe summary source: `{universe_summary}`",
        f"- Selected model set hash: `{prior['selected_model_set_sha256']}`",
        f"- Candidate scan rows: {len(g1_scan_df)}",
        f"- Candidate scan fail-fast streak threshold: {int(args.fail_fast_streak)}",
        f"- Selected offset_scale: {chosen_scale:.3f}",
        "",
        "## Global Prior (Passed Long Universe)",
        "",
        f"- symbols: {', '.join(prior['symbols'])}",
        f"- tp_prior_median: {[round(float(v), 6) for v in prior['tp_prior_median']]}",
        f"- sl_prior_median: {[round(float(v), 6) for v in prior['sl_prior_median']]}",
        "",
        "## SOL Anchor and Bounded Offsets",
        "",
        f"- sol_anchor_tp: {[round(float(v), 6) for v in sol_tp]}",
        f"- sol_anchor_sl: {[round(float(v), 6) for v in sol_sl]}",
        f"- tp_offset_cap: {float(args.tp_offset_cap):.6f}",
        f"- sl_offset_cap: {float(args.sl_offset_cap):.6f}",
        f"- uc_tp_vector: {[round(float(v), 6) for v in uc_cfg_final['tp_vector']]}",
        f"- uc_sl_vector: {[round(float(v), 6) for v in uc_cfg_final['sl_vector']]}",
        "",
        "## Stability and Practical Constraints",
        "",
        f"- min_trades: {int(args.min_trades)}",
        f"- min_split_trades: {int(args.min_split_trades)}",
        f"- min_bucket_support: {int(args.min_bucket_support)}",
        f"- parameter_drift_limit: {float(args.param_drift_limit):.6f} (measured drift=0.0 under fixed-vector design)",
        f"- fatal gates: max_dd<={float(args.fatal_max_dd):.6f} or total_return<={float(args.fatal_total_return):.6f}",
        "",
        "## Candidate Scan",
        "",
        _markdown_table(g1_scan_df),
    ]
    (run_dir / "phaseG1_parameterization_design.md").write_text("\n".join(g1_lines) + "\n", encoding="utf-8")

    # --------------------
    # G2 Ablation Matrix (signal-layer only)
    # --------------------
    variant_defs = [
        VariantConfig("baseline_signal", False, 0, 0, False, False),
        VariantConfig("plus_regime_gate", True, 0, 0, False, False),
        VariantConfig("plus_regime_gate_plus_cooldown4h", True, 4, 0, False, False),
        VariantConfig("plus_regime_gate_plus_cooldown4h_plus_delay0", True, 4, 0, False, False),
        VariantConfig("plus_regime_gate_plus_cooldown4h_plus_delay1", True, 4, 1, False, False),
        VariantConfig("plus_regime_gate_plus_cooldown4h_plus_delay2", True, 4, 2, False, False),
        VariantConfig("plus_uc_params", False, 0, 0, True, False),
        VariantConfig("plus_uc_params_plus_regime_mod", False, 0, 0, True, True),
        VariantConfig("plus_uc_params_plus_regime_mod_plus_regime_gate", True, 0, 0, True, True),
        VariantConfig("plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h", True, 4, 0, True, True),
        VariantConfig("plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay1", True, 4, 1, True, True),
        VariantConfig("plus_uc_params_plus_regime_mod_plus_regime_gate_plus_cooldown4h_plus_delay2", True, 4, 2, True, True),
    ]

    ablation_rows: List[Dict[str, Any]] = []
    split_rows: List[pd.DataFrame] = []

    baseline_expectancy = np.nan
    baseline_metrics_comp: Optional[Dict[str, Any]] = None

    for vc in variant_defs:
        ent = entries_base[["signal_id", "signal_time", "split_id", "signal_tp_mult", "signal_sl_mult", "entry_time", "entry_price"]].copy()
        ent = ent.merge(rep_feat[["signal_id", "cycle", "atr_percentile_1h", "trend_up_1h"]], on="signal_id", how="left")

        if vc.use_uc_params:
            ent = _apply_uc_params(
                ent,
                rep_feat,
                uc_cfg_final,
                use_regime_modifiers=vc.use_regime_modifiers,
                regime_tp_delta=float(args.regime_tp_delta),
                regime_sl_delta=float(args.regime_sl_delta),
            )
            ent = ent.merge(rep_feat[["signal_id", "cycle", "atr_percentile_1h", "trend_up_1h"]], on="signal_id", how="left")

        if vc.regime_gate:
            ent = _regime_gate_filter(
                ent,
                vol_min=float(args.regime_vol_min),
                vol_max=float(args.regime_vol_max),
                require_trend_up=bool(int(args.regime_require_trend_up)),
            )

        if vc.cooldown_hours > 0:
            ent = _apply_cooldown(ent, cooldown_hours=int(vc.cooldown_hours))

        if vc.delay_bars > 0:
            ent = _delay_entries_using_1h_open(
                ent[["signal_id", "signal_time", "split_id", "signal_tp_mult", "signal_sl_mult", "entry_time", "entry_price"]].copy(),
                symbol=symbol,
                offset_hours_after_signal=int(1 + vc.delay_bars),
            )

        ent = ent.dropna(subset=["signal_time", "entry_time", "entry_price", "signal_tp_mult", "signal_sl_mult"]).copy()
        ent = ent.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)

        t = e2._simulate_1h_from_entries(
            entries_df=ent,
            symbol=symbol,
            fee=fee,
            exec_horizon_hours=float(args.exec_horizon_hours),
        )
        t = t.merge(rep_feat[["signal_id", "signal_time", "cycle", "atr_percentile_1h", "trend_up_1h", "regime_bucket", "vol_bucket", "trend_bucket"]], on=["signal_id", "signal_time"], how="left")

        eq_c, m_c, _ = _compute_metrics(t, signals_total=len(ent), initial_equity=init_eq, risk_per_trade=risk_pt)
        _, m_f = _compute_fixed_size_equity_curve(t, signals_total=len(ent), initial_equity=init_eq, risk_per_trade=risk_pt)

        split_m = _compute_split_metrics_single(t, rep_feat, split_definition, initial_equity=init_eq, risk_per_trade=risk_pt)
        split_m.insert(0, "variant", vc.name)
        split_rows.append(split_m)

        losses = t[pd.to_numeric(t["pnl_net_pct"], errors="coerce") < 0].copy()
        adverse_loss_share = float(
            (
                (pd.to_numeric(losses["trend_up_1h"], errors="coerce") < 0.5)
                | (pd.to_numeric(losses["atr_percentile_1h"], errors="coerce") >= 66.6666666667)
            ).mean()
        ) if not losses.empty else np.nan

        min_split_trades = int(pd.to_numeric(split_m["trades_total"], errors="coerce").min()) if not split_m.empty else 0
        support_ok = int(min_split_trades >= int(args.min_split_trades))
        fatal = int(
            (_safe_float(m_c.get("max_drawdown_pct")) <= float(args.fatal_max_dd))
            or (_safe_float(m_c.get("total_return")) <= float(args.fatal_total_return))
        )
        abs_pass = int(
            _safe_float(m_c.get("expectancy_net")) > 0.0
            and _safe_float(m_c.get("total_return")) > 0.0
            and _safe_float(m_c.get("max_drawdown_pct")) > float(args.deploy_max_dd_floor)
            and _safe_float(m_c.get("cvar_5")) > float(args.deploy_cvar5_floor)
            and _safe_float(m_c.get("profit_factor")) >= float(args.deploy_pf_floor)
            and support_ok == 1
        )

        if baseline_metrics_comp is None:
            baseline_metrics_comp = m_c
            baseline_expectancy = _safe_float(m_c.get("expectancy_net"))

        reason = _classify_variant_reason(
            metrics_comp=m_c,
            metrics_fixed=m_f,
            adverse_loss_share=adverse_loss_share,
            baseline_expectancy=baseline_expectancy,
        )

        ablation_rows.append(
            {
                "variant": vc.name,
                "signals_total": int(len(ent)),
                "trades_total": int(m_c.get("trades_total", 0)),
                "regime_gate": int(vc.regime_gate),
                "cooldown_hours": int(vc.cooldown_hours),
                "delay_bars": int(vc.delay_bars),
                "use_uc_params": int(vc.use_uc_params),
                "use_regime_modifiers": int(vc.use_regime_modifiers),
                "expectancy_net": float(m_c.get("expectancy_net", np.nan)),
                "total_return": float(m_c.get("total_return", np.nan)),
                "max_drawdown_pct": float(m_c.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(m_c.get("cvar_5", np.nan)),
                "profit_factor": float(m_c.get("profit_factor", np.nan)),
                "win_rate": float(m_c.get("win_rate", np.nan)),
                "total_return_fixed": float(m_f.get("total_return_fixed", np.nan)),
                "max_drawdown_pct_fixed": float(m_f.get("max_drawdown_pct_fixed", np.nan)),
                "adverse_loss_share": float(adverse_loss_share),
                "min_split_trades": int(min_split_trades),
                "support_ok": int(support_ok),
                "fatal_gate": int(fatal),
                "absolute_practical_pass": int(abs_pass),
                "reason_classification": reason,
            }
        )

        # Save per-variant trades for audit.
        t.to_csv(run_dir / f"phaseG2_trades_{vc.name}.csv", index=False)
        eq_c.to_csv(run_dir / f"phaseG2_equity_{vc.name}.csv", index=False)

    if not ablation_rows:
        raise RuntimeError("No G2 variants were evaluated.")

    ablation_df = pd.DataFrame(ablation_rows)
    if baseline_metrics_comp is None:
        baseline_metrics_comp = {}

    base_row = ablation_df[ablation_df["variant"] == "baseline_signal"]
    if base_row.empty:
        raise RuntimeError("Missing baseline_signal row in G2 ablation matrix.")
    base = base_row.iloc[0]

    for col in ["expectancy_net", "total_return", "max_drawdown_pct", "cvar_5", "profit_factor", "win_rate"]:
        ablation_df[f"delta_{col}_vs_baseline"] = pd.to_numeric(ablation_df[col], errors="coerce") - float(base[col])

    ablation_df.to_csv(run_dir / "phaseG2_ablation_results.csv", index=False)
    pd.concat(split_rows, ignore_index=True).to_csv(run_dir / "phaseG2_split_metrics.csv", index=False)

    # --------------------
    # G3 Practical gate decisions
    # --------------------
    g3_df = ablation_df.copy()
    g3_df["relative_improvement_pass"] = (
        (pd.to_numeric(g3_df["delta_expectancy_net_vs_baseline"], errors="coerce") >= float(args.min_relative_expectancy_delta))
        & (pd.to_numeric(g3_df["delta_max_drawdown_pct_vs_baseline"], errors="coerce") >= float(args.min_relative_maxdd_delta))
    ).astype(int)

    g3_df["final_gate_pass"] = (
        (pd.to_numeric(g3_df["absolute_practical_pass"], errors="coerce") == 1)
        & (pd.to_numeric(g3_df["support_ok"], errors="coerce") == 1)
    ).astype(int)

    g3_df.to_csv(run_dir / "phaseG3_practical_gate_decisions.csv", index=False)

    best_idx = pd.to_numeric(g3_df["expectancy_net"], errors="coerce").idxmax()
    best_row = g3_df.loc[best_idx]

    any_absolute_pass = int((pd.to_numeric(g3_df["final_gate_pass"], errors="coerce") == 1).any())
    if any_absolute_pass == 1:
        final_verdict = "PROCEED_TO_PAPER_SOL"
    else:
        # Conservative path: keep HOLD unless practical gates are truly passed.
        substantial_rel = int(
            _safe_float(best_row.get("delta_expectancy_net_vs_baseline")) >= float(args.fork_expectancy_delta)
            and _safe_float(best_row.get("delta_total_return_vs_baseline")) >= float(args.fork_total_return_delta)
            and int(_safe_float(best_row.get("support_ok"))) == 1
        )
        if substantial_rel == 1 and int(_safe_float(best_row.get("fatal_gate"))) == 0:
            final_verdict = "FORK_SIGNAL_DEFINITION"
        else:
            final_verdict = "HOLD"

    # Gate status summary.
    setup_checks = {
        "symbol_match": int(str(e2_manifest.get("symbol", "")).upper() == symbol),
        "contract_id_match": int(str(contract.get("contract_id", "")) == "SOL_PHASEE2_CANONICAL_V1"),
        "fee_hash_match": int(fee_hash == EXPECTED_PHASEA_FEE_HASH),
        "metrics_hash_match": int(metrics_hash == EXPECTED_PHASEA_METRICS_HASH),
        "subset_hash_match": int(rep_subset_hash_calc == rep_subset_hash_ref == str(e2_manifest.get("representative_subset_sha256", ""))),
        "split_integrity": int(len(split_definition) >= 1),
    }

    run_manifest = {
        "generated_utc": _utc_now().isoformat(),
        "symbol": symbol,
        "contract_id": str(contract.get("contract_id")),
        "e2_dir": str(e2_dir),
        "phase_c_dir": str(phase_c_dir),
        "phase_a_dir": str(phase_a_dir),
        "signal_source_csv": str(signal_source_path),
        "signal_source_sha256": _sha256_file(signal_source_path),
        "fee_model_path": str(fee_model_path),
        "fee_model_sha256": fee_hash,
        "metrics_definition_path": str(metrics_def_path),
        "metrics_definition_sha256": metrics_hash,
        "representative_subset_path": str(rep_subset_path),
        "representative_subset_sha256": rep_subset_hash_calc,
        "split_definition": split_definition,
        "selected_model_set_sha256": prior["selected_model_set_sha256"],
        "global_prior_symbols": prior["symbols"],
        "setup_checks": setup_checks,
        "g1_selected_offset_scale": chosen_scale,
        "g3_final_verdict": final_verdict,
        "best_variant": str(best_row.get("variant")),
    }
    _json_dump(run_dir / "run_manifest.json", run_manifest)

    # Repro + git status.
    repro_lines = [
        "# Repro",
        "",
        "```bash",
        f"cd {PROJECT_ROOT}",
        "analysis/0.87/venv/bin/python scripts/phase_g_sol_pathology_rehab.py "
        f"--symbol {symbol} "
        f"--e2-dir {e2_dir} "
        f"--outdir {args.outdir}",
        "```",
    ]
    (run_dir / "repro.md").write_text("\n".join(repro_lines) + "\n", encoding="utf-8")
    try:
        gs = subprocess.check_output(["git", "status", "--short"], cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.STDOUT)
    except Exception as ex:
        gs = f"git status unavailable: {ex}"
    (run_dir / "git_status.txt").write_text(gs, encoding="utf-8")

    # Consolidated phase report (for required response format).
    report_lines = [
        "# Phase G SOL Pathology-First Rehab",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Symbol: {symbol}",
        f"- Final verdict: **{final_verdict}**",
        "",
        "## 1) Frozen setup confirmation (hashes + contract lock checks)",
        "",
        f"- fee_model_sha256: `{fee_hash}`",
        f"- metrics_definition_sha256: `{metrics_hash}`",
        f"- representative_subset_sha256: `{rep_subset_hash_calc}`",
        f"- selected_model_set_sha256: `{prior['selected_model_set_sha256']}`",
        f"- setup_checks: {json.dumps(setup_checks, sort_keys=True)}",
        "",
        "## 2) G0 drawdown forensics summary (root-cause ranking)",
        "",
        f"- baseline_variant: V4R_EXEC_3M_PHASEC_BEST",
        f"- comp_total_return/maxDD: {_safe_float(m_comp.get('total_return')):.6f} / {_safe_float(m_comp.get('max_drawdown_pct')):.6f}",
        f"- fixed_total_return/maxDD: {_safe_float(m_fix.get('total_return_fixed')):.6f} / {_safe_float(m_fix.get('max_drawdown_pct_fixed')):.6f}",
        f"- adverse_loss_share: {adverse_loss_share:.6f}",
        f"- sl_loss_share: {sl_loss_share:.6f}",
        f"- amplification_ratio: {amp_ratio:.6f}",
        "",
        _markdown_table(root_rank),
        "",
        "## 3) G1 parameterization design (global prior + SOL offsets + constraints)",
        "",
        f"- prior_symbols: {', '.join(prior['symbols'])}",
        f"- selected_offset_scale: {chosen_scale:.3f}",
        f"- uc_tp_vector: {[round(float(v), 6) for v in uc_cfg_final['tp_vector']]}",
        f"- uc_sl_vector: {[round(float(v), 6) for v in uc_cfg_final['sl_vector']]}",
        f"- tp_offset_cap/sl_offset_cap: {float(args.tp_offset_cap):.6f} / {float(args.sl_offset_cap):.6f}",
        f"- constraints(min_trades/min_split/min_bucket): {int(args.min_trades)}/{int(args.min_split_trades)}/{int(args.min_bucket_support)}",
        f"- fail_fast_streak_used: {int(args.fail_fast_streak)}",
        "",
        "## 4) G2 ablation results table",
        "",
        _markdown_table(
            ablation_df[
                [
                    "variant",
                    "signals_total",
                    "trades_total",
                    "expectancy_net",
                    "total_return",
                    "max_drawdown_pct",
                    "cvar_5",
                    "profit_factor",
                    "fatal_gate",
                    "absolute_practical_pass",
                ]
            ].sort_values("expectancy_net", ascending=False)
        ),
        "",
        "## 5) G3 practical gate decisions",
        "",
        _markdown_table(
            g3_df[
                [
                    "variant",
                    "delta_expectancy_net_vs_baseline",
                    "delta_total_return_vs_baseline",
                    "delta_max_drawdown_pct_vs_baseline",
                    "relative_improvement_pass",
                    "absolute_practical_pass",
                    "final_gate_pass",
                    "reason_classification",
                ]
            ].sort_values("delta_expectancy_net_vs_baseline", ascending=False)
        ),
        "",
        "## 6) Final recommendation + next exact prompt",
        "",
        f"- Final recommendation: **{final_verdict}**",
        "",
        "Next exact prompt:",
        "",
        "```text",
        (
            "Phase G follow-up (SOL, contract-locked): keep the same representative subset/hash and rerun only signal-layer changes. "
            "Implement regime gate + 4h cooldown + delayed 1h entry modes with bounded universe-conditioned tp/sl offsets; reject any candidate that fails "
            "absolute practical gates (expectancy>0, total_return>0, maxDD>-0.35, cvar5>-0.0015). "
            "If all fail absolute gates again, freeze optimization and prepare FORK/HOLD memo with root-cause evidence from Phase G0 forensics."
        ),
        "```",
    ]
    (run_dir / "phaseG_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase G SOL pathology-first rehab workflow on contract-locked representative harness.")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--e2-dir", default="reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--universe-summary-json", default="artifacts/reports/universe_20260211_172937/summary.json")
    ap.add_argument("--sol-param-path", default="data/metadata/params/SOLUSDT_C13_active_params_long.json")
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--initial-equity", type=float, default=1.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)

    ap.add_argument("--tp-offset-cap", type=float, default=0.08)
    ap.add_argument("--sl-offset-cap", type=float, default=0.05)
    ap.add_argument("--uc-offset-scales", default="0.0,0.25,0.5,0.75,1.0")
    ap.add_argument("--regime-tp-delta", type=float, default=0.02)
    ap.add_argument("--regime-sl-delta", type=float, default=0.01)

    ap.add_argument("--regime-vol-min", type=float, default=5.0)
    ap.add_argument("--regime-vol-max", type=float, default=90.0)
    ap.add_argument("--regime-require-trend-up", type=int, default=1)

    ap.add_argument("--min-trades", type=int, default=120)
    ap.add_argument("--min-split-trades", type=int, default=40)
    ap.add_argument("--min-bucket-support", type=int, default=30)
    ap.add_argument("--param-drift-limit", type=float, default=0.03)

    ap.add_argument("--fatal-max-dd", type=float, default=-0.95)
    ap.add_argument("--fatal-total-return", type=float, default=-0.95)
    ap.add_argument("--fail-fast-streak", type=int, default=3)

    ap.add_argument("--deploy-max-dd-floor", type=float, default=-0.35)
    ap.add_argument("--deploy-cvar5-floor", type=float, default=-0.0015)
    ap.add_argument("--deploy-pf-floor", type=float, default=1.05)

    ap.add_argument("--min-relative-expectancy-delta", type=float, default=0.00005)
    ap.add_argument("--min-relative-maxdd-delta", type=float, default=0.01)
    ap.add_argument("--fork-expectancy-delta", type=float, default=0.00020)
    ap.add_argument("--fork-total-return-delta", type=float, default=0.10)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    print(str(out))


if __name__ == "__main__":
    main()
