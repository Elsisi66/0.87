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
from scripts import sol_reconcile_truth as recon  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


EXPECTED_PHASEA_FEE_HASH = "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a"
EXPECTED_PHASEA_METRICS_HASH = "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99"
EXPECTED_PHASEC_CFG_HASH = "a285b86c4c22a26976d4a762"
EXPECTED_SIGNAL_SUBSET_HASH = "5e719faf676dffba8d7da926314997182d429361495884b8a870c3393c079bbf"
EXPECTED_SPLIT_HASH = "388ba743b9c16c291385a9ecab6435eabf65eb16f1e1083eee76627193c42c01"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_tag() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _resolve(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def _sha256_signal_subset(df: pd.DataFrame) -> str:
    x = df.copy()
    if "signal_id" not in x.columns or "signal_time" not in x.columns:
        raise RuntimeError("signal subset requires signal_id and signal_time")
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rows = [f"{str(r.signal_id)}|{pd.to_datetime(r.signal_time, utc=True).isoformat()}" for r in x.itertuples(index=False)]
    return _sha256_text("\n".join(rows))


def _ensure_hash(path: Path, expected: str, label: str) -> str:
    got = _sha256_file(path)
    if str(expected).strip() and got != str(expected).strip():
        raise RuntimeError(f"{label} hash mismatch: expected={expected} got={got}")
    return got


def _load_phasec_manifest(phase_c_dir: Path) -> Dict[str, Any]:
    fp = phase_c_dir / "run_manifest.json"
    if not fp.exists():
        raise FileNotFoundError(f"Missing Phase C manifest: {fp}")
    obj = json.loads(fp.read_text(encoding="utf-8"))
    return obj


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
        raise RuntimeError(f"No splits in {path}")
    return out


def _test_indices_from_splits(splits: List[Dict[str, int]]) -> List[int]:
    out: List[int] = []
    for s in splits:
        out.extend(list(range(int(s["test_start"]), int(s["test_end"]))))
    return sorted(set(out))


def _normalize_signals(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    if "tp_mult" in x.columns:
        x["tp_mult"] = pd.to_numeric(x["tp_mult"], errors="coerce")
    elif "strategy_tp_mult" in x.columns:
        x["tp_mult"] = pd.to_numeric(x["strategy_tp_mult"], errors="coerce")
    else:
        x["tp_mult"] = np.nan
    if "sl_mult" in x.columns:
        x["sl_mult"] = pd.to_numeric(x["sl_mult"], errors="coerce")
    elif "strategy_sl_mult" in x.columns:
        x["sl_mult"] = pd.to_numeric(x["strategy_sl_mult"], errors="coerce")
    else:
        x["sl_mult"] = np.nan
    x["atr_percentile_1h"] = pd.to_numeric(x.get("atr_percentile_1h"), errors="coerce")
    x["trend_up_1h"] = pd.to_numeric(x.get("trend_up_1h"), errors="coerce")
    x = x.dropna(subset=["signal_time", "tp_mult", "sl_mult"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    return x


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


def _stratified_sample(df: pd.DataFrame, n: int, seed: int, strata_col: str) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if df.empty:
        return df.copy()
    if n >= len(df):
        return df.sort_values(["signal_time", "signal_id"]).reset_index(drop=True).copy()

    x = df.copy().reset_index(drop=True)
    x["_strata"] = x[strata_col].astype(str).fillna("unknown")
    x["_u"] = np.arange(len(x))
    g = x.groupby("_strata", dropna=False).size().sort_index()
    prop = g / float(g.sum())
    base = np.floor(prop * float(n)).astype(int)
    rem = int(n - int(base.sum()))
    frac = (prop * float(n)) - base
    if rem > 0:
        add_order = frac.sort_values(ascending=False).index.tolist()
        for k in add_order:
            if rem <= 0:
                break
            base[k] += 1
            rem -= 1
    rs = np.random.RandomState(int(seed))
    parts: List[pd.DataFrame] = []
    for k, grp in x.groupby("_strata", dropna=False):
        take = int(min(base.get(k, 0), len(grp)))
        if take <= 0:
            continue
        idx = rs.choice(np.arange(len(grp)), size=take, replace=False)
        parts.append(grp.iloc[np.sort(idx)])
    out = pd.concat(parts, ignore_index=True) if parts else x.iloc[0:0].copy()
    shortfall = int(n - len(out))
    if shortfall > 0:
        used = set(out["_u"].tolist())
        rest = x[~x["_u"].isin(used)].copy()
        if shortfall > len(rest):
            shortfall = len(rest)
        if shortfall > 0:
            idx = rs.choice(np.arange(len(rest)), size=shortfall, replace=False)
            out = pd.concat([out, rest.iloc[np.sort(idx)]], ignore_index=True)
    out = out.drop(columns=["_strata", "_u"], errors="ignore")
    return out.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def _build_splits(n: int, wf_splits: int) -> List[Dict[str, int]]:
    if n <= 0:
        raise RuntimeError("Cannot build splits for empty subset.")
    k = max(1, int(wf_splits))
    if k > n:
        k = n
    edges = np.linspace(0, n, num=k + 1, dtype=int)
    out: List[Dict[str, int]] = []
    for i in range(k):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        if hi <= lo:
            continue
        out.append(
            {
                "split_id": int(i),
                "train_start": 0,
                "train_end": int(lo),
                "test_start": int(lo),
                "test_end": int(hi),
            }
        )
    if not out:
        out = [{"split_id": 0, "train_start": 0, "train_end": 0, "test_start": 0, "test_end": int(n)}]
    return out


def _split_lookup(df_subset: pd.DataFrame, splits: List[Dict[str, int]]) -> Dict[str, int]:
    idx_to_split: Dict[int, int] = {}
    for s in splits:
        sid = int(s["split_id"])
        for i in range(int(s["test_start"]), int(s["test_end"])):
            idx_to_split[int(i)] = sid
    out: Dict[str, int] = {}
    for i, r in df_subset.reset_index(drop=True).iterrows():
        if i in idx_to_split:
            out[str(r["signal_id"])] = int(idx_to_split[i])
    return out


def _latest_phasee_consistency_dir(root: Path) -> Optional[Path]:
    cands = sorted(root.glob("PHASEE_SOL_CONSISTENCY_*"), key=lambda p: p.stat().st_mtime)
    if not cands:
        return None
    return cands[-1]


def _load_market_feature_table(symbol: str) -> pd.DataFrame:
    k = recon._load_symbol_df(symbol=symbol, tf="1h").copy()
    k["Timestamp"] = pd.to_datetime(k["Timestamp"], utc=True, errors="coerce")
    k = k.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).sort_values("Timestamp").reset_index(drop=True)
    close = pd.to_numeric(k["Close"], errors="coerce")
    high = pd.to_numeric(k["High"], errors="coerce")
    low = pd.to_numeric(k["Low"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14, min_periods=14).mean()
    atr_pct = atr14 / close.replace(0.0, np.nan) * 100.0
    atr_pct_rank = atr_pct.rank(method="average", pct=True) * 100.0
    ret24 = close.pct_change(24)
    trend_up = (ret24 > 0).astype(float)
    out = pd.DataFrame(
        {
            "signal_time": pd.to_datetime(k["Timestamp"], utc=True),
            "atr_percentile_1h_calc": pd.to_numeric(atr_pct_rank, errors="coerce"),
            "trend_up_1h_calc": pd.to_numeric(trend_up, errors="coerce"),
        }
    ).dropna(subset=["signal_time"])
    return out.sort_values("signal_time").reset_index(drop=True)


def _attach_universe_features(df: pd.DataFrame, feat_table: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.dropna(subset=["signal_time"]).sort_values("signal_time").reset_index(drop=True)

    if "atr_percentile_1h" not in x.columns:
        x["atr_percentile_1h"] = np.nan
    if "trend_up_1h" not in x.columns:
        x["trend_up_1h"] = np.nan

    x["atr_percentile_1h"] = pd.to_numeric(x["atr_percentile_1h"], errors="coerce")
    x["trend_up_1h"] = pd.to_numeric(x["trend_up_1h"], errors="coerce")

    xf = x.sort_values("signal_time").copy()
    ff = feat_table.sort_values("signal_time").copy()
    xf["_ts_ns"] = pd.to_datetime(xf["signal_time"], utc=True).astype("int64")
    ff["_ts_ns"] = pd.to_datetime(ff["signal_time"], utc=True).astype("int64")
    m = pd.merge_asof(
        xf,
        ff[["_ts_ns", "atr_percentile_1h_calc", "trend_up_1h_calc"]],
        on="_ts_ns",
        direction="backward",
    )
    x["atr_percentile_1h"] = x["atr_percentile_1h"].fillna(m["atr_percentile_1h_calc"])
    x["trend_up_1h"] = x["trend_up_1h"].fillna(m["trend_up_1h_calc"])

    st = pd.to_datetime(x["signal_time"], utc=True)
    x["month"] = st.dt.to_period("M").astype(str)
    x["quarter"] = st.dt.to_period("Q").astype(str)
    x["dow"] = st.dt.day_name().fillna("unknown")
    x["hour"] = st.dt.hour
    x["hour_bucket"] = _hour_bucket(x["hour"])
    x["vol_bucket"] = _vol_bucket(x["atr_percentile_1h"])
    x["trend_bucket"] = x["trend_up_1h"].map(lambda v: "up" if _safe_float(v) >= 0.5 else "down")
    x["trend_bucket"] = x["trend_bucket"].fillna("unknown").astype(str)
    return x


def _distribution_compare(ref_df: pd.DataFrame, cmp_df: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    dims = ["quarter", "month", "vol_bucket", "trend_bucket", "dow", "hour_bucket"]
    rows: List[Dict[str, Any]] = []
    for dim in dims:
        rcount = ref_df[dim].astype(str).value_counts(dropna=False)
        ccount = cmp_df[dim].astype(str).value_counts(dropna=False)
        rshare = rcount / max(1, int(rcount.sum()))
        cshare = ccount / max(1, int(ccount.sum()))
        buckets = sorted(set(rshare.index) | set(cshare.index))
        tvd = 0.5 * float(sum(abs(float(rshare.get(k, 0.0)) - float(cshare.get(k, 0.0))) for k in buckets))
        for b in buckets:
            rows.append(
                {
                    "pair": pair_name,
                    "dimension": dim,
                    "bucket": str(b),
                    "ref_count": int(rcount.get(b, 0)),
                    "cmp_count": int(ccount.get(b, 0)),
                    "ref_share": float(rshare.get(b, 0.0)),
                    "cmp_share": float(cshare.get(b, 0.0)),
                    "abs_delta": float(abs(float(rshare.get(b, 0.0)) - float(cshare.get(b, 0.0)))),
                    "tvd_dimension": float(tvd),
                }
            )
    return pd.DataFrame(rows)


def _coverage_ratio(ref_df: pd.DataFrame, cmp_df: pd.DataFrame, dims: Sequence[str]) -> float:
    vals: List[float] = []
    for dim in dims:
        r = set(ref_df[dim].astype(str).value_counts()[lambda s: s > 0].index.tolist())
        c = set(cmp_df[dim].astype(str).value_counts()[lambda s: s > 0].index.tolist())
        if not r:
            vals.append(float("nan"))
            continue
        vals.append(float(len(r & c) / max(1, len(r))))
    z = [v for v in vals if np.isfinite(v)]
    return float(np.mean(z)) if z else float("nan")


def _ga_rows_to_trades(
    rows_df: pd.DataFrame,
    *,
    mode: str,
    fee: phasec_bt.FeeModel,
    exec_sl_mult: float,
    split_lookup: Dict[str, int],
) -> pd.DataFrame:
    m = str(mode).strip().lower()
    if m not in {"baseline", "exec"}:
        raise RuntimeError(f"Unsupported mode: {mode}")
    x = rows_df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x["split_id"] = x["signal_id"].map(split_lookup).fillna(-1).astype(int)
    x["signal_tp_mult"] = pd.to_numeric(x.get("signal_tp_mult"), errors="coerce")
    x["signal_sl_mult"] = pd.to_numeric(x.get("signal_sl_mult"), errors="coerce")

    if m == "baseline":
        prefix = "baseline"
        risk_mult = 1.0
    else:
        prefix = "exec"
        risk_mult = float(exec_sl_mult)

    out = pd.DataFrame(
        {
            "symbol": x.get("symbol", "SOLUSDT"),
            "signal_id": x["signal_id"],
            "signal_time": x["signal_time"],
            "split_id": x["split_id"],
            "signal_tp_mult": x["signal_tp_mult"],
            "signal_sl_mult": x["signal_sl_mult"],
            "filled": pd.to_numeric(x.get(f"{prefix}_filled", 0), errors="coerce").fillna(0).astype(int),
            "valid_for_metrics": pd.to_numeric(x.get(f"{prefix}_valid_for_metrics", 0), errors="coerce").fillna(0).astype(int),
            "sl_hit": pd.to_numeric(x.get(f"{prefix}_sl_hit", 0), errors="coerce").fillna(0).astype(int),
            "tp_hit": pd.to_numeric(x.get(f"{prefix}_tp_hit", 0), errors="coerce").fillna(0).astype(int),
            "entry_time": pd.to_datetime(x.get(f"{prefix}_entry_time"), utc=True, errors="coerce"),
            "exit_time": pd.to_datetime(x.get(f"{prefix}_exit_time"), utc=True, errors="coerce"),
            "entry_price": pd.to_numeric(x.get(f"{prefix}_entry_price"), errors="coerce"),
            "exit_price": pd.to_numeric(x.get(f"{prefix}_exit_price"), errors="coerce"),
            "exit_reason": x.get(f"{prefix}_exit_reason", "").fillna("").astype(str),
            "mae_pct": pd.to_numeric(x.get(f"{prefix}_mae_pct"), errors="coerce"),
            "mfe_pct": pd.to_numeric(x.get(f"{prefix}_mfe_pct"), errors="coerce"),
            "fill_liquidity_type": x.get(f"{prefix}_fill_liquidity_type", "").fillna("").astype(str).str.lower(),
            "entry_type": x.get(f"{prefix}_entry_type", "market").fillna("market").astype(str),
            "fill_delay_min": pd.to_numeric(x.get(f"{prefix}_fill_delay_min"), errors="coerce"),
            "pnl_net_pct": pd.to_numeric(x.get(f"{prefix}_pnl_net_pct"), errors="coerce"),
            "pnl_gross_pct": pd.to_numeric(x.get(f"{prefix}_pnl_gross_pct"), errors="coerce"),
        }
    )
    out["hold_minutes"] = (
        (pd.to_datetime(out["exit_time"], utc=True, errors="coerce") - pd.to_datetime(out["entry_time"], utc=True, errors="coerce"))
        .dt.total_seconds()
        .div(60.0)
    )
    out["risk_pct"] = (
        (1.0 - pd.to_numeric(out["signal_sl_mult"], errors="coerce")).clip(lower=1e-8) * max(1e-8, float(risk_mult))
    )

    fee_rows: List[Dict[str, Any]] = []
    for r in out.itertuples(index=False):
        ep = _safe_float(getattr(r, "entry_price"))
        xp = _safe_float(getattr(r, "exit_price"))
        liq = str(getattr(r, "fill_liquidity_type", "taker")).lower().strip() or "taker"
        if liq not in {"maker", "taker"}:
            liq = "taker"
        if np.isfinite(ep) and np.isfinite(xp) and ep > 0 and int(getattr(r, "filled", 0)) == 1:
            c = phasec_bt._cost_row(ep, xp, liq, fee)
        else:
            c = {
                "pnl_gross_pct": np.nan,
                "pnl_net_pct": np.nan,
                "entry_fee_bps": np.nan,
                "exit_fee_bps": np.nan,
                "entry_slippage_bps": np.nan,
                "exit_slippage_bps": np.nan,
                "total_cost_bps": np.nan,
            }
        fee_rows.append(c)
    cdf = pd.DataFrame(fee_rows)
    for k in cdf.columns:
        out[k] = pd.to_numeric(cdf[k], errors="coerce")
    return out.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def _subset_metrics(
    trades: pd.DataFrame,
    *,
    signal_ids: set[str],
    signals_total: int,
    initial_equity: float,
    risk_per_trade: float,
) -> Dict[str, Any]:
    x = trades[trades["signal_id"].astype(str).isin(signal_ids)].copy()
    eq, m, _ = phasec_bt._compute_equity_curve(
        x,
        signals_total=int(signals_total),
        initial_equity=float(initial_equity),
        risk_per_trade=float(risk_per_trade),
    )
    out = dict(m)
    out["equity_end"] = float(eq["equity"].iloc[-1]) if not eq.empty else float("nan")
    return out


def _split_metrics(
    trades_by_variant: Dict[str, pd.DataFrame],
    subset_df: pd.DataFrame,
    splits: List[Dict[str, int]],
    *,
    initial_equity: float,
    risk_per_trade: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ss = subset_df.reset_index(drop=True)
    for sp in splits:
        split_id = int(sp["split_id"])
        lo, hi = int(sp["test_start"]), int(sp["test_end"])
        s = ss.iloc[lo:hi].copy()
        sig_ids = set(s["signal_id"].astype(str).tolist())
        sig_n = int(len(s))
        for variant, tdf in trades_by_variant.items():
            m = _subset_metrics(
                tdf,
                signal_ids=sig_ids,
                signals_total=sig_n,
                initial_equity=float(initial_equity),
                risk_per_trade=float(risk_per_trade),
            )
            rows.append(
                {
                    "variant": variant,
                    "split_id": split_id,
                    "signals_total": sig_n,
                    "trades_total": int(m.get("trades_total", 0)),
                    "expectancy_net": float(m.get("expectancy_net", np.nan)),
                    "total_return": float(m.get("total_return", np.nan)),
                    "max_drawdown_pct": float(m.get("max_drawdown_pct", np.nan)),
                    "cvar_5": float(m.get("cvar_5", np.nan)),
                    "profit_factor": float(m.get("profit_factor", np.nan)),
                    "win_rate": float(m.get("win_rate", np.nan)),
                    "taker_share": float(m.get("taker_share", np.nan)),
                    "median_fill_delay_min": float(m.get("median_fill_delay_min", np.nan)),
                    "p95_fill_delay_min": float(m.get("p95_fill_delay_min", np.nan)),
                }
            )
    return pd.DataFrame(rows)


def _comparison_matrix(metrics_df: pd.DataFrame) -> pd.DataFrame:
    idx = {r["variant"]: r for _, r in metrics_df.iterrows()}
    a = idx["V2R_1H_REFERENCE_CONTROL"]
    b = idx["V3R_EXEC_3M_PHASEC_CONTROL"]
    c = idx["V4R_EXEC_3M_PHASEC_BEST"]
    metrics = [
        "expectancy_net",
        "total_return",
        "max_drawdown_pct",
        "cvar_5",
        "profit_factor",
        "win_rate",
        "trades_total",
        "total_fees_paid",
        "taker_share",
        "median_fill_delay_min",
        "p95_fill_delay_min",
    ]
    rows: List[Dict[str, Any]] = []
    for m in metrics:
        av = _safe_float(a.get(m))
        bv = _safe_float(b.get(m))
        cv = _safe_float(c.get(m))
        rows.append(
            {
                "metric": m,
                "v2r_1h_reference_control": av,
                "v3r_exec_3m_phasec_control": bv,
                "v4r_exec_3m_phasec_best": cv,
                "delta_v3r_minus_v2r": float(bv - av) if np.isfinite(bv) and np.isfinite(av) else np.nan,
                "delta_v4r_minus_v3r": float(cv - bv) if np.isfinite(cv) and np.isfinite(bv) else np.nan,
                "delta_v4r_minus_v2r": float(cv - av) if np.isfinite(cv) and np.isfinite(av) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _simulate_1h_from_entries(
    *,
    entries_df: pd.DataFrame,
    symbol: str,
    fee: phasec_bt.FeeModel,
    exec_horizon_hours: float,
) -> pd.DataFrame:
    df1h = recon._load_symbol_df(symbol=symbol, tf="1h")
    ts_pd = pd.to_datetime(df1h["Timestamp"], utc=True, errors="coerce")
    ts_ns = np.array([int(t.value) for t in ts_pd], dtype=np.int64)
    hi = pd.to_numeric(df1h["High"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(df1h["Low"], errors="coerce").to_numpy(dtype=float)
    op = pd.to_numeric(df1h["Open"], errors="coerce").to_numpy(dtype=float)
    cl = pd.to_numeric(df1h["Close"], errors="coerce").to_numpy(dtype=float)

    rows: List[Dict[str, Any]] = []
    for r in entries_df.itertuples(index=False):
        sid = str(getattr(r, "signal_id"))
        st = pd.to_datetime(getattr(r, "signal_time"), utc=True, errors="coerce")
        et = pd.to_datetime(getattr(r, "entry_time"), utc=True, errors="coerce")
        ep = _safe_float(getattr(r, "entry_price"))
        tp_mult = _safe_float(getattr(r, "signal_tp_mult"))
        sl_mult = _safe_float(getattr(r, "signal_sl_mult"))
        split_id = int(getattr(r, "split_id")) if np.isfinite(_safe_float(getattr(r, "split_id", np.nan))) else -1

        base = {
            "symbol": symbol,
            "signal_id": sid,
            "signal_time": st,
            "split_id": split_id,
            "signal_tp_mult": tp_mult,
            "signal_sl_mult": sl_mult,
            "entry_type": "market",
            "fill_liquidity_type": "taker",
        }
        if pd.isna(st) or pd.isna(et) or (not np.isfinite(ep)) or ep <= 0 or (not np.isfinite(tp_mult)) or (not np.isfinite(sl_mult)):
            rows.append(
                {
                    **base,
                    "filled": 0,
                    "valid_for_metrics": 0,
                    "sl_hit": 0,
                    "tp_hit": 0,
                    "entry_time": et,
                    "exit_time": pd.NaT,
                    "entry_price": ep,
                    "exit_price": np.nan,
                    "exit_reason": "invalid_entry",
                    "mae_pct": np.nan,
                    "mfe_pct": np.nan,
                    "hold_minutes": np.nan,
                    "fill_delay_min": np.nan,
                    "risk_pct": max(1e-8, 1.0 - sl_mult) if np.isfinite(sl_mult) else np.nan,
                    "pnl_gross_pct": np.nan,
                    "pnl_net_pct": np.nan,
                    "entry_fee_bps": np.nan,
                    "exit_fee_bps": np.nan,
                    "entry_slippage_bps": np.nan,
                    "exit_slippage_bps": np.nan,
                    "total_cost_bps": np.nan,
                }
            )
            continue

        sl = ep * sl_mult
        tp = ep * tp_mult
        max_ns = int(et.value + float(exec_horizon_hours) * 3600.0 * 1e9)
        i0 = int(np.searchsorted(ts_ns, int(et.value), side="left"))
        if i0 >= len(ts_ns):
            rows.append(
                {
                    **base,
                    "filled": 0,
                    "valid_for_metrics": 0,
                    "sl_hit": 0,
                    "tp_hit": 0,
                    "entry_time": et,
                    "exit_time": pd.NaT,
                    "entry_price": ep,
                    "exit_price": np.nan,
                    "exit_reason": "no_bar_after_entry",
                    "mae_pct": np.nan,
                    "mfe_pct": np.nan,
                    "hold_minutes": np.nan,
                    "fill_delay_min": float((et - st).total_seconds() / 60.0),
                    "risk_pct": max(1e-8, 1.0 - sl_mult),
                    "pnl_gross_pct": np.nan,
                    "pnl_net_pct": np.nan,
                    "entry_fee_bps": np.nan,
                    "exit_fee_bps": np.nan,
                    "entry_slippage_bps": np.nan,
                    "exit_slippage_bps": np.nan,
                    "total_cost_bps": np.nan,
                }
            )
            continue

        mae = 0.0
        mfe = 0.0
        mae_loc = 0
        mfe_loc = 0
        exit_i = i0
        exit_px = float(cl[min(i0, len(cl) - 1)])
        reason = "window_end"
        sl_hit = 0
        tp_hit = 0
        for i in range(i0, len(ts_ns)):
            tns = int(ts_ns[i])
            if tns > max_ns:
                exit_i = max(i0, i - 1)
                exit_px = float(cl[exit_i])
                reason = "window_end"
                break
            h = float(hi[i])
            l = float(lo[i])
            o = float(op[i])
            c = float(cl[i])

            mcur = l / ep - 1.0
            fcur = h / ep - 1.0
            if mcur < mae:
                mae = float(mcur)
                mae_loc = int(i - i0)
            if fcur > mfe:
                mfe = float(fcur)
                mfe_loc = int(i - i0)

            hit_sl = bool(l <= sl)
            hit_tp = bool(h >= tp)
            if hit_sl and hit_tp:
                # Neutral deterministic rule.
                if c >= o:
                    exit_px = float(tp)
                    reason = "tp"
                    tp_hit = 1
                else:
                    exit_px = float(sl)
                    reason = "sl"
                    sl_hit = 1
                exit_i = i
                break
            if hit_sl:
                exit_i = i
                exit_px = float(sl)
                reason = "sl"
                sl_hit = 1
                break
            if hit_tp:
                exit_i = i
                exit_px = float(tp)
                reason = "tp"
                tp_hit = 1
                break
            exit_i = i
            exit_px = float(cl[i])

        xt = pd.to_datetime(int(ts_ns[exit_i]), utc=True)
        et0 = pd.to_datetime(int(ts_ns[i0]), utc=True)
        c = phasec_bt._cost_row(float(ep), float(exit_px), "taker", fee)
        rows.append(
            {
                **base,
                "filled": 1,
                "valid_for_metrics": 1,
                "sl_hit": int(sl_hit),
                "tp_hit": int(tp_hit),
                "entry_time": et,
                "exit_time": xt,
                "entry_price": float(ep),
                "exit_price": float(exit_px),
                "exit_reason": reason,
                "mae_pct": float(mae),
                "mfe_pct": float(mfe),
                "hold_minutes": float((xt - et).total_seconds() / 60.0),
                "fill_delay_min": float((et - st).total_seconds() / 60.0),
                "risk_pct": max(1e-8, 1.0 - sl_mult),
                "pnl_gross_pct": float(c["pnl_gross_pct"]),
                "pnl_net_pct": float(c["pnl_net_pct"]),
                "entry_fee_bps": float(c["entry_fee_bps"]),
                "exit_fee_bps": float(c["exit_fee_bps"]),
                "entry_slippage_bps": float(c["entry_slippage_bps"]),
                "exit_slippage_bps": float(c["exit_slippage_bps"]),
                "total_cost_bps": float(c["total_cost_bps"]),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    symbol = str(args.symbol).strip().upper()
    if symbol != "SOLUSDT":
        raise RuntimeError("Phase E2 is hard-scoped to SOLUSDT only.")

    phase_c_dir = _resolve(args.phase_c_dir)
    phase_a_dir = _resolve(args.phase_a_contract_dir)
    out_root = _resolve(args.outdir)
    run_dir = out_root / f"PHASEE2_SOL_REPRESENTATIVE_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snap = run_dir / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    manifest = _load_phasec_manifest(phase_c_dir)
    if str(manifest.get("symbol", "")).upper() != symbol:
        raise RuntimeError(f"Phase C manifest symbol mismatch: {manifest.get('symbol')}")
    if str(manifest.get("final_selected_cfg_hash", "")) != EXPECTED_PHASEC_CFG_HASH:
        raise RuntimeError(
            f"Phase C cfg hash mismatch: expected={EXPECTED_PHASEC_CFG_HASH} got={manifest.get('final_selected_cfg_hash')}"
        )

    phase_a_contract = manifest.get("phase_a_contract", {})
    fee_model_path = _resolve(str(phase_a_contract.get("fee_model_path", phase_a_dir / "fee_model.json")))
    metrics_def_path = _resolve(str(phase_a_contract.get("metrics_definition_path", phase_a_dir / "metrics_definition.md")))
    subset_path = _resolve(str(manifest.get("signal_subset_path", phase_c_dir / "signal_subset.csv")))
    split_path = _resolve(str(manifest.get("split_definition_path", phase_c_dir / "wf_split_definition.json")))
    signal_source_path = _resolve(str(manifest.get("signal_source_path", args.signal_source_csv)))
    fullscan_csv = _resolve(str(args.fullscan_trades_csv)) if str(args.fullscan_trades_csv).strip() else None

    expected_fee_hash = str(phase_a_contract.get("fee_model_sha256", EXPECTED_PHASEA_FEE_HASH))
    expected_metrics_hash = str(phase_a_contract.get("metrics_definition_sha256", EXPECTED_PHASEA_METRICS_HASH))
    expected_subset_hash = str(manifest.get("signal_subset_hash", EXPECTED_SIGNAL_SUBSET_HASH))
    expected_split_hash = str(manifest.get("split_definition_sha256", EXPECTED_SPLIT_HASH))

    fee_hash = _ensure_hash(fee_model_path, expected_fee_hash, "phase_a fee_model")
    metrics_hash = _ensure_hash(metrics_def_path, expected_metrics_hash, "phase_a metrics_definition")
    subset_hash = _sha256_signal_subset(pd.read_csv(subset_path))
    if subset_hash != expected_subset_hash or subset_hash != EXPECTED_SIGNAL_SUBSET_HASH:
        raise RuntimeError(
            f"Phase C subset hash mismatch: got={subset_hash} expected_manifest={expected_subset_hash} expected_const={EXPECTED_SIGNAL_SUBSET_HASH}"
        )
    split_hash = _ensure_hash(split_path, expected_split_hash, "phase_c split definition")
    if split_hash != EXPECTED_SPLIT_HASH:
        raise RuntimeError(f"Phase C split hash mismatch vs constant: {split_hash} != {EXPECTED_SPLIT_HASH}")

    for fp in [fee_model_path, metrics_def_path, subset_path, split_path, signal_source_path, phase_c_dir / "run_manifest.json"]:
        if fp.exists():
            shutil.copy2(fp, snap / fp.name)
    for fp in [
        PROJECT_ROOT / "scripts" / "phase_e2_sol_representative.py",
        PROJECT_ROOT / "scripts" / "phase_e_sol_consistency.py",
        PROJECT_ROOT / "scripts" / "backtest_exec_phasec_sol.py",
        PROJECT_ROOT / "src" / "execution" / "ga_exec_3m_opt.py",
    ]:
        if fp.exists():
            shutil.copy2(fp, snap / fp.name)

    # Canonical accounting contract (hard lock for E2 variants).
    contract = {
        "contract_id": "SOL_PHASEE2_CANONICAL_V1",
        "symbol": symbol,
        "initial_equity": float(args.initial_equity),
        "position_sizing": "fixed_fractional_risk_per_trade_compounding",
        "risk_per_trade": float(args.risk_per_trade),
        "fee_model_path": str(fee_model_path),
        "fee_model_sha256": fee_hash,
        "metrics_definition_path": str(metrics_def_path),
        "metrics_definition_sha256": metrics_hash,
        "entry_semantics": "next_3m_open_after_1h_signal_for_all_variants",
        "entry_semantics_detail": {
            "v2r_1h_reference": "uses baseline 3m entry timestamps/prices; exits evaluated with 1h control semantics",
            "v3r_exec_control": "same baseline 3m entry timestamps/prices; exits on 3m control path",
            "v4r_exec_phasec_best": "same baseline 3m entry timestamps/prices; exits on Phase C best path",
        },
        "pnl_aggregation_units": "net_return_per_trade_compounded_to_equity",
        "drawdown_convention": "negative_fraction_peak_to_trough",
        "expectancy_formula": "mean(pnl_net_pct) over valid filled trades",
        "cvar_5_formula": "mean of worst 5% pnl_net_pct over valid filled trades",
        "missing_data_handling": "missing slices tracked; invalid fills excluded from per-trade metrics",
        "no_trade_handling": "entry_rate and support explicitly reported; no silent skips",
        "split_aggregation_method": "weighted by signal count",
        "phase_a_contract": str(phase_a_dir),
        "phase_c_source": str(phase_c_dir),
    }
    _json_dump(run_dir / "accounting_contract.json", contract)

    compare_rows = pd.DataFrame(
        [
            {
                "dimension": "initial_equity",
                "fullscan_contract": 10000.0,
                "phasec_frozen_contract": 1.0,
                "phasee2_canonical_contract": float(args.initial_equity),
            },
            {
                "dimension": "position_sizing",
                "fullscan_contract": "native 1h backtester sizing",
                "phasec_frozen_contract": "fixed_fractional_risk_per_trade_compounding",
                "phasee2_canonical_contract": "fixed_fractional_risk_per_trade_compounding",
            },
            {
                "dimension": "fee_model",
                "fullscan_contract": "scan fee/slip params",
                "phasec_frozen_contract": f"phaseA fee model sha={fee_hash}",
                "phasee2_canonical_contract": f"phaseA fee model sha={fee_hash}",
            },
            {
                "dimension": "metrics_formula",
                "fullscan_contract": "scan summary formulas",
                "phasec_frozen_contract": f"phaseA metrics sha={metrics_hash}",
                "phasee2_canonical_contract": f"phaseA metrics sha={metrics_hash}",
            },
            {
                "dimension": "signal_universe",
                "fullscan_contract": "fullscan endogenous",
                "phasec_frozen_contract": "frozen phasec test subset (600)",
                "phasee2_canonical_contract": "deterministic representative subset from source",
            },
            {
                "dimension": "execution_horizon_hours",
                "fullscan_contract": "native 1h backtester",
                "phasec_frozen_contract": float(args.exec_horizon_hours),
                "phasee2_canonical_contract": float(args.exec_horizon_hours),
            },
        ]
    )
    compare_rows.to_csv(run_dir / "accounting_contract_compare.csv", index=False)
    (run_dir / "accounting_contract_compare.md").write_text(
        "\n".join(
            [
                "# Accounting Contract Compare",
                "",
                _markdown_table(compare_rows),
                "",
                "- Canonical contract is hard-locked for all Phase E2 variants.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Universe design and representative subset build.
    source_signals = _normalize_signals(pd.read_csv(signal_source_path))
    phasec_subset_all = _normalize_signals(pd.read_csv(subset_path))
    phasec_splits = _parse_splits(split_path)
    phasec_test_idx = _test_indices_from_splits(phasec_splits)
    phasec_test_subset = phasec_subset_all.iloc[phasec_test_idx].copy().reset_index(drop=True)

    # Keep representative pool constrained to the same source used by current execution pipeline.
    pool = source_signals.copy().reset_index(drop=True)
    if int(args.representative_size) <= 0:
        raise RuntimeError("representative_size must be > 0")
    rep_n = int(min(int(args.representative_size), len(pool)))
    if rep_n < 600:
        raise RuntimeError("representative_size must be >= 600 to improve over Phase C frozen subset.")

    pool["date_bucket"] = pd.to_datetime(pool["signal_time"], utc=True).dt.to_period("Q").astype(str)
    pool["vol_bucket"] = _vol_bucket(pool.get("atr_percentile_1h"))
    pool["trend_bucket"] = pool["trend_up_1h"].map(lambda v: "up" if _safe_float(v) >= 0.5 else "down").fillna("unknown")
    pool["stratum"] = (
        pool["date_bucket"].astype(str)
        + "|"
        + pool["vol_bucket"].astype(str)
        + "|"
        + pool["trend_bucket"].astype(str)
    )

    strata_before = pool["stratum"].value_counts().sort_index().to_dict()
    rep_subset = _stratified_sample(pool, n=rep_n, seed=int(args.seed), strata_col="stratum")
    strata_after = rep_subset["stratum"].value_counts().sort_index().to_dict()
    rep_subset = rep_subset.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rep_subset_hash = _sha256_signal_subset(rep_subset[["signal_id", "signal_time"]].copy())
    rep_subset[["signal_id", "signal_time", "tp_mult", "sl_mult", "atr_percentile_1h", "trend_up_1h"]].to_csv(
        run_dir / "representative_subset_signals.csv", index=False
    )
    (run_dir / "representative_subset_hash.txt").write_text(rep_subset_hash + "\n", encoding="utf-8")

    rep_splits = _build_splits(len(rep_subset), int(args.wf_splits))
    rep_split_lookup = _split_lookup(rep_subset, rep_splits)
    _json_dump(run_dir / "representative_subset_manifest.json", {
        "generated_utc": _utc_now().isoformat(),
        "symbol": symbol,
        "source_signal_csv": str(signal_source_path),
        "source_signal_csv_sha256": _sha256_file(signal_source_path),
        "sampling_method": "A_stratified_chronological_sampling",
        "seed": int(args.seed),
        "representative_size": int(rep_n),
        "bucket_definitions": {
            "date_bucket": "signal_time quarter",
            "vol_bucket": "atr_percentile_1h terciles",
            "trend_bucket": "trend_up_1h binary",
        },
        "stratum_counts_before": strata_before,
        "stratum_counts_after": strata_after,
        "selected_signal_count": int(len(rep_subset)),
        "selected_signal_ids_sha256": _sha256_text("\n".join(rep_subset["signal_id"].astype(str).tolist())),
        "representative_subset_hash": rep_subset_hash,
        "split_definition": rep_splits,
    })

    (run_dir / "universe_design.md").write_text(
        "\n".join(
            [
                "# Universe Design",
                "",
                "- Mode: SOL-only representative evaluation (no new alpha, no multi-coin).",
                "- Source signal file: `{}`".format(signal_source_path),
                "- Frozen Phase C subset reused for comparison only.",
                "- Representative sampling option: A) stratified chronological by quarter+vol_bucket+trend_bucket.",
                "- Deterministic seed: {}".format(int(args.seed)),
                "- Representative subset size: {}".format(int(rep_n)),
                "- Walkforward split count: {}".format(len(rep_splits)),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Distribution / representativeness diagnostics.
    feat_table = _load_market_feature_table(symbol=symbol)

    fullscan_ref_dir = _latest_phasee_consistency_dir(PROJECT_ROOT / "reports" / "execution_layer")
    if fullscan_csv is None:
        if fullscan_ref_dir is None:
            raise RuntimeError("No fullscan trades CSV provided and no PHASEE_SOL_CONSISTENCY_* run found.")
        fullscan_csv = fullscan_ref_dir / "trades_v1_1h_fullscan_reference.csv"
    if not fullscan_csv.exists():
        raise FileNotFoundError(f"Missing fullscan trades csv: {fullscan_csv}")
    fullscan_univ = pd.read_csv(fullscan_csv)
    fullscan_univ = fullscan_univ[["signal_id", "signal_time"]].copy()
    fullscan_univ["signal_id"] = fullscan_univ["signal_id"].astype(str)
    fullscan_univ["signal_time"] = pd.to_datetime(fullscan_univ["signal_time"], utc=True, errors="coerce")
    fullscan_univ = fullscan_univ.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).drop_duplicates(["signal_id", "signal_time"])

    u_full = _attach_universe_features(fullscan_univ.copy(), feat_table)
    u_phasec600 = _attach_universe_features(phasec_test_subset[["signal_id", "signal_time", "atr_percentile_1h", "trend_up_1h"]].copy(), feat_table)
    u_rep = _attach_universe_features(rep_subset[["signal_id", "signal_time", "atr_percentile_1h", "trend_up_1h"]].copy(), feat_table)

    dist_rep_vs_full = _distribution_compare(u_full, u_rep, "rep_vs_fullscan")
    dist_phasec_vs_full = _distribution_compare(u_full, u_phasec600, "phasec600_vs_fullscan")
    dist_rep_vs_full.to_csv(run_dir / "subset_vs_fullscan_distribution.csv", index=False)
    dist_rep_vs_phasec = _distribution_compare(u_phasec600, u_rep, "rep_vs_phasec600")
    dist_rep_vs_phasec.to_csv(run_dir / "subset_vs_phasec600_distribution.csv", index=False)
    regime_cmp = pd.concat([dist_rep_vs_full, dist_phasec_vs_full, dist_rep_vs_phasec], ignore_index=True)
    regime_cmp.to_csv(run_dir / "regime_distribution_compare.csv", index=False)

    key_dims = ["quarter", "vol_bucket", "trend_bucket"]
    rep_tvd_mean = float(
        dist_rep_vs_full[dist_rep_vs_full["dimension"].isin(key_dims)]
        .drop_duplicates(["dimension"])["tvd_dimension"]
        .mean()
    )
    phasec_tvd_mean = float(
        dist_phasec_vs_full[dist_phasec_vs_full["dimension"].isin(key_dims)]
        .drop_duplicates(["dimension"])["tvd_dimension"]
        .mean()
    )
    rep_cov = _coverage_ratio(u_full, u_rep, key_dims)
    phasec_cov = _coverage_ratio(u_full, u_phasec600, key_dims)
    rep_max_delta_key = float(
        dist_rep_vs_full[dist_rep_vs_full["dimension"].isin(key_dims)]["abs_delta"].max()
    )
    phasec_max_delta_key = float(
        dist_phasec_vs_full[dist_phasec_vs_full["dimension"].isin(key_dims)]["abs_delta"].max()
    )
    representative_enough = int(
        np.isfinite(rep_tvd_mean)
        and np.isfinite(phasec_tvd_mean)
        and rep_tvd_mean <= phasec_tvd_mean * 0.95
        and np.isfinite(rep_cov)
        and np.isfinite(phasec_cov)
        and rep_cov >= phasec_cov
        and np.isfinite(rep_max_delta_key)
        and np.isfinite(phasec_max_delta_key)
        and rep_max_delta_key <= phasec_max_delta_key
    )

    rep_lines = [
        "# Representativeness Report",
        "",
        f"- fullscan_universe_signals: {int(len(u_full))}",
        f"- phasec600_signals: {int(len(u_phasec600))}",
        f"- representative_signals: {int(len(u_rep))}",
        f"- key_dims: {', '.join(key_dims)}",
        f"- tvd_mean_rep_vs_full: {rep_tvd_mean:.6f}",
        f"- tvd_mean_phasec600_vs_full: {phasec_tvd_mean:.6f}",
        f"- coverage_rep_vs_full: {rep_cov:.6f}",
        f"- coverage_phasec600_vs_full: {phasec_cov:.6f}",
        f"- max_abs_delta_rep_vs_full(key_dims): {rep_max_delta_key:.6f}",
        f"- max_abs_delta_phasec600_vs_full(key_dims): {phasec_max_delta_key:.6f}",
        "",
        f"- representative_enough: {representative_enough}",
        "",
        "## Key-Dimension TVD Table (rep vs full)",
        "",
    ]
    rep_tvd_tbl = (
        dist_rep_vs_full[dist_rep_vs_full["dimension"].isin(key_dims)][["dimension", "tvd_dimension"]]
        .drop_duplicates()
        .sort_values("dimension")
    )
    rep_lines.append(_markdown_table(rep_tvd_tbl))
    rep_lines.append("")
    rep_lines.append("## Key-Dimension TVD Table (phasec600 vs full)")
    rep_lines.append("")
    phasec_tvd_tbl = (
        dist_phasec_vs_full[dist_phasec_vs_full["dimension"].isin(key_dims)][["dimension", "tvd_dimension"]]
        .drop_duplicates()
        .sort_values("dimension")
    )
    rep_lines.append(_markdown_table(phasec_tvd_tbl))
    (run_dir / "representativeness_report.md").write_text("\n".join(rep_lines) + "\n", encoding="utf-8")

    # Evaluate V2R / V3R / V4R on identical representative subset + canonical contract.
    fee = phasec_bt._load_fee_model(fee_model_path)
    rep_ids = set(rep_subset["signal_id"].astype(str).tolist())
    rep_signals_eval = rep_subset[["signal_id", "signal_time", "tp_mult", "sl_mult"]].copy()
    ga_args = ga_exec.build_arg_parser().parse_args([])
    ga_args.mode = "tight"
    ga_args.force_no_skip = 1
    ga_args.timeframe = "3m"
    ga_args.pre_buffer_hours = 6.0
    ga_args.exec_horizon_hours = float(args.exec_horizon_hours)
    ga_args.cache_dir = "data/processed/_exec_klines_cache"
    ga_args.max_fetch_retries = 8
    ga_args.retry_base_sleep = 0.5
    ga_args.retry_max_sleep = 30.0
    ga_args.fetch_pause_sec = 0.03
    ga_args.fee_bps_maker = float(fee.fee_bps_maker)
    ga_args.fee_bps_taker = float(fee.fee_bps_taker)
    ga_args.slippage_bps_limit = float(fee.slippage_bps_limit)
    ga_args.slippage_bps_market = float(fee.slippage_bps_market)
    ga_args.train_ratio = 0.7
    ga_args.wf_splits = 1
    ga_args.max_signals = 0
    ga_args.signal_order = "latest"
    ga_args.signals_dir = "data/signals"
    ga_args.execution_config = "configs/execution_configs.yaml"
    ga_args.hard_min_trades_overall = 0
    ga_args.hard_min_trade_frac_overall = 0.0
    ga_args.hard_min_trades_symbol = 0
    ga_args.hard_min_trade_frac_symbol = 0.0
    ga_args.hard_min_entry_rate_symbol = 0.0
    ga_args.hard_min_entry_rate_overall = 0.0
    ga_args.hard_max_missing_slice_rate = 1.0
    ga_args.hard_max_taker_share = 1.0
    ga_args.hard_max_median_fill_delay_min = 1e9
    ga_args.hard_max_p95_fill_delay_min = 1e9
    ga_args.tight_min_entry_rate_default = 0.0
    ga_args.tight_max_fill_delay_default = 1e9
    ga_args.tight_max_taker_share_default = 1.0

    full_for_ga = _normalize_signals(pd.read_csv(signal_source_path))
    bundle = ga_exec._build_bundle_for_symbol(
        symbol=symbol,
        signals_df=full_for_ga.copy(),
        signal_csv=signal_source_path,
        constraints={
            "min_entry_rate": 0.0,
            "max_taker_share": 1.0,
            "max_fill_delay_min": 1e9,
            "min_median_entry_improvement_bps": -9999.0,
        },
        args=ga_args,
    )
    bundle.splits = [{"split_id": 0, "train_start": 0, "train_end": 0, "test_start": 0, "test_end": int(len(bundle.contexts))}]

    phasec_best_genome = ga_exec._repair_genome(
        {
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
            "tp_mult": float(args.phasec_tp_mult),
            "sl_mult": float(args.phasec_sl_mult),
            "time_stop_min": int(args.phasec_time_stop_min),
            "break_even_enabled": int(args.phasec_break_even_enabled),
            "break_even_trigger_r": float(args.phasec_break_even_trigger_r),
            "break_even_offset_bps": float(args.phasec_break_even_offset_bps),
            "trailing_enabled": 0,
            "trail_start_r": 2.0,
            "trail_step_bps": 50.0,
            "partial_take_enabled": int(args.phasec_partial_take_enabled),
            "partial_take_r": float(args.phasec_partial_take_r),
            "partial_take_pct": float(args.phasec_partial_take_pct),
            "skip_if_vol_gate": 0,
            "use_signal_quality_gate": 0,
            "min_signal_quality_gate": 0.0,
            "cooldown_min": 0,
        },
        mode="tight",
    )
    e = ga_exec._evaluate_genome(phasec_best_genome, [bundle], ga_args, detailed=True)
    rows_full = e.get("signal_rows_df", pd.DataFrame()).copy()
    if rows_full.empty:
        raise RuntimeError("GA evaluation returned empty signal rows for representative evaluation.")
    rows_full["signal_id"] = rows_full["signal_id"].astype(str)
    rows_full["signal_time"] = pd.to_datetime(rows_full["signal_time"], utc=True, errors="coerce")
    rows_full = rows_full[rows_full["signal_id"].isin(rep_ids)].copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)

    t_v3r = _ga_rows_to_trades(
        rows_full,
        mode="baseline",
        fee=fee,
        exec_sl_mult=1.0,
        split_lookup=rep_split_lookup,
    )
    t_v3r["variant"] = "V3R_EXEC_3M_PHASEC_CONTROL"
    t_v4r = _ga_rows_to_trades(
        rows_full,
        mode="exec",
        fee=fee,
        exec_sl_mult=float(args.phasec_sl_mult),
        split_lookup=rep_split_lookup,
    )
    t_v4r["variant"] = "V4R_EXEC_3M_PHASEC_BEST"
    entries_v2 = pd.DataFrame(
        {
            "signal_id": rows_full["signal_id"].astype(str),
            "signal_time": pd.to_datetime(rows_full["signal_time"], utc=True, errors="coerce"),
            "split_id": rows_full["signal_id"].astype(str).map(rep_split_lookup).fillna(-1).astype(int),
            "signal_tp_mult": pd.to_numeric(rows_full.get("signal_tp_mult"), errors="coerce"),
            "signal_sl_mult": pd.to_numeric(rows_full.get("signal_sl_mult"), errors="coerce"),
            "entry_time": pd.to_datetime(rows_full.get("baseline_entry_time"), utc=True, errors="coerce"),
            "entry_price": pd.to_numeric(rows_full.get("baseline_entry_price"), errors="coerce"),
        }
    )
    t_v2r = _simulate_1h_from_entries(
        entries_df=entries_v2,
        symbol=symbol,
        fee=fee,
        exec_horizon_hours=float(args.exec_horizon_hours),
    )
    t_v2r["variant"] = "V2R_1H_REFERENCE_CONTROL"

    # Persist trade tables for audit trace.
    t_v2r.to_csv(run_dir / "trades_v2r_1h_reference_control.csv", index=False)
    t_v3r.to_csv(run_dir / "trades_v3r_exec_3m_phasec_control.csv", index=False)
    t_v4r.to_csv(run_dir / "trades_v4r_exec_3m_phasec_best.csv", index=False)

    signals_total = int(len(rep_subset))
    init_eq = float(args.initial_equity)
    risk_pt = float(args.risk_per_trade)
    eq2, m2, _ = phasec_bt._compute_equity_curve(
        t_v2r, signals_total=signals_total, initial_equity=init_eq, risk_per_trade=risk_pt
    )
    eq3, m3, _ = phasec_bt._compute_equity_curve(
        t_v3r, signals_total=signals_total, initial_equity=init_eq, risk_per_trade=risk_pt
    )
    eq4, m4, _ = phasec_bt._compute_equity_curve(
        t_v4r, signals_total=signals_total, initial_equity=init_eq, risk_per_trade=risk_pt
    )
    eq2.to_csv(run_dir / "equity_curve_v2r.csv", index=False)
    eq3.to_csv(run_dir / "equity_curve_v3r.csv", index=False)
    eq4.to_csv(run_dir / "equity_curve_v4r.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            {"variant": "V2R_1H_REFERENCE_CONTROL", **m2},
            {"variant": "V3R_EXEC_3M_PHASEC_CONTROL", **m3},
            {"variant": "V4R_EXEC_3M_PHASEC_BEST", **m4},
        ]
    )
    metrics_df["exit_reason_distribution"] = metrics_df["exit_reason_distribution"].map(lambda d: json.dumps(d, sort_keys=True))
    metrics_df.to_csv(run_dir / "metrics_by_variant.csv", index=False)

    comp_df = _comparison_matrix(metrics_df)
    comp_df.to_csv(run_dir / "comparison_matrix.csv", index=False)

    split_df = _split_metrics(
        {
            "V2R_1H_REFERENCE_CONTROL": t_v2r,
            "V3R_EXEC_3M_PHASEC_CONTROL": t_v3r,
            "V4R_EXEC_3M_PHASEC_BEST": t_v4r,
        },
        rep_subset,
        rep_splits,
        initial_equity=init_eq,
        risk_per_trade=risk_pt,
    )
    split_df.to_csv(run_dir / "walkforward_results_by_split.csv", index=False)

    # Risk rollups (variant-row schema, SOL-only).
    rr_rows: List[Dict[str, Any]] = []
    for _, r in metrics_df.iterrows():
        rr_rows.append(
            {
                "variant": r["variant"],
                "scope": "overall",
                "symbols": 1,
                "signals_total": int(r["signals_total"]),
                "trades_total": int(r["trades_total"]),
                "mean_expectancy_net": float(r["expectancy_net"]),
                "total_return": float(r["total_return"]),
                "pnl_net_sum": float(init_eq * (1.0 + float(r["total_return"])) - init_eq),
                "max_drawdown_pct": float(r["max_drawdown_pct"]),
                "cvar_5": float(r["cvar_5"]),
                "win_rate": float(r["win_rate"]),
                "profit_factor": float(r["profit_factor"]),
                "taker_share": float(r["taker_share"]),
                "median_fill_delay_min": float(r["median_fill_delay_min"]),
                "p95_fill_delay_min": float(r["p95_fill_delay_min"]),
            }
        )
    rr_overall = pd.DataFrame(rr_rows)
    rr_overall.to_csv(run_dir / "risk_rollup_overall.csv", index=False)
    rr_by_symbol = rr_overall.copy()
    rr_by_symbol.insert(1, "symbol", symbol)
    rr_by_symbol.to_csv(run_dir / "risk_rollup_by_symbol.csv", index=False)

    # Gate decisions.
    idx = {r["variant"]: r for _, r in metrics_df.iterrows()}
    v2 = idx["V2R_1H_REFERENCE_CONTROL"]
    v3 = idx["V3R_EXEC_3M_PHASEC_CONTROL"]
    v4 = idx["V4R_EXEC_3M_PHASEC_BEST"]

    split_counts = split_df.groupby(["variant", "split_id"], as_index=False)["trades_total"].sum()
    min_split_support = int(split_counts["trades_total"].min()) if not split_counts.empty else 0
    min_split_threshold = int(args.min_split_support)

    contract_gates = {
        "single_contract_locked": 1,
        "all_variants_use_same_subset": int(
            int(v2["signals_total"]) == int(v3["signals_total"]) == int(v4["signals_total"]) == int(rep_n)
        ),
        "all_variants_use_same_splits": int(
            split_df.groupby("variant")["split_id"].nunique().min() == split_df.groupby("variant")["split_id"].nunique().max()
            if not split_df.empty
            else 0
        ),
        "all_variants_use_same_fee_metrics": int(
            fee_hash == EXPECTED_PHASEA_FEE_HASH
            and metrics_hash == EXPECTED_PHASEA_METRICS_HASH
            and np.isfinite(_safe_float(v2["total_fees_paid"]))
            and np.isfinite(_safe_float(v3["total_fees_paid"]))
            and np.isfinite(_safe_float(v4["total_fees_paid"]))
        ),
    }
    representativeness_gates = {
        "representative_enough": int(representative_enough),
        "no_catastrophic_bucket_undercoverage": int(
            np.isfinite(rep_max_delta_key)
            and np.isfinite(phasec_max_delta_key)
            and rep_max_delta_key <= min(float(args.max_bucket_delta), phasec_max_delta_key)
        ),
        "minimum_split_support_ok": int(min_split_support >= min_split_threshold),
    }
    phasec_best_better = int(
        (
            _safe_float(v4["expectancy_net"]) >= _safe_float(v3["expectancy_net"])
        )
        or (
            _safe_float(v4["max_drawdown_pct"]) >= _safe_float(v3["max_drawdown_pct"])
            and _safe_float(v4["cvar_5"]) >= _safe_float(v3["cvar_5"])
        )
    )
    exec_not_unexpectedly_degraded = int(
        np.isfinite(_safe_float(v4["expectancy_net"]))
        and np.isfinite(_safe_float(v3["expectancy_net"]))
        and (_safe_float(v4["expectancy_net"]) - _safe_float(v3["expectancy_net"]) >= -1e-9)
    )

    if (_safe_float(v2["expectancy_net"]) < 0.0) and (_safe_float(v3["expectancy_net"]) < 0.0) and (_safe_float(v4["expectancy_net"]) < 0.0):
        root_label = "likely_signal_quality_issue"
    elif (_safe_float(v3["expectancy_net"]) < _safe_float(v2["expectancy_net"])) and (_safe_float(v4["expectancy_net"]) < _safe_float(v2["expectancy_net"])):
        root_label = "likely_execution_issue"
    elif (not np.isfinite(_safe_float(v2["expectancy_net"]))) or (not np.isfinite(_safe_float(v3["expectancy_net"]))) or (not np.isfinite(_safe_float(v4["expectancy_net"]))):
        root_label = "insufficient_evidence"
    else:
        root_label = "mixed"

    strategy_gates = {
        "phasec_best_ge_phasec_control": int(phasec_best_better),
        "execution_consistency_no_unexpected_degrade": int(exec_not_unexpectedly_degraded),
        "classification_label": root_label,
    }

    harness_trustworthy = int(
        min(contract_gates.values()) == 1 and min(representativeness_gates.values()) == 1
    )
    if harness_trustworthy != 1:
        phase_class = "FAIL_HARNESS_NOT_TRUSTWORTHY"
    elif root_label == "likely_signal_quality_issue":
        phase_class = "PASS_TRUSTWORTHY_HARNESS_PAUSE_SOL_SIGNAL_LAYER"
    else:
        phase_class = "PASS_TRUSTWORTHY_HARNESS_CONTINUE_SOL"

    gates = {
        "contract_integrity_gates": contract_gates,
        "representativeness_gates": representativeness_gates,
        "strategy_interpretation_gates": strategy_gates,
        "thresholds": {
            "min_split_support": min_split_threshold,
            "max_bucket_delta": float(args.max_bucket_delta),
            "relative_tvd_improve_required": 0.05,
        },
        "harness_trustworthy": harness_trustworthy,
        "phase_result_classification": phase_class,
    }
    _json_dump(run_dir / "pass_fail_gates.json", gates)

    # Decision and phase result.
    if phase_class == "PASS_TRUSTWORTHY_HARNESS_CONTINUE_SOL":
        next_prompt = (
            "Phase F (SOL-only): run entry-layer optimization on this exact canonical contract and representative subset, "
            "keeping Phase C best exit fixed as baseline reference."
        )
    elif phase_class == "PASS_TRUSTWORTHY_HARNESS_PAUSE_SOL_SIGNAL_LAYER":
        next_prompt = (
            "Signal-layer remediation (SOL-only): keep canonical contract fixed, diagnose and repair 1h signal quality "
            "on representative subsets before any new execution optimization."
        )
    else:
        next_prompt = (
            "Phase E2 rerun: fix contract/split/subset integrity gate failures, then rerun V2R/V3R/V4R on the same frozen contract."
        )

    decision_lines = [
        "# Phase E2 Decision",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Result classification: **{phase_class}**",
        "",
        "## Questions",
        "",
        f"1) Harness trustworthy? {'YES' if harness_trustworthy == 1 else 'NO'}",
        f"2) SOL 1h signal quality viable on representative contract? {'NO' if root_label == 'likely_signal_quality_issue' else 'UNCLEAR/YES'}",
        f"3) Phase C best adds value vs control? {'YES' if phasec_best_better == 1 else 'NO'}",
        "4) Next phase recommendation:",
        f"- {next_prompt}",
        "",
        "## Headline Metrics",
        "",
        f"- V2R expectancy/return/dd: {_safe_float(v2['expectancy_net']):.6f} / {_safe_float(v2['total_return']):.6f} / {_safe_float(v2['max_drawdown_pct']):.6f}",
        f"- V3R expectancy/return/dd: {_safe_float(v3['expectancy_net']):.6f} / {_safe_float(v3['total_return']):.6f} / {_safe_float(v3['max_drawdown_pct']):.6f}",
        f"- V4R expectancy/return/dd: {_safe_float(v4['expectancy_net']):.6f} / {_safe_float(v4['total_return']):.6f} / {_safe_float(v4['max_drawdown_pct']):.6f}",
        "",
        "## Representativeness",
        "",
        f"- representative_enough: {representative_enough}",
        f"- tvd_mean_rep_vs_full (key dims): {rep_tvd_mean:.6f}",
        f"- tvd_mean_phasec600_vs_full (key dims): {phasec_tvd_mean:.6f}",
        f"- coverage_rep_vs_full (key dims): {rep_cov:.6f}",
        f"- min_split_support: {min_split_support}",
        "",
        "## Recommended Next Prompt",
        "",
        next_prompt,
    ]
    (run_dir / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    # Practical deployment gate memo.
    deploy_gate = {
        "v4_expectancy_positive": int(_safe_float(v4["expectancy_net"]) > 0.0),
        "v4_total_return_positive": int(_safe_float(v4["total_return"]) > 0.0),
        "v4_maxdd_better_than_-0_35": int(_safe_float(v4["max_drawdown_pct"]) > -0.35),
        "v4_cvar5_better_than_-0_0015": int(_safe_float(v4["cvar_5"]) > -0.0015),
        "v4_ge_v3_expectancy": int(_safe_float(v4["expectancy_net"]) >= _safe_float(v3["expectancy_net"])),
    }
    deploy_decision = "DEPLOY_CANDIDATE" if min(deploy_gate.values()) == 1 else "NO_DEPLOY"
    practical_lines = [
        "# Practical Decision",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Absolute gate decision: **{deploy_decision}**",
        "",
        "## Variant Metrics",
        "",
        f"- V2R expectancy/return/maxDD/cvar5/PF/win_rate/fees: {_safe_float(v2['expectancy_net']):.6f} / {_safe_float(v2['total_return']):.6f} / {_safe_float(v2['max_drawdown_pct']):.6f} / {_safe_float(v2['cvar_5']):.6f} / {_safe_float(v2['profit_factor']):.6f} / {_safe_float(v2['win_rate']):.6f} / {_safe_float(v2['total_fees_paid']):.6f}",
        f"- V3R expectancy/return/maxDD/cvar5/PF/win_rate/fees: {_safe_float(v3['expectancy_net']):.6f} / {_safe_float(v3['total_return']):.6f} / {_safe_float(v3['max_drawdown_pct']):.6f} / {_safe_float(v3['cvar_5']):.6f} / {_safe_float(v3['profit_factor']):.6f} / {_safe_float(v3['win_rate']):.6f} / {_safe_float(v3['total_fees_paid']):.6f}",
        f"- V4R expectancy/return/maxDD/cvar5/PF/win_rate/fees: {_safe_float(v4['expectancy_net']):.6f} / {_safe_float(v4['total_return']):.6f} / {_safe_float(v4['max_drawdown_pct']):.6f} / {_safe_float(v4['cvar_5']):.6f} / {_safe_float(v4['profit_factor']):.6f} / {_safe_float(v4['win_rate']):.6f} / {_safe_float(v4['total_fees_paid']):.6f}",
        "",
        "## Relative Deltas",
        "",
    ]
    for _, rr in comp_df.iterrows():
        practical_lines.append(
            f"- {rr['metric']}: v3-v2={_safe_float(rr['delta_v3r_minus_v2r']):.6f}, v4-v3={_safe_float(rr['delta_v4r_minus_v3r']):.6f}, v4-v2={_safe_float(rr['delta_v4r_minus_v2r']):.6f}"
        )
    practical_lines.extend(["", "## Absolute Deploy Gate Checks", ""])
    for k, v in deploy_gate.items():
        practical_lines.append(f"- {k}: {v}")
    (run_dir / "practical_decision.md").write_text("\n".join(practical_lines) + "\n", encoding="utf-8")

    (run_dir / "phase_result.md").write_text(
        "\n".join(
            [
                "Phase: E2 SOL contract-locked representative harness",
                f"Timestamp UTC: {_utc_now().isoformat()}",
                f"Classification: {phase_class}",
                f"Harness trustworthy: {harness_trustworthy}",
                f"Representative enough: {representative_enough}",
                f"Root label: {root_label}",
                f"Artifacts: {run_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Repro + git status + manifest.
    repro_lines = [
        "# Repro",
        "",
        "```bash",
        f"cd {PROJECT_ROOT}",
        "python3 scripts/phase_e2_sol_representative.py "
        f"--symbol {symbol} "
        f"--phase-c-dir {phase_c_dir} "
        f"--phase-a-contract-dir {phase_a_dir} "
        f"--signal-source-csv {signal_source_path} "
        f"--outdir {args.outdir} "
        f"--representative-size {rep_n} "
        f"--wf-splits {int(args.wf_splits)} "
        f"--seed {int(args.seed)}",
        "```",
    ]
    (run_dir / "repro.md").write_text("\n".join(repro_lines) + "\n", encoding="utf-8")
    try:
        gs = subprocess.check_output(["git", "status", "--short"], cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.STDOUT)
    except Exception as ex:
        gs = f"git status unavailable: {ex}"
    (run_dir / "git_status.txt").write_text(gs, encoding="utf-8")

    _json_dump(
        run_dir / "run_manifest.json",
        {
            "generated_utc": _utc_now().isoformat(),
            "symbol": symbol,
            "phase_c_dir": str(phase_c_dir),
            "phase_a_contract_dir": str(phase_a_dir),
            "signal_source_csv": str(signal_source_path),
            "signal_source_sha256": _sha256_file(signal_source_path),
            "phasea_fee_sha256": fee_hash,
            "phasea_metrics_sha256": metrics_hash,
            "phasec_subset_sha256": subset_hash,
            "phasec_split_sha256": split_hash,
            "representative_subset_sha256": rep_subset_hash,
            "seed": int(args.seed),
            "representative_size": int(rep_n),
            "wf_splits": int(len(rep_splits)),
            "split_definition": rep_splits,
            "contract_id": contract["contract_id"],
            "phase_result_classification": phase_class,
            "harness_trustworthy": harness_trustworthy,
            "representative_enough": representative_enough,
            "root_label": root_label,
            "fullscan_reference_csv": str(fullscan_csv),
        },
    )

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase E2 SOL representative, contract-locked evaluation harness.")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--phase-c-dir", default="reports/execution_layer/PHASEC_SOL_20260221_231430")
    ap.add_argument("--phase-a-contract-dir", default="reports/execution_layer/BASELINE_AUDIT_20260221_214310")
    ap.add_argument("--signal-source-csv", default="data/signals/SOLUSDT_signals_1h.csv")
    ap.add_argument("--fullscan-trades-csv", default="")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--representative-size", type=int, default=1200)
    ap.add_argument("--wf-splits", type=int, default=5)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--initial-equity", type=float, default=1.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)
    ap.add_argument("--max-bucket-delta", type=float, default=0.90)
    ap.add_argument("--min-split-support", type=int, default=40)

    ap.add_argument("--phasec-tp-mult", type=float, default=1.0)
    ap.add_argument("--phasec-sl-mult", type=float, default=0.75)
    ap.add_argument("--phasec-time-stop-min", type=int, default=720)
    ap.add_argument("--phasec-break-even-enabled", type=int, default=0)
    ap.add_argument("--phasec-break-even-trigger-r", type=float, default=0.5)
    ap.add_argument("--phasec-break-even-offset-bps", type=float, default=0.0)
    ap.add_argument("--phasec-partial-take-enabled", type=int, default=0)
    ap.add_argument("--phasec-partial-take-r", type=float, default=0.8)
    ap.add_argument("--phasec-partial-take-pct", type=float, default=0.25)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    g = json.loads((out / "pass_fail_gates.json").read_text(encoding="utf-8"))
    cls = str(g.get("phase_result_classification", "UNKNOWN"))
    print(str(out))
    print(cls)


if __name__ == "__main__":
    main()
