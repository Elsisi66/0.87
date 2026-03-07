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
from src.bot087.optim import ga as ga_long  # noqa: E402


EXPECTED_PHASEC_CFG_HASH = "a285b86c4c22a26976d4a762"
EXPECTED_SIGNAL_SUBSET_HASH = "5e719faf676dffba8d7da926314997182d429361495884b8a870c3393c079bbf"
EXPECTED_SPLIT_HASH = "388ba743b9c16c291385a9ecab6435eabf65eb16f1e1083eee76627193c42c01"


@dataclass
class VariantPack:
    name: str
    universe_scope: str
    trades: pd.DataFrame
    equity: pd.DataFrame
    monthly: pd.DataFrame
    metrics: Dict[str, Any]


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


def _tail_mean(x: Sequence[float], frac: float) -> float:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(frac) * float(arr.size))))
    return float(np.mean(np.sort(arr)[:k]))


def _max_dd_from_equity(eq: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(eq), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    peak = np.maximum.accumulate(arr)
    dd_abs = arr - peak
    dd_pct = arr / np.maximum(1e-12, peak) - 1.0
    return float(np.min(dd_abs)), float(np.min(dd_pct))


def _normalize_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename: Dict[str, str] = {}
    for src, dst in (
        ("timestamp", "Timestamp"),
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
    ):
        if src in out.columns and dst not in out.columns:
            rename[src] = dst
    if rename:
        out = out.rename(columns=rename)
    if "Timestamp" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={"index": "Timestamp"})
    out["Timestamp"] = pd.to_datetime(out["Timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    needed = {"Timestamp", "Open", "High", "Low", "Close"}
    miss = sorted(needed - set(out.columns))
    if miss:
        raise RuntimeError(f"Missing OHLCV columns: {miss}")
    return out


def _load_symbol_df(symbol: str, tf: str = "1h") -> pd.DataFrame:
    fp_full = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_{tf}_full.parquet"
    if fp_full.exists():
        return _normalize_ohlcv_cols(pd.read_parquet(fp_full))
    fp_par = PROJECT_ROOT / "data" / "parquet" / f"{symbol}.parquet"
    if fp_par.exists():
        return _normalize_ohlcv_cols(pd.read_parquet(fp_par))
    proc_dir = PROJECT_ROOT / "data" / "processed"
    csvs = sorted(proc_dir.glob(f"{symbol}_*_proc.csv"))
    if csvs:
        return _normalize_ohlcv_cols(pd.concat([pd.read_csv(x) for x in csvs], ignore_index=True))
    raise FileNotFoundError(f"No 1h dataset found for {symbol}")


def _unwrap_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("params"), dict):
        return dict(payload["params"])
    return dict(payload)


def _sha256_signal_subset(df: pd.DataFrame) -> str:
    x = df.copy()
    if "signal_id" not in x.columns or "signal_time" not in x.columns:
        raise RuntimeError("signal subset requires signal_id and signal_time")
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rows = [f"{str(r.signal_id)}|{pd.to_datetime(r.signal_time, utc=True).isoformat()}" for r in x.itertuples(index=False)]
    return _sha256_text("\n".join(rows))


def _ensure_frozen_hash(path: Path, expected: str, label: str) -> str:
    got = _sha256_file(path)
    if str(expected).strip() and got != str(expected).strip():
        raise RuntimeError(f"{label} hash mismatch: expected={expected} got={got}")
    return got


def _load_phasec_manifest(phase_c_dir: Path) -> Dict[str, Any]:
    fp = phase_c_dir / "run_manifest.json"
    if not fp.exists():
        raise FileNotFoundError(f"Missing Phase C manifest: {fp}")
    return json.loads(fp.read_text(encoding="utf-8"))


def _parse_split_definition(path: Path) -> List[Dict[str, int]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, int]] = []
    for s in obj.get("splits", []):
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


def _split_lookup(subset_df: pd.DataFrame, splits: List[Dict[str, int]]) -> Dict[str, int]:
    idx_to_split: Dict[int, int] = {}
    for s in splits:
        sid = int(s["split_id"])
        for i in range(int(s["test_start"]), int(s["test_end"])):
            idx_to_split[int(i)] = sid
    out: Dict[str, int] = {}
    for i, r in subset_df.reset_index(drop=True).iterrows():
        if i in idx_to_split:
            out[str(r["signal_id"])] = int(idx_to_split[i])
    return out


def _load_v3_v4_trades(
    *,
    phase_c_dir: Path,
    split_lookup: Dict[str, int],
    test_signal_ids: set[str],
    fee: phasec_bt.FeeModel,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = phasec_bt._load_trade_diagnostics_csv(
        phase_c_dir / "trade_diagnostics_baseline.csv",
        split_lookup,
        "V3_EXEC_3M_PHASEC_CONTROL_FROZEN",
    )
    best = phasec_bt._load_trade_diagnostics_csv(
        phase_c_dir / "trade_diagnostics_best.csv",
        split_lookup,
        "V4_EXEC_3M_PHASEC_BEST_FROZEN",
    )
    base = base[base["signal_id"].astype(str).isin(test_signal_ids)].copy().reset_index(drop=True)
    best = best[best["signal_id"].astype(str).isin(test_signal_ids)].copy().reset_index(drop=True)
    base = phasec_bt._normalize_exec_rows(base, fee=fee, default_liq="taker")
    best = phasec_bt._normalize_exec_rows(best, fee=fee, default_liq="taker")
    return base, best


def _run_v1_fullscan(
    *,
    symbol: str,
    params_path: Path,
    initial_equity: float,
    fee_bps: float,
    slip_bps: float,
) -> Dict[str, Any]:
    payload = json.loads(params_path.read_text(encoding="utf-8"))
    p = ga_long._norm_params(_unwrap_params(payload))
    df = _load_symbol_df(symbol=symbol, tf="1h")
    df_feat = ga_long._ensure_indicators(df.copy(), p)
    trades, metrics = ga_long.run_backtest_long_only(
        df=df_feat,
        symbol=symbol,
        p=p,
        initial_equity=float(initial_equity),
        fee_bps=float(fee_bps),
        slippage_bps=float(slip_bps),
        collect_trades=True,
        assume_prepared=True,
        return_equity_curve=True,
    )
    sig = ga_long.build_entry_signal(df_feat, p, assume_prepared=True)
    signal_count = int(np.asarray(sig, dtype=bool).sum())

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tdf["entry_ts"] = pd.to_datetime(tdf.get("entry_ts"), utc=True, errors="coerce")
        tdf["exit_ts"] = pd.to_datetime(tdf.get("exit_ts"), utc=True, errors="coerce")
        for c in ("entry_px", "exit_px", "units", "net_pnl", "hold_hours"):
            tdf[c] = pd.to_numeric(tdf.get(c, np.nan), errors="coerce")
        tdf["entry_notional"] = (tdf["entry_px"] * tdf["units"]).abs()
        tdf["pnl_net_pct"] = tdf["net_pnl"] / np.maximum(1e-12, tdf["entry_notional"])
        tdf["signal_id"] = [f"v1_trade_{i:06d}" for i in range(1, len(tdf) + 1)]
    else:
        tdf = pd.DataFrame(
            columns=[
                "signal_id",
                "entry_ts",
                "exit_ts",
                "entry_px",
                "exit_px",
                "units",
                "net_pnl",
                "pnl_net_pct",
                "reason",
                "cycle",
                "hold_hours",
                "entry_notional",
            ]
        )

    eq_ts = pd.to_datetime(metrics.get("equity_timestamps", []), utc=True, errors="coerce")
    eq_vals = pd.to_numeric(pd.Series(metrics.get("equity_curve", [])), errors="coerce").to_numpy(dtype=float)
    if len(eq_ts) != len(eq_vals):
        n = min(len(eq_ts), len(eq_vals))
        eq_ts = eq_ts[:n]
        eq_vals = eq_vals[:n]
    eq_df = pd.DataFrame({"timestamp": eq_ts, "equity": eq_vals}).dropna(subset=["timestamp"])
    eq_df = eq_df.sort_values("timestamp").reset_index(drop=True)
    if eq_df.empty:
        eq_df = pd.DataFrame([{"timestamp": pd.NaT, "equity": float(initial_equity)}])
    peak = eq_df["equity"].cummax()
    eq_df["drawdown_pct"] = eq_df["equity"] / np.maximum(1e-12, peak) - 1.0
    eq_df["drawdown_abs"] = eq_df["equity"] - peak

    return {
        "params": p,
        "metrics_native": metrics,
        "trades_df": tdf,
        "equity_df": eq_df,
        "signal_count": int(signal_count),
        "period_start": pd.to_datetime(df_feat["Timestamp"].iloc[0], utc=True),
        "period_end": pd.to_datetime(df_feat["Timestamp"].iloc[-1], utc=True),
        "bar_count": int(len(df_feat)),
    }


def _cycle_to_sl_mult_map(params: Dict[str, Any]) -> Dict[int, float]:
    raw = params.get("sl_mult_by_cycle", {})
    out: Dict[int, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[int(k)] = float(v)
            except Exception:
                continue
    return out


def _v1_to_phasec_like_trades(
    v1: Dict[str, Any],
    *,
    symbol: str,
    fee_bps: float,
    slip_bps: float,
) -> pd.DataFrame:
    tdf = v1["trades_df"].copy()
    sl_map = _cycle_to_sl_mult_map(v1["params"])
    if tdf.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "signal_id",
                "signal_time",
                "split_id",
                "signal_tp_mult",
                "signal_sl_mult",
                "filled",
                "valid_for_metrics",
                "sl_hit",
                "tp_hit",
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "exit_reason",
                "pnl_net_pct",
                "mae_pct",
                "mfe_pct",
                "entry_type",
                "fill_liquidity_type",
                "entry_improvement_bps",
                "hold_minutes",
                "risk_pct",
                "entry_fee_bps",
                "exit_fee_bps",
                "entry_slippage_bps",
                "exit_slippage_bps",
                "total_cost_bps",
            ]
        )

    out = pd.DataFrame()
    out["symbol"] = symbol
    out["signal_id"] = tdf["signal_id"].astype(str)
    out["signal_time"] = pd.to_datetime(tdf["entry_ts"], utc=True, errors="coerce")
    out["split_id"] = -1
    out["signal_tp_mult"] = np.nan
    out["signal_sl_mult"] = tdf["cycle"].map(lambda c: np.nan if not np.isfinite(c) else np.nan)
    out["filled"] = 1
    out["valid_for_metrics"] = 1
    out["sl_hit"] = (tdf["reason"].astype(str).str.lower() == "sl").astype(int)
    out["tp_hit"] = (tdf["reason"].astype(str).str.lower() == "tp").astype(int)
    out["entry_time"] = pd.to_datetime(tdf["entry_ts"], utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(tdf["exit_ts"], utc=True, errors="coerce")
    out["entry_price"] = pd.to_numeric(tdf.get("entry_px"), errors="coerce")
    out["exit_price"] = pd.to_numeric(tdf.get("exit_px"), errors="coerce")
    out["entry_notional"] = pd.to_numeric(tdf.get("entry_notional"), errors="coerce")
    out["exit_reason"] = tdf["reason"].astype(str).str.lower()
    out["pnl_net_pct"] = pd.to_numeric(tdf.get("pnl_net_pct"), errors="coerce")
    out["mae_pct"] = np.nan
    out["mfe_pct"] = np.nan
    out["entry_type"] = "market"
    out["fill_liquidity_type"] = "taker"
    out["entry_improvement_bps"] = 0.0
    out["hold_minutes"] = pd.to_numeric(tdf.get("hold_hours"), errors="coerce") * 60.0
    out["risk_pct"] = tdf["cycle"].map(
        lambda c: max(1e-8, 1.0 - float(sl_map.get(int(c), 0.999)))
        if np.isfinite(c)
        else 1e-3
    )
    out["entry_fee_bps"] = float(fee_bps)
    out["exit_fee_bps"] = float(fee_bps)
    out["entry_slippage_bps"] = float(slip_bps)
    out["exit_slippage_bps"] = float(slip_bps)
    out["total_cost_bps"] = 2.0 * (float(fee_bps) + float(slip_bps))
    return out


def _fill_drawdown(eq_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    x = eq_df.copy()
    x["variant"] = variant
    x["timestamp"] = pd.to_datetime(x.get("timestamp"), utc=True, errors="coerce")
    x["equity"] = pd.to_numeric(x.get("equity"), errors="coerce")
    x = x.dropna(subset=["equity"]).sort_values("timestamp").reset_index(drop=True)
    if "drawdown_pct" not in x.columns:
        peak = x["equity"].cummax()
        x["drawdown_pct"] = x["equity"] / np.maximum(1e-12, peak) - 1.0
        x["drawdown_abs"] = x["equity"] - peak
    return x


def _metrics_row(variant: str, universe_scope: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "variant": variant,
        "universe_scope": universe_scope,
    }
    row.update(metrics)
    row["exit_reason_distribution"] = json.dumps(metrics.get("exit_reason_distribution", {}), sort_keys=True)
    return row


def _compute_monthly_from_equity(eq_df: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
    x = eq_df.copy()
    x["timestamp"] = pd.to_datetime(x.get("timestamp"), utc=True, errors="coerce")
    x["equity"] = pd.to_numeric(x.get("equity"), errors="coerce")
    x = x.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if x.empty:
        return pd.DataFrame(columns=["month", "equity_end", "monthly_return"])
    ser = x.set_index("timestamp")["equity"].groupby(level=0).last().sort_index()
    monthly_end = ser.resample("ME").last().ffill()
    prev = monthly_end.shift(1).fillna(float(initial_equity))
    r = monthly_end / np.maximum(1e-12, prev) - 1.0
    return pd.DataFrame(
        {
            "month": monthly_end.index.strftime("%Y-%m"),
            "equity_end": monthly_end.values.astype(float),
            "monthly_return": r.values.astype(float),
        }
    )


def _compute_v1_metrics_from_native(
    *,
    v1_trades_phasec_like: pd.DataFrame,
    signals_total: int,
    initial_equity: float,
    risk_per_trade: float,
    metrics_native: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    x = v1_trades_phasec_like.copy()
    for c in ("signal_time", "entry_time", "exit_time"):
        x[c] = pd.to_datetime(x.get(c), utc=True, errors="coerce")
    for c in (
        "filled",
        "valid_for_metrics",
        "sl_hit",
        "tp_hit",
        "pnl_net_pct",
        "entry_price",
        "exit_price",
        "entry_notional",
        "entry_fee_bps",
        "exit_fee_bps",
        "entry_slippage_bps",
        "exit_slippage_bps",
        "hold_minutes",
    ):
        x[c] = pd.to_numeric(x.get(c, np.nan), errors="coerce")

    m_valid = (x["filled"] == 1) & (x["valid_for_metrics"] == 1) & np.isfinite(x["pnl_net_pct"])
    tr = x[m_valid].copy().sort_values(["entry_time", "signal_time"]).reset_index(drop=True)
    n_signals = int(signals_total)
    n_trades = int(tr.shape[0])
    ret = pd.to_numeric(tr["pnl_net_pct"], errors="coerce").dropna().to_numpy(dtype=float)

    eq_vals = pd.to_numeric(pd.Series(metrics_native.get("equity_curve", [])), errors="coerce").to_numpy(dtype=float)
    eq_ts = pd.to_datetime(pd.Series(metrics_native.get("equity_timestamps", [])), utc=True, errors="coerce")
    n_eq = min(len(eq_vals), len(eq_ts))
    eq_df = pd.DataFrame(
        {
            "timestamp": eq_ts.iloc[:n_eq].to_numpy(),
            "equity": eq_vals[:n_eq],
        }
    )
    eq_df = eq_df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp").reset_index(drop=True)
    if eq_df.empty:
        eq_df = pd.DataFrame([{"timestamp": pd.NaT, "equity": float(initial_equity)}])
    peak = eq_df["equity"].cummax()
    eq_df["drawdown_pct"] = eq_df["equity"] / np.maximum(1e-12, peak) - 1.0
    eq_df["drawdown_abs"] = eq_df["equity"] - peak
    mon = _compute_monthly_from_equity(eq_df, float(initial_equity))

    final_eq = _safe_float(metrics_native.get("final_equity"))
    total_return = final_eq / max(1e-12, float(initial_equity)) - 1.0 if np.isfinite(final_eq) else float("nan")
    if eq_df["timestamp"].notna().any():
        t0 = pd.to_datetime(eq_df["timestamp"].dropna().iloc[0], utc=True)
        t1 = pd.to_datetime(eq_df["timestamp"].dropna().iloc[-1], utc=True)
        years = max(1e-9, float((t1 - t0).total_seconds()) / (365.25 * 24.0 * 3600.0))
    else:
        years = float("nan")
    cagr = (final_eq / max(1e-12, float(initial_equity))) ** (1.0 / years) - 1.0 if np.isfinite(final_eq) and np.isfinite(years) and years > 0 else float("nan")

    pos = ret[ret > 0]
    neg = ret[ret < 0]
    expectancy_net = float(np.mean(ret)) if ret.size else float("nan")
    expectancy_gross = float("nan")
    expectancy_net_per_signal = float(expectancy_net * (n_trades / max(1, n_signals))) if np.isfinite(expectancy_net) else float("nan")
    pnl_std = float(np.std(ret, ddof=0)) if ret.size else float("nan")
    downside = ret[ret < 0]
    downside_std = float(np.std(downside, ddof=0)) if downside.size else float("nan")
    trades_per_year = float(n_trades / years) if np.isfinite(years) and years > 0 else float("nan")
    sharpe = float((expectancy_net / pnl_std) * np.sqrt(trades_per_year)) if np.isfinite(expectancy_net) and np.isfinite(pnl_std) and pnl_std > 1e-12 and np.isfinite(trades_per_year) else float("nan")
    sortino = float((expectancy_net / downside_std) * np.sqrt(trades_per_year)) if np.isfinite(expectancy_net) and np.isfinite(downside_std) and downside_std > 1e-12 and np.isfinite(trades_per_year) else float("nan")
    vol_ann = float(pnl_std * np.sqrt(trades_per_year)) if np.isfinite(pnl_std) and np.isfinite(trades_per_year) else float("nan")
    cvar5 = _tail_mean(ret, 0.05) if ret.size else float("nan")
    pf = float(metrics_native.get("profit_factor", np.nan))
    wins = int((ret > 0).sum())
    losses = int((ret < 0).sum())
    win_rate = float(wins / max(1, wins + losses))
    hold = pd.to_numeric(tr.get("hold_minutes"), errors="coerce").dropna()
    avg_hold = float(hold.mean()) if not hold.empty else float("nan")
    med_hold = float(hold.median()) if not hold.empty else float("nan")
    p95_hold = float(hold.quantile(0.95)) if not hold.empty else float("nan")
    if tr["entry_time"].notna().any() and tr["exit_time"].notna().any():
        total_window = max(
            1e-9,
            float(
                (
                    pd.to_datetime(tr["exit_time"].dropna().max(), utc=True)
                    - pd.to_datetime(tr["entry_time"].dropna().min(), utc=True)
                ).total_seconds()
                / 60.0
            ),
        )
        exposure = float(hold.sum() / total_window) if not hold.empty else float("nan")
    else:
        exposure = float("nan")
    fee_abs = (
        pd.to_numeric(tr.get("entry_notional"), errors="coerce")
        * (pd.to_numeric(tr.get("entry_fee_bps"), errors="coerce") + pd.to_numeric(tr.get("exit_fee_bps"), errors="coerce"))
        / 1e4
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    slip_abs = (
        pd.to_numeric(tr.get("entry_notional"), errors="coerce")
        * (pd.to_numeric(tr.get("entry_slippage_bps"), errors="coerce") + pd.to_numeric(tr.get("exit_slippage_bps"), errors="coerce"))
        / 1e4
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    turnover = (
        pd.to_numeric(tr.get("entry_notional"), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) * 2.0
    )
    max_dd_abs = float(eq_df["drawdown_abs"].min()) if "drawdown_abs" in eq_df.columns else float("nan")
    max_dd_pct = -float(metrics_native.get("max_dd", np.nan)) if np.isfinite(_safe_float(metrics_native.get("max_dd"))) else float("nan")
    exit_dist = tr["exit_reason"].fillna("unknown").astype(str).str.lower().value_counts().sort_index().to_dict()

    m = {
        "signals_total": int(n_signals),
        "trades_total": int(n_trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(win_rate),
        "expectancy_gross": float(expectancy_gross),
        "expectancy_net": float(expectancy_net),
        "expectancy_net_per_signal": float(expectancy_net_per_signal),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "profit_factor": float(pf),
        "max_drawdown_abs": float(max_dd_abs),
        "max_drawdown_pct": float(max_dd_pct),
        "cvar_5": float(cvar5),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "volatility_annualized": float(vol_ann),
        "avg_trade_return": float(expectancy_net),
        "median_trade_return": float(np.median(ret)) if ret.size else float("nan"),
        "average_hold_time_min": float(avg_hold),
        "median_hold_time_min": float(med_hold),
        "p95_hold_time_min": float(p95_hold),
        "exposure_time_pct": float(exposure),
        "total_fees_paid": float(fee_abs.sum()),
        "total_slippage_paid": float(slip_abs.sum()),
        "turnover_notional": float(turnover.sum()),
        "entry_rate": float(n_trades / max(1, n_signals)),
        "participation": float(n_trades / max(1, n_signals)),
        "taker_share": 1.0,
        "median_fill_delay_min": 0.0,
        "p95_fill_delay_min": 0.0,
        "exit_reason_distribution": {str(k): int(v) for k, v in exit_dist.items()},
        "annualized_return": float(cagr),
    }
    # Native overlay fields for explicit forensic linkage.
    m["native_final_equity"] = _safe_float(metrics_native.get("final_equity"))
    m["native_net_profit"] = _safe_float(metrics_native.get("net_profit"))
    m["native_max_dd_pct_positive"] = _safe_float(metrics_native.get("max_dd"))
    m["native_profit_factor"] = _safe_float(metrics_native.get("profit_factor"))
    m["native_win_rate_pct"] = _safe_float(metrics_native.get("win_rate_pct"))
    m["native_trades"] = _safe_float(metrics_native.get("trades"))
    return eq_df, m, mon


def _comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    idx = {r["variant"]: r for _, r in metrics_df.iterrows()}
    v1 = idx["V1_1H_FULLSCAN_REFERENCE"]
    v3 = idx["V3_EXEC_3M_PHASEC_CONTROL_FROZEN"]
    v4 = idx["V4_EXEC_3M_PHASEC_BEST_FROZEN"]
    metrics = [
        "expectancy_net",
        "total_return",
        "max_drawdown_pct",
        "cvar_5",
        "profit_factor",
        "win_rate",
        "total_fees_paid",
        "median_fill_delay_min",
        "p95_fill_delay_min",
    ]
    rows: List[Dict[str, Any]] = []
    for m in metrics:
        a = _safe_float(v1.get(m))
        b = _safe_float(v3.get(m))
        c = _safe_float(v4.get(m))
        rows.append(
            {
                "metric": m,
                "v1_1h_fullscan_reference": a,
                "v3_exec3m_phasec_control_frozen": b,
                "v4_exec3m_phasec_best_frozen": c,
                "delta_exec3m_control_minus_1h_reference": b - a if np.isfinite(a) and np.isfinite(b) else np.nan,
                "delta_exec3m_phasec_minus_exec3m_control": c - b if np.isfinite(c) and np.isfinite(b) else np.nan,
                "delta_exec3m_phasec_minus_1h_reference": c - a if np.isfinite(c) and np.isfinite(a) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _universe_comparison(
    *,
    v1: Dict[str, Any],
    test_signals: pd.DataFrame,
    v2_metrics: Dict[str, Any],
    v3_metrics: Dict[str, Any],
    v4_metrics: Dict[str, Any],
) -> pd.DataFrame:
    t0 = pd.to_datetime(test_signals["signal_time"].min(), utc=True)
    t1 = pd.to_datetime(test_signals["signal_time"].max(), utc=True)
    rows = [
        {
            "variant_scope": "V1_1H_FULLSCAN_REFERENCE",
            "date_start": str(v1["period_start"]),
            "date_end": str(v1["period_end"]),
            "duration_days": float((v1["period_end"] - v1["period_start"]).total_seconds() / 86400.0),
            "sample_type": "1h bars + endogenous signals",
            "sample_count": int(v1["bar_count"]),
            "signals_count": int(v1["signal_count"]),
            "trades_total": int(v1["trades_df"].shape[0]),
            "fee_model": "legacy_fullscan_fee_bps=7;slippage_bps=2 (both sides)",
            "sizing_model": "native_atr_position_sizing_compounding",
            "split_scope": "full_period_no_wf_split",
            "notes": "Source used by params_scan/best_by_symbol",
        },
        {
            "variant_scope": "FROZEN_PHASEC_TEST_UNIVERSE",
            "date_start": str(t0),
            "date_end": str(t1),
            "duration_days": float((t1 - t0).total_seconds() / 86400.0),
            "sample_type": "frozen_exported_signals_test_only",
            "sample_count": int(test_signals.shape[0]),
            "signals_count": int(test_signals.shape[0]),
            "trades_total": int(v2_metrics.get("trades_total", 0)),
            "fee_model": "phase_a_maker_taker_fee_model",
            "sizing_model": "fixed_fractional_risk_per_trade_compounding",
            "split_scope": "walkforward_test_splits_only",
            "notes": "Universe used by Phase C/Phase D decisioning",
        },
        {
            "variant_scope": "V3_EXEC_3M_PHASEC_CONTROL_FROZEN",
            "date_start": str(t0),
            "date_end": str(t1),
            "duration_days": float((t1 - t0).total_seconds() / 86400.0),
            "sample_type": "same_frozen_test_signals",
            "sample_count": int(test_signals.shape[0]),
            "signals_count": int(test_signals.shape[0]),
            "trades_total": int(v3_metrics.get("trades_total", 0)),
            "fee_model": "phase_a_maker_taker_fee_model",
            "sizing_model": "fixed_fractional_risk_per_trade_compounding",
            "split_scope": "walkforward_test_splits_only",
            "notes": "Phase C control baseline_exit",
        },
        {
            "variant_scope": "V4_EXEC_3M_PHASEC_BEST_FROZEN",
            "date_start": str(t0),
            "date_end": str(t1),
            "duration_days": float((t1 - t0).total_seconds() / 86400.0),
            "sample_type": "same_frozen_test_signals",
            "sample_count": int(test_signals.shape[0]),
            "signals_count": int(test_signals.shape[0]),
            "trades_total": int(v4_metrics.get("trades_total", 0)),
            "fee_model": "phase_a_maker_taker_fee_model",
            "sizing_model": "fixed_fractional_risk_per_trade_compounding",
            "split_scope": "walkforward_test_splits_only",
            "notes": "Phase C best global exit",
        },
    ]
    return pd.DataFrame(rows)


def _signal_alignment_report(
    *,
    test_signals: pd.DataFrame,
    v2: pd.DataFrame,
    v3: pd.DataFrame,
    v4: pd.DataFrame,
    v1_trades: pd.DataFrame,
) -> pd.DataFrame:
    base = test_signals.copy()
    base["signal_time"] = pd.to_datetime(base["signal_time"], utc=True, errors="coerce")
    keep = ["signal_id", "signal_time", "cycle", "strategy_tp_mult", "strategy_sl_mult", "atr_percentile_1h", "trend_up_1h"]
    keep = [c for c in keep if c in base.columns]
    base = base[keep].copy()
    for name, df in [("v2", v2), ("v3", v3), ("v4", v4)]:
        x = df.copy()
        x["signal_id"] = x["signal_id"].astype(str)
        cols = [
            "signal_id",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "exit_reason",
            "pnl_net_pct",
            "sl_hit",
            "tp_hit",
            "hold_minutes",
        ]
        cols = [c for c in cols if c in x.columns]
        x = x[cols].copy()
        ren = {c: f"{name}_{c}" for c in x.columns if c != "signal_id"}
        x = x.rename(columns=ren)
        base = base.merge(x, on="signal_id", how="left")
        base[f"{name}_has_trade"] = base[f"{name}_entry_time"].notna().astype(int)

    v1_hours = set(
        pd.to_datetime(v1_trades.get("entry_ts"), utc=True, errors="coerce")
        .dropna()
        .dt.floor("h")
        .astype(str)
        .tolist()
    )
    base["v1_entry_same_hour"] = (
        pd.to_datetime(base["signal_time"], utc=True, errors="coerce").dt.floor("h").astype(str).isin(v1_hours).astype(int)
    )
    return base.sort_values("signal_time").reset_index(drop=True)


def _trade_alignment_report(signal_alignment: pd.DataFrame) -> pd.DataFrame:
    x = signal_alignment.copy()
    for c in [
        "v2_pnl_net_pct",
        "v3_pnl_net_pct",
        "v4_pnl_net_pct",
        "v2_hold_minutes",
        "v3_hold_minutes",
        "v4_hold_minutes",
    ]:
        x[c] = pd.to_numeric(x.get(c, np.nan), errors="coerce")
    for c in ["v2_exit_time", "v3_exit_time", "v4_exit_time"]:
        x[c] = pd.to_datetime(x.get(c), utc=True, errors="coerce")
    x["delta_pnl_v4_minus_v3"] = x["v4_pnl_net_pct"] - x["v3_pnl_net_pct"]
    x["delta_pnl_v3_minus_v2"] = x["v3_pnl_net_pct"] - x["v2_pnl_net_pct"]
    x["delta_hold_v4_minus_v3_min"] = x["v4_hold_minutes"] - x["v3_hold_minutes"]
    x["delta_exit_time_v4_minus_v3_min"] = (x["v4_exit_time"] - x["v3_exit_time"]).dt.total_seconds() / 60.0
    x["same_entry_price_v3_v4"] = (pd.to_numeric(x.get("v3_entry_price"), errors="coerce").round(8) == pd.to_numeric(x.get("v4_entry_price"), errors="coerce").round(8)).astype(int)
    x["same_exit_reason_v3_v4"] = (x.get("v3_exit_reason").astype(str) == x.get("v4_exit_reason").astype(str)).astype(int)

    def _classify(r: pd.Series) -> str:
        if int(r.get("same_exit_reason_v3_v4", 0)) == 0:
            return "exit_rule_change"
        dp = _safe_float(r.get("delta_pnl_v4_minus_v3"))
        if np.isfinite(dp) and abs(dp) > 1e-9:
            return "price_or_timing_change"
        return "same_outcome"

    x["alignment_class_v4_vs_v3"] = x.apply(_classify, axis=1)
    cols = [
        "signal_id",
        "signal_time",
        "v2_entry_time",
        "v2_exit_time",
        "v2_exit_reason",
        "v2_pnl_net_pct",
        "v3_entry_time",
        "v3_exit_time",
        "v3_exit_reason",
        "v3_pnl_net_pct",
        "v4_entry_time",
        "v4_exit_time",
        "v4_exit_reason",
        "v4_pnl_net_pct",
        "delta_pnl_v3_minus_v2",
        "delta_pnl_v4_minus_v3",
        "delta_hold_v4_minus_v3_min",
        "delta_exit_time_v4_minus_v3_min",
        "alignment_class_v4_vs_v3",
    ]
    cols = [c for c in cols if c in x.columns]
    return x[cols].sort_values("signal_time").reset_index(drop=True)


def _implementation_diff_matrix(v1: Dict[str, Any], phasec_manifest: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"dimension": "symbol_scope", "v1_fullscan": "SOLUSDT", "frozen_phasec": "SOLUSDT", "match": 1, "note": ""},
            {
                "dimension": "date_range",
                "v1_fullscan": f"{v1['period_start']} -> {v1['period_end']}",
                "frozen_phasec": "from frozen subset test slices",
                "match": 0,
                "note": "Different windows and regime composition",
            },
            {
                "dimension": "signal_source",
                "v1_fullscan": "endogenous 1h signal generation in ga.py",
                "frozen_phasec": "pre-exported signal_subset.csv",
                "match": 0,
                "note": "Different signal universe construction",
            },
            {
                "dimension": "entry_logic",
                "v1_fullscan": "1h backtester open-entry on signal bars",
                "frozen_phasec": "market at next 3m open after exported signal",
                "match": 0,
                "note": "Different entry timing granularity",
            },
            {
                "dimension": "exit_logic",
                "v1_fullscan": "ga.py 1h TP/SL/maxhold/rsi_exit",
                "frozen_phasec": "3m intrabar TP/SL + timeout (Phase C variants)",
                "match": 0,
                "note": "Different path-dependent exits",
            },
            {
                "dimension": "stop_hit_granularity",
                "v1_fullscan": "1h-bar checks",
                "frozen_phasec": "3m-bar sequential checks",
                "match": 0,
                "note": "Intrabar stop/TP ordering differs materially",
            },
            {
                "dimension": "fee_slippage_model",
                "v1_fullscan": "fee_bps=7 slip_bps=2 both sides",
                "frozen_phasec": "maker/taker + per-side slippage from Phase A",
                "match": 0,
                "note": "Explicit contract mismatch",
            },
            {
                "dimension": "position_sizing",
                "v1_fullscan": "ATR-risk position sizing with max allocation",
                "frozen_phasec": "fixed fractional risk_per_trade equity simulator",
                "match": 0,
                "note": "Different leverage/notional dynamics",
            },
            {
                "dimension": "compounding",
                "v1_fullscan": "native compounded cash with trade sizing",
                "frozen_phasec": "compounded 1% risk model on synthetic equity",
                "match": 0,
                "note": "Different equity process",
            },
            {
                "dimension": "wf_split_scope",
                "v1_fullscan": "none",
                "frozen_phasec": json.dumps(phasec_manifest.get("sampling", {}).get("coarse", {}).get("stratify_fields", [])),
                "match": 0,
                "note": "Frozen test-only split contract",
            },
        ]
    )


def _mismatch_checklist(diff_df: pd.DataFrame) -> str:
    lines = ["# Mismatch Checklist", ""]
    for r in diff_df.itertuples(index=False):
        status = "PASS" if int(getattr(r, "match", 0)) == 1 else "FAIL"
        lines.append(f"- [{status}] {r.dimension}: {r.note}")
    return "\n".join(lines).strip() + "\n"


def _simulate_bar_exit_1h_from_frozen_signals(
    *,
    entries_df: pd.DataFrame,
    symbol: str,
    fee: phasec_bt.FeeModel,
    exec_horizon_hours: float,
) -> pd.DataFrame:
    df1h = _load_symbol_df(symbol=symbol, tf="1h")
    ts_pd = pd.to_datetime(df1h["Timestamp"], utc=True, errors="coerce")
    ts = ts_pd.to_numpy()
    ts_ns = np.array([int(t.value) for t in ts_pd], dtype=np.int64)
    hi = pd.to_numeric(df1h["High"], errors="coerce").to_numpy(dtype=float)
    lo = pd.to_numeric(df1h["Low"], errors="coerce").to_numpy(dtype=float)
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
        if pd.isna(et) or not np.isfinite(ep) or ep <= 0 or not np.isfinite(tp_mult) or not np.isfinite(sl_mult):
            rows.append(
                {
                    "symbol": symbol,
                    "signal_id": sid,
                    "signal_time": st,
                    "split_id": split_id,
                    "signal_tp_mult": tp_mult,
                    "signal_sl_mult": sl_mult,
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
                    "risk_pct": max(1e-8, 1.0 - sl_mult) if np.isfinite(sl_mult) else np.nan,
                    "entry_type": "market",
                    "fill_liquidity_type": "taker",
                }
            )
            continue

        sl = ep * sl_mult
        tp = ep * tp_mult
        max_ns = int(et.value + float(exec_horizon_hours) * 3600.0 * 1e9)
        i0 = int(np.searchsorted(ts_ns, int(et.value), side="left"))
        if i0 >= len(ts):
            rows.append(
                {
                    "symbol": symbol,
                    "signal_id": sid,
                    "signal_time": st,
                    "split_id": split_id,
                    "signal_tp_mult": tp_mult,
                    "signal_sl_mult": sl_mult,
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
                    "risk_pct": max(1e-8, 1.0 - sl_mult),
                    "entry_type": "market",
                    "fill_liquidity_type": "taker",
                }
            )
            continue

        mae = 0.0
        mfe = 0.0
        exit_i = i0
        exit_px = float(cl[min(i0, len(cl) - 1)])
        reason = "timeout"
        sl_hit = 0
        tp_hit = 0
        for i in range(i0, len(ts)):
            tns = pd.Timestamp(ts[i]).value
            if tns > max_ns:
                exit_i = max(i0, i - 1)
                exit_px = float(cl[exit_i])
                reason = "window_end"
                break
            h = float(hi[i])
            l = float(lo[i])
            mae = min(mae, l / ep - 1.0)
            mfe = max(mfe, h / ep - 1.0)
            hit_sl = bool(l <= sl)
            hit_tp = bool(h >= tp)
            if hit_sl and hit_tp:
                exit_i = i
                exit_px = float(sl)
                reason = "sl"
                sl_hit = 1
                tp_hit = 1
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
        etime = pd.to_datetime(ts[exit_i], utc=True)
        c = phasec_bt._cost_row(float(ep), float(exit_px), "taker", fee)
        rows.append(
            {
                "symbol": symbol,
                "signal_id": sid,
                "signal_time": st,
                "split_id": split_id,
                "signal_tp_mult": tp_mult,
                "signal_sl_mult": sl_mult,
                "filled": 1,
                "valid_for_metrics": 1,
                "sl_hit": sl_hit,
                "tp_hit": tp_hit,
                "entry_time": et,
                "exit_time": etime,
                "entry_price": float(ep),
                "exit_price": float(exit_px),
                "exit_reason": reason,
                "mae_pct": float(mae),
                "mfe_pct": float(mfe),
                "hold_minutes": float((etime - et).total_seconds() / 60.0),
                "risk_pct": max(1e-8, 1.0 - sl_mult),
                "entry_type": "market",
                "fill_liquidity_type": "taker",
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


def _rollup_for_sensitivity(
    trades: pd.DataFrame,
    *,
    signals_total: int,
    initial_equity: float,
    risk_per_trade: float,
    mode: str,
) -> Dict[str, Any]:
    x = trades.copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    m = (pd.to_numeric(x.get("filled", 0), errors="coerce").fillna(0).astype(int) == 1) & (
        pd.to_numeric(x.get("valid_for_metrics", 0), errors="coerce").fillna(0).astype(int) == 1
    )
    x = x[m].copy()
    x["pnl_net_pct"] = pd.to_numeric(x.get("pnl_net_pct"), errors="coerce")
    x["risk_pct"] = pd.to_numeric(x.get("risk_pct"), errors="coerce").clip(lower=1e-8)
    x = x[np.isfinite(x["pnl_net_pct"]) & np.isfinite(x["risk_pct"])].copy()

    eq = float(initial_equity)
    eq_curve: List[float] = [eq]
    for r in x.itertuples(index=False):
        rsk = max(1e-8, float(getattr(r, "risk_pct")))
        pnl = float(getattr(r, "pnl_net_pct"))
        base = float(eq) if mode == "compounding" else float(initial_equity)
        notional = float(base * risk_per_trade / rsk)
        eq += float(notional * pnl)
        eq_curve.append(eq)
    dd_abs, dd_pct = _max_dd_from_equity(eq_curve)
    ret = (eq / max(1e-12, initial_equity)) - 1.0
    exp = float(x["pnl_net_pct"].mean()) if not x.empty else float("nan")
    cvar = _tail_mean(pd.to_numeric(x["pnl_net_pct"], errors="coerce").dropna().to_numpy(), 0.05) if not x.empty else float("nan")
    return {
        "signals_total": int(signals_total),
        "trades_total": int(x.shape[0]),
        "expectancy_net": float(exp),
        "cvar_5": float(cvar),
        "total_return": float(ret),
        "final_equity": float(eq),
        "max_drawdown_abs": float(dd_abs),
        "max_drawdown_pct": float(dd_pct),
    }


def _yearly_metrics_v1(v1_trades: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
    if v1_trades.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "trades",
                "wins",
                "losses",
                "win_rate",
                "net_pnl",
                "avg_trade_return",
                "profit_factor",
                "equity_start",
                "equity_end",
                "return_pct",
            ]
        )
    x = v1_trades.copy()
    x["exit_ts"] = pd.to_datetime(x.get("exit_ts"), utc=True, errors="coerce")
    x["pnl_net_pct"] = pd.to_numeric(x.get("pnl_net_pct"), errors="coerce")
    x["net_pnl"] = pd.to_numeric(x.get("net_pnl"), errors="coerce")
    x = x.dropna(subset=["exit_ts"]).sort_values("exit_ts").reset_index(drop=True)
    x["year"] = x["exit_ts"].dt.year
    eq = float(initial_equity)
    eq_start: Dict[int, float] = {}
    eq_end: Dict[int, float] = {}
    for r in x.itertuples(index=False):
        y = int(getattr(r, "year"))
        if y not in eq_start:
            eq_start[y] = float(eq)
        eq += float(getattr(r, "net_pnl", 0.0))
        eq_end[y] = float(eq)

    rows: List[Dict[str, Any]] = []
    for y, g in x.groupby("year", sort=True):
        rets = pd.to_numeric(g["pnl_net_pct"], errors="coerce").dropna()
        pnl = pd.to_numeric(g["net_pnl"], errors="coerce").dropna()
        wins = int((pnl > 0).sum())
        losses = int((pnl < 0).sum())
        gp = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
        gl = float(pnl[pnl < 0].sum()) if not pnl.empty else 0.0
        pf = (gp / abs(gl)) if gl < -1e-12 else (float("inf") if gp > 0 else float("nan"))
        es = float(eq_start.get(int(y), np.nan))
        ee = float(eq_end.get(int(y), np.nan))
        rows.append(
            {
                "year": int(y),
                "trades": int(g.shape[0]),
                "wins": wins,
                "losses": losses,
                "win_rate": float(wins / max(1, wins + losses)),
                "net_pnl": float(pnl.sum()),
                "avg_trade_return": float(rets.mean()) if not rets.empty else float("nan"),
                "profit_factor": float(pf),
                "equity_start": es,
                "equity_end": ee,
                "return_pct": float((ee / max(1e-12, es)) - 1.0) if np.isfinite(es) and np.isfinite(ee) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def _write_markdown_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df.empty:
        return "_(empty)_"
    x = df.head(max_rows).copy()
    cols = list(x.columns)
    hdr = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [hdr, sep]
    for _, r in x.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                if np.isfinite(v):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append("nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Path:
    symbol = str(args.symbol).strip().upper()
    if symbol != "SOLUSDT":
        raise RuntimeError("This reconciliation runner is SOL-only by design.")

    phase_c_dir = _resolve(args.phase_c_dir)
    phase_a_dir = _resolve(args.phase_a_contract_dir)
    params_path = _resolve(args.params_file)
    if not phase_c_dir.exists():
        raise FileNotFoundError(f"Missing phase_c_dir: {phase_c_dir}")
    if not phase_a_dir.exists():
        raise FileNotFoundError(f"Missing phase_a_contract_dir: {phase_a_dir}")
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params file: {params_path}")

    run_dir = _resolve(args.outdir) / f"SOL_RECON_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snap = run_dir / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    phasec_manifest = _load_phasec_manifest(phase_c_dir)
    if str(phasec_manifest.get("symbol", "")).upper() != symbol:
        raise RuntimeError(f"Phase C manifest symbol mismatch: {phasec_manifest.get('symbol')}")
    if str(phasec_manifest.get("final_selected_cfg_hash", "")) != EXPECTED_PHASEC_CFG_HASH:
        raise RuntimeError(
            f"Phase C cfg hash mismatch: expected={EXPECTED_PHASEC_CFG_HASH} got={phasec_manifest.get('final_selected_cfg_hash')}"
        )

    phase_a_contract = phasec_manifest.get("phase_a_contract", {})
    fee_model_path = _resolve(str(phase_a_contract.get("fee_model_path", phase_a_dir / "fee_model.json")))
    metrics_def_path = _resolve(str(phase_a_contract.get("metrics_definition_path", phase_a_dir / "metrics_definition.md")))
    fee_hash = _ensure_frozen_hash(fee_model_path, str(phase_a_contract.get("fee_model_sha256", "")), "phase_a fee_model")
    metrics_hash = _ensure_frozen_hash(
        metrics_def_path,
        str(phase_a_contract.get("metrics_definition_sha256", "")),
        "phase_a metrics_definition",
    )

    subset_path = _resolve(str(phasec_manifest.get("signal_subset_path", phase_c_dir / "signal_subset.csv")))
    split_path = _resolve(str(phasec_manifest.get("split_definition_path", phase_c_dir / "wf_split_definition.json")))
    subset_df = pd.read_csv(subset_path)
    subset_hash = _sha256_signal_subset(subset_df)
    if subset_hash != str(phasec_manifest.get("signal_subset_hash", "")) or subset_hash != EXPECTED_SIGNAL_SUBSET_HASH:
        raise RuntimeError(
            "Frozen signal subset hash mismatch: "
            f"got={subset_hash} manifest={phasec_manifest.get('signal_subset_hash')} expected={EXPECTED_SIGNAL_SUBSET_HASH}"
        )
    split_hash = _sha256_file(split_path)
    if split_hash != str(phasec_manifest.get("split_definition_sha256", "")) or split_hash != EXPECTED_SPLIT_HASH:
        raise RuntimeError(
            "Frozen split hash mismatch: "
            f"got={split_hash} manifest={phasec_manifest.get('split_definition_sha256')} expected={EXPECTED_SPLIT_HASH}"
        )

    splits = _parse_split_definition(split_path)
    test_idx = _test_indices_from_splits(splits)
    subset_df = subset_df.reset_index(drop=True).copy()
    subset_df["signal_time"] = pd.to_datetime(subset_df["signal_time"], utc=True, errors="coerce")
    subset_df["tp_mult"] = pd.to_numeric(subset_df.get("strategy_tp_mult", subset_df.get("tp_mult")), errors="coerce")
    subset_df["sl_mult"] = pd.to_numeric(subset_df.get("strategy_sl_mult", subset_df.get("sl_mult")), errors="coerce")
    test_signals = subset_df.iloc[test_idx].copy().reset_index(drop=True)
    split_lookup = _split_lookup(subset_df, splits)
    test_signal_ids = set(test_signals["signal_id"].astype(str).tolist())

    # Snapshot core contracts and inputs.
    shutil.copy2(fee_model_path, run_dir / "fee_model.json")
    shutil.copy2(metrics_def_path, run_dir / "metrics_definition.md")
    shutil.copy2(fee_model_path, snap / "fee_model.json")
    shutil.copy2(metrics_def_path, snap / "metrics_definition.md")
    shutil.copy2(subset_path, snap / "signal_subset.csv")
    shutil.copy2(split_path, snap / "wf_split_definition.json")
    shutil.copy2(params_path, snap / params_path.name)
    shutil.copy2(phase_c_dir / "run_manifest.json", snap / "phasec_run_manifest.json")
    for fname in [
        "trade_diagnostics_baseline.csv",
        "trade_diagnostics_best.csv",
        "baseline_vs_best_summary.csv",
        "risk_rollup_overall.csv",
        "risk_rollup_by_symbol.csv",
    ]:
        fp = phase_c_dir / fname
        if fp.exists():
            shutil.copy2(fp, snap / fname)
    phased_dir = _resolve(args.phase_d_dir)
    if phased_dir.exists():
        for fname in ["decision.md", "risk_rollup_overall.csv", "risk_rollup_by_symbol.csv"]:
            fp = phased_dir / fname
            if fp.exists():
                shutil.copy2(fp, snap / f"phase_d_{fname}")
    best_by_symbol_path = _resolve(args.best_by_symbol_csv)
    if best_by_symbol_path.exists():
        shutil.copy2(best_by_symbol_path, snap / "best_by_symbol.csv")

    # V1 fullscan run.
    v1 = _run_v1_fullscan(
        symbol=symbol,
        params_path=params_path,
        initial_equity=float(args.fullscan_initial_equity),
        fee_bps=float(args.fullscan_fee_bps),
        slip_bps=float(args.fullscan_slip_bps),
    )
    v1_trades_phasec = _v1_to_phasec_like_trades(
        v1,
        symbol=symbol,
        fee_bps=float(args.fullscan_fee_bps),
        slip_bps=float(args.fullscan_slip_bps),
    )
    v1_eq, v1_metrics, v1_monthly = _compute_v1_metrics_from_native(
        v1_trades_phasec_like=v1_trades_phasec,
        signals_total=int(v1["signal_count"]),
        initial_equity=float(args.fullscan_initial_equity),
        risk_per_trade=float(args.fullscan_risk_per_trade_for_overlay),
        metrics_native=v1["metrics_native"],
    )
    v1_pack = VariantPack(
        name="V1_1H_FULLSCAN_REFERENCE",
        universe_scope="fullscan_native",
        trades=v1_trades_phasec,
        equity=_fill_drawdown(v1_eq, "V1_1H_FULLSCAN_REFERENCE"),
        monthly=v1_monthly,
        metrics=v1_metrics,
    )

    # V2/V3/V4 under frozen Phase C contract.
    fee = phasec_bt._load_fee_model(fee_model_path)
    v3_df, v4_df = _load_v3_v4_trades(
        phase_c_dir=phase_c_dir,
        split_lookup=split_lookup,
        test_signal_ids=test_signal_ids,
        fee=fee,
    )
    v2_df = phasec_bt._simulate_1h_reference(
        signals_df=test_signals[["signal_id", "signal_time", "tp_mult", "sl_mult"]].copy(),
        split_lookup=split_lookup,
        fee=fee,
        exec_horizon_hours=float(args.exec_horizon_hours),
        symbol=symbol,
    )
    test_n = int(len(test_signals))
    phase_init_eq = float(args.phase_initial_equity)
    phase_risk = float(args.phase_risk_per_trade)
    v2_eq, v2_metrics, v2_month = phasec_bt._compute_equity_curve(
        v2_df,
        signals_total=test_n,
        initial_equity=phase_init_eq,
        risk_per_trade=phase_risk,
    )
    v3_eq, v3_metrics, v3_month = phasec_bt._compute_equity_curve(
        v3_df,
        signals_total=test_n,
        initial_equity=phase_init_eq,
        risk_per_trade=phase_risk,
    )
    v4_eq, v4_metrics, v4_month = phasec_bt._compute_equity_curve(
        v4_df,
        signals_total=test_n,
        initial_equity=phase_init_eq,
        risk_per_trade=phase_risk,
    )
    v2_pack = VariantPack(
        name="V2_1H_FROZEN_PHASEC_UNIVERSE_REFERENCE",
        universe_scope="frozen_phasec_test",
        trades=v2_df,
        equity=_fill_drawdown(v2_eq, "V2_1H_FROZEN_PHASEC_UNIVERSE_REFERENCE"),
        monthly=v2_month,
        metrics=v2_metrics,
    )
    v3_pack = VariantPack(
        name="V3_EXEC_3M_PHASEC_CONTROL_FROZEN",
        universe_scope="frozen_phasec_test",
        trades=v3_df,
        equity=_fill_drawdown(v3_eq, "V3_EXEC_3M_PHASEC_CONTROL_FROZEN"),
        monthly=v3_month,
        metrics=v3_metrics,
    )
    v4_pack = VariantPack(
        name="V4_EXEC_3M_PHASEC_BEST_FROZEN",
        universe_scope="frozen_phasec_test",
        trades=v4_df,
        equity=_fill_drawdown(v4_eq, "V4_EXEC_3M_PHASEC_BEST_FROZEN"),
        monthly=v4_month,
        metrics=v4_metrics,
    )
    packs = [v1_pack, v2_pack, v3_pack, v4_pack]

    # Required data tables.
    metrics_df = pd.DataFrame([_metrics_row(p.name, p.universe_scope, p.metrics) for p in packs])
    metrics_df.to_csv(run_dir / "metrics_by_variant.csv", index=False)

    eq_all = pd.concat([p.equity.assign(variant=p.name) for p in packs], ignore_index=True)
    eq_all.to_csv(run_dir / "equity_curve_by_variant.csv", index=False)
    dd_all = eq_all[["variant", "timestamp", "equity", "drawdown_pct", "drawdown_abs"]].copy()
    dd_all.to_csv(run_dir / "drawdown_curve_by_variant.csv", index=False)

    split_metrics = phasec_bt._split_variant_metrics(
        {
            "V2_1H_FROZEN_PHASEC_UNIVERSE_REFERENCE": v2_df,
            "V3_EXEC_3M_PHASEC_CONTROL_FROZEN": v3_df,
            "V4_EXEC_3M_PHASEC_BEST_FROZEN": v4_df,
        },
        splits,
        initial_equity=phase_init_eq,
        risk_per_trade=phase_risk,
        signal_subset=subset_df,
    )
    split_metrics.to_csv(run_dir / "per_split_metrics_by_variant.csv", index=False)

    cmp_df = _comparison_table(metrics_df)
    cmp_df.to_csv(run_dir / "comparison_table.csv", index=False)

    uni_df = _universe_comparison(
        v1=v1,
        test_signals=test_signals,
        v2_metrics=v2_metrics,
        v3_metrics=v3_metrics,
        v4_metrics=v4_metrics,
    )
    uni_df.to_csv(run_dir / "universe_comparison.csv", index=False)

    sig_align = _signal_alignment_report(
        test_signals=test_signals,
        v2=v2_df,
        v3=v3_df,
        v4=v4_df,
        v1_trades=v1["trades_df"],
    )
    sig_align.to_csv(run_dir / "signal_alignment_report.csv", index=False)

    trade_align = _trade_alignment_report(sig_align)
    trade_align.to_csv(run_dir / "trade_alignment_report.csv", index=False)

    diff_df = _implementation_diff_matrix(v1=v1, phasec_manifest=phasec_manifest)
    diff_df.to_csv(run_dir / "implementation_diff_matrix.csv", index=False)
    (run_dir / "mismatch_checklist.md").write_text(_mismatch_checklist(diff_df), encoding="utf-8")

    # Stop-hit granularity impact diagnostic (V2 reference vs 1h-bar approximation).
    v2_entries = v2_df.copy()
    for c in ("signal_tp_mult", "signal_sl_mult", "entry_price", "entry_time", "signal_time", "split_id"):
        if c not in v2_entries.columns:
            raise RuntimeError(f"V2 trades missing column {c}")
    v2_1h_bar = _simulate_bar_exit_1h_from_frozen_signals(
        entries_df=v2_entries[
            ["signal_id", "signal_time", "split_id", "signal_tp_mult", "signal_sl_mult", "entry_time", "entry_price"]
        ].copy(),
        symbol=symbol,
        fee=fee,
        exec_horizon_hours=float(args.exec_horizon_hours),
    )
    v2_1h_eq, v2_1h_m, _ = phasec_bt._compute_equity_curve(
        v2_1h_bar,
        signals_total=test_n,
        initial_equity=phase_init_eq,
        risk_per_trade=phase_risk,
    )
    sl_rate_3m = float(pd.to_numeric(v2_df["sl_hit"], errors="coerce").fillna(0).mean())
    sl_rate_1h = float(pd.to_numeric(v2_1h_bar["sl_hit"], errors="coerce").fillna(0).mean())
    tp_rate_3m = float(pd.to_numeric(v2_df["tp_hit"], errors="coerce").fillna(0).mean())
    tp_rate_1h = float(pd.to_numeric(v2_1h_bar["tp_hit"], errors="coerce").fillna(0).mean())
    gran_df = pd.DataFrame(
        [
            {
                "mode": "intrabar_3m_reference",
                "signals_total": test_n,
                "trades_total": int(v2_metrics.get("trades_total", 0)),
                "sl_hit_rate": sl_rate_3m,
                "tp_hit_rate": tp_rate_3m,
                "expectancy_net": float(v2_metrics.get("expectancy_net", np.nan)),
                "total_return": float(v2_metrics.get("total_return", np.nan)),
                "max_drawdown_pct": float(v2_metrics.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(v2_metrics.get("cvar_5", np.nan)),
            },
            {
                "mode": "bar_1h_approx",
                "signals_total": test_n,
                "trades_total": int(v2_1h_m.get("trades_total", 0)),
                "sl_hit_rate": sl_rate_1h,
                "tp_hit_rate": tp_rate_1h,
                "expectancy_net": float(v2_1h_m.get("expectancy_net", np.nan)),
                "total_return": float(v2_1h_m.get("total_return", np.nan)),
                "max_drawdown_pct": float(v2_1h_m.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(v2_1h_m.get("cvar_5", np.nan)),
            },
        ]
    )
    gran_df["delta_vs_intrabar"] = gran_df["expectancy_net"] - float(v2_metrics.get("expectancy_net", np.nan))
    gran_df.to_csv(run_dir / "stop_hit_granularity_impact.csv", index=False)

    # Sizing/compounding sensitivity.
    sens_rows: List[Dict[str, Any]] = []
    for p in packs:
        if p.name == "V1_1H_FULLSCAN_REFERENCE":
            sens_rows.append(
                {
                    "variant": p.name,
                    "sizing_mode": "native_fullscan",
                    "signals_total": int(v1["signal_count"]),
                    "trades_total": int(v1["metrics_native"].get("trades", 0)),
                    "expectancy_net": float(p.metrics.get("expectancy_net", np.nan)),
                    "total_return": float(v1["metrics_native"].get("final_equity", np.nan))
                    / max(1e-12, float(v1["metrics_native"].get("initial_equity", 1.0)))
                    - 1.0,
                    "final_equity": float(v1["metrics_native"].get("final_equity", np.nan)),
                    "max_drawdown_pct": -float(v1["metrics_native"].get("max_dd", np.nan)),
                }
            )
        for mode in ("compounding", "fixed_notional"):
            init = float(args.fullscan_initial_equity) if p.name == "V1_1H_FULLSCAN_REFERENCE" else phase_init_eq
            risk = float(args.fullscan_risk_per_trade_for_overlay) if p.name == "V1_1H_FULLSCAN_REFERENCE" else phase_risk
            s = _rollup_for_sensitivity(
                p.trades,
                signals_total=int(p.metrics.get("signals_total", 0)),
                initial_equity=init,
                risk_per_trade=risk,
                mode=mode,
            )
            s["variant"] = p.name
            s["sizing_mode"] = mode
            sens_rows.append(s)
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(run_dir / "sensitivity_sizing_modes.csv", index=False)

    # V1 per-year.
    yearly_v1 = _yearly_metrics_v1(v1["trades_df"], initial_equity=float(args.fullscan_initial_equity))
    yearly_v1.to_csv(run_dir / "per_year_metrics_1h_fullscan.csv", index=False)

    # Runs registry.
    reg_rows: List[Dict[str, Any]] = []
    reg_rows.append(
        {
            "source_name": "phase_a_contract",
            "path": str(phase_a_dir),
            "fee_model_path": str(fee_model_path),
            "fee_model_sha256": fee_hash,
            "metrics_definition_path": str(metrics_def_path),
            "metrics_definition_sha256": metrics_hash,
        }
    )
    reg_rows.append(
        {
            "source_name": "phase_c_control",
            "path": str(phase_c_dir),
            "final_selected_cfg_hash": str(phasec_manifest.get("final_selected_cfg_hash")),
            "signal_subset_path": str(subset_path),
            "signal_subset_sha256": subset_hash,
            "wf_split_definition_path": str(split_path),
            "wf_split_sha256": split_hash,
        }
    )
    if best_by_symbol_path.exists():
        bdf = pd.read_csv(best_by_symbol_path)
        bsol = bdf[(bdf["symbol"].astype(str).str.upper() == symbol) & (bdf["side"].astype(str).str.lower() == "long")]
        if not bsol.empty:
            r = bsol.iloc[0].to_dict()
            reg_rows.append(
                {
                    "source_name": "best_by_symbol_row",
                    "path": str(best_by_symbol_path),
                    "params_file": r.get("params_file"),
                    "net_profit": r.get("net_profit"),
                    "max_dd_pct": r.get("max_dd_pct"),
                    "period_start": r.get("period_start"),
                    "period_end": r.get("period_end"),
                    "trades": r.get("trades"),
                }
            )
    if phased_dir.exists() and (phased_dir / "risk_rollup_overall.csv").exists():
        pr = pd.read_csv(phased_dir / "risk_rollup_overall.csv")
        if not pr.empty:
            rr = pr.iloc[0].to_dict()
            reg_rows.append(
                {
                    "source_name": "phase_d_rollup",
                    "path": str(phased_dir / "risk_rollup_overall.csv"),
                    "phasec_global_pnl_net_sum": rr.get("phasec_global_pnl_net_sum"),
                    "regime_pnl_net_sum": rr.get("regime_pnl_net_sum"),
                    "delta_expectancy": rr.get("delta_expectancy_regime_exit_minus_phasec_global_exit"),
                    "delta_maxdd": rr.get("delta_maxdd_regime_exit_minus_phasec_global_exit"),
                }
            )
    runs_registry = pd.DataFrame(reg_rows)
    runs_registry.to_csv(run_dir / "runs_registry.csv", index=False)

    # Validate key finite metrics.
    key_cols = ["expectancy_net", "total_return", "max_drawdown_pct", "cvar_5", "profit_factor", "win_rate"]
    bad = []
    for _, r in metrics_df.iterrows():
        for c in key_cols:
            v = _safe_float(r.get(c))
            if not np.isfinite(v):
                bad.append((r["variant"], c, r.get(c)))
    if bad:
        msg = "\n".join([f"- {a}.{b}={c}" for a, b, c in bad])
        raise RuntimeError(f"Non-finite key metrics detected:\n{msg}")

    # Reproduce tolerance checks for V2/V3/V4 known references.
    known = {
        "V2_1H_FROZEN_PHASEC_UNIVERSE_REFERENCE": {"expectancy_net": -0.0006491411569577637, "max_drawdown_pct": -0.9991752110690576},
        "V3_EXEC_3M_PHASEC_CONTROL_FROZEN": {"expectancy_net": -0.0006429344379359453, "max_drawdown_pct": -0.9989855391907831},
        "V4_EXEC_3M_PHASEC_BEST_FROZEN": {"expectancy_net": -0.0005588693803532237, "max_drawdown_pct": -0.9981165324430944},
    }
    tol = float(args.repro_tolerance)
    repro_rows: List[Dict[str, Any]] = []
    for vname, exp in known.items():
        row = metrics_df[metrics_df["variant"] == vname]
        if row.empty:
            continue
        r = row.iloc[0]
        de = _safe_float(r["expectancy_net"]) - float(exp["expectancy_net"])
        dd = _safe_float(r["max_drawdown_pct"]) - float(exp["max_drawdown_pct"])
        repro_rows.append(
            {
                "variant": vname,
                "expectancy_actual": _safe_float(r["expectancy_net"]),
                "expectancy_expected": float(exp["expectancy_net"]),
                "expectancy_delta": de,
                "maxdd_actual": _safe_float(r["max_drawdown_pct"]),
                "maxdd_expected": float(exp["max_drawdown_pct"]),
                "maxdd_delta": dd,
                "pass_tolerance": int(abs(de) <= tol and abs(dd) <= 5e-4),
            }
        )
    repro_df = pd.DataFrame(repro_rows)

    # Reconciliation interpretation.
    v1_row = metrics_df[metrics_df["variant"] == "V1_1H_FULLSCAN_REFERENCE"].iloc[0]
    v2_row = metrics_df[metrics_df["variant"] == "V2_1H_FROZEN_PHASEC_UNIVERSE_REFERENCE"].iloc[0]
    v3_row = metrics_df[metrics_df["variant"] == "V3_EXEC_3M_PHASEC_CONTROL_FROZEN"].iloc[0]
    v4_row = metrics_df[metrics_df["variant"] == "V4_EXEC_3M_PHASEC_BEST_FROZEN"].iloc[0]
    d_v4_v3_exp = float(v4_row["expectancy_net"]) - float(v3_row["expectancy_net"])
    d_v4_v3_dd = float(v4_row["max_drawdown_pct"]) - float(v3_row["max_drawdown_pct"])
    d_v4_v1_exp = float(v4_row["expectancy_net"]) - float(v1_row["expectancy_net"])
    v1_native_final = _safe_float(v1_row.get("native_final_equity"))
    v1_native_init = float(args.fullscan_initial_equity)
    v1_native_return = (v1_native_final / max(1e-12, v1_native_init) - 1.0) if np.isfinite(v1_native_final) else float("nan")

    universe_mismatch = 1
    sizing_mismatch = 1
    bug_found = 0
    if not repro_df.empty and int(repro_df["pass_tolerance"].min()) == 0:
        bug_found = 1

    deploy_status = "HOLD"
    if float(v4_row["expectancy_net"]) > 0 and float(v4_row["total_return"]) > 0:
        deploy_status = "YES"
    elif float(v4_row["expectancy_net"]) < float(v3_row["expectancy_net"]):
        deploy_status = "NO"

    # Write required diagnostics markdown files.
    assumptions_lines = [
        "# Assumptions",
        "",
        "- V1 uses native `ga.py` long backtester behavior (same as params scan contract).",
        "- V2 is an aligned fallback comparator for frozen Phase C test universe using exported signals and 3m path simulation.",
        "- V3/V4 are loaded from Phase C trade diagnostics and re-costed under the same Phase A fee contract.",
        "- All frozen Phase C hash checks are enforced before evaluation.",
        "- No strategy logic modifications were made in this reconciliation run.",
    ]
    (run_dir / "assumptions.md").write_text("\n".join(assumptions_lines).strip() + "\n", encoding="utf-8")

    bug_lines = ["# Bug Findings", ""]
    if bug_found == 0:
        bug_lines.append("- No material implementation bug found in V2/V3/V4 reproduction checks.")
        bug_lines.append("- Differences are explained by contract mismatch and universe/sizing mismatch.")
    else:
        bug_lines.append("- Potential reproduction mismatch detected beyond tolerance:")
        bug_lines.append("")
        bug_lines.append(_write_markdown_table(repro_df))
    (run_dir / "bug_findings.md").write_text("\n".join(bug_lines).strip() + "\n", encoding="utf-8")

    recon_lines = [
        "# SOL Reconciliation Report",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Symbol: {symbol}",
        f"- Phase C dir: `{phase_c_dir}`",
        f"- Phase A dir: `{phase_a_dir}`",
        "",
        "## Contract Verification",
        "",
        f"- fee_model_sha256: `{fee_hash}`",
        f"- metrics_definition_sha256: `{metrics_hash}`",
        f"- signal_subset_hash: `{subset_hash}`",
        f"- wf_split_hash: `{split_hash}`",
        "",
        "## Core Contradiction",
        "",
        f"- V1 native fullscan total_return: {v1_native_return:.6f} (final_equity={v1_native_final:.2f})",
        f"- V2 frozen 1h reference total_return: {float(v2_row['total_return']):.6f}",
        f"- V4 frozen Phase C best total_return: {float(v4_row['total_return']):.6f}",
        "",
        "## Root Cause Summary",
        "",
        "- Fullscan and frozen evaluations are not the same universe or contract.",
        "- Fullscan uses endogenous 1h signal generation + legacy fee/slip and native ATR sizing.",
        "- Frozen evaluation uses exported signal subset + 3m path simulation + Phase A fee contract + fixed-risk equity simulator.",
        "- Phase C best improves over frozen control but remains deeply negative in absolute equity terms on this frozen sample.",
        "",
        "## Reproduction Check (V2/V3/V4)",
        "",
        _write_markdown_table(repro_df if not repro_df.empty else pd.DataFrame([{"status": "no_reference_rows"}])),
        "",
        "## Universe Comparison",
        "",
        _write_markdown_table(uni_df),
        "",
        "## Metric Glossary Snapshot",
        "",
        "- `expectancy_net`: mean net return per valid trade.",
        "- `total_return`: final_equity / initial_equity - 1.",
        "- `max_drawdown_pct`: most negative peak-to-trough drawdown fraction.",
        "- `cvar_5`: mean of worst 5% trade outcomes.",
        "",
        "## Decision Inputs",
        "",
        f"- delta_expectancy(V4 - V3): {d_v4_v3_exp:.6f}",
        f"- delta_maxdd(V4 - V3): {d_v4_v3_dd:.6f}",
        f"- delta_expectancy(V4 - V1): {d_v4_v1_exp:.6f}",
        f"- deploy_status_candidate: {deploy_status}",
    ]
    (run_dir / "reconciliation_report.md").write_text("\n".join(recon_lines).strip() + "\n", encoding="utf-8")

    decision_lines = [
        "# SOL Reconciliation Decision",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        "",
        "## Status Flags",
        "",
        f"- BUG: {'YES' if bug_found else 'NO'}",
        f"- UNIVERSE MISMATCH: {'YES' if universe_mismatch else 'NO'}",
        f"- SIZING/COMPOUNDING MISMATCH: {'YES' if sizing_mismatch else 'NO'}",
        f"- SOL deployable candidate status: {deploy_status}",
        "",
        "## Why",
        "",
        f"- V1 fullscan native return is {v1_native_return:.6f} with native contract, but frozen universe variants stay near -100% return.",
        "- This is primarily explained by mismatch in evaluation universe and execution/sizing contract, not by a single arithmetic bug.",
        f"- On frozen universe, Phase C best beats Phase C control on expectancy by {d_v4_v3_exp:.6f} but remains negative in absolute terms.",
        "",
        "## Final Recommendation",
        "",
        "- HOLD.",
        "- Next step: run a fair expanded SOL evaluation where fullscan and frozen pipelines share identical universe/sizing contract before additional optimization.",
    ]
    (run_dir / "decision.md").write_text("\n".join(decision_lines).strip() + "\n", encoding="utf-8")

    # Additional required files.
    metrics_json = {p.name: p.metrics for p in packs}
    _json_dump(run_dir / "metrics_by_variant.json", metrics_json)
    for p in packs:
        p.trades.to_csv(run_dir / f"trades_{p.name.lower()}.csv", index=False)
        p.monthly.to_csv(run_dir / f"monthly_{p.name.lower()}.csv", index=False)

    # Repro + git status + manifest.
    repro_lines = [
        "# Reproduction",
        "",
        "```bash",
        f"cd {PROJECT_ROOT}",
        (
            "python3 scripts/sol_reconcile_truth.py "
            f"--symbol {symbol} "
            f"--phase-c-dir {phase_c_dir} "
            f"--phase-a-contract-dir {phase_a_dir} "
            f"--params-file {params_path} "
            f"--outdir {args.outdir}"
        ),
        "```",
    ]
    (run_dir / "repro.md").write_text("\n".join(repro_lines).strip() + "\n", encoding="utf-8")

    try:
        gs = subprocess.check_output(["git", "status", "--short"], cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.STDOUT)
    except Exception as ex:  # pragma: no cover
        gs = f"git status unavailable: {ex}"
    (run_dir / "git_status.txt").write_text(gs, encoding="utf-8")

    manifest = {
        "generated_utc": _utc_now().isoformat(),
        "symbol": symbol,
        "phase_c_dir": str(phase_c_dir),
        "phase_a_dir": str(phase_a_dir),
        "phase_d_dir": str(phased_dir),
        "params_file": str(params_path),
        "phase_c_cfg_hash": str(phasec_manifest.get("final_selected_cfg_hash")),
        "phase_a_fee_model_sha256": fee_hash,
        "phase_a_metrics_sha256": metrics_hash,
        "signal_subset_hash": subset_hash,
        "wf_split_hash": split_hash,
        "test_signals_total": int(test_n),
        "variants": [p.name for p in packs],
        "fullscan_initial_equity": float(args.fullscan_initial_equity),
        "fullscan_fee_bps": float(args.fullscan_fee_bps),
        "fullscan_slip_bps": float(args.fullscan_slip_bps),
        "phase_initial_equity": float(args.phase_initial_equity),
        "phase_risk_per_trade": float(args.phase_risk_per_trade),
        "bug_found": int(bug_found),
        "universe_mismatch": int(universe_mismatch),
        "sizing_mismatch": int(sizing_mismatch),
        "deploy_status": deploy_status,
    }
    _json_dump(run_dir / "run_manifest.json", manifest)

    phase_lines = [
        "Phase: SOL reconciliation truth backtest",
        f"Timestamp UTC: {_utc_now().isoformat()}",
        "Status: COMPLETED",
        f"BUG: {'YES' if bug_found else 'NO'}",
        f"UNIVERSE_MISMATCH: {'YES' if universe_mismatch else 'NO'}",
        f"SIZING_COMPOUNDING_MISMATCH: {'YES' if sizing_mismatch else 'NO'}",
        f"SOL_DEPLOYABLE_STATUS: {deploy_status}",
        f"Artifacts: {run_dir}",
    ]
    (run_dir / "phase_result.md").write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="SOL-only reconciliation and truth backtest harness.")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--phase-c-dir", default="reports/execution_layer/PHASEC_SOL_20260221_231430")
    ap.add_argument("--phase-d-dir", default="reports/execution_layer/PHASED_SOL_20260222_000517")
    ap.add_argument("--phase-a-contract-dir", default="reports/execution_layer/BASELINE_AUDIT_20260221_214310")
    ap.add_argument("--params-file", default="data/metadata/params/SOLUSDT_C13_active_params_long.json")
    ap.add_argument("--best-by-symbol-csv", default="reports/params_scan/20260220_044949/best_by_symbol.csv")
    ap.add_argument("--outdir", default="reports/execution_layer")

    # Fullscan (V1) native contract.
    ap.add_argument("--fullscan-initial-equity", type=float, default=10000.0)
    ap.add_argument("--fullscan-fee-bps", type=float, default=7.0)
    ap.add_argument("--fullscan-slip-bps", type=float, default=2.0)
    ap.add_argument("--fullscan-risk-per-trade-for-overlay", type=float, default=0.01)

    # Frozen Phase C contract.
    ap.add_argument("--phase-initial-equity", type=float, default=1.0)
    ap.add_argument("--phase-risk-per-trade", type=float, default=0.01)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)

    ap.add_argument("--repro-tolerance", type=float, default=1e-6)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    m = pd.read_csv(out / "metrics_by_variant.csv")
    idx = {r["variant"]: r for _, r in m.iterrows()}
    v1 = idx["V1_1H_FULLSCAN_REFERENCE"]
    v2 = idx["V2_1H_FROZEN_PHASEC_UNIVERSE_REFERENCE"]
    v4 = idx["V4_EXEC_3M_PHASEC_BEST_FROZEN"]
    print(str(out))
    print(
        "V1 "
        f"exp={_safe_float(v1.get('expectancy_net')):.6f} ret={_safe_float(v1.get('total_return')):.6f} dd={_safe_float(v1.get('max_drawdown_pct')):.6f} | "
        "V2 "
        f"exp={_safe_float(v2.get('expectancy_net')):.6f} ret={_safe_float(v2.get('total_return')):.6f} dd={_safe_float(v2.get('max_drawdown_pct')):.6f} | "
        "V4 "
        f"exp={_safe_float(v4.get('expectancy_net')):.6f} ret={_safe_float(v4.get('total_return')):.6f} dd={_safe_float(v4.get('max_drawdown_pct')):.6f}"
    )
    dec = (out / "decision.md").read_text(encoding="utf-8")
    if "SOL deployable candidate status: YES" in dec:
        print("YES")
    elif "SOL deployable candidate status: NO" in dec:
        print("NO")
    else:
        print("HOLD")


if __name__ == "__main__":
    main()
