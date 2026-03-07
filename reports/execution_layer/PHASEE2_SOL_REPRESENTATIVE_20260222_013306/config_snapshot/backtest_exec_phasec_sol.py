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
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import execution_layer_3m_ict as exec3m  # noqa: E402


EXPECTED_PHASEC_CFG_HASH = "a285b86c4c22a26976d4a762"
EXPECTED_SIGNAL_SUBSET_HASH = "5e719faf676dffba8d7da926314997182d429361495884b8a870c3393c079bbf"
EXPECTED_SPLIT_HASH = "388ba743b9c16c291385a9ecab6435eabf65eb16f1e1083eee76627193c42c01"


@dataclass
class FeeModel:
    fee_bps_maker: float
    fee_bps_taker: float
    slippage_bps_limit: float
    slippage_bps_market: float


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


def _sha256_signal_subset(df: pd.DataFrame) -> str:
    cols = ["signal_id", "signal_time"]
    for c in cols:
        if c not in df.columns:
            raise RuntimeError(f"signal subset missing required column `{c}`")
    rows = []
    x = df.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    for r in x.itertuples(index=False):
        sid = str(getattr(r, "signal_id"))
        st = pd.to_datetime(getattr(r, "signal_time"), utc=True)
        rows.append(f"{sid}|{st.isoformat()}")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def _json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(v, utc=True))
        if isinstance(v, Path):
            return str(v)
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def _load_fee_model(path: Path) -> FeeModel:
    obj = json.loads(path.read_text(encoding="utf-8"))
    tight = obj.get("tight_pipeline_fee_model", {})
    return FeeModel(
        fee_bps_maker=float(tight.get("fee_bps_maker", 2.0)),
        fee_bps_taker=float(tight.get("fee_bps_taker", 4.0)),
        slippage_bps_limit=float(tight.get("slippage_bps_limit", 0.5)),
        slippage_bps_market=float(tight.get("slippage_bps_market", 2.0)),
    )


def _ensure_hash(path: Path, expected: str, label: str) -> str:
    got = _sha256_file(path)
    if str(expected).strip() and got != str(expected).strip():
        raise RuntimeError(f"{label} hash mismatch: expected={expected} got={got}")
    return got


def _read_phasec_contract(phase_c_dir: Path) -> Dict[str, Any]:
    manifest_fp = phase_c_dir / "run_manifest.json"
    if not manifest_fp.exists():
        raise FileNotFoundError(f"Missing Phase C manifest: {manifest_fp}")
    man = json.loads(manifest_fp.read_text(encoding="utf-8"))
    return man


def _load_split_definition(path: Path) -> List[Dict[str, int]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    splits = []
    for s in obj.get("splits", []):
        splits.append(
            {
                "split_id": int(s["split_id"]),
                "train_start": int(s["train_start"]),
                "train_end": int(s["train_end"]),
                "test_start": int(s["test_start"]),
                "test_end": int(s["test_end"]),
            }
        )
    return sorted(splits, key=lambda x: x["split_id"])


def _test_signal_index_set(splits: List[Dict[str, int]]) -> List[int]:
    out: List[int] = []
    for sp in splits:
        out.extend(list(range(int(sp["test_start"]), int(sp["test_end"]))))
    return sorted(set(out))


def _build_split_lookup(subset_df: pd.DataFrame, splits: List[Dict[str, int]]) -> Dict[str, int]:
    idx_to_split: Dict[int, int] = {}
    for sp in splits:
        sid = int(sp["split_id"])
        for i in range(int(sp["test_start"]), int(sp["test_end"])):
            idx_to_split[int(i)] = sid
    out: Dict[str, int] = {}
    for i, r in subset_df.reset_index(drop=True).iterrows():
        if i in idx_to_split:
            out[str(r["signal_id"])] = int(idx_to_split[i])
    return out


def _load_trade_diagnostics_csv(path: Path, split_lookup: Dict[str, int], variant: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing trade diagnostics csv: {path}")
    df = pd.read_csv(path)
    for c in ["signal_time", "entry_time", "exit_time"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    if "signal_id" not in df.columns:
        raise RuntimeError(f"{path} missing signal_id")
    if "split_id" not in df.columns:
        df["split_id"] = df["signal_id"].astype(str).map(split_lookup)
    df["variant"] = str(variant)
    return df


def _cost_row(entry: float, exit_: float, liq: str, fee: FeeModel) -> Dict[str, float]:
    c = exec3m._costed_pnl_long(
        entry_price=float(entry),
        exit_price=float(exit_),
        entry_liquidity_type=str(liq),
        fee_bps_maker=float(fee.fee_bps_maker),
        fee_bps_taker=float(fee.fee_bps_taker),
        slippage_bps_limit=float(fee.slippage_bps_limit),
        slippage_bps_market=float(fee.slippage_bps_market),
    )
    return {
        "pnl_gross_pct": float(c["pnl_gross_pct"]),
        "pnl_net_pct": float(c["pnl_net_pct"]),
        "entry_fee_bps": float(c["entry_fee_bps"]),
        "exit_fee_bps": float(c["exit_fee_bps"]),
        "entry_slippage_bps": float(c["entry_slippage_bps"]),
        "exit_slippage_bps": float(c["exit_slippage_bps"]),
        "total_cost_bps": float(c["total_cost_bps"]),
    }


def _normalize_exec_rows(
    df: pd.DataFrame,
    *,
    fee: FeeModel,
    default_liq: str,
) -> pd.DataFrame:
    x = df.copy()
    for c in ["filled", "valid_for_metrics", "sl_hit", "tp_hit"]:
        x[c] = pd.to_numeric(x.get(c, 0), errors="coerce").fillna(0).astype(int)
    for c in ["entry_price", "exit_price", "signal_tp_mult", "signal_sl_mult", "hold_minutes", "pnl_net_pct"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    if "entry_type" not in x.columns:
        x["entry_type"] = "market"
    x["fill_liquidity_type"] = x["entry_type"].fillna("market").astype(str).str.lower().map(
        lambda s: "maker" if s.startswith("limit") else str(default_liq)
    )
    x["fill_delay_min"] = (
        (pd.to_datetime(x["entry_time"], utc=True, errors="coerce") - pd.to_datetime(x["signal_time"], utc=True, errors="coerce"))
        .dt.total_seconds()
        .div(60.0)
    )
    rows: List[Dict[str, Any]] = []
    for r in x.itertuples(index=False):
        entry = float(getattr(r, "entry_price", np.nan))
        exit_ = float(getattr(r, "exit_price", np.nan))
        liq = str(getattr(r, "fill_liquidity_type", default_liq))
        c = _cost_row(entry, exit_, liq, fee) if (np.isfinite(entry) and np.isfinite(exit_) and entry > 0) else {
            "pnl_gross_pct": np.nan,
            "pnl_net_pct": np.nan,
            "entry_fee_bps": np.nan,
            "exit_fee_bps": np.nan,
            "entry_slippage_bps": np.nan,
            "exit_slippage_bps": np.nan,
            "total_cost_bps": np.nan,
        }
        rows.append(c)
    cdf = pd.DataFrame(rows)
    for k in cdf.columns:
        x[k] = pd.to_numeric(cdf[k], errors="coerce")
    x["risk_pct"] = (1.0 - pd.to_numeric(x.get("signal_sl_mult", np.nan), errors="coerce")).clip(lower=1e-8)
    x["hold_minutes"] = pd.to_numeric(x.get("hold_minutes", np.nan), errors="coerce")
    return x


def _simulate_1h_reference(
    *,
    signals_df: pd.DataFrame,
    split_lookup: Dict[str, int],
    fee: FeeModel,
    exec_horizon_hours: float,
    symbol: str,
) -> pd.DataFrame:
    full_fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"
    if not full_fp.exists():
        raise FileNotFoundError(f"Missing 1h parquet for reference variant: {full_fp}")
    k = pd.read_parquet(full_fp)
    k = exec3m._normalize_ohlcv_cols(k)
    k["Timestamp"] = pd.to_datetime(k["Timestamp"], utc=True, errors="coerce")
    k = k.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).sort_values("Timestamp").reset_index(drop=True)

    ts_ns = np.array([int(t.value) for t in pd.to_datetime(k["Timestamp"], utc=True)], dtype=np.int64)
    open_np = pd.to_numeric(k["Open"], errors="coerce").to_numpy(dtype=float)
    high_np = pd.to_numeric(k["High"], errors="coerce").to_numpy(dtype=float)
    low_np = pd.to_numeric(k["Low"], errors="coerce").to_numpy(dtype=float)
    close_np = pd.to_numeric(k["Close"], errors="coerce").to_numpy(dtype=float)

    rows: List[Dict[str, Any]] = []
    for r in signals_df.itertuples(index=False):
        sid = str(getattr(r, "signal_id"))
        st = pd.to_datetime(getattr(r, "signal_time"), utc=True)
        tp_mult = float(getattr(r, "tp_mult"))
        sl_mult = float(getattr(r, "sl_mult"))
        sig_ns = int(st.value)
        idx = int(np.searchsorted(ts_ns, sig_ns, side="left"))
        base = {
            "symbol": str(symbol),
            "signal_id": sid,
            "signal_time": st,
            "split_id": int(split_lookup.get(sid, -1)),
            "signal_tp_mult": float(tp_mult),
            "signal_sl_mult": float(sl_mult),
            "variant": "1H_REFERENCE_CONTROL",
            "entry_type": "market",
            "fill_liquidity_type": "taker",
            "entry_improvement_bps": 0.0,
        }
        if idx >= len(ts_ns):
            rows.append(
                {
                    **base,
                    "filled": 0,
                    "valid_for_metrics": 0,
                    "sl_hit": 0,
                    "tp_hit": 0,
                    "entry_time": pd.NaT,
                    "exit_time": pd.NaT,
                    "entry_price": np.nan,
                    "exit_price": np.nan,
                    "exit_reason": "no_bar_after_signal",
                    "mae_pct": np.nan,
                    "mfe_pct": np.nan,
                    "fill_delay_min": np.nan,
                    "hold_minutes": np.nan,
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

        entry_price = float(open_np[idx])
        sl = float(entry_price * sl_mult)
        tp = float(entry_price * tp_mult)
        max_exit_ts_ns = exec3m._compute_eval_end_ns(
            entry_ts_ns=int(ts_ns[idx]),
            eval_horizon_hours=float(exec_horizon_hours),
            baseline_exit_time=None,
        )
        sim = exec3m._simulate_path_long(
            ts_ns=ts_ns,
            close=close_np,
            high=high_np,
            low=low_np,
            entry_idx=int(idx),
            entry_price=float(entry_price),
            sl_price=float(sl),
            tp_price=float(tp),
            max_exit_ts_ns=int(max_exit_ts_ns),
        )
        liq = "taker"
        c = _cost_row(float(sim.get("entry_price", np.nan)), float(sim.get("exit_price", np.nan)), liq, fee)
        et = pd.to_datetime(sim.get("entry_time"), utc=True, errors="coerce")
        xt = pd.to_datetime(sim.get("exit_time"), utc=True, errors="coerce")
        rows.append(
            {
                **base,
                "filled": int(bool(sim.get("filled", False))),
                "valid_for_metrics": int(sim.get("valid_for_metrics", 0)),
                "sl_hit": int(bool(sim.get("sl_hit", False))),
                "tp_hit": int(bool(sim.get("tp_hit", False))),
                "entry_time": et,
                "exit_time": xt,
                "entry_price": float(sim.get("entry_price", np.nan)),
                "exit_price": float(sim.get("exit_price", np.nan)),
                "exit_reason": str(sim.get("exit_reason", "")),
                "mae_pct": float(sim.get("mae_pct", np.nan)),
                "mfe_pct": float(sim.get("mfe_pct", np.nan)),
                "fill_delay_min": float((et - st).total_seconds() / 60.0) if pd.notna(et) else np.nan,
                "hold_minutes": float((xt - et).total_seconds() / 60.0) if pd.notna(et) and pd.notna(xt) else np.nan,
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
    df = pd.DataFrame(rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    return df


def _tail_mean(x: np.ndarray, frac: float) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(frac * arr.size)))
    return float(np.mean(np.sort(arr)[:k]))


def _max_drawdown_pct_from_equity(eq: np.ndarray) -> float:
    if eq.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(1e-12, peak) - 1.0
    return float(np.min(dd))


def _compute_equity_curve(
    trades: pd.DataFrame,
    *,
    signals_total: int,
    initial_equity: float,
    risk_per_trade: float,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    x = trades.copy().sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    for c in ["signal_time", "entry_time", "exit_time"]:
        x[c] = pd.to_datetime(x.get(c), utc=True, errors="coerce")
    for c in [
        "filled",
        "valid_for_metrics",
        "sl_hit",
        "tp_hit",
        "pnl_net_pct",
        "pnl_gross_pct",
        "entry_fee_bps",
        "exit_fee_bps",
        "entry_slippage_bps",
        "exit_slippage_bps",
        "total_cost_bps",
        "risk_pct",
        "fill_delay_min",
        "hold_minutes",
    ]:
        x[c] = pd.to_numeric(x.get(c, np.nan), errors="coerce")

    m = (x["filled"] == 1) & (x["valid_for_metrics"] == 1) & x["pnl_net_pct"].notna()
    filled_df = x[m].copy().sort_values(["entry_time", "signal_time"]).reset_index(drop=True)
    n_valid = int(len(filled_df))
    n_signals = int(signals_total)

    equity = float(initial_equity)
    eq_rows: List[Dict[str, Any]] = []
    total_notional = 0.0
    total_fees = 0.0
    total_slip = 0.0

    for r in filled_df.itertuples(index=False):
        risk_pct = float(getattr(r, "risk_pct", np.nan))
        risk_pct = max(1e-8, risk_pct) if np.isfinite(risk_pct) else 1e-3
        pnl = float(getattr(r, "pnl_net_pct"))
        if equity <= 0:
            pos_notional = 0.0
        else:
            pos_notional = float(equity * float(risk_per_trade) / risk_pct)
        trade_pnl_abs = float(pos_notional * pnl)
        fee_bps = float(getattr(r, "entry_fee_bps", 0.0)) + float(getattr(r, "exit_fee_bps", 0.0))
        slip_bps = float(getattr(r, "entry_slippage_bps", 0.0)) + float(getattr(r, "exit_slippage_bps", 0.0))
        fees_abs = float(pos_notional * fee_bps / 1e4)
        slip_abs = float(pos_notional * slip_bps / 1e4)
        total_notional += float(2.0 * pos_notional)
        total_fees += float(fees_abs)
        total_slip += float(slip_abs)
        equity = float(equity + trade_pnl_abs)
        ts = pd.to_datetime(getattr(r, "exit_time"), utc=True, errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(getattr(r, "signal_time"), utc=True, errors="coerce")
        eq_rows.append(
            {
                "timestamp": ts,
                "signal_id": str(getattr(r, "signal_id")),
                "split_id": int(getattr(r, "split_id")) if np.isfinite(getattr(r, "split_id", np.nan)) else -1,
                "equity": float(equity),
                "trade_pnl_abs": float(trade_pnl_abs),
                "trade_pnl_pct": float(pnl),
                "position_notional": float(pos_notional),
                "fees_abs": float(fees_abs),
                "slippage_abs": float(slip_abs),
            }
        )

    eq_df = pd.DataFrame(eq_rows)
    if eq_df.empty:
        eq_df = pd.DataFrame(
            [
                {
                    "timestamp": pd.NaT,
                    "signal_id": "",
                    "split_id": -1,
                    "equity": float(initial_equity),
                    "trade_pnl_abs": 0.0,
                    "trade_pnl_pct": 0.0,
                    "position_notional": 0.0,
                    "fees_abs": 0.0,
                    "slippage_abs": 0.0,
                    "drawdown_pct": 0.0,
                    "drawdown_abs": 0.0,
                }
            ]
        )
    else:
        eq_df = eq_df.sort_values("timestamp").reset_index(drop=True)
        peak = eq_df["equity"].cummax()
        eq_df["drawdown_pct"] = eq_df["equity"] / peak - 1.0
        eq_df["drawdown_abs"] = eq_df["equity"] - peak

    # Monthly returns from equity curve.
    if eq_df["timestamp"].notna().any():
        ser = (
            eq_df.dropna(subset=["timestamp"])
            .set_index(pd.to_datetime(eq_df.dropna(subset=["timestamp"])["timestamp"], utc=True))["equity"]
            .sort_index()
        )
        ser = ser.groupby(level=0).last()
        monthly_end = ser.resample("ME").last().ffill()
        prev = monthly_end.shift(1).fillna(float(initial_equity))
        monthly_ret = monthly_end / np.maximum(1e-12, prev) - 1.0
        monthly_df = pd.DataFrame(
            {
                "month": monthly_end.index.strftime("%Y-%m"),
                "equity_end": monthly_end.values.astype(float),
                "monthly_return": monthly_ret.values.astype(float),
            }
        )
    else:
        monthly_df = pd.DataFrame(columns=["month", "equity_end", "monthly_return"])

    ret_net = pd.to_numeric(filled_df["pnl_net_pct"], errors="coerce").dropna().to_numpy(dtype=float)
    ret_gross = pd.to_numeric(filled_df["pnl_gross_pct"], errors="coerce").dropna().to_numpy(dtype=float)
    pos = ret_net[ret_net > 0]
    neg = ret_net[ret_net < 0]

    expectancy_net = float(np.mean(ret_net)) if ret_net.size else float("nan")
    expectancy_gross = float(np.mean(ret_gross)) if ret_gross.size else float("nan")
    per_signal_vec = np.zeros(n_signals, dtype=float)
    if n_signals == len(x):
        vals = pd.to_numeric(x["pnl_net_pct"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        valid_sig = ((x["filled"] == 1) & (x["valid_for_metrics"] == 1)).to_numpy(dtype=bool)
        per_signal_vec[valid_sig] = vals[valid_sig]
    else:
        per_signal_vec = np.zeros(max(1, n_signals), dtype=float)
    expectancy_net_per_signal = float(np.mean(per_signal_vec)) if per_signal_vec.size else float("nan")

    total_return = float(eq_df["equity"].iloc[-1] / max(1e-12, float(initial_equity)) - 1.0)
    if eq_df["timestamp"].notna().any():
        t0 = pd.to_datetime(eq_df["timestamp"].dropna().iloc[0], utc=True)
        t1 = pd.to_datetime(eq_df["timestamp"].dropna().iloc[-1], utc=True)
        years = max(1e-9, float((t1 - t0).total_seconds()) / (365.25 * 24.0 * 3600.0))
    else:
        years = float("nan")
    cagr = float((eq_df["equity"].iloc[-1] / max(1e-12, float(initial_equity))) ** (1.0 / years) - 1.0) if np.isfinite(years) and years > 0 else float("nan")

    pnl_std = float(np.std(ret_net, ddof=0)) if ret_net.size else float("nan")
    downside = ret_net[ret_net < 0]
    downside_std = float(np.std(downside, ddof=0)) if downside.size else float("nan")
    trades_per_year = float(n_valid / years) if np.isfinite(years) and years > 0 else float("nan")
    sharpe = float((expectancy_net / pnl_std) * np.sqrt(trades_per_year)) if ret_net.size and np.isfinite(pnl_std) and pnl_std > 1e-12 and np.isfinite(trades_per_year) else float("nan")
    sortino = float((expectancy_net / downside_std) * np.sqrt(trades_per_year)) if ret_net.size and np.isfinite(downside_std) and downside_std > 1e-12 and np.isfinite(trades_per_year) else float("nan")
    vol_ann = float(pnl_std * np.sqrt(trades_per_year)) if np.isfinite(pnl_std) and np.isfinite(trades_per_year) else float("nan")

    max_dd_pct = float(eq_df["drawdown_pct"].min()) if "drawdown_pct" in eq_df.columns else float("nan")
    max_dd_abs = float(eq_df["drawdown_abs"].min()) if "drawdown_abs" in eq_df.columns else float("nan")
    cvar_5 = float(_tail_mean(ret_net, 0.05)) if ret_net.size else float("nan")
    profit_factor = float(np.sum(pos) / abs(np.sum(neg))) if neg.size and abs(np.sum(neg)) > 1e-12 else float("inf") if pos.size else float("nan")

    wins = int((ret_net > 0).sum())
    losses = int((ret_net < 0).sum())
    win_rate = float(wins / max(1, n_valid))

    hold = pd.to_numeric(filled_df.get("hold_minutes", np.nan), errors="coerce").dropna()
    avg_hold = float(hold.mean()) if not hold.empty else float("nan")
    med_hold = float(hold.median()) if not hold.empty else float("nan")
    p95_hold = float(hold.quantile(0.95)) if not hold.empty else float("nan")

    if x["signal_time"].notna().any():
        t0s = pd.to_datetime(x["signal_time"].dropna().min(), utc=True)
        t1s = pd.to_datetime(x["exit_time"].dropna().max(), utc=True) if x["exit_time"].notna().any() else pd.to_datetime(x["signal_time"].dropna().max(), utc=True)
        total_window_min = max(1e-9, float((t1s - t0s).total_seconds()) / 60.0)
    else:
        total_window_min = float("nan")
    exposure = float(hold.sum() / total_window_min) if np.isfinite(total_window_min) and total_window_min > 0 and not hold.empty else float("nan")

    m_valid = (x["filled"] == 1) & (x["valid_for_metrics"] == 1)
    delays = pd.to_numeric(x.loc[m_valid, "fill_delay_min"], errors="coerce").dropna()
    median_delay = float(delays.median()) if not delays.empty else float("nan")
    p95_delay = float(delays.quantile(0.95)) if not delays.empty else float("nan")
    liq = x.loc[m_valid, "fill_liquidity_type"].fillna("").astype(str).str.lower() if "fill_liquidity_type" in x.columns else pd.Series(dtype=str)
    taker_share = float((liq == "taker").mean()) if len(liq) else float("nan")

    exit_dist = (
        x.loc[m_valid, "exit_reason"].fillna("unknown").astype(str).str.lower().value_counts(dropna=False).sort_index().to_dict()
        if "exit_reason" in x.columns
        else {}
    )

    metrics = {
        "signals_total": int(n_signals),
        "trades_total": int(n_valid),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(win_rate),
        "expectancy_gross": float(expectancy_gross),
        "expectancy_net": float(expectancy_net),
        "expectancy_net_per_signal": float(expectancy_net_per_signal),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "profit_factor": float(profit_factor),
        "max_drawdown_abs": float(max_dd_abs),
        "max_drawdown_pct": float(max_dd_pct),
        "cvar_5": float(cvar_5),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "volatility_annualized": float(vol_ann),
        "avg_trade_return": float(expectancy_net),
        "median_trade_return": float(np.median(ret_net)) if ret_net.size else float("nan"),
        "average_hold_time_min": float(avg_hold),
        "median_hold_time_min": float(med_hold),
        "p95_hold_time_min": float(p95_hold),
        "exposure_time_pct": float(exposure),
        "total_fees_paid": float(total_fees),
        "total_slippage_paid": float(total_slip),
        "turnover_notional": float(total_notional),
        "entry_rate": float(n_valid / max(1, n_signals)),
        "participation": float(n_valid / max(1, n_signals)),
        "taker_share": float(taker_share),
        "median_fill_delay_min": float(median_delay),
        "p95_fill_delay_min": float(p95_delay),
        "exit_reason_distribution": {str(k): int(v) for k, v in exit_dist.items()},
        "annualized_return": float(cagr),
    }
    return eq_df, metrics, monthly_df


def _finite_check(metrics: Dict[str, Any], keys: Iterable[str]) -> None:
    for k in keys:
        v = metrics.get(k, np.nan)
        if not np.isfinite(float(v)):
            raise RuntimeError(f"Invalid metric `{k}` = {v}")


def _metrics_row(variant: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = {"variant": str(variant)}
    out.update(metrics)
    out["exit_reason_distribution"] = json.dumps(metrics.get("exit_reason_distribution", {}), sort_keys=True)
    return out


def _comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    idx = {r["variant"]: r for _, r in metrics_df.iterrows()}
    a = idx["1H_REFERENCE_CONTROL"]
    b = idx["EXEC_3M_CONTROL"]
    c = idx["EXEC_3M_PHASEC_BEST"]
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
    rows = []
    for m in metrics:
        av = float(a.get(m, np.nan))
        bv = float(b.get(m, np.nan))
        cv = float(c.get(m, np.nan))
        rows.append(
            {
                "metric": str(m),
                "1h_reference_control": av,
                "exec_3m_control": bv,
                "exec_3m_phasec_best": cv,
                "delta_exec3m_control_minus_1h_reference": float(bv - av) if np.isfinite(av) and np.isfinite(bv) else np.nan,
                "delta_exec3m_phasec_minus_exec3m_control": float(cv - bv) if np.isfinite(cv) and np.isfinite(bv) else np.nan,
                "delta_exec3m_phasec_minus_1h_reference": float(cv - av) if np.isfinite(cv) and np.isfinite(av) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _split_variant_metrics(
    trades_by_variant: Dict[str, pd.DataFrame],
    split_def: List[Dict[str, int]],
    *,
    initial_equity: float,
    risk_per_trade: float,
    signal_subset: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ss = signal_subset.reset_index(drop=True)
    for sp in split_def:
        sid = int(sp["split_id"])
        lo, hi = int(sp["test_start"]), int(sp["test_end"])
        split_signals = ss.iloc[lo:hi]
        sig_ids = set(split_signals["signal_id"].astype(str).tolist())
        sig_n = int(len(split_signals))
        for variant, df in trades_by_variant.items():
            x = df[df["signal_id"].astype(str).isin(sig_ids)].copy()
            _, m, _ = _compute_equity_curve(
                x,
                signals_total=sig_n,
                initial_equity=float(initial_equity),
                risk_per_trade=float(risk_per_trade),
            )
            rows.append(
                {
                    "variant": variant,
                    "split_id": sid,
                    "signals_total": sig_n,
                    "trades_total": int(m["trades_total"]),
                    "expectancy_net": float(m["expectancy_net"]),
                    "total_return": float(m["total_return"]),
                    "max_drawdown_pct": float(m["max_drawdown_pct"]),
                    "cvar_5": float(m["cvar_5"]),
                    "profit_factor": float(m["profit_factor"]),
                    "win_rate": float(m["win_rate"]),
                    "taker_share": float(m["taker_share"]),
                    "median_fill_delay_min": float(m["median_fill_delay_min"]),
                    "p95_fill_delay_min": float(m["p95_fill_delay_min"]),
                }
            )
    return pd.DataFrame(rows)


def _write_summary_md(
    path: Path,
    run_dir: Path,
    metrics_df: pd.DataFrame,
    cmp_df: pd.DataFrame,
    phase_c_dir: Path,
    phase_a_dir: Path,
    note_1h: str,
) -> None:
    idx = {r["variant"]: r for _, r in metrics_df.iterrows()}
    a = idx["1H_REFERENCE_CONTROL"]
    b = idx["EXEC_3M_CONTROL"]
    c = idx["EXEC_3M_PHASEC_BEST"]
    lines = [
        "# SOL Execution-Layer Backtest Summary",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Run dir: `{run_dir}`",
        f"- Phase C source: `{phase_c_dir}`",
        f"- Phase A contract: `{phase_a_dir}`",
        f"- 1h reference note: {note_1h}",
        "",
        "## Topline",
        "",
        f"- 1H_REFERENCE_CONTROL expectancy_net / total_return / maxDD: {a['expectancy_net']:.6f} / {a['total_return']:.6f} / {a['max_drawdown_pct']:.6f}",
        f"- EXEC_3M_CONTROL expectancy_net / total_return / maxDD: {b['expectancy_net']:.6f} / {b['total_return']:.6f} / {b['max_drawdown_pct']:.6f}",
        f"- EXEC_3M_PHASEC_BEST expectancy_net / total_return / maxDD: {c['expectancy_net']:.6f} / {c['total_return']:.6f} / {c['max_drawdown_pct']:.6f}",
        "",
        "## Required Deltas",
        "",
    ]
    for _, r in cmp_df.iterrows():
        m = str(r["metric"])
        if m in {"expectancy_net", "total_return", "max_drawdown_pct", "cvar_5", "profit_factor", "win_rate"}:
            lines.append(
                f"- {m}: ctrl-1h={r['delta_exec3m_control_minus_1h_reference']:.6f}, "
                f"phasec-ctrl={r['delta_exec3m_phasec_minus_exec3m_control']:.6f}, "
                f"phasec-1h={r['delta_exec3m_phasec_minus_1h_reference']:.6f}"
            )
    lines.append("")
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_decision_md(
    path: Path,
    metrics_df: pd.DataFrame,
    cmp_df: pd.DataFrame,
    phasec_cfg_hash: str,
    note_1h: str,
) -> str:
    idx = {r["variant"]: r for _, r in metrics_df.iterrows()}
    a = idx["1H_REFERENCE_CONTROL"]
    b = idx["EXEC_3M_CONTROL"]
    c = idx["EXEC_3M_PHASEC_BEST"]

    d_phasec_vs_ctrl = float(c["expectancy_net"] - b["expectancy_net"])
    d_phasec_vs_1h = float(c["expectancy_net"] - a["expectancy_net"])
    better_vs_ctrl = int(np.isfinite(d_phasec_vs_ctrl) and d_phasec_vs_ctrl > 0 and float(c["total_return"]) > float(b["total_return"]))
    better_vs_1h = int(np.isfinite(d_phasec_vs_1h) and d_phasec_vs_1h > 0 and float(c["total_return"]) > float(a["total_return"]))
    dd_not_worse_ctrl = int(np.isfinite(float(c["max_drawdown_pct"])) and np.isfinite(float(b["max_drawdown_pct"])) and float(c["max_drawdown_pct"]) >= float(b["max_drawdown_pct"]) - 1e-12)
    dd_not_worse_1h = int(np.isfinite(float(c["max_drawdown_pct"])) and np.isfinite(float(a["max_drawdown_pct"])) and float(c["max_drawdown_pct"]) >= float(a["max_drawdown_pct"]) - 1e-12)

    positive_practical = int(np.isfinite(float(c["expectancy_net"])) and np.isfinite(float(c["total_return"])) and float(c["expectancy_net"]) > 0.0 and float(c["total_return"]) > 0.0)
    proceed = int(
        positive_practical == 1
        and better_vs_ctrl == 1
        and better_vs_1h == 1
        and dd_not_worse_ctrl == 1
        and dd_not_worse_1h == 1
    )
    rec = "PROCEED_TO_PAPER_SOL" if proceed == 1 else "HOLD"

    lines = [
        "# SOL Backtest Decision",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Phase C cfg hash: `{phasec_cfg_hash}`",
        f"- 1h reference note: {note_1h}",
        "",
        "## Practical Equity Conclusion",
        "",
        f"- Phase C best vs 3m control: expectancy delta = {d_phasec_vs_ctrl:.6f}, total_return delta = {float(c['total_return']) - float(b['total_return']):.6f}",
        f"- Phase C best vs 1h reference: expectancy delta = {d_phasec_vs_1h:.6f}, total_return delta = {float(c['total_return']) - float(a['total_return']):.6f}",
        f"- MaxDD phasec/control/1h: {float(c['max_drawdown_pct']):.6f} / {float(b['max_drawdown_pct']):.6f} / {float(a['max_drawdown_pct']):.6f}",
        "",
        "## Behavior Changes (3m control -> 3m phasec)",
        "",
        f"- Hold time median (min): {float(b['median_hold_time_min']):.2f} -> {float(c['median_hold_time_min']):.2f}",
        f"- Win rate: {float(b['win_rate']):.4f} -> {float(c['win_rate']):.4f}",
        f"- Profit factor: {float(b['profit_factor']):.4f} -> {float(c['profit_factor']):.4f}",
        f"- Fees paid: {float(b['total_fees_paid']):.6f} -> {float(c['total_fees_paid']):.6f}",
        "",
        "## Recommendation",
        "",
        f"- Final: **{rec}**",
    ]
    if rec == "PROCEED_TO_PAPER_SOL":
        lines.extend(
            [
                "- Config to use (Phase C best):",
                "  - tp_mult=1.0, sl_mult=0.75, time_stop_min=720, break_even_enabled=0, break_even_trigger_r=0.5, break_even_offset_bps=0.0, partial_take_enabled=0, partial_take_r=0.8, partial_take_pct=0.25",
                f"  - cfg_hash={phasec_cfg_hash}",
            ]
        )
    else:
        lines.extend(
            [
                "- Blockers:",
                f"  - positive_practical={positive_practical}",
                f"  - better_vs_3m_control={better_vs_ctrl}",
                f"  - better_vs_1h_reference={better_vs_1h}",
                f"  - maxdd_not_worse_vs_control={dd_not_worse_ctrl}",
                f"  - maxdd_not_worse_vs_1h={dd_not_worse_1h}",
            ]
        )
    lines.append("")
    lines.append("## Key Deltas Table")
    lines.append("")
    for _, r in cmp_df.iterrows():
        lines.append(
            f"- {r['metric']}: ctrl-1h={r['delta_exec3m_control_minus_1h_reference']:.6f}, "
            f"phasec-ctrl={r['delta_exec3m_phasec_minus_exec3m_control']:.6f}, "
            f"phasec-1h={r['delta_exec3m_phasec_minus_1h_reference']:.6f}"
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return rec


def run(args: argparse.Namespace) -> Path:
    phase_c_dir = _resolve(args.phase_c_dir)
    if not phase_c_dir.exists():
        raise FileNotFoundError(f"Phase C dir not found: {phase_c_dir}")

    man = _read_phasec_contract(phase_c_dir)
    symbol = str(args.symbol).upper().strip()
    if str(man.get("symbol", "")).upper() != symbol:
        raise RuntimeError(f"Phase C manifest symbol mismatch: expected {symbol} got {man.get('symbol')}")
    cfg_hash = str(man.get("final_selected_cfg_hash", ""))
    if cfg_hash != EXPECTED_PHASEC_CFG_HASH:
        raise RuntimeError(f"Unexpected Phase C cfg hash: expected {EXPECTED_PHASEC_CFG_HASH} got {cfg_hash}")

    phase_a = man.get("phase_a_contract", {})
    phase_a_dir = _resolve(str(phase_a.get("dir", args.phase_a_contract_dir)))
    fee_model_path = _resolve(str(phase_a.get("fee_model_path", phase_a_dir / "fee_model.json")))
    metrics_def_path = _resolve(str(phase_a.get("metrics_definition_path", phase_a_dir / "metrics_definition.md")))
    fee_model_hash = _ensure_hash(fee_model_path, str(phase_a.get("fee_model_sha256", "")), "phase_a fee_model")
    metrics_hash = _ensure_hash(metrics_def_path, str(phase_a.get("metrics_definition_sha256", "")), "phase_a metrics_definition")

    signal_subset_path = _resolve(str(man.get("signal_subset_path", phase_c_dir / "signal_subset.csv")))
    wf_split_path = _resolve(str(man.get("split_definition_path", phase_c_dir / "wf_split_definition.json")))
    signal_subset_df = pd.read_csv(signal_subset_path)
    calc_subset_hash = _sha256_signal_subset(signal_subset_df)
    expected_subset_hash = str(man.get("signal_subset_hash", EXPECTED_SIGNAL_SUBSET_HASH))
    if calc_subset_hash != expected_subset_hash or calc_subset_hash != EXPECTED_SIGNAL_SUBSET_HASH:
        raise RuntimeError(
            f"signal subset hash mismatch: expected(manifest/global)={expected_subset_hash}/{EXPECTED_SIGNAL_SUBSET_HASH} got={calc_subset_hash}"
        )

    split_hash = _sha256_file(wf_split_path)
    expected_split_hash = str(man.get("split_definition_sha256", EXPECTED_SPLIT_HASH))
    if split_hash != expected_split_hash or split_hash != EXPECTED_SPLIT_HASH:
        raise RuntimeError(
            f"wf split hash mismatch: expected(manifest/global)={expected_split_hash}/{EXPECTED_SPLIT_HASH} got={split_hash}"
        )

    split_def = _load_split_definition(wf_split_path)
    test_idx = _test_signal_index_set(split_def)
    subset = signal_subset_df.reset_index(drop=True).copy()
    subset["signal_time"] = pd.to_datetime(subset["signal_time"], utc=True, errors="coerce")
    subset["tp_mult"] = pd.to_numeric(subset.get("strategy_tp_mult", subset.get("tp_mult", np.nan)), errors="coerce")
    subset["sl_mult"] = pd.to_numeric(subset.get("strategy_sl_mult", subset.get("sl_mult", np.nan)), errors="coerce")
    test_signals = subset.iloc[test_idx].copy().reset_index(drop=True)
    split_lookup = _build_split_lookup(subset, split_def)

    run_root = _resolve(args.outdir) / f"BACKTEST_SOL_PHASEC_{_utc_tag()}"
    run_root.mkdir(parents=True, exist_ok=False)
    snap = run_root / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fee_model_path, run_root / "fee_model.json")
    shutil.copy2(metrics_def_path, run_root / "metrics_definition.md")
    shutil.copy2(fee_model_path, snap / "fee_model.json")
    shutil.copy2(metrics_def_path, snap / "metrics_definition.md")
    shutil.copy2(signal_subset_path, snap / "signal_subset.csv")
    (snap / "signal_subset_hash.txt").write_text(calc_subset_hash + "\n", encoding="utf-8")
    shutil.copy2(wf_split_path, snap / "wf_split_definition.json")
    shutil.copy2(phase_c_dir / "run_manifest.json", snap / "phasec_run_manifest.json")
    if (PROJECT_ROOT / "configs" / "execution_configs.yaml").exists():
        shutil.copy2(PROJECT_ROOT / "configs" / "execution_configs.yaml", snap / "execution_configs.yaml")

    fee = _load_fee_model(fee_model_path)

    t_base = _load_trade_diagnostics_csv(phase_c_dir / "trade_diagnostics_baseline.csv", split_lookup, "EXEC_3M_CONTROL")
    t_best = _load_trade_diagnostics_csv(phase_c_dir / "trade_diagnostics_best.csv", split_lookup, "EXEC_3M_PHASEC_BEST")
    test_ids = set(test_signals["signal_id"].astype(str).tolist())
    t_base = t_base[t_base["signal_id"].astype(str).isin(test_ids)].copy().reset_index(drop=True)
    t_best = t_best[t_best["signal_id"].astype(str).isin(test_ids)].copy().reset_index(drop=True)
    t_base = _normalize_exec_rows(t_base, fee=fee, default_liq="taker")
    t_best = _normalize_exec_rows(t_best, fee=fee, default_liq="taker")

    t_1h = _simulate_1h_reference(
        signals_df=test_signals[["signal_id", "signal_time", "tp_mult", "sl_mult"]].copy(),
        split_lookup=split_lookup,
        fee=fee,
        exec_horizon_hours=float(args.exec_horizon_hours),
        symbol=symbol,
    )

    # Save trade tables.
    t_1h.to_csv(run_root / "trades_1h_reference.csv", index=False)
    t_base.to_csv(run_root / "trades_exec3m_control.csv", index=False)
    t_best.to_csv(run_root / "trades_exec3m_phasec.csv", index=False)

    signals_total_test = int(len(test_signals))
    init_eq = float(args.initial_equity)
    risk_pt = float(args.risk_per_trade)

    eq_1h, m_1h, mon_1h = _compute_equity_curve(t_1h, signals_total=signals_total_test, initial_equity=init_eq, risk_per_trade=risk_pt)
    eq_b, m_b, mon_b = _compute_equity_curve(t_base, signals_total=signals_total_test, initial_equity=init_eq, risk_per_trade=risk_pt)
    eq_c, m_c, mon_c = _compute_equity_curve(t_best, signals_total=signals_total_test, initial_equity=init_eq, risk_per_trade=risk_pt)

    _finite_check(m_1h, ["expectancy_net", "total_return", "max_drawdown_pct", "cvar_5"])
    _finite_check(m_b, ["expectancy_net", "total_return", "max_drawdown_pct", "cvar_5"])
    _finite_check(m_c, ["expectancy_net", "total_return", "max_drawdown_pct", "cvar_5"])

    eq_1h.to_csv(run_root / "equity_curve_1h_reference.csv", index=False)
    eq_b.to_csv(run_root / "equity_curve_exec3m_control.csv", index=False)
    eq_c.to_csv(run_root / "equity_curve_exec3m_phasec.csv", index=False)
    mon_1h.to_csv(run_root / "monthly_returns_1h_reference.csv", index=False)
    mon_b.to_csv(run_root / "monthly_returns_exec3m_control.csv", index=False)
    mon_c.to_csv(run_root / "monthly_returns_exec3m_phasec.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            _metrics_row("1H_REFERENCE_CONTROL", m_1h),
            _metrics_row("EXEC_3M_CONTROL", m_b),
            _metrics_row("EXEC_3M_PHASEC_BEST", m_c),
        ]
    )
    metrics_df.to_csv(run_root / "metrics_by_variant.csv", index=False)
    _json_dump(
        run_root / "metrics_by_variant.json",
        {
            "1H_REFERENCE_CONTROL": m_1h,
            "EXEC_3M_CONTROL": m_b,
            "EXEC_3M_PHASEC_BEST": m_c,
        },
    )

    cmp_df = _comparison_table(metrics_df)
    cmp_df.to_csv(run_root / "comparison_table.csv", index=False)

    split_metrics = _split_variant_metrics(
        {
            "1H_REFERENCE_CONTROL": t_1h,
            "EXEC_3M_CONTROL": t_base,
            "EXEC_3M_PHASEC_BEST": t_best,
        },
        split_def,
        initial_equity=init_eq,
        risk_per_trade=risk_pt,
        signal_subset=subset,
    )
    split_metrics.to_csv(run_root / "split_metrics_by_variant.csv", index=False)

    note_1h = (
        "Aligned 1h proxy using same frozen test signals, next-1h-open entry, "
        "same TP/SL multipliers, same 12h horizon, and identical fee/slippage contract."
    )
    _write_summary_md(
        run_root / "summary.md",
        run_dir=run_root,
        metrics_df=metrics_df,
        cmp_df=cmp_df,
        phase_c_dir=phase_c_dir,
        phase_a_dir=phase_a_dir,
        note_1h=note_1h,
    )
    recommendation = _write_decision_md(
        run_root / "decision.md",
        metrics_df=metrics_df,
        cmp_df=cmp_df,
        phasec_cfg_hash=cfg_hash,
        note_1h=note_1h,
    )

    repro_lines = [
        "# Reproduction",
        "",
        "```bash",
        f"cd {PROJECT_ROOT}",
        (
            "python3 scripts/backtest_exec_phasec_sol.py "
            f"--symbol {symbol} "
            f"--phase-c-dir {phase_c_dir} "
            f"--phase-a-contract-dir {phase_a_dir} "
            f"--outdir {args.outdir} "
            f"--risk-per-trade {risk_pt} "
            f"--initial-equity {init_eq} "
            f"--exec-horizon-hours {float(args.exec_horizon_hours)}"
        ),
        "```",
    ]
    (run_root / "repro.md").write_text("\n".join(repro_lines).strip() + "\n", encoding="utf-8")

    try:
        gs = subprocess.check_output(["git", "status", "--short"], cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.STDOUT)
    except Exception as e:  # pragma: no cover
        gs = f"git status unavailable: {e}"
    (run_root / "git_status.txt").write_text(gs, encoding="utf-8")

    run_manifest = {
        "generated_utc": _utc_now().isoformat(),
        "symbol": symbol,
        "phase_c_dir": str(phase_c_dir),
        "phase_c_cfg_hash": cfg_hash,
        "phase_a_dir": str(phase_a_dir),
        "fee_model_path": str(fee_model_path),
        "fee_model_sha256": fee_model_hash,
        "metrics_definition_path": str(metrics_def_path),
        "metrics_definition_sha256": metrics_hash,
        "signal_subset_path": str(signal_subset_path),
        "signal_subset_hash": calc_subset_hash,
        "wf_split_definition_path": str(wf_split_path),
        "wf_split_definition_sha256": split_hash,
        "signals_total_subset": int(len(subset)),
        "signals_total_test": int(signals_total_test),
        "wf_split_count": int(len(split_def)),
        "risk_per_trade": risk_pt,
        "initial_equity": init_eq,
        "exec_horizon_hours": float(args.exec_horizon_hours),
        "recommendation": recommendation,
        "variants": ["1H_REFERENCE_CONTROL", "EXEC_3M_CONTROL", "EXEC_3M_PHASEC_BEST"],
    }
    _json_dump(run_root / "run_manifest.json", run_manifest)

    phase_lines = [
        "Phase: SOL Phase-C post-pass decision-grade backtest",
        f"Timestamp UTC: {_utc_now().isoformat()}",
        "Status: COMPLETED",
        f"Recommendation: {recommendation}",
        f"Signals total test: {signals_total_test}",
        f"Artifacts: {run_root}",
    ]
    (run_root / "phase_result.md").write_text("\n".join(phase_lines).strip() + "\n", encoding="utf-8")
    return run_root


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Decision-grade SOL backtest after Phase C pass (1h ref vs 3m control vs 3m PhaseC best).")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--phase-c-dir", default="reports/execution_layer/PHASEC_SOL_20260221_231430")
    ap.add_argument("--phase-a-contract-dir", default="reports/execution_layer/BASELINE_AUDIT_20260221_214310")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--risk-per-trade", type=float, default=0.01)
    ap.add_argument("--initial-equity", type=float, default=1.0)
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    out = run(args)
    print(str(out))
    m = pd.read_csv(out / "metrics_by_variant.csv")
    idx = {r["variant"]: r for _, r in m.iterrows()}
    a = idx["1H_REFERENCE_CONTROL"]
    b = idx["EXEC_3M_CONTROL"]
    c = idx["EXEC_3M_PHASEC_BEST"]
    print(
        "1H_REFERENCE_CONTROL "
        f"exp={float(a['expectancy_net']):.6f} ret={float(a['total_return']):.6f} dd={float(a['max_drawdown_pct']):.6f} | "
        "EXEC_3M_CONTROL "
        f"exp={float(b['expectancy_net']):.6f} ret={float(b['total_return']):.6f} dd={float(b['max_drawdown_pct']):.6f} | "
        "EXEC_3M_PHASEC_BEST "
        f"exp={float(c['expectancy_net']):.6f} ret={float(c['total_return']):.6f} dd={float(c['max_drawdown_pct']):.6f}"
    )
    dec = (out / "decision.md").read_text(encoding="utf-8")
    if "PROCEED_TO_PAPER_SOL" in dec:
        print("PROCEED_TO_PAPER_SOL")
    else:
        print("HOLD")


if __name__ == "__main__":
    main()
