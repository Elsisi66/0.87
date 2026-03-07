from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _pf(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    gp = float(pnl[pnl > 0.0].sum()) if (pnl > 0.0).any() else 0.0
    gl = float(pnl[pnl < 0.0].sum()) if (pnl < 0.0).any() else 0.0
    return float(gp / abs(gl)) if gl < -1e-12 else (10.0 if gp > 0 else 0.0)


def _max_dd(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    runmax = np.maximum.accumulate(equity)
    dd = (runmax - equity) / np.maximum(runmax, 1e-12)
    return float(dd.max()) if dd.size else 0.0


def audit_baseline(
    *,
    symbol: str,
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_equity: float,
    max_allocation: float,
    equity_cap: float = 100_000.0,
    out_path: str | Path | None = None,
) -> Dict[str, Any]:
    d = df.copy()
    for c in ["Open", "High", "Low", "Close", "ATR"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    if d.empty:
        raise RuntimeError(f"[{symbol}] baseline audit failed: empty df")

    close = d["Close"].astype(float)
    atr = pd.to_numeric(d.get("ATR", pd.Series(np.nan, index=d.index)), errors="coerce")
    atr = atr.replace([np.inf, -np.inf], np.nan).dropna()

    close_min = float(close.min())
    close_med = float(close.median())
    close_max = float(close.max())
    close_ratio = float(close_max / max(close_min, 1e-12))
    atr_min = float(atr.min()) if not atr.empty else 0.0
    atr_med = float(atr.median()) if not atr.empty else 0.0
    atr_max = float(atr.max()) if not atr.empty else 0.0

    violations: List[str] = []
    if not np.isfinite(close_min) or close_min <= 0:
        violations.append("PRICE_NON_POSITIVE")
    if not np.isfinite(close_med) or close_med <= 0:
        violations.append("PRICE_MEDIAN_INVALID")
    close_pct = {
        "p01": float(close.quantile(0.01)),
        "p05": float(close.quantile(0.05)),
        "p50": float(close.quantile(0.50)),
        "p95": float(close.quantile(0.95)),
        "p99": float(close.quantile(0.99)),
    }

    if close_ratio > 1_000_000:
        violations.append("PRICE_SCALE_RATIO_TOO_HIGH")
    if close_max > 1_000_000_000:
        violations.append("PRICE_MAX_TOO_HIGH")

    t = trades_df.copy()
    if t.empty:
        violations.append("NO_TRADES")
    for c in ["entry_px", "exit_px", "units", "net_pnl"]:
        t[c] = pd.to_numeric(t.get(c), errors="coerce")
    t["entry_ts"] = pd.to_datetime(t.get("entry_ts"), utc=True, errors="coerce")
    t["exit_ts"] = pd.to_datetime(t.get("exit_ts"), utc=True, errors="coerce")
    t = t.dropna(subset=["entry_px", "exit_px", "units", "net_pnl"]).sort_values(["entry_ts", "exit_ts"]).reset_index(drop=True)
    if not t.empty:
        prev_exit = t["exit_ts"].shift(1)
        overlap = (t["entry_ts"] < prev_exit).fillna(False)
        if bool(overlap.any()):
            violations.append("OVERLAPPING_POSITIONS")

    sample = []
    cash = float(initial_equity)
    max_units_seen = 0.0
    max_equity_seen = float(initial_equity)
    min_equity_seen = float(initial_equity)
    max_notional_exposure = 0.0
    max_alloc_observed = 0.0
    realized = []
    eps = 1e-9
    close_by_ts = pd.Series(close.values, index=pd.to_datetime(d["Timestamp"], utc=True).astype("int64")).to_dict()
    audit_rows: List[Dict[str, Any]] = []

    for i, r in t.iterrows():
        entry_px = float(r["entry_px"])
        exit_px = float(r["exit_px"])
        units = float(r["units"])
        pnl = float(r["net_pnl"])
        entry_ts = pd.to_datetime(r["entry_ts"], utc=True)
        exit_ts = pd.to_datetime(r["exit_ts"], utc=True)
        if not np.isfinite(entry_px) or not np.isfinite(exit_px) or entry_px <= 0 or exit_px <= 0:
            violations.append("BAD_PRICE_IN_TRADES")
            continue
        if not np.isfinite(units) or units < 0:
            violations.append("BAD_UNITS")
            continue
        max_units_seen = max(max_units_seen, units)

        pnl_calc = float((exit_px - entry_px) * units)
        if abs(pnl_calc - pnl) > max(1e-6, 0.01 * max(1.0, abs(pnl))):
            violations.append("PNL_FORMULA_MISMATCH")

        max_units_allowed = (cash * float(max_allocation)) / max(entry_px, 1e-12)
        if units - max_units_allowed > 1e-9:
            violations.append("UNITS_EXCEED_MAX_ALLOCATION")

        cash_before = float(cash)
        cost = units * entry_px
        if cost - cash > 1e-6:
            violations.append("POSITION_COST_EXCEEDS_CASH")

        if entry_ts.value in close_by_ts:
            c_entry = float(close_by_ts[int(entry_ts.value)])
            if not (0.5 * c_entry <= entry_px <= 1.5 * c_entry):
                violations.append("ENTRY_PRICE_OUTSIDE_BAND")
        else:
            violations.append("ENTRY_TS_NOT_FOUND")
        if exit_ts.value in close_by_ts:
            c_exit = float(close_by_ts[int(exit_ts.value)])
            if not (0.5 * c_exit <= exit_px <= 1.5 * c_exit):
                violations.append("EXIT_PRICE_OUTSIDE_BAND")
        else:
            violations.append("EXIT_TS_NOT_FOUND")

        notional = float(units * entry_px)
        max_notional_exposure = max(max_notional_exposure, notional)
        alloc_obs = notional / max(cash_before, eps)
        max_alloc_observed = max(max_alloc_observed, alloc_obs)

        cash = cash - cost + (units * exit_px)
        max_equity_seen = max(max_equity_seen, cash)
        min_equity_seen = min(min_equity_seen, cash)
        realized.append(pnl)

        audit_rows.append(
            {
                "entry_ts": str(entry_ts),
                "exit_ts": str(exit_ts),
                "entry_px": entry_px,
                "exit_px": exit_px,
                "units": units,
                "pnl": pnl,
                "cash_before": cash_before,
                "cash_after": float(cash),
                "cost": float(cost),
                "max_units_allowed": float(max_units_allowed),
                "alloc_observed": float(alloc_obs),
            }
        )

        if i < 5:
            sample.append(
                {
                    "entry_ts": str(entry_ts),
                    "exit_ts": str(exit_ts),
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "units": units,
                    "pnl": pnl,
                    "cash_before": cash_before,
                    "cash_after": float(cash),
                }
            )

        if cash < -1e-6:
            violations.append("NEGATIVE_EQUITY")
            break
        if cash > float(equity_cap):
            violations.append("EQUITY_CAP_BREACH")
            break

    pnl_np = np.asarray(realized, dtype=float)
    eq = np.concatenate([[float(initial_equity)], float(initial_equity) + np.cumsum(pnl_np)]) if pnl_np.size else np.array([float(initial_equity)], dtype=float)
    audit = {
        "symbol": symbol.upper(),
        "price_stats": {
            "close_min": close_min,
            "close_median": close_med,
            "close_max": close_max,
            "close_ratio_max_min": close_ratio,
            "close_percentiles": close_pct,
            "atr_min": atr_min,
            "atr_median": atr_med,
            "atr_max": atr_max,
        },
        "position_stats": {
            "max_units_seen": float(max_units_seen),
            "max_notional_exposure": float(max_notional_exposure),
            "max_alloc_observed": float(max_alloc_observed),
            "max_equity_seen": float(max_equity_seen),
            "min_equity_seen": float(min_equity_seen),
            "initial_equity": float(initial_equity),
            "max_allocation": float(max_allocation),
            "equity_cap": float(equity_cap),
        },
        "pnl_stats": {
            "trade_count": int(len(t)),
            "net": float(pnl_np.sum()) if pnl_np.size else 0.0,
            "pf": float(_pf(pnl_np)),
            "dd": float(_max_dd(eq)),
        },
        "sample_trades": sample,
        "worst_trades": (
            pd.DataFrame(audit_rows)
            .sort_values("pnl", ascending=True)
            .head(5)
            .to_dict(orient="records")
            if audit_rows
            else []
        ),
        "violations": sorted(set(violations)),
        "ok": len(set(violations)) == 0,
    }

    if out_path is not None:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        pd.DataFrame(audit_rows).to_csv(op.parent / "audit_trades.csv", index=False)
        if not audit["ok"]:
            snap = {
                "symbol": symbol.upper(),
                "violation_count": len(audit["violations"]),
                "violations": audit["violations"],
                "last_trade": audit_rows[-1] if audit_rows else None,
            }
            (op.parent / "debug_snapshot.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")

    if not audit["ok"]:
        raise RuntimeError(f"[{symbol}] baseline audit violations: {audit['violations']}")
    return audit
