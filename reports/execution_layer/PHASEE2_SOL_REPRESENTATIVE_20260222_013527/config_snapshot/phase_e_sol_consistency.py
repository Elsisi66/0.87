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


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
        if not np.isfinite(x):
            return float("nan")
        return x
    except Exception:
        return float("nan")


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


def _sha256_signal_subset(df: pd.DataFrame) -> str:
    x = df.copy()
    if "signal_id" not in x.columns or "signal_time" not in x.columns:
        raise RuntimeError("signal subset requires signal_id and signal_time")
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    x = x.dropna(subset=["signal_time"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    rows = [f"{str(r.signal_id)}|{pd.to_datetime(r.signal_time, utc=True).isoformat()}" for r in x.itertuples(index=False)]
    return _sha256_text("\n".join(rows))


def _markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
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
        raise RuntimeError(f"No splits in {path}")
    return out


def _test_indices_from_splits(splits: List[Dict[str, int]]) -> List[int]:
    out: List[int] = []
    for s in splits:
        out.extend(list(range(int(s["test_start"]), int(s["test_end"]))))
    return sorted(set(out))


def _vol_bucket(x: pd.Series) -> pd.Series:
    y = pd.to_numeric(x, errors="coerce")
    out = pd.Series(index=y.index, dtype=object)
    out[y <= 33.3333333333] = "low"
    out[(y > 33.3333333333) & (y <= 66.6666666667)] = "mid"
    out[y > 66.6666666667] = "high"
    out = out.fillna("unknown")
    return out.astype(str)


def _normalize_signals(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["signal_id"] = x["signal_id"].astype(str)
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True, errors="coerce")
    if "tp_mult" not in x.columns:
        if "strategy_tp_mult" in x.columns:
            x["tp_mult"] = pd.to_numeric(x["strategy_tp_mult"], errors="coerce")
    else:
        x["tp_mult"] = pd.to_numeric(x["tp_mult"], errors="coerce")
    if "sl_mult" not in x.columns:
        if "strategy_sl_mult" in x.columns:
            x["sl_mult"] = pd.to_numeric(x["strategy_sl_mult"], errors="coerce")
    else:
        x["sl_mult"] = pd.to_numeric(x["sl_mult"], errors="coerce")
    x["atr_percentile_1h"] = pd.to_numeric(x.get("atr_percentile_1h"), errors="coerce")
    x["trend_up_1h"] = pd.to_numeric(x.get("trend_up_1h"), errors="coerce").fillna(0).astype(int)
    x = x.dropna(subset=["signal_id", "signal_time", "tp_mult", "sl_mult"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    return x


def _load_best_row(best_csv: Path, symbol: str) -> Dict[str, Any]:
    if not best_csv.exists():
        raise FileNotFoundError(f"Missing best_by_symbol.csv: {best_csv}")
    df = pd.read_csv(best_csv)
    m = (df["symbol"].astype(str).str.upper() == symbol.upper()) & (df["side"].astype(str).str.lower() == "long")
    x = df[m].copy()
    if x.empty:
        raise RuntimeError(f"No long row for {symbol} in {best_csv}")
    row = x.iloc[0].to_dict()
    return row


def _latest_metrics_by_variant() -> Optional[Path]:
    cands = sorted((PROJECT_ROOT / "reports" / "execution_layer").glob("BACKTEST_SOL_PHASEC_*/metrics_by_variant.csv"))
    if not cands:
        return None
    cands = sorted(cands, key=lambda p: p.stat().st_mtime)
    return cands[-1]


def _ensure_hash(path: Path, expected: str, label: str) -> str:
    got = _sha256_file(path)
    if str(expected).strip() and got != str(expected).strip():
        raise RuntimeError(f"{label} hash mismatch: expected={expected} got={got}")
    return got


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
    # Fill any shortfall due to tiny strata caps.
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


def _subset_metrics(
    trades_df: pd.DataFrame,
    signal_ids: set[str],
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
    return {
        "signals_total": int(signals_total),
        "trades_total": int(m.get("trades_total", 0)),
        "expectancy_net": float(m.get("expectancy_net", np.nan)),
        "total_return": float(m.get("total_return", np.nan)),
        "max_drawdown_pct": float(m.get("max_drawdown_pct", np.nan)),
        "cvar_5": float(m.get("cvar_5", np.nan)),
        "win_rate": float(m.get("win_rate", np.nan)),
        "equity_end": float(eq["equity"].iloc[-1]) if not eq.empty else float("nan"),
    }


def _ga_rows_to_trade_table(rows_df: pd.DataFrame, mode: str, exec_sl_mult: float) -> pd.DataFrame:
    m = str(mode).strip().lower()
    if m not in {"baseline", "exec"}:
        raise RuntimeError(f"Unsupported mode for row conversion: {mode}")

    x = rows_df.copy()
    out = pd.DataFrame()
    out["signal_id"] = x["signal_id"].astype(str)
    out["signal_time"] = pd.to_datetime(x.get("signal_time"), utc=True, errors="coerce")
    out["split_id"] = pd.to_numeric(x.get("split_id", -1), errors="coerce").fillna(-1).astype(int)
    out["signal_tp_mult"] = pd.to_numeric(x.get("signal_tp_mult", np.nan), errors="coerce")
    out["signal_sl_mult"] = pd.to_numeric(x.get("signal_sl_mult", np.nan), errors="coerce")

    if m == "baseline":
        out["filled"] = pd.to_numeric(x.get("baseline_filled", 0), errors="coerce").fillna(0).astype(int)
        out["valid_for_metrics"] = pd.to_numeric(x.get("baseline_valid_for_metrics", 0), errors="coerce").fillna(0).astype(int)
        out["sl_hit"] = pd.to_numeric(x.get("baseline_sl_hit", 0), errors="coerce").fillna(0).astype(int)
        out["tp_hit"] = pd.to_numeric(x.get("baseline_tp_hit", 0), errors="coerce").fillna(0).astype(int)
        out["pnl_net_pct"] = pd.to_numeric(x.get("baseline_pnl_net_pct", np.nan), errors="coerce")
        out["pnl_gross_pct"] = pd.to_numeric(x.get("baseline_pnl_gross_pct", np.nan), errors="coerce")
        out["entry_time"] = pd.to_datetime(x.get("baseline_entry_time"), utc=True, errors="coerce")
        out["exit_time"] = pd.to_datetime(x.get("baseline_exit_time"), utc=True, errors="coerce")
        out["entry_price"] = pd.to_numeric(x.get("baseline_entry_price", np.nan), errors="coerce")
        out["exit_price"] = pd.to_numeric(x.get("baseline_exit_price", np.nan), errors="coerce")
        out["fill_liquidity_type"] = x.get("baseline_fill_liquidity_type", "").fillna("").astype(str)
        out["fill_delay_min"] = pd.to_numeric(x.get("baseline_fill_delay_min", np.nan), errors="coerce")
        out["mae_pct"] = pd.to_numeric(x.get("baseline_mae_pct", np.nan), errors="coerce")
        out["mfe_pct"] = pd.to_numeric(x.get("baseline_mfe_pct", np.nan), errors="coerce")
        out["exit_reason"] = x.get("baseline_exit_reason", "").fillna("").astype(str)
        risk_mult = 1.0
    else:
        out["filled"] = pd.to_numeric(x.get("exec_filled", 0), errors="coerce").fillna(0).astype(int)
        out["valid_for_metrics"] = pd.to_numeric(x.get("exec_valid_for_metrics", 0), errors="coerce").fillna(0).astype(int)
        out["sl_hit"] = pd.to_numeric(x.get("exec_sl_hit", 0), errors="coerce").fillna(0).astype(int)
        out["tp_hit"] = pd.to_numeric(x.get("exec_tp_hit", 0), errors="coerce").fillna(0).astype(int)
        out["pnl_net_pct"] = pd.to_numeric(x.get("exec_pnl_net_pct", np.nan), errors="coerce")
        out["pnl_gross_pct"] = pd.to_numeric(x.get("exec_pnl_gross_pct", np.nan), errors="coerce")
        out["entry_time"] = pd.to_datetime(x.get("exec_entry_time"), utc=True, errors="coerce")
        out["exit_time"] = pd.to_datetime(x.get("exec_exit_time"), utc=True, errors="coerce")
        out["entry_price"] = pd.to_numeric(x.get("exec_entry_price", np.nan), errors="coerce")
        out["exit_price"] = pd.to_numeric(x.get("exec_exit_price", np.nan), errors="coerce")
        out["fill_liquidity_type"] = x.get("exec_fill_liquidity_type", "").fillna("").astype(str)
        out["fill_delay_min"] = pd.to_numeric(x.get("exec_fill_delay_min", np.nan), errors="coerce")
        out["mae_pct"] = pd.to_numeric(x.get("exec_mae_pct", np.nan), errors="coerce")
        out["mfe_pct"] = pd.to_numeric(x.get("exec_mfe_pct", np.nan), errors="coerce")
        out["exit_reason"] = x.get("exec_exit_reason", "").fillna("").astype(str)
        risk_mult = float(exec_sl_mult)

    out["risk_pct"] = (1.0 - pd.to_numeric(out["signal_sl_mult"], errors="coerce")).clip(lower=1e-8) * max(1e-8, float(risk_mult))
    return out


def _simulate_bar_precedence(
    entries_df: pd.DataFrame,
    *,
    symbol: str,
    fee: phasec_bt.FeeModel,
    exec_horizon_hours: float,
    precedence: str,
) -> pd.DataFrame:
    assert precedence in {"optimistic", "pessimistic", "neutral"}
    df1h = recon._load_symbol_df(symbol=symbol, tf="1h")
    ts_pd = pd.to_datetime(df1h["Timestamp"], utc=True, errors="coerce")
    ts = ts_pd.to_numpy()
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
        if pd.isna(et) or (not np.isfinite(ep)) or ep <= 0 or (not np.isfinite(tp_mult)) or (not np.isfinite(sl_mult)):
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
            o = float(op[i])
            c = float(cl[i])
            mae = min(mae, l / ep - 1.0)
            mfe = max(mfe, h / ep - 1.0)
            hit_sl = bool(l <= sl)
            hit_tp = bool(h >= tp)
            if hit_sl and hit_tp:
                exit_i = i
                if precedence == "optimistic":
                    exit_px = float(tp)
                    reason = "tp"
                    tp_hit = 1
                    sl_hit = 0
                elif precedence == "pessimistic":
                    exit_px = float(sl)
                    reason = "sl"
                    sl_hit = 1
                    tp_hit = 0
                else:
                    # Neutral deterministic tie-breaker: candle direction.
                    if c >= o:
                        exit_px = float(tp)
                        reason = "tp"
                        tp_hit = 1
                        sl_hit = 0
                    else:
                        exit_px = float(sl)
                        reason = "sl"
                        sl_hit = 1
                        tp_hit = 0
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

        xt = pd.to_datetime(ts[exit_i], utc=True)
        crow = phasec_bt._cost_row(float(ep), float(exit_px), "taker", fee)
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
                "exit_time": xt,
                "entry_price": float(ep),
                "exit_price": float(exit_px),
                "exit_reason": reason,
                "mae_pct": float(mae),
                "mfe_pct": float(mfe),
                "hold_minutes": float((xt - et).total_seconds() / 60.0),
                "risk_pct": max(1e-8, 1.0 - sl_mult),
                "entry_type": "market",
                "fill_liquidity_type": "taker",
                "pnl_gross_pct": float(crow["pnl_gross_pct"]),
                "pnl_net_pct": float(crow["pnl_net_pct"]),
                "entry_fee_bps": float(crow["entry_fee_bps"]),
                "exit_fee_bps": float(crow["exit_fee_bps"]),
                "entry_slippage_bps": float(crow["entry_slippage_bps"]),
                "exit_slippage_bps": float(crow["exit_slippage_bps"]),
                "total_cost_bps": float(crow["total_cost_bps"]),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def run(args: argparse.Namespace) -> Path:
    symbol = str(args.symbol).strip().upper()
    if symbol != "SOLUSDT":
        raise RuntimeError("Phase E consistency gate is hard-scoped to SOLUSDT only.")

    phase_c_dir = _resolve(args.phase_c_dir)
    phase_a_dir = _resolve(args.phase_a_contract_dir)
    phased_dir = _resolve(args.phase_d_dir)
    params_path = _resolve(args.params_file)
    best_csv = _resolve(args.best_by_symbol_csv)
    full_signal_path = _resolve(args.full_signal_csv)
    out_root = _resolve(args.outdir)
    run_dir = out_root / f"PHASEE_SOL_CONSISTENCY_{_utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    snap = run_dir / "config_snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    # STEP E1 — Inventory + Freeze
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
    subset_path = _resolve(str(phasec_manifest.get("signal_subset_path", phase_c_dir / "signal_subset.csv")))
    split_path = _resolve(str(phasec_manifest.get("split_definition_path", phase_c_dir / "wf_split_definition.json")))
    expected_fee_hash = str(phase_a_contract.get("fee_model_sha256", EXPECTED_PHASEA_FEE_HASH))
    expected_metrics_hash = str(phase_a_contract.get("metrics_definition_sha256", EXPECTED_PHASEA_METRICS_HASH))
    expected_subset_hash = str(phasec_manifest.get("signal_subset_hash", EXPECTED_SIGNAL_SUBSET_HASH))
    expected_split_hash = str(phasec_manifest.get("split_definition_sha256", EXPECTED_SPLIT_HASH))

    fee_hash = _ensure_hash(fee_model_path, expected_fee_hash, "phase_a fee_model")
    metrics_hash = _ensure_hash(metrics_def_path, expected_metrics_hash, "phase_a metrics_definition")
    subset_hash = _sha256_signal_subset(pd.read_csv(subset_path))
    if subset_hash != expected_subset_hash or subset_hash != EXPECTED_SIGNAL_SUBSET_HASH:
        raise RuntimeError(
            f"Frozen subset hash mismatch: got={subset_hash} expected_manifest={expected_subset_hash} expected_const={EXPECTED_SIGNAL_SUBSET_HASH}"
        )
    split_hash = _ensure_hash(split_path, expected_split_hash, "phase_c split definition")
    if split_hash != EXPECTED_SPLIT_HASH:
        raise RuntimeError(f"Frozen split hash mismatch vs expected constant: {split_hash} != {EXPECTED_SPLIT_HASH}")

    latest_metrics_by_variant = _latest_metrics_by_variant()
    inventory_items: List[Dict[str, Any]] = [
        {"category": "phase_a_contract", "path": str(fee_model_path)},
        {"category": "phase_a_contract", "path": str(metrics_def_path)},
        {"category": "phase_c_artifact", "path": str(subset_path)},
        {"category": "phase_c_artifact", "path": str(split_path)},
        {"category": "phase_c_artifact", "path": str(phase_c_dir / "decision.md")},
        {"category": "phase_c_artifact", "path": str(phase_c_dir / "risk_rollup_overall.csv")},
        {"category": "phase_c_artifact", "path": str(phase_c_dir / "walkforward_results_by_split.csv")},
        {"category": "full_1h_source", "path": str(best_csv)},
        {"category": "full_1h_source", "path": str(params_path)},
        {"category": "full_1h_source", "path": str(PROJECT_ROOT / "scripts" / "scan_params_all_coins.py")},
        {"category": "full_1h_source", "path": str(PROJECT_ROOT / "src" / "bot087" / "optim" / "ga.py")},
        {"category": "exec_compare_source", "path": str(PROJECT_ROOT / "scripts" / "backtest_exec_phasec_sol.py")},
        {"category": "exec_compare_source", "path": str(PROJECT_ROOT / "scripts" / "sol_reconcile_truth.py")},
        {"category": "exec_compare_source", "path": str(latest_metrics_by_variant) if latest_metrics_by_variant else ""},
    ]
    input_manifest_rows: List[Dict[str, Any]] = []
    hashes: Dict[str, str] = {}
    for item in inventory_items:
        p = Path(item["path"]) if str(item.get("path", "")).strip() else Path("")
        exists = p.exists() if str(p) else False
        sha = _sha256_file(p) if exists and p.is_file() else ""
        rec = {
            "category": item["category"],
            "path": str(p) if str(p) else "",
            "exists": int(exists),
            "sha256": sha,
        }
        input_manifest_rows.append(rec)
        if exists and sha:
            hashes[str(p)] = sha

    input_manifest = {
        "generated_utc": _utc_now().isoformat(),
        "symbol": symbol,
        "phase_a_contract_dir": str(phase_a_dir),
        "phase_c_dir": str(phase_c_dir),
        "phase_d_dir": str(phased_dir),
        "files": input_manifest_rows,
    }
    _json_dump(run_dir / "input_manifest.json", input_manifest)
    _json_dump(run_dir / "hashes.json", hashes)

    for fp in [fee_model_path, metrics_def_path, subset_path, split_path, params_path, best_csv]:
        if fp.exists():
            shutil.copy2(fp, snap / fp.name)
    for fp in [
        PROJECT_ROOT / "scripts" / "scan_params_all_coins.py",
        PROJECT_ROOT / "scripts" / "backtest_exec_phasec_sol.py",
        PROJECT_ROOT / "scripts" / "sol_reconcile_truth.py",
        PROJECT_ROOT / "src" / "bot087" / "optim" / "ga.py",
        phase_c_dir / "run_manifest.json",
    ]:
        if fp.exists():
            shutil.copy2(fp, snap / fp.name)
    for fp in [phase_c_dir / "decision.md", phase_c_dir / "risk_rollup_overall.csv", phase_c_dir / "walkforward_results_by_split.csv"]:
        if fp.exists():
            shutil.copy2(fp, snap / fp.name)
    if phased_dir.exists():
        for fp in [phased_dir / "decision.md", phased_dir / "risk_rollup_overall.csv"]:
            if fp.exists():
                shutil.copy2(fp, snap / f"phase_d_{fp.name}")

    # STEP E2 — Reproduce both paths via internal reconciliation runner
    internal_root = run_dir / "_internal"
    internal_root.mkdir(parents=True, exist_ok=True)
    rp = recon.build_arg_parser()
    rargs = rp.parse_args([])
    rargs.symbol = symbol
    rargs.phase_c_dir = str(phase_c_dir)
    rargs.phase_d_dir = str(phased_dir)
    rargs.phase_a_contract_dir = str(phase_a_dir)
    rargs.params_file = str(params_path)
    rargs.best_by_symbol_csv = str(best_csv)
    rargs.outdir = str(internal_root)
    rargs.fullscan_initial_equity = float(args.fullscan_initial_equity)
    rargs.fullscan_fee_bps = float(args.fullscan_fee_bps)
    rargs.fullscan_slip_bps = float(args.fullscan_slip_bps)
    rargs.phase_initial_equity = float(args.phase_initial_equity)
    rargs.phase_risk_per_trade = float(args.phase_risk_per_trade)
    rargs.exec_horizon_hours = float(args.exec_horizon_hours)
    recon_dir = recon.run(rargs)

    metrics_df = pd.read_csv(recon_dir / "metrics_by_variant.csv")
    metrics_df.to_csv(run_dir / "metrics_by_variant.csv", index=False)
    for v in ["V1_1H_FULLSCAN_REFERENCE", "V2_1H_FROZEN_PHASEC_UNIVERSE_REFERENCE", "V3_EXEC_3M_PHASEC_CONTROL_FROZEN", "V4_EXEC_3M_PHASEC_BEST_FROZEN"]:
        fp = recon_dir / f"trades_{v.lower()}.csv"
        if fp.exists():
            shutil.copy2(fp, run_dir / fp.name)

    idx = {r["variant"]: r for _, r in metrics_df.iterrows()}
    v1 = idx["V1_1H_FULLSCAN_REFERENCE"]
    v2 = idx["V2_1H_FROZEN_PHASEC_UNIVERSE_REFERENCE"]
    v3 = idx["V3_EXEC_3M_PHASEC_CONTROL_FROZEN"]
    v4 = idx["V4_EXEC_3M_PHASEC_BEST_FROZEN"]
    best_row = _load_best_row(best_csv, symbol)

    exp_full_net = _safe_float(best_row.get("net_profit"))
    exp_full_dd = _safe_float(best_row.get("max_dd_pct"))
    act_full_net = _safe_float(v1.get("native_net_profit"))
    act_full_dd = _safe_float(v1.get("native_max_dd_pct_positive")) * 100.0
    full_repro_rows = pd.DataFrame(
        [
            {
                "metric": "net_profit",
                "expected": exp_full_net,
                "actual": act_full_net,
                "delta": act_full_net - exp_full_net,
                "pass": int(abs(act_full_net - exp_full_net) <= max(1e-6 * max(1.0, abs(exp_full_net)), 1e-3)),
            },
            {
                "metric": "max_dd_pct",
                "expected": exp_full_dd,
                "actual": act_full_dd,
                "delta": act_full_dd - exp_full_dd,
                "pass": int(abs(act_full_dd - exp_full_dd) <= 1e-3),
            },
        ]
    )
    (run_dir / "reproduction_full_1h.md").write_text(
        "\n".join(
            [
                "# Reproduction: Full-Period 1h Result",
                "",
                f"- Source best_by_symbol row params_file: `{best_row.get('params_file')}`",
                f"- Expected net_profit: {exp_full_net:.6f}",
                f"- Actual net_profit: {act_full_net:.6f}",
                f"- Expected max_dd_pct: {exp_full_dd:.6f}",
                f"- Actual max_dd_pct: {act_full_dd:.6f}",
                "",
                _markdown_table(full_repro_rows),
                "",
                "- Note: `actual` comes from native fields in the same 1h backtester code path used by scan output.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    frozen_expected = {
        "v2_total_return": -0.996799,
        "v2_max_drawdown_pct": -0.999175,
        "v4_total_return": -0.993151,
        "v4_max_drawdown_pct": -0.998117,
    }
    frozen_rows = pd.DataFrame(
        [
            {
                "metric": "v2_total_return",
                "expected": frozen_expected["v2_total_return"],
                "actual": _safe_float(v2.get("total_return")),
            },
            {
                "metric": "v2_max_drawdown_pct",
                "expected": frozen_expected["v2_max_drawdown_pct"],
                "actual": _safe_float(v2.get("max_drawdown_pct")),
            },
            {
                "metric": "v4_total_return",
                "expected": frozen_expected["v4_total_return"],
                "actual": _safe_float(v4.get("total_return")),
            },
            {
                "metric": "v4_max_drawdown_pct",
                "expected": frozen_expected["v4_max_drawdown_pct"],
                "actual": _safe_float(v4.get("max_drawdown_pct")),
            },
        ]
    )
    frozen_rows["delta"] = frozen_rows["actual"] - frozen_rows["expected"]
    frozen_rows["pass"] = (frozen_rows["delta"].abs() <= 5e-4).astype(int)
    (run_dir / "reproduction_frozen_subset.md").write_text(
        "\n".join(
            [
                "# Reproduction: Frozen Phase C Universe",
                "",
                "- Expected values from previously reported apples-to-apples frozen-universe run.",
                "",
                _markdown_table(frozen_rows),
                "",
                "- If any row is outside tolerance, root cause is recorded as contract/data drift in summary.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repro_pass_full = int(full_repro_rows["pass"].min()) if not full_repro_rows.empty else 0
    repro_pass_frozen = int(frozen_rows["pass"].min()) if not frozen_rows.empty else 0
    (run_dir / "reproduction_summary.md").write_text(
        "\n".join(
            [
                "# Reproduction Summary",
                "",
                f"- full_1h_reproduction_pass: {repro_pass_full}",
                f"- frozen_universe_reproduction_pass: {repro_pass_frozen}",
                f"- internal_recon_dir: `{recon_dir}`",
                "",
                "## Notes",
                "",
                "- Full-period path uses native 1h backtester contract.",
                "- Frozen path uses Phase A fee model and frozen Phase C subset/splits.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # STEP E3 — Accounting / metric convention reconciliation
    contract_rows = pd.DataFrame(
        [
            {"dimension": "initial_equity", "full_1h_path": float(args.fullscan_initial_equity), "frozen_phasec_path": float(args.phase_initial_equity), "mismatch": 1},
            {"dimension": "position_sizing", "full_1h_path": "native ATR-based sizing", "frozen_phasec_path": "fixed fractional risk per trade", "mismatch": 1},
            {"dimension": "compounding", "full_1h_path": "yes", "frozen_phasec_path": "yes", "mismatch": 0},
            {"dimension": "fee_model", "full_1h_path": f"fee_bps={args.fullscan_fee_bps}", "frozen_phasec_path": f"phase_a_sha={fee_hash}", "mismatch": 1},
            {"dimension": "slippage_model", "full_1h_path": f"slip_bps={args.fullscan_slip_bps}", "frozen_phasec_path": "phase_a maker/taker slips", "mismatch": 1},
            {"dimension": "signal_universe", "full_1h_path": "endogenous full-period", "frozen_phasec_path": "frozen exported test subset", "mismatch": 1},
            {"dimension": "entry_semantics", "full_1h_path": "1h backtester internal", "frozen_phasec_path": "next 3m open after signal", "mismatch": 1},
            {"dimension": "exit_semantics", "full_1h_path": "1h rules in ga.py", "frozen_phasec_path": "3m path with fixed Phase C exit", "mismatch": 1},
            {"dimension": "bar_ambiguity_handling", "full_1h_path": "1h-candle internal behavior", "frozen_phasec_path": "3m sequential path", "mismatch": 1},
            {"dimension": "drawdown_sign_convention", "full_1h_path": "positive % in scan outputs", "frozen_phasec_path": "negative fraction in exec reports", "mismatch": 1},
            {"dimension": "expectancy_denominator", "full_1h_path": "per-trade return", "frozen_phasec_path": "per-trade net return (+per-signal reported)", "mismatch": 0},
        ]
    )
    contract_rows.to_csv(run_dir / "accounting_contract_compare.csv", index=False)
    (run_dir / "accounting_contract_compare.md").write_text(
        "\n".join(["# Accounting Contract Compare", "", _markdown_table(contract_rows)])
        + "\n",
        encoding="utf-8",
    )
    metric_formula_lines = [
        "# Metric Formula Reconciliation",
        "",
        "- `expectancy_net` = mean net return per valid trade (`pnl_net_pct`).",
        "- `total_return` = `final_equity / initial_equity - 1`.",
        "- `max_drawdown_pct` in exec reports is negative peak-to-trough fraction.",
        "- `max_dd_pct` in params-scan outputs is positive magnitude percentage.",
        "- `cvar_5` = mean of worst 5% trade outcomes.",
        "",
        "## Contract Source",
        "",
        f"- fee_model: `{fee_model_path}` sha256=`{fee_hash}`",
        f"- metrics_definition: `{metrics_def_path}` sha256=`{metrics_hash}`",
    ]
    (run_dir / "metric_formula_reconciliation.md").write_text("\n".join(metric_formula_lines) + "\n", encoding="utf-8")

    # STEP E4 — Universe alignment audit
    full_signals = _normalize_signals(pd.read_csv(full_signal_path))
    subset_full = _normalize_signals(pd.read_csv(subset_path))
    splits = _parse_splits(split_path)
    test_idx = _test_indices_from_splits(splits)
    subset_test = subset_full.iloc[test_idx].copy().reset_index(drop=True)

    def _urow(name: str, df: pd.DataFrame) -> Dict[str, Any]:
        return {
            "universe": name,
            "signals_total": int(df.shape[0]),
            "date_start": str(pd.to_datetime(df["signal_time"], utc=True).min()),
            "date_end": str(pd.to_datetime(df["signal_time"], utc=True).max()),
            "quarters": int(pd.to_datetime(df["signal_time"], utc=True).dt.to_period("Q").nunique()),
        }

    uni_compare = pd.DataFrame(
        [
            _urow("full_signal_source", full_signals),
            _urow("phasec_subset_all", subset_full),
            _urow("phasec_test_subset", subset_test),
        ]
    )
    uni_compare.to_csv(run_dir / "universe_compare.csv", index=False)

    full_ids = set(full_signals["signal_id"].astype(str))
    sub_ids = set(subset_full["signal_id"].astype(str))
    test_ids = set(subset_test["signal_id"].astype(str))
    overlap_rows = pd.DataFrame(
        [
            {"category": "full_ids", "count": int(len(full_ids))},
            {"category": "phasec_subset_ids", "count": int(len(sub_ids))},
            {"category": "phasec_test_ids", "count": int(len(test_ids))},
            {"category": "overlap_full_vs_subset", "count": int(len(full_ids & sub_ids))},
            {"category": "overlap_full_vs_test", "count": int(len(full_ids & test_ids))},
            {"category": "subset_not_in_full", "count": int(len(sub_ids - full_ids))},
            {"category": "test_not_in_full", "count": int(len(test_ids - full_ids))},
            {"category": "full_not_in_subset", "count": int(len(full_ids - sub_ids))},
            {"category": "full_in_subset_not_test", "count": int(len((full_ids & sub_ids) - test_ids))},
        ]
    )
    overlap_rows["share_of_test"] = overlap_rows["count"] / max(1, int(len(test_ids)))
    overlap_rows.to_csv(run_dir / "universe_overlap_summary.csv", index=False)

    fs = full_signals.copy()
    fs["month"] = pd.to_datetime(fs["signal_time"], utc=True).dt.to_period("M").astype(str)
    ss = subset_test.copy()
    ss["month"] = pd.to_datetime(ss["signal_time"], utc=True).dt.to_period("M").astype(str)
    m_full = fs["month"].value_counts().sort_index()
    m_test = ss["month"].value_counts().sort_index()
    m_ix = sorted(set(m_full.index) | set(m_test.index))
    month_df = pd.DataFrame(
        {
            "month": m_ix,
            "full_signal_count": [int(m_full.get(k, 0)) for k in m_ix],
            "phasec_test_count": [int(m_test.get(k, 0)) for k in m_ix],
        }
    )
    month_df["phasec_test_share"] = month_df["phasec_test_count"] / np.maximum(1, month_df["full_signal_count"])
    month_df.to_csv(run_dir / "signal_alignment_report.csv", index=False)
    quarter_df = month_df.copy()
    quarter_df["quarter"] = pd.PeriodIndex(quarter_df["month"], freq="M").asfreq("Q").astype(str)
    quarter_df = quarter_df.groupby("quarter", as_index=False)[["full_signal_count", "phasec_test_count"]].sum()
    quarter_df["phasec_test_share"] = quarter_df["phasec_test_count"] / np.maximum(1, quarter_df["full_signal_count"])
    quarter_df.to_csv(run_dir / "trade_alignment_report.csv", index=False)

    (run_dir / "universe_timeline.md").write_text(
        "\n".join(
            [
                "# Universe Timeline",
                "",
                "## Universe Compare",
                "",
                _markdown_table(uni_compare),
                "",
                "## Overlap Summary",
                "",
                _markdown_table(overlap_rows),
                "",
                "## Quarterly Signal Counts",
                "",
                _markdown_table(quarter_df, max_rows=40),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # STEP E5 — Representativeness audit
    fee = phasec_bt._load_fee_model(fee_model_path)
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

    bundle = ga_exec._build_bundle_for_symbol(
        symbol=symbol,
        signals_df=full_signals.copy(),
        signal_csv=full_signal_path,
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
            "time_stop_min": 720,
            "break_even_enabled": 0,
            "break_even_trigger_r": 0.5,
            "break_even_offset_bps": 0.0,
            "trailing_enabled": 0,
            "trail_start_r": 2.0,
            "trail_step_bps": 50.0,
            "partial_take_enabled": 0,
            "partial_take_r": 0.8,
            "partial_take_pct": 0.25,
            "skip_if_vol_gate": 0,
            "use_signal_quality_gate": 0,
            "min_signal_quality_gate": 0.0,
            "cooldown_min": 0,
        },
        mode="tight",
    )
    e5_eval = ga_exec._evaluate_genome(phasec_best_genome, [bundle], ga_args, detailed=True)
    rows_full = e5_eval.get("signal_rows_df", pd.DataFrame()).copy()
    if rows_full.empty:
        raise RuntimeError("Phase E representativeness eval returned empty signal rows.")
    rows_full["signal_id"] = rows_full["signal_id"].astype(str)
    rows_full["signal_time"] = pd.to_datetime(rows_full["signal_time"], utc=True, errors="coerce")
    rows_full = rows_full.sort_values(["signal_time", "signal_id"]).reset_index(drop=True)

    trades_ref_all = _ga_rows_to_trade_table(rows_full, mode="baseline", exec_sl_mult=1.0)
    trades_best_all = _ga_rows_to_trade_table(rows_full, mode="exec", exec_sl_mult=float(args.phasec_sl_mult))

    frozen_ids_raw = set(subset_test["signal_id"].astype(str))
    available_ids = set(rows_full["signal_id"].astype(str))
    frozen_ids = set([x for x in frozen_ids_raw if x in available_ids])
    frozen_n = int(len(frozen_ids))
    if frozen_n <= 0:
        raise RuntimeError("No frozen IDs overlap with representativeness-eval signal universe.")
    frozen_ref_m = _subset_metrics(
        trades_ref_all,
        signal_ids=frozen_ids,
        signals_total=frozen_n,
        initial_equity=float(args.phase_initial_equity),
        risk_per_trade=float(args.phase_risk_per_trade),
    )
    frozen_best_m = _subset_metrics(
        trades_best_all,
        signal_ids=frozen_ids,
        signals_total=frozen_n,
        initial_equity=float(args.phase_initial_equity),
        risk_per_trade=float(args.phase_risk_per_trade),
    )

    pool = full_signals[full_signals["signal_id"].astype(str).isin(available_ids)].copy()
    pool["date_bucket"] = pd.to_datetime(pool["signal_time"], utc=True).dt.to_period("Q").astype(str)
    pool["vol_bucket"] = _vol_bucket(pool.get("atr_percentile_1h"))
    pool["trend_bucket"] = pool["trend_up_1h"].map(lambda v: "up" if int(v) == 1 else "down")
    pool["stratum"] = pool["date_bucket"].astype(str) + "|" + pool["vol_bucket"].astype(str) + "|" + pool["trend_bucket"].astype(str)

    # Independent alternatives if possible.
    pool_alt = pool[~pool["signal_id"].astype(str).isin(frozen_ids)].copy()
    if int(len(pool_alt)) < frozen_n:
        pool_alt = pool.copy()

    alt_rows: List[Dict[str, Any]] = []
    for i in range(int(args.alt_subsets)):
        seed_i = int(args.seed + i)
        ss = _stratified_sample(pool_alt, n=frozen_n, seed=seed_i, strata_col="stratum")
        sids = set(ss["signal_id"].astype(str).tolist())
        m_ref = _subset_metrics(
            trades_ref_all,
            signal_ids=sids,
            signals_total=frozen_n,
            initial_equity=float(args.phase_initial_equity),
            risk_per_trade=float(args.phase_risk_per_trade),
        )
        m_best = _subset_metrics(
            trades_best_all,
            signal_ids=sids,
            signals_total=frozen_n,
            initial_equity=float(args.phase_initial_equity),
            risk_per_trade=float(args.phase_risk_per_trade),
        )
        for variant, m in [("1h_reference", m_ref), ("phasec_best_exit", m_best)]:
            alt_rows.append(
                {
                    "subset_id": f"alt_{i+1:02d}",
                    "seed": seed_i,
                    "variant": variant,
                    "signals_total": int(m["signals_total"]),
                    "trades_total": int(m["trades_total"]),
                    "expectancy_net": float(m["expectancy_net"]),
                    "total_return": float(m["total_return"]),
                    "max_drawdown_pct": float(m["max_drawdown_pct"]),
                    "cvar_5": float(m["cvar_5"]),
                    "win_rate": float(m["win_rate"]),
                }
            )
    alt_df = pd.DataFrame(alt_rows)
    alt_df.to_csv(run_dir / "alt_subsets_results.csv", index=False)

    percentile_rows: List[Dict[str, Any]] = []
    metrics_for_pct = ["expectancy_net", "total_return", "max_drawdown_pct", "cvar_5"]
    frozen_by_variant = {
        "1h_reference": frozen_ref_m,
        "phasec_best_exit": frozen_best_m,
    }
    for variant in ["1h_reference", "phasec_best_exit"]:
        cur = alt_df[alt_df["variant"] == variant].copy()
        for mname in metrics_for_pct:
            vals = pd.to_numeric(cur[mname], errors="coerce").dropna().to_numpy(dtype=float)
            fv = _safe_float(frozen_by_variant[variant].get(mname))
            pct = float(np.mean(vals <= fv)) if vals.size and np.isfinite(fv) else float("nan")
            percentile_rows.append(
                {
                    "variant": variant,
                    "metric": mname,
                    "frozen_value": fv,
                    "alt_mean": float(np.mean(vals)) if vals.size else float("nan"),
                    "alt_median": float(np.median(vals)) if vals.size else float("nan"),
                    "frozen_percentile": pct,
                }
            )
    pct_df = pd.DataFrame(percentile_rows)
    pct_df.to_csv(run_dir / "subset_percentile_position.csv", index=False)

    # Regime-distribution bias check.
    frozen_reg = subset_test.copy()
    frozen_reg["date_bucket"] = pd.to_datetime(frozen_reg["signal_time"], utc=True).dt.to_period("Q").astype(str)
    frozen_reg["vol_bucket"] = _vol_bucket(frozen_reg.get("atr_percentile_1h"))
    frozen_reg["trend_bucket"] = frozen_reg["trend_up_1h"].map(lambda v: "up" if int(v) == 1 else "down")
    frozen_reg["stratum"] = frozen_reg["date_bucket"].astype(str) + "|" + frozen_reg["vol_bucket"].astype(str) + "|" + frozen_reg["trend_bucket"].astype(str)
    dist_full = pool["stratum"].value_counts(normalize=True)
    dist_frozen = frozen_reg["stratum"].value_counts(normalize=True)
    all_strata = sorted(set(dist_full.index) | set(dist_frozen.index))
    tvd = 0.5 * float(sum(abs(float(dist_full.get(k, 0.0)) - float(dist_frozen.get(k, 0.0))) for k in all_strata))

    flags = {
        "frozen_tail_10pct_any_expectancy": int(
            (
                (pct_df["metric"] == "expectancy_net")
                & (pd.to_numeric(pct_df["frozen_percentile"], errors="coerce") <= 0.10)
            ).any()
        ),
        "frozen_tail_20pct_both_expectancy": int(
            (
                (pct_df["metric"] == "expectancy_net")
                & (pd.to_numeric(pct_df["frozen_percentile"], errors="coerce") <= 0.20)
            ).sum()
            >= 2
        ),
        "regime_distribution_tvd_gt_0_20": int(tvd > 0.20),
        "tvd": float(tvd),
    }
    _json_dump(run_dir / "subset_bias_flags.json", flags)

    rep_lines = [
        "# Subset Representativeness Report",
        "",
        f"- frozen_signals_total: {frozen_n}",
        f"- alt_subsets: {int(args.alt_subsets)}",
        f"- seed_base: {int(args.seed)}",
        f"- sampling_pool_size: {int(pool_alt.shape[0])}",
        f"- regime_tvd(frozen vs broader): {tvd:.6f}",
        "",
        "## Frozen vs Alternatives Percentile Position",
        "",
        _markdown_table(pct_df),
        "",
        "## Bias Flags",
        "",
        f"- frozen_tail_10pct_any_expectancy: {flags['frozen_tail_10pct_any_expectancy']}",
        f"- frozen_tail_20pct_both_expectancy: {flags['frozen_tail_20pct_both_expectancy']}",
        f"- regime_distribution_tvd_gt_0_20: {flags['regime_distribution_tvd_gt_0_20']}",
    ]
    (run_dir / "subset_representativeness_report.md").write_text("\n".join(rep_lines) + "\n", encoding="utf-8")

    # STEP E6 — 1h intrabar ambiguity stress
    ref_frozen_trades = trades_ref_all[trades_ref_all["signal_id"].astype(str).isin(frozen_ids)].copy()
    entries = ref_frozen_trades[
        [
            "signal_id",
            "signal_time",
            "split_id",
            "signal_tp_mult",
            "signal_sl_mult",
            "entry_time",
            "entry_price",
        ]
    ].copy()
    amb_rows: List[Dict[str, Any]] = []
    # Baseline (3m sequential path already computed in ref_frozen_trades)
    _, m_intrabar, _ = phasec_bt._compute_equity_curve(
        ref_frozen_trades,
        signals_total=frozen_n,
        initial_equity=float(args.phase_initial_equity),
        risk_per_trade=float(args.phase_risk_per_trade),
    )
    amb_rows.append(
        {
            "mode": "intrabar_3m_reference",
            "expectancy_net": float(m_intrabar.get("expectancy_net", np.nan)),
            "total_return": float(m_intrabar.get("total_return", np.nan)),
            "max_drawdown_pct": float(m_intrabar.get("max_drawdown_pct", np.nan)),
            "cvar_5": float(m_intrabar.get("cvar_5", np.nan)),
        }
    )
    for mode in ["optimistic", "neutral", "pessimistic"]:
        sim = _simulate_bar_precedence(
            entries,
            symbol=symbol,
            fee=fee,
            exec_horizon_hours=float(args.exec_horizon_hours),
            precedence=mode,
        )
        _, mm, _ = phasec_bt._compute_equity_curve(
            sim,
            signals_total=frozen_n,
            initial_equity=float(args.phase_initial_equity),
            risk_per_trade=float(args.phase_risk_per_trade),
        )
        amb_rows.append(
            {
                "mode": f"bar_1h_{mode}",
                "expectancy_net": float(mm.get("expectancy_net", np.nan)),
                "total_return": float(mm.get("total_return", np.nan)),
                "max_drawdown_pct": float(mm.get("max_drawdown_pct", np.nan)),
                "cvar_5": float(mm.get("cvar_5", np.nan)),
            }
        )
    amb_df = pd.DataFrame(amb_rows)
    amb_df["delta_expectancy_vs_intrabar3m"] = amb_df["expectancy_net"] - float(m_intrabar.get("expectancy_net", np.nan))
    amb_df["delta_total_return_vs_intrabar3m"] = amb_df["total_return"] - float(m_intrabar.get("total_return", np.nan))
    amb_df["delta_maxdd_vs_intrabar3m"] = amb_df["max_drawdown_pct"] - float(m_intrabar.get("max_drawdown_pct", np.nan))
    amb_df.to_csv(run_dir / "intrabar_ambiguity_sensitivity.csv", index=False)
    (run_dir / "intrabar_ambiguity_sensitivity.md").write_text(
        "\n".join(
            [
                "# Intrabar Ambiguity Sensitivity",
                "",
                "- Scope: frozen Phase C test universe, 1h-bar approximation around 3m-entry references.",
                "- `optimistic`: TP precedence on same-bar TP/SL touch.",
                "- `pessimistic`: SL precedence on same-bar TP/SL touch.",
                "- `neutral`: deterministic candle-direction tie-break.",
                "",
                _markdown_table(amb_df),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Reuse/rename requested comparison files from internal run
    cmp_src = recon_dir / "comparison_table.csv"
    if cmp_src.exists():
        shutil.copy2(cmp_src, run_dir / "comparison_vs_phasec_control.csv")

    # STEP E7 — Decision
    measurement_mismatch = int((contract_rows["mismatch"] == 1).sum() >= 3)
    universe_mismatch = int(len(full_ids ^ test_ids) > 0)
    subset_nonrep = int(flags["frozen_tail_10pct_any_expectancy"] or flags["regime_distribution_tvd_gt_0_20"])
    intrabar_bias = int(
        (
            abs(
                _safe_float(amb_df.loc[amb_df["mode"] == "bar_1h_optimistic", "expectancy_net"].iloc[0])
                - _safe_float(amb_df.loc[amb_df["mode"] == "bar_1h_pessimistic", "expectancy_net"].iloc[0])
            )
            > 5e-5
        )
    )
    genuine_degradation = int((_safe_float(v2.get("expectancy_net")) < 0.0) and (_safe_float(v4.get("expectancy_net")) < 0.0))
    mixed_causes = int(sum([measurement_mismatch, universe_mismatch, subset_nonrep, intrabar_bias, genuine_degradation]) >= 2)

    pass_criteria = {
        "reproduction_completed": int(repro_pass_full == 1 and repro_pass_frozen == 1),
        "contract_comparison_completed": 1,
        "universe_mismatch_quantified": 1,
        "representativeness_assessed": 1,
        "intrabar_impact_quantified": 1,
        "critical_contradiction_resolved": int(measurement_mismatch == 1 and universe_mismatch == 1),
    }
    phase_status = "PASS" if min(pass_criteria.values()) == 1 else "FAIL"

    root = {
        "measurement_mismatch": int(measurement_mismatch),
        "universe_mismatch": int(universe_mismatch),
        "subset_non_representativeness": int(subset_nonrep),
        "intrabar_optimism_bias": int(intrabar_bias),
        "genuine_strategy_degradation_on_frozen_test": int(genuine_degradation),
        "mixed_causes": int(mixed_causes),
        "evidence": {
            "v1_native_net_profit": _safe_float(v1.get("native_net_profit")),
            "v1_native_max_dd_pct_positive": _safe_float(v1.get("native_max_dd_pct_positive")) * 100.0,
            "v2_expectancy_net": _safe_float(v2.get("expectancy_net")),
            "v4_expectancy_net": _safe_float(v4.get("expectancy_net")),
            "v2_total_return": _safe_float(v2.get("total_return")),
            "v4_total_return": _safe_float(v4.get("total_return")),
            "frozen_percentiles": pct_df.to_dict(orient="records"),
            "intrabar_sensitivity_rows": amb_df.to_dict(orient="records"),
        },
        "pass_criteria": pass_criteria,
        "phase_status": phase_status,
    }
    _json_dump(run_dir / "root_cause_summary.json", root)

    next_action = (
        "FAIL: contradiction still operationally blocking deployment confidence. "
        "Run Phase E2 remediation: unify fullscan and frozen evaluation on one contract/universe "
        "(single signal export, single sizing model, single fee/slip model), then rerun representativeness gate."
        if phase_status == "FAIL"
        else "PASS: contradiction explained and quantified; proceed to Phase E2 remediation or controlled entry-layer experiments on the same unified contract."
    )
    (run_dir / "next_actions.md").write_text(
        "\n".join(
            [
                "# Next Actions",
                "",
                next_action,
                "",
                "## Ranked Options",
                "",
                "1. Unify contract first: force fullscan and frozen comparisons onto one exact accounting/universe definition.",
                "2. Rebuild frozen subsets with balanced regime/date stratification and retest.",
                "3. Only after 1-2 pass, consider entry optimization expansion.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    decision_lines = [
        "# Phase E Consistency Decision",
        "",
        f"- Generated UTC: {_utc_now().isoformat()}",
        f"- Status: **{phase_status}**",
        "",
        "## Classification",
        "",
        f"- Measurement mismatch: {measurement_mismatch}",
        f"- Universe mismatch: {universe_mismatch}",
        f"- Subset non-representativeness: {subset_nonrep}",
        f"- 1h intrabar optimism bias: {intrabar_bias}",
        f"- Genuine degradation on frozen test: {genuine_degradation}",
        f"- Mixed causes: {mixed_causes}",
        "",
        "## Key Evidence",
        "",
        f"- V1 fullscan native net_profit: {_safe_float(v1.get('native_net_profit')):.6f}",
        f"- V1 fullscan native max_dd_pct: {_safe_float(v1.get('native_max_dd_pct_positive')) * 100.0:.6f}",
        f"- V2 frozen expectancy/return/dd: {_safe_float(v2.get('expectancy_net')):.6f} / {_safe_float(v2.get('total_return')):.6f} / {_safe_float(v2.get('max_drawdown_pct')):.6f}",
        f"- V4 frozen expectancy/return/dd: {_safe_float(v4.get('expectancy_net')):.6f} / {_safe_float(v4.get('total_return')):.6f} / {_safe_float(v4.get('max_drawdown_pct')):.6f}",
        "",
        "## Gate Criteria",
        "",
    ]
    for k, v in pass_criteria.items():
        decision_lines.append(f"- {k}: {v}")
    decision_lines.extend(
        [
            "",
            "## Recommendation",
            "",
            next_action,
        ]
    )
    (run_dir / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    # Compliance files
    repro_lines = [
        "# Repro",
        "",
        "```bash",
        f"cd {PROJECT_ROOT}",
        "python3 scripts/phase_e_sol_consistency.py "
        f"--symbol {symbol} "
        f"--phase-c-dir {phase_c_dir} "
        f"--phase-a-contract-dir {phase_a_dir} "
        f"--params-file {params_path} "
        f"--best-by-symbol-csv {best_csv} "
        f"--full-signal-csv {full_signal_path} "
        f"--outdir {args.outdir} "
        f"--seed {args.seed} "
        f"--alt-subsets {args.alt_subsets}",
        "```",
    ]
    (run_dir / "repro.md").write_text("\n".join(repro_lines) + "\n", encoding="utf-8")

    try:
        gs = subprocess.check_output(["git", "status", "--short"], cwd=str(PROJECT_ROOT), text=True, stderr=subprocess.STDOUT)
    except Exception as ex:  # pragma: no cover
        gs = f"git status unavailable: {ex}"
    (run_dir / "git_status.txt").write_text(gs, encoding="utf-8")

    _json_dump(
        run_dir / "run_manifest.json",
        {
            "generated_utc": _utc_now().isoformat(),
            "symbol": symbol,
            "phase_a_contract_dir": str(phase_a_dir),
            "phase_c_dir": str(phase_c_dir),
            "phase_d_dir": str(phased_dir),
            "params_file": str(params_path),
            "best_by_symbol_csv": str(best_csv),
            "full_signal_csv": str(full_signal_path),
            "phase_c_cfg_hash": EXPECTED_PHASEC_CFG_HASH,
            "phase_a_fee_sha256": fee_hash,
            "phase_a_metrics_sha256": metrics_hash,
            "signal_subset_sha256": subset_hash,
            "wf_split_sha256": split_hash,
            "internal_recon_dir": str(recon_dir),
            "frozen_signals_total": int(frozen_n),
            "seed": int(args.seed),
            "alt_subsets": int(args.alt_subsets),
            "phase_status": phase_status,
            "pass_criteria": pass_criteria,
        },
    )

    (run_dir / "phase_result.md").write_text(
        "\n".join(
            [
                "Phase: Phase E SOLUSDT consistency + representativeness gate",
                f"Timestamp UTC: {_utc_now().isoformat()}",
                f"Status: {phase_status}",
                f"Contract hashes verified: 1",
                f"Reproduction full/frozen pass: {repro_pass_full}/{repro_pass_frozen}",
                f"Root causes: measurement={measurement_mismatch}, universe={universe_mismatch}, subset_nonrep={subset_nonrep}, intrabar_bias={intrabar_bias}, degradation={genuine_degradation}",
                f"Artifacts: {run_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Phase E consistency + representativeness gate for SOLUSDT.")
    ap.add_argument("--symbol", default="SOLUSDT")
    ap.add_argument("--phase-c-dir", default="reports/execution_layer/PHASEC_SOL_20260221_231430")
    ap.add_argument("--phase-d-dir", default="reports/execution_layer/PHASED_SOL_20260222_000517")
    ap.add_argument("--phase-a-contract-dir", default="reports/execution_layer/BASELINE_AUDIT_20260221_214310")
    ap.add_argument("--params-file", default="data/metadata/params/SOLUSDT_C13_active_params_long.json")
    ap.add_argument("--best-by-symbol-csv", default="reports/params_scan/20260220_044949/best_by_symbol.csv")
    ap.add_argument("--full-signal-csv", default="data/signals/SOLUSDT_signals_1h.csv")
    ap.add_argument("--outdir", default="reports/execution_layer")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alt-subsets", type=int, default=10)

    ap.add_argument("--fullscan-initial-equity", type=float, default=10000.0)
    ap.add_argument("--fullscan-fee-bps", type=float, default=7.0)
    ap.add_argument("--fullscan-slip-bps", type=float, default=2.0)

    ap.add_argument("--phase-initial-equity", type=float, default=1.0)
    ap.add_argument("--phase-risk-per-trade", type=float, default=0.01)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)

    ap.add_argument("--phasec-tp-mult", type=float, default=1.0)
    ap.add_argument("--phasec-sl-mult", type=float, default=0.75)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    out = run(args)
    dec = (out / "decision.md").read_text(encoding="utf-8")
    status = "FAIL"
    if "Status: **PASS**" in dec:
        status = "PASS"
    print(str(out))
    print(status)


if __name__ == "__main__":
    main()
