#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import execution_layer_3m_ict as exec3m  # noqa: E402


KILLZONES_DEFAULT: List[Tuple[int, int]] = [(7 * 60, 10 * 60), (13 * 60, 16 * 60)]
ENTRY_MODES = ["market", "limit", "hybrid"]
DEFAULT_CANONICAL_FEE_MODEL_PATH = "reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json"
DEFAULT_CANONICAL_METRICS_DEFINITION_PATH = "reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md"
DEFAULT_EXPECTED_FEE_MODEL_SHA256 = "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a"
DEFAULT_EXPECTED_METRICS_DEFINITION_SHA256 = "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99"


@dataclass
class SignalContext:
    symbol: str
    signal_id: str
    signal_time: pd.Timestamp
    signal_ts_ns: int
    tp_mult_sig: float
    sl_mult_sig: float
    quality: float
    baseline_entry_time: Optional[pd.Timestamp]
    baseline_exit_time: Optional[pd.Timestamp]
    baseline_exit_reason: str
    baseline_filled: int
    baseline_valid_for_metrics: int
    baseline_sl_hit: int
    baseline_tp_hit: int
    baseline_same_bar_hit: int
    baseline_invalid_stop_geometry: int
    baseline_invalid_tp_geometry: int
    baseline_entry_type: str
    baseline_entry_price: float
    baseline_exit_price: float
    baseline_fill_liq: str
    baseline_fill_delay_min: float
    baseline_mae_pct: float
    baseline_mfe_pct: float
    baseline_pnl_gross_pct: float
    baseline_pnl_net_pct: float
    ts_ns: np.ndarray
    open_np: np.ndarray
    high_np: np.ndarray
    low_np: np.ndarray
    close_np: np.ndarray
    atr_np: np.ndarray
    swing_high: np.ndarray


@dataclass
class SymbolBundle:
    symbol: str
    signals_csv: Path
    contexts: List[SignalContext]
    splits: List[Dict[str, int]]
    constraints: Dict[str, float]


_WORKER_STATE: Dict[str, Any] = {}


def _utc_now() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    return now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p.resolve()
    return (PROJECT_ROOT / p).resolve()


def _as_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def _json_dump(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _write_yaml_like(path: Path, obj: Any) -> None:
    # JSON is valid YAML 1.2 and keeps dependencies minimal.
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _copy_file_bytes(src: Path, dst: Path) -> None:
    dst.write_bytes(src.read_bytes())


def _validate_and_lock_frozen_artifacts(args: argparse.Namespace, run_dir: Path) -> Dict[str, Any]:
    fee_src = _resolve_path(args.canonical_fee_model_path)
    metrics_src = _resolve_path(args.canonical_metrics_definition_path)
    if not fee_src.exists():
        raise SystemExit(f"Canonical fee model file not found: {fee_src}")
    if not metrics_src.exists():
        raise SystemExit(f"Canonical metrics definition file not found: {metrics_src}")

    fee_sha = _sha256_file(fee_src)
    metrics_sha = _sha256_file(metrics_src)

    expected_fee_sha = str(args.expected_fee_model_sha256).strip().lower()
    expected_metrics_sha = str(args.expected_metrics_definition_sha256).strip().lower()
    allow_mismatch = int(args.allow_freeze_hash_mismatch) == 1

    fee_hash_match = int((not expected_fee_sha) or (fee_sha == expected_fee_sha))
    metrics_hash_match = int((not expected_metrics_sha) or (metrics_sha == expected_metrics_sha))

    if (fee_hash_match == 0 or metrics_hash_match == 0) and (not allow_mismatch):
        raise SystemExit(
            "Frozen artifact hash mismatch. "
            f"fee_sha={fee_sha}, expected_fee_sha={expected_fee_sha or 'unset'}, "
            f"metrics_sha={metrics_sha}, expected_metrics_sha={expected_metrics_sha or 'unset'}"
        )

    try:
        fee_obj = json.loads(fee_src.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"Failed to parse canonical fee model JSON: {fee_src} ({exc})")
    if not isinstance(fee_obj, dict):
        raise SystemExit(f"Canonical fee model must be a JSON object: {fee_src}")

    fee_blocks: List[Dict[str, Any]] = []
    for block_key in ("tight_pipeline_fee_model", "ga_pipeline_fee_model"):
        blk = fee_obj.get(block_key)
        if isinstance(blk, dict):
            fee_blocks.append(blk)
    fee_blocks.append(fee_obj)

    def _canonical_fee_value(field: str) -> float:
        for blk in fee_blocks:
            raw = blk.get(field, None)
            try:
                v = float(raw)
            except Exception:
                continue
            if np.isfinite(v):
                return v
        return float("nan")

    fee_fields = ("fee_bps_maker", "fee_bps_taker", "slippage_bps_limit", "slippage_bps_market")
    fee_value_match = 1
    fee_value_diffs: Dict[str, Dict[str, float]] = {}
    for k in fee_fields:
        canonical_v = _canonical_fee_value(k)
        arg_v = float(getattr(args, k))
        if (not np.isfinite(canonical_v)) or (not np.isfinite(arg_v)) or abs(canonical_v - arg_v) > 1e-12:
            fee_value_match = 0
            fee_value_diffs[k] = {"canonical": canonical_v, "args": arg_v}
    if fee_value_match == 0 and (not allow_mismatch):
        raise SystemExit(
            "Fee/slippage args mismatch canonical frozen fee model "
            f"{fee_src}. Diffs: {json.dumps(fee_value_diffs, sort_keys=True)}"
        )

    fee_dst = run_dir / "fee_model.json"
    metrics_dst = run_dir / "metrics_definition.md"
    _copy_file_bytes(fee_src, fee_dst)
    _copy_file_bytes(metrics_src, metrics_dst)

    fee_dst_sha = _sha256_file(fee_dst)
    metrics_dst_sha = _sha256_file(metrics_dst)

    out = {
        "generated_utc": _utc_now().isoformat(),
        "canonical_fee_model_path": str(fee_src),
        "canonical_metrics_definition_path": str(metrics_src),
        "expected_fee_model_sha256": expected_fee_sha,
        "expected_metrics_definition_sha256": expected_metrics_sha,
        "canonical_fee_model_sha256": fee_sha,
        "canonical_metrics_definition_sha256": metrics_sha,
        "run_fee_model_sha256": fee_dst_sha,
        "run_metrics_definition_sha256": metrics_dst_sha,
        "fee_hash_match_expected": int(fee_hash_match),
        "metrics_hash_match_expected": int(metrics_hash_match),
        "fee_value_match_args": int(fee_value_match),
        "fee_value_diffs": fee_value_diffs,
        "copied_verbatim_fee": int(fee_sha == fee_dst_sha),
        "copied_verbatim_metrics": int(metrics_sha == metrics_dst_sha),
        "allow_freeze_hash_mismatch": int(allow_mismatch),
        "freeze_lock_pass": int(fee_hash_match == 1 and metrics_hash_match == 1 and fee_value_match == 1),
    }
    _json_dump(run_dir / "freeze_lock_validation.json", out)
    return out


def _metrics_definition_text() -> str:
    lines: List[str] = []
    lines.append("# Metrics Definition")
    lines.append("")
    lines.append("- Baseline entry: market at next 3m open after 1h signal timestamp (UTC).")
    lines.append("- Candidate entry/exit decisions must use only information up to decision bar (no lookahead).")
    lines.append("- `mean_expectancy_net`: mean per-signal pnl vector where non-filled/invalid rows contribute 0.")
    lines.append("- `pnl_net_sum`: sum of per-signal net pnl values on TEST rows.")
    lines.append("- `cvar_5`: mean of worst 5% per-signal pnl outcomes.")
    lines.append("- `max_drawdown`: peak-to-trough drawdown on cumulative per-signal pnl curve.")
    lines.append("- Entry-conditioned metrics (`taker_share`, `SL_hit_rate_valid`, delays) use valid filled rows only.")
    return "\n".join(lines).strip() + "\n"


def _parse_scalar(raw: str) -> Any:
    s = str(raw).strip()
    if not s:
        return ""
    if s.lower() in {"true", "yes"}:
        return 1
    if s.lower() in {"false", "no"}:
        return 0
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def _load_execution_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    out: Dict[str, Any] = {}
    cur_symbol: Optional[str] = None
    cur_section: Optional[str] = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if indent == 0 and stripped.endswith(":"):
            cur_symbol = stripped[:-1].strip()
            if cur_symbol:
                out[cur_symbol] = {}
                cur_section = None
            continue
        if cur_symbol is None:
            continue
        if indent == 2:
            if stripped.endswith(":"):
                key = stripped[:-1].strip()
                out[cur_symbol][key] = {}
                cur_section = key
            elif ":" in stripped:
                key, val = stripped.split(":", 1)
                out[cur_symbol][key.strip()] = _parse_scalar(val)
                cur_section = None
            continue
        if indent == 4 and cur_section and isinstance(out[cur_symbol].get(cur_section), dict) and ":" in stripped:
            key, val = stripped.split(":", 1)
            out[cur_symbol][cur_section][key.strip()] = _parse_scalar(val)
    return out


def _symbol_exec_config(all_cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    if not isinstance(all_cfg, dict):
        return {}
    sym_u = str(symbol).upper()
    if sym_u in all_cfg and isinstance(all_cfg[sym_u], dict):
        return dict(all_cfg[sym_u])
    symbols_map = all_cfg.get("symbols")
    if isinstance(symbols_map, dict) and sym_u in symbols_map and isinstance(symbols_map[sym_u], dict):
        return dict(symbols_map[sym_u])
    return {}


def _find_latest_signal_csv(symbol: str, signals_dir: Path, reports_dir: Path) -> Optional[Path]:
    sym_u = str(symbol).upper()
    direct = signals_dir / f"{sym_u}_signals_1h.csv"
    if direct.exists():
        return direct.resolve()
    cands = sorted(reports_dir.glob(f"*/{sym_u}_signals_1h.csv"), key=lambda p: p.parent.name)
    if cands:
        return cands[-1].resolve()
    return None


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if str(args.symbol).strip():
        return [str(args.symbol).strip().upper()]
    if str(args.symbols).strip():
        return [x.strip().upper() for x in str(args.symbols).split(",") if x.strip()]

    scan_dir = _resolve_path(args.scan_dir) if str(args.scan_dir).strip() else exec3m._find_latest_scan_dir()
    best_csv = _resolve_path(args.best_csv) if str(args.best_csv).strip() else (scan_dir / "best_by_symbol.csv").resolve()
    df = pd.read_csv(best_csv)
    if df.empty:
        raise SystemExit(f"No rows in {best_csv}")
    rows = df.copy()
    if "side" in rows.columns:
        rows = rows[rows["side"].astype(str).str.lower() == "long"].copy()
    if "pass" in rows.columns:
        rows = rows[rows["pass"].map(exec3m._as_bool)].copy()
    if rows.empty:
        rows = df.copy()
    rows["score"] = pd.to_numeric(rows.get("score"), errors="coerce").fillna(-1e18)
    rows = rows.sort_values(["score", "symbol"], ascending=[False, True]).reset_index(drop=True)
    rk = max(1, int(args.rank))
    if rk > len(rows):
        raise SystemExit(f"Requested rank={rk}, available={len(rows)}")
    return [str(rows.iloc[rk - 1].get("symbol", "")).strip().upper()]


def _load_signals_for_symbol(symbol: str, args: argparse.Namespace) -> Tuple[pd.DataFrame, Path]:
    reports_dir = _resolve_path("reports/execution_layer")
    signals_dir = _resolve_path(args.signals_dir)
    signals_dir.mkdir(parents=True, exist_ok=True)

    signal_csv: Optional[Path] = None
    if str(args.signals_csv).strip():
        signal_csv = _resolve_path(args.signals_csv)
        if not signal_csv.exists():
            raise FileNotFoundError(f"signals csv not found: {signal_csv}")
    else:
        signal_csv = _find_latest_signal_csv(symbol=symbol, signals_dir=signals_dir, reports_dir=reports_dir)
    if signal_csv is None:
        raise FileNotFoundError(
            f"No 1h signal export found for {symbol}. Expected data/signals/{symbol}_signals_1h.csv or reports/execution_layer/*/{symbol}_signals_1h.csv"
        )

    df = pd.read_csv(signal_csv)
    if "signal_time" not in df.columns:
        raise SystemExit(f"Missing signal_time in {signal_csv}")
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True, errors="coerce")
    df = df[df["signal_time"].notna()].copy()
    if "direction" in df.columns:
        df = df[df["direction"].astype(str).str.lower() == "long"].copy()

    if "strategy_tp_mult" in df.columns:
        df["tp_mult"] = pd.to_numeric(df["strategy_tp_mult"], errors="coerce")
    elif "tp_mult" in df.columns:
        df["tp_mult"] = pd.to_numeric(df["tp_mult"], errors="coerce")
    else:
        raise SystemExit(f"Missing strategy_tp_mult/tp_mult in {signal_csv}")

    if "strategy_sl_mult" in df.columns:
        df["sl_mult"] = pd.to_numeric(df["strategy_sl_mult"], errors="coerce")
    elif "sl_mult" in df.columns:
        df["sl_mult"] = pd.to_numeric(df["sl_mult"], errors="coerce")
    else:
        raise SystemExit(f"Missing strategy_sl_mult/sl_mult in {signal_csv}")

    if "signal_id" not in df.columns:
        df["signal_id"] = [f"sig_{i:05d}" for i in range(1, len(df) + 1)]

    if "signal_quality" in df.columns:
        df["signal_quality"] = pd.to_numeric(df["signal_quality"], errors="coerce")
    else:
        df["signal_quality"] = np.nan

    df = df.dropna(subset=["tp_mult", "sl_mult"]).sort_values("signal_time").reset_index(drop=True)

    max_signals = int(args.max_signals)
    if max_signals > 0 and len(df) > max_signals:
        if str(args.signal_order).lower() == "latest":
            df = df.iloc[-max_signals:].copy()
        else:
            df = df.iloc[:max_signals].copy()
        df = df.sort_values("signal_time").reset_index(drop=True)

    return df, signal_csv


def _build_walkforward_splits(n: int, train_ratio: float, n_splits: int) -> List[Dict[str, int]]:
    if n <= 1:
        return [{"split_id": 0, "train_start": 0, "train_end": 0, "test_start": 0, "test_end": n}]

    tr = _clamp(train_ratio, 0.1, 0.95)
    initial_train = max(1, int(round(n * tr)))
    if initial_train >= n:
        initial_train = n - 1

    rem = n - initial_train
    if rem <= 0:
        return [{"split_id": 0, "train_start": 0, "train_end": n - 1, "test_start": n - 1, "test_end": n}]

    n_splits = max(1, int(n_splits))
    step = max(1, int(math.floor(rem / n_splits)))
    splits: List[Dict[str, int]] = []
    start = initial_train
    for i in range(n_splits):
        test_start = start + i * step
        if test_start >= n:
            break
        if i == n_splits - 1:
            test_end = n
        else:
            test_end = min(n, test_start + step)
        if test_end <= test_start:
            continue
        splits.append(
            {
                "split_id": int(i),
                "train_start": 0,
                "train_end": int(test_start),
                "test_start": int(test_start),
                "test_end": int(test_end),
            }
        )
    if not splits:
        splits = [{"split_id": 0, "train_start": 0, "train_end": n - 1, "test_start": n - 1, "test_end": n}]
    return splits


def _max_consecutive_losses(pnl: np.ndarray) -> int:
    cur = 0
    best = 0
    for x in pnl:
        if np.isfinite(x) and x < 0.0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _max_drawdown(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return float("nan")
    cum = np.cumsum(np.nan_to_num(pnl, nan=0.0))
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.nanmin(dd)) if dd.size else float("nan")


def _tail_mean(arr: np.ndarray, q: float) -> float:
    x = arr[np.isfinite(arr)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(float(q) * x.size)))
    xs = np.sort(x)
    return float(np.mean(xs[:k]))


def _improvement_ratio_abs(exec_val: float, base_val: float) -> float:
    if not np.isfinite(exec_val) or not np.isfinite(base_val):
        return float("nan")
    b = abs(float(base_val))
    if b <= 1e-12:
        return float("nan")
    e = abs(float(exec_val))
    return float((b - e) / b)


def _is_in_killzone(ts: pd.Timestamp) -> bool:
    m = int(ts.hour) * 60 + int(ts.minute)
    for s, e in KILLZONES_DEFAULT:
        if int(s) <= m < int(e):
            return True
    return False


def _costed_pnl_from_legs(
    *,
    entry_price: float,
    legs: List[Tuple[float, float]],
    entry_liquidity_type: str,
    fee_bps_maker: float,
    fee_bps_taker: float,
    slippage_bps_limit: float,
    slippage_bps_market: float,
) -> Dict[str, float]:
    e = float(entry_price)
    if (not np.isfinite(e)) or e <= 0.0 or not legs:
        return {
            "pnl_gross_pct": float("nan"),
            "pnl_net_pct": float("nan"),
            "entry_fee_bps": float("nan"),
            "exit_fee_bps": float("nan"),
            "entry_slippage_bps": float("nan"),
            "exit_slippage_bps": float("nan"),
            "total_cost_bps": float("nan"),
        }

    liq = str(entry_liquidity_type).strip().lower()
    is_maker = liq == "maker"
    entry_fee_bps = float(fee_bps_maker if is_maker else fee_bps_taker)
    exit_fee_bps = float(fee_bps_taker)
    entry_slip_bps = float(slippage_bps_limit if is_maker else slippage_bps_market)
    exit_slip_bps = float(slippage_bps_market)

    total_frac = sum(max(0.0, float(f)) for f, _ in legs)
    if total_frac <= 1e-12:
        return {
            "pnl_gross_pct": float("nan"),
            "pnl_net_pct": float("nan"),
            "entry_fee_bps": float("nan"),
            "exit_fee_bps": float("nan"),
            "entry_slippage_bps": float("nan"),
            "exit_slippage_bps": float("nan"),
            "total_cost_bps": float("nan"),
        }

    norm_legs = [(float(f) / total_frac, float(px)) for f, px in legs]
    gross = float(sum(f * ((px / e) - 1.0) for f, px in norm_legs if np.isfinite(px)))

    entry_eff = float(e * (1.0 + entry_slip_bps / 1e4))
    exit_eff_w = 0.0
    for f, px in norm_legs:
        if np.isfinite(px):
            exit_eff_w += float(f) * float(px * (1.0 - exit_slip_bps / 1e4))
    net = float((exit_eff_w / entry_eff) - 1.0 - (entry_fee_bps + exit_fee_bps) / 1e4)

    return {
        "pnl_gross_pct": float(gross),
        "pnl_net_pct": float(net),
        "entry_fee_bps": float(entry_fee_bps),
        "exit_fee_bps": float(exit_fee_bps),
        "entry_slippage_bps": float(entry_slip_bps),
        "exit_slippage_bps": float(exit_slip_bps),
        "total_cost_bps": float((gross - net) * 1e4),
    }


def _simulate_dynamic_exit_long(
    *,
    ts_ns: np.ndarray,
    close_np: np.ndarray,
    high_np: np.ndarray,
    low_np: np.ndarray,
    entry_idx: int,
    entry_price: float,
    tp_mult_sig: float,
    sl_mult_sig: float,
    genome: Dict[str, Any],
    max_exit_ts_ns: Optional[int],
) -> Dict[str, Any]:
    n = len(ts_ns)
    if n == 0 or entry_idx < 0 or entry_idx >= n or (not np.isfinite(entry_price)) or entry_price <= 0:
        return {
            "filled": False,
            "exit_reason": "invalid_entry",
            "invalid_stop_geometry": 1,
            "invalid_tp_geometry": 1,
            "valid_for_metrics": 0,
            "same_bar_hit": 0,
            "sl_hit": False,
            "tp_hit": False,
            "legs": [],
            "exit_idx": int(entry_idx),
            "exit_price": float("nan"),
            "exit_time": None,
            "mae_pct": float("nan"),
            "mfe_pct": float("nan"),
            "time_to_mae_min": float("nan"),
            "time_to_mfe_min": float("nan"),
        }

    if max_exit_ts_ns is None:
        end_idx = n - 1
    else:
        end_idx = exec3m._idx_at_or_before_ts(ts_ns=ts_ns, target_ns=int(max_exit_ts_ns), min_idx=int(entry_idx), max_idx=n - 1)

    base_risk_pct = max(1e-6, 1.0 - float(sl_mult_sig))
    base_reward_pct = max(1e-6, float(tp_mult_sig) - 1.0)
    risk_pct = max(1e-8, base_risk_pct * float(genome["sl_mult"]))
    reward_pct = max(1e-8, base_reward_pct * float(genome["tp_mult"]))

    initial_sl = float(entry_price * (1.0 - risk_pct))
    tp_px = float(entry_price * (1.0 + reward_pct))

    inv_stop = int((not np.isfinite(initial_sl)) or initial_sl >= entry_price)
    inv_tp = int((not np.isfinite(tp_px)) or tp_px <= entry_price)
    valid = int((inv_stop == 0) and (inv_tp == 0))
    if valid == 0:
        et = pd.to_datetime(int(ts_ns[entry_idx]), utc=True)
        return {
            "filled": True,
            "exit_reason": "invalid_geometry",
            "invalid_stop_geometry": int(inv_stop),
            "invalid_tp_geometry": int(inv_tp),
            "valid_for_metrics": 0,
            "same_bar_hit": 0,
            "sl_hit": False,
            "tp_hit": False,
            "legs": [(1.0, float(entry_price))],
            "exit_idx": int(entry_idx),
            "exit_price": float(entry_price),
            "exit_time": et,
            "mae_pct": float("nan"),
            "mfe_pct": float("nan"),
            "time_to_mae_min": float("nan"),
            "time_to_mfe_min": float("nan"),
            "sl_price": float(initial_sl),
            "tp_price": float(tp_px),
        }

    R = max(1e-8, float(entry_price - initial_sl))
    stop_px = float(initial_sl)
    highest = float(entry_price)
    same_bar = 0

    break_even_enabled = _as_bool(genome["break_even_enabled"])
    break_even_trigger_r = float(genome["break_even_trigger_r"])
    break_even_offset_bps = float(genome["break_even_offset_bps"])
    trailing_enabled = _as_bool(genome["trailing_enabled"])
    trail_start_r = float(genome["trail_start_r"])
    trail_step_bps = float(genome["trail_step_bps"])
    partial_enabled = _as_bool(genome["partial_take_enabled"])
    partial_r = float(genome["partial_take_r"])
    partial_pct = float(genome["partial_take_pct"])
    time_stop_min = int(round(float(genome["time_stop_min"])))

    partial_px = float(entry_price + partial_r * R)
    be_armed = False
    trail_armed = False
    partial_done = False

    remain = 1.0
    legs: List[Tuple[float, float]] = []
    sl_hit = False
    tp_hit = False
    exit_reason = "window_end"
    exit_idx = int(end_idx)
    exit_px = float(close_np[exit_idx]) if np.isfinite(close_np[exit_idx]) else float(entry_price)

    for i in range(int(entry_idx), int(end_idx) + 1):
        hi = float(high_np[i])
        lo = float(low_np[i])
        cl = float(close_np[i]) if np.isfinite(close_np[i]) else float(entry_price)

        if np.isfinite(hi):
            highest = max(highest, hi)

        hit_sl = np.isfinite(lo) and lo <= float(stop_px)
        hit_tp = np.isfinite(hi) and hi >= float(tp_px)

        if hit_sl and hit_tp:
            same_bar = 1
            hit_tp = False  # conservative ordering: SL first

        if hit_sl:
            if remain > 1e-12:
                legs.append((float(remain), float(stop_px)))
                remain = 0.0
            sl_hit = True
            exit_reason = "sl"
            exit_idx = int(i)
            exit_px = float(stop_px)
            break

        if partial_enabled and (not partial_done) and remain > 1e-12 and np.isfinite(hi) and hi >= partial_px:
            take = min(remain, float(partial_pct))
            if take > 1e-12:
                legs.append((float(take), float(partial_px)))
                remain = float(remain - take)
                partial_done = True

        if hit_tp:
            if remain > 1e-12:
                legs.append((float(remain), float(tp_px)))
                remain = 0.0
            tp_hit = True
            exit_reason = "tp"
            exit_idx = int(i)
            exit_px = float(tp_px)
            break

        if time_stop_min > 0:
            elapsed_min = float((pd.to_datetime(int(ts_ns[i]), utc=True) - pd.to_datetime(int(ts_ns[entry_idx]), utc=True)).total_seconds() / 60.0)
            if elapsed_min >= float(time_stop_min) and remain > 1e-12:
                legs.append((float(remain), float(cl)))
                remain = 0.0
                exit_reason = "time_stop"
                exit_idx = int(i)
                exit_px = float(cl)
                break

        if break_even_enabled and (not be_armed) and np.isfinite(hi) and hi >= float(entry_price + break_even_trigger_r * R):
            be_px = float(entry_price * (1.0 + break_even_offset_bps / 1e4))
            stop_px = max(stop_px, be_px)
            be_armed = True

        if trailing_enabled and np.isfinite(hi) and hi >= float(entry_price + trail_start_r * R):
            trail_armed = True
        if trail_armed:
            tr_px = float(highest * (1.0 - trail_step_bps / 1e4))
            stop_px = max(stop_px, tr_px)

    if remain > 1e-12:
        legs.append((float(remain), float(exit_px)))

    lows = np.asarray(low_np[entry_idx : exit_idx + 1], dtype=float)
    highs = np.asarray(high_np[entry_idx : exit_idx + 1], dtype=float)
    if lows.size:
        mae = float(np.nanmin(lows) / entry_price - 1.0)
        mae_loc = int(np.nanargmin(lows))
    else:
        mae = float("nan")
        mae_loc = 0
    if highs.size:
        mfe = float(np.nanmax(highs) / entry_price - 1.0)
        mfe_loc = int(np.nanargmax(highs))
    else:
        mfe = float("nan")
        mfe_loc = 0

    mae_idx = int(entry_idx + mae_loc)
    mfe_idx = int(entry_idx + mfe_loc)
    et0 = pd.to_datetime(int(ts_ns[entry_idx]), utc=True)

    return {
        "filled": True,
        "exit_reason": str(exit_reason),
        "invalid_stop_geometry": int(inv_stop),
        "invalid_tp_geometry": int(inv_tp),
        "valid_for_metrics": int(valid),
        "same_bar_hit": int(same_bar),
        "sl_hit": bool(sl_hit),
        "tp_hit": bool(tp_hit),
        "legs": [(float(f), float(px)) for f, px in legs],
        "exit_idx": int(exit_idx),
        "exit_price": float(exit_px),
        "exit_time": pd.to_datetime(int(ts_ns[exit_idx]), utc=True),
        "mae_pct": float(mae),
        "mfe_pct": float(mfe),
        "time_to_mae_min": float((pd.to_datetime(int(ts_ns[mae_idx]), utc=True) - et0).total_seconds() / 60.0),
        "time_to_mfe_min": float((pd.to_datetime(int(ts_ns[mfe_idx]), utc=True) - et0).total_seconds() / 60.0),
        "sl_price": float(initial_sl),
        "tp_price": float(tp_px),
    }


def _simulate_candidate_signal(
    *,
    ctx: SignalContext,
    genome: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    last_entry_time: Optional[pd.Timestamp],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "symbol": ctx.symbol,
        "signal_id": ctx.signal_id,
        "signal_time": str(ctx.signal_time),
        "signal_tp_mult": float(ctx.tp_mult_sig),
        "signal_sl_mult": float(ctx.sl_mult_sig),
        "baseline_filled": int(ctx.baseline_filled),
        "baseline_valid_for_metrics": int(ctx.baseline_valid_for_metrics),
        "baseline_sl_hit": int(ctx.baseline_sl_hit),
        "baseline_tp_hit": int(ctx.baseline_tp_hit),
        "baseline_pnl_net_pct": float(ctx.baseline_pnl_net_pct),
        "baseline_pnl_gross_pct": float(ctx.baseline_pnl_gross_pct),
        "baseline_fill_liquidity_type": str(ctx.baseline_fill_liq),
        "baseline_fill_delay_min": float(ctx.baseline_fill_delay_min),
        "baseline_mae_pct": float(ctx.baseline_mae_pct),
        "baseline_mfe_pct": float(ctx.baseline_mfe_pct),
        "baseline_entry_time": str(ctx.baseline_entry_time) if ctx.baseline_entry_time is not None else "",
        "baseline_exit_time": str(ctx.baseline_exit_time) if ctx.baseline_exit_time is not None else "",
        "baseline_exit_reason": str(ctx.baseline_exit_reason),
        "baseline_same_bar_hit": int(ctx.baseline_same_bar_hit),
        "baseline_invalid_stop_geometry": int(ctx.baseline_invalid_stop_geometry),
        "baseline_invalid_tp_geometry": int(ctx.baseline_invalid_tp_geometry),
        "baseline_entry_type": str(ctx.baseline_entry_type),
        "baseline_entry_price": float(ctx.baseline_entry_price),
        "baseline_exit_price": float(ctx.baseline_exit_price),
        "exec_filled": 0,
        "exec_valid_for_metrics": 0,
        "exec_sl_hit": 0,
        "exec_tp_hit": 0,
        "exec_pnl_net_pct": float("nan"),
        "exec_pnl_gross_pct": float("nan"),
        "exec_fill_liquidity_type": "",
        "exec_fill_delay_min": float("nan"),
        "exec_mae_pct": float("nan"),
        "exec_mfe_pct": float("nan"),
        "entry_improvement_bps": float("nan"),
        "exec_skip_reason": "",
        "lookahead_violation": 0,
        "constraint_fail_reason": "",
        "missing_slice_flag": 0,
    }

    if last_entry_time is not None and int(genome.get("cooldown_min", 0)) > 0:
        delta_min = float((ctx.signal_time - last_entry_time).total_seconds() / 60.0)
        if delta_min < float(genome["cooldown_min"]):
            out["exec_skip_reason"] = "cooldown"
            return out

    ts_ns = ctx.ts_ns
    n = len(ts_ns)
    if n == 0:
        out["exec_skip_reason"] = "no_3m_data"
        out["missing_slice_flag"] = 1
        return out

    sig_idx = int(np.searchsorted(ts_ns, int(ctx.signal_ts_ns), side="left"))
    if sig_idx >= n:
        out["exec_skip_reason"] = "no_bar_after_signal"
        out["missing_slice_flag"] = 1
        return out

    open_np = ctx.open_np
    high_np = ctx.high_np
    low_np = ctx.low_np
    close_np = ctx.close_np
    atr_np = ctx.atr_np

    entry_ref = float(open_np[sig_idx])
    if (not np.isfinite(entry_ref)) or entry_ref <= 0.0:
        out["exec_skip_reason"] = "bad_entry_ref"
        out["missing_slice_flag"] = 1
        return out

    # Optional quality gate (only if data exists)
    if _as_bool(genome.get("use_signal_quality_gate", 0)) and np.isfinite(ctx.quality):
        if float(ctx.quality) < float(genome.get("min_signal_quality_gate", 0.0)):
            out["exec_skip_reason"] = "signal_quality_gate"
            return out

    # Vol gate at signal anchor uses only history up to signal index.
    atr_ref_idx = max(0, sig_idx - 1)
    atr_ref = float(atr_np[atr_ref_idx]) if atr_np.size else float("nan")
    hist_start = max(0, atr_ref_idx - 7 * 24 * 20)
    hist = np.asarray(atr_np[hist_start:atr_ref_idx], dtype=float)
    hist = hist[np.isfinite(hist)]
    atr_z = 0.0
    if hist.size >= 30 and np.isfinite(atr_ref):
        mean = float(np.nanmean(hist))
        std = float(np.nanstd(hist))
        atr_z = float((atr_ref - mean) / std) if std > 1e-12 else 0.0

    if _as_bool(genome["micro_vol_filter"]) and np.isfinite(atr_z) and atr_z > float(genome["vol_threshold"]):
        if _as_bool(genome.get("skip_if_vol_gate", 0)) and (not _as_bool(eval_cfg.get("force_no_skip", 0))):
            out["exec_skip_reason"] = "volatility_gate"
            return out

    spread_proxy_bps = float((high_np[sig_idx] - low_np[sig_idx]) / max(1e-12, close_np[sig_idx]) * 1e4) if np.isfinite(close_np[sig_idx]) else float("nan")
    if np.isfinite(spread_proxy_bps) and spread_proxy_bps > float(genome["spread_guard_bps"]):
        if not _as_bool(eval_cfg.get("force_no_skip", 0)):
            out["exec_skip_reason"] = "spread_guard"
            return out

    anchor_idx = int(sig_idx)
    if _as_bool(genome["mss_displacement_gate"]):
        disp = exec3m._find_displacement_long(
            open_=open_np,
            high=high_np,
            close=close_np,
            atr=atr_np,
            swing_high=ctx.swing_high,
            start_idx=int(sig_idx),
            max_confirm_bars=max(1, int(math.ceil(float(genome["max_fill_delay_min"]) / 3.0))),
            body_atr_mult=1.0,
            body_pct_thr=0.002,
            last_idx=n - 1,
        )
        if disp is None:
            if not _as_bool(eval_cfg.get("force_no_skip", 0)):
                out["exec_skip_reason"] = "no_displacement"
                return out
        else:
            anchor_idx = min(n - 1, int(disp["disp_idx"]) + 1)

    max_fill_bars = max(0, int(math.ceil(float(genome["max_fill_delay_min"]) / 3.0)))
    fallback_bars = max(0, int(math.ceil(float(genome["fallback_delay_min"]) / 3.0)))
    fill_end_idx = min(n - 1, anchor_idx + max_fill_bars)

    fill_idx: Optional[int] = None
    fill_px = float("nan")
    fill_type = ""
    mode = str(genome["entry_mode"])

    if mode == "market":
        fill_idx = int(anchor_idx)
        fill_px = float(open_np[fill_idx])
        fill_type = "market"
    else:
        limit_off = max(0.0, float(genome["limit_offset_bps"]))
        limit_px = float(entry_ref * (1.0 - limit_off / 1e4))
        for i in range(int(anchor_idx), int(fill_end_idx) + 1):
            if np.isfinite(low_np[i]) and float(low_np[i]) <= limit_px:
                fill_idx = int(i)
                fill_px = float(limit_px)
                fill_type = "limit"
                break

        want_fallback = _as_bool(genome["fallback_to_market"]) or _as_bool(eval_cfg.get("force_no_skip", 0))
        if fill_idx is None and mode in {"limit", "hybrid"} and want_fallback:
            m_idx = min(n - 1, anchor_idx + fallback_bars)
            if m_idx <= n - 1:
                fill_idx = int(m_idx)
                fill_px = float(open_np[m_idx])
                fill_type = "market_fallback"

    if fill_idx is None:
        out["exec_skip_reason"] = "timeout_no_fill"
        return out

    fill_time = pd.to_datetime(int(ts_ns[int(fill_idx)]), utc=True)
    if _as_bool(genome["killzone_filter"]) and (not _is_in_killzone(fill_time)):
        if not _as_bool(eval_cfg.get("force_no_skip", 0)):
            out["exec_skip_reason"] = "outside_killzone"
            return out

    if ctx.baseline_exit_time is not None and fill_time > exec3m._to_utc_ts(ctx.baseline_exit_time):
        out["exec_skip_reason"] = "after_baseline_exit"
        return out

    improve_bps = float((entry_ref - fill_px) / entry_ref * 1e4) if np.isfinite(entry_ref) and entry_ref > 0 else float("nan")
    out["entry_improvement_bps"] = float(improve_bps)
    if np.isfinite(improve_bps) and improve_bps < float(genome["min_entry_improvement_bps_gate"]):
        if not _as_bool(eval_cfg.get("force_no_skip", 0)):
            out["exec_skip_reason"] = "entry_improvement_gate"
            return out

    max_exit_ts_ns = exec3m._compute_eval_end_ns(
        entry_ts_ns=int(ts_ns[int(fill_idx)]),
        eval_horizon_hours=float(eval_cfg["exec_horizon_hours"]),
        baseline_exit_time=ctx.baseline_exit_time,
    )

    exit_res = _simulate_dynamic_exit_long(
        ts_ns=ts_ns,
        close_np=close_np,
        high_np=high_np,
        low_np=low_np,
        entry_idx=int(fill_idx),
        entry_price=float(fill_px),
        tp_mult_sig=float(ctx.tp_mult_sig),
        sl_mult_sig=float(ctx.sl_mult_sig),
        genome=genome,
        max_exit_ts_ns=int(max_exit_ts_ns),
    )

    liq = "maker" if str(fill_type).startswith("limit") else "taker"
    if _as_bool(genome["partial_take_enabled"]):
        cost = _costed_pnl_from_legs(
            entry_price=float(fill_px),
            legs=list(exit_res.get("legs", [])),
            entry_liquidity_type=liq,
            fee_bps_maker=float(eval_cfg["fee_bps_maker"]),
            fee_bps_taker=float(eval_cfg["fee_bps_taker"]),
            slippage_bps_limit=float(eval_cfg["slippage_bps_limit"]),
            slippage_bps_market=float(eval_cfg["slippage_bps_market"]),
        )
    else:
        cost = exec3m._costed_pnl_long(
            entry_price=float(fill_px),
            exit_price=float(exit_res.get("exit_price", np.nan)),
            entry_liquidity_type=liq,
            fee_bps_maker=float(eval_cfg["fee_bps_maker"]),
            fee_bps_taker=float(eval_cfg["fee_bps_taker"]),
            slippage_bps_limit=float(eval_cfg["slippage_bps_limit"]),
            slippage_bps_market=float(eval_cfg["slippage_bps_market"]),
        )

    out.update(
        {
            "exec_filled": 1,
            "exec_valid_for_metrics": int(exit_res.get("valid_for_metrics", 0)),
            "exec_sl_hit": int(bool(exit_res.get("sl_hit", False))),
            "exec_tp_hit": int(bool(exit_res.get("tp_hit", False))),
            "exec_pnl_net_pct": float(cost["pnl_net_pct"]),
            "exec_pnl_gross_pct": float(cost["pnl_gross_pct"]),
            "exec_fill_liquidity_type": str(liq),
            "exec_fill_delay_min": float((fill_time - ctx.signal_time).total_seconds() / 60.0),
            "exec_mae_pct": float(exit_res.get("mae_pct", np.nan)),
            "exec_mfe_pct": float(exit_res.get("mfe_pct", np.nan)),
            "exec_skip_reason": "",
            "exec_exit_reason": str(exit_res.get("exit_reason", "")),
            "exec_entry_time": str(fill_time),
            "exec_exit_time": str(exit_res.get("exit_time", "")),
            "exec_entry_price": float(fill_px),
            "exec_exit_price": float(exit_res.get("exit_price", np.nan)),
            "exec_same_bar_hit": int(exit_res.get("same_bar_hit", 0)),
            "exec_invalid_stop_geometry": int(exit_res.get("invalid_stop_geometry", 0)),
            "exec_invalid_tp_geometry": int(exit_res.get("invalid_tp_geometry", 0)),
            "exec_entry_type": str(fill_type),
        }
    )
    return out


def _rollup_mode(rows_df: pd.DataFrame, mode: str) -> Dict[str, float]:
    mode = str(mode).strip().lower()
    if mode == "baseline":
        fill_col = "baseline_filled"
        valid_col = "baseline_valid_for_metrics"
        sl_col = "baseline_sl_hit"
        pnl_col = "baseline_pnl_net_pct"
        liq_col = "baseline_fill_liquidity_type"
        delay_col = "baseline_fill_delay_min"
        improve_col = None
    else:
        fill_col = "exec_filled"
        valid_col = "exec_valid_for_metrics"
        sl_col = "exec_sl_hit"
        pnl_col = "exec_pnl_net_pct"
        liq_col = "exec_fill_liquidity_type"
        delay_col = "exec_fill_delay_min"
        improve_col = "entry_improvement_bps"

    n = int(len(rows_df))
    if n == 0:
        return {
            "signals_total": 0,
            "entries_valid": 0,
            "entry_rate": float("nan"),
            "pnl_net_sum": float("nan"),
            "mean_expectancy_net": float("nan"),
            "pnl_std": float("nan"),
            "worst_decile_mean": float("nan"),
            "cvar_5": float("nan"),
            "max_consecutive_losses": 0,
            "SL_hit_rate_valid": float("nan"),
            "taker_share": float("nan"),
            "median_fill_delay_min": float("nan"),
            "p95_fill_delay_min": float("nan"),
            "median_entry_improvement_bps": 0.0,
            "max_drawdown": float("nan"),
        }

    filled = pd.to_numeric(rows_df.get(fill_col, 0), errors="coerce").fillna(0).astype(int)
    valid = pd.to_numeric(rows_df.get(valid_col, 0), errors="coerce").fillna(0).astype(int)
    sl_hit = pd.to_numeric(rows_df.get(sl_col, 0), errors="coerce").fillna(0).astype(int)
    pnl_raw = pd.to_numeric(rows_df.get(pnl_col, np.nan), errors="coerce")

    mask = (filled == 1) & (valid == 1)
    entries = int(mask.sum())
    pnl_sig = np.zeros(n, dtype=float)
    good = mask & pnl_raw.notna()
    pnl_sig[good.to_numpy(dtype=bool)] = pnl_raw[good].to_numpy(dtype=float)

    liq = rows_df.get(liq_col, pd.Series(["" for _ in range(n)])).fillna("").astype(str).str.lower()
    taker_share = float(((liq == "taker") & mask).sum() / entries) if entries > 0 else float("nan")
    delay = pd.to_numeric(rows_df.get(delay_col, np.nan), errors="coerce")
    med_delay = float(delay[mask].median()) if entries > 0 and delay[mask].notna().any() else float("nan")
    p95_delay = float(delay[mask].quantile(0.95)) if entries > 0 and delay[mask].notna().any() else float("nan")

    if improve_col is not None:
        improve = pd.to_numeric(rows_df.get(improve_col, np.nan), errors="coerce")
        med_improve = float(improve[mask].median()) if entries > 0 and improve[mask].notna().any() else float("nan")
    else:
        med_improve = 0.0

    pnl_sum = float(np.sum(pnl_sig))
    exp = float(np.mean(pnl_sig))
    std = float(np.std(pnl_sig, ddof=0))

    return {
        "signals_total": int(n),
        "entries_valid": int(entries),
        "entry_rate": float(entries / max(1, n)),
        "pnl_net_sum": float(pnl_sum),
        "mean_expectancy_net": float(exp),
        "pnl_std": float(std),
        "worst_decile_mean": float(_tail_mean(pnl_sig, 0.10)),
        "cvar_5": float(_tail_mean(pnl_sig, 0.05)),
        "max_consecutive_losses": int(_max_consecutive_losses(pnl_sig)),
        "SL_hit_rate_valid": float(((sl_hit == 1) & mask).sum() / entries) if entries > 0 else float("nan"),
        "taker_share": float(taker_share),
        "median_fill_delay_min": float(med_delay),
        "p95_fill_delay_min": float(p95_delay),
        "median_entry_improvement_bps": float(med_improve),
        "max_drawdown": float(_max_drawdown(pnl_sig)),
    }


def _weighted_avg(vals: Iterable[float], w: Iterable[float]) -> float:
    x = np.asarray(list(vals), dtype=float)
    wt = np.asarray(list(w), dtype=float)
    m = np.isfinite(x) & np.isfinite(wt) & (wt > 0)
    if not np.any(m):
        return float("nan")
    return float(np.sum(x[m] * wt[m]) / np.sum(wt[m]))


def _aggregate_rows(rows_df: pd.DataFrame) -> Dict[str, Any]:
    b = _rollup_mode(rows_df, "baseline")
    e = _rollup_mode(rows_df, "exec")
    return {
        "baseline": b,
        "exec": e,
        "delta_expectancy_exec_minus_baseline": float(e["mean_expectancy_net"] - b["mean_expectancy_net"]),
        "delta_cvar5_exec_minus_baseline": float(e["cvar_5"] - b["cvar_5"]),
        "delta_max_drawdown_exec_minus_baseline": float(e["max_drawdown"] - b["max_drawdown"]),
        "cvar_improve_ratio": float(_improvement_ratio_abs(e["cvar_5"], b["cvar_5"])),
        "maxdd_improve_ratio": float(_improvement_ratio_abs(e["max_drawdown"], b["max_drawdown"])),
    }


def _symbol_thresholds(bundle: SymbolBundle, genome: Dict[str, Any], mode: str, args: argparse.Namespace) -> Dict[str, float]:
    cons = dict(bundle.constraints)
    if str(mode).lower() == "tight":
        min_entry = float(cons.get("min_entry_rate", args.tight_min_entry_rate_default))
        max_delay = float(cons.get("max_fill_delay_min", args.tight_max_fill_delay_default))
        max_taker = float(cons.get("max_taker_share", args.tight_max_taker_share_default))
    else:
        min_entry = float(cons.get("min_entry_rate", args.min_entry_rate_default))
        max_delay = float(cons.get("max_fill_delay_min", args.max_fill_delay_default))
        max_taker = float(cons.get("max_taker_share", args.max_taker_share_default))

    gene_taker_cap = float(genome.get("max_taker_share", 1.0))
    max_taker = min(max_taker, gene_taker_cap)

    min_entry_improve = max(float(cons.get("min_median_entry_improvement_bps", -9999.0)), float(genome.get("min_entry_improvement_bps_gate", 0.0)))

    return {
        "min_entry_rate": float(min_entry),
        "max_fill_delay_min": float(max_delay),
        "max_taker_share": float(max_taker),
        "min_median_entry_improvement_bps": float(min_entry_improve),
    }


def _evaluate_genome(
    genome: Dict[str, Any],
    bundles: List[SymbolBundle],
    args: argparse.Namespace,
    detailed: bool,
) -> Dict[str, Any]:
    t0 = time.time()
    mode = str(args.mode).lower()
    force_no_skip = int(args.force_no_skip) == 1

    struct_fail_reasons: List[str] = []
    if float(genome["fallback_delay_min"]) > float(genome["max_fill_delay_min"]):
        struct_fail_reasons.append("fallback_delay_gt_max_fill")
    if _as_bool(genome["trailing_enabled"]) and _as_bool(genome["break_even_enabled"]):
        if float(genome["trail_start_r"]) < float(genome["break_even_trigger_r"]):
            struct_fail_reasons.append("trail_start_lt_be_trigger")
    if _as_bool(genome["partial_take_enabled"]) and float(genome["partial_take_r"]) >= float(genome["tp_mult"]):
        struct_fail_reasons.append("partial_r_gte_tp_mult")
    if float(genome["tp_mult"]) <= 0.0:
        struct_fail_reasons.append("tp_mult_nonpositive")
    if float(genome["sl_mult"]) <= 0.0:
        struct_fail_reasons.append("sl_mult_nonpositive")

    split_rows: List[Dict[str, Any]] = []
    symbol_rows: List[Dict[str, Any]] = []
    all_signal_rows: List[Dict[str, Any]] = []

    participation_fail: List[str] = []
    realism_fail: List[str] = []
    nan_fail: List[str] = []
    data_quality_fail: List[str] = []
    split_fail: List[str] = []

    lookahead_violations = 0
    expected_split_count = int(sum(len(b.splits) for b in bundles))

    eval_cfg = {
        "exec_horizon_hours": float(args.exec_horizon_hours),
        "fee_bps_maker": float(args.fee_bps_maker),
        "fee_bps_taker": float(args.fee_bps_taker),
        "slippage_bps_limit": float(args.slippage_bps_limit),
        "slippage_bps_market": float(args.slippage_bps_market),
        "force_no_skip": int(force_no_skip),
    }

    fee_model_identical = int(
        np.isfinite(eval_cfg["fee_bps_maker"])
        and np.isfinite(eval_cfg["fee_bps_taker"])
        and np.isfinite(eval_cfg["slippage_bps_limit"])
        and np.isfinite(eval_cfg["slippage_bps_market"])
    )
    if fee_model_identical == 0:
        struct_fail_reasons.append("fee_model_invalid")

    for bundle in bundles:
        symbol_all_rows: List[Dict[str, Any]] = []
        thresholds = _symbol_thresholds(bundle=bundle, genome=genome, mode=mode, args=args)

        for sp in bundle.splits:
            idx0 = int(sp["test_start"])
            idx1 = int(sp["test_end"])
            split_signal_rows: List[Dict[str, Any]] = []
            last_entry_time: Optional[pd.Timestamp] = None

            for ctx in bundle.contexts[idx0:idx1]:
                row = _simulate_candidate_signal(
                    ctx=ctx,
                    genome=genome,
                    eval_cfg=eval_cfg,
                    last_entry_time=last_entry_time,
                )
                row["split_id"] = int(sp["split_id"])
                row["split_test_start"] = int(idx0)
                row["split_test_end"] = int(idx1)
                if int(row.get("exec_filled", 0)) == 1:
                    et = pd.to_datetime(row.get("exec_entry_time"), utc=True, errors="coerce")
                    if pd.notna(et):
                        last_entry_time = et
                lookahead_violations += int(row.get("lookahead_violation", 0))
                split_signal_rows.append(row)

            df_split = pd.DataFrame(split_signal_rows)
            split_roll = _aggregate_rows(df_split)
            b = split_roll["baseline"]
            e = split_roll["exec"]

            split_rows.append(
                {
                    "symbol": bundle.symbol,
                    "split_id": int(sp["split_id"]),
                    "test_start": int(idx0),
                    "test_end": int(idx1),
                    "signals_total": int(e["signals_total"]),
                    "baseline_entries_valid": int(b["entries_valid"]),
                    "exec_entries_valid": int(e["entries_valid"]),
                    "baseline_mean_expectancy_net": float(b["mean_expectancy_net"]),
                    "exec_mean_expectancy_net": float(e["mean_expectancy_net"]),
                    "delta_expectancy_exec_minus_baseline": float(split_roll["delta_expectancy_exec_minus_baseline"]),
                    "baseline_cvar_5": float(b["cvar_5"]),
                    "exec_cvar_5": float(e["cvar_5"]),
                    "cvar_improve_ratio": float(split_roll["cvar_improve_ratio"]),
                    "baseline_max_drawdown": float(b["max_drawdown"]),
                    "exec_max_drawdown": float(e["max_drawdown"]),
                    "maxdd_improve_ratio": float(split_roll["maxdd_improve_ratio"]),
                    "exec_entry_rate": float(e["entry_rate"]),
                    "exec_taker_share": float(e["taker_share"]),
                    "exec_median_fill_delay_min": float(e["median_fill_delay_min"]),
                    "exec_p95_fill_delay_min": float(e["p95_fill_delay_min"]),
                    "exec_median_entry_improvement_bps": float(e["median_entry_improvement_bps"]),
                }
            )
            symbol_all_rows.extend(split_signal_rows)

        df_symbol = pd.DataFrame(symbol_all_rows).sort_values("signal_time").reset_index(drop=True)
        symbol_roll = _aggregate_rows(df_symbol)
        b = symbol_roll["baseline"]
        e = symbol_roll["exec"]

        signals_sym = int(e["signals_total"])
        entries_sym = int(e["entries_valid"])
        min_trades_symbol = max(int(args.hard_min_trades_symbol), int(math.ceil(float(args.hard_min_trade_frac_symbol) * max(1, signals_sym))))
        min_entry_rate_symbol = max(float(args.hard_min_entry_rate_symbol), float(thresholds["min_entry_rate"]))

        s_entry_pass = int(np.isfinite(e["entry_rate"]) and e["entry_rate"] >= min_entry_rate_symbol)
        s_trade_count_pass = int(entries_sym >= int(min_trades_symbol))
        if s_entry_pass == 0:
            participation_fail.append(f"{bundle.symbol}:entry_rate")
        if s_trade_count_pass == 0:
            participation_fail.append(f"{bundle.symbol}:trades<{min_trades_symbol}")

        max_taker_symbol = min(float(args.hard_max_taker_share), float(thresholds["max_taker_share"]))
        max_delay_symbol = min(float(args.hard_max_median_fill_delay_min), float(thresholds["max_fill_delay_min"]))
        s_taker_pass = int(np.isfinite(e["taker_share"]) and e["taker_share"] <= max_taker_symbol)
        s_delay_pass = int(np.isfinite(e["median_fill_delay_min"]) and e["median_fill_delay_min"] <= max_delay_symbol)
        s_p95_pass = int(np.isfinite(e["p95_fill_delay_min"]) and e["p95_fill_delay_min"] <= float(args.hard_max_p95_fill_delay_min))
        s_improve_pass = int(np.isfinite(e["median_entry_improvement_bps"]) and e["median_entry_improvement_bps"] >= float(thresholds["min_median_entry_improvement_bps"]))
        if s_taker_pass == 0:
            realism_fail.append(f"{bundle.symbol}:taker_share")
        if s_delay_pass == 0:
            realism_fail.append(f"{bundle.symbol}:median_fill_delay")
        if s_p95_pass == 0:
            realism_fail.append(f"{bundle.symbol}:p95_fill_delay")

        miss_rate_sym = float(pd.to_numeric(df_symbol.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_symbol.empty else 0.0
        s_data_pass = int(miss_rate_sym <= float(args.hard_max_missing_slice_rate))
        if s_data_pass == 0:
            data_quality_fail.append(f"{bundle.symbol}:missing_slice_rate>{float(args.hard_max_missing_slice_rate):.4f}")

        req_symbol = [
            float(e["mean_expectancy_net"]),
            float(e["cvar_5"]),
            float(e["max_drawdown"]),
            float(e["entry_rate"]),
            float(e["taker_share"]),
            float(e["median_fill_delay_min"]),
            float(e["p95_fill_delay_min"]),
        ]
        s_nan_pass = int(all(np.isfinite(v) for v in req_symbol))
        if s_nan_pass == 0:
            nan_fail.append(f"{bundle.symbol}:nan_or_inf")

        symbol_rows.append(
            {
                "symbol": bundle.symbol,
                "signals_total": int(signals_sym),
                "exec_entries_valid": int(entries_sym),
                "baseline_mean_expectancy_net": float(b["mean_expectancy_net"]),
                "exec_mean_expectancy_net": float(e["mean_expectancy_net"]),
                "delta_expectancy_exec_minus_baseline": float(symbol_roll["delta_expectancy_exec_minus_baseline"]),
                "baseline_pnl_net_sum": float(b["pnl_net_sum"]),
                "exec_pnl_net_sum": float(e["pnl_net_sum"]),
                "baseline_cvar_5": float(b["cvar_5"]),
                "exec_cvar_5": float(e["cvar_5"]),
                "cvar_improve_ratio": float(symbol_roll["cvar_improve_ratio"]),
                "baseline_max_drawdown": float(b["max_drawdown"]),
                "exec_max_drawdown": float(e["max_drawdown"]),
                "maxdd_improve_ratio": float(symbol_roll["maxdd_improve_ratio"]),
                "baseline_SL_hit_rate_valid": float(b["SL_hit_rate_valid"]),
                "exec_SL_hit_rate_valid": float(e["SL_hit_rate_valid"]),
                "exec_entry_rate": float(e["entry_rate"]),
                "exec_taker_share": float(e["taker_share"]),
                "exec_median_fill_delay_min": float(e["median_fill_delay_min"]),
                "exec_p95_fill_delay_min": float(e["p95_fill_delay_min"]),
                "exec_median_entry_improvement_bps": float(e["median_entry_improvement_bps"]),
                "missing_slice_rate": float(miss_rate_sym),
                "threshold_min_entry_rate": float(min_entry_rate_symbol),
                "threshold_min_trades": int(min_trades_symbol),
                "threshold_max_taker_share": float(max_taker_symbol),
                "threshold_max_fill_delay_min": float(max_delay_symbol),
                "threshold_max_p95_fill_delay_min": float(args.hard_max_p95_fill_delay_min),
                "threshold_min_median_entry_improvement_bps": float(thresholds["min_median_entry_improvement_bps"]),
                "pass_entry_rate": int(s_entry_pass),
                "pass_trade_count": int(s_trade_count_pass),
                "pass_taker_share": int(s_taker_pass),
                "pass_fill_delay": int(s_delay_pass),
                "pass_p95_fill_delay": int(s_p95_pass),
                "pass_entry_improvement": int(s_improve_pass),
                "pass_data_quality": int(s_data_pass),
                "pass_nan_finite": int(s_nan_pass),
            }
        )
        all_signal_rows.extend(symbol_all_rows)

    df_all = pd.DataFrame(all_signal_rows).sort_values("signal_time").reset_index(drop=True)
    overall_roll = _aggregate_rows(df_all)
    b = overall_roll["baseline"]
    e = overall_roll["exec"]

    df_split_all = pd.DataFrame(split_rows)
    split_expectancy = pd.to_numeric(df_split_all.get("exec_mean_expectancy_net", np.nan), errors="coerce")
    min_split_expectancy = float(split_expectancy.min()) if not split_expectancy.empty else float("nan")
    med_split_expectancy = float(split_expectancy.median()) if not split_expectancy.empty else float("nan")
    std_split_expectancy = float(split_expectancy.std(ddof=0)) if not split_expectancy.empty else float("nan")

    if int(len(split_rows)) != int(expected_split_count):
        split_fail.append(f"split_count:{len(split_rows)}!={expected_split_count}")
    if split_expectancy.empty or split_expectancy.isna().any():
        split_fail.append("split_metrics_missing_or_nan")

    overall_signals = int(e["signals_total"])
    overall_entries = int(e["entries_valid"])
    min_trades_overall = max(int(args.hard_min_trades_overall), int(math.ceil(float(args.hard_min_trade_frac_overall) * max(1, overall_signals))))
    overall_entry_rate_pass = int(np.isfinite(e["entry_rate"]) and e["entry_rate"] >= float(args.hard_min_entry_rate_overall))
    overall_trade_count_pass = int(overall_entries >= int(min_trades_overall))
    if overall_entry_rate_pass == 0:
        participation_fail.append("overall:entry_rate")
    if overall_trade_count_pass == 0:
        participation_fail.append(f"overall:trades<{min_trades_overall}")

    overall_taker_pass = int(np.isfinite(e["taker_share"]) and e["taker_share"] <= float(args.hard_max_taker_share))
    overall_median_delay_pass = int(np.isfinite(e["median_fill_delay_min"]) and e["median_fill_delay_min"] <= float(args.hard_max_median_fill_delay_min))
    overall_p95_delay_pass = int(np.isfinite(e["p95_fill_delay_min"]) and e["p95_fill_delay_min"] <= float(args.hard_max_p95_fill_delay_min))
    if overall_taker_pass == 0:
        realism_fail.append("overall:taker_share")
    if overall_median_delay_pass == 0:
        realism_fail.append("overall:median_fill_delay")
    if overall_p95_delay_pass == 0:
        realism_fail.append("overall:p95_fill_delay")

    missing_slice_rate = float(pd.to_numeric(df_all.get("missing_slice_flag", 0), errors="coerce").fillna(0).mean()) if not df_all.empty else float("nan")
    data_quality_pass = int(np.isfinite(missing_slice_rate) and missing_slice_rate <= float(args.hard_max_missing_slice_rate))
    if data_quality_pass == 0:
        data_quality_fail.append(f"overall:missing_slice_rate>{float(args.hard_max_missing_slice_rate):.4f}")

    req_overall = [
        float(e["mean_expectancy_net"]),
        float(e["cvar_5"]),
        float(e["max_drawdown"]),
        float(e["entry_rate"]),
        float(e["taker_share"]),
        float(e["median_fill_delay_min"]),
        float(e["p95_fill_delay_min"]),
    ]
    nan_pass = int(all(np.isfinite(v) for v in req_overall))
    if nan_pass == 0:
        nan_fail.append("overall:nan_or_inf")

    split_pass = int(len(split_fail) == 0)
    if lookahead_violations > 0:
        struct_fail_reasons.append("lookahead_violation")
    constraint_pass = int(len(struct_fail_reasons) == 0 and fee_model_identical == 1 and split_pass == 1)
    participation_pass = int(len(participation_fail) == 0)
    realism_pass = int(len(realism_fail) == 0)
    viability_pass = int(participation_pass == 1 and realism_pass == 1)

    invalid_reasons: List[str] = []
    invalid_reasons.extend(struct_fail_reasons)
    invalid_reasons.extend(sorted(set(participation_fail)))
    invalid_reasons.extend(sorted(set(realism_fail)))
    invalid_reasons.extend(sorted(set(nan_fail)))
    invalid_reasons.extend(sorted(set(data_quality_fail)))
    invalid_reasons.extend(sorted(set(split_fail)))

    hard_invalid = int(
        (constraint_pass == 0)
        or (participation_pass == 0)
        or (realism_pass == 0)
        or (nan_pass == 0)
        or (data_quality_pass == 0)
        or (split_pass == 0)
    )
    valid_for_ranking = int(hard_invalid == 0)

    cvar_gate_pass = int(np.isfinite(overall_roll["cvar_improve_ratio"]) and overall_roll["cvar_improve_ratio"] >= float(args.gate_cvar_improve_min))
    maxdd_gate_pass = int(np.isfinite(overall_roll["maxdd_improve_ratio"]) and overall_roll["maxdd_improve_ratio"] >= float(args.gate_maxdd_improve_min))

    rank_key = (
        int(valid_for_ranking),
        int(constraint_pass),
        int(participation_pass),
        int(realism_pass),
        float(e["mean_expectancy_net"]) if np.isfinite(e["mean_expectancy_net"]) else -1e9,
        float(overall_roll["cvar_improve_ratio"]) if np.isfinite(overall_roll["cvar_improve_ratio"]) else -1e9,
        float(overall_roll["maxdd_improve_ratio"]) if np.isfinite(overall_roll["maxdd_improve_ratio"]) else -1e9,
        float(med_split_expectancy) if np.isfinite(med_split_expectancy) else -1e9,
        -float(std_split_expectancy) if np.isfinite(std_split_expectancy) else -1e9,
    )

    out: Dict[str, Any] = {
        "constraint_pass": int(constraint_pass),
        "constraint_fail_reason": "|".join(sorted(set(struct_fail_reasons))),
        "participation_pass": int(participation_pass),
        "participation_fail_reason": "|".join(sorted(set(participation_fail))),
        "realism_pass": int(realism_pass),
        "realism_fail_reason": "|".join(sorted(set(realism_fail))),
        "nan_pass": int(nan_pass),
        "nan_fail_reason": "|".join(sorted(set(nan_fail))),
        "data_quality_pass": int(data_quality_pass),
        "data_quality_fail_reason": "|".join(sorted(set(data_quality_fail))),
        "split_pass": int(split_pass),
        "split_fail_reason": "|".join(sorted(set(split_fail))),
        "hard_invalid": int(hard_invalid),
        "valid_for_ranking": int(valid_for_ranking),
        "invalid_reason": "|".join(sorted(set(invalid_reasons))),
        "viability_pass": int(viability_pass),
        "viability_fail_reason": "|".join(sorted(set(participation_fail + realism_fail))),
        "fee_model_identical": int(fee_model_identical),
        "lookahead_violations": int(lookahead_violations),
        "split_count": int(len(split_rows)),
        "expected_split_count": int(expected_split_count),
        "overall_signals_total": int(overall_signals),
        "overall_entries_valid": int(overall_entries),
        "overall_min_trades_required": int(min_trades_overall),
        "overall_entry_rate": float(e["entry_rate"]),
        "overall_exec_expectancy_net": float(e["mean_expectancy_net"]),
        "overall_baseline_expectancy_net": float(b["mean_expectancy_net"]),
        "overall_delta_expectancy_exec_minus_baseline": float(overall_roll["delta_expectancy_exec_minus_baseline"]),
        "overall_exec_pnl_net_sum": float(e["pnl_net_sum"]),
        "overall_baseline_pnl_net_sum": float(b["pnl_net_sum"]),
        "overall_exec_cvar_5": float(e["cvar_5"]),
        "overall_baseline_cvar_5": float(b["cvar_5"]),
        "overall_cvar_improve_ratio": float(overall_roll["cvar_improve_ratio"]),
        "overall_exec_max_drawdown": float(e["max_drawdown"]),
        "overall_baseline_max_drawdown": float(b["max_drawdown"]),
        "overall_maxdd_improve_ratio": float(overall_roll["maxdd_improve_ratio"]),
        "overall_exec_taker_share": float(e["taker_share"]),
        "overall_exec_median_fill_delay_min": float(e["median_fill_delay_min"]),
        "overall_exec_p95_fill_delay_min": float(e["p95_fill_delay_min"]),
        "overall_exec_median_entry_improvement_bps": float(e["median_entry_improvement_bps"]),
        "overall_missing_slice_rate": float(missing_slice_rate),
        "overall_exec_sl_hit_rate_valid": float(e["SL_hit_rate_valid"]),
        "overall_baseline_sl_hit_rate_valid": float(b["SL_hit_rate_valid"]),
        "overall_exec_worst_decile_mean": float(e["worst_decile_mean"]),
        "overall_baseline_worst_decile_mean": float(b["worst_decile_mean"]),
        "overall_exec_pnl_std": float(e["pnl_std"]),
        "overall_baseline_pnl_std": float(b["pnl_std"]),
        "min_split_expectancy_net": float(min_split_expectancy),
        "median_split_expectancy_net": float(med_split_expectancy),
        "std_split_expectancy_net": float(std_split_expectancy),
        "tail_gate_pass_cvar": int(cvar_gate_pass),
        "tail_gate_pass_maxdd": int(maxdd_gate_pass),
        "rank_key": [float(x) for x in rank_key],
        "eval_time_sec": float(time.time() - t0),
    }

    if detailed:
        out["split_rows_df"] = df_split_all
        out["symbol_rows_df"] = pd.DataFrame(symbol_rows).sort_values("symbol").reset_index(drop=True)
        out["signal_rows_df"] = df_all

    return out


def _worker_init(bundles: List[SymbolBundle], args_dict: Dict[str, Any]) -> None:
    global _WORKER_STATE
    ns = argparse.Namespace(**args_dict)
    _WORKER_STATE = {"bundles": bundles, "args": ns}


def _worker_eval(genome: Dict[str, Any]) -> Dict[str, Any]:
    bundles = _WORKER_STATE["bundles"]
    args = _WORKER_STATE["args"]
    return _evaluate_genome(genome=genome, bundles=bundles, args=args, detailed=False)


def _genome_hash(genome: Dict[str, Any]) -> str:
    txt = json.dumps(genome, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:24]


def _participation_risk_score_bucket(g: Dict[str, Any]) -> Tuple[int, str]:
    entry_mode = str(g.get("entry_mode", "market"))
    fallback_to_market = int(_as_bool(g.get("fallback_to_market", 1)))
    limit_offset_bps = float(g.get("limit_offset_bps", 0.0))
    max_fill_delay_min = float(g.get("max_fill_delay_min", 0.0))
    fallback_delay_min = float(g.get("fallback_delay_min", 0.0))
    micro_vol = int(_as_bool(g.get("micro_vol_filter", 0)))
    vol_thr = float(g.get("vol_threshold", 2.5))
    killzone = int(_as_bool(g.get("killzone_filter", 0)))
    mss_disp = int(_as_bool(g.get("mss_displacement_gate", 0)))
    use_quality = int(_as_bool(g.get("use_signal_quality_gate", 0)))
    min_quality = float(g.get("min_signal_quality_gate", 0.0))
    min_improve = float(g.get("min_entry_improvement_bps_gate", 0.0))
    spread_guard = float(g.get("spread_guard_bps", 100.0))
    cooldown = float(g.get("cooldown_min", 0.0))

    score = 0
    if mss_disp == 1:
        score += 2
    if micro_vol == 1:
        score += 1
        if vol_thr <= 2.0:
            score += 1
    if use_quality == 1:
        score += 1
        if min_quality >= 0.20:
            score += 1
    if entry_mode in {"limit", "hybrid"} and fallback_to_market == 0:
        score += 2
    if entry_mode in {"limit", "hybrid"} and limit_offset_bps >= 6.0:
        score += 1
    if max_fill_delay_min <= 15.0:
        score += 1
    if fallback_delay_min >= 6.0:
        score += 1
    if min_improve >= 0.75:
        score += 1
    if spread_guard <= 30.0:
        score += 2
    elif spread_guard <= 45.0:
        score += 1
    if killzone == 1:
        score += 1
    if cooldown >= 30.0:
        score += 1

    if score <= 2:
        bucket = "low"
    elif score <= 5:
        bucket = "medium"
    else:
        bucket = "high"
    return int(score), bucket


def _invalid_by_construction_reasons(g: Dict[str, Any], mode: str) -> List[str]:
    tight = str(mode).lower() == "tight"
    if not tight:
        return []

    reasons: List[str] = []
    entry_mode = str(g.get("entry_mode", "market"))
    fallback_to_market = int(_as_bool(g.get("fallback_to_market", 1)))
    limit_offset_bps = float(g.get("limit_offset_bps", 0.0))
    max_fill_delay_min = float(g.get("max_fill_delay_min", 0.0))
    micro_vol = int(_as_bool(g.get("micro_vol_filter", 0)))
    skip_vol = int(_as_bool(g.get("skip_if_vol_gate", 0)))
    killzone = int(_as_bool(g.get("killzone_filter", 0)))
    mss_disp = int(_as_bool(g.get("mss_displacement_gate", 0)))
    use_quality = int(_as_bool(g.get("use_signal_quality_gate", 0)))
    min_improve = float(g.get("min_entry_improvement_bps_gate", 0.0))
    vol_thr = float(g.get("vol_threshold", 2.5))
    spread_guard = float(g.get("spread_guard_bps", 100.0))

    if entry_mode in {"limit", "hybrid"} and fallback_to_market == 0 and limit_offset_bps >= 15.0 and max_fill_delay_min <= 12.0:
        reasons.append("strict_limit_no_fallback_low_delay")
    if micro_vol == 1 and skip_vol == 1 and vol_thr <= 1.2:
        reasons.append("aggressive_vol_skip")
    if (killzone + mss_disp + use_quality + int(micro_vol == 1 and skip_vol == 1)) >= 3:
        reasons.append("stacked_restrictive_filters")
    if min_improve >= 6.0 and entry_mode in {"limit", "hybrid"} and fallback_to_market == 0:
        reasons.append("high_improvement_gate_without_fallback")
    # Phase N: spread_guard + displacement + strict entry-improvement are dominant participation killers.
    if spread_guard <= 30.0 and mss_disp == 1:
        reasons.append("spread_guard_too_strict_with_displacement")
    if mss_disp == 1 and limit_offset_bps >= 4.0 and max_fill_delay_min <= 18.0:
        reasons.append("displacement_with_strict_limit_and_short_cancel")
    if mss_disp == 1 and min_improve >= 0.75:
        reasons.append("displacement_with_strict_entry_improvement_gate")
    if mss_disp == 1 and use_quality == 1 and micro_vol == 1:
        reasons.append("mss_quality_microvol_stack")
    score, bucket = _participation_risk_score_bucket(g)
    if bucket == "high" and score >= 8:
        reasons.append("high_participation_risk_bundle")

    return reasons


def _apply_feasibility_repairs(out: Dict[str, Any], mode: str, repair_hist: Optional[Counter]) -> Dict[str, Any]:
    tight = str(mode).lower() == "tight"
    if not tight:
        return out

    def _hit(key: str) -> None:
        if repair_hist is not None:
            repair_hist[key] += 1

    if out["entry_mode"] in {"limit", "hybrid"} and int(out["fallback_to_market"]) == 0:
        out["fallback_to_market"] = 1
        _hit("enforce_fallback_for_nonmarket")

    if out["entry_mode"] in {"limit", "hybrid"} and float(out["limit_offset_bps"]) > 6.0:
        out["limit_offset_bps"] = 6.0
        _hit("cap_limit_offset_bps_tight")

    if int(out["fallback_to_market"]) == 1 and int(out["fallback_delay_min"]) > 3:
        out["fallback_delay_min"] = 3
        _hit("cap_fallback_delay_min_tight")

    if float(out["min_entry_improvement_bps_gate"]) > 0.75:
        out["min_entry_improvement_bps_gate"] = 0.75
        _hit("cap_min_entry_improvement_gate_tight")

    if float(out["spread_guard_bps"]) < 45.0:
        out["spread_guard_bps"] = 45.0
        _hit("raise_spread_guard_floor_tight")

    if int(out["killzone_filter"]) == 1:
        out["killzone_filter"] = 0
        _hit("disable_killzone_filter_tight")

    if int(out["micro_vol_filter"]) == 1 and float(out["vol_threshold"]) < 2.5:
        out["vol_threshold"] = 2.5
        _hit("raise_vol_threshold_under_micro_filter")

    if int(out["micro_vol_filter"]) == 1 and int(out["skip_if_vol_gate"]) == 1 and float(out["vol_threshold"]) < 3.5:
        out["skip_if_vol_gate"] = 0
        _hit("disable_skip_if_vol_gate_under_micro_filter")

    if int(out["mss_displacement_gate"]) == 1 and int(out["max_fill_delay_min"]) < 24:
        out["max_fill_delay_min"] = 24
        if int(out["fallback_delay_min"]) > int(out["max_fill_delay_min"]):
            out["fallback_delay_min"] = int(out["max_fill_delay_min"])
        _hit("ensure_fill_window_for_displacement_gate")

    if int(out["mss_displacement_gate"]) == 1 and float(out["limit_offset_bps"]) > 2.0:
        out["limit_offset_bps"] = 2.0
        _hit("cap_limit_offset_with_displacement")

    if int(out["mss_displacement_gate"]) == 1 and float(out["min_entry_improvement_bps_gate"]) > 0.25:
        out["min_entry_improvement_bps_gate"] = 0.25
        _hit("cap_entry_improvement_with_displacement")

    if int(out["mss_displacement_gate"]) == 1 and int(out["fallback_to_market"]) == 1 and int(out["fallback_delay_min"]) > 2:
        out["fallback_delay_min"] = 2
        _hit("tighten_fallback_delay_with_displacement")

    if int(out["mss_displacement_gate"]) == 1 and float(out["spread_guard_bps"]) < 60.0:
        out["spread_guard_bps"] = 60.0
        _hit("raise_spread_guard_with_displacement")

    if int(out["use_signal_quality_gate"]) == 1 and float(out["min_signal_quality_gate"]) > 0.35:
        out["min_signal_quality_gate"] = 0.35
        _hit("cap_signal_quality_gate_tight")

    if int(out["cooldown_min"]) > 20:
        out["cooldown_min"] = 20
        _hit("cap_cooldown_min_tight")

    restrictive = int(out["mss_displacement_gate"]) + int(out["use_signal_quality_gate"]) + int(out["micro_vol_filter"] and out["skip_if_vol_gate"])
    if restrictive >= 2 and int(out["use_signal_quality_gate"]) == 1 and float(out["min_signal_quality_gate"]) > 0.25:
        out["min_signal_quality_gate"] = 0.25
        _hit("soften_quality_gate_under_stack")

    # If very low limit offset is sampled, strict entry-improvement gate becomes self-contradictory.
    if out["entry_mode"] in {"limit", "hybrid"} and float(out["limit_offset_bps"]) <= 0.5 and float(out["min_entry_improvement_bps_gate"]) > 0.10:
        out["min_entry_improvement_bps_gate"] = 0.10
        _hit("align_entry_improve_with_limit_offset")

    # Avoid over-stacking restrictive controls in tight mode.
    stack = int(out["mss_displacement_gate"]) + int(out["micro_vol_filter"]) + int(out["use_signal_quality_gate"])
    if stack >= 3:
        out["use_signal_quality_gate"] = 0
        out["min_signal_quality_gate"] = 0.0
        _hit("remove_quality_gate_from_restrictive_stack")

    score, bucket = _participation_risk_score_bucket(out)
    if bucket == "high" and score >= 7:
        out["mss_displacement_gate"] = 0
        out["micro_vol_filter"] = 0
        out["use_signal_quality_gate"] = 0
        out["min_signal_quality_gate"] = 0.0
        out["spread_guard_bps"] = max(float(out["spread_guard_bps"]), 60.0)
        out["fallback_to_market"] = 1
        out["fallback_delay_min"] = min(int(out["fallback_delay_min"]), 2)
        _hit("collapse_high_risk_bundle_to_feasible_prior")

    return out


def _random_genome(
    rng: random.Random,
    mode: str,
    *,
    repair_hist: Optional[Counter] = None,
    reject_hist: Optional[Counter] = None,
    sampler_telemetry: Optional[Counter] = None,
) -> Dict[str, Any]:
    tight = str(mode).lower() == "tight"
    g: Dict[str, Any] = {}
    max_resample = 20 if tight else 8
    for _ in range(max_resample):
        if tight:
            # Phase N evidence: spread_guard and displacement-stack are dominant entry killers.
            # Keep exploration, but bias to participation-feasible regions.
            feasible_prior = rng.random() < 0.80
            if feasible_prior:
                fallback_to_market = 1 if rng.random() < 0.98 else 0
                g = {
                    "entry_mode": rng.choices(ENTRY_MODES, weights=[0.15, 0.20, 0.65], k=1)[0],
                    "limit_offset_bps": rng.uniform(0.0, 3.0),
                    "max_fill_delay_min": rng.randint(18, 45),
                    "fallback_to_market": fallback_to_market,
                    "fallback_delay_min": 0,
                    "max_taker_share": rng.uniform(0.0, 0.25),
                    "micro_vol_filter": 1 if rng.random() < 0.15 else 0,
                    "vol_threshold": rng.uniform(2.5, 6.0),
                    "spread_guard_bps": rng.uniform(45.0, 100.0),
                    "killzone_filter": 0,
                    "mss_displacement_gate": 1 if rng.random() < 0.10 else 0,
                    "min_entry_improvement_bps_gate": rng.uniform(0.0, 0.35),
                    "tp_mult": rng.uniform(0.5, 3.0),
                    "sl_mult": rng.uniform(0.3, 2.0),
                    "time_stop_min": rng.randint(0, 72 * 60),
                    "break_even_enabled": rng.choice([0, 1]),
                    "break_even_trigger_r": rng.uniform(0.25, 1.5),
                    "break_even_offset_bps": rng.uniform(0.0, 10.0),
                    "trailing_enabled": rng.choice([0, 1]),
                    "trail_start_r": rng.uniform(0.5, 2.0),
                    "trail_step_bps": rng.uniform(1.0, 50.0),
                    "partial_take_enabled": rng.choice([0, 1]),
                    "partial_take_r": rng.uniform(0.3, 1.5),
                    "partial_take_pct": rng.uniform(0.1, 0.9),
                    "skip_if_vol_gate": 1 if rng.random() < 0.05 else 0,
                    "use_signal_quality_gate": 1 if rng.random() < 0.10 else 0,
                    "min_signal_quality_gate": rng.uniform(0.0, 0.20),
                    "cooldown_min": rng.randint(0, 20),
                }
            else:
                fallback_to_market = 1 if rng.random() < 0.85 else 0
                g = {
                    "entry_mode": rng.choices(ENTRY_MODES, weights=[0.35, 0.20, 0.45], k=1)[0],
                    "limit_offset_bps": rng.uniform(0.0, 8.0),
                    "max_fill_delay_min": rng.randint(6, 45),
                    "fallback_to_market": fallback_to_market,
                    "fallback_delay_min": 0,
                    "max_taker_share": rng.uniform(0.0, 0.25),
                    "micro_vol_filter": 1 if rng.random() < 0.30 else 0,
                    "vol_threshold": rng.uniform(1.5, 6.0),
                    "spread_guard_bps": rng.uniform(30.0, 100.0),
                    "killzone_filter": 1 if rng.random() < 0.02 else 0,
                    "mss_displacement_gate": 1 if rng.random() < 0.25 else 0,
                    "min_entry_improvement_bps_gate": rng.uniform(0.0, 1.5),
                    "tp_mult": rng.uniform(0.5, 3.0),
                    "sl_mult": rng.uniform(0.3, 2.0),
                    "time_stop_min": rng.randint(0, 72 * 60),
                    "break_even_enabled": rng.choice([0, 1]),
                    "break_even_trigger_r": rng.uniform(0.25, 1.5),
                    "break_even_offset_bps": rng.uniform(0.0, 10.0),
                    "trailing_enabled": rng.choice([0, 1]),
                    "trail_start_r": rng.uniform(0.5, 2.0),
                    "trail_step_bps": rng.uniform(1.0, 50.0),
                    "partial_take_enabled": rng.choice([0, 1]),
                    "partial_take_r": rng.uniform(0.3, 1.5),
                    "partial_take_pct": rng.uniform(0.1, 0.9),
                    "skip_if_vol_gate": 1 if rng.random() < 0.15 else 0,
                    "use_signal_quality_gate": 1 if rng.random() < 0.25 else 0,
                    "min_signal_quality_gate": rng.uniform(0.0, 0.40),
                    "cooldown_min": rng.randint(0, 45),
                }
            if sampler_telemetry is not None:
                sampler_telemetry["proposal_feasible_prior_count" if feasible_prior else "proposal_exploration_count"] += 1
        else:
            g = {
                "entry_mode": rng.choice(ENTRY_MODES),
                "limit_offset_bps": rng.uniform(0.0, 30.0),
                "max_fill_delay_min": rng.randint(0, 180),
                "fallback_to_market": rng.choice([0, 1]),
                "fallback_delay_min": 0,
                "max_taker_share": rng.uniform(0.0, 1.0),
                "micro_vol_filter": rng.choice([0, 1]),
                "vol_threshold": rng.uniform(1.0, 4.0),
                "spread_guard_bps": rng.uniform(0.0, 40.0),
                "killzone_filter": rng.choice([0, 1]),
                "mss_displacement_gate": rng.choice([0, 1]),
                "min_entry_improvement_bps_gate": rng.uniform(0.0, 20.0),
                "tp_mult": rng.uniform(0.5, 3.0),
                "sl_mult": rng.uniform(0.3, 2.0),
                "time_stop_min": rng.randint(0, 72 * 60),
                "break_even_enabled": rng.choice([0, 1]),
                "break_even_trigger_r": rng.uniform(0.25, 1.5),
                "break_even_offset_bps": rng.uniform(0.0, 10.0),
                "trailing_enabled": rng.choice([0, 1]),
                "trail_start_r": rng.uniform(0.5, 2.0),
                "trail_step_bps": rng.uniform(1.0, 50.0),
                "partial_take_enabled": rng.choice([0, 1]),
                "partial_take_r": rng.uniform(0.3, 1.5),
                "partial_take_pct": rng.uniform(0.1, 0.9),
                "skip_if_vol_gate": rng.choice([0, 1]),
                "use_signal_quality_gate": rng.choice([0, 1]),
                "min_signal_quality_gate": rng.uniform(0.0, 1.0),
                "cooldown_min": rng.randint(0, 240),
            }
        g["fallback_delay_min"] = rng.randint(0, int(g["max_fill_delay_min"]))
        pre_score, pre_bucket = _participation_risk_score_bucket(g)
        if sampler_telemetry is not None:
            sampler_telemetry["proposed_pre_repair_total"] += 1
            sampler_telemetry[f"risk_bucket_pre::{pre_bucket}"] += 1
        reject_reasons = _invalid_by_construction_reasons(g, mode=mode)
        if not reject_reasons:
            break
        if reject_hist is not None:
            for reason in reject_reasons:
                reject_hist[reason] += 1
        if sampler_telemetry is not None:
            sampler_telemetry["rejected_invalid_construction_total"] += 1
            for reason in reject_reasons:
                sampler_telemetry[f"reject_reason::{reason}"] += 1
    else:
        if reject_hist is not None:
            reject_hist["resample_budget_exhausted"] += 1
        if sampler_telemetry is not None:
            sampler_telemetry["resample_budget_exhausted"] += 1

    g_post = _repair_genome(g, mode=mode, repair_hist=repair_hist)
    post_score, post_bucket = _participation_risk_score_bucket(g_post)
    if sampler_telemetry is not None:
        sampler_telemetry["proposed_post_repair_total"] += 1
        sampler_telemetry[f"risk_bucket_post::{post_bucket}"] += 1
        if post_score < pre_score:
            sampler_telemetry["repaired_for_participation_risk_total"] += 1
        if post_bucket != pre_bucket:
            sampler_telemetry["risk_bucket_changed_by_repair_total"] += 1
    return g_post


def _repair_genome(g: Dict[str, Any], mode: str, repair_hist: Optional[Counter] = None) -> Dict[str, Any]:
    out = dict(g)
    out["entry_mode"] = str(out.get("entry_mode", "market"))
    if out["entry_mode"] not in ENTRY_MODES:
        out["entry_mode"] = "market"

    tight = str(mode).lower() == "tight"
    out["limit_offset_bps"] = _clamp(float(out.get("limit_offset_bps", 0.0)), 0.0, 30.0)
    out["max_fill_delay_min"] = int(round(_clamp(float(out.get("max_fill_delay_min", 0.0)), 0.0, 45.0 if tight else 180.0)))
    out["fallback_to_market"] = int(_as_bool(out.get("fallback_to_market", 1)))
    out["fallback_delay_min"] = int(round(_clamp(float(out.get("fallback_delay_min", 0.0)), 0.0, float(out["max_fill_delay_min"]))))
    out["max_taker_share"] = _clamp(float(out.get("max_taker_share", 1.0)), 0.0, 0.25 if tight else 1.0)
    out["micro_vol_filter"] = int(_as_bool(out.get("micro_vol_filter", 0)))
    out["vol_threshold"] = _clamp(float(out.get("vol_threshold", 2.5)), 0.5, 6.0)
    out["spread_guard_bps"] = _clamp(float(out.get("spread_guard_bps", 10.0)), 0.0, 100.0)
    out["killzone_filter"] = int(_as_bool(out.get("killzone_filter", 0)))
    out["mss_displacement_gate"] = int(_as_bool(out.get("mss_displacement_gate", 0)))
    out["min_entry_improvement_bps_gate"] = _clamp(float(out.get("min_entry_improvement_bps_gate", 0.0)), 0.0, 20.0)

    out["tp_mult"] = _clamp(float(out.get("tp_mult", 1.0)), 0.5, 3.0)
    out["sl_mult"] = _clamp(float(out.get("sl_mult", 1.0)), 0.3, 2.0)
    out["time_stop_min"] = int(round(_clamp(float(out.get("time_stop_min", 0.0)), 0.0, 72.0 * 60.0)))

    out["break_even_enabled"] = int(_as_bool(out.get("break_even_enabled", 0)))
    out["break_even_trigger_r"] = _clamp(float(out.get("break_even_trigger_r", 1.0)), 0.25, 1.5)
    out["break_even_offset_bps"] = _clamp(float(out.get("break_even_offset_bps", 0.0)), 0.0, 10.0)

    out["trailing_enabled"] = int(_as_bool(out.get("trailing_enabled", 0)))
    out["trail_start_r"] = _clamp(float(out.get("trail_start_r", 1.0)), 0.5, 2.0)
    out["trail_step_bps"] = _clamp(float(out.get("trail_step_bps", 10.0)), 1.0, 50.0)

    out["partial_take_enabled"] = int(_as_bool(out.get("partial_take_enabled", 0)))
    out["partial_take_r"] = _clamp(float(out.get("partial_take_r", 0.5)), 0.3, 1.5)
    out["partial_take_pct"] = _clamp(float(out.get("partial_take_pct", 0.5)), 0.1, 0.9)

    out["skip_if_vol_gate"] = int(_as_bool(out.get("skip_if_vol_gate", 0)))
    out["use_signal_quality_gate"] = int(_as_bool(out.get("use_signal_quality_gate", 0)))
    out["min_signal_quality_gate"] = _clamp(float(out.get("min_signal_quality_gate", 0.0)), 0.0, 1.0)
    out["cooldown_min"] = int(round(_clamp(float(out.get("cooldown_min", 0.0)), 0.0, 240.0)))

    if out["trailing_enabled"] == 1 and out["break_even_enabled"] == 1 and out["trail_start_r"] < out["break_even_trigger_r"]:
        out["trail_start_r"] = float(out["break_even_trigger_r"])

    if out["partial_take_enabled"] == 1 and out["partial_take_r"] >= out["tp_mult"]:
        out["partial_take_r"] = max(0.3, min(1.5, out["tp_mult"] - 1e-3))

    return _apply_feasibility_repairs(out, mode=mode, repair_hist=repair_hist)


def _mutate_genome(
    g: Dict[str, Any],
    rng: random.Random,
    mut_rate: float,
    mode: str,
    *,
    repair_hist: Optional[Counter] = None,
) -> Dict[str, Any]:
    out = dict(g)
    keys = list(out.keys())
    for k in keys:
        if rng.random() > float(mut_rate):
            continue
        if k == "entry_mode":
            cands = [x for x in ENTRY_MODES if x != out[k]]
            out[k] = rng.choice(cands) if cands else out[k]
        elif k in {
            "fallback_to_market",
            "micro_vol_filter",
            "killzone_filter",
            "mss_displacement_gate",
            "break_even_enabled",
            "trailing_enabled",
            "partial_take_enabled",
            "skip_if_vol_gate",
            "use_signal_quality_gate",
        }:
            out[k] = int(1 - int(_as_bool(out[k])))
        elif k in {"max_fill_delay_min", "fallback_delay_min", "time_stop_min", "cooldown_min"}:
            lo_hi = {
                "max_fill_delay_min": (0, 45 if str(mode).lower() == "tight" else 180),
                "fallback_delay_min": (0, int(out.get("max_fill_delay_min", 0))),
                "time_stop_min": (0, 72 * 60),
                "cooldown_min": (0, 240),
            }
            lo, hi = lo_hi[k]
            step = max(1, int((hi - lo) / 10))
            out[k] = int(_clamp(int(out[k]) + rng.randint(-step, step), lo, hi))
        else:
            lo_hi = {
                "limit_offset_bps": (0.0, 30.0),
                "max_taker_share": (0.0, 0.25 if str(mode).lower() == "tight" else 1.0),
                "vol_threshold": (0.5, 6.0),
                "spread_guard_bps": (0.0, 100.0),
                "min_entry_improvement_bps_gate": (0.0, 20.0),
                "tp_mult": (0.5, 3.0),
                "sl_mult": (0.3, 2.0),
                "break_even_trigger_r": (0.25, 1.5),
                "break_even_offset_bps": (0.0, 10.0),
                "trail_start_r": (0.5, 2.0),
                "trail_step_bps": (1.0, 50.0),
                "partial_take_r": (0.3, 1.5),
                "partial_take_pct": (0.1, 0.9),
                "min_signal_quality_gate": (0.0, 1.0),
            }
            if k in lo_hi:
                lo, hi = lo_hi[k]
                width = float(hi - lo)
                cur = float(out[k])
                cur += rng.gauss(0.0, 0.12 * width)
                out[k] = _clamp(cur, lo, hi)
    return _repair_genome(out, mode=mode, repair_hist=repair_hist)


def _crossover(
    a: Dict[str, Any],
    b: Dict[str, Any],
    rng: random.Random,
    mode: str,
    *,
    repair_hist: Optional[Counter] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    keys = sorted(a.keys())
    if rng.random() < 0.5:
        # Uniform crossover.
        c1: Dict[str, Any] = {}
        c2: Dict[str, Any] = {}
        for k in keys:
            if rng.random() < 0.5:
                c1[k] = a[k]
                c2[k] = b[k]
            else:
                c1[k] = b[k]
                c2[k] = a[k]
    else:
        # 2-point crossover.
        i = rng.randint(0, max(0, len(keys) - 1))
        j = rng.randint(0, max(0, len(keys) - 1))
        lo, hi = min(i, j), max(i, j)
        c1 = dict(a)
        c2 = dict(b)
        for k in keys[lo : hi + 1]:
            c1[k], c2[k] = c2[k], c1[k]
    return (
        _repair_genome(c1, mode=mode, repair_hist=repair_hist),
        _repair_genome(c2, mode=mode, repair_hist=repair_hist),
    )


def _dedupe_population_pre_eval(
    *,
    population: List[Dict[str, Any]],
    target_size: int,
    rng: random.Random,
    mode: str,
    eval_cache: Dict[str, Dict[str, Any]],
    repair_hist: Optional[Counter],
    reject_hist: Optional[Counter],
    sampler_telemetry: Optional[Counter],
    avoid_cache_hashes: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    seen = set()
    unique_pop: List[Dict[str, Any]] = []
    removed = 0

    for g in population:
        h = _genome_hash(g)
        if h in seen:
            removed += 1
            continue
        seen.add(h)
        unique_pop.append(g)

    refill = 0
    cache_skip = 0
    refill_dup = 0
    max_attempts = max(10, int(target_size) * 50)
    attempts = 0

    while len(unique_pop) < int(target_size) and attempts < max_attempts:
        attempts += 1
        cand = _random_genome(
            rng=rng,
            mode=mode,
            repair_hist=repair_hist,
            reject_hist=reject_hist,
            sampler_telemetry=sampler_telemetry,
        )
        h = _genome_hash(cand)
        if h in seen:
            refill_dup += 1
            continue
        if avoid_cache_hashes and (h in eval_cache):
            cache_skip += 1
            continue
        seen.add(h)
        unique_pop.append(cand)
        refill += 1

    while len(unique_pop) < int(target_size):
        cand = _random_genome(
            rng=rng,
            mode=mode,
            repair_hist=repair_hist,
            reject_hist=reject_hist,
            sampler_telemetry=sampler_telemetry,
        )
        unique_pop.append(cand)
        refill += 1

    info = {
        "pre_eval_target_size": int(target_size),
        "pre_eval_unique_size": int(len(unique_pop)),
        "pre_eval_duplicates_removed": int(removed),
        "pre_eval_refill_count": int(refill),
        "pre_eval_refill_duplicate_skip_count": int(refill_dup),
        "pre_eval_refill_cache_skip_count": int(cache_skip),
    }
    return unique_pop, info


def _tournament_select(pop: List[Dict[str, Any]], records: Dict[str, Dict[str, Any]], rng: random.Random, k: int) -> Dict[str, Any]:
    k = max(2, min(int(k), len(pop)))
    cand_idx = rng.sample(range(len(pop)), k)
    best = pop[cand_idx[0]]
    best_h = _genome_hash(best)
    best_key = tuple(records[best_h]["metrics"]["rank_key"])
    for idx in cand_idx[1:]:
        g = pop[idx]
        h = _genome_hash(g)
        rk = tuple(records[h]["metrics"]["rank_key"])
        if rk > best_key:
            best = g
            best_h = h
            best_key = rk
    return dict(best)


def _pareto_front(df: pd.DataFrame, obj_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    x = df.copy().reset_index(drop=True)
    vals = x[list(obj_cols)].to_numpy(dtype=float)
    n = len(x)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        vi = vals[i]
        for j in range(n):
            if i == j or not keep[j]:
                continue
            vj = vals[j]
            if np.all(vj >= vi) and np.any(vj > vi):
                keep[i] = False
                break
    return x[keep].copy().reset_index(drop=True)


def _flatten_record(genome_hash: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "genome_hash": genome_hash,
        "first_generation": int(rec.get("first_generation", -1)),
        "last_generation": int(rec.get("last_generation", -1)),
        "seen_count": int(rec.get("seen_count", 1)),
    }
    for k, v in rec.get("metrics", {}).items():
        if k in {"split_rows_df", "symbol_rows_df", "signal_rows_df"}:
            continue
        out[k] = v
    for k, v in rec.get("genome", {}).items():
        out[f"g_{k}"] = v
    return out


def _build_bundle_for_symbol(
    *,
    symbol: str,
    signals_df: pd.DataFrame,
    signal_csv: Path,
    constraints: Dict[str, float],
    args: argparse.Namespace,
) -> SymbolBundle:
    pre_h = float(args.pre_buffer_hours)
    hor_h = float(args.exec_horizon_hours)

    min_signal = pd.to_datetime(signals_df["signal_time"].min(), utc=True)
    max_signal = pd.to_datetime(signals_df["signal_time"].max(), utc=True)
    all_start = min_signal - pd.Timedelta(hours=pre_h)
    all_end = max_signal + pd.Timedelta(hours=hor_h)

    cache_root = _resolve_path(args.cache_dir)
    df3m_all = exec3m._load_or_fetch_klines(
        symbol=str(symbol),
        timeframe=str(args.timeframe),
        start_ts=all_start,
        end_ts=all_end,
        cache_root=cache_root,
        max_retries=int(args.max_fetch_retries),
        retry_base_sleep_sec=float(args.retry_base_sleep),
        retry_max_sleep_sec=float(args.retry_max_sleep),
        pause_sec=float(args.fetch_pause_sec),
    )
    df3m_all = exec3m._normalize_ohlcv_cols(df3m_all)

    contexts: List[SignalContext] = []
    for _, r in signals_df.iterrows():
        signal_time = exec3m._to_utc_ts(r["signal_time"])
        start_ts = signal_time - pd.Timedelta(hours=pre_h)
        end_ts = signal_time + pd.Timedelta(hours=hor_h)
        df3m = df3m_all[(df3m_all["Timestamp"] >= start_ts) & (df3m_all["Timestamp"] < end_ts)].copy().reset_index(drop=True)

        baseline = exec3m._simulate_baseline_long(
            df3m=df3m,
            signal_time=signal_time,
            tp_mult=float(r["tp_mult"]),
            sl_mult=float(r["sl_mult"]),
            eval_horizon_hours=float(args.exec_horizon_hours),
        )
        b_liq = exec3m._liquidity_type_from_entry_type(baseline.get("entry_type", "")) if bool(baseline.get("filled", False)) else ""
        b_cost = exec3m._costed_pnl_long(
            entry_price=baseline.get("entry_price"),
            exit_price=baseline.get("exit_price"),
            entry_liquidity_type=b_liq,
            fee_bps_maker=float(args.fee_bps_maker),
            fee_bps_taker=float(args.fee_bps_taker),
            slippage_bps_limit=float(args.slippage_bps_limit),
            slippage_bps_market=float(args.slippage_bps_market),
        )

        if df3m.empty:
            ctx = SignalContext(
                symbol=str(symbol),
                signal_id=str(r["signal_id"]),
                signal_time=signal_time,
                signal_ts_ns=int(signal_time.value),
                tp_mult_sig=float(r["tp_mult"]),
                sl_mult_sig=float(r["sl_mult"]),
                quality=float(pd.to_numeric(pd.Series([r.get("signal_quality", np.nan)]), errors="coerce").iloc[0]),
                baseline_entry_time=baseline.get("entry_time"),
                baseline_exit_time=baseline.get("exit_time"),
                baseline_exit_reason=str(baseline.get("exit_reason", "")),
                baseline_filled=int(bool(baseline.get("filled", False))),
                baseline_valid_for_metrics=int(bool(baseline.get("valid_for_metrics", 0))),
                baseline_sl_hit=int(bool(baseline.get("sl_hit", False))),
                baseline_tp_hit=int(bool(baseline.get("tp_hit", False))),
                baseline_same_bar_hit=int(baseline.get("same_bar_hit", 0)),
                baseline_invalid_stop_geometry=int(baseline.get("invalid_stop_geometry", 0)),
                baseline_invalid_tp_geometry=int(baseline.get("invalid_tp_geometry", 0)),
                baseline_entry_type=str(baseline.get("entry_type", "")),
                baseline_entry_price=float(baseline.get("entry_price", np.nan)),
                baseline_exit_price=float(baseline.get("exit_price", np.nan)),
                baseline_fill_liq=str(b_liq),
                baseline_fill_delay_min=float(baseline.get("fill_delay_minutes", np.nan)),
                baseline_mae_pct=float(baseline.get("mae_pct", np.nan)),
                baseline_mfe_pct=float(baseline.get("mfe_pct", np.nan)),
                baseline_pnl_gross_pct=float(b_cost["pnl_gross_pct"]),
                baseline_pnl_net_pct=float(b_cost["pnl_net_pct"]),
                ts_ns=np.array([], dtype=np.int64),
                open_np=np.array([], dtype=float),
                high_np=np.array([], dtype=float),
                low_np=np.array([], dtype=float),
                close_np=np.array([], dtype=float),
                atr_np=np.array([], dtype=float),
                swing_high=np.array([], dtype=bool),
            )
            contexts.append(ctx)
            continue

        x = df3m.copy()
        x["ATR14"] = exec3m._compute_atr14(x)
        ts_ser = pd.to_datetime(x["Timestamp"], utc=True, errors="coerce")
        open_ser = pd.to_numeric(x["Open"], errors="coerce")
        high_ser = pd.to_numeric(x["High"], errors="coerce")
        low_ser = pd.to_numeric(x["Low"], errors="coerce")
        close_ser = pd.to_numeric(x["Close"], errors="coerce")
        atr_ser = pd.to_numeric(x["ATR14"], errors="coerce")
        good = ts_ser.notna() & open_ser.notna() & high_ser.notna() & low_ser.notna() & close_ser.notna() & atr_ser.notna()

        ts_ok = ts_ser[good].tolist()
        ts_ns = np.array([int(t.value) for t in ts_ok], dtype=np.int64)
        open_np = open_ser[good].to_numpy(dtype=float)
        high_np = high_ser[good].to_numpy(dtype=float)
        low_np = low_ser[good].to_numpy(dtype=float)
        close_np = close_ser[good].to_numpy(dtype=float)
        atr_np = atr_ser[good].to_numpy(dtype=float)
        _, swing_high = exec3m._detect_swings(low=low_np, high=high_np, k=2)

        ctx = SignalContext(
            symbol=str(symbol),
            signal_id=str(r["signal_id"]),
            signal_time=signal_time,
            signal_ts_ns=int(signal_time.value),
            tp_mult_sig=float(r["tp_mult"]),
            sl_mult_sig=float(r["sl_mult"]),
            quality=float(pd.to_numeric(pd.Series([r.get("signal_quality", np.nan)]), errors="coerce").iloc[0]),
            baseline_entry_time=baseline.get("entry_time"),
            baseline_exit_time=baseline.get("exit_time"),
            baseline_exit_reason=str(baseline.get("exit_reason", "")),
            baseline_filled=int(bool(baseline.get("filled", False))),
            baseline_valid_for_metrics=int(bool(baseline.get("valid_for_metrics", 0))),
            baseline_sl_hit=int(bool(baseline.get("sl_hit", False))),
            baseline_tp_hit=int(bool(baseline.get("tp_hit", False))),
            baseline_same_bar_hit=int(baseline.get("same_bar_hit", 0)),
            baseline_invalid_stop_geometry=int(baseline.get("invalid_stop_geometry", 0)),
            baseline_invalid_tp_geometry=int(baseline.get("invalid_tp_geometry", 0)),
            baseline_entry_type=str(baseline.get("entry_type", "")),
            baseline_entry_price=float(baseline.get("entry_price", np.nan)),
            baseline_exit_price=float(baseline.get("exit_price", np.nan)),
            baseline_fill_liq=str(b_liq),
            baseline_fill_delay_min=float(baseline.get("fill_delay_minutes", np.nan)),
            baseline_mae_pct=float(baseline.get("mae_pct", np.nan)),
            baseline_mfe_pct=float(baseline.get("mfe_pct", np.nan)),
            baseline_pnl_gross_pct=float(b_cost["pnl_gross_pct"]),
            baseline_pnl_net_pct=float(b_cost["pnl_net_pct"]),
            ts_ns=ts_ns,
            open_np=open_np,
            high_np=high_np,
            low_np=low_np,
            close_np=close_np,
            atr_np=atr_np,
            swing_high=swing_high,
        )
        contexts.append(ctx)

    splits = _build_walkforward_splits(n=len(contexts), train_ratio=float(args.train_ratio), n_splits=int(args.wf_splits))

    return SymbolBundle(
        symbol=str(symbol),
        signals_csv=signal_csv,
        contexts=contexts,
        splits=splits,
        constraints=dict(constraints),
    )


def _prepare_bundles(args: argparse.Namespace) -> Tuple[List[SymbolBundle], Dict[str, Any]]:
    symbols = _resolve_symbols(args)
    if not symbols:
        raise SystemExit("No symbols resolved")

    exec_cfg_path = _resolve_path(args.execution_config)
    all_cfg = _load_execution_config(exec_cfg_path)

    bundles: List[SymbolBundle] = []
    load_meta: Dict[str, Any] = {
        "symbols": symbols,
        "execution_config_path": str(exec_cfg_path),
        "signals": {},
        "bundle_sizes": {},
    }

    for symbol in symbols:
        sdf, scsv = _load_signals_for_symbol(symbol=symbol, args=args)
        s_cfg = _symbol_exec_config(all_cfg, symbol)
        cons = s_cfg.get("tight_constraints") if str(args.mode).lower() == "tight" else s_cfg.get("constraints")
        if not isinstance(cons, dict):
            cons = {}
        bundle = _build_bundle_for_symbol(
            symbol=symbol,
            signals_df=sdf,
            signal_csv=scsv,
            constraints={
                "min_entry_rate": float(cons.get("min_entry_rate", args.tight_min_entry_rate_default if str(args.mode).lower() == "tight" else args.min_entry_rate_default)),
                "max_taker_share": float(cons.get("max_taker_share", args.tight_max_taker_share_default if str(args.mode).lower() == "tight" else args.max_taker_share_default)),
                "max_fill_delay_min": float(cons.get("max_fill_delay_min", args.tight_max_fill_delay_default if str(args.mode).lower() == "tight" else args.max_fill_delay_default)),
                "min_median_entry_improvement_bps": float(cons.get("min_median_entry_improvement_bps", 0.0)),
            },
            args=args,
        )
        bundles.append(bundle)
        load_meta["signals"][symbol] = str(scsv)
        load_meta["bundle_sizes"][symbol] = {
            "signals": int(len(bundle.contexts)),
            "splits": int(len(bundle.splits)),
        }

    return bundles, load_meta


def _evaluate_population(
    *,
    population: List[Dict[str, Any]],
    eval_cache: Dict[str, Dict[str, Any]],
    bundles: List[SymbolBundle],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    pop_hashes = [_genome_hash(g) for g in population]
    uncached_items: List[Tuple[str, Dict[str, Any]]] = []
    uncached_seen = set()
    for h, g in zip(pop_hashes, population):
        if h not in eval_cache and h not in uncached_seen:
            uncached_items.append((h, g))
            uncached_seen.add(h)

    t0 = time.time()
    eval_time = 0.0
    if uncached_items:
        if int(args.workers) > 1 and len(uncached_items) > 1:
            mp_ctx = mp.get_context("fork")
            args_dict = vars(args).copy()
            with ProcessPoolExecutor(
                max_workers=int(args.workers),
                mp_context=mp_ctx,
                initializer=_worker_init,
                initargs=(bundles, args_dict),
            ) as ex:
                fut_map = {ex.submit(_worker_eval, g): (h, g) for h, g in uncached_items}
                for fut in as_completed(fut_map):
                    h, g = fut_map[fut]
                    met = fut.result()
                    eval_cache[h] = {
                        "genome": dict(g),
                        "metrics": dict(met),
                        "first_generation": -1,
                        "last_generation": -1,
                        "seen_count": 0,
                    }
        else:
            for h, g in uncached_items:
                met = _evaluate_genome(genome=g, bundles=bundles, args=args, detailed=False)
                eval_cache[h] = {
                    "genome": dict(g),
                    "metrics": dict(met),
                    "first_generation": -1,
                    "last_generation": -1,
                    "seen_count": 0,
                }
    eval_time = float(time.time() - t0)

    records: List[Dict[str, Any]] = []
    for h, g in zip(pop_hashes, population):
        rec = eval_cache[h]
        records.append({"hash": h, "genome": g, "metrics": rec["metrics"]})

    log = {
        "uncached_count": int(len(uncached_items)),
        "population_size": int(len(population)),
        "population_unique_hashes": int(len(set(pop_hashes))),
        "cache_hit_count": int(len(population) - len(uncached_items)),
        "cache_hit_rate": float((len(population) - len(uncached_items)) / max(1, len(population))),
        "eval_time_sec": float(eval_time),
        "avg_eval_time_per_new": float(eval_time / max(1, len(uncached_items))) if uncached_items else 0.0,
    }
    return records, log


def _sort_records_desc(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(records, key=lambda r: tuple(r["metrics"]["rank_key"]), reverse=True)


def _save_resume_state(
    *,
    path: Path,
    generation_done: int,
    population_next: List[Dict[str, Any]],
    eval_cache: Dict[str, Dict[str, Any]],
    rng: random.Random,
    np_rng: np.random.Generator,
    best_hash: str,
) -> None:
    state = {
        "generation_done": int(generation_done),
        "population": population_next,
        "eval_cache": eval_cache,
        "py_random_state_b64": base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii"),
        "np_random_state": np_rng.bit_generator.state,
        "best_hash": str(best_hash),
    }
    with path.open("wb") as f:
        pickle.dump(state, f)


def _load_resume_state(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        st = pickle.load(f)
    if not isinstance(st, dict):
        raise SystemExit(f"Invalid resume state: {path}")
    return st


def _repro_cmd(args: argparse.Namespace) -> str:
    cmd = [
        "python3 -m src.execution.ga_exec_3m_opt",
        f"--symbols {args.symbols}" if str(args.symbols).strip() else (f"--symbol {args.symbol}" if str(args.symbol).strip() else f"--rank {int(args.rank)}"),
        f"--mode {args.mode}",
        f"--pop {int(args.pop)}",
        f"--gens {int(args.gens)}",
        f"--workers {int(args.workers)}",
        f"--seed {int(args.seed)}",
        f"--export-topk {int(args.export_topk)}",
        f"--execution-config {args.execution_config}",
        f"--canonical-fee-model-path {args.canonical_fee_model_path}",
        f"--canonical-metrics-definition-path {args.canonical_metrics_definition_path}",
        f"--expected-fee-model-sha256 {args.expected_fee_model_sha256}",
        f"--expected-metrics-definition-sha256 {args.expected_metrics_definition_sha256}",
        f"--allow-freeze-hash-mismatch {int(args.allow_freeze_hash_mismatch)}",
        f"--dedupe-avoid-cache-hashes {int(args.dedupe_avoid_cache_hashes)}",
        f"--timeframe {args.timeframe}",
        f"--pre-buffer-hours {float(args.pre_buffer_hours)}",
        f"--exec-horizon-hours {float(args.exec_horizon_hours)}",
        f"--max-signals {int(args.max_signals)}",
    ]
    return " \\\n  ".join([x for x in cmd if x])


def run(args: argparse.Namespace) -> Path:
    if str(args.resume).strip():
        run_dir = _resolve_path(args.resume)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = _resolve_path(args.outdir) / f"GA_EXEC_OPT_{_utc_tag()}"
        run_dir.mkdir(parents=True, exist_ok=True)

    freeze_lock = _validate_and_lock_frozen_artifacts(args=args, run_dir=run_dir)
    bundles, load_meta = _prepare_bundles(args)

    ga_cfg = {
        "generated_utc": _utc_now().isoformat(),
        "mode": str(args.mode),
        "symbols": [b.symbol for b in bundles],
        "population": int(args.pop),
        "generations": int(args.gens),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "selection": {"type": "tournament", "k": int(args.tournament_k)},
        "elitism_rate": float(args.elitism_rate),
        "mutation_rate_start": float(args.mutation_rate_start),
        "mutation_rate_end": float(args.mutation_rate_end),
        "immigrant_rate": float(args.immigrant_rate),
        "early_stop_patience": int(args.early_stop_patience),
        "gate_stall_generations": int(args.gate_stall_generations),
        "fees": {
            "fee_bps_maker": float(args.fee_bps_maker),
            "fee_bps_taker": float(args.fee_bps_taker),
            "slippage_bps_limit": float(args.slippage_bps_limit),
            "slippage_bps_market": float(args.slippage_bps_market),
        },
        "gates": {
            "cvar_improve_min": float(args.gate_cvar_improve_min),
            "maxdd_improve_min": float(args.gate_maxdd_improve_min),
        },
        "signals": load_meta,
        "freeze_lock": {
            "canonical_fee_model_path": str(freeze_lock["canonical_fee_model_path"]),
            "canonical_metrics_definition_path": str(freeze_lock["canonical_metrics_definition_path"]),
            "canonical_fee_model_sha256": str(freeze_lock["canonical_fee_model_sha256"]),
            "canonical_metrics_definition_sha256": str(freeze_lock["canonical_metrics_definition_sha256"]),
            "freeze_lock_pass": int(freeze_lock["freeze_lock_pass"]),
        },
        "gene_space": {
            "entry_mode": ENTRY_MODES,
            "limit_offset_bps": [0.0, 30.0],
            "max_fill_delay_min": [0, 180],
            "fallback_to_market": [0, 1],
            "fallback_delay_min": [0, "max_fill_delay_min"],
            "max_taker_share": [0.0, 1.0],
            "micro_vol_filter": [0, 1],
            "vol_threshold": [0.5, 6.0],
            "spread_guard_bps": [0.0, 100.0],
            "killzone_filter": [0, 1],
            "mss_displacement_gate": [0, 1],
            "min_entry_improvement_bps_gate": [0.0, 20.0],
            "tp_mult": [0.5, 3.0],
            "sl_mult": [0.3, 2.0],
            "time_stop_min": [0, 4320],
            "break_even_enabled": [0, 1],
            "break_even_trigger_r": [0.25, 1.5],
            "break_even_offset_bps": [0.0, 10.0],
            "trailing_enabled": [0, 1],
            "trail_start_r": [0.5, 2.0],
            "trail_step_bps": [1.0, 50.0],
            "partial_take_enabled": [0, 1],
            "partial_take_r": [0.3, 1.5],
            "partial_take_pct": [0.1, 0.9],
            "skip_if_vol_gate": [0, 1],
            "use_signal_quality_gate": [0, 1],
            "min_signal_quality_gate": [0.0, 1.0],
            "cooldown_min": [0, 240],
        },
        "sampler_strategy": {
            "pre_eval_genome_hash_dedupe": 1,
            "avoid_cached_hashes_on_refill": int(args.dedupe_avoid_cache_hashes),
            "constraint_first_repair": 1,
            "invalid_by_construction_reject": 1,
            "tight_mode_feasible_prior_weight": 0.80,
            "participation_risk_repair": 1,
            "participation_risk_reject_high_bundle": 1,
        },
    }
    _write_yaml_like(run_dir / "ga_config.yaml", ga_cfg)

    rng = random.Random(int(args.seed))
    np_rng = np.random.default_rng(int(args.seed))
    eval_cache: Dict[str, Dict[str, Any]] = {}
    sampler_repair_hist: Counter = Counter()
    sampler_reject_hist: Counter = Counter()
    sampler_telemetry: Counter = Counter()
    pre_eval_duplicates_removed_total = 0
    pre_eval_refill_total = 0
    pre_eval_refill_dup_skip_total = 0
    pre_eval_refill_cache_skip_total = 0

    population: List[Dict[str, Any]] = []
    start_gen = 0
    best_hash_global = ""

    resume_path = run_dir / "resume_state.pkl"
    if str(args.resume).strip() and resume_path.exists():
        st = _load_resume_state(resume_path)
        population = [
            _repair_genome(dict(g), mode=str(args.mode), repair_hist=sampler_repair_hist)
            for g in st.get("population", [])
        ]
        eval_cache = dict(st.get("eval_cache", {}))
        if "py_random_state_b64" in st:
            rng.setstate(pickle.loads(base64.b64decode(st["py_random_state_b64"])))
        if "np_random_state" in st:
            np_rng.bit_generator.state = st["np_random_state"]
        start_gen = int(st.get("generation_done", -1)) + 1
        best_hash_global = str(st.get("best_hash", ""))

    if not population:
        population = [
            _random_genome(
                rng=rng,
                mode=str(args.mode),
                repair_hist=sampler_repair_hist,
                reject_hist=sampler_reject_hist,
                sampler_telemetry=sampler_telemetry,
            )
            for _ in range(int(args.pop))
        ]

    best_key_seen: Optional[Tuple[float, ...]] = None
    no_improve_gens = 0
    no_viable_gens = 0
    invalid_reason_hist = Counter()

    for gen in range(int(start_gen), int(args.gens)):
        population, dedupe_info = _dedupe_population_pre_eval(
            population=population,
            target_size=len(population),
            rng=rng,
            mode=str(args.mode),
            eval_cache=eval_cache,
            repair_hist=sampler_repair_hist,
            reject_hist=sampler_reject_hist,
            sampler_telemetry=sampler_telemetry,
            avoid_cache_hashes=(int(args.dedupe_avoid_cache_hashes) == 1),
        )
        pre_eval_duplicates_removed_total += int(dedupe_info["pre_eval_duplicates_removed"])
        pre_eval_refill_total += int(dedupe_info["pre_eval_refill_count"])
        pre_eval_refill_dup_skip_total += int(dedupe_info["pre_eval_refill_duplicate_skip_count"])
        pre_eval_refill_cache_skip_total += int(dedupe_info["pre_eval_refill_cache_skip_count"])

        recs, perf = _evaluate_population(population=population, eval_cache=eval_cache, bundles=bundles, args=args)
        ranked = _sort_records_desc(recs)
        valid_ranked = [r for r in ranked if int(r["metrics"].get("valid_for_ranking", 0)) == 1]
        best_rec = valid_ranked[0] if valid_ranked else ranked[0]
        best_hash = str(best_rec["hash"])
        best_key = tuple(best_rec["metrics"]["rank_key"])
        best_hash_global = best_hash

        # Track visibility metadata in eval cache.
        for rr in recs:
            h = str(rr["hash"])
            info = eval_cache[h]
            if int(info.get("seen_count", 0)) <= 0:
                info["first_generation"] = int(gen)
            info["last_generation"] = int(gen)
            info["seen_count"] = int(info.get("seen_count", 0)) + 1

        valid_count = int(sum(int(r["metrics"].get("valid_for_ranking", 0)) for r in ranked))
        if valid_count == 0:
            no_viable_gens += 1
        else:
            no_viable_gens = 0

        if best_key_seen is None or best_key > best_key_seen:
            best_key_seen = best_key
            no_improve_gens = 0
        else:
            no_improve_gens += 1

        fail_hist = Counter()
        constraint_fail_count = 0
        participation_fail_count = 0
        nan_fail_count = 0
        realism_fail_count = 0
        for r in ranked:
            m = r["metrics"]
            if int(m.get("constraint_pass", 0)) == 0:
                constraint_fail_count += 1
            if int(m.get("participation_pass", 0)) == 0:
                participation_fail_count += 1
            if int(m.get("nan_pass", 0)) == 0:
                nan_fail_count += 1
            if int(m.get("realism_pass", 0)) == 0:
                realism_fail_count += 1
            reason_blob = str(m.get("invalid_reason", "")).strip()
            if reason_blob:
                for part in [x.strip() for x in reason_blob.split("|") if x.strip()]:
                    invalid_reason_hist[part] += 1
            if int(m.get("constraint_pass", 0)) == 0:
                reason = str(m.get("constraint_fail_reason", "constraint_fail")).strip() or "constraint_fail"
                fail_hist[reason] += 1
            elif int(m.get("participation_pass", 0)) == 0:
                reason = str(m.get("participation_fail_reason", "participation_fail")).strip() or "participation_fail"
                fail_hist[reason] += 1
            elif int(m.get("realism_pass", 0)) == 0:
                reason = str(m.get("realism_fail_reason", "realism_fail")).strip() or "realism_fail"
                fail_hist[reason] += 1
            elif int(m.get("nan_pass", 0)) == 0:
                reason = str(m.get("nan_fail_reason", "nan_fail")).strip() or "nan_fail"
                fail_hist[reason] += 1
            elif int(m.get("data_quality_pass", 0)) == 0:
                reason = str(m.get("data_quality_fail_reason", "data_quality_fail")).strip() or "data_quality_fail"
                fail_hist[reason] += 1
            elif int(m.get("viability_pass", 0)) == 0:
                reason = str(m.get("viability_fail_reason", "viability_fail")).strip() or "viability_fail"
                fail_hist[reason] += 1

        gen_status = {
            "generated_utc": _utc_now().isoformat(),
            "generation": int(gen),
            "population_size": int(len(population)),
            "best_hash": best_hash,
            "best_rank_key": list(best_key),
            "best_metrics": {k: v for k, v in best_rec["metrics"].items() if k not in {"split_rows_df", "symbol_rows_df", "signal_rows_df"}},
            "best_genome": best_rec["genome"],
            "valid_count": int(valid_count),
            "invalid_count": int(len(ranked) - valid_count),
            "constraint_fail_count": int(constraint_fail_count),
            "participation_fail_count": int(participation_fail_count),
            "nan_fail_count": int(nan_fail_count),
            "realism_fail_count": int(realism_fail_count),
            "constraint_fail_hist": dict(fail_hist),
            "invalid_reason_histogram": dict(sorted(invalid_reason_hist.items())),
            "eval_perf": perf,
            "pre_eval_dedupe": dedupe_info,
            "pre_eval_duplicates_removed_total": int(pre_eval_duplicates_removed_total),
            "pre_eval_refill_total": int(pre_eval_refill_total),
            "pre_eval_refill_duplicate_skip_total": int(pre_eval_refill_dup_skip_total),
            "pre_eval_refill_cache_skip_total": int(pre_eval_refill_cache_skip_total),
            "sampler_reject_histogram": dict(sorted(sampler_reject_hist.items())),
            "sampler_repair_histogram": dict(sorted(sampler_repair_hist.items())),
            "sampler_telemetry": dict(sorted(sampler_telemetry.items())),
            "no_improve_gens": int(no_improve_gens),
            "no_viable_gens": int(no_viable_gens),
        }
        _json_dump(run_dir / "gen_status.json", gen_status)
        _json_dump(run_dir / "invalid_reason_histogram.json", dict(sorted(invalid_reason_hist.items())))
        _json_dump(run_dir / f"population_gen_{gen:04d}.json", {"generation": gen, "population": population, "best_hash": best_hash})

        if no_improve_gens >= int(args.early_stop_patience):
            print(f"early_stop: no improvement for {no_improve_gens} generations", flush=True)
            _save_resume_state(
                path=resume_path,
                generation_done=int(gen),
                population_next=population,
                eval_cache=eval_cache,
                rng=rng,
                np_rng=np_rng,
                best_hash=best_hash_global,
            )
            break
        if no_viable_gens >= int(args.gate_stall_generations):
            print(f"early_stop: viability gates unmet for {no_viable_gens} generations", flush=True)
            _save_resume_state(
                path=resume_path,
                generation_done=int(gen),
                population_next=population,
                eval_cache=eval_cache,
                rng=rng,
                np_rng=np_rng,
                best_hash=best_hash_global,
            )
            break

        if gen == int(args.gens) - 1:
            _save_resume_state(
                path=resume_path,
                generation_done=int(gen),
                population_next=population,
                eval_cache=eval_cache,
                rng=rng,
                np_rng=np_rng,
                best_hash=best_hash_global,
            )
            continue

        # Breed next generation.
        elite_n = max(1, int(round(float(args.elitism_rate) * len(population))))
        imm_n = int(round(float(args.immigrant_rate) * len(population)))
        new_pop: List[Dict[str, Any]] = [dict(ranked[i]["genome"]) for i in range(min(elite_n, len(ranked)))]

        prog = float(gen + 1) / max(1.0, float(args.gens))
        mut_rate = float(args.mutation_rate_start) + (float(args.mutation_rate_end) - float(args.mutation_rate_start)) * prog
        mut_rate = _clamp(mut_rate, 0.01, 0.9)

        while len(new_pop) < max(0, len(population) - imm_n):
            p1 = _tournament_select(pop=population, records=eval_cache, rng=rng, k=int(args.tournament_k))
            p2 = _tournament_select(pop=population, records=eval_cache, rng=rng, k=int(args.tournament_k))
            c1, c2 = _crossover(p1, p2, rng=rng, mode=str(args.mode), repair_hist=sampler_repair_hist)
            c1 = _mutate_genome(c1, rng=rng, mut_rate=mut_rate, mode=str(args.mode), repair_hist=sampler_repair_hist)
            c2 = _mutate_genome(c2, rng=rng, mut_rate=mut_rate, mode=str(args.mode), repair_hist=sampler_repair_hist)
            new_pop.append(c1)
            if len(new_pop) < max(0, len(population) - imm_n):
                new_pop.append(c2)

        while len(new_pop) < len(population):
            new_pop.append(
                _random_genome(
                    rng=rng,
                    mode=str(args.mode),
                    repair_hist=sampler_repair_hist,
                    reject_hist=sampler_reject_hist,
                    sampler_telemetry=sampler_telemetry,
                )
            )

        population = new_pop[: len(population)]

        _save_resume_state(
            path=resume_path,
            generation_done=int(gen),
            population_next=population,
            eval_cache=eval_cache,
            rng=rng,
            np_rng=np_rng,
            best_hash=best_hash_global,
        )

    # Final sorted unique records.
    flat_rows: List[Dict[str, Any]] = []
    ranked_all: List[Dict[str, Any]] = []
    for h, rec in eval_cache.items():
        ranked_all.append({"hash": h, "genome": rec["genome"], "metrics": rec["metrics"]})
        flat_rows.append(_flatten_record(h, rec))
    ranked_all = _sort_records_desc(ranked_all)
    genomes_df = pd.DataFrame(flat_rows)
    genomes_df = genomes_df.sort_values(
        [
            "valid_for_ranking",
            "constraint_pass",
            "participation_pass",
            "realism_pass",
            "overall_exec_expectancy_net",
            "overall_cvar_improve_ratio",
            "overall_maxdd_improve_ratio",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).reset_index(drop=True)
    genomes_csv = run_dir / "genomes.csv"
    genomes_df.to_csv(genomes_csv, index=False)
    invalid_hist_final = Counter()
    for rec in eval_cache.values():
        reason_blob = str(rec.get("metrics", {}).get("invalid_reason", "")).strip()
        if not reason_blob:
            continue
        for part in [x.strip() for x in reason_blob.split("|") if x.strip()]:
            invalid_hist_final[part] += 1
    _json_dump(run_dir / "invalid_reason_histogram.json", dict(sorted(invalid_hist_final.items())))

    metric_sig_cols = [
        "overall_exec_expectancy_net",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "overall_entry_rate",
        "overall_entries_valid",
        "overall_exec_taker_share",
        "overall_exec_median_fill_delay_min",
    ]
    for c in metric_sig_cols:
        if c not in genomes_df.columns:
            genomes_df[c] = np.nan
    metric_sig = (
        genomes_df[metric_sig_cols]
        .copy()
        .round(12)
        .apply(lambda r: "|".join([str(x) for x in r.tolist()]), axis=1)
        if not genomes_df.empty
        else pd.Series(dtype=str)
    )
    metric_sig_counts = metric_sig.value_counts() if len(metric_sig) > 0 else pd.Series(dtype=int)
    metric_dup_rows = int((metric_sig.map(metric_sig_counts) > 1).sum()) if len(metric_sig) > 0 else 0
    metric_unique = int(metric_sig_counts.shape[0]) if len(metric_sig_counts) > 0 else 0

    run_manifest = {
        "generated_utc": _utc_now().isoformat(),
        "run_dir": str(run_dir),
        "symbols": [b.symbol for b in bundles],
        "mode": str(args.mode),
        "seed": int(args.seed),
        "population": int(args.pop),
        "generations": int(args.gens),
        "evaluated_unique_genomes": int(len(genomes_df)),
        "valid_for_ranking_count": int(pd.to_numeric(genomes_df.get("valid_for_ranking", 0), errors="coerce").fillna(0).astype(int).sum()) if not genomes_df.empty else 0,
        "invalid_reason_histogram": dict(sorted(invalid_hist_final.items())),
        "freeze_lock": freeze_lock,
        "sampler_repair_histogram": dict(sorted(sampler_repair_hist.items())),
        "sampler_reject_histogram": dict(sorted(sampler_reject_hist.items())),
        "sampler_telemetry": dict(sorted(sampler_telemetry.items())),
        "pre_eval_duplicates_removed_total": int(pre_eval_duplicates_removed_total),
        "pre_eval_refill_total": int(pre_eval_refill_total),
        "pre_eval_refill_duplicate_skip_total": int(pre_eval_refill_dup_skip_total),
        "pre_eval_refill_cache_skip_total": int(pre_eval_refill_cache_skip_total),
        "metric_signature_duplicate_rows": int(metric_dup_rows),
        "metric_signature_unique_count": int(metric_unique),
        "effective_trials_proxy": float(metric_unique),
    }
    _json_dump(run_dir / "run_manifest.json", run_manifest)

    if not ranked_all:
        raise SystemExit("No evaluated genomes")

    valid_ranked_all = [r for r in ranked_all if int(r["metrics"].get("valid_for_ranking", 0)) == 1]
    if not valid_ranked_all:
        raise SystemExit("No valid genomes passed hard constraints; best selection aborted.")
    best = valid_ranked_all[0]
    best_hash = str(best["hash"])
    best_genome = dict(best["genome"])
    best_eval = _evaluate_genome(genome=best_genome, bundles=bundles, args=args, detailed=True)

    # Repro check (same genome, same data, same metrics must match).
    best_eval_2 = _evaluate_genome(genome=best_genome, bundles=bundles, args=args, detailed=False)
    repro_check_pass = int(
        np.isfinite(best_eval["overall_exec_expectancy_net"])
        and np.isfinite(best_eval_2["overall_exec_expectancy_net"])
        and abs(float(best_eval["overall_exec_expectancy_net"]) - float(best_eval_2["overall_exec_expectancy_net"])) <= 1e-12
    )

    best_json = {
        "generated_utc": _utc_now().isoformat(),
        "best_hash": best_hash,
        "genome": best_genome,
        "metrics": {k: v for k, v in best_eval.items() if k not in {"split_rows_df", "symbol_rows_df", "signal_rows_df"}},
        "repro_check_pass": int(repro_check_pass),
    }
    _json_dump(run_dir / "best_genome.json", best_json)

    top_k = max(1, int(args.export_topk))
    top_list = []
    for r in ranked_all[:top_k]:
        top_list.append(
            {
                "hash": str(r["hash"]),
                "genome": r["genome"],
                "metrics": {k: v for k, v in r["metrics"].items() if k not in {"split_rows_df", "symbol_rows_df", "signal_rows_df"}},
            }
        )
    _json_dump(run_dir / "top_k_genomes.json", top_list)

    pareto_src = genomes_df[(pd.to_numeric(genomes_df.get("valid_for_ranking", 0), errors="coerce").fillna(0).astype(int) == 1)].copy()
    keep_cols = [
        "genome_hash",
        "valid_for_ranking",
        "constraint_pass",
        "participation_pass",
        "realism_pass",
        "viability_pass",
        "overall_exec_expectancy_net",
        "overall_cvar_improve_ratio",
        "overall_maxdd_improve_ratio",
        "overall_entry_rate",
        "overall_exec_taker_share",
        "overall_exec_median_fill_delay_min",
    ]
    for c in keep_cols:
        if c not in pareto_src.columns:
            pareto_src[c] = np.nan
    finite_mask = (
        pd.to_numeric(pareto_src["overall_exec_expectancy_net"], errors="coerce").notna()
        & pd.to_numeric(pareto_src["overall_cvar_improve_ratio"], errors="coerce").notna()
        & pd.to_numeric(pareto_src["overall_maxdd_improve_ratio"], errors="coerce").notna()
    )
    pareto_src = pareto_src[finite_mask].copy()
    pareto_df = _pareto_front(
        pareto_src[keep_cols + [c for c in pareto_src.columns if c.startswith("g_")]],
        obj_cols=["overall_exec_expectancy_net", "overall_cvar_improve_ratio", "overall_maxdd_improve_ratio"],
    ) if not pareto_src.empty else pd.DataFrame(columns=keep_cols)
    pareto_df.to_csv(run_dir / "pareto_front.csv", index=False)

    split_df = best_eval["split_rows_df"] if isinstance(best_eval.get("split_rows_df"), pd.DataFrame) else pd.DataFrame()
    sym_df = best_eval["symbol_rows_df"] if isinstance(best_eval.get("symbol_rows_df"), pd.DataFrame) else pd.DataFrame()

    split_fp = run_dir / "walkforward_results_by_split.csv"
    sym_fp = run_dir / "risk_rollup_by_symbol.csv"
    ov_fp = run_dir / "risk_rollup_overall.csv"

    split_df.to_csv(split_fp, index=False)
    sym_df.to_csv(sym_fp, index=False)

    overall_row = {
        "scope": "overall",
        "symbols": int(len(sym_df)),
        "signals_total": int(best_eval.get("overall_signals_total", 0)),
        "baseline_mean_expectancy_net": float(best_eval.get("overall_baseline_expectancy_net", np.nan)),
        "exec_mean_expectancy_net": float(best_eval.get("overall_exec_expectancy_net", np.nan)),
        "delta_expectancy_exec_minus_baseline": float(best_eval.get("overall_delta_expectancy_exec_minus_baseline", np.nan)),
        "baseline_pnl_net_sum": float(best_eval.get("overall_baseline_pnl_net_sum", np.nan)),
        "exec_pnl_net_sum": float(best_eval.get("overall_exec_pnl_net_sum", np.nan)),
        "baseline_cvar_5": float(best_eval.get("overall_baseline_cvar_5", np.nan)),
        "exec_cvar_5": float(best_eval.get("overall_exec_cvar_5", np.nan)),
        "cvar_improve_ratio": float(best_eval.get("overall_cvar_improve_ratio", np.nan)),
        "baseline_max_drawdown": float(best_eval.get("overall_baseline_max_drawdown", np.nan)),
        "exec_max_drawdown": float(best_eval.get("overall_exec_max_drawdown", np.nan)),
        "maxdd_improve_ratio": float(best_eval.get("overall_maxdd_improve_ratio", np.nan)),
        "exec_entry_rate": float(best_eval.get("overall_entry_rate", np.nan)),
        "exec_taker_share": float(best_eval.get("overall_exec_taker_share", np.nan)),
        "exec_median_fill_delay_min": float(best_eval.get("overall_exec_median_fill_delay_min", np.nan)),
        "exec_p95_fill_delay_min": float(best_eval.get("overall_exec_p95_fill_delay_min", np.nan)),
        "exec_median_entry_improvement_bps": float(best_eval.get("overall_exec_median_entry_improvement_bps", np.nan)),
        "exec_sl_hit_rate_valid": float(best_eval.get("overall_exec_sl_hit_rate_valid", np.nan)),
        "baseline_sl_hit_rate_valid": float(best_eval.get("overall_baseline_sl_hit_rate_valid", np.nan)),
        "min_split_expectancy_net": float(best_eval.get("min_split_expectancy_net", np.nan)),
    }
    pd.DataFrame([overall_row]).to_csv(ov_fp, index=False)

    # Rubric decision.
    expect_pass = int(np.isfinite(overall_row["exec_mean_expectancy_net"]) and np.isfinite(overall_row["baseline_mean_expectancy_net"]) and overall_row["exec_mean_expectancy_net"] >= overall_row["baseline_mean_expectancy_net"])
    cvar_pass = int(np.isfinite(overall_row["cvar_improve_ratio"]) and overall_row["cvar_improve_ratio"] >= float(args.gate_cvar_improve_min))
    maxdd_pass = int(np.isfinite(overall_row["maxdd_improve_ratio"]) and overall_row["maxdd_improve_ratio"] >= float(args.gate_maxdd_improve_min))
    taker_pass = int(np.isfinite(overall_row["exec_taker_share"]) and overall_row["exec_taker_share"] <= float(args.decision_taker_max))
    delay_pass = int(np.isfinite(overall_row["exec_median_fill_delay_min"]) and overall_row["exec_median_fill_delay_min"] <= float(args.decision_delay_max))
    p95_delay_pass = int(np.isfinite(overall_row["exec_p95_fill_delay_min"]) and overall_row["exec_p95_fill_delay_min"] <= float(args.hard_max_p95_fill_delay_min))

    if not split_df.empty:
        split_min = float(pd.to_numeric(split_df["exec_mean_expectancy_net"], errors="coerce").min())
        split_med = float(pd.to_numeric(split_df["exec_mean_expectancy_net"], errors="coerce").median())
        stability_pass = int(np.isfinite(split_min) and np.isfinite(split_med) and split_min >= (split_med - abs(split_med) * float(args.stability_drawdown_mult)))
    else:
        split_min = float("nan")
        split_med = float("nan")
        stability_pass = 0

    entry_gate_pass = 1
    if not sym_df.empty:
        entry_gate_pass = int(
            (pd.to_numeric(sym_df.get("pass_entry_rate", 0), errors="coerce").fillna(0).astype(int) == 1).all()
            and (pd.to_numeric(sym_df.get("pass_trade_count", 0), errors="coerce").fillna(0).astype(int) == 1).all()
        )

    deploy = int(
        expect_pass == 1
        and cvar_pass == 1
        and maxdd_pass == 1
        and taker_pass == 1
        and delay_pass == 1
        and p95_delay_pass == 1
        and entry_gate_pass == 1
        and stability_pass == 1
    )

    decision_lines: List[str] = []
    decision_lines.append("# GA Exec 3m Decision")
    decision_lines.append("")
    decision_lines.append(f"- Generated UTC: {_utc_now().isoformat()}")
    decision_lines.append(f"- Run dir: `{run_dir}`")
    decision_lines.append(f"- Best genome hash: `{best_hash}`")
    decision_lines.append(f"- Repro check pass: {int(repro_check_pass)}")
    decision_lines.append("")
    decision_lines.append("## Baseline vs Best (Overall TEST-only)")
    decision_lines.append("")
    decision_lines.append(f"- baseline_expectancy_net: {overall_row['baseline_mean_expectancy_net']:.6f}")
    decision_lines.append(f"- exec_expectancy_net: {overall_row['exec_mean_expectancy_net']:.6f}")
    decision_lines.append(f"- delta_expectancy_exec_minus_baseline: {overall_row['delta_expectancy_exec_minus_baseline']:.6f}")
    decision_lines.append(f"- baseline_cvar_5: {overall_row['baseline_cvar_5']:.6f}")
    decision_lines.append(f"- exec_cvar_5: {overall_row['exec_cvar_5']:.6f}")
    decision_lines.append(f"- cvar_improve_ratio: {overall_row['cvar_improve_ratio']:.6f}")
    decision_lines.append(f"- baseline_max_drawdown: {overall_row['baseline_max_drawdown']:.6f}")
    decision_lines.append(f"- exec_max_drawdown: {overall_row['exec_max_drawdown']:.6f}")
    decision_lines.append(f"- maxdd_improve_ratio: {overall_row['maxdd_improve_ratio']:.6f}")
    decision_lines.append(f"- exec_entry_rate: {overall_row['exec_entry_rate']:.6f}")
    decision_lines.append(f"- exec_taker_share: {overall_row['exec_taker_share']:.6f}")
    decision_lines.append(f"- exec_median_fill_delay_min: {overall_row['exec_median_fill_delay_min']:.2f}")
    decision_lines.append(f"- exec_p95_fill_delay_min: {overall_row['exec_p95_fill_delay_min']:.2f}")
    decision_lines.append("")
    decision_lines.append("## Gate Table")
    decision_lines.append("")
    decision_lines.append(f"- expectancy >= baseline: {expect_pass}")
    decision_lines.append(f"- cvar_improve >= {float(args.gate_cvar_improve_min):.0%}: {cvar_pass}")
    decision_lines.append(f"- maxdd_improve >= {float(args.gate_maxdd_improve_min):.0%}: {maxdd_pass}")
    decision_lines.append(f"- taker_share <= {float(args.decision_taker_max):.2f}: {taker_pass}")
    decision_lines.append(f"- median_fill_delay <= {float(args.decision_delay_max):.0f} min: {delay_pass}")
    decision_lines.append(f"- p95_fill_delay <= {float(args.hard_max_p95_fill_delay_min):.0f} min: {p95_delay_pass}")
    decision_lines.append(f"- per-symbol entry-rate gates pass: {entry_gate_pass}")
    decision_lines.append(f"- split stability pass: {stability_pass} (min={split_min:.6f}, median={split_med:.6f})" if np.isfinite(split_min) and np.isfinite(split_med) else "- split stability pass: 0")
    decision_lines.append("")
    decision_lines.append("## Final")
    decision_lines.append("")
    decision_lines.append(f"- Decision: **{'DEPLOY' if deploy == 1 else 'NO-DEPLOY'}**")
    decision_lines.append("")
    decision_lines.append("## Artifacts")
    decision_lines.append("")
    decision_lines.append(f"- genomes: `{genomes_csv}`")
    decision_lines.append(f"- best genome: `{run_dir / 'best_genome.json'}`")
    decision_lines.append(f"- top-k: `{run_dir / 'top_k_genomes.json'}`")
    decision_lines.append(f"- pareto front: `{run_dir / 'pareto_front.csv'}`")
    decision_lines.append(f"- split rollup: `{split_fp}`")
    decision_lines.append(f"- symbol risk rollup: `{sym_fp}`")
    decision_lines.append(f"- overall risk rollup: `{ov_fp}`")
    (run_dir / "decision.md").write_text("\n".join(decision_lines).strip() + "\n", encoding="utf-8")

    # Deployment-compatible config export.
    deploy_cfg: Dict[str, Any] = {}
    for b in bundles:
        deploy_cfg[b.symbol] = {
            "exec_mode": "ga_exec_3m_opt",
            "entry_mode": best_genome["entry_mode"],
            "limit_offset_bps": float(best_genome["limit_offset_bps"]),
            "max_fill_delay_min": int(best_genome["max_fill_delay_min"]),
            "fallback_to_market": int(best_genome["fallback_to_market"]),
            "fallback_delay_min": int(best_genome["fallback_delay_min"]),
            "micro_vol_filter": int(best_genome["micro_vol_filter"]),
            "vol_threshold": float(best_genome["vol_threshold"]),
            "spread_guard_bps": float(best_genome["spread_guard_bps"]),
            "killzone_filter": int(best_genome["killzone_filter"]),
            "mss_displacement_gate": int(best_genome["mss_displacement_gate"]),
            "min_entry_improvement_bps_gate": float(best_genome["min_entry_improvement_bps_gate"]),
            "tp_mult": float(best_genome["tp_mult"]),
            "sl_mult": float(best_genome["sl_mult"]),
            "time_stop_min": int(best_genome["time_stop_min"]),
            "break_even_enabled": int(best_genome["break_even_enabled"]),
            "break_even_trigger_r": float(best_genome["break_even_trigger_r"]),
            "break_even_offset_bps": float(best_genome["break_even_offset_bps"]),
            "trailing_enabled": int(best_genome["trailing_enabled"]),
            "trail_start_r": float(best_genome["trail_start_r"]),
            "trail_step_bps": float(best_genome["trail_step_bps"]),
            "partial_take_enabled": int(best_genome["partial_take_enabled"]),
            "partial_take_r": float(best_genome["partial_take_r"]),
            "partial_take_pct": float(best_genome["partial_take_pct"]),
            "skip_if_vol_gate": int(best_genome["skip_if_vol_gate"]),
            "cooldown_min": int(best_genome["cooldown_min"]),
            "constraints": b.constraints,
        }
    _write_yaml_like(run_dir / "best_genome_as_execution_config.yaml", deploy_cfg)

    repro_lines: List[str] = []
    repro_lines.append("# Repro")
    repro_lines.append("")
    repro_lines.append(f"- Generated UTC: {_utc_now().isoformat()}")
    repro_lines.append(f"- Seed: {int(args.seed)}")
    repro_lines.append(f"- Run dir: `{run_dir}`")
    repro_lines.append("")
    repro_lines.append("## Fresh Run")
    repro_lines.append("")
    repro_lines.append("```bash")
    repro_lines.append(_repro_cmd(args))
    repro_lines.append("```")
    repro_lines.append("")
    repro_lines.append("## Resume")
    repro_lines.append("")
    repro_lines.append("```bash")
    repro_lines.append(f"python3 -m src.execution.ga_exec_3m_opt --resume {run_dir}")
    repro_lines.append("```")
    (run_dir / "repro.md").write_text("\n".join(repro_lines).strip() + "\n", encoding="utf-8")

    # Persist last state snapshot with final generation marker.
    _save_resume_state(
        path=resume_path,
        generation_done=int(args.gens) - 1,
        population_next=population,
        eval_cache=eval_cache,
        rng=rng,
        np_rng=np_rng,
        best_hash=best_hash,
    )

    print(str(run_dir))
    print(str(genomes_csv))
    print(str(run_dir / "best_genome.json"))
    print(str(run_dir / "decision.md"))
    return run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="GA optimizer for 3m execution layer (entry + exit) on fixed 1h signals.")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--rank", type=int, default=1)
    ap.add_argument("--scan-dir", default="")
    ap.add_argument("--best-csv", default="")

    ap.add_argument("--signals-dir", default="data/signals")
    ap.add_argument("--signals-csv", default="")
    ap.add_argument("--signal-order", choices=["latest", "oldest"], default="latest")
    ap.add_argument("--max-signals", type=int, default=2000)

    ap.add_argument("--walkforward", action="store_true")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--wf-splits", type=int, default=5)

    ap.add_argument("--mode", choices=["tight", "normal"], default="tight")
    ap.add_argument("--force-no-skip", type=int, default=0)

    ap.add_argument("--timeframe", default="3m")
    ap.add_argument("--pre-buffer-hours", type=float, default=6.0)
    ap.add_argument("--exec-horizon-hours", type=float, default=12.0)
    ap.add_argument("--cache-dir", default="data/processed/_exec_klines_cache")
    ap.add_argument("--max-fetch-retries", type=int, default=8)
    ap.add_argument("--retry-base-sleep", type=float, default=0.5)
    ap.add_argument("--retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--fetch-pause-sec", type=float, default=0.03)

    ap.add_argument("--execution-config", default="configs/execution_configs.yaml")
    ap.add_argument("--fee-bps-maker", type=float, default=2.0)
    ap.add_argument("--fee-bps-taker", type=float, default=4.0)
    ap.add_argument("--slippage-bps-limit", type=float, default=0.5)
    ap.add_argument("--slippage-bps-market", type=float, default=2.0)
    ap.add_argument("--canonical-fee-model-path", default=DEFAULT_CANONICAL_FEE_MODEL_PATH)
    ap.add_argument("--canonical-metrics-definition-path", default=DEFAULT_CANONICAL_METRICS_DEFINITION_PATH)
    ap.add_argument("--expected-fee-model-sha256", default=DEFAULT_EXPECTED_FEE_MODEL_SHA256)
    ap.add_argument("--expected-metrics-definition-sha256", default=DEFAULT_EXPECTED_METRICS_DEFINITION_SHA256)
    ap.add_argument("--allow-freeze-hash-mismatch", type=int, default=0)

    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default="")
    ap.add_argument("--export-topk", type=int, default=20)

    ap.add_argument("--tournament-k", type=int, default=5)
    ap.add_argument("--elitism-rate", type=float, default=0.05)
    ap.add_argument("--mutation-rate-start", type=float, default=0.15)
    ap.add_argument("--mutation-rate-end", type=float, default=0.05)
    ap.add_argument("--immigrant-rate", type=float, default=0.05)
    ap.add_argument("--early-stop-patience", type=int, default=12)
    ap.add_argument("--gate-stall-generations", type=int, default=10)
    ap.add_argument("--dedupe-avoid-cache-hashes", type=int, default=1)

    ap.add_argument("--min-entry-rate-default", type=float, default=0.55)
    ap.add_argument("--max-fill-delay-default", type=float, default=90.0)
    ap.add_argument("--max-taker-share-default", type=float, default=0.40)
    ap.add_argument("--tight-min-entry-rate-default", type=float, default=0.55)
    ap.add_argument("--tight-max-fill-delay-default", type=float, default=45.0)
    ap.add_argument("--tight-max-taker-share-default", type=float, default=0.25)

    # Hard anti-cheat/validity gates.
    ap.add_argument("--hard-min-trades-overall", type=int, default=200)
    ap.add_argument("--hard-min-trade-frac-overall", type=float, default=0.15)
    ap.add_argument("--hard-min-trades-symbol", type=int, default=50)
    ap.add_argument("--hard-min-trade-frac-symbol", type=float, default=0.10)
    ap.add_argument("--hard-min-entry-rate-symbol", type=float, default=0.55)
    ap.add_argument("--hard-min-entry-rate-overall", type=float, default=0.70)
    ap.add_argument("--hard-max-missing-slice-rate", type=float, default=0.02)
    ap.add_argument("--hard-max-taker-share", type=float, default=0.25)
    ap.add_argument("--hard-max-median-fill-delay-min", type=float, default=45.0)
    ap.add_argument("--hard-max-p95-fill-delay-min", type=float, default=180.0)

    ap.add_argument("--gate-cvar-improve-min", type=float, default=0.15)
    ap.add_argument("--gate-maxdd-improve-min", type=float, default=0.15)
    ap.add_argument("--decision-taker-max", type=float, default=0.25)
    ap.add_argument("--decision-delay-max", type=float, default=45.0)
    ap.add_argument("--stability-drawdown-mult", type=float, default=2.0)

    ap.add_argument("--outdir", default="reports/execution_layer")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
