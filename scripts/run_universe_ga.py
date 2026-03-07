#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve()
for _p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
    if (_p / "data").is_dir() and (_p / "src").is_dir():
        PROJECT_ROOT = _p
        break

os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.cli import universe_pipeline as up  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402
from src.bot087.optim import ga_short  # noqa: E402


SUMMARY_COLUMNS = [
    "symbol",
    "side",
    "test_net",
    "test_pf",
    "test_dd",
    "test_trades",
    "stability",
    "PASS/FAIL",
    "fail_reasons",
    "param_path",
    "run_id",
    "updated_utc",
    "executed_generations",
    "symbol_first_ts_utc",
]

SKIP_SYMBOLS_DEFAULT = {"BTCUSDT", "SOLUSDT", "ADAUSDT", "ETHUSDT"}
RANKING_FORMULA = "rank = abs(rsi_prev-50.0) / ((atr_prev/max(close,eps)) + eps)"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}", flush=True)


def _apply_thread_caps() -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(os.environ.get(var, "1"))
    _log(
        "thread_caps="
        f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']} "
        f"MKL_NUM_THREADS={os.environ['MKL_NUM_THREADS']} "
        f"OPENBLAS_NUM_THREADS={os.environ['OPENBLAS_NUM_THREADS']} "
        f"NUMEXPR_NUM_THREADS={os.environ['NUMEXPR_NUM_THREADS']}"
    )


def _safe_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _dedupe_symbols(raw: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for sym in raw:
        if not sym:
            continue
        s = str(sym).upper().strip()
        if "__" in s:
            s = s.split("__", 1)[0]
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _extract_symbols_from_csv(path: Path) -> List[str]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    if "symbol" not in df.columns:
        return []
    return _dedupe_symbols([str(x) for x in df["symbol"].tolist()])


def _symbols_from_summary_reports() -> Tuple[List[str], str]:
    files_all = sorted(
        (PROJECT_ROOT / "artifacts" / "reports").glob("**/summary.csv"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    files = [p for p in files_all if not any(part.startswith("universe_") for part in p.parts)]
    for fp in files:
        syms = _extract_symbols_from_csv(fp)
        if syms:
            return syms, str(fp.resolve())
    return [], ""


def _extract_symbols_from_metadata_file(path: Path) -> List[str]:
    ext = path.suffix.lower()
    if ext == ".csv":
        try:
            df = pd.read_csv(path)
            if "symbol" in df.columns:
                return _dedupe_symbols([str(x) for x in df["symbol"].tolist()])
            if "symbols" in df.columns:
                vals: List[str] = []
                for v in df["symbols"].tolist():
                    vals.extend([x.strip() for x in str(v).split(",") if x.strip()])
                return _dedupe_symbols(vals)
        except Exception:
            return []
    if ext in {".txt", ".list"}:
        try:
            return _dedupe_symbols([ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()])
        except Exception:
            return []
    if ext == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(payload, dict):
            if isinstance(payload.get("selected"), list):
                vals = [str(x.get("symbol", "")) for x in payload["selected"] if isinstance(x, dict)]
                return _dedupe_symbols(vals)
            if isinstance(payload.get("symbols"), list):
                return _dedupe_symbols([str(x) for x in payload["symbols"]])
            if str(payload.get("symbol", "")).strip():
                return _dedupe_symbols([str(payload.get("symbol"))])
        if isinstance(payload, list):
            if payload and isinstance(payload[0], dict):
                return _dedupe_symbols([str(x.get("symbol", "")) for x in payload if isinstance(x, dict)])
            return _dedupe_symbols([str(x) for x in payload])
    return []


def _symbols_from_metadata_universe() -> Tuple[List[str], str]:
    base = PROJECT_ROOT / "data" / "metadata"
    candidates = sorted(
        [p for p in base.glob("**/*") if p.is_file() and "universe" in p.name.lower()],
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    all_syms: List[str] = []
    for fp in candidates:
        all_syms.extend(_extract_symbols_from_metadata_file(fp))
    params_dir = base / "params"
    if params_dir.exists():
        for fp in sorted(params_dir.glob("*_C13_active_params_long.json")):
            name = fp.name
            sym = name.replace("_C13_active_params_long.json", "").strip()
            if sym:
                all_syms.append(sym)
        for fp in sorted(params_dir.glob("*_C13_active_params_short.json")):
            name = fp.name
            sym = name.replace("_C13_active_params_short.json", "").strip()
            if sym:
                all_syms.append(sym)
    syms = _dedupe_symbols(all_syms)
    if syms:
        src = str(candidates[0].resolve()) if candidates else str((base / "params").resolve())
        return syms, src
    return [], ""


def _symbols_from_cli(args: argparse.Namespace) -> List[str]:
    return _csv_symbols(str(args.symbols))


def _csv_symbols(value: str) -> List[str]:
    raw = [s.strip() for s in str(value).split(",")] if str(value).strip() else []
    return _dedupe_symbols(raw)


def _resolve_symbols(args: argparse.Namespace) -> Tuple[List[str], str]:
    syms, src = _symbols_from_summary_reports()
    if syms:
        return syms, f"artifacts_reports_summary:{src}"
    syms, src = _symbols_from_metadata_universe()
    if syms:
        return syms, f"metadata_universe:{src}"
    syms = _symbols_from_cli(args)
    if syms:
        return syms, "cli_symbols"
    raise SystemExit("No symbols found from summary.csv, metadata universe file, or --symbols")


def _parse_skip_symbols(args: argparse.Namespace) -> set:
    raw = [x.strip().upper() for x in str(args.skip_symbols).split(",") if x.strip()]
    return set(raw)


def _stop_repo_processes(max_age_sec: int = 24 * 3600) -> List[Dict[str, Any]]:
    keywords = (
        "optimize_overlay",
        "run_universe_ga",
        "run_ga",
        "ga_short",
        "ga.py",
        "bot087.cli.universe_pipeline",
    )
    killed: List[Dict[str, Any]] = []
    me = os.getpid()
    p = subprocess.run(
        ["ps", "-eo", "pid=,etimes=,args="],
        check=False,
        capture_output=True,
        text=True,
    )
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) != 3:
            continue
        try:
            pid = int(parts[0])
            etimes = int(parts[1])
        except Exception:
            continue
        cmd = parts[2]
        if pid == me:
            continue
        if etimes > int(max_age_sec):
            continue
        if "python" not in cmd:
            continue
        if str(PROJECT_ROOT) not in cmd:
            continue
        if not any(k in cmd for k in keywords):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append({"pid": int(pid), "etimes_sec": int(etimes), "cmd": cmd})
        except ProcessLookupError:
            continue
        except Exception as ex:
            _log(f"housekeeping: failed to SIGTERM pid={pid}: {type(ex).__name__}: {ex}")

    # Brief grace, then hard kill survivors.
    if killed:
        time.sleep(1.5)
        for rec in list(killed):
            pid = int(rec["pid"])
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            except Exception:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    return killed


def _is_recent(path: Path, max_age_sec: float) -> bool:
    try:
        return (time.time() - path.stat().st_mtime) <= max_age_sec
    except Exception:
        return False


def _cleanup_recent_artifacts(exclude: Optional[List[Path]] = None) -> List[Path]:
    removed: List[Path] = []
    max_age = 24 * 3600
    exclude_set = {str(p.resolve()) for p in (exclude or []) if p}

    for root in [PROJECT_ROOT / "artifacts" / "runs"]:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not _is_recent(child, max_age):
                continue
            if str(child.resolve()) in exclude_set:
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
                removed.append(child)
            except Exception as ex:
                _log(f"housekeeping: failed to remove {child}: {type(ex).__name__}: {ex}")

    reports_root = PROJECT_ROOT / "artifacts" / "reports"
    if reports_root.exists():
        for child in reports_root.glob("universe_*"):
            if not _is_recent(child, max_age):
                continue
            if str(child.resolve()) in exclude_set:
                continue
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
                removed.append(child)
            except Exception as ex:
                _log(f"housekeeping: failed to remove {child}: {type(ex).__name__}: {ex}")

    for root in [PROJECT_ROOT / "artifacts"]:
        if not root.exists():
            continue
        for child in root.rglob("*"):
            name = child.name.lower()
            if "crash" not in name:
                continue
            if not _is_recent(child, max_age):
                continue
            if child.is_file():
                try:
                    child.unlink(missing_ok=True)
                    removed.append(child)
                except Exception as ex:
                    _log(f"housekeeping: failed to remove {child}: {type(ex).__name__}: {ex}")
    return removed


def _make_fetch_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        start=str(args.start),
        end=str(args.end),
        force_download=bool(args.force_download),
        offline_fallback=bool(args.offline_fallback),
        http_retries=int(args.http_retries),
        http_retry_base_sleep=float(args.http_retry_base_sleep),
        http_retry_max_sleep=float(args.http_retry_max_sleep),
    )


def _prepare_symbol_df(
    symbol: str,
    args: argparse.Namespace,
    fetch_args: SimpleNamespace,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    if bool(args.local_only):
        df = up._load_existing_full(symbol)
        if df.empty:
            raise RuntimeError("missing_local_1h_data")
        seed = up._load_long_seed(symbol)
        df = ga_long._ensure_indicators(df.copy(), ga_long._norm_params(seed))
    else:
        df = up._prepare_symbol_dataset(symbol, fetch_args)
    df = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] <= end_ts)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("empty_data_after_filter")
    return df


def _long_cfg(args: argparse.Namespace) -> ga_long.GAConfig:
    return ga_long.GAConfig(
        pop_size=int(args.pop_size),
        generations=int(args.ga_generations),
        n_procs=int(args.eval_procs),
        mc_splits=int(args.mc_splits),
        train_days=int(args.train_days),
        val_days=int(args.val_days),
        test_days=int(args.test_days),
        seed=int(args.seed),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        initial_equity=float(args.initial_equity),
        min_trades_train=int(args.min_trades_train),
        min_trades_val=int(args.min_trades_val),
        resume=False,
        early_stop_patience=int(args.early_stop),
    )


def _short_cfg(args: argparse.Namespace) -> ga_short.GAConfig:
    return ga_short.GAConfig(
        pop_size=int(args.pop_size),
        generations=int(args.ga_generations),
        n_procs=int(args.eval_procs),
        mc_splits=int(args.mc_splits),
        train_days=int(args.train_days),
        val_days=int(args.val_days),
        test_days=int(args.test_days),
        seed=int(args.seed),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        initial_equity=float(args.initial_equity),
        min_trades_train=int(args.min_trades_train),
        min_trades_val=int(args.min_trades_val),
        resume=False,
        early_stop_patience=int(args.early_stop),
    )


def _pass_fail(
    *,
    test_pf: float,
    test_dd: float,
    test_trades: float,
    stability: float,
    pf_min: float,
    dd_max: float,
    trades_min: float,
    stability_min: float,
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    if float(test_pf) < float(pf_min):
        reasons.append(f"pf<{pf_min}")
    if float(test_dd) > float(dd_max):
        reasons.append(f"dd>{dd_max}")
    if float(test_trades) < float(trades_min):
        reasons.append(f"trades<{trades_min}")
    if float(stability) < float(stability_min):
        reasons.append(f"stability<{stability_min}")
    return ("PASS" if not reasons else "FAIL", reasons)


def _save_active_side_params(
    *,
    symbol: str,
    side: str,
    params: Dict[str, Any],
    pipeline_run_id: str,
    ga_report: Dict[str, Any],
    test_metrics: Dict[str, Any],
    stability: float,
    pass_fail: str,
    fail_reasons: List[str],
    generations_target: int,
    generations_ran: int,
    early_stop: bool,
) -> Path:
    out = PROJECT_ROOT / "data" / "metadata" / "params" / f"{symbol}_C13_active_params_{side}.json"
    payload = {
        "symbol": symbol,
        "side": side,
        "params": params,
        "meta": {
            "saved_at_utc": _utc_now_iso(),
            "pipeline_run_id": pipeline_run_id,
            "ga_run_id": ga_report.get("run_id"),
            "ga_symbol": ga_report.get("symbol"),
            "ga_saved": ga_report.get("saved", {}),
            "test_metrics": test_metrics,
            "stability": float(stability),
            "pass_fail": str(pass_fail),
            "fail_reasons": list(fail_reasons),
            "generations_target": int(generations_target),
            "generations_ran": int(generations_ran),
            "early_stop": bool(early_stop),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _count_generations_ran(run_dir: Path) -> int:
    return sum(1 for _ in run_dir.glob("gen_*.json"))


def _read_generations_target_from_final(run_dir: Path) -> Optional[int]:
    fp = run_dir / "final_report.json"
    if not fp.exists():
        return None
    try:
        payload = json.loads(fp.read_text(encoding="utf-8"))
        return int(payload.get("cfg", {}).get("generations"))
    except Exception:
        return None


def _base_row(symbol: str, side: str, symbol_first_ts_utc: str, run_id: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "side": side,
        "test_net": 0.0,
        "test_pf": 0.0,
        "test_dd": 1.0,
        "test_trades": 0.0,
        "stability": 0.0,
        "PASS/FAIL": "FAIL",
        "fail_reasons": "",
        "param_path": "",
        "run_id": run_id,
        "updated_utc": _utc_now_iso(),
        "executed_generations": 0,
        "symbol_first_ts_utc": symbol_first_ts_utc,
    }


def _write_side_ga_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_summary(outputs_dir: Path, rows: List[Dict[str, Any]], meta: Dict[str, Any]) -> Tuple[Path, Path]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = outputs_dir / "summary.csv"
    json_path = outputs_dir / "summary.json"

    ordered = sorted(rows, key=lambda r: (str(r.get("symbol", "")), str(r.get("side", ""))))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        for row in ordered:
            w.writerow({k: row.get(k, "") for k in SUMMARY_COLUMNS})

    payload = {"meta": meta, "rows": ordered}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def _yearly_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["year", "trades", "net_profit", "profit_factor", "win_rate_pct", "avg_trade"]
    if trades_df.empty or "exit_ts" not in trades_df.columns or "net_pnl" not in trades_df.columns:
        return pd.DataFrame(columns=cols)

    t = trades_df.copy()
    t["exit_ts"] = pd.to_datetime(t["exit_ts"], utc=True, errors="coerce")
    t = t.dropna(subset=["exit_ts"])
    if t.empty:
        return pd.DataFrame(columns=cols)
    t["year"] = t["exit_ts"].dt.year.astype(int)
    t["net_pnl"] = pd.to_numeric(t["net_pnl"], errors="coerce").fillna(0.0)

    rows: List[Dict[str, Any]] = []
    for year, grp in t.groupby("year", sort=True):
        pnl = grp["net_pnl"].to_numpy(dtype=float)
        wins = pnl[pnl > 0.0]
        losses = pnl[pnl < 0.0]
        gp = float(wins.sum()) if wins.size else 0.0
        gl = float(losses.sum()) if losses.size else 0.0
        if gl < -1e-9:
            pf = float(gp / abs(gl))
        else:
            pf = 10.0 if gp > 0.0 else 0.0
        rows.append(
            {
                "year": int(year),
                "trades": int(len(grp)),
                "net_profit": float(pnl.sum()) if pnl.size else 0.0,
                "profit_factor": float(pf),
                "win_rate_pct": float((pnl > 0.0).mean() * 100.0) if pnl.size else 0.0,
                "avg_trade": float(pnl.mean()) if pnl.size else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=cols)


@dataclass
class SimModel:
    symbol: str
    side: str
    symbol_first_ts_ns: int
    params: Dict[str, Any]
    ts_ns: np.ndarray
    idx_by_ts: Dict[int, int]
    sig: np.ndarray
    cycles: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    atr_prev: np.ndarray
    rsi_prev: np.ndarray


def _entry_rank(m: SimModel, idx: int) -> float:
    eps = 1e-9
    signal_strength_proxy = abs(float(m.rsi_prev[idx]) - 50.0)
    recent_vol = float(m.atr_prev[idx]) / max(abs(float(m.c[idx])), eps)
    return float(signal_strength_proxy / (recent_vol + eps))


def _build_sim_model(row: Dict[str, Any], df_base: pd.DataFrame) -> SimModel:
    symbol = str(row["symbol"])
    side = str(row["side"])
    param_path = Path(str(row["param_path"]))
    raw = json.loads(param_path.read_text(encoding="utf-8"))
    params = dict(raw["params"]) if isinstance(raw, dict) and isinstance(raw.get("params"), dict) else dict(raw)

    if side == "long":
        p = ga_long._norm_params(params)
        df_feat = ga_long._ensure_indicators(df_base.copy(), p)
        sig = ga_long.build_entry_signal(df_feat, p, assume_prepared=True)
        cycles = ga_long._shift_cycles(
            ga_long.compute_cycles(df_feat, p),
            shift=int(p.get("cycle_shift", 1)),
            fill=int(p.get("cycle_fill", 2)),
        )
    else:
        p = ga_short._norm_params(params)
        df_feat = ga_short._ensure_indicators(df_base.copy(), p)
        sig = ga_short.build_entry_signal_short(df_feat, p)
        cycles = ga_short._shift_cycles(
            ga_short.compute_cycles(df_feat, p),
            shift=int(p.get("cycle_shift", 1)),
            fill=int(p.get("cycle_fill", 2)),
        )

    ts = pd.to_datetime(df_feat["Timestamp"], utc=True, errors="coerce")
    ts_ns = ts.astype("int64").to_numpy()
    idx_by_ts = {int(v): int(i) for i, v in enumerate(ts_ns.tolist())}

    first_ts = pd.to_datetime(str(row.get("symbol_first_ts_utc", "")), utc=True, errors="coerce")
    first_ts_ns = int(first_ts.value) if pd.notna(first_ts) else int(ts_ns.min()) if len(ts_ns) else 0
    return SimModel(
        symbol=symbol,
        side=side,
        symbol_first_ts_ns=int(first_ts_ns),
        params=p,
        ts_ns=ts_ns,
        idx_by_ts=idx_by_ts,
        sig=np.asarray(sig, dtype=bool),
        cycles=np.asarray(cycles, dtype=np.int16),
        o=df_feat["Open"].astype(float).to_numpy(),
        h=df_feat["High"].astype(float).to_numpy(),
        l=df_feat["Low"].astype(float).to_numpy(),
        c=df_feat["Close"].astype(float).to_numpy(),
        atr_prev=df_feat["ATR"].astype(float).shift(1).fillna(0.0).to_numpy(),
        rsi_prev=df_feat["RSI"].astype(float).shift(1).fillna(50.0).to_numpy(),
    )


def _max_dd(equity: List[float]) -> float:
    if not equity:
        return 0.0
    arr = np.array(equity, dtype=float)
    runmax = np.maximum.accumulate(arr)
    dd = (runmax - arr) / np.maximum(runmax, 1e-9)
    return float(dd.max()) if dd.size else 0.0


def _run_universe_simulation(models: List[SimModel], args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim_path = out_dir / "universe_sim.csv"
    summary_path = out_dir / "universe_sim_summary.json"

    if not models:
        pd.DataFrame(columns=["timestamp", "chosen_symbol", "side", "action", "price", "equity"]).to_csv(sim_path, index=False)
        payload = {
            "final_equity": float(args.universe_initial_equity),
            "max_dd": 0.0,
            "trades": 0,
            "avg_hold_hours": 0.0,
            "exposure": 0.0,
            "ranking_formula": RANKING_FORMULA,
            "note": "No passed symbols; universe simulation skipped.",
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    timeline = pd.date_range(start=up._to_utc_ts(args.start), end=up._inclusive_end_ts(args.end), freq="1h", tz="UTC")
    ts_union = [int(ts.value) for ts in timeline]
    cash = float(args.universe_initial_equity)
    open_pos: Optional[Dict[str, Any]] = None

    trade_rows: List[Dict[str, Any]] = []
    sim_rows: List[Dict[str, Any]] = []
    equity_track: List[float] = []
    exposure_bars = 0

    for ts_ns in ts_union:
        ts = pd.to_datetime(ts_ns, utc=True)
        actions: List[str] = []
        chosen_symbol = ""
        chosen_side = ""
        action_price: Optional[float] = None

        # Exit first.
        if open_pos is not None:
            m = open_pos["model"]
            idx = m.idx_by_ts.get(ts_ns)
            if idx is not None:
                open_pos["last_idx"] = idx
                hold = int(idx - open_pos["entry_i"])
                reason: Optional[str] = None
                exit_exec_px: Optional[float] = None

                if m.side == "long":
                    tp_px = float(open_pos["entry_px"] * open_pos["tp_mult"])
                    sl_px = float(open_pos["entry_px"] * open_pos["sl_mult"])
                    hit_sl = bool(m.l[idx] <= sl_px)
                    hit_tp = bool(m.h[idx] >= tp_px)
                    if hit_sl and hit_tp:
                        reason, exit_exec_px = "sl", sl_px
                    elif hit_sl:
                        reason, exit_exec_px = "sl", sl_px
                    elif hit_tp:
                        reason, exit_exec_px = "tp", tp_px
                    elif hold >= int(m.params.get("max_hold_hours", 48)):
                        reason, exit_exec_px = "maxhold", float(m.o[idx])
                    else:
                        ex = float(m.params["exit_rsi_by_cycle"][int(open_pos["entry_cycle"])])
                        pnl_ratio = float(m.c[idx] / open_pos["entry_px"]) if open_pos["entry_px"] > 0 else 1.0
                        if float(m.rsi_prev[idx]) < ex and pnl_ratio > 1.0:
                            reason, exit_exec_px = "rsi_exit", float(m.o[idx])

                    if reason is not None and exit_exec_px is not None:
                        sell_px = ga_long._apply_cost(float(exit_exec_px), float(args.fee_bps), float(args.slip_bps), "sell")
                        proceeds = float(open_pos["units"] * sell_px)
                        cash += proceeds
                        net_pnl = float((sell_px - open_pos["entry_px"]) * open_pos["units"])
                        trade_rows.append(
                            {
                                "symbol": m.symbol,
                                "side": m.side,
                                "entry_ts": pd.to_datetime(open_pos["entry_ts_ns"], utc=True).isoformat(),
                                "exit_ts": ts.isoformat(),
                                "entry_px": float(open_pos["entry_px"]),
                                "exit_px": float(sell_px),
                                "units": float(open_pos["units"]),
                                "reason": reason,
                                "hold_hours": float(hold),
                                "net_pnl": float(net_pnl),
                                "rank_score": float(open_pos.get("rank_at_entry", 0.0)),
                            }
                        )
                        actions.append("exit_long")
                        chosen_symbol = m.symbol
                        chosen_side = m.side
                        action_price = float(sell_px)
                        open_pos = None
                else:
                    tp_px = float(open_pos["entry_px"] * open_pos["tp_mult"])
                    sl_px = float(open_pos["entry_px"] * open_pos["sl_mult"])
                    hit_tp = bool(m.l[idx] <= tp_px)
                    hit_sl = bool(m.h[idx] >= sl_px)
                    if hit_sl and hit_tp:
                        reason, exit_exec_px = "sl", sl_px
                    elif hit_sl:
                        reason, exit_exec_px = "sl", sl_px
                    elif hit_tp:
                        reason, exit_exec_px = "tp", tp_px
                    elif hold >= int(m.params.get("max_hold_hours", 48)):
                        reason, exit_exec_px = "maxhold", float(m.o[idx])
                    else:
                        ex = float(m.params["exit_rsi_by_cycle"][int(open_pos["entry_cycle"])])
                        pnl_ratio = (open_pos["entry_px"] / float(m.c[idx])) if (float(m.c[idx]) > 0 and open_pos["entry_px"] > 0) else 1.0
                        if float(m.rsi_prev[idx]) > ex and pnl_ratio > 1.0:
                            reason, exit_exec_px = "rsi_exit", float(m.o[idx])

                    if reason is not None and exit_exec_px is not None:
                        buy_px = ga_short._apply_cost(float(exit_exec_px), float(args.fee_bps), float(args.slip_bps), "buy")
                        cost = float(open_pos["units"] * buy_px)
                        cash -= cost
                        net_pnl = float((open_pos["entry_px"] - buy_px) * open_pos["units"])
                        trade_rows.append(
                            {
                                "symbol": m.symbol,
                                "side": m.side,
                                "entry_ts": pd.to_datetime(open_pos["entry_ts_ns"], utc=True).isoformat(),
                                "exit_ts": ts.isoformat(),
                                "entry_px": float(open_pos["entry_px"]),
                                "exit_px": float(buy_px),
                                "units": float(open_pos["units"]),
                                "reason": reason,
                                "hold_hours": float(hold),
                                "net_pnl": float(net_pnl),
                                "rank_score": float(open_pos.get("rank_at_entry", 0.0)),
                            }
                        )
                        actions.append("exit_short")
                        chosen_symbol = m.symbol
                        chosen_side = m.side
                        action_price = float(buy_px)
                        open_pos = None

        # Entry when flat.
        if open_pos is None:
            candidates: List[Tuple[float, str, str, int, SimModel]] = []
            for m in models:
                if ts_ns < int(m.symbol_first_ts_ns):
                    continue
                idx = m.idx_by_ts.get(ts_ns)
                if idx is None:
                    continue
                if idx >= len(m.sig):
                    continue
                if bool(m.sig[idx]):
                    candidates.append((_entry_rank(m, idx), m.symbol, m.side, int(idx), m))

            if candidates:
                # Deterministic tie-break: highest rank, then symbol, then side.
                candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
                rank, _sym, _side, idx, m = candidates[0]

                equity_now = float(cash)
                if m.side == "long":
                    buy_px = ga_long._apply_cost(float(m.o[idx]), float(args.fee_bps), float(args.slip_bps), "buy")
                    atrv = float(m.atr_prev[idx])
                    risk = float(m.params.get("risk_per_trade", 0.02))
                    max_alloc = float(m.params.get("max_allocation", 0.7))
                    atr_k = float(m.params.get("atr_k", 1.0))
                    cap = float(m.params.get("equity_sizing_cap", 0.0))
                    eq_for_size = float(min(equity_now, cap)) if cap > 0.0 else equity_now
                    units = float(ga_long._position_size(eq_for_size, buy_px, atrv, risk, max_alloc, atr_k))
                    if units > 0.0:
                        cost = float(units * buy_px)
                        if cost > cash:
                            units = float(cash / buy_px)
                            cost = float(units * buy_px)
                        if units > 0.0 and cost > 0.0:
                            cash -= cost
                            entry_cycle = int(m.cycles[idx])
                            open_pos = {
                                "model": m,
                                "entry_ts_ns": ts_ns,
                                "entry_i": int(idx),
                                "entry_cycle": int(entry_cycle),
                                "entry_px": float(buy_px),
                                "units": float(units),
                                "tp_mult": float(m.params["tp_mult_by_cycle"][entry_cycle]),
                                "sl_mult": float(m.params["sl_mult_by_cycle"][entry_cycle]),
                                "last_idx": int(idx),
                                "rank_at_entry": float(rank),
                            }
                            actions.append("enter_long")
                            chosen_symbol = m.symbol
                            chosen_side = m.side
                            action_price = float(buy_px)
                else:
                    sell_px = ga_short._apply_cost(float(m.o[idx]), float(args.fee_bps), float(args.slip_bps), "sell")
                    atrv = float(m.atr_prev[idx])
                    risk = float(m.params.get("risk_per_trade", 0.02))
                    max_alloc = float(m.params.get("max_allocation", 0.7))
                    atr_k = float(m.params.get("atr_k", 1.0))
                    units = float(ga_short._position_size_short(equity_now, sell_px, atrv, risk, max_alloc, atr_k))
                    if units > 0.0:
                        proceeds = float(units * sell_px)
                        cash += proceeds
                        entry_cycle = int(m.cycles[idx])
                        open_pos = {
                            "model": m,
                            "entry_ts_ns": ts_ns,
                            "entry_i": int(idx),
                            "entry_cycle": int(entry_cycle),
                            "entry_px": float(sell_px),
                            "units": float(units),
                            "tp_mult": float(m.params["tp_mult_by_cycle"][entry_cycle]),
                            "sl_mult": float(m.params["sl_mult_by_cycle"][entry_cycle]),
                            "last_idx": int(idx),
                            "rank_at_entry": float(rank),
                        }
                        actions.append("enter_short")
                        chosen_symbol = m.symbol
                        chosen_side = m.side
                        action_price = float(sell_px)

        # Mark-to-market equity.
        if open_pos is None:
            eq_now = float(cash)
        else:
            exposure_bars += 1
            m = open_pos["model"]
            idx = m.idx_by_ts.get(ts_ns, int(open_pos.get("last_idx", open_pos["entry_i"])))
            open_pos["last_idx"] = int(idx)
            mid = float(m.c[idx])
            if m.side == "long":
                eq_now = float(cash + open_pos["units"] * mid)
            else:
                eq_now = float(cash - open_pos["units"] * mid)
        equity_track.append(float(eq_now))
        sim_rows.append(
            {
                "timestamp": ts.isoformat(),
                "chosen_symbol": chosen_symbol,
                "side": chosen_side,
                "action": "+".join(actions) if actions else "hold",
                "price": float(action_price) if action_price is not None else "",
                "equity": float(eq_now),
            }
        )

    # Force close at end.
    if open_pos is not None:
        m = open_pos["model"]
        idx = int(open_pos.get("last_idx", open_pos["entry_i"]))
        final_ts = pd.to_datetime(int(m.ts_ns[idx]), utc=True)
        if m.side == "long":
            sell_px = ga_long._apply_cost(float(m.c[idx]), float(args.fee_bps), float(args.slip_bps), "sell")
            cash += float(open_pos["units"] * sell_px)
            net_pnl = float((sell_px - open_pos["entry_px"]) * open_pos["units"])
        else:
            buy_px = ga_short._apply_cost(float(m.c[idx]), float(args.fee_bps), float(args.slip_bps), "buy")
            cash -= float(open_pos["units"] * buy_px)
            sell_px = buy_px
            net_pnl = float((open_pos["entry_px"] - buy_px) * open_pos["units"])
        trade_rows.append(
            {
                "symbol": m.symbol,
                "side": m.side,
                "entry_ts": pd.to_datetime(open_pos["entry_ts_ns"], utc=True).isoformat(),
                "exit_ts": final_ts.isoformat(),
                "entry_px": float(open_pos["entry_px"]),
                "exit_px": float(sell_px),
                "units": float(open_pos["units"]),
                "reason": "eod",
                "hold_hours": float(max(0, idx - open_pos["entry_i"])),
                "net_pnl": float(net_pnl),
                "rank_score": float(open_pos.get("rank_at_entry", 0.0)),
            }
        )
        sim_rows.append(
            {
                "timestamp": final_ts.isoformat(),
                "chosen_symbol": m.symbol,
                "side": m.side,
                "action": "exit_eod",
                "price": float(sell_px),
                "equity": float(cash),
            }
        )
        equity_track.append(float(cash))

    pd.DataFrame(sim_rows).to_csv(sim_path, index=False)
    trades_df = pd.DataFrame(trade_rows)
    avg_hold = float(trades_df["hold_hours"].mean()) if ("hold_hours" in trades_df.columns and not trades_df.empty) else 0.0
    exposure = float(exposure_bars / max(1, len(ts_union)))
    summary = {
        "final_equity": float(cash),
        "max_dd": float(_max_dd(equity_track)),
        "trades": int(len(trades_df)),
        "avg_hold_hours": avg_hold,
        "exposure": exposure,
        "ranking_formula": RANKING_FORMULA,
        "selected_models": [
            {"symbol": m.symbol, "side": m.side}
            for m in sorted(models, key=lambda x: (x.symbol, x.side))
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


_CYCLE_KEYS = ["willr_by_cycle", "tp_mult_by_cycle", "sl_mult_by_cycle", "exit_rsi_by_cycle"]


def _merge_c13_params(side: str, params_c1: Dict[str, Any], params_c3: Dict[str, Any]) -> Dict[str, Any]:
    norm = ga_long._norm_params if side == "long" else ga_short._norm_params
    p1 = norm(dict(params_c1))
    p3 = norm(dict(params_c3))
    out = dict(p3)
    for key in _CYCLE_KEYS:
        a = list(p1.get(key, []))
        b = list(p3.get(key, []))
        if len(a) == 5 and len(b) == 5:
            b[1] = float(a[1])
            out[key] = b
    out["trade_cycles"] = [1, 3]
    out["require_trade_cycles"] = True
    out["cycle_shift"] = 1
    out["cycle_fill"] = 2
    return norm(out)


def _run_side_backtest_with_params(
    *,
    side: str,
    symbol: str,
    df: pd.DataFrame,
    params: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if side == "long":
        _, m = ga_long.run_backtest_long_only(
            df,
            symbol=symbol,
            p=ga_long._norm_params(params),
            initial_equity=float(args.initial_equity),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            collect_trades=False,
        )
        return dict(m)
    _, m = ga_short.run_backtest_short_only(
        df,
        symbol=symbol,
        p=ga_short._norm_params(params),
        initial_equity=float(args.initial_equity),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
    )
    return dict(m)


def _trades_gate_value(
    *,
    side: str,
    symbol: str,
    df_test: pd.DataFrame,
    merged_params: Dict[str, Any],
    c13_metrics: Dict[str, Any],
    trades_min: float,
    args: argparse.Namespace,
) -> Tuple[float, str, float]:
    c13_trades = float(c13_metrics.get("trades", 0.0))
    p2 = dict(merged_params)
    p2["trade_cycles"] = [2]
    p2["require_trade_cycles"] = True
    c2_metrics = _run_side_backtest_with_params(side=side, symbol=symbol, df=df_test, params=p2, args=args)
    c2_trades = float(c2_metrics.get("trades", 0.0))
    if c13_trades >= float(trades_min):
        return c13_trades, "c13_combined", c2_trades
    if c2_trades >= float(trades_min):
        return c2_trades, "cycle2_fallback", c2_trades
    return c13_trades, "c13_combined", c2_trades


def _zero_gen_reason(run_dir: Path, cfg_generations: Optional[int], requested_generations: int) -> str:
    if cfg_generations is not None and int(cfg_generations) != int(requested_generations):
        return "wrong_entrypoint_cfg_generations_mismatch"
    if run_dir.exists() and (run_dir / "final_report.json").exists():
        return "resume_bug_or_early_exit_before_gen1"
    if run_dir.exists():
        return "ga_crash_or_early_exit_before_gen1"
    return "ga_run_dir_missing"


def _append_existing_c13_rows(
    *,
    include_symbols: List[str],
    args: argparse.Namespace,
    pipeline_run_id: str,
    out_dir: Path,
    fetch_args: SimpleNamespace,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    data_by_symbol: Dict[str, pd.DataFrame],
    rows: List[Dict[str, Any]],
) -> None:
    existing_keys = {(str(r.get("symbol", "")), str(r.get("side", ""))) for r in rows}
    for sym in include_symbols:
        if (sym, "long") in existing_keys and (sym, "short") in existing_keys:
            continue

        symbol_first_ts = ""
        try:
            df = data_by_symbol.get(sym)
            if df is None:
                df = _prepare_symbol_df(sym, args, fetch_args, start_ts, end_ts)
                data_by_symbol[sym] = df
            symbol_first_ts = pd.to_datetime(df["Timestamp"].min(), utc=True).isoformat()
            df_train, df_test = up._build_train_test(df, args.test_start, args.test_end)
        except Exception as ex:
            reason = f"include_existing_data_prep_failed:{type(ex).__name__}"
            _log(f"{sym}: include existing data prep failed ({type(ex).__name__}: {ex})")
            for side in ("long", "short"):
                if (sym, side) in existing_keys:
                    continue
                row = _base_row(sym, side, symbol_first_ts, pipeline_run_id)
                row["fail_reasons"] = reason
                row["updated_utc"] = _utc_now_iso()
                rows.append(row)
                existing_keys.add((sym, side))
                _write_side_ga_report(
                    out_dir / sym / f"{side}_ga_report.json",
                    {
                        "configured_generations": int(args.ga_generations),
                        "executed_generations": 0,
                        "early_stop_triggered": False,
                        "best_score": None,
                        "test_net": 0.0,
                        "test_pf": 0.0,
                        "test_dd": 1.0,
                        "test_trades": 0.0,
                        "stability": 0.0,
                        "PASS/FAIL": "FAIL",
                        "fail_reasons": [reason],
                        "param_path": "",
                        "symbol_first_ts_utc": symbol_first_ts,
                        "source": "existing_c13_param",
                    },
                )
            continue

        for side in ("long", "short"):
            if (sym, side) in existing_keys:
                continue

            row = _base_row(sym, side, symbol_first_ts, pipeline_run_id)
            row["executed_generations"] = 0
            reasons: List[str] = []
            trades_gate_source = "c13_combined"
            c13_trades = 0.0
            c2_trades = 0.0
            param_fp = PROJECT_ROOT / "data" / "metadata" / "params" / f"{sym}_C13_active_params_{side}.json"

            if not param_fp.exists():
                reasons = [f"include_existing_param_missing:{param_fp.name}"]
            else:
                try:
                    raw = json.loads(param_fp.read_text(encoding="utf-8"))
                    params = dict(raw["params"]) if isinstance(raw, dict) and isinstance(raw.get("params"), dict) else dict(raw)
                    if side == "long":
                        p = ga_long._norm_params(params)
                        stability = up._long_stability(df_train, p, _long_cfg(args))
                    else:
                        p = ga_short._norm_params(params)
                        stability = up._short_stability(df_train, p, _short_cfg(args))
                    test_metrics = _run_side_backtest_with_params(
                        side=side,
                        symbol=sym,
                        df=df_test,
                        params=p,
                        args=args,
                    )
                    trades_gate, trades_gate_source, c2_trades = _trades_gate_value(
                        side=side,
                        symbol=sym,
                        df_test=df_test,
                        merged_params=p,
                        c13_metrics=test_metrics,
                        trades_min=float(args.pass_trades_min),
                        args=args,
                    )
                    c13_trades = float(test_metrics.get("trades", 0.0))
                    row["test_net"] = float(test_metrics.get("net_profit", 0.0))
                    row["test_pf"] = float(test_metrics.get("profit_factor", 0.0))
                    row["test_dd"] = float(test_metrics.get("max_dd", 1.0))
                    row["test_trades"] = float(trades_gate)
                    row["stability"] = float(stability)
                    row["param_path"] = str(param_fp.resolve())
                    row["trades_gate_source"] = str(trades_gate_source)
                    row["test_trades_c13"] = float(c13_trades)
                    row["test_trades_c2"] = float(c2_trades)

                    pass_fail, reasons = _pass_fail(
                        test_pf=float(row["test_pf"]),
                        test_dd=float(row["test_dd"]),
                        test_trades=float(row["test_trades"]),
                        stability=float(row["stability"]),
                        pf_min=float(args.pass_pf_min),
                        dd_max=float(args.pass_dd_max),
                        trades_min=float(args.pass_trades_min),
                        stability_min=float(args.pass_stability_min),
                    )
                    if float(trades_gate) < float(args.pass_trades_min):
                        reasons = [r for r in reasons if not str(r).startswith("trades<")]
                        reasons.append(
                            f"trades<{args.pass_trades_min}(c13={c13_trades:.1f},c2={float(c2_trades):.1f})"
                        )
                    row["PASS/FAIL"] = pass_fail if not reasons else "FAIL"
                except Exception as ex:
                    reasons = [f"include_existing_eval_failed:{type(ex).__name__}"]
                    _log(f"{sym} {side}: include existing eval failed ({type(ex).__name__}: {ex})")

            if reasons:
                row["PASS/FAIL"] = "FAIL"
                row["fail_reasons"] = ",".join(reasons)
            row["updated_utc"] = _utc_now_iso()
            rows.append(row)
            existing_keys.add((sym, side))
            _write_side_ga_report(
                out_dir / sym / f"{side}_ga_report.json",
                {
                    "configured_generations": int(args.ga_generations),
                    "executed_generations": int(row["executed_generations"]),
                    "early_stop_triggered": False,
                    "best_score": None,
                    "test_net": float(row["test_net"]),
                    "test_pf": float(row["test_pf"]),
                    "test_dd": float(row["test_dd"]),
                    "test_trades": float(row["test_trades"]),
                    "stability": float(row["stability"]),
                    "PASS/FAIL": str(row["PASS/FAIL"]),
                    "fail_reasons": [x for x in str(row["fail_reasons"]).split(",") if x],
                    "param_path": str(row["param_path"]),
                    "symbol_first_ts_utc": symbol_first_ts,
                    "trades_gate_source": str(row.get("trades_gate_source", trades_gate_source)),
                    "test_trades_c13": float(row.get("test_trades_c13", c13_trades)),
                    "test_trades_c2": float(row.get("test_trades_c2", c2_trades)),
                    "source": "existing_c13_param",
                },
            )
            _log(
                f"{sym} {side}: include_existing_c13 {row['PASS/FAIL']} "
                f"pf={row['test_pf']:.3f} dd={row['test_dd']:.3f} trades={row['test_trades']:.1f} "
                f"stability={row['stability']:.3f}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="", help="Optional CSV symbols fallback, used only if sources are missing")
    ap.add_argument("--skip-symbols", default=",".join(sorted(SKIP_SYMBOLS_DEFAULT)))
    ap.add_argument(
        "--include-existing-c13",
        default="",
        help="CSV symbols to include from existing *_C13_active_params_{long,short}.json without GA re-optimization",
    )
    ap.add_argument("--max-symbols", type=int, default=0, help="Optional limit for selected symbols after filtering")

    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--test-end", default="2025-12-31")

    ap.add_argument("--gens", "--ga-generations", dest="ga_generations", type=int, default=40)
    ap.add_argument("--early-stop", type=int, default=10)
    ap.add_argument("--eval-procs", type=int, default=3)
    ap.add_argument("--output-dir", default="")

    ap.add_argument("--pop-size", type=int, default=8)
    ap.add_argument("--mc-splits", type=int, default=3)
    ap.add_argument("--train-days", type=int, default=540)
    ap.add_argument("--val-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--min-trades-train", type=int, default=40)
    ap.add_argument("--min-trades-val", type=int, default=15)

    ap.add_argument("--pass-pf-min", type=float, default=1.5)
    ap.add_argument("--pass-dd-max", type=float, default=0.20)
    ap.add_argument("--pass-trades-min", type=float, default=30.0)
    ap.add_argument("--pass-stability-min", type=float, default=0.75)

    ap.add_argument("--backtest-initial-equity", type=float, default=100.0)
    ap.add_argument("--universe-initial-equity", type=float, default=350.0)

    ap.add_argument("--http-retries", type=int, default=8)
    ap.add_argument("--http-retry-base-sleep", type=float, default=0.5)
    ap.add_argument("--http-retry-max-sleep", type=float, default=30.0)
    ap.add_argument("--offline-fallback", dest="offline_fallback", action="store_true", default=True)
    ap.add_argument("--no-offline-fallback", dest="offline_fallback", action="store_false")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--local-only", dest="local_only", action="store_true", default=True)
    ap.add_argument("--no-local-only", dest="local_only", action="store_false")
    ap.add_argument("--skip-housekeeping", action="store_true")
    args = ap.parse_args()

    _apply_thread_caps()
    pipeline_run_id = _safe_run_id()
    out_dir = Path(args.output_dir) if str(args.output_dir).strip() else (PROJECT_ROOT / "artifacts" / "reports" / f"universe_{pipeline_run_id}")
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _log(f"run_id={pipeline_run_id}")
    _log(f"output_dir={out_dir}")

    symbols_raw, symbol_source = _resolve_symbols(args)
    skip_symbols = _parse_skip_symbols(args)
    symbols = [s for s in symbols_raw if s not in skip_symbols]
    if not symbols and symbol_source.startswith("artifacts_reports_summary:"):
        alt, alt_src = _symbols_from_metadata_universe()
        if alt:
            symbols = [s for s in alt if s not in skip_symbols]
            symbol_source = f"metadata_universe:{alt_src} (fallback_after_skip)"
    if not symbols:
        alt = _symbols_from_cli(args)
        if alt:
            symbols = [s for s in alt if s not in skip_symbols]
            symbol_source = "cli_symbols (fallback_after_skip)"
    if not symbols:
        raise SystemExit(f"No symbols left after skip set={sorted(skip_symbols)} from source={symbol_source}")
    if int(args.max_symbols) > 0:
        symbols = symbols[: int(args.max_symbols)]
    _log(f"symbols_source={symbol_source}")
    _log(f"symbols_selected={len(symbols)} symbols={symbols}")
    _log(
        "pass_criteria="
        f"pf>={args.pass_pf_min} dd<={args.pass_dd_max} "
        f"trades>={args.pass_trades_min} stability>={args.pass_stability_min}"
    )

    if not args.skip_housekeeping:
        killed = _stop_repo_processes(max_age_sec=24 * 3600)
        removed = _cleanup_recent_artifacts(exclude=[out_dir])
        for rec in killed:
            _log(f"housekeeping_stopped pid={rec['pid']} age_sec={rec['etimes_sec']} cmd={rec['cmd']}")
        for p in removed:
            _log(f"housekeeping_deleted {p}")
        _log(f"housekeeping_summary stopped={len(killed)} deleted={len(removed)}")
    else:
        _log("housekeeping: skipped by flag")

    fetch_args = _make_fetch_args(args)
    start_ts = up._to_utc_ts(args.start)
    end_ts = up._inclusive_end_ts(args.end)
    rows: List[Dict[str, Any]] = []
    data_by_symbol: Dict[str, pd.DataFrame] = {}
    ga_runs_total = 0
    ga_runs_with_gens = 0

    for sym in symbols:
        _log(f"{sym}: phase2 prepare 1h data + indicators")
        try:
            df = _prepare_symbol_df(sym, args, fetch_args, start_ts, end_ts)
        except Exception as ex:
            _log(f"{sym}: data prep failed ({type(ex).__name__}: {ex})")
            for side in ("long", "short"):
                row = _base_row(sym, side, "", pipeline_run_id)
                row["fail_reasons"] = f"data_fetch_failed:{type(ex).__name__}"
                rows.append(row)
                _write_side_ga_report(
                    out_dir / sym / f"{side}_ga_report.json",
                    {
                        "configured_generations": int(args.ga_generations),
                        "executed_generations": 0,
                        "early_stop_triggered": False,
                        "best_score": None,
                        "test_net": 0.0,
                        "test_pf": 0.0,
                        "test_dd": 1.0,
                        "test_trades": 0.0,
                        "stability": 0.0,
                        "PASS/FAIL": "FAIL",
                        "fail_reasons": [row["fail_reasons"]],
                        "param_path": "",
                        "symbol_first_ts_utc": "",
                    },
                )
            continue

        symbol_first_ts = pd.to_datetime(df["Timestamp"].min(), utc=True).isoformat()
        data_by_symbol[sym] = df
        try:
            df_train, df_test = up._build_train_test(df, args.test_start, args.test_end)
        except Exception as ex:
            _log(f"{sym}: split failed ({type(ex).__name__}: {ex})")
            for side in ("long", "short"):
                row = _base_row(sym, side, symbol_first_ts, pipeline_run_id)
                row["fail_reasons"] = f"data_prep_error:{type(ex).__name__}"
                rows.append(row)
                _write_side_ga_report(
                    out_dir / sym / f"{side}_ga_report.json",
                    {
                        "configured_generations": int(args.ga_generations),
                        "executed_generations": 0,
                        "early_stop_triggered": False,
                        "best_score": None,
                        "test_net": 0.0,
                        "test_pf": 0.0,
                        "test_dd": 1.0,
                        "test_trades": 0.0,
                        "stability": 0.0,
                        "PASS/FAIL": "FAIL",
                        "fail_reasons": [row["fail_reasons"]],
                        "param_path": "",
                        "symbol_first_ts_utc": symbol_first_ts,
                    },
                )
            continue

        for side in ("long", "short"):
            row = _base_row(sym, side, symbol_first_ts, pipeline_run_id)
            cycle_params: Dict[int, Dict[str, Any]] = {}
            cycle_details: List[Dict[str, Any]] = []
            generation_reasons: List[str] = []
            executed_total = 0
            best_score: Optional[float] = None
            early_stop_any = False

            for cycle in (1, 3):
                ga_runs_total += 1
                ga_symbol = f"{sym}__UNIVERSE_{side.upper()}_C{cycle}"
                try:
                    if side == "long":
                        cfg = _long_cfg(args)
                        seed = up._load_long_seed(sym)
                        seed["trade_cycles"] = [int(cycle)]
                        seed["require_trade_cycles"] = True
                        best_params, report = ga_long.run_ga_montecarlo(ga_symbol, df_train, ga_long._norm_params(seed), cfg)
                        best_params = ga_long._norm_params(best_params)
                    else:
                        cfg = _short_cfg(args)
                        seed = up._load_short_seed(sym)
                        seed["trade_cycles"] = [int(cycle)]
                        seed["require_trade_cycles"] = True
                        best_params, report = ga_short.run_ga_montecarlo(ga_symbol, df_train, ga_short._norm_params(seed), cfg)
                        best_params = ga_short._norm_params(best_params)

                    run_dir = Path(str(report.get("saved", {}).get("run_dir", "")))
                    gens_ran = _count_generations_ran(run_dir) if run_dir.exists() else 0
                    cfg_generations = _read_generations_target_from_final(run_dir) if run_dir.exists() else None
                    early_stop_c = bool(gens_ran > 0 and gens_ran < int(args.ga_generations))
                    score_c = None
                    if isinstance(report.get("best_overall"), dict) and report.get("best_overall", {}).get("fitness") is not None:
                        score_c = float(report["best_overall"]["fitness"])

                    cycle_details.append(
                        {
                            "cycle": int(cycle),
                            "ga_symbol": ga_symbol,
                            "ga_run_id": str(report.get("run_id", "")),
                            "run_dir": str(run_dir),
                            "executed_generations": int(gens_ran),
                            "configured_generations": int(args.ga_generations),
                            "early_stop_triggered": bool(early_stop_c),
                            "best_score": score_c,
                        }
                    )
                    executed_total += int(gens_ran)
                    early_stop_any = bool(early_stop_any or early_stop_c)
                    cycle_params[int(cycle)] = best_params
                    if score_c is not None:
                        best_score = score_c if best_score is None else max(best_score, score_c)
                    if int(gens_ran) > 0:
                        ga_runs_with_gens += 1
                    else:
                        generation_reasons.append(f"{_zero_gen_reason(run_dir, cfg_generations, int(args.ga_generations))}:cycle{cycle}")
                    _log(f"{sym} {side} cycle{cycle}: executed_generations={gens_ran}/{args.ga_generations}")
                except Exception as ex:
                    generation_reasons.append(f"ga_crash:{type(ex).__name__}:cycle{cycle}")
                    cycle_details.append(
                        {
                            "cycle": int(cycle),
                            "ga_symbol": ga_symbol,
                            "ga_run_id": "",
                            "run_dir": "",
                            "executed_generations": 0,
                            "configured_generations": int(args.ga_generations),
                            "early_stop_triggered": False,
                            "best_score": None,
                            "error": f"{type(ex).__name__}: {ex}",
                        }
                    )
                    _log(f"{sym} {side} cycle{cycle}: optimization failed ({type(ex).__name__}: {ex})")

            if 1 in cycle_params and 3 in cycle_params:
                merged_params = _merge_c13_params(side, cycle_params[1], cycle_params[3])
                if side == "long":
                    stability = up._long_stability(df_train, merged_params, _long_cfg(args))
                else:
                    stability = up._short_stability(df_train, merged_params, _short_cfg(args))
                test_metrics = _run_side_backtest_with_params(
                    side=side,
                    symbol=sym,
                    df=df_test,
                    params=merged_params,
                    args=args,
                )
                trades_gate, trades_gate_source, c2_trades = _trades_gate_value(
                    side=side,
                    symbol=sym,
                    df_test=df_test,
                    merged_params=merged_params,
                    c13_metrics=test_metrics,
                    trades_min=float(args.pass_trades_min),
                    args=args,
                )
                c13_trades = float(test_metrics.get("trades", 0.0))
                row["test_net"] = float(test_metrics.get("net_profit", 0.0))
                row["test_pf"] = float(test_metrics.get("profit_factor", 0.0))
                row["test_dd"] = float(test_metrics.get("max_dd", 1.0))
                row["test_trades"] = float(trades_gate)
                row["stability"] = float(stability)
                row["executed_generations"] = int(executed_total)

                pass_fail, reasons = _pass_fail(
                    test_pf=float(row["test_pf"]),
                    test_dd=float(row["test_dd"]),
                    test_trades=float(row["test_trades"]),
                    stability=float(row["stability"]),
                    pf_min=float(args.pass_pf_min),
                    dd_max=float(args.pass_dd_max),
                    trades_min=float(args.pass_trades_min),
                    stability_min=float(args.pass_stability_min),
                )
                if float(trades_gate) < float(args.pass_trades_min):
                    reasons = [r for r in reasons if not str(r).startswith("trades<")]
                    reasons.append(
                        f"trades<{args.pass_trades_min}(c13={c13_trades:.1f},c2={float(c2_trades):.1f})"
                    )
                reasons.extend(generation_reasons)
                if reasons:
                    pass_fail = "FAIL"

                param_path = _save_active_side_params(
                    symbol=sym,
                    side=side,
                    params=merged_params,
                    pipeline_run_id=pipeline_run_id,
                    ga_report={"run_id": pipeline_run_id, "symbol": f"{sym}__UNIVERSE_{side.upper()}_C13", "saved": {"run_dir": ""}},
                    test_metrics=test_metrics,
                    stability=float(stability),
                    pass_fail=pass_fail,
                    fail_reasons=reasons,
                    generations_target=int(args.ga_generations),
                    generations_ran=int(executed_total),
                    early_stop=bool(early_stop_any),
                )
                row["PASS/FAIL"] = pass_fail
                row["fail_reasons"] = ",".join(reasons)
                row["param_path"] = str(param_path.resolve())
                row["trades_gate_source"] = str(trades_gate_source)
                row["test_trades_c13"] = float(c13_trades)
                row["test_trades_c2"] = float(c2_trades)
            else:
                row["executed_generations"] = int(executed_total)
                all_reasons = generation_reasons + ["missing_cycle_params_for_c13"]
                row["PASS/FAIL"] = "FAIL"
                row["fail_reasons"] = ",".join(all_reasons)

            row["updated_utc"] = _utc_now_iso()
            rows.append(row)

            _write_side_ga_report(
                out_dir / sym / f"{side}_ga_report.json",
                {
                    "configured_generations": int(args.ga_generations),
                    "executed_generations": int(row["executed_generations"]),
                    "early_stop_triggered": bool(early_stop_any),
                    "best_score": best_score,
                    "test_net": float(row["test_net"]),
                    "test_pf": float(row["test_pf"]),
                    "test_dd": float(row["test_dd"]),
                    "test_trades": float(row["test_trades"]),
                    "stability": float(row["stability"]),
                    "PASS/FAIL": str(row["PASS/FAIL"]),
                    "fail_reasons": [x for x in str(row["fail_reasons"]).split(",") if x],
                    "param_path": str(row["param_path"]),
                    "symbol_first_ts_utc": symbol_first_ts,
                    "cycle_details": cycle_details,
                    "trades_gate_source": str(row.get("trades_gate_source", "c13_combined")),
                    "test_trades_c13": float(row.get("test_trades_c13", row["test_trades"])),
                    "test_trades_c2": float(row.get("test_trades_c2", 0.0)),
                },
            )
            _log(
                f"{sym} {side}: {row['PASS/FAIL']} executed_generations={row['executed_generations']} "
                f"pf={row['test_pf']:.3f} dd={row['test_dd']:.3f} trades={row['test_trades']:.1f} "
                f"stability={row['stability']:.3f}"
            )

    include_existing = _csv_symbols(str(args.include_existing_c13))
    if include_existing:
        _log(f"including_existing_c13 symbols={include_existing}")
        _append_existing_c13_rows(
            include_symbols=include_existing,
            args=args,
            pipeline_run_id=pipeline_run_id,
            out_dir=out_dir,
            fetch_args=fetch_args,
            start_ts=start_ts,
            end_ts=end_ts,
            data_by_symbol=data_by_symbol,
            rows=rows,
        )

    meta = {
        "run_id": pipeline_run_id,
        "generated_utc": _utc_now_iso(),
        "start": str(args.start),
        "end": str(args.end),
        "test_start": str(args.test_start),
        "test_end": str(args.test_end),
        "ga_generations": int(args.ga_generations),
        "early_stop": int(args.early_stop),
        "symbols_count": len(symbols),
        "symbol_source": symbol_source,
        "skip_symbols": sorted(skip_symbols),
        "include_existing_c13": include_existing,
        "pass_criteria": {
            "pf_min": float(args.pass_pf_min),
            "dd_max": float(args.pass_dd_max),
            "trades_min": float(args.pass_trades_min),
            "stability_min": float(args.pass_stability_min),
        },
        "ranking_formula": RANKING_FORMULA,
    }
    summary_csv, summary_json = _write_summary(out_dir, rows, meta)
    _log(f"summary_written csv={summary_csv} json={summary_json}")
    _log(f"ga_generation_proof runs_with_executed_generations_gt0={ga_runs_with_gens} total_ga_runs={ga_runs_total}")
    if ga_runs_with_gens <= 0:
        raise RuntimeError("validation_failed:no_ga_run_executed_generations_gt_0")

    passed_rows = [r for r in rows if str(r.get("PASS/FAIL", "")) == "PASS" and str(r.get("param_path", "")).strip()]
    _log(f"passed_rows={len(passed_rows)}")

    for r in passed_rows:
        sym = str(r["symbol"])
        side = str(r["side"])
        if sym not in data_by_symbol:
            continue
        df = data_by_symbol[sym].copy()
        fp = Path(str(r["param_path"]))
        raw = json.loads(fp.read_text(encoding="utf-8"))
        params = dict(raw["params"]) if isinstance(raw, dict) and isinstance(raw.get("params"), dict) else dict(raw)
        bt_start = max(up._to_utc_ts(args.start), pd.to_datetime(str(r["symbol_first_ts_utc"]), utc=True))
        bt_end = up._inclusive_end_ts(args.end)
        df = df[(df["Timestamp"] >= bt_start) & (df["Timestamp"] <= bt_end)].reset_index(drop=True)
        if df.empty:
            continue

        if side == "long":
            p = ga_long._norm_params(params)
            df = ga_long._ensure_indicators(df, p)
            trades, m = ga_long.run_backtest_long_only(
                df,
                symbol=sym,
                p=p,
                initial_equity=float(args.backtest_initial_equity),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slip_bps),
                collect_trades=True,
                assume_prepared=True,
            )
        else:
            p = ga_short._norm_params(params)
            df = ga_short._ensure_indicators(df, p)
            trades, m = ga_short.run_backtest_short_only(
                df,
                symbol=sym,
                p=p,
                initial_equity=float(args.backtest_initial_equity),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slip_bps),
            )

        trades_df = pd.DataFrame(trades)
        yearly = _yearly_breakdown(trades_df)
        report_path = out_dir / sym / f"{side}_backtest_summary.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "final_equity": float(m.get("final_equity", args.backtest_initial_equity)),
            "net_profit": float(m.get("net_profit", 0.0)),
            "trades": int(float(m.get("trades", 0.0))),
            "win_rate": float(m.get("win_rate_pct", 0.0)),
            "pf": float(m.get("profit_factor", 0.0)),
            "max_dd": float(m.get("max_dd", 0.0)),
            "yearly_breakdown": yearly.to_dict("records"),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        _log(f"backtest_written {report_path}")

    sim_models = []
    for r in passed_rows:
        sym = str(r["symbol"])
        if sym not in data_by_symbol:
            continue
        sim_models.append(_build_sim_model(r, data_by_symbol[sym]))
    universe_summary = _run_universe_simulation(sim_models, args, out_dir)
    _log(
        f"universe_done trades={universe_summary.get('trades', 0)} "
        f"final_equity={universe_summary.get('final_equity', 0.0):.4f} "
        f"max_dd={universe_summary.get('max_dd', 0.0):.4f}"
    )


if __name__ == "__main__":
    main()
