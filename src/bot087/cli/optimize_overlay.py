from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import logging
import multiprocessing
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.bot087.execution.execution_eval import (
    ExecutionEvalConfig,
    _alignment_check,
    _ensure_cache_for_windows,
    _load_1s_data,
    _metrics_from_pnl,
    _sec_arrays,
    _slice_idx,
    purge_in_memory_1s_cache,
    run_entry_overlay_backtest_from_df,
)
from src.bot087.optim.ga import (
    GAConfig,
    _apply_cost,
    _ensure_indicators,
    _norm_params,
    _position_size,
    _shift_cycles,
    compute_cycles,
    make_mc_splits,
    run_backtest_long_only,
)


def _detect_project_root() -> Path:
    env_root = os.getenv("BOT087_PROJECT_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    for base in [here, *here.parents]:
        if (base / "src" / "bot087").exists() or (base / "pyproject.toml").exists():
            return base.resolve()
    return Path.cwd().resolve()


PROJECT_ROOT = _detect_project_root()
_LOG = logging.getLogger("bot087.optimize_overlay")
_SORT_COLUMNS = ["valid", "median_val_score", "stability_pct", "median_val_dd", "median_val_pf"]
_SORT_ASCENDING = [False, False, False, True, False]
_CANDIDATE_KEYS = [
    "overlay_mode",
    "overlay_policy",
    "overlay_behavior",
    "overlay_window_sec",
    "pullback_dip_bps",
    "bounce_confirm_n",
    "break_bps",
    "adx_strong",
    "use_sep_bypass",
    "sep_k",
    "cap_mult",
]
_STATS_KEYS = [
    "cid",
    "valid",
    "invalid_reason",
    "median_val_score",
    "iqr_val_score",
    "median_val_net",
    "median_val_pf",
    "median_val_dd",
    "median_val_trades",
    "median_skip_rate",
    "median_edge_decay",
    "stability_pct",
    "pos_splits",
    "total_splits",
]
_TOPK_CAP = 200
_WINDOW_CHOICES = (10, 20, 30, 45, 60)
_BREAKOUT_LOOKBACK_SEC = 10


class _ContextDefaultsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key in ("symbol", "phase", "progress"):
            if not hasattr(record, key):
                setattr(record, key, "-")
        return True


class _UTCFormatter(logging.Formatter):
    converter = time.gmtime


def _configure_logging() -> None:
    fmt = "%(asctime)sZ %(levelname)s symbol=%(symbol)s phase=%(phase)s progress=%(progress)s %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_UTCFormatter(fmt=fmt, datefmt="%Y-%m-%dT%H:%M:%S"))
    handler.addFilter(_ContextDefaultsFilter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    mp_logger = multiprocessing.get_logger()
    mp_logger.handlers.clear()
    mp_logger.setLevel(logging.INFO)
    mp_logger.addHandler(handler)
    mp_logger.propagate = False


def _log_info(message: str, *, symbol: str = "-", phase: str = "-", progress: str = "-") -> None:
    _LOG.info(message, extra={"symbol": symbol, "phase": phase, "progress": progress})


def _print_line(message: str) -> None:
    print(message, flush=True)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:
        dfd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except Exception:
        pass


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or path.stat().st_size == 0
    cols = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _pack_py_random_state(state: Any) -> Dict[str, Any]:
    version, internal, gauss_next = state
    return {"version": int(version), "internal": list(internal), "gauss_next": gauss_next}


def _unpack_py_random_state(packed: Dict[str, Any]) -> Any:
    version = int(packed["version"])
    internal = packed["internal"]
    if isinstance(internal, list):
        internal = tuple(tuple(x) if isinstance(x, list) else x for x in internal)
    return (version, internal, packed.get("gauss_next", None))


def _pack_np_random_state(state: Tuple[Any, ...]) -> Dict[str, Any]:
    bitgen, keys, pos, has_gauss, cached = state
    return {
        "bitgen": str(bitgen),
        "keys": keys.tolist() if hasattr(keys, "tolist") else list(keys),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _unpack_np_random_state(packed: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        str(packed["bitgen"]),
        np.array(packed["keys"], dtype=np.uint32),
        int(packed["pos"]),
        int(packed["has_gauss"]),
        float(packed["cached_gaussian"]),
    )


@dataclass(frozen=True)
class SymbolPaths:
    search_out: Path
    best_out: Path
    test_out: Path
    status_out: Path
    ckpt_out: Path
    crash_out: Path
    partial_jsonl: Path
    random_csv: Path
    ga_csv: Path
    invalid_reasons_out: Path


@dataclass
class EvalProgress:
    symbol: str
    phase: str
    split_eval_count: int = 0
    split_eval_total: int = 0


@dataclass
class TopKBuffer:
    cap: int = _TOPK_CAP
    rows: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def _rank(self, row: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
        return (
            1.0 if bool(row.get("valid", False)) else 0.0,
            float(row.get("median_val_score", -1e15)),
            float(row.get("stability_pct", 0.0)),
            -float(row.get("median_val_dd", 1e9)),
            float(row.get("median_val_pf", 0.0)),
        )

    def add(self, row: Dict[str, Any]) -> None:
        cid = str(row.get("cid", "")).strip()
        if not cid:
            return
        prev = self.rows.get(cid)
        if prev is None or self._rank(row) > self._rank(prev):
            self.rows[cid] = dict(row)
        if len(self.rows) > self.cap * 2:
            self.trim()

    def extend(self, rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            if isinstance(row, dict):
                self.add(row)

    def trim(self) -> None:
        ranked = sorted(self.rows.values(), key=self._rank, reverse=True)[: self.cap]
        self.rows = {str(r["cid"]): r for r in ranked if "cid" in r}

    def sorted(self) -> List[Dict[str, Any]]:
        self.trim()
        return sorted(self.rows.values(), key=self._rank, reverse=True)

    def top(self, n: int) -> List[Dict[str, Any]]:
        return self.sorted()[: max(0, int(n))]


def _rss_mb() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))
    except Exception:
        try:
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
        except Exception:
            return 0.0
    return 0.0


def _total_ram_mb() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.virtual_memory().total / (1024 * 1024))
    except Exception:
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
        except Exception:
            return 0.0
    return 0.0


def _log_warn(message: str, *, symbol: str = "-", phase: str = "-", progress: str = "-") -> None:
    _LOG.warning(message, extra={"symbol": symbol, "phase": phase, "progress": progress})


def _memory_guard(args: argparse.Namespace, *, symbol: str, phase: str, progress: str) -> None:
    limit = float(getattr(args, "_effective_max_rss_mb", 0.0))
    if limit <= 0.0:
        return
    rss = _rss_mb()
    if rss <= 0.0:
        return
    if rss > limit:
        purged = purge_in_memory_1s_cache()
        gc.collect()
        rss_after = _rss_mb()
        _log_warn(
            f"RSS over limit rss_mb={rss:.1f} limit_mb={limit:.1f}; purged_1s_cache_entries={purged}; rss_after_mb={rss_after:.1f}",
            symbol=symbol,
            phase=phase,
            progress=progress,
        )


def _artifact_paths(symbol: str) -> SymbolPaths:
    sym = symbol.upper()
    reports = (PROJECT_ROOT / "artifacts" / "reports").resolve()
    return SymbolPaths(
        search_out=reports / f"overlay_ultra_search_{sym}.csv",
        best_out=reports / f"overlay_ultra_best_{sym}.json",
        test_out=reports / f"overlay_ultra_test_{sym}.json",
        status_out=reports / f"overlay_ultra_status_{sym}.json",
        ckpt_out=reports / f"overlay_ultra_ckpt_{sym}.json",
        crash_out=reports / f"overlay_ultra_crash_{sym}.txt",
        partial_jsonl=reports / f"overlay_ultra_partial_{sym}.jsonl",
        random_csv=reports / f"overlay_ultra_random_{sym}.csv",
        ga_csv=reports / f"overlay_ultra_ga_{sym}.csv",
        invalid_reasons_out=reports / f"invalid_reasons_{sym}.json",
    )


def _args_hash(args: argparse.Namespace, symbol: str) -> str:
    payload = {
        k: v
        for k, v in vars(args).items()
        if k not in {"symbols", "fresh"} and not str(k).startswith("_")
    }
    payload["symbol"] = symbol.upper()
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _candidate_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if any(k not in row for k in _CANDIDATE_KEYS):
        return None
    try:
        use_sep_raw = row["use_sep_bypass"]
        return {
            "overlay_mode": str(row["overlay_mode"]),
            "overlay_policy": str(row["overlay_policy"]),
            "overlay_behavior": str(row["overlay_behavior"]),
            "overlay_window_sec": int(row["overlay_window_sec"]),
            "pullback_dip_bps": int(row["pullback_dip_bps"]),
            "bounce_confirm_n": int(row["bounce_confirm_n"]),
            "break_bps": int(row["break_bps"]),
            "adx_strong": float(row["adx_strong"]),
            "use_sep_bypass": _parse_bool(use_sep_raw) if isinstance(use_sep_raw, str) else bool(use_sep_raw),
            "sep_k": float(row["sep_k"]),
            "cap_mult": int(row["cap_mult"]),
        }
    except Exception:
        return None


def _stats_from_row(row: Dict[str, Any]) -> Optional["CandidateStats"]:
    if any(k not in row for k in _STATS_KEYS):
        return None
    try:
        valid_raw = row["valid"]
        return CandidateStats(
            cid=str(row["cid"]),
            valid=_parse_bool(valid_raw) if isinstance(valid_raw, str) else bool(valid_raw),
            invalid_reason=str(row["invalid_reason"]),
            median_val_score=float(row["median_val_score"]),
            iqr_val_score=float(row["iqr_val_score"]),
            median_val_net=float(row["median_val_net"]),
            median_val_pf=float(row["median_val_pf"]),
            median_val_dd=float(row["median_val_dd"]),
            median_val_trades=float(row["median_val_trades"]),
            median_skip_rate=float(row["median_skip_rate"]),
            median_edge_decay=float(row["median_edge_decay"]),
            stability_pct=float(row["stability_pct"]),
            pos_splits=int(row["pos_splits"]),
            total_splits=int(row["total_splits"]),
        )
    except Exception:
        return None


def _sort_search_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    for col in _SORT_COLUMNS:
        if col not in df.columns:
            if col == "valid":
                df[col] = False
            else:
                df[col] = 0.0
    df = df.sort_values(_SORT_COLUMNS, ascending=_SORT_ASCENDING).reset_index(drop=True)
    return df.to_dict("records")


def _rank_candidates(cache: Dict[str, "CandidateStats"], cand_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for cid, st in cache.items():
        c = cand_map.get(cid)
        if c is None:
            continue
        rows.append({"cid": cid, **c, **st.__dict__})
    return _sort_search_rows(rows)


def _candidate_only(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _candidate_from_row(row)


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _parse_bool(v: Any) -> bool:
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


def _load_params(symbol: str, btc_params: str) -> Tuple[Dict[str, Any], Path]:
    if symbol.upper() == "BTCUSDT":
        ppath = (PROJECT_ROOT / btc_params).resolve()
    else:
        ppath = PROJECT_ROOT / "data" / "metadata" / "params" / f"{symbol.upper()}_active_params.json"
    raw = json.loads(ppath.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
        raw = raw["params"]
    p = _norm_params(dict(raw))
    p["cycle_shift"] = 1
    p["two_candle_confirm"] = False
    p["require_trade_cycles"] = True
    p["max_allocation"] = min(0.7, float(p.get("max_allocation", 0.7)))
    return p, ppath


def _load_df(symbol: str, start: str, end: str) -> pd.DataFrame:
    fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol.upper()}_1h_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Missing 1h parquet: {fp}")
    df = pd.read_parquet(fp)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp")
    df = df.drop_duplicates(subset=["Timestamp"], keep="last").reset_index(drop=True)
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    df = df[(df["Timestamp"] >= pd.Timestamp(start, tz="UTC")) & (df["Timestamp"] <= pd.Timestamp(end, tz="UTC"))]
    if df.empty:
        raise RuntimeError(f"{symbol}: empty frame after date filter")
    return df.reset_index(drop=True)


def _split_outer(df: pd.DataFrame, test_start: str, test_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if test_start.strip() and test_end.strip():
        ts0 = pd.Timestamp(test_start, tz="UTC")
        ts1 = pd.Timestamp(test_end, tz="UTC")
        test = df[(df["Timestamp"] >= ts0) & (df["Timestamp"] <= ts1)].reset_index(drop=True)
        pre = df[df["Timestamp"] < ts0].reset_index(drop=True)
    else:
        n = len(df)
        k = max(1, int(round(n * 0.20)))
        pre = df.iloc[:-k].reset_index(drop=True)
        test = df.iloc[-k:].reset_index(drop=True)
    if pre.empty or test.empty:
        raise RuntimeError("outer split failed: empty pre or test")
    return pre, test


def _candidate_id(c: Dict[str, Any]) -> str:
    s = json.dumps(c, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _overlay_cfg_from_candidate(c: Dict[str, Any], args: argparse.Namespace) -> ExecutionEvalConfig:
    behavior = str(c["overlay_behavior"])
    fallback = behavior == "fallback_to_open"
    return ExecutionEvalConfig(
        mode="klines1s",
        market="spot",
        window_sec=15,
        cache_cap_gb=float(args.cache_cap_gb),
        cap_gb=float(args.cache_cap_gb),
        cache_root=str((PROJECT_ROOT / args.cache_root).resolve()),
        fetch_workers=max(1, int(args.fetch_workers)),
        cache_1s=bool(args.cache_1s),
        max_cache_mb=float(args.max_cache_mb),
        alignment_max_gap_sec=2.0,
        alignment_open_tol_pct=0.01,
        overlay_mode=str(c["overlay_mode"]),
        overlay_window_sec=int(c["overlay_window_sec"]),
        overlay_pullback_dip_bps=float(c["pullback_dip_bps"]),
        overlay_bounce_confirm_n=int(c["bounce_confirm_n"]),
        overlay_breakout_bps=float(c["break_bps"]),
        overlay_breakout_lookback_sec=10,
        overlay_fallback_to_open=bool(fallback),
        overlay_skip_if_no_trigger=not bool(fallback),
        overlay_policy=str(c["overlay_policy"]),
        adx_strong=float(c["adx_strong"]),
        use_sep_bypass=bool(c["use_sep_bypass"]),
        sep_k=float(c["sep_k"]),
        cap_mult=float(c["cap_mult"]),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        initial_equity=float(args.initial_equity),
    )


def _sanitize_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(c)
    mode = str(out.get("overlay_mode", "pullback")).lower()
    if mode not in {"pullback", "breakout"}:
        mode = "pullback"
    out["overlay_mode"] = mode

    policy = str(out.get("overlay_policy", "conditional")).lower()
    out["overlay_policy"] = policy if policy in {"always", "conditional"} else "conditional"

    behavior = str(out.get("overlay_behavior", "fallback_to_open")).lower()
    out["overlay_behavior"] = behavior if behavior in {"skip_if_no_trigger", "fallback_to_open"} else "fallback_to_open"

    w_raw = int(out.get("overlay_window_sec", _WINDOW_CHOICES[0]))
    out["overlay_window_sec"] = int(min(_WINDOW_CHOICES, key=lambda x: (abs(x - w_raw), x)))
    out["cap_mult"] = int(np.clip(int(out.get("cap_mult", 3)), 2, 10))
    out["adx_strong"] = float(np.clip(float(out.get("adx_strong", 25.0)), 18.0, 40.0))
    out["sep_k"] = float(np.clip(float(out.get("sep_k", 0.35)), 0.10, 1.00))
    out["use_sep_bypass"] = bool(out.get("use_sep_bypass", True))

    if mode == "pullback":
        out["pullback_dip_bps"] = int(np.clip(int(out.get("pullback_dip_bps", 20)), 2, 40))
        out["bounce_confirm_n"] = int(np.clip(int(out.get("bounce_confirm_n", 2)), 1, 5))
        out["break_bps"] = int(np.clip(int(out.get("break_bps", 6)), 1, 12))
    else:
        out["pullback_dip_bps"] = 2
        out["bounce_confirm_n"] = 1
        out["break_bps"] = int(np.clip(int(out.get("break_bps", 6)), 1, 12))
    return out


def _random_candidate(rng: random.Random) -> Dict[str, Any]:
    mode = "pullback" if rng.random() < 0.84 else "breakout"
    policy = "conditional" if rng.random() < 0.84 else "always"
    behavior = "fallback_to_open" if rng.random() < 0.96 else "skip_if_no_trigger"
    window = int(rng.choices(_WINDOW_CHOICES, weights=(1, 7, 8, 5, 2), k=1)[0])
    c = {
        "overlay_mode": mode,
        "overlay_policy": policy,
        "overlay_behavior": behavior,
        "overlay_window_sec": window,
        "pullback_dip_bps": rng.randint(14, 24) if mode == "pullback" else 2,
        "bounce_confirm_n": rng.randint(1, 2) if mode == "pullback" else 1,
        "break_bps": rng.randint(5, 8) if mode == "breakout" else rng.randint(2, 5),
        "adx_strong": round(rng.uniform(22.0, 30.0), 3),
        "use_sep_bypass": bool(rng.random() < 0.94),
        "sep_k": round(rng.uniform(0.28, 0.62), 4),
        "cap_mult": rng.randint(2, 6),
    }
    return _sanitize_candidate(c)


def _mutate(base: Dict[str, Any], rng: random.Random, rate: float = 0.35) -> Dict[str, Any]:
    c = dict(base)
    if rng.random() < rate:
        c["overlay_mode"] = rng.choice(["pullback", "breakout"])
    if rng.random() < rate:
        c["overlay_policy"] = rng.choice(["always", "conditional"])
    if rng.random() < rate:
        c["overlay_behavior"] = rng.choice(["skip_if_no_trigger", "fallback_to_open"])
    if rng.random() < rate:
        c["overlay_window_sec"] = int(rng.choice(_WINDOW_CHOICES))
    if rng.random() < rate:
        c["pullback_dip_bps"] = int(np.clip(c["pullback_dip_bps"] + rng.randint(-5, 5), 2, 40))
    if rng.random() < rate:
        c["bounce_confirm_n"] = int(np.clip(c["bounce_confirm_n"] + rng.randint(-1, 1), 1, 5))
    if rng.random() < rate:
        c["break_bps"] = int(np.clip(c["break_bps"] + rng.randint(-2, 2), 1, 12))
    if rng.random() < rate:
        c["adx_strong"] = float(np.clip(c["adx_strong"] + rng.uniform(-3.0, 3.0), 18.0, 40.0))
    if rng.random() < rate:
        c["use_sep_bypass"] = bool(not c["use_sep_bypass"])
    if rng.random() < rate:
        c["sep_k"] = float(np.clip(c["sep_k"] + rng.uniform(-0.15, 0.15), 0.10, 1.00))
    if rng.random() < rate:
        c["cap_mult"] = int(np.clip(c["cap_mult"] + rng.randint(-2, 2), 1, 10))
    return _sanitize_candidate(c)


def _cross(a: Dict[str, Any], b: Dict[str, Any], rng: random.Random) -> Dict[str, Any]:
    out = {}
    for k in a.keys():
        out[k] = a[k] if rng.random() < 0.5 else b[k]
    return _sanitize_candidate(out)


@dataclass(frozen=True)
class CandidateStats:
    cid: str
    valid: bool
    invalid_reason: str
    median_val_score: float
    iqr_val_score: float
    median_val_net: float
    median_val_pf: float
    median_val_dd: float
    median_val_trades: float
    median_skip_rate: float
    median_edge_decay: float
    stability_pct: float
    pos_splits: int
    total_splits: int


@dataclass(frozen=True)
class CandidateEvalResult:
    valid: bool
    score: float
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    reason_if_invalid: str
    stats: CandidateStats


def _candidate_metrics_from_stats(st: CandidateStats) -> Dict[str, Any]:
    return {
        "median_val_score": float(st.median_val_score),
        "iqr_val_score": float(st.iqr_val_score),
        "median_val_net": float(st.median_val_net),
        "median_val_pf": float(st.median_val_pf),
        "median_val_dd": float(st.median_val_dd),
        "median_val_trades": float(st.median_val_trades),
        "median_skip_rate": float(st.median_skip_rate),
        "median_edge_decay": float(st.median_edge_decay),
        "stability_pct": float(st.stability_pct),
        "pos_splits": int(st.pos_splits),
        "total_splits": int(st.total_splits),
    }


def _candidate_summary(c: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(c, dict):
        return None
    return {
        "overlay_policy": str(c.get("overlay_policy", "")),
        "overlay_mode": str(c.get("overlay_mode", "")),
        "overlay_behavior": str(c.get("overlay_behavior", "")),
        "overlay_window_sec": int(c.get("overlay_window_sec", 0)),
        "cap_mult": int(c.get("cap_mult", 0)),
    }


def _classify_invalid_reason(reason: str) -> str:
    r = str(reason or "").lower()
    if not r:
        return "unknown"
    if _constraint_reason_parts(r):
        return "constraint_fail"
    if "stability" in r:
        return "stability_fail"
    if "edge_decay" in r:
        return "edge_decay_fail"
    if "pf" in r or "profit_factor" in r:
        return "pf_fail"
    if "dd" in r:
        return "dd_fail"
    if "extra trades" in r or "overaly_trades" in r or "overlay created extra trades" in r:
        return "extra_trades"
    if "equity cap breach" in r or "negative cash" in r or "invariant" in r:
        return "invariant_violation"
    if "runtime:" in r:
        return "runtime_error"
    return "other"


def _is_constraint_reason(reason: str) -> bool:
    r = str(reason or "").strip().lower()
    return r.startswith("constraint_fail")


def _constraint_reason_parts(reason: str) -> List[str]:
    raw = str(reason or "").strip()
    if not _is_constraint_reason(raw):
        return []
    detail = raw.split(":", 1)[1] if ":" in raw else ""
    parts = [p.strip().lower() for p in detail.split(",") if p.strip()]
    if not parts:
        return ["constraint_fail"]
    return sorted(dict.fromkeys(parts))


def _constraint_reason_text(reasons: List[str]) -> str:
    parts = [str(p).strip().lower() for p in reasons if str(p).strip()]
    if not parts:
        return "CONSTRAINT_FAIL"
    return f"CONSTRAINT_FAIL:{','.join(sorted(dict.fromkeys(parts)))}"


def _score(net: float, dd: float, trades: float, init_eq: float, dd_penalty: float, trade_penalty: float) -> float:
    return float(net - dd_penalty * dd * init_eq - trade_penalty * trades)


def _evaluate_candidate(
    *,
    symbol: str,
    base_params: Dict[str, Any],
    dfi_pre: pd.DataFrame,
    split_windows: List[Tuple[int, int]],
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    run_tag: str,
    cache: Dict[str, CandidateStats],
    progress: Optional[EvalProgress] = None,
) -> CandidateStats:
    cid = _candidate_id(candidate)
    if cid in cache:
        return cache[cid]

    nets: List[float] = []
    pfs: List[float] = []
    dds: List[float] = []
    trades_n: List[float] = []
    scores: List[float] = []
    skips: List[float] = []
    edges: List[float] = []
    pos = 0
    invalid_reason = ""

    p = dict(base_params)
    p["equity_sizing_cap"] = float(args.initial_equity) * float(candidate["cap_mult"])
    ov_cfg = _overlay_cfg_from_candidate(candidate, args)

    for si, (va0, va1) in enumerate(split_windows):
        if progress is not None:
            progress.split_eval_count += 1
            if progress.split_eval_count % 100 == 0:
                total_s = progress.split_eval_total
                split_prog = f"{progress.split_eval_count}/{total_s}" if total_s > 0 else str(progress.split_eval_count)
                _log_info(
                    "split_eval heartbeat",
                    symbol=progress.symbol,
                    phase=progress.phase,
                    progress=split_prog,
                )
        if va1 <= va0:
            invalid_reason = "EMPTY_SEGMENT"
            break
        try:
            trades, m_base = run_backtest_long_only(
                df=dfi_pre,
                symbol=symbol,
                p=p,
                initial_equity=float(args.initial_equity),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slip_bps),
                collect_trades=True,
                start_idx=va0,
                end_idx=va1,
                assume_prepared=True,
            )
            tr_base = pd.DataFrame(trades)
            out = run_entry_overlay_backtest_from_df(
                symbol=symbol,
                df=dfi_pre,
                p=p,
                cfg=ov_cfg,
                initial_equity=float(args.initial_equity),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slip_bps),
                baseline_trades=tr_base,
                fetch_log_path=str(PROJECT_ROOT / "artifacts" / "execution_overlay" / symbol / run_tag / f"{cid}_s{si:03d}.jsonl"),
                start_idx=va0,
                end_idx=va1,
                assume_prepared=True,
            )
            m_ov = out["metrics"]
            dbg = out["debug"]
            if int(dbg.get("overlay_trades_count", 0)) > int(dbg.get("opportunities_count", 0)):
                invalid_reason = "OVERLAY_TRADES_GT_OPPORTUNITIES"
                break
            if int(dbg.get("overlay_trades_count", 0)) > int(dbg.get("baseline_trades_count", 0)):
                invalid_reason = "OVERLAY_TRADES_GT_BASELINE"
                break

            bnet = float(m_base.get("net_profit", 0.0))
            onet = float(m_ov.get("net_profit", 0.0))
            edge = float(onet / bnet) if bnet > 0 else (float(onet / abs(bnet)) if bnet < 0 else 0.0)

            pf = float(m_ov.get("profit_factor", 0.0))
            dd = float(m_ov.get("max_dd", 1.0))
            trn = float(m_ov.get("trades", 0.0))
            sc = _score(onet, dd, trn, float(args.initial_equity), float(args.dd_penalty), float(args.trade_penalty))

            nets.append(onet)
            pfs.append(pf)
            dds.append(dd)
            trades_n.append(trn)
            scores.append(sc)
            skips.append(float(dbg.get("skip_rate", 0.0)))
            edges.append(edge)
            if onet > 0:
                pos += 1
        except MemoryError:
            raise
        except Exception as e:
            invalid_reason = f"RUNTIME:{e}"
            break

    total = len(split_windows)
    if invalid_reason or len(scores) < total:
        st = CandidateStats(
            cid=cid,
            valid=False,
            invalid_reason=invalid_reason or "INCOMPLETE_SPLITS",
            median_val_score=-1e15,
            iqr_val_score=0.0,
            median_val_net=-1e9,
            median_val_pf=0.0,
            median_val_dd=1.0,
            median_val_trades=0.0,
            median_skip_rate=1.0,
            median_edge_decay=-1e9,
            stability_pct=0.0,
            pos_splits=0,
            total_splits=total,
        )
        cache[cid] = st
        return st

    med_score = float(np.median(scores))
    iqr_score = float(np.percentile(scores, 75) - np.percentile(scores, 25))
    med_net = float(np.median(nets))
    med_pf = float(np.median(pfs))
    med_dd = float(np.median(dds))
    med_tr = float(np.median(trades_n))
    med_skip = float(np.median(skips))
    med_edge = float(np.median(edges))
    stability = float(pos / max(1, total))

    fail_reasons: List[str] = []
    if stability < float(args.stability_min):
        fail_reasons.append("stability_fail")
    if med_pf < 1.20:
        fail_reasons.append("pf_fail")
    if med_dd > 0.25:
        fail_reasons.append("dd_fail")
    if med_edge < float(args.edge_decay_min):
        fail_reasons.append("edge_decay_fail")
    eligible = len(fail_reasons) == 0
    st = CandidateStats(
        cid=cid,
        valid=bool(eligible),
        invalid_reason="" if eligible else _constraint_reason_text(fail_reasons),
        median_val_score=med_score,
        iqr_val_score=iqr_score,
        median_val_net=med_net,
        median_val_pf=med_pf,
        median_val_dd=med_dd,
        median_val_trades=med_tr,
        median_skip_rate=med_skip,
        median_edge_decay=med_edge,
        stability_pct=stability * 100.0,
        pos_splits=pos,
        total_splits=total,
    )
    cache[cid] = st
    return st


def _fitness(st: CandidateStats, target_skip: float, skip_penalty: float) -> float:
    if not st.valid:
        return -1e15
    return float(
        st.median_val_net
        - st.median_val_dd * 0.45 * 100.0
        - 0.8 * st.median_val_trades
        - skip_penalty * abs(st.median_skip_rate - target_skip)
    )


def _save_active_configs(
    *,
    symbol: str,
    strategy_params_path: Path,
    candidate: Dict[str, Any],
    args: argparse.Namespace,
    run_tag: str,
    test_payload: Dict[str, Any],
    select_payload: Dict[str, Any],
) -> None:
    overlay_path = PROJECT_ROOT / "data" / "metadata" / "params" / f"{symbol}_overlay_1s.json"
    payload = {
        "symbol": symbol,
        "run_id": run_tag,
        "overlay_type": "ultra",
        "candidate": candidate,
        "constraints": {
            "initial_equity": float(args.initial_equity),
            "max_allocation": 0.7,
            "edge_decay_min": float(args.edge_decay_min),
            "stability_min": float(args.stability_min),
        },
        "selection": select_payload,
        "test": test_payload,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    hashv = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    payload["hash"] = hashv
    _write_json(overlay_path, payload)

    trade_cfg = {
        "symbol": symbol,
        "strategy_params_path": str(strategy_params_path),
        "overlay_params_path": str(overlay_path),
        "execution_assumptions": {
            "fee_bps": float(args.fee_bps),
            "slippage_bps": float(args.slip_bps),
            "cycle_shift": 1,
        },
        "sizing_settings": {
            "max_allocation": 0.7,
            "cap_mult": int(candidate["cap_mult"]),
            "equity_sizing_cap": float(args.initial_equity) * int(candidate["cap_mult"]),
            "initial_equity": float(args.initial_equity),
        },
        "meta": {
            "run_id": run_tag,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "test_metrics": test_payload,
            "selection_metrics": select_payload,
        },
    }
    _write_json(PROJECT_ROOT / "data" / "metadata" / "params" / f"{symbol}_trade_config.json", trade_cfg)


def _run_ultra_for_symbol(symbol: str, args: argparse.Namespace, run_tag: str) -> Dict[str, Any]:
    symbol = symbol.upper()
    paths = _artifact_paths(symbol)
    args_hash = _args_hash(args, symbol)
    run_started = time.monotonic()

    p_base, ppath = _load_params(symbol, args.btc_params)
    d = _ensure_indicators(_load_df(symbol, args.start, args.end), p_base)
    pre, test = _split_outer(d, args.test_start, args.test_end)

    split_seeds = [int(x.strip()) for x in str(args.split_seeds).split(",") if x.strip()]
    split_windows: List[Tuple[int, int]] = []
    for s in split_seeds:
        cfg = GAConfig(
            mc_splits=int(args.mc_splits),
            train_days=int(args.train_days),
            val_days=int(args.val_days),
            test_days=int(args.test_days),
            seed=s,
            initial_equity=float(args.initial_equity),
        )
        for _tr0, _tr1, va0, va1, _te0, _te1 in make_mc_splits(pre, cfg, gen=0):
            split_windows.append((va0, va1))
    if not split_windows:
        raise RuntimeError(f"{symbol}: no inner splits")

    random_target = int(args.random_samples)
    ga_target = int(args.ga_generations)
    pop_size = int(args.pop_size)
    fetch_fail_count = 0
    fetch_fail_reasons: Dict[str, int] = {"dns": 0, "429": 0, "timeout": 0, "other": 0}
    fetch_failed_split_count = 0
    fetch_failed_opp_count = 0
    fetch_fail_log_path = ""

    rng_seed = int(args.seed) + (abs(hash(symbol)) % 100000)
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed & 0xFFFFFFFF)

    cache: Dict[str, CandidateStats] = {}
    cand_map: Dict[str, Dict[str, Any]] = {}
    seen_random: set[str] = set()
    pop: List[Dict[str, Any]] = []
    best_fit = -1e18
    best_c: Optional[Dict[str, Any]] = None
    stale = 0
    start_gen = 0
    phase = "random"
    random_invalid_count = 0
    ga_invalid_count = 0
    best_random_score = -1e15
    current_gen: Optional[int] = None

    random_topk = TopKBuffer(cap=_TOPK_CAP)
    overall_topk = TopKBuffer(cap=max(_TOPK_CAP, pop_size * 3))

    random_progress = EvalProgress(symbol=symbol, phase="random", split_eval_total=max(1, random_target * len(split_windows)))
    ga_progress = EvalProgress(symbol=symbol, phase="ga", split_eval_total=max(1, ga_target * pop_size * len(split_windows)))

    def _row_from_candidate(
        row_type: str,
        candidate: Dict[str, Any],
        stats: CandidateStats,
        *,
        gen: Optional[int] = None,
        fitness: Optional[float] = None,
    ) -> Dict[str, Any]:
        cid = _candidate_id(candidate)
        row: Dict[str, Any] = {"row_type": row_type, "cid": cid, **candidate, **stats.__dict__}
        if gen is not None:
            row["gen"] = int(gen)
        if fitness is not None:
            row["fitness"] = float(fitness)
        return row

    def _record_row(row: Dict[str, Any], *, is_random: bool) -> None:
        _append_jsonl(paths.partial_jsonl, row)
        _append_csv_row(paths.random_csv if is_random else paths.ga_csv, row)
        overall_topk.add(row)
        if is_random:
            random_topk.add(row)
        cid = str(row.get("cid", "")).strip()
        cand = _candidate_from_row(row)
        st = _stats_from_row(row)
        if cid and cand is not None:
            cand_map[cid] = cand
        if cid and st is not None:
            cache[cid] = st

    def _write_status(
        *,
        status_phase: str,
        candidate_index: int,
        generation_index: Optional[int],
        invalid_count: int,
        note: str = "",
    ) -> None:
        payload = {
            "symbol": symbol,
            "run_id": run_tag,
            "args_hash": args_hash,
            "phase": status_phase,
            "candidate_index": int(candidate_index),
            "generation_index": int(generation_index) if generation_index is not None else None,
            "best_score": float(best_fit if best_fit > -1e17 else best_random_score),
            "best_candidate": best_c,
            "invalid_count": int(invalid_count),
            "rss_mb": float(_rss_mb()),
            "elapsed_sec": float(time.monotonic() - run_started),
            "note": str(note),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(paths.status_out, payload)

    def _save_ckpt(
        *,
        ckpt_phase: str,
        candidate_index: int,
        generation_index: Optional[int],
        invalid_count: int,
    ) -> None:
        top_rows = overall_topk.top(max(50, pop_size))
        random_rows = random_topk.top(max(50, pop_size))
        top_rows_valid = [r for r in top_rows if bool(r.get("valid", False))][: max(50, pop_size)]
        ckpt_best_cfg = (
            dict(best_c)
            if isinstance(best_c, dict)
            else (
                dict(best_valid_candidate)
                if isinstance(best_valid_candidate, dict)
                else (_candidate_only(top_rows_valid[0]) if top_rows_valid else (_candidate_only(top_rows[0]) if top_rows else None))
            )
        )
        ckpt_best_score = (
            float(best_fit)
            if best_fit > -1e17
            else (float(best_valid_score) if best_valid_score > -1e17 else float(best_invalid_score))
        )
        payload = {
            "symbol": symbol,
            "run_id": run_tag,
            "args_hash": args_hash,
            "phase": ckpt_phase,
            "candidate_index": int(candidate_index),
            "generation_index": int(generation_index) if generation_index is not None else None,
            "best_config": ckpt_best_cfg,
            "best_score": float(ckpt_best_score),
            "best_valid_candidate": best_valid_candidate,
            "best_valid_score": float(best_valid_score),
            "best_invalid_candidate": best_invalid_candidate,
            "best_invalid_score": float(best_invalid_score),
            "best_invalid_reason": str(best_invalid_reason),
            "top_k": top_rows,
            "top_k_valid": top_rows_valid,
            "random_top_k": random_rows,
            "seen_cids": sorted(seen_random),
            "population": pop,
            "stale": int(stale),
            "random_invalid_count": int(random_invalid_count),
            "ga_invalid_count": int(ga_invalid_count),
            "split_eval_count_random": int(random_progress.split_eval_count),
            "split_eval_count_ga": int(ga_progress.split_eval_count),
            "invalid_count": int(invalid_count),
            "partial_results_path": str(paths.partial_jsonl.resolve()),
            "rng_state_python": _pack_py_random_state(rng.getstate()),
            "rng_state_numpy": _pack_np_random_state(np.random.get_state()),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(paths.ckpt_out, payload)

    def _load_existing_stage_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        s1: List[Dict[str, Any]] = []
        s2: List[Dict[str, Any]] = []
        s3: List[Dict[str, Any]] = []
        if not paths.random_csv.exists() or paths.random_csv.stat().st_size <= 0:
            return s1, s2, s3
        try:
            df = pd.read_csv(paths.random_csv)
        except Exception:
            return s1, s2, s3
        if df.empty:
            return s1, s2, s3
        rows = df.to_dict("records")
        for row in rows:
            stage = str(row.get("stage", "")).lower()
            if stage == "stage1" or str(row.get("row_type", "")).lower() == "random_s1":
                s1.append(row)
            elif stage == "stage2" or str(row.get("row_type", "")).lower() == "random_s2":
                s2.append(row)
            elif stage == "stage3" or str(row.get("row_type", "")).lower() == "random_s3":
                s3.append(row)
            cid = str(row.get("cid", "")).strip()
            cand = _candidate_from_row(row)
            if cid and cand is not None:
                cand_map[cid] = cand
        return s1, s2, s3

    stage1_rows: List[Dict[str, Any]] = []
    stage2_rows: List[Dict[str, Any]] = []
    stage3_rows: List[Dict[str, Any]] = []

    ckpt = None if bool(args.fresh) else _read_json(paths.ckpt_out)
    if ckpt and ckpt.get("symbol") == symbol and ckpt.get("args_hash") == args_hash:
        phase = str(ckpt.get("phase", "random")).lower()
        run_tag = str(ckpt.get("run_id", run_tag))
        start_gen = max(0, int(ckpt.get("generation_index", -1)) + 1) if phase == "ga" else 0
        seen_random = {str(x) for x in ckpt.get("seen_cids", []) if str(x)}
        stale = int(ckpt.get("stale", 0))
        random_invalid_count = int(ckpt.get("random_invalid_count", ckpt.get("invalid_count", 0)))
        ga_invalid_count = int(ckpt.get("ga_invalid_count", 0))
        random_progress.split_eval_count = int(ckpt.get("split_eval_count_random", 0))
        ga_progress.split_eval_count = int(ckpt.get("split_eval_count_ga", 0))
        if isinstance(ckpt.get("best_config"), dict):
            best_c = dict(ckpt["best_config"])
        if ckpt.get("best_score") is not None:
            try:
                best_fit = float(ckpt["best_score"])
            except Exception:
                best_fit = -1e18
        pop = [dict(x) for x in ckpt.get("population", []) if isinstance(x, dict)]
        for pc in pop:
            cand_map[_candidate_id(pc)] = dict(pc)
        top_rows = [r for r in ckpt.get("top_k", []) if isinstance(r, dict)]
        random_rows = [r for r in ckpt.get("random_top_k", []) if isinstance(r, dict)]
        overall_topk.extend(top_rows)
        random_topk.extend(random_rows if random_rows else top_rows)
        for row in top_rows + random_rows:
            cid = str(row.get("cid", "")).strip()
            cand = _candidate_from_row(row)
            st = _stats_from_row(row)
            if cid and cand is not None:
                cand_map[cid] = cand
            if cid and st is not None:
                cache[cid] = st
        best_random_score = float(max([r.get("median_val_score", -1e15) for r in random_topk.sorted()] or [-1e15]))
        try:
            if isinstance(ckpt.get("rng_state_python"), dict):
                rng.setstate(_unpack_py_random_state(ckpt["rng_state_python"]))
            if isinstance(ckpt.get("rng_state_numpy"), dict):
                np.random.set_state(_unpack_np_random_state(ckpt["rng_state_numpy"]))
        except Exception:
            _log_warn("failed to restore RNG state from checkpoint", symbol=symbol, phase="resume", progress="-")
        _log_info(
            f"resuming from checkpoint phase={phase} random_done={len(seen_random)} start_gen={start_gen}",
            symbol=symbol,
            phase="resume",
            progress="-",
        )
    else:
        if bool(args.fresh):
            _log_info("fresh run requested; ignoring checkpoints", symbol=symbol, phase="resume", progress="-")
        elif ckpt is not None:
            _log_info("checkpoint args mismatch; starting fresh", symbol=symbol, phase="resume", progress="-")
        _atomic_write_text(paths.partial_jsonl, "")
        _atomic_write_text(paths.random_csv, "")
        _atomic_write_text(paths.ga_csv, "")

    if phase == "ga" and len(seen_random) < random_target:
        phase = "random"
        start_gen = 0
        pop = []

    random_started = time.monotonic()
    last_random_status_ts = time.monotonic()
    current_phase = "random"
    try:
        while len(seen_random) < random_target:
            c = _random_candidate(rng)
            cid = _candidate_id(c)
            if cid in seen_random:
                continue
            seen_random.add(cid)
            cand_map[cid] = dict(c)

            st = _evaluate_candidate(
                symbol=symbol,
                base_params=p_base,
                dfi_pre=pre,
                split_windows=split_windows,
                candidate=c,
                args=args,
                run_tag=run_tag,
                cache=cache,
                progress=random_progress,
            )
            row = _row_from_candidate("random", c, st)
            _record_row(row, is_random=True)
            if not st.valid:
                random_invalid_count += 1
            else:
                best_random_score = max(best_random_score, float(st.median_val_score))

            done = len(seen_random)
            if done % 10 == 0 or done == random_target:
                elapsed = time.monotonic() - random_started
                eta = (elapsed / done) * max(0, random_target - done) if done > 0 else 0.0
                _log_info(
                    (
                        f"random heartbeat best_median_score={best_random_score:.6f} "
                        f"invalid_count={random_invalid_count} elapsed_sec={elapsed:.1f} eta_sec={eta:.1f}"
                    ),
                    symbol=symbol,
                    phase="random",
                    progress=f"{done}/{random_target}",
                )
                _memory_guard(args, symbol=symbol, phase="random", progress=f"{done}/{random_target}")
            now = time.monotonic()
            if now - last_random_status_ts >= 120.0 or done == random_target:
                _write_status(
                    status_phase="random",
                    candidate_index=done,
                    generation_index=None,
                    invalid_count=random_invalid_count,
                )
                last_random_status_ts = now
            if done % 20 == 0 or done == random_target:
                _save_ckpt(
                    ckpt_phase="random",
                    candidate_index=done,
                    generation_index=None,
                    invalid_count=random_invalid_count,
                )

        top80_rows = random_topk.top(80)
        if not top80_rows:
            top80_rows = overall_topk.top(80)
        if not top80_rows:
            raise RuntimeError(f"{symbol}: no candidates available for GA")

        if not pop:
            if start_gen > 0:
                start_gen = 0
            pop = [cand_map[str(r["cid"])] for r in top80_rows if str(r.get("cid", "")) in cand_map]
            while len(pop) < pop_size:
                pop.append(_random_candidate(rng))
            pop = pop[:pop_size]

        current_phase = "ga"
        ga_started = time.monotonic()
        for gen in range(start_gen, ga_target):
            current_gen = gen
            scored: List[Tuple[float, Dict[str, Any], CandidateStats]] = []
            for c in pop:
                cid = _candidate_id(c)
                cand_map[cid] = dict(c)
                st = _evaluate_candidate(
                    symbol=symbol,
                    base_params=p_base,
                    dfi_pre=pre,
                    split_windows=split_windows,
                    candidate=c,
                    args=args,
                    run_tag=run_tag,
                    cache=cache,
                    progress=ga_progress,
                )
                fit = _fitness(st, float(args.target_skip_rate), float(args.skip_penalty))
                scored.append((fit, c, st))
            scored.sort(key=lambda x: x[0], reverse=True)
            bf, bc, bst = scored[0]
            gen_invalid_count = sum(1 for _, _, st in scored if not st.valid)
            ga_invalid_count = gen_invalid_count
            median_val_score = float(np.median([st.median_val_score for _, _, st in scored])) if scored else -1e15
            row = _row_from_candidate("ga", bc, bst, gen=gen, fitness=bf)
            _record_row(row, is_random=False)

            if bf > best_fit:
                best_fit = bf
                best_c = dict(bc)
                stale = 0
            else:
                stale += 1

            _log_info(
                (
                    f"ga heartbeat best_fitness={bf:.6f} median_val_score={median_val_score:.6f} "
                    f"invalid_count={gen_invalid_count} elapsed_sec={time.monotonic() - ga_started:.1f}"
                ),
                symbol=symbol,
                phase="ga",
                progress=f"{gen + 1}/{ga_target}",
            )
            _memory_guard(args, symbol=symbol, phase="ga", progress=f"{gen + 1}/{ga_target}")

            if (gen + 1) % 10 == 0:
                _write_status(
                    status_phase="ga",
                    candidate_index=len(seen_random),
                    generation_index=gen,
                    invalid_count=gen_invalid_count,
                )

            stop_now = stale >= int(args.early_stop)
            if not stop_now:
                elites = [x[1] for x in scored[: max(8, pop_size // 5)]]
                new_pop = [dict(e) for e in elites]
                while len(new_pop) < pop_size:
                    p1 = rng.choice(elites)
                    p2 = rng.choice(elites)
                    ch = _cross(p1, p2, rng)
                    ch = _mutate(ch, rng, rate=0.35)
                    new_pop.append(ch)
                pop = new_pop[:pop_size]

            _save_ckpt(
                ckpt_phase="ga",
                candidate_index=len(seen_random),
                generation_index=gen,
                invalid_count=gen_invalid_count,
            )
            if stop_now:
                _log_info(
                    f"ga early stop triggered after {stale} stale generations",
                    symbol=symbol,
                    phase="ga",
                    progress=f"{gen + 1}/{ga_target}",
                )
                break
    except MemoryError:
        _save_ckpt(
            ckpt_phase=current_phase,
            candidate_index=len(seen_random),
            generation_index=current_gen if current_phase == "ga" else None,
            invalid_count=ga_invalid_count if current_phase == "ga" else random_invalid_count,
        )
        raise

    if best_c is None:
        top = overall_topk.top(1)
        if top:
            best_c = _candidate_only(top[0])
    if best_c is None:
        raise RuntimeError(f"{symbol}: unable to resolve best candidate")

    best_st = _evaluate_candidate(
        symbol=symbol,
        base_params=p_base,
        dfi_pre=pre,
        split_windows=split_windows,
        candidate=best_c,
        args=args,
        run_tag=run_tag,
        cache=cache,
        progress=ga_progress,
    )

    p_test = dict(p_base)
    p_test["equity_sizing_cap"] = float(args.initial_equity) * float(best_c["cap_mult"])
    ov_cfg = _overlay_cfg_from_candidate(best_c, args)
    try:
        tr_b, m_b = run_backtest_long_only(
            df=test,
            symbol=symbol,
            p=p_test,
            initial_equity=float(args.initial_equity),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            collect_trades=True,
            assume_prepared=True,
        )
        out_t = run_entry_overlay_backtest_from_df(
            symbol=symbol,
            df=test,
            p=p_test,
            cfg=ov_cfg,
            initial_equity=float(args.initial_equity),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            baseline_trades=pd.DataFrame(tr_b),
            fetch_log_path=str((PROJECT_ROOT / "artifacts" / "execution_overlay" / symbol / run_tag / "ultra_test.jsonl").resolve()),
            assume_prepared=True,
        )
        m_o = out_t["metrics"]
        dbg_t = out_t["debug"]
        bnet = float(m_b.get("net_profit", 0.0))
        onet = float(m_o.get("net_profit", 0.0))
        edge_t = float(onet / bnet) if bnet > 0 else (float(onet / abs(bnet)) if bnet < 0 else 0.0)
    except MemoryError:
        raise
    except Exception:
        m_b = {"net_profit": 0.0, "profit_factor": 0.0, "max_dd": 1.0}
        m_o = {"net_profit": -1e9, "profit_factor": 0.0, "max_dd": 1.0}
        dbg_t = {"skip_rate": 1.0}
        edge_t = -1e9

    pass_fail = bool(
        edge_t >= float(args.edge_decay_min)
        and best_st.stability_pct >= float(args.stability_min) * 100.0
        and float(m_o.get("profit_factor", 0.0)) >= 1.20
        and float(m_o.get("max_dd", 1.0)) <= 0.25
    )

    top_rows = overall_topk.top(200)
    if top_rows:
        pd.DataFrame(top_rows).to_csv(paths.search_out, index=False)
    else:
        _atomic_write_text(paths.search_out, "")

    best_payload = {"symbol": symbol, "run_id": run_tag, "candidate": best_c, "validation": best_st.__dict__}
    _write_json(paths.best_out, best_payload)
    test_payload = {
        "symbol": symbol,
        "run_id": run_tag,
        "candidate": best_c,
        "baseline_test": m_b,
        "overlay_test": m_o,
        "edge_decay_test": edge_t,
        "skip_rate_test": float(dbg_t.get("skip_rate", 0.0)),
        "fetch_fail_count": int(fetch_fail_count),
        "fetch_fail_reasons": {k: int(v) for k, v in sorted(fetch_fail_reasons.items(), key=lambda kv: kv[0])},
        "fetch_failed_split_count": int(fetch_failed_split_count),
        "fetch_failed_opp_count": int(fetch_failed_opp_count),
        "fetch_fail_log_path": str(fetch_fail_log_path),
        "pass": pass_fail,
    }
    _write_json(paths.test_out, test_payload)
    _write_invalid_hist()
    _write_status(
        status_phase="done",
        candidate_index=len(seen_random),
        generation_index=current_gen,
        invalid_count=ga_invalid_count if current_gen is not None else random_invalid_count,
        note="completed",
    )

    if _parse_bool(args.save_active):
        _save_active_configs(
            symbol=symbol,
            strategy_params_path=ppath,
            candidate=best_c,
            args=args,
            run_tag=run_tag,
            test_payload=test_payload,
            select_payload=best_payload["validation"],
        )

    _log_info(f"symbol completed in {time.monotonic() - run_started:.1f}s", symbol=symbol, phase="done", progress="-")
    return {
        "symbol": symbol,
        "run_id": run_tag,
        "baseline_test_net": float(m_b.get("net_profit", 0.0)),
        "overlay_test_net": float(m_o.get("net_profit", 0.0)),
        "baseline_test_pf": float(m_b.get("profit_factor", 0.0)),
        "overlay_test_pf": float(m_o.get("profit_factor", 0.0)),
        "baseline_test_dd": float(m_b.get("max_dd", 1.0)),
        "overlay_test_dd": float(m_o.get("max_dd", 1.0)),
        "stability_pct": float(best_st.stability_pct),
        "skip_rate": float(dbg_t.get("skip_rate", 0.0)),
        "cap_mult": int(best_c["cap_mult"]),
        "policy": str(best_c["overlay_policy"]),
        "mode": str(best_c["overlay_mode"]),
        "behavior": str(best_c["overlay_behavior"]),
        "PASS/FAIL": "PASS" if pass_fail else "FAIL",
    }


@dataclass
class _SplitBase:
    split_id: int
    seed: int
    seed_rank: int
    split_rank: int
    va0: int
    va1: int
    baseline_net: float
    ts_ns_1h: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    rsi_prev: np.ndarray
    cycles: np.ndarray
    opp_entry_idx: np.ndarray
    opp_ts_ns: np.ndarray
    opp_1h_open_px: np.ndarray
    adx_prev: np.ndarray
    slope_prev: np.ndarray
    close_prev: np.ndarray
    ema_long_prev: np.ndarray
    ema_span_prev: np.ndarray
    atr_prev: np.ndarray
    ema_sep_prev: np.ndarray
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]]

    @property
    def opp_count(self) -> int:
        return int(self.opp_entry_idx.size)


@dataclass
class _SplitPrepared:
    split_id: int
    seed: int
    seed_rank: int
    split_rank: int
    va0: int
    va1: int
    baseline_net: float
    ts_ns_1h: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    rsi_prev: np.ndarray
    cycles: np.ndarray
    opp_entry_idx: np.ndarray
    opp_ts_ns: np.ndarray
    opp_1h_open_px: np.ndarray
    adx_prev: np.ndarray
    slope_prev: np.ndarray
    close_prev: np.ndarray
    ema_long_prev: np.ndarray
    ema_span_prev: np.ndarray
    atr_prev: np.ndarray
    ema_sep_prev: np.ndarray
    align_ok: np.ndarray
    first_open_1s: np.ndarray
    mismatch_pct: np.ndarray
    sec_len_max: np.ndarray
    sec_open: np.ndarray
    sec_high: np.ndarray
    sec_low: np.ndarray
    sec_close: np.ndarray
    sec_atr: np.ndarray
    sec_ema: np.ndarray
    len_by_w: np.ndarray
    next_open_tplus1s: np.ndarray
    min_px_in_window: np.ndarray
    idx_min: np.ndarray
    max_px_in_window: np.ndarray
    max_px_after_min: np.ndarray
    fetch_failed_mask: np.ndarray

    @property
    def opp_count(self) -> int:
        return int(self.opp_entry_idx.size)


@dataclass(frozen=True)
class _SplitEval:
    ok: bool
    invalid_reason: str
    net: float
    pf: float
    dd: float
    trades: float
    skip_rate: float
    edge_decay: float


@dataclass
class _ProfileBreakdown:
    opp_precompute_sec: float = 0.0
    window_precompute_sec: float = 0.0
    candidate_eval_sec_total: float = 0.0
    candidate_eval_count: int = 0


def _window_choice(v: int) -> int:
    iv = int(v)
    return int(min(_WINDOW_CHOICES, key=lambda x: (abs(x - iv), x)))


def _window_choice_index(v: int) -> int:
    w = _window_choice(v)
    return int(_WINDOW_CHOICES.index(w))


def _stage_split_ids(splits: List[_SplitPrepared], mc_splits: int) -> Tuple[List[int], List[int], List[int]]:
    by_seed: Dict[int, List[_SplitPrepared]] = {}
    for sp in splits:
        by_seed.setdefault(int(sp.seed_rank), []).append(sp)
    for rank in by_seed:
        by_seed[rank] = sorted(by_seed[rank], key=lambda x: x.split_rank)
    ranks = sorted(by_seed.keys())
    if not ranks:
        return [], [], []
    stage1: List[int] = []
    stage2: List[int] = []
    stage3: List[int] = []
    first = by_seed[ranks[0]]
    stage1 = [sp.split_id for sp in first[: min(6, len(first))]]
    for rank in ranks[:2]:
        stage2.extend(sp.split_id for sp in by_seed[rank][: min(15, len(by_seed[rank]))])
    for rank in ranks:
        stage3.extend(sp.split_id for sp in by_seed[rank][: min(int(mc_splits), len(by_seed[rank]))])
    stage1 = sorted(dict.fromkeys(stage1))
    stage2 = sorted(dict.fromkeys(stage2))
    stage3 = sorted(dict.fromkeys(stage3))
    return stage1, stage2, stage3


def _build_split_bases(
    *,
    symbol: str,
    args: argparse.Namespace,
    p_base: Dict[str, Any],
    pre: pd.DataFrame,
    split_seeds: List[int],
    run_tag: str,
    exec_root: Path,
) -> Tuple[List[_SplitBase], List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    max_w = int(max(_WINDOW_CHOICES))
    split_bases: List[_SplitBase] = []
    all_windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []

    for seed_rank, seed in enumerate(split_seeds):
        cfg = GAConfig(
            mc_splits=int(args.mc_splits),
            train_days=int(args.train_days),
            val_days=int(args.val_days),
            test_days=int(args.test_days),
            seed=int(seed),
            initial_equity=float(args.initial_equity),
        )
        seed_splits = list(make_mc_splits(pre, cfg, gen=0))
        for split_rank, (_tr0, _tr1, va0, va1, _te0, _te1) in enumerate(seed_splits):
            split_id = len(split_bases)
            dsplit = pre.iloc[int(va0) : int(va1)].reset_index(drop=True)
            if dsplit.empty:
                continue

            ts = pd.to_datetime(dsplit["Timestamp"], utc=True, errors="coerce")
            ts_ns_1h = ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            o = pd.to_numeric(dsplit["Open"], errors="coerce").to_numpy(dtype=float)
            h = pd.to_numeric(dsplit["High"], errors="coerce").to_numpy(dtype=float)
            l = pd.to_numeric(dsplit["Low"], errors="coerce").to_numpy(dtype=float)
            c = pd.to_numeric(dsplit["Close"], errors="coerce").to_numpy(dtype=float)
            rsi_prev = pd.to_numeric(dsplit["RSI"], errors="coerce").shift(1).fillna(50.0).to_numpy(dtype=float)
            atr_prev_all = pd.to_numeric(dsplit["ATR"], errors="coerce").shift(1).fillna(0.0).to_numpy(dtype=float)
            close_prev_all = pd.to_numeric(dsplit["Close"], errors="coerce").shift(1).bfill().fillna(0.0).to_numpy(dtype=float)
            slope_prev_all = (
                dsplit.get("EMA_200_SLOPE", pd.Series(0.0, index=dsplit.index))
                .astype(float)
                .shift(1)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            adx_prev_all = (
                dsplit.get("ADX", pd.Series(0.0, index=dsplit.index))
                .astype(float)
                .shift(1)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
            ema_long_col = f"EMA_{int(p_base.get('ema_trend_long', 120))}"
            ema_span_col = f"EMA_{int(p_base.get('ema_span', 35))}"
            ema_long_prev_all = (
                dsplit[ema_long_col].astype(float).shift(1).ffill().to_numpy(dtype=float)
                if ema_long_col in dsplit.columns
                else dsplit["EMA_200"].astype(float).shift(1).ffill().to_numpy(dtype=float)
            )
            ema_span_prev_all = (
                dsplit[ema_span_col].astype(float).shift(1).ffill().to_numpy(dtype=float)
                if ema_span_col in dsplit.columns
                else dsplit["EMA_200"].astype(float).shift(1).ffill().to_numpy(dtype=float)
            )
            ema_sep_prev_all = np.abs(ema_span_prev_all - ema_long_prev_all)
            cycles = _shift_cycles(
                compute_cycles(dsplit, p_base),
                shift=int(p_base.get("cycle_shift", 1)),
                fill=int(p_base.get("cycle_fill", 2)),
            ).astype(np.int16)

            p_pre = dict(p_base)
            p_pre["equity_sizing_cap"] = float(args.initial_equity)
            base_trades, m_base = run_backtest_long_only(
                df=dsplit,
                symbol=symbol,
                p=p_pre,
                initial_equity=float(args.initial_equity),
                fee_bps=float(args.fee_bps),
                slippage_bps=float(args.slip_bps),
                collect_trades=True,
                assume_prepared=True,
            )
            baseline_net = float(m_base.get("net_profit", 0.0))

            opp_df = pd.DataFrame(base_trades)
            if opp_df.empty:
                opp_entry_idx = np.array([], dtype=np.int32)
                opp_ts_ns = np.array([], dtype=np.int64)
                opp_1h_open_px = np.array([], dtype=np.float32)
                adx_prev = np.array([], dtype=np.float32)
                slope_prev = np.array([], dtype=np.float32)
                close_prev = np.array([], dtype=np.float32)
                ema_long_prev = np.array([], dtype=np.float32)
                ema_span_prev = np.array([], dtype=np.float32)
                atr_prev = np.array([], dtype=np.float32)
                ema_sep_prev = np.array([], dtype=np.float32)
            else:
                entry_ts = pd.to_datetime(opp_df["entry_ts"], utc=True, errors="coerce")
                entry_ns = entry_ts.to_numpy(dtype="datetime64[ns]").astype(np.int64)
                idx = np.searchsorted(ts_ns_1h, entry_ns, side="left").astype(np.int64)
                valid = (idx >= 0) & (idx < len(ts_ns_1h))
                if valid.any():
                    valid = valid & (ts_ns_1h[idx] == entry_ns)
                idx = idx[valid]
                opp_ts_ns = entry_ns[valid]
                opp_entry_idx = idx.astype(np.int32)
                opp_1h_open_px = o[idx].astype(np.float32)
                adx_prev = adx_prev_all[idx].astype(np.float32)
                slope_prev = slope_prev_all[idx].astype(np.float32)
                close_prev = close_prev_all[idx].astype(np.float32)
                ema_long_prev = ema_long_prev_all[idx].astype(np.float32)
                ema_span_prev = ema_span_prev_all[idx].astype(np.float32)
                atr_prev = atr_prev_all[idx].astype(np.float32)
                ema_sep_prev = ema_sep_prev_all[idx].astype(np.float32)

            opp_path = exec_root / f"opportunities_split_{split_id:03d}.npz"
            np.savez_compressed(
                opp_path,
                split_id=np.array([split_id], dtype=np.int32),
                seed=np.array([seed], dtype=np.int32),
                va0=np.array([va0], dtype=np.int32),
                va1=np.array([va1], dtype=np.int32),
                opp_ts=opp_ts_ns,
                opp_entry_idx=opp_entry_idx,
                opp_1h_open_px=opp_1h_open_px,
                adx_prev=adx_prev,
                slope_prev=slope_prev,
                close_prev=close_prev,
                ema_long_prev=ema_long_prev,
                ema_span_prev=ema_span_prev,
                atr_prev=atr_prev,
            )

            windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
            for ns in opp_ts_ns:
                t0 = pd.to_datetime(int(ns), utc=True)
                t1 = t0 + pd.Timedelta(seconds=max_w)
                windows.append((t0, t1))
            all_windows.extend(windows)

            split_bases.append(
                _SplitBase(
                    split_id=split_id,
                    seed=int(seed),
                    seed_rank=int(seed_rank),
                    split_rank=int(split_rank),
                    va0=int(va0),
                    va1=int(va1),
                    baseline_net=float(baseline_net),
                    ts_ns_1h=ts_ns_1h.astype(np.int64),
                    o=o.astype(float),
                    h=h.astype(float),
                    l=l.astype(float),
                    c=c.astype(float),
                    rsi_prev=rsi_prev.astype(float),
                    cycles=cycles.astype(np.int16),
                    opp_entry_idx=opp_entry_idx,
                    opp_ts_ns=opp_ts_ns,
                    opp_1h_open_px=opp_1h_open_px,
                    adx_prev=adx_prev,
                    slope_prev=slope_prev,
                    close_prev=close_prev,
                    ema_long_prev=ema_long_prev,
                    ema_span_prev=ema_span_prev,
                    atr_prev=atr_prev,
                    ema_sep_prev=ema_sep_prev,
                    windows=windows,
                )
            )
    return split_bases, all_windows


def _prepare_split_1s_summaries(
    *,
    symbol: str,
    args: argparse.Namespace,
    run_tag: str,
    exec_root: Path,
    split_bases: List[_SplitBase],
) -> Tuple[List[_SplitPrepared], Dict[str, Any]]:
    max_w = int(max(_WINDOW_CHOICES))
    n_w = len(_WINDOW_CHOICES)
    cache_root = (PROJECT_ROOT / str(args.cache_root)).resolve()
    fetch_log = (exec_root / "prefetch_1s.jsonl").resolve()
    sec_cfg = ExecutionEvalConfig(
        mode="klines1s",
        market="spot",
        window_sec=1,
        cache_cap_gb=float(args.cache_cap_gb),
        cap_gb=float(args.cache_cap_gb),
        cache_root=str(cache_root),
        fetch_workers=max(1, int(args.fetch_workers)),
        cache_1s=bool(args.cache_1s),
        max_cache_mb=float(args.max_cache_mb),
        alignment_max_gap_sec=2.0,
        alignment_open_tol_pct=0.01,
        overlay_mode="pullback",
        overlay_window_sec=max_w,
        overlay_breakout_lookback_sec=_BREAKOUT_LOOKBACK_SEC,
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        initial_equity=float(args.initial_equity),
    )

    reports_root = (PROJECT_ROOT / "artifacts" / "reports").resolve()
    fetch_failures_path = reports_root / f"overlay_ultra_fetch_failures_{symbol.upper()}_{run_tag}.jsonl"
    all_windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for sb in split_bases:
        all_windows.extend(sb.windows)
    failed_windows: List[Dict[str, Any]] = []
    if all_windows:
        failed_windows = _ensure_cache_for_windows(
            symbol=symbol,
            windows=all_windows,
            cfg=sec_cfg,
            cache_root=cache_root,
            fetch_log_path=fetch_log,
            failed_windows_path=fetch_failures_path,
        )

    fetch_reason_hist: Dict[str, int] = {"dns": 0, "429": 0, "timeout": 0, "other": 0}
    failed_ranges_ns: List[Tuple[int, int]] = []
    for row in failed_windows:
        key = str(row.get("reason", "other")).strip().lower()
        if key not in fetch_reason_hist:
            key = "other"
        fetch_reason_hist[key] = int(fetch_reason_hist.get(key, 0) + 1)
        try:
            s_ms = int(row.get("start_ms"))
            e_ms = int(row.get("end_ms"))
        except Exception:
            continue
        s_ns = int(s_ms) * 1_000_000
        e_ns = int(e_ms) * 1_000_000
        if e_ns > s_ns:
            failed_ranges_ns.append((s_ns, e_ns))

    failed_split_count = 0
    failed_opp_count = 0

    prepared: List[_SplitPrepared] = []
    for sb in split_bases:
        n_opp = int(sb.opp_count)
        fetch_failed_mask = np.zeros(n_opp, dtype=bool)
        if failed_ranges_ns and sb.windows:
            for oi, (ws, we) in enumerate(sb.windows):
                ws_ns = int(_to_utc_ts(ws).value)
                we_ns = int(_to_utc_ts(we).value)
                for fs_ns, fe_ns in failed_ranges_ns:
                    if ws_ns < fe_ns and fs_ns < we_ns:
                        fetch_failed_mask[oi] = True
                        break
        split_failed_n = int(np.count_nonzero(fetch_failed_mask))
        if split_failed_n > 0:
            failed_split_count += 1
            failed_opp_count += split_failed_n

        align_ok = np.zeros(n_opp, dtype=bool)
        first_open_1s = np.full(n_opp, np.nan, dtype=np.float32)
        mismatch_pct = np.full(n_opp, np.nan, dtype=np.float32)
        sec_len_max = np.zeros(n_opp, dtype=np.int16)
        sec_open = np.full((n_opp, max_w), np.nan, dtype=np.float32)
        sec_high = np.full((n_opp, max_w), np.nan, dtype=np.float32)
        sec_low = np.full((n_opp, max_w), np.nan, dtype=np.float32)
        sec_close = np.full((n_opp, max_w), np.nan, dtype=np.float32)
        sec_atr = np.full((n_opp, max_w), np.nan, dtype=np.float32)
        sec_ema = np.full((n_opp, max_w), np.nan, dtype=np.float32)
        len_by_w = np.zeros((n_w, n_opp), dtype=np.int16)
        next_open_tplus1s = np.full((n_w, n_opp), np.nan, dtype=np.float32)
        min_px_in_window = np.full((n_w, n_opp), np.nan, dtype=np.float32)
        idx_min = np.full((n_w, n_opp), -1, dtype=np.int16)
        max_px_in_window = np.full((n_w, n_opp), np.nan, dtype=np.float32)
        max_px_after_min = np.full((n_w, n_opp), np.nan, dtype=np.float32)

        if sb.windows:
            sec_df = _load_1s_data(
                symbol=symbol,
                windows=sb.windows,
                cfg=sec_cfg,
                cache_root=cache_root,
                fetch_log_path=fetch_log,
            )
            sec = _sec_arrays(sec_df)
        else:
            sec = {
                "ts_ns": np.array([], dtype=np.int64),
                "open": np.array([], dtype=float),
                "high": np.array([], dtype=float),
                "low": np.array([], dtype=float),
                "close": np.array([], dtype=float),
                "vol": np.array([], dtype=float),
            }

        for oi in range(n_opp):
            t0 = pd.to_datetime(int(sb.opp_ts_ns[oi]), utc=True)
            ent_al = _alignment_check(
                sec,
                t0,
                float(sb.opp_1h_open_px[oi]),
                window_sec=max_w,
                max_gap_sec=float(sec_cfg.alignment_max_gap_sec),
                tol_pct=float(sec_cfg.alignment_open_tol_pct),
            )
            align_ok[oi] = bool(ent_al.ok)
            if np.isfinite(ent_al.first_open):
                first_open_1s[oi] = float(ent_al.first_open)
            if np.isfinite(ent_al.mismatch_pct):
                mismatch_pct[oi] = float(ent_al.mismatch_pct)

            i0 = int(ent_al.i0)
            i1 = int(ent_al.i1)
            n = max(0, min(max_w, i1 - i0))
            sec_len_max[oi] = int(n)
            if n > 0:
                op = np.asarray(sec["open"][i0 : i0 + n], dtype=float)
                hi = np.asarray(sec["high"][i0 : i0 + n], dtype=float)
                lo = np.asarray(sec["low"][i0 : i0 + n], dtype=float)
                cl = np.asarray(sec["close"][i0 : i0 + n], dtype=float)
                sec_open[oi, :n] = op.astype(np.float32)
                sec_high[oi, :n] = hi.astype(np.float32)
                sec_low[oi, :n] = lo.astype(np.float32)
                sec_close[oi, :n] = cl.astype(np.float32)
                prev_close = np.concatenate(([cl[0]], cl[:-1]))
                tr = np.maximum.reduce([hi - lo, np.abs(hi - prev_close), np.abs(lo - prev_close)])
                atr = pd.Series(tr).ewm(span=5, adjust=False).mean().to_numpy(dtype=float)
                ema = pd.Series(cl).ewm(span=5, adjust=False).mean().to_numpy(dtype=float)
                sec_atr[oi, :n] = atr.astype(np.float32)
                sec_ema[oi, :n] = ema.astype(np.float32)

            for wi, w in enumerate(_WINDOW_CHOICES):
                j0, j1 = _slice_idx(sec, t0, int(w))
                nw = max(0, min(max_w, int(j1 - j0)))
                len_by_w[wi, oi] = int(nw)
                if nw <= 0:
                    continue
                opw = np.asarray(sec["open"][j0 : j0 + nw], dtype=float)
                hiw = np.asarray(sec["high"][j0 : j0 + nw], dtype=float)
                low = np.asarray(sec["low"][j0 : j0 + nw], dtype=float)
                if nw > 1 and np.isfinite(opw[1]):
                    next_open_tplus1s[wi, oi] = float(opw[1])
                if np.isfinite(low).any():
                    low_safe = np.where(np.isfinite(low), low, np.inf)
                    m_idx = int(np.argmin(low_safe))
                    if np.isfinite(low_safe[m_idx]):
                        idx_min[wi, oi] = int(m_idx)
                        min_px_in_window[wi, oi] = float(low_safe[m_idx])
                if np.isfinite(hiw).any():
                    hi_safe = np.where(np.isfinite(hiw), hiw, -np.inf)
                    max_px_in_window[wi, oi] = float(np.max(hi_safe))
                    m_idx = int(idx_min[wi, oi])
                    if m_idx >= 0 and m_idx < nw:
                        max_px_after_min[wi, oi] = float(np.max(hi_safe[m_idx:]))

        win_path = exec_root / f"win_summaries_split_{sb.split_id:03d}.npz"
        np.savez_compressed(
            win_path,
            split_id=np.array([sb.split_id], dtype=np.int32),
            window_choices=np.array(_WINDOW_CHOICES, dtype=np.int16),
            opp_ts_ns=sb.opp_ts_ns.astype(np.int64),
            len_by_w=len_by_w,
            next_open_px_at_tplus1s=next_open_tplus1s,
            min_px_in_window=min_px_in_window,
            idx_min=idx_min,
            max_px_in_window=max_px_in_window,
            max_px_after_min=max_px_after_min,
            sec_len_max=sec_len_max,
            sec_open=sec_open,
            sec_high=sec_high,
            sec_low=sec_low,
            sec_close=sec_close,
            sec_atr=sec_atr,
            sec_ema=sec_ema,
            fetch_failed_mask=fetch_failed_mask.astype(np.uint8),
        )

        prepared.append(
            _SplitPrepared(
                split_id=sb.split_id,
                seed=sb.seed,
                seed_rank=sb.seed_rank,
                split_rank=sb.split_rank,
                va0=sb.va0,
                va1=sb.va1,
                baseline_net=sb.baseline_net,
                ts_ns_1h=sb.ts_ns_1h,
                o=sb.o,
                h=sb.h,
                l=sb.l,
                c=sb.c,
                rsi_prev=sb.rsi_prev,
                cycles=sb.cycles,
                opp_entry_idx=sb.opp_entry_idx,
                opp_ts_ns=sb.opp_ts_ns,
                opp_1h_open_px=sb.opp_1h_open_px,
                adx_prev=sb.adx_prev,
                slope_prev=sb.slope_prev,
                close_prev=sb.close_prev,
                ema_long_prev=sb.ema_long_prev,
                ema_span_prev=sb.ema_span_prev,
                atr_prev=sb.atr_prev,
                ema_sep_prev=sb.ema_sep_prev,
                align_ok=align_ok,
                first_open_1s=first_open_1s,
                mismatch_pct=mismatch_pct,
                sec_len_max=sec_len_max,
                sec_open=sec_open,
                sec_high=sec_high,
                sec_low=sec_low,
                sec_close=sec_close,
                sec_atr=sec_atr,
                sec_ema=sec_ema,
                len_by_w=len_by_w,
                next_open_tplus1s=next_open_tplus1s,
                min_px_in_window=min_px_in_window,
                idx_min=idx_min,
                max_px_in_window=max_px_in_window,
                max_px_after_min=max_px_after_min,
                fetch_failed_mask=fetch_failed_mask,
            )
        )
    fetch_summary = {
        "window_fail_count": int(len(failed_windows)),
        "reason_hist": {k: int(v) for k, v in sorted(fetch_reason_hist.items(), key=lambda kv: kv[0])},
        "failed_split_count": int(failed_split_count),
        "failed_opp_count": int(failed_opp_count),
        "failed_windows_path": str(fetch_failures_path.resolve()),
    }
    return prepared, fetch_summary


def _evaluate_split_fast(
    *,
    symbol: str,
    split: _SplitPrepared,
    candidate: Dict[str, Any],
    p_base: Dict[str, Any],
    args: argparse.Namespace,
) -> _SplitEval:
    try:
        if split.opp_count <= 0:
            return _SplitEval(
                ok=True,
                invalid_reason="",
                net=0.0,
                pf=0.0,
                dd=0.0,
                trades=0.0,
                skip_rate=0.0,
                edge_decay=0.0,
            )

        mode = str(candidate["overlay_mode"]).lower()
        policy = str(candidate["overlay_policy"]).lower()
        behavior = str(candidate["overlay_behavior"]).lower()
        fallback_to_open = behavior == "fallback_to_open"
        skip_if_no_trigger = not fallback_to_open
        use_sep_bypass = bool(candidate["use_sep_bypass"])
        sep_k = float(candidate["sep_k"])
        adx_strong = float(candidate["adx_strong"])
        dip_bps = float(candidate["pullback_dip_bps"])
        bounce_n = max(1, int(candidate["bounce_confirm_n"]))
        break_bps = float(candidate["break_bps"])
        bump = break_bps / 1e4
        w_idx = _window_choice_index(int(candidate["overlay_window_sec"]))

        max_hold = int(p_base.get("max_hold_hours", 48))
        risk_per_trade = float(p_base.get("risk_per_trade", 0.02))
        max_alloc = float(p_base.get("max_allocation", 0.7))
        atr_k = float(p_base.get("atr_k", 1.0))
        tp_by_cycle = list(p_base.get("tp_mult_by_cycle", [1.02] * 5))
        sl_by_cycle = list(p_base.get("sl_mult_by_cycle", [0.98] * 5))
        exit_rsi_by_cycle = list(p_base.get("exit_rsi_by_cycle", [50.0] * 5))

        sizing_equity_cap = float(args.initial_equity) * float(int(candidate["cap_mult"]))
        cap_equity_limit = float(args.initial_equity) * float(int(candidate["cap_mult"]))
        fee = float(args.fee_bps)
        slip = float(args.slip_bps)

        cash = float(args.initial_equity)
        pnls: List[float] = []
        skip_count = 0
        last_exit_ns = np.iinfo(np.int64).min

        for oi in range(split.opp_count):
            if oi < int(split.fetch_failed_mask.size) and bool(split.fetch_failed_mask[oi]):
                skip_count += 1
                continue

            t0_ns = int(split.opp_ts_ns[oi])
            if t0_ns < int(last_exit_ns):
                skip_count += 1
                continue

            i = int(split.opp_entry_idx[oi])
            if i < 0 or i >= len(split.o):
                return _SplitEval(False, "MISSING_ENTRY_BAR", -1e9, 0.0, 1.0, 0.0, 1.0, -1e9)

            entry_open_1h = float(split.opp_1h_open_px[oi])
            entry_raw = float(entry_open_1h)
            overlay_triggered = False

            if bool(split.align_ok[oi]):
                bypass_overlay = False
                if policy == "conditional":
                    sep_ok = True
                    if use_sep_bypass:
                        sep_ok = bool(float(split.ema_sep_prev[oi]) >= sep_k * max(float(split.atr_prev[oi]), 1e-12))
                    bypass_overlay = bool(
                        float(split.adx_prev[oi]) >= adx_strong
                        and float(split.slope_prev[oi]) > 0.0
                        and float(split.close_prev[oi]) > float(split.ema_long_prev[oi])
                        and sep_ok
                    )

                if not bypass_overlay and mode in {"breakout", "pullback"}:
                    n_w = int(split.len_by_w[w_idx, oi])
                    if n_w <= 0:
                        if skip_if_no_trigger:
                            skip_count += 1
                            continue
                    elif mode == "breakout":
                        triggered = False
                        if n_w > _BREAKOUT_LOOKBACK_SEC:
                            hi = split.sec_high[oi, :n_w].astype(float)
                            op = split.sec_open[oi, :n_w].astype(float)
                            for j in range(_BREAKOUT_LOOKBACK_SEC, n_w):
                                micro_high = float(np.nanmax(hi[j - _BREAKOUT_LOOKBACK_SEC : j]))
                                if not np.isfinite(micro_high):
                                    continue
                                level = micro_high * (1.0 + bump)
                                hj = float(hi[j])
                                if np.isfinite(hj) and hj >= level:
                                    jn = j + 1
                                    if jn >= n_w:
                                        break
                                    nxt = float(op[jn])
                                    if not np.isfinite(nxt) or nxt <= 0.0:
                                        break
                                    entry_raw = float(max(nxt, level))
                                    overlay_triggered = True
                                    triggered = True
                                    break
                        if not triggered and skip_if_no_trigger:
                            skip_count += 1
                            continue
                    else:
                        if n_w < 4:
                            if skip_if_no_trigger:
                                skip_count += 1
                                continue
                        else:
                            op = split.sec_open[oi, :n_w].astype(float)
                            lo = split.sec_low[oi, :n_w].astype(float)
                            cl = split.sec_close[oi, :n_w].astype(float)
                            atr = split.sec_atr[oi, :n_w].astype(float)
                            ema = split.sec_ema[oi, :n_w].astype(float)
                            first_open = float(op[0])
                            dip_level = first_open * (1.0 - dip_bps / 1e4)
                            dip_seen = False
                            triggered = False
                            for j in range(1, n_w):
                                if not np.isfinite(lo[j]) or not np.isfinite(cl[j]) or not np.isfinite(ema[j]):
                                    continue
                                dip_cond = (float(lo[j]) <= dip_level) or (
                                    np.isfinite(atr[j]) and float(lo[j]) <= first_open - float(atr[j])
                                )
                                if dip_cond:
                                    dip_seen = True
                                if not dip_seen:
                                    continue
                                start_j = j - bounce_n + 1
                                if start_j < 1:
                                    continue
                                rising = bool(np.all(cl[start_j : j + 1] > cl[start_j - 1 : j]))
                                crossed_up = bool(float(cl[j]) > float(ema[j]))
                                if rising and crossed_up:
                                    jn = j + 1
                                    if jn >= n_w:
                                        break
                                    nxt = float(op[jn])
                                    if not np.isfinite(nxt) or nxt <= 0.0:
                                        break
                                    entry_raw = float(nxt)
                                    overlay_triggered = True
                                    triggered = True
                                    break
                            if not triggered and skip_if_no_trigger:
                                skip_count += 1
                                continue

            buy_px = float(_apply_cost(float(entry_raw), fee, slip, "buy"))
            if not np.isfinite(buy_px) or buy_px <= 0.0:
                buy_px = float(_apply_cost(entry_open_1h, fee, slip, "buy"))
                if not np.isfinite(buy_px) or buy_px <= 0.0:
                    return _SplitEval(False, "BAD_FILL_SANITY_ENTRY", -1e9, 0.0, 1.0, 0.0, 1.0, -1e9)

            equity = float(cash)
            equity_for_size = float(min(equity, sizing_equity_cap)) if sizing_equity_cap > 0.0 else float(equity)
            size = _position_size(equity_for_size, buy_px, float(split.atr_prev[oi]), risk_per_trade, max_alloc, atr_k)
            if size <= 0.0:
                skip_count += 1
                continue
            cost = float(size * buy_px)
            if cost > cash:
                size = float(cash / max(buy_px, 1e-12))
                cost = float(size * buy_px)
            if size <= 0.0:
                skip_count += 1
                continue
            cash -= float(cost)

            entry_cycle = int(split.cycles[i])
            if entry_cycle < 0 or entry_cycle >= len(tp_by_cycle):
                entry_cycle = max(0, min(len(tp_by_cycle) - 1, entry_cycle))
            tp_mult = float(tp_by_cycle[entry_cycle])
            sl_mult = float(sl_by_cycle[entry_cycle])
            entry_px = float(buy_px)

            exit_i = len(split.o) - 1
            exit_exec_px = float(split.c[-1])
            for j in range(i + 1, len(split.o)):
                hold = j - i
                tp_px = entry_px * tp_mult
                sl_px = entry_px * sl_mult
                hit_sl = bool(split.l[j] <= sl_px)
                hit_tp = bool(split.h[j] >= tp_px)
                if hit_sl and hit_tp:
                    exit_exec_px = float(sl_px)
                    exit_i = j
                    break
                if hit_sl:
                    exit_exec_px = float(sl_px)
                    exit_i = j
                    break
                if hit_tp:
                    exit_exec_px = float(tp_px)
                    exit_i = j
                    break
                if hold >= max_hold:
                    exit_exec_px = float(split.o[j])
                    exit_i = j
                    break
                ex = float(exit_rsi_by_cycle[entry_cycle]) if entry_cycle < len(exit_rsi_by_cycle) else 50.0
                pnl_ratio = float(split.c[j] / entry_px) if entry_px > 0.0 else 1.0
                if (float(split.rsi_prev[j]) < ex) and (pnl_ratio > 1.0):
                    exit_exec_px = float(split.o[j])
                    exit_i = j
                    break

            sell_px = float(_apply_cost(float(exit_exec_px), fee, slip, "sell"))
            proceeds = float(size * sell_px)
            cash += float(proceeds)
            if cash > cap_equity_limit + 1e-6:
                return _SplitEval(False, f"BUG: equity cap breach for {symbol} in overlay replay: cash={cash} limit={cap_equity_limit}", -1e9, 0.0, 1.0, 0.0, 1.0, -1e9)
            pnl = float((sell_px - entry_px) * size)
            pnls.append(float(pnl))
            last_exit_ns = int(split.ts_ns_1h[exit_i])

        pnl_np = np.array(pnls, dtype=float)
        metrics = _metrics_from_pnl(float(args.initial_equity), pnl_np)
        if float(metrics["max_dd"]) > 1.000001:
            return _SplitEval(False, f"BUG: overlay replay DD out of range for long-only {symbol}: {metrics['max_dd']}", -1e9, 0.0, 1.0, 0.0, 1.0, -1e9)
        trades = float(metrics.get("trades", 0.0))
        if int(trades) > int(split.opp_count):
            return _SplitEval(False, f"BUG: overlay created extra trades for {symbol}: {int(trades)}>{split.opp_count}", -1e9, 0.0, 1.0, 0.0, 1.0, -1e9)
        bnet = float(split.baseline_net)
        onet = float(metrics.get("net_profit", 0.0))
        edge = float(onet / bnet) if bnet > 0 else (float(onet / abs(bnet)) if bnet < 0 else 0.0)
        return _SplitEval(
            ok=True,
            invalid_reason="",
            net=onet,
            pf=float(metrics.get("profit_factor", 0.0)),
            dd=float(metrics.get("max_dd", 1.0)),
            trades=trades,
            skip_rate=float(skip_count / max(1, split.opp_count)),
            edge_decay=edge,
        )
    except MemoryError:
        raise
    except Exception as e:
        return _SplitEval(False, f"RUNTIME:{e}", -1e9, 0.0, 1.0, 0.0, 1.0, -1e9)


def _evaluate_candidate_fast(
    *,
    symbol: str,
    candidate: Dict[str, Any],
    split_ids: List[int],
    split_map: Dict[int, _SplitPrepared],
    split_cache: Dict[Tuple[str, int], _SplitEval],
    args: argparse.Namespace,
    p_base: Dict[str, Any],
    progress: Optional[EvalProgress],
    profile: _ProfileBreakdown,
) -> CandidateEvalResult:
    cid = _candidate_id(candidate)
    nets: List[float] = []
    pfs: List[float] = []
    dds: List[float] = []
    trades_n: List[float] = []
    scores: List[float] = []
    skips: List[float] = []
    edges: List[float] = []
    pos = 0
    invalid_reason = ""

    for sid in split_ids:
        key = (cid, int(sid))
        sv = split_cache.get(key)
        if sv is None:
            t0 = time.perf_counter()
            sv = _evaluate_split_fast(
                symbol=symbol,
                split=split_map[int(sid)],
                candidate=candidate,
                p_base=p_base,
                args=args,
            )
            profile.candidate_eval_sec_total += float(time.perf_counter() - t0)
            profile.candidate_eval_count += 1
            split_cache[key] = sv
            if progress is not None:
                progress.split_eval_count += 1
                if progress.split_eval_count % 100 == 0:
                    total_s = progress.split_eval_total
                    split_prog = f"{progress.split_eval_count}/{total_s}" if total_s > 0 else str(progress.split_eval_count)
                    _log_info(
                        "split_eval heartbeat",
                        symbol=progress.symbol,
                        phase=progress.phase,
                        progress=split_prog,
                    )
        if not sv.ok:
            invalid_reason = str(sv.invalid_reason or "INCOMPLETE_SPLITS")
            break
        onet = float(sv.net)
        pf = float(sv.pf)
        dd = float(sv.dd)
        trn = float(sv.trades)
        sc = _score(onet, dd, trn, float(args.initial_equity), float(args.dd_penalty), float(args.trade_penalty))
        nets.append(onet)
        pfs.append(pf)
        dds.append(dd)
        trades_n.append(trn)
        scores.append(sc)
        skips.append(float(sv.skip_rate))
        edges.append(float(sv.edge_decay))
        if onet > 0:
            pos += 1

    total = len(split_ids)
    bad_stats = CandidateStats(
        cid=cid,
        valid=False,
        invalid_reason=invalid_reason or "INCOMPLETE_SPLITS",
        median_val_score=-1e15,
        iqr_val_score=0.0,
        median_val_net=-1e9,
        median_val_pf=0.0,
        median_val_dd=1.0,
        median_val_trades=0.0,
        median_skip_rate=1.0,
        median_edge_decay=-1e9,
        stability_pct=0.0,
        pos_splits=0,
        total_splits=total,
    )
    if invalid_reason or len(scores) < total:
        return CandidateEvalResult(
            valid=False,
            score=float(-1e15),
            config=dict(candidate),
            metrics=_candidate_metrics_from_stats(bad_stats),
            reason_if_invalid=str(bad_stats.invalid_reason),
            stats=bad_stats,
        )

    med_score = float(np.median(scores))
    iqr_score = float(np.percentile(scores, 75) - np.percentile(scores, 25))
    med_net = float(np.median(nets))
    med_pf = float(np.median(pfs))
    med_dd = float(np.median(dds))
    med_tr = float(np.median(trades_n))
    med_skip = float(np.median(skips))
    med_edge = float(np.median(edges))
    stability = float(pos / max(1, total))
    fail_reasons: List[str] = []
    if stability < float(args.stability_min):
        fail_reasons.append("stability_fail")
    if med_pf < 1.20:
        fail_reasons.append("pf_fail")
    if med_dd > 0.25:
        fail_reasons.append("dd_fail")
    if med_edge < float(args.edge_decay_min):
        fail_reasons.append("edge_decay_fail")
    eligible = len(fail_reasons) == 0
    st = CandidateStats(
        cid=cid,
        valid=eligible,
        invalid_reason="" if eligible else _constraint_reason_text(fail_reasons),
        median_val_score=med_score,
        iqr_val_score=iqr_score,
        median_val_net=med_net,
        median_val_pf=med_pf,
        median_val_dd=med_dd,
        median_val_trades=med_tr,
        median_skip_rate=med_skip,
        median_edge_decay=med_edge,
        stability_pct=stability * 100.0,
        pos_splits=pos,
        total_splits=total,
    )
    return CandidateEvalResult(
        valid=bool(st.valid),
        score=float(st.median_val_score),
        config=dict(candidate),
        metrics=_candidate_metrics_from_stats(st),
        reason_if_invalid=str(st.invalid_reason),
        stats=st,
    )


def _run_ultra_for_symbol_fast(symbol: str, args: argparse.Namespace, run_tag: str) -> Dict[str, Any]:
    symbol = symbol.upper()
    paths = _artifact_paths(symbol)
    args_hash = _args_hash(args, symbol)
    run_started = time.monotonic()
    profile = _ProfileBreakdown()

    p_base, ppath = _load_params(symbol, args.btc_params)
    d = _ensure_indicators(_load_df(symbol, args.start, args.end), p_base)
    pre, test = _split_outer(d, args.test_start, args.test_end)
    split_seeds = [int(x.strip()) for x in str(args.split_seeds).split(",") if x.strip()]
    exec_root = (PROJECT_ROOT / "artifacts" / "execution_overlay" / symbol / run_tag).resolve()
    exec_root.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    split_bases, _all_windows = _build_split_bases(
        symbol=symbol,
        args=args,
        p_base=p_base,
        pre=pre,
        split_seeds=split_seeds,
        run_tag=run_tag,
        exec_root=exec_root,
    )
    profile.opp_precompute_sec = float(time.perf_counter() - t0)
    total_opp = int(sum(sb.opp_count for sb in split_bases))
    _log_info(
        f"opportunity precompute complete splits={len(split_bases)} opportunities={total_opp} elapsed_sec={profile.opp_precompute_sec:.2f}",
        symbol=symbol,
        phase="precompute",
        progress="opportunities",
    )
    if not split_bases:
        raise RuntimeError(f"{symbol}: no inner splits")

    t1 = time.perf_counter()
    _log_info(
        "window summary precompute starting",
        symbol=symbol,
        phase="precompute",
        progress="windows",
    )
    prepared_splits, fetch_prefetch = _prepare_split_1s_summaries(
        symbol=symbol,
        args=args,
        run_tag=run_tag,
        exec_root=exec_root,
        split_bases=split_bases,
    )
    fetch_fail_count = int(fetch_prefetch.get("window_fail_count", 0))
    fetch_fail_reasons: Dict[str, int] = {
        "dns": int(dict(fetch_prefetch.get("reason_hist", {})).get("dns", 0)),
        "429": int(dict(fetch_prefetch.get("reason_hist", {})).get("429", 0)),
        "timeout": int(dict(fetch_prefetch.get("reason_hist", {})).get("timeout", 0)),
        "other": int(dict(fetch_prefetch.get("reason_hist", {})).get("other", 0)),
    }
    fetch_failed_split_count = int(fetch_prefetch.get("failed_split_count", 0))
    fetch_failed_opp_count = int(fetch_prefetch.get("failed_opp_count", 0))
    fetch_fail_log_path = str(fetch_prefetch.get("failed_windows_path", ""))
    profile.window_precompute_sec = float(time.perf_counter() - t1)
    _log_info(
        f"window summary precompute complete splits={len(prepared_splits)} elapsed_sec={profile.window_precompute_sec:.2f}",
        symbol=symbol,
        phase="precompute",
        progress="windows",
    )
    if fetch_fail_count > 0:
        _log_warn(
            (
                f"prefetch had failed windows count={fetch_fail_count} "
                f"dns={fetch_fail_reasons['dns']} timeout={fetch_fail_reasons['timeout']} "
                f"429={fetch_fail_reasons['429']} other={fetch_fail_reasons['other']} "
                f"affected_splits={fetch_failed_split_count} affected_opportunities={fetch_failed_opp_count} "
                f"report={fetch_fail_log_path}"
            ),
            symbol=symbol,
            phase="precompute",
            progress="windows",
        )
    if not prepared_splits:
        raise RuntimeError(f"{symbol}: no prepared splits")

    split_map: Dict[int, _SplitPrepared] = {sp.split_id: sp for sp in prepared_splits}
    stage1_ids, stage2_ids, stage3_ids = _stage_split_ids(prepared_splits, int(args.mc_splits))
    if not stage3_ids:
        raise RuntimeError(f"{symbol}: stage3 split set is empty")

    random_target = int(args.random_samples)
    ga_target = int(args.ga_generations)
    pop_size = int(args.pop_size)

    rng_seed = int(args.seed) + (abs(hash(symbol)) % 100000)
    rng = random.Random(rng_seed)
    np.random.seed(rng_seed & 0xFFFFFFFF)

    split_eval_cache: Dict[Tuple[str, int], _SplitEval] = {}
    cand_map: Dict[str, Dict[str, Any]] = {}
    seen_random: set[str] = set()
    pop: List[Dict[str, Any]] = []
    best_fit = -1e18
    best_c: Optional[Dict[str, Any]] = None
    best_valid_candidate: Optional[Dict[str, Any]] = None
    best_valid_stats: Optional[CandidateStats] = None
    best_valid_score = -1e18
    best_invalid_candidate: Optional[Dict[str, Any]] = None
    best_invalid_score = -1e18
    best_invalid_reason = ""
    stale = 0
    start_gen = 0
    random_invalid_count = 0
    random_constraint_fail_count = 0
    ga_invalid_count = 0
    ga_constraint_fail_count = 0
    best_random_score = -1e15
    current_gen: Optional[int] = None
    invalid_reason_counts: Dict[str, int] = {}
    constraint_fail_reasons: Dict[str, int] = {}
    stage1_rows: List[Dict[str, Any]] = []
    stage2_rows: List[Dict[str, Any]] = []
    stage3_rows: List[Dict[str, Any]] = []

    random_topk = TopKBuffer(cap=_TOPK_CAP)
    overall_topk = TopKBuffer(cap=max(_TOPK_CAP, pop_size * 3))
    random_progress_total = (
        max(1, random_target * max(1, len(stage1_ids)))
        + max(1, int(np.ceil(random_target * 0.15)) * max(1, len(stage2_ids)))
        + max(1, int(np.ceil(random_target * 0.15 * 0.25)) * max(1, len(stage3_ids)))
    )
    random_progress = EvalProgress(symbol=symbol, phase="random", split_eval_total=int(random_progress_total))
    ga_progress = EvalProgress(symbol=symbol, phase="ga", split_eval_total=max(1, ga_target * pop_size * len(stage3_ids)))

    def _row_from_candidate(
        row_type: str,
        stage: str,
        candidate: Dict[str, Any],
        stats: CandidateStats,
        *,
        gen: Optional[int] = None,
        fitness: Optional[float] = None,
    ) -> Dict[str, Any]:
        cid = _candidate_id(candidate)
        row: Dict[str, Any] = {"row_type": row_type, "stage": stage, "cid": cid, **candidate, **stats.__dict__}
        if gen is not None:
            row["gen"] = int(gen)
        if fitness is not None:
            row["fitness"] = float(fitness)
        return row

    def _record_row(row: Dict[str, Any], *, is_random: bool, track_random_topk: bool = False) -> None:
        _append_jsonl(paths.partial_jsonl, row)
        _append_csv_row(paths.random_csv if is_random else paths.ga_csv, row)
        overall_topk.add(row)
        if is_random and track_random_topk:
            random_topk.add(row)
        cid = str(row.get("cid", "")).strip()
        cand = _candidate_from_row(row)
        if cid and cand is not None:
            cand_map[cid] = cand

    def _record_invalid_reason(reason: str) -> None:
        raw = str(reason or "").strip()
        cparts = _constraint_reason_parts(raw)
        if cparts:
            for part in cparts:
                constraint_fail_reasons[part] = int(constraint_fail_reasons.get(part, 0) + 1)
                invalid_reason_counts[part] = int(invalid_reason_counts.get(part, 0) + 1)
            return
        key = _classify_invalid_reason(raw)
        invalid_reason_counts[key] = int(invalid_reason_counts.get(key, 0) + 1)

    def _write_invalid_hist() -> None:
        payload = {
            "symbol": symbol,
            "run_id": run_tag,
            "candidate_index": int(len(seen_random)),
            "invalid_count": int(random_invalid_count),
            "constraint_fail_count": int(random_constraint_fail_count),
            "reject_count": int(random_invalid_count + random_constraint_fail_count),
            "invalid_reasons": {k: int(v) for k, v in sorted(invalid_reason_counts.items(), key=lambda kv: kv[0])},
            "constraint_fail_reasons": {k: int(v) for k, v in sorted(constraint_fail_reasons.items(), key=lambda kv: kv[0])},
            "fetch_fail_count": int(fetch_fail_count),
            "fetch_fail_reasons": {k: int(v) for k, v in sorted(fetch_fail_reasons.items(), key=lambda kv: kv[0])},
            "fetch_failed_split_count": int(fetch_failed_split_count),
            "fetch_failed_opp_count": int(fetch_failed_opp_count),
            "fetch_fail_log_path": str(fetch_fail_log_path),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(paths.invalid_reasons_out, payload)

    def _update_best_random(eval_result: CandidateEvalResult, *, record_invalid_reason: bool = True) -> None:
        nonlocal best_valid_candidate, best_valid_stats, best_valid_score
        nonlocal best_invalid_candidate, best_invalid_score, best_invalid_reason
        nonlocal best_random_score, best_c
        if eval_result.valid:
            s = float(eval_result.score)
            if s > best_valid_score:
                best_valid_score = s
                best_valid_candidate = dict(eval_result.config)
                best_valid_stats = eval_result.stats
                best_c = dict(eval_result.config)
            best_random_score = max(best_random_score, s)
        else:
            s = float(eval_result.score)
            if s > best_invalid_score:
                best_invalid_score = s
                best_invalid_candidate = dict(eval_result.config)
                best_invalid_reason = str(eval_result.reason_if_invalid)
            if record_invalid_reason:
                _record_invalid_reason(eval_result.reason_if_invalid)

    def _write_status(
        *,
        status_phase: str,
        candidate_index: int,
        generation_index: Optional[int],
        invalid_count: int,
        note: str = "",
    ) -> None:
        best_candidate_payload = dict(best_valid_candidate) if isinstance(best_valid_candidate, dict) else None
        best_any_candidate_payload = (
            best_candidate_payload
            if best_candidate_payload is not None
            else (dict(best_invalid_candidate) if isinstance(best_invalid_candidate, dict) else None)
        )
        best_score_payload = (
            float(best_fit)
            if best_fit > -1e17
            else (float(best_valid_score) if best_valid_score > -1e17 else float(best_invalid_score))
        )
        payload = {
            "symbol": symbol,
            "run_id": run_tag,
            "args_hash": args_hash,
            "phase": status_phase,
            "candidate_index": int(candidate_index),
            "generation_index": int(generation_index) if generation_index is not None else None,
            "best_score": float(best_score_payload),
            "best_candidate": best_candidate_payload,
            "best_candidate_summary": _candidate_summary(best_candidate_payload),
            "best_any_candidate": best_any_candidate_payload,
            "best_any_candidate_summary": _candidate_summary(best_any_candidate_payload),
            "best_valid_candidate": best_valid_candidate,
            "best_invalid_candidate": best_invalid_candidate,
            "best_invalid_reason": str(best_invalid_reason),
            "invalid_reason_counts": {k: int(v) for k, v in sorted(invalid_reason_counts.items(), key=lambda kv: kv[0])},
            "constraint_fail_reasons": {k: int(v) for k, v in sorted(constraint_fail_reasons.items(), key=lambda kv: kv[0])},
            "invalid_count": int(invalid_count),
            "constraint_fail_count": int(random_constraint_fail_count if generation_index is None else ga_constraint_fail_count),
            "reject_count": int(
                int(invalid_count)
                + int(random_constraint_fail_count if generation_index is None else ga_constraint_fail_count)
            ),
            "random_invalid_count": int(random_invalid_count),
            "random_constraint_fail_count": int(random_constraint_fail_count),
            "ga_invalid_count": int(ga_invalid_count),
            "ga_constraint_fail_count": int(ga_constraint_fail_count),
            "fetch_fail_count": int(fetch_fail_count),
            "fetch_fail_reasons": {k: int(v) for k, v in sorted(fetch_fail_reasons.items(), key=lambda kv: kv[0])},
            "fetch_failed_split_count": int(fetch_failed_split_count),
            "fetch_failed_opp_count": int(fetch_failed_opp_count),
            "fetch_fail_log_path": str(fetch_fail_log_path),
            "rss_mb": float(_rss_mb()),
            "elapsed_sec": float(time.monotonic() - run_started),
            "opp_precompute_sec": float(profile.opp_precompute_sec),
            "window_precompute_sec": float(profile.window_precompute_sec),
            "candidate_eval_avg_sec": float(profile.candidate_eval_sec_total / profile.candidate_eval_count) if profile.candidate_eval_count > 0 else 0.0,
            "note": str(note),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(paths.status_out, payload)

    def _save_ckpt(
        *,
        ckpt_phase: str,
        candidate_index: int,
        generation_index: Optional[int],
        invalid_count: int,
    ) -> None:
        top_rows = overall_topk.top(max(50, pop_size))
        random_rows = random_topk.top(max(50, pop_size))
        top_rows_valid = [r for r in top_rows if bool(r.get("valid", False))][: max(50, pop_size)]
        ckpt_best_cfg = (
            dict(best_valid_candidate)
            if isinstance(best_valid_candidate, dict)
            else (
                dict(best_c)
                if isinstance(best_c, dict)
                else (_candidate_only(top_rows_valid[0]) if top_rows_valid else (_candidate_only(top_rows[0]) if top_rows else None))
            )
        )
        ckpt_best_score = (
            float(best_fit)
            if best_fit > -1e17
            else (float(best_valid_score) if best_valid_score > -1e17 else float(best_invalid_score))
        )
        payload = {
            "symbol": symbol,
            "run_id": run_tag,
            "args_hash": args_hash,
            "phase": ckpt_phase,
            "candidate_index": int(candidate_index),
            "generation_index": int(generation_index) if generation_index is not None else None,
            "best_config": ckpt_best_cfg,
            "best_score": float(ckpt_best_score),
            "best_valid_candidate": best_valid_candidate,
            "best_valid_score": float(best_valid_score),
            "best_invalid_candidate": best_invalid_candidate,
            "best_invalid_score": float(best_invalid_score),
            "best_invalid_reason": str(best_invalid_reason),
            "top_k": top_rows,
            "top_k_valid": top_rows_valid,
            "random_top_k": random_rows,
            "seen_cids": sorted(seen_random),
            "population": pop,
            "stale": int(stale),
            "random_invalid_count": int(random_invalid_count),
            "random_constraint_fail_count": int(random_constraint_fail_count),
            "ga_invalid_count": int(ga_invalid_count),
            "ga_constraint_fail_count": int(ga_constraint_fail_count),
            "fetch_fail_count": int(fetch_fail_count),
            "fetch_fail_reasons": {k: int(v) for k, v in sorted(fetch_fail_reasons.items(), key=lambda kv: kv[0])},
            "fetch_failed_split_count": int(fetch_failed_split_count),
            "fetch_failed_opp_count": int(fetch_failed_opp_count),
            "fetch_fail_log_path": str(fetch_fail_log_path),
            "split_eval_count_random": int(random_progress.split_eval_count),
            "split_eval_count_ga": int(ga_progress.split_eval_count),
            "invalid_count": int(invalid_count),
            "constraint_fail_count": int(random_constraint_fail_count if generation_index is None else ga_constraint_fail_count),
            "partial_results_path": str(paths.partial_jsonl.resolve()),
            "rng_state_python": _pack_py_random_state(rng.getstate()),
            "rng_state_numpy": _pack_np_random_state(np.random.get_state()),
            "opp_precompute_sec": float(profile.opp_precompute_sec),
            "window_precompute_sec": float(profile.window_precompute_sec),
            "candidate_eval_sec_total": float(profile.candidate_eval_sec_total),
            "candidate_eval_count": int(profile.candidate_eval_count),
            "invalid_reason_counts": {k: int(v) for k, v in sorted(invalid_reason_counts.items(), key=lambda kv: kv[0])},
            "constraint_fail_reasons": {k: int(v) for k, v in sorted(constraint_fail_reasons.items(), key=lambda kv: kv[0])},
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(paths.ckpt_out, payload)

    def _load_existing_stage_rows() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        s1: List[Dict[str, Any]] = []
        s2: List[Dict[str, Any]] = []
        s3: List[Dict[str, Any]] = []
        if not paths.random_csv.exists() or paths.random_csv.stat().st_size <= 0:
            return s1, s2, s3
        try:
            df = pd.read_csv(paths.random_csv)
        except Exception:
            return s1, s2, s3
        if df.empty:
            return s1, s2, s3
        rows = df.to_dict("records")
        for row in rows:
            cid = str(row.get("cid", "")).strip()
            cand = _candidate_from_row(row)
            if not cid or cand is None:
                continue
            stage = str(row.get("stage", "")).lower()
            row_type = str(row.get("row_type", "")).lower()
            if stage == "stage1" or row_type in {"random", "random_s1"}:
                s1.append(row)
            elif stage == "stage2" or row_type == "random_s2":
                s2.append(row)
            elif stage == "stage3" or row_type == "random_s3":
                s3.append(row)
            cand_map[cid] = cand
        return s1, s2, s3

    ckpt = None if bool(args.fresh) else _read_json(paths.ckpt_out)
    if ckpt and ckpt.get("symbol") == symbol and ckpt.get("args_hash") == args_hash:
        phase = str(ckpt.get("phase", "random")).lower()
        run_tag = str(ckpt.get("run_id", run_tag))
        random_invalid_count = int(ckpt.get("random_invalid_count", ckpt.get("invalid_count", 0)))
        random_constraint_fail_count = int(ckpt.get("random_constraint_fail_count", ckpt.get("constraint_fail_count", 0)))
        ga_invalid_count = int(ckpt.get("ga_invalid_count", 0))
        ga_constraint_fail_count = int(ckpt.get("ga_constraint_fail_count", 0))
        random_progress.split_eval_count = int(ckpt.get("split_eval_count_random", 0))
        ga_progress.split_eval_count = int(ckpt.get("split_eval_count_ga", 0))
        stale = int(ckpt.get("stale", 0))
        profile.opp_precompute_sec = float(ckpt.get("opp_precompute_sec", profile.opp_precompute_sec))
        profile.window_precompute_sec = float(ckpt.get("window_precompute_sec", profile.window_precompute_sec))
        profile.candidate_eval_sec_total = float(ckpt.get("candidate_eval_sec_total", profile.candidate_eval_sec_total))
        profile.candidate_eval_count = int(ckpt.get("candidate_eval_count", profile.candidate_eval_count))
        invalid_reason_counts = {str(k): int(v) for k, v in dict(ckpt.get("invalid_reason_counts", {})).items()}
        constraint_fail_reasons = {str(k): int(v) for k, v in dict(ckpt.get("constraint_fail_reasons", {})).items()}
        fetch_fail_count = int(ckpt.get("fetch_fail_count", fetch_fail_count))
        loaded_fetch_reasons = dict(ckpt.get("fetch_fail_reasons", {}))
        fetch_fail_reasons = {
            "dns": int(loaded_fetch_reasons.get("dns", fetch_fail_reasons.get("dns", 0))),
            "429": int(loaded_fetch_reasons.get("429", fetch_fail_reasons.get("429", 0))),
            "timeout": int(loaded_fetch_reasons.get("timeout", fetch_fail_reasons.get("timeout", 0))),
            "other": int(loaded_fetch_reasons.get("other", fetch_fail_reasons.get("other", 0))),
        }
        fetch_failed_split_count = int(ckpt.get("fetch_failed_split_count", fetch_failed_split_count))
        fetch_failed_opp_count = int(ckpt.get("fetch_failed_opp_count", fetch_failed_opp_count))
        fetch_fail_log_path = str(ckpt.get("fetch_fail_log_path", fetch_fail_log_path))
        if isinstance(ckpt.get("best_config"), dict):
            best_c = dict(ckpt["best_config"])
        if isinstance(ckpt.get("best_valid_candidate"), dict):
            best_valid_candidate = dict(ckpt["best_valid_candidate"])
            best_c = dict(ckpt["best_valid_candidate"]) if best_c is None else best_c
        if ckpt.get("best_valid_score") is not None:
            best_valid_score = float(ckpt.get("best_valid_score", best_valid_score))
            best_random_score = max(best_random_score, best_valid_score)
        if isinstance(ckpt.get("best_invalid_candidate"), dict):
            best_invalid_candidate = dict(ckpt["best_invalid_candidate"])
        if ckpt.get("best_invalid_score") is not None:
            best_invalid_score = float(ckpt.get("best_invalid_score", best_invalid_score))
        best_invalid_reason = str(ckpt.get("best_invalid_reason", best_invalid_reason))
        if ckpt.get("best_score") is not None:
            try:
                best_fit = float(ckpt["best_score"])
            except Exception:
                best_fit = -1e18
        pop = [dict(x) for x in ckpt.get("population", []) if isinstance(x, dict)]
        seen_random = {str(x) for x in ckpt.get("seen_cids", []) if str(x)}
        for row in [r for r in ckpt.get("top_k", []) if isinstance(r, dict)]:
            overall_topk.add(row)
            cid = str(row.get("cid", "")).strip()
            cand = _candidate_from_row(row)
            if cid and cand is not None:
                cand_map[cid] = cand
        for row in [r for r in ckpt.get("top_k_valid", []) if isinstance(r, dict)]:
            overall_topk.add(row)
            cid = str(row.get("cid", "")).strip()
            cand = _candidate_from_row(row)
            if cid and cand is not None:
                cand_map[cid] = cand
        for row in [r for r in ckpt.get("random_top_k", []) if isinstance(r, dict)]:
            random_topk.add(row)
            overall_topk.add(row)
            cid = str(row.get("cid", "")).strip()
            cand = _candidate_from_row(row)
            if cid and cand is not None:
                cand_map[cid] = cand
        stage1_rows, stage2_rows, stage3_rows = _load_existing_stage_rows()
        if stage1_rows or stage2_rows or stage3_rows:
            for row in stage1_rows + stage2_rows + stage3_rows:
                overall_topk.add(row)
                stage = str(row.get("stage", "")).lower()
                row_type = str(row.get("row_type", "")).lower()
                if stage == "stage3" or row_type == "random_s3":
                    random_topk.add(row)

            stage1_cids = {
                str(row.get("cid", "")).strip()
                for row in stage1_rows
                if str(row.get("cid", "")).strip()
            }
            seen_random.update(stage1_cids)

            derived_random_invalid = 0
            derived_random_constraint = 0
            derived_invalid_reason_counts: Dict[str, int] = {}
            derived_constraint_fail_reasons: Dict[str, int] = {}
            for row in stage1_rows:
                cid = str(row.get("cid", "")).strip()
                if not cid:
                    continue
                cand = cand_map.get(cid)
                if cand is None:
                    continue
                valid_raw = row.get("valid", False)
                valid = _parse_bool(valid_raw) if isinstance(valid_raw, str) else bool(valid_raw)
                score = float(row.get("median_val_score", -1e15))
                reason = str(row.get("invalid_reason", ""))
                if valid:
                    if score > best_valid_score:
                        best_valid_score = score
                        best_valid_candidate = dict(cand)
                        best_c = dict(cand)
                        st_loaded = _stats_from_row(row)
                        if st_loaded is not None:
                            best_valid_stats = st_loaded
                    best_random_score = max(best_random_score, score)
                else:
                    if score > best_invalid_score:
                        best_invalid_score = score
                        best_invalid_candidate = dict(cand)
                        best_invalid_reason = reason
                    cparts = _constraint_reason_parts(reason)
                    if cparts:
                        derived_random_constraint += 1
                        for part in cparts:
                            derived_constraint_fail_reasons[part] = int(derived_constraint_fail_reasons.get(part, 0) + 1)
                            derived_invalid_reason_counts[part] = int(derived_invalid_reason_counts.get(part, 0) + 1)
                    else:
                        derived_random_invalid += 1
                        key = _classify_invalid_reason(reason)
                        derived_invalid_reason_counts[key] = int(derived_invalid_reason_counts.get(key, 0) + 1)

            random_invalid_count = max(random_invalid_count, derived_random_invalid)
            random_constraint_fail_count = max(random_constraint_fail_count, derived_random_constraint)
            for k, v in derived_invalid_reason_counts.items():
                invalid_reason_counts[k] = max(int(invalid_reason_counts.get(k, 0)), int(v))
            for k, v in derived_constraint_fail_reasons.items():
                constraint_fail_reasons[k] = max(int(constraint_fail_reasons.get(k, 0)), int(v))

            if best_valid_candidate is not None and best_c is None:
                best_c = dict(best_valid_candidate)
            if best_valid_score > -1e17:
                best_random_score = max(best_random_score, best_valid_score)

        try:
            if isinstance(ckpt.get("rng_state_python"), dict):
                rng.setstate(_unpack_py_random_state(ckpt["rng_state_python"]))
            if isinstance(ckpt.get("rng_state_numpy"), dict):
                np.random.set_state(_unpack_np_random_state(ckpt["rng_state_numpy"]))
        except Exception:
            _log_warn("failed to restore RNG state from checkpoint", symbol=symbol, phase="resume", progress="-")
        if phase.startswith("ga") and pop:
            start_gen = max(0, int(ckpt.get("generation_index", -1)) + 1)
            _log_info(
                f"resuming from checkpoint phase={phase} random_done={len(seen_random)} start_gen={start_gen}",
                symbol=symbol,
                phase="resume",
                progress="-",
            )
        else:
            _log_info(
                f"resuming random phase from checkpoint phase={phase} candidate_index={len(seen_random)}",
                symbol=symbol,
                phase="resume",
                progress="-",
            )
            pop = []
            start_gen = 0
    else:
        if bool(args.fresh):
            _log_info("fresh run requested; ignoring checkpoints", symbol=symbol, phase="resume", progress="-")
        elif ckpt is not None:
            _log_info("checkpoint args mismatch; starting fresh", symbol=symbol, phase="resume", progress="-")
        _atomic_write_text(paths.partial_jsonl, "")
        _atomic_write_text(paths.random_csv, "")
        _atomic_write_text(paths.ga_csv, "")
        stage1_rows = []
        stage2_rows = []
        stage3_rows = []

    random_started = time.monotonic()
    last_random_status_ts = time.monotonic()

    while len(seen_random) < random_target:
        c = _random_candidate(rng)
        cid = _candidate_id(c)
        if cid in seen_random:
            continue
        seen_random.add(cid)
        cand_map[cid] = dict(c)
        res1 = _evaluate_candidate_fast(
            symbol=symbol,
            candidate=c,
            split_ids=stage1_ids,
            split_map=split_map,
            split_cache=split_eval_cache,
            args=args,
            p_base=p_base,
            progress=random_progress,
            profile=profile,
        )
        st1 = res1.stats
        row = _row_from_candidate("random_s1", "stage1", c, st1)
        stage1_rows.append(row)
        _record_row(row, is_random=True, track_random_topk=False)
        _update_best_random(res1, record_invalid_reason=True)
        if not res1.valid:
            if _constraint_reason_parts(res1.reason_if_invalid):
                random_constraint_fail_count += 1
            else:
                random_invalid_count += 1

        done = len(seen_random)
        if done % 10 == 0 or done == random_target:
            elapsed = time.monotonic() - random_started
            eta = (elapsed / done) * max(0, random_target - done) if done > 0 else 0.0
            _write_invalid_hist()
            _log_info(
                (
                    f"random_s1 heartbeat best_median_score={best_random_score:.6f} "
                    f"invalid_count={random_invalid_count} constraint_fail_count={random_constraint_fail_count} "
                    f"elapsed_sec={elapsed:.1f} eta_sec={eta:.1f}"
                ),
                symbol=symbol,
                phase="random",
                progress=f"{done}/{random_target}",
            )
            _log_info(
                f"invalid reasons: {json.dumps(dict(sorted(invalid_reason_counts.items())), sort_keys=True)}",
                symbol=symbol,
                phase="random",
                progress=f"{done}/{random_target}",
            )
            _log_info(
                f"constraint_fail_reasons={json.dumps(dict(sorted(constraint_fail_reasons.items())), sort_keys=True, separators=(',', ':'))}",
                symbol=symbol,
                phase="random",
                progress=f"{done}/{random_target}",
            )
            _memory_guard(args, symbol=symbol, phase="random", progress=f"{done}/{random_target}")
        now = time.monotonic()
        if now - last_random_status_ts >= 120.0 or done == random_target:
            _write_status(
                status_phase="random_stage1",
                candidate_index=done,
                generation_index=None,
                invalid_count=random_invalid_count,
            )
            last_random_status_ts = now
        if done % 20 == 0 or done == random_target:
            _save_ckpt(
                ckpt_phase="random_stage1",
                candidate_index=done,
                generation_index=None,
                invalid_count=random_invalid_count,
            )

    sorted_stage1 = _sort_search_rows(stage1_rows)
    keep1 = max(1, int(np.ceil(len(sorted_stage1) * 0.15)))
    survivors1 = sorted_stage1[:keep1]
    _log_info(
        f"stage1 complete candidates={len(stage1_rows)} survivors={len(survivors1)}",
        symbol=symbol,
        phase="random",
        progress=f"{len(stage1_rows)}/{len(stage1_rows)}",
    )

    completed_stage2_cids = {str(r.get("cid", "")).strip() for r in stage2_rows if str(r.get("cid", "")).strip()}
    for idx, row1 in enumerate(survivors1, start=1):
        cid = str(row1["cid"])
        if cid in completed_stage2_cids:
            continue
        c = cand_map.get(cid)
        if c is None:
            continue
        res2 = _evaluate_candidate_fast(
            symbol=symbol,
            candidate=c,
            split_ids=stage2_ids if stage2_ids else stage3_ids,
            split_map=split_map,
            split_cache=split_eval_cache,
            args=args,
            p_base=p_base,
            progress=random_progress,
            profile=profile,
        )
        st2 = res2.stats
        row2 = _row_from_candidate("random_s2", "stage2", c, st2)
        stage2_rows.append(row2)
        completed_stage2_cids.add(cid)
        _record_row(row2, is_random=True, track_random_topk=False)
        _update_best_random(res2, record_invalid_reason=False)
        if idx % 10 == 0 or idx == len(survivors1):
            _write_invalid_hist()
            _log_info(
                (
                    "random_s2 heartbeat "
                    f"best_median_score={best_random_score:.6f} "
                    f"invalid_count={random_invalid_count} constraint_fail_count={random_constraint_fail_count}"
                ),
                symbol=symbol,
                phase="random",
                progress=f"{idx}/{len(survivors1)}",
            )
            _memory_guard(args, symbol=symbol, phase="random", progress=f"{idx}/{len(survivors1)}")
        if idx % 20 == 0 or idx == len(survivors1):
            _save_ckpt(
                ckpt_phase="random_stage2",
                candidate_index=len(seen_random),
                generation_index=None,
                invalid_count=random_invalid_count,
            )

    sorted_stage2 = _sort_search_rows(stage2_rows)
    keep2 = max(1, int(np.ceil(len(sorted_stage2) * 0.25)))
    survivors2 = sorted_stage2[:keep2]
    _log_info(
        f"stage2 complete candidates={len(stage2_rows)} survivors={len(survivors2)}",
        symbol=symbol,
        phase="random",
        progress=f"{len(stage2_rows)}/{len(stage2_rows)}",
    )

    completed_stage3_cids = {str(r.get("cid", "")).strip() for r in stage3_rows if str(r.get("cid", "")).strip()}
    for idx, row2 in enumerate(survivors2, start=1):
        cid = str(row2["cid"])
        if cid in completed_stage3_cids:
            continue
        c = cand_map.get(cid)
        if c is None:
            continue
        res3 = _evaluate_candidate_fast(
            symbol=symbol,
            candidate=c,
            split_ids=stage3_ids,
            split_map=split_map,
            split_cache=split_eval_cache,
            args=args,
            p_base=p_base,
            progress=random_progress,
            profile=profile,
        )
        st3 = res3.stats
        row3 = _row_from_candidate("random_s3", "stage3", c, st3)
        stage3_rows.append(row3)
        completed_stage3_cids.add(cid)
        _record_row(row3, is_random=True, track_random_topk=True)
        _update_best_random(res3, record_invalid_reason=False)
        if idx % 10 == 0 or idx == len(survivors2):
            _write_invalid_hist()
            _log_info(
                (
                    "random_s3 heartbeat "
                    f"best_median_score={best_random_score:.6f} "
                    f"invalid_count={random_invalid_count} constraint_fail_count={random_constraint_fail_count}"
                ),
                symbol=symbol,
                phase="random",
                progress=f"{idx}/{len(survivors2)}",
            )
            _memory_guard(args, symbol=symbol, phase="random", progress=f"{idx}/{len(survivors2)}")
            _write_status(
                status_phase="random_stage3",
                candidate_index=len(seen_random),
                generation_index=None,
                invalid_count=random_invalid_count,
            )
        if idx % 20 == 0 or idx == len(survivors2):
            _save_ckpt(
                ckpt_phase="random_stage3",
                candidate_index=len(seen_random),
                generation_index=None,
                invalid_count=random_invalid_count,
            )

    stage3_sorted = _sort_search_rows(stage3_rows) if stage3_rows else _sort_search_rows(stage2_rows if stage2_rows else stage1_rows)
    if not stage3_sorted:
        raise RuntimeError(f"{symbol}: no candidates available for GA")

    if not pop:
        elites_for_seed = stage3_sorted[:50]
        for row in elites_for_seed:
            cid = str(row["cid"])
            if cid in cand_map:
                pop.append(dict(cand_map[cid]))
        while len(pop) < pop_size:
            pop.append(_random_candidate(rng))
        pop = pop[:pop_size]

    if best_valid_candidate is not None:
        best_c = dict(best_valid_candidate)
    elif best_c is None and stage3_sorted:
        best_c = dict(cand_map[str(stage3_sorted[0]["cid"])])
    if best_fit <= -1e17:
        if best_valid_score > -1e17:
            best_fit = float(best_valid_score)
        elif stage3_sorted:
            best_fit = float(stage3_sorted[0].get("median_val_score", -1e15))

    ga_started = time.monotonic()
    for gen in range(start_gen, ga_target):
        current_gen = gen
        scored: List[Tuple[float, Dict[str, Any], CandidateStats]] = []
        for c in pop:
            res_g = _evaluate_candidate_fast(
                symbol=symbol,
                candidate=c,
                split_ids=stage3_ids,
                split_map=split_map,
                split_cache=split_eval_cache,
                args=args,
                p_base=p_base,
                progress=ga_progress,
                profile=profile,
            )
            st = res_g.stats
            fit = _fitness(st, float(args.target_skip_rate), float(args.skip_penalty))
            scored.append((fit, c, st))
        scored.sort(key=lambda x: x[0], reverse=True)
        bf, bc, bst = scored[0]
        gen_invalid_count = sum(1 for _, _, st in scored if (not st.valid and not _constraint_reason_parts(st.invalid_reason)))
        ga_constraint_fail_count = sum(1 for _, _, st in scored if (not st.valid and _constraint_reason_parts(st.invalid_reason)))
        ga_invalid_count = gen_invalid_count
        median_val_score = float(np.median([st.median_val_score for _, _, st in scored])) if scored else -1e15
        row = _row_from_candidate("ga", "ga", bc, bst, gen=gen, fitness=bf)
        _record_row(row, is_random=False, track_random_topk=False)

        if bf > best_fit:
            best_fit = bf
            best_c = dict(bc)
            stale = 0
        else:
            stale += 1

        _log_info(
            (
                f"ga heartbeat best_fitness={bf:.6f} median_val_score={median_val_score:.6f} "
                f"invalid_count={gen_invalid_count} elapsed_sec={time.monotonic() - ga_started:.1f}"
            ),
            symbol=symbol,
            phase="ga",
            progress=f"{gen + 1}/{ga_target}",
        )
        _memory_guard(args, symbol=symbol, phase="ga", progress=f"{gen + 1}/{ga_target}")

        if (gen + 1) % 10 == 0:
            _write_status(
                status_phase="ga",
                candidate_index=len(seen_random),
                generation_index=gen,
                invalid_count=gen_invalid_count,
            )

        stop_now = stale >= int(args.early_stop)
        if not stop_now:
            elites = [x[1] for x in scored[: max(8, pop_size // 5)]]
            new_pop = [dict(e) for e in elites]
            while len(new_pop) < pop_size:
                p1 = rng.choice(elites)
                p2 = rng.choice(elites)
                ch = _cross(p1, p2, rng)
                ch = _mutate(ch, rng, rate=0.35)
                new_pop.append(ch)
            pop = new_pop[:pop_size]

        _save_ckpt(
            ckpt_phase="ga",
            candidate_index=len(seen_random),
            generation_index=gen,
            invalid_count=gen_invalid_count,
        )
        if stop_now:
            _log_info(
                f"ga early stop triggered after {stale} stale generations",
                symbol=symbol,
                phase="ga",
                progress=f"{gen + 1}/{ga_target}",
            )
            break

    if best_c is None:
        top = overall_topk.top(1)
        if top:
            best_c = _candidate_only(top[0])
    if best_c is None:
        raise RuntimeError(f"{symbol}: unable to resolve best candidate")

    best_res = _evaluate_candidate_fast(
        symbol=symbol,
        candidate=best_c,
        split_ids=stage3_ids,
        split_map=split_map,
        split_cache=split_eval_cache,
        args=args,
        p_base=p_base,
        progress=ga_progress,
        profile=profile,
    )
    best_st = best_res.stats

    p_test = dict(p_base)
    p_test["equity_sizing_cap"] = float(args.initial_equity) * float(best_c["cap_mult"])
    ov_cfg = _overlay_cfg_from_candidate(best_c, args)
    try:
        tr_b, m_b = run_backtest_long_only(
            df=test,
            symbol=symbol,
            p=p_test,
            initial_equity=float(args.initial_equity),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            collect_trades=True,
            assume_prepared=True,
        )
        out_t = run_entry_overlay_backtest_from_df(
            symbol=symbol,
            df=test,
            p=p_test,
            cfg=ov_cfg,
            initial_equity=float(args.initial_equity),
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            baseline_trades=pd.DataFrame(tr_b),
            fetch_log_path=str((PROJECT_ROOT / "artifacts" / "execution_overlay" / symbol / run_tag / "ultra_test.jsonl").resolve()),
            assume_prepared=True,
        )
        m_o = out_t["metrics"]
        dbg_t = out_t["debug"]
        bnet = float(m_b.get("net_profit", 0.0))
        onet = float(m_o.get("net_profit", 0.0))
        edge_t = float(onet / bnet) if bnet > 0 else (float(onet / abs(bnet)) if bnet < 0 else 0.0)
    except MemoryError:
        raise
    except Exception:
        m_b = {"net_profit": 0.0, "profit_factor": 0.0, "max_dd": 1.0}
        m_o = {"net_profit": -1e9, "profit_factor": 0.0, "max_dd": 1.0}
        dbg_t = {"skip_rate": 1.0}
        edge_t = -1e9

    pass_fail = bool(
        edge_t >= float(args.edge_decay_min)
        and best_st.stability_pct >= float(args.stability_min) * 100.0
        and float(m_o.get("profit_factor", 0.0)) >= 1.20
        and float(m_o.get("max_dd", 1.0)) <= 0.25
    )

    top_rows = overall_topk.top(200)
    if top_rows:
        pd.DataFrame(top_rows).to_csv(paths.search_out, index=False)
    else:
        _atomic_write_text(paths.search_out, "")

    best_payload = {"symbol": symbol, "run_id": run_tag, "candidate": best_c, "validation": best_st.__dict__}
    _write_json(paths.best_out, best_payload)
    test_payload = {
        "symbol": symbol,
        "run_id": run_tag,
        "candidate": best_c,
        "baseline_test": m_b,
        "overlay_test": m_o,
        "edge_decay_test": edge_t,
        "skip_rate_test": float(dbg_t.get("skip_rate", 0.0)),
        "fetch_fail_count": int(fetch_fail_count),
        "fetch_fail_reasons": {k: int(v) for k, v in sorted(fetch_fail_reasons.items(), key=lambda kv: kv[0])},
        "fetch_failed_split_count": int(fetch_failed_split_count),
        "fetch_failed_opp_count": int(fetch_failed_opp_count),
        "fetch_fail_log_path": str(fetch_fail_log_path),
        "pass": pass_fail,
    }
    _write_json(paths.test_out, test_payload)
    _write_status(
        status_phase="done",
        candidate_index=len(seen_random),
        generation_index=current_gen,
        invalid_count=ga_invalid_count if current_gen is not None else random_invalid_count,
        note="completed",
    )

    if _parse_bool(args.save_active):
        _save_active_configs(
            symbol=symbol,
            strategy_params_path=ppath,
            candidate=best_c,
            args=args,
            run_tag=run_tag,
            test_payload=test_payload,
            select_payload=best_payload["validation"],
        )

    if bool(getattr(args, "profile", False)):
        avg_eval = float(profile.candidate_eval_sec_total / profile.candidate_eval_count) if profile.candidate_eval_count > 0 else 0.0
        _print_line(
            (
                f"profile symbol={symbol} opp_precompute_sec={profile.opp_precompute_sec:.3f} "
                f"window_precompute_sec={profile.window_precompute_sec:.3f} "
                f"candidate_eval_avg_sec={avg_eval:.6f} eval_calls={profile.candidate_eval_count}"
            )
        )
        _log_info(
            (
                f"profile opp_precompute_sec={profile.opp_precompute_sec:.3f} "
                f"window_precompute_sec={profile.window_precompute_sec:.3f} "
                f"candidate_eval_avg_sec={avg_eval:.6f} eval_calls={profile.candidate_eval_count}"
            ),
            symbol=symbol,
            phase="profile",
            progress="-",
        )

    _log_info(f"symbol completed in {time.monotonic() - run_started:.1f}s", symbol=symbol, phase="done", progress="-")
    return {
        "symbol": symbol,
        "run_id": run_tag,
        "baseline_test_net": float(m_b.get("net_profit", 0.0)),
        "overlay_test_net": float(m_o.get("net_profit", 0.0)),
        "baseline_test_pf": float(m_b.get("profit_factor", 0.0)),
        "overlay_test_pf": float(m_o.get("profit_factor", 0.0)),
        "baseline_test_dd": float(m_b.get("max_dd", 1.0)),
        "overlay_test_dd": float(m_o.get("max_dd", 1.0)),
        "stability_pct": float(best_st.stability_pct),
        "skip_rate": float(dbg_t.get("skip_rate", 0.0)),
        "cap_mult": int(best_c["cap_mult"]),
        "policy": str(best_c["overlay_policy"]),
        "mode": str(best_c["overlay_mode"]),
        "behavior": str(best_c["overlay_behavior"]),
        "fetch_fail_count": int(fetch_fail_count),
        "fetch_fail_dns": int(fetch_fail_reasons.get("dns", 0)),
        "fetch_fail_429": int(fetch_fail_reasons.get("429", 0)),
        "fetch_fail_timeout": int(fetch_fail_reasons.get("timeout", 0)),
        "fetch_fail_other": int(fetch_fail_reasons.get("other", 0)),
        "PASS/FAIL": "PASS" if pass_fail else "FAIL",
    }


def _run_symbol_worker(symbol: str, args: argparse.Namespace, run_tag: str) -> Dict[str, Any]:
    if not logging.getLogger().handlers:
        _configure_logging()
    try:
        return _run_ultra_for_symbol_fast(symbol=symbol, args=args, run_tag=run_tag)
    except MemoryError:
        _LOG.exception("symbol worker memory error", extra={"symbol": symbol, "phase": "worker", "progress": "-"})
        raise
    except Exception:
        _LOG.exception("symbol worker failed", extra={"symbol": symbol, "phase": "worker", "progress": "-"})
        raise


def _write_crash_report(symbol: str, run_tag: str, tb: str) -> Path:
    cp = _artifact_paths(symbol)
    crash_payload = (
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"symbol={symbol}\n"
        f"run_id={run_tag}\n"
        f"last_checkpoint_path={cp.ckpt_out.resolve()}\n\n"
        f"{tb}"
    )
    _atomic_write_text(cp.crash_out, crash_payload)
    return cp.crash_out


def main() -> None:
    _configure_logging()
    _print_line(f"startup_utc={datetime.now(timezone.utc).isoformat()}")
    _print_line(f"argv={sys.argv}")

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ADAUSDT,AVAXUSDT,SOLUSDT")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--initial-equity", type=float, default=100.0)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--btc-params", default="data/metadata/params/BTCUSDT_C13_active_params.json")
    ap.add_argument("--cache-root", default="data/processed/execution_1s")
    ap.add_argument("--cache-cap-gb", type=float, default=20.0)
    ap.add_argument("--fetch-workers", type=int, default=2)
    ap.add_argument("--eval-procs", type=int, default=1)
    ap.add_argument("--symbols-mode", choices=["sequential", "parallel"], default="sequential")
    ap.add_argument("--max-cache-mb", type=float, default=256.0)
    ap.add_argument("--max-rss-mb", type=float, default=0.0)
    ap.add_argument("--cache-1s", default="true")
    ap.add_argument("--search", choices=["grid", "ultra"], default="ultra")
    ap.add_argument("--mc-splits", type=int, default=30)
    ap.add_argument("--split-seeds", default="101,202,303")
    ap.add_argument("--random-samples", type=int, default=2000)
    ap.add_argument("--ga-generations", type=int, default=250)
    ap.add_argument("--pop-size", type=int, default=80)
    ap.add_argument("--early-stop", type=int, default=60)
    ap.add_argument("--train-days", type=int, default=540)
    ap.add_argument("--val-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=180)
    ap.add_argument("--test-start", default="2024-01-01")
    ap.add_argument("--test-end", default="2025-12-31")
    ap.add_argument("--dd-penalty", type=float, default=0.45)
    ap.add_argument("--trade-penalty", type=float, default=0.8)
    ap.add_argument("--edge-decay-min", type=float, default=0.75)
    ap.add_argument("--stability-min", type=float, default=0.75)
    ap.add_argument("--target-skip-rate", type=float, default=0.25)
    ap.add_argument("--skip-penalty", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-active", default="true")
    ap.add_argument("--fresh", action="store_true", help="Ignore matching checkpoints and restart symbol search from scratch.")
    ap.add_argument("--profile", action="store_true", help="Print runtime breakdown for precompute and candidate evaluation.")
    args = ap.parse_args()
    args.cache_1s = _parse_bool(args.cache_1s)
    args.fetch_workers = max(1, int(args.fetch_workers))
    args.eval_procs = max(1, int(args.eval_procs))
    args.max_cache_mb = max(0.0, float(args.max_cache_mb))
    args.max_rss_mb = float(args.max_rss_mb)
    total_ram_mb = _total_ram_mb()
    args._effective_max_rss_mb = (
        float(args.max_rss_mb)
        if float(args.max_rss_mb) > 0.0
        else (float(total_ram_mb) * 0.70 if total_ram_mb > 0.0 else 6000.0)
    )

    if abs(float(args.initial_equity) - 100.0) > 1e-9:
        raise RuntimeError("initial_equity must be 100.0")

    syms = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    run_tag = _utc_tag()
    _print_line(f"parsed_args={json.dumps(vars(args), sort_keys=True, default=str)}")
    _print_line(f"project_root={PROJECT_ROOT.resolve()}")
    _print_line(f"run_id={run_tag}")
    _print_line(f"symbols={syms}")
    _print_line(f"python={sys.version.replace(os.linesep, ' ')} pid={os.getpid()}")

    rows: List[Dict[str, Any]] = []
    if args.symbols_mode == "parallel":
        futures: Dict[Any, str] = {}
        with ProcessPoolExecutor(max_workers=max(1, int(args.eval_procs))) as pool:
            for s in syms:
                futures[pool.submit(_run_symbol_worker, s, args, run_tag)] = s
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    row = fut.result()
                    rows.append(row)
                    _print_line(f"[{sym}] {row['PASS/FAIL']} mode={row['mode']} policy={row['policy']} cap_mult={row['cap_mult']}")
                except Exception:
                    tb = traceback.format_exc()
                    _print_line(tb.rstrip())
                    crash_path = _write_crash_report(sym, run_tag, tb)
                    _print_line(f"crash_report={crash_path}")
                    raise
    else:
        current_symbol: Optional[str] = None
        try:
            for s in syms:
                current_symbol = s
                _print_line(f"\n=== ULTRA OVERLAY {s} ===")
                row = _run_symbol_worker(s, args, run_tag=run_tag)
                rows.append(row)
                _print_line(f"[{s}] {row['PASS/FAIL']} mode={row['mode']} policy={row['policy']} cap_mult={row['cap_mult']}")
        except Exception:
            tb = traceback.format_exc()
            _print_line(tb.rstrip())
            if current_symbol:
                crash_path = _write_crash_report(current_symbol, run_tag, tb)
                _print_line(f"crash_report={crash_path}")
            raise

    out = pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
    out_path = (PROJECT_ROOT / "artifacts" / "reports" / "overlay_ultra_final_table.csv").resolve()
    out.to_csv(out_path, index=False)
    _print_line("\n=== ULTRA DONE ===")
    _print_line(out.to_string(index=False))
    _print_line(f"final_table={out_path}")


if __name__ == "__main__":
    main()
