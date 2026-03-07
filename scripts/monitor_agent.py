#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


THREAD_CAP_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
RUN_ID_RE = re.compile(r"(?<!\d)(\d{8}_\d{6})(?!\d)")
RUN_ID_ASSIGN_RE = re.compile(r"\brun_id=(\d{8}_\d{6})\b")
GEN_BEST_RE = re.compile(r"\[gen\s*([0-9]+)\]\s+best_score=([-+0-9.eE]+)")
GA_HEARTBEAT_RE = re.compile(
    r"\bphase=ga\b.*?\bprogress=([0-9]+)/([0-9]+)\b.*?\bga heartbeat\b.*?\bbest_fitness=([-+0-9.eE]+)"
)
BEST_VALID_INLINE_RE = re.compile(r"\bbest_valid_score[=: ]\s*([-+0-9.eE]+)")
BEST_ANY_INLINE_RE = re.compile(r"\bbest_any_score[=: ]\s*([-+0-9.eE]+)")
GEN_INLINE_RE = re.compile(r"\bgeneration(?:_index)?[=: ]\s*([0-9]+)\b")
SYMBOL_PATH_RE = re.compile(r"([A-Z0-9]{2,20}USDT)")
EPOCH_2017 = datetime(2017, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class Thresholds:
    min_pf: float
    max_dd: float
    min_trades: int
    min_stability: float


@dataclass
class MonitorConfig:
    repo_root: Path
    run_id_arg: str
    poll_sec: int
    stagnant_gens: int
    thresholds: Thresholds
    test_only: bool


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_iso() -> str:
    return utc_now().isoformat()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value from: {value!r}")


def apply_thread_caps() -> None:
    for var in THREAD_CAP_VARS:
        os.environ[var] = "1"


def normalize_field_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(",", "")
    pct = s.endswith("%")
    if pct:
        s = s[:-1].strip()
    try:
        v = float(s)
    except ValueError:
        return None
    if not math.isfinite(v):
        return None
    return v


def safe_int(value: Any) -> Optional[int]:
    v = safe_float(value)
    if v is None:
        return None
    try:
        return int(round(v))
    except Exception:
        return None


def safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "pass", "passed"}:
        return True
    if s in {"0", "false", "f", "no", "n", "fail", "failed"}:
        return False
    return None


def parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    # Accept "YYYY-MM-DD HH:MM:SS+00:00"
    if " " in s and "T" not in s:
        s = s.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        try:
            dt = datetime.strptime(s[:10], "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def fmt_num(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def fmt_int(value: Optional[int]) -> str:
    return "-" if value is None else str(int(value))


def infer_symbol_side_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    symbol = None
    side = None
    joined = "/".join(path.parts)
    m = SYMBOL_PATH_RE.search(joined.upper())
    if m:
        symbol = m.group(1).upper()
    low = joined.lower()
    if "short" in low:
        side = "short"
    elif "long" in low:
        side = "long"
    return symbol, side


def split_reasons(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for x in value:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    if not s:
        return []
    # tolerate JSON-ish arrays serialized as string
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return split_reasons(arr)
        except Exception:
            pass
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def normalize_dd(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    if v > 1.0 and v <= 100.0:
        return v / 100.0
    return v


def normalize_ratio_0_1(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    if v > 1.0 and v <= 100.0:
        return v / 100.0
    return v


def normalize_win_rate_pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    v = float(value)
    if v <= 1.0:
        return v * 100.0
    return v


def duration_years(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if start is None or end is None:
        return None
    sec = (end - start).total_seconds()
    if sec <= 0:
        return None
    return sec / (365.25 * 24.0 * 3600.0)


def tail_lines(path: Path, max_lines: int = 500, max_bytes: int = 512 * 1024) -> List[str]:
    try:
        size = path.stat().st_size
    except Exception:
        return []
    if size <= 0:
        return []
    read_size = min(size, max_bytes)
    try:
        with path.open("rb") as f:
            f.seek(size - read_size)
            data = f.read(read_size)
    except Exception:
        return []
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        return lines[-max_lines:]
    return lines


def scan_files(
    root: Path,
    predicate,
    *,
    max_depth: int,
    max_results: int,
    skip_dir_names: Optional[set] = None,
) -> List[Tuple[Path, float]]:
    if not root.exists() or not root.is_dir():
        return []
    skip = skip_dir_names or set()
    root_parts = len(root.parts)
    out: List[Tuple[Path, float]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        cur = Path(dirpath)
        depth = len(cur.parts) - root_parts
        if depth >= max_depth:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in skip and d != "__pycache__"]
        for name in filenames:
            p = cur / name
            if not predicate(p):
                continue
            try:
                mt = p.stat().st_mtime
            except Exception:
                continue
            out.append((p, mt))
            if len(out) >= max_results:
                return out
    return out


def latest_files_with_run_id(root: Path, *, max_depth: int, max_scan: int = 1500) -> List[Tuple[Path, float, str]]:
    if not root.exists():
        return []
    candidates: List[Tuple[Path, float]] = scan_files(
        root,
        lambda p: p.is_file(),
        max_depth=max_depth,
        max_results=max_scan,
        skip_dir_names={"monitor"},
    )
    out: List[Tuple[Path, float, str]] = []
    for p, mt in sorted(candidates, key=lambda x: x[1], reverse=True):
        rid = None
        for probe in (str(p), p.name, p.parent.name):
            m = RUN_ID_RE.search(probe)
            if m:
                rid = m.group(1)
                break
        if rid is None and p.suffix.lower() in {".log", ".txt", ".csv"}:
            for line in tail_lines(p, max_lines=80, max_bytes=64 * 1024):
                m = RUN_ID_ASSIGN_RE.search(line)
                if m:
                    rid = m.group(1)
                    break
        if rid:
            out.append((p, mt, rid))
    return out


def discover_run_id(repo_root: Path) -> str:
    artifacts = repo_root / "artifacts"
    roots = [
        artifacts / "reports",
        artifacts / "runs",
        artifacts / "universe_runs",
    ]
    for root in roots:
        found = latest_files_with_run_id(root, max_depth=6)
        if found:
            found.sort(key=lambda x: x[1], reverse=True)
            return found[0][2]
    # fallback: scan known logs for run_id=
    log_roots = [repo_root / "logs", artifacts / "reports", artifacts / "runs"]
    log_files: List[Tuple[Path, float]] = []
    for root in log_roots:
        log_files.extend(
            scan_files(
                root,
                lambda p: p.is_file() and (p.suffix.lower() in {".log", ".txt"} or "log" in p.name.lower()),
                max_depth=6,
                max_results=300,
                skip_dir_names={"monitor"},
            )
        )
    for p, _ in sorted(log_files, key=lambda x: x[1], reverse=True):
        for line in tail_lines(p, max_lines=120, max_bytes=128 * 1024):
            m = RUN_ID_ASSIGN_RE.search(line)
            if m:
                return m.group(1)
    return "unknown"


def find_log_candidates(repo_root: Path, run_id: str) -> List[Path]:
    artifacts = repo_root / "artifacts"
    roots = [repo_root / "logs", artifacts / "reports", artifacts / "runs"]
    scored: List[Tuple[int, float, Path]] = []
    seen: set = set()
    for root in roots:
        for p, mt in scan_files(
            root,
            lambda x: x.is_file() and (x.suffix.lower() in {".log", ".txt"} or "log" in x.name.lower()),
            max_depth=6,
            max_results=400,
            skip_dir_names={"monitor"},
        ):
            if p in seen:
                continue
            seen.add(p)
            path_s = str(p)
            score = 0
            if run_id and run_id != "unknown" and run_id in path_s:
                score += 8
            if "universe" in p.name.lower() or "ga" in p.name.lower():
                score += 3
            score += int(min(5.0, max(0.0, (time.time() - mt) / 3600.0)) * -1)
            scored.append((score, mt, p))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out = [p for _, _, p in scored[:12]]
    return out


def find_summary_candidates(repo_root: Path, run_id: str) -> List[Path]:
    artifacts = repo_root / "artifacts"
    roots: List[Path] = []
    if run_id and run_id != "unknown":
        roots.extend(
            [
                artifacts / "universe_runs" / run_id,
                artifacts / "reports" / f"universe_{run_id}",
                artifacts / "runs" / run_id,
            ]
        )
    roots.extend([artifacts / "reports", artifacts / "runs", artifacts / "universe_runs"])
    selected: List[Tuple[Path, float]] = []
    seen: set = set()
    json_keywords = ("summary", "universe", "result", "report", "backtest", "status", "final")

    def is_candidate(path: Path) -> bool:
        n = path.name.lower()
        ext = path.suffix.lower()
        if ext == ".csv":
            return ("summary" in n) or ("universe" in n) or ("results" in n)
        if ext == ".jsonl":
            if run_id and run_id != "unknown":
                return run_id in str(path)
            return any(k in n for k in json_keywords)
        if ext == ".json":
            if run_id and run_id != "unknown" and run_id in str(path):
                return True
            return any(k in n for k in json_keywords)
        return False

    for root in roots:
        if not root.exists():
            continue
        for p, mt in scan_files(
            root,
            lambda x: x.is_file() and is_candidate(x),
            max_depth=7,
            max_results=500,
            skip_dir_names={"monitor"},
        ):
            if p in seen:
                continue
            seen.add(p)
            selected.append((p, mt))

    if not selected:
        for p, mt in scan_files(
            artifacts,
            lambda x: x.is_file() and is_candidate(x),
            max_depth=6,
            max_results=600,
            skip_dir_names={"monitor"},
        ):
            if p in seen:
                continue
            seen.add(p)
            selected.append((p, mt))

    selected.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in selected[:240]]


def parse_csv_rows(path: Path, *, max_tail_rows: int = 3000, full_size_limit: int = 5 * 1024 * 1024) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        size = path.stat().st_size
    except Exception:
        return rows
    if size <= 0:
        return rows
    try:
        if size <= full_size_limit:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
            return rows

        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            header = f.readline()
        if not header.strip():
            return rows
        tail = tail_lines(path, max_lines=max_tail_rows, max_bytes=3 * 1024 * 1024)
        if not tail:
            return rows
        payload = "\n".join([header.rstrip("\n")] + tail)
        reader = csv.DictReader(payload.splitlines())
        for row in reader:
            rows.append(dict(row))
        return rows
    except Exception:
        return rows


def parse_json_rows(path: Path, run_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        size = path.stat().st_size
    except Exception:
        return rows
    if size <= 0:
        return rows
    # avoid loading very large JSON blobs unrelated to summaries
    if size > 12 * 1024 * 1024 and (run_id == "unknown" or run_id not in str(path)):
        return rows
    payload = read_json(path, None)
    if payload is None:
        return rows
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                rows.append(dict(item))
        return rows
    if isinstance(payload, dict):
        meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
        for key in ("rows", "records", "results", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    enriched = dict(item)
                    # useful meta propagated when row lacks these fields
                    for mk in ("run_id", "pipeline_run_id", "start", "end", "test_start", "test_end"):
                        if mk in meta and mk not in enriched:
                            enriched[mk] = meta.get(mk)
                    rows.append(enriched)
                if rows:
                    return rows
        rows.append(dict(payload))
    return rows


def parse_jsonl_rows(path: Path, *, max_lines_small: int = 2500, max_bytes_large: int = 2 * 1024 * 1024) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        size = path.stat().st_size
    except Exception:
        return rows
    if size <= 0:
        return rows

    lines: List[str]
    if size <= max_bytes_large:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except Exception:
            return rows
    else:
        lines = tail_lines(path, max_lines=max_lines_small, max_bytes=max_bytes_large)

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if not s.startswith("{"):
            continue
        try:
            item = json.loads(s)
        except Exception:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def alias_value(norm: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        nk = normalize_field_name(key)
        if nk in norm:
            v = norm[nk]
            if v is not None and str(v).strip() != "":
                return v
    return None


def normalize_row(raw: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        nk = normalize_field_name(k)
        if nk not in out:
            out[nk] = v
    return out


def canonicalize_record(
    raw: Mapping[str, Any],
    *,
    source_file: Path,
    source_mtime: float,
    run_id_target: str,
    test_only: bool,
) -> Optional[Dict[str, Any]]:
    norm = normalize_row(raw)
    path_symbol, path_side = infer_symbol_side_from_path(source_file)
    symbol = alias_value(norm, ["symbol", "coin", "asset", "ticker", "pair", "instrument"])
    if symbol is None:
        symbol = path_symbol
    if symbol is None:
        return None
    symbol = str(symbol).strip().upper()
    if not symbol:
        return None

    side_raw = alias_value(norm, ["side", "direction", "position_side"])
    side = str(side_raw).strip().lower() if side_raw is not None else (path_side or "na")
    if side not in {"long", "short"}:
        side = path_side or side or "na"

    # test metrics get priority; when unavailable, fallback to generic
    pf_test = safe_float(alias_value(norm, ["test_pf", "pf_test", "profit_factor_test", "test_profit_factor"]))
    pf_gen = safe_float(alias_value(norm, ["profit_factor", "pf", "overlay_test_pf"]))
    dd_test = normalize_dd(safe_float(alias_value(norm, ["test_dd", "dd_test", "max_dd_test", "drawdown_test"])))
    dd_gen = normalize_dd(safe_float(alias_value(norm, ["max_dd", "dd", "drawdown", "overlay_test_dd"])))
    trades_test = safe_float(alias_value(norm, ["test_trades", "trades_test", "trade_count_test"]))
    trades_gen = safe_float(alias_value(norm, ["trades", "trade_count", "n_trades"]))
    net_test = safe_float(alias_value(norm, ["test_net", "net_test", "pnl_test", "overlay_test_net"]))
    net_gen = safe_float(alias_value(norm, ["net_profit", "net", "pnl", "final_equity", "overlay_net"]))
    stability = normalize_ratio_0_1(
        safe_float(alias_value(norm, ["stability", "stability_pct", "stability_score", "robustness"]))
    )
    cagr_pct = safe_float(alias_value(norm, ["cagr_pct", "cagr", "test_cagr_pct"]))
    return_pct = safe_float(alias_value(norm, ["return_pct", "ret_pct", "roi_pct"]))
    net_profit = safe_float(alias_value(norm, ["net_profit", "test_net", "pnl"]))
    win_rate_pct = normalize_win_rate_pct(safe_float(alias_value(norm, ["win_rate_pct", "win_rate", "wr"])))

    period_start_raw = alias_value(
        norm,
        [
            "period_start",
            "start",
            "start_date",
            "test_start",
            "listing_first_ts",
            "symbol_first_ts_utc",
            "listing_first_ts_utc",
        ],
    )
    period_end_raw = alias_value(norm, ["period_end", "end", "end_date", "test_end"])
    period_start = parse_timestamp(period_start_raw)
    period_end = parse_timestamp(period_end_raw)

    run_id = alias_value(norm, ["run_id", "pipeline_run_id", "ga_run_id"])
    run_id = str(run_id).strip() if run_id is not None else ""

    pass_raw = alias_value(norm, ["PASS/FAIL", "PASS_FAIL", "pass_fail", "status", "result"])
    fail_reasons_raw = alias_value(norm, ["fail_reasons", "failure_reasons", "reason", "reasons"])
    eligible_2017 = safe_bool(alias_value(norm, ["eligible_2017"]))

    pf = pf_test if pf_test is not None else pf_gen
    dd = dd_test if dd_test is not None else dd_gen
    trades = trades_test if trades_test is not None else trades_gen
    net = net_test if net_test is not None else net_gen
    if test_only:
        # keep fallback for files that only have generic fields
        pf = pf_test if pf_test is not None else pf
        dd = dd_test if dd_test is not None else dd
        trades = trades_test if trades_test is not None else trades
        net = net_test if net_test is not None else net

    rec = {
        "symbol": symbol,
        "side": side,
        "pf": pf,
        "dd": dd,
        "trades": trades,
        "stability": stability,
        "net": net,
        "cagr_pct": cagr_pct,
        "return_pct": return_pct,
        "net_profit": net_profit,
        "win_rate_pct": win_rate_pct,
        "period_start": period_start.isoformat() if period_start else "",
        "period_end": period_end.isoformat() if period_end else "",
        "period_start_dt": period_start,
        "period_end_dt": period_end,
        "eligible_2017": eligible_2017,
        "raw_pass": str(pass_raw).strip() if pass_raw is not None else "",
        "raw_fail_reasons": split_reasons(fail_reasons_raw),
        "run_id": run_id,
        "source_file": str(source_file),
        "source_mtime": float(source_mtime),
        "run_match": bool(run_id_target and run_id_target != "unknown" and (run_id == run_id_target or run_id_target in str(source_file))),
    }

    # useful GA hints from summary rows
    rec["best_score"] = safe_float(alias_value(norm, ["best_score", "best_fitness", "fitness"]))
    rec["generations_ran"] = safe_int(alias_value(norm, ["generations_ran", "executed_generations", "generation_index", "gen"]))
    return rec


def parse_summary_records(files: Sequence[Path], run_id: str, test_only: bool) -> List[Dict[str, Any]]:
    all_records: List[Dict[str, Any]] = []
    for path in files:
        try:
            mt = path.stat().st_mtime
        except Exception:
            continue
        ext = path.suffix.lower()
        rows: List[Dict[str, Any]]
        if ext == ".csv":
            rows = parse_csv_rows(path)
        elif ext == ".json":
            rows = parse_json_rows(path, run_id=run_id)
        elif ext == ".jsonl":
            rows = parse_jsonl_rows(path)
        else:
            continue
        for raw in rows:
            rec = canonicalize_record(
                raw,
                source_file=path,
                source_mtime=mt,
                run_id_target=run_id,
                test_only=test_only,
            )
            if rec is not None:
                all_records.append(rec)
    return all_records


def choose_records(records: Sequence[Dict[str, Any]], run_id: str) -> List[Dict[str, Any]]:
    if not records:
        return []
    selected = list(records)
    if run_id and run_id != "unknown":
        matched = [r for r in selected if r.get("run_match")]
        if matched:
            selected = matched

    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in sorted(selected, key=lambda r: (bool(r.get("run_match")), float(r.get("source_mtime", 0.0))), reverse=True):
        key = (str(rec.get("symbol", "")).upper(), str(rec.get("side", "na")).lower())
        if key not in by_key:
            by_key[key] = dict(rec)
            continue
        cur = by_key[key]
        # merge missing fields from older sources
        for field in ("pf", "dd", "trades", "stability", "net", "cagr_pct", "return_pct", "net_profit", "win_rate_pct"):
            if cur.get(field) is None and rec.get(field) is not None:
                cur[field] = rec.get(field)
        if not cur.get("period_start") and rec.get("period_start"):
            cur["period_start"] = rec.get("period_start")
            cur["period_start_dt"] = rec.get("period_start_dt")
        if not cur.get("period_end") and rec.get("period_end"):
            cur["period_end"] = rec.get("period_end")
            cur["period_end_dt"] = rec.get("period_end_dt")
    out = list(by_key.values())
    out.sort(key=lambda r: (str(r.get("symbol", "")), str(r.get("side", ""))))
    return out


def compute_sanity_flags(rec: Mapping[str, Any], thresholds: Thresholds) -> List[str]:
    flags: List[str] = []
    ret = safe_float(rec.get("return_pct"))
    net_profit = safe_float(rec.get("net_profit"))
    cagr = safe_float(rec.get("cagr_pct"))
    pf = safe_float(rec.get("pf"))
    dd = safe_float(rec.get("dd"))
    trades = safe_float(rec.get("trades"))
    wr = safe_float(rec.get("win_rate_pct"))
    start = rec.get("period_start_dt")
    end = rec.get("period_end_dt")

    if ret is not None and net_profit is not None:
        denom = max(1.0, abs(ret), abs(net_profit))
        if abs(ret - net_profit) <= 1e-9 * denom:
            flags.append("return_pct_equals_net_profit")

    if cagr is not None and pf is not None:
        if cagr > 250.0 and pf < thresholds.min_pf:
            flags.append("cagr_extreme_with_low_pf")

    years = duration_years(start, end)
    if trades is not None and trades > 0:
        if years is None:
            years = 1.0
        tpy = trades / max(1e-9, years)
        if tpy > 2000.0:
            flags.append("trades_per_year_gt_2000")
        elif tpy < 5.0:
            flags.append("trades_per_year_lt_5")

    if pf is not None and wr is not None and pf > 0 and 0.0 < wr < 100.0:
        wr_ratio = wr / 100.0
        implied = pf * (1.0 - wr_ratio) / wr_ratio
        if implied > 30.0:
            flags.append("implied_avgwin_avgloss_gt_30x")

    huge_returns = False
    if ret is not None and abs(ret) >= 100.0:
        huge_returns = True
    if cagr is not None and cagr >= 200.0:
        huge_returns = True
    if net_profit is not None and abs(net_profit) >= 1000.0:
        huge_returns = True
    if huge_returns:
        if dd is None:
            flags.append("max_dd_missing_with_huge_returns")
        elif abs(dd) <= 1e-12:
            flags.append("max_dd_zero_with_huge_returns")

    return flags


def evaluate_pass_fail(rec: Mapping[str, Any], thresholds: Thresholds) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    pf = safe_float(rec.get("pf"))
    dd = safe_float(rec.get("dd"))
    trades = safe_float(rec.get("trades"))
    stability = safe_float(rec.get("stability"))

    if pf is None:
        reasons.append("pf_missing")
    elif pf < thresholds.min_pf:
        reasons.append(f"pf<{thresholds.min_pf:g}")

    if dd is None:
        reasons.append("dd_missing")
    elif dd > thresholds.max_dd:
        reasons.append(f"dd>{thresholds.max_dd:g}")

    if trades is None:
        reasons.append("trades_missing")
    elif trades < float(thresholds.min_trades):
        reasons.append(f"trades<{thresholds.min_trades}")

    if stability is None:
        reasons.append("stability_missing")
    elif stability < thresholds.min_stability:
        reasons.append(f"stability<{thresholds.min_stability:g}")

    return ("PASS" if not reasons else "FAIL"), reasons


def eligible_2017(rec: Mapping[str, Any]) -> bool:
    raw = rec.get("eligible_2017")
    b = safe_bool(raw)
    if b is not None:
        return b
    start = rec.get("period_start_dt")
    if isinstance(start, datetime):
        return start <= EPOCH_2017
    return True


def build_universe_rows(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("pass_fail") != "PASS":
            continue
        if not eligible_2017(rec):
            continue
        pf = safe_float(rec.get("pf"))
        dd = safe_float(rec.get("dd"))
        trades = safe_float(rec.get("trades"))
        stability = safe_float(rec.get("stability"))
        if pf is None or dd is None or trades is None or stability is None:
            continue
        dd_clamped = min(0.99, max(0.0, dd))
        if trades <= 0.0 or stability <= 0.0:
            continue
        pf_component = pf ** 1.0
        dd_component = (1.0 - dd_clamped) ** 1.5
        trades_component = math.log(1.0 + max(0.0, trades))
        stability_component = stability
        score = pf_component * dd_component * trades_component * stability_component
        rows.append(
            {
                "symbol": rec.get("symbol", ""),
                "side": rec.get("side", ""),
                "score": score,
                "pf_component": pf_component,
                "dd_component": dd_component,
                "trades_component": trades_component,
                "stability_component": stability_component,
                "pf": pf,
                "dd": dd_clamped,
                "trades": trades,
                "stability": stability,
                "period_start": rec.get("period_start", ""),
                "source_file": rec.get("source_file", ""),
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    for idx, row in enumerate(rows, 1):
        row["rank"] = idx
    return rows


def write_universe_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "rank",
        "symbol",
        "side",
        "score",
        "pf_component",
        "dd_component",
        "trades_component",
        "stability_component",
        "pf",
        "dd",
        "trades",
        "stability",
        "period_start",
        "source_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})


def read_new_lines_with_offset(path: Path, old_offset: int, *, max_read_bytes: int = 2 * 1024 * 1024) -> Tuple[List[str], int]:
    try:
        size = path.stat().st_size
    except Exception:
        return [], old_offset
    if size <= 0:
        return [], 0
    offset = max(0, int(old_offset))
    if offset > size:
        offset = 0
    if size - offset > max_read_bytes:
        offset = max(0, size - max_read_bytes)
    try:
        with path.open("rb") as f:
            f.seek(offset)
            data = f.read(size - offset)
    except Exception:
        return [], offset
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    return lines, size


def parse_ga_events_from_line(line: str) -> Dict[str, Any]:
    event: Dict[str, Any] = {}
    m = GEN_BEST_RE.search(line)
    if m:
        event["generation"] = safe_int(m.group(1))
        event["best_valid_score"] = safe_float(m.group(2))
        return event

    m = GA_HEARTBEAT_RE.search(line)
    if m:
        progress = safe_int(m.group(1))
        event["generation"] = (progress - 1) if progress is not None else None
        event["best_valid_score"] = safe_float(m.group(3))
        return event

    # parse inline JSON-ish telemetry lines with generation / best_valid_score / best_any_score
    gen = None
    mg = GEN_INLINE_RE.search(line)
    if mg:
        gen = safe_int(mg.group(1))
        event["generation"] = gen
    mv = BEST_VALID_INLINE_RE.search(line)
    if mv:
        event["best_valid_score"] = safe_float(mv.group(1))
    ma = BEST_ANY_INLINE_RE.search(line)
    if ma:
        event["best_any_score"] = safe_float(ma.group(1))
    return event


def update_ga_state_from_logs(
    log_files: Sequence[Path],
    log_offsets: Dict[str, int],
    prev_ga: Mapping[str, Any],
    stagnant_gens: int,
) -> Tuple[Dict[str, Any], Dict[str, int], int]:
    ga = {
        "last_gen": safe_int(prev_ga.get("last_gen")),
        "best_valid_score": safe_float(prev_ga.get("best_valid_score")),
        "best_any_score": safe_float(prev_ga.get("best_any_score")),
        "best_gen": safe_int(prev_ga.get("best_gen")),
        "lines_read": 0,
    }
    if ga["last_gen"] is None:
        ga["last_gen"] = -1
    if ga["best_gen"] is None:
        ga["best_gen"] = -1
    offsets = dict(log_offsets)
    events = 0

    for path in log_files:
        key = str(path)
        old = int(offsets.get(key, 0))
        lines, new_offset = read_new_lines_with_offset(path, old)
        offsets[key] = new_offset
        ga["lines_read"] += len(lines)
        for line in lines:
            ev = parse_ga_events_from_line(line)
            if not ev:
                continue
            gen = ev.get("generation")
            best_valid = ev.get("best_valid_score")
            best_any = ev.get("best_any_score")
            touched = False
            if gen is not None:
                g = int(gen)
                if g > int(ga["last_gen"]):
                    ga["last_gen"] = g
                touched = True
            if best_valid is not None:
                score = float(best_valid)
                if ga["best_valid_score"] is None or score > float(ga["best_valid_score"]):
                    ga["best_valid_score"] = score
                    if gen is not None:
                        ga["best_gen"] = int(gen)
                touched = True
            if best_any is not None:
                score_any = float(best_any)
                if ga["best_any_score"] is None or score_any > float(ga["best_any_score"]):
                    ga["best_any_score"] = score_any
                touched = True
            if touched:
                events += 1

    if int(ga["last_gen"]) >= 0 and int(ga["best_gen"]) < 0:
        ga["best_gen"] = int(ga["last_gen"])
    gens_since = 0
    if int(ga["last_gen"]) >= 0 and int(ga["best_gen"]) >= 0:
        gens_since = max(0, int(ga["last_gen"]) - int(ga["best_gen"]))
    ga["gens_since_improvement"] = gens_since
    ga["stagnant"] = gens_since >= int(stagnant_gens)
    return ga, offsets, events


def update_ga_state_from_summary_hints(ga: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return ga
    last_gen_hint = ga.get("last_gen", -1)
    best_score_hint = ga.get("best_valid_score")
    for rec in records:
        g = safe_int(rec.get("generations_ran"))
        if g is not None:
            # generations_ran is count, convert to zero-based index
            idx = max(-1, g - 1)
            if idx > int(last_gen_hint):
                last_gen_hint = idx
        bs = safe_float(rec.get("best_score"))
        if bs is not None:
            if best_score_hint is None or bs > float(best_score_hint):
                best_score_hint = bs
    if int(ga.get("last_gen", -1)) < int(last_gen_hint):
        ga["last_gen"] = int(last_gen_hint)
    if ga.get("best_valid_score") is None and best_score_hint is not None:
        ga["best_valid_score"] = float(best_score_hint)
        ga["best_gen"] = int(ga.get("last_gen", -1))
    return ga


def render_report_md(
    *,
    run_id: str,
    poll_sec: int,
    thresholds: Thresholds,
    stagnant_gens: int,
    ga_state: Mapping[str, Any],
    records: Sequence[Dict[str, Any]],
    universe_rows: Sequence[Dict[str, Any]],
    fail_hist: Counter,
    flag_hist: Counter,
    source_files: Sequence[str],
) -> str:
    lines: List[str] = []
    lines.append(f"# Monitor Report: {run_id}")
    lines.append("")
    lines.append(f"- last_updated_utc: {utc_iso()}")
    lines.append(f"- poll_sec: {poll_sec}")
    lines.append(f"- thresholds: min_pf={thresholds.min_pf:g}, max_dd={thresholds.max_dd:g}, min_trades={thresholds.min_trades}, min_stability={thresholds.min_stability:g}")
    lines.append(f"- stagnant_gens: {stagnant_gens}")
    lines.append("")
    lines.append("## GA Status")
    last_gen = safe_int(ga_state.get("last_gen"))
    best_valid = safe_float(ga_state.get("best_valid_score"))
    best_any = safe_float(ga_state.get("best_any_score"))
    gens_since = safe_int(ga_state.get("gens_since_improvement"))
    lines.append(f"- last_gen_seen: {fmt_int(last_gen)}")
    lines.append(f"- best_valid_score: {fmt_num(best_valid, 6)}")
    lines.append(f"- best_any_score: {fmt_num(best_any, 6)}")
    lines.append(f"- gens_since_improvement: {fmt_int(gens_since)}")
    if bool(ga_state.get("stagnant")):
        lines.append(f"- STAGNATION WARNING: no improvement for {fmt_int(gens_since)} generations")
    lines.append("")
    lines.append("## Audited Coins")
    lines.append("| symbol | side | PF | DD | trades | stability | PASS/FAIL | reasons | sanity_flags |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|---|")
    for rec in records:
        reasons = ",".join(rec.get("reasons", [])) if rec.get("reasons") else "-"
        flags = ",".join(rec.get("sanity_flags", [])) if rec.get("sanity_flags") else "-"
        lines.append(
            "| {symbol} | {side} | {pf} | {dd} | {trades} | {stability} | {pfail} | {reasons} | {flags} |".format(
                symbol=rec.get("symbol", ""),
                side=rec.get("side", ""),
                pf=fmt_num(safe_float(rec.get("pf")), 4),
                dd=fmt_num(safe_float(rec.get("dd")), 4),
                trades=fmt_num(safe_float(rec.get("trades")), 1),
                stability=fmt_num(safe_float(rec.get("stability")), 4),
                pfail=rec.get("pass_fail", ""),
                reasons=reasons,
                flags=flags,
            )
        )
    if not records:
        lines.append("| - | - | - | - | - | - | - | no records found yet | - |")
    lines.append("")
    lines.append("## Universe")
    if universe_rows:
        lines.append("| rank | symbol | side | score | PF | DD | trades | stability |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for row in universe_rows:
            lines.append(
                "| {rank} | {symbol} | {side} | {score} | {pf} | {dd} | {trades} | {stability} |".format(
                    rank=row.get("rank", ""),
                    symbol=row.get("symbol", ""),
                    side=row.get("side", ""),
                    score=fmt_num(safe_float(row.get("score")), 6),
                    pf=fmt_num(safe_float(row.get("pf")), 4),
                    dd=fmt_num(safe_float(row.get("dd")), 4),
                    trades=fmt_num(safe_float(row.get("trades")), 1),
                    stability=fmt_num(safe_float(row.get("stability")), 4),
                )
            )
        top_n = min(3, len(universe_rows))
        lines.append("")
        lines.append(f"Top {top_n} notes:")
        for row in universe_rows[:top_n]:
            lines.append(
                "- {symbol} {side}: PF={pf:.3f}, DD={dd:.3f}, trades={trades:.1f}, stability={stability:.3f}, score={score:.6f}".format(
                    symbol=row.get("symbol", ""),
                    side=row.get("side", ""),
                    pf=float(row.get("pf", 0.0)),
                    dd=float(row.get("dd", 0.0)),
                    trades=float(row.get("trades", 0.0)),
                    stability=float(row.get("stability", 0.0)),
                    score=float(row.get("score", 0.0)),
                )
            )
    else:
        lines.append("No passed/eligible symbols for universe ranking yet.")
    lines.append("")
    lines.append("## Action List")
    lines.append("Top failure reasons:")
    if fail_hist:
        for reason, cnt in fail_hist.most_common(8):
            lines.append(f"- {reason}: {cnt}")
    else:
        lines.append("- none")
    lines.append("Top sanity flags:")
    if flag_hist:
        for flag, cnt in flag_hist.most_common(8):
            lines.append(f"- {flag}: {cnt}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Sources:")
    if source_files:
        for sf in source_files[:20]:
            lines.append(f"- {sf}")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


class MonitorAgent:
    def __init__(self, cfg: MonitorConfig):
        self.cfg = cfg
        self._stop = False
        self._resolved_run_id: Optional[str] = None
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        _ = signum
        self._stop = True

    def resolve_run_id(self) -> str:
        if self.cfg.run_id_arg != "auto":
            return self.cfg.run_id_arg
        # lock onto the first discovered run_id to avoid bouncing between runs
        if self._resolved_run_id and self._resolved_run_id != "unknown":
            return self._resolved_run_id
        rid = discover_run_id(self.cfg.repo_root)
        self._resolved_run_id = rid
        return rid

    def poll_once(self) -> Dict[str, Any]:
        run_id = self.resolve_run_id()
        out_dir = self.cfg.repo_root / "artifacts" / "monitor" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        state_path = out_dir / ".state.json"
        status_path = out_dir / "status.json"
        flags_path = out_dir / "flags.json"
        report_path = out_dir / "report.md"
        universe_path = out_dir / "universe.csv"

        state = read_json(state_path, {})
        if not isinstance(state, dict):
            state = {}
        if state.get("run_id") != run_id:
            state = {"run_id": run_id, "log_offsets": {}, "ga": {}}

        log_offsets = state.get("log_offsets") if isinstance(state.get("log_offsets"), dict) else {}
        ga_prev = state.get("ga") if isinstance(state.get("ga"), dict) else {}

        log_files = find_log_candidates(self.cfg.repo_root, run_id)
        ga_state, log_offsets_new, events = update_ga_state_from_logs(
            log_files=log_files,
            log_offsets={str(k): int(v) for k, v in log_offsets.items()},
            prev_ga=ga_prev,
            stagnant_gens=self.cfg.stagnant_gens,
        )

        summary_files = find_summary_candidates(self.cfg.repo_root, run_id)
        all_records = parse_summary_records(summary_files, run_id=run_id, test_only=self.cfg.test_only)
        records = choose_records(all_records, run_id=run_id)

        audited: List[Dict[str, Any]] = []
        fail_hist: Counter = Counter()
        flag_hist: Counter = Counter()
        for rec in records:
            pass_fail, reasons = evaluate_pass_fail(rec, self.cfg.thresholds)
            sanity_flags = compute_sanity_flags(rec, self.cfg.thresholds)
            rec["pass_fail"] = pass_fail
            rec["reasons"] = reasons
            rec["sanity_flags"] = sanity_flags
            audited.append(rec)
            if pass_fail == "FAIL":
                fail_hist.update(reasons)
            flag_hist.update(sanity_flags)

        ga_state = update_ga_state_from_summary_hints(ga_state, audited)
        if safe_int(ga_state.get("best_gen")) is None:
            ga_state["best_gen"] = safe_int(ga_state.get("last_gen")) if safe_int(ga_state.get("last_gen")) is not None else -1
        if safe_int(ga_state.get("last_gen")) is None:
            ga_state["last_gen"] = -1
        if safe_int(ga_state.get("best_gen")) is None:
            ga_state["best_gen"] = -1
        ga_state["gens_since_improvement"] = max(0, int(ga_state["last_gen"]) - int(ga_state["best_gen"]))
        ga_state["stagnant"] = int(ga_state["gens_since_improvement"]) >= int(self.cfg.stagnant_gens)
        ga_state["events_in_poll"] = int(events)
        ga_state["log_files"] = [str(p) for p in log_files]

        universe_rows = build_universe_rows(audited)
        write_universe_csv(universe_path, universe_rows)

        flags_payload = {
            "run_id": run_id,
            "updated_utc": utc_iso(),
            "records": [
                {
                    "symbol": rec.get("symbol"),
                    "side": rec.get("side"),
                    "PASS/FAIL": rec.get("pass_fail"),
                    "reasons": rec.get("reasons", []),
                    "sanity_flags": rec.get("sanity_flags", []),
                    "metrics": {
                        "pf": rec.get("pf"),
                        "dd": rec.get("dd"),
                        "trades": rec.get("trades"),
                        "stability": rec.get("stability"),
                        "net": rec.get("net"),
                        "cagr_pct": rec.get("cagr_pct"),
                        "win_rate_pct": rec.get("win_rate_pct"),
                    },
                    "eligible_2017": eligible_2017(rec),
                    "period_start": rec.get("period_start", ""),
                    "source_file": rec.get("source_file", ""),
                }
                for rec in audited
            ],
        }
        write_json(flags_path, flags_payload)

        status_payload = {
            "run_id": run_id,
            "updated_utc": utc_iso(),
            "poll_sec": int(self.cfg.poll_sec),
            "stagnant_gens": int(self.cfg.stagnant_gens),
            "thresholds": {
                "min_pf": float(self.cfg.thresholds.min_pf),
                "max_dd": float(self.cfg.thresholds.max_dd),
                "min_trades": int(self.cfg.thresholds.min_trades),
                "min_stability": float(self.cfg.thresholds.min_stability),
            },
            "ga": {
                "last_gen": int(ga_state.get("last_gen", -1)),
                "best_valid_score": ga_state.get("best_valid_score"),
                "best_any_score": ga_state.get("best_any_score"),
                "best_gen": int(ga_state.get("best_gen", -1)),
                "gens_since_improvement": int(ga_state.get("gens_since_improvement", 0)),
                "stagnant": bool(ga_state.get("stagnant", False)),
                "events_in_poll": int(ga_state.get("events_in_poll", 0)),
                "lines_read": int(ga_state.get("lines_read", 0)),
                "log_files": ga_state.get("log_files", []),
            },
            "audit": {
                "records_total": len(audited),
                "pass_count": sum(1 for r in audited if r.get("pass_fail") == "PASS"),
                "fail_count": sum(1 for r in audited if r.get("pass_fail") != "PASS"),
                "source_files": [str(p) for p in summary_files[:80]],
                "top_fail_reasons": fail_hist.most_common(10),
                "top_sanity_flags": flag_hist.most_common(10),
            },
            "outputs": {
                "report_md": str(report_path),
                "status_json": str(status_path),
                "flags_json": str(flags_path),
                "universe_csv": str(universe_path),
            },
        }
        write_json(status_path, status_payload)

        report_text = render_report_md(
            run_id=run_id,
            poll_sec=self.cfg.poll_sec,
            thresholds=self.cfg.thresholds,
            stagnant_gens=self.cfg.stagnant_gens,
            ga_state=ga_state,
            records=audited,
            universe_rows=universe_rows,
            fail_hist=fail_hist,
            flag_hist=flag_hist,
            source_files=[str(p) for p in summary_files],
        )
        report_path.write_text(report_text, encoding="utf-8")

        new_state = {
            "run_id": run_id,
            "updated_utc": utc_iso(),
            "ga": {
                "last_gen": int(ga_state.get("last_gen", -1)),
                "best_valid_score": ga_state.get("best_valid_score"),
                "best_any_score": ga_state.get("best_any_score"),
                "best_gen": int(ga_state.get("best_gen", -1)),
            },
            "log_offsets": log_offsets_new,
        }
        write_json(state_path, new_state)

        pass_count = status_payload["audit"]["pass_count"]
        total = status_payload["audit"]["records_total"]
        print(
            "[monitor] {ts} run_id={rid} gen={gen} best={best} stagnant={stale} pass={passed}/{total}".format(
                ts=utc_now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                rid=run_id,
                gen=int(ga_state.get("last_gen", -1)),
                best=fmt_num(safe_float(ga_state.get("best_valid_score")), 4),
                stale=int(ga_state.get("gens_since_improvement", 0)),
                passed=pass_count,
                total=total,
            ),
            flush=True,
        )

        return status_payload

    def run_forever(self) -> None:
        while not self._stop:
            started = time.time()
            try:
                self.poll_once()
            except Exception as ex:
                run_id = self.resolve_run_id()
                print(
                    f"[monitor] {utc_now().strftime('%Y-%m-%dT%H:%M:%SZ')} run_id={run_id} error={type(ex).__name__}: {ex}",
                    flush=True,
                )
            elapsed = time.time() - started
            sleep_sec = max(1.0, float(self.cfg.poll_sec) - elapsed)
            end_time = time.time() + sleep_sec
            while not self._stop and time.time() < end_time:
                time.sleep(min(1.0, end_time - time.time()))


def run_self_test() -> int:
    apply_thread_caps()
    with tempfile.TemporaryDirectory(prefix="monitor_agent_selftest_") as td:
        root = Path(td)
        rid = "20260211_160149"
        (root / "logs").mkdir(parents=True, exist_ok=True)
        (root / "artifacts" / "reports" / f"universe_{rid}").mkdir(parents=True, exist_ok=True)

        # fake GA log
        log_path = root / "logs" / "ga_monitor_test.log"
        log_path.write_text(
            "\n".join(
                [
                    "startup_utc=2026-02-11T16:01:00+00:00",
                    f"run_id={rid}",
                    "[gen 00] best_score=100.0 net=10.0 trades=20.0 win=40.0% dd=10.0% pf=1.20 (train_net=12.0 test_net=5.0)",
                    "[gen 01] best_score=130.0 net=11.0 trades=21.0 win=41.0% dd=10.0% pf=1.30 (train_net=13.0 test_net=6.0)",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        summary_csv = root / "artifacts" / "reports" / f"universe_{rid}" / "summary.csv"
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "symbol",
                    "side",
                    "test_net",
                    "test_pf",
                    "test_dd",
                    "test_trades",
                    "stability",
                    "PASS_FAIL",
                    "fail_reasons",
                    "run_id",
                    "period_start",
                    "period_end",
                    "win_rate_pct",
                    "cagr_pct",
                    "return_pct",
                    "net_profit",
                    "eligible_2017",
                ]
            )
            w.writerow(
                [
                    "BTCUSDT",
                    "long",
                    "400",
                    "1.5",
                    "0.12",
                    "120",
                    "0.9",
                    "PASS",
                    "",
                    rid,
                    "2017-01-01",
                    "2025-12-31",
                    "57",
                    "45",
                    "40",
                    "400",
                    "true",
                ]
            )
            w.writerow(
                [
                    "NEWUSDT",
                    "short",
                    "220",
                    "1.6",
                    "0.10",
                    "90",
                    "0.85",
                    "PASS",
                    "",
                    rid,
                    "2018-05-01",
                    "2025-12-31",
                    "52",
                    "75",
                    "22",
                    "220",
                    "false",
                ]
            )
            w.writerow(
                [
                    "ETHUSDT",
                    "short",
                    "-10",
                    "0.9",
                    "0.32",
                    "14",
                    "0.4",
                    "FAIL",
                    "pf<1.2,dd>0.2",
                    rid,
                    "2017-01-01",
                    "2025-12-31",
                    "48",
                    "10",
                    "-1",
                    "-10",
                    "true",
                ]
            )

        cfg = MonitorConfig(
            repo_root=root,
            run_id_arg="auto",
            poll_sec=60,
            stagnant_gens=2,
            thresholds=Thresholds(min_pf=1.2, max_dd=0.2, min_trades=30, min_stability=0.75),
            test_only=False,
        )
        agent = MonitorAgent(cfg)
        status1 = agent.poll_once()
        # append one more non-improving generation to validate stagnation increment
        with log_path.open("a", encoding="utf-8") as f:
            f.write("[gen 02] best_score=120.0 net=9.0 trades=22.0 win=39.0% dd=11.0% pf=1.10 (train_net=11.0 test_net=4.0)\n")
        status2 = agent.poll_once()

        out_dir = root / "artifacts" / "monitor" / rid
        required = [
            out_dir / "status.json",
            out_dir / "report.md",
            out_dir / "flags.json",
            out_dir / "universe.csv",
        ]
        for p in required:
            if not p.exists():
                print(f"SELF-TEST FAIL: missing {p}", file=sys.stderr)
                return 1

        if int(status2["ga"]["last_gen"]) < 2:
            print("SELF-TEST FAIL: GA gen parsing did not advance to 2", file=sys.stderr)
            return 1
        if int(status2["audit"]["records_total"]) < 2:
            print("SELF-TEST FAIL: expected audited records", file=sys.stderr)
            return 1
        # NEWUSDT must be excluded from universe due listing after 2017
        uni_rows = parse_csv_rows(out_dir / "universe.csv")
        syms = {str(r.get("symbol", "")).upper() for r in uni_rows}
        if "NEWUSDT" in syms:
            print("SELF-TEST FAIL: listing-after-2017 symbol should be excluded from universe ranking", file=sys.stderr)
            return 1

        print(f"SELF-TEST PASS: outputs at {out_dir}")
        print(f"SELF-TEST STATUS: run_id={status1['run_id']} gen={status2['ga']['last_gen']} pass={status2['audit']['pass_count']}/{status2['audit']['records_total']}")
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Lightweight monitor agent for GA progress and summary audits.")
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]), help="Project root path")
    ap.add_argument("--run-id", default="auto", help="auto|<run_id>")
    ap.add_argument("--poll-sec", type=int, default=60)
    ap.add_argument("--stagnant-gens", type=int, default=10)
    ap.add_argument("--min-pf", type=float, default=1.2)
    ap.add_argument("--max-dd", type=float, default=0.2)
    ap.add_argument("--min-trades", type=int, default=30)
    ap.add_argument("--min-stability", type=float, default=0.75)
    ap.add_argument("--test-only", default="false", help="true|false")
    ap.add_argument("--self-test", action="store_true", help="run built-in lightweight self test and exit")
    ap.add_argument("--once", action="store_true", help="run one poll and exit")
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    apply_thread_caps()

    if args.self_test:
        return run_self_test()

    repo_root = Path(args.repo_root).resolve()
    cfg = MonitorConfig(
        repo_root=repo_root,
        run_id_arg=str(args.run_id).strip() or "auto",
        poll_sec=max(60, int(args.poll_sec)),
        stagnant_gens=max(1, int(args.stagnant_gens)),
        thresholds=Thresholds(
            min_pf=float(args.min_pf),
            max_dd=float(args.max_dd),
            min_trades=max(0, int(args.min_trades)),
            min_stability=float(args.min_stability),
        ),
        test_only=parse_bool(args.test_only),
    )
    agent = MonitorAgent(cfg)
    if args.once:
        agent.poll_once()
        return 0
    agent.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
