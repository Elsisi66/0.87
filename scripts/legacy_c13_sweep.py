#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except ModuleNotFoundError:
    # Allow `python3 scripts/legacy_c13_sweep.py ...` to work on hosts where
    # deps are installed only in the repo venv.
    _venv_python = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    if __name__ == "__main__" and _venv_python.exists():
        os.execv(str(_venv_python), [str(_venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])
    raise


PROJECT_ROOT = Path(__file__).resolve()
for _p in [PROJECT_ROOT] + list(PROJECT_ROOT.parents):
    if (_p / "data").is_dir() and (_p / "src").is_dir():
        PROJECT_ROOT = _p
        break

os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_nla import _ensure_ohlc_cols, _load_df, run_backtest_nla  # noqa: E402
from src.bot087.optim.ga import _ensure_indicators, _norm_params  # noqa: E402
from src.bot087.params.portable import adapt_params  # noqa: E402


CSV_COLUMNS = [
    "symbol",
    "status",
    "fail_reason",
    "trades",
    "win_rate",
    "pf",
    "net",
    "dd",
    "final_equity",
    "start",
    "end",
    "notes",
]

VALID_STATUSES = {"OK", "NO_DATA", "ERROR", "INVALID_PARAMS", "ZERO_TRADES"}
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "legacy_c13_sweep_summary.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_date_only(value: str) -> bool:
    try:
        return len(str(value).strip()) == 10
    except Exception:
        return False


def _inclusive_end_ts(value: str) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    if _is_date_only(value):
        return ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def _normalize_symbol(raw: str) -> str:
    s = str(raw).strip().upper().replace(" ", "")
    if not s:
        return ""
    if "/" in s:
        parts = [x for x in s.split("/") if x]
        if len(parts) == 2:
            s = parts[0] + parts[1]
    known_quotes = ("USDT", "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "USD")
    if any(s.endswith(q) and len(s) > len(q) for q in known_quotes):
        return s
    if s.isalpha():
        return f"{s}USDT"
    return s


def _dedupe_symbols(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        sym = _normalize_symbol(raw)
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _extract_universe_symbols(payload: Any) -> List[str]:
    if isinstance(payload, dict):
        if isinstance(payload.get("symbols"), list):
            return _dedupe_symbols([str(x) for x in payload["symbols"]])
        if isinstance(payload.get("selected"), list):
            vals = [
                str(item.get("symbol", ""))
                for item in payload["selected"]
                if isinstance(item, dict)
            ]
            return _dedupe_symbols(vals)
        if isinstance(payload.get("selected_models"), list):
            vals = [
                str(item.get("symbol", ""))
                for item in payload["selected_models"]
                if isinstance(item, dict)
            ]
            return _dedupe_symbols(vals)
        if payload.get("symbol"):
            return _dedupe_symbols([str(payload.get("symbol"))])
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict):
            vals = [str(item.get("symbol", "")) for item in payload if isinstance(item, dict)]
            return _dedupe_symbols(vals)
        return _dedupe_symbols([str(x) for x in payload])
    return []


def _find_universe_file() -> Tuple[Optional[Path], str]:
    p = PROJECT_ROOT / "config" / "universe.json"
    if p.exists():
        return p, "config/universe.json"

    alt_dir = PROJECT_ROOT / "artifacts" / "universe"
    if alt_dir.exists():
        files = sorted(
            [x for x in alt_dir.glob("*.json") if x.is_file()],
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0], f"fallback:{files[0].relative_to(PROJECT_ROOT)}"
    return None, "missing"


def _load_universe_defaults() -> Dict[str, Any]:
    universe_fp, source = _find_universe_file()
    if universe_fp is None:
        return {
            "symbols": [],
            "start": None,
            "end": None,
            "source": source,
            "notes": "universe file not found",
        }

    payload = _load_json(universe_fp)
    start = payload.get("start") if isinstance(payload, dict) else None
    end = payload.get("end") if isinstance(payload, dict) else None
    symbols = _extract_universe_symbols(payload)

    note = ""
    if source.startswith("fallback:"):
        note = f"used_universe_fallback={source.split(':', 1)[1]}"
    return {
        "symbols": symbols,
        "start": str(start) if start else None,
        "end": str(end) if end else None,
        "source": source,
        "notes": note,
    }


def _validate_params(p: Dict[str, Any]) -> None:
    for key in ("willr_by_cycle", "tp_mult_by_cycle", "sl_mult_by_cycle", "exit_rsi_by_cycle"):
        v = p.get(key)
        if not isinstance(v, list) or len(v) != 5:
            raise ValueError(f"Invalid {key}: expected len=5 list")
        for item in v:
            float(item)

    cycles = p.get("trade_cycles")
    if not isinstance(cycles, list) or not cycles:
        raise ValueError("Invalid trade_cycles")
    for c in cycles:
        int(c)


def _resolve_path(path_value: str) -> Path:
    p = Path(path_value).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def _short_exc(exc: Exception) -> str:
    msg = str(exc).strip()
    if not msg:
        msg = exc.__class__.__name__
    return msg[:240]


def _blank_row(symbol: str, start: Optional[str], end: Optional[str]) -> Dict[str, Any]:
    row = {k: "" for k in CSV_COLUMNS}
    row["symbol"] = symbol
    row["status"] = "ERROR"
    row["start"] = start or ""
    row["end"] = end or ""
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Legacy C13 sweep with old params template.")
    ap.add_argument("--symbols", default=None, help="Optional comma-separated symbols, e.g. BTC,ADA,SOL,AVAX")
    ap.add_argument("--params_template", required=True, help="Path to older C13 params JSON template")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (inclusive); default universe start")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (inclusive); default universe end")
    ap.add_argument("--out", default="reports/legacy_c13_sweep.csv", help="Output CSV path")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    universe = _load_universe_defaults()

    if args.symbols:
        symbols = _dedupe_symbols([x for x in str(args.symbols).split(",") if x.strip()])
    else:
        symbols = _dedupe_symbols(universe.get("symbols", []))
        if not symbols:
            raise SystemExit("No symbols provided and no symbols found in config/universe.json")

    start = str(args.start).strip() if args.start else universe.get("start")
    end = str(args.end).strip() if args.end else universe.get("end")

    params_template_fp = _resolve_path(str(args.params_template))
    out_csv_fp = _resolve_path(str(args.out))
    summary_fp = DEFAULT_SUMMARY_PATH.resolve()

    template_obj = _load_json(params_template_fp)
    if not isinstance(template_obj, dict):
        raise SystemExit(f"params template must be a JSON object: {params_template_fp}")

    rows: List[Dict[str, Any]] = []
    counts: Counter = Counter()

    for symbol in symbols:
        row = _blank_row(symbol=symbol, start=start, end=end)
        notes: List[str] = []
        if universe.get("notes"):
            notes.append(str(universe["notes"]))

        try:
            adapted = adapt_params(template_obj, symbol=symbol)
            adapt_notes = str(adapted.pop("__adapt_notes", "")).strip()
            if adapt_notes:
                notes.append(adapt_notes)

            prev_cycles = adapted.get("trade_cycles")
            adapted["trade_cycles"] = [1, 3]
            adapted["require_trade_cycles"] = True
            if prev_cycles != [1, 3]:
                notes.append(f"forced_trade_cycles={prev_cycles}->{[1,3]}")

            params = _norm_params(adapted)
            _validate_params(params)
        except Exception as exc:
            row["status"] = "INVALID_PARAMS"
            row["fail_reason"] = _short_exc(exc)
            row["notes"] = "; ".join(notes)
            rows.append(row)
            counts[row["status"]] += 1
            continue

        try:
            df = _load_df(symbol, tf="1h")
            df = _ensure_ohlc_cols(df)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

            if start:
                start_ts = pd.to_datetime(start, utc=True)
                df = df[df["Timestamp"] >= start_ts].reset_index(drop=True)
            if end:
                end_ts = _inclusive_end_ts(end)
                df = df[df["Timestamp"] <= end_ts].reset_index(drop=True)
        except FileNotFoundError as exc:
            row["status"] = "NO_DATA"
            row["fail_reason"] = _short_exc(exc)
            row["notes"] = "; ".join(notes)
            rows.append(row)
            counts[row["status"]] += 1
            continue
        except Exception as exc:
            row["status"] = "ERROR"
            row["fail_reason"] = _short_exc(exc)
            row["notes"] = "; ".join(notes)
            rows.append(row)
            counts[row["status"]] += 1
            continue

        if df.empty:
            row["status"] = "NO_DATA"
            row["fail_reason"] = "No data after date filter"
            row["notes"] = "; ".join(notes)
            rows.append(row)
            counts[row["status"]] += 1
            continue

        try:
            dfi = _ensure_indicators(df, params)
            _, metrics = run_backtest_nla(
                df=dfi,
                symbol=symbol,
                p=params,
                initial_equity=10_000.0,
                fee_bps=7.0,
                slip_bps=2.0,
                cycle_shift=int(params.get("cycle_shift", 1)),
            )
        except Exception as exc:
            row["status"] = "ERROR"
            row["fail_reason"] = _short_exc(exc)
            row["notes"] = "; ".join(notes)
            rows.append(row)
            counts[row["status"]] += 1
            continue

        trades = int(float(metrics.get("trades", 0.0)))
        row["trades"] = trades
        row["win_rate"] = float(metrics.get("win_rate_pct", 0.0))
        row["pf"] = float(metrics.get("profit_factor", 0.0))
        row["net"] = float(metrics.get("net_profit", 0.0))
        row["dd"] = float(metrics.get("max_dd_pct", 0.0))
        row["final_equity"] = float(metrics.get("final_equity", 0.0))
        row["notes"] = "; ".join(notes)

        if trades <= 0:
            row["status"] = "ZERO_TRADES"
            row["fail_reason"] = "No trades generated"
        else:
            row["status"] = "OK"
            row["fail_reason"] = ""

        if row["status"] not in VALID_STATUSES:
            row["status"] = "ERROR"
            row["fail_reason"] = f"Unexpected status normalization: {row['status']}"

        rows.append(row)
        counts[row["status"]] += 1

    _write_csv(out_csv_fp, rows)

    summary_payload = {
        "generated_utc": _utc_now_iso(),
        "params_template": str(params_template_fp),
        "out_csv": str(out_csv_fp),
        "summary_csv_rows": len(rows),
        "symbols": symbols,
        "start": start,
        "end": end,
        "universe_source": universe.get("source"),
        "counts_by_status": {
            "OK": int(counts.get("OK", 0)),
            "NO_DATA": int(counts.get("NO_DATA", 0)),
            "ERROR": int(counts.get("ERROR", 0)),
            "INVALID_PARAMS": int(counts.get("INVALID_PARAMS", 0)),
            "ZERO_TRADES": int(counts.get("ZERO_TRADES", 0)),
        },
    }
    summary_fp.parent.mkdir(parents=True, exist_ok=True)
    summary_fp.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"wrote_csv={out_csv_fp}")
    print(f"wrote_summary={summary_fp}")
    print(f"counts_by_status={json.dumps(summary_payload['counts_by_status'], sort_keys=True)}")


if __name__ == "__main__":
    main()
