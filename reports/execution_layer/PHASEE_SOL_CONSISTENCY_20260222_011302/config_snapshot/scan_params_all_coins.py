#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.optim import ga as ga_long  # noqa: E402
from src.bot087.optim import ga_short  # noqa: E402


SYMBOL_RE = re.compile(r"([A-Z0-9]{2,20}USDT)")


@dataclass
class Thresholds:
    min_net_profit: float
    min_profit_factor: float
    min_cagr_pct: float
    max_dd_pct: float
    min_trades: float
    min_trades_per_year: float


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(raw, dict) and isinstance(raw.get("params"), dict):
        return dict(raw["params"])
    return dict(raw)


def normalize_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename: Dict[str, str] = {}
    for src, dst in (("timestamp", "Timestamp"), ("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")):
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
        raise ValueError(f"missing columns {miss}")
    return out


def load_symbol_df(symbol: str, tf: str = "1h") -> pd.DataFrame:
    full_fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_{tf}_full.parquet"
    if full_fp.exists():
        return normalize_ohlcv_cols(pd.read_parquet(full_fp))

    par_fp = PROJECT_ROOT / "data" / "parquet" / f"{symbol}.parquet"
    if par_fp.exists():
        return normalize_ohlcv_cols(pd.read_parquet(par_fp))

    proc_dir = PROJECT_ROOT / "data" / "processed"
    csv_files = sorted(proc_dir.glob(f"{symbol}_*_proc.csv"))
    if csv_files:
        df = pd.concat([pd.read_csv(fp) for fp in csv_files], ignore_index=True)
        return normalize_ohlcv_cols(df)

    raise FileNotFoundError(f"No data found for {symbol}")


def discover_param_files(params_dir: Path, patterns: Iterable[str]) -> List[Path]:
    found: List[Path] = []
    seen = set()
    for pat in patterns:
        for fp in sorted(params_dir.glob(pat)):
            if fp.is_file() and fp.suffix.lower() == ".json":
                key = str(fp.resolve())
                if key not in seen:
                    seen.add(key)
                    found.append(fp)
    return found


def extract_symbol_from_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    m = SYMBOL_RE.search(str(value).upper())
    return m.group(1) if m else None


def infer_symbol_side(params_file: Path, payload: Dict[str, Any]) -> Tuple[str, str, str]:
    candidates: List[Any] = [params_file.stem]
    source = "filename"

    if isinstance(payload, dict):
        for key in ("symbol", "ga_symbol", "name"):
            if key in payload:
                candidates.append(payload.get(key))
        meta = payload.get("meta")
        if isinstance(meta, dict):
            for key in ("symbol", "ga_symbol"):
                if key in meta:
                    candidates.append(meta.get(key))
            ga_saved = meta.get("ga_saved")
            if isinstance(ga_saved, dict):
                candidates.append(ga_saved.get("active_params"))

    symbol: Optional[str] = None
    for cand in candidates:
        sym = extract_symbol_from_text(cand)
        if sym:
            symbol = sym
            source = "json" if cand is not params_file.stem else "filename"
            break

    if not symbol:
        raise ValueError("Could not infer symbol")

    side: Optional[str] = None
    if isinstance(payload, dict):
        s = str(payload.get("side", "")).strip().lower()
        if s in {"long", "short"}:
            side = s
        meta = payload.get("meta")
        if not side and isinstance(meta, dict):
            ms = str(meta.get("side", "")).strip().lower()
            if ms in {"long", "short"}:
                side = ms

    name_l = params_file.name.lower()
    if not side:
        if "short" in name_l:
            side = "short"
        elif "long" in name_l:
            side = "long"

    if not side:
        sym_blob = " ".join(str(x).lower() for x in candidates if x is not None)
        side = "short" if "short" in sym_blob else "long"

    return symbol, side, source


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    seconds = (end - start).total_seconds()
    return max(0.0, float(seconds / (365.25 * 24.0 * 3600.0)))


def normalize_dd_pct(metrics: Dict[str, Any]) -> Optional[float]:
    if "max_dd_pct" in metrics:
        raw = metrics.get("max_dd_pct")
    else:
        raw = metrics.get("max_dd")
    if raw is None:
        return None
    try:
        x = float(raw)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x * 100.0 if x <= 1.5 else x


def safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def compute_pass_score(
    metrics: Dict[str, Any],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    thresholds: Thresholds,
) -> Dict[str, Any]:
    years = years_between(period_start, period_end)
    initial = safe_float(metrics.get("initial_equity")) or 0.0
    final = safe_float(metrics.get("final_equity")) or initial
    net_profit = safe_float(metrics.get("net_profit"))
    if net_profit is None:
        net_profit = final - initial

    return_pct = ((final / initial) - 1.0) * 100.0 if initial > 0 else 0.0
    cagr_pct = 0.0
    if years > 0 and initial > 0 and final > 0:
        cagr_pct = ((final / initial) ** (1.0 / years) - 1.0) * 100.0

    trades = safe_float(metrics.get("trades")) or 0.0
    trades_per_year = (trades / years) if years > 0 else 0.0
    profit_factor = safe_float(metrics.get("profit_factor")) or 0.0
    max_dd = safe_float(metrics.get("max_dd"))
    max_dd_pct = normalize_dd_pct(metrics)

    score = -1e18
    if max_dd_pct is not None:
        score = (cagr_pct * profit_factor) / (1.0 + max_dd_pct)

    trades_ok = trades >= thresholds.min_trades
    if not trades_ok and years > 0:
        trades_ok = trades_per_year >= thresholds.min_trades_per_year

    pass_flag = (
        (net_profit > thresholds.min_net_profit)
        and (profit_factor >= thresholds.min_profit_factor)
        and (cagr_pct >= thresholds.min_cagr_pct)
        and (max_dd_pct is not None and max_dd_pct <= thresholds.max_dd_pct)
        and trades_ok
    )

    return {
        "period_start": str(period_start),
        "period_end": str(period_end),
        "years": years,
        "initial_equity": initial,
        "final_equity": final,
        "net_profit": net_profit,
        "return_pct": return_pct,
        "cagr_pct": cagr_pct,
        "trades": trades,
        "trades_per_year": trades_per_year,
        "profit_factor": profit_factor,
        "max_dd": max_dd,
        "max_dd_pct": max_dd_pct,
        "score": score,
        "pass": bool(pass_flag),
    }


def flatten_for_csv(record: Dict[str, Any]) -> Dict[str, Any]:
    flat = {k: v for k, v in record.items() if k not in {"metrics", "error_trace"}}
    metrics = record.get("metrics", {})
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                if k not in flat:
                    flat[k] = v
    return flat


def to_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def write_results_md(
    out_md: Path,
    rows: List[Dict[str, Any]],
    thresholds: Thresholds,
    ranking_formula: str,
    best_side: str,
    top_n: int,
) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    pass_rows = [r for r in ok_rows if bool(r.get("pass"))]
    err_rows = [r for r in rows if r.get("status") == "error"]

    rank_rows = sorted(ok_rows, key=lambda r: float(r.get("score", -1e18)), reverse=True)
    top_rows = rank_rows[:top_n]

    lines: List[str] = []
    lines.append("# Params Scan Results")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Total runs: {len(rows)}")
    lines.append(f"- Successful runs: {len(ok_rows)}")
    lines.append(f"- Errors: {len(err_rows)}")
    lines.append(f"- Passing runs: {len(pass_rows)}")
    lines.append("")
    lines.append("## Score & Pass Rules")
    lines.append("")
    lines.append(f"- Score formula: `{ranking_formula}`")
    lines.append("- Pass thresholds:")
    lines.append(f"  - `net_profit > {thresholds.min_net_profit}`")
    lines.append(f"  - `profit_factor >= {thresholds.min_profit_factor}`")
    lines.append(f"  - `cagr_pct >= {thresholds.min_cagr_pct}`")
    lines.append(f"  - `max_dd_pct <= {thresholds.max_dd_pct}`")
    lines.append(f"  - `trades >= {thresholds.min_trades}` OR `trades_per_year >= {thresholds.min_trades_per_year}`")
    lines.append(f"- `best_by_symbol.csv` side filter: `{best_side}`")
    lines.append("")

    lines.append(f"## Top {len(top_rows)} By Score")
    lines.append("")
    lines.append("| rank | symbol | side | pass | score | cagr_pct | pf | max_dd_pct | net_profit | trades | params_file |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for idx, r in enumerate(top_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(r.get("symbol", "")),
                    str(r.get("side", "")),
                    str(bool(r.get("pass"))),
                    f"{float(r.get('score', 0.0)):.6f}",
                    f"{float(r.get('cagr_pct', 0.0)):.4f}",
                    f"{float(r.get('profit_factor', 0.0)):.4f}",
                    f"{float(r.get('max_dd_pct', 0.0)):.4f}",
                    f"{float(r.get('net_profit', 0.0)):.4f}",
                    f"{float(r.get('trades', 0.0)):.1f}",
                    str(r.get("params_file", "")),
                ]
            )
            + " |"
        )

    if pass_rows:
        lines.append("")
        lines.append("## Passing Runs")
        lines.append("")
        pass_sorted = sorted(pass_rows, key=lambda r: float(r.get("score", -1e18)), reverse=True)
        for r in pass_sorted:
            lines.append(
                f"- `{r.get('symbol')}` ({r.get('side')}): score={float(r.get('score', 0.0)):.6f}, "
                f"cagr={float(r.get('cagr_pct', 0.0)):.2f}%, pf={float(r.get('profit_factor', 0.0)):.3f}, "
                f"dd={float(r.get('max_dd_pct', 0.0)):.2f}%, params=`{r.get('params_file')}`"
            )

    if err_rows:
        lines.append("")
        lines.append("## Errors (first 20)")
        lines.append("")
        for r in err_rows[:20]:
            lines.append(
                f"- `{r.get('params_file')}` -> {r.get('error_type', 'Error')}: {r.get('error_message', '')}"
            )

    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_scan(args: argparse.Namespace) -> Path:
    params_dir = (PROJECT_ROOT / args.params_dir).resolve()
    patterns = [x.strip() for x in str(args.params_globs).split(",") if x.strip()]
    files = discover_param_files(params_dir, patterns)
    if not files:
        raise SystemExit(f"No params files found in {params_dir} for patterns={patterns}")

    run_id = args.run_id.strip() if str(args.run_id).strip() else utc_tag()
    out_dir = (PROJECT_ROOT / "reports" / "params_scan" / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = Thresholds(
        min_net_profit=float(args.min_net_profit),
        min_profit_factor=float(args.min_profit_factor),
        min_cagr_pct=float(args.min_cagr_pct),
        max_dd_pct=float(args.max_dd_pct),
        min_trades=float(args.min_trades),
        min_trades_per_year=float(args.min_trades_per_year),
    )

    ranking_formula = "score = (cagr_pct * profit_factor) / (1 + max_dd_pct)"

    df_cache: Dict[str, pd.DataFrame] = {}
    records: List[Dict[str, Any]] = []

    for idx, params_fp in enumerate(files, start=1):
        rel_fp = to_rel(params_fp)
        base: Dict[str, Any] = {
            "run_id": run_id,
            "idx": idx,
            "params_file": rel_fp,
            "status": "error",
            "symbol": "",
            "side": "",
            "symbol_infer_source": "",
            "tf": args.tf,
            "fee_bps": float(args.fee_bps),
            "slip_bps": float(args.slip_bps),
            "initial_equity": float(args.initial_equity),
            "score": -1e18,
            "pass": False,
        }
        try:
            payload = load_json(params_fp)
            symbol, side, sym_source = infer_symbol_side(params_fp, payload)

            if args.side_filter != "all" and side != args.side_filter:
                continue

            if symbol not in df_cache:
                df_cache[symbol] = load_symbol_df(symbol=symbol, tf=args.tf)
            df_base = df_cache[symbol]

            raw_params = unwrap_params(payload)
            if side == "short":
                p = ga_short._norm_params(raw_params)
                df_feat = ga_short._ensure_indicators(df_base.copy(), p)
                _, metrics = ga_short.run_backtest_short_only(
                    df=df_feat,
                    symbol=symbol,
                    p=p,
                    initial_equity=float(args.initial_equity),
                    fee_bps=float(args.fee_bps),
                    slippage_bps=float(args.slip_bps),
                )
            else:
                p = ga_long._norm_params(raw_params)
                df_feat = ga_long._ensure_indicators(df_base.copy(), p)
                _, metrics = ga_long.run_backtest_long_only(
                    df=df_feat,
                    symbol=symbol,
                    p=p,
                    initial_equity=float(args.initial_equity),
                    fee_bps=float(args.fee_bps),
                    slippage_bps=float(args.slip_bps),
                    collect_trades=False,
                    assume_prepared=True,
                )

            if df_feat.empty:
                raise ValueError("Prepared dataframe is empty")

            summary = compute_pass_score(
                metrics=metrics,
                period_start=pd.to_datetime(df_feat["Timestamp"].iloc[0], utc=True),
                period_end=pd.to_datetime(df_feat["Timestamp"].iloc[-1], utc=True),
                thresholds=thresholds,
            )

            rec = {
                **base,
                **summary,
                "status": "ok",
                "symbol": symbol,
                "side": side,
                "symbol_infer_source": sym_source,
                "metrics": metrics,
            }
            records.append(rec)
        except Exception as ex:
            tb = traceback.format_exc(limit=5)
            rec = {
                **base,
                "status": "error",
                "error_type": type(ex).__name__,
                "error_message": str(ex),
                "error_trace": tb,
            }
            records.append(rec)

    if not records:
        raise SystemExit("No records produced. Check --side-filter and discovered params files.")

    results_jsonl = out_dir / "results.jsonl"
    with results_jsonl.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True, default=str) + "\n")

    csv_rows = [flatten_for_csv(r) for r in records]
    df_csv = pd.DataFrame(csv_rows)
    sort_cols = [c for c in ("status", "pass", "score") if c in df_csv.columns]
    if sort_cols:
        asc = [True, False, False][: len(sort_cols)]
        df_csv = df_csv.sort_values(sort_cols, ascending=asc)
    df_csv.to_csv(out_dir / "results.csv", index=False)

    ok_df = df_csv[df_csv.get("status") == "ok"].copy()
    if args.best_side != "any":
        ok_df = ok_df[ok_df.get("side") == args.best_side].copy()

    best_cols = [
        "symbol",
        "side",
        "params_file",
        "score",
        "pass",
        "period_start",
        "period_end",
        "years",
        "initial_equity",
        "final_equity",
        "net_profit",
        "return_pct",
        "cagr_pct",
        "trades",
        "trades_per_year",
        "win_rate_pct",
        "profit_factor",
        "max_dd",
        "max_dd_pct",
    ]

    if not ok_df.empty:
        ok_df["score"] = pd.to_numeric(ok_df["score"], errors="coerce").fillna(-1e18)
        ok_df = ok_df.sort_values(["symbol", "score"], ascending=[True, False])
        best_df = ok_df.groupby("symbol", as_index=False).head(1).reset_index(drop=True)
        keep_cols = [c for c in best_cols if c in best_df.columns]
        best_df = best_df[keep_cols]
    else:
        best_df = pd.DataFrame(columns=best_cols)

    best_df.to_csv(out_dir / "best_by_symbol.csv", index=False)

    write_results_md(
        out_md=out_dir / "results.md",
        rows=records,
        thresholds=thresholds,
        ranking_formula=ranking_formula,
        best_side=args.best_side,
        top_n=int(args.top_n),
    )

    scan_meta = {
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "params_dir": str(params_dir),
        "params_globs": patterns,
        "tf": args.tf,
        "fee_bps": float(args.fee_bps),
        "slip_bps": float(args.slip_bps),
        "initial_equity": float(args.initial_equity),
        "side_filter": args.side_filter,
        "best_side": args.best_side,
        "ranking_formula": ranking_formula,
        "thresholds": {
            "min_net_profit": thresholds.min_net_profit,
            "min_profit_factor": thresholds.min_profit_factor,
            "min_cagr_pct": thresholds.min_cagr_pct,
            "max_dd_pct": thresholds.max_dd_pct,
            "min_trades": thresholds.min_trades,
            "min_trades_per_year": thresholds.min_trades_per_year,
        },
        "counts": {
            "total": int(len(records)),
            "ok": int(sum(1 for r in records if r.get("status") == "ok")),
            "error": int(sum(1 for r in records if r.get("status") == "error")),
            "pass": int(sum(1 for r in records if r.get("status") == "ok" and bool(r.get("pass")))),
        },
    }
    (out_dir / "scan_meta.json").write_text(json.dumps(scan_meta, indent=2), encoding="utf-8")

    print(str(out_dir))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Scan active params across all coins using existing backtesters.")
    ap.add_argument("--params-dir", default="data/metadata/params")
    ap.add_argument("--params-globs", default="*active*params*.json")
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--side-filter", choices=["all", "long", "short"], default="all")
    ap.add_argument("--best-side", choices=["any", "long", "short"], default="long")

    ap.add_argument("--min-net-profit", type=float, default=0.0)
    ap.add_argument("--min-profit-factor", type=float, default=1.15)
    ap.add_argument("--min-cagr-pct", type=float, default=15.0)
    ap.add_argument("--max-dd-pct", type=float, default=35.0)
    ap.add_argument("--min-trades", type=float, default=50.0)
    ap.add_argument("--min-trades-per-year", type=float, default=10.0)

    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--run-id", default="", help="Optional run folder name under reports/params_scan")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_scan(args)


if __name__ == "__main__":
    main()
