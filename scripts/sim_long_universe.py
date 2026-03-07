#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bot087.optim import ga as ga_long  # noqa: E402


@dataclass
class CoinModel:
    symbol: str
    params_file: Path
    first_ts: pd.Timestamp
    last_ts: pd.Timestamp


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in {"1", "true", "yes", "y", "t"}


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    seconds = (end - start).total_seconds()
    return max(0.0, float(seconds / (365.25 * 24.0 * 3600.0)))


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    arr = series.astype(float)
    runmax = arr.cummax()
    dd = (runmax - arr) / runmax.clip(lower=1e-9)
    return float(dd.max()) if len(dd) else 0.0


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unwrap_params(payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("params"), dict):
        return dict(payload["params"])
    return dict(payload)


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
    fp_full = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_{tf}_full.parquet"
    if fp_full.exists():
        return normalize_ohlcv_cols(pd.read_parquet(fp_full))

    fp_par = PROJECT_ROOT / "data" / "parquet" / f"{symbol}.parquet"
    if fp_par.exists():
        return normalize_ohlcv_cols(pd.read_parquet(fp_par))

    proc_dir = PROJECT_ROOT / "data" / "processed"
    files = sorted(proc_dir.glob(f"{symbol}_*_proc.csv"))
    if files:
        return normalize_ohlcv_cols(pd.concat([pd.read_csv(x) for x in files], ignore_index=True))

    raise FileNotFoundError(f"No data found for {symbol}")


def resolve_path(path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def find_latest_scan_dir() -> Path:
    base = PROJECT_ROOT / "reports" / "params_scan"
    if not base.exists():
        raise SystemExit(f"Missing directory: {base}")
    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        raise SystemExit(f"No scan runs found under {base}")
    dirs.sort(key=lambda d: d.name)
    return dirs[-1]


def determine_window(models: List[CoinModel], years: float) -> Tuple[List[CoinModel], Dict[str, str], pd.Timestamp, pd.Timestamp]:
    dropped: Dict[str, str] = {}
    selected = list(models)
    delta = pd.to_timedelta(float(years) * 365.25, unit="D")

    while True:
        if not selected:
            raise RuntimeError("No symbols left after history coverage checks")
        common_end = min(m.last_ts for m in selected)
        common_start = common_end - delta
        insufficient = [m for m in selected if m.first_ts > common_start]
        if not insufficient:
            return selected, dropped, common_start, common_end
        for m in insufficient:
            dropped[m.symbol] = (
                f"insufficient_history:first={m.first_ts.isoformat()} window_start={common_start.isoformat()}"
            )
        selected = [m for m in selected if m.symbol not in {x.symbol for x in insufficient}]


def build_universe_report_md(
    out_md: Path,
    run_info: Dict[str, Any],
    per_coin: List[Dict[str, Any]],
    dropped: Dict[str, str],
) -> None:
    lines: List[str] = []
    lines.append("# Long Universe Simulation")
    lines.append("")
    lines.append(f"- Generated UTC: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Scan dir: `{run_info['scan_dir']}`")
    lines.append(f"- Input best CSV: `{run_info['best_csv']}`")
    lines.append(f"- Capital (EUR): {run_info['capital_eur']:.2f}")
    lines.append(f"- Allocation per coin (EUR): {run_info['allocation_per_coin']:.6f}")
    lines.append(f"- Coins used: {run_info['coins_used']}")
    lines.append(f"- Window start: {run_info['window_start']}")
    lines.append(f"- Window end: {run_info['window_end']}")
    lines.append(f"- Years actual: {run_info['years']:.6f}")
    lines.append("")

    lines.append("## Universe Metrics")
    lines.append("")
    lines.append(f"- Initial equity: {run_info['initial_equity']:.6f}")
    lines.append(f"- Final equity: {run_info['final_equity']:.6f}")
    lines.append(f"- Net profit: {run_info['net_profit']:.6f}")
    lines.append(f"- Return %: {run_info['return_pct']:.6f}")
    lines.append(f"- CAGR %: {run_info['cagr_pct']:.6f}")
    lines.append(f"- Max drawdown: {run_info['max_dd']:.6f}")
    lines.append(f"- Max drawdown %: {run_info['max_dd_pct']:.6f}")
    lines.append("")

    lines.append("## Per-Coin Final Contributions")
    lines.append("")
    lines.append("| symbol | params_file | initial_alloc_eur | final_equity | net_profit | return_pct | trades | pf | max_dd_pct |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(per_coin, key=lambda x: float(x.get("final_equity", 0.0)), reverse=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("symbol", "")),
                    str(row.get("params_file", "")),
                    f"{float(row.get('initial_equity', 0.0)):.6f}",
                    f"{float(row.get('final_equity', 0.0)):.6f}",
                    f"{float(row.get('net_profit', 0.0)):.6f}",
                    f"{float(row.get('return_pct', 0.0)):.6f}",
                    f"{float(row.get('trades', 0.0)):.1f}",
                    f"{float(row.get('profit_factor', 0.0)):.6f}",
                    f"{float(row.get('max_dd_pct', 0.0)):.6f}",
                ]
            )
            + " |"
        )

    if dropped:
        lines.append("")
        lines.append("## Dropped Symbols")
        lines.append("")
        for sym, reason in sorted(dropped.items()):
            lines.append(f"- `{sym}`: {reason}")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Strategy logic is unchanged; simulation reuses `run_backtest_long_only` from `src/bot087/optim/ga.py`.")
    lines.append("- Equity curves were exported using the optional `return_equity_curve=True` hook in that same backtest function.")

    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def run_sim(args: argparse.Namespace) -> Path:
    scan_dir = resolve_path(args.scan_dir) if str(args.scan_dir).strip() else find_latest_scan_dir()
    best_csv = resolve_path(args.best_csv) if str(args.best_csv).strip() else (scan_dir / "best_by_symbol.csv").resolve()
    if not best_csv.exists():
        raise SystemExit(f"Missing best_by_symbol.csv: {best_csv}")

    out_dir = resolve_path(args.output_dir) if str(args.output_dir).strip() else scan_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    curves_dir = out_dir / "equity_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    best_df = pd.read_csv(best_csv)
    if best_df.empty:
        raise SystemExit(f"No rows in {best_csv}")

    if "side" not in best_df.columns:
        best_df["side"] = "long"

    mask_pass = best_df["pass"].map(as_bool)
    best_df = best_df[mask_pass].copy()
    best_df = best_df[best_df["side"].astype(str).str.lower() == "long"].copy()

    if best_df.empty:
        raise SystemExit("No passing long rows found in best_by_symbol.csv")

    data_cache: Dict[str, pd.DataFrame] = {}
    models: List[CoinModel] = []
    dropped: Dict[str, str] = {}

    for _, row in best_df.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        params_file = resolve_path(str(row.get("params_file", "")).strip())
        if not symbol or not params_file.exists():
            dropped[symbol or "UNKNOWN"] = f"missing_params_file:{params_file}"
            continue
        try:
            if symbol not in data_cache:
                data_cache[symbol] = load_symbol_df(symbol=symbol, tf=args.tf)
            df = data_cache[symbol]
            if df.empty:
                dropped[symbol] = "empty_dataframe"
                continue
            models.append(
                CoinModel(
                    symbol=symbol,
                    params_file=params_file,
                    first_ts=pd.to_datetime(df["Timestamp"].iloc[0], utc=True),
                    last_ts=pd.to_datetime(df["Timestamp"].iloc[-1], utc=True),
                )
            )
        except Exception as ex:
            dropped[symbol] = f"data_load_failed:{type(ex).__name__}:{ex}"

    if not models:
        raise SystemExit("No usable passing long symbols after loading datasets")

    selected, dropped_cov, window_start, window_end = determine_window(models=models, years=float(args.years))
    dropped.update(dropped_cov)

    if not selected:
        raise SystemExit("No symbols left after history coverage filter")

    allocation = float(args.capital_eur) / float(len(selected))

    per_coin_rows: List[Dict[str, Any]] = []
    coin_curves: Dict[str, pd.DataFrame] = {}

    for model in selected:
        df_full = data_cache[model.symbol]
        df_slice = df_full[(df_full["Timestamp"] >= window_start) & (df_full["Timestamp"] <= window_end)].reset_index(drop=True)
        if df_slice.empty:
            dropped[model.symbol] = "empty_window_after_slice"
            continue

        payload = load_json(model.params_file)
        p = ga_long._norm_params(unwrap_params(payload))
        df_feat = ga_long._ensure_indicators(df_slice.copy(), p)
        _, metrics = ga_long.run_backtest_long_only(
            df=df_feat,
            symbol=model.symbol,
            p=p,
            initial_equity=allocation,
            fee_bps=float(args.fee_bps),
            slippage_bps=float(args.slip_bps),
            collect_trades=False,
            assume_prepared=True,
            return_equity_curve=True,
        )

        ts = metrics.get("equity_timestamps", [])
        eq = metrics.get("equity_curve", [])
        if not isinstance(ts, list) or not isinstance(eq, list) or not ts or len(ts) != len(eq):
            dropped[model.symbol] = "invalid_equity_curve_from_backtest"
            continue

        col = f"{model.symbol}_equity"
        curve_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts, utc=True, errors="coerce"),
                col: pd.to_numeric(eq, errors="coerce"),
            }
        ).dropna(subset=["timestamp"]).sort_values("timestamp")

        if curve_df.empty:
            dropped[model.symbol] = "empty_equity_curve"
            continue

        curve_df = curve_df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        curve_df.to_csv(curves_dir / f"{model.symbol}_equity.csv", index=False)
        coin_curves[model.symbol] = curve_df

        initial_eq = float(metrics.get("initial_equity", allocation))
        final_eq = float(metrics.get("final_equity", initial_eq))
        net = float(metrics.get("net_profit", final_eq - initial_eq))
        ret = ((final_eq / initial_eq) - 1.0) * 100.0 if initial_eq > 0 else 0.0
        dd_raw = metrics.get("max_dd_pct", metrics.get("max_dd", 0.0))
        dd_pct = float(dd_raw) * 100.0 if float(dd_raw) <= 1.5 else float(dd_raw)

        per_coin_rows.append(
            {
                "symbol": model.symbol,
                "params_file": str(model.params_file),
                "initial_equity": initial_eq,
                "final_equity": final_eq,
                "net_profit": net,
                "return_pct": ret,
                "trades": float(metrics.get("trades", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 0.0)),
                "max_dd": float(metrics.get("max_dd", 0.0)),
                "max_dd_pct": dd_pct,
            }
        )

    if not per_coin_rows:
        raise SystemExit("No coin produced a valid equity curve for universe simulation")

    used_symbols = [r["symbol"] for r in per_coin_rows]
    timeline = pd.date_range(start=window_start, end=window_end, freq=args.freq, tz="UTC")
    universe_df = pd.DataFrame({"timestamp": timeline})

    for sym in used_symbols:
        col = f"{sym}_equity"
        curve = coin_curves[sym][["timestamp", col]].copy()
        universe_df = universe_df.merge(curve, on="timestamp", how="left")
        universe_df[col] = pd.to_numeric(universe_df[col], errors="coerce").ffill().fillna(allocation)

    coin_cols = [f"{sym}_equity" for sym in used_symbols]
    universe_df["universe_equity"] = universe_df[coin_cols].sum(axis=1)

    initial_total = float(args.capital_eur)
    final_total = float(universe_df["universe_equity"].iloc[-1]) if not universe_df.empty else initial_total
    net_profit = final_total - initial_total
    years_actual = years_between(
        pd.to_datetime(universe_df["timestamp"].iloc[0], utc=True),
        pd.to_datetime(universe_df["timestamp"].iloc[-1], utc=True),
    ) if not universe_df.empty else 0.0
    return_pct = ((final_total / initial_total) - 1.0) * 100.0 if initial_total > 0 else 0.0
    cagr_pct = ((final_total / initial_total) ** (1.0 / years_actual) - 1.0) * 100.0 if (years_actual > 0 and initial_total > 0 and final_total > 0) else 0.0
    max_dd = max_drawdown(universe_df["universe_equity"])

    universe_df.to_csv(out_dir / "universe_equity.csv", index=False)
    pd.DataFrame(per_coin_rows).sort_values("final_equity", ascending=False).to_csv(out_dir / "universe_per_coin.csv", index=False)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scan_dir": str(scan_dir),
        "best_csv": str(best_csv),
        "capital_eur": float(args.capital_eur),
        "allocation_per_coin": float(allocation),
        "years_requested": float(args.years),
        "years": float(years_actual),
        "window_start": str(window_start),
        "window_end": str(window_end),
        "coins_used": len(per_coin_rows),
        "symbols_used": used_symbols,
        "initial_equity": float(initial_total),
        "final_equity": float(final_total),
        "net_profit": float(net_profit),
        "return_pct": float(return_pct),
        "cagr_pct": float(cagr_pct),
        "max_dd": float(max_dd),
        "max_dd_pct": float(max_dd * 100.0),
        "fee_bps": float(args.fee_bps),
        "slip_bps": float(args.slip_bps),
        "dropped_symbols": dropped,
        "per_coin": per_coin_rows,
    }
    (out_dir / "universe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    build_universe_report_md(
        out_md=out_dir / "universe_report.md",
        run_info=summary,
        per_coin=per_coin_rows,
        dropped=dropped,
    )

    print(str(out_dir))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Simple long-universe simulation from passing best_by_symbol rows.")
    ap.add_argument("--scan-dir", default="", help="Path to reports/params_scan/<run_id>. Defaults to latest run.")
    ap.add_argument("--best-csv", default="", help="Path to best_by_symbol.csv. Defaults to <scan_dir>/best_by_symbol.csv")
    ap.add_argument("--output-dir", default="", help="Output directory. Defaults to scan_dir.")
    ap.add_argument("--capital-eur", type=float, default=250.0)
    ap.add_argument("--years", type=float, default=7.0)
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--freq", default="1h")
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slip-bps", type=float, default=2.0)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_sim(args)


if __name__ == "__main__":
    main()
