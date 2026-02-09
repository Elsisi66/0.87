from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.bot087.execution.execution_eval import (
    ExecutionEvalConfig,
    evaluate_execution_from_trades,
    load_and_prepare_execution_data,
)
from src.bot087.optim.execution_ga import ExecutionGAConfig, run_execution_ga


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _save_report(base_dir: Path, symbol: str, payload: dict, adjusted_df: pd.DataFrame | None = None) -> Path:
    run_dir = base_dir / symbol.upper() / _utc_tag()
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "exec_check_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if adjusted_df is not None and not adjusted_df.empty:
        adjusted_df.to_csv(run_dir / "adjusted_trades.csv", index=False)
    return report_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--trades", required=True, help="Backtest trades file (csv/parquet/json)")
    ap.add_argument("--mode", choices=["klines1s", "aggtrades"], default="klines1s")
    ap.add_argument("--window", type=int, default=15, help="Execution lookup window in seconds")
    ap.add_argument("--entry-delay", type=int, default=0)
    ap.add_argument("--exit-delay", type=int, default=0)
    ap.add_argument("--fee-bps", type=float, default=7.0)
    ap.add_argument("--slippage-bps", type=float, default=2.0)
    ap.add_argument("--cap-gb", type=float, default=20.0)
    ap.add_argument("--cache-root", default="data/processed/execution_1s")
    ap.add_argument("--report-root", default="artifacts/reports/execution")

    ap.add_argument("--run-ga", action="store_true", help="Optimize execution knobs with GA on loaded trades + 1s cache")
    ap.add_argument("--ga-pop", type=int, default=28)
    ap.add_argument("--ga-gens", type=int, default=50)
    ap.add_argument("--ga-early-stop", type=int, default=20)
    ap.add_argument("--ga-seed", type=int, default=42)
    args = ap.parse_args()

    cfg = ExecutionEvalConfig(
        mode=args.mode,
        window_sec=int(args.window),
        entry_delay_sec=int(args.entry_delay),
        exit_delay_sec=int(args.exit_delay),
        cap_gb=float(args.cap_gb),
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slippage_bps),
        cache_root=str(Path(args.cache_root).resolve()),
    )

    out = evaluate_execution_from_trades(symbol=args.symbol.upper(), trades_path=args.trades, cfg=cfg)
    adjusted_df = out.get("adjusted_trades")
    summary = out.get("summary", {})

    payload = {
        "symbol": out.get("symbol", args.symbol.upper()),
        "mode": cfg.mode,
        "window_sec": cfg.window_sec,
        "entry_delay_sec": cfg.entry_delay_sec,
        "exit_delay_sec": cfg.exit_delay_sec,
        "summary": summary,
        "cache_root": out.get("cache_root"),
        "time_range": out.get("time_range"),
    }

    if args.run_ga:
        trades, sec_df, time_range = load_and_prepare_execution_data(symbol=args.symbol.upper(), trades_path=args.trades, cfg=cfg)
        ga_cfg = ExecutionGAConfig(
            pop_size=int(args.ga_pop),
            generations=int(args.ga_gens),
            elite_k=6,
            mutation_rate=0.4,
            seed=int(args.ga_seed),
            early_stop_patience=int(args.ga_early_stop),
        )
        best_ind, ga_report = run_execution_ga(
            symbol=args.symbol.upper(),
            trades=trades,
            sec_df=sec_df,
            base_eval_cfg=cfg,
            ga_cfg=ga_cfg,
        )
        payload["ga"] = {
            "best_individual": best_ind,
            "history": ga_report.get("history", []),
            "time_range": time_range,
        }

    report_path = _save_report(Path(args.report_root).resolve(), args.symbol.upper(), payload, adjusted_df=adjusted_df)

    print("=== EXEC CHECK ===")
    print(f"symbol: {payload['symbol']}")
    print(f"mode: {payload['mode']}")
    print(f"window_sec: {payload['window_sec']}")
    print(f"trades: {summary.get('trades', 0)}")
    print(f"orig_net: {summary.get('orig_net', 0.0):.2f}")
    print(f"adj_net: {summary.get('adj_net', 0.0):.2f}")
    print(f"delta_net: {summary.get('delta_net', 0.0):.2f}")
    print(f"improved_ratio: {summary.get('improved_ratio', 0.0):.3f}")
    if "ga" in payload:
        print(f"ga_best: {payload['ga']['best_individual']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
