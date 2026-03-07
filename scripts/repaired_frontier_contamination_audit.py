#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
import sys

os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import backtest_exec_phasec_sol as phasec_bt  # noqa: E402
from scripts import phase_u_combined_1h3m_pilot as phase_u  # noqa: E402
from scripts import rebaseline_1h_contract_repair as rebaseline_1h  # noqa: E402
from scripts import scan_params_all_coins as scan  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402


PHASE_H_REP_SUBSET = (
    PROJECT_ROOT
    / "reports"
    / "execution_layer"
    / "PHASEE2_SOL_REPRESENTATIVE_20260222_021052"
    / "representative_subset_signals.csv"
)

DISCOVERY_EXPECTED = {
    "phase_i": "PHASEI_EXECAWARE_1H_GA_EXPANSION_20260224_012237",
    "params_scan": "20260220_044949",
    "repaired_baseline": "1H_CONTRACT_REPAIR_REBASELINE_20260301_140650",
}


@dataclass
class CandidateArtifacts:
    phase_i_dir: Path
    params_scan_dir: Path
    repaired_baseline_dir: Path
    phase_j0_dir: Optional[Path]
    repaired_signal_root: Optional[Path]


@dataclass
class AuditState:
    blocked: bool
    block_reason: str
    recommendation: str
    report_lines: List[str]
    manifest: Dict[str, Any]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_tag() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(pd.to_datetime(v, utc=True))
        if isinstance(v, Path):
            return str(v)
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if math.isfinite(x) else float("nan")


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def spearman_rank_corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    if np.all(a_arr == a_arr[0]) or np.all(b_arr == b_arr[0]):
        return float("nan")
    return float(np.corrcoef(a_arr, b_arr)[0, 1])


def tail_mean(series: pd.Series, frac: float) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(frac) * float(arr.size))))
    arr.sort()
    return float(arr[:k].mean())


def max_drawdown_from_returns(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    eq = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peaks, 1e-12) - 1.0
    return float(dd.min())


def profit_factor_from_returns(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    gains = arr[arr > 0.0].sum()
    losses = -arr[arr < 0.0].sum()
    if losses <= 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return float(gains / losses)


def markdown_table(df: pd.DataFrame, n: int = 8) -> str:
    if df.empty:
        return "| (empty) |\n| --- |\n| (no rows) |"
    view = df.head(n).copy()
    cols = [str(c) for c in view.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, sep]
    for _, row in view.iterrows():
        vals: List[str] = []
        for c in view.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append("" if not math.isfinite(v) else f"{v:.10g}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def find_latest_complete(root: Path, pattern: str, required: Iterable[str]) -> Optional[Path]:
    cands = sorted([p for p in root.glob(pattern) if p.is_dir()], key=lambda p: p.name)
    for cand in reversed(cands):
        if all((cand / req).exists() for req in required):
            return cand.resolve()
    return None


def discover_artifacts(args: argparse.Namespace) -> CandidateArtifacts:
    exec_root = PROJECT_ROOT / "reports" / "execution_layer"
    params_root = PROJECT_ROOT / "reports" / "params_scan"

    phase_i_dir = (
        Path(args.phase_i_dir).resolve()
        if args.phase_i_dir
        else find_latest_complete(
            exec_root,
            "PHASEI_EXECAWARE_1H_GA_EXPANSION_*",
            ["phaseI2_ga_results.csv", "phaseI_frontier_comparison_vs_H.csv", "phaseI_run_manifest.json"],
        )
    )
    params_scan_dir = (
        Path(args.params_scan_dir).resolve()
        if args.params_scan_dir
        else find_latest_complete(params_root, "*", ["best_by_symbol.csv", "universe_summary.json"])
    )
    repaired_baseline_dir = (
        Path(args.repaired_baseline_dir).resolve()
        if args.repaired_baseline_dir
        else find_latest_complete(exec_root, "1H_CONTRACT_REPAIR_REBASELINE_*", ["repaired_1h_reference_summary.csv"])
    )
    phase_j0_dir = (
        Path(args.phase_j0_dir).resolve()
        if args.phase_j0_dir
        else find_latest_complete(exec_root, "PHASEJ0_POST_PHASEI_RECOVERY_*", ["phaseJ05_tradeoff_frontier.csv"])
    )
    repaired_signal_root: Optional[Path] = None
    try:
        repaired_signal_root = rebaseline_1h._latest_multicoin_signal_root().resolve()  # pylint: disable=protected-access
    except Exception:
        repaired_signal_root = None

    if phase_i_dir is None:
        raise FileNotFoundError("Missing latest complete PHASEI_EXECAWARE_1H_GA_EXPANSION_* directory")
    if params_scan_dir is None:
        raise FileNotFoundError("Missing latest complete reports/params_scan/* directory with best_by_symbol.csv")
    if repaired_baseline_dir is None:
        raise FileNotFoundError("Missing latest complete 1H_CONTRACT_REPAIR_REBASELINE_* directory")

    return CandidateArtifacts(
        phase_i_dir=phase_i_dir,
        params_scan_dir=params_scan_dir,
        repaired_baseline_dir=repaired_baseline_dir,
        phase_j0_dir=phase_j0_dir,
        repaired_signal_root=repaired_signal_root,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Repaired-contract contamination audit for the existing 1h discovery stack")
    p.add_argument("--outdir", default="reports/execution_layer")
    p.add_argument("--phase-i-dir", default="")
    p.add_argument("--phase-j0-dir", default="")
    p.add_argument("--params-scan-dir", default="")
    p.add_argument("--repaired-baseline-dir", default="")
    p.add_argument("--topk-list", default="5,10,20")
    p.add_argument("--min-frontier-coverage", type=float, default=0.95)
    p.add_argument("--require-full-universe-coverage", type=int, default=1)
    return p


def parse_topk_list(text: str) -> List[int]:
    vals: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    vals = sorted(set(v for v in vals if v > 0))
    return vals or [5, 10, 20]


def canonical_phase_i_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["valid_for_ranking"] = pd.to_numeric(out.get("valid_for_ranking"), errors="coerce").fillna(0).astype(int)
    out["duplicate_of_candidate_id"] = out.get("duplicate_of_candidate_id", "").fillna("").astype(str)
    out["OJ2"] = pd.to_numeric(out.get("OJ2"), errors="coerce")
    out["delta_expectancy_vs_exec_baseline"] = pd.to_numeric(out.get("delta_expectancy_vs_exec_baseline"), errors="coerce")
    out = out[(out["valid_for_ranking"] == 1) & (out["duplicate_of_candidate_id"].str.strip() == "")]
    out = out.sort_values(["OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False]).reset_index(drop=True)
    out["legacy_old_rank"] = np.arange(1, len(out) + 1)
    return out


def load_thresholds(scan_meta: Optional[Dict[str, Any]] = None) -> scan.Thresholds:
    src = dict((scan_meta or {}).get("thresholds", {}) or {})
    return scan.Thresholds(
        min_net_profit=float(src.get("min_net_profit", 0.0)),
        min_profit_factor=float(src.get("min_profit_factor", 1.15)),
        min_cagr_pct=float(src.get("min_cagr_pct", 15.0)),
        max_dd_pct=float(src.get("max_dd_pct", 35.0)),
        min_trades=float(src.get("min_trades", 50.0)),
        min_trades_per_year=float(src.get("min_trades_per_year", 10.0)),
    )


def prep_full_df(
    symbol: str,
    params_dict: Dict[str, Any],
    cache: Dict[Tuple[str, str], pd.DataFrame],
    raw_cache: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    fingerprint = phase_u.param_fingerprint(params_dict)
    key = (symbol, fingerprint)
    if key in cache:
        return cache[key]
    if symbol not in raw_cache:
        raw_cache[symbol] = scan.load_symbol_df(symbol=symbol, tf="1h")
    df_base = raw_cache[symbol]
    df_feat = ga_long._ensure_indicators(df_base.copy(), params_dict)  # pylint: disable=protected-access
    df_feat["Timestamp"] = pd.to_datetime(df_feat["Timestamp"], utc=True, errors="coerce")
    df_feat = df_feat.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    cache[key] = df_feat
    return df_feat


def build_signal_table(
    *,
    symbol: str,
    params_dict: Dict[str, Any],
    df_feat: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    sig = np.asarray(ga_long.build_entry_signal(df_feat, params_dict, assume_prepared=True), dtype=bool)
    cycles_raw = ga_long.compute_cycles(df_feat, params_dict)
    cycles = ga_long._shift_cycles(  # pylint: disable=protected-access
        cycles_raw,
        shift=int(params_dict.get("cycle_shift", 1)),
        fill=int(params_dict.get("cycle_fill", 2)),
    )
    ts = pd.to_datetime(df_feat["Timestamp"], utc=True, errors="coerce")
    mask = sig & (ts >= period_start) & (ts <= period_end)
    rows: List[Dict[str, Any]] = []
    for idx in np.flatnonzero(mask):
        cyc = int(cycles[idx])
        rows.append(
            {
                "signal_id": f"{symbol}_{pd.to_datetime(ts.iloc[idx], utc=True).isoformat()}",
                "signal_time": pd.to_datetime(ts.iloc[idx], utc=True),
                "tp_mult": float(params_dict["tp_mult_by_cycle"][cyc]),
                "sl_mult": float(params_dict["sl_mult_by_cycle"][cyc]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["signal_id", "signal_time", "tp_mult", "sl_mult"])
    return pd.DataFrame(rows).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)


def load_market_arrays(symbol: str, market_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if symbol in market_cache:
        return market_cache[symbol]
    full_fp = PROJECT_ROOT / "data" / "processed" / "_full" / f"{symbol}_1h_full.parquet"
    if not full_fp.exists():
        raise FileNotFoundError(f"Missing 1h parquet for repaired reference variant: {full_fp}")
    k = pd.read_parquet(full_fp)
    k = phasec_bt.exec3m._normalize_ohlcv_cols(k)  # pylint: disable=protected-access
    k["Timestamp"] = pd.to_datetime(k["Timestamp"], utc=True, errors="coerce")
    k = k.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"]).sort_values("Timestamp").reset_index(drop=True)
    payload = {
        "ts_ns": np.array([int(t.value) for t in pd.to_datetime(k["Timestamp"], utc=True)], dtype=np.int64),
        "open": pd.to_numeric(k["Open"], errors="coerce").to_numpy(dtype=float),
        "high": pd.to_numeric(k["High"], errors="coerce").to_numpy(dtype=float),
        "low": pd.to_numeric(k["Low"], errors="coerce").to_numpy(dtype=float),
        "close": pd.to_numeric(k["Close"], errors="coerce").to_numpy(dtype=float),
    }
    market_cache[symbol] = payload
    return payload


def simulate_repaired_returns(
    *,
    signals_df: pd.DataFrame,
    symbol: str,
    fee_model: phasec_bt.FeeModel,
    exec_horizon_hours: float,
    market_cache: Dict[str, Dict[str, Any]],
) -> pd.Series:
    if signals_df.empty:
        return pd.Series(dtype=float)
    market = load_market_arrays(symbol, market_cache)
    ts_ns = market["ts_ns"]
    open_np = market["open"]
    high_np = market["high"]
    low_np = market["low"]
    close_np = market["close"]
    out: List[float] = []
    for r in signals_df.itertuples(index=False):
        st = pd.to_datetime(getattr(r, "signal_time"), utc=True)
        tp_mult = float(getattr(r, "tp_mult"))
        sl_mult = float(getattr(r, "sl_mult"))
        idx = int(np.searchsorted(ts_ns, int(st.value), side="left"))
        if idx >= len(ts_ns):
            continue
        entry_price = float(open_np[idx])
        eval_start_idx = int(idx + 1)
        if eval_start_idx >= len(ts_ns):
            continue
        max_exit_ts_ns = phasec_bt.exec3m._compute_eval_end_ns(  # pylint: disable=protected-access
            entry_ts_ns=int(ts_ns[idx]),
            eval_horizon_hours=float(exec_horizon_hours),
            baseline_exit_time=None,
        )
        sim = phasec_bt.exec3m._simulate_path_long(  # pylint: disable=protected-access
            ts_ns=ts_ns,
            close=close_np,
            high=high_np,
            low=low_np,
            entry_idx=int(eval_start_idx),
            entry_price=float(entry_price),
            sl_price=float(entry_price * sl_mult),
            tp_price=float(entry_price * tp_mult),
            max_exit_ts_ns=int(max_exit_ts_ns),
        )
        if int(sim.get("filled", 0)) != 1 or int(sim.get("valid_for_metrics", 0)) != 1:
            continue
        exit_px = safe_float(sim.get("exit_price"))
        if not math.isfinite(exit_px):
            continue
        costs = phasec_bt._cost_row(float(entry_price), float(exit_px), "taker", fee_model)  # pylint: disable=protected-access
        pnl = safe_float(costs.get("pnl_net_pct"))
        if math.isfinite(pnl):
            out.append(pnl)
    return pd.Series(out, dtype=float)


def summarize_repaired_returns(returns: pd.Series, initial_equity: float) -> Dict[str, Any]:
    returns = pd.to_numeric(returns, errors="coerce")
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        equity_curve = np.array([float(initial_equity)], dtype=float)
        final_equity = float(initial_equity)
    else:
        equity_curve = np.empty(len(returns) + 1, dtype=float)
        equity_curve[0] = float(initial_equity)
        for i, ret in enumerate(returns.to_numpy(dtype=float), start=1):
            equity_curve[i] = float(equity_curve[i - 1] * (1.0 + float(ret)))
        final_equity = float(equity_curve[-1])
    peaks = np.maximum.accumulate(equity_curve)
    dd = equity_curve / np.maximum(peaks, 1e-12) - 1.0
    cvar_5 = tail_mean(returns, 0.05)
    expectancy = float(returns.mean()) if len(returns) else float("nan")
    profit_factor = profit_factor_from_returns(returns)
    wins = int((returns > 0.0).sum())
    losses = int((returns <= 0.0).sum())
    max_dd_pct = abs(float(dd.min())) * 100.0 if dd.size else 0.0
    net_profit = float(final_equity - initial_equity)
    metrics = {
        "initial_equity": float(initial_equity),
        "final_equity": final_equity,
        "net_profit": net_profit,
        "trades": float(len(returns)),
        "wins": float(wins),
        "losses": float(losses),
        "win_rate_pct": (float(wins) / float(len(returns)) * 100.0) if len(returns) else 0.0,
        "gross_profit": float(returns[returns > 0.0].sum()) if len(returns) else 0.0,
        "gross_loss": float((-returns[returns < 0.0].sum())) if len(returns) else 0.0,
        "max_dd": abs(float(dd.min())) if dd.size else 0.0,
        "profit_factor": profit_factor if math.isfinite(profit_factor) else 0.0,
        "avg_win": float(returns[returns > 0.0].mean()) if (returns > 0.0).any() else 0.0,
        "avg_loss": float(returns[returns <= 0.0].mean()) if (returns <= 0.0).any() else 0.0,
    }
    return {
        "repaired_trade_count": int(len(returns)),
        "repaired_expectancy_net": expectancy,
        "repaired_cvar_5": cvar_5,
        "repaired_max_dd_pct": max_dd_pct,
        "repaired_win_rate": (float(wins) / float(len(returns))) if len(returns) else 0.0,
        "repaired_profit_factor": profit_factor,
        "repaired_final_equity": final_equity,
        "repaired_cagr_pct": float("nan"),
        "repaired_metrics_compat": metrics,
    }


def evaluate_repaired_1h_candidate(
    *,
    symbol: str,
    params_dict: Dict[str, Any],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    initial_equity: float,
    fee_model: phasec_bt.FeeModel,
    df_cache: Dict[Tuple[str, str], pd.DataFrame],
    raw_cache: Dict[str, pd.DataFrame],
    market_cache: Dict[str, Dict[str, Any]],
    thresholds: scan.Thresholds,
) -> Dict[str, Any]:
    df_feat = prep_full_df(symbol, params_dict, df_cache, raw_cache)
    signals_df = build_signal_table(symbol=symbol, params_dict=params_dict, df_feat=df_feat, period_start=period_start, period_end=period_end)
    returns = simulate_repaired_returns(
        signals_df=signals_df,
        symbol=symbol,
        fee_model=fee_model,
        exec_horizon_hours=float(max(1.0, safe_float(params_dict.get("max_hold_hours")))),
        market_cache=market_cache,
    )
    summary = summarize_repaired_returns(returns, initial_equity)
    compat = summary["repaired_metrics_compat"]
    score_pack = scan.compute_pass_score(
        metrics=compat,
        period_start=period_start,
        period_end=period_end,
        thresholds=thresholds,
    )
    summary["repaired_1h_score"] = float(score_pack["score"])
    summary["repaired_pass"] = bool(score_pack["pass"])
    summary["repaired_pass_detail"] = score_pack
    summary["repaired_cagr_pct"] = safe_float(score_pack["cagr_pct"])
    summary["repaired_final_equity"] = safe_float(score_pack["final_equity"])
    return summary


def evaluate_legacy_1h_candidate_or_equivalent(
    *,
    symbol: str,
    params_dict: Dict[str, Any],
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    initial_equity: float,
    df_cache: Dict[Tuple[str, str], pd.DataFrame],
    raw_cache: Dict[str, pd.DataFrame],
    thresholds: scan.Thresholds,
    fee_bps: float,
    slippage_bps: float,
) -> Dict[str, Any]:
    df_feat = prep_full_df(symbol, params_dict, df_cache, raw_cache)
    ts = pd.to_datetime(df_feat["Timestamp"], utc=True, errors="coerce")
    mask = (ts >= period_start) & (ts <= period_end)
    idx = np.flatnonzero(mask.to_numpy(dtype=bool))
    if idx.size == 0:
        metrics = {
            "initial_equity": float(initial_equity),
            "final_equity": float(initial_equity),
            "net_profit": 0.0,
            "trades": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "win_rate_pct": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "max_dd": 0.0,
            "profit_factor": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }
        trades = []
    else:
        start_idx = int(idx[0])
        end_idx = int(idx[-1] + 1)
        trades, metrics = ga_long.run_backtest_long_only(
            df=df_feat,
            symbol=symbol,
            p=params_dict,
            initial_equity=float(initial_equity),
            fee_bps=float(fee_bps),
            slippage_bps=float(slippage_bps),
            collect_trades=True,
            start_idx=start_idx,
            end_idx=end_idx,
            assume_prepared=True,
        )
    score_pack = scan.compute_pass_score(metrics, period_start, period_end, thresholds)
    return {"trades": trades, "metrics": metrics, "score_pack": score_pack}


def compare_float(a: Any, b: Any, tol: float) -> bool:
    af = safe_float(a)
    bf = safe_float(b)
    if not (math.isfinite(af) and math.isfinite(bf)):
        return False
    return abs(af - bf) <= tol


def phase_i_parity_check(
    phase_i_df: pd.DataFrame,
    manifest: Dict[str, Any],
    df_cache: Dict[Tuple[str, str], pd.DataFrame],
    raw_cache: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    parity_rows: List[Dict[str, Any]] = []
    phase_h_dir = Path(str(manifest.get("phase_h_source_dir", ""))).resolve()
    rep_subset_path = PHASE_H_REP_SUBSET if PHASE_H_REP_SUBSET.exists() else None
    if rep_subset_path is None:
        return pd.DataFrame(), {"ok": False, "reason": f"missing representative subset file: {PHASE_H_REP_SUBSET}"}
    rep_subset = pd.read_csv(rep_subset_path)
    active_meta = manifest["i1_meta"]["h1_manifest_reused"]["active_params_meta"]["row"]
    symbol = str(active_meta["symbol"]).upper()

    sample_n = min(25, len(phase_i_df))
    sample_df = phase_i_df.head(sample_n).copy()
    mismatch_count = 0
    exact_sig_hash_count = 0
    for _, row in sample_df.iterrows():
        params = ga_long._norm_params(json.loads(str(row["params_json"])))  # pylint: disable=protected-access
        df_feat = prep_full_df(symbol, params, df_cache, raw_cache)
        rep_idx = phase_u.build_rep_subset_with_idx(rep_subset, df_feat)
        active_ids, diag = phase_u.active_signal_ids_for_params(df_feat=df_feat, params=params, rep_idx=rep_idx)
        reproduced_hash = phase_u.sha256_text("|".join(sorted(active_ids)))
        stored_signals = int(pd.to_numeric(pd.Series([row.get("signals_active")]), errors="coerce").fillna(-1).iloc[0])
        repro_signals = int(diag.get("active_signals", -1))
        stored_rate = safe_float(row.get("active_rate_vs_rep"))
        repro_rate = safe_float(diag.get("active_rate_vs_rep"))
        hash_match = int(str(row.get("signal_signature", "")) == str(reproduced_hash))
        signals_match = int(stored_signals == repro_signals)
        rate_match = int(compare_float(stored_rate, repro_rate, 1e-12))
        if hash_match == 0 or signals_match == 0 or rate_match == 0:
            mismatch_count += 1
        if hash_match == 1:
            exact_sig_hash_count += 1
        parity_rows.append(
            {
                "scope": "phase_i_sample",
                "candidate_id": str(row["candidate_id"]),
                "stored_signals_active": stored_signals,
                "reproduced_signals_active": repro_signals,
                "signals_match": signals_match,
                "stored_active_rate_vs_rep": stored_rate,
                "reproduced_active_rate_vs_rep": repro_rate,
                "rate_match": rate_match,
                "stored_signal_signature": str(row.get("signal_signature", "")),
                "reproduced_signal_signature": str(reproduced_hash),
                "signal_signature_match": hash_match,
                "source_subset_csv": str(rep_subset_path),
                "source_phase_h_dir": str(phase_h_dir),
            }
        )
    parity_df = pd.DataFrame(parity_rows)
    ok = bool(mismatch_count == 0)
    return parity_df, {
        "ok": ok,
        "sample_size": sample_n,
        "mismatch_count": mismatch_count,
        "exact_signal_signature_matches": exact_sig_hash_count,
        "reason": "" if ok else f"{mismatch_count} / {sample_n} sampled Phase I candidates failed signal reconstruction parity",
        "source_subset_csv": str(rep_subset_path),
    }


def universe_parity_check(
    universe_df: pd.DataFrame,
    df_cache: Dict[Tuple[str, str], pd.DataFrame],
    raw_cache: Dict[str, pd.DataFrame],
    thresholds: scan.Thresholds,
    fee_bps: float,
    slippage_bps: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mismatch = 0
    for _, row in universe_df.iterrows():
        params_path = (PROJECT_ROOT / str(row["params_file"])).resolve()
        payload = scan.load_json(params_path)
        params = ga_long._norm_params(scan.unwrap_params(payload))  # pylint: disable=protected-access
        period_start = pd.to_datetime(row["period_start"], utc=True)
        period_end = pd.to_datetime(row["period_end"], utc=True)
        initial_equity = safe_float(row["initial_equity"])
        legacy = evaluate_legacy_1h_candidate_or_equivalent(
            symbol=str(row["symbol"]).upper(),
            params_dict=params,
            period_start=period_start,
            period_end=period_end,
            initial_equity=initial_equity,
            df_cache=df_cache,
            raw_cache=raw_cache,
            thresholds=thresholds,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
        )
        metrics = legacy["score_pack"]
        stored_score = safe_float(row["score"])
        stored_cagr = safe_float(row["cagr_pct"])
        stored_pf = safe_float(row["profit_factor"])
        stored_dd = safe_float(row["max_dd_pct"])
        stored_trades = safe_float(row["trades"])
        score_match = int(compare_float(stored_score, metrics["score"], 1e-6))
        cagr_match = int(compare_float(stored_cagr, metrics["cagr_pct"], 1e-6))
        pf_match = int(compare_float(stored_pf, metrics["profit_factor"], 1e-6))
        dd_match = int(compare_float(stored_dd, metrics["max_dd_pct"], 1e-6))
        trades_match = int(compare_float(stored_trades, metrics["trades"], 1e-6))
        if min(score_match, cagr_match, pf_match, dd_match, trades_match) == 0:
            mismatch += 1
        rows.append(
            {
                "scope": "universe_row",
                "symbol": str(row["symbol"]).upper(),
                "params_file": str(row["params_file"]),
                "stored_score": stored_score,
                "reproduced_score": safe_float(metrics["score"]),
                "score_match": score_match,
                "stored_cagr_pct": stored_cagr,
                "reproduced_cagr_pct": safe_float(metrics["cagr_pct"]),
                "cagr_match": cagr_match,
                "stored_profit_factor": stored_pf,
                "reproduced_profit_factor": safe_float(metrics["profit_factor"]),
                "profit_factor_match": pf_match,
                "stored_max_dd_pct": stored_dd,
                "reproduced_max_dd_pct": safe_float(metrics["max_dd_pct"]),
                "max_dd_match": dd_match,
                "stored_trades": stored_trades,
                "reproduced_trades": safe_float(metrics["trades"]),
                "trades_match": trades_match,
                "stored_pass": int(to_bool(row["pass"])),
                "reproduced_pass": int(bool(metrics["pass"])),
            }
        )
    parity_df = pd.DataFrame(rows)
    ok = bool(mismatch == 0)
    return parity_df, {
        "ok": ok,
        "sample_size": int(len(universe_df)),
        "mismatch_count": int(mismatch),
        "reason": "" if ok else f"{mismatch} / {len(universe_df)} universe rows failed exact legacy parity",
    }


def repaired_smoke_check(
    universe_results: pd.DataFrame,
    repaired_baseline_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    targets = ["SOLUSDT", "NEARUSDT", "AVAXUSDT"]
    mismatch = 0
    for sym in targets:
        a = universe_results[universe_results["symbol"] == sym]
        b = repaired_baseline_df[repaired_baseline_df["symbol"] == sym]
        if a.empty or b.empty:
            mismatch += 1
            rows.append(
                {
                    "symbol": sym,
                    "present_in_universe_results": int(not a.empty),
                    "present_in_repaired_baseline": int(not b.empty),
                    "trade_count_match": 0,
                    "expectancy_match": 0,
                    "cvar_match": 0,
                    "maxdd_match": 0,
                }
            )
            continue
        ar = a.iloc[0]
        br = b.iloc[0]
        trade_match = int(int(ar["repaired_trades"]) == int(br["after_trade_count"]))
        exp_tol = 3e-4
        cvar_tol = 2e-4
        baseline_dd_pct = abs(safe_float(br["after_maxdd"])) * 100.0
        dd_tol = max(5.0, 0.40 * baseline_dd_pct)
        exp_match = int(compare_float(ar["repaired_expectancy_net"], br["after_expectancy_net"], exp_tol))
        cvar_match = int(compare_float(ar["repaired_cvar_pct_5"], br["after_cvar_5"], cvar_tol))
        dd_match = int(compare_float(ar["repaired_max_dd_pct"], baseline_dd_pct, dd_tol))
        if min(trade_match, exp_match, cvar_match, dd_match) == 0:
            mismatch += 1
        rows.append(
            {
                "symbol": sym,
                "present_in_universe_results": 1,
                "present_in_repaired_baseline": 1,
                "trade_count_match": trade_match,
                "expectancy_match": exp_match,
                "cvar_match": cvar_match,
                "maxdd_match": dd_match,
                "computed_trade_count": int(ar["repaired_trades"]),
                "baseline_trade_count": int(br["after_trade_count"]),
                "computed_expectancy": float(ar["repaired_expectancy_net"]),
                "baseline_expectancy": float(br["after_expectancy_net"]),
                "expectancy_tolerance": exp_tol,
                "computed_cvar_5": float(ar["repaired_cvar_pct_5"]),
                "baseline_cvar_5": float(br["after_cvar_5"]),
                "cvar_tolerance": cvar_tol,
                "computed_max_dd_pct": float(ar["repaired_max_dd_pct"]),
                "baseline_max_dd_pct": baseline_dd_pct,
                "max_dd_tolerance_pct": dd_tol,
            }
        )
    df = pd.DataFrame(rows)
    ok = bool(mismatch == 0)
    return df, {
        "ok": ok,
        "sample_size": len(targets),
        "mismatch_count": mismatch,
        "reason": "" if ok else f"{mismatch} / {len(targets)} repaired smoke checks diverged from repaired baseline summary",
    }


def compute_frontier_instability(frontier_df: pd.DataFrame, topk_list: List[int]) -> Dict[str, Any]:
    ranked = frontier_df.sort_values(
        ["repaired_pass_proxy_for_frontier", "repaired_1h_score", "repaired_expectancy_net"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    ranked["repaired_rank"] = np.arange(1, len(ranked) + 1)
    rank_map = dict(zip(ranked["candidate_id"].astype(str), ranked["repaired_rank"].astype(int)))
    out_df = frontier_df.copy()
    out_df["repaired_rank"] = out_df["candidate_id"].astype(str).map(rank_map).astype(int)
    out_df["rank_delta"] = out_df["repaired_rank"] - out_df["legacy_old_rank"].astype(int)
    old_ranks = out_df["legacy_old_rank"].to_numpy(dtype=float)
    new_ranks = out_df["repaired_rank"].to_numpy(dtype=float)
    topk_stats: Dict[str, Any] = {}
    old_ids = out_df.sort_values("legacy_old_rank")["candidate_id"].astype(str).tolist()
    new_ids = out_df.sort_values("repaired_rank")["candidate_id"].astype(str).tolist()
    for k in topk_list:
        old_top = set(old_ids[:k])
        new_top = set(new_ids[:k])
        shared = len(old_top & new_top)
        turnover = 1.0 - (shared / float(max(1, k)))
        topk_stats[f"top{k}_shared"] = shared
        topk_stats[f"top{k}_turnover"] = turnover
        topk_stats[f"top{k}_legacy_retained_pct"] = shared / float(max(1, k))
    top20 = out_df.sort_values("legacy_old_rank").head(20).copy()
    pass_flips = int(
        (
            top20["legacy_valid_for_ranking"].astype(int)
            != top20["repaired_pass_proxy_for_frontier"].astype(int)
        ).sum()
    )
    return {
        "df": out_df,
        "spearman": spearman_rank_corr(old_ranks, new_ranks),
        "topk": topk_stats,
        "top20_proxy_flip_rate": pass_flips / float(max(1, len(top20))),
        "top20_proxy_flip_count": pass_flips,
    }


def compute_universe_diff(universe_df: pd.DataFrame, topk_list: List[int]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = universe_df.copy()
    out["legacy_pass"] = out["legacy_pass"].astype(int)
    out["repaired_pass"] = out["repaired_pass"].astype(int)
    out = out.sort_values(["repaired_pass", "repaired_score", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
    out["repaired_rank"] = np.arange(1, len(out) + 1)
    out["rank_delta"] = out["repaired_rank"] - out["legacy_rank"]
    out["score_delta"] = out["repaired_score"] - out["legacy_score"]
    actions: List[str] = []
    for _, row in out.iterrows():
        lp = int(row["legacy_pass"])
        rp = int(row["repaired_pass"])
        if lp == 1 and rp == 1:
            actions.append("STAY_PASS")
        elif lp == 1 and rp == 0:
            actions.append("DROP_FROM_PASS")
        elif lp == 0 and rp == 1:
            actions.append("NEW_PASS")
        else:
            actions.append("STAY_FAIL")
    out["membership_action"] = actions

    legacy_pass = set(out.loc[out["legacy_pass"] == 1, "symbol"].astype(str))
    repaired_pass = set(out.loc[out["repaired_pass"] == 1, "symbol"].astype(str))
    inter = legacy_pass & repaired_pass
    union = legacy_pass | repaired_pass
    jaccard = len(inter) / float(max(1, len(union)))
    legacy_flips = int((out["membership_action"] == "DROP_FROM_PASS").sum())
    legacy_pass_ct = int(max(1, (out["legacy_pass"] == 1).sum()))
    shared_df = out[(out["legacy_pass"] == 1) & (out["repaired_pass"] == 1)].copy()
    shared_spearman = spearman_rank_corr(
        shared_df["legacy_rank"].to_numpy(dtype=float),
        shared_df["repaired_rank"].to_numpy(dtype=float),
    ) if len(shared_df) >= 2 else float("nan")
    topk_stats: Dict[str, Any] = {}
    old_pass_df = out[out["legacy_pass"] == 1].sort_values(["legacy_rank", "symbol"])
    new_pass_df = out[out["repaired_pass"] == 1].sort_values(["repaired_rank", "symbol"])
    old_ids = old_pass_df["symbol"].astype(str).tolist()
    new_ids = new_pass_df["symbol"].astype(str).tolist()
    for k in [x for x in topk_list if x <= max(len(old_ids), len(new_ids), 1)]:
        old_top = set(old_ids[:k])
        new_top = set(new_ids[:k])
        shared = len(old_top & new_top)
        turnover = 1.0 - (shared / float(max(1, k)))
        topk_stats[f"top{k}_shared"] = shared
        topk_stats[f"top{k}_turnover"] = turnover
    stats = {
        "passed_set_jaccard": jaccard,
        "stay_pass_count": int((out["membership_action"] == "STAY_PASS").sum()),
        "drop_from_pass_count": int((out["membership_action"] == "DROP_FROM_PASS").sum()),
        "new_pass_count": int((out["membership_action"] == "NEW_PASS").sum()),
        "stay_fail_count": int((out["membership_action"] == "STAY_FAIL").sum()),
        "drop_from_pass_rate_vs_legacy": legacy_flips / float(legacy_pass_ct),
        "shared_pass_rank_spearman": shared_spearman,
        "topk": topk_stats,
        "legacy_pass_set": sorted(legacy_pass),
        "repaired_pass_set": sorted(repaired_pass),
    }
    return out, stats


def classify_contamination(universe_stats: Dict[str, Any], frontier_stats: Dict[str, Any]) -> str:
    jaccard = safe_float(universe_stats["passed_set_jaccard"])
    flip_rate = safe_float(universe_stats["drop_from_pass_rate_vs_legacy"])
    top10_turnover = safe_float(frontier_stats["topk"].get("top10_turnover"))
    phase_spearman = safe_float(frontier_stats["spearman"])
    if (
        (math.isfinite(jaccard) and jaccard < 0.60)
        or (math.isfinite(flip_rate) and flip_rate > 0.50)
        or (math.isfinite(top10_turnover) and top10_turnover > 0.60)
        or (math.isfinite(phase_spearman) and phase_spearman < 0.50)
    ):
        return "HIGH_CONTAMINATION"
    if (
        (math.isfinite(jaccard) and 0.60 <= jaccard < 0.80)
        or (math.isfinite(flip_rate) and 0.20 < flip_rate <= 0.50)
        or (math.isfinite(top10_turnover) and 0.30 < top10_turnover <= 0.60)
        or (math.isfinite(phase_spearman) and 0.50 <= phase_spearman < 0.80)
    ):
        return "MODERATE_CONTAMINATION"
    if (
        math.isfinite(jaccard)
        and jaccard >= 0.80
        and math.isfinite(flip_rate)
        and flip_rate <= 0.20
        and math.isfinite(top10_turnover)
        and top10_turnover <= 0.30
        and math.isfinite(phase_spearman)
        and phase_spearman >= 0.80
    ):
        return "LOW_CONTAMINATION"
    return "MODERATE_CONTAMINATION"


def map_recommendation(
    *,
    contamination: str,
    universe_stats: Dict[str, Any],
    frontier_stats: Dict[str, Any],
    parity_ok: bool,
) -> str:
    if not parity_ok:
        return "BLOCKED"
    jaccard = safe_float(universe_stats["passed_set_jaccard"])
    drops = int(universe_stats["drop_from_pass_count"])
    top10_turnover = safe_float(frontier_stats["topk"].get("top10_turnover"))
    phase_spearman = safe_float(frontier_stats["spearman"])
    if (
        contamination == "LOW_CONTAMINATION"
        and math.isfinite(jaccard)
        and jaccard >= 0.80
        and drops <= 2
    ):
        return "KEEP_REPAIRED_FRONTIER_AND_CONTINUE"
    if (
        contamination == "MODERATE_CONTAMINATION"
        and math.isfinite(top10_turnover)
        and top10_turnover <= 0.60
        and math.isfinite(phase_spearman)
        and phase_spearman >= 0.50
    ):
        return "RERANK_AND_REBUILD_UNIVERSE_ONLY"
    if (
        contamination == "HIGH_CONTAMINATION"
        or (
            math.isfinite(top10_turnover)
            and top10_turnover > 0.60
            and math.isfinite(jaccard)
            and jaccard < 0.70
        )
    ):
        return "FULL_1H_GA_RERUN_RECOMMENDED"
    return "RERANK_AND_REBUILD_UNIVERSE_ONLY"


def write_report(
    *,
    out_path: Path,
    artifacts: CandidateArtifacts,
    manifest: Dict[str, Any],
    phase_i_parity_summary: Dict[str, Any],
    universe_parity_summary: Dict[str, Any],
    repaired_smoke_summary: Dict[str, Any],
    frontier_coverage: Dict[str, Any],
    universe_coverage: Dict[str, Any],
    frontier_stats: Dict[str, Any],
    universe_stats: Dict[str, Any],
    contamination: str,
    recommendation: str,
    phase_i_parity_df: pd.DataFrame,
    universe_parity_df: pd.DataFrame,
    repaired_smoke_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Repaired Frontier Contamination Report")
    lines.append("")
    lines.append(f"- Generated UTC: `{utc_now().isoformat()}`")
    lines.append("- Audit scope: repaired 1h-only upstream contamination audit; intentionally excludes the 3m execution layer.")
    lines.append("")
    lines.append("## Discovered Artifacts")
    lines.append(f"- Phase I frontier: `{artifacts.phase_i_dir}`")
    lines.append(f"- Params scan universe: `{artifacts.params_scan_dir}`")
    lines.append(f"- Repaired baseline: `{artifacts.repaired_baseline_dir}`")
    lines.append(f"- Phase J0 appendix: `{artifacts.phase_j0_dir}`" if artifacts.phase_j0_dir else "- Phase J0 appendix: not used")
    lines.append(f"- Repaired signal root / fee model source: `{artifacts.repaired_signal_root}`" if artifacts.repaired_signal_root else "- Repaired signal root / fee model source: not found (fallback fixed 1h taker costs not used)")
    lines.append("")
    lines.append("## Discovery Validation")
    lines.append(f"- Phase I discovery target matched expected: `{int(artifacts.phase_i_dir.name == DISCOVERY_EXPECTED['phase_i'])}`")
    lines.append(f"- Params scan discovery target matched expected: `{int(artifacts.params_scan_dir.name == DISCOVERY_EXPECTED['params_scan'])}`")
    lines.append(f"- Repaired baseline discovery target matched expected: `{int(artifacts.repaired_baseline_dir.name == DISCOVERY_EXPECTED['repaired_baseline'])}`")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Phase I coverage: `{frontier_coverage['ok_rows']}/{frontier_coverage['total_rows']}` = `{frontier_coverage['coverage_pct']:.4%}`")
    lines.append(f"- Universe coverage: `{universe_coverage['ok_rows']}/{universe_coverage['total_rows']}` = `{universe_coverage['coverage_pct']:.4%}`")
    if frontier_coverage["failed_rows"]:
        lines.append(f"- Phase I failed rows: `{', '.join(frontier_coverage['failed_rows'][:20])}`")
    if universe_coverage["failed_rows"]:
        lines.append(f"- Universe failed rows: `{', '.join(universe_coverage['failed_rows'][:20])}`")
    lines.append("")
    lines.append("## Legacy Reconstruction Parity")
    lines.append(f"- Phase I signal reconstruction parity: `ok={int(phase_i_parity_summary['ok'])}`, sample=`{phase_i_parity_summary.get('sample_size', 0)}`, mismatches=`{phase_i_parity_summary.get('mismatch_count', 0)}`")
    if phase_i_parity_summary.get("reason"):
        lines.append(f"- Phase I parity note: `{phase_i_parity_summary['reason']}`")
    lines.append(f"- Universe legacy evaluator parity: `ok={int(universe_parity_summary['ok'])}`, sample=`{universe_parity_summary.get('sample_size', 0)}`, mismatches=`{universe_parity_summary.get('mismatch_count', 0)}`")
    if universe_parity_summary.get("reason"):
        lines.append(f"- Universe parity note: `{universe_parity_summary['reason']}`")
    lines.append(f"- Repaired evaluator smoke check (SOL/NEAR/AVAX): `ok={int(repaired_smoke_summary['ok'])}`, mismatches=`{repaired_smoke_summary.get('mismatch_count', 0)}`")
    if repaired_smoke_summary.get("reason"):
        lines.append(f"- Repaired smoke note: `{repaired_smoke_summary['reason']}`")
    lines.append("")
    if not phase_i_parity_df.empty:
        lines.append("### Phase I Parity Sample")
        lines.append("")
        lines.append(markdown_table(phase_i_parity_df, n=8))
        lines.append("")
    if not universe_parity_df.empty:
        lines.append("### Universe Parity Sample")
        lines.append("")
        lines.append(markdown_table(universe_parity_df, n=8))
        lines.append("")
    if not repaired_smoke_df.empty:
        lines.append("### Repaired Smoke Sample")
        lines.append("")
        lines.append(markdown_table(repaired_smoke_df, n=len(repaired_smoke_df)))
        lines.append("")
    lines.append("## Phase I Frontier Comparison")
    lines.append("- Old score = legacy GA objective `OJ2`.")
    lines.append("- New score = repaired 1h symbol-equity score `(cagr_pct * profit_factor) / (1 + max_dd_pct)`.")
    lines.append(f"- Spearman(old rank, repaired rank): `{frontier_stats['spearman']:.6f}`" if math.isfinite(safe_float(frontier_stats["spearman"])) else "- Spearman(old rank, repaired rank): `nan`")
    for k, v in sorted(frontier_stats["topk"].items()):
        lines.append(f"- {k}: `{v}`")
    lines.append(f"- Top-20 repaired pass-proxy flips (secondary annotation): `{frontier_stats['top20_proxy_flip_count']}` / `20` = `{frontier_stats['top20_proxy_flip_rate']:.4%}`")
    lines.append("")
    lines.append("## Universe Membership Comparison")
    lines.append(f"- Passed-set Jaccard: `{universe_stats['passed_set_jaccard']:.6f}`")
    lines.append(f"- STAY_PASS: `{universe_stats['stay_pass_count']}`")
    lines.append(f"- DROP_FROM_PASS: `{universe_stats['drop_from_pass_count']}`")
    lines.append(f"- NEW_PASS: `{universe_stats['new_pass_count']}`")
    lines.append(f"- STAY_FAIL: `{universe_stats['stay_fail_count']}`")
    lines.append(f"- Drop rate vs legacy pass set: `{universe_stats['drop_from_pass_rate_vs_legacy']:.4%}`")
    lines.append(f"- Shared-pass rank Spearman: `{universe_stats['shared_pass_rank_spearman']:.6f}`" if math.isfinite(safe_float(universe_stats["shared_pass_rank_spearman"])) else "- Shared-pass rank Spearman: `nan`")
    for k, v in sorted(universe_stats["topk"].items()):
        lines.append(f"- {k}: `{v}`")
    lines.append(f"- Legacy pass set: `{', '.join(universe_stats['legacy_pass_set'])}`")
    lines.append(f"- Repaired pass set: `{', '.join(universe_stats['repaired_pass_set'])}`")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- Contamination severity: `{contamination}`")
    lines.append(f"- Final recommendation: `{recommendation}`")
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_phase_i_rows(
    *,
    phase_i_df: pd.DataFrame,
    manifest: Dict[str, Any],
    fee_model: phasec_bt.FeeModel,
    df_cache: Dict[Tuple[str, str], pd.DataFrame],
    raw_cache: Dict[str, pd.DataFrame],
    market_cache: Dict[str, Dict[str, Any]],
    thresholds: scan.Thresholds,
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, Any]] = []
    failed: List[str] = []
    active_meta = manifest["i1_meta"]["h1_manifest_reused"]["active_params_meta"]["row"]
    symbol = str(active_meta["symbol"]).upper()
    period_start = pd.to_datetime(active_meta["period_start"], utc=True)
    period_end = pd.to_datetime(active_meta["period_end"], utc=True)
    initial_equity = safe_float(active_meta.get("initial_equity"))
    for _, row in phase_i_df.iterrows():
        cid = str(row["candidate_id"])
        try:
            params = ga_long._norm_params(json.loads(str(row["params_json"])))  # pylint: disable=protected-access
            repaired = evaluate_repaired_1h_candidate(
                symbol=symbol,
                params_dict=params,
                period_start=period_start,
                period_end=period_end,
                initial_equity=initial_equity,
                fee_model=fee_model,
                df_cache=df_cache,
                raw_cache=raw_cache,
                market_cache=market_cache,
                thresholds=thresholds,
            )
            pass_proxy = int(
                repaired["repaired_trade_count"] >= 20
                and safe_float(repaired["repaired_expectancy_net"]) > 0.0
                and safe_float(repaired["repaired_profit_factor"]) > 1.0
                and safe_float(repaired["repaired_max_dd_pct"]) <= 25.0
            )
            rows.append(
                {
                    "audit_scope": "phase_i_frontier",
                    "candidate_id": cid,
                    "candidate_hash": str(row.get("candidate_hash", "")),
                    "param_hash": str(row.get("param_hash", "")),
                    "seed_origin": str(row.get("seed_origin", "")),
                    "legacy_valid_for_ranking": int(row.get("valid_for_ranking", 0)),
                    "legacy_old_score_field": "OJ2",
                    "legacy_old_score_value": safe_float(row.get("OJ2")),
                    "legacy_old_rank": int(row["legacy_old_rank"]),
                    "legacy_old_rank_tuple": json.dumps([safe_float(row.get("OJ2")), safe_float(row.get("delta_expectancy_vs_exec_baseline"))]),
                    "eval_symbol": symbol,
                    "eval_period_start_utc": str(period_start),
                    "eval_period_end_utc": str(period_end),
                    "repaired_trade_count": int(repaired["repaired_trade_count"]),
                    "repaired_expectancy_net": safe_float(repaired["repaired_expectancy_net"]),
                    "repaired_cvar_5": safe_float(repaired["repaired_cvar_5"]),
                    "repaired_max_dd_pct": safe_float(repaired["repaired_max_dd_pct"]),
                    "repaired_profit_factor": safe_float(repaired["repaired_profit_factor"]),
                    "repaired_final_equity": safe_float(repaired["repaired_final_equity"]),
                    "repaired_cagr_pct": safe_float(repaired["repaired_cagr_pct"]),
                    "repaired_1h_score": safe_float(repaired["repaired_1h_score"]),
                    "repaired_pass_proxy_for_frontier": pass_proxy,
                    "repaired_rank": -1,
                    "rank_delta": float("nan"),
                    "pass_flip": int(int(row.get("valid_for_ranking", 0)) != pass_proxy),
                    "coverage_ok": 1,
                    "coverage_failure_reason": "",
                }
            )
        except Exception as exc:
            failed.append(f"{cid}:{type(exc).__name__}")
            rows.append(
                {
                    "audit_scope": "phase_i_frontier",
                    "candidate_id": cid,
                    "candidate_hash": str(row.get("candidate_hash", "")),
                    "param_hash": str(row.get("param_hash", "")),
                    "seed_origin": str(row.get("seed_origin", "")),
                    "legacy_valid_for_ranking": int(row.get("valid_for_ranking", 0)),
                    "legacy_old_score_field": "OJ2",
                    "legacy_old_score_value": safe_float(row.get("OJ2")),
                    "legacy_old_rank": int(row["legacy_old_rank"]),
                    "legacy_old_rank_tuple": json.dumps([safe_float(row.get("OJ2")), safe_float(row.get("delta_expectancy_vs_exec_baseline"))]),
                    "eval_symbol": symbol,
                    "eval_period_start_utc": str(period_start),
                    "eval_period_end_utc": str(period_end),
                    "repaired_trade_count": float("nan"),
                    "repaired_expectancy_net": float("nan"),
                    "repaired_cvar_5": float("nan"),
                    "repaired_max_dd_pct": float("nan"),
                    "repaired_profit_factor": float("nan"),
                    "repaired_final_equity": float("nan"),
                    "repaired_cagr_pct": float("nan"),
                    "repaired_1h_score": float("nan"),
                    "repaired_pass_proxy_for_frontier": 0,
                    "repaired_rank": -1,
                    "rank_delta": float("nan"),
                    "pass_flip": float("nan"),
                    "coverage_ok": 0,
                    "coverage_failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows), failed


def build_universe_rows(
    *,
    universe_df: pd.DataFrame,
    fee_model: phasec_bt.FeeModel,
    df_cache: Dict[Tuple[str, str], pd.DataFrame],
    raw_cache: Dict[str, pd.DataFrame],
    market_cache: Dict[str, Dict[str, Any]],
    thresholds: scan.Thresholds,
) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, Any]] = []
    failed: List[str] = []
    for _, row in universe_df.iterrows():
        symbol = str(row["symbol"]).upper()
        try:
            params_path = (PROJECT_ROOT / str(row["params_file"])).resolve()
            payload = scan.load_json(params_path)
            params = ga_long._norm_params(scan.unwrap_params(payload))  # pylint: disable=protected-access
            period_start = pd.to_datetime(row["period_start"], utc=True)
            period_end = pd.to_datetime(row["period_end"], utc=True)
            initial_equity = safe_float(row["initial_equity"])
            repaired = evaluate_repaired_1h_candidate(
                symbol=symbol,
                params_dict=params,
                period_start=period_start,
                period_end=period_end,
                initial_equity=initial_equity,
                fee_model=fee_model,
                df_cache=df_cache,
                raw_cache=raw_cache,
                market_cache=market_cache,
                thresholds=thresholds,
            )
            detail = repaired["repaired_pass_detail"]
            rows.append(
                {
                    "symbol": symbol,
                    "side": str(row["side"]),
                    "params_file": str(row["params_file"]),
                    "legacy_pass": int(to_bool(row["pass"])),
                    "legacy_score": safe_float(row["score"]),
                    "legacy_rank": int(row["legacy_rank"]),
                    "legacy_cagr_pct": safe_float(row["cagr_pct"]),
                    "legacy_profit_factor": safe_float(row["profit_factor"]),
                    "legacy_max_dd_pct": safe_float(row["max_dd_pct"]),
                    "legacy_trades": int(safe_float(row["trades"])),
                    "repaired_pass": int(bool(repaired["repaired_pass"])),
                    "repaired_score": safe_float(repaired["repaired_1h_score"]),
                    "repaired_rank": -1,
                    "repaired_cagr_pct": safe_float(detail["cagr_pct"]),
                    "repaired_profit_factor": safe_float(detail["profit_factor"]),
                    "repaired_max_dd_pct": safe_float(detail["max_dd_pct"]),
                    "repaired_trades": int(safe_float(detail["trades"])),
                    "repaired_expectancy_net": safe_float(repaired["repaired_expectancy_net"]),
                    "repaired_cvar_pct_5": safe_float(repaired["repaired_cvar_5"]),
                    "membership_action": "",
                    "score_delta": float("nan"),
                    "rank_delta": float("nan"),
                }
            )
        except Exception as exc:
            failed.append(f"{symbol}:{type(exc).__name__}")
    return pd.DataFrame(rows), failed


def write_blocked(reason: str, next_cmd: str, paths: Sequence[Path]) -> None:
    print(reason)
    print(next_cmd)
    print(", ".join(str(p) for p in paths))


def main() -> None:
    args = build_arg_parser().parse_args()
    topk_list = parse_topk_list(args.topk_list)
    out_root = (PROJECT_ROOT / args.outdir).resolve()
    run_dir = out_root / f"REPAIRED_FRONTIER_CONTAMINATION_AUDIT_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        artifacts = discover_artifacts(args)
    except Exception as exc:
        missing = str(exc)
        next_cmd = "ls -R /root/analysis/0.87/reports/execution_layer /root/analysis/0.87/reports/params_scan"
        write_blocked(missing, next_cmd, [PROJECT_ROOT / "reports" / "execution_layer", PROJECT_ROOT / "reports" / "params_scan"])
        return

    phase_i_csv = artifacts.phase_i_dir / "phaseI2_ga_results.csv"
    phase_i_frontier_compare = artifacts.phase_i_dir / "phaseI_frontier_comparison_vs_H.csv"
    phase_i_manifest_path = artifacts.phase_i_dir / "phaseI_run_manifest.json"
    phase_i_shortlist = artifacts.phase_i_dir / "phaseI2_shortlist_significance.csv"
    best_by_symbol_csv = artifacts.params_scan_dir / "best_by_symbol.csv"
    scan_meta_path = artifacts.params_scan_dir / "scan_meta.json"
    repaired_summary_csv = artifacts.repaired_baseline_dir / "repaired_1h_reference_summary.csv"

    required = [phase_i_csv, phase_i_frontier_compare, phase_i_manifest_path, best_by_symbol_csv, repaired_summary_csv]
    missing_required = [p for p in required if not p.exists()]
    if missing_required:
        next_cmd = "ls -l " + " ".join(str(p.parent) for p in missing_required)
        write_blocked("missing required artifact(s): " + ", ".join(str(p) for p in missing_required), next_cmd, missing_required)
        return

    phase_i_raw = pd.read_csv(phase_i_csv)
    phase_i_df = canonical_phase_i_rows(phase_i_raw)
    phase_i_manifest = load_json(phase_i_manifest_path)
    best_by_symbol_df = pd.read_csv(best_by_symbol_csv)
    scan_meta = load_json(scan_meta_path) if scan_meta_path.exists() else {}
    thresholds = load_thresholds(scan_meta)
    legacy_fee_bps = float(scan_meta.get("fee_bps", 4.0))
    legacy_slip_bps = float(scan_meta.get("slip_bps", 2.0))
    best_by_symbol_df["legacy_rank"] = (
        best_by_symbol_df.assign(
            pass_num=best_by_symbol_df["pass"].map(lambda x: int(to_bool(x))),
            score_num=pd.to_numeric(best_by_symbol_df["score"], errors="coerce"),
        )
        .sort_values(["pass_num", "score_num", "symbol"], ascending=[False, False, True])
        .reset_index(drop=True)
        .assign(tmp_rank=lambda d: np.arange(1, len(d) + 1))
        .set_index("symbol")["tmp_rank"]
        .reindex(best_by_symbol_df["symbol"])
        .to_numpy()
    )
    repaired_baseline_df = pd.read_csv(repaired_summary_csv)

    shortlist_df = pd.read_csv(phase_i_shortlist)
    reconstructed_top = phase_i_df["candidate_id"].astype(str).head(min(10, len(shortlist_df))).tolist()
    shortlist_top = shortlist_df["candidate_id"].astype(str).head(min(10, len(phase_i_df))).tolist()
    phase_i_rank_reconstruction_ok = int(reconstructed_top == shortlist_top)

    score_recalc = (
        pd.to_numeric(best_by_symbol_df["cagr_pct"], errors="coerce")
        * pd.to_numeric(best_by_symbol_df["profit_factor"], errors="coerce")
        / (1.0 + pd.to_numeric(best_by_symbol_df["max_dd_pct"], errors="coerce"))
    )
    score_formula_ok = int(np.allclose(score_recalc.to_numpy(dtype=float), pd.to_numeric(best_by_symbol_df["score"], errors="coerce").to_numpy(dtype=float), atol=1e-9, rtol=0.0))

    if artifacts.repaired_signal_root is None:
        reason = "repaired signal root / fee model source could not be discovered"
        next_cmd = "ls -1 /root/analysis/0.87/reports/execution_layer/MULTICOIN_MODELA_AUDIT_*"
        write_blocked(reason, next_cmd, [PROJECT_ROOT / "reports" / "execution_layer"])
        return

    fee_model = phasec_bt._load_fee_model(artifacts.repaired_signal_root / "fee_model.json")  # pylint: disable=protected-access
    df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    raw_cache: Dict[str, pd.DataFrame] = {}
    market_cache: Dict[str, Dict[str, Any]] = {}

    phase_i_parity_df, phase_i_parity_summary = phase_i_parity_check(phase_i_df, phase_i_manifest, df_cache, raw_cache)
    universe_parity_df, universe_parity_summary = universe_parity_check(
        best_by_symbol_df,
        df_cache,
        raw_cache,
        thresholds,
        legacy_fee_bps,
        legacy_slip_bps,
    )

    if phase_i_rank_reconstruction_ok != 1 or score_formula_ok != 1:
        reason = (
            f"legacy reconstruction validation failed: phase_i_rank_reconstruction_ok={phase_i_rank_reconstruction_ok}, "
            f"universe_score_formula_ok={score_formula_ok}"
        )
        next_cmd = "/root/analysis/0.87/.venv/bin/python /root/analysis/0.87/scripts/repaired_frontier_contamination_audit.py"
        paths = [phase_i_shortlist, best_by_symbol_csv]
        write_blocked(reason, next_cmd, paths)
        return

    if not phase_i_parity_summary["ok"] or not universe_parity_summary["ok"]:
        reason = "legacy reconstruction parity is too weak to trust the audit"
        if phase_i_parity_summary.get("reason"):
            reason += f"; {phase_i_parity_summary['reason']}"
        if universe_parity_summary.get("reason"):
            reason += f"; {universe_parity_summary['reason']}"
        next_cmd = "/root/analysis/0.87/.venv/bin/python /root/analysis/0.87/scripts/repaired_frontier_contamination_audit.py"
        paths = [phase_i_csv, best_by_symbol_csv]
        write_blocked(reason, next_cmd, paths)
        return

    frontier_df, frontier_failed = build_phase_i_rows(
        phase_i_df=phase_i_df,
        manifest=phase_i_manifest,
        fee_model=fee_model,
        df_cache=df_cache,
        raw_cache=raw_cache,
        market_cache=market_cache,
        thresholds=thresholds,
    )
    universe_results_df, universe_failed = build_universe_rows(
        universe_df=best_by_symbol_df,
        fee_model=fee_model,
        df_cache=df_cache,
        raw_cache=raw_cache,
        market_cache=market_cache,
        thresholds=thresholds,
    )

    frontier_ok_rows = int((frontier_df["coverage_ok"] == 1).sum())
    frontier_cov_pct = frontier_ok_rows / float(max(1, len(frontier_df)))
    universe_cov_pct = len(universe_results_df) / float(max(1, len(best_by_symbol_df)))
    frontier_coverage = {
        "ok_rows": frontier_ok_rows,
        "total_rows": int(len(frontier_df)),
        "coverage_pct": frontier_cov_pct,
        "failed_rows": frontier_failed,
    }
    universe_coverage = {
        "ok_rows": int(len(universe_results_df)),
        "total_rows": int(len(best_by_symbol_df)),
        "coverage_pct": universe_cov_pct,
        "failed_rows": universe_failed,
    }

    coverage_block = False
    coverage_reason = ""
    if frontier_cov_pct < float(args.min_frontier_coverage):
        coverage_block = True
        coverage_reason = f"Phase I frontier coverage below threshold: {frontier_cov_pct:.4%} < {float(args.min_frontier_coverage):.4%}"
    if not coverage_block and int(args.require_full_universe_coverage) == 1 and len(universe_results_df) != len(best_by_symbol_df):
        coverage_block = True
        coverage_reason = f"Universe coverage below required full coverage: {len(universe_results_df)}/{len(best_by_symbol_df)}"
    if coverage_block:
        cov_fail_df = frontier_df[frontier_df["coverage_ok"] == 0][["candidate_id", "coverage_failure_reason"]].copy()
        if universe_failed:
            extra = pd.DataFrame({"candidate_id": universe_failed, "coverage_failure_reason": ["universe_evaluation_failed"] * len(universe_failed)})
            cov_fail_df = pd.concat([cov_fail_df, extra], ignore_index=True)
        if not cov_fail_df.empty:
            cov_fail_df.to_csv(run_dir / "repaired_frontier_coverage_failures.csv", index=False)
        next_cmd = "/root/analysis/0.87/.venv/bin/python /root/analysis/0.87/scripts/repaired_frontier_contamination_audit.py"
        paths = [phase_i_csv, best_by_symbol_csv]
        write_blocked(coverage_reason, next_cmd, paths)
        return

    universe_results_df = universe_results_df.sort_values(["legacy_rank", "symbol"]).reset_index(drop=True)
    repaired_smoke_df, repaired_smoke_summary = repaired_smoke_check(universe_results_df, repaired_baseline_df)
    if not repaired_smoke_summary["ok"]:
        next_cmd = "/root/analysis/0.87/.venv/bin/python /root/analysis/0.87/scripts/repaired_frontier_contamination_audit.py"
        write_blocked("legacy reconstruction parity is too weak to trust the audit; " + repaired_smoke_summary["reason"], next_cmd, [repaired_summary_csv])
        return

    frontier_stats = compute_frontier_instability(frontier_df[frontier_df["coverage_ok"] == 1].copy(), topk_list)
    frontier_out_df = frontier_stats["df"].copy()
    universe_diff_df, universe_stats = compute_universe_diff(universe_results_df.copy(), topk_list)
    contamination = classify_contamination(universe_stats, frontier_stats)
    recommendation = map_recommendation(
        contamination=contamination,
        universe_stats=universe_stats,
        frontier_stats=frontier_stats,
        parity_ok=True,
    )

    frontier_out_df.to_csv(run_dir / "repaired_frontier_rerank_audit.csv", index=False)
    universe_diff_df[
        [
            "symbol",
            "side",
            "params_file",
            "legacy_pass",
            "legacy_score",
            "legacy_rank",
            "legacy_cagr_pct",
            "legacy_profit_factor",
            "legacy_max_dd_pct",
            "legacy_trades",
            "repaired_pass",
            "repaired_score",
            "repaired_rank",
            "repaired_cagr_pct",
            "repaired_profit_factor",
            "repaired_max_dd_pct",
            "repaired_trades",
            "membership_action",
            "score_delta",
            "rank_delta",
        ]
    ].to_csv(run_dir / "repaired_universe_membership_diff.csv", index=False)

    if artifacts.phase_j0_dir is not None:
        j0_df = pd.read_csv(artifacts.phase_j0_dir / "phaseJ05_tradeoff_frontier.csv")
        pd.DataFrame(
            {
                "phase_j0_dir": [str(artifacts.phase_j0_dir)],
                "rows": [int(len(j0_df))],
                "re_scored": [0],
                "reason": ["missing candidate definitions / params_json; appendix only"],
            }
        ).to_csv(run_dir / "repaired_frontier_phasej0_appendix.csv", index=False)

    manifest = {
        "generated_utc": utc_now().isoformat(),
        "artifact_paths": {
            "phase_i_dir": str(artifacts.phase_i_dir),
            "params_scan_dir": str(artifacts.params_scan_dir),
            "repaired_baseline_dir": str(artifacts.repaired_baseline_dir),
            "phase_j0_dir": str(artifacts.phase_j0_dir) if artifacts.phase_j0_dir else "",
            "repaired_signal_root": str(artifacts.repaired_signal_root) if artifacts.repaired_signal_root else "",
            "phase_h_rep_subset": str(PHASE_H_REP_SUBSET) if PHASE_H_REP_SUBSET.exists() else "",
        },
        "validation": {
            "phase_i_rank_reconstruction_ok": phase_i_rank_reconstruction_ok,
            "universe_score_formula_ok": score_formula_ok,
            "phase_i_parity_summary": phase_i_parity_summary,
            "universe_parity_summary": universe_parity_summary,
            "repaired_smoke_summary": repaired_smoke_summary,
        },
        "coverage": {
            "frontier": frontier_coverage,
            "universe": universe_coverage,
        },
        "metrics": {
            "frontier": {k: v for k, v in frontier_stats.items() if k != "df"},
            "universe": universe_stats,
            "contamination": contamination,
        },
        "recommendation": recommendation,
        "notes": [
            "Primary decision weighting: universe pass-set instability, then universe rank instability, then Phase I top-k instability, then Phase I repaired pass-proxy flips.",
            "Phase I repaired pass proxy is secondary annotation only.",
            "Audit intentionally excludes the 3m execution layer.",
        ],
    }
    json_dump(run_dir / "repaired_frontier_contamination_manifest.json", manifest)
    write_report(
        out_path=run_dir / "repaired_frontier_contamination_report.md",
        artifacts=artifacts,
        manifest=manifest,
        phase_i_parity_summary=phase_i_parity_summary,
        universe_parity_summary=universe_parity_summary,
        repaired_smoke_summary=repaired_smoke_summary,
        frontier_coverage=frontier_coverage,
        universe_coverage=universe_coverage,
        frontier_stats=frontier_stats,
        universe_stats=universe_stats,
        contamination=contamination,
        recommendation=recommendation,
        phase_i_parity_df=phase_i_parity_df,
        universe_parity_df=universe_parity_df,
        repaired_smoke_df=repaired_smoke_df,
    )

    print(json.dumps(
        {
            "run_dir": str(run_dir),
            "recommendation": recommendation,
            "contamination": contamination,
            "frontier_coverage_pct": frontier_cov_pct,
            "universe_coverage_pct": universe_cov_pct,
            "universe_passed_set_jaccard": universe_stats["passed_set_jaccard"],
            "phase_i_spearman": frontier_stats["spearman"],
            "phase_i_top10_turnover": frontier_stats["topk"].get("top10_turnover"),
        },
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
