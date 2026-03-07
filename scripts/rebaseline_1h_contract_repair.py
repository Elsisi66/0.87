#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import backtest_exec_phasec_sol as phasec_bt  # noqa: E402


TARGET_SYMBOLS = ["SOLUSDT", "NEARUSDT", "AVAXUSDT"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_tag() -> str:
    return _utc_now().strftime("%Y%m%d_%H%M%S")


def _tail_mean(values: pd.Series, frac: float) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    k = max(1, int(math.ceil(float(frac) * float(arr.size))))
    arr.sort()
    return float(arr[:k].mean())


def _max_drawdown_from_returns(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + arr)
    peaks = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peaks, 1e-12) - 1.0
    return float(dd.min())


def _rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _latest_multicoin_signal_root() -> Path:
    roots = sorted((PROJECT_ROOT / "reports" / "execution_layer").glob("MULTICOIN_MODELA_AUDIT_*"))
    roots = [x for x in roots if (x / "_signal_inputs").exists() and (x / "fee_model.json").exists()]
    if not roots:
        raise FileNotFoundError("No MULTICOIN_MODELA_AUDIT_* signal root with _signal_inputs and fee_model.json found")
    return roots[-1]


def _load_symbols() -> list[str]:
    universe_fp = PROJECT_ROOT / "paper_trading" / "config" / "resolved_universe.json"
    if not universe_fp.exists():
        raise FileNotFoundError(f"Missing resolved universe file: {universe_fp}")
    payload = json.loads(universe_fp.read_text(encoding="utf-8"))
    symbols = [str(x).upper() for x in payload.get("symbols", [])]
    ordered: list[str] = []
    for sym in TARGET_SYMBOLS + symbols:
        if sym and sym not in ordered:
            ordered.append(sym)
    return ordered


def _load_signal_frame(signal_root: Path, symbol: str) -> pd.DataFrame:
    fp = signal_root / "_signal_inputs" / f"{symbol}_signals_1h.csv"
    if not fp.exists():
        raise FileNotFoundError(f"Missing signal file for {symbol}: {fp}")
    df = pd.read_csv(fp)
    req = {"signal_id", "signal_time", "strategy_tp_mult", "strategy_sl_mult"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise RuntimeError(f"Signal file missing required columns for {symbol}: {miss}")
    out = pd.DataFrame(
        {
            "signal_id": df["signal_id"].astype(str),
            "signal_time": pd.to_datetime(df["signal_time"], utc=True, errors="coerce"),
            "tp_mult": pd.to_numeric(df["strategy_tp_mult"], errors="coerce"),
            "sl_mult": pd.to_numeric(df["strategy_sl_mult"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["signal_time", "tp_mult", "sl_mult"]).sort_values(["signal_time", "signal_id"]).reset_index(drop=True)
    return out


def _trade_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["filled"] = pd.to_numeric(out.get("filled"), errors="coerce").fillna(0).astype(int)
    out["valid_for_metrics"] = pd.to_numeric(out.get("valid_for_metrics"), errors="coerce").fillna(0).astype(int)
    out["hold_minutes"] = pd.to_numeric(out.get("hold_minutes"), errors="coerce")
    out["pnl_net_pct"] = pd.to_numeric(out.get("pnl_net_pct"), errors="coerce")
    out["entry_time"] = pd.to_datetime(out.get("entry_time"), utc=True, errors="coerce")
    out["exit_time"] = pd.to_datetime(out.get("exit_time"), utc=True, errors="coerce")
    out["exit_reason"] = out.get("exit_reason", "").astype(str)
    return out[(out["filled"] == 1) & (out["valid_for_metrics"] == 1)].copy()


def _same_bar_mask(df: pd.DataFrame) -> pd.Series:
    return (df["entry_time"].notna()) & (df["exit_time"].notna()) & (df["entry_time"] == df["exit_time"])


def _metrics_row(symbol: str, before: pd.DataFrame, after: pd.DataFrame, total_signals: int) -> dict[str, Any]:
    b = _trade_subset(before)
    a = _trade_subset(after)
    b_same = _same_bar_mask(b)
    a_same = _same_bar_mask(a)
    b_zero = pd.to_numeric(b["hold_minutes"], errors="coerce").fillna(-1.0) == 0.0
    a_zero = pd.to_numeric(a["hold_minutes"], errors="coerce").fillna(-1.0) == 0.0
    b_same_sl = b_same & (b["exit_reason"] == "sl")
    a_same_sl = a_same & (a["exit_reason"] == "sl")

    return {
        "symbol": symbol,
        "total_signals": int(total_signals),
        "before_trade_count": int(len(b)),
        "after_trade_count": int(len(a)),
        "before_expectancy_net": float(pd.to_numeric(b["pnl_net_pct"], errors="coerce").mean()) if len(b) else float("nan"),
        "after_expectancy_net": float(pd.to_numeric(a["pnl_net_pct"], errors="coerce").mean()) if len(a) else float("nan"),
        "before_cvar_5": _tail_mean(b["pnl_net_pct"], 0.05),
        "after_cvar_5": _tail_mean(a["pnl_net_pct"], 0.05),
        "before_maxdd": _max_drawdown_from_returns(b.sort_values("entry_time")["pnl_net_pct"]),
        "after_maxdd": _max_drawdown_from_returns(a.sort_values("entry_time")["pnl_net_pct"]),
        "before_median_hold_min": float(pd.to_numeric(b["hold_minutes"], errors="coerce").median()) if len(b) else float("nan"),
        "after_median_hold_min": float(pd.to_numeric(a["hold_minutes"], errors="coerce").median()) if len(a) else float("nan"),
        "before_zero_hold_count": int(b_zero.sum()),
        "after_zero_hold_count": int(a_zero.sum()),
        "before_zero_hold_rate": _rate(int(b_zero.sum()), int(len(b))),
        "after_zero_hold_rate": _rate(int(a_zero.sum()), int(len(a))),
        "before_same_bar_exit_count": int(b_same.sum()),
        "after_same_bar_exit_count": int(a_same.sum()),
        "before_same_bar_exit_rate": _rate(int(b_same.sum()), int(len(b))),
        "after_same_bar_exit_rate": _rate(int(a_same.sum()), int(len(a))),
        "before_same_bar_sl_count": int(b_same_sl.sum()),
        "after_same_bar_sl_count": int(a_same_sl.sum()),
        "before_same_bar_sl_rate": _rate(int(b_same_sl.sum()), int(len(b))),
        "after_same_bar_sl_rate": _rate(int(a_same_sl.sum()), int(len(a))),
    }


def _aggregate_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not rows:
        return {
            "symbol": label,
            "total_signals": 0,
            "before_trade_count": 0,
            "after_trade_count": 0,
            "before_expectancy_net": float("nan"),
            "after_expectancy_net": float("nan"),
            "before_cvar_5": float("nan"),
            "after_cvar_5": float("nan"),
            "before_maxdd": float("nan"),
            "after_maxdd": float("nan"),
            "before_median_hold_min": float("nan"),
            "after_median_hold_min": float("nan"),
            "before_zero_hold_count": 0,
            "after_zero_hold_count": 0,
            "before_zero_hold_rate": 0.0,
            "after_zero_hold_rate": 0.0,
            "before_same_bar_exit_count": 0,
            "after_same_bar_exit_count": 0,
            "before_same_bar_exit_rate": 0.0,
            "after_same_bar_exit_rate": 0.0,
            "before_same_bar_sl_count": 0,
            "after_same_bar_sl_count": 0,
            "before_same_bar_sl_rate": 0.0,
            "after_same_bar_sl_rate": 0.0,
        }
    out: dict[str, Any] = {"symbol": label}
    out["total_signals"] = int(sum(int(x["total_signals"]) for x in rows))
    out["before_trade_count"] = int(sum(int(x["before_trade_count"]) for x in rows))
    out["after_trade_count"] = int(sum(int(x["after_trade_count"]) for x in rows))
    out["before_zero_hold_count"] = int(sum(int(x["before_zero_hold_count"]) for x in rows))
    out["after_zero_hold_count"] = int(sum(int(x["after_zero_hold_count"]) for x in rows))
    out["before_same_bar_exit_count"] = int(sum(int(x["before_same_bar_exit_count"]) for x in rows))
    out["after_same_bar_exit_count"] = int(sum(int(x["after_same_bar_exit_count"]) for x in rows))
    out["before_same_bar_sl_count"] = int(sum(int(x["before_same_bar_sl_count"]) for x in rows))
    out["after_same_bar_sl_count"] = int(sum(int(x["after_same_bar_sl_count"]) for x in rows))
    out["before_zero_hold_rate"] = _rate(out["before_zero_hold_count"], out["before_trade_count"])
    out["after_zero_hold_rate"] = _rate(out["after_zero_hold_count"], out["after_trade_count"])
    out["before_same_bar_exit_rate"] = _rate(out["before_same_bar_exit_count"], out["before_trade_count"])
    out["after_same_bar_exit_rate"] = _rate(out["after_same_bar_exit_count"], out["after_trade_count"])
    out["before_same_bar_sl_rate"] = _rate(out["before_same_bar_sl_count"], out["before_trade_count"])
    out["after_same_bar_sl_rate"] = _rate(out["after_same_bar_sl_count"], out["after_trade_count"])
    for key in ("before_expectancy_net", "after_expectancy_net", "before_cvar_5", "after_cvar_5", "before_maxdd", "after_maxdd", "before_median_hold_min", "after_median_hold_min"):
        vals = [float(x[key]) for x in rows if pd.notna(x[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def _classify(summary_df: pd.DataFrame) -> tuple[str, bool, str]:
    target = summary_df[summary_df["symbol"].isin(TARGET_SYMBOLS)].copy()
    aggregate = summary_df[summary_df["symbol"] == "UNIVERSE_ALL"].copy()
    agg_exp = float(aggregate.iloc[0]["after_expectancy_net"]) if not aggregate.empty else float("nan")
    agg_dd = float(aggregate.iloc[0]["after_maxdd"]) if not aggregate.empty else float("nan")
    target_pos = int((pd.to_numeric(target["after_expectancy_net"], errors="coerce") > 0).sum())
    target_nonpos = int((pd.to_numeric(target["after_expectancy_net"], errors="coerce") <= 0).sum())
    target_trade_floor_ok = bool((pd.to_numeric(target["after_trade_count"], errors="coerce") >= 20).all())
    target_zero_rates = pd.to_numeric(target["after_zero_hold_rate"], errors="coerce")
    target_before_zero_rates = pd.to_numeric(target["before_zero_hold_rate"], errors="coerce")
    zero_repair_effective = bool((target_zero_rates < target_before_zero_rates).all()) if len(target) else False

    if target_pos == len(target) and pd.notna(agg_exp) and agg_exp > 0 and target_trade_floor_ok:
        if pd.notna(agg_dd) and agg_dd < -0.35:
            return "1H_BASELINE_WEAKENS_BUT_SURVIVES", True, "expectancy remains positive after chronology repair but aggregate drawdown is materially worse"
        return "1H_BASELINE_SURVIVES_REPAIR", True, "target trio remains profitable after chronology repair"
    if target_pos >= max(1, len(target) - 1) and pd.notna(agg_exp) and agg_exp > 0 and zero_repair_effective:
        return "1H_BASELINE_WEAKENS_BUT_SURVIVES", True, "same-bar exits were removed and most priority symbols remain profitable, but the edge is weaker"
    if target_nonpos >= max(1, len(target) - 1) or (pd.notna(agg_exp) and agg_exp <= 0):
        return "1H_SIGNAL_REDESIGN_REQUIRED", False, "repaired chronology removes the prior timing advantage and the current signal family no longer clears profitability thresholds"
    return "1H_BASELINE_INVALIDATED", False, "repaired chronology materially weakens the current signal family beyond the acceptable baseline"


def main() -> None:
    run_root = PROJECT_ROOT / "reports" / "execution_layer" / f"1H_CONTRACT_REPAIR_REBASELINE_{_utc_tag()}"
    run_root.mkdir(parents=True, exist_ok=True)

    signal_root = _latest_multicoin_signal_root()
    fee = phasec_bt._load_fee_model(signal_root / "fee_model.json")  # pylint: disable=protected-access
    symbols = _load_symbols()

    report_lines = [
        "# 1H Contract Repair Report",
        "",
        f"- Generated UTC: `{_utc_now().isoformat()}`",
        f"- Signal source root: `{signal_root}`",
        f"- Universe source: `{PROJECT_ROOT / 'paper_trading' / 'config' / 'resolved_universe.json'}`",
        "",
        "## H0 Frozen Broken Behavior",
        "- Pre-patch 1h reference entry fill was created at the signal bar open inside `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference` (`entry_price = open_np[idx]`).",
        "- Pre-patch same-parent-bar exit evaluation began immediately from the same bar by passing `entry_idx=int(idx)` into `scripts/execution_layer_3m_ict.py::_simulate_path_long`, whose scan loop runs `for i in range(entry_idx, ...)`.",
        "- Pre-patch stop/target construction used raw entry open in `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference` (`sl = entry_price * sl_mult`, `tp = entry_price * tp_mult`).",
        "- Pre-patch hold duration collapsed to zero whenever the same parent bar hit the stop because `hold_minutes = (exit_time - entry_time)` and both timestamps were the same bar.",
        "",
        "## H1 Repaired Contract",
        "- Patched `scripts/backtest_exec_phasec_sol.py::_simulate_1h_reference` to preserve the same entry event but start exit evaluation from the first full subsequent 1h bar (`idx + 1`) by default.",
        "- Added explicit export fields: `entry_parent_bar_time`, `exit_eval_start_time`, `exit_eval_bar_time`, and `defer_exit_to_next_bar`.",
        "- Signal generation input stayed unchanged; only the 1h exit chronology was rebased.",
        "",
    ]

    before_frames: list[pd.DataFrame] = []
    after_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []

    for symbol in symbols:
        sig = _load_signal_frame(signal_root, symbol)
        split_lookup = {str(x): 0 for x in sig["signal_id"].tolist()}
        before = phasec_bt._simulate_1h_reference(  # pylint: disable=protected-access
            signals_df=sig,
            split_lookup=split_lookup,
            fee=fee,
            exec_horizon_hours=12.0,
            symbol=symbol,
            defer_exit_to_next_bar=False,
        )
        after = phasec_bt._simulate_1h_reference(  # pylint: disable=protected-access
            signals_df=sig,
            split_lookup=split_lookup,
            fee=fee,
            exec_horizon_hours=12.0,
            symbol=symbol,
            defer_exit_to_next_bar=True,
        )
        before["symbol"] = symbol
        after["symbol"] = symbol
        before_frames.append(before)
        after_frames.append(after)
        summary_rows.append(_metrics_row(symbol, before, after, len(sig)))

    summary_rows.append(_aggregate_row(summary_rows, "UNIVERSE_ALL"))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_root / "repaired_1h_reference_summary.csv", index=False)

    repaired_trades = pd.concat(after_frames, ignore_index=True).sort_values(["symbol", "signal_time", "signal_id"]).reset_index(drop=True)
    repaired_trades.to_csv(run_root / "repaired_1h_trades.csv", index=False)

    zero_df = summary_df[
        [
            "symbol",
            "before_trade_count",
            "before_zero_hold_count",
            "before_zero_hold_rate",
            "after_trade_count",
            "after_zero_hold_count",
            "after_zero_hold_rate",
        ]
    ].copy()
    zero_df.to_csv(run_root / "zero_hold_before_after_1h.csv", index=False)

    same_df = summary_df[
        [
            "symbol",
            "before_trade_count",
            "before_same_bar_exit_count",
            "before_same_bar_exit_rate",
            "before_same_bar_sl_count",
            "before_same_bar_sl_rate",
            "after_trade_count",
            "after_same_bar_exit_count",
            "after_same_bar_exit_rate",
            "after_same_bar_sl_count",
            "after_same_bar_sl_rate",
        ]
    ].copy()
    same_df.to_csv(run_root / "same_bar_exit_before_after_1h.csv", index=False)

    classification, signal_stands, rationale = _classify(summary_df)
    target_view = summary_df[summary_df["symbol"].isin(TARGET_SYMBOLS)].copy()
    report_lines.extend(
        [
            "## H2/H4 Rebaseline Scope",
            f"- Symbols processed: `{', '.join(summary_df[summary_df['symbol'] != 'UNIVERSE_ALL']['symbol'].tolist())}`",
            "- Priority symbols included: `SOLUSDT`, `NEARUSDT`, `AVAXUSDT`",
            "",
            "## H3 Signal Layer Reassessment",
            f"- Classification: `{classification}`",
            f"- Signal layer still stands: `{int(signal_stands)}`",
            f"- Rationale: `{rationale}`",
            "",
            "## Target Symbol Snapshot",
        ]
    )
    for row in target_view.itertuples(index=False):
        report_lines.append(
            f"- {row.symbol}: expectancy `{row.after_expectancy_net:.8f}`, cvar_5 `{row.after_cvar_5:.8f}`, maxdd `{row.after_maxdd:.8f}`, same_bar_rate `{row.after_same_bar_exit_rate:.4%}`, zero_hold_rate `{row.after_zero_hold_rate:.4%}`"
        )

    (run_root / "contract_repair_1h_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    reassessment_lines = [
        "# Signal Layer Reassessment",
        "",
        f"- Classification: `{classification}`",
        f"- Signal layer still stands: `{int(signal_stands)}`",
        f"- Rationale: `{rationale}`",
        "",
        "## Priority Symbols",
    ]
    for row in target_view.itertuples(index=False):
        delta_exp = float(row.after_expectancy_net) - float(row.before_expectancy_net)
        delta_dd = float(row.after_maxdd) - float(row.before_maxdd)
        reassessment_lines.append(
            f"- {row.symbol}: expectancy `{row.after_expectancy_net:.8f}` (delta `{delta_exp:.8f}`), cvar_5 `{row.after_cvar_5:.8f}`, maxdd `{row.after_maxdd:.8f}` (delta `{delta_dd:.8f}`), trade_count `{row.after_trade_count}`, median_hold_min `{row.after_median_hold_min:.2f}`"
        )
    reassessment_lines.extend(
        [
            "",
            "## Interpretation",
            "- The signal timestamps were held constant; only the 1h exit chronology changed.",
            "- Any drop in expectancy here is therefore attributable to removing the same-parent-bar exit path, not to a different signal family.",
            "- The repaired same-bar and zero-hold diagnostics are in the companion CSV files.",
        ]
    )
    (run_root / "signal_layer_reassessment.md").write_text("\n".join(reassessment_lines) + "\n", encoding="utf-8")

    classification_payload = {
        "generated_utc": _utc_now().isoformat(),
        "classification": classification,
        "signal_layer_still_stands": bool(signal_stands),
        "rationale": rationale,
        "signal_source_root": str(signal_root),
        "symbols_processed": summary_df[summary_df["symbol"] != "UNIVERSE_ALL"]["symbol"].tolist(),
        "priority_symbols": TARGET_SYMBOLS,
        "target_metrics": {
            row.symbol: {
                "after_expectancy_net": float(row.after_expectancy_net),
                "after_cvar_5": float(row.after_cvar_5),
                "after_maxdd": float(row.after_maxdd),
                "after_trade_count": int(row.after_trade_count),
            }
            for row in target_view.itertuples(index=False)
        },
    }
    (run_root / "next_state_classification.json").write_text(json.dumps(classification_payload, indent=2), encoding="utf-8")

    print(json.dumps({"run_root": str(run_root), "classification": classification, "signal_layer_still_stands": bool(signal_stands)}, indent=2))


if __name__ == "__main__":
    main()
