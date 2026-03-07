#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bucket_vol_from_percentile(v: float) -> str:
    if not np.isfinite(v):
        return "vol_unknown"
    if float(v) <= 33.333333:
        return "vol_low"
    if float(v) <= 66.666667:
        return "vol_mid"
    return "vol_high"


def _bucket_vol_from_expanding_quantiles(
    atr: pd.Series,
    min_hist: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    # Strict no-lookahead: q33/q66 at t use history up to t-1.
    s = pd.to_numeric(atr, errors="coerce")
    q33 = s.expanding(min_periods=int(min_hist)).quantile(1.0 / 3.0).shift(1)
    q66 = s.expanding(min_periods=int(min_hist)).quantile(2.0 / 3.0).shift(1)
    out = pd.Series(["vol_unknown"] * len(s), index=s.index, dtype=object)
    m = s.notna() & q33.notna() & q66.notna()
    out.loc[m & (s <= q33)] = "vol_low"
    out.loc[m & (s > q33) & (s <= q66)] = "vol_mid"
    out.loc[m & (s > q66)] = "vol_high"
    return out, q33, q66


def _bucket_trend(v: float) -> str:
    if not np.isfinite(v):
        return "trend_flat"
    return "trend_up" if float(v) >= 0.5 else "trend_down"


def _bucket_session(ts: pd.Timestamp) -> str:
    if pd.isna(ts):
        return "session_unknown"
    h = int(ts.hour)
    if 0 <= h < 7:
        return "session_asia"
    if 7 <= h < 13:
        return "session_eu"
    if 13 <= h < 20:
        return "session_us"
    return "session_late"


def build_regime_labels(
    *,
    signal_df: pd.DataFrame,
    min_hist_bars: int = 50,
    use_trend: int = 1,
    use_session: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    df = signal_df.copy()
    if "signal_id" not in df.columns:
        df["signal_id"] = [f"sig_{i:05d}" for i in range(1, len(df) + 1)]
    if "signal_time" not in df.columns:
        raise ValueError("signal_time missing from signal subset")
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True, errors="coerce")
    df = df[df["signal_time"].notna()].sort_values("signal_time").reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid signal rows after timestamp parsing")

    # Volatility regime at signal time, no lookahead.
    vol_source = ""
    q33 = pd.Series([np.nan] * len(df))
    q66 = pd.Series([np.nan] * len(df))
    if "atr_percentile_1h" in df.columns and pd.to_numeric(df["atr_percentile_1h"], errors="coerce").notna().any():
        vol_source = "atr_percentile_1h"
        ap = pd.to_numeric(df["atr_percentile_1h"], errors="coerce")
        df["vol_regime"] = ap.map(_bucket_vol_from_percentile)
    elif "atr_1h" in df.columns and pd.to_numeric(df["atr_1h"], errors="coerce").notna().any():
        vol_source = "atr_1h_expanding_terciles_shift1"
        vol_bucket, q33, q66 = _bucket_vol_from_expanding_quantiles(
            atr=pd.to_numeric(df["atr_1h"], errors="coerce"),
            min_hist=int(min_hist_bars),
        )
        df["vol_regime"] = vol_bucket
    else:
        vol_source = "missing"
        df["vol_regime"] = "vol_unknown"

    # Trend regime at signal time, from exported 1h state only.
    trend_source = ""
    if int(use_trend) == 1 and "trend_up_1h" in df.columns:
        trend_source = "trend_up_1h"
        trend = pd.to_numeric(df["trend_up_1h"], errors="coerce")
        df["trend_regime"] = trend.map(_bucket_trend)
    else:
        trend_source = "disabled_or_missing"
        df["trend_regime"] = "trend_flat"

    df["session_bucket"] = df["signal_time"].map(_bucket_session) if int(use_session) == 1 else "session_disabled"

    if int(use_session) == 1:
        df["combined_regime"] = (
            df["vol_regime"].astype(str)
            + "|"
            + df["trend_regime"].astype(str)
            + "|"
            + df["session_bucket"].astype(str)
        )
    else:
        df["combined_regime"] = df["vol_regime"].astype(str) + "|" + df["trend_regime"].astype(str)

    labels = df[
        [
            "signal_id",
            "signal_time",
            "vol_regime",
            "trend_regime",
            "session_bucket",
            "combined_regime",
        ]
    ].copy()
    labels["vol_q33_used"] = pd.to_numeric(q33, errors="coerce")
    labels["vol_q66_used"] = pd.to_numeric(q66, errors="coerce")

    cov = (
        labels.groupby(["combined_regime"], dropna=False)
        .agg(signals=("signal_id", "count"))
        .reset_index()
        .sort_values(["signals", "combined_regime"], ascending=[False, True])
        .reset_index(drop=True)
    )

    diag_lines: List[str] = []
    diag_lines.append("# Regime Diagnostics")
    diag_lines.append("")
    diag_lines.append(f"- Generated UTC: {_utc_now_iso()}")
    diag_lines.append(f"- Signals labeled: {len(labels)}")
    diag_lines.append(f"- Vol source: `{vol_source}`")
    diag_lines.append(f"- Trend source: `{trend_source}`")
    diag_lines.append(f"- use_session: {int(use_session)}")
    diag_lines.append(f"- Combined regimes: {cov['combined_regime'].nunique() if not cov.empty else 0}")
    diag_lines.append("")
    diag_lines.append("## Coverage")
    diag_lines.append("")
    for _, r in cov.head(20).iterrows():
        diag_lines.append(f"- {r['combined_regime']}: {int(r['signals'])}")
    diag_text = "\n".join(diag_lines).strip() + "\n"
    return labels, cov, diag_text


def run(args: argparse.Namespace) -> Path:
    sig_csv = Path(args.signal_subset_csv).resolve()
    if not sig_csv.exists():
        raise FileNotFoundError(f"signal_subset_csv not found: {sig_csv}")
    out_dir = Path(args.outdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sdf = pd.read_csv(sig_csv)
    labels, cov, diag_text = build_regime_labels(
        signal_df=sdf,
        min_hist_bars=int(args.min_hist_bars),
        use_trend=int(args.use_trend),
        use_session=int(args.use_session),
    )
    labels_fp = out_dir / "regime_labels.csv"
    cov_fp = out_dir / "regime_coverage.csv"
    diag_fp = out_dir / "regime_diagnostics.md"
    labels.to_csv(labels_fp, index=False)
    cov.to_csv(cov_fp, index=False)
    diag_fp.write_text(diag_text, encoding="utf-8")

    meta = {
        "generated_utc": _utc_now_iso(),
        "signal_subset_csv": str(sig_csv),
        "rows_in": int(len(sdf)),
        "rows_labeled": int(len(labels)),
        "combined_regime_count": int(cov["combined_regime"].nunique() if not cov.empty else 0),
        "use_trend": int(args.use_trend),
        "use_session": int(args.use_session),
        "min_hist_bars": int(args.min_hist_bars),
    }
    (out_dir / "regime_features_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(str(labels_fp))
    print(str(cov_fp))
    print(str(diag_fp))
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build no-lookahead regime labels from frozen signal subset.")
    ap.add_argument("--signal-subset-csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-hist-bars", type=int, default=50)
    ap.add_argument("--use-trend", type=int, default=1)
    ap.add_argument("--use-session", type=int, default=0)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
