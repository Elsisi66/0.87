#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_ae_signal_labeling as ae  # noqa: E402
from scripts import phase_af_ah_sizing_autorun as af  # noqa: E402
from scripts import phase_u_combined_1h3m_pilot as pu  # noqa: E402
from scripts import sol_reconcile_truth as recon  # noqa: E402
from src.bot087.optim import ga as ga_long  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


LOCKED = {
    "frozen_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "canonical_fee_model": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "canonical_metrics_definition": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
    "phase_ae_dir": "/root/analysis/0.87/reports/execution_layer/PHASEAE_SIGNAL_LABELING_20260223_111116",
    "phase_f_dir": "/root/analysis/0.87/reports/execution_layer/PHASEF_NO_GO_RECOVERY_20260223_193804",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def to_num(x: Any) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def safe_div(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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


def write_text(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def markdown_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return "_(none)_"
    x = df.loc[:, [c for c in cols if c in df.columns]].copy()
    lines: List[str] = []
    lines.append("| " + " | ".join(x.columns.tolist()) + " |")
    lines.append("| " + " | ".join(["---"] * len(x.columns)) + " |")
    for r in x.itertuples(index=False):
        vals: List[str] = []
        for v in r:
            if isinstance(v, float):
                vals.append(f"{v:.8g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def spearman_no_scipy(a: pd.Series, b: pd.Series) -> float:
    xa = to_num(a)
    xb = to_num(b)
    m = xa.notna() & xb.notna()
    if int(m.sum()) < 3:
        return float("nan")
    ra = xa[m].rank(method="average").to_numpy(dtype=float)
    rb = xb[m].rank(method="average").to_numpy(dtype=float)
    sa = float(np.std(ra, ddof=0))
    sb = float(np.std(rb, ddof=0))
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def normalize_01(s: pd.Series) -> pd.Series:
    x = to_num(s)
    lo = float(np.nanquantile(x.dropna().to_numpy(dtype=float), 0.10)) if x.notna().any() else float("nan")
    hi = float(np.nanquantile(x.dropna().to_numpy(dtype=float), 0.90)) if x.notna().any() else float("nan")
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) <= 1e-12:
        return pd.Series(np.zeros(len(x), dtype=float), index=x.index)
    z = (x - lo) / (hi - lo)
    return z.clip(lower=0.0, upper=1.0).fillna(0.0)


def validate_contract(run_dir: Path, rep_csv: Path, fee_path: Path, metrics_path: Path, seed: int) -> Dict[str, Any]:
    for fp in (rep_csv, fee_path, metrics_path):
        if not fp.exists():
            raise FileNotFoundError(f"Missing locked input: {fp}")

    fee_sha = sha256_file(fee_path)
    met_sha = sha256_file(metrics_path)
    if fee_sha != LOCKED["expected_fee_sha"]:
        raise RuntimeError(f"Fee hash mismatch: {fee_sha} != {LOCKED['expected_fee_sha']}")
    if met_sha != LOCKED["expected_metrics_sha"]:
        raise RuntimeError(f"Metrics hash mismatch: {met_sha} != {LOCKED['expected_metrics_sha']}")

    sig_raw = pd.read_csv(rep_csv)
    sig = ae.ensure_signals_schema(sig_raw)
    if sig.empty:
        raise RuntimeError("Frozen subset empty after schema normalization")

    lock_args = ae.build_args(signals_csv=rep_csv, seed=int(seed))
    lock_args.allow_freeze_hash_mismatch = 0
    lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=lock_args, run_dir=run_dir)
    if int(lock_validation.get("freeze_lock_pass", 0)) != 1:
        raise RuntimeError("freeze_lock_pass != 1")

    return {
        "generated_utc": utc_now(),
        "subset_rows_raw": int(len(sig_raw)),
        "subset_rows_normalized": int(len(sig)),
        "fee_sha256": fee_sha,
        "metrics_sha256": met_sha,
        "freeze_lock_validation": lock_validation,
    }


def build_signal_feature_frame(rep_subset: pd.DataFrame, df_feat: pd.DataFrame) -> pd.DataFrame:
    rep_idx = pu.build_rep_subset_with_idx(rep_subset=rep_subset, df_feat=df_feat)
    feat_cols = [
        "RSI",
        "ADX",
        "WILLR",
        "ATR",
        "RV_24",
        "EMA_200_SLOPE",
        "MACD",
        "MACD_SIGNAL",
        "PLUS_DI",
        "MINUS_DI",
    ]
    out = rep_idx[["signal_id", "signal_time", "tp_mult", "sl_mult", "atr_percentile_1h", "trend_up_1h", "_full_idx"]].copy()
    n = len(out)
    for c in feat_cols:
        out[f"f1h_{c.lower()}"] = np.nan
    for i in range(n):
        idx = int(out.iloc[i]["_full_idx"])
        if 0 <= idx < len(df_feat):
            for c in feat_cols:
                out.at[i, f"f1h_{c.lower()}"] = float(to_num(pd.Series([df_feat.iloc[idx][c]])).iloc[0]) if c in df_feat.columns else float("nan")
    out["session_bucket"] = ae.session_bucket(pd.to_datetime(out["signal_time"], utc=True, errors="coerce"))
    out["vol_bucket"] = ae.vol_bucket(out["atr_percentile_1h"])
    out["trend_bucket"] = np.where(to_num(out["trend_up_1h"]) >= 0.5, "up", "down")

    # Legacy score proxy (old-style 1h ranking signal).
    out["legacy_score_raw"] = (
        0.30 * normalize_01(out["f1h_adx"])
        + 0.20 * (1.0 - normalize_01(out["f1h_rsi"]))
        + 0.20 * (1.0 - normalize_01(out["f1h_willr"]))
        + 0.15 * normalize_01(out["trend_up_1h"])
        + 0.15 * (1.0 - normalize_01(out["atr_percentile_1h"]))
    )
    out["legacy_score_raw"] = to_num(out["legacy_score_raw"]).clip(lower=0.0, upper=1.0)
    return out


def evaluate_g1_tables(
    run_dir: Path,
    rep_subset: pd.DataFrame,
    signal_feat: pd.DataFrame,
    exec_choices: Sequence[pu.ExecChoice],
    seed: int,
) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    eval_art_dir = run_dir / "phaseG1_eval_artifacts"
    eval_art_dir.mkdir(parents=True, exist_ok=True)
    route_sets = af.route_signal_sets(rep_subset)
    rid_list = sorted(route_sets.keys())
    eval_id = 0
    for rid in rid_list:
        sdf = route_sets[rid].copy().reset_index(drop=True)
        for choice in exec_choices:
            if choice.genome is None:
                continue
            eval_id += 1
            met, sig, _split, _args, bundle = ae.evaluate_exact(
                run_dir=eval_art_dir,
                signals_df=sdf,
                genome=copy.deepcopy(choice.genome),
                seed=int(seed) + eval_id,
                name=f"g1_{rid}_{choice.exec_choice_id}",
            )
            pre = ae.build_preentry_features(bundle, choice.genome)
            lbl = ae.build_trade_labels(sig.merge(pre, on=["signal_id", "signal_time"], how="left"))
            m = lbl.merge(
                signal_feat.drop(columns=["_full_idx"], errors="ignore"),
                on=["signal_id", "signal_time", "tp_mult", "sl_mult", "atr_percentile_1h", "trend_up_1h"],
                how="left",
            )
            m["route_id"] = rid
            m["exec_choice_id"] = choice.exec_choice_id
            m["exec_genome_hash"] = choice.genome_hash
            m = m.sort_values(["signal_time_utc", "signal_id"]).reset_index(drop=True)
            n = len(m)
            if n > 0:
                m["subperiod_id"] = np.clip(np.floor(np.linspace(0, 3, n, endpoint=False)).astype(int), 0, 2)
            else:
                m["subperiod_id"] = np.nan
            # Fragility markers.
            m["fragility_long_delay"] = (to_num(m["fill_delay_min"]) > 20.0).astype(int)
            m["fragility_taker"] = (to_num(m["taker_flag"]) >= 1).astype(int)
            m["fragility_fee_dominated"] = to_num(m["y_fee_dominated"]).fillna(0).astype(int)
            rows.append(m)
    if not rows:
        return pd.DataFrame()
    g1 = pd.concat(rows, axis=0, ignore_index=True)
    return g1


def g2_diagnostics(g1: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = g1.copy()
    valid = x[x["entry_for_labels"] == 1].copy()
    if valid.empty:
        raise RuntimeError("No valid entry rows in G1 table")

    # ranking diagnostics
    diag_rows: List[Dict[str, Any]] = []
    for (choice, rid), g in valid.groupby(["exec_choice_id", "route_id"], dropna=False):
        for metric in [
            "pnl_net_trade_notional_dec",
            "y_toxic_trade",
            "y_cluster_loss",
            "y_tail_loss",
            "fragility_long_delay",
            "fragility_taker",
        ]:
            s = spearman_no_scipy(g["legacy_score_raw"], g[metric])
            diag_rows.append(
                {
                    "scope": "route_choice",
                    "exec_choice_id": str(choice),
                    "route_id": str(rid),
                    "metric": metric,
                    "spearman_legacy_vs_metric": float(s),
                    "support_n": int(len(g)),
                }
            )
    for choice, g in valid.groupby("exec_choice_id", dropna=False):
        for metric in ["pnl_net_trade_notional_dec", "y_toxic_trade", "y_cluster_loss", "y_tail_loss"]:
            s = spearman_no_scipy(g["legacy_score_raw"], g[metric])
            diag_rows.append(
                {
                    "scope": "choice_all_routes",
                    "exec_choice_id": str(choice),
                    "route_id": "ALL",
                    "metric": metric,
                    "spearman_legacy_vs_metric": float(s),
                    "support_n": int(len(g)),
                }
            )
    diag_df = pd.DataFrame(diag_rows).sort_values(["scope", "exec_choice_id", "route_id", "metric"]).reset_index(drop=True)
    diag_df.to_csv(out_dir / "phaseG2_ranking_diagnostics.csv", index=False)

    # bucket tests
    bt_rows: List[Dict[str, Any]] = []
    for choice, g in valid.groupby("exec_choice_id", dropna=False):
        try:
            b = pd.qcut(to_num(g["legacy_score_raw"]), q=10, duplicates="drop")
            g2 = g.copy()
            g2["score_bucket"] = b.astype(str)
        except Exception:
            g2 = g.copy()
            g2["score_bucket"] = "all"
        for bk, z in g2.groupby("score_bucket", dropna=False):
            bt_rows.append(
                {
                    "exec_choice_id": str(choice),
                    "score_bucket": str(bk),
                    "support_n": int(len(z)),
                    "post_exec_expectancy": float(to_num(z["pnl_net_trade_notional_dec"]).mean()),
                    "toxic_rate": float(to_num(z["y_toxic_trade"]).mean()),
                    "cluster_rate": float(to_num(z["y_cluster_loss"]).mean()),
                    "tail_rate": float(to_num(z["y_tail_loss"]).mean()),
                    "avg_fill_delay_min": float(to_num(z["fill_delay_min"]).mean()),
                    "taker_rate": float(to_num(z["taker_flag"]).mean()),
                }
            )
    bt_df = pd.DataFrame(bt_rows).sort_values(["exec_choice_id", "score_bucket"]).reset_index(drop=True)
    bt_df.to_csv(out_dir / "phaseG2_bucket_tests.csv", index=False)

    # feature stability by split
    feats = [
        "legacy_score_raw",
        "f1h_rsi",
        "f1h_adx",
        "f1h_willr",
        "f1h_rv_24",
        "f1h_ema_200_slope",
        "atr_percentile_1h",
        "trend_up_1h",
    ]
    outcomes = ["pnl_net_trade_notional_dec", "y_toxic_trade", "y_cluster_loss", "y_tail_loss"]
    st_rows: List[Dict[str, Any]] = []
    for choice, g in valid.groupby("exec_choice_id", dropna=False):
        for f in feats:
            for y in outcomes:
                overall = spearman_no_scipy(g[f], g[y])
                signs: List[int] = []
                vals: List[float] = []
                for sid, h in g.groupby("split_id", dropna=True):
                    if len(h) < 20:
                        continue
                    s = spearman_no_scipy(h[f], h[y])
                    if np.isfinite(s):
                        vals.append(float(s))
                        signs.append(int(np.sign(s)))
                overall_sign = int(np.sign(overall)) if np.isfinite(overall) else 0
                if overall_sign == 0:
                    stable = float(np.mean([int(v > 0) for v in vals])) if vals else float("nan")
                else:
                    stable = float(np.mean([int(np.sign(v) == overall_sign) for v in vals])) if vals else float("nan")
                st_rows.append(
                    {
                        "exec_choice_id": str(choice),
                        "feature": f,
                        "outcome": y,
                        "overall_spearman": float(overall),
                        "split_count": int(len(vals)),
                        "stable_sign_frac": float(stable),
                        "split_mean_spearman": float(np.mean(vals)) if vals else float("nan"),
                        "split_std_spearman": float(np.std(vals, ddof=0)) if vals else float("nan"),
                    }
                )
    st_df = pd.DataFrame(st_rows).sort_values(["exec_choice_id", "feature", "outcome"]).reset_index(drop=True)
    st_df.to_csv(out_dir / "phaseG2_feature_stability.csv", index=False)

    # report
    top_diag = (
        diag_df[(diag_df["metric"] == "pnl_net_trade_notional_dec") & (diag_df["scope"] == "choice_all_routes")]
        .sort_values("spearman_legacy_vs_metric", ascending=False)
        .head(10)
    )
    lines = []
    lines.append("# Phase G2 Diagnostics Report")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append(f"- Valid rows used: `{len(valid)}`")
    lines.append("")
    lines.append("## Legacy Score Correlation vs Post-Exec PnL")
    lines.append("")
    lines.append(markdown_table(top_diag, ["exec_choice_id", "spearman_legacy_vs_metric", "support_n"]))
    lines.append("")
    lines.append("## Feature Stability (sample)")
    lines.append("")
    lines.append(markdown_table(st_df.head(20), ["exec_choice_id", "feature", "outcome", "overall_spearman", "stable_sign_frac", "split_count"]))
    write_text(out_dir / "phaseG2_diagnostics_report.md", "\n".join(lines))

    return diag_df, bt_df, st_df


def candidate_objective_scores(row: pd.Series) -> Dict[str, float]:
    # Prototypes are execution-aware and robustness-aware.
    de = float(row.get("delta_expectancy_vs_exec_baseline", np.nan))
    cvar = float(row.get("cvar_improve_ratio", np.nan))
    dd = float(row.get("maxdd_improve_ratio", np.nan))
    tox = float(row.get("avg_toxic_proxy", np.nan))
    clu = float(row.get("avg_cluster_proxy", np.nan))
    tail = float(row.get("avg_tail_proxy", np.nan))
    conc = float(row.get("dominant_regime_share", np.nan))
    min_split = float(row.get("min_split_expectancy_net", np.nan))
    entry = float(row.get("entry_rate", np.nan))

    # Guard NaNs.
    for v in [de, cvar, dd, tox, clu, tail, conc, min_split, entry]:
        if not np.isfinite(v):
            return {k: float("nan") for k in ["OJ1", "OJ2", "OJ3", "OJ4", "OJ5"]}

    oj1 = de - 0.0003 * tox - 0.0002 * clu
    oj2 = de + 0.40 * cvar + 0.30 * dd - 0.0002 * tail
    oj3 = oj2 - 0.08 * max(0.0, conc - 0.55)  # anti-concentration.
    oj4 = oj2 + 0.20 * min_split  # robust min-split proxy.
    oj5 = oj3 - 0.05 * max(0.0, 0.85 - entry)  # participation floor proxy.
    return {"OJ1": float(oj1), "OJ2": float(oj2), "OJ3": float(oj3), "OJ4": float(oj4), "OJ5": float(oj5)}


def g3_candidate_bench(
    run_dir: Path,
    rep_subset: pd.DataFrame,
    signal_feat: pd.DataFrame,
    g1_table: pd.DataFrame,
    seed: int,
    candidate_count: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[str, Any]]:
    out_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    exec_choices, _meta = pu.load_exec_choices(out_root)
    ch_map = {c.exec_choice_id: c for c in exec_choices}
    if "E1" not in ch_map or "E2" not in ch_map:
        raise RuntimeError("Required execution choices E1/E2 not found")
    eval_choices = [ch_map["E1"], ch_map["E2"]]

    # Base signal engine context.
    params_path, base_params_raw, params_meta = pu.load_active_sol_params()
    base_params = ga_long._norm_params(copy.deepcopy(base_params_raw))
    df1h = recon._load_symbol_df("SOLUSDT", tf="1h")
    df_feat = ga_long._ensure_indicators(df1h.copy(), base_params)
    df_feat = ga_long._prepare_signal_df(df_feat, assume_prepared=False)
    rep_idx = pu.build_rep_subset_with_idx(rep_subset=rep_subset, df_feat=df_feat)

    exec_args = pu.build_exec_args(signals_csv=Path(LOCKED["frozen_subset_csv"]), mode="tight", seed=int(seed))
    bundles, _load_meta = ga_exec._prepare_bundles(exec_args)
    if not bundles:
        raise RuntimeError("No bundle prepared for G3")
    base_bundle = bundles[0]

    # Candidate set.
    cands = pu.generate_1h_candidates(base_params=base_params, n_total=int(candidate_count), seed=int(seed))
    # Ensure base first (generator already does this).
    if not cands:
        raise RuntimeError("No candidates generated")

    # Build baseline metrics by choice from base candidate.
    base_active_ids, base_diag = pu.active_signal_ids_for_params(df_feat=df_feat, params=cands[0]["params"], rep_idx=rep_idx)
    base_bundle_sel = pu.build_candidate_bundle(base_bundle=base_bundle, active_ids=base_active_ids, args=exec_args)
    base_by_choice: Dict[str, Dict[str, Any]] = {}
    for ch in eval_choices:
        base_by_choice[ch.exec_choice_id] = pu.eval_exec_choice(bundle=base_bundle_sel, args=exec_args, choice=ch)

    # Signal-level proxy maps from G1.
    g1v = g1_table[g1_table["entry_for_labels"] == 1].copy()
    proxy_map = (
        g1v.groupby("signal_id", dropna=False)
        .agg(
            toxic_rate=("y_toxic_trade", "mean"),
            cluster_rate=("y_cluster_loss", "mean"),
            tail_rate=("y_tail_loss", "mean"),
        )
        .reset_index()
    )
    proxy_map["signal_id"] = proxy_map["signal_id"].astype(str)
    proxy_idx = proxy_map.set_index("signal_id")

    sf = signal_feat.copy()
    sf["signal_id"] = sf["signal_id"].astype(str)
    sf["regime_key"] = sf["vol_bucket"].astype(str) + "|" + sf["trend_bucket"].astype(str)

    rows: List[Dict[str, Any]] = []
    choice_rows: List[Dict[str, Any]] = []
    for cand in cands:
        cid = str(cand["signal_candidate_id"])
        active_ids, diag = pu.active_signal_ids_for_params(df_feat=df_feat, params=cand["params"], rep_idx=rep_idx)
        signal_sig = pu.sha256_text("|".join(sorted(active_ids)))

        # Proxies from selected signal set.
        if active_ids:
            sid = pd.Index(sorted(active_ids))
            px = proxy_idx.reindex(sid)
            avg_toxic = float(np.nanmean(to_num(px["toxic_rate"])))
            avg_cluster = float(np.nanmean(to_num(px["cluster_rate"])))
            avg_tail = float(np.nanmean(to_num(px["tail_rate"])))
            sub = sf[sf["signal_id"].isin(active_ids)].copy()
            if sub.empty:
                dom_session = float("nan")
                dom_regime = float("nan")
            else:
                dom_session = float(sub["session_bucket"].value_counts(normalize=True, dropna=False).iloc[0])
                dom_regime = float(sub["regime_key"].value_counts(normalize=True, dropna=False).iloc[0])

            # subperiod proxy from G1 signal outcomes on selected ids.
            gp = g1v[g1v["signal_id"].isin(active_ids)].copy()
            if gp.empty:
                min_subperiod_proxy = float("nan")
                subperiod_proxy_pass = 0
            else:
                sub_agg = gp.groupby(["exec_choice_id", "route_id", "subperiod_id"], dropna=False)["pnl_net_trade_notional_dec"].mean().reset_index()
                min_subperiod_proxy = float(to_num(sub_agg["pnl_net_trade_notional_dec"]).min()) if not sub_agg.empty else float("nan")
                subperiod_proxy_pass = int(np.isfinite(min_subperiod_proxy) and min_subperiod_proxy > 0.0)
        else:
            avg_toxic = float("nan")
            avg_cluster = float("nan")
            avg_tail = float("nan")
            dom_session = float("nan")
            dom_regime = float("nan")
            min_subperiod_proxy = float("nan")
            subperiod_proxy_pass = 0

        bundle = pu.build_candidate_bundle(base_bundle=base_bundle, active_ids=active_ids, args=exec_args)
        if len(bundle.contexts) == 0:
            row = {
                "candidate_id": cid,
                "candidate_name": str(cand["name"]),
                "candidate_kind": str(cand["kind"]),
                "param_hash": str(cand["param_hash"]),
                "signal_signature": signal_sig,
                "signals_active": int(len(active_ids)),
                "active_rate_vs_rep": float(diag["active_rate_vs_rep"]),
                "valid_for_ranking": 0,
                "exec_expectancy_net": float("nan"),
                "delta_expectancy_vs_exec_baseline": float("nan"),
                "cvar_improve_ratio": float("nan"),
                "maxdd_improve_ratio": float("nan"),
                "min_split_expectancy_net": float("nan"),
                "entries_valid": 0,
                "entry_rate": 0.0,
                "taker_share": float("nan"),
                "p95_fill_delay_min": float("nan"),
                "route_subperiod_proxy_pass": int(subperiod_proxy_pass),
                "min_subperiod_expectancy_proxy": float(min_subperiod_proxy),
                "avg_toxic_proxy": float(avg_toxic),
                "avg_cluster_proxy": float(avg_cluster),
                "avg_tail_proxy": float(avg_tail),
                "dominant_session_share": float(dom_session),
                "dominant_regime_share": float(dom_regime),
                "invalid_reason": "no_signals",
            }
            rows.append(row)
            continue

        choice_metrics: List[Dict[str, Any]] = []
        invalid: List[str] = []
        for ch in eval_choices:
            met = pu.eval_exec_choice(bundle=bundle, args=exec_args, choice=ch)
            b = base_by_choice[ch.exec_choice_id]
            de = float(met.get("overall_exec_expectancy_net", np.nan) - b.get("overall_exec_expectancy_net", np.nan))
            cvar_imp = safe_div(
                abs(float(b.get("overall_exec_cvar_5", np.nan))) - abs(float(met.get("overall_exec_cvar_5", np.nan))),
                abs(float(b.get("overall_exec_cvar_5", np.nan))),
            )
            dd_imp = safe_div(
                abs(float(b.get("overall_exec_max_drawdown", np.nan))) - abs(float(met.get("overall_exec_max_drawdown", np.nan))),
                abs(float(b.get("overall_exec_max_drawdown", np.nan))),
            )
            cm = {
                "candidate_id": cid,
                "exec_choice_id": ch.exec_choice_id,
                "overall_exec_expectancy_net": float(met.get("overall_exec_expectancy_net", np.nan)),
                "delta_expectancy_vs_exec_baseline": de,
                "cvar_improve_ratio": float(cvar_imp),
                "maxdd_improve_ratio": float(dd_imp),
                "entries_valid": int(met.get("overall_entries_valid", 0)),
                "entry_rate": float(met.get("overall_entry_rate", np.nan)),
                "taker_share": float(met.get("overall_exec_taker_share", np.nan)),
                "p95_fill_delay_min": float(met.get("overall_exec_p95_fill_delay_min", np.nan)),
                "min_split_expectancy_net": float(met.get("min_split_expectancy_net", np.nan)),
                "valid_for_ranking": int(met.get("valid_for_ranking", 0)),
                "invalid_reason": str(met.get("invalid_reason", "")),
            }
            if int(cm["valid_for_ranking"]) != 1 and cm["invalid_reason"]:
                invalid.append(cm["invalid_reason"])
            choice_metrics.append(cm)
            choice_rows.append(cm)

        cm_df = pd.DataFrame(choice_metrics)
        row = {
            "candidate_id": cid,
            "candidate_name": str(cand["name"]),
            "candidate_kind": str(cand["kind"]),
            "param_hash": str(cand["param_hash"]),
            "signal_signature": signal_sig,
            "signals_active": int(len(active_ids)),
            "active_rate_vs_rep": float(diag["active_rate_vs_rep"]),
            "valid_for_ranking": int((to_num(cm_df["valid_for_ranking"]) == 1).all()),
            "exec_expectancy_net": float(to_num(cm_df["overall_exec_expectancy_net"]).mean()),
            "delta_expectancy_vs_exec_baseline": float(to_num(cm_df["delta_expectancy_vs_exec_baseline"]).mean()),
            "cvar_improve_ratio": float(to_num(cm_df["cvar_improve_ratio"]).mean()),
            "maxdd_improve_ratio": float(to_num(cm_df["maxdd_improve_ratio"]).mean()),
            "min_split_expectancy_net": float(to_num(cm_df["min_split_expectancy_net"]).min()),
            "entries_valid": int(to_num(cm_df["entries_valid"]).min()),
            "entry_rate": float(to_num(cm_df["entry_rate"]).min()),
            "taker_share": float(to_num(cm_df["taker_share"]).max()),
            "p95_fill_delay_min": float(to_num(cm_df["p95_fill_delay_min"]).max()),
            "route_subperiod_proxy_pass": int(subperiod_proxy_pass),
            "min_subperiod_expectancy_proxy": float(min_subperiod_proxy),
            "avg_toxic_proxy": float(avg_toxic),
            "avg_cluster_proxy": float(avg_cluster),
            "avg_tail_proxy": float(avg_tail),
            "dominant_session_share": float(dom_session),
            "dominant_regime_share": float(dom_regime),
            "invalid_reason": "|".join(sorted(set([x for x in invalid if x]))),
        }
        row.update(candidate_objective_scores(pd.Series(row)))
        rows.append(row)

    res = pd.DataFrame(rows).sort_values(["valid_for_ranking", "OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False, False]).reset_index(drop=True)
    choice_df = pd.DataFrame(choice_rows).sort_values(["candidate_id", "exec_choice_id"]).reset_index(drop=True)

    # invalid histogram
    invalid_hist: Dict[str, int] = {}
    for _, r in res.iterrows():
        if int(r["valid_for_ranking"]) == 1:
            continue
        rs = str(r.get("invalid_reason", "")).strip()
        if not rs:
            invalid_hist["unknown_invalid"] = int(invalid_hist.get("unknown_invalid", 0) + 1)
            continue
        for part in rs.split("|"):
            part = part.strip()
            if not part:
                continue
            invalid_hist[part] = int(invalid_hist.get(part, 0) + 1)

    # top-K per objective.
    top_rows: List[Dict[str, Any]] = []
    for obj in ["OJ1", "OJ2", "OJ3", "OJ4", "OJ5"]:
        z = res.copy()
        z = z.sort_values(["valid_for_ranking", obj, "delta_expectancy_vs_exec_baseline"], ascending=[False, False, False]).head(5)
        for _, r in z.iterrows():
            top_rows.append(
                {
                    "objective_id": obj,
                    "rank_in_objective": int(len([x for x in top_rows if x.get("objective_id") == obj]) + 1),
                    "candidate_id": str(r["candidate_id"]),
                    "candidate_name": str(r["candidate_name"]),
                    "valid_for_ranking": int(r["valid_for_ranking"]),
                    "objective_score": float(r[obj]),
                    "exec_expectancy_net": float(r["exec_expectancy_net"]),
                    "delta_expectancy_vs_exec_baseline": float(r["delta_expectancy_vs_exec_baseline"]),
                    "cvar_improve_ratio": float(r["cvar_improve_ratio"]),
                    "maxdd_improve_ratio": float(r["maxdd_improve_ratio"]),
                    "min_split_expectancy_net": float(r["min_split_expectancy_net"]),
                    "entry_rate": float(r["entry_rate"]),
                    "route_subperiod_proxy_pass": int(r["route_subperiod_proxy_pass"]),
                    "invalid_reason": str(r["invalid_reason"]),
                }
            )
    topk = pd.DataFrame(top_rows)

    # candidate-level ranking diagnostics for G2 from legacy 1h objective vs exec metrics
    # Build legacy candidate score via 1h backtest only.
    cand_diag_rows: List[Dict[str, Any]] = []
    for cand in cands:
        cid = str(cand["signal_candidate_id"])
        try:
            _tr, m = ga_long.run_backtest_long_only(
                df_feat,
                symbol="SOLUSDT",
                p=cand["params"],
                initial_equity=1000.0,
                fee_bps=2.0,
                slippage_bps=2.0,
                collect_trades=False,
                assume_prepared=True,
            )
            legacy_score = float(m.get("net_profit", np.nan) - 0.4 * abs(float(m.get("max_dd", np.nan))))
        except Exception:
            legacy_score = float("nan")
        cand_diag_rows.append({"candidate_id": cid, "legacy_1h_score": legacy_score})
    cand_diag = pd.DataFrame(cand_diag_rows)
    res = res.merge(cand_diag, on="candidate_id", how="left")
    if not topk.empty:
        topk = topk.merge(res[["candidate_id", "legacy_1h_score"]], on="candidate_id", how="left")

    return res, topk, invalid_hist, {"base_candidate_id": cands[0]["signal_candidate_id"], "params_path": str(params_path), "params_meta": params_meta}


def g4_decision(res: pd.DataFrame, topk: pd.DataFrame, out_dir: Path, bench_meta: Dict[str, Any]) -> Tuple[str, str, Optional[str]]:
    # identify base row
    base_id = str(bench_meta.get("base_candidate_id", "P00"))
    base = res[res["candidate_id"] == base_id].copy()
    if base.empty:
        base = res.head(1).copy()
    b = base.iloc[0]

    z = res[res["candidate_id"] != str(b["candidate_id"])].copy()
    pass_like = z[
        (to_num(z["valid_for_ranking"]) == 1)
        & (to_num(z["delta_expectancy_vs_exec_baseline"]) >= 5e-5)
        & (to_num(z["cvar_improve_ratio"]) >= 0.0)
        & (to_num(z["maxdd_improve_ratio"]) >= 0.0)
        & (to_num(z["min_subperiod_expectancy_proxy"]) >= float(to_num(pd.Series([b["min_subperiod_expectancy_proxy"]])).iloc[0]) - 2e-5)
        & (to_num(z["entry_rate"]) >= 0.75 * float(b["entry_rate"]))
        & (to_num(z["entries_valid"]) >= 0.75 * float(b["entries_valid"]))
    ].copy()

    if pass_like.empty:
        cls = "STOP_NO_GO"
        reason = "Prototype bench showed no material additive signal-side gain with non-degrading robustness proxies."
        prompt = None
    else:
        cls = "CONTINUE_READY_FOR_PHASEH"
        reason = "At least one execution-aware objective produced material additive gain with non-degrading robustness proxies."
        prompt = (
            "ROLE\n"
            "You are in Phase H execution-aware 1h GA pilot mode.\n\n"
            "MISSION\n"
            "Run a bounded execution-aware 1h GA (no full marathon) around top Phase G objective prototypes and candidate neighborhoods.\n\n"
            "RULES\n"
            "1) Hard gates unchanged.\n"
            "2) Keep frozen execution contract lock (E1/E2 scoring mandatory).\n"
            "3) Candidate budget bounded (e.g., 128x4 equivalent).\n"
            "4) Stop NO_GO if additive signal-side gains collapse under split/route robustness checks.\n"
            "5) Output full duplicate-adjusted significance and strict decision pack."
        )
        write_text(out_dir / "ready_to_launch_phaseH_execaware_1h_ga_prompt.txt", prompt)

    lines = []
    lines.append("# Phase G Decision")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append(f"- Classification: **{cls}**")
    lines.append(f"- Reason: {reason}")
    lines.append(f"- Base candidate id: `{b['candidate_id']}`")
    lines.append(f"- Base metrics: delta=`{float(b['delta_expectancy_vs_exec_baseline']):.8f}`, cvar_imp=`{float(b['cvar_improve_ratio']):.6f}`, maxdd_imp=`{float(b['maxdd_improve_ratio']):.6f}`")
    lines.append(f"- Candidates meeting GO rule: `{len(pass_like)}`")
    if len(pass_like):
        lines.append("")
        lines.append("## GO-Qualified Candidates")
        lines.append("")
        lines.append(markdown_table(pass_like.head(10), ["candidate_id", "candidate_name", "delta_expectancy_vs_exec_baseline", "cvar_improve_ratio", "maxdd_improve_ratio", "entry_rate", "min_subperiod_expectancy_proxy"]))
    write_text(out_dir / "phaseG_decision_next_step.md", "\n".join(lines))
    return cls, reason, prompt


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase G upstream objective redesign (execution-aware 1h, no full GA)")
    ap.add_argument("--seed", type=int, default=20260223)
    ap.add_argument("--candidate-count", type=int, default=24)
    args = ap.parse_args()

    out_root = (PROJECT_ROOT / "reports" / "execution_layer").resolve()
    run_dir = out_root / f"PHASEG_EXECAWARE_OBJECTIVE_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()

    rep_csv = Path(LOCKED["frozen_subset_csv"]).resolve()
    fee_path = Path(LOCKED["canonical_fee_model"]).resolve()
    met_path = Path(LOCKED["canonical_metrics_definition"]).resolve()

    try:
        lock_obj = validate_contract(run_dir=run_dir, rep_csv=rep_csv, fee_path=fee_path, metrics_path=met_path, seed=int(args.seed))
    except Exception as e:
        write_text(run_dir / "phaseG_infra_fail.md", f"Contract/data validation failed: {e}")
        manifest = {
            "furthest_phase": "G0",
            "classification": "STOP_INFRA",
            "mainline_status": "STOP_INFRA",
            "reason": str(e),
            "run_dir": str(run_dir),
        }
        json_dump(run_dir / "phaseG_run_manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True))
        return
    json_dump(run_dir / "phaseG_freeze_lock_validation.json", lock_obj)

    # Load context
    rep_subset = ae.ensure_signals_schema(pd.read_csv(rep_csv))
    params_path, base_params_raw, params_meta = pu.load_active_sol_params()
    base_params = ga_long._norm_params(copy.deepcopy(base_params_raw))
    df1h = recon._load_symbol_df("SOLUSDT", tf="1h")
    df_feat = ga_long._ensure_indicators(df1h.copy(), base_params)
    df_feat = ga_long._prepare_signal_df(df_feat, assume_prepared=False)
    signal_feat = build_signal_feature_frame(rep_subset=rep_subset, df_feat=df_feat)

    # G1
    exec_choices, _exec_meta = pu.load_exec_choices(out_root)
    e1e2 = [c for c in exec_choices if c.exec_choice_id in {"E1", "E2"}]
    g1 = evaluate_g1_tables(run_dir=run_dir, rep_subset=rep_subset, signal_feat=signal_feat, exec_choices=e1e2, seed=int(args.seed))
    if g1.empty:
        write_text(run_dir / "phaseG_infra_fail.md", "G1 execution-aware table is empty.")
        manifest = {
            "furthest_phase": "G1",
            "classification": "STOP_INFRA",
            "mainline_status": "STOP_INFRA",
            "reason": "empty G1 table",
            "run_dir": str(run_dir),
        }
        json_dump(run_dir / "phaseG_run_manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True))
        return
    g1.to_parquet(run_dir / "phaseG1_execaware_signal_table.parquet", index=False)

    write_text(
        run_dir / "phaseG1_label_dictionary.md",
        "\n".join(
            [
                "# Phase G1 Label Dictionary",
                "",
                "- `signal_id`, `signal_time`: 1h signal identity/time from frozen representative subset.",
                "- `exec_choice_id`: execution policy (`E1` or `E2`) used for downstream labeling.",
                "- `route_id`: route partition (`route1_holdout`, `route2_reslice`).",
                "- `subperiod_id`: chronological thirds within route for robustness slicing.",
                "- `entry_for_labels`: 1 if execution outcome valid for label metrics.",
                "- `pnl_net_trade_notional_dec`: downstream executed net return.",
                "- `y_toxic_trade`: AE composite toxic label (tail OR cluster OR large adverse excursion).",
                "- `y_cluster_loss`, `y_tail_loss`, `y_sl_loss`: cluster/tail/SL-specific labels.",
                "- `prior_loss_streak_len`, `prior_rolling_tail_count_20`, `prior_rolling_loss_rate_5`: prior-only context labels.",
                "- `fragility_long_delay`, `fragility_taker`, `fragility_fee_dominated`: execution fragility markers.",
                "- `legacy_score_raw`: current/legacy 1h rank proxy from 1h indicators.",
                "- `f1h_*`: baseline 1h indicator metadata aligned at signal time.",
            ]
        ),
    )

    dq = g1.copy()
    dq_rows = []
    for c in [
        "pnl_net_trade_notional_dec",
        "fill_delay_min",
        "f1h_rsi",
        "f1h_adx",
        "f1h_willr",
        "legacy_score_raw",
        "y_toxic_trade",
        "y_cluster_loss",
        "y_tail_loss",
    ]:
        dq_rows.append(
            {
                "column": c,
                "missing_rate": float(1.0 - np.mean(to_num(dq[c]).notna() if c != "legacy_score_raw" else dq[c].notna())),
                "support_n": int(len(dq)),
            }
        )
    dq_df = pd.DataFrame(dq_rows)
    write_text(
        run_dir / "phaseG1_data_quality_report.md",
        "\n".join(
            [
                "# Phase G1 Data Quality Report",
                "",
                f"- Generated UTC: {utc_now()}",
                f"- Rows: `{len(g1)}`",
                f"- Entry-for-label rows: `{int(to_num(g1['entry_for_labels']).sum())}`",
                f"- Unique signals: `{g1['signal_id'].nunique()}`",
                f"- Routes: `{sorted(g1['route_id'].dropna().unique().tolist())}`",
                f"- Exec choices: `{sorted(g1['exec_choice_id'].dropna().unique().tolist())}`",
                "",
                "## Missingness Snapshot",
                "",
                markdown_table(dq_df, ["column", "missing_rate", "support_n"]),
            ]
        ),
    )

    # G2
    diag_df, bt_df, st_df = g2_diagnostics(g1=g1, out_dir=run_dir)

    # If diagnostics show no useful relationship, stop no-go.
    usable_diag = diag_df[(diag_df["metric"] == "pnl_net_trade_notional_dec") & (diag_df["scope"] == "choice_all_routes")].copy()
    best_abs_corr = float(np.nanmax(np.abs(to_num(usable_diag["spearman_legacy_vs_metric"]).to_numpy(dtype=float)))) if not usable_diag.empty else float("nan")
    if (not np.isfinite(best_abs_corr)) or best_abs_corr < 0.02:
        write_text(
            run_dir / "phaseG_decision_next_step.md",
            "\n".join(
                [
                    "# Phase G Decision",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    "- Classification: **STOP_NO_GO**",
                    "- Mainline status: **STOP_NO_GO**",
                    "- Reason: ranking diagnostics show no useful signal-feature relationship with downstream execution outcomes.",
                ]
            ),
        )
        manifest = {
            "generated_utc": utc_now(),
            "run_dir": str(run_dir),
            "furthest_phase": "G2",
            "classification": "STOP_NO_GO",
            "mainline_status": "STOP_NO_GO",
            "reason": f"best_abs_corr={best_abs_corr}",
            "duration_sec": float(time.time() - t0),
        }
        json_dump(run_dir / "phaseG_run_manifest.json", manifest)
        print(json.dumps(manifest, sort_keys=True))
        return

    # G3
    res, topk, invalid_hist, bench_meta = g3_candidate_bench(
        run_dir=run_dir,
        rep_subset=rep_subset,
        signal_feat=signal_feat,
        g1_table=g1,
        seed=int(args.seed),
        candidate_count=int(args.candidate_count),
    )
    res.to_csv(run_dir / "phaseG3_candidate_results.csv", index=False)
    topk.to_csv(run_dir / "phaseG3_topk_by_objective.csv", index=False)
    json_dump(run_dir / "phaseG3_invalid_reason_histogram.json", invalid_hist)

    write_text(
        run_dir / "phaseG3_objective_prototypes.md",
        "\n".join(
            [
                "# Phase G3 Objective Prototypes",
                "",
                "All objectives are execution-aware and use E1/E2 candidate metrics with robustness proxies.",
                "",
                "- `OJ1 = delta_expectancy_vs_exec_baseline - 0.0003*avg_toxic_proxy - 0.0002*avg_cluster_proxy`",
                "- `OJ2 = delta_expectancy_vs_exec_baseline + 0.40*cvar_improve_ratio + 0.30*maxdd_improve_ratio - 0.0002*avg_tail_proxy`",
                "- `OJ3 = OJ2 - 0.08*max(0, dominant_regime_share-0.55)` (anti-concentration penalty)",
                "- `OJ4 = OJ2 + 0.20*min_split_expectancy_net` (robust min-split emphasis)",
                "- `OJ5 = OJ3 - 0.05*max(0, 0.85-entry_rate)` (participation floor penalty)",
                "",
                "Scoring/ranking is prototype-only in this phase; no GA search was launched.",
            ]
        ),
    )

    lines = []
    lines.append("# Phase G3 Bench Report")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append(f"- Candidate count: `{len(res)}`")
    lines.append(f"- valid_for_ranking count: `{int((to_num(res['valid_for_ranking']) == 1).sum())}`")
    lines.append("")
    lines.append("## Top candidates by OJ2")
    lines.append("")
    z = res.sort_values(["valid_for_ranking", "OJ2", "delta_expectancy_vs_exec_baseline"], ascending=[False, False, False]).head(10)
    lines.append(
        markdown_table(
            z,
            [
                "candidate_id",
                "candidate_name",
                "valid_for_ranking",
                "OJ2",
                "exec_expectancy_net",
                "delta_expectancy_vs_exec_baseline",
                "cvar_improve_ratio",
                "maxdd_improve_ratio",
                "min_split_expectancy_net",
                "entry_rate",
                "invalid_reason",
            ],
        )
    )
    write_text(run_dir / "phaseG3_bench_report.md", "\n".join(lines))

    # G4 decision
    cls, reason, prompt = g4_decision(res=res, topk=topk, out_dir=run_dir, bench_meta=bench_meta)
    mainline_status = cls
    manifest = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "classification": cls,
        "mainline_status": mainline_status,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "source_phase_ae": LOCKED["phase_ae_dir"],
        "source_phase_f": LOCKED["phase_f_dir"],
    }
    json_dump(run_dir / "phaseG_run_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
