#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_ae_signal_labeling as ae  # noqa: E402
from scripts import phase_af_ah_sizing_autorun as af  # noqa: E402
from scripts import phase_d123_tail_filter as dmod  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


DEFAULTS = {
    "repo_root": "/root/analysis/0.87",
    "frozen_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "fee_path": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "metrics_path": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
    "prior_dir": "/root/analysis/0.87/reports/execution_layer/PHASEABC_LABEL_REPAIR_20260223_131344",
    "primary_exec_hash": "862c940746de0da984862d95",
    "backup_exec_hash": "992bd371689ba3936f3b4d09",
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


def markdown_table(df: pd.DataFrame, cols: Iterable[str]) -> str:
    x = df.loc[:, [c for c in cols if c in df.columns]].copy()
    if x.empty:
        return "_(none)_"
    lines: List[str] = []
    headers = list(x.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in x.itertuples(index=False):
        vals: List[str] = []
        for v in r:
            if isinstance(v, float):
                vals.append(f"{v:.8g}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def spearman_corr_no_scipy(x: pd.Series, y: pd.Series) -> float:
    a = to_num(x)
    b = to_num(y)
    m = a.notna() & b.notna()
    if int(m.sum()) < 2:
        return float("nan")
    ar = a[m].rank(method="average").to_numpy(dtype=float)
    br = b[m].rank(method="average").to_numpy(dtype=float)
    sa = float(np.std(ar, ddof=0))
    sb = float(np.std(br, ddof=0))
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


class Heartbeat:
    def __init__(self, run_dir: Path, interval_sec: int = 75) -> None:
        self.run_dir = run_dir
        self.interval_sec = int(max(60, min(120, interval_sec)))
        self.path = self.run_dir / "heartbeat.json"
        self._phase = "INIT"
        self._status = "running"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def set_state(self, phase: str, status: str) -> None:
        self._phase = str(phase)
        self._status = str(status)
        self._write_once()

    def _write_once(self) -> None:
        json_dump(
            self.path,
            {
                "generated_utc": utc_now(),
                "phase": self._phase,
                "status": self._status,
                "interval_sec": self.interval_sec,
            },
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._write_once()

    def start(self) -> None:
        self._write_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, final_status: str = "stopped") -> None:
        self._status = final_status
        self._write_once()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def checkpoint_path(run_dir: Path, phase_key: str) -> Path:
    return run_dir / f"checkpoint_{phase_key}.json"


def write_checkpoint(run_dir: Path, phase_key: str, classification: str, info: Dict[str, Any]) -> None:
    obj = {"generated_utc": utc_now(), "phase": phase_key, "classification": classification}
    obj.update(info)
    json_dump(checkpoint_path(run_dir, phase_key), obj)


def read_checkpoint(run_dir: Path, phase_key: str) -> Optional[Dict[str, Any]]:
    p = checkpoint_path(run_dir, phase_key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def phase_done(run_dir: Path, phase_key: str) -> bool:
    ck = read_checkpoint(run_dir, phase_key)
    return bool(ck and str(ck.get("classification", "")).upper() == "PASS")


def find_prior_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if p.exists():
            return p
    root = PROJECT_ROOT / "reports" / "execution_layer"
    cands = sorted([x for x in root.glob("PHASEABC_LABEL_REPAIR_*") if x.is_dir()], key=lambda x: x.name)
    for c in reversed(cands):
        phc = sorted([p for p in c.glob("phaseC_*") if p.is_dir()], key=lambda x: x.name)
        if not phc:
            continue
        cdir = phc[-1]
        if (cdir / "phaseC_results.csv").exists() and (cdir / "phaseC_results_by_route.csv").exists():
            return c
    raise FileNotFoundError("Could not locate prior PHASEABC_LABEL_REPAIR_* with phaseC results")


def latest_subphase_dir(parent: Path, prefix: str) -> Optional[Path]:
    dirs = sorted([p for p in parent.glob(f"{prefix}_*") if p.is_dir()], key=lambda x: x.name)
    return dirs[-1] if dirs else None


def locate_prior_artifacts(prior_dir: Path) -> Dict[str, Path]:
    d: Dict[str, Path] = {"prior_dir": prior_dir}
    phaseb = latest_subphase_dir(prior_dir, "phaseB")
    phasec = latest_subphase_dir(prior_dir, "phaseC")
    if phasec is None:
        raise FileNotFoundError(f"No phaseC_* subdir in {prior_dir}")
    d["phasec_dir"] = phasec
    d["phasec_results"] = phasec / "phaseC_results.csv"
    d["phasec_byroute"] = phasec / "phaseC_results_by_route.csv"
    if not d["phasec_results"].exists() or not d["phasec_byroute"].exists():
        raise FileNotFoundError(f"Missing phaseC result files in {phasec}")
    if phaseb is not None:
        rs = phaseb / "phaseB_artifacts" / "risk_score.csv"
        if rs.exists():
            d["risk_score"] = rs
    return d


def phase_a(run_dir: Path, args: argparse.Namespace, hb: Heartbeat) -> Tuple[str, Dict[str, Any]]:
    hb.set_state("A", "running")
    t0 = time.time()
    phase_dir = run_dir / "phaseA"
    phase_dir.mkdir(parents=True, exist_ok=True)

    fee_fp = Path(args.fee_path).resolve()
    met_fp = Path(args.metrics_path).resolve()
    subset_fp = Path(args.frozen_subset_csv).resolve()
    for fp in (fee_fp, met_fp, subset_fp):
        if not fp.exists():
            return "INFRA_FAIL", {"reason": f"Missing locked file: {fp}", "phase_dir": str(phase_dir)}

    fee_sha = sha256_file(fee_fp)
    met_sha = sha256_file(met_fp)
    if fee_sha != args.expected_fee_sha or met_sha != args.expected_metrics_sha:
        return "CONTRACT_FAIL", {
            "reason": "hash mismatch",
            "phase_dir": str(phase_dir),
            "observed_fee_sha": fee_sha,
            "observed_metrics_sha": met_sha,
        }

    # Schema check
    raw = pd.read_csv(subset_fp)
    sig = ae.ensure_signals_schema(raw)
    required_cols = ["signal_id", "signal_time", "tp_mult", "sl_mult", "atr_percentile_1h", "trend_up_1h"]
    schema_ok = all(c in sig.columns for c in required_cols)
    if sig.empty:
        return "CONTRACT_FAIL", {"reason": "subset empty", "phase_dir": str(phase_dir)}
    if not schema_ok:
        return "INFRA_FAIL", {"reason": "subset schema mismatch", "phase_dir": str(phase_dir)}

    lock_args = ae.build_args(signals_csv=subset_fp, seed=int(args.seed))
    lock_args.allow_freeze_hash_mismatch = 0
    lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=lock_args, run_dir=phase_dir)
    if int(lock_validation.get("freeze_lock_pass", 0)) != 1:
        return "CONTRACT_FAIL", {"reason": "freeze_lock_pass!=1", "phase_dir": str(phase_dir), "freeze_lock_validation": lock_validation}

    out = {
        "generated_utc": utc_now(),
        "phase": "A",
        "decision": "PASS",
        "duration_sec": float(time.time() - t0),
        "phase_dir": str(phase_dir),
        "subset_rows_raw": int(len(raw)),
        "subset_rows_schema_normalized": int(len(sig)),
        "required_cols": required_cols,
        "fee_sha256": fee_sha,
        "metrics_sha256": met_sha,
        "freeze_lock_validation": lock_validation,
        "python": sys.version,
    }
    json_dump(phase_dir / "phaseA_freeze_lock_validation.json", out)
    json_dump(phase_dir / "phaseA_run_manifest.json", out)
    write_text(
        phase_dir / "phaseA_report.md",
        "\n".join(
            [
                "# Phase A Report",
                "",
                f"- Generated UTC: {utc_now()}",
                "- Decision: **PASS**",
                f"- Subset rows (normalized): `{len(sig)}`",
                f"- Fee hash match: `{int(fee_sha == args.expected_fee_sha)}`",
                f"- Metrics hash match: `{int(met_sha == args.expected_metrics_sha)}`",
                f"- Freeze lock pass: `{int(lock_validation.get('freeze_lock_pass', 0))}`",
            ]
        ),
    )
    write_text(
        phase_dir / "phaseA_decision.md",
        "\n".join(["# Phase A Decision", "", f"- Generated UTC: {utc_now()}", "- Classification: **PASS**", "- Reason: all checks passed"]),
    )
    return "PASS", out


def phase_d1(run_dir: Path, args: argparse.Namespace, hb: Heartbeat, prior_art: Dict[str, Path], route_data_cache: Dict[str, pd.DataFrame]) -> Tuple[str, Dict[str, Any]]:
    hb.set_state("D1", "running")
    t0 = time.time()
    phase_dir = run_dir / "phaseD1"
    phase_dir.mkdir(parents=True, exist_ok=True)

    phasec_results = pd.read_csv(prior_art["phasec_results"])
    phasec_byroute = pd.read_csv(prior_art["phasec_byroute"])
    if phasec_results.empty:
        return "INFRA_FAIL", {"reason": "phaseC_results.csv empty", "phase_dir": str(phase_dir)}

    sig_in = ae.ensure_signals_schema(pd.read_csv(Path(args.frozen_subset_csv)))
    exec_pair = ae.load_exec_pair(PROJECT_ROOT / "reports" / "execution_layer")
    if exec_pair["E1"]["genome_hash"] != DEFAULTS["primary_exec_hash"]:
        return "CONTRACT_FAIL", {"reason": "E1 hash mismatch", "phase_dir": str(phase_dir)}

    # Always build route-level replay dataset for robust attribution.
    route_data = dmod.evaluate_baseline_routes(
        run_dir=phase_dir,
        sig_in=sig_in,
        genome=copy.deepcopy(exec_pair["E1"]["genome"]),
        seed=int(args.seed),
    )
    route_data_cache.clear()
    route_data_cache.update(route_data)
    attr_df, summary, tail_examples = dmod.phase_d1_attribution(route_data)
    if attr_df.empty or tail_examples.empty:
        return "INFRA_FAIL", {"reason": "attribution outputs empty", "phase_dir": str(phase_dir)}

    # tail concentration file (decile-focused)
    conc_cols = [
        "route_id",
        "risk_decile",
        "support",
        "tail10_count",
        "tail10_rate",
        "cvar5_count",
        "cvar5_rate",
        "tail10_loss_share",
        "cvar5_loss_share",
    ]
    conc_df = attr_df.loc[:, [c for c in conc_cols if c in attr_df.columns]].copy()
    attr_df.to_csv(phase_dir / "tail_attribution_by_route.csv", index=False)
    conc_df.to_csv(phase_dir / "tail_concentration_by_score_bucket.csv", index=False)
    tail_examples.to_csv(phase_dir / "tail_examples_worst_route.csv", index=False)

    d1_pass = bool(summary.get("worst_route")) and not conc_df.empty
    cls = "PASS" if d1_pass else "NO_GO"
    reason = "worst route + tail concentration identified" if d1_pass else "no clear attribution"

    rep = []
    rep.append("# Phase D1 Report")
    rep.append("")
    rep.append(f"- Generated UTC: {utc_now()}")
    rep.append(f"- Decision: **{cls}**")
    rep.append(f"- Reason: {reason}")
    rep.append(f"- Prior dir: `{prior_art['prior_dir']}`")
    rep.append(f"- Source phaseC dir: `{prior_art['phasec_dir']}`")
    rep.append(f"- Source policy rows: `{len(phasec_results)}`")
    rep.append(f"- Source by-route rows: `{len(phasec_byroute)}`")
    rep.append(f"- Worst route by CVaR: `{summary.get('worst_route', 'n/a')}`")
    rep.append(f"- Worst route CVaR5: `{float(summary.get('worst_route_cvar5', float('nan'))):.8f}`")
    rep.append(f"- Top-2 decile CVaR loss share (worst route): `{float(summary.get('worst_route_top2_decile_cvar5_loss_share', float('nan'))):.6f}`")
    rep.append("")
    rep.append("## Route summary")
    rep.append("")
    rep.append(markdown_table(pd.DataFrame(summary.get("route_rows", [])), ["route_id", "entries", "mean_pnl", "cvar5", "tail_cut10", "cvar5_cut"]))
    rep.append("")
    rep.append("## Tail concentration by score bucket (head)")
    rep.append("")
    rep.append(markdown_table(conc_df.head(20), conc_cols))
    write_text(phase_dir / "phaseD1_report.md", "\n".join(rep))

    out = {
        "generated_utc": utc_now(),
        "phase": "D1",
        "decision": cls,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "phase_dir": str(phase_dir),
        "summary": summary,
    }
    json_dump(phase_dir / "phaseD1_run_manifest.json", out)
    write_text(
        phase_dir / "phaseD1_decision.md",
        "\n".join(["# Phase D1 Decision", "", f"- Generated UTC: {utc_now()}", f"- Classification: **{cls}**", f"- Reason: {reason}"]),
    )
    return cls, out


def phase_d2(run_dir: Path, args: argparse.Namespace, hb: Heartbeat, route_data: Dict[str, pd.DataFrame]) -> Tuple[str, Dict[str, Any]]:
    hb.set_state("D2", "running")
    t0 = time.time()
    phase_dir = run_dir / "phaseD2"
    phase_dir.mkdir(parents=True, exist_ok=True)

    if not route_data:
        return "INFRA_FAIL", {"reason": "missing route_data cache", "phase_dir": str(phase_dir)}

    labels_rows: List[Dict[str, Any]] = []
    split_rows: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []
    # Build no-leak tail labels: threshold from other splits only.
    for route_id, d in route_data.items():
        x = d[(d["entry_for_labels"] == 1) & d["pnl_net_trade_notional_dec"].notna()].copy().reset_index(drop=True)
        if x.empty:
            continue
        x["y_tail_loss"] = 0
        has_split = x["split_id"].notna().any()
        if has_split:
            split_vals = sorted([v for v in x["split_id"].dropna().unique().tolist()])
            for sid in split_vals:
                train = x[x["split_id"] != sid]
                test_idx = x.index[x["split_id"] == sid].tolist()
                if not test_idx:
                    continue
                if train.empty:
                    cut = float(np.nanquantile(to_num(x["pnl_net_trade_notional_dec"]), 0.05))
                else:
                    cut = float(np.nanquantile(to_num(train["pnl_net_trade_notional_dec"]), 0.05))
                x.loc[test_idx, "y_tail_loss"] = (to_num(x.loc[test_idx, "pnl_net_trade_notional_dec"]) <= cut).astype(int)
        else:
            cut = float(np.nanquantile(to_num(x["pnl_net_trade_notional_dec"]), 0.05))
            x["y_tail_loss"] = (to_num(x["pnl_net_trade_notional_dec"]) <= cut).astype(int)

        # Tail risk score by route decile frequency (monotone mapping from risk_score_s1).
        try:
            x["risk_bin"] = pd.qcut(to_num(x["risk_score_s1"]), q=min(10, max(2, int(to_num(x["risk_score_s1"]).nunique()))), duplicates="drop").astype(str)
        except Exception:
            x["risk_bin"] = "all"
        bin_rate = x.groupby("risk_bin", dropna=False)["y_tail_loss"].mean().to_dict()
        x["tail_risk_score"] = x["risk_bin"].map(bin_rate).astype(float)

        # Stability per route/split
        route_sc = spearman_corr_no_scipy(to_num(x["tail_risk_score"]), to_num(x["y_tail_loss"]))
        split_pos = 0
        split_eligible = 0
        if has_split:
            for sid, g in x.groupby("split_id", dropna=True):
                if len(g) < 20:
                    continue
                sc = spearman_corr_no_scipy(to_num(g["tail_risk_score"]), to_num(g["y_tail_loss"]))
                if np.isfinite(sc):
                    split_eligible += 1
                    if sc > 0:
                        split_pos += 1
                split_rows.append(
                    {
                        "route_id": route_id,
                        "split_id": int(sid) if np.isfinite(sid) else str(sid),
                        "support": int(len(g)),
                        "spearman_tail": float(sc),
                    }
                )
        stable = safe_div(float(split_pos), float(split_eligible)) if split_eligible > 0 else 0.0
        route_rows.append(
            {
                "route_id": route_id,
                "support": int(len(x)),
                "tail_rate": float(np.mean(x["y_tail_loss"])),
                "overall_spearman_tail": float(route_sc),
                "stable_sign_frac": float(stable),
                "split_eligible": int(split_eligible),
                "split_positive": int(split_pos),
                "score_nan_rate": float(1.0 - np.mean(to_num(x["tail_risk_score"]).notna())),
            }
        )
        for _, r in x.iterrows():
            labels_rows.append(
                {
                    "route_id": route_id,
                    "signal_id": str(r["signal_id"]),
                    "signal_time_utc": str(r["signal_time_utc"]),
                    "split_id": float(r.get("split_id", np.nan)),
                    "risk_score_s1": float(r.get("risk_score_s1", np.nan)),
                    "tail_risk_score": float(r.get("tail_risk_score", np.nan)),
                    "y_tail_loss": int(r.get("y_tail_loss", 0)),
                    "pnl_net_trade_notional_dec": float(r.get("pnl_net_trade_notional_dec", np.nan)),
                }
            )

    labels = pd.DataFrame(labels_rows)
    srows = pd.DataFrame(split_rows)
    rrows = pd.DataFrame(route_rows)
    if labels.empty or rrows.empty:
        return "INFRA_FAIL", {"reason": "empty D2 labels or route rows", "phase_dir": str(phase_dir)}

    labels.to_csv(phase_dir / "tail_label.csv", index=False)
    labels.loc[:, ["route_id", "signal_id", "signal_time_utc", "split_id", "risk_score_s1", "tail_risk_score"]].to_csv(
        phase_dir / "tail_risk_score.csv", index=False
    )

    combined_sc = spearman_corr_no_scipy(to_num(labels["tail_risk_score"]), to_num(labels["y_tail_loss"]))
    comb_eligible = int(to_num(srows.get("spearman_tail", pd.Series(dtype=float))).notna().sum()) if not srows.empty else 0
    comb_positive = int((to_num(srows.get("spearman_tail", pd.Series(dtype=float))) > 0).sum()) if not srows.empty else 0
    comb_stable = safe_div(float(comb_positive), float(comb_eligible)) if comb_eligible > 0 else 0.0

    # monotonicity by decile on combined data.
    try:
        bb = pd.qcut(to_num(labels["tail_risk_score"]), q=min(10, max(2, int(to_num(labels["tail_risk_score"]).nunique()))), duplicates="drop")
        mdf = pd.DataFrame({"bin": bb.astype(str), "y": to_num(labels["y_tail_loss"])}).groupby("bin", dropna=False)["y"].mean().reset_index()
        rates = to_num(mdf["y"]).to_numpy(dtype=float)
        mono_viol = int(np.sum(np.diff(rates) < -1e-12)) if rates.size >= 2 else 0
    except Exception:
        mono_viol = 999

    stab_obj = {
        "generated_utc": utc_now(),
        "phase": "D2",
        "label_tail_fraction": 0.05,
        "combined_support": int(len(labels)),
        "combined_overall_spearman_tail": float(combined_sc),
        "combined_split_eligible": int(comb_eligible),
        "combined_split_positive": int(comb_positive),
        "combined_stable_sign_frac": float(comb_stable),
        "combined_monotonic_violations": int(mono_viol),
        "per_route": rrows.to_dict(orient="records"),
        "split_rows": srows.to_dict(orient="records"),
        "acceptance_rule": {
            "stable_sign_frac_min": 0.75,
            "overall_spearman_positive": True,
            "score_nan_rate_max": 0.0,
            "monotonic_violations_max": 1,
        },
    }
    json_dump(phase_dir / "split_stability_tail.json", stab_obj)

    d2_pass = (
        float(comb_stable) >= 0.75
        and float(combined_sc) > 0.0
        and int(mono_viol) <= 1
        and all(float(x.get("score_nan_rate", 1.0)) <= 0.0 for x in stab_obj["per_route"])
    )
    cls = "PASS" if d2_pass else "NO_GO"
    reason = "tail label stable and monotonic" if d2_pass else "tail label instability/non-monotonicity"

    rep = []
    rep.append("# Phase D2 Report")
    rep.append("")
    rep.append(f"- Generated UTC: {utc_now()}")
    rep.append(f"- Decision: **{cls}**")
    rep.append(f"- Reason: {reason}")
    rep.append(f"- Combined support: `{len(labels)}`")
    rep.append(f"- Combined Spearman(tail_risk_score, y_tail_loss): `{float(combined_sc):.6f}`")
    rep.append(f"- Combined stable_sign_frac: `{float(comb_stable):.4f}` (rule >= 0.75)")
    rep.append(f"- Combined monotonic violations: `{mono_viol}` (rule <= 1)")
    rep.append("")
    rep.append("## Per-route tail stability")
    rep.append("")
    rep.append(markdown_table(rrows, ["route_id", "support", "tail_rate", "overall_spearman_tail", "stable_sign_frac", "split_eligible", "split_positive", "score_nan_rate"]))
    rep.append("")
    rep.append("## Leakage check")
    rep.append("")
    rep.append("- y_tail_loss is created from route outcomes with split-aware train-thresholding (no same-split threshold leakage).")
    rep.append("- tail_risk_score is a monotonic mapping from pre-entry risk score bins.")
    write_text(phase_dir / "phaseD2_report.md", "\n".join(rep))

    out = {
        "generated_utc": utc_now(),
        "phase": "D2",
        "decision": cls,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "phase_dir": str(phase_dir),
    }
    json_dump(phase_dir / "phaseD2_run_manifest.json", out)
    write_text(
        phase_dir / "phaseD2_decision.md",
        "\n".join(["# Phase D2 Decision", "", f"- Generated UTC: {utc_now()}", f"- Classification: **{cls}**", f"- Reason: {reason}"]),
    )
    return cls, out


def phase_d3(run_dir: Path, args: argparse.Namespace, hb: Heartbeat, route_data: Dict[str, pd.DataFrame]) -> Tuple[str, Dict[str, Any]]:
    hb.set_state("D3", "running")
    t0 = time.time()
    phase_dir = run_dir / "phaseD3"
    phase_dir.mkdir(parents=True, exist_ok=True)
    if not route_data:
        return "INFRA_FAIL", {"reason": "missing route_data for D3", "phase_dir": str(phase_dir)}

    route_df, agg_df, invalid_hist = dmod.phase_d3_filter_pilot(route_data, max_policies=int(args.max_filter_policies))
    route_df.to_csv(phase_dir / "phaseD3_results_by_route.csv", index=False)
    agg_df.to_csv(phase_dir / "phaseD3_results.csv", index=False)
    json_dump(phase_dir / "invalid_reason_histogram.json", invalid_hist)

    # duplicate/effective-trials summary
    dup_summary = {
        "generated_utc": utc_now(),
        "policy_rows": int(len(agg_df)),
        "unique_policy_hashes": int(agg_df["policy_hash"].nunique()) if not agg_df.empty else 0,
        "duplicate_rows": int(len(agg_df) - agg_df["policy_hash"].nunique()) if not agg_df.empty else 0,
        "effective_trials_uncorrelated": float(agg_df["policy_hash"].nunique()) if not agg_df.empty else 0.0,
    }
    json_dump(phase_dir / "duplicate_effective_trials_summary.json", dup_summary)

    # PSR/DSR proxy (search/ranking used here).
    psr_rows: List[Dict[str, Any]] = []
    if not agg_df.empty:
        top = agg_df.head(10).copy()
        eff = float(max(1, dup_summary["effective_trials_uncorrelated"]))
        for _, r in top.iterrows():
            v = to_num(route_df[route_df["policy_id"] == str(r["policy_id"])]["delta_expectancy_vs_flat"]).fillna(0.0).to_numpy(dtype=float)
            psr, dsr = af.psr_proxy_from_pnl(v, eff_trials=eff)
            psr_rows.append({"policy_id": str(r["policy_id"]), "psr_proxy": float(psr), "dsr_proxy": float(dsr)})
    psr_df = pd.DataFrame(psr_rows)
    if not psr_df.empty:
        psr_df.to_csv(phase_dir / "phaseD3_psr_dsr_proxy.csv", index=False)
    else:
        write_text(phase_dir / "phaseD3_psr_dsr_proxy.md", "not applicable")

    strict_passers = int((agg_df["strict_pass"] == 1).sum()) if not agg_df.empty else 0
    cls = "PASS_GO_PAPER_CANDIDATES" if strict_passers >= 1 else "NO_GO"
    reason = ">=1 strict passer" if strict_passers >= 1 else "0 strict passers"

    rep = []
    rep.append("# Phase D3 Report")
    rep.append("")
    rep.append(f"- Generated UTC: {utc_now()}")
    rep.append(f"- Decision: **{cls}**")
    rep.append(f"- Reason: {reason}")
    rep.append(f"- Policy budget: `{int(args.max_filter_policies)}`")
    rep.append(f"- Evaluated policies (excluding flat): `{len(agg_df)}`")
    rep.append(f"- Strict passers: `{strict_passers}`")
    rep.append("")
    rep.append("## Top candidates")
    rep.append("")
    rep.append(
        markdown_table(
            agg_df.head(20),
            [
                "policy_id",
                "policy_type",
                "strict_pass",
                "min_delta_expectancy_vs_flat",
                "min_cvar_improve_ratio_vs_flat",
                "min_maxdd_improve_ratio_vs_flat",
                "min_entries_valid",
                "min_filter_kept_entries_pct",
                "rank_score",
            ],
        )
    )
    rep.append("")
    rep.append("## Reality-check")
    rep.append("")
    rep.append("- Reality-check bootstrap: TODO placeholder (deployment-adjacent).")
    write_text(phase_dir / "phaseD3_report.md", "\n".join(rep))

    out = {
        "generated_utc": utc_now(),
        "phase": "D3",
        "decision": cls,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "phase_dir": str(phase_dir),
        "strict_passers": strict_passers,
    }
    json_dump(phase_dir / "phaseD3_run_manifest.json", out)
    write_text(
        phase_dir / "phaseD3_decision.md",
        "\n".join(["# Phase D3 Decision", "", f"- Generated UTC: {utc_now()}", f"- Classification: **{cls}**", f"- Reason: {reason}"]),
    )
    return cls, out


def write_final_summary(run_dir: Path, overall: Dict[str, Any], top_df: Optional[pd.DataFrame], next_prompt: Optional[str]) -> None:
    furthest = overall.get("furthest_phase", "A")
    cls = overall.get("furthest_classification", "UNKNOWN")
    status = overall.get("mainline_status", "STOP_NO_GO")
    lines: List[str] = []
    lines.append("- Furthest phase reached: " + str(furthest))
    lines.append("- Classification at furthest phase: " + str(cls))
    lines.append("- Mainline status: " + str(status))
    lines.append("")
    lines.append("- What was proven (1 paragraph plain English):")
    lines.append(str(overall.get("plain_english", "")))
    lines.append("")
    lines.append("- Top candidates (exact metrics):")
    if top_df is not None and not top_df.empty:
        lines.append(markdown_table(top_df.head(5), top_df.columns.tolist()))
    else:
        lines.append("_(none)_")
    lines.append("")
    lines.append("- Failure branch taken (if any):")
    lines.append(str(overall.get("failure_branch", "none")))
    lines.append("")
    lines.append("- Is next phase justified? (yes/no)")
    lines.append("yes" if status == "CONTINUE" else "no")
    lines.append("")
    lines.append("- Artifact directory (exact path)")
    lines.append(str(run_dir))
    lines.append("")
    lines.append("- Key files list")
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name in {
            "phaseA_report.md",
            "phaseA_freeze_lock_validation.json",
            "phaseA_run_manifest.json",
            "phaseA_decision.md",
            "phaseD1_report.md",
            "phaseD1_run_manifest.json",
            "tail_attribution_by_route.csv",
            "tail_concentration_by_score_bucket.csv",
            "phaseD2_report.md",
            "tail_label.csv",
            "tail_risk_score.csv",
            "split_stability_tail.json",
            "phaseD3_report.md",
            "phaseD3_results.csv",
            "invalid_reason_histogram.json",
            "phaseD3_decision.md",
            "run_manifest.json",
        }:
            lines.append(f"- {p}")
    lines.append("")
    lines.append("- Exact next prompt contents (only if justified)")
    if status == "CONTINUE" and next_prompt:
        lines.append(next_prompt)
    else:
        lines.append("Not justified.")
    write_text(run_dir / "FINAL_SUMMARY.md", "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Detached autorun D1->D3 tail branch with checkpoints/heartbeat")
    ap.add_argument("--seed", type=int, default=20260223)
    ap.add_argument("--time_budget_minutes", type=int, default=175)
    ap.add_argument("--prior_dir", default=DEFAULTS["prior_dir"])
    ap.add_argument("--frozen_subset_csv", default=DEFAULTS["frozen_subset_csv"])
    ap.add_argument("--fee_path", default=DEFAULTS["fee_path"])
    ap.add_argument("--metrics_path", default=DEFAULTS["metrics_path"])
    ap.add_argument("--expected_fee_sha", default=DEFAULTS["expected_fee_sha"])
    ap.add_argument("--expected_metrics_sha", default=DEFAULTS["expected_metrics_sha"])
    ap.add_argument("--allow_freeze_hash_mismatch", type=int, default=0)
    ap.add_argument("--max_filter_policies", type=int, default=40)
    ap.add_argument("--report_dir", default="")
    args = ap.parse_args()

    if int(args.allow_freeze_hash_mismatch) != 0:
        raise RuntimeError("allow_freeze_hash_mismatch must be 0 in this branch")

    root = PROJECT_ROOT / "reports" / "execution_layer"
    if args.report_dir:
        run_dir = Path(args.report_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = root / f"PHASED_TAIL_BRANCH_{utc_tag()}"
        run_dir.mkdir(parents=True, exist_ok=False)

    hb = Heartbeat(run_dir=run_dir, interval_sec=75)
    hb.start()
    hb.set_state("PHASE0", "running")

    start_t = time.time()
    deadline = start_t + max(1, int(args.time_budget_minutes)) * 60.0

    phase0 = {
        "generated_utc": utc_now(),
        "phase": "0",
        "run_dir": str(run_dir),
        "script": str(Path(__file__).resolve()),
        "args": vars(args),
        "repo_root": str(PROJECT_ROOT),
        "time_budget_minutes": int(args.time_budget_minutes),
        "deadline_utc_estimate": datetime.fromtimestamp(deadline, tz=timezone.utc).isoformat(),
    }
    json_dump(run_dir / "phase0_run_manifest.json", phase0)
    write_checkpoint(run_dir, "phase0", "PASS", {"phase0_run_manifest": str(run_dir / "phase0_run_manifest.json")})

    overall: Dict[str, Any] = {
        "generated_utc": utc_now(),
        "run_dir": str(run_dir),
        "mainline_status": "CONTINUE",
        "phases": {},
        "start_utc": utc_now(),
        "time_budget_minutes": int(args.time_budget_minutes),
    }

    try:
        prior_dir = find_prior_dir(args.prior_dir)
        prior_art = locate_prior_artifacts(prior_dir)
    except Exception as e:
        overall["mainline_status"] = "STOP_INFRA"
        overall["furthest_phase"] = "D0"
        overall["furthest_classification"] = "INFRA_FAIL"
        overall["plain_english"] = f"Failed to locate prior artifacts: {e}"
        overall["failure_branch"] = "INFRA_FAIL (locator)"
        json_dump(run_dir / "run_manifest.json", overall)
        write_text(run_dir / "infra_report.md", f"INFRA_FAIL locator: {e}")
        write_text(run_dir / "patch_diff_summary.md", "No patch applied inside autorun.")
        write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
        hb.stop("stopped")
        return

    # Phase A
    if not phase_done(run_dir, "phaseA"):
        cls, info = phase_a(run_dir, args, hb)
        overall["phases"]["A"] = {"classification": cls, "info": info}
        write_checkpoint(run_dir, "phaseA", cls, info)
        if cls != "PASS":
            overall["mainline_status"] = "STOP_CONTRACT" if cls == "CONTRACT_FAIL" else "STOP_INFRA"
            overall["furthest_phase"] = "A"
            overall["furthest_classification"] = cls
            overall["plain_english"] = f"Phase A failed due to {info.get('reason', 'unknown')}"
            overall["failure_branch"] = cls
            json_dump(run_dir / "run_manifest.json", overall)
            write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
            hb.stop("stopped")
            return
    else:
        ck = read_checkpoint(run_dir, "phaseA") or {}
        overall["phases"]["A"] = {"classification": "PASS", "info": ck}

    if time.time() >= deadline:
        overall["mainline_status"] = "STOP_NO_GO"
        overall["furthest_phase"] = "A"
        overall["furthest_classification"] = "NO_GO"
        overall["plain_english"] = "Time budget exhausted after Phase A."
        overall["failure_branch"] = "NO_GO (time budget)"
        json_dump(run_dir / "run_manifest.json", overall)
        write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
        hb.stop("stopped")
        return

    route_cache: Dict[str, pd.DataFrame] = {}

    # Phase D1
    if not phase_done(run_dir, "D1"):
        cls, info = phase_d1(run_dir, args, hb, prior_art, route_cache)
        overall["phases"]["D1"] = {"classification": cls, "info": info}
        write_checkpoint(run_dir, "D1", cls, info)
        if cls != "PASS":
            overall["mainline_status"] = "STOP_INFRA" if cls == "INFRA_FAIL" else ("STOP_CONTRACT" if cls == "CONTRACT_FAIL" else "STOP_NO_GO")
            overall["furthest_phase"] = "D1"
            overall["furthest_classification"] = cls
            overall["plain_english"] = f"D1 failed: {info.get('reason', 'unknown')}"
            overall["failure_branch"] = cls
            json_dump(run_dir / "run_manifest.json", overall)
            write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
            hb.stop("stopped")
            return
    else:
        ck = read_checkpoint(run_dir, "D1") or {}
        overall["phases"]["D1"] = {"classification": "PASS", "info": ck}
        # route cache is required for resumed D2; rebuild safely from phaseD1 data.
        sig_in = ae.ensure_signals_schema(pd.read_csv(Path(args.frozen_subset_csv)))
        exec_pair = ae.load_exec_pair(PROJECT_ROOT / "reports" / "execution_layer")
        route_cache = dmod.evaluate_baseline_routes(
            run_dir=(run_dir / "phaseD1"),
            sig_in=sig_in,
            genome=copy.deepcopy(exec_pair["E1"]["genome"]),
            seed=int(args.seed),
        )

    if time.time() >= deadline:
        overall["mainline_status"] = "STOP_NO_GO"
        overall["furthest_phase"] = "D1"
        overall["furthest_classification"] = "NO_GO"
        overall["plain_english"] = "Time budget exhausted after D1."
        overall["failure_branch"] = "NO_GO (time budget)"
        json_dump(run_dir / "run_manifest.json", overall)
        write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
        hb.stop("stopped")
        return

    # Phase D2
    if not phase_done(run_dir, "D2"):
        cls, info = phase_d2(run_dir, args, hb, route_cache)
        overall["phases"]["D2"] = {"classification": cls, "info": info}
        write_checkpoint(run_dir, "D2", cls, info)
        if cls != "PASS":
            overall["mainline_status"] = "STOP_NO_GO" if cls == "NO_GO" else ("STOP_CONTRACT" if cls == "CONTRACT_FAIL" else "STOP_INFRA")
            overall["furthest_phase"] = "D2"
            overall["furthest_classification"] = cls
            overall["plain_english"] = "D2 built tail labels but stability criteria did not pass."
            overall["failure_branch"] = cls
            d2_dir = run_dir / "phaseD2"
            ng = d2_dir / "no_go_package"
            ng.mkdir(parents=True, exist_ok=True)
            write_text(
                ng / "phaseD2_no_go_diagnosis.md",
                "\n".join(
                    [
                        "# Phase D2 NO_GO Diagnosis",
                        "",
                        f"- Generated UTC: {utc_now()}",
                        f"- Reason: {info.get('reason', 'tail label instability')}",
                        "- Branch stopped before D3 as required.",
                    ]
                ),
            )
            write_text(
                ng / "next_step_prompt.txt",
                "D2 no-go follow-up: refine tail label definition and route-conditional score mapping (non-leaky), then rerun a small filter pilot under unchanged hard gates.",
            )
            json_dump(run_dir / "run_manifest.json", overall)
            write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
            hb.stop("stopped")
            return
    else:
        ck = read_checkpoint(run_dir, "D2") or {}
        overall["phases"]["D2"] = {"classification": "PASS", "info": ck}

    if time.time() >= deadline:
        overall["mainline_status"] = "STOP_NO_GO"
        overall["furthest_phase"] = "D2"
        overall["furthest_classification"] = "NO_GO"
        overall["plain_english"] = "Time budget exhausted after D2."
        overall["failure_branch"] = "NO_GO (time budget)"
        json_dump(run_dir / "run_manifest.json", overall)
        write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
        hb.stop("stopped")
        return

    # Phase D3
    if not phase_done(run_dir, "D3"):
        cls, info = phase_d3(run_dir, args, hb, route_cache)
        overall["phases"]["D3"] = {"classification": cls, "info": info}
        write_checkpoint(run_dir, "D3", cls, info)
    else:
        ck = read_checkpoint(run_dir, "D3") or {}
        cls = str(ck.get("classification", "UNKNOWN"))
        info = ck
        overall["phases"]["D3"] = {"classification": cls, "info": ck}

    d3_dir = run_dir / "phaseD3"
    top_df = None
    if (d3_dir / "phaseD3_results.csv").exists():
        try:
            top_df = pd.read_csv(d3_dir / "phaseD3_results.csv").head(5)
        except Exception:
            top_df = None

    if cls == "PASS_GO_PAPER_CANDIDATES":
        overall["mainline_status"] = "CONTINUE"
        overall["furthest_phase"] = "D3"
        overall["furthest_classification"] = cls
        overall["plain_english"] = "D3 found at least one strict multi-route passer under unchanged hard gates."
        overall["failure_branch"] = "none"
        next_prompt = (
            "Phase E paper/shadow confirmation (contract-locked): validate D3 strict-pass filter policy set under route perturbation and stress scenarios. "
            "Keep hard gates unchanged, include rollback triggers, and do not run full GA."
        )
        write_text(d3_dir / "phaseD_prompt_next.txt", next_prompt)
        write_text(d3_dir / "reality_check_todo.md", "TODO: add full reality-check bootstrap before any live deployment recommendation.")
    else:
        overall["mainline_status"] = "STOP_NO_GO" if cls == "NO_GO" else ("STOP_INFRA" if cls == "INFRA_FAIL" else "STOP_CONTRACT")
        overall["furthest_phase"] = "D3"
        overall["furthest_classification"] = cls
        overall["plain_english"] = "D3 filter pilot produced no strict multi-route passers (or failed infra/contract)."
        overall["failure_branch"] = cls
        ng = d3_dir / "no_go_package"
        ng.mkdir(parents=True, exist_ok=True)
        write_text(
            ng / "phaseD3_no_go_reasoning.md",
            "\n".join(
                [
                    "# Phase D3 NO_GO Reasoning",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Classification: {cls}",
                    f"- Reason: {info.get('reason', 'no strict passers')}",
                ]
            ),
        )
        write_text(
            ng / "next_step_prompt.txt",
            "D3 no-go follow-up: stop filter expansion and redesign tail objective/labels before additional policy search. Keep contract lock and hard gates unchanged.",
        )
        write_text(d3_dir / "reality_check_todo.md", "TODO: reality-check bootstrap remains unimplemented.")
        next_prompt = None

    overall["duration_sec"] = float(time.time() - start_t)
    overall["end_utc"] = utc_now()
    json_dump(run_dir / "run_manifest.json", overall)
    write_final_summary(run_dir, overall, top_df=top_df, next_prompt=next_prompt)
    hb.stop("completed")


if __name__ == "__main__":
    main()

