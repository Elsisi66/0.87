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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("BOT087_PROJECT_ROOT", str(PROJECT_ROOT))

from scripts import phase_ae_signal_labeling as ae  # noqa: E402
from scripts import phase_d123_tail_filter as dmod  # noqa: E402
from src.execution import ga_exec_3m_opt as ga_exec  # noqa: E402


DEFAULTS = {
    "repo_root": "/root/analysis/0.87",
    "phaseD_dir": "/root/analysis/0.87/reports/execution_layer/PHASED_TAIL_BRANCH_20260223_134324",
    "frozen_subset_csv": "/root/analysis/0.87/reports/execution_layer/PHASEE2_SOL_REPRESENTATIVE_20260222_021052/representative_subset_signals.csv",
    "fee_path": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/fee_model.json",
    "metrics_path": "/root/analysis/0.87/reports/execution_layer/BASELINE_AUDIT_20260221_214310/metrics_definition.md",
    "expected_fee_sha": "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a",
    "expected_metrics_sha": "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99",
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
        if isinstance(v, (datetime, pd.Timestamp)):
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
    headers = list(x.columns)
    lines: List[str] = []
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
    if not ck:
        return False
    cls = str(ck.get("classification", "")).upper()
    return cls in {"PASS", "WEAK_PASS", "PASS_GO_PAPER_SHADOW"}


def find_phase_d_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if p.exists():
            return p
    root = PROJECT_ROOT / "reports" / "execution_layer"
    cands = sorted([p for p in root.glob("PHASED_TAIL_BRANCH_*") if p.is_dir()], key=lambda x: x.name)
    for c in reversed(cands):
        if (c / "phaseD3" / "phaseD3_results.csv").exists():
            return c
    raise FileNotFoundError("No PHASED_TAIL_BRANCH_* with phaseD3 results found")


def load_phase_d_candidates(
    phase_d_dir: Path,
    *,
    candidates_json: str = "",
    include_near_miss: bool = True,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], pd.DataFrame]:
    d3 = phase_d_dir / "phaseD3" / "phaseD3_results.csv"
    if not d3.exists():
        raise FileNotFoundError(f"Missing phaseD3 results: {d3}")
    d3df = pd.read_csv(d3)
    if d3df.empty:
        raise RuntimeError("phaseD3_results.csv is empty")

    # Build policy lookup from generator to recover exact params.
    lookup: Dict[str, Dict[str, Any]] = {}
    for p in dmod.build_filter_policies(max_policies=256):
        key_id = str(p.get("policy_id", ""))
        key_hash = dmod.policy_hash(p)
        lookup[f"id::{key_id}"] = p
        lookup[f"hash::{key_hash}"] = p

    selected: List[Dict[str, Any]] = []
    if candidates_json:
        raw = json.loads(Path(candidates_json).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise RuntimeError("candidates_json must be a list")
        for obj in raw:
            pid = str(obj.get("policy_id", ""))
            ph = str(obj.get("policy_hash", ""))
            pol = lookup.get(f"hash::{ph}") or lookup.get(f"id::{pid}")
            if pol is None:
                raise RuntimeError(f"Policy not reconstructable from id/hash: {pid}/{ph}")
            selected.append(pol)
    else:
        s = d3df[d3df["strict_pass"] == 1].copy()
        s = s.sort_values(["rank_score", "min_delta_expectancy_vs_flat"], ascending=[False, False]).head(3)
        for _, r in s.iterrows():
            pid = str(r["policy_id"])
            ph = str(r["policy_hash"])
            pol = lookup.get(f"hash::{ph}") or lookup.get(f"id::{pid}")
            if pol is None:
                raise RuntimeError(f"Strict-passer policy unresolved: {pid}/{ph}")
            selected.append(pol)

    # Ensure mandatory 3 are included (if available).
    must_ids = ["skip_streak5_cool60m", "skip_streak4_cool60m", "skip_risk_ge_0.70"]
    for mid in must_ids:
        if all(str(p.get("policy_id")) != mid for p in selected) and f"id::{mid}" in lookup:
            selected.append(lookup[f"id::{mid}"])

    # Deduplicate by hash.
    uniq: Dict[str, Dict[str, Any]] = {}
    for p in selected:
        uniq[dmod.policy_hash(p)] = p
    selected = list(uniq.values())

    near = None
    if include_near_miss:
        nm = d3df[d3df["strict_pass"] == 0].copy()
        if not nm.empty:
            nm = nm.sort_values(["rank_score"], ascending=[False]).head(1)
            r = nm.iloc[0]
            pid = str(r["policy_id"])
            ph = str(r["policy_hash"])
            pol = lookup.get(f"hash::{ph}") or lookup.get(f"id::{pid}")
            if pol is not None:
                near = pol
    return selected, near, d3df


def load_locked_route_data(
    *,
    phase_dir: Path,
    frozen_subset_csv: Path,
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    sig_in = ae.ensure_signals_schema(pd.read_csv(frozen_subset_csv))
    exec_pair = ae.load_exec_pair(PROJECT_ROOT / "reports" / "execution_layer")
    if exec_pair["E1"]["genome_hash"] != DEFAULTS["primary_exec_hash"]:
        raise RuntimeError("Locked E1 hash mismatch")
    if exec_pair["E2"]["genome_hash"] != DEFAULTS["backup_exec_hash"]:
        raise RuntimeError("Locked E2 hash mismatch")
    data = dmod.evaluate_baseline_routes(run_dir=phase_dir, sig_in=sig_in, genome=copy.deepcopy(exec_pair["E1"]["genome"]), seed=int(seed))
    return data, exec_pair


def apply_scenario_to_route_df(route_df: pd.DataFrame, scenario: Dict[str, Any]) -> pd.DataFrame:
    x = route_df.copy().reset_index(drop=True)
    x["signal_time_utc"] = pd.to_datetime(x["signal_time_utc"], utc=True, errors="coerce")
    trim_head = float(scenario.get("trim_head_pct", 0.0))
    trim_tail = float(scenario.get("trim_tail_pct", 0.0))
    if trim_head > 0 or trim_tail > 0:
        x = x.sort_values(["signal_time_utc", "signal_id"]).reset_index(drop=True)
        n = len(x)
        lo = int(math.floor(n * trim_head))
        hi = n - int(math.floor(n * trim_tail))
        if hi <= lo:
            x = x.iloc[0:0].copy()
        else:
            x = x.iloc[lo:hi].copy().reset_index(drop=True)

    if x.empty:
        x["pnl_scenario"] = np.nan
        x["fill_delay_scenario"] = np.nan
        return x

    valid = (x["entry_for_labels"] == 1) & to_num(x["pnl_net_trade_notional_dec"]).notna()
    net = to_num(x["pnl_net_trade_notional_dec"]).fillna(0.0)
    fee_drag = to_num(x.get("fee_drag_trade", pd.Series(np.zeros(len(x), dtype=float)))).fillna(0.0)
    gross = to_num(x.get("pnl_gross_trade_notional_dec", net + fee_drag)).fillna(net + fee_drag)

    maker_mult = float(scenario.get("maker_fee_mult", 1.0))
    taker_mult = float(scenario.get("taker_fee_mult", 1.0))
    cost_mult = float(scenario.get("cost_multiplier", 1.0))
    taker_flag = (to_num(x.get("taker_flag", 0)).fillna(0.0) >= 0.5).astype(int)
    fee_mult_vec = np.where(taker_flag.to_numpy(dtype=int) == 1, taker_mult, maker_mult)
    fee_adj = fee_drag.to_numpy(dtype=float) * fee_mult_vec * cost_mult
    net_adj = gross.to_numpy(dtype=float) - fee_adj

    extra_slip_bps = float(scenario.get("extra_slippage_bps", 0.0))
    if extra_slip_bps != 0.0:
        net_adj = net_adj - (extra_slip_bps / 1e4)

    entry_delay_bars = float(scenario.get("entry_delay_bars", 0.0))
    exit_delay_bars = float(scenario.get("exit_delay_bars", 0.0))
    lat_pen_bps = float(scenario.get("latency_penalty_bps_per_bar", 1.0))
    latency_bars_total = entry_delay_bars + exit_delay_bars
    if latency_bars_total > 0.0:
        net_adj = net_adj - (latency_bars_total * lat_pen_bps / 1e4)

    spread_mult = float(scenario.get("spread_multiplier", 1.0))
    if spread_mult > 1.0 and "pre3m_spread_proxy_bps" in x.columns:
        spread_bps = to_num(x["pre3m_spread_proxy_bps"]).fillna(0.0).to_numpy(dtype=float)
        net_adj = net_adj - ((spread_mult - 1.0) * spread_bps / 1e4)

    out = np.full(len(x), np.nan, dtype=float)
    out[valid.to_numpy(dtype=bool)] = net_adj[valid.to_numpy(dtype=bool)]
    x["pnl_scenario"] = out

    fill_delay = to_num(x.get("fill_delay_min", np.nan)).to_numpy(dtype=float)
    fill_delay_adj = fill_delay.copy()
    if latency_bars_total > 0:
        fill_delay_adj = fill_delay_adj + 3.0 * latency_bars_total
    x["fill_delay_scenario"] = fill_delay_adj
    return x


def eval_policy_metrics_on_scenario(df_s: pd.DataFrame, keep_mask: pd.Series) -> Dict[str, Any]:
    x = df_s.copy().reset_index(drop=True)
    if x.empty:
        return {
            "exec_expectancy_net": float("nan"),
            "exec_cvar_5": float("nan"),
            "exec_max_drawdown": float("nan"),
            "entries_valid": 0,
            "entry_rate": float("nan"),
            "taker_share": float("nan"),
            "p95_fill_delay_min": float("nan"),
            "min_split_expectancy_net": float("nan"),
            "pnl_vector": np.array([], dtype=float),
            "valid_mask": np.array([], dtype=bool),
        }

    valid_orig = (x["entry_for_labels"] == 1) & to_num(x["pnl_scenario"]).notna()
    keep = keep_mask.reindex(x.index).fillna(True).astype(bool)
    valid = valid_orig & keep

    n = len(x)
    pnl = np.zeros(n, dtype=float)
    pnl[valid.to_numpy(dtype=bool)] = to_num(x.loc[valid, "pnl_scenario"]).to_numpy(dtype=float)
    exp = float(np.mean(pnl)) if n else float("nan")
    k5 = max(1, int(math.ceil(0.05 * max(1, len(pnl)))))
    cvar = float(np.mean(np.sort(pnl)[:k5])) if len(pnl) else float("nan")
    cum = np.cumsum(np.nan_to_num(pnl, nan=0.0))
    peak = np.maximum.accumulate(cum) if cum.size else np.array([], dtype=float)
    dd = cum - peak if cum.size else np.array([], dtype=float)
    mdd = float(np.nanmin(dd)) if dd.size else float("nan")

    entries = int(valid.sum())
    entry_rate = float(entries / max(1, n))
    taker = float(np.nanmean(to_num(x.loc[valid, "taker_flag"]))) if entries > 0 else float("nan")
    delays = to_num(x.loc[valid, "fill_delay_scenario"]).dropna().to_numpy(dtype=float)
    p95 = float(np.quantile(delays, 0.95)) if delays.size else float("nan")

    z = x.copy()
    z["pnl_eval"] = 0.0
    z.loc[valid, "pnl_eval"] = to_num(z.loc[valid, "pnl_scenario"])
    split_exp = z.groupby("split_id", dropna=True)["pnl_eval"].mean()
    min_split = float(split_exp.min()) if not split_exp.empty else float("nan")

    return {
        "exec_expectancy_net": float(exp),
        "exec_cvar_5": float(cvar),
        "exec_max_drawdown": float(mdd),
        "entries_valid": int(entries),
        "entry_rate": float(entry_rate),
        "taker_share": float(taker),
        "p95_fill_delay_min": float(p95),
        "min_split_expectancy_net": float(min_split),
        "pnl_vector": pnl,
        "valid_mask": valid.to_numpy(dtype=bool),
    }


def aggregate_scenario_policy(route_rows: pd.DataFrame) -> pd.DataFrame:
    if route_rows.empty:
        return pd.DataFrame()
    agg = (
        route_rows.groupby(["scenario_id", "policy_id", "policy_type", "policy_hash"], dropna=False)
        .agg(
            routes_tested=("route_id", "nunique"),
            min_delta_expectancy_vs_flat=("delta_expectancy_vs_flat", "min"),
            min_cvar_improve_ratio_vs_flat=("cvar_improve_ratio_vs_flat", "min"),
            min_maxdd_improve_ratio_vs_flat=("maxdd_improve_ratio_vs_flat", "min"),
            mean_delta_expectancy_vs_flat=("delta_expectancy_vs_flat", "mean"),
            mean_cvar_improve_ratio_vs_flat=("cvar_improve_ratio_vs_flat", "mean"),
            mean_maxdd_improve_ratio_vs_flat=("maxdd_improve_ratio_vs_flat", "mean"),
            min_entries_valid=("entries_valid", "min"),
            min_entry_rate=("entry_rate", "min"),
            min_filter_kept_entries_pct=("filter_kept_entries_pct", "min"),
            no_pathology=("no_pathology", "min"),
        )
        .reset_index()
    )
    agg["strict_pass"] = (
        (to_num(agg["min_delta_expectancy_vs_flat"]) > 0.0)
        & (to_num(agg["min_cvar_improve_ratio_vs_flat"]) >= 0.0)
        & (to_num(agg["min_maxdd_improve_ratio_vs_flat"]) > 0.0)
        & (to_num(agg["no_pathology"]) == 1)
    ).astype(int)
    return agg


def evaluate_policies_for_scenarios(
    *,
    route_data: Dict[str, pd.DataFrame],
    policies: List[Dict[str, Any]],
    scenarios: List[Dict[str, Any]],
    use_routes: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    route_rows: List[Dict[str, Any]] = []
    route_ids = use_routes if use_routes else sorted(route_data.keys())
    for sc in scenarios:
        sid = str(sc["scenario_id"])
        for route_id in route_ids:
            if route_id not in route_data:
                continue
            d_raw = route_data[route_id]
            d = apply_scenario_to_route_df(d_raw, sc)
            if d.empty:
                # Put a failing row per policy for traceability.
                for pol in policies:
                    route_rows.append(
                        {
                            "scenario_id": sid,
                            "route_id": route_id,
                            "policy_id": str(pol["policy_id"]),
                            "policy_type": str(pol.get("type", "flat")),
                            "policy_hash": dmod.policy_hash(pol),
                            "filter_kept_entries_pct": float("nan"),
                            "delta_expectancy_vs_flat": float("nan"),
                            "cvar_improve_ratio_vs_flat": float("nan"),
                            "maxdd_improve_ratio_vs_flat": float("nan"),
                            "exec_expectancy_net": float("nan"),
                            "exec_cvar_5": float("nan"),
                            "exec_max_drawdown": float("nan"),
                            "entries_valid": 0,
                            "entry_rate": float("nan"),
                            "taker_share": float("nan"),
                            "p95_fill_delay_min": float("nan"),
                            "min_split_expectancy_net": float("nan"),
                            "no_pathology": 0,
                        }
                    )
                continue

            flat = eval_policy_metrics_on_scenario(d, keep_mask=pd.Series(np.ones(len(d), dtype=bool), index=d.index))
            for pol in policies:
                keep = dmod.apply_filter_policy(d, pol)
                met = eval_policy_metrics_on_scenario(d, keep_mask=keep)
                filter_kept = safe_div(float(met["entries_valid"]), float(flat["entries_valid"])) if int(flat["entries_valid"]) > 0 else float("nan")
                cvar_imp = safe_div(abs(float(flat["exec_cvar_5"])) - abs(float(met["exec_cvar_5"])), abs(float(flat["exec_cvar_5"])))
                dd_imp = safe_div(abs(float(flat["exec_max_drawdown"])) - abs(float(met["exec_max_drawdown"])), abs(float(flat["exec_max_drawdown"])))
                no_path = int(
                    np.isfinite(met["exec_expectancy_net"])
                    and np.isfinite(met["exec_cvar_5"])
                    and np.isfinite(met["exec_max_drawdown"])
                    and np.isfinite(cvar_imp)
                    and np.isfinite(dd_imp)
                    and int(met["entries_valid"]) > 0
                )
                route_rows.append(
                    {
                        "scenario_id": sid,
                        "route_id": route_id,
                        "policy_id": str(pol["policy_id"]),
                        "policy_type": str(pol.get("type", "flat")),
                        "policy_hash": dmod.policy_hash(pol),
                        "filter_kept_entries_pct": float(filter_kept),
                        "delta_expectancy_vs_flat": float(met["exec_expectancy_net"] - flat["exec_expectancy_net"]),
                        "cvar_improve_ratio_vs_flat": float(cvar_imp),
                        "maxdd_improve_ratio_vs_flat": float(dd_imp),
                        "exec_expectancy_net": float(met["exec_expectancy_net"]),
                        "exec_cvar_5": float(met["exec_cvar_5"]),
                        "exec_max_drawdown": float(met["exec_max_drawdown"]),
                        "entries_valid": int(met["entries_valid"]),
                        "entry_rate": float(met["entry_rate"]),
                        "taker_share": float(met["taker_share"]),
                        "p95_fill_delay_min": float(met["p95_fill_delay_min"]),
                        "min_split_expectancy_net": float(met["min_split_expectancy_net"]),
                        "no_pathology": int(no_path),
                    }
                )
    rdf = pd.DataFrame(route_rows)
    agg = aggregate_scenario_policy(rdf)
    return rdf, agg


def bootstrap_pass_rate(
    *,
    route_data: Dict[str, pd.DataFrame],
    policy: Dict[str, Any],
    scenario: Dict[str, Any],
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    route_vectors: List[Dict[str, np.ndarray]] = []
    for route_id, d_raw in route_data.items():
        d = apply_scenario_to_route_df(d_raw, scenario)
        if d.empty:
            continue
        flat = eval_policy_metrics_on_scenario(d, keep_mask=pd.Series(np.ones(len(d), dtype=bool), index=d.index))
        keep = dmod.apply_filter_policy(d, policy)
        met = eval_policy_metrics_on_scenario(d, keep_mask=keep)
        route_vectors.append(
            {
                "route_id": route_id,
                "flat": np.asarray(flat["pnl_vector"], dtype=float),
                "cand": np.asarray(met["pnl_vector"], dtype=float),
            }
        )

    if not route_vectors:
        return {"boot_n": int(n_boot), "pass_rate": 0.0, "min_cvar_q05": float("nan"), "min_cvar_mean": float("nan")}

    passes = 0
    min_cvars: List[float] = []
    for _ in range(int(n_boot)):
        route_metrics: List[Tuple[float, float, float]] = []
        for rv in route_vectors:
            fv = rv["flat"]
            cv = rv["cand"]
            n = len(fv)
            if n <= 1:
                continue
            idx = rng.integers(0, n, size=n)
            f = fv[idx]
            c = cv[idx]
            f_exp = float(np.mean(f))
            c_exp = float(np.mean(c))
            k5 = max(1, int(math.ceil(0.05 * n)))
            f_cvar = float(np.mean(np.sort(f)[:k5]))
            c_cvar = float(np.mean(np.sort(c)[:k5]))
            f_cum = np.cumsum(np.nan_to_num(f, nan=0.0))
            c_cum = np.cumsum(np.nan_to_num(c, nan=0.0))
            f_dd = float(np.nanmin(f_cum - np.maximum.accumulate(f_cum)))
            c_dd = float(np.nanmin(c_cum - np.maximum.accumulate(c_cum)))
            delta = c_exp - f_exp
            cvar_imp = safe_div(abs(f_cvar) - abs(c_cvar), abs(f_cvar))
            dd_imp = safe_div(abs(f_dd) - abs(c_dd), abs(f_dd))
            route_metrics.append((delta, cvar_imp, dd_imp))
        if not route_metrics:
            continue
        min_delta = float(min(x[0] for x in route_metrics))
        min_cvar = float(min(x[1] for x in route_metrics))
        min_dd = float(min(x[2] for x in route_metrics))
        min_cvars.append(min_cvar)
        if min_delta > 0.0 and min_cvar >= 0.0 and min_dd > 0.0:
            passes += 1
    arr = np.asarray(min_cvars, dtype=float) if min_cvars else np.array([], dtype=float)
    return {
        "boot_n": int(n_boot),
        "pass_rate": float(passes / max(1, int(n_boot))),
        "min_cvar_mean": float(np.nanmean(arr)) if arr.size else float("nan"),
        "min_cvar_q05": float(np.nanquantile(arr, 0.05)) if arr.size else float("nan"),
        "min_cvar_q50": float(np.nanquantile(arr, 0.50)) if arr.size else float("nan"),
    }


def subperiod_checks(
    *,
    route_data: Dict[str, pd.DataFrame],
    policy: Dict[str, Any],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for route_id, d_raw in route_data.items():
        d = apply_scenario_to_route_df(d_raw, scenario)
        if d.empty:
            continue
        d = d.sort_values(["signal_time_utc", "signal_id"]).reset_index(drop=True)
        n = len(d)
        if n < 9:
            continue
        bins = np.floor(np.linspace(0, 3, n, endpoint=False)).astype(int)
        bins = np.clip(bins, 0, 2)
        d["subperiod_id"] = bins
        for sub in [0, 1, 2]:
            g = d[d["subperiod_id"] == sub].copy()
            if g.empty:
                continue
            flat = eval_policy_metrics_on_scenario(g, keep_mask=pd.Series(np.ones(len(g), dtype=bool), index=g.index))
            keep = dmod.apply_filter_policy(g, policy)
            met = eval_policy_metrics_on_scenario(g, keep_mask=keep)
            delta = float(met["exec_expectancy_net"] - flat["exec_expectancy_net"])
            cvar_imp = safe_div(abs(float(flat["exec_cvar_5"])) - abs(float(met["exec_cvar_5"])), abs(float(flat["exec_cvar_5"])))
            rows.append(
                {
                    "route_id": route_id,
                    "subperiod_id": int(sub),
                    "delta_expectancy_vs_flat": float(delta),
                    "cvar_improve_ratio_vs_flat": float(cvar_imp),
                    "entries_valid": int(met["entries_valid"]),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return {"min_subperiod_delta": float("nan"), "min_subperiod_cvar_improve": float("nan"), "subperiod_rows": []}
    return {
        "min_subperiod_delta": float(to_num(df["delta_expectancy_vs_flat"]).min()),
        "min_subperiod_cvar_improve": float(to_num(df["cvar_improve_ratio_vs_flat"]).min()),
        "subperiod_rows": df.to_dict(orient="records"),
    }


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
            return "INFRA_FAIL", {"reason": f"Missing locked input: {fp}", "phase_dir": str(phase_dir)}

    fee_sha = sha256_file(fee_fp)
    met_sha = sha256_file(met_fp)
    if fee_sha != args.expected_fee_sha or met_sha != args.expected_metrics_sha:
        return "CONTRACT_FAIL", {
            "reason": "hash mismatch",
            "observed_fee_sha": fee_sha,
            "observed_metrics_sha": met_sha,
            "phase_dir": str(phase_dir),
        }

    raw = pd.read_csv(subset_fp)
    sig = ae.ensure_signals_schema(raw)
    if sig.empty:
        return "CONTRACT_FAIL", {"reason": "frozen subset empty", "phase_dir": str(phase_dir)}
    required_cols = ["signal_id", "signal_time", "tp_mult", "sl_mult", "atr_percentile_1h", "trend_up_1h"]
    if not all(c in sig.columns for c in required_cols):
        return "INFRA_FAIL", {"reason": "subset schema mismatch", "phase_dir": str(phase_dir)}

    lock_args = ae.build_args(signals_csv=subset_fp, seed=int(args.seed))
    lock_args.allow_freeze_hash_mismatch = 0
    lock_validation = ga_exec._validate_and_lock_frozen_artifacts(args=lock_args, run_dir=phase_dir)
    if int(lock_validation.get("freeze_lock_pass", 0)) != 1:
        return "CONTRACT_FAIL", {"reason": "freeze lock validation failed", "phase_dir": str(phase_dir), "freeze_lock_validation": lock_validation}

    out = {
        "generated_utc": utc_now(),
        "phase": "A",
        "decision": "PASS",
        "duration_sec": float(time.time() - t0),
        "phase_dir": str(phase_dir),
        "fee_sha256": fee_sha,
        "metrics_sha256": met_sha,
        "subset_rows_raw": int(len(raw)),
        "subset_rows_schema_normalized": int(len(sig)),
        "required_cols": required_cols,
        "freeze_lock_validation": lock_validation,
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
                f"- Fee hash match: `{int(fee_sha == args.expected_fee_sha)}`",
                f"- Metrics hash match: `{int(met_sha == args.expected_metrics_sha)}`",
                f"- Freeze lock pass: `{int(lock_validation.get('freeze_lock_pass', 0))}`",
                f"- Subset rows: `{len(sig)}`",
            ]
        ),
    )
    write_text(phase_dir / "phaseA_decision.md", "# Phase A Decision\n\n- Classification: **PASS**\n- Reason: contract lock checks passed.")
    return "PASS", out


def phase_e1(
    run_dir: Path,
    args: argparse.Namespace,
    hb: Heartbeat,
    phase_d_dir: Path,
    route_data: Dict[str, pd.DataFrame],
    candidate_policies: List[Dict[str, Any]],
    near_miss_policy: Optional[Dict[str, Any]],
    d3df: pd.DataFrame,
) -> Tuple[str, Dict[str, Any]]:
    hb.set_state("E1", "running")
    t0 = time.time()
    phase_dir = run_dir / "phaseE1"
    phase_dir.mkdir(parents=True, exist_ok=True)

    policies = list(candidate_policies)
    if near_miss_policy is not None:
        policies.append(near_miss_policy)
    flat = {"policy_id": "flat_baseline", "type": "flat"}
    policies = [flat] + policies

    base_scenario = {"scenario_id": "base"}
    rdf, agg = evaluate_policies_for_scenarios(route_data=route_data, policies=policies, scenarios=[base_scenario], use_routes=args.routes)
    agg = agg[agg["policy_id"] != "flat_baseline"].copy().reset_index(drop=True)
    if agg.empty:
        return "NO_GO", {"reason": "E1 produced empty aggregate results", "phase_dir": str(phase_dir)}

    prior = d3df[d3df["policy_id"].isin(agg["policy_id"])].copy()
    cmp = agg.merge(
        prior[
            [
                "policy_id",
                "min_delta_expectancy_vs_flat",
                "min_cvar_improve_ratio_vs_flat",
                "min_maxdd_improve_ratio_vs_flat",
                "strict_pass",
            ]
        ].rename(
            columns={
                "min_delta_expectancy_vs_flat": "prior_min_delta_expectancy_vs_flat",
                "min_cvar_improve_ratio_vs_flat": "prior_min_cvar_improve_ratio_vs_flat",
                "min_maxdd_improve_ratio_vs_flat": "prior_min_maxdd_improve_ratio_vs_flat",
                "strict_pass": "prior_strict_pass",
            }
        ),
        on="policy_id",
        how="left",
    )

    cmp["delta_abs_diff"] = np.abs(
        to_num(cmp["min_delta_expectancy_vs_flat"]) - to_num(cmp["prior_min_delta_expectancy_vs_flat"])
    )
    cmp["delta_tol"] = np.maximum(np.abs(to_num(cmp["prior_min_delta_expectancy_vs_flat"])) * 0.20, 5e-5)
    cmp["delta_within_tol"] = (to_num(cmp["delta_abs_diff"]) <= to_num(cmp["delta_tol"])).astype(int)
    cmp["cvar_nonneg"] = (to_num(cmp["min_cvar_improve_ratio_vs_flat"]) >= 0.0).astype(int)
    cmp["maxdd_pos"] = (to_num(cmp["min_maxdd_improve_ratio_vs_flat"]) > 0.0).astype(int)

    cmp.to_csv(phase_dir / "phaseE1_repro_results.csv", index=False)
    rdf.to_csv(phase_dir / "phaseE1_repro_results_by_route.csv", index=False)

    top1 = cmp[cmp["policy_id"] == "skip_streak5_cool60m"].copy()
    top1_repro = bool((to_num(top1.get("strict_pass")) == 1).any()) if not top1.empty else False
    at_least_one = bool((to_num(cmp["strict_pass"]) == 1).any())
    cls = "PASS" if (top1_repro and at_least_one) else "NO_GO"
    reason = "top candidate reproduced strict pass" if cls == "PASS" else "no strict passer reproduced"

    lines = []
    lines.append("# Phase E1 Reproduction Report")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append(f"- Decision: **{cls}**")
    lines.append(f"- Reason: {reason}")
    lines.append(f"- Phase D source: `{phase_d_dir}`")
    lines.append("")
    lines.append("## Candidate reproduction table")
    lines.append("")
    lines.append(
        markdown_table(
            cmp,
            [
                "policy_id",
                "strict_pass",
                "prior_strict_pass",
                "min_delta_expectancy_vs_flat",
                "prior_min_delta_expectancy_vs_flat",
                "delta_abs_diff",
                "delta_tol",
                "delta_within_tol",
                "min_cvar_improve_ratio_vs_flat",
                "cvar_nonneg",
                "min_maxdd_improve_ratio_vs_flat",
                "maxdd_pos",
            ],
        )
    )
    write_text(phase_dir / "phaseE1_repro_report.md", "\n".join(lines))
    out = {
        "generated_utc": utc_now(),
        "phase": "E1",
        "decision": cls,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "phase_dir": str(phase_dir),
    }
    json_dump(phase_dir / "phaseE1_run_manifest.json", out)
    write_text(phase_dir / "phaseE1_decision.md", f"# Phase E1 Decision\n\n- Classification: **{cls}**\n- Reason: {reason}")
    return cls, out


def phase_e2(
    run_dir: Path,
    args: argparse.Namespace,
    hb: Heartbeat,
    route_data: Dict[str, pd.DataFrame],
    candidate_policies: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    hb.set_state("E2", "running")
    t0 = time.time()
    phase_dir = run_dir / "phaseE2"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Route perturb scenarios (lightweight and contract-safe).
    scenarios = [
        {"scenario_id": "base"},
        {"scenario_id": "maker_bias", "maker_fee_mult": 0.80, "taker_fee_mult": 0.70},
        {"scenario_id": "taker_bias", "maker_fee_mult": 1.40, "taker_fee_mult": 1.60},
        {"scenario_id": "trim_head10", "trim_head_pct": 0.10},
        {"scenario_id": "trim_tail10", "trim_tail_pct": 0.10},
    ]
    rdf, agg = evaluate_policies_for_scenarios(route_data=route_data, policies=candidate_policies, scenarios=scenarios, use_routes=args.routes)
    if agg.empty:
        return "NO_GO", {"reason": "empty route perturb aggregate", "phase_dir": str(phase_dir)}
    matrix = agg.sort_values(["policy_id", "scenario_id"]).reset_index(drop=True)
    matrix.to_csv(phase_dir / "route_perturb_matrix.csv", index=False)
    rdf.to_csv(phase_dir / "route_perturb_matrix_by_route.csv", index=False)

    # Bootstrap pass rates on base scenario.
    boot_obj: Dict[str, Any] = {"generated_utc": utc_now(), "n_boot": int(args.bootstrap_n), "policies": {}}
    for i, pol in enumerate(candidate_policies):
        boot = bootstrap_pass_rate(
            route_data=route_data,
            policy=pol,
            scenario={"scenario_id": "base"},
            n_boot=int(args.bootstrap_n),
            seed=int(args.seed) + 2000 + i,
        )
        boot_obj["policies"][str(pol["policy_id"])] = boot
    json_dump(phase_dir / "bootstrap_pass_rates.json", boot_obj)

    top_id = "skip_streak5_cool60m"
    top_row = matrix[(matrix["policy_id"] == top_id) & (matrix["scenario_id"] == "base")]
    if top_row.empty:
        return "NO_GO", {"reason": f"{top_id} missing from route perturb matrix", "phase_dir": str(phase_dir)}

    top_boot = boot_obj["policies"].get(top_id, {})
    pass_rate = float(top_boot.get("pass_rate", 0.0))
    top_base_strict = int(top_row.iloc[0]["strict_pass"]) == 1

    # per-route cvar drift for top candidate across all perturb scenarios.
    top_route = rdf[rdf["policy_id"] == top_id].copy()
    cvar_min = float(to_num(top_route["cvar_improve_ratio_vs_flat"]).min()) if not top_route.empty else float("nan")
    eps = 1e-6

    if top_base_strict and pass_rate >= 0.70 and cvar_min >= -eps:
        cls = "PASS"
        reason = "base strict pass + bootstrap >=70% + no route-level CVaR drift below epsilon"
    elif top_base_strict and (0.55 <= pass_rate < 0.70) and (-5e-4 <= cvar_min < -1e-4):
        cls = "WEAK_PASS"
        reason = "bootstrap 55-69% with tiny negative CVaR drift (weak pass allowed)"
    else:
        cls = "NO_GO"
        reason = f"bootstrap/cvar robustness failed (pass_rate={pass_rate:.3f}, min_route_cvar={cvar_min:.6f})"

    lines: List[str] = []
    lines.append("# Phase E2 Route Perturbation Report")
    lines.append("")
    lines.append(f"- Generated UTC: {utc_now()}")
    lines.append(f"- Decision: **{cls}**")
    lines.append(f"- Reason: {reason}")
    lines.append(f"- Top candidate `{top_id}` bootstrap pass rate: `{pass_rate:.4f}`")
    lines.append(f"- Top candidate min route-level CVaR improve across perturbs: `{cvar_min:.6f}`")
    lines.append("")
    lines.append("## Scenario matrix (aggregate)")
    lines.append("")
    lines.append(
        markdown_table(
            matrix,
            [
                "scenario_id",
                "policy_id",
                "strict_pass",
                "min_delta_expectancy_vs_flat",
                "min_cvar_improve_ratio_vs_flat",
                "min_maxdd_improve_ratio_vs_flat",
                "min_entries_valid",
                "min_filter_kept_entries_pct",
            ],
        )
    )
    write_text(phase_dir / "phaseE2_route_perturb_report.md", "\n".join(lines))

    out = {
        "generated_utc": utc_now(),
        "phase": "E2",
        "decision": cls,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "phase_dir": str(phase_dir),
        "top_candidate_id": top_id,
        "top_candidate_bootstrap_pass_rate": pass_rate,
        "top_candidate_min_route_cvar_improve": cvar_min,
    }
    json_dump(phase_dir / "phaseE2_run_manifest.json", out)
    write_text(phase_dir / "phaseE2_decision.md", f"# Phase E2 Decision\n\n- Classification: **{cls}**\n- Reason: {reason}")
    return cls, out


def phase_e3(
    run_dir: Path,
    args: argparse.Namespace,
    hb: Heartbeat,
    route_data: Dict[str, pd.DataFrame],
    candidate_policies: List[Dict[str, Any]],
    e2_cls: str,
) -> Tuple[str, Dict[str, Any]]:
    hb.set_state("E3", "running")
    t0 = time.time()
    phase_dir = run_dir / "phaseE3"
    phase_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {"scenario_id": "S00_base"},
        {"scenario_id": "S01_cost125", "cost_multiplier": 1.25},
        {"scenario_id": "S02_cost150", "cost_multiplier": 1.50},
        {"scenario_id": "S03_slip_p1", "extra_slippage_bps": 1.0},
        {"scenario_id": "S04_slip_p2", "extra_slippage_bps": 2.0},
        {"scenario_id": "S05_lat_entry1", "entry_delay_bars": 1, "latency_penalty_bps_per_bar": 1.0},
        {"scenario_id": "S06_lat_exit1", "exit_delay_bars": 1, "latency_penalty_bps_per_bar": 1.0},
        {"scenario_id": "S07_lat_both1", "entry_delay_bars": 1, "exit_delay_bars": 1, "latency_penalty_bps_per_bar": 1.0},
        {"scenario_id": "S08_spread15", "spread_multiplier": 1.5},
        {"scenario_id": "S09_cost125_slip1", "cost_multiplier": 1.25, "extra_slippage_bps": 1.0},
        {"scenario_id": "S10_cost150_slip2_lat1", "cost_multiplier": 1.50, "extra_slippage_bps": 2.0, "entry_delay_bars": 1, "latency_penalty_bps_per_bar": 1.0},
        {"scenario_id": "S11_spread15_lat1", "spread_multiplier": 1.5, "entry_delay_bars": 1, "latency_penalty_bps_per_bar": 1.0},
    ]

    rdf, agg = evaluate_policies_for_scenarios(route_data=route_data, policies=candidate_policies, scenarios=scenarios, use_routes=args.routes)
    if agg.empty:
        return "NO_GO", {"reason": "empty stress aggregate", "phase_dir": str(phase_dir)}

    # Add subperiod constraints.
    sub_rows: List[Dict[str, Any]] = []
    for _, r in agg.iterrows():
        sid = str(r["scenario_id"])
        pid = str(r["policy_id"])
        pol = next((p for p in candidate_policies if str(p["policy_id"]) == pid), None)
        sc = next((s for s in scenarios if str(s["scenario_id"]) == sid), None)
        if pol is None or sc is None:
            continue
        sub = subperiod_checks(route_data=route_data, policy=pol, scenario=sc)
        sub_rows.append(
            {
                "scenario_id": sid,
                "policy_id": pid,
                "min_subperiod_delta": float(sub["min_subperiod_delta"]),
                "min_subperiod_cvar_improve": float(sub["min_subperiod_cvar_improve"]),
            }
        )
    sub_df = pd.DataFrame(sub_rows)
    if not sub_df.empty:
        agg = agg.merge(sub_df, on=["scenario_id", "policy_id"], how="left")
    else:
        agg["min_subperiod_delta"] = np.nan
        agg["min_subperiod_cvar_improve"] = np.nan

    # Pathology check.
    key_cols = [
        "min_delta_expectancy_vs_flat",
        "min_cvar_improve_ratio_vs_flat",
        "min_maxdd_improve_ratio_vs_flat",
        "min_subperiod_delta",
        "min_subperiod_cvar_improve",
    ]
    pathology_mask = np.zeros(len(agg), dtype=bool)
    for c in key_cols:
        pathology_mask |= ~np.isfinite(to_num(agg[c]).to_numpy(dtype=float))
    if bool(pathology_mask.any()):
        path_rows = agg[pathology_mask].copy()
        path_rows.to_csv(phase_dir / "pathology_rows.csv", index=False)
        write_text(
            phase_dir / "pathology_report.md",
            "\n".join(
                [
                    "# Phase E3 Pathology Report",
                    "",
                    f"- Generated UTC: {utc_now()}",
                    f"- Pathology rows: {int(pathology_mask.sum())}",
                    "- Stopping due to NaN/inf metric pathology.",
                ]
            ),
        )
        return "PATHOLOGY", {"reason": "NaN/inf metrics in stress matrix", "phase_dir": str(phase_dir)}

    # Candidate scenario pass flags.
    agg["scenario_pass"] = (
        (to_num(agg["strict_pass"]) == 1)
        & (to_num(agg["min_subperiod_delta"]) > 0.0)
        & (to_num(agg["min_subperiod_cvar_improve"]) >= 0.0)
        & (to_num(agg["min_filter_kept_entries_pct"]) >= 0.60)
    ).astype(int)

    pass_threshold = 0.70 if str(e2_cls).upper() == "WEAK_PASS" else 0.60
    cand_rows: List[Dict[str, Any]] = []
    for pid, g in agg.groupby("policy_id", dropna=False):
        g = g.sort_values("scenario_id")
        base = g[g["scenario_id"] == "S00_base"]
        base_ok = int(base.iloc[0]["strict_pass"]) == 1 if not base.empty else False
        scen_pass_rate = float(np.mean(to_num(g["scenario_pass"]) == 1))
        min_sub_delta = float(to_num(g["min_subperiod_delta"]).min())
        min_sub_cvar = float(to_num(g["min_subperiod_cvar_improve"]).min())
        min_kept = float(to_num(g["min_filter_kept_entries_pct"]).min())
        cand_rows.append(
            {
                "policy_id": str(pid),
                "base_strict_pass": int(base_ok),
                "scenario_pass_rate": float(scen_pass_rate),
                "min_subperiod_delta": min_sub_delta,
                "min_subperiod_cvar_improve": min_sub_cvar,
                "min_filter_kept_entries_pct": min_kept,
                "passes_e3_rule": int(base_ok and scen_pass_rate >= pass_threshold and min_sub_delta > 0 and min_sub_cvar >= 0 and min_kept >= 0.60),
                "mean_min_delta": float(to_num(g["min_delta_expectancy_vs_flat"]).mean()),
                "mean_min_cvar_imp": float(to_num(g["min_cvar_improve_ratio_vs_flat"]).mean()),
                "mean_min_dd_imp": float(to_num(g["min_maxdd_improve_ratio_vs_flat"]).mean()),
            }
        )
    cdf = pd.DataFrame(cand_rows).sort_values(["passes_e3_rule", "scenario_pass_rate", "mean_min_delta"], ascending=[False, False, False]).reset_index(drop=True)

    # Invalid reason histogram.
    invalid_hist: Dict[str, int] = {}
    bad = agg[agg["scenario_pass"] != 1].copy()
    for _, r in bad.iterrows():
        if float(r["min_delta_expectancy_vs_flat"]) <= 0.0:
            invalid_hist["non_positive_expectancy_delta"] = int(invalid_hist.get("non_positive_expectancy_delta", 0) + 1)
        if float(r["min_cvar_improve_ratio_vs_flat"]) < 0.0:
            invalid_hist["negative_cvar_improvement"] = int(invalid_hist.get("negative_cvar_improvement", 0) + 1)
        if float(r["min_maxdd_improve_ratio_vs_flat"]) <= 0.0:
            invalid_hist["non_positive_maxdd_improvement"] = int(invalid_hist.get("non_positive_maxdd_improvement", 0) + 1)
        if float(r["min_subperiod_delta"]) <= 0.0:
            invalid_hist["subperiod_non_positive_expectancy"] = int(invalid_hist.get("subperiod_non_positive_expectancy", 0) + 1)
        if float(r["min_subperiod_cvar_improve"]) < 0.0:
            invalid_hist["subperiod_negative_cvar"] = int(invalid_hist.get("subperiod_negative_cvar", 0) + 1)
        if float(r["min_filter_kept_entries_pct"]) < 0.60:
            invalid_hist["trade_kill_too_high"] = int(invalid_hist.get("trade_kill_too_high", 0) + 1)
    json_dump(phase_dir / "invalid_reason_histogram.json", invalid_hist)

    agg.to_csv(phase_dir / "stress_matrix_results.csv", index=False)
    rdf.to_csv(phase_dir / "stress_matrix_results_by_route.csv", index=False)
    cdf.to_csv(phase_dir / "phaseE3_candidate_summary.csv", index=False)

    worst_case = (
        agg.sort_values(["min_delta_expectancy_vs_flat", "min_cvar_improve_ratio_vs_flat", "min_maxdd_improve_ratio_vs_flat"], ascending=[True, True, True]).head(10)
    )
    json_dump(
        phase_dir / "worst_case_scenario_summary.json",
        {"generated_utc": utc_now(), "pass_threshold": pass_threshold, "rows": worst_case.to_dict(orient="records")},
    )

    has_pass = bool((to_num(cdf["passes_e3_rule"]) == 1).any())
    cls = "PASS" if has_pass else "NO_GO"
    reason = ">=1 candidate passed stress coverage rule" if has_pass else "no candidate met stress coverage rule"

    rep = []
    rep.append("# Phase E3 Stress Report")
    rep.append("")
    rep.append(f"- Generated UTC: {utc_now()}")
    rep.append(f"- Decision: **{cls}**")
    rep.append(f"- Reason: {reason}")
    rep.append(f"- E2 inherited class: `{e2_cls}`")
    rep.append(f"- Scenario pass threshold: `{pass_threshold:.2f}`")
    rep.append("")
    rep.append("## Candidate stress summary")
    rep.append("")
    rep.append(
        markdown_table(
            cdf,
            [
                "policy_id",
                "passes_e3_rule",
                "base_strict_pass",
                "scenario_pass_rate",
                "min_subperiod_delta",
                "min_subperiod_cvar_improve",
                "min_filter_kept_entries_pct",
                "mean_min_delta",
                "mean_min_cvar_imp",
                "mean_min_dd_imp",
            ],
        )
    )
    write_text(phase_dir / "phaseE3_stress_report.md", "\n".join(rep))

    out = {
        "generated_utc": utc_now(),
        "phase": "E3",
        "decision": cls,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "phase_dir": str(phase_dir),
        "pass_threshold": float(pass_threshold),
        "passing_candidates": cdf[cdf["passes_e3_rule"] == 1]["policy_id"].astype(str).tolist(),
    }
    json_dump(phase_dir / "phaseE3_run_manifest.json", out)
    write_text(phase_dir / "phaseE3_decision.md", f"# Phase E3 Decision\n\n- Classification: **{cls}**\n- Reason: {reason}")
    return cls, out


def phase_e4(
    run_dir: Path,
    args: argparse.Namespace,
    hb: Heartbeat,
    phase_d_dir: Path,
    candidate_policies: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    hb.set_state("E4", "running")
    t0 = time.time()
    phase_dir = run_dir / "phaseE4"
    phase_dir.mkdir(parents=True, exist_ok=True)

    csum_fp = run_dir / "phaseE3" / "phaseE3_candidate_summary.csv"
    if not csum_fp.exists():
        return "INFRA_FAIL", {"reason": "missing phaseE3 candidate summary", "phase_dir": str(phase_dir)}
    csum = pd.read_csv(csum_fp)
    winners = csum[csum["passes_e3_rule"] == 1].copy()
    if winners.empty:
        return "NO_GO", {"reason": "no E3 winners for paper package", "phase_dir": str(phase_dir)}
    winners = winners.sort_values(["scenario_pass_rate", "mean_min_delta"], ascending=[False, False]).reset_index(drop=True)
    primary_id = str(winners.iloc[0]["policy_id"])
    backup_id = str(winners.iloc[1]["policy_id"]) if len(winners) > 1 else primary_id

    pol_lookup = {str(p["policy_id"]): p for p in candidate_policies}
    primary_pol = pol_lookup.get(primary_id, {"policy_id": primary_id, "type": "unknown"})
    backup_pol = pol_lookup.get(backup_id, {"policy_id": backup_id, "type": "unknown"})

    cfg = {
        "generated_utc": utc_now(),
        "mode": "paper_shadow_only",
        "phase_d_source": str(phase_d_dir),
        "phase_e_source": str(run_dir),
        "primary_policy": primary_pol,
        "backup_policy": backup_pol,
        "hard_gates_unchanged": True,
        "freeze_lock": {
            "frozen_subset_csv": str(args.frozen_subset_csv),
            "canonical_fee_model_path": str(args.fee_path),
            "canonical_metrics_definition_path": str(args.metrics_path),
            "expected_fee_sha256": str(args.expected_fee_sha),
            "expected_metrics_sha256": str(args.expected_metrics_sha),
            "allow_freeze_hash_mismatch": 0,
        },
    }
    json_dump(phase_dir / "paper_policy_config.json", cfg)

    write_text(
        phase_dir / "paper_run_checklist.md",
        "\n".join(
            [
                "# Paper Run Checklist",
                "",
                "1. Run primary and flat baseline in parallel shadow books.",
                "2. Track per-route delta expectancy and per-route CVaR daily.",
                "3. Verify entry_rate and kept_entries_pct remain within expected bounds.",
                "4. Track slippage/fee drift against modeled bounds each day.",
                "5. Maintain backup policy ready; switch only on rollback trigger.",
            ]
        ),
    )

    write_text(
        phase_dir / "rollback_triggers.md",
        "\n".join(
            [
                "# Rollback Triggers",
                "",
                "- If rolling 30-trade CVaR is worse than flat baseline by >15%, disable policy.",
                "- If entry_rate drops below 60% of expected baseline for >1 day, disable policy.",
                "- If worst-route delta expectancy is negative for 2 consecutive days, disable policy.",
                "- If effective fees/slippage exceed modeled bounds by >25% for 1 day, disable policy.",
            ]
        ),
    )
    write_text(
        phase_dir / "reality_check_TODO.md",
        "\n".join(
            [
                "# Reality Check TODO",
                "",
                "- Implement full White-style reality-check bootstrap before any live deployment consideration.",
                "- Add execution-venue drift alerts and intraday fee/slippage reconciliation.",
            ]
        ),
    )

    cls = "PASS_GO_PAPER_SHADOW"
    reason = "paper/shadow package generated from stress-passing candidate set"
    out = {
        "generated_utc": utc_now(),
        "phase": "E4",
        "decision": cls,
        "reason": reason,
        "duration_sec": float(time.time() - t0),
        "phase_dir": str(phase_dir),
        "primary_policy_id": primary_id,
        "backup_policy_id": backup_id,
    }
    json_dump(phase_dir / "phaseE4_run_manifest.json", out)
    write_text(phase_dir / "phaseE4_decision.md", f"# Phase E4 Decision\n\n- Classification: **{cls}**\n- Reason: {reason}")
    return cls, out


def write_final_summary(
    run_dir: Path,
    overall: Dict[str, Any],
    top_df: Optional[pd.DataFrame],
    next_prompt: Optional[str],
) -> None:
    lines: List[str] = []
    lines.append(f"- Furthest phase reached: {overall.get('furthest_phase', 'A')}")
    lines.append(f"- Classification at furthest phase: {overall.get('furthest_classification', 'UNKNOWN')}")
    lines.append(f"- Mainline status: {overall.get('mainline_status', 'STOP_NO_GO')}")
    lines.append("")
    lines.append("- What was proven (1 paragraph)")
    lines.append(str(overall.get("plain_english", "")))
    lines.append("")
    lines.append("- Top candidates (exact metrics)")
    if top_df is not None and not top_df.empty:
        lines.append(markdown_table(top_df, top_df.columns.tolist()))
    else:
        lines.append("_(none)_")
    lines.append("")
    lines.append("- Failure branch taken (if any)")
    lines.append(str(overall.get("failure_branch", "none")))
    lines.append("")
    lines.append("- Is next phase justified? (yes/no)")
    lines.append("yes" if str(overall.get("mainline_status")) == "CONTINUE" else "no")
    lines.append("")
    lines.append("- Artifact directory (exact path)")
    lines.append(str(run_dir))
    lines.append("")
    lines.append("- Key files list")
    keys = {
        "phaseA_freeze_lock_validation.json",
        "phaseA_report.md",
        "phaseA_decision.md",
        "phaseE1_repro_report.md",
        "phaseE1_repro_results.csv",
        "phaseE1_decision.md",
        "route_perturb_matrix.csv",
        "bootstrap_pass_rates.json",
        "phaseE2_decision.md",
        "stress_matrix_results.csv",
        "worst_case_scenario_summary.json",
        "invalid_reason_histogram.json",
        "phaseE3_decision.md",
        "paper_policy_config.json",
        "paper_run_checklist.md",
        "rollback_triggers.md",
        "phaseE4_decision.md",
        "run_manifest.json",
    }
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name in keys:
            lines.append(f"- {p}")
    lines.append("")
    lines.append("- Exact next prompt contents (only if justified)")
    if next_prompt and str(overall.get("mainline_status")) == "CONTINUE":
        lines.append(next_prompt)
    else:
        lines.append("Not justified.")
    write_text(run_dir / "FINAL_SUMMARY.md", "\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase E paper confirmation autorun (repro + route perturb + stress, no GA)")
    ap.add_argument("--seed", type=int, default=20260223)
    ap.add_argument("--time_budget_minutes", type=int, default=175)
    ap.add_argument("--phaseD_dir", default=DEFAULTS["phaseD_dir"])
    ap.add_argument("--frozen_subset_csv", default=DEFAULTS["frozen_subset_csv"])
    ap.add_argument("--fee_path", default=DEFAULTS["fee_path"])
    ap.add_argument("--metrics_path", default=DEFAULTS["metrics_path"])
    ap.add_argument("--expected_fee_sha", default=DEFAULTS["expected_fee_sha"])
    ap.add_argument("--expected_metrics_sha", default=DEFAULTS["expected_metrics_sha"])
    ap.add_argument("--allow_freeze_hash_mismatch", type=int, default=0)
    ap.add_argument("--candidates_json", default="")
    ap.add_argument("--routes", default="")
    ap.add_argument("--stress_profile", default="standard")
    ap.add_argument("--bootstrap_n", type=int, default=300)
    ap.add_argument("--report_dir", default="")
    args = ap.parse_args()

    if int(args.allow_freeze_hash_mismatch) != 0:
        raise RuntimeError("allow_freeze_hash_mismatch must be 0")

    root = PROJECT_ROOT / "reports" / "execution_layer"
    if args.report_dir:
        run_dir = Path(args.report_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = root / f"PHASEE_PAPER_CONFIRM_{utc_tag()}"
        run_dir.mkdir(parents=True, exist_ok=False)

    hb = Heartbeat(run_dir=run_dir, interval_sec=75)
    hb.start()
    hb.set_state("PHASE0", "running")
    t0 = time.time()
    deadline = t0 + max(1, int(args.time_budget_minutes)) * 60.0

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

    # Resolve routes.
    args.routes = [r.strip() for r in str(args.routes).split(",") if r.strip()]
    try:
        phase_d_dir = find_phase_d_dir(args.phaseD_dir)
        cand_pols, near_miss_pol, d3df = load_phase_d_candidates(phase_d_dir, candidates_json=str(args.candidates_json), include_near_miss=True)
        if not args.routes:
            by_route = phase_d_dir / "phaseD3" / "phaseD3_results_by_route.csv"
            if by_route.exists():
                bdf = pd.read_csv(by_route)
                rr = [str(x) for x in bdf["route_id"].dropna().unique().tolist()]
                args.routes = rr if rr else ["route1_holdout", "route2_reslice"]
            else:
                args.routes = ["route1_holdout", "route2_reslice"]
    except Exception as e:
        overall["mainline_status"] = "STOP_INFRA"
        overall["furthest_phase"] = "0"
        overall["furthest_classification"] = "INFRA_FAIL"
        overall["plain_english"] = f"Failed to initialize Phase E inputs: {e}"
        overall["failure_branch"] = "INFRA_FAIL"
        write_text(run_dir / "infra_report.md", f"Initialization failure: {e}")
        json_dump(run_dir / "run_manifest.json", overall)
        write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
        hb.stop("stopped")
        return

    route_cache: Dict[str, pd.DataFrame] = {}

    # Phase A
    if not phase_done(run_dir, "phaseA"):
        cls, info = phase_a(run_dir, args, hb)
        overall["phases"]["A"] = {"classification": cls, "info": info}
        write_checkpoint(run_dir, "phaseA", cls, info)
        if cls != "PASS":
            overall["mainline_status"] = "STOP_CONTRACT" if cls == "CONTRACT_FAIL" else "STOP_INFRA"
            overall["furthest_phase"] = "A"
            overall["furthest_classification"] = cls
            overall["plain_english"] = f"Phase A failed: {info.get('reason', 'unknown')}"
            overall["failure_branch"] = cls
            json_dump(run_dir / "run_manifest.json", overall)
            write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
            hb.stop("stopped")
            return

    # Build route cache once after contract pass.
    try:
        route_cache, _exec_pair = load_locked_route_data(
            phase_dir=run_dir / "phaseE1",
            frozen_subset_csv=Path(args.frozen_subset_csv).resolve(),
            seed=int(args.seed),
        )
    except Exception as e:
        overall["mainline_status"] = "STOP_INFRA"
        overall["furthest_phase"] = "E0"
        overall["furthest_classification"] = "INFRA_FAIL"
        overall["plain_english"] = f"Failed to build route cache: {e}"
        overall["failure_branch"] = "INFRA_FAIL"
        write_text(run_dir / "infra_report.md", f"Route cache failure: {e}")
        json_dump(run_dir / "run_manifest.json", overall)
        write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
        hb.stop("stopped")
        return

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

    # Phase E1
    if not phase_done(run_dir, "E1"):
        cls, info = phase_e1(run_dir, args, hb, phase_d_dir, route_cache, cand_pols, near_miss_pol, d3df)
        overall["phases"]["E1"] = {"classification": cls, "info": info}
        write_checkpoint(run_dir, "E1", cls, info)
        if cls != "PASS":
            overall["mainline_status"] = "STOP_NO_GO"
            overall["furthest_phase"] = "E1"
            overall["furthest_classification"] = cls
            overall["plain_english"] = f"Phase E1 failed to reproduce strict-pass candidates: {info.get('reason', 'unknown')}."
            overall["failure_branch"] = cls
            json_dump(run_dir / "run_manifest.json", overall)
            write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
            hb.stop("stopped")
            return

    if time.time() >= deadline:
        overall["mainline_status"] = "STOP_NO_GO"
        overall["furthest_phase"] = "E1"
        overall["furthest_classification"] = "NO_GO"
        overall["plain_english"] = "Time budget exhausted after Phase E1."
        overall["failure_branch"] = "NO_GO (time budget)"
        json_dump(run_dir / "run_manifest.json", overall)
        write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
        hb.stop("stopped")
        return

    # Phase E2
    if not phase_done(run_dir, "E2"):
        cls, info = phase_e2(run_dir, args, hb, route_cache, cand_pols)
        overall["phases"]["E2"] = {"classification": cls, "info": info}
        write_checkpoint(run_dir, "E2", cls, info)
        if cls == "NO_GO":
            overall["mainline_status"] = "STOP_NO_GO"
            overall["furthest_phase"] = "E2"
            overall["furthest_classification"] = cls
            overall["plain_english"] = "Route perturbation/bootstrapping did not support robust strict-pass behavior."
            overall["failure_branch"] = cls
            ng = run_dir / "phaseE2" / "no_go_package"
            ng.mkdir(parents=True, exist_ok=True)
            write_text(ng / "failure_diagnosis.md", f"Phase E2 NO_GO: {info.get('reason', 'unknown')}")
            write_text(ng / "next_step_prompt.txt", "Refine route-aligned risk controls before additional promotion attempts; keep hard gates unchanged.")
            json_dump(run_dir / "run_manifest.json", overall)
            write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
            hb.stop("stopped")
            return
        if cls not in {"PASS", "WEAK_PASS"}:
            overall["mainline_status"] = "STOP_INFRA"
            overall["furthest_phase"] = "E2"
            overall["furthest_classification"] = cls
            overall["plain_english"] = "Phase E2 failed due to infrastructure/pathology."
            overall["failure_branch"] = cls
            json_dump(run_dir / "run_manifest.json", overall)
            write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
            hb.stop("stopped")
            return
    else:
        ck = read_checkpoint(run_dir, "E2") or {}
        cls = str(ck.get("classification", "PASS"))
    e2_cls = cls

    if time.time() >= deadline:
        overall["mainline_status"] = "STOP_NO_GO"
        overall["furthest_phase"] = "E2"
        overall["furthest_classification"] = "NO_GO"
        overall["plain_english"] = "Time budget exhausted after Phase E2."
        overall["failure_branch"] = "NO_GO (time budget)"
        json_dump(run_dir / "run_manifest.json", overall)
        write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
        hb.stop("stopped")
        return

    # Phase E3
    if not phase_done(run_dir, "E3"):
        cls, info = phase_e3(run_dir, args, hb, route_cache, cand_pols, e2_cls)
        overall["phases"]["E3"] = {"classification": cls, "info": info}
        write_checkpoint(run_dir, "E3", cls, info)
        if cls == "PATHOLOGY":
            overall["mainline_status"] = "STOP_NO_GO"
            overall["furthest_phase"] = "E3"
            overall["furthest_classification"] = "PATHOLOGY"
            overall["plain_english"] = "Stress validation hit NaN/inf pathology and was stopped."
            overall["failure_branch"] = "STOP_PATHOLOGY"
            json_dump(run_dir / "run_manifest.json", overall)
            write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
            hb.stop("stopped")
            return
        if cls != "PASS":
            overall["mainline_status"] = "STOP_NO_GO"
            overall["furthest_phase"] = "E3"
            overall["furthest_classification"] = cls
            overall["plain_english"] = "No candidate retained strict-pass robustness under stress coverage."
            overall["failure_branch"] = cls
            ng = run_dir / "phaseE3" / "no_go_package"
            ng.mkdir(parents=True, exist_ok=True)
            write_text(ng / "failure_diagnosis.md", f"Phase E3 NO_GO: {info.get('reason', 'unknown')}")
            write_text(ng / "next_step_prompt.txt", "Investigate scenario-specific breakpoints and redesign filter controls before new promotion runs.")
            json_dump(run_dir / "run_manifest.json", overall)
            top_fail = pd.read_csv(run_dir / "phaseE3" / "phaseE3_candidate_summary.csv").head(5) if (run_dir / "phaseE3" / "phaseE3_candidate_summary.csv").exists() else None
            write_final_summary(run_dir, overall, top_df=top_fail, next_prompt=None)
            hb.stop("stopped")
            return

    # Phase E4
    if not phase_done(run_dir, "E4"):
        cls, info = phase_e4(run_dir, args, hb, phase_d_dir, cand_pols)
        overall["phases"]["E4"] = {"classification": cls, "info": info}
        write_checkpoint(run_dir, "E4", cls, info)
        if cls != "PASS_GO_PAPER_SHADOW":
            overall["mainline_status"] = "STOP_NO_GO" if cls == "NO_GO" else "STOP_INFRA"
            overall["furthest_phase"] = "E4"
            overall["furthest_classification"] = cls
            overall["plain_english"] = "Paper/shadow package generation failed."
            overall["failure_branch"] = cls
            json_dump(run_dir / "run_manifest.json", overall)
            write_final_summary(run_dir, overall, top_df=None, next_prompt=None)
            hb.stop("stopped")
            return

    overall["mainline_status"] = "CONTINUE"
    overall["furthest_phase"] = "E4"
    overall["furthest_classification"] = "GO_PAPER_SHADOW"
    overall["plain_english"] = "At least one candidate held strict-pass behavior through route perturbation and stress coverage, so paper/shadow promotion package is justified."
    overall["failure_branch"] = "none"
    overall["duration_sec"] = float(time.time() - t0)
    overall["end_utc"] = utc_now()
    json_dump(run_dir / "run_manifest.json", overall)

    top_df = None
    csum_fp = run_dir / "phaseE3" / "phaseE3_candidate_summary.csv"
    if csum_fp.exists():
        try:
            top_df = pd.read_csv(csum_fp).head(5)
        except Exception:
            top_df = None
    next_prompt = (
        "Phase F paper/shadow live-sim confirmation (contract-locked): run 2-4 week shadow tracking for the primary and backup "
        "Phase E policy, monitor rollback triggers daily, and do not alter hard gates or execution mechanics."
    )
    write_text(run_dir / "phaseE_next_prompt.txt", next_prompt)
    write_final_summary(run_dir, overall, top_df=top_df, next_prompt=next_prompt)
    hb.stop("completed")


if __name__ == "__main__":
    main()
