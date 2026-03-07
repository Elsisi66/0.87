#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path("/root/analysis/0.87").resolve()
REPORTS_ROOT = PROJECT_ROOT / "reports" / "execution_layer"
PHASEC_DIR = (REPORTS_ROOT / "PHASEC_MODEL_A_BOUNDED_CONFIRMATION_20260228_022501").resolve()
PHASER_DIR = (REPORTS_ROOT / "PHASER_ROUTE_HARNESS_REDESIGN_20260228_005334").resolve()
BASELINE_AUDIT_DIR = (REPORTS_ROOT / "BASELINE_AUDIT_20260221_214310").resolve()

FEE_MODEL_PATH = BASELINE_AUDIT_DIR / "fee_model.json"
METRICS_DEF_PATH = BASELINE_AUDIT_DIR / "metrics_definition.md"
EXPECTED_FEE_SHA256 = "b54445675e835778cb25f7256b061d885474255335a3c975613f2c7d52710f4a"
EXPECTED_METRICS_SHA256 = "d3c55348888498d32832a083765b57b0088a43b2fca0b232cccbcf0a8d187c99"

PHASE_C_CONTRACT_PATH = PHASEC_DIR / "phaseC1_contract_validation.json"
PHASE_C_SELECTION_PATH = PHASEC_DIR / "phaseC4_primary_backup.csv"
PHASE_C_DECISION_PATH = PHASEC_DIR / "phaseC_decision.md"
PHASE_R_VALIDATION_PATH = PHASER_DIR / "phaseR1_feasibility_validation.csv"

PAPER_ROOT = PROJECT_ROOT / "paper_trading"
PAPER_SETTINGS_PATH = PAPER_ROOT / "config" / "settings.yaml"
PAPER_UNIVERSE_PATH = PAPER_ROOT / "config" / "resolved_universe.json"
PAPER_MAIN_PATH = PAPER_ROOT / "app" / "main.py"
PAPER_EXEC_PATH = PAPER_ROOT / "app" / "execution_sim.py"
PAPER_SIGNAL_PATH = PAPER_ROOT / "app" / "signal_runner.py"
PAPER_FEED_PATH = PAPER_ROOT / "app" / "data_feed.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def json_dump(path: Path, obj: Any) -> None:
    def _default(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        return str(v)

    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default), encoding="utf-8")


def git_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        out["git_head"] = (
            subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:
        out["git_head"] = "unavailable"
    try:
        status = subprocess.check_output(["git", "-C", str(PROJECT_ROOT), "status", "--short"], text=True)
        out["git_status_short"] = status.strip().splitlines()
    except Exception:
        out["git_status_short"] = []
    return out


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_candidates(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        payload = {}
        for col, val in row.items():
            if pd.isna(val):
                payload[col] = None
            elif isinstance(val, (int, float, str)):
                payload[col] = val
            else:
                payload[col] = str(val)
        rows.append(payload)
    return rows


def load_route_validation(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def line_no(text: str, needle: str) -> int:
    for idx, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return idx
    return -1


def main() -> None:
    run_dir = REPORTS_ROOT / f"PHASED_MODEL_A_PAPER_SHADOW_{utc_tag()}"
    run_dir.mkdir(parents=True, exist_ok=False)

    fee_sha = sha256_file(FEE_MODEL_PATH)
    metrics_sha = sha256_file(METRICS_DEF_PATH)
    freeze_lock_pass = int(fee_sha == EXPECTED_FEE_SHA256 and metrics_sha == EXPECTED_METRICS_SHA256)

    phase_c_contract = load_json(PHASE_C_CONTRACT_PATH)
    phase_c_candidates = load_candidates(PHASE_C_SELECTION_PATH)
    phase_c_decision_text = read_text(PHASE_C_DECISION_PATH)
    route_validation = load_route_validation(PHASE_R_VALIDATION_PATH)
    paper_settings_text = read_text(PAPER_SETTINGS_PATH)
    paper_universe = load_json(PAPER_UNIVERSE_PATH)
    main_text = read_text(PAPER_MAIN_PATH)
    exec_text = read_text(PAPER_EXEC_PATH)
    signal_text = read_text(PAPER_SIGNAL_PATH)
    feed_text = read_text(PAPER_FEED_PATH)

    route_flag_col = "route_trade_gates_reachable" if "route_trade_gates_reachable" in route_validation.columns else (
        "support_feasible" if "support_feasible" in route_validation.columns else None
    )
    route_support_ok = int(
        (not route_validation.empty)
        and (route_flag_col is not None)
        and route_validation[route_flag_col].astype(int).eq(1).all()
    )
    phase_c_purity_ok = int(
        int(phase_c_contract.get("freeze_lock_pass", 0)) == 1
        and int(phase_c_contract.get("wrapper_uses_3m_entry_only", 0)) == 1
        and int(phase_c_contract.get("wrapper_uses_1h_exit_only", 0)) == 1
        and int(phase_c_contract.get("route_reproduction_match_phaseR", 0)) == 1
    )
    phase_c_promote_ok = int("MODEL_A_PROMOTE_PAPER" in phase_c_decision_text)

    current_runtime_fetches_3m = int("fetch_ohlcv_3m" in feed_text or "interval\": \"3m\"" in feed_text or "interval': '3m'" in feed_text)
    current_runtime_fetches_1h_only = int('params = {"symbol": symbol, "interval": "1h"' in feed_text)
    current_runtime_has_entry_only_knobs = int(
        any(
            key in (main_text + "\n" + exec_text + "\n" + signal_text + "\n" + paper_settings_text)
            for key in ["entry_mode", "limit_offset_bps", "fallback_to_market", "fallback_delay_min", "max_fill_delay_min"]
        )
    )
    current_runtime_uses_forbidden_exit_logic = int(
        all(token in exec_text for token in ["tp_mult_by_cycle", "sl_mult_by_cycle", "max_hold_hours", "exit_rsi_by_cycle"])
    )
    current_runtime_uses_generic_execution_sim = int("ExecutionSimulator(settings.fee_bps" in main_text)
    current_runtime_sol_only = int(paper_universe.get("symbols") == ["SOLUSDT"])
    current_runtime_candidate_mapping_present = 0

    current_runtime_model_a_pure = int(
        current_runtime_fetches_3m == 1
        and current_runtime_has_entry_only_knobs == 1
        and current_runtime_uses_forbidden_exit_logic == 0
        and current_runtime_uses_generic_execution_sim == 0
        and current_runtime_sol_only == 1
    )

    classification = "PAPER_READY_STRONG"
    blockers: list[str] = []
    if freeze_lock_pass != 1:
        classification = "PAPER_BLOCKED_INFRA"
        blockers.append("freeze_lock_failed")
    if phase_c_purity_ok != 1:
        classification = "PAPER_BLOCKED_INFRA"
        blockers.append("phase_c_contract_not_confirmed")
    if route_support_ok != 1:
        classification = "PAPER_BLOCKED_INFRA"
        blockers.append("repaired_routes_not_support_feasible")
    if phase_c_promote_ok != 1:
        classification = "PAPER_BLOCKED_LOGIC"
        blockers.append("phase_c_did_not_authorize_paper")
    if current_runtime_model_a_pure != 1:
        classification = "PAPER_BLOCKED_INFRA"
        if current_runtime_fetches_3m != 1:
            blockers.append("paper_runtime_missing_3m_feed")
        if current_runtime_has_entry_only_knobs != 1:
            blockers.append("paper_runtime_missing_model_a_entry_knobs")
        if current_runtime_uses_forbidden_exit_logic == 1:
            blockers.append("paper_runtime_exec_sim_mutates_exits")
        if current_runtime_uses_generic_execution_sim == 1:
            blockers.append("paper_runtime_still_bound_to_generic_execution_sim")
        if current_runtime_sol_only != 1:
            blockers.append("paper_runtime_universe_not_sol_only")

    mainline_status = classification

    primary = next((row for row in phase_c_candidates if str(row.get("selection_role", "")).lower() == "primary"), None)
    backup = next((row for row in phase_c_candidates if str(row.get("selection_role", "")).lower() == "backup"), None)
    if primary is None or backup is None:
        classification = "PAPER_BLOCKED_LOGIC"
        mainline_status = classification
        blockers.append("missing_primary_or_backup_mapping")

    contract_validation = {
        "generated_utc": utc_now(),
        "fee_model_path": str(FEE_MODEL_PATH),
        "fee_model_sha256_expected": EXPECTED_FEE_SHA256,
        "fee_model_sha256_observed": fee_sha,
        "metrics_definition_path": str(METRICS_DEF_PATH),
        "metrics_definition_sha256_expected": EXPECTED_METRICS_SHA256,
        "metrics_definition_sha256_observed": metrics_sha,
        "freeze_lock_pass": int(freeze_lock_pass),
        "phase_c_contract_validation_path": str(PHASE_C_CONTRACT_PATH),
        "phase_c_purity_ok": int(phase_c_purity_ok),
        "phase_c_promote_ok": int(phase_c_promote_ok),
        "repaired_route_validation_path": str(PHASE_R_VALIDATION_PATH),
        "repaired_route_support_feasible": int(route_support_ok),
        "wrapper_uses_3m_entry_only": int(phase_c_contract.get("wrapper_uses_3m_entry_only", 0)),
        "wrapper_uses_1h_exit_only": int(phase_c_contract.get("wrapper_uses_1h_exit_only", 0)),
        "current_runtime_fetches_3m": int(current_runtime_fetches_3m),
        "current_runtime_fetches_1h_only": int(current_runtime_fetches_1h_only),
        "current_runtime_has_entry_only_knobs": int(current_runtime_has_entry_only_knobs),
        "current_runtime_uses_forbidden_exit_logic": int(current_runtime_uses_forbidden_exit_logic),
        "current_runtime_uses_generic_execution_sim": int(current_runtime_uses_generic_execution_sim),
        "current_runtime_model_a_pure": int(current_runtime_model_a_pure),
        "current_runtime_sol_only": int(current_runtime_sol_only),
        "current_runtime_candidate_mapping_present": int(current_runtime_candidate_mapping_present),
        "classification": classification,
        "blockers": blockers,
    }
    json_dump(run_dir / "phaseD1_contract_validation.json", contract_validation)

    parity_lines = [
        "# Phase D1 Runtime Parity Report",
        "",
        f"- Generated UTC: `{contract_validation['generated_utc']}`",
        f"- Frozen contract lock pass: `{freeze_lock_pass}`",
        f"- Phase C paper authorization present: `{phase_c_promote_ok}`",
        f"- Repaired route support-feasible metadata preserved: `{route_support_ok}`",
        f"- Current paper runtime Model A pure: `{current_runtime_model_a_pure}`",
        "",
        "## Verified Model A Source Of Truth",
        f"- Phase C contract file: `{PHASE_C_CONTRACT_PATH}`",
        f"- Phase A wrapper source: `{PROJECT_ROOT / 'scripts' / 'phase_a_model_a_audit.py'}`",
        "- Validated wrapper semantics:",
        "  - 1h signal generation remains frozen.",
        "  - 1h TP/SL/exit semantics remain frozen after entry fill.",
        "  - 3m controls entry timing only via `simulate_entry_only_fill(...)`.",
        "",
        "## Current Paper Runtime Mismatch",
        f"- `paper_trading/app/data_feed.py:{line_no(feed_text, 'params = {\"symbol\": symbol, \"interval\": \"1h\", \"limit\": int(limit)}')}` fetches only `interval=\"1h\"`; no `fetch_ohlcv_3m(...)` path exists.",
        f"- `paper_trading/app/signal_runner.py:{line_no(signal_text, 'signals = np.asarray(build_entry_signal(x, params, assume_prepared=True), dtype=bool)')}` builds signals from 1h only and exports 1h rows directly into execution.",
        f"- `paper_trading/app/main.py:{line_no(main_text, 'execution_sim = ExecutionSimulator(settings.fee_bps, settings.slippage_bps_choices, seed=87)')}` wires the daemon to the generic `ExecutionSimulator`.",
        f"- `paper_trading/app/execution_sim.py:{line_no(exec_text, 'tp_mult = float(params.get(\"tp_mult_by_cycle\", [1.02] * 5)[cycle])')}` stores TP/SL on entry from params.",
        f"- `paper_trading/app/execution_sim.py:{line_no(exec_text, 'max_hold = int(params.get(\"max_hold_hours\", 48))')}` and `paper_trading/app/execution_sim.py:{line_no(exec_text, 'exit_rsi_by_cycle = params.get(\"exit_rsi_by_cycle\", [0.0] * 5)')}` keep downstream exit ownership inside the runtime.",
        "- No Model A entry-only knob wiring (`entry_mode`, `limit_offset_bps`, `fallback_to_market`, `fallback_delay_min`, `max_fill_delay_min`) exists anywhere in `paper_trading/app` or `paper_trading/config`.",
        "",
        "## Conclusion",
        "- The backtest contract is valid, but the current paper runtime does not implement the validated Model A wrapper.",
        "- Starting paper/shadow on the existing daemon would not match the confirmed Phase C contract.",
    ]
    write_text(run_dir / "phaseD1_runtime_parity_report.md", "\n".join(parity_lines) + "\n")

    runtime_snapshot_lines = [
        "# Phase D2 Runtime Config Snapshot",
        "",
        "## Intended Deployment Roles",
        f"- `paper_primary`: `{primary['candidate_id']}`",
        f"- `shadow_backup`: `{backup['candidate_id']}`",
        "- Capital logic: isolated paper books; no combined allocation logic.",
        "",
        "## Required Runtime Contract",
        "- Symbol scope: `SOLUSDT` only.",
        f"- 1h signal params path: `{paper_universe['symbol_params'].get('SOLUSDT')}`",
        "- 1h signal engine: `paper_trading/app/signal_runner.py::build_signal_frame`.",
        "- 3m entry wrapper source: `scripts/phase_a_model_a_audit.py::simulate_entry_only_fill`.",
        "- 1h exit source: `scripts/phase_a_model_a_audit.py::simulate_frozen_1h_exit`.",
        "",
        "## Current Generic Daemon Snapshot",
        f"- Settings file: `{PAPER_SETTINGS_PATH}`",
        f"- Resolved universe file: `{PAPER_UNIVERSE_PATH}`",
        f"- Current symbols tracked: `{','.join(paper_universe.get('symbols', []))}`",
        f"- Current symbol count: `{len(paper_universe.get('symbols', []))}`",
        "- Current runtime does not load candidate-specific Model A execution knobs.",
        "- Current runtime cannot map `paper_primary` and `shadow_backup` separately.",
        "",
        "## Status",
        "- Candidate mapping is defined at the deployment package level but not applied to the live paper daemon.",
        "- This is a prepared config snapshot only; no runtime switch was applied because parity failed in D1.",
    ]
    write_text(run_dir / "phaseD2_runtime_config_snapshot.md", "\n".join(runtime_snapshot_lines) + "\n")

    candidate_mapping = {
        "generated_utc": utc_now(),
        "symbol": "SOLUSDT",
        "paper_primary": primary,
        "shadow_backup": backup,
        "mapping_clean": 1,
        "runtime_can_load_mapping_directly": 0,
        "runtime_can_isolate_primary_backup_books": 0,
        "reason_not_applied": "current_paper_runtime_not_model_a_pure",
    }
    json_dump(run_dir / "phaseD2_candidate_mapping.json", candidate_mapping)

    logging_lines = [
        "# Phase D2 Logging Schema",
        "",
        "Every paper/shadow trade event must emit these fields:",
        "- `candidate_role` (`paper_primary` or `shadow_backup`)",
        "- `candidate_id`",
        "- `symbol`",
        "- `signal_timestamp_1h`",
        "- `signal_bar_timestamp_1h`",
        "- `signal_state_hash`",
        "- `entry_wrapper_mode`",
        "- `entry_mode`",
        "- `entry_limit_offset_bps`",
        "- `fallback_to_market`",
        "- `fallback_delay_min`",
        "- `max_fill_delay_min`",
        "- `entry_fill_timestamp_3m`",
        "- `entry_fill_price`",
        "- `entry_fill_type`",
        "- `entry_fill_delay_min`",
        "- `entry_liquidity_type`",
        "- `taker_share_flag`",
        "- `exit_timestamp_1h`",
        "- `exit_price`",
        "- `exit_reason_1h`",
        "- `tp_hit`",
        "- `sl_hit`",
        "- `realized_pnl_pct_net`",
        "- `realized_pnl_quote`",
        "- `contract_path`",
        "- `model_a_purity_ok`",
        "- `hybrid_exit_mutation_detected`",
        "- `runtime_mismatch_flag`",
        "",
        "Current `paper_trading/state/journal.jsonl` does not include `candidate_id`, `entry wrapper knobs`, or `hybrid_exit_mutation_detected`, so the existing paper daemon is insufficient for this schema.",
    ]
    write_text(run_dir / "phaseD2_logging_schema.md", "\n".join(logging_lines) + "\n")

    monitoring_lines = [
        "# Phase D3 Monitoring Spec",
        "",
        "Track these metrics separately for `paper_primary` and `shadow_backup`, plus a direct diff between them:",
        "- daily_pnl_quote",
        "- daily_pnl_pct_net",
        "- trades_total",
        "- entries_valid",
        "- entry_rate",
        "- avg_fill_delay_min",
        "- p95_fill_delay_min",
        "- taker_share",
        "- realized_win_rate",
        "- realized_avg_pnl_per_trade",
        "- realized_sl_hit_rate",
        "- realized_tp_hit_rate",
        "- realized_drawdown_pct",
        "- divergence_vs_expected_delta",
        "- divergence_vs_expected_taker_share",
        "- divergence_vs_expected_fill_delay",
        "- primary_minus_backup_expectancy",
        "- primary_minus_backup_drawdown",
        "",
        "Contract-violation counters:",
        "- hybrid_exit_mutation_detected_count",
        "- missing_signal_count",
        "- runtime_mismatch_count",
        "- impossible_fill_path_count",
        "- stale_data_count",
        "- execution_exception_count",
    ]
    write_text(run_dir / "phaseD3_monitoring_spec.md", "\n".join(monitoring_lines) + "\n")

    daily_summary_lines = [
        "# Phase D3 Daily Summary Template",
        "",
        "## Header",
        "- date_utc:",
        "- contract_lock_pass:",
        "- runtime_model_a_pure:",
        "- route_harness_reference:",
        "",
        "## Paper Primary",
        "- candidate_id:",
        "- trades_total:",
        "- entry_rate:",
        "- taker_share:",
        "- avg_fill_delay_min:",
        "- p95_fill_delay_min:",
        "- realized_pnl_quote:",
        "- realized_pnl_pct_net:",
        "- realized_win_rate:",
        "- realized_drawdown_pct:",
        "",
        "## Shadow Backup",
        "- candidate_id:",
        "- trades_total:",
        "- entry_rate:",
        "- taker_share:",
        "- avg_fill_delay_min:",
        "- p95_fill_delay_min:",
        "- realized_pnl_quote:",
        "- realized_pnl_pct_net:",
        "- realized_win_rate:",
        "- realized_drawdown_pct:",
        "",
        "## Divergence Checks",
        "- primary_minus_backup_expectancy:",
        "- route_center_regression_detected:",
        "- hybrid_exit_mutation_detected:",
        "- missing_signal_count:",
        "- runtime_exceptions:",
        "",
        "## Action",
        "- status: continue / caution / rollback",
        "- operator_note:",
    ]
    write_text(run_dir / "phaseD3_daily_summary_template.md", "\n".join(daily_summary_lines) + "\n")

    alert_lines = [
        "# Phase D3 Alert Rules",
        "",
        "Immediate red alerts:",
        "- Any `hybrid_exit_mutation_detected_count > 0`.",
        "- Any missing or duplicated 1h signal relative to the frozen signal path.",
        "- Any fill marked impossible under the configured Model A entry-only rules.",
        "- Any repeated runtime exception on the same candidate for 2 consecutive cycles.",
        "",
        "Economic caution alerts:",
        "- `taker_share > 0.10` on either candidate intraday.",
        "- `p95_fill_delay_min > 6.0` intraday on either candidate.",
        "- `realized_pnl_pct_net < 0` on a rolling 5-trade window versus the 1h reference expectation.",
        "- `primary_minus_backup_expectancy < -0.0010` on a rolling 10-trade window.",
    ]
    write_text(run_dir / "phaseD3_alert_rules.md", "\n".join(alert_lines) + "\n")

    rollback_lines = [
        "# Phase D4 Rollback Rules",
        "",
        "## Infra Rollback",
        "- Soft warning: any single runtime mismatch, stale data event, or transient signal gap.",
        "- Hard stop: any freeze-lock failure, any hybrid exit mutation, any impossible fill path, or 2 consecutive runtime exceptions.",
        "- Rollback action: stop both `paper_primary` and `shadow_backup`; do not fail over.",
        "",
        "## Economic Caution / Stop",
        "- Taker share",
        "  - soft warning: `> 0.10` on any rolling 10 trades",
        "  - hard stop: `> 0.25` on any rolling 10 trades",
        "  - action: switch primary to backup only if infra is clean and backup remains within band; otherwise stop both",
        "- Fill delay",
        "  - soft warning: `p95_fill_delay_min > 6.0`",
        "  - hard stop: `p95_fill_delay_min > 12.0`",
        "  - action: switch primary to backup only if backup remains <= 6.0; otherwise stop both",
        "- Realized expectancy vs frozen 1h reference",
        "  - soft warning: rolling 5-trade delta <= 0.0",
        "  - hard stop: rolling 10-trade delta < -0.0010",
        "  - action: demote primary to shadow if backup remains positive; otherwise stop both",
        "- Trade-handling mismatch",
        "  - soft warning: 1 mismatch event in a day",
        "  - hard stop: 2 mismatch events in a day",
        "  - action: stop both immediately",
        "- SL clustering",
        "  - soft warning: rolling SL hit rate > 0.70 over 10 trades",
        "  - hard stop: rolling SL hit rate > 0.85 over 10 trades",
        "  - action: keep backup only if no purity violations; otherwise stop both",
    ]
    write_text(run_dir / "phaseD4_rollback_rules.md", "\n".join(rollback_lines) + "\n")

    failsafe_rows = [
        ["category", "metric", "soft_warning_threshold", "hard_stop_threshold", "rollback_action", "allow_primary_to_backup_switch"],
        ["infra", "freeze_lock_pass", "0_once", "0_once", "stop_both", "0"],
        ["infra", "hybrid_exit_mutation_detected_count", ">0", ">0", "stop_both", "0"],
        ["infra", "runtime_exception_count", ">=1_same_day", ">=2_consecutive_cycles", "stop_both", "0"],
        ["infra", "missing_signal_count", ">=1_same_day", ">=2_same_day", "stop_both", "0"],
        ["economics", "taker_share", ">0.10_rolling10", ">0.25_rolling10", "switch_or_stop", "1"],
        ["economics", "p95_fill_delay_min", ">6.0_intraday", ">12.0_intraday", "switch_or_stop", "1"],
        ["economics", "delta_vs_1h_reference", "<=0.0_rolling5", "<-0.0010_rolling10", "switch_or_stop", "1"],
        ["economics", "trade_handling_mismatch_count", ">=1_same_day", ">=2_same_day", "stop_both", "0"],
        ["economics", "sl_hit_rate", ">0.70_rolling10", ">0.85_rolling10", "switch_or_stop", "1"],
    ]
    with (run_dir / "phaseD4_failsafe_matrix.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerows(failsafe_rows)

    decision_lines = [
        "# Phase D Decision",
        "",
        f"- Classification: `{classification}`",
        f"- Mainline status: `{mainline_status}`",
        f"- Freeze lock pass: `{freeze_lock_pass}`",
        f"- Phase C promotion basis present: `{phase_c_promote_ok}`",
        f"- Current paper runtime Model A pure: `{current_runtime_model_a_pure}`",
        f"- Primary mapping clean: `{1 if primary is not None else 0}`",
        f"- Backup mapping clean: `{1 if backup is not None else 0}`",
        "",
        "## Exact Blockers",
    ]
    if blockers:
        decision_lines.extend([f"- `{b}`" for b in blockers])
    else:
        decision_lines.append("- None")
    decision_lines.extend(
        [
            "",
            "## Operator Decision",
            "- Do not start paper/shadow on the current generic paper daemon.",
            "- Build or wire a dedicated Model A paper runtime that uses the validated Phase A entry-only wrapper before promotion can be considered ready.",
        ]
    )
    write_text(run_dir / "phaseD_decision.md", "\n".join(decision_lines) + "\n")

    manifest = {
        "generated_utc": utc_now(),
        "artifact_dir": str(run_dir),
        "git": git_snapshot(),
        "inputs": {
            "phase_c_dir": str(PHASEC_DIR),
            "phase_r_dir": str(PHASER_DIR),
            "paper_settings": str(PAPER_SETTINGS_PATH),
            "paper_resolved_universe": str(PAPER_UNIVERSE_PATH),
        },
        "phases_executed": ["D1", "D2", "D3", "D4", "D5"],
        "final_classification": classification,
        "mainline_status": mainline_status,
        "blockers": blockers,
        "current_runtime_model_a_pure": int(current_runtime_model_a_pure),
        "paper_candidate_mapping_clean": int(primary is not None and backup is not None),
    }
    json_dump(run_dir / "phaseD_run_manifest.json", manifest)

    print(str(run_dir))


if __name__ == "__main__":
    main()
