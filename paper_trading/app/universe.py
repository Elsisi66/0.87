from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Settings
from .utils.io import atomic_write_json
from .utils.time_utils import utc_iso


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def _resolve_path(project_root: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    p = Path(path_value)
    if p.exists():
        return p.resolve()
    rel = (project_root / path_value).resolve()
    if rel.exists():
        return rel
    return None


def _resolve_path_from_root(root: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    p = Path(path_value)
    if p.exists():
        return p.resolve()
    rel = (root / path_value).resolve()
    if rel.exists():
        return rel
    return None


def _guess_quote_asset(symbol: str) -> str:
    for quote in ["USDT", "BUSD", "USDC", "EUR", "BTC", "ETH"]:
        if symbol.endswith(quote):
            return quote
    return "UNKNOWN"


def _find_best_params_for_symbol(project_root: Path, symbol: str) -> Path | None:
    params_dir = project_root / "data" / "metadata" / "params"
    patterns = [
        f"{symbol}__UNIVERSE_LONG_active_params.json",
        f"{symbol}__UNIVERSE_LONG_C1_active_params.json",
        f"{symbol}__UNIVERSE_LONG_C3_active_params.json",
        f"{symbol}_C13_active_params_long.json",
        f"{symbol}_active_params.json",
        f"{symbol}_active_params_long.json",
    ]
    for name in patterns:
        candidate = params_dir / name
        if candidate.exists():
            return candidate.resolve()

    wildcard_hits = sorted(params_dir.glob(f"{symbol}*active_params*json"))
    for hit in wildcard_hits:
        if "short" in hit.name.lower():
            continue
        return hit.resolve()
    return None


def _canonical_from_best_by_symbol(project_root: Path) -> tuple[list[dict[str, str]], str] | None:
    base = project_root / "reports" / "params_scan"
    if not base.exists():
        return None

    candidates = sorted(base.glob("*/best_by_symbol.csv"), key=lambda p: p.parent.name, reverse=True)
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "symbol" not in df.columns or "pass" not in df.columns:
            continue

        x = df.copy()
        if "side" in x.columns:
            x = x[x["side"].astype(str).str.lower() == "long"]
        x = x[x["pass"].map(_truthy)]
        if x.empty:
            continue

        rows: list[dict[str, str]] = []
        for _, row in x.sort_values("symbol").iterrows():
            symbol = str(row["symbol"]).strip().upper()
            params_path = _resolve_path(project_root, str(row.get("params_file", "")))
            if params_path is None:
                params_path = _find_best_params_for_symbol(project_root, symbol)
            if params_path is None:
                continue
            rows.append({"symbol": symbol, "params_path": str(params_path)})

        if rows:
            dedup = {row["symbol"]: row for row in rows}
            return list(dedup.values()), str(path.resolve())
    return None


def _canonical_from_universe_summary(project_root: Path) -> tuple[list[dict[str, str]], str] | None:
    base = project_root / "artifacts" / "reports"
    if not base.exists():
        return None

    candidates = sorted(base.glob("universe_*/summary.csv"), key=lambda p: p.parent.name, reverse=True)
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "symbol" not in df.columns:
            continue

        x = df.copy()
        if "side" in x.columns:
            x = x[x["side"].astype(str).str.lower() == "long"]

        status_col = "PASS/FAIL" if "PASS/FAIL" in x.columns else ("pass" if "pass" in x.columns else None)
        if status_col is None:
            continue

        if status_col == "PASS/FAIL":
            x = x[x[status_col].astype(str).str.upper() == "PASS"]
        else:
            x = x[x[status_col].map(_truthy)]

        if x.empty:
            continue

        rows: list[dict[str, str]] = []
        for _, row in x.sort_values("symbol").iterrows():
            symbol = str(row["symbol"]).strip().upper()
            params_path = _resolve_path(project_root, str(row.get("param_path", "")))
            if params_path is None:
                params_path = _resolve_path(project_root, str(row.get("params_file", "")))
            if params_path is None:
                params_path = _find_best_params_for_symbol(project_root, symbol)
            if params_path is None:
                continue
            rows.append({"symbol": symbol, "params_path": str(params_path)})

        if rows:
            dedup = {row["symbol"]: row for row in rows}
            return list(dedup.values()), str(path.resolve())
    return None


def _from_whitelist(project_root: Path) -> tuple[list[dict[str, str]], str] | None:
    candidates = [
        project_root / "configs" / "universe_whitelist.txt",
        project_root / "configs" / "universe_whitelist.csv",
        project_root / "paper_trading" / "config" / "universe_whitelist.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue
        symbols: list[str] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip().upper()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                symbols.extend([x.strip().upper() for x in line.split(",") if x.strip()])
            else:
                symbols.append(line)

        rows: list[dict[str, str]] = []
        for symbol in sorted(set(symbols)):
            params_path = _find_best_params_for_symbol(project_root, symbol)
            if params_path is None:
                continue
            rows.append({"symbol": symbol, "params_path": str(params_path)})

        if rows:
            return rows, str(path.resolve())
    return None


def _from_manual_fallback(settings: Settings) -> tuple[list[dict[str, str]], str]:
    path = settings.config_dir / "manual_universe.txt"
    if not path.exists():
        path.write_text("SOLUSDT\n", encoding="utf-8")

    symbols: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip().upper()
        if not line or line.startswith("#"):
            continue
        symbols.append(line)

    rows: list[dict[str, str]] = []
    for symbol in sorted(set(symbols)):
        params_path = _find_best_params_for_symbol(settings.project_root, symbol)
        if params_path is None:
            continue
        rows.append({"symbol": symbol, "params_path": str(params_path)})

    return rows, str(path.resolve())


def _from_repaired_posture_pack(settings: Settings) -> tuple[list[dict[str, str]], str, str] | None:
    freeze_dir = _resolve_path_from_root(settings.project_root, settings.repaired_posture_freeze_dir)
    subset_csv = _resolve_path_from_root(settings.project_root, settings.repaired_active_subset_csv)
    params_dir = _resolve_path_from_root(settings.project_root, settings.repaired_active_params_dir)
    if freeze_dir is None or subset_csv is None or params_dir is None:
        return None
    if not subset_csv.exists() or not params_dir.exists():
        return None

    df = pd.read_csv(subset_csv)
    if "symbol" not in df.columns:
        return None

    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        posture = str(row.get("posture", "ACTIVE")).strip().upper()
        if posture != "ACTIVE":
            continue
        winner_id = str(row.get("winner_config_id", "")).strip()
        params_path = params_dir / f"{symbol}_repaired_selected_params.json"
        if not params_path.exists():
            continue
        rows.append(
            {
                "symbol": symbol,
                "params_path": str(params_path.resolve()),
                "winner_config_id": winner_id,
            }
        )

    if not rows:
        return None
    dedup = {row["symbol"]: row for row in rows}
    return list(dedup.values()), str(subset_csv.resolve()), str(freeze_dir.resolve())


@dataclass
class ResolvedUniverse:
    source_priority: str
    source_path: str
    symbols: list[str]
    symbol_params: dict[str, str]
    quote_assets: dict[str, str]
    unresolved_symbols: list[str]
    winner_config_ids: dict[str, str]
    posture_freeze_dir: str | None = None


def resolve_universe(settings: Settings) -> ResolvedUniverse:
    project_root = settings.project_root

    posture_resolution = _from_repaired_posture_pack(settings)
    if settings.require_repaired_posture_pack:
        if posture_resolution is None:
            raise RuntimeError(
                "repaired posture pack required but not resolvable: "
                f"freeze_dir={settings.repaired_posture_freeze_dir} "
                f"active_subset={settings.repaired_active_subset_csv} "
                f"active_params_dir={settings.repaired_active_params_dir}"
            )
        rows, source_path, freeze_dir = posture_resolution
        source_priority = "repaired_posture_active_subset"
    elif posture_resolution is not None:
        rows, source_path, freeze_dir = posture_resolution
        source_priority = "repaired_posture_active_subset"
    else:
        freeze_dir = None
        resolution = _canonical_from_best_by_symbol(project_root)
        source_priority = "canonical_passed_report"

        if resolution is None:
            resolution = _canonical_from_universe_summary(project_root)
            source_priority = "best_by_symbol_or_ranking"

        if resolution is None:
            resolution = _from_whitelist(project_root)
            source_priority = "config_whitelist"

        if resolution is None:
            resolution = _from_manual_fallback(settings)
            source_priority = "manual_fallback"

        rows, source_path = resolution

    symbol_params: dict[str, str] = {}
    unresolved: list[str] = []
    winner_config_ids: dict[str, str] = {}
    for row in rows:
        symbol = row["symbol"].upper()
        params_path = Path(row["params_path"])
        if not params_path.exists():
            unresolved.append(symbol)
            continue
        symbol_params[symbol] = str(params_path.resolve())
        winner_config_ids[symbol] = str(row.get("winner_config_id", "")).strip()

    symbols = sorted(symbol_params.keys())
    quote_assets = {symbol: _guess_quote_asset(symbol) for symbol in symbols}

    resolved = ResolvedUniverse(
        source_priority=source_priority,
        source_path=source_path,
        symbols=symbols,
        symbol_params=symbol_params,
        quote_assets=quote_assets,
        unresolved_symbols=sorted(set(unresolved)),
        winner_config_ids=winner_config_ids,
        posture_freeze_dir=freeze_dir,
    )

    payload = {
        "generated_utc": utc_iso(),
        "source_priority": resolved.source_priority,
        "source_path": resolved.source_path,
        "posture_freeze_dir": resolved.posture_freeze_dir,
        "symbols": resolved.symbols,
        "symbol_params": resolved.symbol_params,
        "winner_config_ids": resolved.winner_config_ids,
        "quote_assets": resolved.quote_assets,
        "unresolved_symbols": resolved.unresolved_symbols,
    }
    atomic_write_json(settings.config_dir / "resolved_universe.json", payload)
    return resolved
