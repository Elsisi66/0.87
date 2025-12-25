from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.bot087.strategy.params import StrategyParams


# -------------------------
# Project root discovery
# -------------------------

def _project_root_from_file() -> Path:
    """
    Find repo root by walking up until we see both /src and /data folders.
    Works on VPS and local.
    """
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data").is_dir() and (p / "src").is_dir():
            return p
    # fallback: src/bot087/strategy/param_store.py -> parents[3] is repo root
    # (repo)/src/bot087/strategy/param_store.py
    try:
        return Path(__file__).resolve().parents[3]
    except Exception as e:
        raise RuntimeError("Could not find project root") from e


PROJECT_ROOT = _project_root_from_file()
PARAMS_DIR = PROJECT_ROOT / "data" / "metadata" / "params"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Helpers
# -------------------------

def _active_path(symbol: str) -> Path:
    return PARAMS_DIR / f"{symbol.upper()}_active_params.json"


def _to_params_dict(p: Union[StrategyParams, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(p, dict):
        return dict(p)
    if is_dataclass(p):
        return asdict(p)
    return dict(vars(p))


def _get(d: Dict[str, Any], k: str, default: Any = None) -> Any:
    v = d.get(k, default)
    return default if v is None else v


def _cycle_list(d: Dict[str, Any], key_prefix: str, fallback: float, n: int = 5) -> List[float]:
    """
    Reads keys like:
      willr_cycle0..4  OR  tp_mult_cycle0..4  etc.
    """
    out: List[float] = []
    for c in range(n):
        v = d.get(f"{key_prefix}{c}", None)
        out.append(float(fallback if v is None else v))
    return out


def _ensure_cycle_lists(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize old JSON formats into StrategyParams expected fields.

    Supports:
      - willr_by_cycle / tp_mult_by_cycle / sl_mult_by_cycle / exit_rsi_by_cycle (list length 5)
      - willr_cycle0..4 / tp_mult_cycle0..4 / sl_mult_cycle0..4 / exit_rsi_cycle0..4 (scalars)
    """
    p = dict(params)

    # Base fallbacks if per-cycle scalars exist but list is missing
    willr_base = float(_get(p, "willr_max", -30.0))
    tp_base = float(_get(p, "profit_target_mult", 1.05))
    sl_base = float(_get(p, "stop_loss_mult", 0.97))
    exit_rsi_base = float(_get(p, "exit_rsi_threshold", 50.0))

    def _ensure_list(field: str, scalar_prefix: str, fallback: float) -> None:
        v = p.get(field, None)
        if isinstance(v, list) and len(v) == 5:
            p[field] = [float(x) for x in v]
            return
        # try scalar-per-cycle
        p[field] = _cycle_list(p, scalar_prefix, fallback)

    _ensure_list("willr_by_cycle", "willr_cycle", willr_base)
    _ensure_list("tp_mult_by_cycle", "tp_mult_cycle", tp_base)
    _ensure_list("sl_mult_by_cycle", "sl_mult_cycle", sl_base)
    _ensure_list("exit_rsi_by_cycle", "exit_rsi_cycle", exit_rsi_base)

    # allow_hours should be list or None
    ah = p.get("allow_hours", None)
    if ah is None:
        p["allow_hours"] = []
    elif not isinstance(ah, list):
        p["allow_hours"] = list(ah)  # best effort

    return p


# -------------------------
# Public API
# -------------------------

def load_active_params(symbol: str) -> StrategyParams:
    """
    Loads active params from:
      data/metadata/params/<SYMBOL>_active_params.json

    Accepts formats:
      - {"params": {...}, "meta": {...}}
      - flat dict {...}
    If missing -> returns StrategyParams() defaults.
    """
    path = _active_path(symbol)
    if not path.exists():
        return StrategyParams()

    raw = json.loads(path.read_text(encoding="utf-8"))
    params_dict = raw.get("params", raw) if isinstance(raw, dict) else {}
    if not isinstance(params_dict, dict):
        params_dict = {}

    params_dict = _ensure_cycle_lists(params_dict)

    # filter unknown keys so old json doesn't explode
    valid = set(StrategyParams.__dataclass_fields__.keys())
    cleaned = {k: v for k, v in params_dict.items() if k in valid}

    return StrategyParams(**cleaned)


def save_active_params(
    symbol: str,
    params: Union[StrategyParams, Dict[str, Any]],
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save active params to:
      data/metadata/params/<SYMBOL>_active_params.json
    Returns the filepath written.
    """
    path = _active_path(symbol)

    payload = {
        "symbol": symbol.upper(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "params": _to_params_dict(params),
        "meta": meta or {},
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


# --- Compatibility alias (older code) ---
def save_params(symbol: str, params: Union[StrategyParams, Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> str:
    return save_active_params(symbol, params, meta=meta)
