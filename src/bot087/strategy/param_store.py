import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.bot087.strategy.params import StrategyParams


def _project_root_from_file() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "data").is_dir() and (p / "src").is_dir():
            return p
    raise RuntimeError("Could not find project root")


def _get(d: Dict[str, Any], k: str, default: Any = None) -> Any:
    v = d.get(k, default)
    return default if v is None else v


def _cycle_list(d: Dict[str, Any], key_prefix: str, fallback: float, n: int = 5) -> List[float]:
    out: List[float] = []
    for c in range(n):
        v = d.get(f"{key_prefix}{c}", None)
        out.append(float(fallback if v is None else v))
    return out


def load_active_params(symbol: str) -> StrategyParams:
    root = _project_root_from_file()
    fp = root / "artifacts" / "params" / symbol.upper() / "active.json"
    if not fp.exists():
        raise FileNotFoundError(f"Missing params file: {fp}")

    raw = json.loads(fp.read_text(encoding="utf-8"))
    p = raw.get("best_params", raw)  # supports both formats

    # Base fallbacks
    willr_base = float(_get(p, "willr_max", -30.0))
    tp_base = float(_get(p, "profit_target_mult", 1.05))
    sl_base = float(_get(p, "stop_loss_mult", 0.97))
    exit_rsi_base = float(_get(p, "exit_rsi_threshold", 50.0))

    return StrategyParams(
        entry_rsi_min=float(p["entry_rsi_min"]),
        entry_rsi_max=float(p["entry_rsi_max"]),
        entry_rsi_buffer=float(_get(p, "entry_rsi_buffer", 0.0)),

        willr_floor=float(_get(p, "willr_floor", -100.0)),
        willr_max=float(_get(p, "willr_max", -30.0)),
        willr_by_cycle=_cycle_list(p, "willr_cycle", willr_base),

        ema_span=int(_get(p, "ema_span", 35)),
        ema_trend_long=int(_get(p, "ema_trend_long", 120)),
        ema_align=bool(_get(p, "ema_align", True)),
        require_ema200_slope=bool(_get(p, "require_ema200_slope", True)),

        profit_target_mult=float(_get(p, "profit_target_mult", tp_base)),
        stop_loss_mult=float(_get(p, "stop_loss_mult", sl_base)),
        max_hold_hours=int(_get(p, "max_hold_hours", 48)),

        tp_mult_by_cycle=_cycle_list(p, "tp_mult_cycle", tp_base),
        sl_mult_by_cycle=_cycle_list(p, "sl_mult_cycle", sl_base),
        exit_rsi_by_cycle=_cycle_list(p, "exit_rsi_cycle", exit_rsi_base),

        risk_per_trade=float(_get(p, "risk_per_trade", 0.02)),
        max_allocation=float(_get(p, "max_allocation", 0.7)),
        atr_k=float(_get(p, "atr_k", 1.0)),

        use_vol_filter=bool(_get(p, "use_vol_filter", False)),
        vol_tail_percentile=float(_get(p, "vol_tail_percentile", 0.55)),

        allow_hours=list(_get(p, "allow_hours", [])),

        trade_cycles=[1, 2],  # your current rule
    )
# src/bot087/strategy/param_store.py

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from src.bot087.strategy.params import StrategyParams


def _project_root() -> str:
    # src/bot087/strategy/param_store.py -> go up 4 levels to repo root
    here = os.path.abspath(__file__)
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


PROJECT_ROOT = _project_root()
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")
PARAMS_DIR = os.path.join(METADATA_DIR, "params")
os.makedirs(PARAMS_DIR, exist_ok=True)


def _active_path(symbol: str) -> str:
    return os.path.join(PARAMS_DIR, f"{symbol}_active_params.json")


def _to_params_dict(p: Union[StrategyParams, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(p, dict):
        return dict(p)
    if is_dataclass(p):
        return asdict(p)
    return dict(vars(p))


def load_active_params(symbol: str) -> StrategyParams:
    """
    Load 'active' params for a symbol. If missing, return defaults.
    """
    path = _active_path(symbol)
    if not os.path.exists(path):
        return StrategyParams()

    with open(path, "r") as f:
        raw = json.load(f)

    # accept either {"params": {...}} or flat dict
    params_dict = raw.get("params", raw) if isinstance(raw, dict) else {}
    if not isinstance(params_dict, dict):
        params_dict = {}

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
    Save 'active' params for a symbol.
    Returns the filepath written.
    """
    path = _active_path(symbol)

    payload = {
        "symbol": symbol,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "params": _to_params_dict(params),
        "meta": meta or {},
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    return path


# --- Compatibility alias (so older code can call save_params) ---
def save_params(symbol: str, params: Union[StrategyParams, Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> str:
    return save_active_params(symbol, params, meta=meta)
