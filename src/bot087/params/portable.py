from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


# Keys consumed by the current C13/NLA backtest flow.
_USED_PARAM_KEYS: Set[str] = {
    "entry_rsi_min",
    "entry_rsi_max",
    "entry_rsi_buffer",
    "willr_floor",
    "willr_by_cycle",
    "ema_span",
    "ema_trend_long",
    "ema_align",
    "require_ema200_slope",
    "adx_min",
    "require_plus_di",
    "tp_mult_by_cycle",
    "sl_mult_by_cycle",
    "exit_rsi_by_cycle",
    "risk_per_trade",
    "max_allocation",
    "atr_k",
    "allow_hours",
    "trade_cycles",
    "require_trade_cycles",
    "max_hold_hours",
    "breakout_window",
    "breakout_atr_mult",
    "cycle_shift",
    "cycle_fill",
    "cycle1_adx_boost",
    "cycle1_ema_sep_atr",
    "two_candle_confirm",
}


def _legacy_cycle_keys() -> Set[str]:
    out: Set[str] = set()
    for base in ("willr", "tp_mult", "sl_mult", "exit_rsi"):
        for idx in range(5):
            out.add(f"{base}_cycle{idx}")
    return out


_LEGACY_KEYS = _legacy_cycle_keys()


def _unwrap_template(template: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(template.get("params"), dict):
        return dict(template["params"])
    return dict(template)


def _as_csv(values: Iterable[str]) -> str:
    vals = [str(v).strip() for v in values if str(v).strip()]
    return ",".join(vals) if vals else "none"


def adapt_params(template: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Keep only known/used keys from an older params template.

    Returns a params dict plus an internal `__adapt_notes` string describing
    what was dropped during adaptation.
    """
    if not isinstance(template, dict):
        raise ValueError("template must be a JSON object")

    src = _unwrap_template(template)
    kept: Dict[str, Any] = {}
    dropped: List[str] = []
    allowed = _USED_PARAM_KEYS | _LEGACY_KEYS

    for key, value in src.items():
        if key in allowed:
            kept[key] = value
        else:
            dropped.append(str(key))

    kept["__adapt_notes"] = (
        f"adapted_for={str(symbol).upper()}; dropped_keys={_as_csv(sorted(dropped))}"
    )
    return kept
