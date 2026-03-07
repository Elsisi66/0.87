from __future__ import annotations

from typing import Any


def default_portfolio(start_equity_eur: float) -> dict[str, Any]:
    return {
        "cash_eur": float(start_equity_eur),
        "initial_equity_eur": float(start_equity_eur),
        "realized_pnl_eur": 0.0,
        "fees_paid_eur": 0.0,
        "slippage_paid_eur": 0.0,
        "trade_count_opened": 0,
        "trade_count_closed": 0,
        "wins": 0,
        "losses": 0,
        "last_summary_date": None,
        "last_updated_utc": None,
        "degraded_mode": False,
        "mode_note": "initialized",
    }


def unrealized_pnl_eur(positions: dict[str, Any], mark_prices_quote: dict[str, float], quote_to_eur: dict[str, float]) -> float:
    total = 0.0
    for symbol, pos in positions.items():
        qty = float(pos.get("units", 0.0))
        if qty <= 0:
            continue
        mark_px_quote = float(mark_prices_quote.get(symbol, pos.get("entry_px_quote", 0.0)))
        entry_px_quote = float(pos.get("entry_px_quote", 0.0))
        fx = float(quote_to_eur.get(symbol, 1.0))
        total += (mark_px_quote - entry_px_quote) * qty * fx
    return float(total)


def total_equity_eur(
    portfolio: dict[str, Any],
    positions: dict[str, Any],
    mark_prices_quote: dict[str, float],
    quote_to_eur: dict[str, float],
) -> float:
    cash = float(portfolio.get("cash_eur", 0.0))
    mtm = 0.0
    for symbol, pos in positions.items():
        qty = float(pos.get("units", 0.0))
        if qty <= 0:
            continue
        mark_px_quote = float(mark_prices_quote.get(symbol, pos.get("entry_px_quote", 0.0)))
        fx = float(quote_to_eur.get(symbol, 1.0))
        mtm += qty * mark_px_quote * fx
    return float(cash + mtm)
