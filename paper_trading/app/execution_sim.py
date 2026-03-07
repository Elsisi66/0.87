from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from src.bot087.optim.ga import _apply_cost, _position_size


@dataclass
class ExecutionResult:
    events: list[dict[str, Any]]
    opened: int
    closed: int


class ExecutionSimulator:
    def __init__(
        self,
        fee_bps: float,
        slippage_choices_bps: list[int],
        seed: int = 42,
        *,
        defer_exit_to_next_bar: bool = True,
    ) -> None:
        self.fee_bps = float(fee_bps)
        self.slippage_choices_bps = [int(x) for x in slippage_choices_bps]
        self.rng = random.Random(seed)
        self.defer_exit_to_next_bar = bool(defer_exit_to_next_bar)
        self._guard_stats = {
            "same_bar_exit_attempts": 0,
            "exits_deferred_to_next_bar": 0,
            "exits_blocked_pre_entry": 0,
        }

    def reset_guard_stats(self) -> None:
        self._guard_stats = {
            "same_bar_exit_attempts": 0,
            "exits_deferred_to_next_bar": 0,
            "exits_blocked_pre_entry": 0,
        }

    def snapshot_guard_stats(self) -> dict[str, int]:
        return {
            "same_bar_exit_attempts": int(self._guard_stats.get("same_bar_exit_attempts", 0)),
            "exits_deferred_to_next_bar": int(self._guard_stats.get("exits_deferred_to_next_bar", 0)),
            "exits_blocked_pre_entry": int(self._guard_stats.get("exits_blocked_pre_entry", 0)),
        }

    def sample_slippage_bps(self) -> int:
        return int(self.rng.choice(self.slippage_choices_bps))

    @staticmethod
    def _cost_components_eur(qty: float, raw_px_quote: float, quote_to_eur: float, fee_bps: float, slip_bps: float) -> tuple[float, float]:
        fee_cost = qty * raw_px_quote * (fee_bps / 1e4) * quote_to_eur
        slip_cost = qty * raw_px_quote * (slip_bps / 1e4) * quote_to_eur
        return float(fee_cost), float(slip_cost)

    def process_bar(
        self,
        *,
        symbol: str,
        row: dict[str, Any],
        params: dict[str, Any],
        quote_to_eur: float,
        portfolio: dict[str, Any],
        positions: dict[str, Any],
    ) -> ExecutionResult:
        events: list[dict[str, Any]] = []
        opened = 0
        closed = 0

        signal = bool(row.get("SIGNAL", False))
        bar_ts = str(row["Timestamp"])
        bar_open = float(row["Open"])
        bar_high = float(row["High"])
        bar_low = float(row["Low"])
        bar_close = float(row["Close"])
        atr_prev = float(row.get("ATR_PREV", 0.0))
        rsi_prev = float(row.get("RSI_PREV", 50.0))
        bar_index = int(row.get("BAR_INDEX", 0))
        cycle = int(row.get("CYCLE", 2))

        pos = positions.get(symbol)

        if pos is None and signal:
            cash_eur = float(portfolio.get("cash_eur", 0.0))
            if cash_eur > 0:
                slip_bps = float(self.sample_slippage_bps())
                buy_px_quote = float(_apply_cost(bar_open, self.fee_bps, slip_bps, "buy"))

                entry_px_eur = buy_px_quote * quote_to_eur
                atr_eur = atr_prev * quote_to_eur
                risk_per_trade = float(params.get("risk_per_trade", 0.02))
                max_alloc = float(params.get("max_allocation", 0.7))
                atr_k = float(params.get("atr_k", 1.0))

                size = float(_position_size(cash_eur, entry_px_eur, atr_eur, risk_per_trade, max_alloc, atr_k))
                if size > 0:
                    entry_cost_eur = size * entry_px_eur
                    if entry_cost_eur > cash_eur:
                        size = cash_eur / max(entry_px_eur, 1e-9)
                        entry_cost_eur = size * entry_px_eur

                    if size > 0:
                        open_seq = int(portfolio.get("trade_count_opened", 0)) + 1
                        trade_id = f"{symbol}:{bar_ts}:{open_seq}"
                        fee_cost_eur, slip_cost_eur = self._cost_components_eur(
                            size,
                            bar_open,
                            quote_to_eur,
                            self.fee_bps,
                            slip_bps,
                        )

                        portfolio["cash_eur"] = float(cash_eur - entry_cost_eur)
                        portfolio["fees_paid_eur"] = float(portfolio.get("fees_paid_eur", 0.0) + fee_cost_eur)
                        portfolio["slippage_paid_eur"] = float(portfolio.get("slippage_paid_eur", 0.0) + slip_cost_eur)
                        portfolio["trade_count_opened"] = int(portfolio.get("trade_count_opened", 0) + 1)

                        tp_mult = float(params.get("tp_mult_by_cycle", [1.02] * 5)[cycle])
                        sl_mult = float(params.get("sl_mult_by_cycle", [0.98] * 5)[cycle])

                        positions[symbol] = {
                            "symbol": symbol,
                            "trade_id": trade_id,
                            "units": float(size),
                            "signal_ts": bar_ts,
                            "entry_ts": bar_ts,
                            "entry_bar_index": bar_index,
                            "entry_cycle": cycle,
                            "entry_px_quote": buy_px_quote,
                            "entry_raw_open_quote": bar_open,
                            "entry_px_eur": entry_px_eur,
                            "entry_cost_eur": entry_cost_eur,
                            "entry_fee_eur": fee_cost_eur,
                            "entry_slippage_eur": slip_cost_eur,
                            "tp_mult": tp_mult,
                            "sl_mult": sl_mult,
                            "entry_slippage_bps": slip_bps,
                            "entry_fee_bps": self.fee_bps,
                        }
                        opened += 1
                        events.append(
                            {
                                "event": "fill_open",
                                "trade_id": trade_id,
                                "symbol": symbol,
                                "signal_time": bar_ts,
                                "entry_time": bar_ts,
                                "bar_ts": bar_ts,
                                "side": "LONG",
                                "units": float(size),
                                "entry_px_quote": buy_px_quote,
                                "entry_raw_open_quote": bar_open,
                                "entry_cost_eur": entry_cost_eur,
                                "fee_bps": self.fee_bps,
                                "slippage_bps": slip_bps,
                                "entry_fee_eur": fee_cost_eur,
                                "entry_slippage_eur": slip_cost_eur,
                                "cycle": cycle,
                            }
                        )

        pos = positions.get(symbol)
        if pos is not None:
            qty = float(pos["units"])
            entry_px_quote = float(pos["entry_px_quote"])
            entry_bar_index = int(pos["entry_bar_index"])
            entry_cycle = int(pos["entry_cycle"])
            tp_mult = float(pos["tp_mult"])
            sl_mult = float(pos["sl_mult"])

            hold = int(bar_index - entry_bar_index)
            max_hold = int(params.get("max_hold_hours", 48))
            tp_px = entry_px_quote * tp_mult
            sl_px = entry_px_quote * sl_mult

            if self.defer_exit_to_next_bar and hold < 0:
                self._guard_stats["exits_blocked_pre_entry"] = int(self._guard_stats["exits_blocked_pre_entry"]) + 1
                events.append(
                    {
                        "event": "exit_blocked_pre_entry",
                        "symbol": symbol,
                        "bar_ts": bar_ts,
                        "entry_ts": pos.get("entry_ts"),
                        "hold_hours": hold,
                    }
                )
                events.insert(
                    0,
                    {
                        "event": "signal_decision",
                        "symbol": symbol,
                        "bar_ts": bar_ts,
                        "signal": signal,
                        "cycle": cycle,
                        "has_position": symbol in positions,
                    },
                )
                return ExecutionResult(events=events, opened=opened, closed=closed)

            if self.defer_exit_to_next_bar and hold == 0:
                hit_sl = bar_low <= sl_px
                hit_tp = bar_high >= tp_px
                hit_maxhold = hold >= max_hold
                exit_rsi_by_cycle = params.get("exit_rsi_by_cycle", [0.0] * 5)
                ex = float(exit_rsi_by_cycle[entry_cycle])
                pnl_ratio = (bar_close / entry_px_quote) if entry_px_quote > 0 else 1.0
                hit_rsi = bool(rsi_prev < ex and pnl_ratio > 1.0)
                attempted = bool(hit_sl or hit_tp or hit_maxhold or hit_rsi)
                if attempted:
                    self._guard_stats["same_bar_exit_attempts"] = int(self._guard_stats["same_bar_exit_attempts"]) + 1
                    self._guard_stats["exits_deferred_to_next_bar"] = int(self._guard_stats["exits_deferred_to_next_bar"]) + 1
                    events.append(
                        {
                            "event": "exit_deferred_to_next_bar",
                            "symbol": symbol,
                            "bar_ts": bar_ts,
                            "entry_ts": pos.get("entry_ts"),
                            "hold_hours": hold,
                            "hit_sl": int(hit_sl),
                            "hit_tp": int(hit_tp),
                            "hit_maxhold": int(hit_maxhold),
                            "hit_rsi": int(hit_rsi),
                        }
                    )
                events.insert(
                    0,
                    {
                        "event": "signal_decision",
                        "symbol": symbol,
                        "bar_ts": bar_ts,
                        "signal": signal,
                        "cycle": cycle,
                        "has_position": symbol in positions,
                    },
                )
                return ExecutionResult(events=events, opened=opened, closed=closed)

            exit_reason = None
            exit_raw_quote = None

            hit_sl = bar_low <= sl_px
            hit_tp = bar_high >= tp_px
            if hit_sl and hit_tp:
                exit_reason = "sl"
                exit_raw_quote = sl_px
            elif hit_sl:
                exit_reason = "sl"
                exit_raw_quote = sl_px
            elif hit_tp:
                exit_reason = "tp"
                exit_raw_quote = tp_px
            elif hold >= max_hold:
                exit_reason = "maxhold"
                exit_raw_quote = bar_open
            else:
                exit_rsi_by_cycle = params.get("exit_rsi_by_cycle", [0.0] * 5)
                ex = float(exit_rsi_by_cycle[entry_cycle])
                pnl_ratio = (bar_close / entry_px_quote) if entry_px_quote > 0 else 1.0
                if rsi_prev < ex and pnl_ratio > 1.0:
                    exit_reason = "rsi_exit"
                    exit_raw_quote = bar_open

            if exit_reason is not None and exit_raw_quote is not None:
                slip_bps = float(self.sample_slippage_bps())
                sell_px_quote = float(_apply_cost(float(exit_raw_quote), self.fee_bps, slip_bps, "sell"))
                proceeds_eur = qty * sell_px_quote * quote_to_eur
                entry_cost_eur = float(pos.get("entry_cost_eur", qty * entry_px_quote * quote_to_eur))
                net_pnl_eur = proceeds_eur - entry_cost_eur
                pnl_pct = 100.0 * net_pnl_eur / entry_cost_eur if abs(entry_cost_eur) > 1e-12 else 0.0

                fee_cost_eur, slip_cost_eur = self._cost_components_eur(
                    qty,
                    float(exit_raw_quote),
                    quote_to_eur,
                    self.fee_bps,
                    slip_bps,
                )

                portfolio["cash_eur"] = float(portfolio.get("cash_eur", 0.0) + proceeds_eur)
                portfolio["realized_pnl_eur"] = float(portfolio.get("realized_pnl_eur", 0.0) + net_pnl_eur)
                portfolio["fees_paid_eur"] = float(portfolio.get("fees_paid_eur", 0.0) + fee_cost_eur)
                portfolio["slippage_paid_eur"] = float(portfolio.get("slippage_paid_eur", 0.0) + slip_cost_eur)
                portfolio["trade_count_closed"] = int(portfolio.get("trade_count_closed", 0) + 1)
                if net_pnl_eur > 0:
                    portfolio["wins"] = int(portfolio.get("wins", 0) + 1)
                elif net_pnl_eur < 0:
                    portfolio["losses"] = int(portfolio.get("losses", 0) + 1)

                positions.pop(symbol, None)
                closed += 1
                events.append(
                    {
                        "event": "fill_close",
                        "trade_id": pos.get("trade_id"),
                        "symbol": symbol,
                        "signal_time": pos.get("signal_ts", pos.get("entry_ts")),
                        "entry_time": pos.get("entry_ts"),
                        "exit_time": bar_ts,
                        "bar_ts": bar_ts,
                        "entry_ts": pos.get("entry_ts"),
                        "side": "LONG",
                        "units": qty,
                        "entry_px_quote": entry_px_quote,
                        "exit_px_quote": sell_px_quote,
                        "exit_raw_quote": float(exit_raw_quote),
                        "reason": exit_reason,
                        "hold_hours": hold,
                        "fee_bps": self.fee_bps,
                        "slippage_bps": slip_bps,
                        "entry_slippage_bps": pos.get("entry_slippage_bps"),
                        "entry_fee_eur": float(pos.get("entry_fee_eur", 0.0)),
                        "entry_slippage_eur": float(pos.get("entry_slippage_eur", 0.0)),
                        "exit_fee_eur": fee_cost_eur,
                        "exit_slippage_eur": slip_cost_eur,
                        "fees_eur": float(pos.get("entry_fee_eur", 0.0)) + fee_cost_eur,
                        "slippage_eur": float(pos.get("entry_slippage_eur", 0.0)) + slip_cost_eur,
                        "net_pnl_eur": net_pnl_eur,
                        "pnl_pct": pnl_pct,
                        "entry_cycle": entry_cycle,
                    }
                )

        events.insert(
            0,
            {
                "event": "signal_decision",
                "symbol": symbol,
                "bar_ts": bar_ts,
                "signal": signal,
                "cycle": cycle,
                "has_position": symbol in positions,
            },
        )

        return ExecutionResult(events=events, opened=opened, closed=closed)
