from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import requests

from .config import Settings
from .health import HealthSnapshot
from .portfolio import total_equity_eur, unrealized_pnl_eur
from .utils.io import atomic_write_json, atomic_write_text
from .utils.redact import redact_text
from .utils.retry import RetryConfig, retry_call
from .utils.time_utils import date_yyyymmdd, utc_iso


@dataclass
class NotifyResult:
    sent: bool
    reason: str


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _day_bounds_utc(day: datetime) -> tuple[datetime, datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start, end


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    except ValueError:
        return None


def _pct(numer: float, denom: float) -> float:
    if abs(denom) < 1e-12:
        return 0.0
    return 100.0 * numer / denom


def _profit_factor(pnls: list[float]) -> float | None:
    wins = sum(x for x in pnls if x > 0)
    losses = abs(sum(x for x in pnls if x < 0))
    if losses <= 1e-12:
        if wins > 0:
            return float("inf")
        return None
    return wins / losses


def _max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    max_dd = 0.0
    for value in values:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
    return float(max_dd)


def build_daily_summary(
    *,
    settings: Settings,
    target_day_utc: datetime,
    portfolio: dict[str, Any],
    positions: dict[str, Any],
    mark_prices_quote: dict[str, float],
    quote_to_eur_map: dict[str, float],
    health_snapshot: HealthSnapshot,
) -> dict[str, Any]:
    journal_path = settings.state_dir / "journal.jsonl"
    rows = _iter_jsonl(journal_path)

    day_start, day_end = _day_bounds_utc(target_day_utc)

    day_events: list[dict[str, Any]] = []
    for row in rows:
        ts = _parse_ts(row.get("event_recorded_ts") or row.get("ts_utc") or row.get("bar_ts"))
        if ts is None:
            continue
        if day_start <= ts < day_end:
            day_events.append(row)

    fills_open = [x for x in day_events if x.get("event") == "fill_open"]
    fills_close = [x for x in day_events if x.get("event") == "fill_close"]
    equity_points = [x for x in day_events if x.get("event") == "equity_snapshot"]

    close_pnls = [float(x.get("net_pnl_eur", 0.0)) for x in fills_close]
    wins = [x for x in close_pnls if x > 0]
    losses = [x for x in close_pnls if x < 0]

    opened = len(fills_open)
    closed = len(fills_close)
    win_rate = (len(wins) / closed * 100.0) if closed else 0.0
    avg_win = mean(wins) if wins else 0.0
    avg_loss = mean(losses) if losses else 0.0
    pf = _profit_factor(close_pnls)

    per_symbol: dict[str, float] = {}
    for fill in fills_close:
        symbol = str(fill.get("symbol", "UNKNOWN"))
        per_symbol[symbol] = per_symbol.get(symbol, 0.0) + float(fill.get("net_pnl_eur", 0.0))

    slips = []
    for fill in fills_open + fills_close:
        if "slippage_bps" in fill:
            try:
                slips.append(int(float(fill["slippage_bps"])))
            except (TypeError, ValueError):
                continue

    slip_dist = {str(bps): slips.count(bps) for bps in settings.slippage_bps_choices}
    slip_mean = mean(slips) if slips else 0.0
    slip_median = median(slips) if slips else 0.0

    end_equity = total_equity_eur(portfolio, positions, mark_prices_quote, quote_to_eur_map)
    unrealized = unrealized_pnl_eur(positions, mark_prices_quote, quote_to_eur_map)
    realized_total = float(portfolio.get("realized_pnl_eur", 0.0))

    prior_date = (target_day_utc - timedelta(days=1)).strftime("%Y%m%d")
    prior_summary_path = settings.reports_dir / f"daily_summary_{prior_date}.json"
    prior_summary = json.loads(prior_summary_path.read_text(encoding="utf-8")) if prior_summary_path.exists() else None
    prior_end_equity = float(prior_summary.get("end_equity_eur", portfolio.get("initial_equity_eur", 0.0))) if prior_summary else float(
        portfolio.get("initial_equity_eur", 0.0)
    )

    start_equity = prior_end_equity
    day_realized = sum(close_pnls)
    day_total = day_realized + unrealized

    if equity_points:
        values = [float(x.get("equity_eur", 0.0)) for x in equity_points]
        max_intraday_dd = _max_drawdown(values)
    else:
        max_intraday_dd = 0.0

    summary = {
        "generated_utc": utc_iso(),
        "date_utc": target_day_utc.strftime("%Y-%m-%d"),
        "date_range_utc": f"{day_start.isoformat()} -> {day_end.isoformat()}",
        "start_equity_eur": float(start_equity),
        "end_equity_eur": float(end_equity),
        "daily_realized_pnl_eur": float(day_realized),
        "daily_total_pnl_eur": float(day_total),
        "pnl_pct_vs_initial": float(_pct(end_equity - float(portfolio.get("initial_equity_eur", 0.0)), float(portfolio.get("initial_equity_eur", 0.0)))),
        "pnl_pct_vs_prior_day": float(_pct(end_equity - prior_end_equity, prior_end_equity)),
        "trades_opened": opened,
        "trades_closed": closed,
        "win_rate_pct": float(win_rate),
        "avg_win_eur": float(avg_win),
        "avg_loss_eur": float(avg_loss),
        "profit_factor": "inf" if pf == float("inf") else (float(pf) if pf is not None else None),
        "max_intraday_drawdown_pct": float(max_intraday_dd * 100.0),
        "open_positions_count": int(len(positions)),
        "per_symbol_contribution_eur": dict(sorted(per_symbol.items())),
        "slippage_mean_bps": float(slip_mean),
        "slippage_median_bps": float(slip_median),
        "slippage_distribution_counts": slip_dist,
        "error_recovery_counts": dict(health_snapshot.counters),
        "strategy_health": health_snapshot.strategy_health,
        "degraded_mode": bool(health_snapshot.degraded_mode),
        "degraded_note": "local/degraded path active" if health_snapshot.degraded_mode else "",
    }

    day_tag = target_day_utc.strftime("%Y%m%d")
    json_path = settings.reports_dir / f"daily_summary_{day_tag}.json"
    md_path = settings.reports_dir / f"daily_summary_{day_tag}.md"
    atomic_write_json(json_path, summary)

    lines = [
        f"# Daily Paper Summary ({summary['date_utc']})",
        "",
        f"- Date range (UTC): `{summary['date_range_utc']}`",
        f"- Start equity (EUR): `{summary['start_equity_eur']:.4f}`",
        f"- End equity (EUR): `{summary['end_equity_eur']:.4f}`",
        f"- Daily realized PnL (EUR): `{summary['daily_realized_pnl_eur']:.4f}`",
        f"- Daily total PnL (EUR): `{summary['daily_total_pnl_eur']:.4f}`",
        f"- PnL % vs initial 320 EUR: `{summary['pnl_pct_vs_initial']:.4f}%`",
        f"- PnL % vs prior day equity: `{summary['pnl_pct_vs_prior_day']:.4f}%`",
        f"- Trades opened/closed: `{opened}/{closed}`",
        f"- Win rate: `{summary['win_rate_pct']:.2f}%`",
        f"- Avg win / Avg loss (EUR): `{summary['avg_win_eur']:.4f}` / `{summary['avg_loss_eur']:.4f}`",
        f"- Profit factor: `{summary['profit_factor']}`",
        f"- Max intraday drawdown: `{summary['max_intraday_drawdown_pct']:.4f}%`",
        f"- Open positions: `{summary['open_positions_count']}`",
        f"- Slippage mean/median bps: `{summary['slippage_mean_bps']:.2f}` / `{summary['slippage_median_bps']:.2f}`",
        f"- Slippage distribution: `{summary['slippage_distribution_counts']}`",
        f"- Error/recovery counts: `{summary['error_recovery_counts']}`",
        f"- Strategy health: `{summary['strategy_health']}`",
    ]
    if summary["degraded_mode"]:
        lines.append(f"- Degraded mode note: `{summary['degraded_note']}`")

    if summary["per_symbol_contribution_eur"]:
        lines.append("")
        lines.append("## Per-Symbol Contribution (EUR)")
        for symbol, pnl in summary["per_symbol_contribution_eur"].items():
            lines.append(f"- {symbol}: `{pnl:.4f}`")

    atomic_write_text(md_path, "\n".join(lines) + "\n")
    return summary


class TelegramNotifier:
    def __init__(self, settings: Settings, logger) -> None:
        self.settings = settings
        self.logger = logger
        self.session = requests.Session()
        self.retry_cfg = RetryConfig(attempts=3, base_delay_sec=1.0, max_delay_sec=6.0, jitter_sec=0.35)

    def _send_text(self, text: str) -> None:
        if not self.settings.telegram_token or not self.settings.telegram_chat_id:
            raise RuntimeError("telegram credentials not configured")

        url = f"https://api.telegram.org/bot{self.settings.telegram_token}/sendMessage"
        payload = {
            "chat_id": self.settings.telegram_chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }

        def call() -> None:
            resp = self.session.post(url, data=payload, timeout=12)
            resp.raise_for_status()

        retry_call(call, cfg=self.retry_cfg)

    def send_daily_summary(self, summary: dict[str, Any]) -> NotifyResult:
        if not self.settings.telegram_token or not self.settings.telegram_chat_id:
            return NotifyResult(sent=False, reason="missing_telegram_credentials")

        text = (
            f"Paper Daily Summary {summary['date_utc']}\n"
            f"Start/End EUR: {summary['start_equity_eur']:.2f} -> {summary['end_equity_eur']:.2f}\n"
            f"Realized/Total PnL EUR: {summary['daily_realized_pnl_eur']:.2f} / {summary['daily_total_pnl_eur']:.2f}\n"
            f"Opened/Closed: {summary['trades_opened']}/{summary['trades_closed']}\n"
            f"WinRate: {summary['win_rate_pct']:.2f}%  PF: {summary['profit_factor']}\n"
            f"OpenPos: {summary['open_positions_count']}  DD: {summary['max_intraday_drawdown_pct']:.2f}%\n"
            f"Health: {summary['strategy_health']}  Degraded: {summary['degraded_mode']}"
        )
        try:
            self._send_text(text)
            return NotifyResult(sent=True, reason="sent")
        except Exception as exc:
            err_text = redact_text(str(exc), [self.settings.telegram_token, self.settings.telegram_chat_id])
            self.logger.error("telegram_send_failed err=%s", err_text)
            return NotifyResult(sent=False, reason=f"telegram_error:{err_text}")

    def send_trade_fill(self, fill: dict[str, Any]) -> NotifyResult:
        if not self.settings.telegram_token or not self.settings.telegram_chat_id:
            return NotifyResult(sent=False, reason="missing_telegram_credentials")

        event_type = str(fill.get("event", "fill"))
        symbol = str(fill.get("symbol", "UNKNOWN"))
        bar_ts = str(fill.get("bar_ts", ""))

        if event_type == "fill_open":
            text = (
                f"Paper Trade Open {symbol}\n"
                f"Bar: {bar_ts}\n"
                f"Units: {float(fill.get('units', 0.0)):.8f}\n"
                f"Entry: {float(fill.get('entry_px_quote', 0.0)):.8f}\n"
                f"Cost EUR: {float(fill.get('entry_cost_eur', 0.0)):.2f}\n"
                f"Slip/Fee bps: {fill.get('slippage_bps')} / {fill.get('fee_bps')}"
            )
        elif event_type == "fill_close":
            text = (
                f"Paper Trade Close {symbol}\n"
                f"Bar: {bar_ts}\n"
                f"Reason: {fill.get('reason')}\n"
                f"Exit: {float(fill.get('exit_px_quote', 0.0)):.8f}\n"
                f"PnL EUR: {float(fill.get('net_pnl_eur', 0.0)):.2f}\n"
                f"Hold hours: {fill.get('hold_hours')}\n"
                f"Slip/Fee bps: {fill.get('slippage_bps')} / {fill.get('fee_bps')}"
            )
        else:
            text = f"Paper Trade Event {symbol}\nType: {event_type}\nBar: {bar_ts}"

        try:
            self._send_text(text)
            return NotifyResult(sent=True, reason="sent")
        except Exception as exc:
            err_text = redact_text(str(exc), [self.settings.telegram_token, self.settings.telegram_chat_id])
            self.logger.error("telegram_trade_send_failed err=%s", err_text)
            return NotifyResult(sent=False, reason=f"telegram_error:{err_text}")

    def send_reset_completed(self, reset_meta: dict[str, Any]) -> NotifyResult:
        if not self.settings.telegram_token or not self.settings.telegram_chat_id:
            return NotifyResult(sent=False, reason="missing_telegram_credentials")

        text = (
            "Paper Reset Completed\n"
            f"Start capital EUR: {float(reset_meta.get('start_equity_eur', 0.0)):.2f}\n"
            f"Start from bar: {reset_meta.get('start_from_bar_ts')}\n"
            f"Reset mode: {reset_meta.get('reset_mode')}\n"
            f"Symbols: {len(reset_meta.get('symbols', []))}"
        )
        try:
            self._send_text(text)
            return NotifyResult(sent=True, reason="sent")
        except Exception as exc:
            err_text = redact_text(str(exc), [self.settings.telegram_token, self.settings.telegram_chat_id])
            self.logger.error("telegram_reset_send_failed err=%s", err_text)
            return NotifyResult(sent=False, reason=f"telegram_error:{err_text}")

    def send_fatal_error(self, message: str) -> NotifyResult:
        if not self.settings.telegram_token or not self.settings.telegram_chat_id:
            return NotifyResult(sent=False, reason="missing_telegram_credentials")
        text = f"Paper Trader Fatal Error\n{message}"
        try:
            self._send_text(text)
            return NotifyResult(sent=True, reason="sent")
        except Exception as exc:
            err_text = redact_text(str(exc), [self.settings.telegram_token, self.settings.telegram_chat_id])
            self.logger.error("telegram_fatal_send_failed err=%s", err_text)
            return NotifyResult(sent=False, reason=f"telegram_error:{err_text}")



def should_emit_daily_summary(
    *,
    now_utc: datetime,
    daily_summary_hour_utc: int,
    last_summary_date: str | None,
) -> bool:
    today = date_yyyymmdd(now_utc)
    if last_summary_date == today:
        return False
    return now_utc.hour >= int(daily_summary_hour_utc)
