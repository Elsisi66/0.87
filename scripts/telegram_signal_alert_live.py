#!/usr/bin/env python3
# telegram_signal_alert_live.py
#
# Live 1h (or any ccxt-supported TF) scanner:
# - pulls LIVE OHLCV from Binance via ccxt (NOT parquet)
# - computes indicators locally (EMA/RSI/ATR/WILLR/ADX/DI + EMA200 slope)
# - computes cycles with NO-LOOKAHEAD logic
# - checks a 1-candle entry condition (NO 2-candle confirm)
# - sends a Telegram message ONLY when a long entry is detected
#
# Install deps:
#   pip install -U ccxt requests pandas numpy
#
# Telegram:
#   export TELEGRAM_BOT_TOKEN="123:ABC..."
#   export TELEGRAM_CHAT_ID="123456789"
#
# Example:
#   python3 scripts/telegram_signal_alert_live.py \
#     --symbols BTC/USDT \
#     --tf 1h \
#     --only-cycles 3 \
#     --cycle-shift 1 \
#     --dry-run \
#     --verbose

import os
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

try:
    import ccxt
except Exception as e:
    raise SystemExit("Missing dependency: ccxt. Install with: pip install -U ccxt") from e


# -------------------------
# Params normalization
# -------------------------
def _norm_params(seed: Dict[str, Any]) -> Dict[str, Any]:
    p = dict(seed)

    def _list_from_cycles(prefix: str, fallback: Optional[List[float]] = None) -> List[float]:
        if prefix in p and isinstance(p[prefix], list) and len(p[prefix]) == 5:
            return [float(x) for x in p[prefix]]

        base = prefix.replace("_by_cycle", "")
        out = []
        ok = True
        for i in range(5):
            k = f"{base}_cycle{i}"
            if k in p and p[k] is not None:
                out.append(float(p[k]))
            else:
                ok = False
                break
        if ok and len(out) == 5:
            return out

        if fallback is not None and isinstance(fallback, list) and len(fallback) == 5:
            return [float(x) for x in fallback]
        return [0.0] * 5

    p["willr_by_cycle"] = _list_from_cycles("willr_by_cycle", fallback=p.get("willr_by_cycle"))
    p["tp_mult_by_cycle"] = _list_from_cycles("tp_mult_by_cycle", fallback=p.get("tp_mult_by_cycle"))
    p["sl_mult_by_cycle"] = _list_from_cycles("sl_mult_by_cycle", fallback=p.get("sl_mult_by_cycle"))
    p["exit_rsi_by_cycle"] = _list_from_cycles("exit_rsi_by_cycle", fallback=p.get("exit_rsi_by_cycle"))

    p.setdefault("willr_floor", -100.0)
    p.setdefault("willr_max", -30.0)
    p.setdefault("ema_span", 35)
    p.setdefault("ema_trend_long", 120)
    p.setdefault("ema_align", True)
    p.setdefault("require_ema200_slope", True)
    p.setdefault("adx_min", 18.0)
    p.setdefault("require_plus_di", True)

    p.setdefault("entry_rsi_min", 50.0)
    p.setdefault("entry_rsi_max", 65.0)
    p.setdefault("entry_rsi_buffer", 2.0)

    p.setdefault("profit_target_mult", 1.06)
    p.setdefault("stop_loss_mult", 0.98)
    p.setdefault("max_hold_hours", 48)

    p.setdefault("risk_per_trade", 0.02)
    p.setdefault("max_allocation", 0.7)
    p.setdefault("atr_k", 1.0)

    p.setdefault("use_vol_filter", False)
    p.setdefault("vol_tail_percentile", 0.55)
    p.setdefault("allow_hours", None)

    # cycles allowed to trade
    p.setdefault("trade_cycles", [1, 2])

    # signal module toggles (at least one True)
    p.setdefault("use_sig_baseline", True)
    p.setdefault("use_sig_breakout", False)
    p.setdefault("use_sig_pullback", False)

    # breakout params
    p.setdefault("breakout_window", 20)
    p.setdefault("breakout_atr_mult", 1.5)

    return p


# -------------------------
# Indicators (no external libs)
# -------------------------
def _ensure_indicators(df: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    df = df.copy()

    for col in ["Open", "High", "Low", "Close"]:
        if col not in df.columns:
            raise ValueError(f"Missing OHLC column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    ema_span = int(p.get("ema_span", 35))
    ema_long = int(p.get("ema_trend_long", 120))
    for n in {ema_span, ema_long, 200}:
        col = f"EMA_{n}"
        if col not in df.columns:
            df[col] = _ema(close, n)

    if "EMA_200_SLOPE" not in df.columns:
        df["EMA_200_SLOPE"] = df["EMA_200"].diff().fillna(0.0)

    # RSI (Wilder via EMA)
    if "RSI" not in df.columns:
        period = 14
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        df["RSI"] = rsi.fillna(50.0).clip(0.0, 100.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    if "ATR" not in df.columns:
        period = 14
        df["ATR"] = tr.ewm(alpha=1 / period, adjust=False).mean().fillna(0.0)

    if "WILLR" not in df.columns:
        period = 14
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        denom = (hh - ll).replace(0.0, np.nan)
        willr = -100.0 * (hh - close) / denom
        df["WILLR"] = willr.fillna(-50.0).clip(-100.0, 0.0)

    need_adx = ("ADX" not in df.columns) or ("PLUS_DI" not in df.columns) or ("MINUS_DI" not in df.columns)
    if need_adx:
        period = 14
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

        tr_sm = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_sm = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()
        minus_sm = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean()

        plus_di = 100.0 * (plus_sm / tr_sm.replace(0.0, np.nan))
        minus_di = 100.0 * (minus_sm / tr_sm.replace(0.0, np.nan))

        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        df["PLUS_DI"] = plus_di.fillna(0.0)
        df["MINUS_DI"] = minus_di.fillna(0.0)
        df["ADX"] = adx.fillna(0.0)

    return df


# -------------------------
# Cycles (no lookahead)
# -------------------------
def compute_cycles(df: pd.DataFrame, p: Dict[str, Any]) -> np.ndarray:
    """
    5 regimes:
      0 downtrend
      1 expansion
      2 accumulation
      3 breakout
      4 distribution

    NO LOOKAHEAD: recent_high uses rolling max of previous closes (shifted by 1).
    """
    close = df["Close"].astype(float).to_numpy()

    ema_long_col = f"EMA_{int(p['ema_trend_long'])}"
    if ema_long_col in df.columns:
        ema_long = df[ema_long_col].astype(float).ffill().to_numpy()
    else:
        ema_long = df["EMA_200"].astype(float).ffill().to_numpy()

    slope = df.get("EMA_200_SLOPE", pd.Series(np.zeros(len(df)), index=df.index)).astype(float).fillna(0.0).to_numpy()
    adx = df.get("ADX", pd.Series(np.zeros(len(df)), index=df.index)).astype(float).fillna(0.0).to_numpy()
    rsi = df.get("RSI", pd.Series(np.full(len(df), 50.0), index=df.index)).astype(float).fillna(50.0).to_numpy()
    atr = df.get("ATR", pd.Series(np.zeros(len(df)), index=df.index)).astype(float).fillna(0.0).to_numpy()

    above = close > ema_long
    s_up = slope > 0.0
    s_down = slope < 0.0

    adx_thr = float(p.get("adx_min", 18.0))
    adx_high = adx >= adx_thr
    rsi_high = rsi > float(p.get("entry_rsi_max", 65.0))

    w = int(p.get("breakout_window", 20))
    prev = pd.Series(close).shift(1)
    recent_high = prev.rolling(w, min_periods=w).max().to_numpy()
    gap = close - recent_high
    breakout_up = (close > recent_high) & (gap > atr * float(p.get("breakout_atr_mult", 1.5)))
    breakout_up = np.nan_to_num(breakout_up, nan=False).astype(bool)

    n = len(df)
    cycles = np.full(n, 2, dtype=np.int8)

    cycles[breakout_up] = 3
    mask_down = (~breakout_up) & s_down & (~above) & adx_high
    cycles[mask_down] = 0
    mask_exp = (~breakout_up) & (~mask_down) & s_up & above & adx_high
    cycles[mask_exp] = 1
    mask_dist = (~breakout_up) & (~mask_down) & (~mask_exp) & rsi_high & (~adx_high)
    cycles[mask_dist] = 4
    return cycles


def shift_cycles_forward(cycles: np.ndarray, n: int) -> np.ndarray:
    """
    Shift cycles forward by N bars:
      out[i] = cycles[i-n] for i>=n
    This is the strict "use previous bar's regime" safety knob (NLA).
    """
    if n <= 0:
        return cycles.copy()
    out = cycles.copy()
    out[n:] = cycles[:-n]
    out[:n] = cycles[0] if len(cycles) else 2
    return out


# -------------------------
# Entry signal (1-candle, NLA)
# -------------------------
def _hour_ok(ts: pd.Timestamp, allow_hours: Optional[List[int]]) -> bool:
    if not allow_hours:
        return True
    allowed = set(int(h) for h in allow_hours)
    return int(ts.hour) in allowed


def evaluate_live_entry(df: pd.DataFrame, p: Dict[str, Any], cycle_shift: int = 1) -> Tuple[bool, Dict[str, Any]]:
    """
    Live alert logic:
    - Use LAST CLOSED candle as the 'prev candle' features.
    - If conditions pass on that closed candle, alert to enter NOW (next candle open).
    This is 1-candle entry (no 2-candle confirm).
    """
    if len(df) < 250:
        return False, {"error": f"not enough candles: {len(df)} (need ~250+ for EMA200/ADX stability)"}

    # Ensure Timestamp is tz-aware UTC
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # Compute indicators
    df = _ensure_indicators(df, p)

    # Compute cycles (no lookahead) and apply shift safety
    raw_cycles = compute_cycles(df, p)
    cycles = shift_cycles_forward(raw_cycles, int(cycle_shift))

    # Use last CLOSED candle as prev features
    i = len(df) - 1
    ts_prev = df.loc[i, "Timestamp"]

    # Features from this closed candle (prev candle)
    rsi_prev = float(df.loc[i, "RSI"])
    willr_prev = float(df.loc[i, "WILLR"])
    atr_prev = float(df.loc[i, "ATR"])
    close_prev = float(df.loc[i, "Close"])

    ema_long_col = f"EMA_{int(p['ema_trend_long'])}"
    ema_span_col = f"EMA_{int(p['ema_span'])}"
    ema_long_prev = float(df.loc[i, ema_long_col]) if ema_long_col in df.columns else float(df.loc[i, "EMA_200"])
    ema_span_prev = float(df.loc[i, ema_span_col]) if ema_span_col in df.columns else ema_long_prev

    slope_prev = float(df.loc[i, "EMA_200_SLOPE"]) if "EMA_200_SLOPE" in df.columns else 0.0
    adx_prev = float(df.loc[i, "ADX"]) if "ADX" in df.columns else 0.0
    plus_prev = float(df.loc[i, "PLUS_DI"]) if "PLUS_DI" in df.columns else 0.0
    minus_prev = float(df.loc[i, "MINUS_DI"]) if "MINUS_DI" in df.columns else 0.0

    cycle_prev = int(cycles[i]) if len(cycles) else 2

    # Trend filter (same spirit as GA)
    close_ok = close_prev > ema_long_prev
    ema_ok = (ema_span_prev > ema_long_prev) if bool(p.get("ema_align", True)) else True
    adx_ok = adx_prev >= float(p.get("adx_min", 18.0))
    di_ok = (plus_prev > minus_prev) if bool(p.get("require_plus_di", True)) else True
    slope_ok = (slope_prev > 0.0) if bool(p.get("require_ema200_slope", True)) else True
    trend_ok = bool(close_ok and ema_ok and adx_ok and di_ok and slope_ok)

    # Cycle filter
    trade_cycles = set(int(x) for x in p.get("trade_cycles", [1, 2]))
    cyc_ok = (cycle_prev in trade_cycles) if bool(p.get("require_trade_cycles", True)) else True

    hour_ok = _hour_ok(ts_prev, p.get("allow_hours"))

    # Signal 0: baseline (RSI band + WILLR by cycle)
    rsi_min = float(p["entry_rsi_min"])
    rsi_max = float(p["entry_rsi_max"])
    rsi_band = (rsi_prev >= rsi_min) and (rsi_prev <= rsi_max)

    willr_floor = float(p.get("willr_floor", -100.0))
    willr_thr = float(p["willr_by_cycle"][cycle_prev]) if 0 <= cycle_prev < 5 else float(p.get("willr_max", -30.0))
    willr_ok = (willr_prev >= willr_floor) and (willr_prev < willr_thr)

    sig_baseline = bool(rsi_band and willr_ok and trend_ok and cyc_ok and hour_ok)

    # Signal 1: breakout
    w = int(p.get("breakout_window", 20))
    closes = df["Close"].astype(float).to_numpy()
    recent_high = pd.Series(closes).shift(1).rolling(w, min_periods=w).max().to_numpy()
    rh = float(recent_high[i]) if not math.isnan(float(recent_high[i])) else float("nan")
    gap = close_prev - rh if not math.isnan(rh) else float("nan")
    sig_breakout = False
    if not math.isnan(rh):
        sig_breakout = bool(
            (cycle_prev == 3)
            and (close_prev > rh)
            and (gap > atr_prev * float(p.get("breakout_atr_mult", 1.5)))
            and trend_ok
            and hour_ok
        )

    # Signal 2: pullback
    sig_pullback = bool(
        (cycle_prev in {1, 2})
        and trend_ok
        and hour_ok
        and (rsi_prev <= (rsi_min + 2.0))
        and (willr_prev < min(-55.0, willr_thr))
    )

    use0 = bool(p.get("use_sig_baseline", True))
    use1 = bool(p.get("use_sig_breakout", False))
    use2 = bool(p.get("use_sig_pullback", False))

    fired_modules = []
    sig = False
    if use0 and sig_baseline:
        sig = True
        fired_modules.append("baseline")
    if use1 and sig_breakout:
        sig = True
        fired_modules.append("breakout")
    if use2 and sig_pullback:
        sig = True
        fired_modules.append("pullback")

    # Provide TP/SL multipliers for this cycle (what your backtest uses)
    tp_mult = float(p["tp_mult_by_cycle"][cycle_prev]) if 0 <= cycle_prev < 5 else float(p.get("profit_target_mult", 1.06))
    sl_mult = float(p["sl_mult_by_cycle"][cycle_prev]) if 0 <= cycle_prev < 5 else float(p.get("stop_loss_mult", 0.98))

    details = {
        "ts_prev": str(ts_prev),
        "cycle": cycle_prev,
        "fired_modules": fired_modules,
        "trend_ok": trend_ok,
        "cyc_ok": bool(cyc_ok),
        "hour_ok": bool(hour_ok),
        "rsi": rsi_prev,
        "willr": willr_prev,
        "adx": adx_prev,
        "plus_di": plus_prev,
        "minus_di": minus_prev,
        "ema_long": ema_long_prev,
        "ema_span": ema_span_prev,
        "ema200_slope": slope_prev,
        "atr": atr_prev,
        "close": close_prev,
        "recent_high": rh,
        "gap": gap,
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "cycle_shift": int(cycle_shift),
    }

    return sig, details


# -------------------------
# Live data fetch (ccxt)
# -------------------------
def fetch_ohlcv_live(exchange_id: str, symbol: str, tf: str, limit: int, verbose: bool = False) -> pd.DataFrame:
    ex_class = getattr(ccxt, exchange_id, None)
    if ex_class is None:
        raise ValueError(f"Unsupported exchange_id={exchange_id}. Example: binance, binanceusdm")

    ex = ex_class({"enableRateLimit": True})
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=int(limit))

    if not ohlcv or len(ohlcv) < 50:
        raise ValueError(f"Not enough OHLCV returned for {symbol} {tf}. got={len(ohlcv) if ohlcv else 0}")

    df = pd.DataFrame(ohlcv, columns=["Timestamp_ms", "Open", "High", "Low", "Close", "Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["Timestamp_ms"])

    if verbose:
        t0 = df["Timestamp"].iloc[0]
        t1 = df["Timestamp"].iloc[-1]
        print(f"[data] {exchange_id} {symbol} {tf} rows={len(df)} {t0} -> {t1}")

    return df


# -------------------------
# Telegram
# -------------------------
def tg_send(token: str, chat_id: str, text: str, dry_run: bool) -> None:
    if dry_run:
        print("\n----- TELEGRAM (dry-run) -----")
        print(text)
        print("----- /TELEGRAM -----\n")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    r = requests.post(url, json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram send failed: {r.status_code} {r.text}")


# -------------------------
# Params loading
# -------------------------
def _try_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def load_params_for_symbol(params_arg: Optional[str], symbol_norm: str, project_root: Path) -> Tuple[Dict[str, Any], str]:
    """
    Priority:
      1) --params if provided
      2) artifacts/ga/<SYMBOL>/best_params.json
      3) data/metadata/params/<SYMBOL>_active_params.json (if format {params:{...}})
    """
    if params_arg:
        p = _try_load_json(Path(params_arg).expanduser())
        if p is None:
            raise FileNotFoundError(f"Could not load params from --params: {params_arg}")
        return _norm_params(p), str(Path(params_arg).resolve())

    # symbol like BTC/USDT -> BTCUSDT folder naming
    folder_sym = symbol_norm.replace("/", "")
    p1 = project_root / "artifacts" / "ga" / folder_sym / "best_params.json"
    j1 = _try_load_json(p1)
    if j1 is not None:
        return _norm_params(j1), str(p1.resolve())

    p2 = project_root / "data" / "metadata" / "params" / f"{folder_sym}_active_params.json"
    j2 = _try_load_json(p2)
    if j2 is not None:
        if isinstance(j2, dict) and "params" in j2 and isinstance(j2["params"], dict):
            return _norm_params(j2["params"]), str(p2.resolve())
        if isinstance(j2, dict):
            return _norm_params(j2), str(p2.resolve())

    raise FileNotFoundError(
        f"Could not find params for {symbol_norm}. Looked in:\n"
        f"  {p1}\n  {p2}\n"
        f"or pass --params /path/to/params.json"
    )


# -------------------------
# State (avoid duplicate alerts)
# -------------------------
def load_state(state_file: Path) -> Dict[str, Any]:
    if state_file.exists():
        try:
            return json.load(open(state_file, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state_file: Path, st: Dict[str, Any]) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


# -------------------------
# Formatting
# -------------------------
def format_alert(symbol: str, details: Dict[str, Any]) -> str:
    ts = details.get("ts_prev")
    cyc = details.get("cycle")
    fired = ",".join(details.get("fired_modules", [])) or "none"

    close_px = details.get("close")
    rsi = details.get("rsi")
    willr = details.get("willr")
    adx = details.get("adx")
    plus_di = details.get("plus_di")
    minus_di = details.get("minus_di")

    tp_mult = details.get("tp_mult")
    sl_mult = details.get("sl_mult")

    # Entry reference: "enter now" near next candle open ~ last close
    # TP/SL reference (multipliers like your backtest)
    tp_px = close_px * tp_mult if close_px and tp_mult else None
    sl_px = close_px * sl_mult if close_px and sl_mult else None

    lines = []
    lines.append(f"📈 LONG SIGNAL: {symbol}")
    lines.append(f"Prev candle close (UTC): {ts}")
    lines.append(f"Cycle: {cyc} | fired: {fired} | cycle_shift: {details.get('cycle_shift')}")
    lines.append("")
    lines.append(f"Entry reference (market near open): {close_px:.6f}")
    if tp_px is not None and sl_px is not None:
        lines.append(f"TP (mult {tp_mult:.4f}): {tp_px:.6f}")
        lines.append(f"SL (mult {sl_mult:.4f}): {sl_px:.6f}")
    lines.append("")
    lines.append(f"RSI: {rsi:.2f} | WILLR: {willr:.2f} | ATR: {details.get('atr'):.6f}")
    lines.append(f"ADX: {adx:.2f} | +DI: {plus_di:.2f} | -DI: {minus_di:.2f}")
    lines.append(f"Trend OK: {details.get('trend_ok')} | Cycle OK: {details.get('cyc_ok')} | Hour OK: {details.get('hour_ok')}")
    return "\n".join(lines)


# -------------------------
# CLI
# -------------------------
def parse_int_list(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out if out else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols. Prefer ccxt format like BTC/USDT,ETH/USDT")
    ap.add_argument("--tf", default="1h", help="Timeframe (ccxt), e.g. 1h, 4h, 15m")
    ap.add_argument("--params", default=None, help="Optional params json to use for ALL symbols")
    ap.add_argument("--only-cycles", default=None, help="Only alert if cycle in list, e.g. 3 or 1,3")
    ap.add_argument("--cycle-shift", type=int, default=1, help="Shift cycles forward by N bars (strict NLA: 1)")
    ap.add_argument("--exchange", default="binance", help="ccxt exchange id: binance, binanceusdm, etc.")
    ap.add_argument("--limit", type=int, default=600, help="Candles to fetch (>=250 recommended)")
    ap.add_argument("--dry-run", action="store_true", help="Print instead of sending Telegram")
    ap.add_argument("--verbose", action="store_true")

    # Telegram creds (either args or env)
    ap.add_argument("--tg-token", default=os.getenv("TELEGRAM_BOT_TOKEN", ""), help="Telegram bot token (or env TELEGRAM_BOT_TOKEN)")
    ap.add_argument("--tg-chat", default=os.getenv("TELEGRAM_CHAT_ID", ""), help="Telegram chat id (or env TELEGRAM_CHAT_ID)")

    # State file to avoid spamming the same candle
    ap.add_argument("--state-file", default="output/telegram_alert_state.json")

    args = ap.parse_args()

    if not args.dry_run:
        if not args.tg_token or not args.tg_chat:
            raise SystemExit("Missing Telegram creds. Provide --tg-token and --tg-chat or set env TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.")

    project_root = Path.cwd().resolve()
    only_cycles = parse_int_list(args.only_cycles)

    state_file = Path(args.state_file).expanduser()
    state = load_state(state_file)

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not syms:
        raise SystemExit("No symbols parsed from --symbols")

    now_utc = datetime.now(timezone.utc).isoformat()

    for sym in syms:
        # Normalize symbol:
        # - accept BTCUSDT and convert to BTC/USDT for ccxt if possible
        sym_norm = sym.upper()
        if "/" not in sym_norm:
            if sym_norm.endswith("USDT"):
                sym_norm = sym_norm[:-4] + "/USDT"
            else:
                # fallback: ccxt might still accept, but binance usually wants BTC/USDT
                pass

        try:
            params, params_path = load_params_for_symbol(args.params, sym_norm, project_root)
        except Exception as e:
            print(f"[WARN] {sym_norm}: params load failed: {e}")
            continue

        params = _norm_params(params)

        if args.verbose:
            print(f"[params] {sym_norm} using: {params_path}")

        try:
            df = fetch_ohlcv_live(args.exchange, sym_norm, args.tf, args.limit, verbose=args.verbose)
        except Exception as e:
            print(f"[WARN] {sym_norm}: fetch failed: {e}")
            continue

        # IMPORTANT:
        # For safety we evaluate entry using LAST CLOSED candle.
        # ccxt returns the latest candle which *may be in-progress* depending on timing.
        # To be strict, drop the last candle if it's likely in-progress by checking if its timestamp is too recent.
        # For 1h this is easy; for other TF, we just drop last row always (safe, but one candle delayed).
        if len(df) > 2:
            df_eval = df.iloc[:-1].reset_index(drop=True)
        else:
            df_eval = df

        sig, details = evaluate_live_entry(df_eval, params, cycle_shift=int(args.cycle_shift))

        if "error" in details:
            if args.verbose:
                print(f"[skip] {sym_norm}: {details['error']}")
            continue

        # Only cycles filter
        if only_cycles is not None and int(details.get("cycle", -999)) not in set(only_cycles):
            if args.verbose:
                print(f"[no] {sym_norm}: signal={sig} cycle={details.get('cycle')} (filtered by only-cycles={only_cycles})")
            continue

        # De-dup: one alert per prev-candle timestamp
        key = sym_norm.replace("/", "")
        last_ts = state.get(key, {}).get("last_ts_prev")
        ts_prev = details.get("ts_prev")
        if last_ts == ts_prev:
            if args.verbose:
                print(f"[dup] {sym_norm}: already alerted for {ts_prev}")
            continue

        if sig:
            msg = format_alert(sym_norm, details)
            try:
                tg_send(args.tg_token, args.tg_chat, msg, dry_run=args.dry_run)
                print(f"[ALERT] {sym_norm} sent at {now_utc} (prev={ts_prev})")
                state[key] = {"last_ts_prev": ts_prev, "sent_at_utc": now_utc}
                save_state(state_file, state)
            except Exception as e:
                print(f"[ERR] {sym_norm}: telegram send failed: {e}")
        else:
            if args.verbose:
                print(f"[no] {sym_norm}: no entry. cycle={details.get('cycle')} trend_ok={details.get('trend_ok')}")

if __name__ == "__main__":
    main()
