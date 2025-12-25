# src/bot087/optim/ga.py
import os
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import tempfile


import numpy as np
import pandas as pd
from multiprocessing import Pool


# =========================
# Config
# =========================

@dataclass(frozen=True)
class GAConfig:
    # GA
    pop_size: int = 40
    generations: int = 30
    elite_k: int = 6
    mutation_rate: float = 0.35
    mutation_strength: float = 1.0
    n_procs: int = 3

    # Monte Carlo walk-forward splits per generation
    mc_splits: int = 6
    train_days: int = 540  # ~18 months
    val_days: int = 180    # 6 months
    test_days: int = 180   # 6 months
    seed: int = 42

    # Execution model
    fee_bps: float = 7.0
    slippage_bps: float = 2.0
    initial_equity: float = 10_000.0

    # Constraints
    min_trades_train: int = 40
    min_trades_val: int = 15

    # Entry behavior
    two_candle_confirm: bool = True  # require signal for 2 consecutive closed candles
    require_trade_cycles: bool = True

    # Fitness weights
    w_train: float = 0.7
    w_val: float = 0.3
    dd_penalty: float = 0.45     # multiplied by max_dd * initial_equity
    trade_penalty: float = 0.8   # multiplied by n_trades (fee/churn pressure)
    bad_val_penalty: float = 1200.0  # if val net is negative

    # Saving
    resume: bool = True


# =========================
# Paths / saving
# =========================
def _ensure_indicators(df: pd.DataFrame, p: Dict[str, Any]) -> pd.DataFrame:
    """
    If processed CSVs are missing indicators, compute them here so signals are not killed by 0-defaults.
    Computes: EMA_{ema_span}, EMA_{ema_trend_long}, EMA_200, EMA_200_SLOPE, RSI, ATR, WILLR, PLUS_DI, MINUS_DI, ADX
    """
    df = df.copy()

    # enforce numeric OHLC
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low  = df["Low"].astype(float)

    # --- EMAs
    def _ema(s: pd.Series, span: int) -> pd.Series:
        return s.ewm(span=span, adjust=False).mean()

    ema_span = int(p.get("ema_span", 35))
    ema_long = int(p.get("ema_trend_long", 120))
    for n in {ema_span, ema_long, 200}:
        col = f"EMA_{n}"
        if col not in df.columns:
            df[col] = _ema(close, n)

    # --- EMA slope
    if "EMA_200_SLOPE" not in df.columns:
        df["EMA_200_SLOPE"] = df["EMA_200"].diff().fillna(0.0)

    # --- RSI (Wilder-style via EMA)
    if "RSI" not in df.columns:
        period = 14
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        df["RSI"] = rsi.fillna(50.0).clip(0.0, 100.0)

    # --- True Range (needed for ATR + ADX)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # --- ATR
    if "ATR" not in df.columns:
        period = 14
        df["ATR"] = tr.ewm(alpha=1/period, adjust=False).mean().fillna(0.0)

    # --- Williams %R
    if "WILLR" not in df.columns:
        period = 14
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        denom = (hh - ll).replace(0.0, np.nan)
        willr = -100.0 * (hh - close) / denom
        df["WILLR"] = willr.fillna(-50.0).clip(-100.0, 0.0)

    # --- ADX / DI
    need_adx = ("ADX" not in df.columns) or ("PLUS_DI" not in df.columns) or ("MINUS_DI" not in df.columns)
    if need_adx:
        period = 14
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

        tr_sm = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_sm = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
        minus_sm = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()

        plus_di = 100.0 * (plus_sm / tr_sm.replace(0.0, np.nan))
        minus_di = 100.0 * (minus_sm / tr_sm.replace(0.0, np.nan))

        dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan))
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        df["PLUS_DI"] = plus_di.fillna(0.0)
        df["MINUS_DI"] = minus_di.fillna(0.0)
        df["ADX"] = adx.fillna(0.0)

    return df

def _project_root() -> Path:
    env = os.getenv("BOT087_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    # assume running from repo root
    return Path.cwd().resolve()


def _utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, payload: Any) -> None:
    """
    Atomic JSON write so a crash doesn't corrupt checkpoint/snapshots.
    """
    _ensure_dir(path.parent)

    tmp = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
        tmp = Path(tmp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
    finally:
        if tmp is not None and tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass



def _json_load(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


# =========================
# Parameter handling (dict-based)
# =========================

def _norm_params(seed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize params into the keys this GA/backtest expects.
    Accepts either:
      - list form (willr_by_cycle, tp_mult_by_cycle, sl_mult_by_cycle, exit_rsi_by_cycle)
      - scalar-per-cycle form (willr_cycle0..4, tp_mult_cycle0..4, sl_mult_cycle0..4, exit_rsi_cycle0..4)
    """
    p = dict(seed)

    def _list_from_cycles(prefix: str, fallback: Optional[List[float]] = None) -> List[float]:
        if prefix in p and isinstance(p[prefix], list) and len(p[prefix]) == 5:
            return [float(x) for x in p[prefix]]

        base = prefix.replace("_by_cycle", "")  # FIX
        out = []
        ok = True
        for i in range(5):
            k = f"{base}_cycle{i}"  # FIX
            if k in p and p[k] is not None:
                out.append(float(p[k]))
            else:
                ok = False
                break

        if ok and len(out) == 5:
            return out
        if fallback is not None:
            return list(map(float, fallback))
        return [0.0] * 5

    # Required lists
    p["willr_by_cycle"] = _list_from_cycles("willr_by_cycle", fallback=p.get("willr_by_cycle"))
    p["tp_mult_by_cycle"] = _list_from_cycles("tp_mult_by_cycle", fallback=p.get("tp_mult_by_cycle"))
    p["sl_mult_by_cycle"] = _list_from_cycles("sl_mult_by_cycle", fallback=p.get("sl_mult_by_cycle"))
    p["exit_rsi_by_cycle"] = _list_from_cycles("exit_rsi_by_cycle", fallback=p.get("exit_rsi_by_cycle"))

    # Required scalars with safe defaults
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

    # Cycles we trade (your default was 1 & 2)
    p.setdefault("trade_cycles", [1, 2])

    # 3-signal toggles (at least one must be True)
    p.setdefault("use_sig_baseline", True)
    p.setdefault("use_sig_breakout", False)
    p.setdefault("use_sig_pullback", False)

    # Breakout params
    p.setdefault("breakout_window", 20)
    p.setdefault("breakout_atr_mult", 1.5)

    return p


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def _mut_gauss(x: float, sigma: float) -> float:
    return float(x + random.gauss(0.0, sigma))


def _mut_list_gauss(xs: List[float], sigma: float, lo: float, hi: float) -> List[float]:
    out = []
    for v in xs:
        out.append(_clip(_mut_gauss(float(v), sigma), lo, hi))
    return out


def _random_trade_cycles() -> List[int]:
    # You *said* 1 & 2 are tradable. We'll allow GA to optionally include 3, but it must pay via fitness.
    base = [1, 2]
    if random.random() < 0.25:
        base = [1, 2, 3]
    return base


def mutate_params(p: Dict[str, Any], strength: float, rate: float) -> Dict[str, Any]:
    """
    Mutate in-place style (returns a new dict).
    """
    q = dict(p)
    sigma_rsi = 2.5 * strength
    sigma_willr = 6.0 * strength
    sigma_exit = 3.0 * strength

    if random.random() < rate:
        q["entry_rsi_min"] = _clip(_mut_gauss(float(q["entry_rsi_min"]), sigma_rsi), 5.0, 80.0)
    if random.random() < rate:
        q["entry_rsi_max"] = _clip(_mut_gauss(float(q["entry_rsi_max"]), sigma_rsi), 20.0, 95.0)
    if q["entry_rsi_min"] >= q["entry_rsi_max"]:
        q["entry_rsi_max"] = min(95.0, q["entry_rsi_min"] + 5.0)

    if random.random() < rate:
        q["entry_rsi_buffer"] = _clip(_mut_gauss(float(q["entry_rsi_buffer"]), 1.0 * strength), 0.0, 12.0)

    if random.random() < rate:
        q["willr_by_cycle"] = _mut_list_gauss(list(q["willr_by_cycle"]), sigma_willr, -100.0, -1.0)

    # TP/SL per cycle
    if random.random() < rate:
        q["tp_mult_by_cycle"] = _mut_list_gauss(list(q["tp_mult_by_cycle"]), 0.01 * strength, 1.005, 1.25)
    if random.random() < rate:
        q["sl_mult_by_cycle"] = _mut_list_gauss(list(q["sl_mult_by_cycle"]), 0.01 * strength, 0.85, 0.999)

    if random.random() < rate:
        q["exit_rsi_by_cycle"] = _mut_list_gauss(list(q["exit_rsi_by_cycle"]), sigma_exit, 10.0, 95.0)

    # Hold / risk knobs (bounded)
    if random.random() < rate:
        q["max_hold_hours"] = int(_clip(_mut_gauss(float(q["max_hold_hours"]), 8.0 * strength), 6.0, 120.0))
    if random.random() < rate:
        q["risk_per_trade"] = _clip(_mut_gauss(float(q["risk_per_trade"]), 0.003 * strength), 0.003, 0.04)
    if random.random() < rate:
        q["max_allocation"] = _clip(_mut_gauss(float(q["max_allocation"]), 0.05 * strength), 0.05, 0.95)
    if random.random() < rate:
        q["atr_k"] = _clip(_mut_gauss(float(q["atr_k"]), 0.25 * strength), 0.3, 4.0)

    # 3-signal toggles
    if random.random() < 0.15:
        q["use_sig_baseline"] = bool(random.getrandbits(1))
    if random.random() < 0.15:
        q["use_sig_breakout"] = bool(random.getrandbits(1))
    if random.random() < 0.15:
        q["use_sig_pullback"] = bool(random.getrandbits(1))

    if not (q["use_sig_baseline"] or q["use_sig_breakout"] or q["use_sig_pullback"]):
        q["use_sig_baseline"] = True  # never allow zero-signal individuals

    # trade cycles
    if random.random() < 0.12:
        q["trade_cycles"] = _random_trade_cycles()

    return _norm_params(q)


def crossover(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Blend numerical fields and pick discrete bits.
    """
    c = dict(a)
    alpha = random.random()

    def blend(x, y):
        return float(alpha * float(x) + (1.0 - alpha) * float(y))

    c["entry_rsi_min"] = blend(a["entry_rsi_min"], b["entry_rsi_min"])
    c["entry_rsi_max"] = blend(a["entry_rsi_max"], b["entry_rsi_max"])
    c["entry_rsi_buffer"] = blend(a["entry_rsi_buffer"], b["entry_rsi_buffer"])

    c["willr_by_cycle"] = [blend(x, y) for x, y in zip(a["willr_by_cycle"], b["willr_by_cycle"])]
    c["tp_mult_by_cycle"] = [blend(x, y) for x, y in zip(a["tp_mult_by_cycle"], b["tp_mult_by_cycle"])]
    c["sl_mult_by_cycle"] = [blend(x, y) for x, y in zip(a["sl_mult_by_cycle"], b["sl_mult_by_cycle"])]
    c["exit_rsi_by_cycle"] = [blend(x, y) for x, y in zip(a["exit_rsi_by_cycle"], b["exit_rsi_by_cycle"])]

    c["max_hold_hours"] = int(round(blend(a["max_hold_hours"], b["max_hold_hours"])))
    c["risk_per_trade"] = blend(a["risk_per_trade"], b["risk_per_trade"])
    c["max_allocation"] = blend(a["max_allocation"], b["max_allocation"])
    c["atr_k"] = blend(a["atr_k"], b["atr_k"])

    # discrete picks
    c["use_sig_baseline"] = a["use_sig_baseline"] if random.random() < 0.5 else b["use_sig_baseline"]
    c["use_sig_breakout"] = a["use_sig_breakout"] if random.random() < 0.5 else b["use_sig_breakout"]
    c["use_sig_pullback"] = a["use_sig_pullback"] if random.random() < 0.5 else b["use_sig_pullback"]
    if not (c["use_sig_baseline"] or c["use_sig_breakout"] or c["use_sig_pullback"]):
        c["use_sig_baseline"] = True

    c["trade_cycles"] = a["trade_cycles"] if random.random() < 0.5 else b["trade_cycles"]
    return _norm_params(c)


# =========================
# Regime computation (no lookahead)
# =========================

def compute_cycles(df: pd.DataFrame, p: Dict[str, Any]) -> np.ndarray:
    """
    5 regimes:
      0 downtrend
      1 expansion
      2 accumulation
      3 breakout
      4 distribution
    Uses t-1 features (rolling highs shifted).
    """
    close = df["Close"].astype(float).to_numpy()
    ema_long_col = f"EMA_{int(p['ema_trend_long'])}"
    ema_long = df[ema_long_col].astype(float).ffill().to_numpy() if ema_long_col in df.columns else df["EMA_200"].astype(float).ffill().to_numpy()

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
    # rolling max of *previous* closes (shifted)
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


# =========================
# Signals (3 modules + 2-candle confirm) with no lookahead entries
# =========================

def _hour_mask(ts: pd.Series, allow_hours: Optional[List[int]]) -> np.ndarray:
    if not allow_hours:
        return np.ones(len(ts), dtype=bool)
    allowed = set(int(h) for h in allow_hours)
    return ts.dt.hour.apply(lambda h: h in allowed).to_numpy(dtype=bool)


def build_entry_signal(df: pd.DataFrame, p: Dict[str, Any]) -> np.ndarray:
    """
    Entry decision uses t-1 feature values and enters at bar t open.
    """
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    cycles = compute_cycles(df, p)

    # Use previous candle features for entry decision
    rsi_prev = df["RSI"].astype(float).shift(1).fillna(50.0).to_numpy()
    willr_prev = df["WILLR"].astype(float).shift(1).fillna(-50.0).to_numpy()
    atr_prev = df["ATR"].astype(float).shift(1).fillna(0.0).to_numpy()
    close_prev = df["Close"].astype(float).shift(1).fillna(df["Close"].astype(float)).to_numpy()

    ema_long_col = f"EMA_{int(p['ema_trend_long'])}"
    ema_span_col = f"EMA_{int(p['ema_span'])}"
    ema_long_prev = df[ema_long_col].astype(float).shift(1).ffill().to_numpy() if ema_long_col in df.columns else df["EMA_200"].astype(float).shift(1).ffill().to_numpy()
    ema_span_prev = df[ema_span_col].astype(float).shift(1).ffill().to_numpy() if ema_span_col in df.columns else ema_long_prev

    slope_prev = df.get("EMA_200_SLOPE", pd.Series(0.0, index=df.index)).astype(float).shift(1).fillna(0.0).to_numpy()
    adx_prev = df.get("ADX", pd.Series(0.0, index=df.index)).astype(float).shift(1).fillna(0.0).to_numpy()
    plus_prev = df.get("PLUS_DI", pd.Series(0.0, index=df.index)).astype(float).shift(1).fillna(0.0).to_numpy()
    minus_prev = df.get("MINUS_DI", pd.Series(0.0, index=df.index)).astype(float).shift(1).fillna(0.0).to_numpy()

    # Trend filter (prev candle)
    close_ok = close_prev > ema_long_prev
    ema_ok = (ema_span_prev > ema_long_prev) if bool(p.get("ema_align", True)) else True
    adx_ok = adx_prev >= float(p.get("adx_min", 18.0))
    di_ok = (plus_prev > minus_prev) if bool(p.get("require_plus_di", True)) else True
    slope_ok = (slope_prev > 0.0) if bool(p.get("require_ema200_slope", True)) else True
    trend_ok = close_ok & ema_ok & adx_ok & di_ok & slope_ok

    # Cycle filter
    trade_cycles = set(int(x) for x in p.get("trade_cycles", [1, 2]))
    if bool(p.get("require_trade_cycles", True)):
        cyc_ok = np.array([int(c) in trade_cycles for c in cycles], dtype=bool)
    else:
        cyc_ok = np.ones(len(df), dtype=bool)

    # Hours
    hour_ok = _hour_mask(df["Timestamp"], p.get("allow_hours"))

    # --- Signal 0: baseline (RSI band + WILLR by cycle)
    rsi_min = float(p["entry_rsi_min"])
    rsi_max = float(p["entry_rsi_max"])
    rsi_band = (rsi_prev >= rsi_min) & (rsi_prev <= rsi_max)

    willr_floor = float(p.get("willr_floor", -100.0))
    willr_thr = np.array([float(p["willr_by_cycle"][int(c)]) for c in cycles], dtype=float)
    willr_ok = (willr_prev >= willr_floor) & (willr_prev < willr_thr)

    sig_baseline = rsi_band & willr_ok & trend_ok & cyc_ok & hour_ok

    # --- Signal 1: breakout (cycle==3 + gap > ATR mult)
    w = int(p.get("breakout_window", 20))
    prev_close_ser = pd.Series(df["Close"].astype(float).to_numpy()).shift(1)
    recent_high = prev_close_ser.rolling(w, min_periods=w).max().to_numpy()
    gap = df["Close"].astype(float).to_numpy() - recent_high
    sig_breakout = (cycles == 3) & (df["Close"].astype(float).to_numpy() > recent_high) & (gap > atr_prev * float(p.get("breakout_atr_mult", 1.5)))
    sig_breakout = np.nan_to_num(sig_breakout, nan=False).astype(bool) & trend_ok & hour_ok

    # --- Signal 2: pullback (in expansion/accum, RSI near min and WILLR oversold)
    pull_cyc_ok = np.array([int(c) in {1, 2} for c in cycles], dtype=bool)
    sig_pullback = pull_cyc_ok & trend_ok & hour_ok & (rsi_prev <= (rsi_min + 2.0)) & (willr_prev < np.minimum(-55.0, willr_thr))

    # Combine selected signals
    use0 = bool(p.get("use_sig_baseline", True))
    use1 = bool(p.get("use_sig_breakout", False))
    use2 = bool(p.get("use_sig_pullback", False))

    sig = np.zeros(len(df), dtype=bool)
    if use0:
        sig |= sig_baseline
    if use1:
        sig |= sig_breakout
    if use2:
        sig |= sig_pullback

    # 2-candle confirmation (prev 2 bars must be true); entry happens at bar t
    if bool(p.get("two_candle_confirm", True)):
        sig = sig & np.roll(sig, 1) & np.roll(sig, 2)
        sig[:2] = False

    return sig


# =========================
# Backtest (long-only, enter at OPEN, exit with intrabar TP/SL using HIGH/LOW)
# =========================

def _apply_cost(price: float, fee_bps: float, slip_bps: float, side: str) -> float:
    fee = fee_bps / 1e4
    slip = slip_bps / 1e4
    if side == "buy":
        return price * (1.0 + fee + slip)
    return price * (1.0 - fee - slip)


def _position_size(equity: float, entry_px: float, atr: float, risk_per_trade: float, max_alloc: float, atr_k: float) -> float:
    if equity <= 0.0 or entry_px <= 0.0 or atr <= 0.0:
        return 0.0
    u_risk = (equity * risk_per_trade) / (atr_k * atr)
    u_afford = (equity * max_alloc) / entry_px
    return max(0.0, min(u_risk, u_afford))


def run_backtest_long_only(
    df: pd.DataFrame,
    symbol: str,
    p: Dict[str, Any],
    initial_equity: float,
    fee_bps: float,
    slippage_bps: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # entry signal uses t-1 features; enter at t open
    p2 = dict(p)
    p2["two_candle_confirm"] = bool(p.get("two_candle_confirm", True))
    sig = build_entry_signal(df, p2)

    # Prepare arrays
    ts = df["Timestamp"].to_numpy()
    o = df["Open"].astype(float).to_numpy()
    h = df["High"].astype(float).to_numpy()
    l = df["Low"].astype(float).to_numpy()
    c = df["Close"].astype(float).to_numpy()
    atr_prev = df["ATR"].astype(float).shift(1).fillna(0.0).to_numpy()
    rsi_prev = df["RSI"].astype(float).shift(1).fillna(50.0).to_numpy()

    cycles = compute_cycles(df, p)

    cash = float(initial_equity)
    units = 0.0
    entry_px = 0.0
    entry_ts = None
    entry_cycle = None
    entry_i = -1
    tp_mult = 1.0
    sl_mult = 1.0

    equity_curve = []
    trades: List[Dict[str, Any]] = []

    max_hold = int(p.get("max_hold_hours", 48))
    risk_per_trade = float(p.get("risk_per_trade", 0.02))
    max_alloc = float(p.get("max_allocation", 0.7))
    atr_k = float(p.get("atr_k", 1.0))

    for i in range(len(df)):
        mid = c[i]
        equity = cash + units * mid
        equity_curve.append(equity)

        # ENTRY
        if units == 0.0 and sig[i]:
            # enter at OPEN[i]
            buy_px = _apply_cost(o[i], fee_bps, slippage_bps, "buy")
            atrv = float(atr_prev[i])
            size = _position_size(equity, buy_px, atrv, risk_per_trade, max_alloc, atr_k)
            if size <= 0.0:
                continue
            cost = size * buy_px
            if cost > cash:
                size = cash / buy_px
                cost = size * buy_px
            if size <= 0.0:
                continue

            units = size
            cash -= cost
            entry_px = buy_px
            entry_ts = ts[i]
            entry_cycle = int(cycles[i])
            entry_i = i

            # cycle TP/SL
            tp_mult = float(p["tp_mult_by_cycle"][entry_cycle]) if entry_cycle is not None else float(p.get("profit_target_mult", 1.06))
            sl_mult = float(p["sl_mult_by_cycle"][entry_cycle]) if entry_cycle is not None else float(p.get("stop_loss_mult", 0.98))
            continue

        # EXIT
        if units > 0.0 and entry_ts is not None:
            hold = i - entry_i
            # TP/SL levels (based on entry price with costs)
            tp_px = entry_px * tp_mult
            sl_px = entry_px * sl_mult

            exit_reason = None
            exit_exec_px = None

            # intrabar TP/SL using high/low
            hit_sl = l[i] <= sl_px
            hit_tp = h[i] >= tp_px

            if hit_sl and hit_tp:
                # conservative: assume worst case
                exit_reason = "sl"
                exit_exec_px = sl_px
            elif hit_sl:
                exit_reason = "sl"
                exit_exec_px = sl_px
            elif hit_tp:
                exit_reason = "tp"
                exit_exec_px = tp_px
            elif hold >= max_hold:
                exit_reason = "maxhold"
                exit_exec_px = o[i]  # exit at open
            else:
                # early RSI profit-only exit based on entry cycle threshold (uses prev candle RSI)
                ex = float(p["exit_rsi_by_cycle"][int(entry_cycle)]) if entry_cycle is not None else 50.0
                pnl_ratio = c[i] / entry_px if entry_px > 0 else 1.0
                if (rsi_prev[i] < ex) and (pnl_ratio > 1.0):
                    exit_reason = "rsi_exit"
                    exit_exec_px = o[i]

            if exit_reason is not None and exit_exec_px is not None:
                sell_px = _apply_cost(float(exit_exec_px), fee_bps, slippage_bps, "sell")
                proceeds = units * sell_px
                cash += proceeds

                gross = (sell_px - entry_px) * units
                hold_hours = float(hold)

                trades.append({
                    "symbol": symbol,
                    "cycle": int(entry_cycle) if entry_cycle is not None else None,
                    "entry_ts": str(pd.to_datetime(entry_ts, utc=True)),
                    "exit_ts": str(pd.to_datetime(ts[i], utc=True)),
                    "entry_px": float(entry_px),
                    "exit_px": float(sell_px),
                    "units": float(units),
                    "reason": exit_reason,
                    "net_pnl": float(gross),
                    "hold_hours": float(hold_hours),
                })

                units = 0.0
                entry_px = 0.0
                entry_ts = None
                entry_cycle = None
                entry_i = -1

    # force close
    if units > 0.0:
        sell_px = _apply_cost(float(c[-1]), fee_bps, slippage_bps, "sell")
        cash += units * sell_px
        gross = (sell_px - entry_px) * units
        trades.append({
            "symbol": symbol,
            "cycle": int(entry_cycle) if entry_cycle is not None else None,
            "entry_ts": str(pd.to_datetime(entry_ts, utc=True)) if entry_ts is not None else None,
            "exit_ts": str(pd.to_datetime(ts[-1], utc=True)),
            "entry_px": float(entry_px),
            "exit_px": float(sell_px),
            "units": float(units),
            "reason": "eod",
            "net_pnl": float(gross),
            "hold_hours": float(max(0, len(df) - 1 - entry_i)),
        })
        units = 0.0

    final_equity = float(cash)
    net_profit = float(final_equity - initial_equity)

    # max dd
    eq = np.array(equity_curve, dtype=float) if equity_curve else np.array([initial_equity], dtype=float)
    runmax = np.maximum.accumulate(eq)
    dd = (runmax - eq) / np.maximum(runmax, 1e-9)
    max_dd = float(dd.max()) if dd.size else 0.0

    # win rate, profit factor, avg win/loss
    pnls = np.array([t["net_pnl"] for t in trades], dtype=float) if trades else np.array([], dtype=float)
    wins = pnls[pnls > 0.0]
    losses = pnls[pnls < 0.0]
    win_rate = float((pnls > 0.0).mean() * 100.0) if pnls.size else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if losses.size and abs(losses.sum()) > 1e-9 else (10.0 if wins.size else 0.0)
    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0

    metrics = {
        "initial_equity": float(initial_equity),
        "final_equity": float(final_equity),
        "net_profit": float(net_profit),
        "trades": float(len(trades)),
        "win_rate_pct": float(win_rate),
        "max_dd": float(max_dd),
        "profit_factor": float(pf),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
    }
    return trades, metrics


# =========================
# Monte Carlo splits
# =========================

def make_mc_splits(df: pd.DataFrame, cfg: GAConfig, gen: int) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Returns list of splits as (tr0,tr1, va0,va1, te0,te1) indices, contiguous.
    """
    n = len(df)
    bars_train = cfg.train_days * 24
    bars_val = cfg.val_days * 24
    bars_test = cfg.test_days * 24
    total = bars_train + bars_val + bars_test

    if n < total + 50:
        raise ValueError(f"Not enough rows for MC split: have={n} need~{total}")

    rng = random.Random(cfg.seed + gen * 9973)
    splits = []
    for _ in range(cfg.mc_splits):
        start = rng.randint(0, n - total - 1)
        tr0 = start
        tr1 = tr0 + bars_train
        va0 = tr1
        va1 = va0 + bars_val
        te0 = va1
        te1 = te0 + bars_test
        splits.append((tr0, tr1, va0, va1, te0, te1))
    return splits


# =========================
# Fitness
# =========================

def _segment_score(m: Dict[str, float], cfg: GAConfig, min_trades: int) -> float:
    if m["trades"] < float(min_trades):
        return -1e9
    score = m["net_profit"]
    score -= cfg.dd_penalty * m["max_dd"] * cfg.initial_equity
    score -= cfg.trade_penalty * m["trades"]
    return float(score)


_GLOBAL_DF: Optional[pd.DataFrame] = None


def _init_worker(df: pd.DataFrame):
    global _GLOBAL_DF
    _GLOBAL_DF = df


def _eval_worker(args):
    ind, symbol, cfg_dict, splits = args
    cfg = GAConfig(**cfg_dict)
    assert _GLOBAL_DF is not None
    df = _GLOBAL_DF

    # ensure required flags exist
    ind = _norm_params(ind)
    ind["two_candle_confirm"] = bool(cfg.two_candle_confirm)
    ind["require_trade_cycles"] = bool(cfg.require_trade_cycles)

    train_scores = []
    val_scores = []
    test_scores = []
    train_nets = []
    val_nets = []
    test_nets = []

    # aggregate metrics
    agg = {
        "train": {"net": 0.0, "trades": 0.0, "win": 0.0, "dd": 0.0, "pf": 0.0},
        "val":   {"net": 0.0, "trades": 0.0, "win": 0.0, "dd": 0.0, "pf": 0.0},
        "test":  {"net": 0.0, "trades": 0.0, "win": 0.0, "dd": 0.0, "pf": 0.0},
    }

    for (tr0, tr1, va0, va1, te0, te1) in splits:
        df_tr = df.iloc[tr0:tr1].reset_index(drop=True)
        df_va = df.iloc[va0:va1].reset_index(drop=True)
        df_te = df.iloc[te0:te1].reset_index(drop=True)

        _, m_tr = run_backtest_long_only(df_tr, symbol, ind, cfg.initial_equity, cfg.fee_bps, cfg.slippage_bps)
        _, m_va = run_backtest_long_only(df_va, symbol, ind, cfg.initial_equity, cfg.fee_bps, cfg.slippage_bps)
        _, m_te = run_backtest_long_only(df_te, symbol, ind, cfg.initial_equity, cfg.fee_bps, cfg.slippage_bps)

        s_tr = _segment_score(m_tr, cfg, cfg.min_trades_train)
        s_va = _segment_score(m_va, cfg, cfg.min_trades_val)
        s_te = _segment_score(m_te, cfg, 0)

        train_scores.append(s_tr)
        val_scores.append(s_va)
        test_scores.append(s_te)

        train_nets.append(m_tr["net_profit"])
        val_nets.append(m_va["net_profit"])
        test_nets.append(m_te["net_profit"])

        agg["train"]["net"] += m_tr["net_profit"]
        agg["train"]["trades"] += m_tr["trades"]
        agg["train"]["win"] += m_tr["win_rate_pct"]
        agg["train"]["dd"] += m_tr["max_dd"]
        agg["train"]["pf"] += m_tr["profit_factor"]

        agg["val"]["net"] += m_va["net_profit"]
        agg["val"]["trades"] += m_va["trades"]
        agg["val"]["win"] += m_va["win_rate_pct"]
        agg["val"]["dd"] += m_va["max_dd"]
        agg["val"]["pf"] += m_va["profit_factor"]

        agg["test"]["net"] += m_te["net_profit"]
        agg["test"]["trades"] += m_te["trades"]
        agg["test"]["win"] += m_te["win_rate_pct"]
        agg["test"]["dd"] += m_te["max_dd"]
        agg["test"]["pf"] += m_te["profit_factor"]

    k = float(len(splits))
    for seg in ["train", "val", "test"]:
        for kk in agg[seg].keys():
            agg[seg][kk] = float(agg[seg][kk] / k)

    train_score = float(np.mean(train_scores))
    val_score = float(np.mean(val_scores))
    test_score = float(np.mean(test_scores))

    fitness = cfg.w_train * train_score + cfg.w_val * val_score
    if agg["val"]["net"] < 0:
        fitness -= cfg.bad_val_penalty

    return {
        "ind": ind,
        "fitness": float(fitness),
        "train_score": float(train_score),
        "val_score": float(val_score),
        "test_score": float(test_score),
        "avg": agg,
    }


# =========================
# Checkpointing
# =========================

def _cfg_fingerprint(cfg: GAConfig) -> Dict[str, Any]:
    # stable subset for mismatch detection
    keep = asdict(cfg)
    return keep


def _save_checkpoint(root: Path, symbol: str, payload: Dict[str, Any]) -> None:
    _ensure_dir(root)
    ck = root / "checkpoint_latest.json"
    _json_dump(ck, payload)


def _load_checkpoint(root: Path) -> Optional[Dict[str, Any]]:
    ck = root / "checkpoint_latest.json"
    if not ck.exists():
        return None
    try:
        return _json_load(ck)
    except Exception:
        return None


def _append_history(hist_path: Path, row: Dict[str, Any]) -> None:
    _ensure_dir(hist_path.parent)
    header = not hist_path.exists()
    cols = list(row.keys())
    line = ",".join(str(row[c]) for c in cols)
    if header:
        with open(hist_path, "w") as f:
            f.write(",".join(cols) + "\n")
            f.write(line + "\n")
    else:
        with open(hist_path, "a") as f:
            f.write(line + "\n")
# =========================
# RNG state packing (JSON-safe)
# =========================

def _pack_py_random_state(state) -> Dict[str, Any]:
    version, internal, gauss_next = state
    return {"version": int(version), "internal": list(internal), "gauss_next": gauss_next}


def _unpack_py_random_state(packed: Dict[str, Any]):
    version = int(packed["version"])
    internal = packed["internal"]
    if isinstance(internal, list):
        internal = tuple(tuple(x) if isinstance(x, list) else x for x in internal)
    gauss_next = packed.get("gauss_next", None)
    return (version, internal, gauss_next)


def _pack_np_random_state(state) -> Dict[str, Any]:
    bitgen, keys, pos, has_gauss, cached = state
    return {
        "bitgen": str(bitgen),
        "keys": keys.tolist() if hasattr(keys, "tolist") else list(keys),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached),
    }


def _unpack_np_random_state(packed: Dict[str, Any]):
    bitgen = packed["bitgen"]
    keys = np.array(packed["keys"], dtype=np.uint32)
    pos = int(packed["pos"])
    has_gauss = int(packed["has_gauss"])
    cached = float(packed["cached_gaussian"])
    return (bitgen, keys, pos, has_gauss, cached)


# =========================
# Public API
# =========================

def run_ga_montecarlo(
    symbol: str,
    df: pd.DataFrame,
    seed_params: Dict[str, Any],
    cfg: GAConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Main entrypoint.
    Returns (best_params_dict, report_dict).
    Also saves progress + best params on disk.
    """
    root = _project_root()
    out_root = root / "artifacts" / "ga" / symbol
    runs_root = out_root / "runs"
    _ensure_dir(runs_root)

    run_id = _utc_tag()
    run_dir = runs_root / run_id
    _ensure_dir(run_dir)

    # Normalize data
    df = df.copy()
    if "Timestamp" not in df.columns:
        raise ValueError("df must contain Timestamp column")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # Normalize seed params
    seed = _norm_params(seed_params)
    seed["two_candle_confirm"] = bool(cfg.two_candle_confirm)
    seed["require_trade_cycles"] = bool(cfg.require_trade_cycles)

    # Resume if possible
    start_gen = 0
    population: List[Dict[str, Any]] = []
    best_overall = None

    ck = _load_checkpoint(out_root) if cfg.resume else None
    if ck is not None:
        if ck.get("cfg_fingerprint") == _cfg_fingerprint(cfg) and ck.get("symbol") == symbol:
            start_gen = int(ck.get("gen", 0)) + 1
            population = ck.get("population", [])
            best_overall = ck.get("best_overall", None)

            # continue same run_id so files stay together
            if "run_id" in ck:
                run_id = str(ck["run_id"])

            if "py_random_state" in ck:
                random.setstate(_unpack_py_random_state(ck["py_random_state"]))
            if "np_random_state" in ck:
                np.random.set_state(_unpack_np_random_state(ck["np_random_state"]))

            print(f"[GA] Resuming: gen={start_gen} pop={len(population)} run_id={run_id}", flush=True)
        else:
            print("[GA] Checkpoint exists but config mismatch -> starting fresh", flush=True)

    if not population:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        population = [seed]
        while len(population) < cfg.pop_size:
            population.append(mutate_params(seed, cfg.mutation_strength, rate=1.0))

    hist_path = out_root / "history.csv"
    best_path = out_root / "best_params.json"
    active_path = root / "data" / "metadata" / "params" / f"{symbol}_active_params.json"
    _ensure_dir(active_path.parent)

    cfg_dict = asdict(cfg)

    with Pool(processes=cfg.n_procs, initializer=_init_worker, initargs=(df,)) as pool:
        for gen in range(start_gen, cfg.generations):
            t0 = time.time()
            splits = make_mc_splits(df, cfg, gen)

            tasks = [(ind, symbol, cfg_dict, splits) for ind in population]
            scored = pool.map(_eval_worker, tasks)
            scored.sort(key=lambda x: x["fitness"], reverse=True)

            best = scored[0]
            avg = best["avg"]
            top_k = min(8, len(scored))
            gen_payload = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "run_id": run_id,
                "gen": gen,
                "splits": splits,
                "best": best,
                "top": scored[:top_k],
            }
            _json_dump(run_dir / f"gen_{gen:03d}.json", gen_payload)

            # Print progress
            print(
                f"[gen {gen:02d}] best_score={best['fitness']:.2f} "
                f"net={avg['val']['net']:.2f} trades={avg['val']['trades']:.1f} "
                f"win={avg['val']['win']:.2f}% dd={avg['val']['dd']*100:.1f}% pf={avg['val']['pf']:.2f} "
                f"(train_net={avg['train']['net']:.2f} test_net={avg['test']['net']:.2f})",
                flush=True
            )

            # Track best overall (by fitness)
            improved = False
            if best_overall is None or best["fitness"] > best_overall["fitness"]:
                best_overall = best
                improved = True
                _json_dump(best_path, best_overall["ind"])

                # also write active params
                active_payload = {
                    "symbol": symbol,
                    "params": best_overall["ind"],
                    "meta": {
                        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
                        "run_id": run_id,
                        "gen": gen,
                        "fitness": best_overall["fitness"],
                        "avg": best_overall["avg"],
                        "cfg": cfg_dict,
                    },
                }
                _json_dump(active_path, active_payload)

                # per-run snapshot
                _json_dump(run_dir / f"best_gen_{gen:02d}.json", active_payload)

            # History row
            row = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "gen": gen,
                "fitness": round(best["fitness"], 6),
                "val_net": round(avg["val"]["net"], 6),
                "val_trades": round(avg["val"]["trades"], 6),
                "val_win": round(avg["val"]["win"], 6),
                "val_dd": round(avg["val"]["dd"], 6),
                "val_pf": round(avg["val"]["pf"], 6),
                "train_net": round(avg["train"]["net"], 6),
                "test_net": round(avg["test"]["net"], 6),
                "improved": int(improved),
                "sec": round(time.time() - t0, 3),
            }
            _append_history(hist_path, row)

            # Checkpoint every generation
            ck_payload = {
                "symbol": symbol,
                "run_id": run_id,
                "gen": gen,
                "cfg_fingerprint": _cfg_fingerprint(cfg),
                "population": population,
                "best_overall": best_overall,
                "py_random_state": _pack_py_random_state(random.getstate()),
                "np_random_state": _pack_np_random_state(np.random.get_state()),
            }

            _save_checkpoint(out_root, symbol, ck_payload)

            # Breed next population
            elites = [x["ind"] for x in scored[: max(2, cfg.elite_k)]]
            new_pop = elites.copy()
            while len(new_pop) < cfg.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = crossover(p1, p2)
                child = mutate_params(child, cfg.mutation_strength, cfg.mutation_rate)
                new_pop.append(child)
            population = new_pop

    # Final report
    report = {
        "symbol": symbol,
        "run_id": run_id,
        "saved": {
            "checkpoint_latest": str((out_root / "checkpoint_latest.json").resolve()),
            "history_csv": str(hist_path.resolve()),
            "best_params": str(best_path.resolve()),
            "active_params": str(active_path.resolve()),
            "run_dir": str(run_dir.resolve()),
        },
        "best_overall": best_overall,
        "cfg": cfg_dict,
    }
    _json_dump(run_dir / "final_report.json", report)

    # Return best
    best_params = best_overall["ind"] if best_overall else seed
    return best_params, report


# Backward-compatible alias
def run_ga(*args, **kwargs):
    return run_ga_montecarlo(*args, **kwargs)
