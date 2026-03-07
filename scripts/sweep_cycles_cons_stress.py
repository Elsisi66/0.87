#!/usr/bin/env python3
from __future__ import annotations
import os, sys, json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SYMBOLS = ["ETHUSDT","AVAXUSDT","BNBUSDT"]
CYCLES = [1,2,3,4,5]
FEE_BPS=25.0
SLIP_BPS=10.0
RISK=0.005
MAX_ALLOC=0.30
TF="1h"

def load_df(symbol: str) -> pd.DataFrame:
    fp = PROJECT_ROOT/"data/processed/_full"/f"{symbol}_{TF}_full.parquet"
    if not fp.exists():
        raise FileNotFoundError(fp)
    df = pd.read_parquet(fp)
    if "Timestamp" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp":"Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    return df

def load_active(symbol: str) -> dict:
    fp = PROJECT_ROOT/"data/metadata/params"/f"{symbol}_active_params.json"
    d = json.load(open(fp))
    return d.get("params", d)

def trades_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df is None or len(trades_df)==0:
        return dict(trades=0,wins=0,losses=0,win_rate_pct=0.0,net_profit=0.0,profit_factor=0.0)
    cand=[c for c in trades_df.columns if any(k in c.lower() for k in ["pnl","profit","net"])]
    pnl=None
    for c in cand:
        s=pd.to_numeric(trades_df[c], errors="coerce")
        if s.notna().sum()>0:
            pnl=c; break
    if pnl is None:
        return dict(trades=len(trades_df),wins=0,losses=0,win_rate_pct=0.0,net_profit=0.0,profit_factor=0.0)
    p=pd.to_numeric(trades_df[pnl], errors="coerce").fillna(0.0)
    wins=p[p>0]; losses=p[p<0]
    pf = wins.sum()/(-losses.sum()) if (-losses.sum())>0 else float("inf") if wins.sum()>0 else 0.0
    return dict(
        trades=int(len(p)),
        wins=int((p>0).sum()),
        losses=int((p<0).sum()),
        win_rate_pct=float((p>0).mean()*100.0) if len(p) else 0.0,
        net_profit=float(p.sum()),
        profit_factor=float(pf),
    )

def main():
    from src.bot087.optim.ga import _norm_params, _ensure_indicators, run_backtest_long_only

    rows=[]
    for sym in SYMBOLS:
        p = load_active(sym)
        # conservative sizing
        p["risk_per_trade"]=RISK
        p["max_allocation"]=MAX_ALLOC

        df = load_df(sym)

        # compute indicators once with a normalized base
        p_base = _norm_params(dict(p))
        df_ind = _ensure_indicators(df.copy(), p_base)

        for c in CYCLES:
            p2 = dict(p)
            p2["trade_cycles"]=[c]
            p2 = _norm_params(p2)

            trades, _m = run_backtest_long_only(
                df_ind,
                symbol=sym,
                p=p2,
                initial_equity=10_000.0,
                fee_bps=FEE_BPS,
                slippage_bps=SLIP_BPS,
            )
            tdf = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame(trades)
            m = trades_metrics(tdf)
            rows.append(dict(symbol=sym, cycles=str([c]), **m))

            print(sym, "cycle", c, "-> trades", m["trades"], "net", round(m["net_profit"],2), "pf", round(m["profit_factor"],4))
    out = PROJECT_ROOT/"output"/"stress_reports"
    out.mkdir(parents=True, exist_ok=True)
    fp = out/"cycle_sweep_cons_stress.csv"
    pd.DataFrame(rows).to_csv(fp, index=False)
    print("\n[SAVED]", fp)

if __name__=="__main__":
    main()
