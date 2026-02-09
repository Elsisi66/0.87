from __future__ import annotations

import pandas as pd

from src.bot087.execution.execution_eval import ExecutionEvalConfig, evaluate_with_sec_data


def test_window_zero_reproduces_original_fills() -> None:
    trades = pd.DataFrame(
        [
            {
                "symbol": "TESTUSDT",
                "cycle": 1,
                "entry_ts": "2024-01-01T00:00:00Z",
                "exit_ts": "2024-01-01T01:00:00Z",
                "entry_open_raw": 100.0,
                "exit_open_raw": 110.0,
                "entry_px": 100.09,
                "exit_px": 109.901,
                "units": 2.0,
                "fee_bps": 7.0,
                "slippage_bps": 2.0,
                "net_pnl": 19.622,
            }
        ]
    )

    sec_df = pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])

    cfg = ExecutionEvalConfig(
        window_sec=0,
        model="default",
        overlay_mode="none",
        overlay_partial_tp_frac=0.0,
        initial_equity=1_000.0,
        fee_bps=7.0,
        slippage_bps=2.0,
    )

    out = evaluate_with_sec_data(symbol="TESTUSDT", trades=trades, sec_df=sec_df, cfg=cfg)
    tr = out["trade_level"]
    s = out["summary"]

    assert len(tr) == 1
    r = tr.iloc[0]
    assert abs(float(r["entry_fill_new"]) - float(r["entry_fill_old"])) < 1e-12
    assert abs(float(r["exit_fill_new"]) - float(r["exit_fill_old"])) < 1e-12
    assert abs(float(r["pnl_after"]) - float(r["pnl_before"])) < 1e-12

    assert abs(float(s["net_after"]) - float(s["net_before"])) < 1e-12
    assert float(s["fallback_rate"]) == 0.0
    assert float(s["alignment_fail_rate"]) == 0.0
