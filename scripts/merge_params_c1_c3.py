# scripts/merge_params_c1_c3.py
import argparse, json
from copy import deepcopy
from datetime import datetime, timezone

CYCLE_KEYS = ["willr_by_cycle", "tp_mult_by_cycle", "sl_mult_by_cycle", "exit_rsi_by_cycle"]

def load_params(path: str):
    with open(path, "r") as f:
        obj = json.load(f)
    # active_params files usually look like {"symbol":..., "params":..., "meta":...}
    if isinstance(obj, dict) and "params" in obj and isinstance(obj["params"], dict):
        return obj, obj["params"]
    # otherwise treat whole json as params dict
    return {"symbol": None, "params": obj, "meta": {}}, obj

def ensure_list5(p, k):
    v = p.get(k)
    if not (isinstance(v, list) and len(v) == 5):
        raise ValueError(f"{k} must be a list of length 5. Got: {type(v)} len={len(v) if isinstance(v,list) else 'NA'}")
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", required=True, help="Cycle-1 active params json (e.g. BTCUSDT_C1_active_params.json)")
    ap.add_argument("--c3", required=True, help="Cycle-3/primary active params json (e.g. BTCUSDT_active_params.json)")
    ap.add_argument("--out", required=True, help="Output combined json (e.g. BTCUSDT_C13_active_params.json)")
    ap.add_argument("--base", choices=["c1","c3"], default="c3",
                    help="Which file provides the GLOBAL (scalar) params. Default: c3 (usually safer to preserve cycle3 edge).")
    args = ap.parse_args()

    c1_obj, c1 = load_params(args.c1)
    c3_obj, c3 = load_params(args.c3)

    base_obj, base_p = (c1_obj, deepcopy(c1)) if args.base == "c1" else (c3_obj, deepcopy(c3))

    # Validate per-cycle lists exist
    for k in CYCLE_KEYS:
        ensure_list5(c1, k)
        ensure_list5(c3, k)
        ensure_list5(base_p, k)

    # Patch: cycle index 1 from c1, cycle index 3 from c3
    for k in CYCLE_KEYS:
        base_p[k][1] = float(c1[k][1])
        base_p[k][3] = float(c3[k][3])

    # Make sure we actually trade both cycles
    base_p["trade_cycles"] = [1, 3]
    base_p["require_trade_cycles"] = True

    # Keep strict no-lookahead defaults (don’t let these drift)
    base_p["cycle_shift"] = 1
    base_p["cycle_fill"]  = int(base_p.get("cycle_fill", 2))
    base_p["two_candle_confirm"] = bool(base_p.get("two_candle_confirm", False))

    out_obj = {
        "symbol": base_obj.get("symbol") or c1_obj.get("symbol") or c3_obj.get("symbol") or "BTCUSDT",
        "params": base_p,
        "meta": {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "merged_from": {"c1": args.c1, "c3": args.c3},
            "base": args.base,
            "note": "Merged cycle-1 per-cycle values from C1 and cycle-3 per-cycle values from C3. Global params taken from base.",
        },
    }

    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2)

    print(f"OK wrote: {args.out}")
    print("trade_cycles =", out_obj["params"].get("trade_cycles"))
    for k in CYCLE_KEYS:
        print(k, "C1=", out_obj["params"][k][1], "C3=", out_obj["params"][k][3])

if __name__ == "__main__":
    main()
