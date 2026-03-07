#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ART_DIR = PROJECT_ROOT / "artifacts" / "params"
META_DIR = PROJECT_ROOT / "data" / "metadata" / "params"
OUT_DIR = PROJECT_ROOT / "output"

META_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["ADAUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "BNBUSDT", "XRPUSDT", "BTCUSDT"]
USE_GA_FOR = {"BTCUSDT", "XRPUSDT"}  # per your request


def _unwrap_params(d: dict) -> dict:
    if isinstance(d, dict) and isinstance(d.get("params"), dict):
        return d["params"]
    return d


def _find_artifacts_active(symbol: str) -> Path:
    # supports both "_active.json" and "active.json"
    p1 = ART_DIR / symbol / "_active.json"
    p2 = ART_DIR / symbol / "active.json"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing artifacts active params for {symbol}: tried {p1} and {p2}")


def _find_ga_best(symbol: str) -> Path:
    """
    Try to auto-find GA-produced params for symbol under output/.
    Looks for files containing the symbol and 'params'/'best' in name.
    """
    if not OUT_DIR.exists():
        raise FileNotFoundError("output/ folder not found, cannot auto-locate GA params")

    candidates = []
    for fp in OUT_DIR.rglob("*.json"):
        name = fp.name.lower()
        if symbol.lower() in name and ("param" in name or "best" in name):
            candidates.append(fp)

    # prefer files deeper in symbol folder and with 'best' in name
    def score(fp: Path) -> int:
        n = fp.name.lower()
        s = 0
        if "best" in n: s += 50
        if "active" in n: s += 10
        if "seed" in n: s += 5
        if symbol.lower() in str(fp.parent).lower(): s += 20
        s += len(str(fp)) // 50
        return s

    candidates.sort(key=score, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Could not auto-find GA params for {symbol} under output/.")
    return candidates[0]


def _load_json(fp: Path) -> dict:
    with open(fp, "r") as f:
        return json.load(f)


def _write_seed(symbol: str, params: dict, source: Path) -> Path:
    params = dict(params)
    params["trade_cycles"] = [3]  # enforce cycle 3 only

    out_fp = META_DIR / f"{symbol}_seed_params.json"
    payload = {"params": params, "_source": str(source)}
    out_fp.write_text(json.dumps(payload, indent=2))
    return out_fp


def main():
    print(f"[INFO] Writing seeds into: {META_DIR}")

    for sym in SYMBOLS:
        try:
            if sym in USE_GA_FOR:
                src = _find_ga_best(sym)
                raw = _load_json(src)
                params = _unwrap_params(raw)
                out_fp = _write_seed(sym, params, src)
                print(f"[OK] {sym}: seed <- GA {src} -> {out_fp}")
            else:
                src = _find_artifacts_active(sym)
                raw = _load_json(src)
                params = _unwrap_params(raw)
                out_fp = _write_seed(sym, params, src)
                print(f"[OK] {sym}: seed <- ACTIVE {src} -> {out_fp}")
        except Exception as e:
            print(f"[FAIL] {sym}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
