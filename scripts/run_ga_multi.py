# scripts/run_ga_multi.py
import os
import sys
import argparse
import importlib.util
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["BOT087_PROJECT_ROOT"] = str(PROJECT_ROOT)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _import_run_one_symbol():
    """
    Import run_one_symbol from scripts/run_ga_symbol.py without relying on 'scripts' being a package.
    """
    mod_path = PROJECT_ROOT / "scripts" / "run_ga_symbol.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"Missing: {mod_path}")

    spec = importlib.util.spec_from_file_location("run_ga_symbol", str(mod_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for: {mod_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    if not hasattr(module, "run_one_symbol"):
        raise AttributeError("run_ga_symbol.py does not define run_one_symbol()")

    return module.run_one_symbol


DEFAULT_SYMBOLS = ["XRPUSDT", "ADAUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "BNBUSDT"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated list. Default: XRP,ADA,ETH,SOL,AVAX,BNB",
    )
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--n-procs", type=int, default=3)
    ap.add_argument("--smoke-rows", type=int, default=20000)
    args = ap.parse_args()

    run_one_symbol = _import_run_one_symbol()

    symbols: List[str] = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    for sym in symbols:
        run_one_symbol(
            symbol=sym,
            tf=args.tf,
            resume=(not args.no_resume),
            n_procs=args.n_procs,
            smoke_rows=args.smoke_rows,
        )


if __name__ == "__main__":
    main()
