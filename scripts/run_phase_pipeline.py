#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: str) -> int:
    print(f"$ {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=str(PROJECT_ROOT))
    return int(p.returncode)


def run(args: argparse.Namespace) -> int:
    phase = str(args.phase).upper().strip()
    continue_on_fail = int(args.continue_on_fail) == 1

    cmds: list[str] = []
    if phase == "A":
        cmd = ".venv/bin/python scripts/compare_baseline_pipelines.py"
        if str(args.tight_dir).strip():
            cmd += f" --tight-dir {shlex.quote(str(args.tight_dir))}"
        if str(args.ga_dir).strip():
            cmd += f" --ga-dir {shlex.quote(str(args.ga_dir))}"
        cmds = [cmd]
    elif phase == "B":
        symbol = str(args.symbol).strip().upper() if str(args.symbol).strip() else str(args.symbols).split(",")[0].strip().upper()
        sig_csv = f"data/signals/{symbol}_signals_1h.csv"
        cmds = [
            (
                ".venv/bin/python -m src.execution.ga_exec_3m_opt "
                f"--symbol {symbol} --signals-csv {sig_csv} --max-signals {int(args.max_signals)} "
                f"--mode {args.mode} --force-no-skip 1 --pop {int(args.pop)} --gens {int(args.gens)} "
                f"--workers {int(args.workers)} --seed {int(args.seed)} "
                "--hard-max-taker-share 1.0 --hard-max-median-fill-delay-min 180 --hard-max-p95-fill-delay-min 360 "
                "--outdir reports/execution_layer"
            ),
            ".venv/bin/python scripts/ga_patch_report.py",
        ]
    elif phase == "C":
        sym_arg = str(args.symbol).strip().upper() if str(args.symbol).strip() else str(args.symbols).strip().upper()
        cmd = (
            ".venv/bin/python scripts/exit_sweep.py "
            f"--{'symbol' if ',' not in sym_arg else 'symbols'} {sym_arg} "
            f"--mode {args.mode} --max-signals {int(args.max_signals)} --wf-splits {int(args.wf_splits)} "
            f"--max-configs {int(args.max_configs)} --outdir reports/execution_layer"
        )
        cmds = [cmd]
    elif phase in {"D", "E", "F", "G", "H"}:
        print(f"Phase {phase} is not yet implemented in this orchestrator.", flush=True)
        return 3
    else:
        print(f"Unknown phase: {phase}", flush=True)
        return 2

    rc = 0
    for c in cmds:
        rc = _run(c)
        if rc != 0 and not continue_on_fail:
            return rc
    return rc


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run execution-layer roadmap phases.")
    ap.add_argument("--phase", required=True, help="A|B|C|D|E|F|G|H")
    ap.add_argument("--resume", default="")
    ap.add_argument("--symbols", default="SOLUSDT,AVAXUSDT,NEARUSDT")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--mode", choices=["tight", "normal"], default="normal")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--continue-on-fail", type=int, default=0)

    ap.add_argument("--tight-dir", default="")
    ap.add_argument("--ga-dir", default="")

    ap.add_argument("--max-signals", type=int, default=2000)
    ap.add_argument("--wf-splits", type=int, default=5)
    ap.add_argument("--max-configs", type=int, default=0)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=2)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()

