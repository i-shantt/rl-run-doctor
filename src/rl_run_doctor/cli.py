"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .api import diagnose, diagnose_prefixes
from .trace import Trace


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="rl-run-doctor", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="name the likely cause of a run's behaviour")
    d.add_argument("trace")
    d.add_argument("--model", required=True, help="fitted diagnoser JSON")
    d.add_argument("--prefixes", action="store_true", help="print the score at every prefix")

    i = sub.add_parser("inspect", help="show what signals a trace carries")
    i.add_argument("trace")

    args = ap.parse_args(argv)

    if args.cmd == "inspect":
        t = Trace(args.trace)
        print(f"{Path(args.trace).name}: {len(t)} records, {len(t.keys)} signals")
        print(f"  env steps {t.env_steps[0]} .. {t.env_steps[-1]}")
        for k in t.keys:
            v = t.get(k)
            print(f"  {k:<28} first={v[0]:>12.4f}  last={v[-1]:>12.4f}")
        return 0

    if args.prefixes:
        for step, score in diagnose_prefixes(args.trace, args.model):
            print(f"{step:>10}  {score:.4f}")
        return 0

    print(diagnose(args.trace, args.model))
    return 0


if __name__ == "__main__":
    sys.exit(main())
