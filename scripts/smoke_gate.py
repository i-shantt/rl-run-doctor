"""Per-cell smoke gate. Run this before building any corpus.

For every (env, algo, pathology) cell it asks two questions:

  1. Is the **control** healthy on this env/algo? If not, the cell is unusable -- there is nothing
     to fall from, and a failure label would be meaningless.
  2. Does the **injection** push runs below the healthy band derived from the controls?

Cells that fail either question are dropped and printed. This exists because a sibling project
measured that most induced pathologies leave final performance intact, and that collapse-proneness
was strongly task-specific: 26 conditions produced only 2 reliable failure vehicles, and one task
gave zero collapses across 29 cells. Discovering that after generating a full corpus is expensive;
discovering it here is cheap.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from testbed.corpus.runner import RunSpec, run_one
from testbed.health import healthy_band, run_health
from testbed.inject.pathologies import applicable

ENVS = ["cartpole", "deep_sea", "chain_rho"]
ALGOS = ["ppo", "dqn"]


def _run(args: tuple[RunSpec, str]) -> tuple[str, float, float, int, float]:
    spec, out_dir = args
    t0 = time.time()
    path = run_one(spec, out_dir)
    h = run_health(path)
    return spec.run_id, h.windowed, h.peak, h.n_evals, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", default="scratch/smoke")
    ap.add_argument("--workers", type=int, default=min(10, mp.cpu_count()))
    ap.add_argument("--envs", nargs="*", default=ENVS)
    ap.add_argument("--algos", nargs="*", default=ALGOS)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[RunSpec, str]] = []
    for env_name, algo in itertools.product(args.envs, args.algos):
        for path in applicable(algo, env_name):  # type: ignore[arg-type]
            for seed in range(args.seeds):
                jobs.append((RunSpec(env_name, algo, path.name, seed), str(out_dir)))

    print(f"{len(jobs)} runs across {len(args.envs)}x{len(args.algos)} cells "
          f"on {args.workers} workers\n")
    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        results = list(pool.imap_unordered(_run, jobs))
    print(f"all runs finished in {time.time()-t0:.0f}s\n")

    by_run = {rid: (w, pk, n, dt) for rid, w, pk, n, dt in results}

    report: dict[str, dict] = {}
    for env_name, algo in itertools.product(args.envs, args.algos):
        paths = applicable(algo, env_name)  # type: ignore[arg-type]
        if not paths:
            continue
        controls = [
            by_run[RunSpec(env_name, algo, "P0_control", s).run_id][0]
            for s in range(args.seeds)
        ]
        band = healthy_band(controls, q=0.0)  # with few seeds, use the worst control as the floor

        print(f"=== {env_name} / {algo} ===")
        print(f"  control: {np.round(controls,1).tolist()}  -> floor {band.threshold:.1f}")

        cell: dict[str, dict] = {}
        for p in paths:
            if p.is_control:
                continue
            vals = [
                by_run[RunSpec(env_name, algo, p.name, s).run_id][0] for s in range(args.seeds)
            ]
            n_failed = sum(1 for v in vals if band.failed(v))
            verdict = "KEEP" if n_failed == args.seeds else (
                "partial" if n_failed else "NO EFFECT"
            )
            drop = band.mean - float(np.mean(vals))
            print(
                f"  {p.name:<24} {np.round(vals,1).tolist():<22} "
                f"drop {drop:+7.1f}  {n_failed}/{args.seeds} below floor  [{verdict}]"
            )
            cell[p.name] = {
                "values": vals,
                "n_below_floor": n_failed,
                "verdict": verdict,
                "mean_drop_vs_control": drop,
            }
        report[f"{env_name}/{algo}"] = {
            "control": controls,
            "floor": band.threshold,
            "control_mean": band.mean,
            "pathologies": cell,
        }
        print()

    keep = [
        f"{cell}:{p}"
        for cell, d in report.items()
        for p, v in d["pathologies"].items()
        if v["verdict"] == "KEEP"
    ]
    total = sum(len(d["pathologies"]) for d in report.values())
    print(f"SUMMARY: {len(keep)}/{total} pathology-cells reliably degrade performance")
    for k in keep:
        print(f"  keep  {k}")

    dest = out_dir / "smoke_report.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
