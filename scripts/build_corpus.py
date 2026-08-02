"""Build the labelled corpus, using only cells the smoke gate cleared.

Reads the gate's report and includes a pathology-cell only if it reliably degraded held-out
performance across every smoke seed. Cells that did not are recorded in the manifest as excluded,
with their measured effect, so the corpus documents what failed to fail rather than silently
omitting it.

Controls get more seeds than pathologies: the healthy band is what every failure label is measured
against, so its width is the thing least worth being uncertain about.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

from testbed.corpus.runner import RunSpec, run_one
from testbed.health import healthy_band, run_health
from testbed.label import label_run


def _run(args: tuple[RunSpec, str]) -> tuple[str, str, float]:
    spec, out_dir = args
    path = run_one(spec, out_dir)
    return spec.run_id, str(path), run_health(path).windowed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", default="scratch/smoke/smoke_report.json")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--seeds", type=int, default=8, help="seeds per pathology cell")
    ap.add_argument("--control-seeds", type=int, default=16)
    ap.add_argument("--workers", type=int, default=min(10, mp.cpu_count()))
    ap.add_argument(
        "--include-partial",
        action="store_true",
        help="also include cells that degraded on some but not all smoke seeds",
    )
    args = ap.parse_args()

    smoke = json.loads(Path(args.smoke).read_text())
    out_dir = Path(args.out) / "traces"
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted = {"KEEP"} | ({"partial"} if args.include_partial else set())

    jobs: list[tuple[RunSpec, str]] = []
    included: list[dict] = []
    excluded: list[dict] = []
    for cell, d in sorted(smoke.items()):
        env_name, algo = cell.split("/")
        kept = [p for p, v in d["pathologies"].items() if v["verdict"] in accepted]
        for p, v in sorted(d["pathologies"].items()):
            record = {"cell": cell, "pathology": p, **v}
            (included if p in kept else excluded).append(record)
        if not kept:
            continue  # a cell with no working pathology contributes no controls either
        for s in range(args.control_seeds):
            jobs.append((RunSpec(env_name, algo, "P0_control", s), str(out_dir)))
        for p in kept:
            for s in range(args.seeds):
                jobs.append((RunSpec(env_name, algo, p, s), str(out_dir)))

    if not jobs:
        print("no cells survived the smoke gate; there is no corpus to build.")
        print("that is a result, not an error -- see docs/NEGATIVE_RESULTS.md")
        return

    print(f"{len(jobs)} runs from {len(included)} accepted cells "
          f"({len(excluded)} excluded) on {args.workers} workers")
    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        results = list(pool.imap_unordered(_run, jobs))
    print(f"finished in {time.time()-t0:.0f}s")

    # Re-derive the healthy band from the full control set, then label everything against it.
    by_cell: dict[str, list[str]] = {}
    for run_id, path, _w in results:
        env_name, algo, _p, _s = run_id.split("__")
        by_cell.setdefault(f"{env_name}/{algo}", []).append(path)

    manifest: list[dict] = []
    for cell, paths in sorted(by_cell.items()):
        controls = [run_health(p).windowed for p in paths if "P0_control" in p]
        band = healthy_band(controls, q=5.0)
        print(f"{cell}: control n={band.n} mean={band.mean:.1f} sd={band.std:.1f} "
              f"floor={band.threshold:.1f}")
        for p in sorted(paths):
            lab = label_run(p, floor=band.threshold)
            manifest.append(
                {
                    "run_id": lab.run_id,
                    "path": str(Path(p).relative_to(Path(args.out))),
                    "cell": cell,
                    "env": lab.env_name,
                    "algo": lab.algo,
                    "pathology": lab.pathology,
                    "seed": lab.seed,
                    "failed": lab.failed,
                    "windowed_eval": lab.windowed,
                    "degrade_step": lab.degrade_step,
                    "floor": band.threshold,
                }
            )

    dest = Path(args.out)
    (dest / "manifest.jsonl").write_text(
        "\n".join(json.dumps(m, sort_keys=True) for m in manifest) + "\n"
    )
    (dest / "excluded_cells.json").write_text(json.dumps(excluded, indent=2, sort_keys=True))

    n_failed = sum(1 for m in manifest if m["failed"])
    print(f"\n{len(manifest)} runs, {n_failed} labelled failed "
          f"({n_failed/max(len(manifest),1):.0%})")
    by_path: dict[str, list[bool]] = {}
    for m in manifest:
        by_path.setdefault(m["pathology"], []).append(m["failed"])
    for p, flags in sorted(by_path.items()):
        print(f"  {p:<26} {sum(flags):>3}/{len(flags):<3} failed")
    print(f"\nwrote {dest/'manifest.jsonl'} and {dest/'excluded_cells.json'}")


if __name__ == "__main__":
    main()
