"""Read `credit_study.jsonl` and ask whether any estimator beats chance by more than noise.

The study's table reports per-cell means, and at one seed those means flip sign between adjacent
densities -- which is what noise looks like. This aggregates the *seed* means and reports a
confidence interval, so a cell is only reported as signal when it clears zero (Spearman) or 0.5
(decision AUC) across seeds. The seed is the unit of resampling: episodes within a seed share a
trained policy and are not independent.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def ci(vals: list[float]) -> tuple[float, float]:
    """Mean and half-width of a 95% interval across seeds, on the finite values only."""
    xs = [v for v in vals if math.isfinite(v)]
    if not xs:
        return float("nan"), float("nan")
    m = float(np.mean(xs))
    if len(xs) < 2:
        return m, float("nan")
    sem = float(np.std(xs, ddof=1)) / math.sqrt(len(xs))
    return m, 1.96 * sem


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/credit_study.jsonl")
    args = ap.parse_args()

    recs = [json.loads(line) for line in Path(args.raw).read_text().splitlines() if line.strip()]
    cells: dict[tuple[int, float, str], dict[str, list[float]]] = defaultdict(
        lambda: {"spearman": [], "decision_auc": []}
    )
    for r in recs:
        for name, sc in r["scores"].items():
            c = cells[r["steps"], r["rho"], name]
            c["spearman"].append(sc["spearman"])
            c["decision_auc"].append(sc["decision_auc"])

    n_seeds = len({r["seed"] for r in recs})
    print(f"{len(recs)} cells, {n_seeds} seeds. 95% CI across seeds.\n")
    print("Cells where the interval excludes the null (Spearman 0, decision AUC 0.5):\n")

    hits = []
    for (steps, rho, name), c in sorted(cells.items()):
        for metric, null in [("spearman", 0.0), ("decision_auc", 0.5)]:
            m, h = ci(c[metric])
            if not math.isfinite(m) or not math.isfinite(h):
                continue
            if abs(m - null) > h:
                hits.append((steps, rho, name, metric, m, h))

    if not hits:
        print("  none.")
    for steps, rho, name, metric, m, h in sorted(hits, key=lambda x: -abs(x[4])):
        print(f"  {steps:>6,} steps  rho={rho:<5} {name:<22} {metric:<13} {m:+.3f} +/- {h:.3f}")

    # The pre-registered prediction is about GAE's Spearman: >0.7 at high density, <0.3 at rho=0.1.
    print("\nPre-registered prediction -- gae(l=0.95) Spearman vs exact credit:")
    for steps in sorted({r["steps"] for r in recs}):
        row = f"  {steps:>6,} steps  "
        for rho in sorted({r["rho"] for r in recs}, reverse=True):
            key = (steps, rho, "gae(l=0.95)")
            m, h = ci(cells[key]["spearman"]) if key in cells else (float("nan"), float("nan"))
            if not math.isfinite(m):
                cell = "n/a"
            else:
                cell = f"{m:+.2f}+/-{h:.2f}" if math.isfinite(h) else f"{m:+.2f}"
            row += f"rho={rho}: {cell:<14}"
        print(row)

    print("\nBest Spearman achieved by any estimator in any cell:")
    means = ((ci(c["spearman"])[0], k) for k, c in cells.items())
    best = max(((m, k) for m, k in means if math.isfinite(m)), default=(float("nan"), None))
    print(f"  {best[0]:+.3f} at {best[1]}")


if __name__ == "__main__":
    main()
