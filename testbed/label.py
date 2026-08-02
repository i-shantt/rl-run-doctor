"""Labelling runs from the held-out evaluator.

This is the only module allowed to read the `oracle` block. It converts a trace into (a) whether
the run failed and (b) *when* it began to fail, which is the reference point lead time is measured
against.

"When it failed" is deliberately the first evaluation of a **persistent** fall below the healthy
floor, not the first dip. A single greedy evaluation is noisy enough that a momentary dip is
common in healthy runs, and anchoring lead time to noise would make an early-warning claim
meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .telemetry import load_trace

PERSIST = 2  # consecutive sub-floor evaluations required to call it a real degradation


@dataclass(frozen=True)
class RunLabel:
    run_id: str
    pathology: str
    env_name: str
    algo: str
    seed: int
    failed: bool
    windowed: float
    # Environment step at which degradation is first confirmed. None for healthy runs.
    degrade_step: int | None
    eval_steps: list[int]
    eval_curve: list[float]


def label_run(path: str | Path, floor: float, persist: int = PERSIST) -> RunLabel:
    recs = load_trace(path, include_oracle=True)
    meta_spec = _read_spec(path)

    curve: list[float] = []
    steps: list[int] = []
    for r in recs:
        v = r.get("oracle", {}).get("eval_return")
        if v is not None:
            curve.append(float(v))
            steps.append(int(r["env_steps"]))
    if not curve:
        raise ValueError(f"{path} has no held-out evaluations")

    arr = np.asarray(curve)
    windowed = float(arr[-3:].mean())

    degrade_step: int | None = None
    below = arr <= floor
    for i in range(len(below) - persist + 1):
        if below[i : i + persist].all():
            degrade_step = steps[i]
            break

    failed = windowed <= floor
    # A run whose trailing window is healthy did not fail, whatever happened mid-run.
    if not failed:
        degrade_step = None

    return RunLabel(
        run_id=meta_spec["run_id"],
        pathology=meta_spec["pathology"],
        env_name=meta_spec["env_name"],
        algo=meta_spec["algo"],
        seed=meta_spec["seed"],
        failed=failed,
        windowed=windowed,
        degrade_step=degrade_step,
        eval_steps=steps,
        eval_curve=curve,
    )


def _read_spec(path: str | Path) -> dict:
    from .telemetry import read_meta

    meta = read_meta(path)
    spec = meta["spec"]
    return {
        "run_id": meta["run_id"],
        "pathology": spec["pathology"],
        "env_name": spec["env_name"],
        "algo": spec["algo"],
        "seed": int(spec["seed"]),
    }
