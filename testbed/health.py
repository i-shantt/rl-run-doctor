"""Deciding whether a run failed.

Two rules, both learned the hard way elsewhere:

1. **Never the training return.** Several pathologies leave it flat or raise it. The label comes
   only from the held-out evaluator.

2. **Never a single snapshot.** One greedy evaluation of a DQN policy is close to a coin flip;
   measured on CartPole, scoring the final eval alone rather than a trailing window inflated the
   across-seed standard deviation from 68 to 161 on the same runs. Health is a window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .telemetry import load_trace

WINDOW = 3


@dataclass(frozen=True)
class RunHealth:
    path: str
    eval_curve: list[float]
    eval_steps: list[int]
    windowed: float  # mean of the last WINDOW evaluations
    peak: float
    max_drawdown: float  # largest fall from a running peak, in eval units

    @property
    def n_evals(self) -> int:
        return len(self.eval_curve)


def run_health(path: str | Path, window: int = WINDOW) -> RunHealth:
    recs = load_trace(path, include_oracle=True)
    curve: list[float] = []
    steps: list[int] = []
    for r in recs:
        v = r.get("oracle", {}).get("eval_return")
        if v is not None:
            curve.append(float(v))
            steps.append(int(r["env_steps"]))
    if not curve:
        raise ValueError(f"{path} contains no held-out evaluations")

    arr = np.asarray(curve, dtype=np.float64)
    running_peak = np.maximum.accumulate(arr)
    drawdown = float(np.max(running_peak - arr))
    return RunHealth(
        path=str(path),
        eval_curve=curve,
        eval_steps=steps,
        windowed=float(arr[-window:].mean()),
        peak=float(arr.max()),
        max_drawdown=drawdown,
    )


@dataclass(frozen=True)
class HealthyBand:
    """The healthy control distribution, from which the failure threshold is derived."""

    n: int
    mean: float
    std: float
    lo: float  # 5th percentile of control windowed health
    threshold: float  # a run at or below this counts as failed

    def failed(self, windowed: float) -> bool:
        return windowed <= self.threshold


def healthy_band(control_windowed: list[float], q: float = 5.0) -> HealthyBand:
    """Derive the failure threshold from controls rather than picking a number.

    The threshold is the `q`-th percentile of control health. A pathology only counts as a failure
    vehicle if it pushes runs below the range the healthy control already occupies -- which, for a
    high-variance algorithm like DQN, is a genuinely demanding bar and is meant to be.
    """
    if len(control_windowed) < 2:
        raise ValueError("need at least 2 control runs to derive a band")
    arr = np.asarray(control_windowed, dtype=np.float64)
    lo = float(np.percentile(arr, q))
    return HealthyBand(
        n=len(arr),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)),
        lo=lo,
        threshold=lo,
    )
