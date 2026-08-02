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


REL_MARGIN = 0.02


@dataclass(frozen=True)
class HealthyBand:
    """The healthy control distribution, from which the failure threshold is derived."""

    n: int
    mean: float
    std: float
    lo: float  # q-th percentile of control windowed health
    margin: float
    threshold: float  # a run *strictly below* this counts as failed

    def failed(self, windowed: float) -> bool:
        return windowed < self.threshold


def healthy_band(
    control_windowed: list[float], q: float = 5.0, rel_margin: float = REL_MARGIN
) -> HealthyBand:
    """Derive the failure threshold from controls rather than picking a number.

    Two details that are not cosmetic:

    **Strictly below, plus a margin.** With `<=` and no margin, a saturated control makes every
    run that matches it perfectly count as a failure. Measured here: `chain_rho/ppo` controls both
    scored 1.000, so the floor was 1.000, and four pathologies that also scored 1.000 were reported
    as reliable failure vehicles. The margin is a proportion of the control mean, so it scales with
    whatever the return happens to be measured in.

    **The floor is a percentile of the controls, not their mean.** A pathology has to push a run
    below the range the healthy control already occupies. For a high-variance algorithm like DQN
    that is a demanding bar, and it is meant to be -- anything easier is measuring seed noise.
    """
    if len(control_windowed) < 2:
        raise ValueError("need at least 2 control runs to derive a band")
    arr = np.asarray(control_windowed, dtype=np.float64)
    lo = float(np.percentile(arr, q))
    margin = max(rel_margin * abs(float(arr.mean())), 1e-9)
    return HealthyBand(
        n=len(arr),
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        lo=lo,
        margin=margin,
        threshold=lo - margin,
    )


def random_policy_return(env_name: str, env_kwargs: dict, n_episodes: int, seed: int = 0) -> float:
    """Mean return of a uniform-random policy on the held-out seeds.

    The reference for "is the control actually learning anything". Without it a cell whose control
    never leaves the floor still produces a floor, and every pathology gets compared against a
    broken baseline.
    """
    from .envs import eval_seeds, make

    rng = np.random.default_rng(seed)
    env = make(env_name, **env_kwargs)
    totals = []
    for s in eval_seeds(n_episodes):
        env.reset(seed=s)
        total = 0.0
        for _ in range(env.max_steps):
            res = env.step(int(rng.integers(0, env.n_actions)))
            total += res.reward
            if res.terminated or res.truncated:
                break
        totals.append(total)
    return float(np.mean(totals))
