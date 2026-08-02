"""Credit-assignment estimators, scored against exact truth.

Every one of these is what some algorithm actually uses as its per-step learning signal:

  `uniform`        the whole trajectory's return spread evenly -- what a trajectory-level,
                   critic-free method effectively does
  `return_to_go`   discounted future reward from each step (REINFORCE)
  `td0`            one-step temporal difference with a learned value function
  `gae`            generalised advantage estimation, the PPO default

On an environment with an analytically exact per-step contribution, these can be scored directly
rather than judged by the downstream performance of a training run that bundles them with a dozen
confounders. Spearman rather than Pearson: what matters for learning is whether a method *ranks*
the steps that mattered above the ones that did not, not whether it reproduces their magnitude.
"""

from __future__ import annotations

import numpy as np


def uniform_credit(rewards: np.ndarray) -> np.ndarray:
    """Trajectory return, spread evenly over its steps."""
    n = rewards.size
    return np.full(n, float(rewards.sum()) / max(n, 1), dtype=np.float64)


def return_to_go(rewards: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    out = np.zeros_like(rewards, dtype=np.float64)
    acc = 0.0
    for t in range(rewards.size - 1, -1, -1):
        acc = float(rewards[t]) + gamma * acc
        out[t] = acc
    return out


def td0_advantage(rewards: np.ndarray, values: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    n = rewards.size
    next_v = np.append(values[1:], 0.0)
    return rewards + gamma * next_v - values[:n]


def gae_advantage(
    rewards: np.ndarray, values: np.ndarray, gamma: float = 1.0, lam: float = 0.95
) -> np.ndarray:
    n = rewards.size
    next_v = np.append(values[1:], 0.0)
    delta = rewards + gamma * next_v - values[:n]
    out = np.zeros(n, dtype=np.float64)
    acc = 0.0
    for t in range(n - 1, -1, -1):
        acc = float(delta[t]) + gamma * lam * acc
        out[t] = acc
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, with ties averaged. Returns nan when either side is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        return float("nan")

    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(x.size, dtype=np.float64)
        r[order] = np.arange(x.size, dtype=np.float64)
        # average ranks within tie groups
        xs = x[order]
        i = 0
        while i < x.size:
            j = i
            while j + 1 < x.size and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i : j + 1]] = (i + j) / 2.0
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    sa, sb = ra.std(), rb.std()
    if sa == 0 or sb == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def decision_step_auc(assigned: np.ndarray, is_decision: np.ndarray) -> float:
    """Probability that a decision step is ranked above a routine step.

    A more forgiving reading than Spearman against exact credit: it asks only whether the method
    can tell the steps that could possibly matter from the ones that provably cannot. A method
    scoring 0.5 here is not assigning credit at all.
    """
    pos = assigned[is_decision.astype(bool)]
    neg = assigned[~is_decision.astype(bool)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    wins = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((wins + 0.5 * ties) / (pos.size * neg.size))
