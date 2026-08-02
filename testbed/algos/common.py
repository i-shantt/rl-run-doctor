"""Pieces shared by the algorithms: observation normalisation and held-out evaluation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..envs import Env, assert_training_seed, eval_seeds


class RunningNorm:
    """Welford running mean/var over observations.

    `frozen` exists so that "the observation normaliser stopped tracking" is an injectable
    pathology rather than an accident.
    """

    def __init__(self, dim: int, enabled: bool = True) -> None:
        self.enabled = enabled
        self.frozen = False
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        if not self.enabled or self.frozen:
            return
        x = np.atleast_2d(x)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return x.astype(np.float32)
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


def evaluate(
    env: Env,
    act_greedy: Callable[[np.ndarray], int],
    n_episodes: int,
    obs_norm: RunningNorm | None = None,
) -> float:
    """Mean return over the held-out seed block, acting greedily.

    This is the label. It is deliberately not the training return, which rises during several of
    the pathologies we care about.
    """
    returns: list[float] = []
    for seed in eval_seeds(n_episodes):
        obs = env.reset(seed=seed)
        total = 0.0
        for _ in range(env.max_steps):
            o = obs_norm(obs) if obs_norm is not None else obs
            res = env.step(act_greedy(o))
            total += res.reward
            obs = res.obs
            if res.terminated or res.truncated:
                break
        returns.append(total)
    return float(np.mean(returns))


def rollout_seed_stream(base_seed: int, reseed: bool = True) -> Callable[[], int]:
    """Episode seed generator for training.

    With `reseed=False` every episode gets the same seed -- the "environment was never reseeded"
    bug, which silently turns a run into single-episode overfitting while every training curve
    still looks plausible.
    """
    assert_training_seed(base_seed)
    state = {"i": 0}

    def next_seed() -> int:
        if not reseed:
            return base_seed
        state["i"] += 1
        s = base_seed * 100_003 + state["i"]
        return int(s % 999_983)  # stays well below the held-out block

    return next_seed
