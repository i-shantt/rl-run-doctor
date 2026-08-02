"""Deep Sea: sparse-reward hard exploration, in the style of bsuite.

An N-step episode descending an N x N grid. At each cell one of the two actions means "right" and
the other "left", under a fixed but scrambled per-cell mapping, so the agent cannot learn a
constant action. Going right costs a little; only the bottom-right corner pays.

Chosen because its failure is *unambiguous*: an agent that never reaches the corner has a return of
at most 0, and one that solves it has ~1. There is no partial credit to hide in.
"""

from __future__ import annotations

import numpy as np

from .base import StepResult


class DeepSea:
    obs_dim: int
    n_actions = 2

    def __init__(self, size: int = 12, layout_seed: int = 0) -> None:
        self.size = size
        self.obs_dim = size * 2  # one-hot row ++ one-hot column
        self.max_steps = size
        self.move_cost = 0.01 / size

        # The action->direction scramble is a property of the environment, fixed for the whole
        # experiment, not resampled per episode. This is what makes the task hard.
        rng = np.random.default_rng(layout_seed)
        self._action_right = rng.integers(0, 2, size=(size, size)).astype(np.int64)

        self._row = 0
        self._col = 0

    def _obs(self) -> np.ndarray:
        o = np.zeros(self.obs_dim, dtype=np.float32)
        o[self._row] = 1.0
        o[self.size + self._col] = 1.0
        return o

    def reset(self, seed: int) -> np.ndarray:
        # Deep Sea is fully deterministic; the seed is accepted for interface uniformity.
        # Held-out evaluation here means "a separate greedy rollout", not "a different layout".
        del seed
        self._row = 0
        self._col = 0
        return self._obs()

    def step(self, action: int) -> StepResult:
        go_right = int(action) == int(self._action_right[self._row, self._col])

        reward = 0.0
        if go_right:
            reward -= self.move_cost
            new_col = min(self._col + 1, self.size - 1)
        else:
            new_col = max(self._col - 1, 0)

        self._row += 1
        self._col = new_col

        terminated = self._row >= self.size
        if terminated and self._col == self.size - 1:
            reward += 1.0

        obs = self._obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        return StepResult(obs=obs, reward=reward, terminated=terminated, truncated=False)
