"""Chain-rho: delayed reward with an analytically exact per-step credit.

A length-L episode. A fixed fraction `rho` of the steps are *decision* steps, where taking the
cued action contributes to the final payout; the rest are *routine* steps, where the action is
irrelevant. Nothing is paid out until the terminal step.

Two jobs:

1. A credit-assignment pathology with a tunable severity. Lowering `rho` dilutes the learning
   signal without changing the horizon, which is the cleaner knob -- the turn-level to
   trajectory-level SNR falls as rho^(-1/2) (arXiv:2606.22164).

2. **Exact ground truth for Phase 3.** The counterfactual contribution of the action at step t is
   known in closed form:

       credit(t) = 1 / n_decision   if t is a decision step and the cued action was taken
       credit(t) = 0                otherwise

   Flipping the action at a single decision step changes the return by exactly 1/n_decision, and
   flipping it at a routine step changes it by exactly 0. So any credit-assignment algorithm's
   output can be scored against truth rather than against a proxy.
"""

from __future__ import annotations

import numpy as np

from .base import StepResult


class ChainRho:
    # [t/L, is_decision, cue]
    obs_dim = 3
    n_actions = 2

    def __init__(self, length: int = 24, rho: float = 0.5, layout_seed: int = 0) -> None:
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"rho must be in (0, 1], got {rho}")
        self.length = length
        self.max_steps = length
        self.rho = rho

        rng = np.random.default_rng(layout_seed)
        n_decision = max(1, int(round(rho * length)))
        idx = rng.choice(length, size=n_decision, replace=False)
        self._is_decision = np.zeros(length, dtype=bool)
        self._is_decision[idx] = True
        self.n_decision = int(n_decision)

        # Which action is correct at each step. Only meaningful on decision steps, but it is
        # surfaced in the observation everywhere so the cue channel has no distributional tell
        # that leaks which steps matter.
        self._cue = rng.integers(0, 2, size=length).astype(np.int64)

        self._t = 0
        self._correct = 0
        self._credit_trace: list[float] = []

    def _obs(self) -> np.ndarray:
        t = min(self._t, self.length - 1)
        return np.array(
            [self._t / self.length, float(self._is_decision[t]), float(self._cue[t])],
            dtype=np.float32,
        )

    def reset(self, seed: int) -> np.ndarray:
        del seed  # layout is fixed; the episode is deterministic given the action sequence
        self._t = 0
        self._correct = 0
        self._credit_trace = []
        return self._obs()

    def step(self, action: int) -> StepResult:
        t = self._t
        is_dec = bool(self._is_decision[t])
        took_cued = int(action) == int(self._cue[t])

        # Exact counterfactual contribution of this action to the final return.
        credit = (1.0 / self.n_decision) if (is_dec and took_cued) else 0.0
        self._credit_trace.append(credit)

        if is_dec and took_cued:
            self._correct += 1

        self._t += 1
        terminated = self._t >= self.length
        reward = (self._correct / self.n_decision) if terminated else 0.0

        obs = self._obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        return StepResult(
            obs=obs, reward=reward, terminated=terminated, truncated=False, credit=credit
        )

    def true_credit(self) -> np.ndarray:
        """Per-step exact credit for the episode just completed."""
        return np.asarray(self._credit_trace, dtype=np.float64)
