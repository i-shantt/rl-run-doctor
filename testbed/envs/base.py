"""Minimal environment interface.

Deliberately not `gymnasium`. The testbed needs exact control over seeding so that the
"environment was never reseeded" pathology is something we *inject*, not something we inherit
from a framework's RNG conventions.

Every env is:
  - deterministic given (seed, action sequence)
  - cheap enough to run thousands of episodes on CPU
  - able to expose a held-out evaluation set of seeds that training never touches
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# Seeds >= this are reserved for held-out evaluation and must never be used for training.
# Ground truth for "did this run fail" is measured only on these.
EVAL_SEED_BASE = 1_000_000


@dataclass(frozen=True)
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    # Per-step credit ground truth, where the env can supply it analytically. None otherwise.
    credit: float | None = None


class Env(Protocol):
    obs_dim: int
    n_actions: int
    max_steps: int

    def reset(self, seed: int) -> np.ndarray: ...

    def step(self, action: int) -> StepResult: ...


def eval_seeds(n: int) -> list[int]:
    """The held-out seed block. Training must never see these."""
    return [EVAL_SEED_BASE + i for i in range(n)]


def assert_training_seed(seed: int) -> None:
    if seed >= EVAL_SEED_BASE:
        raise ValueError(
            f"seed {seed} is in the held-out evaluation block (>= {EVAL_SEED_BASE}). "
            "Training must never touch these, or the ground-truth label leaks."
        )
