"""CartPole, from scratch.

Standard physics and thresholds (Barto/Sutton/Anderson via the classic gym implementation) so that
numbers are comparable to everything else in the literature. Written out rather than imported so
that reset/seed behaviour is fully under our control.
"""

from __future__ import annotations

import numpy as np

from .base import StepResult

GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
LENGTH = 0.5  # actually half the pole's length
POLEMASS_LENGTH = MASS_POLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02  # seconds between state updates

THETA_THRESHOLD = 12 * 2 * np.pi / 360  # 12 degrees, in radians
X_THRESHOLD = 2.4


class CartPole:
    """Dense-reward, easy-credit control task. The 'does anything work at all' env."""

    obs_dim = 4
    n_actions = 2

    def __init__(self, max_steps: int = 500) -> None:
        self.max_steps = max_steps
        self._state = np.zeros(4, dtype=np.float64)
        self._t = 0

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        self._state = rng.uniform(-0.05, 0.05, size=4)
        self._t = 0
        return self._state.astype(np.float32)

    def step(self, action: int) -> StepResult:
        x, x_dot, theta, theta_dot = self._state
        force = FORCE_MAG if action == 1 else -FORCE_MAG
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        temp = (force + POLEMASS_LENGTH * theta_dot**2 * sin_t) / TOTAL_MASS
        theta_acc = (GRAVITY * sin_t - cos_t * temp) / (
            LENGTH * (4.0 / 3.0 - MASS_POLE * cos_t**2 / TOTAL_MASS)
        )
        x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_t / TOTAL_MASS

        # Euler, matching the reference implementation.
        x += TAU * x_dot
        x_dot += TAU * x_acc
        theta += TAU * theta_dot
        theta_dot += TAU * theta_acc
        self._state = np.array([x, x_dot, theta, theta_dot])
        self._t += 1

        terminated = bool(
            x < -X_THRESHOLD
            or x > X_THRESHOLD
            or theta < -THETA_THRESHOLD
            or theta > THETA_THRESHOLD
        )
        truncated = self._t >= self.max_steps
        return StepResult(
            obs=self._state.astype(np.float32),
            reward=1.0,
            terminated=terminated,
            truncated=truncated,
        )
