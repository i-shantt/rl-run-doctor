"""Credit-estimator tests on inputs whose correct answer is known analytically."""

from __future__ import annotations

import numpy as np
import pytest

from rl_run_doctor.credit import (
    decision_step_auc,
    gae_advantage,
    return_to_go,
    spearman,
    td0_advantage,
    uniform_credit,
)


def test_spearman_matches_known_cases() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(spearman(x, np.ones(4)))


def test_spearman_handles_ties() -> None:
    a = np.array([1.0, 1.0, 2.0, 2.0])
    b = np.array([5.0, 5.0, 9.0, 9.0])
    assert spearman(a, b) == pytest.approx(1.0)


def test_uniform_credit_cannot_rank_anything() -> None:
    """The failure mode of trajectory-level credit, stated as a test.

    Every step gets the same number, so no ordering information survives at all -- rank
    correlation with truth is undefined and decision/routine separation is exactly chance.
    """
    rewards = np.zeros(10)
    rewards[-1] = 1.0
    assigned = uniform_credit(rewards)
    assert np.allclose(assigned, 0.1)
    is_decision = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    assert decision_step_auc(assigned, is_decision) == pytest.approx(0.5)
    assert np.isnan(spearman(assigned, is_decision.astype(float)))


def test_return_to_go_is_flat_under_terminal_only_reward() -> None:
    """Undiscounted return-to-go gives every step the same value when the reward is terminal,
    so it is no better than uniform on exactly the tasks credit assignment is hard for."""
    rewards = np.zeros(8)
    rewards[-1] = 1.0
    rtg = return_to_go(rewards, gamma=1.0)
    assert np.allclose(rtg, 1.0)

    discounted = return_to_go(rewards, gamma=0.9)
    assert discounted[0] < discounted[-1]  # discounting does break the tie


def test_gae_with_an_informative_critic_ranks_decision_steps_first() -> None:
    """If the critic knows which steps matter, GAE inherits that and separates them."""
    n = 12
    rewards = np.zeros(n)
    rewards[-1] = 1.0
    is_decision = np.zeros(n)
    is_decision[[2, 5, 9]] = 1.0
    # V(s_t) is the value *before* the action at t, so a decision taken at t shows up as a jump
    # between V(s_t) and V(s_{t+1}) -- the cumulative sum has to be shifted by one. Getting this
    # backwards makes TD credit land on the step before each decision, which looks almost right
    # and is completely wrong.
    values = np.concatenate([[0.0], np.cumsum(is_decision)[:-1]]) / 3.0
    adv = gae_advantage(rewards, values, gamma=1.0, lam=0.0)  # lam=0 => pure TD(0)
    assert decision_step_auc(adv, is_decision) == pytest.approx(1.0)


def test_td0_equals_gae_at_lambda_zero() -> None:
    rng = np.random.default_rng(0)
    rewards = rng.normal(size=10)
    values = rng.normal(size=10)
    np.testing.assert_allclose(
        td0_advantage(rewards, values, gamma=0.99),
        gae_advantage(rewards, values, gamma=0.99, lam=0.0),
        atol=1e-12,
    )


def test_decision_step_auc_is_symmetric_about_chance() -> None:
    assigned = np.array([3.0, 1.0, 2.0, 0.0])
    is_decision = np.array([1, 0, 1, 0])
    good = decision_step_auc(assigned, is_decision)
    bad = decision_step_auc(-assigned, is_decision)
    assert good + bad == pytest.approx(1.0)
