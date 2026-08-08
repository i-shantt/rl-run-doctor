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


def test_terminal_only_reward_makes_trajectory_credit_constant() -> None:
    """With one lump at the end, `uniform` and undiscounted return-to-go carry no per-step info.

    Not a quirk of the fit -- it is why both score exactly 0.50 decision AUC in every cell of the
    Phase 3 study, and why Spearman against them is undefined rather than low.
    """
    rewards = np.array([0.0] * 23 + [0.75])
    assert np.unique(uniform_credit(rewards)).size == 1
    assert np.unique(return_to_go(rewards, gamma=1.0)).size == 1


def test_discounted_return_to_go_is_exactly_the_step_index() -> None:
    """On a terminal-only reward, rtg(g<1) = R * g^(L-1-t): monotone in t, so rho vs index is +1.

    The Phase 3 study measures exactly +1.000 for this in all 15 cells. That is analytic, not
    empirical, and it is why the estimator's apparent correlation with exact credit is decided by
    where the decision steps happen to sit rather than by anything it computed.
    """
    rewards = np.array([0.0] * 23 + [1.0])
    rtg = return_to_go(rewards, gamma=0.99)
    assert spearman(rtg, np.arange(rtg.size, dtype=float)) == pytest.approx(1.0)


@pytest.mark.parametrize("v_end", [1.03, 0.97])
def test_gae_on_a_terminal_reward_is_the_negated_step_index(v_end: float) -> None:
    """GAE sums the *remaining* future TD errors, so its window shrinks monotonically with t.

    On a terminal-only reward that makes it a strictly decreasing function of the step index --
    it ranks by recency and never by causal relevance, measured at -1.00 in every density cell at
    40k steps. Both signs of terminal critic bias are checked because the direction of the bias is
    not what drives it; the shrinking window is.
    """
    rewards = np.array([0.0] * 23 + [1.0])
    values = np.linspace(0.80, v_end, 24)  # a critic that is a smooth function of t, as measured
    adv = gae_advantage(rewards, values, gamma=1.0, lam=0.95)
    assert spearman(adv, np.arange(adv.size, dtype=float)) == pytest.approx(-1.0)


def test_gae_is_exactly_zero_on_a_flat_critic() -> None:
    """The degenerate corner: V constant *and* equal to the return makes every TD error zero.

    Worth pinning because it is the limit the trained runs approach -- as the critic converges the
    advantages collapse toward zero and what survives is pure discount profile, which is why the
    step-index correlation sharpens from -0.47 at 4k steps to -1.00 at 40k.
    """
    rewards = np.array([0.0] * 23 + [1.0])
    adv = gae_advantage(rewards, np.full(24, 1.0), gamma=1.0, lam=0.95)
    np.testing.assert_allclose(adv, 0.0, atol=1e-12)
