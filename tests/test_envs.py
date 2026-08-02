"""Environment contract tests.

These pin the properties the corpus depends on. If any of these break, run labels become
untrustworthy and every downstream number is garbage.
"""

from __future__ import annotations

import numpy as np
import pytest

from testbed.envs import EVAL_SEED_BASE, ChainRho, assert_training_seed, env_names, eval_seeds, make


@pytest.mark.parametrize("name", env_names())
def test_determinism_given_seed_and_actions(name: str) -> None:
    """Same seed + same actions => byte-identical trajectory. Non-negotiable."""

    def rollout() -> list[tuple[np.ndarray, float, bool]]:
        env = make(name)
        obs = env.reset(seed=7)
        rng = np.random.default_rng(123)
        out = [(obs.copy(), 0.0, False)]
        for _ in range(env.max_steps):
            a = int(rng.integers(0, env.n_actions))
            r = env.step(a)
            out.append((r.obs.copy(), r.reward, r.terminated))
            if r.terminated or r.truncated:
                break
        return out

    a, b = rollout(), rollout()
    assert len(a) == len(b)
    for (oa, ra, ta), (ob, rb, tb) in zip(a, b, strict=True):
        np.testing.assert_array_equal(oa, ob)
        assert ra == rb
        assert ta == tb


@pytest.mark.parametrize("name", env_names())
def test_obs_shape_and_dtype(name: str) -> None:
    env = make(name)
    obs = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    assert obs.dtype == np.float32


@pytest.mark.parametrize("name", env_names())
def test_episode_terminates_within_max_steps(name: str) -> None:
    env = make(name)
    env.reset(seed=0)
    for i in range(env.max_steps):
        r = env.step(i % env.n_actions)
        if r.terminated or r.truncated:
            return
    raise AssertionError(f"{name} ran past max_steps={env.max_steps} without ending")


def test_eval_seed_block_is_quarantined() -> None:
    seeds = eval_seeds(4)
    assert all(s >= EVAL_SEED_BASE for s in seeds)
    assert_training_seed(EVAL_SEED_BASE - 1)  # fine
    with pytest.raises(ValueError, match="held-out"):
        assert_training_seed(seeds[0])


def test_chain_rho_credit_matches_counterfactual_exactly() -> None:
    """The whole Phase-3 claim rests on this: flipping one action moves the return by exactly
    the credit that env reported for that step."""
    env = ChainRho(length=16, rho=0.5, layout_seed=3)
    rng = np.random.default_rng(0)
    actions = [int(rng.integers(0, 2)) for _ in range(env.length)]

    def run(acts: list[int]) -> tuple[float, np.ndarray]:
        env.reset(seed=0)
        total = 0.0
        for a in acts:
            r = env.step(a)
            total += r.reward
        return total, env.true_credit()

    base_return, credit = run(actions)

    for t in range(env.length):
        flipped = list(actions)
        flipped[t] = 1 - flipped[t]
        flipped_return, _ = run(flipped)
        delta = flipped_return - base_return
        # Flipping away from the cued action loses exactly credit[t];
        # flipping toward it gains exactly 1/n_decision. Routine steps move nothing.
        if not env._is_decision[t]:
            assert delta == pytest.approx(0.0, abs=1e-12), f"routine step {t} moved the return"
        else:
            expected = -credit[t] if credit[t] > 0 else 1.0 / env.n_decision
            assert delta == pytest.approx(expected, abs=1e-12), f"decision step {t}"


def test_chain_rho_credit_sums_to_return() -> None:
    env = ChainRho(length=20, rho=0.4, layout_seed=1)
    env.reset(seed=0)
    total = 0.0
    for t in range(env.length):
        total += env.step(int(env._cue[t])).reward  # play perfectly
    assert total == pytest.approx(1.0)
    assert env.true_credit().sum() == pytest.approx(1.0)
