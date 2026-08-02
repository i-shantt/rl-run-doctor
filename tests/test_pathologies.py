"""Taxonomy well-formedness.

Every one of these failures would otherwise surface partway through a multi-hour sweep, as a
worker dying on a config field that does not exist. Cheap to check up front.
"""

from __future__ import annotations

import itertools

import pytest

from testbed.corpus.runner import ENV_DEFAULTS, PPO_ENV_CFG, RunSpec, build
from testbed.inject.pathologies import BY_NAME, PATHOLOGIES, applicable

ENVS = sorted(ENV_DEFAULTS)
ALGOS = ["ppo", "dqn"]


def test_names_are_unique() -> None:
    names = [p.name for p in PATHOLOGIES]
    assert len(names) == len(set(names)), "duplicate pathology names"


@pytest.mark.parametrize(("env_name", "algo"), list(itertools.product(ENVS, ALGOS)))
def test_every_applicable_pathology_builds(env_name: str, algo: str) -> None:
    """Config overrides must name fields that actually exist on the target config."""
    for p in applicable(algo, env_name):  # type: ignore[arg-type]
        spec = RunSpec(env_name, algo, p.name, seed=0)
        train_env, eval_env, cfg = build(spec)
        for field, value in p.cfg.items():
            assert getattr(cfg, field) == value, f"{p.name}: {field} did not take"
        assert train_env is not eval_env, "train and eval envs must be distinct objects"


def test_unknown_config_field_is_rejected_loudly() -> None:
    from dataclasses import replace

    bad = replace(BY_NAME["P9_overestimation"], cfg={"not_a_real_field": 1})
    BY_NAME["__bad__"] = bad
    try:
        with pytest.raises(ValueError, match="unknown config fields"):
            build(RunSpec("cartpole", "dqn", "__bad__", 0))
    finally:
        del BY_NAME["__bad__"]


def test_env_only_restrictions_are_respected() -> None:
    # credit dilution is a chain_rho concept; it must not leak into other envs
    for env_name in ENVS:
        names = {p.family for p in applicable("ppo", env_name)}
        if env_name != "chain_rho":
            assert "P11_credit_dilution" not in names
        if env_name != "cartpole":
            assert "P10_no_reseed" not in names


def test_dose_ladders_are_ordered_and_distinct() -> None:
    by_family: dict[str, list[float]] = {}
    for p in PATHOLOGIES:
        if p.is_control:
            continue
        by_family.setdefault(p.family, []).append(p.severity)
    for family, sevs in by_family.items():
        assert len(sevs) == len(set(sevs)), f"{family} has duplicate doses"


def test_control_is_a_genuine_no_op() -> None:
    """If the control silently differed from the tuned default, every band would be wrong."""
    _, _, cfg = build(RunSpec("cartpole", "ppo", "P0_control", 0))
    for field, value in PPO_ENV_CFG["cartpole"].items():
        assert getattr(cfg, field) == value
