"""Environments for the testbed."""

from __future__ import annotations

from typing import Any

from .base import EVAL_SEED_BASE, Env, StepResult, assert_training_seed, eval_seeds
from .cartpole import CartPole
from .chain_rho import ChainRho
from .deep_sea import DeepSea

_REGISTRY = {
    "cartpole": CartPole,
    "deep_sea": DeepSea,
    "chain_rho": ChainRho,
}


def make(name: str, **kwargs: Any) -> Env:
    if name not in _REGISTRY:
        raise KeyError(f"unknown env {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)  # type: ignore[abstract]


def env_names() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "EVAL_SEED_BASE",
    "CartPole",
    "ChainRho",
    "DeepSea",
    "Env",
    "StepResult",
    "assert_training_seed",
    "env_names",
    "eval_seeds",
    "make",
]
