"""Turn a (env, algo, pathology, seed) tuple into a trace on disk.

Everything needed to reproduce a run lives in the trace's meta header, so analysis never depends on
the script that produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..algos.dqn import DQNConfig, train_dqn
from ..algos.ppo import PPOConfig, train_ppo
from ..envs import make
from ..inject.pathologies import BY_NAME
from ..inject.ramp import Ramp
from ..telemetry import TraceWriter

# Per-environment defaults. PPO rollout lengths are whole multiples of the episode length where the
# environment has a fixed one, so a rollout never straddles a partial episode.
ENV_DEFAULTS: dict[str, dict[str, Any]] = {
    "cartpole": {},
    "deep_sea": {"size": 10},
    "chain_rho": {"length": 24, "rho": 0.5},
}

PPO_ENV_CFG: dict[str, dict[str, Any]] = {
    # lr tuned: the original 3e-4/0.01 control averaged 365 on CartPole, which was weak enough
    # that three injected "pathologies" scored *better* than it. A 3x3 sweep put 1e-3/0.01 at a
    # 464 mean with a 392 worst seed. (3e-3/0.0 is better still at 500/500/500, but a control
    # pinned to the ceiling leaves nothing to distinguish mild degradation from noise.)
    "cartpole": {
        "total_steps": 120_000,
        "n_steps": 512,
        "eval_every": 4,
        "eval_episodes": 20,
        "lr": 1e-3,
    },
    "deep_sea": {"total_steps": 40_000, "n_steps": 500, "eval_every": 4, "eval_episodes": 5,
                 "ent_coef": 0.02},
    "chain_rho": {"total_steps": 40_000, "n_steps": 480, "eval_every": 4, "eval_episodes": 5},
}

DQN_ENV_CFG: dict[str, dict[str, Any]] = {
    "cartpole": {"total_steps": 60_000, "eval_every": 4_000, "eval_episodes": 20},
    "deep_sea": {"total_steps": 40_000, "eval_every": 2_000, "eval_episodes": 5,
                 "eps_decay_steps": 20_000},
    "chain_rho": {"total_steps": 40_000, "eval_every": 2_000, "eval_episodes": 5},
}


@dataclass(frozen=True)
class RunSpec:
    env_name: str
    algo: str
    pathology: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.env_name}__{self.algo}__{self.pathology}__s{self.seed}"


def _jsonable(obj: Any) -> Any:
    """Ramps are dataclasses, not JSON scalars. The manifest has to record them, because a ramped
    run is not reproducible from a config field alone."""
    if isinstance(obj, Ramp):
        return obj.to_dict()
    if isinstance(obj, list | tuple):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def build(spec: RunSpec) -> tuple[Any, Any, Any]:
    """Return (train_env, eval_env, cfg) with the pathology applied."""
    path = BY_NAME[spec.pathology]

    env_kwargs = {**ENV_DEFAULTS[spec.env_name], **path.env}
    train_env = make(spec.env_name, **env_kwargs)
    eval_env = make(spec.env_name, **env_kwargs)

    if spec.algo == "ppo":
        cfg: Any = PPOConfig(seed=spec.seed, **PPO_ENV_CFG[spec.env_name])
    elif spec.algo == "dqn":
        cfg = DQNConfig(seed=spec.seed, **DQN_ENV_CFG[spec.env_name])
    else:
        raise ValueError(f"unknown algo {spec.algo!r}")

    unknown = set(path.cfg) - set(vars(cfg))
    if unknown:
        raise ValueError(f"{spec.pathology} sets unknown config fields for {spec.algo}: {unknown}")
    cfg = replace(cfg, **path.cfg)
    return train_env, eval_env, cfg


def run_one(spec: RunSpec, out_dir: str | Path) -> Path:
    train_env, eval_env, cfg = build(spec)
    path = BY_NAME[spec.pathology]
    out = Path(out_dir) / f"{spec.run_id}.jsonl.gz"

    meta = {
        "run_id": spec.run_id,
        "spec": asdict(spec),
        "pathology": {
            "name": path.name,
            "mechanism": path.mechanism,
            "cfg_overrides": _jsonable(path.cfg),
            "env_overrides": _jsonable(path.env),
            "cliff": path.cliff,
        },
        "cfg": cfg.to_dict(),
        "env_kwargs": {**ENV_DEFAULTS[spec.env_name], **path.env},
    }

    with TraceWriter(out, meta=meta) as w:
        if spec.algo == "ppo":
            train_ppo(train_env, cfg, writer=w, eval_env=eval_env)
        else:
            train_dqn(train_env, cfg, writer=w, eval_env=eval_env)
    return out
