"""The failure taxonomy.

Each pathology is a *diff against a config*, never a patched code path. A corpus run is therefore
reproducible from its manifest without the code that generated it, and the "injection" is auditable
by reading a dict.

Naming: `P0` is the healthy control. Everything else is one named mechanism. Several are expected
*not* to produce a failure -- a sibling project measured that most induced pathologies leave final
performance intact -- and the smoke gate exists to find out which. Pathologies that do not
reliably degrade performance are dropped from the corpus and reported, not quietly retried.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Algo = Literal["ppo", "dqn"]


@dataclass(frozen=True)
class Pathology:
    name: str
    algos: tuple[Algo, ...]
    mechanism: str
    # Overrides applied to the algorithm config.
    cfg: dict[str, Any] = field(default_factory=dict)
    # Overrides applied to the environment constructor.
    env: dict[str, Any] = field(default_factory=dict)
    # Envs this only makes sense on. Empty = any.
    only_envs: tuple[str, ...] = ()

    @property
    def is_control(self) -> bool:
        return not self.cfg and not self.env


PATHOLOGIES: tuple[Pathology, ...] = (
    Pathology(
        name="P0_control",
        algos=("ppo", "dqn"),
        mechanism="Nothing injected. Defines the healthy band.",
    ),
    # ---- PPO ----------------------------------------------------------------
    Pathology(
        name="P1_entropy_collapse",
        algos=("ppo",),
        mechanism="Entropy bonus removed and learning rate raised; the policy commits early.",
        cfg={"ent_coef": 0.0, "lr": 1e-3},
    ),
    Pathology(
        name="P2_trust_region_blowup",
        algos=("ppo",),
        mechanism="PPO clipping and gradient clipping both removed, learning rate 10x.",
        cfg={"clip_coef": None, "max_grad_norm": None, "lr": 3e-3},
    ),
    Pathology(
        name="P3_obs_norm_freeze",
        algos=("ppo",),
        mechanism="Observation normaliser stops tracking early; inputs drift out of its range.",
        cfg={"freeze_obs_norm_after": 3},
    ),
    Pathology(
        name="P4_reward_scale_drift",
        algos=("ppo",),
        mechanism="Rewards scaled 1000x with advantage normalisation off.",
        cfg={"reward_scale": 1000.0, "normalize_advantage": False},
    ),
    # ---- DQN ----------------------------------------------------------------
    Pathology(
        name="P5_no_target_network",
        algos=("dqn",),
        mechanism="Bootstrap from the online network; the classic route to value divergence.",
        cfg={"use_target_network": False, "double_q": False},
    ),
    Pathology(
        name="P6_stale_target",
        algos=("dqn",),
        mechanism="Target network almost never refreshed, so targets lag the policy badly.",
        cfg={"target_update_interval": 20_000},
    ),
    Pathology(
        name="P7_replay_staleness",
        algos=("dqn",),
        mechanism="Buffer never evicts, so training is dominated by early bad-policy data.",
        cfg={"buffer_size": 1_000_000},
    ),
    Pathology(
        name="P8_plasticity_loss",
        algos=("dqn",),
        mechanism="Update-to-data ratio raised 8x with no resets; units go dormant.",
        cfg={"replay_ratio": 8.0, "reset_interval": None},
    ),
    Pathology(
        name="P9_overestimation",
        algos=("dqn",),
        mechanism="Double-Q disabled; the max operator's bias is left uncorrected.",
        cfg={"double_q": False},
    ),
    # ---- Both ---------------------------------------------------------------
    Pathology(
        name="P10_no_reseed",
        algos=("ppo", "dqn"),
        mechanism=(
            "Environment reseeded to the same value every episode. Training curves look normal "
            "while the agent overfits a single initial state."
        ),
        cfg={"reseed_envs": False},
        only_envs=("cartpole",),  # the only env here with seed-dependent episodes
    ),
    Pathology(
        name="P11_credit_dilution",
        algos=("ppo", "dqn"),
        mechanism="Decision density cut so the payout depends on very few of the actions taken.",
        env={"rho": 0.05},
        only_envs=("chain_rho",),
    ),
)

BY_NAME = {p.name: p for p in PATHOLOGIES}


def applicable(algo: Algo, env_name: str) -> list[Pathology]:
    out = []
    for p in PATHOLOGIES:
        if algo not in p.algos:
            continue
        if p.only_envs and env_name not in p.only_envs:
            continue
        out.append(p)
    return out
