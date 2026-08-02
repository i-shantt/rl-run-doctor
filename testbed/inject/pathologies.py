"""The failure taxonomy, as dose ladders rather than single guesses.

The first version of this file picked one severity per mechanism and asked "does it break?". That
question has a bad property: a `no` conflates "this mechanism is harmless" with "I guessed too
gentle a dose", and a `yes` tells you nothing about how much of the fault was needed. Three of
twenty-five cells answered yes, which is uninformative either way.

Each family now spans several severities, so what comes out is a dose-response curve: the point at
which a mechanism starts to bite, or the demonstration that it does not bite anywhere in a range
that reaches well past anything a practitioner would plausibly configure.

Each level is still a *diff against a config*, never a patched code path, so a run stays
reproducible from its manifest alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Algo = Literal["ppo", "dqn"]


@dataclass(frozen=True)
class Pathology:
    name: str  # e.g. "P6_stale_target@5000"
    family: str  # e.g. "P6_stale_target"
    severity: float  # ordinal dose, comparable within a family only
    algos: tuple[Algo, ...]
    mechanism: str
    cfg: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    only_envs: tuple[str, ...] = ()

    @property
    def is_control(self) -> bool:
        return self.family == "P0_control"


def _ladder(
    family: str,
    algos: tuple[Algo, ...],
    mechanism: str,
    field_name: str,
    doses: list[Any],
    *,
    base_cfg: dict[str, Any] | None = None,
    env_field: bool = False,
    only_envs: tuple[str, ...] = (),
) -> list[Pathology]:
    out = []
    for d in doses:
        label = "off" if d is None else str(d)
        cfg = dict(base_cfg or {})
        env: dict[str, Any] = {}
        if env_field:
            env[field_name] = d
        else:
            cfg[field_name] = d
        out.append(
            Pathology(
                name=f"{family}@{label}",
                family=family,
                severity=float("inf") if d is None else float(d),
                algos=algos,
                mechanism=f"{mechanism} ({field_name}={label})",
                cfg=cfg,
                env=env,
                only_envs=only_envs,
            )
        )
    return out


CONTROL = Pathology(
    name="P0_control",
    family="P0_control",
    severity=0.0,
    algos=("ppo", "dqn"),
    mechanism="Nothing injected. Defines the healthy band.",
)

PATHOLOGIES: tuple[Pathology, ...] = (
    CONTROL,
    # ---- DQN: target lag. Control is 200; the ladder runs out to never-refreshed. ----------
    *_ladder(
        "P6_stale_target",
        ("dqn",),
        "Target network refreshed ever more rarely, so bootstrap targets lag the policy",
        "target_update_interval",
        [1_000, 5_000, 20_000, 100_000],
    ),
    Pathology(
        name="P5_no_target_network",
        family="P5_no_target_network",
        severity=1.0,
        algos=("dqn",),
        mechanism="Bootstrap directly from the online network: the target lag ladder's limit case",
        cfg={"use_target_network": False, "double_q": False},
    ),
    # ---- DQN: update-to-data ratio. Control is 1.0. -----------------------------------------
    *_ladder(
        "P8_plasticity_loss",
        ("dqn",),
        "Update-to-data ratio raised with no resets; units go dormant",
        "replay_ratio",
        [2.0, 4.0, 8.0],
    ),
    # ---- DQN: replay staleness. Control buffer is 50k against 60k steps. ---------------------
    *_ladder(
        "P7_replay_staleness",
        ("dqn",),
        "Buffer large enough that early bad-policy data is never evicted",
        "buffer_size",
        [200_000, 1_000_000],
    ),
    Pathology(
        name="P9_overestimation",
        family="P9_overestimation",
        severity=1.0,
        algos=("dqn",),
        mechanism="Double-Q disabled; the max operator's bias is left uncorrected",
        cfg={"double_q": False},
    ),
    # ---- PPO: trust region removed, then heated. Control lr is env-specific. ------------------
    *_ladder(
        "P2_trust_region_blowup",
        ("ppo",),
        "PPO clipping and gradient clipping removed, then the learning rate raised",
        "lr",
        [3e-3, 1e-2, 3e-2, 1e-1],
        base_cfg={"clip_coef": None, "max_grad_norm": None},
    ),
    # ---- PPO: reward scale, with advantage normalisation off so the scale survives. -----------
    *_ladder(
        "P4_reward_scale_drift",
        ("ppo",),
        "Rewards rescaled with advantage normalisation off, so the scale reaches the gradient",
        "reward_scale",
        [100.0, 1_000.0, 10_000.0, 100_000.0],
        base_cfg={"normalize_advantage": False},
    ),
    # ---- PPO: observation normaliser frozen, earlier is worse. --------------------------------
    *_ladder(
        "P3_obs_norm_freeze",
        ("ppo",),
        "Observation normaliser stops tracking after N updates; inputs drift out of its range",
        "freeze_obs_norm_after",
        [1, 3, 10],
    ),
    # ---- PPO: entropy bonus removed. Only meaningful where exploration matters. ---------------
    *_ladder(
        "P1_entropy_collapse",
        ("ppo",),
        "Entropy bonus removed, so the policy commits before it has explored",
        "ent_coef",
        [0.005, 0.0],
        only_envs=("deep_sea",),
    ),
    # ---- Both: the seeding bug. Binary by nature. ---------------------------------------------
    Pathology(
        name="P10_no_reseed",
        family="P10_no_reseed",
        severity=1.0,
        algos=("ppo", "dqn"),
        mechanism=(
            "Environment reseeded to the same value every episode, so the agent overfits a single "
            "initial state while every training curve looks normal"
        ),
        cfg={"reseed_envs": False},
        only_envs=("cartpole",),
    ),
    # ---- Both: credit dilution. Control density is 0.5. ---------------------------------------
    *_ladder(
        "P11_credit_dilution",
        ("ppo", "dqn"),
        "Decision density cut so the payout depends on very few of the actions taken",
        "rho",
        [0.25, 0.1, 0.05],
        env_field=True,
        only_envs=("chain_rho",),
    ),
)

BY_NAME = {p.name: p for p in PATHOLOGIES}
FAMILIES = sorted({p.family for p in PATHOLOGIES if not p.is_control})


def applicable(algo: Algo, env_name: str) -> list[Pathology]:
    out = []
    for p in PATHOLOGIES:
        if algo not in p.algos:
            continue
        if p.only_envs and env_name not in p.only_envs:
            continue
        out.append(p)
    return out
