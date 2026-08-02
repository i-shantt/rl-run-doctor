"""DQN, instrumented.

PPO cannot express the most distinctive classic-RL pathologies -- there is no bootstrapped target
to diverge and no replay buffer to go stale. DQN carries those: target-network removal, replay
staleness, and the high-replay-ratio plasticity loss that motivated the resets literature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ..envs import Env
from ..telemetry import TraceWriter, UpdateRecord
from .common import RunningNorm, evaluate, rollout_seed_stream
from .nets import MLP, dormant_fraction, effective_rank, grad_norm, make_probe_batch, weight_norm


@dataclass
class DQNConfig:
    # Defaults tuned on CartPole for the healthiest control available: 6-seed mean of a trailing
    # eval window ~322 at 60k steps. The band is wide (118-500) and that is DQN, not a bug --
    # training longer makes the mean *worse*, so "healthy" is defined as a band in the corpus
    # rather than as a point.
    total_steps: int = 60_000
    buffer_size: int = 50_000
    batch_size: int = 128
    lr: float = 5e-4
    gamma: float = 0.99
    # Gradient steps per environment step. Raising this without resets is the canonical route to
    # plasticity loss.
    replay_ratio: float = 1.0
    learning_starts: int = 1_000
    train_frequency: int = 2
    use_target_network: bool = True
    target_update_interval: int = 200
    double_q: bool = True
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 10_000
    max_grad_norm: float | None = 10.0
    normalize_obs: bool = True
    reward_scale: float = 1.0
    hidden: tuple[int, ...] = (64, 64)
    # Periodic full re-initialisation of the network (Nikishin et al. resets). None = never.
    reset_interval: int | None = None
    eval_every: int = 4_000
    eval_episodes: int = 20
    probe_size: int = 256
    log_every: int = 1_000
    seed: int = 0
    reseed_envs: bool = True

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["hidden"] = list(self.hidden)
        return d


class ReplayBuffer:
    """Uniform replay that also records *when* each transition was written.

    The insertion step is what makes replay staleness measurable: the age distribution of a sampled
    batch is a signal, and it is one a practitioner never normally looks at.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.added_at = np.zeros(capacity, dtype=np.int64)
        self.size = 0
        self.ptr = 0

    def add(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        step: int,
    ) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = float(done)
        self.added_at[i] = step
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "act": self.act[idx],
            "rew": self.rew[idx],
            "next_obs": self.next_obs[idx],
            "done": self.done[idx],
            "added_at": self.added_at[idx],
        }


def train_dqn(
    env: Env,
    cfg: DQNConfig,
    writer: TraceWriter | None = None,
    eval_env: Env | None = None,
) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    def build() -> MLP:
        return MLP(env.obs_dim, env.n_actions, cfg.hidden)

    q = build()
    target = build()
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)

    buf = ReplayBuffer(cfg.buffer_size, env.obs_dim)
    obs_norm = RunningNorm(env.obs_dim, enabled=cfg.normalize_obs)
    probe = make_probe_batch(env.obs_dim, cfg.probe_size, seed=cfg.seed + 991)
    next_seed = rollout_seed_stream(cfg.seed, reseed=cfg.reseed_envs)
    eval_env = eval_env if eval_env is not None else env

    def act_greedy(o: np.ndarray) -> int:
        with torch.no_grad():
            return int(
                torch.argmax(q(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))).item()
            )

    raw_obs = env.reset(seed=next_seed())
    ep_return = 0.0
    recent_returns: list[float] = []
    update = 0

    # Accumulators reset at every log boundary.
    acc = {k: 0.0 for k in ("loss", "td_abs", "td_mean", "q_mean", "q_max", "tgt_gap", "gnorm", "age")}
    acc_n = 0
    grad_step_debt = 0.0

    for step in range(cfg.total_steps):
        eps = max(
            cfg.eps_end,
            cfg.eps_start + (cfg.eps_end - cfg.eps_start) * (step / max(cfg.eps_decay_steps, 1)),
        )
        obs_norm.update(raw_obs)
        o = obs_norm(raw_obs)
        if rng.random() < eps:
            action = int(rng.integers(0, env.n_actions))
        else:
            action = act_greedy(o)

        res = env.step(action)
        ep_return += res.reward
        done = res.terminated or res.truncated
        # Bootstrapping must not be cut by a time-limit truncation; only true termination ends
        # the value backup.
        buf.add(o, action, res.reward * cfg.reward_scale, obs_norm(res.obs), res.terminated, step)
        raw_obs = res.obs
        if done:
            recent_returns.append(ep_return)
            ep_return = 0.0
            raw_obs = env.reset(seed=next_seed())

        if step >= cfg.learning_starts and step % cfg.train_frequency == 0:
            grad_step_debt += cfg.replay_ratio * cfg.train_frequency
            while grad_step_debt >= 1.0:
                grad_step_debt -= 1.0
                batch = buf.sample(cfg.batch_size, rng)
                obs_b = torch.as_tensor(batch["obs"])
                act_b = torch.as_tensor(batch["act"])
                rew_b = torch.as_tensor(batch["rew"])
                nobs_b = torch.as_tensor(batch["next_obs"])
                done_b = torch.as_tensor(batch["done"])

                qsa = q(obs_b).gather(1, act_b.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    bootstrap_net = target if cfg.use_target_network else q
                    if cfg.double_q:
                        next_act = torch.argmax(q(nobs_b), dim=1, keepdim=True)
                        next_q = bootstrap_net(nobs_b).gather(1, next_act).squeeze(1)
                    else:
                        next_q = bootstrap_net(nobs_b).max(dim=1).values
                    tgt = rew_b + cfg.gamma * next_q * (1.0 - done_b)

                td = qsa - tgt
                loss = 0.5 * (td**2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = grad_norm(q)
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(q.parameters(), cfg.max_grad_norm)
                opt.step()
                update += 1

                with torch.no_grad():
                    all_q = q(obs_b)
                    acc["loss"] += float(loss.item())
                    acc["td_abs"] += float(td.abs().mean().item())
                    acc["td_mean"] += float(td.mean().item())
                    acc["q_mean"] += float(all_q.mean().item())
                    acc["q_max"] += float(all_q.max(dim=1).values.mean().item())
                    acc["tgt_gap"] += float((qsa - tgt).mean().item())
                    acc["gnorm"] += gn
                    acc["age"] += float(np.mean(step - batch["added_at"]))
                acc_n += 1

                if cfg.use_target_network and update % cfg.target_update_interval == 0:
                    target.load_state_dict(q.state_dict())

        if cfg.reset_interval is not None and step > 0 and step % cfg.reset_interval == 0:
            fresh = build()
            q.load_state_dict(fresh.state_dict())
            target.load_state_dict(fresh.state_dict())
            opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)

        if step % cfg.log_every == 0 and step >= cfg.learning_starts:
            n = max(acc_n, 1)
            signals = {
                "train_return": float(np.mean(recent_returns[-20:]))
                if recent_returns
                else float("nan"),
                "epsilon": eps,
                "loss": acc["loss"] / n,
                "td_abs_mean": acc["td_abs"] / n,
                "td_mean": acc["td_mean"] / n,
                "q_mean": acc["q_mean"] / n,
                "q_max_mean": acc["q_max"] / n,
                "target_gap": acc["tgt_gap"] / n,
                "grad_norm": acc["gnorm"] / n,
                "replay_age_mean": acc["age"] / n,
                "buffer_fill": buf.size / buf.capacity,
                "grad_steps": float(update),
                "weight_norm": weight_norm(q),
                "dormant_frac": dormant_fraction(q, probe),
                "effective_rank": effective_rank(q, probe),
                "lr": cfg.lr,
            }
            oracle: dict[str, float] = {}
            if step % cfg.eval_every == 0:
                oracle["eval_return"] = evaluate(
                    eval_env, act_greedy, cfg.eval_episodes, obs_norm if cfg.normalize_obs else None
                )
            if writer is not None:
                writer.write(
                    UpdateRecord(update=update, env_steps=step, signals=signals, oracle=oracle)
                )
            acc = {k: 0.0 for k in acc}
            acc_n = 0

    final_eval = evaluate(
        eval_env, act_greedy, cfg.eval_episodes, obs_norm if cfg.normalize_obs else None
    )
    return {"final_eval_return": final_eval, "grad_steps": update, "env_steps": cfg.total_steps}
