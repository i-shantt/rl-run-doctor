"""PPO, instrumented.

Every knob a pathology needs to perturb is a config field, so an injection is a diff in a dataclass
rather than a patched code path. That keeps the corpus reproducible from its manifest alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from ..envs import Env
from ..inject.ramp import Ramp
from ..telemetry import TraceWriter, UpdateRecord
from .common import RunningNorm, evaluate, rollout_seed_stream
from .nets import MLP, dormant_fraction, effective_rank, grad_norm, make_probe_batch, weight_norm


@dataclass
class PPOConfig:
    total_steps: int = 60_000
    n_steps: int = 512
    n_epochs: int = 4
    minibatch_size: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    # None removes the trust region entirely -- the "unclipped, hot" pathology.
    clip_coef: float | None = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float | None = 0.5
    normalize_advantage: bool = True
    normalize_obs: bool = True
    reward_scale: float = 1.0
    hidden: tuple[int, ...] = (64, 64)
    eval_every: int = 5
    eval_episodes: int = 20
    probe_size: int = 256
    seed: int = 0
    reseed_envs: bool = True
    # After this many updates the observation normaliser stops tracking. None = never.
    freeze_obs_norm_after: int | None = None
    # Fields that move during training. See testbed/inject/ramp.py.
    ramps: tuple[Ramp, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["hidden"] = list(self.hidden)
        d["ramps"] = [r.to_dict() for r in self.ramps]
        return d


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: tuple[int, ...]) -> None:
        super().__init__()
        self.actor = MLP(obs_dim, n_actions, hidden)
        self.critic = MLP(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(obs), self.critic(obs).squeeze(-1)


def train_ppo(
    env: Env,
    cfg: PPOConfig,
    writer: TraceWriter | None = None,
    eval_env: Env | None = None,
) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(False)
    rng = np.random.default_rng(cfg.seed)

    net = ActorCritic(env.obs_dim, env.n_actions, cfg.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    obs_norm = RunningNorm(env.obs_dim, enabled=cfg.normalize_obs)
    probe = make_probe_batch(env.obs_dim, cfg.probe_size, seed=cfg.seed + 991)
    next_seed = rollout_seed_stream(cfg.seed, reseed=cfg.reseed_envs)
    eval_env = eval_env if eval_env is not None else env

    def act_greedy(o: np.ndarray) -> int:
        with torch.no_grad():
            logits, _ = net(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(logits, dim=-1).item())

    raw_obs = env.reset(seed=next_seed())
    ep_return = 0.0
    recent_returns: list[float] = []
    env_steps = 0
    update = 0
    n_updates = max(1, cfg.total_steps // cfg.n_steps)

    while update < n_updates:
        progress = update / max(n_updates - 1, 1)
        ramped = {r.field: r.value(progress) for r in cfg.ramps}
        eff_lr = ramped.get("lr", cfg.lr)
        eff_ent = ramped.get("ent_coef", cfg.ent_coef)
        eff_rscale = ramped.get("reward_scale", cfg.reward_scale)
        if cfg.ramps:
            for g in opt.param_groups:
                g["lr"] = eff_lr

        if cfg.freeze_obs_norm_after is not None and update >= cfg.freeze_obs_norm_after:
            obs_norm.frozen = True

        obs_buf = np.zeros((cfg.n_steps, env.obs_dim), dtype=np.float32)
        act_buf = np.zeros(cfg.n_steps, dtype=np.int64)
        logp_buf = np.zeros(cfg.n_steps, dtype=np.float32)
        rew_buf = np.zeros(cfg.n_steps, dtype=np.float32)
        done_buf = np.zeros(cfg.n_steps, dtype=np.float32)
        val_buf = np.zeros(cfg.n_steps, dtype=np.float32)

        for t in range(cfg.n_steps):
            obs_norm.update(raw_obs)
            o = obs_norm(raw_obs)
            obs_buf[t] = o
            with torch.no_grad():
                logits, value = net(torch.as_tensor(o).unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                logp_buf[t] = float(dist.log_prob(a).item())
                val_buf[t] = float(value.item())
            action = int(a.item())
            act_buf[t] = action

            res = env.step(action)
            env_steps += 1
            ep_return += res.reward
            rew_buf[t] = res.reward * eff_rscale
            done = res.terminated or res.truncated
            done_buf[t] = float(done)
            raw_obs = res.obs
            if done:
                recent_returns.append(ep_return)
                ep_return = 0.0
                raw_obs = env.reset(seed=next_seed())

        # GAE. Bootstrap from the value of the state we stopped on.
        with torch.no_grad():
            _, last_val = net(torch.as_tensor(obs_norm(raw_obs)).unsqueeze(0))
            last_val_f = float(last_val.item())
        adv = np.zeros(cfg.n_steps, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(cfg.n_steps)):
            next_val = last_val_f if t == cfg.n_steps - 1 else val_buf[t + 1]
            next_nonterm = 1.0 - done_buf[t]
            delta = rew_buf[t] + cfg.gamma * next_val * next_nonterm - val_buf[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterm * gae
            adv[t] = gae
        ret = adv + val_buf

        obs_t = torch.as_tensor(obs_buf)
        act_t = torch.as_tensor(act_buf)
        # .detach() is load-bearing: the old log-probs are a constant of the update. Building the
        # ratio from a tensor that still carries grad_fn silently changes the objective.
        old_logp_t = torch.as_tensor(logp_buf).detach()
        adv_t = torch.as_tensor(adv)
        ret_t = torch.as_tensor(ret)

        pl = vl = ent = kl = clipfrac = 0.0
        gnorm = 0.0
        n_batches = 0
        idx = np.arange(cfg.n_steps)
        for _ in range(cfg.n_epochs):
            rng.shuffle(idx)
            for start in range(0, cfg.n_steps, cfg.minibatch_size):
                mb = idx[start : start + cfg.minibatch_size]
                if len(mb) < 2:
                    continue
                mb_t = torch.as_tensor(mb)
                logits, value = net(obs_t[mb_t])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb_t])
                entropy = dist.entropy().mean()

                logratio = new_logp - old_logp_t[mb_t]
                ratio = logratio.exp()

                mb_adv = adv_t[mb_t]
                if cfg.normalize_advantage and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std(unbiased=True) + 1e-8)

                if cfg.clip_coef is None:
                    policy_loss = -(ratio * mb_adv).mean()
                    mb_clipfrac = 0.0
                else:
                    unclipped = ratio * mb_adv
                    clipped = torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * mb_adv
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    with torch.no_grad():
                        mb_clipfrac = float(
                            ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                        )

                value_loss = 0.5 * ((value - ret_t[mb_t]) ** 2).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - eff_ent * entropy

                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm += grad_norm(net)
                if cfg.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()

                with torch.no_grad():
                    # Schulman's low-variance KL estimator.
                    approx_kl = float(((ratio - 1) - logratio).mean().item())
                pl += float(policy_loss.item())
                vl += float(value_loss.item())
                ent += float(entropy.item())
                kl += approx_kl
                clipfrac += mb_clipfrac
                n_batches += 1

        n_batches = max(n_batches, 1)
        y, yhat = ret, val_buf
        var_y = float(np.var(y))
        expl_var = float("nan") if var_y == 0 else 1.0 - float(np.var(y - yhat)) / var_y

        signals = {
            "train_return": (
                float(np.mean(recent_returns[-20:])) if recent_returns else float("nan")
            ),
            "policy_loss": pl / n_batches,
            "value_loss": vl / n_batches,
            "entropy": ent / n_batches,
            "approx_kl": kl / n_batches,
            "clip_frac": clipfrac / n_batches,
            "grad_norm": gnorm / n_batches,
            "explained_variance": expl_var,
            "adv_mean": float(adv.mean()),
            "adv_std": float(adv.std()),
            "value_mean": float(val_buf.mean()),
            "value_std": float(val_buf.std()),
            "return_mean": float(ret.mean()),
            "weight_norm": weight_norm(net),
            "dormant_frac_actor": dormant_fraction(net.actor, probe),
            "dormant_frac_critic": dormant_fraction(net.critic, probe),
            "effective_rank_actor": effective_rank(net.actor, probe),
            "effective_rank_critic": effective_rank(net.critic, probe),
        }

        oracle: dict[str, float] = {}
        if update % cfg.eval_every == 0 or update == n_updates - 1:
            oracle["eval_return"] = evaluate(
                eval_env, act_greedy, cfg.eval_episodes, obs_norm if cfg.normalize_obs else None
            )

        if writer is not None:
            writer.write(
                UpdateRecord(update=update, env_steps=env_steps, signals=signals, oracle=oracle)
            )
        update += 1

    final_eval = evaluate(
        eval_env, act_greedy, cfg.eval_episodes, obs_norm if cfg.normalize_obs else None
    )
    return {
        "final_eval_return": final_eval,
        "updates": update,
        "env_steps": env_steps,
        # In-process handle for analyses that need the trained critic (the credit study). Not
        # serialised; corpus runs ignore it.
        "net": net,
        "obs_norm": obs_norm,
    }
