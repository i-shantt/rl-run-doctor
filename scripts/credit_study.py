"""Phase 3: score credit-assignment estimators against exact per-step truth.

`chain_rho` knows the counterfactual contribution of every action in closed form, so the estimators
that real algorithms actually learn from can be scored directly, instead of being judged by the
final performance of a training run that bundles them with a dozen other confounders.

Sweeps decision density, because the interesting question is not "which method is best" but "where
does each one stop carrying information".
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rl_run_doctor.credit import (
    decision_step_auc,
    gae_advantage,
    return_to_go,
    spearman,
    td0_advantage,
    uniform_credit,
)
from testbed.algos import PPOConfig, train_ppo
from testbed.envs import ChainRho, eval_seeds

LENGTH = 24
RHOS = [1.0, 0.5, 0.25, 0.1]


def collect(net, obs_norm, env: ChainRho, n_episodes: int) -> list[dict[str, np.ndarray]]:
    """Roll the trained policy and record what each estimator would see."""
    out = []
    for seed in eval_seeds(n_episodes):
        obs = env.reset(seed=seed)
        rewards, values, is_dec = [], [], []
        for t in range(env.length):
            o = obs_norm(obs)
            with torch.no_grad():
                logits, v = net(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
                a = int(torch.argmax(logits, dim=-1).item())
            values.append(float(v.item()))
            is_dec.append(float(env._is_decision[t]))
            res = env.step(a)
            rewards.append(res.reward)
            obs = res.obs
            if res.terminated:
                break
        out.append(
            {
                "rewards": np.asarray(rewards, dtype=np.float64),
                "values": np.asarray(values, dtype=np.float64),
                "is_decision": np.asarray(is_dec, dtype=np.float64),
                "exact": env.true_credit(),
            }
        )
    return out


def estimators(ep: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    r, v = ep["rewards"], ep["values"]
    return {
        "uniform": uniform_credit(r),
        "return_to_go(g=1.0)": return_to_go(r, gamma=1.0),
        "return_to_go(g=0.99)": return_to_go(r, gamma=0.99),
        "td0": td0_advantage(r, v, gamma=1.0),
        "gae(l=0.95)": gae_advantage(r, v, gamma=1.0, lam=0.95),
        "gae(l=0.0)": gae_advantage(r, v, gamma=1.0, lam=0.0),
    }


def one_rho(args: tuple[float, int, int]) -> tuple[float, int, dict[str, dict[str, float]]]:
    rho, seed, episodes = args
    env = ChainRho(length=LENGTH, rho=rho, layout_seed=0)
    cfg = replace(
        PPOConfig(),
        total_steps=40_000,
        n_steps=480,
        eval_every=1000,
        eval_episodes=5,
        seed=seed,
    )
    res = train_ppo(env, cfg, eval_env=ChainRho(length=LENGTH, rho=rho, layout_seed=0))
    eps = collect(res["net"], res["obs_norm"], env, episodes)

    scores: dict[str, dict[str, float]] = {}
    for name in estimators(eps[0]):
        rhos, aucs = [], []
        for ep in eps:
            est = estimators(ep)[name]
            rhos.append(spearman(est, ep["exact"]))
            aucs.append(decision_step_auc(est, ep["is_decision"]))
        scores[name] = {
            "spearman": float(np.nanmean(rhos)) if np.any(np.isfinite(rhos)) else float("nan"),
            "decision_auc": float(np.nanmean(aucs)) if np.any(np.isfinite(aucs)) else float("nan"),
        }
    return rho, seed, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="results/credit_study.json")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = [(rho, s, args.episodes) for rho in RHOS for s in range(args.seeds)]
    with mp.Pool(processes=args.workers) as pool:
        results = pool.map(one_rho, jobs)

    agg: dict[float, dict[str, dict[str, list[float]]]] = {}
    for rho, _seed, scores in results:
        for name, sc in scores.items():
            d = agg.setdefault(rho, {}).setdefault(name, {"spearman": [], "decision_auc": []})
            d["spearman"].append(sc["spearman"])
            d["decision_auc"].append(sc["decision_auc"])

    names = list(agg[RHOS[0]])
    print(f"\nchain_rho(length={LENGTH}), {args.seeds} seeds x {args.episodes} episodes")
    print("Spearman rho vs exact per-step credit  /  P(decision step ranked above routine)\n")
    header = "estimator".ljust(22) + "".join(f"{'rho=' + str(r):>20}" for r in RHOS)
    print(header)
    print("-" * len(header))
    for name in names:
        row = name.ljust(22)
        for r in RHOS:
            sp = np.nanmean(agg[r][name]["spearman"])
            au = np.nanmean(agg[r][name]["decision_auc"])
            sp_s = "  n/a" if not np.isfinite(sp) else f"{sp:+.2f}"
            row += f"{sp_s:>9} /{au:6.2f}    "
        print(row)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {str(k): {n: v for n, v in d.items()} for k, d in agg.items()}, indent=2, sort_keys=True
        )
    )
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
