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
# 1.0 is retained but is structurally unmeasurable: at full density there are no routine steps, so
# `decision_step_auc` has no negative class, and a competent policy makes `true_credit` constant so
# Spearman is undefined too. 0.75 is the highest density that can actually be scored, and is the
# anchor the Phase 3 prediction is tested at. See docs/NEGATIVE_RESULTS.md.
RHOS = [1.0, 0.75, 0.5, 0.25, 0.1]
# Training budgets at which the estimators are scored. Scoring only the converged policy would
# measure credit assignment at the point where a good critic has already driven every advantage
# to ~0, and would answer a question nobody asks -- the estimators matter while they are still
# driving updates. An unconverged policy also errs on decision steps, which is what makes exact
# credit non-constant and rho=1.0 scoreable at all.
CHECKPOINTS = [4_000, 12_000, 40_000]


def collect(net, obs_norm, env: ChainRho, n_episodes: int) -> list[dict[str, np.ndarray]]:
    """Roll the trained policy and record what each estimator would see.

    Actions are **sampled**, not argmaxed. `ChainRho` is deterministic given the action sequence
    and ignores its reset seed, so a greedy policy replays one identical episode `n_episodes`
    times and the spread across episodes is zero by construction. Sampling is also the honest
    distribution to score on: these estimators are what PPO consumes on-policy, and a policy that
    never errs generates a constant `true_credit`, which makes Spearman undefined.
    """
    out = []
    for seed in eval_seeds(n_episodes):
        gen = torch.Generator().manual_seed(seed)
        obs = env.reset(seed=seed)
        rewards, values, is_dec = [], [], []
        for t in range(env.length):
            o = obs_norm(obs)
            with torch.no_grad():
                logits, v = net(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                a = int(torch.multinomial(probs, 1, generator=gen).item())
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


def one_rho(
    args: tuple[float, int, int, int, bool],
) -> tuple[float, int, int, dict[str, dict[str, float]], dict[str, float]]:
    rho, seed, steps, episodes, vary_layout = args
    # The layout varies with the seed. Holding it at 0 makes every number conditional on one
    # arrangement of decision steps, and the across-seed CI cannot see that: once the policy
    # converges its trajectories are identical, so five seeds report the same value to three
    # decimals and the interval collapses to +/-0.000. That reads as precision and is not.
    layout = seed if vary_layout else 0
    env = ChainRho(length=LENGTH, rho=rho, layout_seed=layout)
    cfg = replace(
        PPOConfig(),
        total_steps=steps,
        n_steps=480,
        eval_every=1000,
        eval_episodes=5,
        seed=seed,
    )
    res = train_ppo(env, cfg, eval_env=ChainRho(length=LENGTH, rho=rho, layout_seed=layout))
    eps = collect(res["net"], res["obs_norm"], env, episodes)

    # Competence diagnostics. Without these a low correlation is unreadable: it could mean the
    # estimator fails to rank the steps that mattered, or it could mean the policy never solved the
    # task and there was nothing to rank. `hit_rate` is the fraction of decision steps where the
    # cued action was taken (0.5 = chance); `degenerate` is the fraction of episodes where exact
    # credit is constant, which is where Spearman is undefined rather than low.
    hit, degen = [], []
    for ep in eps:
        dec = ep["is_decision"].astype(bool)
        hit.append(float((ep["exact"][dec] > 0).mean()) if dec.any() else float("nan"))
        degen.append(float(np.unique(ep["exact"]).size < 2))
    diag = {
        "hit_rate": float(np.nanmean(hit)),
        "degenerate_frac": float(np.mean(degen)),
        "n_decision": int(env.n_decision),
        "n_routine": int(env.length - env.n_decision),
    }

    per_ep = [estimators(ep) for ep in eps]
    scores: dict[str, dict[str, float]] = {}
    for name in per_ep[0]:
        rhos, aucs, idxs = [], [], []
        for ep, est_map in zip(eps, per_ep, strict=True):
            est = est_map[name]
            rhos.append(spearman(est, ep["exact"]))
            aucs.append(decision_step_auc(est, ep["is_decision"]))
            # The control that makes the other two readable. On a terminal-only reward an
            # estimator can rank steps purely by recency and never touch causal relevance, so a
            # near-+/-1 value here means the estimator is the step index in disguise.
            idxs.append(spearman(est, np.arange(est.size, dtype=np.float64)))

        def mean(xs: list[float]) -> float:
            return float(np.nanmean(xs)) if np.any(np.isfinite(xs)) else float("nan")

        scores[name] = {
            "spearman": mean(rhos),
            "decision_auc": mean(aucs),
            "step_index_rho": mean(idxs),
        }
    return rho, seed, steps, scores, diag


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--out", default="results/credit_study.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--vary-layout",
        action="store_true",
        help="draw a fresh decision-step layout per seed, so the across-seed CI covers it",
    )
    args = ap.parse_args()

    jobs = [
        (rho, s, steps, args.episodes, args.vary_layout)
        for steps in CHECKPOINTS
        for rho in RHOS
        for s in range(args.seeds)
    ]

    # Flushed as produced: a bug in the reporting below then costs seconds, not the whole sweep.
    raw = Path(args.out).with_suffix(".jsonl")
    raw.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with mp.Pool(processes=args.workers) as pool, raw.open("w") as fh:
        for i, (rho, seed, steps, scores, diag) in enumerate(pool.imap_unordered(one_rho, jobs), 1):
            rec = {"rho": rho, "seed": seed, "steps": steps, "scores": scores, "diag": diag}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            results.append(rec)
            print(
                f"[{i}/{len(jobs)}] steps={steps} rho={rho} seed={seed} "
                f"hit_rate={diag['hit_rate']:.2f} degenerate={diag['degenerate_frac']:.2f}",
                flush=True,
            )

    def mean_or_nan(xs: list[float]) -> float:
        finite = [x for x in xs if np.isfinite(x)]
        return float(np.mean(finite)) if finite else float("nan")

    def cell(x: float, fmt: str) -> str:
        return "n/a" if not np.isfinite(x) else format(x, fmt)

    agg: dict[tuple[int, float], dict[str, dict[str, list[float]]]] = {}
    diags: dict[tuple[int, float], list[dict[str, float]]] = {}
    for rec in results:
        key = (rec["steps"], rec["rho"])
        diags.setdefault(key, []).append(rec["diag"])
        for name, sc in rec["scores"].items():
            d = agg.setdefault(key, {}).setdefault(
                name, {"spearman": [], "decision_auc": [], "step_index_rho": []}
            )
            for metric in ("spearman", "decision_auc", "step_index_rho"):
                d[metric].append(sc[metric])

    names = list(agg[CHECKPOINTS[0], RHOS[0]])
    print(f"\nchain_rho(length={LENGTH}), {args.seeds} seeds x {args.episodes} episodes")
    print("Spearman rho vs exact per-step credit  /  P(decision step ranked above routine)")
    print("[t] = Spearman vs the step index; near +/-1 means the estimator is recency, not credit.")
    for steps in CHECKPOINTS:
        print(f"\n--- after {steps:,} env steps " + "-" * 60)
        header = "estimator".ljust(22) + "".join(f"{'rho=' + str(r):>26}" for r in RHOS)
        print(header)
        for name in names:
            row = name.ljust(22)
            for r in RHOS:
                sp = cell(mean_or_nan(agg[steps, r][name]["spearman"]), "+.2f")
                au = cell(mean_or_nan(agg[steps, r][name]["decision_auc"]), ".2f")
                ix = cell(mean_or_nan(agg[steps, r][name]["step_index_rho"]), "+.2f")
                row += f"{sp:>9} /{au:>6}  [t]{ix:>6}   "
            print(row)
        # Read the table only alongside these. A low score where hit_rate is near 0.5 means the
        # policy never learned the task, not that the estimator failed; a cell where every episode
        # is degenerate has no Spearman to report at all.
        for label, key_ in [
            ("  hit_rate on decision steps", "hit_rate"),
            ("  episodes w/ constant credit", "degenerate_frac"),
        ]:
            row = label.ljust(22)
            for r in RHOS:
                v = mean_or_nan([d[key_] for d in diags[steps, r]])
                row += f"{cell(v, '.2f'):>9}                "
            print(row)

    row = "decision/routine".ljust(22)
    for r in RHOS:
        d = diags[CHECKPOINTS[0], r][0]
        row += f"{str(d['n_decision']) + '/' + str(d['n_routine']):>9}                "
    print("\n" + row)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(
            {
                "scores": {f"{s}|{r}": d for (s, r), d in agg.items()},
                "diagnostics": {f"{s}|{r}": v for (s, r), v in diags.items()},
                "config": {
                    "length": LENGTH,
                    "seeds": args.seeds,
                    "episodes": args.episodes,
                    "rhos": RHOS,
                    "checkpoints": CHECKPOINTS,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
