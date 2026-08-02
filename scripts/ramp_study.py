"""Does a gradually-arriving fault give any warning?

The dose sweep produced a corpus in which every labelled failure was instantaneous, so lead time
was negative everywhere. Ramped faults are the test of whether that was a property of reinforcement
learning or a property of how the faults were injected.

Each ramp has two onsets, and reporting both is the point:

  `degrade_step`  when the held-out return actually fell -- the label.
  `cliff_step`    when the ramped parameter crossed the value the dose sweep showed to be fatal,
                  known by construction and independent of any evaluation.

A detector that alarms after `cliff_step` but before `degrade_step` is doing the only useful thing
available: it saw the training signals react to a setting that had become destructive, before that
destruction reached the return.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

from rl_run_doctor.diagnose import LinearDiagnoser
from rl_run_doctor.eval import RunScores, lead_time_at_fpr
from rl_run_doctor.signals import featurize
from rl_run_doctor.trace import Trace
from testbed.corpus.runner import RunSpec, run_one
from testbed.health import healthy_band, run_health
from testbed.label import label_run

CELLS = [("cartpole", "ppo", ["R1_lr_ramp", "R2_reward_scale_ramp"]),
         ("cartpole", "dqn", ["R3_target_lag_ramp", "R4_replay_ratio_ramp"])]
MIN_PREFIX = 6


def _run(job: tuple[RunSpec, str]) -> str:
    spec, out_dir = job
    p = Path(out_dir) / f"{spec.run_id}.jsonl.gz"
    if p.exists():
        try:
            run_health(p)
            return str(p)
        except Exception:
            p.unlink(missing_ok=True)
    return str(run_one(spec, out_dir))


def cliff_step(trace: Trace) -> int | None:
    """Env step at which the ramped parameter passed its measured fatal value."""
    meta_path = trace.meta.get("pathology", {})
    cliff = meta_path.get("cliff")
    ramps = meta_path.get("cfg_overrides", {}).get("ramps") or []
    if cliff is None or not ramps:
        return None
    from testbed.inject.ramp import Ramp

    frac = Ramp(**ramps[0]).crosses_at(float(cliff))
    if frac is None:
        return None
    return int(frac * int(trace.env_steps[-1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--dqn-seeds", type=int, default=3, help="R4 is very expensive")
    ap.add_argument("--out", default="scratch/ramp")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for env_name, algo, ramps in CELLS:
        n = args.seeds if algo == "ppo" else args.dqn_seeds
        for s in range(max(n, args.seeds)):
            jobs.append((RunSpec(env_name, algo, "P0_control", s), str(out_dir)))
        for r in ramps:
            for s in range(n):
                jobs.append((RunSpec(env_name, algo, r, s), str(out_dir)))

    print(f"{len(jobs)} runs on {args.workers} workers")
    with mp.Pool(processes=args.workers) as pool:
        list(pool.imap_unordered(_run, jobs))

    report: dict[str, dict] = {}
    for env_name, algo, _ramps in CELLS:
        cell = f"{env_name}/{algo}"
        paths = sorted(out_dir.glob(f"{env_name}__{algo}__*.jsonl.gz"))
        controls = [run_health(p).windowed for p in paths if "__P0_control__" in p.name]
        if len(controls) < 2:
            continue
        band = healthy_band(controls)
        print(f"\n=== {cell} ===")
        print(f"  control n={band.n} mean={band.mean:.1f} floor={band.threshold:.1f}")

        scored: list[RunScores] = []
        rows: list[dict] = []
        for p in paths:
            lab = label_run(p, floor=band.threshold)
            t = Trace(p)
            cs = cliff_step(t)
            h = run_health(p)
            rows.append({"path": p, "lab": lab, "trace": t, "cliff": cs, "health": h})
            if "__P0_control__" not in p.name:
                rel = (h.peak - h.windowed) / max(abs(h.peak), 1e-9)
                print(f"  {p.name.split('__')[2]:<22} s{lab.seed} "
                      f"peak={h.peak:6.1f} final={h.windowed:6.1f} "
                      f"lost={rel:5.0%} failed={str(lab.failed):<5} "
                      f"degrade@{lab.degrade_step} cliff@{cs}")

        # Fit on this cell and replay the alarm over prefixes.
        keys = sorted(set.intersection(*[set(r["trace"].keys) for r in rows]))
        from sklearn.linear_model import LogisticRegression

        feats, ys, groups = [], [], []
        for r in rows:
            t = r["trace"]
            for f in (0.4, 0.6, 0.8, 1.0):
                n = max(MIN_PREFIX, int(round(f * len(t))))
                if n <= len(t):
                    feats.append(featurize(t.prefix(n), keys))
                    ys.append(r["lab"].pathology.split("@")[0])
                    groups.append(r["lab"].seed)
        names = sorted({k for f in feats for k in f})
        x = np.vstack([[f.get(n, 0.0) if np.isfinite(f.get(n, 0.0)) else 0.0 for n in names]
                       for f in feats])
        if len(set(ys)) < 2:
            print("  only one class; skipping detector")
            continue
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale == 0] = 1.0
        clf = LogisticRegression(max_iter=4000, C=0.05).fit((x - mean) / scale, ys)
        diag = LinearDiagnoser(names, list(clf.classes_), clf.coef_, clf.intercept_, mean, scale)

        for r in rows:
            t = r["trace"]
            steps, sc = [], []
            for n in range(MIN_PREFIX, len(t) + 1):
                pr = t.prefix(n)
                sc.append(diag.anomaly_score(featurize(pr, keys)))
                steps.append(int(pr.env_steps[-1]))
            scored.append(RunScores(r["lab"].run_id, np.asarray(steps), np.asarray(sc),
                                    r["lab"].failed, r["lab"].degrade_step))

        if any(s.failed for s in scored) and any(not s.failed for s in scored):
            lt = lead_time_at_fpr(scored, alpha=args.alpha)
            print(f"  vs return drop : {lt.summary()}")
            # Same alarms, measured against the parameter-space onset instead.
            vs_cliff = []
            for r, s in zip(rows, scored, strict=True):
                if not s.failed or r["cliff"] is None:
                    continue
                idx = np.nonzero(s.scores > lt.threshold)[0]
                if idx.size:
                    vs_cliff.append(float(int(s.steps[idx[0]]) - r["cliff"]))
            if vs_cliff:
                print(f"  vs cliff cross : median {np.median(vs_cliff):+.0f} steps "
                      f"({np.mean([v > 0 for v in vs_cliff]):.0%} fired after the parameter "
                      f"became fatal)")
            report[cell] = {
                "median_lead_vs_return": lt.median_lead,
                "detection_rate": lt.detection_rate,
                "fpr": lt.false_alarm_rate,
                "median_vs_cliff": float(np.median(vs_cliff)) if vs_cliff else None,
            }
        else:
            print("  lead time: not computable (need both failed and healthy runs)")

    dest = out_dir / "ramp_report.json"
    dest.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
