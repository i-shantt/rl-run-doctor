"""Fit the attribution model and run the controls that decide whether to believe it.

Three evaluations, and the last two matter more than the first:

  **grouped CV** -- leave-one-seed-out, so the same seed never spans train and test.

  **leave-one-pathology-out** -- fit on every failure mode but one, test on the held-out one. A
  detector that only recognises mechanisms it was trained on is not an early-warning system, and
  the pre-registered expectation is that this lands near chance.

  **negative controls** -- a step-index-only model, a train-return-only model, and a
  shuffled-label model. If step-index alone comes within 0.15 accuracy of the real thing, the
  corpus is time-confounded and the headline number is an artifact of when injections bite rather
  than of what they do.

Lead time reuses the same fitted model applied to prefixes. That mismatch -- trained on completed
runs, scored on partial ones -- is the actual deployment situation, and is reported rather than
engineered away.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from rl_run_doctor.diagnose import LinearDiagnoser
from rl_run_doctor.eval import RunScores, lead_time_at_fpr
from rl_run_doctor.signals import featurize
from rl_run_doctor.trace import Trace

MIN_PREFIX = 6  # a detector cannot say anything before it has seen a baseline


def load_corpus(corpus_dir: Path) -> list[dict]:
    rows = [json.loads(x) for x in (corpus_dir / "manifest.jsonl").read_text().splitlines() if x]
    for r in rows:
        r["trace"] = Trace(corpus_dir / r["path"])
    return rows


# Fractions of each run used as additional training examples. Two jobs at once: it multiplies a
# tiny sample (a cell has tens of runs against ~150 features, which saturates a logistic model's
# probabilities at exactly 1.0 and makes any calibrated alarm threshold unreachable), and it
# removes the train/score mismatch, since the model is scored on prefixes at deployment.
TRAIN_FRACS = (0.4, 0.6, 0.8, 1.0)

# Deliberately strong. With this many correlated shape features per signal, a weakly regularised
# fit is a lookup table that reports 1.00 confidence on everything.
C_REG = 0.05


def _feature_row(feats: dict[str, float], names: list[str]) -> np.ndarray:
    row = np.zeros(len(names))
    for j, n in enumerate(names):
        v = feats.get(n, 0.0)
        row[j] = v if np.isfinite(v) else 0.0
    return row


def build_xy(
    rows: list[dict], keys: list[str], fracs: tuple[float, ...] = (1.0,)
) -> tuple[np.ndarray, list[str], list[str], list[int]]:
    """Feature matrix over (run, prefix fraction) pairs. Groups are seeds, so a seed never spans
    train and test even when it contributes several prefixes."""
    feats: list[dict[str, float]] = []
    y: list[str] = []
    groups: list[int] = []
    for r in rows:
        t = r["trace"]
        n_rec = len(t)
        for f in fracs:
            n = max(MIN_PREFIX, int(round(f * n_rec)))
            if n > n_rec:
                continue
            feats.append(featurize(t.prefix(n), keys))
            y.append(r["pathology"])
            groups.append(r["seed"])
    names = sorted({k for f in feats for k in f})
    x = np.vstack([_feature_row(f, names) for f in feats])
    return x, names, y, groups


def fit(x: np.ndarray, y: list[str]) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    xs = (x - mean) / scale
    # multinomial is the default for multi-class in current sklearn; the explicit kwarg was
    # removed, so passing it raises rather than being ignored.
    clf = LogisticRegression(max_iter=4000, C=C_REG)
    clf.fit(xs, y)
    return clf, mean, scale


def grouped_cv(
    x: np.ndarray,
    y: list[str],
    groups: list[int],
    x_eval: np.ndarray | None = None,
    y_eval: list[str] | None = None,
    groups_eval: list[int] | None = None,
) -> float:
    """Leave-one-seed-out. Trains on prefixes, scores on whole runs when an eval set is given."""
    y_arr = np.asarray(y)
    g = np.asarray(groups)
    xe = x if x_eval is None else x_eval
    ye = y_arr if y_eval is None else np.asarray(y_eval)
    ge = g if groups_eval is None else np.asarray(groups_eval)

    correct = total = 0
    for held in sorted(set(groups)):
        tr = g != held
        te = ge == held
        if len(set(y_arr[tr])) < 2 or te.sum() == 0:
            continue
        clf, m, s = fit(x[tr], list(y_arr[tr]))
        pred = clf.predict((xe[te] - m) / s)
        correct += int((pred == ye[te]).sum())
        total += int(te.sum())
    return correct / max(total, 1)


def leave_one_pathology_out(x: np.ndarray, y: list[str]) -> dict[str, float]:
    """Can the model place a mechanism it has never seen? Scored as 'did it at least not call it
    healthy', which is the weakest useful claim."""
    y_arr = np.asarray(y)
    out: dict[str, float] = {}
    for held in sorted(set(y) - {"P0_control"}):
        tr, te = y_arr != held, y_arr == held
        if te.sum() == 0 or len(set(y_arr[tr])) < 2:
            continue
        clf, m, s = fit(x[tr], list(y_arr[tr]))
        pred = clf.predict((x[te] - m) / s)
        out[held] = float(np.mean(pred != "P0_control"))
    return out


def prefix_scores(rows: list[dict], diag: LinearDiagnoser, keys: list[str]) -> list[RunScores]:
    out = []
    for r in rows:
        t: Trace = r["trace"]
        steps, scores = [], []
        for n in range(MIN_PREFIX, len(t) + 1):
            p = t.prefix(n)
            scores.append(diag.anomaly_score(featurize(p, keys)))
            steps.append(int(p.env_steps[-1]))
        if not steps:
            continue
        out.append(
            RunScores(
                run_id=r["run_id"],
                steps=np.asarray(steps),
                scores=np.asarray(scores),
                failed=bool(r["failed"]),
                degrade_step=r["degrade_step"],
            )
        )
    return out


def analyse(rows: list[dict], title: str, out: Path, alpha: float) -> None:
    keys = sorted(set.intersection(*[set(r["trace"].keys) for r in rows]))
    if len({r["pathology"] for r in rows}) < 2:
        print(f"\n=== {title} ===\n  only one class present; nothing to attribute")
        return

    x, names, y, groups = build_xy(rows, keys, fracs=TRAIN_FRACS)
    xe, names_e, ye, ge = build_xy(rows, keys, fracs=(1.0,))
    assert names == names_e, "prefix and full-run feature spaces diverged"

    print(f"\n=== {title} ===")
    counts = {c: y.count(c) for c in sorted(set(y))}
    print(f"{len(rows)} runs -> {len(y)} training rows, {len(names)} features, "
          f"{len(keys)} shared signals")
    print(f"  classes: {counts}")

    acc = grouped_cv(x, y, groups, xe, ye, ge)
    chance = max(np.bincount(np.unique(ye, return_inverse=True)[1])) / len(ye)
    print(f"  leave-one-seed-out accuracy : {acc:.3f}   (majority class {chance:.3f})")

    ni = names.index("_n_records")
    acc_step = grouped_cv(x[:, [ni]], y, groups, xe[:, [ni]], ye, ge)
    ret_cols = [i for i, n in enumerate(names) if n.startswith("train_return.")]
    acc_ret = (
        grouped_cv(x[:, ret_cols], y, groups, xe[:, ret_cols], ye, ge)
        if ret_cols else float("nan")
    )
    # Shuffle at the *run* level and rebuild both matrices from the permuted labels. Permuting
    # the training vector alone leaves each run's several prefixes with different random labels
    # while the eval labels stay true, which leaks: it scored 0.625 against a 0.521 majority
    # class, i.e. a model trained on noise beating the trivial baseline.
    rng = np.random.default_rng(0)
    perm = list(rng.permutation([r["pathology"] for r in rows]))
    shuffled_rows = [dict(r, pathology=lab) for r, lab in zip(rows, perm, strict=True)]
    xs_, _, ys_, gs_ = build_xy(shuffled_rows, keys, fracs=TRAIN_FRACS)
    xse, _, yse, gse = build_xy(shuffled_rows, keys, fracs=(1.0,))
    acc_shuf = grouped_cv(xs_, ys_, gs_, xse, yse, gse)
    print(f"  control: step-index only    : {acc_step:.3f}")
    print(f"  control: train-return only  : {acc_ret:.3f}")
    print(f"  control: shuffled labels    : {acc_shuf:.3f}")
    margin = acc - acc_step
    print(f"  margin over step-index      : {margin:+.3f}  "
          f"[{'OK' if margin >= 0.15 else 'TIME-CONFOUNDED'}]")

    lopo = leave_one_pathology_out(x, y)
    if lopo:
        print("  leave-one-pathology-out (fraction not called healthy):")
        for k, v in sorted(lopo.items()):
            print(f"      {k:<26} {v:.2f}")

    clf, mean, scale = fit(x, y)
    diag = LinearDiagnoser(
        feature_names=names, classes=list(clf.classes_), coef=clf.coef_,
        intercept=clf.intercept_, mean=mean, scale=scale,
    )
    scored = prefix_scores(rows, diag, keys)
    if any(s.failed for s in scored) and any(not s.failed for s in scored):
        print(f"  {lead_time_at_fpr(scored, alpha=alpha).summary()}")
    else:
        print("  lead time: not computable (need both failed and healthy runs)")

    out.mkdir(parents=True, exist_ok=True)
    diag.save(out / f"diagnoser_{title.replace('/', '_').replace(' ', '_')}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rows = load_corpus(Path(args.corpus))
    if not rows:
        print("empty corpus")
        return
    out = Path(args.out)

    for cell in sorted({r["cell"] for r in rows}):
        analyse([r for r in rows if r["cell"] == cell], cell, out, args.alpha)

    # Pooled within an algorithm. The features are deliberately baseline-relative shape
    # descriptors rather than levels, precisely so a CartPole gradient norm and a DeepSea gradient
    # norm are comparable. If that design works, one model spans the environments; if it does not,
    # this is where it shows.
    print("\n" + "=" * 60)
    print("POOLED ACROSS ENVIRONMENTS (the claim the features were designed for)")
    print("=" * 60)
    for algo in sorted({r["algo"] for r in rows}):
        pooled = [r for r in rows if r["algo"] == algo]
        envs = sorted({r["env"] for r in pooled})
        analyse(pooled, f"all-envs {algo} ({'+'.join(envs)})", out, args.alpha)

    print(f"\nmodels written to {args.out}/")


if __name__ == "__main__":
    main()
