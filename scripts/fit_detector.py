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


def build_xy(rows: list[dict], keys: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    feats = [featurize(r["trace"], keys) for r in rows]
    names = sorted({k for f in feats for k in f})
    x = np.zeros((len(feats), len(names)))
    for i, f in enumerate(feats):
        for j, n in enumerate(names):
            v = f.get(n, 0.0)
            x[i, j] = v if np.isfinite(v) else 0.0
    y = [r["pathology"] for r in rows]
    return x, names, y


def fit(x: np.ndarray, y: list[str]) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    xs = (x - mean) / scale
    clf = LogisticRegression(max_iter=4000, C=0.5, multi_class="multinomial")
    clf.fit(xs, y)
    return clf, mean, scale


def grouped_cv(x: np.ndarray, y: list[str], groups: list[int]) -> float:
    y_arr = np.asarray(y)
    g = np.asarray(groups)
    correct = total = 0
    for held in sorted(set(groups)):
        tr, te = g != held, g == held
        if len(set(y_arr[tr])) < 2 or te.sum() == 0:
            continue
        clf, m, s = fit(x[tr], list(y_arr[tr]))
        pred = clf.predict((x[te] - m) / s)
        correct += int((pred == y_arr[te]).sum())
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--out", default="results")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    rows = load_corpus(corpus)
    if not rows:
        print("empty corpus")
        return

    for cell in sorted({r["cell"] for r in rows}):
        cell_rows = [r for r in rows if r["cell"] == cell]
        keys = sorted(set.intersection(*[set(r["trace"].keys) for r in cell_rows]))
        x, names, y = build_xy(cell_rows, keys)
        groups = [r["seed"] for r in cell_rows]

        print(f"\n=== {cell} ===")
        print(f"{len(cell_rows)} runs, {len(set(y))} classes, {len(names)} features")

        acc = grouped_cv(x, y, groups)
        chance = max(np.bincount(np.unique(y, return_inverse=True)[1])) / len(y)
        print(f"  leave-one-seed-out accuracy : {acc:.3f}   (majority class {chance:.3f})")

        # --- negative controls -------------------------------------------
        step_only = np.array([[f] for f in x[:, names.index("_n_records")]])
        acc_step = grouped_cv(step_only, y, groups)
        ret_cols = [i for i, n in enumerate(names) if n.startswith("train_return.")]
        acc_ret = grouped_cv(x[:, ret_cols], y, groups) if ret_cols else float("nan")
        rng = np.random.default_rng(0)
        acc_shuf = grouped_cv(x, list(rng.permutation(y)), groups)
        print(f"  control: step-index only    : {acc_step:.3f}")
        print(f"  control: train-return only  : {acc_ret:.3f}")
        print(f"  control: shuffled labels    : {acc_shuf:.3f}")
        margin = acc - acc_step
        flag = "OK" if margin >= 0.15 else "TIME-CONFOUNDED"
        print(f"  margin over step-index      : {margin:+.3f}  [{flag}]")

        lopo = leave_one_pathology_out(x, y)
        if lopo:
            print("  leave-one-pathology-out (fraction not called healthy):")
            for k, v in sorted(lopo.items()):
                print(f"      {k:<26} {v:.2f}")

        # --- lead time ---------------------------------------------------
        clf, mean, scale = fit(x, y)
        diag = LinearDiagnoser(
            feature_names=names,
            classes=list(clf.classes_),
            coef=clf.coef_,
            intercept=clf.intercept_,
            mean=mean,
            scale=scale,
        )
        scored = prefix_scores(cell_rows, diag, keys)
        if any(s.failed for s in scored) and any(not s.failed for s in scored):
            lt = lead_time_at_fpr(scored, alpha=args.alpha)
            print(f"  {lt.summary()}")
        else:
            print("  lead time: not computable (need both failed and healthy runs)")

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        diag.save(out / f"diagnoser_{cell.replace('/', '_')}.json")

    print(f"\nmodels written to {args.out}/")


if __name__ == "__main__":
    main()
