"""Public API and the shipped scorer.

The scorer is what actually installs into someone's training image, so its persistence round-trip
and its refusal to guess without a model both matter more than they look.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rl_run_doctor import LinearDiagnoser, Trace, diagnose, diagnose_prefixes, featurize
from testbed.telemetry import TraceWriter, UpdateRecord


def _trace(path: Path, n: int = 40, drift: float = 0.0) -> Path:
    meta = {
        "run_id": "x__ppo__P0_control__s0",
        "spec": {"env_name": "cartpole", "algo": "ppo", "pathology": "P0_control", "seed": 0},
    }
    with TraceWriter(path, meta=meta) as w:
        for i in range(n):
            w.write(
                UpdateRecord(
                    update=i,
                    env_steps=i * 512,
                    signals={"grad_norm": 1.0 + drift * i, "entropy": 0.7 - 0.001 * i},
                    oracle={"eval_return": 100.0},
                )
            )
    return path


def _names(path: Path) -> list[str]:
    return sorted(featurize(Trace(path)).keys())


def _model(names: list[str]) -> LinearDiagnoser:
    """A two-class model that fires on a rising gradient norm and nothing else."""
    coef = np.zeros((2, len(names)))
    if "grad_norm.z_vs_baseline" in names:
        coef[1, names.index("grad_norm.z_vs_baseline")] = 3.0
    return LinearDiagnoser(
        feature_names=names,
        classes=["P0_control", "P2_trust_region_blowup"],
        coef=coef,
        intercept=np.zeros(2),
        mean=np.zeros(len(names)),
        scale=np.ones(len(names)),
    )


def test_diagnose_returns_a_named_verdict_pointing_at_the_driving_signal(tmp_path: Path) -> None:
    p = _trace(tmp_path / "a.jsonl.gz", drift=0.5)
    rep = diagnose(p, _model(_names(p)))
    assert rep.verdict == "P2_trust_region_blowup"
    assert 0.0 <= rep.anomaly_score <= 1.0
    top = rep.diagnosis.evidence[0][0]
    assert top.startswith("grad_norm"), f"evidence should name the driving signal, got {top}"


def test_a_flat_run_produces_no_evidence_either_way(tmp_path: Path) -> None:
    """With nothing moving, every feature is zero and the model is left at its prior.

    The assertion is *not* that the score is near zero -- a zero-weight model with no intercept is
    genuinely undecided, and 0.5 is the honest answer. What must hold is that an uninformative run
    never scores higher than one carrying real evidence.
    """
    flat = _trace(tmp_path / "b.jsonl.gz", drift=0.0)
    drifting = _trace(tmp_path / "b2.jsonl.gz", drift=0.5)
    names = _names(flat)
    m = _model(names)
    flat_rep, drift_rep = diagnose(flat, m), diagnose(drifting, m)

    assert flat_rep.verdict == "P0_control"
    assert flat_rep.anomaly_score <= 0.5, "no evidence must not read as positive evidence"
    assert drift_rep.anomaly_score > flat_rep.anomaly_score


def test_model_round_trips_through_json(tmp_path: Path) -> None:
    """The artifact is what ships; if it does not survive a save/load it is not a deliverable."""
    p = _trace(tmp_path / "c.jsonl.gz", drift=0.5)
    m = _model(_names(p))
    dest = tmp_path / "m.json"
    m.save(dest)
    loaded = LinearDiagnoser.load(dest)
    a, b = diagnose(p, m), diagnose(p, loaded)
    assert a.verdict == b.verdict
    assert a.anomaly_score == pytest.approx(b.anomaly_score)


def test_prefix_scores_are_ordered_and_accumulate_evidence(tmp_path: Path) -> None:
    p = _trace(tmp_path / "d.jsonl.gz", n=40, drift=0.5)
    scores = diagnose_prefixes(p, _model(_names(p)))
    assert len(scores) == 40 - 6 + 1
    steps = [s for s, _ in scores]
    assert steps == sorted(steps)
    assert scores[-1][1] > scores[0][1]


def test_shape_mismatch_between_coef_and_features_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        LinearDiagnoser(
            feature_names=["a", "b"],
            classes=["x", "y", "z"],
            coef=np.zeros((2, 2)),
            intercept=np.zeros(3),
            mean=np.zeros(2),
            scale=np.ones(2),
        )


def test_empty_trace_is_an_error_not_a_verdict(tmp_path: Path) -> None:
    p = _trace(tmp_path / "e.jsonl.gz", n=0)
    with pytest.raises(ValueError, match="no update records"):
        diagnose(p, _model(["_n_records"]))
