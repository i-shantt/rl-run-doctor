"""Health, banding and labelling.

The first two tests pin bugs that were live in this repo and that inflated the smoke gate's result
from 3 usable cells to 11. Both produced plausible output rather than an error, which is why they
survived a full 74-run sweep before anyone noticed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from testbed.health import healthy_band, random_policy_return, run_health
from testbed.label import label_run
from testbed.telemetry import TraceWriter, UpdateRecord


def _write_trace(path: Path, evals: list[float], *, every: int = 1000) -> Path:
    meta = {
        "run_id": "t__ppo__P0_control__s0",
        "spec": {"env_name": "cartpole", "algo": "ppo", "pathology": "P0_control", "seed": 0},
    }
    with TraceWriter(path, meta=meta) as w:
        for i, v in enumerate(evals):
            w.write(
                UpdateRecord(
                    update=i,
                    env_steps=i * every,
                    signals={"train_return": 1.0},
                    oracle={"eval_return": v},
                )
            )
    return path


def test_a_run_matching_a_saturated_control_is_not_a_failure(tmp_path: Path) -> None:
    """The bug that inflated the gate: with `<=` against a floor taken from controls that had
    saturated at the ceiling, a run scoring *identically to the control* was labelled failed.
    Four chain_rho/ppo pathologies scoring exactly 1.000 against a 1.000 control were reported as
    reliable failure vehicles."""
    band = healthy_band([1.0, 1.0])
    assert not band.failed(1.0), "a run equal to the control must not count as failed"
    assert band.threshold < 1.0, "the floor needs a margin, or ties become failures"
    assert band.failed(0.90)


def test_margin_scales_with_the_units_of_the_return() -> None:
    """CartPole returns run to 500 and DeepSea to 1.0; a fixed absolute margin cannot serve both."""
    small = healthy_band([1.0, 1.0])
    large = healthy_band([400.0, 400.0])
    assert small.margin == pytest.approx(0.02, rel=1e-6)
    assert large.margin == pytest.approx(8.0, rel=1e-6)


def test_random_policy_baseline_exposes_a_control_that_learned_nothing() -> None:
    """The second bug: the gate documented a control-health check and never performed one, so
    chain_rho/dqn supplied a floor from a control scoring 0.472 against a random policy's 0.483."""
    rand_chain = random_policy_return("chain_rho", {"length": 24, "rho": 0.5}, n_episodes=10)
    # Random play on chain_rho gets roughly half the decision steps right, so the baseline is far
    # from zero. A control near this value has learned nothing at all.
    assert 0.3 < rand_chain < 0.7

    rand_cart = random_policy_return("cartpole", {}, n_episodes=10)
    assert rand_cart < 60, "a random CartPole policy should fall over quickly"


def test_windowed_health_is_steadier_than_the_final_snapshot(tmp_path: Path) -> None:
    """Measured on real runs: scoring one final evaluation instead of a trailing window inflated
    the across-seed standard deviation from 68 to 161."""
    noisy = _write_trace(tmp_path / "a.jsonl.gz", [300, 320, 310, 305, 40])
    h = run_health(noisy)
    assert h.windowed == pytest.approx((310 + 305 + 40) / 3)
    assert h.eval_curve[-1] == 40
    assert h.windowed > h.eval_curve[-1], "the window must not be dominated by one bad snapshot"


def test_max_drawdown_captures_the_learn_then_diverge_shape(tmp_path: Path) -> None:
    """The signature of value divergence: a run that reaches a good policy and then loses it."""
    p = _write_trace(tmp_path / "b.jsonl.gz", [10, 200, 440, 300, 60])
    h = run_health(p)
    assert h.peak == 440
    assert h.max_drawdown == pytest.approx(380)


def test_label_requires_a_persistent_fall_not_a_dip(tmp_path: Path) -> None:
    """A single sub-floor evaluation is common in healthy runs. Anchoring lead time to a dip would
    make the early-warning claim meaningless."""
    dip = _write_trace(tmp_path / "c.jsonl.gz", [300, 300, 50, 300, 300, 300])
    lab = label_run(dip, floor=100.0)
    assert not lab.failed
    assert lab.degrade_step is None

    real = _write_trace(tmp_path / "d.jsonl.gz", [300, 300, 50, 40, 30, 20])
    lab2 = label_run(real, floor=100.0)
    assert lab2.failed
    assert lab2.degrade_step == 2000, "degradation is dated to the first of the persistent run"


def test_healthy_trailing_window_overrides_a_mid_run_collapse(tmp_path: Path) -> None:
    """A run that dipped and recovered did not fail, whatever happened in the middle."""
    p = _write_trace(tmp_path / "e.jsonl.gz", [300, 20, 20, 300, 310, 305])
    lab = label_run(p, floor=100.0)
    assert not lab.failed
    assert lab.degrade_step is None


def test_band_needs_more_than_one_control() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        healthy_band([1.0])


def test_no_control_can_be_failed_by_a_band_derived_from_those_controls() -> None:
    """Self-consistency. An interpolated quantile on a small sample sits above the smallest
    observation, so a 5th-percentile floor put the worse of two CartPole DQN controls (209.1
    against 395.3, floor 218.4) into the failed class. On a 60-run pilot that mislabelled 3 of 10
    controls, which would have taught the detector that healthy runs are failures."""
    for controls in ([209.117, 395.3], [1.0, 1.0], [0.99, 0.991], [10.0, 20.0, 30.0, 400.0]):
        band = healthy_band(controls)
        for c in controls:
            assert not band.failed(c), f"control {c} failed against its own band {band}"


def test_floor_never_rises_above_the_worst_control() -> None:
    band = healthy_band([209.117, 395.3], q=50.0)  # even a deliberately silly quantile
    assert band.threshold <= min(209.117, 395.3)
