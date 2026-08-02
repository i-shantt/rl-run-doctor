"""Lead-time metric tests, on synthetic scores where the right answer is known by construction."""

from __future__ import annotations

import numpy as np
import pytest

from rl_run_doctor.eval import RunScores, calibrate_threshold, lead_time_at_fpr


def _run(rid: str, scores: list[float], failed: bool, degrade: int | None) -> RunScores:
    steps = np.arange(len(scores)) * 100
    return RunScores(rid, steps, np.asarray(scores, dtype=float), failed, degrade)


def test_perfect_detector_gets_positive_lead() -> None:
    healthy = [_run(f"h{i}", [0.0] * 10, False, None) for i in range(5)]
    # Score rises at index 2 (step 200); degradation confirmed at step 700.
    failed = [_run(f"f{i}", [0, 0, 5, 5, 5, 5, 5, 5, 5, 5], True, 700) for i in range(3)]
    res = lead_time_at_fpr(healthy + failed, alpha=0.05)
    assert res.false_alarm_rate == 0.0
    assert res.detection_rate == 1.0
    assert res.median_lead == pytest.approx(500.0)
    assert res.frac_positive_lead == 1.0


def test_late_detector_reports_negative_lead_rather_than_hiding_it() -> None:
    healthy = [_run(f"h{i}", [0.0] * 10, False, None) for i in range(5)]
    # Alarm at step 800, but the return already fell at step 300.
    failed = [_run("f0", [0] * 8 + [9, 9], True, 300)]
    res = lead_time_at_fpr(healthy + failed, alpha=0.05)
    assert res.median_lead == pytest.approx(-500.0)
    assert res.frac_positive_lead == 0.0


def test_missed_failure_counts_against_detection_rate() -> None:
    healthy = [_run(f"h{i}", [0.0] * 10, False, None) for i in range(5)]
    failed = [
        _run("f0", [5.0] * 10, True, 500),  # detected
        _run("f1", [0.0] * 10, True, 500),  # never alarms
    ]
    res = lead_time_at_fpr(healthy + failed, alpha=0.05)
    assert res.detection_rate == pytest.approx(0.5)


def test_threshold_is_calibrated_on_healthy_runs_only() -> None:
    """If the threshold moved with the failed runs' scores, the label would be leaking in."""
    healthy = [_run(f"h{i}", [0.0, 1.0], False, None) for i in range(20)]
    thr_a = calibrate_threshold(healthy, alpha=0.05)
    thr_b = calibrate_threshold(healthy, alpha=0.05)
    assert thr_a == thr_b
    assert thr_a > 1.0  # above every healthy peak, so none of them alarm


def test_alpha_controls_the_false_alarm_rate() -> None:
    rng = np.random.default_rng(0)
    healthy = [_run(f"h{i}", list(rng.normal(0, 1, 20)), False, None) for i in range(100)]
    failed = [_run("f0", [10.0] * 20, True, 1000)]
    strict = lead_time_at_fpr(healthy + failed, alpha=0.01)
    loose = lead_time_at_fpr(healthy + failed, alpha=0.25)
    assert strict.false_alarm_rate <= 0.02
    assert loose.false_alarm_rate > strict.false_alarm_rate
    assert strict.threshold > loose.threshold


def test_no_failed_runs_is_an_error_not_a_zero() -> None:
    healthy = [_run("h0", [0.0], False, None)]
    with pytest.raises(ValueError, match="no failed runs"):
        lead_time_at_fpr(healthy, alpha=0.05)
