"""Lead time at a fixed false-alarm rate.

The number a practitioner actually needs, and the one no published RL diagnostic reports. Existing
work reports AUROC over runs or a metric trajectory; neither answers "if I let this fire, how much
warning do I get, and how often will it cry wolf on a healthy run?"

Definitions used here:

* A detector emits a score for every **prefix** of a run. Scores must be causal -- computed from
  records up to that point only.
* A run *alarms* at the first prefix whose score exceeds the threshold.
* The threshold is calibrated on healthy runs so that at most `alpha` of them alarm at any point.
  Calibrating on the failed runs, or on all runs pooled, would leak the label.
* Lead time for a failed run is `degrade_step - alarm_step`. Positive means the alarm came before
  the held-out return actually fell. Negative means the reward curve got there first, and the
  detector added nothing.

A detector that never alarms on a failed run gets no lead time and counts as a miss; it does not
get to skip the case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RunScores:
    run_id: str
    steps: np.ndarray  # env step at each prefix
    scores: np.ndarray  # detector score at each prefix
    failed: bool
    degrade_step: int | None


@dataclass(frozen=True)
class LeadTimeResult:
    alpha: float
    threshold: float
    n_healthy: int
    n_failed: int
    false_alarm_rate: float
    detection_rate: float  # fraction of failed runs that alarmed at all
    lead_times: list[float]  # one per *detected* failed run, in env steps
    median_lead: float
    mean_lead: float
    frac_positive_lead: float  # detected runs where the alarm preceded degradation

    def summary(self) -> str:
        return (
            f"alpha={self.alpha:.2f} thr={self.threshold:.3f} "
            f"FPR={self.false_alarm_rate:.2f} detect={self.detection_rate:.2f} "
            f"median_lead={self.median_lead:+.0f} steps "
            f"({self.frac_positive_lead:.0%} of detections were early)"
        )


def _first_alarm(steps: np.ndarray, scores: np.ndarray, thr: float) -> int | None:
    idx = np.nonzero(scores > thr)[0]
    return int(steps[idx[0]]) if idx.size else None


def calibrate_threshold(healthy: list[RunScores], alpha: float) -> float:
    """Smallest threshold whose false-alarm rate over healthy runs is <= alpha.

    Uses each healthy run's *maximum* score, since a run alarms if it ever crosses. With `alpha`
    small and few healthy runs the threshold is pinned just above the worst healthy run, which is
    the honest answer -- you cannot demonstrate a 1% false-alarm rate with 20 healthy runs, and
    the reported FPR makes that visible.
    """
    if not healthy:
        raise ValueError("need healthy runs to calibrate a false-alarm rate")
    peaks = np.array([float(np.nanmax(h.scores)) if h.scores.size else -np.inf for h in healthy])
    peaks = peaks[np.isfinite(peaks)]
    if peaks.size == 0:
        return 0.0
    # Threshold at the (1 - alpha) quantile of healthy peaks, nudged up so ties do not alarm.
    q = float(np.quantile(peaks, 1.0 - alpha))
    return float(np.nextafter(q, np.inf))


def lead_time_at_fpr(runs: list[RunScores], alpha: float = 0.05) -> LeadTimeResult:
    healthy = [r for r in runs if not r.failed]
    failed = [r for r in runs if r.failed]
    if not failed:
        raise ValueError("no failed runs: lead time is undefined")

    thr = calibrate_threshold(healthy, alpha)

    n_false = sum(1 for h in healthy if _first_alarm(h.steps, h.scores, thr) is not None)
    fpr = n_false / max(len(healthy), 1)

    leads: list[float] = []
    n_detected = 0
    for f in failed:
        alarm = _first_alarm(f.steps, f.scores, thr)
        if alarm is None:
            continue
        n_detected += 1
        if f.degrade_step is None:
            # Failed by trailing window but never crossed the floor persistently mid-run; there is
            # no reference point, so it counts as a detection with no lead time measured.
            continue
        leads.append(float(f.degrade_step - alarm))

    detection = n_detected / len(failed)
    if leads:
        median_lead = float(np.median(leads))
        mean_lead = float(np.mean(leads))
        frac_pos = float(np.mean([x > 0 for x in leads]))
    else:
        median_lead = mean_lead = 0.0
        frac_pos = 0.0

    return LeadTimeResult(
        alpha=alpha,
        threshold=thr,
        n_healthy=len(healthy),
        n_failed=len(failed),
        false_alarm_rate=fpr,
        detection_rate=detection,
        lead_times=leads,
        median_lead=median_lead,
        mean_lead=mean_lead,
        frac_positive_lead=frac_pos,
    )
