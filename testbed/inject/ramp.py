"""Faults that arrive gradually.

Every pathology in the first taxonomy is set once and held, so the outcome is binary: the
configuration is either survivable or it is not. That produced a corpus in which the mechanisms
severe enough to label were also too fast to warn about, and the ones slow enough to warn about
never degraded enough to be labelled.

A ramp starts at the control's value and moves toward a destructive one during the run, crossing
the cliff the dose sweep located. That buys two things:

* a genuine onset, with a stretch of training where the setting is bad but not yet fatal -- which
  is the only regime in which early warning can exist at all;
* a *second* ground truth. The step at which the parameter crosses the measured cliff is known by
  construction, independent of the held-out evaluation. If a detector fires before the return
  falls, the parameter-crossing step says whether it was reading real evidence or getting lucky.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Ramp:
    """Move `field` from `start` to `end` between two fractions of training."""

    field: str
    start: float
    end: float
    start_frac: float = 0.0
    end_frac: float = 1.0
    # Geometric interpolation, for quantities that live on a multiplicative scale (learning
    # rates, reward scales, refresh intervals). Linear interpolation from 200 to 100_000 spends
    # almost the whole run in a regime the sweep showed to be harmless.
    log: bool = True

    def value(self, progress: float) -> float:
        if progress <= self.start_frac:
            return self.start
        if progress >= self.end_frac:
            return self.end
        span = max(self.end_frac - self.start_frac, 1e-9)
        t = (progress - self.start_frac) / span
        if self.log and self.start > 0 and self.end > 0:
            lo, hi = math.log(self.start), math.log(self.end)
            return float(math.exp(lo + t * (hi - lo)))
        return float(self.start + t * (self.end - self.start))

    def crosses_at(self, threshold: float) -> float | None:
        """Fraction of training at which the ramp passes `threshold`, or None if it never does.

        This is the parameter-space onset: the point after which the run is configured into the
        regime the dose sweep showed to be fatal.
        """
        lo, hi = min(self.start, self.end), max(self.start, self.end)
        if not (lo <= threshold <= hi):
            return None
        if self.log and self.start > 0 and self.end > 0:
            num = math.log(threshold) - math.log(self.start)
            den = math.log(self.end) - math.log(self.start)
        else:
            num = threshold - self.start
            den = self.end - self.start
        if den == 0:
            return None
        t = num / den
        return self.start_frac + t * (self.end_frac - self.start_frac)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_ramps(cfg: Any, ramps: tuple[Ramp, ...], progress: float) -> dict[str, float]:
    """Return the ramped field values for this point in training.

    Returns rather than mutates: the config stays the immutable record of what was *configured*,
    and the training loop uses the returned values for this update only.
    """
    return {r.field: r.value(progress) for r in ramps}
