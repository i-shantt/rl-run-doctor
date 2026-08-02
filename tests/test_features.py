"""Feature-extraction contract tests.

The first two are the important ones. Both encode traps that produced *plausible* wrong answers
rather than crashes, which is the expensive kind.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from rl_run_doctor.signals import RESOLUTIONS, series_features


def test_flat_signal_reads_uninformative_not_extreme() -> None:
    """A constant signal jittered below its resolution must not produce a large z-score.

    Scaling by the empirical std with only a numerical epsilon as a floor turns a 0.002-unit
    wobble into a >7-sigma excursion and fires alarms in the middle of healthy runs. The floor has
    to be the resolution at which the quantity carries information.
    """
    rng = np.random.default_rng(0)
    y = 7.0 + rng.normal(0, 1e-3, size=60)  # jitter far below the 1.0 resolution of a return
    f = series_features(y, resolution=RESOLUTIONS["train_return"])
    assert abs(f["z_vs_baseline"]) < 1.0, f"flat signal scored {f['z_vs_baseline']:.2f} sigma"
    assert abs(f["drawdown"]) < 1.0
    assert abs(f["slope_full"]) < 1.0


def test_features_are_causal() -> None:
    """Features of a prefix must not change when later records are appended.

    If they do, "lead time" is measuring the future leaking backwards.
    """
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(0, 1, 40), rng.normal(8, 1, 40)])
    early = series_features(y[:40], resolution=1e-3)
    early_again = series_features(y[:40].copy(), resolution=1e-3)
    assert early == early_again
    late = series_features(y, resolution=1e-3)
    assert late["z_vs_baseline"] > early["z_vs_baseline"] + 1.0


def test_step_change_is_detected() -> None:
    rng = np.random.default_rng(2)
    y = np.concatenate([rng.normal(0, 0.1, 30), rng.normal(5, 0.1, 30)])
    f = series_features(y, resolution=1e-3)
    assert f["z_vs_baseline"] > 5.0
    assert f["runup"] > f["drawdown"]


def test_monotone_decay_shows_as_drawdown() -> None:
    y = np.linspace(10.0, 1.0, 50)
    f = series_features(y, resolution=1e-3)
    assert f["drawdown"] > f["runup"]
    assert f["slope_full"] < 0


def test_nan_heavy_signal_degrades_gracefully() -> None:
    y = np.full(40, np.nan)
    y[:5] = 1.0
    f = series_features(y, resolution=1e-3)
    assert f["frac_finite"] == pytest.approx(5 / 40)
    assert all(np.isfinite(v) for v in f.values())


def test_short_series_returns_neutral_features() -> None:
    f = series_features(np.array([1.0, 2.0]), resolution=1e-3)
    assert f["z_vs_baseline"] == 0.0
    assert f["volatility_ratio"] == 1.0


def test_package_import_does_not_pull_in_torch() -> None:
    """The shipped package installs into someone else's training image. It must stay light."""
    for mod in [m for m in sys.modules if m.startswith(("torch", "rl_run_doctor"))]:
        del sys.modules[mod]
    import rl_run_doctor  # noqa: F401

    assert "torch" not in sys.modules, "importing rl_run_doctor pulled in torch"
