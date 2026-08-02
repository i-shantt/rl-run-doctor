"""Turning a signal history into features a classifier can use.

Everything here is **causal**: features at index `t` are computed from records `0..t` only. That is
what makes lead time measurable -- a feature that peeks ahead would produce a detector that "fires
early" purely by construction.

Levels are mostly useless across environments (a CartPole gradient norm and a DeepSea gradient norm
are not comparable), so features are shape descriptors: change relative to an early baseline,
trend, drawdown, and volatility. The baseline window is skipped rather than scored, because a
detector cannot say anything before it has seen a healthy stretch to compare against.
"""

from __future__ import annotations

import numpy as np

BASELINE_FRAC = 0.2
MIN_BASELINE = 3


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _safe_scale(baseline: np.ndarray, resolution: float) -> float:
    """Scale for a z-like statistic.

    The floor is the *resolution at which the quantity carries information*, never a numerical
    epsilon. A constant signal must read as uninformative; scaling it by 1e-8 turns rounding noise
    into a huge z-score and manufactures alarms on healthy runs.
    """
    if baseline.size < 2:
        return max(resolution, 1e-12)
    return max(float(baseline.std(ddof=1)), resolution)


def _slope(y: np.ndarray) -> float:
    if y.size < 3:
        return 0.0
    x = np.arange(y.size, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y0 = y - y.mean()
    denom = float((x**2).sum())
    return float((x * y0).sum() / denom) if denom > 0 else 0.0


def series_features(y: np.ndarray, resolution: float) -> dict[str, float]:
    """Shape descriptors for one signal, using an early baseline as the reference."""
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    out = {
        "z_vs_baseline": 0.0,
        "rel_change": 0.0,
        "slope_recent": 0.0,
        "slope_full": 0.0,
        "drawdown": 0.0,
        "runup": 0.0,
        "volatility_ratio": 1.0,
        "frac_finite": 0.0,
    }
    finite = _finite(y)
    out["frac_finite"] = float(finite.size) / max(n, 1)
    if finite.size < MIN_BASELINE + 2:
        return out

    nb = max(MIN_BASELINE, int(round(BASELINE_FRAC * finite.size)))
    nb = min(nb, finite.size - 2)
    base, rest = finite[:nb], finite[nb:]
    base_mean = float(base.mean())
    scale = _safe_scale(base, resolution)

    recent = rest[-max(3, rest.size // 4):]
    out["z_vs_baseline"] = (float(recent.mean()) - base_mean) / scale
    denom = max(abs(base_mean), resolution)
    out["rel_change"] = (float(recent.mean()) - base_mean) / denom
    out["slope_full"] = _slope(finite) / scale
    out["slope_recent"] = _slope(recent) / scale

    running_max = np.maximum.accumulate(finite)
    running_min = np.minimum.accumulate(finite)
    out["drawdown"] = float(np.max(running_max - finite)) / scale
    out["runup"] = float(np.max(finite - running_min)) / scale

    base_vol = _safe_scale(base, resolution)
    rest_vol = float(rest.std(ddof=1)) if rest.size > 1 else 0.0
    out["volatility_ratio"] = rest_vol / base_vol
    return out


# The resolution at which each signal carries information. Anything below this is noise, and a
# flat signal at this scale must read as uninformative rather than extreme.
RESOLUTIONS: dict[str, float] = {
    "train_return": 1.0,
    "policy_loss": 1e-3,
    "value_loss": 1e-3,
    "loss": 1e-3,
    "entropy": 1e-3,
    "approx_kl": 1e-5,
    "clip_frac": 1e-3,
    "grad_norm": 1e-3,
    "explained_variance": 1e-2,
    "adv_mean": 1e-3,
    "adv_std": 1e-3,
    "value_mean": 1e-2,
    "value_std": 1e-2,
    "return_mean": 1e-2,
    "weight_norm": 1e-2,
    "dormant_frac": 1e-3,
    "dormant_frac_actor": 1e-3,
    "dormant_frac_critic": 1e-3,
    "effective_rank": 0.5,
    "effective_rank_actor": 0.5,
    "effective_rank_critic": 0.5,
    "td_abs_mean": 1e-3,
    "td_mean": 1e-3,
    "q_mean": 1e-2,
    "q_max_mean": 1e-2,
    "target_gap": 1e-3,
    "replay_age_mean": 1.0,
    "buffer_fill": 1e-3,
    "grad_steps": 1.0,
    "epsilon": 1e-3,
    "lr": 1e-8,
}

DEFAULT_RESOLUTION = 1e-3


def featurize(trace, keys: list[str] | None = None) -> dict[str, float]:
    """Flatten a trace into a named feature vector.

    `train_return` is included: it is something a practitioner genuinely has during training, and
    excluding it would understate the baseline the detector must beat.
    """
    keys = keys if keys is not None else trace.keys
    feats: dict[str, float] = {}
    for k in keys:
        if k not in trace.signals:
            continue
        res = RESOLUTIONS.get(k, DEFAULT_RESOLUTION)
        for name, val in series_features(trace.get(k), res).items():
            feats[f"{k}.{name}"] = val
    feats["_n_records"] = float(len(trace))
    return feats


def feature_matrix(
    traces: list, keys: list[str]
) -> tuple[np.ndarray, list[str]]:
    """Stack featurized traces into (n_runs, n_features) with a shared column order."""
    rows = [featurize(t, keys) for t in traces]
    names = sorted({k for r in rows for k in r})
    x = np.zeros((len(rows), len(names)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, name in enumerate(names):
            v = r.get(name, 0.0)
            x[i, j] = v if np.isfinite(v) else 0.0
    return x, names
