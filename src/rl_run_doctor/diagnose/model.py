"""The shipped scorer: numpy only, loaded from exported weights.

Fitting happens offline with whatever library is convenient; what installs into someone's training
image is a JSON of feature names, standardisation constants, and coefficients, plus the ~20 lines
of numpy needed to evaluate them. This keeps the dependency promise without giving up a real
classifier, and it makes the model auditable -- you can read the weights.

Standardisation constants are part of the artifact. Re-deriving them from the run being diagnosed
would mean a single run is scored against its own statistics, so a uniformly broken run would look
perfectly normal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Diagnosis:
    verdict: str
    confidence: float
    ranked: list[tuple[str, float]]
    evidence: list[tuple[str, float]]  # (feature, signed contribution), largest first

    def explain(self, top: int = 4) -> str:
        lines = [f"{self.verdict}  (confidence {self.confidence:.2f})"]
        for name, contrib in self.evidence[:top]:
            lines.append(f"    {name:<40} {contrib:+.2f}")
        return "\n".join(lines)


class LinearDiagnoser:
    """Multinomial logistic scorer over standardised features."""

    def __init__(
        self,
        feature_names: list[str],
        classes: list[str],
        coef: np.ndarray,
        intercept: np.ndarray,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        self.feature_names = list(feature_names)
        self.classes = list(classes)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        if self.coef.shape != (len(self.classes), len(self.feature_names)):
            raise ValueError(
                f"coef shape {self.coef.shape} does not match "
                f"{len(self.classes)} classes x {len(self.feature_names)} features"
            )

    # -- persistence ------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(
            {
                "feature_names": self.feature_names,
                "classes": self.classes,
                "coef": self.coef.tolist(),
                "intercept": self.intercept.tolist(),
                "mean": self.mean.tolist(),
                "scale": self.scale.tolist(),
            },
            indent=2,
        )

    @classmethod
    def load(cls, path: str | Path) -> LinearDiagnoser:
        d = json.loads(Path(path).read_text())
        return cls(
            d["feature_names"],
            d["classes"],
            np.asarray(d["coef"]),
            np.asarray(d["intercept"]),
            np.asarray(d["mean"]),
            np.asarray(d["scale"]),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    # -- scoring ----------------------------------------------------------
    def _vector(self, feats: dict[str, float]) -> np.ndarray:
        x = np.array([feats.get(n, 0.0) for n in self.feature_names], dtype=np.float64)
        return np.where(np.isfinite(x), x, 0.0)

    def _standardise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.where(self.scale > 0, self.scale, 1.0)

    def probabilities(self, feats: dict[str, float]) -> np.ndarray:
        z = self.coef @ self._standardise(self._vector(feats)) + self.intercept
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def anomaly_score(self, feats: dict[str, float]) -> float:
        """One number for "this run does not look healthy".

        Defined as 1 - P(control), so the same fitted model serves both the alarm and the
        attribution rather than needing a second calibration.
        """
        p = self.probabilities(feats)
        if "P0_control" in self.classes:
            return float(1.0 - p[self.classes.index("P0_control")])
        return float(1.0 - p.max())

    def diagnose(self, feats: dict[str, float]) -> Diagnosis:
        p = self.probabilities(feats)
        order = np.argsort(-p)
        best = int(order[0])
        xs = self._standardise(self._vector(feats))
        contrib = self.coef[best] * xs
        ev_order = np.argsort(-np.abs(contrib))
        return Diagnosis(
            verdict=self.classes[best],
            confidence=float(p[best]),
            ranked=[(self.classes[i], float(p[i])) for i in order],
            evidence=[(self.feature_names[i], float(contrib[i])) for i in ev_order],
        )
