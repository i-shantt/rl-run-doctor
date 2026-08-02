"""The public entry point.

Takes a trace and a fitted diagnoser and returns a named verdict with its evidence. Deliberately
refuses to guess when it has no model: shipping a default before one has been validated would mean
the first thing a user sees is a confident answer from a classifier nobody has scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .diagnose import Diagnosis, LinearDiagnoser
from .signals import featurize
from .trace import Trace


@dataclass(frozen=True)
class Report:
    run: str
    diagnosis: Diagnosis
    anomaly_score: float
    n_records: int

    @property
    def verdict(self) -> str:
        return self.diagnosis.verdict

    def __str__(self) -> str:
        head = (
            f"{Path(self.run).name}: {self.diagnosis.verdict} "
            f"(confidence {self.diagnosis.confidence:.2f}, "
            f"anomaly {self.anomaly_score:.2f}, {self.n_records} records)"
        )
        lines = [head, "  evidence:"]
        for name, contrib in self.diagnosis.evidence[:5]:
            lines.append(f"    {name:<44} {contrib:+.2f}")
        lines.append("  alternatives:")
        for cls, p in self.diagnosis.ranked[1:4]:
            lines.append(f"    {cls:<44} {p:.2f}")
        return "\n".join(lines)


def diagnose(trace_path: str | Path, model: str | Path | LinearDiagnoser) -> Report:
    """Name the most likely cause of a run's behaviour.

    `model` is required. There is no bundled default yet, because none has been validated -- see
    docs/NEGATIVE_RESULTS.md for the controls a model has to clear before one ships here.
    """
    diag = model if isinstance(model, LinearDiagnoser) else LinearDiagnoser.load(model)
    trace = Trace(trace_path)
    if len(trace) == 0:
        raise ValueError(f"{trace_path} contains no update records")
    feats = featurize(trace)
    return Report(
        run=str(trace_path),
        diagnosis=diag.diagnose(feats),
        anomaly_score=diag.anomaly_score(feats),
        n_records=len(trace),
    )


def diagnose_prefixes(
    trace_path: str | Path, model: str | Path | LinearDiagnoser, min_records: int = 6
) -> list[tuple[int, float]]:
    """(env_step, anomaly score) for every prefix, for replaying what a live monitor would have
    seen. Causal by construction: each point uses only the records up to it."""
    diag = model if isinstance(model, LinearDiagnoser) else LinearDiagnoser.load(model)
    trace = Trace(trace_path)
    out = []
    for n in range(min_records, len(trace) + 1):
        p = trace.prefix(n)
        out.append((int(p.env_steps[-1]), diag.anomaly_score(featurize(p))))
    return out
