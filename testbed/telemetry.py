"""Run telemetry, with the label physically separated from the evidence.

Every record has two halves:

  `signals` -- everything a detector is allowed to see. Things a real practitioner has during
               training: losses, gradient norms, entropy, Q statistics, replay ages, weight norms.

  `oracle`  -- the held-out evaluation used to decide whether the run failed. This is the *label*.
               A detector that reads it is scoring itself against its own input.

The split is enforced by the loader, not by discipline: `load_trace` returns signals only unless
you pass `include_oracle=True`, which the detector code never does. This mirrors a trap that cost
real time on a sibling project, where an oracle-derived quantity leaked in through a diagnostic
nobody thought of as oracle data.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass
class UpdateRecord:
    update: int
    env_steps: int
    signals: dict[str, float] = field(default_factory=dict)
    oracle: dict[str, float] = field(default_factory=dict)


class TraceWriter:
    """Append-only gzipped JSONL.

    Records are flushed as they are produced so that a crashed or killed run still yields an
    analysable prefix, and so re-analysis never means re-training.
    """

    def __init__(self, path: str | Path, meta: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Held open for the life of the writer on purpose -- TraceWriter is itself the
        # context manager, and records are flushed as they are produced so a killed run still
        # leaves an analysable prefix.
        self._fh = gzip.open(self.path, "wt", encoding="utf-8")  # noqa: SIM115
        header = {"kind": "meta", "schema_version": SCHEMA_VERSION, **meta}
        self._fh.write(json.dumps(header, sort_keys=True) + "\n")
        self._fh.flush()

    def write(self, rec: UpdateRecord) -> None:
        payload = {"kind": "update", **asdict(rec)}
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _iter_lines(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_meta(path: str | Path) -> dict[str, Any]:
    for obj in _iter_lines(path):
        if obj.get("kind") == "meta":
            return obj
    raise ValueError(f"{path} has no meta header")


def load_trace(path: str | Path, *, include_oracle: bool = False) -> list[dict[str, Any]]:
    """Load update records.

    By default the `oracle` block is stripped. Detector code must never pass
    `include_oracle=True`; only labelling and evaluation code may.
    """
    out: list[dict[str, Any]] = []
    for obj in _iter_lines(path):
        if obj.get("kind") != "update":
            continue
        rec = {
            "update": obj["update"],
            "env_steps": obj["env_steps"],
            "signals": obj.get("signals", {}),
        }
        if include_oracle:
            rec["oracle"] = obj.get("oracle", {})
        out.append(rec)
    return out
