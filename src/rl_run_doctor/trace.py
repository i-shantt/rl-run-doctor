"""Reading run traces. numpy only -- this module ships.

Deliberately duplicates the small amount of read logic in the testbed rather than importing it:
the installed package must not drag in torch, and `testbed` does.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np


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


class Trace:
    """A run's signal history, as parallel arrays.

    The `oracle` block is dropped on load and there is no option to keep it. Diagnostic code
    physically cannot read the label.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.meta = read_meta(path)
        steps: list[int] = []
        updates: list[int] = []
        rows: list[dict[str, float]] = []
        for obj in _iter_lines(path):
            if obj.get("kind") != "update":
                continue
            steps.append(int(obj["env_steps"]))
            updates.append(int(obj["update"]))
            rows.append(obj.get("signals", {}))

        self.env_steps = np.asarray(steps, dtype=np.int64)
        self.updates = np.asarray(updates, dtype=np.int64)
        self.keys: list[str] = sorted({k for r in rows for k in r})
        self.signals: dict[str, np.ndarray] = {
            k: np.asarray([float(r.get(k, np.nan)) for r in rows], dtype=np.float64)
            for k in self.keys
        }

    def __len__(self) -> int:
        return int(self.env_steps.shape[0])

    def get(self, key: str) -> np.ndarray:
        return self.signals[key]

    def prefix(self, n: int) -> Trace:
        """A view of the first `n` records. Used to keep detection causal."""
        clone = object.__new__(Trace)
        clone.path = self.path
        clone.meta = self.meta
        clone.env_steps = self.env_steps[:n]
        clone.updates = self.updates[:n]
        clone.keys = list(self.keys)
        clone.signals = {k: v[:n] for k, v in self.signals.items()}
        return clone
