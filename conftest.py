"""Make `src/` and the repo root importable during tests without an editable install."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
