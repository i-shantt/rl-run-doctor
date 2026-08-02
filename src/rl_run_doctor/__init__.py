"""rl-run-doctor: name why a reinforcement-learning run failed.

Imports numpy and nothing else. `torch` must never enter `sys.modules` as a result of importing
this package -- it is meant to install into an existing training image.
"""

from .api import Report, diagnose, diagnose_prefixes
from .diagnose import Diagnosis, LinearDiagnoser
from .signals import feature_matrix, featurize, series_features
from .trace import Trace, read_meta

__version__ = "0.0.1.dev0"

__all__ = [
    "Diagnosis",
    "LinearDiagnoser",
    "Report",
    "Trace",
    "__version__",
    "diagnose",
    "diagnose_prefixes",
    "feature_matrix",
    "featurize",
    "read_meta",
    "series_features",
]
