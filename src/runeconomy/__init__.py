"""RunEconomy core package."""

from .analysis import analyze_trace
from .models import Trace, TraceValidationError
from .workflow import depends_on, step, track_workflow

__version__ = "0.3.0"

__all__ = [
    "Trace",
    "TraceValidationError",
    "__version__",
    "analyze_trace",
    "depends_on",
    "step",
    "track_workflow",
]
