"""RunEconomy core package."""

from .analysis import analyze_trace
from .models import Trace, TraceValidationError

__version__ = "0.2.0"

__all__ = ["Trace", "TraceValidationError", "__version__", "analyze_trace"]
