"""One bounded retry policy shared by cold-starting backend operations."""

from __future__ import annotations

import time
from collections.abc import Callable


class StartupRetry:
    """Bound retries to an explicit startup window and notify only once."""

    def __init__(
        self,
        seconds: float,
        *,
        on_retry: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        interval_seconds: float = 2.0,
    ) -> None:
        if seconds < 0:
            raise ValueError("startup retry seconds must be non-negative")
        self._clock = clock
        self._sleep = sleep
        self._deadline = clock() + seconds
        self._enabled = seconds > 0
        self._on_retry = on_retry
        self._notified = False
        self._interval_seconds = interval_seconds

    def available(self) -> bool:
        return self._enabled and self._clock() < self._deadline

    def wait(self) -> None:
        if not self.available():
            return
        if not self._notified and self._on_retry is not None:
            self._on_retry(max(self._deadline - self._clock(), 0.0))
        self._notified = True
        self._sleep(min(self._interval_seconds, max(self._deadline - self._clock(), 0.0)))
