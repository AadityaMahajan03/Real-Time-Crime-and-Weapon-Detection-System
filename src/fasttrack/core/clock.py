"""Clock abstraction for deterministic temporal testing and production timing."""

from __future__ import annotations

import abc
import time
from datetime import datetime, timezone
from typing import Optional


class Clock(abc.ABC):
    """Abstract interface for time retrieval and pacing."""

    @abc.abstractmethod
    def now_utc(self) -> datetime:
        """Return the current UTC timestamp."""

    @abc.abstractmethod
    def monotonic(self) -> float:
        """Return a monotonic clock value in seconds."""

    @abc.abstractmethod
    def sleep(self, seconds: float) -> None:
        """Sleep or advance time for the specified duration."""


class SystemClock(Clock):
    """Production clock implementation backed by the host operating system."""

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class SimulatedClock(Clock):
    """Controllable virtual clock for deterministic unit and integration tests.

    Allows tests to simulate minutes or hours of streaming video and incident
    timeouts in milliseconds without real delays.
    """

    def __init__(
        self,
        start_utc: Optional[datetime] = None,
        start_monotonic: float = 1000.0,
    ) -> None:
        if start_utc is None:
            self._current_utc = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        else:
            self._current_utc = (
                start_utc if start_utc.tzinfo is not None else start_utc.replace(tzinfo=timezone.utc)
            )
        self._current_monotonic = float(start_monotonic)

    def now_utc(self) -> datetime:
        return self._current_utc

    def monotonic(self) -> float:
        return self._current_monotonic

    def sleep(self, seconds: float) -> None:
        """Instantly advances the virtual clock by the given duration."""
        if seconds > 0:
            self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Advance the simulated timeline by seconds."""
        from datetime import timedelta

        self._current_utc += timedelta(seconds=seconds)
        self._current_monotonic += seconds

    def set_time(self, utc_dt: datetime, monotonic_val: Optional[float] = None) -> None:
        """Directly set the simulated clock state."""
        self._current_utc = utc_dt if utc_dt.tzinfo is not None else utc_dt.replace(tzinfo=timezone.utc)
        if monotonic_val is not None:
            self._current_monotonic = monotonic_val
