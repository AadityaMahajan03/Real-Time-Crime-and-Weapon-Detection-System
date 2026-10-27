"""Reconnection strategy and exponential backoff with jitter for network streams."""

from __future__ import annotations

import random
from typing import Optional


class ReconnectManager:
    """Manages reconnect delays with exponential backoff and jitter."""

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
        multiplier: float = 2.0,
        jitter_factor: float = 0.25,
        max_attempts: Optional[int] = None,
    ) -> None:
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter_factor = jitter_factor
        self.max_attempts = max_attempts
        self.current_attempt = 0
        self._current_delay = initial_delay

    def can_retry(self) -> bool:
        """Return True if retry limit has not been exceeded."""
        if self.max_attempts is None:
            return True
        return self.current_attempt < self.max_attempts

    def get_next_delay(self) -> float:
        """Calculate and return next backoff delay in seconds with jitter."""
        self.current_attempt += 1

        # Calculate base exponential delay
        delay = self._current_delay
        self._current_delay = min(self.max_delay, self._current_delay * self.multiplier)

        # Apply random jitter (+/- jitter_factor)
        jitter = delay * self.jitter_factor * (random.uniform(-1.0, 1.0))
        actual_delay = max(0.1, delay + jitter)
        return round(actual_delay, 3)

    def record_success(self) -> None:
        """Reset retry counters after a successful sustained connection."""
        self.current_attempt = 0
        self._current_delay = self.initial_delay
