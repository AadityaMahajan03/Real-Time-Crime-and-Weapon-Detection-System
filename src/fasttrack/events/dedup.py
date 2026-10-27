"""Rate limiting, alert-storm protection, and violence hysteresis state machine."""

from __future__ import annotations

from collections import deque
from typing import Dict, Optional

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.logging import get_logger

logger = get_logger("events.dedup")


class AlertStormLimiter:
    """Per-camera and global rate limiter to prevent operator alert fatigue during events."""

    def __init__(
        self,
        camera_limit_per_min: int = 6,
        global_limit_per_min: int = 24,
        window_seconds: float = 60.0,
        clock: Optional[Clock] = None,
    ) -> None:
        self.camera_limit = camera_limit_per_min
        self.global_limit = global_limit_per_min
        self.window_sec = window_seconds
        self.clock = clock or SystemClock()

        # Sliding window timestamp deques: camera_id -> deque of monotonic timestamps
        self._camera_timestamps: Dict[str, deque[float]] = {}
        self._global_timestamps: deque[float] = deque()
        self._suppressed_counts: Dict[str, int] = {}

    def _prune(self, dq: deque[float], cutoff: float) -> None:
        while dq and dq[0] < cutoff:
            dq.popleft()

    def should_allow_alert(self, camera_id: str) -> bool:
        """Evaluate if alert notification is permitted or if it exceeds storm limits."""
        now = self.clock.monotonic()
        cutoff = now - self.window_sec

        # Prune global
        self._prune(self._global_timestamps, cutoff)

        # Prune camera
        if camera_id not in self._camera_timestamps:
            self._camera_timestamps[camera_id] = deque()
            self._suppressed_counts[camera_id] = 0
        cam_dq = self._camera_timestamps[camera_id]
        self._prune(cam_dq, cutoff)

        # Check limits
        if len(self._global_timestamps) >= self.global_limit:
            self._suppressed_counts[camera_id] = self._suppressed_counts.get(camera_id, 0) + 1
            logger.debug("Global alert storm rate limit reached. Suppressing alert.")
            return False

        if len(cam_dq) >= self.camera_limit:
            self._suppressed_counts[camera_id] = self._suppressed_counts.get(camera_id, 0) + 1
            logger.debug("Camera %s alert storm rate limit reached. Suppressing alert.", camera_id)
            return False

        # Alert allowed; record timestamp
        self._global_timestamps.append(now)
        cam_dq.append(now)
        return True

    def get_suppressed_count(self, camera_id: str) -> int:
        """Return total suppressed alert count for camera."""
        return self._suppressed_counts.get(camera_id, 0)

    def reset(self) -> None:
        """Reset all rate limiting state."""
        self._camera_timestamps.clear()
        self._global_timestamps.clear()
        self._suppressed_counts.clear()


class ViolenceHysteresis:
    """Hysteresis filter for violence classification with dual thresholds, consecutive windows, and cooldown."""

    def __init__(
        self,
        t_on: float = 0.65,
        t_off: float = 0.40,
        consecutive_windows: int = 2,
        cooldown_seconds: float = 5.0,
        clock: Optional[Clock] = None,
    ) -> None:
        self.t_on = t_on
        self.t_off = t_off
        self.consecutive_windows = consecutive_windows
        self.cooldown_sec = cooldown_seconds
        self.clock = clock or SystemClock()

        self._active = False
        self._consecutive_high_count = 0
        self._last_trigger_ts = -1000.0

    @property
    def is_active(self) -> bool:
        """Return True if violence state is currently triggered."""
        return self._active

    def update(self, score: float) -> bool:
        """Process new window score and return True if confirmed violence candidate."""
        now = self.clock.monotonic()

        # Cooldown guard: if recently triggered, prevent new trigger until cooldown passes
        if (now - self._last_trigger_ts) < self.cooldown_sec:
            # During cooldown, check if score falls below release threshold to clear state
            if score < self.t_off:
                self._active = False
                self._consecutive_high_count = 0
            return self._active

        if not self._active:
            if score >= self.t_on:
                self._consecutive_high_count += 1
                if self._consecutive_high_count >= self.consecutive_windows:
                    self._active = True
                    self._last_trigger_ts = now
                    logger.info("Violence hysteresis triggered (score: %.3f >= %.2f)", score, self.t_on)
                    return True
            else:
                self._consecutive_high_count = 0
            return False
        else:
            # Active state: release only when score drops below T_off
            if score < self.t_off:
                self._active = False
                self._consecutive_high_count = 0
                logger.info("Violence hysteresis released (score: %.3f < %.2f)", score, self.t_off)
            return self._active
