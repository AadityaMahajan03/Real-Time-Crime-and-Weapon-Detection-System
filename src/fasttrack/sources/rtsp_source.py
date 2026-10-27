"""Authorized RTSP/IP network stream adapter with watchdog, backoff, and buffer flushing."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Optional

import cv2

from src.fasttrack.core.errors import SourceReadError
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import Frame
from src.fasttrack.sources.base import VideoSource
from src.fasttrack.sources.reconnect import ReconnectManager

logger = get_logger("sources.rtsp")


class RTSPVideoSource(VideoSource):
    """Network camera source adapter with automatic reconnection and watchdog."""

    def __init__(
        self,
        source_id: str,
        rtsp_url: str,
        transport: str = "tcp",
        timeout_seconds: float = 5.0,
        max_reconnect_attempts: Optional[int] = 10,
        buffer_size: int = 1,
    ) -> None:
        super().__init__(source_id, rtsp_url)
        self.transport = transport.lower()
        self.timeout_seconds = timeout_seconds
        self.buffer_size = buffer_size
        self._cap: Optional[cv2.VideoCapture] = None
        self._reconnect_mgr = ReconnectManager(
            initial_delay=1.0,
            max_delay=30.0,
            multiplier=2.0,
            jitter_factor=0.25,
            max_attempts=max_reconnect_attempts,
        )
        self._last_frame_time_monotonic = 0.0
        self._last_seen_utc: Optional[datetime] = None

    @property
    def masked_url(self) -> str:
        """Return credentials-masked URL safe for logging and display."""
        if "@" in self.source_ref:
            try:
                prefix, rest = self.source_ref.split("://", 1)
                _, host = rest.split("@", 1)
                return f"{prefix}://***@{host}"
            except Exception:
                return "rtsp://***"
        return self.source_ref

    def _open_capture(self) -> bool:
        """Attempt to establish socket and open RTSP stream with TCP/UDP flags."""
        logger.info("Connecting to RTSP source %s (%s)...", self.source_id, self.masked_url)

        # Set FFMPEG RTSP transport flags for low-latency streaming
        options = f"rtsp_transport;{self.transport}|stimeout;{int(self.timeout_seconds * 1_000_000)}"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options

        self._cap = cv2.VideoCapture(self.source_ref, cv2.CAP_FFMPEG)

        # Limit internal buffer to 1 frame to eliminate network queue lag
        if self._cap.isOpened():
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self.buffer_size))
            except Exception:
                pass

            raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._metadata["fps"] = self._sanitize_fps(raw_fps)
            self._metadata["width"] = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._metadata["height"] = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._metadata["transport"] = self.transport

            self._is_opened = True
            self._last_frame_time_monotonic = time.monotonic()
            self._last_seen_utc = datetime.now(timezone.utc)
            self._reconnect_mgr.record_success()
            logger.info("Successfully established RTSP connection for %s.", self.source_id)
            return True

        logger.warning("Failed to open RTSP stream for %s.", self.source_id)
        return False

    def open(self) -> bool:
        """Open RTSP stream, returning True or raising SourceReadError if all retries fail."""
        if self._open_capture():
            return True

        # Initial reconnect attempts
        return self._reconnect()

    def _reconnect(self) -> bool:
        """Execute exponential backoff reconnection loop."""
        self.close()

        while self._reconnect_mgr.can_retry():
            delay = self._reconnect_mgr.get_next_delay()
            logger.warning(
                "RTSP stream %s disconnected. Reconnect attempt %d in %.2fs...",
                self.source_id,
                self._reconnect_mgr.current_attempt,
                delay,
            )
            time.sleep(delay)

            if self._open_capture():
                return True

        logger.error(
            "RTSP source %s exceeded maximum reconnection attempts (%s).",
            self.source_id,
            self._reconnect_mgr.max_attempts,
        )
        raise SourceReadError(f"RTSP stream {self.source_id} failed after maximum retry attempts.")

    def read(self) -> Optional[Frame]:
        """Read frame with watchdog monitoring for frozen streams and codec failures."""
        if not self._is_opened or self._cap is None:
            if not self._reconnect():
                return None

        # Watchdog: Check if stream has been silent longer than timeout_seconds
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_frame_time_monotonic > self.timeout_seconds:
            logger.warning(
                "Watchdog triggered for %s: No frames received for %.1fs. Reconnecting...",
                self.source_id,
                now_monotonic - self._last_frame_time_monotonic,
            )
            if not self._reconnect():
                return None

        try:
            ret, raw_image = self._cap.read()
        except Exception as e:
            logger.warning("Codec / network decode error on RTSP stream %s: %s", self.source_id, e)
            ret, raw_image = False, None

        if not ret or raw_image is None or raw_image.size == 0:
            self._consecutive_bad_frames += 1
            if self._consecutive_bad_frames >= self._max_consecutive_bad_frames:
                logger.warning(
                    "Source %s consecutive read failures reached %d. Reconnecting...",
                    self.source_id,
                    self._consecutive_bad_frames,
                )
                self._reconnect()
            return None

        # Successful read; update watchdog timestamps and counters
        self._consecutive_bad_frames = 0
        self._last_frame_time_monotonic = time.monotonic()
        self._last_seen_utc = datetime.now(timezone.utc)
        self._reconnect_mgr.record_success()

        processed_image = self._preprocess_frame(raw_image)

        frame = Frame(
            image=processed_image,
            ts_utc=self._last_seen_utc,
            source_ts=self._last_frame_time_monotonic,
            frame_idx=self._frame_count,
            source_id=self.source_id,
            metadata={
                "transport": self.transport,
                "is_downscaled": self._metadata["is_downscaled"],
                "last_seen_utc": self._last_seen_utc.isoformat(),
            },
        )

        self._frame_count += 1
        return frame

    def close(self) -> None:
        """Release network capture handles."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing RTSP capture for %s: %s", self.source_id, e)
            self._cap = None
        self._is_opened = False
