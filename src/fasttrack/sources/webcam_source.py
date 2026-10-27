"""USB and integrated laptop webcam source adapter with warm-up settling and health monitoring."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import List, Optional

import cv2
import numpy as np

from src.fasttrack.core.errors import SourceReadError
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import Frame
from src.fasttrack.sources.base import VideoSource

logger = get_logger("sources.webcam")


class WebcamVideoSource(VideoSource):
    """Source adapter for USB webcams and integrated cameras."""

    def __init__(
        self,
        source_id: str,
        device_index: int | str = 0,
        warmup_frames: int = 5,
        fps_override: Optional[float] = None,
    ) -> None:
        super().__init__(source_id, f"webcam:{device_index}")
        self.device_index = int(device_index)
        self.warmup_frames = warmup_frames
        self.fps_override = fps_override
        self._cap: Optional[cv2.VideoCapture] = None
        self._prev_frame_gray: Optional[np.ndarray] = None
        self._frozen_frame_counter = 0
        self._max_frozen_frames = 60

    @classmethod
    def enumerate_cameras(cls, max_to_check: int = 4) -> List[int]:
        """Utility to discover connected and accessible webcam device indices."""
        available: List[int] = []
        for index in range(max_to_check):
            cap = cv2.VideoCapture(index, cv2.CAP_ANY)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(index)
                cap.release()
        return available

    def open(self) -> bool:
        """Open webcam capture and discard warm-up frames to allow exposure to settle."""
        try:
            # On Windows, try cv2.CAP_DSHOW for fastest device initialization
            self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                # Fallback to default API
                self._cap = cv2.VideoCapture(self.device_index)

            if not self._cap.isOpened():
                raise SourceReadError(
                    f"Failed to access webcam device {self.device_index} (permission denied or device busy)."
                )

            # Get or override FPS
            raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
            self._metadata["fps"] = self.fps_override if self.fps_override else self._sanitize_fps(raw_fps)
            self._metadata["width"] = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._metadata["height"] = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self._metadata["device_index"] = self.device_index

            # Discard initial warm-up frames while auto-exposure / white balance settles
            logger.info(
                "Source %s opened; discarding %d warm-up frames for sensor auto-exposure.",
                self.source_id,
                self.warmup_frames,
            )
            for _ in range(self.warmup_frames):
                self._cap.read()

            self._is_opened = True
            self._frame_count = 0
            self._frozen_frame_counter = 0
            self._last_read_time = time.monotonic()
            return True

        except Exception as e:
            self.close()
            raise SourceReadError(f"Exception initializing webcam {self.device_index}: {e}")

    def read(self) -> Optional[Frame]:
        """Read live webcam frame, checking for hot-unplug, black, and frozen frames."""
        if not self._is_opened or self._cap is None:
            return None

        ret, raw_image = self._cap.read()
        now_monotonic = time.monotonic()
        now_utc = datetime.now(timezone.utc)

        if not ret or raw_image is None or raw_image.size == 0:
            self._consecutive_bad_frames += 1
            logger.warning(
                "Webcam %s read failed (Hot-unplug or device disconnect suspected: %d/%d).",
                self.source_id,
                self._consecutive_bad_frames,
                self._max_consecutive_bad_frames,
            )
            if self._consecutive_bad_frames >= self._max_consecutive_bad_frames:
                self.close()
                raise SourceReadError(
                    f"Webcam {self.device_index} disconnected or unresponsive after {self._consecutive_bad_frames} attempts."
                )
            return None

        self._consecutive_bad_frames = 0
        processed_image = self._preprocess_frame(raw_image)

        # Diagnostics: Check for black / blank frames (lens cap or disconnected sensor)
        frame_std = float(np.std(processed_image))
        is_black_frame = frame_std < 2.0
        if is_black_frame:
            logger.debug("Source %s: Black frame detected (std=%.2f).", self.source_id, frame_std)

        # Diagnostics: Check for frozen frames
        gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        if self._prev_frame_gray is not None and self._prev_frame_gray.shape == gray.shape:
            diff = np.mean(np.abs(gray.astype(float) - self._prev_frame_gray.astype(float)))
            if diff < 0.05:
                self._frozen_frame_counter += 1
                if self._frozen_frame_counter % 30 == 0:
                    logger.warning(
                        "Source %s: Frame appears frozen for %d consecutive frames.",
                        self.source_id,
                        self._frozen_frame_counter,
                    )
            else:
                self._frozen_frame_counter = 0

        self._prev_frame_gray = gray
        self._last_read_time = now_monotonic

        frame = Frame(
            image=processed_image,
            ts_utc=now_utc,
            source_ts=now_monotonic,
            frame_idx=self._frame_count,
            source_id=self.source_id,
            metadata={
                "is_black_frame": is_black_frame,
                "frozen_frame_count": self._frozen_frame_counter,
                "is_downscaled": self._metadata["is_downscaled"],
            },
        )

        self._frame_count += 1
        return frame

    def close(self) -> None:
        """Release webcam handle."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing webcam %s: %s", self.source_id, e)
            self._cap = None
        self._is_opened = False
        self._prev_frame_gray = None
