"""Abstract base class and common interface for all video source adapters."""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.fasttrack.core.errors import SourceReadError
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import Frame

logger = get_logger("sources.base")

# Maximum dimension before early downscaling to protect host RAM/VRAM
MAX_FRAME_DIMENSION = 1920
DEFAULT_FALLBACK_FPS = 25.0


class VideoSource(abc.ABC):
    """Unified video source interface for files, webcams, and RTSP streams."""

    def __init__(self, source_id: str, source_ref: str) -> None:
        self.source_id = source_id
        self.source_ref = source_ref
        self._is_opened = False
        self._frame_count = 0
        self._consecutive_bad_frames = 0
        self._max_consecutive_bad_frames = 30
        self._last_read_time: float = 0.0
        self._metadata: Dict[str, Any] = {
            "fps": DEFAULT_FALLBACK_FPS,
            "width": 0,
            "height": 0,
            "rotation": 0,
            "total_frames": -1,
            "is_downscaled": False,
        }

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return source metadata dictionary."""
        return dict(self._metadata)

    @property
    def is_alive(self) -> bool:
        """Return True if source is opened and actively reading."""
        return self._is_opened

    @abc.abstractmethod
    def open(self) -> bool:
        """Open and initialize capture resource. Returns True on success."""

    @abc.abstractmethod
    def read(self) -> Optional[Frame]:
        """Read and return next decoded Frame or None on EOF / fatal failure."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release capture handles, threads, and hardware resources."""

    def _sanitize_fps(self, raw_fps: float) -> float:
        """Sanitize reported FPS, falling back to 25.0 if zero, negative, or NaN."""
        if raw_fps is None or np.isnan(raw_fps) or raw_fps <= 0.1 or raw_fps > 240.0:
            logger.warning(
                "Source %s reported invalid FPS (%s). Defaulting to %.1f.",
                self.source_id,
                raw_fps,
                DEFAULT_FALLBACK_FPS,
            )
            return DEFAULT_FALLBACK_FPS
        return float(raw_fps)

    def _preprocess_frame(self, raw_image: np.ndarray) -> np.ndarray:
        """Verify, rotate if necessary, and downscale 4K frames early."""
        if raw_image is None or raw_image.size == 0:
            raise SourceReadError(f"Received empty frame from source {self.source_id}")

        h, w = raw_image.shape[:2]

        # Early downscaling for huge resolutions (e.g. 4K, 8K)
        if max(h, w) > MAX_FRAME_DIMENSION:
            scale = MAX_FRAME_DIMENSION / float(max(h, w))
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            raw_image = cv2.resize(raw_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self._metadata["is_downscaled"] = True
            self._metadata["width"] = new_w
            self._metadata["height"] = new_h

        return raw_image

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
