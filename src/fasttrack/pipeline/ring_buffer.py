"""In-memory circular ring buffer of JPEG-compressed frames for bounded pre-roll storage."""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

from src.fasttrack.core.types import Frame


class FrameRingBuffer:
    """Thread-safe circular ring buffer storing pre-roll frames as compressed JPEG bytes.

    Compressing frames on ingestion reduces memory usage by >95% (from ~2.7MB per 1080p frame
    down to ~70KB), ensuring bounded heap utilization across multi-camera streams.
    """

    def __init__(self, capacity_seconds: float = 4.0, jpeg_quality: int = 85) -> None:
        self.capacity_seconds = capacity_seconds
        self.jpeg_quality = int(jpeg_quality)
        self._lock = threading.Lock()
        # Stores (source_ts, ts_utc, jpeg_bytes, frame_idx)
        self._buffer: Deque[Tuple[float, datetime, bytes, int]] = deque(maxlen=200)

    def append(self, frame: Frame) -> None:
        """Compress and append frame to the ring buffer, evicting frames older than capacity."""
        if frame.image is None or frame.image.size == 0:
            return

        # Compress to JPEG bytes
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        success, enc = cv2.imencode(".jpg", frame.image, encode_params)
        if not success:
            return
        jpeg_bytes = enc.tobytes()

        with self._lock:
            self._buffer.append((frame.source_ts, frame.ts_utc, jpeg_bytes, frame.frame_idx))

            # Evict frames older than capacity_seconds
            newest_ts = frame.source_ts
            min_ts = newest_ts - self.capacity_seconds
            while self._buffer and self._buffer[0][0] < min_ts:
                self._buffer.popleft()

    def get_all_frames(self) -> List[Tuple[float, datetime, np.ndarray, int]]:
        """Retrieve and decode all buffered frames into BGR numpy arrays."""
        with self._lock:
            items = list(self._buffer)

        decoded_frames: List[Tuple[float, datetime, np.ndarray, int]] = []
        for source_ts, ts_utc, jpeg_bytes, f_idx in items:
            np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                decoded_frames.append((source_ts, ts_utc, img, f_idx))

        return decoded_frames

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Decode and return the single newest frame."""
        with self._lock:
            if not self._buffer:
                return None
            _, _, jpeg_bytes, _ = self._buffer[-1]

        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def clear(self) -> None:
        """Clear all buffered frames."""
        with self._lock:
            self._buffer.clear()
