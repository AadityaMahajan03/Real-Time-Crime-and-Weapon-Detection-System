"""Recorded video file source adapter with format validation, 4K downscaling, and loop support."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2

from src.fasttrack.core.errors import SourceReadError, ValidationError
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import Frame
from src.fasttrack.sources.base import DEFAULT_FALLBACK_FPS, VideoSource

logger = get_logger("sources.file")

# Permitted video container extensions
VALID_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB limit


class FileVideoSource(VideoSource):
    """Source adapter for recorded surveillance video files."""

    def __init__(
        self,
        source_id: str,
        file_path: str | Path,
        loop: bool = False,
        min_clip_duration_sec: float = 2.0,
    ) -> None:
        super().__init__(source_id, str(file_path))
        self.path = Path(file_path)
        self.loop = loop
        self.min_clip_duration_sec = min_clip_duration_sec
        self._cap: Optional[cv2.VideoCapture] = None
        self._start_wall_time = datetime.now(timezone.utc)
        self._last_valid_frame: Optional[Frame] = None
        self._total_frames = 0
        self._fps = DEFAULT_FALLBACK_FPS

    def _validate_file(self) -> None:
        """Validate file presence, extension, size, and header integrity."""
        if not self.path.exists():
            raise SourceReadError(f"Video file does not exist: {self.path}")

        if not self.path.is_file():
            raise SourceReadError(f"Path is not a regular file: {self.path}")

        # Check extension
        ext = self.path.suffix.lower()
        if ext not in VALID_EXTENSIONS:
            raise ValidationError(
                f"Unsupported video file extension '{ext}'. Allowed: {', '.join(sorted(VALID_EXTENSIONS))}"
            )

        # Cap upload/file size
        file_size = self.path.stat().st_size
        if file_size == 0:
            raise SourceReadError(f"Video file is empty (0 bytes): {self.path}")

        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValidationError(f"File size ({file_size} bytes) exceeds 2GB maximum limit.")

    def open(self) -> bool:
        """Open video file and validate by decoding the initial frame."""
        self._validate_file()

        # OpenCV capture initialization with stringified path
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise SourceReadError(f"OpenCV failed to open video file: {self.path}")

        # Retrieve video properties
        raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = self._sanitize_fps(raw_fps)
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Test decode of the first frame to guarantee file is not corrupted or audio-only
        ret, test_frame = self._cap.read()
        if not ret or test_frame is None or test_frame.size == 0:
            self.close()
            raise SourceReadError(
                f"File opened but could not decode first frame (audio-only or corrupt stream): {self.path}"
            )

        # Reset read cursor to beginning
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Calculate duration and check for short clips
        duration_sec = (self._total_frames / self._fps) if self._fps > 0 and self._total_frames > 0 else 0.0
        insufficient_frames = duration_sec > 0 and duration_sec < self.min_clip_duration_sec

        self._metadata.update(
            {
                "fps": self._fps,
                "width": w,
                "height": h,
                "total_frames": self._total_frames,
                "duration_seconds": round(duration_sec, 2),
                "insufficient_frames": insufficient_frames,
                "file_path": str(self.path.resolve()),
            }
        )

        if insufficient_frames:
            logger.warning(
                "Source %s is shorter (%.2fs) than minimum clip window (%.2fs); marked 'insufficient_frames'.",
                self.source_id,
                duration_sec,
                self.min_clip_duration_sec,
            )

        self._is_opened = True
        self._frame_count = 0
        self._consecutive_bad_frames = 0
        return True

    def read(self) -> Optional[Frame]:
        """Read and return next decoded Frame, handling EOF, looping, and bad frames."""
        if not self._is_opened or self._cap is None:
            return None

        while True:
            ret, raw_image = self._cap.read()

            if not ret or raw_image is None or raw_image.size == 0:
                # Reached end-of-file or encountered bad frame
                current_pos = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Check if this is genuine EOF
                if current_pos >= self._total_frames or current_pos == 0:
                    if self.loop:
                        logger.debug("Source %s reached EOF; looping to start.", self.source_id)
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self._consecutive_bad_frames = 0
                        continue
                    logger.info("Source %s reached EOF.", self.source_id)
                    return None

                # Mid-stream corruption: skip and count
                self._consecutive_bad_frames += 1
                logger.warning(
                    "Bad or corrupted frame %d on source %s (Consecutive: %d/%d).",
                    current_pos,
                    self.source_id,
                    self._consecutive_bad_frames,
                    self._max_consecutive_bad_frames,
                )

                if self._consecutive_bad_frames >= self._max_consecutive_bad_frames:
                    logger.error(
                        "Source %s exceeded max consecutive bad frames (%d). Aborting.",
                        self.source_id,
                        self._max_consecutive_bad_frames,
                    )
                    self.close()
                    raise SourceReadError(
                        f"Stream corrupted beyond recovery: {self._consecutive_bad_frames} consecutive read failures."
                    )
                continue

            # Successfully decoded a frame; reset corruption counter
            self._consecutive_bad_frames = 0
            processed_image = self._preprocess_frame(raw_image)

            # Monotonic time calculated from video frame index / FPS
            source_pts = float(self._frame_count) / self._fps
            ts_utc = self._start_wall_time + timedelta(seconds=source_pts)

            frame = Frame(
                image=processed_image,
                ts_utc=ts_utc,
                source_ts=source_pts,
                frame_idx=self._frame_count,
                source_id=self.source_id,
                metadata={
                    "is_downscaled": self._metadata["is_downscaled"],
                    "insufficient_frames": self._metadata["insufficient_frames"],
                },
            )

            self._frame_count += 1
            self._last_valid_frame = frame
            return frame

    def close(self) -> None:
        """Release video capture handle."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                logger.warning("Error releasing capture for %s: %s", self.source_id, e)
            self._cap = None
        self._is_opened = False
