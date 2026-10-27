"""Continuous 60-second segmented rolling camera recorder."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from src.fasttrack.core.logging import get_logger
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.repositories import RecordingRepository
from src.fasttrack.storage.video_encoder import FFmpegVideoEncoder

logger = get_logger("services.recorder")


class CameraSegmentRecorder:
    """Records continuous 60-second H.264 video segments for a single camera feed."""

    def __init__(
        self,
        camera_id: str,
        db_manager: DatabaseManager,
        segment_duration_sec: int = 60,
        fps: float = 25.0,
        recordings_root: str = "data/recordings",
    ) -> None:
        self.camera_id = camera_id
        self.db_manager = db_manager
        self.repo = RecordingRepository(db_manager)
        self.segment_duration_sec = segment_duration_sec
        self.fps = max(1.0, float(fps))
        self.encoder = FFmpegVideoEncoder()

        self.storage_dir = Path(recordings_root) / camera_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._current_segment_id: Optional[str] = None
        self._current_start_utc: Optional[str] = None
        self._current_frames: List[np.ndarray] = []
        self._threads: List[threading.Thread] = []
        self._lock = threading.Lock()
        self._is_running = True

    def push_frame(self, frame_bgr: np.ndarray) -> None:
        """Push frame into current rolling segment; rotates every segment_duration_sec."""
        with self._lock:
            if not self._is_running:
                return

            now_utc = datetime.now(timezone.utc).isoformat()

            # Start new segment if needed
            if not self._current_segment_id:
                self._current_segment_id = f"seg-{self.camera_id}-{uuid.uuid4().hex[:10]}"
                self._current_start_utc = now_utc
                self._current_frames = []

            self._current_frames.append(frame_bgr.copy())

            # Check if duration limit reached
            frame_limit = int(self.segment_duration_sec * self.fps)
            if len(self._current_frames) >= frame_limit:
                self._flush_and_rotate(now_utc)

    def _flush_and_rotate(self, end_utc: str) -> None:
        """Finalize current segment on disk & in database, then prepare next segment."""
        if not self._current_segment_id or not self._current_frames:
            return

        seg_id = self._current_segment_id
        start_utc = self._current_start_utc or end_utc
        frames_to_save = self._current_frames

        # Reset pointers immediately for continuous capture
        self._current_segment_id = f"seg-{self.camera_id}-{uuid.uuid4().hex[:10]}"
        self._current_start_utc = end_utc
        self._current_frames = []

        # Background thread to encode segment without blocking frame ingestion
        def _encode_worker():
            try:
                date_str = start_utc[:10]
                day_dir = self.storage_dir / date_str
                day_dir.mkdir(parents=True, exist_ok=True)
                seg_path = day_dir / f"{seg_id}.mp4"

                # Initial database entry
                self.repo.create_segment(
                    segment_id=seg_id,
                    camera_id=self.camera_id,
                    segment_path=str(seg_path).replace("\\", "/"),
                    start_utc=start_utc,
                    duration_sec=self.segment_duration_sec,
                )

                # Encode H.264
                meta = self.encoder.encode_h264_clip(
                    frames=frames_to_save,
                    output_path=seg_path,
                    fps=self.fps,
                )

                # Update database record
                self.repo.close_segment(
                    segment_id=seg_id,
                    end_utc=end_utc,
                    size_bytes=meta["size_bytes"],
                    sha256=meta["sha256"],
                    status="completed",
                )
                logger.debug("Segment %s finalized (%d MB, %s)", seg_id, meta["size_bytes"] // (1024*1024), meta["sha256"][:8])

            except Exception as e:
                logger.error("Failed to encode segment %s: %s", seg_id, e)
                try:
                    self.repo.close_segment(seg_id, end_utc, 0, "", status="corrupt")
                except Exception:
                    pass

        t = threading.Thread(target=_encode_worker, name=f"SegEnc_{seg_id}", daemon=True)
        t.start()
        self._threads.append(t)

    def close(self, wait: bool = True) -> None:
        """Flush and close currently recording segment cleanly."""
        with self._lock:
            self._is_running = False
            if self._current_segment_id and self._current_frames:
                now_utc = datetime.now(timezone.utc).isoformat()
                self._flush_and_rotate(now_utc)

        if wait:
            for t in list(self._threads):
                t.join(timeout=10.0)
