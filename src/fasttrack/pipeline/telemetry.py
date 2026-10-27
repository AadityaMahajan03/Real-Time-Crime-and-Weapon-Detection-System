"""Telemetry and health diagnostics collector for multi-camera video pipelines."""

from __future__ import annotations

import threading
from collections import deque
from typing import Any, Deque, Dict, Optional

import psutil
import torch

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import HealthSample
from src.fasttrack.storage.db import DatabaseManager

logger = get_logger("pipeline.telemetry")


class CameraMetrics:
    """Internal metrics accumulator for an individual camera stream."""

    def __init__(self, window_sec: float = 3.0) -> None:
        self.window_sec = window_sec
        self.ingested_timestamps: Deque[float] = deque()
        self.processed_timestamps: Deque[float] = deque()
        self.latencies_ms: Deque[float] = deque(maxlen=50)
        self.dropped_frames: int = 0
        self.last_latency_ms: float = 0.0

    def prune(self, now_mono: float) -> None:
        cutoff = now_mono - self.window_sec
        while self.ingested_timestamps and self.ingested_timestamps[0] < cutoff:
            self.ingested_timestamps.popleft()
        while self.processed_timestamps and self.processed_timestamps[0] < cutoff:
            self.processed_timestamps.popleft()

    @property
    def fps(self) -> float:
        """Calculate processing FPS over the sliding window."""
        count = len(self.processed_timestamps)
        if count < 2:
            return 0.0
        duration = self.processed_timestamps[-1] - self.processed_timestamps[0]
        return round(float(count - 1) / duration, 1) if duration > 0.05 else 0.0

    @property
    def ingestion_fps(self) -> float:
        """Calculate source ingestion FPS over the sliding window."""
        count = len(self.ingested_timestamps)
        if count < 2:
            return 0.0
        duration = self.ingested_timestamps[-1] - self.ingested_timestamps[0]
        return round(float(count - 1) / duration, 1) if duration > 0.05 else 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Calculate mean pipeline latency over recent frames."""
        if not self.latencies_ms:
            return self.last_latency_ms
        return round(sum(self.latencies_ms) / float(len(self.latencies_ms)), 1)


class TelemetryCollector:
    """Collects real-time performance telemetry and system resource metrics across cameras."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        clock: Optional[Clock] = None,
        window_sec: float = 3.0,
    ) -> None:
        self.db_manager = db_manager
        self.clock = clock or SystemClock()
        self.window_sec = window_sec
        self._lock = threading.Lock()
        self._camera_metrics: Dict[str, CameraMetrics] = {}

    def _get_or_create(self, camera_id: str) -> CameraMetrics:
        if camera_id not in self._camera_metrics:
            self._camera_metrics[camera_id] = CameraMetrics(window_sec=self.window_sec)
        return self._camera_metrics[camera_id]

    def record_frame_ingested(self, camera_id: str) -> None:
        """Record timestamp of a frame received from the video source."""
        now = self.clock.monotonic()
        with self._lock:
            m = self._get_or_create(camera_id)
            m.ingested_timestamps.append(now)
            m.prune(now)

    def record_frame_processed(self, camera_id: str, latency_ms: float) -> None:
        """Record successful completion of an inference/event pass and its latency."""
        now = self.clock.monotonic()
        with self._lock:
            m = self._get_or_create(camera_id)
            m.processed_timestamps.append(now)
            m.latencies_ms.append(float(latency_ms))
            m.last_latency_ms = float(latency_ms)
            m.prune(now)

    def record_frame_dropped(self, camera_id: str) -> None:
        """Increment count of dropped frames due to queue saturation (backpressure)."""
        with self._lock:
            m = self._get_or_create(camera_id)
            m.dropped_frames += 1

    def get_system_resources(self) -> Dict[str, float]:
        """Query host CPU, RAM, and GPU utilization metrics."""
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent

        gpu_util = 0.0
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0)
                total = torch.cuda.get_device_properties(0).total_memory
                gpu_util = round((allocated / total) * 100.0, 1)
            except Exception:
                gpu_util = 0.0

        return {"cpu_percent": cpu, "ram_percent": ram, "gpu_util": gpu_util}

    def emit_sample(self, camera_id: str) -> HealthSample:
        """Generate a HealthSample measuring the current performance snapshot."""
        now_utc = self.clock.now_utc()
        now_mono = self.clock.monotonic()

        with self._lock:
            m = self._get_or_create(camera_id)
            m.prune(now_mono)
            fps = m.fps
            lat = m.avg_latency_ms
            dropped = m.dropped_frames

        sys_metrics = self.get_system_resources()

        return HealthSample(
            ts_utc=now_utc,
            camera_id=camera_id,
            fps=fps,
            latency_ms=lat,
            dropped_frames=dropped,
            gpu_util=sys_metrics["gpu_util"],
            cpu_percent=sys_metrics["cpu_percent"],
            ram_percent=sys_metrics["ram_percent"],
        )

    def persist_sample(self, sample: HealthSample) -> None:
        """Persist a diagnostic health sample to the SQLite database."""
        if not self.db_manager:
            return

        try:
            self.db_manager.execute_non_query(
                """
                INSERT INTO health_samples (ts_utc, camera_id, fps, latency_ms, dropped_frames, gpu_util, cpu_percent, ram_percent)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sample.ts_utc.isoformat(),
                    sample.camera_id,
                    sample.fps,
                    sample.latency_ms,
                    sample.dropped_frames,
                    sample.gpu_util,
                    sample.cpu_percent,
                    sample.ram_percent,
                ),
            )
        except Exception as e:
            logger.warning("Failed to persist health sample for camera %s: %s", sample.camera_id, e)

    def get_metrics(self, camera_id: str) -> Dict[str, Any]:
        """Fetch current live metrics dictionary for UI display."""
        now_mono = self.clock.monotonic()
        with self._lock:
            m = self._get_or_create(camera_id)
            m.prune(now_mono)
            fps = m.fps
            ingest_fps = m.ingestion_fps
            avg_lat = m.avg_latency_ms
            last_lat = m.last_latency_ms
            dropped = m.dropped_frames

        sys_res = self.get_system_resources()
        return {
            "camera_id": camera_id,
            "fps": fps,
            "ingestion_fps": ingest_fps,
            "latency_ms": avg_lat,
            "last_latency_ms": last_lat,
            "dropped_frames": dropped,
            "cpu_percent": sys_res["cpu_percent"],
            "ram_percent": sys_res["ram_percent"],
            "gpu_util": sys_res["gpu_util"],
        }
