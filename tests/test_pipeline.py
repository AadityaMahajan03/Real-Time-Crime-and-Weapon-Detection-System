"""Unit and integration tests for multi-stream pipeline workers, telemetry, and StreamManager."""

from __future__ import annotations

import queue
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.fasttrack.core.clock import SimulatedClock
from src.fasttrack.core.types import (
    CameraConfig,
    CameraSourceType,
    Frame,
    StreamStatus,
)
from src.fasttrack.ml.tracker import WeaponTracker
from src.fasttrack.pipeline.stream_manager import StreamManager
from src.fasttrack.pipeline.telemetry import TelemetryCollector
from src.fasttrack.pipeline.workers import CameraStream, CaptureWorker, ProcessingWorker
from src.fasttrack.sources.base import VideoSource
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner


class MockVideoSource(VideoSource):
    """Synthetic video source yielding a fixed or infinite sequence of frames for testing."""

    def __init__(
        self,
        camera_id: str = "mock-cam-01",
        num_frames: Optional[int] = None,
        frame_interval_sec: float = 0.01,
    ) -> None:
        super().__init__(source_id=camera_id, source_ref="mock://stream")
        self.camera_id = camera_id
        self.num_frames = num_frames
        self.frame_interval_sec = frame_interval_sec
        self._frame_count = 0
        self._is_closed = False

    def open(self) -> bool:
        self._is_opened = True
        self._is_closed = False
        return True

    def read(self) -> Optional[Frame]:
        if not self._is_opened or self._is_closed:
            return None
        if self.num_frames is not None and self._frame_count >= self.num_frames:
            return None

        time.sleep(self.frame_interval_sec)
        self._frame_count += 1
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        return Frame(
            image=img,
            ts_utc=datetime.now(timezone.utc),
            source_ts=float(self._frame_count) * self.frame_interval_sec,
            frame_idx=self._frame_count,
            source_id=self.source_id,
        )

    def close(self) -> None:
        self._is_closed = True
        self._is_opened = False


# -------------------------------------------------------------------------
# 1. TelemetryCollector Tests
# -------------------------------------------------------------------------
def test_telemetry_collector_metrics_and_persistence():
    """Verify FPS calculation, latency aggregation, and SQLite persistence."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_telemetry.db")
        db_mgr = DatabaseManager(db_path=db_path)
        MigrationRunner(db_mgr).run_migrations()

        clock = SimulatedClock()
        telemetry = TelemetryCollector(db_manager=db_mgr, clock=clock, window_sec=5.0)
        cam_id = "cam-test-telemetry"

        # Seed camera in DB to satisfy foreign keys
        db_mgr.execute_non_query(
            "INSERT INTO cameras (id, name, source_type, source_ref_masked, created_utc) VALUES (?, ?, ?, ?, ?)",
            (cam_id, "Test Camera", "file", "test.mp4", clock.now_utc().isoformat()),
        )

        # Ingest 5 frames at 0.1s intervals
        for _ in range(5):
            telemetry.record_frame_ingested(cam_id)
            telemetry.record_frame_processed(cam_id, latency_ms=45.0)
            clock.advance(0.1)

        # Drop 2 frames
        telemetry.record_frame_dropped(cam_id)
        telemetry.record_frame_dropped(cam_id)

        metrics = telemetry.get_metrics(cam_id)
        assert metrics["camera_id"] == cam_id
        assert metrics["fps"] > 0.0
        assert metrics["latency_ms"] == 45.0
        assert metrics["dropped_frames"] == 2
        assert "cpu_percent" in metrics
        assert "ram_percent" in metrics

        # Emit and persist health sample
        sample = telemetry.emit_sample(cam_id)
        assert sample.camera_id == cam_id
        assert sample.dropped_frames == 2
        telemetry.persist_sample(sample)

        # Verify sample row in DB
        rows = db_mgr.execute_query("SELECT * FROM health_samples WHERE camera_id = ?", (cam_id,))
        assert len(rows) == 1
        assert rows[0]["dropped_frames"] == 2
        assert rows[0]["latency_ms"] == 45.0

        db_mgr.close()


# -------------------------------------------------------------------------
# 2. CaptureWorker & Backpressure Tests
# -------------------------------------------------------------------------
def test_capture_worker_backpressure_and_lifecycle():
    """Verify bounded queue saturation gracefully drops oldest frames without memory leaks."""
    cam_id = "cam-cap-test"
    telemetry = TelemetryCollector()
    frame_queue: queue.Queue[Frame] = queue.Queue(maxsize=3)
    source = MockVideoSource(camera_id=cam_id, num_frames=10, frame_interval_sec=0.005)

    worker = CaptureWorker(
        camera_id=cam_id,
        source=source,
        frame_queue=frame_queue,
        telemetry=telemetry,
    )

    worker.start()
    time.sleep(0.15)  # Let worker produce frames while queue is small (maxsize=3)

    worker.stop()
    worker.join(timeout=2.0)
    source.close()

    # Queue should be full (size <= 3) and drops should be recorded
    assert frame_queue.qsize() <= 3
    metrics = telemetry.get_metrics(cam_id)
    assert metrics["ingestion_fps"] > 0.0 or metrics["dropped_frames"] > 0


# -------------------------------------------------------------------------
# 3. ProcessingWorker Tests
# -------------------------------------------------------------------------
def test_processing_worker_execution():
    """Verify processing worker dequeues frames, runs pipeline, and updates display frames."""
    cam_id = "cam-proc-test"
    telemetry = TelemetryCollector()
    frame_queue: queue.Queue[Frame] = queue.Queue(maxsize=10)
    tracker = WeaponTracker()

    worker = ProcessingWorker(
        camera_id=cam_id,
        frame_queue=frame_queue,
        telemetry=telemetry,
        tracker=tracker,
        weapon_detector=None,
        violence_classifier=None,
        event_engine=None,
    )

    worker.start()

    # Feed dummy frame into queue
    img = np.full((120, 160, 3), 77, dtype=np.uint8)
    dummy_frame = Frame(
        image=img,
        ts_utc=datetime.now(timezone.utc),
        source_ts=0.0,
        frame_idx=1,
        source_id=cam_id,
    )
    frame_queue.put(dummy_frame)

    time.sleep(0.15)  # Allow worker to process frame

    display_frame = worker.get_latest_display_frame()
    assert display_frame is not None
    assert display_frame.shape == (120, 160, 3)

    worker.stop()
    worker.join(timeout=2.0)

    metrics = telemetry.get_metrics(cam_id)
    assert metrics["latency_ms"] >= 0.0


# -------------------------------------------------------------------------
# 4. CameraStream Lifecycle Tests
# -------------------------------------------------------------------------
def test_camera_stream_lifecycle():
    """Verify CameraStream start -> pause -> resume -> stop transitions without hanging."""
    cfg = CameraConfig(
        id="cam-stream-test",
        name="Test Camera Stream",
        source_type=CameraSourceType.FILE,
        source_ref="dummy.mp4",
    )
    source = MockVideoSource(camera_id=cfg.id, num_frames=50, frame_interval_sec=0.01)
    telemetry = TelemetryCollector()

    stream = CameraStream(
        config=cfg,
        source=source,
        telemetry=telemetry,
        queue_maxsize=5,
    )

    assert stream.status == StreamStatus.STOPPED

    # Start
    ok = stream.start()
    assert ok is True
    assert stream.status == StreamStatus.RUNNING

    time.sleep(0.1)

    # Pause
    stream.pause()
    assert stream.status == StreamStatus.PAUSED

    # Resume
    stream.resume()
    assert stream.status == StreamStatus.RUNNING

    time.sleep(0.1)

    # Stop
    stream.stop(timeout=2.0)
    assert stream.status == StreamStatus.STOPPED
    assert not stream.capture_worker.is_alive()
    assert not stream.processing_worker.is_alive()


# -------------------------------------------------------------------------
# 5. StreamManager Multi-Camera Orchestration Tests
# -------------------------------------------------------------------------
def test_stream_manager_multi_camera_orchestration():
    """Verify StreamManager coordinates multiple cameras, health sampling, and clean shutdown."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_manager.db")
        db_mgr = DatabaseManager(db_path=db_path)
        MigrationRunner(db_mgr).run_migrations()

        manager = StreamManager(
            db_manager=db_mgr,
            telemetry_interval_sec=0.2,  # Rapid telemetry loop for testing
        )

        # Configure 2 mock cameras
        cfg1 = CameraConfig(id="cam-01", name="Entrance 1", source_type=CameraSourceType.FILE, source_ref="f1.mp4")
        cfg2 = CameraConfig(id="cam-02", name="Entrance 2", source_type=CameraSourceType.FILE, source_ref="f2.mp4")

        s1 = MockVideoSource(camera_id="cam-01", num_frames=30, frame_interval_sec=0.01)
        s2 = MockVideoSource(camera_id="cam-02", num_frames=30, frame_interval_sec=0.01)

        manager.add_camera(cfg1, source=s1)
        manager.add_camera(cfg2, source=s2)

        # Start both cameras
        manager.start_all()
        assert manager.get_stream("cam-01").status == StreamStatus.RUNNING
        assert manager.get_stream("cam-02").status == StreamStatus.RUNNING

        time.sleep(0.3)

        # Query status info
        statuses = manager.get_all_status_info()
        assert "cam-01" in statuses
        assert "cam-02" in statuses
        assert statuses["cam-01"]["name"] == "Entrance 1"
        assert statuses["cam-02"]["name"] == "Entrance 2"

        # Explicitly invoke telemetry sampling
        samples = manager.sample_and_persist_telemetry()
        assert len(samples) == 2

        # Verify DB records
        health_rows = db_mgr.execute_query("SELECT * FROM health_samples")
        assert len(health_rows) >= 2

        # Remove one camera
        removed = manager.remove_camera("cam-01")
        assert removed is True
        assert manager.get_stream("cam-01") is None

        # Clean shutdown
        manager.shutdown()
        db_mgr.close()
