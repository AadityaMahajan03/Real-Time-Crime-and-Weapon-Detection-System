"""StreamManager: Central multi-camera lifecycle coordinator and telemetry manager."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import (
    CameraConfig,
    HealthSample,
    StreamStatus,
)
from src.fasttrack.ml.violence_detector import ViolenceClassifier
from src.fasttrack.ml.weapon_detector import WeaponDetector
from src.fasttrack.pipeline.telemetry import TelemetryCollector
from src.fasttrack.pipeline.workers import CameraStream
from src.fasttrack.sources import create_video_source
from src.fasttrack.sources.base import VideoSource
from src.fasttrack.storage.db import DatabaseManager

if TYPE_CHECKING:
    from src.fasttrack.events.engine import EventEngine

logger = get_logger("pipeline.stream_manager")


class StreamManager:
    """Manages the lifecycle, execution, and health monitoring of all camera video streams."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        event_engine: Optional[EventEngine] = None,
        telemetry: Optional[TelemetryCollector] = None,
        weapon_detector: Optional[WeaponDetector] = None,
        telemetry_interval_sec: float = 5.0,
        clock: Optional[Clock] = None,
    ) -> None:
        self.db_manager = db_manager
        self.clock = clock or SystemClock()
        self.telemetry = telemetry or TelemetryCollector(db_manager=self.db_manager, clock=self.clock)
        if event_engine is not None:
            self.event_engine = event_engine
        else:
            from src.fasttrack.events.engine import EventEngine

            self.event_engine = EventEngine(db_manager=self.db_manager, clock=self.clock)
        self.weapon_detector = weapon_detector
        self.telemetry_interval_sec = telemetry_interval_sec

        self._model_lock = threading.Lock()
        self._streams: Dict[str, CameraStream] = {}
        self._streams_lock = threading.Lock()

        # Telemetry background sampler thread
        self._telemetry_stop_event = threading.Event()
        self._telemetry_thread = threading.Thread(
            target=self._telemetry_loop, name="TelemetrySampler", daemon=True
        )
        self._telemetry_thread.start()

    def add_camera(
        self,
        config: CameraConfig,
        source: Optional[VideoSource] = None,
    ) -> CameraStream:
        """Register and configure a camera stream, initializing dedicated classifiers and workers."""
        cid = config.id
        with self._streams_lock:
            if cid in self._streams:
                logger.warning("Camera %s already registered; stopping existing stream.", cid)
                self._streams[cid].stop()

            # Create video source if not provided
            cam_source = source
            if cam_source is None:
                cam_source = create_video_source(camera=config)

            # Register camera in EventEngine
            self.event_engine.register_camera(config)

            # Per-camera violence classifier instance with isolated rolling buffer
            violence_classifier = ViolenceClassifier()

            stream = CameraStream(
                config=config,
                source=cam_source,
                telemetry=self.telemetry,
                weapon_detector=self.weapon_detector,
                violence_classifier=violence_classifier,
                event_engine=self.event_engine,
                model_lock=self._model_lock,
                clock=self.clock,
            )

            self._streams[cid] = stream
            logger.info("Added camera %s (%s) to StreamManager.", cid, config.name)
            return stream

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and unregister a camera stream."""
        with self._streams_lock:
            if camera_id in self._streams:
                stream = self._streams.pop(camera_id)
                stream.stop()
                logger.info("Removed camera %s from StreamManager.", camera_id)
                return True
            return False

    def start_camera(self, camera_id: str) -> bool:
        """Start streaming pipeline for a specific camera."""
        with self._streams_lock:
            stream = self._streams.get(camera_id)
            if stream is None:
                logger.error("Camera %s not found in StreamManager.", camera_id)
                return False
            return stream.start()

    def stop_camera(self, camera_id: str) -> None:
        """Stop streaming pipeline for a specific camera."""
        with self._streams_lock:
            stream = self._streams.get(camera_id)
            if stream is not None:
                stream.stop()

    def pause_camera(self, camera_id: str) -> None:
        """Temporarily pause a camera stream."""
        with self._streams_lock:
            stream = self._streams.get(camera_id)
            if stream is not None:
                stream.pause()

    def resume_camera(self, camera_id: str) -> None:
        """Resume a paused camera stream."""
        with self._streams_lock:
            stream = self._streams.get(camera_id)
            if stream is not None:
                stream.resume()

    def start_all(self) -> None:
        """Start all registered camera streaming pipelines."""
        with self._streams_lock:
            for cid, stream in self._streams.items():
                if stream.status != StreamStatus.RUNNING:
                    stream.start()

    def stop_all(self) -> None:
        """Stop all running camera streaming pipelines."""
        with self._streams_lock:
            for cid, stream in self._streams.items():
                stream.stop()

    def get_stream(self, camera_id: str) -> Optional[CameraStream]:
        """Fetch CameraStream instance by ID."""
        with self._streams_lock:
            return self._streams.get(camera_id)

    def get_latest_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Fetch latest processed video frame for UI live view."""
        with self._streams_lock:
            stream = self._streams.get(camera_id)
            if stream is not None:
                return stream.get_latest_frame()
            return None

    def sync_from_db(self) -> Dict[str, Any]:
        """Synchronize running camera workers with the database's enabled status."""
        if not self.db_manager:
            return {}

        from src.fasttrack.storage.repositories import CameraRepository
        repo = CameraRepository(self.db_manager)
        db_cameras = repo.get_all()

        results = {"started": [], "stopped": []}

        for c in db_cameras:
            cid = c["id"]
            is_enabled = bool(c["enabled"]) if "enabled" in c.keys() else True

            with self._streams_lock:
                stream_exists = cid in self._streams
                stream_running = stream_exists and self._streams[cid].status == StreamStatus.RUNNING

            if is_enabled:
                if not stream_exists:
                    from src.fasttrack.core.types import CameraSourceType
                    try:
                        src_t = CameraSourceType(c["source_type"].lower())
                    except Exception:
                        src_t = CameraSourceType.FILE

                    cfg = CameraConfig(
                        id=cid,
                        name=c["name"],
                        source_type=src_t,
                        source_ref=c["source_ref_masked"],
                        enabled=True,
                    )
                    self.add_camera(cfg)
                    self.start_camera(cid)
                    results["started"].append(cid)
                elif not stream_running:
                    self.start_camera(cid)
                    results["started"].append(cid)
            else:
                if stream_running:
                    self.stop_camera(cid)
                    results["stopped"].append(cid)

        return results

    def set_camera_state(self, camera_id: str, enabled: bool) -> None:
        """Persist camera state to DB and synchronize worker."""
        if self.db_manager:
            from src.fasttrack.storage.repositories import CameraRepository
            repo = CameraRepository(self.db_manager)
            repo.update_enabled(camera_id, enabled)
        self.sync_from_db()

    def set_all_cameras_state(self, enabled: bool) -> None:
        """Bulk enable/disable all cameras in DB and synchronize workers."""
        if self.db_manager:
            from src.fasttrack.storage.repositories import CameraRepository
            repo = CameraRepository(self.db_manager)
            repo.set_all_enabled(enabled)
        self.sync_from_db()

    def set_active_subset(self, active_camera_ids: List[str]) -> None:
        """Bulk activate a selected subset of cameras and deactivate all others."""
        if self.db_manager:
            from src.fasttrack.storage.repositories import CameraRepository
            repo = CameraRepository(self.db_manager)
            repo.set_subset_enabled(active_camera_ids)
        self.sync_from_db()

    def get_all_status_info(self) -> Dict[str, Dict[str, Any]]:
        """Fetch live status and telemetry metrics for all registered cameras."""
        with self._streams_lock:
            return {cid: stream.get_status_info() for cid, stream in self._streams.items()}

    def sample_and_persist_telemetry(self) -> List[HealthSample]:
        """Sample current telemetry metrics across cameras and persist to the database."""
        samples: List[HealthSample] = []
        with self._streams_lock:
            active_ids = list(self._streams.keys())

        for cid in active_ids:
            sample = self.telemetry.emit_sample(cid)
            self.telemetry.persist_sample(sample)
            samples.append(sample)

        return samples

    def _telemetry_loop(self) -> None:
        """Background thread periodically recording health samples for active cameras."""
        while not self._telemetry_stop_event.is_set():
            time.sleep(self.telemetry_interval_sec)
            try:
                self.sample_and_persist_telemetry()
            except Exception as e:
                logger.error("Telemetry sampling error: %s", e)

    def shutdown(self) -> None:
        """Cleanly terminate all camera pipelines, sampler threads, and resources."""
        logger.info("Shutting down StreamManager...")
        self._telemetry_stop_event.set()
        self.stop_all()
        logger.info("StreamManager shutdown complete.")
