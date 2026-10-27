"""Multi-threaded pipeline workers for asynchronous capture, inference, and event dispatch."""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import (
    CameraConfig,
    Detection,
    Frame,
    StreamStatus,
)
from src.fasttrack.ml.tracker import WeaponTracker
from src.fasttrack.ml.violence_detector import ViolenceClassifier
from src.fasttrack.ml.weapon_detector import WeaponDetector
from src.fasttrack.pipeline.telemetry import TelemetryCollector
from src.fasttrack.sources.base import VideoSource

if TYPE_CHECKING:
    from src.fasttrack.events.engine import EventEngine

logger = get_logger("pipeline.workers")

DEFAULT_QUEUE_MAXSIZE = 10


class CaptureWorker(threading.Thread):
    """Worker thread ingesting video frames from a VideoSource into a bounded FIFO queue."""

    def __init__(
        self,
        camera_id: str,
        source: VideoSource,
        frame_queue: queue.Queue[Frame],
        telemetry: TelemetryCollector,
        clock: Optional[Clock] = None,
    ) -> None:
        super().__init__(name=f"CaptureWorker-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.source = source
        self.frame_queue = frame_queue
        self.telemetry = telemetry
        self.clock = clock or SystemClock()

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def stop(self) -> None:
        """Signal worker to terminate execution."""
        self._stop_event.set()

    def pause(self) -> None:
        """Temporarily pause frame ingestion."""
        self._pause_event.set()

    def resume(self) -> None:
        """Resume frame ingestion after pause."""
        self._pause_event.clear()

    def run(self) -> None:
        """Execution loop reading frames and enqueuing with drop-oldest backpressure."""
        logger.info("CaptureWorker started for camera %s", self.camera_id)
        if not self.source.is_alive:
            try:
                self.source.open()
            except Exception as e:
                logger.error("Failed to open video source for %s: %s", self.camera_id, e)

        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            try:
                frame = self.source.read()
            except Exception as e:
                logger.error("Error reading frame on %s: %s", self.camera_id, e)
                time.sleep(0.1)
                continue

            if frame is None:
                # End of file or disconnected stream awaiting reconnect
                time.sleep(0.02)
                continue

            self.telemetry.record_frame_ingested(self.camera_id)

            # Enqueue with bounded backpressure (LIFO / drop oldest on saturation)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    # Drop oldest unread frame to ensure real-time latency
                    _ = self.frame_queue.get_nowait()
                    self.telemetry.record_frame_dropped(self.camera_id)
                except queue.Empty:
                    pass
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    self.telemetry.record_frame_dropped(self.camera_id)

        logger.info("CaptureWorker stopped for camera %s", self.camera_id)


class ProcessingWorker(threading.Thread):
    """Worker thread running weapon detection, tracking, violence inference, and event dispatch."""

    def __init__(
        self,
        camera_id: str,
        frame_queue: queue.Queue[Frame],
        telemetry: TelemetryCollector,
        tracker: WeaponTracker,
        weapon_detector: Optional[WeaponDetector] = None,
        violence_classifier: Optional[ViolenceClassifier] = None,
        event_engine: Optional[EventEngine] = None,
        model_lock: Optional[threading.Lock] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        super().__init__(name=f"ProcessingWorker-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.frame_queue = frame_queue
        self.telemetry = telemetry
        self.tracker = tracker
        self.weapon_detector = weapon_detector
        self.violence_classifier = violence_classifier
        self.event_engine = event_engine
        self.model_lock = model_lock or threading.Lock()
        self.clock = clock or SystemClock()

        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_detections: List[Detection] = []
        self._latest_violence_score: float = 0.0
        self._frame_lock = threading.Lock()

    def stop(self) -> None:
        """Signal worker to terminate execution."""
        self._stop_event.set()

    def pause(self) -> None:
        """Pause processing."""
        self._pause_event.set()

    def resume(self) -> None:
        """Resume processing."""
        self._pause_event.clear()

    def get_latest_display_frame(self) -> Optional[np.ndarray]:
        """Return the most recently processed frame."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_detections(self) -> List[Detection]:
        """Return recent detections for UI preview."""
        with self._frame_lock:
            return list(self._latest_detections)

    def run(self) -> None:
        """Processing loop: dequeue, inference, tracking, triage, event generation."""
        logger.info("ProcessingWorker started for camera %s", self.camera_id)
        while not self._stop_event.is_set():
            if self._pause_event.is_set():
                time.sleep(0.05)
                continue

            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            t_start = time.perf_counter()

            # 1. Weapon Detection & Tracking
            weapon_dets: List[Detection] = []
            if self.weapon_detector is not None:
                try:
                    with self.model_lock:
                        raw_dets = self.weapon_detector.detect(frame)
                    # Update K-of-N confirmation tracker
                    confirmed_dets, _ = self.tracker.update(raw_dets)
                    weapon_dets = confirmed_dets
                except Exception as e:
                    logger.error("Weapon detection error on %s: %s", self.camera_id, e)

            # 2. Spatiotemporal Violence Classification
            violence_prob: Optional[float] = None
            if self.violence_classifier is not None:
                try:
                    person_boxes = getattr(self.weapon_detector, "last_detected_persons", None)
                    with self.model_lock:
                        pred = self.violence_classifier.push_frame(frame, person_boxes=person_boxes)
                    if pred is not None and pred.is_buffer_ready:
                        violence_prob = pred.calibrated_score
                        self._latest_violence_score = pred.calibrated_score
                except Exception as e:
                    logger.error("Violence classification error on %s: %s", self.camera_id, e)

            # 3. Central Event Engine processing
            if self.event_engine is not None:
                try:
                    self.event_engine.process_frame(
                        frame=frame,
                        weapon_detections=weapon_dets,
                        violence_prob=violence_prob,
                    )
                except Exception as e:
                    logger.error("EventEngine processing error on %s: %s", self.camera_id, e)

            # Record telemetry latency
            t_latency = (time.perf_counter() - t_start) * 1000.0
            self.telemetry.record_frame_processed(self.camera_id, t_latency)

            # Cache latest display frame
            with self._frame_lock:
                self._latest_frame = frame.image
                self._latest_detections = weapon_dets

        logger.info("ProcessingWorker stopped for camera %s", self.camera_id)


class CameraStream:
    """Encapsulates the end-to-end streaming pipeline for a single camera."""

    def __init__(
        self,
        config: CameraConfig,
        source: VideoSource,
        telemetry: TelemetryCollector,
        weapon_detector: Optional[WeaponDetector] = None,
        violence_classifier: Optional[ViolenceClassifier] = None,
        event_engine: Optional[EventEngine] = None,
        model_lock: Optional[threading.Lock] = None,
        queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE,
        clock: Optional[Clock] = None,
    ) -> None:
        self.config = config
        self.camera_id = config.id
        self.source = source
        self.telemetry = telemetry
        self.weapon_detector = weapon_detector
        self.violence_classifier = violence_classifier
        self.event_engine = event_engine
        self.model_lock = model_lock or threading.Lock()
        self.clock = clock or SystemClock()

        self.frame_queue: queue.Queue[Frame] = queue.Queue(maxsize=queue_maxsize)
        self.tracker = WeaponTracker(k=3, n=5, clock=self.clock)

        self.status = StreamStatus.STOPPED
        self._error_msg: Optional[str] = None

        self.capture_worker = CaptureWorker(
            camera_id=self.camera_id,
            source=self.source,
            frame_queue=self.frame_queue,
            telemetry=self.telemetry,
            clock=self.clock,
        )

        self.processing_worker = ProcessingWorker(
            camera_id=self.camera_id,
            frame_queue=self.frame_queue,
            telemetry=self.telemetry,
            tracker=self.tracker,
            weapon_detector=self.weapon_detector,
            violence_classifier=self.violence_classifier,
            event_engine=self.event_engine,
            model_lock=self.model_lock,
            clock=self.clock,
        )

    def start(self) -> bool:
        """Start both capture and processing workers."""
        if self.status in (StreamStatus.RUNNING, StreamStatus.STARTING):
            return True

        self.status = StreamStatus.STARTING
        try:
            self.capture_worker.start()
            self.processing_worker.start()
            self.status = StreamStatus.RUNNING
            self._error_msg = None
            logger.info("CameraStream for %s started successfully.", self.camera_id)
            return True
        except Exception as e:
            self.status = StreamStatus.ERROR
            self._error_msg = str(e)
            logger.error("Failed to start CameraStream for %s: %s", self.camera_id, e)
            return False

    def pause(self) -> None:
        """Pause capture and processing."""
        if self.status == StreamStatus.RUNNING:
            self.capture_worker.pause()
            self.processing_worker.pause()
            self.status = StreamStatus.PAUSED

    def resume(self) -> None:
        """Resume capture and processing from pause."""
        if self.status == StreamStatus.PAUSED:
            self.capture_worker.resume()
            self.processing_worker.resume()
            self.status = StreamStatus.RUNNING

    def stop(self, timeout: float = 2.0) -> None:
        """Signal workers to stop and release source resources."""
        self.status = StreamStatus.STOPPED
        self.capture_worker.stop()
        self.processing_worker.stop()

        if self.capture_worker.is_alive():
            self.capture_worker.join(timeout=timeout)
        if self.processing_worker.is_alive():
            self.processing_worker.join(timeout=timeout)

        try:
            self.source.close()
        except Exception as e:
            logger.warning("Error closing source for %s: %s", self.camera_id, e)

        logger.info("CameraStream for %s stopped.", self.camera_id)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Fetch latest processed video frame for preview."""
        return self.processing_worker.get_latest_display_frame()

    def get_status_info(self) -> Dict[str, Any]:
        """Return status descriptor and live telemetry metrics."""
        metrics = self.telemetry.get_metrics(self.camera_id)
        return {
            "camera_id": self.camera_id,
            "name": self.config.name,
            "status": self.status.value,
            "error": self._error_msg,
            "queue_depth": self.frame_queue.qsize(),
            **metrics,
        }
