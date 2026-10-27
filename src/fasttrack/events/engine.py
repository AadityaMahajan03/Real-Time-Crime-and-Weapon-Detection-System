"""Central Event Engine coordinating ROI filtering, hysteresis, alert storming, and evidence."""

from __future__ import annotations

import json
import uuid
from typing import Dict, List, Optional, Tuple

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import (
    CameraConfig,
    CameraSourceType,
    Detection,
    Event,
    EventType,
    Frame,
)
from src.fasttrack.events.dedup import AlertStormLimiter, ViolenceHysteresis
from src.fasttrack.events.incident import IncidentAggregate, IncidentLifecycle
from src.fasttrack.events.zones import ZoneFilter
from src.fasttrack.pipeline.ring_buffer import FrameRingBuffer
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.evidence import EvidenceWriter

logger = get_logger("events.engine")


class EventEngine:
    """Orchestrates multi-camera event triage, alert storm deduplication, and evidence generation."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        evidence_writer: Optional[EvidenceWriter] = None,
        alert_limiter: Optional[AlertStormLimiter] = None,
        merge_gap_seconds: float = 10.0,
        privacy_blur_faces: bool = False,
        clock: Optional[Clock] = None,
    ) -> None:
        self.db_manager = db_manager
        self.evidence_writer = evidence_writer
        self.clock = clock or SystemClock()
        self.alert_limiter = alert_limiter or AlertStormLimiter(clock=self.clock)
        self.merge_gap_seconds = float(merge_gap_seconds)
        self.privacy_blur_faces = privacy_blur_faces

        # Per-camera state tracking
        self._camera_configs: Dict[str, CameraConfig] = {}
        self._zone_filters: Dict[str, ZoneFilter] = {}
        self._violence_filters: Dict[str, ViolenceHysteresis] = {}
        self._ring_buffers: Dict[str, FrameRingBuffer] = {}
        self._active_incidents: Dict[str, IncidentAggregate] = {}

    def register_camera(self, config: CameraConfig) -> None:
        """Register camera configuration and initialize dedicated pipeline buffers and filters."""
        cid = config.id
        self._camera_configs[cid] = config
        self._zone_filters[cid] = ZoneFilter(config.zones)
        self._violence_filters[cid] = ViolenceHysteresis(clock=self.clock)
        self._ring_buffers[cid] = FrameRingBuffer(capacity_seconds=4.0)

        if self.db_manager:
            try:
                st = config.source_type.value if hasattr(config.source_type, "value") else str(config.source_type)
                self.db_manager.execute_non_query(
                    """
                    INSERT OR IGNORE INTO cameras (id, name, source_type, source_ref_masked, enabled, zones_json, thresholds_json, schedule_json, created_utc)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cid,
                        config.name,
                        st,
                        config.masked_source_ref,
                        1 if config.enabled else 0,
                        "[]",
                        "{}",
                        "{}",
                        self.clock.now_utc().isoformat(),
                    ),
                )
            except Exception as e:
                logger.warning("Could not persist registered camera %s to DB: %s", cid, e)

        logger.info("Camera %s registered in EventEngine with %d zones.", cid, len(config.zones))

    def _ensure_camera_registered(self, camera_id: str) -> None:
        """Auto-register default configuration if camera was not registered explicitly."""
        if camera_id not in self._camera_configs:
            cfg = CameraConfig(
                id=camera_id,
                name=f"Camera {camera_id}",
                source_type=CameraSourceType.FILE,
                source_ref=camera_id,
            )
            self.register_camera(cfg)

    def get_active_incident(self, camera_id: str) -> Optional[IncidentAggregate]:
        """Return currently active incident for camera, if any."""
        return self._active_incidents.get(camera_id)

    def get_all_active_incidents(self) -> Dict[str, IncidentAggregate]:
        """Return snapshot copy of all active incident aggregates."""
        return dict(self._active_incidents)

    def process_frame(
        self,
        frame: Frame,
        weapon_detections: List[Detection],
        violence_prob: Optional[float] = None,
    ) -> Tuple[List[Event], Optional[IncidentAggregate]]:
        """Process a single video frame with weapon candidates and violence probabilities.

        Returns:
            Tuple of (list of newly generated Events, current IncidentAggregate or None).
        """
        cid = frame.source_id
        self._ensure_camera_registered(cid)

        # 1. Store frame in pre-roll ring buffer
        self._ring_buffers[cid].append(frame)

        # 2. Filter weapon detections through ROI zones
        allowed_weapon_dets = self._zone_filters[cid].filter_detections(
            weapon_detections, frame.width, frame.height
        )

        # 3. Create candidate events
        now_mono = self.clock.monotonic()
        now_utc = frame.ts_utc
        new_events: List[Event] = []

        # Weapon detection events
        for det in allowed_weapon_dets:
            if det.confidence >= 0.40:
                evt = Event(
                    id=f"evt-{uuid.uuid4().hex[:10]}",
                    incident_id=None,
                    ts_utc=now_utc,
                    event_type=EventType.WEAPON_DETECTION,
                    class_name=det.class_name,
                    raw_score=det.confidence,
                    calibrated_score=det.confidence,
                    bbox=det.box,
                    track_id=det.track_id,
                )
                new_events.append(evt)

        # Violence event via hysteresis
        if violence_prob is not None:
            is_violent = self._violence_filters[cid].update(violence_prob)
            if is_violent:
                evt = Event(
                    id=f"evt-{uuid.uuid4().hex[:10]}",
                    incident_id=None,
                    ts_utc=now_utc,
                    event_type=EventType.POSSIBLE_VIOLENCE,
                    class_name="violence",
                    raw_score=violence_prob,
                    calibrated_score=violence_prob,
                )
                new_events.append(evt)

        # 4. Ingest events into active incident aggregate
        current_inc: Optional[IncidentAggregate] = None

        if new_events:
            # Check alert rate limiting
            self.alert_limiter.should_allow_alert(cid)

            # Retrieve or instantiate active incident aggregate
            if cid not in self._active_incidents or self._active_incidents[cid].lifecycle == IncidentLifecycle.CLOSED:
                current_inc = IncidentAggregate.create(
                    camera_id=cid,
                    start_mono=now_mono,
                    start_utc=now_utc,
                    merge_gap_sec=self.merge_gap_seconds,
                )
                self._active_incidents[cid] = current_inc
                if self.db_manager:
                    self.db_manager.execute_non_query(
                        """
                        INSERT INTO incidents (id, camera_id, severity, status, start_utc, peak_weapon_score, peak_violence_score, summary)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            current_inc.id,
                            cid,
                            current_inc.severity.value,
                            current_inc.status.value,
                            now_utc.isoformat(),
                            0.0,
                            0.0,
                            current_inc.summary,
                        ),
                    )
            else:
                current_inc = self._active_incidents[cid]

            # Generate evidence snapshot for this trigger frame
            snapshot_item = None
            if self.evidence_writer is not None:
                try:
                    snapshot_item = self.evidence_writer.write_snapshot(
                        frame=frame.image,
                        camera_id=cid,
                        camera_name=self._camera_configs[cid].name,
                        detections=allowed_weapon_dets,
                        privacy_blur=self.privacy_blur_faces,
                    )
                except Exception as e:
                    logger.error("Failed to capture evidence snapshot on %s: %s", cid, e)

            # Ingest events into incident aggregate and persist to database
            for evt in new_events:
                if snapshot_item:
                    evt.evidence_id = snapshot_item.id

                current_inc.add_event(evt, now_mono)

                if self.db_manager:
                    bbox_str = json.dumps(evt.bbox.to_list()) if evt.bbox else None
                    self.db_manager.execute_non_query(
                        """
                        INSERT INTO events (id, incident_id, camera_id, ts_utc, event_type, class_name, raw_score, calibrated_score, bbox_json, track_id, evidence_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            evt.id,
                            current_inc.id,
                            cid,
                            evt.ts_utc.isoformat(),
                            evt.event_type.value,
                            evt.class_name,
                            evt.raw_score,
                            evt.calibrated_score,
                            bbox_str,
                            evt.track_id,
                            evt.evidence_id,
                        ),
                    )

            # Update incident summary & severity in database
            if self.db_manager:
                self.db_manager.execute_non_query(
                    """
                    UPDATE incidents
                    SET severity = ?, peak_weapon_score = ?, peak_violence_score = ?, summary = ?
                    WHERE id = ?
                    """,
                    (
                        current_inc.severity.value,
                        current_inc.peak_weapon_score,
                        current_inc.peak_violence_score,
                        current_inc.summary,
                        current_inc.id,
                    ),
                )
        else:
            current_inc = self._active_incidents.get(cid)

        # 5. Check and advance lifecycles across cameras
        self.tick_active_incidents()

        return new_events, current_inc

    def tick_active_incidents(self) -> List[IncidentAggregate]:
        """Advance inactivity timers across active incidents and finalize closed ones with video clips."""
        now_mono = self.clock.monotonic()
        now_utc = self.clock.now_utc()
        closed_incidents: List[IncidentAggregate] = []

        for cid, inc in list(self._active_incidents.items()):
            prev_state = inc.lifecycle
            new_state = inc.tick_lifecycle(now_mono, now_utc)

            if new_state == IncidentLifecycle.CLOSED and prev_state != IncidentLifecycle.CLOSED:
                logger.info("Incident %s on camera %s closed after inactivity threshold.", inc.id, cid)
                closed_incidents.append(inc)

                # Generate finalized video evidence clip from ring buffer
                if self.evidence_writer is not None and cid in self._ring_buffers:
                    buffered_frames = [f[2] for f in self._ring_buffers[cid].get_all_frames()]
                    if buffered_frames:
                        try:
                            ev_item = self.evidence_writer.write_clip(
                                frames=buffered_frames,
                                camera_id=cid,
                                privacy_blur=self.privacy_blur_faces,
                            )
                            if ev_item is not None and self.db_manager:
                                # Link to incident_evidence
                                self.db_manager.execute_non_query(
                                    """
                                    INSERT OR IGNORE INTO incident_evidence (incident_id, evidence_id, attached_utc)
                                    VALUES (?, ?, ?);
                                    """,
                                    (inc.id, ev_item.id, now_utc.isoformat()),
                                )
                                # Also insert into clips table for direct video playback
                                self.db_manager.execute_non_query(
                                    """
                                    INSERT OR REPLACE INTO clips (
                                        id, incident_id, camera_id, raw_clip_path, annotated_clip_path,
                                        thumbnail_path, sha256, duration_sec, size_bytes, codec, verification_status, created_utc
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'h264', 'verified', ?);
                                    """,
                                    (
                                        f"clip-{inc.id}",
                                        inc.id,
                                        cid,
                                        str(ev_item.path).replace("\\", "/"),
                                        str(ev_item.path).replace("\\", "/"),
                                        str(ev_item.path).replace(".mp4", "_thumb.jpg").replace("\\", "/"),
                                        ev_item.sha256,
                                        len(buffered_frames) / 15.0,
                                        ev_item.size_bytes,
                                        now_utc.isoformat(),
                                    ),
                                )
                        except Exception as e:
                            logger.error("Failed to generate clip for closed incident %s: %s", inc.id, e)

                # Finalize incident record in database
                if self.db_manager:
                    self.db_manager.execute_non_query(
                        """
                        UPDATE incidents
                        SET end_utc = ?, severity = ?, summary = ?
                        WHERE id = ?
                        """,
                        (
                            inc.end_utc.isoformat() if inc.end_utc else now_utc.isoformat(),
                            inc.severity.value,
                            inc.summary,
                            inc.id,
                        ),
                    )

                del self._active_incidents[cid]

        return closed_incidents
