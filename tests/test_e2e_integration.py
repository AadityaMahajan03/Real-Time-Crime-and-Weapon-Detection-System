"""End-to-End integration test covering ingestion, inference, triage, evidence, notification, and operator audit."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

from src.fasttrack.analytics.engine import AnalyticsEngine
from src.fasttrack.auth.service import AuthService
from src.fasttrack.core.clock import SimulatedClock
from src.fasttrack.core.config import AppConfig
from src.fasttrack.core.types import (
    BoundingBox,
    Detection,
    Event,
    EventType,
    Frame,
    IncidentSeverity,
    IncidentStatus,
    UserRole,
)
from src.fasttrack.events.dedup import AlertStormLimiter
from src.fasttrack.events.incident import IncidentAggregate
from src.fasttrack.events.severity import calculate_severity
from src.fasttrack.ml.benchmark import BenchmarkService
from src.fasttrack.ml.tracker import WeaponTracker
from src.fasttrack.ml.violence_detector import ViolenceClassifier
from src.fasttrack.ml.weapon_detector import WeaponDetector
from src.fasttrack.notify.dispatcher import NotificationDispatcher
from src.fasttrack.pipeline.ring_buffer import FrameRingBuffer
from src.fasttrack.review.service import ReviewService
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.evidence import EvidenceWriter


def test_e2e_full_system_lifecycle(tmp_path: Path, in_memory_db: DatabaseManager) -> None:
    """Test full system lifecycle across all 9 architectural layers."""
    # 1. System Configuration, Clock & Storage
    sim_clock = SimulatedClock()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    cfg.storage.evidence_dir = str(evidence_dir)

    # Register camera in DB
    cam_id = "cam_e2e_01"
    in_memory_db.execute_non_query(
        """
        INSERT INTO cameras (id, name, source_type, source_ref_masked, enabled, zones_json, thresholds_json, schedule_json, created_utc)
        VALUES (?, ?, ?, ?, 1, '[]', '{}', '{}', ?)
        """,
        (cam_id, "Main Courtyard CCTV", "file", "synthetic_test.mp4", sim_clock.now_utc().isoformat()),
    )

    # 2. Ring Buffer Ingestion & Violence Stream
    ring_buffer = FrameRingBuffer(capacity_seconds=3.0, jpeg_quality=85)
    violence_clf = ViolenceClassifier()
    latest_violence_pred = None

    # Ingest 25 synthetic frames (2.5s duration) into ring buffer and violence classifier
    for i in range(25):
        frame_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add synthetic person and motion artifact
        cv2.rectangle(frame_img, (100, 100), (250, 400), (200, 200, 200), -1)
        sim_clock.advance(0.1)
        frame_obj = Frame(
            image=frame_img,
            ts_utc=sim_clock.now_utc(),
            source_ts=sim_clock.monotonic(),
            frame_idx=i,
            source_id=cam_id,
        )
        ring_buffer.append(frame_obj)
        pred = violence_clf.push_frame(frame_obj)
        if pred is not None:
            latest_violence_pred = pred

    buffered_frames = ring_buffer.get_all_frames()
    assert len(buffered_frames) > 0
    assert latest_violence_pred is not None
    assert 0.0 <= latest_violence_pred.raw_score <= 1.0
    assert 0.0 <= latest_violence_pred.calibrated_score <= 1.0

    # 3. Model Inference Pipeline
    # Weapon Detector & Tracker
    weapon_detector = WeaponDetector()
    first_frame_obj = Frame(
        image=buffered_frames[0][2],
        ts_utc=buffered_frames[0][1],
        source_ts=buffered_frames[0][0],
        frame_idx=buffered_frames[0][3],
        source_id=cam_id,
    )
    sample_det = weapon_detector.detect(first_frame_obj)
    assert isinstance(sample_det, list)
    tracker = WeaponTracker(k=2, n=3, clock=sim_clock)

    raw_detections = [
        Detection(
            box=BoundingBox(120.0, 150.0, 180.0, 240.0),
            class_name="knife",
            confidence=0.88,
            source_id=cam_id,
            timestamp_utc=sim_clock.now_utc(),
        )
    ]
    # Pass through tracker (requires k=2 confirmations)
    confirmed_detections = []
    for _ in range(3):
        conf, _ = tracker.update(raw_detections)
        if conf:
            confirmed_detections = conf

    assert len(confirmed_detections) > 0
    confirmed_weapon = confirmed_detections[0]
    assert confirmed_weapon.class_name == "knife"

    # 4. Event Engine & Severity Triage
    # Test calculate_severity: confirmed weapon + confirmed violence -> HIGH
    sev_high = calculate_severity(
        has_confirmed_weapon=True,
        has_confirmed_violence=True,
    )
    assert sev_high == IncidentSeverity.HIGH

    # Incident Aggregate lifecycle
    inc_agg = IncidentAggregate.create(
        camera_id=cam_id,
        start_mono=sim_clock.monotonic(),
        start_utc=sim_clock.now_utc(),
        merge_gap_sec=2.0,
    )
    event_obj = Event(
        id="ev-e2e-01",
        incident_id=inc_agg.id,
        ts_utc=sim_clock.now_utc(),
        event_type=EventType.WEAPON_DETECTION,
        class_name="knife",
        raw_score=0.88,
        calibrated_score=0.88,
        track_id=confirmed_weapon.track_id,
    )
    inc_agg.add_event(event_obj, mono_ts=sim_clock.monotonic())
    assert inc_agg.lifecycle.value == "ACTIVE"

    # 5. Evidence Generation & Cryptographic Integrity
    evidence_writer = EvidenceWriter(db_manager=in_memory_db, base_dir=str(evidence_dir), clock=sim_clock)
    frames_for_clip = [f[2] for f in buffered_frames]

    clip_item = evidence_writer.write_clip(
        frames=frames_for_clip,
        camera_id=cam_id,
        fps=10.0,
    )
    assert clip_item is not None
    assert os.path.exists(clip_item.path)
    assert len(clip_item.sha256) == 64

    snapshot_item = evidence_writer.write_snapshot(
        frame=frames_for_clip[-1],
        camera_id=cam_id,
        privacy_blur=True,
    )
    assert snapshot_item is not None
    assert os.path.exists(snapshot_item.path)
    assert len(snapshot_item.sha256) == 64

    # Verify cryptographic integrity on disk vs DB hash
    is_valid, _ = evidence_writer.verify_evidence_integrity(clip_item.id)
    assert is_valid is True

    # 6. Database Storage & Persistence
    now_str = sim_clock.now_utc().isoformat()
    in_memory_db.execute_non_query(
        """
        INSERT INTO incidents
        (id, camera_id, severity, status, start_utc, end_utc, peak_weapon_score, peak_violence_score, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            inc_agg.id,
            cam_id,
            IncidentSeverity.HIGH.value,
            IncidentStatus.NEW.value,
            now_str,
            None,
            0.88,
            0.85,
            "Detected knife and possible violence",
        ),
    )

    # Verify query
    inc_row = in_memory_db.execute_query("SELECT * FROM incidents WHERE id = ?", (inc_agg.id,))
    assert len(inc_row) == 1
    assert inc_row[0]["severity"] == "HIGH"
    assert inc_row[0]["status"] == "new"

    # 7. Notification Dispatcher
    dispatcher = NotificationDispatcher(clock=sim_clock)
    inc_entity = inc_agg.to_incident_record()
    inc_entity.severity = IncidentSeverity.HIGH
    dispatch_res = dispatcher.dispatch_incident_alert(inc_entity, camera_name="Main Courtyard CCTV")
    assert isinstance(dispatch_res, dict)

    # Test alert storm limiter
    storm_limiter = AlertStormLimiter(camera_limit_per_min=5, clock=sim_clock)
    assert storm_limiter.should_allow_alert(cam_id) is True

    # 8. Operator Review & Immutable Audit Trail
    auth_svc = AuthService(db_manager=in_memory_db)
    user_created = auth_svc.create_user("operator_john", "P@ssword1234!", role=UserRole.REVIEWER)
    assert user_created is not None
    assert user_created.username == "operator_john"
    auth_user = auth_svc.authenticate("operator_john", "P@ssword1234!")
    assert auth_user is not None

    review_svc = ReviewService(db_manager=in_memory_db, clock=sim_clock)

    # Operator marks reviewed
    review_action = review_svc.mark_reviewed(
        incident_id=inc_agg.id,
        user_id=auth_user.id,
        note="Confirmed knife threat in courtyard. Escalating to field team.",
    )
    assert review_action is not None
    assert review_action.action == "mark_reviewed"

    # Operator escalates
    esc_action = review_svc.escalate(
        incident_id=inc_agg.id,
        user_id=auth_user.id,
        reason="P1-EMERGENCY",
        note="Emergency dispatch authorized by shift supervisor.",
    )
    assert esc_action is not None
    assert esc_action.action == "escalate"

    # Export immutable audit trail
    audit_file = review_svc.export_audit_log(
        output_dir=str(tmp_path / "audit"),
        incident_id=inc_agg.id,
    )
    assert audit_file.exists()
    with open(audit_file, "r", encoding="utf-8") as f:
        audit_payload = json.load(f)
    assert audit_payload["record_count"] == 2
    assert len(audit_payload["sha256"]) == 64
    assert audit_payload["records"][0]["action"] == "escalate"
    assert audit_payload["records"][1]["action"] == "mark_reviewed"

    # 9. Operational Analytics & Model Benchmark Catalog
    analytics = AnalyticsEngine(db_manager=in_memory_db)
    kpis = analytics.get_kpis()
    assert kpis["total_incidents"] == 1
    assert kpis["high_severity_count"] == 1

    benchmark_svc = BenchmarkService(eval_dir="reports/eval", db_manager=in_memory_db)
    synced_models = benchmark_svc.sync_model_versions_to_db()
    assert synced_models == 4
    listed_models = benchmark_svc.list_model_versions()
    assert len(listed_models) == 4
