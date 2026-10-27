"""Unit and integration test suite for FastTrack Event Engine and Evidence Management."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.fasttrack.core.clock import SimulatedClock
from src.fasttrack.core.types import (
    BoundingBox,
    CameraConfig,
    CameraSourceType,
    Detection,
    Event,
    EventType,
    Frame,
    IncidentSeverity,
    Zone,
    ZoneType,
)
from src.fasttrack.events.dedup import AlertStormLimiter, ViolenceHysteresis
from src.fasttrack.events.engine import EventEngine
from src.fasttrack.events.incident import IncidentAggregate, IncidentLifecycle
from src.fasttrack.events.severity import calculate_severity, get_severity_rationale
from src.fasttrack.events.zones import ZoneFilter
from src.fasttrack.pipeline.ring_buffer import FrameRingBuffer
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.evidence import EvidenceWriter
from src.fasttrack.storage.migrations import MigrationRunner
from src.fasttrack.storage.retention import cleanup_expired_evidence


def make_dummy_frame(width: int = 640, height: int = 480, source_ts: float = 0.0) -> Frame:
    """Helper to create dummy synthetic frame."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(
        image=img,
        ts_utc=datetime.now(timezone.utc),
        source_ts=source_ts,
        frame_idx=1,
        source_id="cam-test-1",
    )


# -------------------------------------------------------------------------
# 1. ZoneFilter Tests
# -------------------------------------------------------------------------
def test_zone_filter_include_and_ignore():
    """Verify include zone containment and ignore zone exclusion."""
    # Define include zone covering left half [0, 0] to [0.5, 1.0]
    include_zone = Zone(
        name="left_hallway",
        zone_type=ZoneType.INCLUDE,
        polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
    )
    # Define ignore zone covering bottom-left corner [0, 0.7] to [0.2, 1.0] (e.g. poster/desk)
    ignore_zone = Zone(
        name="desk_poster",
        zone_type=ZoneType.IGNORE,
        polygon=[(0.0, 0.7), (0.2, 0.7), (0.2, 1.0), (0.0, 1.0)],
    )

    zf = ZoneFilter(zones=[include_zone, ignore_zone])

    w, h = 1000, 1000

    # Detection 1: Center is at x=250, y=250 (norm: 0.25, 0.25) -> Inside include, not in ignore -> Allowed
    det1 = Detection(
        box=BoundingBox(200, 200, 300, 300),
        class_name="knife",
        confidence=0.85,
    )
    assert zf.is_detection_allowed(det1, w, h) is True

    # Detection 2: Center is at x=750, y=250 (norm: 0.75, 0.25) -> Outside include -> Disallowed
    det2 = Detection(
        box=BoundingBox(700, 200, 800, 300),
        class_name="knife",
        confidence=0.85,
    )
    assert zf.is_detection_allowed(det2, w, h) is False

    # Detection 3: Center is at x=100, y=850 (norm: 0.10, 0.85) -> Inside ignore zone -> Disallowed
    det3 = Detection(
        box=BoundingBox(50, 800, 150, 900),
        class_name="knife",
        confidence=0.85,
    )
    assert zf.is_detection_allowed(det3, w, h) is False

    # Filter detections list
    filtered = zf.filter_detections([det1, det2, det3], w, h)
    assert len(filtered) == 1
    assert filtered[0].class_name == "knife"


def test_zone_filter_no_zones():
    """Verify default behavior when no zones configured: all regions allowed."""
    zf = ZoneFilter(zones=[])
    det = Detection(box=BoundingBox(10, 10, 50, 50), class_name="gun", confidence=0.9)
    assert zf.is_detection_allowed(det, 640, 480) is True


# -------------------------------------------------------------------------
# 2. AlertStormLimiter Tests
# -------------------------------------------------------------------------
def test_alert_storm_limiter_rate_limiting():
    """Verify camera and global rate limits with sliding window time expiry."""
    clock = SimulatedClock()
    limiter = AlertStormLimiter(
        camera_limit_per_min=3,
        global_limit_per_min=5,
        window_seconds=60.0,
        clock=clock,
    )

    cam1 = "cam-01"
    cam2 = "cam-02"

    # Send 3 alerts on cam1 -> all allowed
    assert limiter.should_allow_alert(cam1) is True
    assert limiter.should_allow_alert(cam1) is True
    assert limiter.should_allow_alert(cam1) is True

    # 4th alert on cam1 -> suppressed by camera limit
    assert limiter.should_allow_alert(cam1) is False
    assert limiter.get_suppressed_count(cam1) == 1

    # cam2 has sent 0 alerts so far, global total is 3. Send 2 alerts on cam2 -> allowed
    assert limiter.should_allow_alert(cam2) is True
    assert limiter.should_allow_alert(cam2) is True

    # Global total is now 5 (3 from cam1, 2 from cam2). 3rd alert on cam2 -> suppressed by global limit
    assert limiter.should_allow_alert(cam2) is False

    # Advance clock by 61 seconds (past window)
    clock.advance(61.0)

    # Now alerts on both cameras should be permitted again
    assert limiter.should_allow_alert(cam1) is True
    assert limiter.should_allow_alert(cam2) is True


# -------------------------------------------------------------------------
# 3. ViolenceHysteresis Tests
# -------------------------------------------------------------------------
def test_violence_hysteresis_state_transitions():
    """Verify dual-threshold hysteresis with consecutive window confirmation and cooldown."""
    clock = SimulatedClock()
    vh = ViolenceHysteresis(
        t_on=0.65,
        t_off=0.40,
        consecutive_windows=2,
        cooldown_seconds=5.0,
        clock=clock,
    )

    assert vh.is_active is False

    # Single high score -> not active yet (requires 2 consecutive)
    assert vh.update(0.70) is False
    assert vh.is_active is False

    # Intermediate dip -> resets consecutive counter
    assert vh.update(0.50) is False
    assert vh.is_active is False

    # Window 1 high
    assert vh.update(0.72) is False
    # Window 2 high -> TRIGGERS active
    assert vh.update(0.68) is True
    assert vh.is_active is True

    # Advance clock past cooldown
    clock.advance(6.0)

    # Intermediate score (0.50) is above T_off (0.40) -> remains active
    assert vh.update(0.50) is True
    assert vh.is_active is True

    # Score drops below T_off (0.35 < 0.40) -> releases active state
    assert vh.update(0.35) is False
    assert vh.is_active is False


# -------------------------------------------------------------------------
# 4. Severity & IncidentAggregate Tests
# -------------------------------------------------------------------------
def test_severity_calculation():
    """Verify rule-based severity triage without score summing."""
    assert calculate_severity(has_confirmed_weapon=True, has_confirmed_violence=True) == IncidentSeverity.HIGH
    assert calculate_severity(has_confirmed_weapon=True, has_confirmed_violence=False) == IncidentSeverity.MEDIUM
    assert calculate_severity(has_confirmed_weapon=False, has_confirmed_violence=True) == IncidentSeverity.MEDIUM
    assert calculate_severity(has_confirmed_weapon=False, has_confirmed_violence=False, has_weak_signal=True) == IncidentSeverity.LOW
    assert calculate_severity(has_confirmed_weapon=False, has_confirmed_violence=False, has_weak_signal=False) == IncidentSeverity.INFO

    rationale = get_severity_rationale(IncidentSeverity.HIGH)
    assert "Co-occurring" in rationale


def test_incident_aggregate_lifecycle():
    """Verify IncidentAggregate state transitions: OPEN -> ACTIVE -> CLOSING -> CLOSED."""
    start_utc = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)
    inc = IncidentAggregate.create(
        camera_id="cam-north",
        start_mono=100.0,
        start_utc=start_utc,
        merge_gap_sec=10.0,
    )

    assert inc.lifecycle == IncidentLifecycle.OPEN
    assert inc.severity == IncidentSeverity.INFO

    # Add weapon event at mono 102.0
    evt1 = Event(
        id="evt-1",
        incident_id=None,
        ts_utc=start_utc,
        event_type=EventType.WEAPON_DETECTION,
        class_name="knife",
        raw_score=0.80,
        calibrated_score=0.80,
    )
    inc.add_event(evt1, mono_ts=102.0)
    assert inc.lifecycle == IncidentLifecycle.ACTIVE
    assert inc.severity == IncidentSeverity.MEDIUM
    assert inc.peak_weapon_score == 0.80

    # Add violence event at mono 103.0 -> severity escalates to HIGH
    evt2 = Event(
        id="evt-2",
        incident_id=None,
        ts_utc=start_utc,
        event_type=EventType.POSSIBLE_VIOLENCE,
        class_name="violence",
        raw_score=0.75,
        calibrated_score=0.75,
    )
    inc.add_event(evt2, mono_ts=103.0)
    assert inc.severity == IncidentSeverity.HIGH
    assert inc.peak_violence_score == 0.75

    # Tick at mono 107.0 (elapsed 4.0s < 5.0s half-gap) -> remains ACTIVE
    state = inc.tick_lifecycle(now_mono=107.0, now_utc=start_utc)
    assert state == IncidentLifecycle.ACTIVE

    # Tick at mono 109.0 (elapsed 6.0s >= 5.0s half-gap) -> transitions to CLOSING
    state = inc.tick_lifecycle(now_mono=109.0, now_utc=start_utc)
    assert state == IncidentLifecycle.CLOSING

    # Tick at mono 114.0 (elapsed 11.0s >= 10.0s merge gap) -> transitions to CLOSED
    state = inc.tick_lifecycle(now_mono=114.0, now_utc=start_utc)
    assert state == IncidentLifecycle.CLOSED
    assert inc.end_utc is not None

    # Conversion to domain entity
    record = inc.to_incident_record()
    assert record.id == inc.id
    assert len(record.events) == 2


# -------------------------------------------------------------------------
# 5. FrameRingBuffer Tests
# -------------------------------------------------------------------------
def test_frame_ring_buffer_compression_and_retrieval():
    """Verify JPEG compression, bounded eviction, and frame decoding in ring buffer."""
    buf = FrameRingBuffer(capacity_seconds=2.0, jpeg_quality=75)

    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    now = datetime.now(timezone.utc)

    # Ingest 3 frames at timestamps 0.0, 1.0, 2.5
    f1 = Frame(image=img, ts_utc=now, source_ts=0.0, frame_idx=0, source_id="cam")
    f2 = Frame(image=img, ts_utc=now, source_ts=1.0, frame_idx=1, source_id="cam")
    f3 = Frame(image=img, ts_utc=now, source_ts=2.5, frame_idx=2, source_id="cam")

    buf.append(f1)
    buf.append(f2)
    # At source_ts=2.5, f1 (source_ts=0.0) is older than 2.5 - 2.0 = 0.5s -> evicted
    buf.append(f3)

    all_frames = buf.get_all_frames()
    assert len(all_frames) == 2
    assert all_frames[0][0] == 1.0
    assert all_frames[1][0] == 2.5

    # Check decoded image shape
    assert all_frames[0][2].shape == (100, 100, 3)

    latest = buf.get_latest_frame()
    assert latest is not None
    assert latest.shape == (100, 100, 3)

    buf.clear()
    assert buf.get_latest_frame() is None


# -------------------------------------------------------------------------
# 6. EvidenceWriter & Retention Tests
# -------------------------------------------------------------------------
def test_evidence_writer_atomic_and_integrity():
    """Verify snapshot and clip writing with atomic persistence and SHA-256 integrity."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = str(tmp_path / "test.db")
        ev_dir = str(tmp_path / "evidence")

        db_mgr = DatabaseManager(db_path=db_path)
        MigrationRunner(db_mgr).run_migrations()

        writer = EvidenceWriter(db_manager=db_mgr, base_dir=ev_dir)

        # 1. Snapshot test
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        det = Detection(
            box=BoundingBox(50, 50, 150, 150),
            class_name="knife",
            confidence=0.88,
        )

        snap = writer.write_snapshot(
            frame=dummy_img,
            camera_id="cam-gate",
            camera_name="Main Gate",
            detections=[det],
            privacy_blur=True,
        )

        assert snap.id.startswith("ev-snap-")
        assert Path(snap.path).exists()
        assert snap.size_bytes > 0
        assert len(snap.sha256) == 64

        # Verify integrity in DB
        ok, msg = writer.verify_evidence_integrity(snap.id)
        assert ok is True
        assert "Integrity verified" in msg

        # Tamper check: alter file contents
        with open(snap.path, "ab") as f:
            f.write(b"corrupt")
        ok_tampered, msg_tampered = writer.verify_evidence_integrity(snap.id)
        assert ok_tampered is False
        assert "failure" in msg_tampered

        # 2. Clip test
        clip_frames = [dummy_img.copy() for _ in range(5)]
        clip = writer.write_clip(
            frames=clip_frames,
            camera_id="cam-gate",
            fps=10.0,
        )
        assert clip is not None
        assert clip.kind == "clip"
        assert Path(clip.path).exists()
        assert clip.size_bytes > 0

        ok_clip, _ = writer.verify_evidence_integrity(clip.id)
        assert ok_clip is True

        # 3. Disk space degradation test
        writer.min_free_disk_bytes = 100 * 1024 * 1024 * 1024 * 1024  # 100 TB (force fail)
        assert writer.is_clip_writing_permitted() is False
        clip_degraded = writer.write_clip(clip_frames, "cam-gate")
        assert clip_degraded is None  # Degraded to snapshot-only mode

        db_mgr.close()


def test_retention_cleanup_job():
    """Verify automated evidence retention purging of expired artifacts."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = str(tmp_path / "test_retention.db")
        ev_dir = str(tmp_path / "evidence")

        db_mgr = DatabaseManager(db_path=db_path)
        MigrationRunner(db_mgr).run_migrations()

        clock = SimulatedClock()
        writer = EvidenceWriter(db_manager=db_mgr, base_dir=ev_dir, clock=clock)

        # Write snapshot with past expiration date
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        past_expiry = clock.now_utc() - timedelta(days=1)
        snap = writer.write_snapshot(
            frame=img,
            camera_id="cam-1",
            expires_utc=past_expiry,
        )

        assert Path(snap.path).exists()

        # Seed camera and incident, and link an event to verify audit logging
        db_mgr.execute_non_query(
            "INSERT INTO cameras (id, name, source_type, source_ref_masked, created_utc) VALUES (?, ?, ?, ?, ?)",
            ("cam-1", "Test Camera 1", "file", "test.mp4", clock.now_utc().isoformat()),
        )
        inc_id = "inc-test-ret"
        db_mgr.execute_non_query(
            "INSERT INTO incidents (id, camera_id, severity, start_utc) VALUES (?, ?, ?, ?)",
            (inc_id, "cam-1", "MEDIUM", clock.now_utc().isoformat()),
        )
        db_mgr.execute_non_query(
            """
            INSERT INTO events (id, incident_id, camera_id, ts_utc, event_type, class_name, raw_score, calibrated_score, evidence_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("evt-ret-1", inc_id, "cam-1", clock.now_utc().isoformat(), "weapon_detection", "knife", 0.8, 0.8, snap.id),
        )

        # Create an orphaned .tmp file to test orphan cleanup
        tmp_file = Path(ev_dir) / "snapshots" / "orphan.tmp"
        tmp_file.write_bytes(b"temp_data")
        assert tmp_file.exists()

        # Run retention cleanup
        stats = cleanup_expired_evidence(
            db_manager=db_mgr,
            evidence_dir=ev_dir,
            clock=clock,
        )

        assert stats["total_purged"] == 1
        assert stats["bytes_freed"] > 0
        assert stats["tmp_files_cleaned"] == 1
        assert not Path(snap.path).exists()
        assert not tmp_file.exists()

        # Verify DB row deleted
        ev_rows = db_mgr.execute_query("SELECT id FROM evidence WHERE id = ?", (snap.id,))
        assert len(ev_rows) == 0

        # Verify audit review action inserted
        audit_rows = db_mgr.execute_query(
            "SELECT action, user_id FROM review_actions WHERE incident_id = ?",
            (inc_id,),
        )
        assert len(audit_rows) == 1
        assert audit_rows[0]["action"] == "evidence_purged"

        db_mgr.close()


# -------------------------------------------------------------------------
# 7. EventEngine Integration Tests
# -------------------------------------------------------------------------
def test_event_engine_full_lifecycle():
    """Verify EventEngine end-to-end: detection ingestion, triage, evidence snapshot, and clip generation."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = str(tmp_path / "test_engine.db")
        ev_dir = str(tmp_path / "evidence")

        db_mgr = DatabaseManager(db_path=db_path)
        MigrationRunner(db_mgr).run_migrations()

        clock = SimulatedClock()
        ev_writer = EvidenceWriter(db_manager=db_mgr, base_dir=ev_dir, clock=clock)

        engine = EventEngine(
            db_manager=db_mgr,
            evidence_writer=ev_writer,
            merge_gap_seconds=10.0,
            clock=clock,
        )

        cam_id = "cam-entrance"
        cfg = CameraConfig(
            id=cam_id,
            name="Entrance Gate",
            source_type=CameraSourceType.FILE,
            source_ref="test.mp4",
        )
        engine.register_camera(cfg)

        # Frame 1: Frame without detections
        f1 = make_dummy_frame(source_ts=clock.monotonic())
        f1.source_id = cam_id
        evts, inc = engine.process_frame(f1, weapon_detections=[], violence_prob=0.1)
        assert len(evts) == 0
        assert inc is None

        # Frame 2: Weapon detected
        clock.advance(1.0)
        f2 = make_dummy_frame(source_ts=clock.monotonic())
        f2.source_id = cam_id
        weapon_det = Detection(
            box=BoundingBox(100, 100, 200, 200),
            class_name="knife",
            confidence=0.85,
            track_id=1,
            source_id=cam_id,
        )
        evts2, inc2 = engine.process_frame(f2, weapon_detections=[weapon_det], violence_prob=0.2)
        assert len(evts2) == 1
        assert evts2[0].event_type == EventType.WEAPON_DETECTION
        assert inc2 is not None
        assert inc2.severity == IncidentSeverity.MEDIUM
        assert evts2[0].evidence_id is not None  # Evidence snapshot generated

        # Frame 3: Consecutive high violence scores (hysteresis requires 2 consecutive)
        clock.advance(1.0)
        f3 = make_dummy_frame(source_ts=clock.monotonic())
        f3.source_id = cam_id
        # First violence spike (not triggered yet)
        evts3, inc3 = engine.process_frame(f3, weapon_detections=[], violence_prob=0.75)
        assert len(evts3) == 0

        clock.advance(1.0)
        f4 = make_dummy_frame(source_ts=clock.monotonic())
        f4.source_id = cam_id
        # Second violence spike (triggers violence)
        evts4, inc4 = engine.process_frame(f4, weapon_detections=[], violence_prob=0.78)
        assert len(evts4) == 1
        assert evts4[0].event_type == EventType.POSSIBLE_VIOLENCE
        assert inc4.severity == IncidentSeverity.HIGH  # High because weapon + violence co-occur

        # Check DB records
        inc_rows = db_mgr.execute_query("SELECT id, severity FROM incidents WHERE id = ?", (inc4.id,))
        assert len(inc_rows) == 1
        assert inc_rows[0]["severity"] == "HIGH"

        evt_rows = db_mgr.execute_query("SELECT id, event_type FROM events WHERE incident_id = ?", (inc4.id,))
        assert len(evt_rows) == 2

        # Fast forward time beyond merge gap (10 seconds)
        clock.advance(12.0)
        closed = engine.tick_active_incidents()
        assert len(closed) == 1
        assert closed[0].id == inc4.id

        # Verify clip was generated for the closed incident
        clip_rows = db_mgr.execute_query("SELECT id, kind, path FROM evidence WHERE kind = 'clip'")
        assert len(clip_rows) >= 1
        assert Path(clip_rows[0]["path"]).exists()

        # Verify incident marked closed in DB
        updated_inc = db_mgr.execute_query("SELECT end_utc FROM incidents WHERE id = ?", (inc4.id,))
        assert updated_inc[0]["end_utc"] is not None

        db_mgr.close()
