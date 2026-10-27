"""Unit tests verifying repository abstractions and V002 schema migrations."""

from pathlib import Path

import pytest

from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner
from src.fasttrack.storage.repositories import (
    AuditRepository,
    CameraRepository,
    ClipRepository,
    IncidentRepository,
    RecordingRepository,
)


@pytest.fixture
def test_db(tmp_path: Path) -> DatabaseManager:
    db_file = tmp_path / "test_fasttrack.db"
    db = DatabaseManager(str(db_file))
    runner = MigrationRunner(db, migrations_dir="src/fasttrack/storage/migrations")
    runner.run_migrations()
    return db


def test_camera_repository(test_db: DatabaseManager):
    repo = CameraRepository(test_db)
    # 1. Create camera
    cam = repo.create(
        camera_id="cam-test-01",
        name="Test Corridor",
        source_type="file",
        source_ref_masked="data/test.mp4",
        enabled=True,
    )
    assert cam["id"] == "cam-test-01"
    assert cam["enabled"] == 1

    # 2. Toggle enabled
    repo.update_enabled("cam-test-01", False)
    cam_updated = repo.get_by_id("cam-test-01")
    assert cam_updated["enabled"] == 0

    # 3. Enable all / subset
    repo.set_all_enabled(True)
    assert repo.get_by_id("cam-test-01")["enabled"] == 1


def test_incident_and_clip_repository(test_db: DatabaseManager):
    cam_repo = CameraRepository(test_db)
    cam_repo.create("cam-test-02", "Gate Alpha", "webcam", "0", enabled=True)

    inc_repo = IncidentRepository(test_db)
    clip_repo = ClipRepository(test_db)

    # 1. Create incident
    inc = inc_repo.create(
        incident_id="inc-test-01",
        camera_id="cam-test-02",
        severity="HIGH",
        start_utc="2026-10-25T01:00:00Z",
        peak_weapon_score=0.92,
        peak_violence_score=0.45,
        summary="Weapon detected near gate",
    )
    assert inc["id"] == "inc-test-01"

    # 2. Create and link clip
    clip = clip_repo.create(
        clip_id="clip-test-01",
        incident_id="inc-test-01",
        camera_id="cam-test-02",
        raw_clip_path="/data/clips/raw.mp4",
        annotated_clip_path="/data/clips/ann.mp4",
        thumbnail_path="/data/clips/thumb.jpg",
        sha256="abc12345",
        duration_sec=12.5,
        size_bytes=1048576,
    )
    assert clip["id"] == "clip-test-01"

    # 3. Retrieve incident with nested clips
    detail = inc_repo.get_by_id("inc-test-01")
    assert detail is not None
    assert len(detail["clips"]) == 1
    assert detail["clips"][0]["sha256"] == "abc12345"

    # 4. Check real KPIs calculation
    kpis = inc_repo.get_real_kpis()
    assert kpis["total_incidents"] == 1
    assert kpis["open_incidents"] == 1


def test_recording_repository_and_locking(test_db: DatabaseManager):
    cam_repo = CameraRepository(test_db)
    cam_repo.create("cam-test-03", "Bay East", "rtsp", "rtsp://localhost:554/live")

    rec_repo = RecordingRepository(test_db)

    # Create segment
    seg = rec_repo.create_segment(
        segment_id="seg-001",
        camera_id="cam-test-03",
        segment_path="/data/recordings/seg_001.mp4",
        start_utc="2026-10-25T01:00:00Z",
    )
    assert seg["status"] == "recording"

    # Close segment
    rec_repo.close_segment("seg-001", "2026-10-25T01:01:00Z", 2048576, "sha-seg-1")
    updated = rec_repo.get_by_id("seg-001")
    assert updated["status"] == "completed"
    assert updated["is_locked"] == 0

    # Lock segment for incident
    rec_repo.lock_segments_for_incident("cam-test-03", "2026-10-25T01:00:30Z", "2026-10-25T01:00:45Z")
    locked = rec_repo.get_by_id("seg-001")
    assert locked["is_locked"] == 1


def test_audit_hash_chain(test_db: DatabaseManager):
    audit_repo = AuditRepository(test_db)

    # Append entries
    e1 = audit_repo.log_action("Operator_Lead", "CONFIRM_THREAT", "incident", "inc-01", "Threat confirmed")
    e2 = audit_repo.log_action("Reviewer_Patel", "ESCALATE", "incident", "inc-01", "Police dispatched")

    assert e2["prev_hash"] == e1["entry_hash"]

    # Verify chain
    valid, msg = audit_repo.verify_audit_chain()
    assert valid is True
    assert "Verified 2 entries" in msg
