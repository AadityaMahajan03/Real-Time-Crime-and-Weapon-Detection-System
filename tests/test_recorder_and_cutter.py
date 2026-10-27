"""Unit tests verifying continuous recording, clip cutting, and locked retention."""

from pathlib import Path

import numpy as np
import pytest

from src.fasttrack.services.clip_cutter import ClipCutterService
from src.fasttrack.services.recorder import CameraSegmentRecorder
from src.fasttrack.services.retention import RetentionService
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner
from src.fasttrack.storage.repositories import (
    CameraRepository,
    ClipRepository,
    IncidentRepository,
    RecordingRepository,
)


@pytest.fixture
def test_setup(tmp_path: Path):
    db_file = tmp_path / "rec_test.db"
    db = DatabaseManager(str(db_file))
    runner = MigrationRunner(db, migrations_dir="src/fasttrack/storage/migrations")
    runner.run_migrations()

    cam_repo = CameraRepository(db)
    cam_repo.create("cam-rec-01", "Sector 7 Hallway", "file", "0")

    return db, tmp_path


def test_segment_recorder_lifecycle(test_setup):
    db, tmp_path = test_setup
    rec_root = tmp_path / "recordings"

    # Recorder configured with 1-second segment duration at 10 fps
    recorder = CameraSegmentRecorder(
        camera_id="cam-rec-01",
        db_manager=db,
        segment_duration_sec=1,
        fps=10.0,
        recordings_root=str(rec_root),
    )

    # Push 15 frames to trigger rotation
    dummy_frame = np.full((120, 160, 3), 45, dtype=np.uint8)
    for _ in range(15):
        recorder.push_frame(dummy_frame)

    recorder.close()

    # Verify segment was registered in database
    rec_repo = RecordingRepository(db)
    segments = rec_repo.get_segments_for_range(
        "cam-rec-01", "2026-01-01T00:00:00Z", "2026-12-31T23:59:59Z"
    )
    assert len(segments) >= 1
    assert segments[0]["camera_id"] == "cam-rec-01"


def test_clip_cutter_and_evidence_extraction(test_setup):
    db, tmp_path = test_setup
    inc_repo = IncidentRepository(db)
    inc_repo.create("inc-cut-01", "cam-rec-01", "HIGH", "2026-10-25T01:00:00Z")

    cutter = ClipCutterService(db_manager=db, clips_root=str(tmp_path / "clips"))

    # Pass fallback frames to simulate immediate extraction
    sample_frames = [np.full((100, 120, 3), i * 10 % 255, dtype=np.uint8) for i in range(25)]

    cutter.request_clip(
        incident_id="inc-cut-01",
        camera_id="cam-rec-01",
        trigger_utc="2026-10-25T01:00:00Z",
        duration_sec=1.0,
        pre_roll_sec=0.5,
        post_roll_sec=0.5,
        fallback_frames=sample_frames,
    )

    # Wait for queue processing
    cutter._queue.join()
    cutter.stop()

    clip_repo = ClipRepository(db)
    clips = clip_repo.get_by_incident_id("inc-cut-01")

    assert len(clips) == 1
    clip = clips[0]
    assert Path(clip["raw_clip_path"]).exists()
    assert Path(clip["annotated_clip_path"]).exists()
    assert Path(clip["thumbnail_path"]).exists()
    assert len(clip["sha256"]) == 64


def test_retention_never_deletes_locked_incident_segments(test_setup):
    db, tmp_path = test_setup
    rec_repo = RecordingRepository(db)

    # 1. Create an old unlocked segment (September 2026)
    seg1_path = tmp_path / "seg_old_unlocked.mp4"
    seg1_path.write_bytes(b"dummy1")
    rec_repo.create_segment("seg-old-1", "cam-rec-01", str(seg1_path), "2026-09-01T00:00:00Z")
    rec_repo.close_segment("seg-old-1", "2026-09-01T00:01:00Z", 6, "hash1", status="completed")

    # 2. Create an old LOCKED segment (linked to an incident on Oct 1)
    seg2_path = tmp_path / "seg_old_locked.mp4"
    seg2_path.write_bytes(b"dummy2")
    rec_repo.create_segment("seg-old-2", "cam-rec-01", str(seg2_path), "2026-10-01T00:00:00Z")
    rec_repo.close_segment("seg-old-2", "2026-10-01T00:01:00Z", 6, "hash2", status="completed")
    rec_repo.lock_segments_for_incident("cam-rec-01", "2026-10-01T00:00:00Z", "2026-10-01T00:01:00Z")

    retention = RetentionService(db_manager=db, max_retention_days=7)
    purged_count = retention.enforce_retention()

    # Assert unlocked segment was purged, but locked segment was PRESERVED
    assert purged_count == 1
    assert not seg1_path.exists()
    assert seg2_path.exists()  # Evidence locked segment preserved!
    assert rec_repo.get_by_id("seg-old-1") is None
    assert rec_repo.get_by_id("seg-old-2") is not None
