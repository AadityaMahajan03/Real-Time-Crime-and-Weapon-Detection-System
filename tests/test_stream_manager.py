"""Unit tests verifying StreamManager multi-camera fleet controls and DB synchronization."""

from pathlib import Path

import pytest

from src.fasttrack.core.types import StreamStatus
from src.fasttrack.pipeline.stream_manager import StreamManager
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner
from src.fasttrack.storage.repositories import CameraRepository


@pytest.fixture
def manager_setup(tmp_path: Path):
    db_file = tmp_path / "stream_test.db"
    db = DatabaseManager(str(db_file))
    runner = MigrationRunner(db, migrations_dir="src/fasttrack/storage/migrations")
    runner.run_migrations()

    cam_repo = CameraRepository(db)
    # Register 3 cameras: 2 enabled, 1 disabled
    cam_repo.create("cam-fleet-01", "North Gate", "file", "0", enabled=True)
    cam_repo.create("cam-fleet-02", "South Corridor", "file", "0", enabled=True)
    cam_repo.create("cam-fleet-03", "East Loading Dock", "file", "0", enabled=False)

    sm = StreamManager(db_manager=db)
    yield sm, cam_repo
    sm.shutdown()


def test_sync_active_cameras_from_db(manager_setup):
    sm, cam_repo = manager_setup

    # Synchronize from DB
    res = sm.sync_from_db()

    # cam-fleet-01 and cam-fleet-02 should be started
    assert "cam-fleet-01" in res["started"]
    assert "cam-fleet-02" in res["started"]
    assert "cam-fleet-03" not in res["started"]

    s1 = sm.get_stream("cam-fleet-01")
    s2 = sm.get_stream("cam-fleet-02")
    assert s1 is not None and s1.status == StreamStatus.RUNNING
    assert s2 is not None and s2.status == StreamStatus.RUNNING


def test_bulk_camera_state_activation(manager_setup):
    sm, cam_repo = manager_setup

    # 1. Deactivate All
    sm.set_all_cameras_state(False)
    all_cams = cam_repo.get_all()
    assert all(c["enabled"] == 0 for c in all_cams)

    # 2. Activate All
    sm.set_all_cameras_state(True)
    all_cams = cam_repo.get_all()
    assert all(c["enabled"] == 1 for c in all_cams)

    # 3. Set Subset Active (only cam-fleet-03)
    sm.set_active_subset(["cam-fleet-03"])
    c1 = cam_repo.get_by_id("cam-fleet-01")
    c3 = cam_repo.get_by_id("cam-fleet-03")
    assert c1["enabled"] == 0
    assert c3["enabled"] == 1
