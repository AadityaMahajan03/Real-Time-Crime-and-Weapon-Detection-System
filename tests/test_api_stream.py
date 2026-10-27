"""Tests verifying FastAPI HTTP 206 Partial Content video streaming endpoint."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.fasttrack.api.main import app
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner
from src.fasttrack.storage.repositories import CameraRepository, ClipRepository, IncidentRepository
from src.fasttrack.storage.video_encoder import FFmpegVideoEncoder


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    test_db_path = tmp_path / "test_api.db"
    db = DatabaseManager(str(test_db_path))
    runner = MigrationRunner(db, migrations_dir="src/fasttrack/storage/migrations")
    runner.run_migrations()

    # Monkeypatch config db path
    from src.fasttrack.core.config import get_config
    cfg = get_config()
    monkeypatch.setattr(cfg.storage, "db_path", str(test_db_path))

    # Create dummy camera and incident
    cam_repo = CameraRepository(db)
    cam_repo.create("cam-stream-01", "Gate Cam", "webcam", "0")
    inc_repo = IncidentRepository(db)
    inc_repo.create("inc-stream-01", "cam-stream-01", "HIGH", "2026-10-25T01:00:00Z")

    # Generate small H.264 video
    encoder = FFmpegVideoEncoder()
    frames = [np.full((120, 160, 3), 100, dtype=np.uint8) for _ in range(25)]
    clip_file = tmp_path / "stream_clip.mp4"
    meta = encoder.encode_h264_clip(frames, clip_file, fps=25.0)

    # Register in clips table
    clip_repo = ClipRepository(db)
    clip_repo.create(
        clip_id="clip-stream-01",
        incident_id="inc-stream-01",
        camera_id="cam-stream-01",
        raw_clip_path=str(clip_file),
        annotated_clip_path=str(clip_file),
        thumbnail_path=str(clip_file),
        sha256=meta["sha256"],
        duration_sec=meta["duration_sec"],
        size_bytes=meta["size_bytes"],
    )

    return TestClient(app)


def test_stream_clip_full_http_200(client: TestClient):
    resp = client.get("/api/clips/clip-stream-01/stream")
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "video/mp4"
    assert resp.headers["Accept-Ranges"] == "bytes"
    assert len(resp.content) > 0


def test_stream_clip_range_http_206(client: TestClient):
    headers = {"Range": "bytes=0-1023"}
    resp = client.get("/api/clips/clip-stream-01/stream", headers=headers)
    assert resp.status_code == 206
    assert resp.headers["Content-Type"] == "video/mp4"
    assert "Content-Range" in resp.headers
    assert resp.headers["Content-Range"].startswith("bytes 0-1023/")
    assert len(resp.content) == 1024
