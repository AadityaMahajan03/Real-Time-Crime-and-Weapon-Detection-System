"""Unit tests for configuration loading, validation, and secret masking."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.fasttrack.core.config import AppConfig
from src.fasttrack.core.types import CameraConfig, CameraSourceType


def test_default_config_initialization() -> None:
    config = AppConfig()
    assert config.app.name == "FAST-TRACK AI"
    assert config.weapon_detector.input_size == 640
    assert config.violence_classifier.model_architecture == "s3d"
    assert config.event_engine.incident_merge_gap_seconds == 10.0


def test_camera_masked_source_ref() -> None:
    # RTSP URL with credentials must be masked
    cam_rtsp = CameraConfig(
        id="c1",
        name="Test RTSP",
        source_type=CameraSourceType.RTSP,
        source_ref="rtsp://operator:secretPassword123@192.168.1.50:554/stream1",
    )
    assert "secretPassword123" not in cam_rtsp.masked_source_ref
    assert cam_rtsp.masked_source_ref == "rtsp://***@192.168.1.50:554/stream1"

    # Webcam or file path is untouched
    cam_webcam = CameraConfig(
        id="c2",
        name="Test Webcam",
        source_type=CameraSourceType.WEBCAM,
        source_ref="0",
    )
    assert cam_webcam.masked_source_ref == "0"


def test_config_env_override(monkeypatch: pytest.MonkeyPatch, temp_dir: Path) -> None:
    monkeypatch.setenv("FASTTRACK_ENV", "production")
    monkeypatch.setenv("FASTTRACK_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FASTTRACK_DATA_DIR", str(temp_dir))

    cfg = AppConfig.load()
    assert cfg.app.environment == "production"
    assert cfg.logging.level == "DEBUG"
    assert str(temp_dir) in cfg.storage.db_path
