"""Unit tests for logging configuration and credential masking."""

from __future__ import annotations

import logging
from pathlib import Path

from src.fasttrack.core.logging import _mask_secrets, get_logger, setup_logging


def test_secret_masking_in_rtsp_urls() -> None:
    raw_msg = "Connecting to rtsp://admin:superSecretPass99@10.0.0.5:554/live on Cam-01"
    sanitized = _mask_secrets(raw_msg)
    assert "superSecretPass99" not in sanitized
    assert "admin" not in sanitized
    assert "rtsp://***:***@10.0.0.5:554/live" in sanitized


def test_logger_setup_creates_file(temp_dir: Path) -> None:
    log_file = temp_dir / "test_run.log"
    # Ensure root fasttrack logger handlers are reset
    root_logger = logging.getLogger("fasttrack")
    root_logger.handlers.clear()

    logger = setup_logging(level="DEBUG", log_file=str(log_file))
    logger.info("Test event for fasttrack system: rtsp://user:pass@camera.local:554/feed")

    # Flush handlers
    for h in list(logger.handlers):
        h.flush()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test event for fasttrack system" in content
    assert "pass@" not in content
    assert "rtsp://***:***@camera.local" in content

    # Clean up file handlers so Windows file locks are released
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)


def test_get_logger_namespacing() -> None:
    child_logger = get_logger("detector.weapon")
    assert child_logger.name == "fasttrack.detector.weapon"
