"""Shared pytest test fixtures for FastTrack AI."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import pytest

from src.fasttrack.core.clock import SimulatedClock
from src.fasttrack.core.config import AppConfig
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner


@pytest.fixture
def sim_clock() -> SimulatedClock:
    """Fixture providing a deterministic controllable clock."""
    start_time = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    return SimulatedClock(start_utc=start_time, start_monotonic=1000.0)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Fixture providing a temporary directory cleaned up after test."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        yield Path(td)


@pytest.fixture
def in_memory_db() -> Generator[DatabaseManager, None, None]:
    """Fixture providing an in-memory database with migrations applied."""
    db = DatabaseManager(":memory:")
    runner = MigrationRunner(db, migrations_dir="src/fasttrack/storage/migrations")
    runner.run_migrations()
    yield db
    db.close()


@pytest.fixture
def test_config(temp_dir: Path) -> AppConfig:
    """Fixture providing custom AppConfig pointing to temp storage."""
    cfg = AppConfig()
    cfg.storage.db_path = str(temp_dir / "test.db")
    cfg.storage.evidence_dir = str(temp_dir / "evidence")
    cfg.storage.logs_dir = str(temp_dir / "logs")
    return cfg
