"""Unit tests verifying the Acoustic Alarm Service state machine, hysteresis, and acknowledgment."""

import time
from pathlib import Path

import pytest

from src.fasttrack.services.alarm import AlarmPattern, AlarmService, AlarmState
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner
from src.fasttrack.storage.repositories import CameraRepository


@pytest.fixture
def alarm_service(tmp_path: Path):
    db_file = tmp_path / "alarm_test.db"
    db = DatabaseManager(str(db_file))
    runner = MigrationRunner(db, migrations_dir="src/fasttrack/storage/migrations")
    runner.run_migrations()

    cam_repo = CameraRepository(db)
    cam_repo.create("cam-siren-01", "Vault Room", "file", "0")

    # Fast cooldown (0.3s) and escalation (1.0s) for rapid automated testing
    service = AlarmService(
        db_manager=db,
        high_threshold=0.60,
        low_threshold=0.35,
        cooldown_sec=0.3,
        escalation_timeout_sec=1.0,
        server_audio_enabled=False,  # Mute physical speaker during unit test
    )
    yield service
    service.stop()


def test_alarm_trigger_and_hysteresis(alarm_service: AlarmService):
    # 1. Initial State: IDLE
    assert alarm_service.state == AlarmState.IDLE

    # 2. Score below high threshold (0.45 < 0.60): does not trigger
    alarm_service.update_threat("cam-siren-01", weapon_score=0.45, violence_score=0.10)
    assert alarm_service.state == AlarmState.IDLE

    # 3. High score (0.75 >= 0.60): triggers weapon alarm
    alarm_service.update_threat("cam-siren-01", weapon_score=0.75, violence_score=0.20)
    assert alarm_service.state == AlarmState.ACTIVE
    assert alarm_service.pattern == AlarmPattern.CONTINUOUS_WEAPON

    # 4. Hysteresis score drop to 0.40 (above low_threshold 0.35): sustains alarm!
    alarm_service.update_threat("cam-siren-01", weapon_score=0.40, violence_score=0.10)
    assert alarm_service.state == AlarmState.ACTIVE

    # 5. Threat completely cleared: waits cooldown then stops
    time.sleep(0.4)
    assert alarm_service.state == AlarmState.IDLE


def test_alarm_operator_acknowledgment(alarm_service: AlarmService):
    alarm_service.update_threat("cam-siren-01", weapon_score=0.85, violence_score=0.0)
    assert alarm_service.state == AlarmState.ACTIVE

    # Operator acknowledges
    alarm_service.acknowledge(operator_id="Operator_Lead", notes="False positive confirmed")
    assert alarm_service.state == AlarmState.ACKNOWLEDGED

    # Database verification
    recent = alarm_service.alarm_repo.get_recent(limit=5)
    assert len(recent) >= 1
    assert recent[0]["acknowledged"] == 1
    assert recent[0]["ack_by"] == "Operator_Lead"
