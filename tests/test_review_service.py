"""Unit and integration test suite for AuthService, RBAC, and ReviewService."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.fasttrack.auth.service import (
    AuthService,
    can_user_perform,
    hash_password,
    verify_password,
)
from src.fasttrack.core.clock import SimulatedClock
from src.fasttrack.core.errors import FastTrackError
from src.fasttrack.core.types import IncidentSeverity, IncidentStatus, UserRole
from src.fasttrack.review.service import ReviewService
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner


# -------------------------------------------------------------------------
# 1. AuthService & RBAC Tests
# -------------------------------------------------------------------------
def test_password_hashing_and_verification():
    """Verify Argon2id password hashing and constant-time verification."""
    pwd = "SecureSurveillancePassword123!"
    hashed = hash_password(pwd)

    assert hashed.startswith("$argon2")
    assert verify_password(pwd, hashed) is True
    assert verify_password("WrongPassword", hashed) is False
    assert verify_password("", hashed) is False

    with pytest.raises(ValueError):
        hash_password("")


def test_auth_service_crud_and_authentication():
    """Verify user registration, login, and disabled user rejection."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_auth.db")
        db_mgr = DatabaseManager(db_path=db_path)
        MigrationRunner(db_mgr).run_migrations()

        clock = SimulatedClock()
        auth = AuthService(db_manager=db_mgr, clock=clock)

        # Register reviewer
        user = auth.create_user("operator_bob", "Password456!", role=UserRole.REVIEWER)
        assert user.username == "operator_bob"
        assert user.role == UserRole.REVIEWER

        # Duplicate username rejection
        with pytest.raises(FastTrackError):
            auth.create_user("operator_bob", "OtherPass")

        # Successful authentication
        auth_user = auth.authenticate("operator_bob", "Password456!")
        assert auth_user is not None
        assert auth_user.id == user.id

        # Failed authentication: wrong password
        assert auth.authenticate("operator_bob", "BadPassword") is None

        # Failed authentication: unknown user
        assert auth.authenticate("unknown_user", "AnyPassword") is None

        # Disabled user authentication rejection
        db_mgr.execute_non_query("UPDATE users SET active = 0 WHERE username = ?", ("operator_bob",))
        assert auth.authenticate("operator_bob", "Password456!") is None

        db_mgr.close()


def test_rbac_permission_matrix():
    """Verify role-based action authorization boundaries."""
    # Viewer: read-only
    assert can_user_perform(UserRole.VIEWER, "view_incidents") is True
    assert can_user_perform(UserRole.VIEWER, "view_forensics") is True
    assert can_user_perform(UserRole.VIEWER, "mark_reviewed") is False
    assert can_user_perform(UserRole.VIEWER, "manage_cameras") is False

    # Reviewer: triage operations
    assert can_user_perform(UserRole.REVIEWER, "mark_reviewed") is True
    assert can_user_perform(UserRole.REVIEWER, "mark_false_alarm") is True
    assert can_user_perform(UserRole.REVIEWER, "escalate") is True
    assert can_user_perform(UserRole.REVIEWER, "add_note") is True
    assert can_user_perform(UserRole.REVIEWER, "manage_cameras") is False

    # Admin: full access
    assert can_user_perform(UserRole.ADMIN, "mark_reviewed") is True
    assert can_user_perform(UserRole.ADMIN, "manage_cameras") is True
    assert can_user_perform(UserRole.ADMIN, "manage_users") is True
    assert can_user_perform(UserRole.ADMIN, "export_audit") is True


# -------------------------------------------------------------------------
# 2. ReviewService Workflow Tests
# -------------------------------------------------------------------------
def test_review_service_incident_workflow():
    """Verify incident state transitions, mandatory rationale validation, and audit recording."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        db_path = str(Path(tmp_dir) / "test_review.db")
        db_mgr = DatabaseManager(db_path=db_path)
        MigrationRunner(db_mgr).run_migrations()

        clock = SimulatedClock()
        service = ReviewService(db_manager=db_mgr, clock=clock)

        # Seed camera and initial incident
        cam_id = "cam-rev-01"
        inc_id = "inc-rev-101"
        now_iso = clock.now_utc().isoformat()

        db_mgr.execute_non_query(
            "INSERT INTO cameras (id, name, source_type, source_ref_masked, created_utc) VALUES (?, ?, ?, ?, ?)",
            (cam_id, "Lobby Camera", "file", "lobby.mp4", now_iso),
        )
        db_mgr.execute_non_query(
            """
            INSERT INTO incidents (id, camera_id, severity, status, start_utc, peak_weapon_score, peak_violence_score, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (inc_id, cam_id, IncidentSeverity.MEDIUM.value, IncidentStatus.NEW.value, now_iso, 0.75, 0.0, "Detected knife"),
        )

        # 1. Action: mark_reviewed
        act1 = service.mark_reviewed(inc_id, user_id="operator_alice", note="Confirmed single knife candidate.")
        assert act1.action == "mark_reviewed"
        assert act1.incident_id == inc_id

        detail = service.get_incident_detail(inc_id)
        assert detail is not None
        assert detail["status"] == IncidentStatus.REVIEWED.value
        assert len(detail["review_actions"]) == 1

        # 2. Action: add_note (does not alter status)
        act2 = service.add_note(inc_id, user_id="operator_alice", note="Notified on-duty floor security.")
        assert act2.action == "add_note"
        detail2 = service.get_incident_detail(inc_id)
        assert detail2["status"] == IncidentStatus.REVIEWED.value
        assert len(detail2["review_actions"]) == 2

        # 3. Action: escalate
        act3 = service.escalate(inc_id, user_id="senior_supervisor", reason="High-traffic lobby zone threat")
        assert act3.action == "escalate"
        detail3 = service.get_incident_detail(inc_id)
        assert detail3["status"] == IncidentStatus.ESCALATED.value
        assert len(detail3["review_actions"]) == 3

        # 4. Action: mark_false_alarm with mandatory rationale
        act4 = service.mark_false_alarm(
            inc_id,
            user_id="lead_investigator",
            reason="Floor maintenance utility box misidentified as bladed object",
            note="Reviewed high-res frame with supervisor.",
        )
        assert act4.action == "mark_false_alarm"
        detail4 = service.get_incident_detail(inc_id)
        assert detail4["status"] == IncidentStatus.FALSE_ALARM.value
        assert len(detail4["review_actions"]) == 4

        # Error cases: empty rationale must be rejected
        with pytest.raises(FastTrackError):
            service.mark_false_alarm(inc_id, user_id="op", reason="   ")

        with pytest.raises(FastTrackError):
            service.escalate(inc_id, user_id="op", reason="")

        with pytest.raises(FastTrackError):
            service.add_note(inc_id, user_id="op", note="  ")

        # Non-existent incident
        with pytest.raises(FastTrackError):
            service.mark_reviewed("non-existent-inc", user_id="op")

        db_mgr.close()


def test_audit_log_export_and_integrity():
    """Verify exporting audit logs to JSON and CSV formats with SHA-256 integrity checksum."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = str(tmp_path / "test_audit.db")
        export_dir = str(tmp_path / "reports_audit")

        db_mgr = DatabaseManager(db_path=db_path)
        MigrationRunner(db_mgr).run_migrations()

        clock = SimulatedClock()
        service = ReviewService(db_manager=db_mgr, clock=clock)

        # Seed data
        cam_id = "cam-aud"
        inc_id = "inc-aud-01"
        now_iso = clock.now_utc().isoformat()

        db_mgr.execute_non_query(
            "INSERT INTO cameras (id, name, source_type, source_ref_masked, created_utc) VALUES (?, ?, ?, ?, ?)",
            (cam_id, "Vault Cam", "file", "vault.mp4", now_iso),
        )
        db_mgr.execute_non_query(
            "INSERT INTO incidents (id, camera_id, severity, status, start_utc) VALUES (?, ?, ?, ?, ?)",
            (inc_id, cam_id, "HIGH", "new", now_iso),
        )

        service.mark_reviewed(inc_id, user_id="operator_1", note="First review note")
        service.escalate(inc_id, user_id="supervisor_1", reason="Vault zone alarm")

        # 1. Export JSON
        json_path = service.export_audit_log(output_dir=export_dir, export_format="json")
        assert json_path.exists()
        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["record_count"] == 2
        assert len(payload["sha256"]) == 64
        assert len(payload["records"]) == 2

        # 2. Export CSV
        csv_path = service.export_audit_log(output_dir=export_dir, export_format="csv")
        assert csv_path.exists()
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3  # header + 2 records

        db_mgr.close()
