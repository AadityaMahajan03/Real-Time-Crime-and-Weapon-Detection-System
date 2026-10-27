"""Unit tests for SQLite connection manager, WAL mode, and migrations."""

from __future__ import annotations

import pytest

from src.fasttrack.core.errors import DatabaseError
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner


def test_migrations_create_expected_tables(in_memory_db: DatabaseManager) -> None:
    # Query sqlite_master to verify tables exist
    rows = in_memory_db.execute_query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    table_names = {row["name"] for row in rows}

    expected_tables = {
        "_schema_migrations",
        "cameras",
        "incidents",
        "events",
        "evidence",
        "review_actions",
        "users",
        "model_versions",
        "evaluation_runs",
        "health_samples",
        "settings",
    }
    assert expected_tables.issubset(table_names)


def test_migration_runner_idempotency(in_memory_db: DatabaseManager) -> None:
    runner = MigrationRunner(in_memory_db, migrations_dir="src/fasttrack/storage/migrations")

    # Second run should find 0 pending migrations
    applied_count = runner.run_migrations()
    assert applied_count == 0


def test_foreign_key_enforcement(in_memory_db: DatabaseManager) -> None:
    # Inserting an incident referencing a non-existent camera must fail due to foreign keys
    with pytest.raises(DatabaseError):
        in_memory_db.execute_non_query(
            """
            INSERT INTO incidents (id, camera_id, severity, status, start_utc, summary)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            ("inc-invalid", "cam-nonexistent", "HIGH", "new", "2026-01-01T00:00:00Z", "Test summary"),
        )


def test_parameterized_crud_operations(in_memory_db: DatabaseManager) -> None:
    # Insert valid camera
    in_memory_db.execute_non_query(
        """
        INSERT INTO cameras (id, name, source_type, source_ref_masked, enabled, created_utc)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        ("cam-test", "Gate 1", "file", "data/test.mp4", 1, "2026-01-01T00:00:00Z"),
    )

    # Insert incident linked to camera
    in_memory_db.execute_non_query(
        """
        INSERT INTO incidents (id, camera_id, severity, status, start_utc, summary)
        VALUES (?, ?, ?, ?, ?, ?);
        """,
        ("inc-001", "cam-test", "MEDIUM", "new", "2026-01-01T00:05:00Z", "Detected object: knife (0.75)"),
    )

    # Query incident
    rows = in_memory_db.execute_query("SELECT * FROM incidents WHERE id = ?;", ("inc-001",))
    assert len(rows) == 1
    assert rows[0]["camera_id"] == "cam-test"
    assert rows[0]["severity"] == "MEDIUM"
    assert "knife (0.75)" in rows[0]["summary"]
