"""Versioned database migrations runner for FastTrack AI."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from src.fasttrack.core.errors import DatabaseError
from src.fasttrack.core.logging import get_logger
from src.fasttrack.storage.db import DatabaseManager

logger = get_logger("storage.migrations")

_MIGRATION_PATTERN = re.compile(r"^(V\d+)__(.+)\.sql$")


class MigrationRunner:
    """Manages forward schema migrations for SQLite database."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        migrations_dir: str = "src/fasttrack/storage/migrations",
    ) -> None:
        self.db = db_manager
        self.migrations_dir = Path(migrations_dir)

    def _ensure_migrations_table(self) -> None:
        """Create the schema migrations tracking table if not present."""
        query = """
        CREATE TABLE IF NOT EXISTS _schema_migrations (
            version TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            applied_utc TEXT NOT NULL
        );
        """
        self.db.execute_non_query(query)

    def get_applied_versions(self) -> List[str]:
        """Fetch list of migration versions already applied to the database."""
        self._ensure_migrations_table()
        rows = self.db.execute_query("SELECT version FROM _schema_migrations ORDER BY version ASC;")
        return [row["version"] for row in rows]

    def get_pending_migrations(self) -> List[Tuple[str, str, Path]]:
        """Identify available migration files that have not yet been applied."""
        if not self.migrations_dir.exists():
            return []

        applied = set(self.get_applied_versions())
        pending: List[Tuple[str, str, Path]] = []

        files = sorted(self.migrations_dir.glob("V*.sql"))
        for file in files:
            match = _MIGRATION_PATTERN.match(file.name)
            if match:
                version, description = match.group(1), match.group(2)
                if version not in applied:
                    pending.append((version, description, file))

        return pending

    def run_migrations(self) -> int:
        """Apply all pending migrations in order. Returns count of applied migrations."""
        self._ensure_migrations_table()
        pending = self.get_pending_migrations()

        if not pending:
            logger.info("Database schema is up to date. No pending migrations.")
            return 0

        applied_count = 0
        for version, desc, path in pending:
            logger.info("Applying migration %s: %s (%s)", version, desc, path.name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    sql_script = f.read()

                # Execute migration script inside a transaction
                with self.db.connection() as conn:
                    cursor = conn.cursor()
                    cursor.executescript(sql_script)
                    now_utc = datetime.now(timezone.utc).isoformat()
                    cursor.execute(
                        "INSERT INTO _schema_migrations (version, description, applied_utc) VALUES (?, ?, ?);",
                        (version, desc, now_utc),
                    )
                applied_count += 1
                logger.info("Successfully applied migration %s.", version)
            except Exception as e:
                logger.error("Failed to apply migration %s: %s", version, e)
                raise DatabaseError(f"Migration {version} failed: {e}")

        return applied_count
