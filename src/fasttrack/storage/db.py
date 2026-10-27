"""SQLite connection manager with WAL mode, foreign keys, and retry logic."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator, List, Tuple

from src.fasttrack.core.errors import DatabaseError
from src.fasttrack.core.logging import get_logger

logger = get_logger("storage.db")


class DatabaseManager:
    """Manages thread-safe SQLite connections configured for concurrent operations."""

    def __init__(self, db_path: str = "data/fasttrack.db", busy_timeout_ms: int = 5000) -> None:
        self.db_path = db_path
        self.busy_timeout_ms = busy_timeout_ms
        self._local = threading.local()

        # Ensure directory exists if not in-memory
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    def _create_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection with WAL and pragmas."""
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=self.busy_timeout_ms / 1000.0,
                detect_types=sqlite3.PARSE_DECLTYPES,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()
            if self.db_path != ":memory:":
                cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute(f"PRAGMA busy_timeout = {self.busy_timeout_ms};")
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.close()
            return conn
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to connect to database at {self.db_path}: {e}")

    def get_connection(self) -> sqlite3.Connection:
        """Get or create a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = self._create_connection()
        return self._local.conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a thread-local connection with auto-commit on success."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except sqlite3.OperationalError as e:
            conn.rollback()
            if "locked" in str(e).lower() or "busy" in str(e).lower():
                logger.warning("Database locked; rolling back: %s", e)
            raise DatabaseError(f"Database operational error: {e}")
        except Exception as e:
            conn.rollback()
            raise DatabaseError(f"Transaction failed: {e}")

    def execute_query(
        self, query: str, params: Tuple[Any, ...] = ()
    ) -> List[sqlite3.Row]:
        """Execute a read query and return all rows."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_non_query(self, query: str, params: Tuple[Any, ...] = ()) -> int:
        """Execute an insert/update/delete query and return rowcount."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount

    def execute_scalar(self, query: str, params: Tuple[Any, ...] = ()) -> Any:
        """Execute query returning a single scalar value from the first column of the first row."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            if row is not None:
                return row[0]
            return None

    def close(self) -> None:
        """Close connection for current thread."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            try:
                self._local.conn.close()
            except Exception:
                pass
            self._local.conn = None
