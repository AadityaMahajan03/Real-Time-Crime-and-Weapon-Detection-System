"""Repository for continuous camera surveillance video segments."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.fasttrack.storage.repositories.base import BaseRepository


class RecordingRepository(BaseRepository):
    """Data access layer for 60-second continuous camera recording segments."""

    def create_segment(
        self,
        segment_id: str,
        camera_id: str,
        segment_path: str,
        start_utc: str,
        duration_sec: int = 60,
    ) -> Dict[str, Any]:
        now_utc = datetime.now(timezone.utc).isoformat()
        self.execute_non_query(
            """
            INSERT INTO recordings (id, camera_id, segment_path, start_utc, duration_sec, status, created_utc)
            VALUES (?, ?, ?, ?, ?, 'recording', ?);
            """,
            (segment_id, camera_id, segment_path, start_utc, duration_sec, now_utc),
        )
        return self.get_by_id(segment_id) or {}

    def close_segment(
        self,
        segment_id: str,
        end_utc: str,
        size_bytes: int,
        sha256: str,
        status: str = "completed",
    ) -> None:
        self.execute_non_query(
            """
            UPDATE recordings
            SET end_utc = ?, size_bytes = ?, sha256 = ?, status = ?
            WHERE id = ?;
            """,
            (end_utc, size_bytes, sha256, status, segment_id),
        )

    def get_by_id(self, segment_id: str) -> Optional[Dict[str, Any]]:
        rows = self.execute_query("SELECT * FROM recordings WHERE id = ?;", (segment_id,))
        return rows[0] if rows else None

    def get_segments_for_range(
        self,
        camera_id: str,
        start_utc: str,
        end_utc: str,
    ) -> List[Dict[str, Any]]:
        """Fetch all completed segments that overlap the specified time window."""
        return self.execute_query(
            """
            SELECT * FROM recordings
            WHERE camera_id = ?
              AND status = 'completed'
              AND (
                (start_utc <= ? AND (end_utc >= ? OR end_utc IS NULL))
                OR (start_utc >= ? AND start_utc <= ?)
              )
            ORDER BY start_utc ASC;
            """,
            (camera_id, end_utc, start_utc, start_utc, end_utc),
        )

    def lock_segments_for_incident(
        self,
        camera_id: str,
        start_utc: str,
        end_utc: str,
    ) -> int:
        """Mark all segments covering an incident as locked (is_locked=1) so retention cleanup never deletes them."""
        return self.execute_non_query(
            """
            UPDATE recordings
            SET is_locked = 1
            WHERE camera_id = ?
              AND (
                (start_utc <= ? AND (end_utc >= ? OR end_utc IS NULL))
                OR (start_utc >= ? AND start_utc <= ?)
              );
            """,
            (camera_id, end_utc, start_utc, start_utc, end_utc),
        )

    def get_unlocked_older_than(self, cutoff_utc: str) -> List[Dict[str, Any]]:
        """Find unlocked segments older than the retention cutoff timestamp."""
        return self.execute_query(
            """
            SELECT * FROM recordings
            WHERE is_locked = 0
              AND start_utc < ?
            ORDER BY start_utc ASC;
            """,
            (cutoff_utc,),
        )

    def delete_segment_record(self, segment_id: str) -> None:
        self.execute_non_query("DELETE FROM recordings WHERE id = ? AND is_locked = 0;", (segment_id,))
