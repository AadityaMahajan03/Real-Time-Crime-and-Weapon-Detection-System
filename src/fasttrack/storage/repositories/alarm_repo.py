"""Repository for acoustic siren alerts and operator acknowledgment audit."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.fasttrack.storage.repositories.base import BaseRepository


class AlarmRepository(BaseRepository):
    """Data access layer for security acoustic sirens."""

    def create_alarm(
        self,
        alarm_id: str,
        incident_id: Optional[str],
        camera_id: str,
        alarm_type: str,
        pattern: str = "continuous",
    ) -> Dict[str, Any]:
        now_utc = datetime.now(timezone.utc).isoformat()
        self.execute_non_query(
            """
            INSERT INTO alarms (id, incident_id, camera_id, alarm_type, pattern, start_utc, acknowledged)
            VALUES (?, ?, ?, ?, ?, ?, 0);
            """,
            (alarm_id, incident_id, camera_id, alarm_type, pattern, now_utc),
        )
        return self.get_by_id(alarm_id) or {}

    def stop_alarm(self, alarm_id: str, duration_sec: float) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        self.execute_non_query(
            """
            UPDATE alarms
            SET stop_utc = ?, duration_sec = ?
            WHERE id = ?;
            """,
            (now_utc, duration_sec, alarm_id),
        )

    def acknowledge_alarm(self, alarm_id: str, ack_by: str, notes: str = "") -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        self.execute_non_query(
            """
            UPDATE alarms
            SET acknowledged = 1, ack_by = ?, ack_utc = ?, notes = ?
            WHERE id = ?;
            """,
            (ack_by, now_utc, notes, alarm_id),
        )

    def get_by_id(self, alarm_id: str) -> Optional[Dict[str, Any]]:
        rows = self.execute_query("SELECT * FROM alarms WHERE id = ?;", (alarm_id,))
        return rows[0] if rows else None

    def get_active_alarms(self) -> List[Dict[str, Any]]:
        return self.execute_query(
            "SELECT * FROM alarms WHERE stop_utc IS NULL ORDER BY start_utc DESC;"
        )

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.execute_query(
            "SELECT * FROM alarms ORDER BY start_utc DESC LIMIT ?;",
            (limit,),
        )
