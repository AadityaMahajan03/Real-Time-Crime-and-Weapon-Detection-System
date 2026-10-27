"""Incident repository managing security alert lifecycle, metrics, and evidence linking."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.fasttrack.storage.repositories.base import BaseRepository


class IncidentRepository(BaseRepository):
    """Data access layer for security threat incidents and real-time operational analytics."""

    def get_all(
        self,
        limit: int = 100,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        camera_id: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = """
        SELECT i.*, c.name as camera_name, c.source_type, c.source_ref_masked
        FROM incidents i
        LEFT JOIN cameras c ON i.camera_id = c.id
        WHERE 1=1
        """
        params: List[Any] = []

        if severity and severity.lower() != "all":
            query += " AND LOWER(i.severity) = ?"
            params.append(severity.lower())

        if status and status.lower() != "all":
            query += " AND LOWER(i.status) = ?"
            params.append(status.lower())

        if camera_id and camera_id.lower() != "all":
            query += " AND i.camera_id = ?"
            params.append(camera_id)

        if search and search.strip():
            query += " AND (i.id LIKE ? OR i.summary LIKE ? OR c.name LIKE ?)"
            s_param = f"%{search.strip()}%"
            params.extend([s_param, s_param, s_param])

        query += " ORDER BY i.start_utc DESC LIMIT ?;"
        params.append(limit)

        return self.execute_query(query, tuple(params))

    def get_by_id(self, incident_id: str) -> Optional[Dict[str, Any]]:
        rows = self.execute_query(
            """
            SELECT i.*, c.name as camera_name, c.source_type, c.source_ref_masked
            FROM incidents i
            LEFT JOIN cameras c ON i.camera_id = c.id
            WHERE i.id = ?;
            """,
            (incident_id,),
        )
        if not rows:
            return None

        incident = dict(rows[0])

        # Fetch associated clips
        incident["clips"] = self.execute_query(
            "SELECT * FROM clips WHERE incident_id = ? ORDER BY created_utc DESC;",
            (incident_id,),
        )

        # Fetch associated evidence snapshots
        incident["evidence"] = self.execute_query(
            """
            SELECT e.* FROM evidence e
            JOIN incident_evidence ie ON e.id = ie.evidence_id
            WHERE ie.incident_id = ?
            ORDER BY e.created_utc ASC;
            """,
            (incident_id,),
        )

        # Fetch review actions / audit trail
        incident["review_actions"] = self.execute_query(
            "SELECT * FROM review_actions WHERE incident_id = ? ORDER BY ts_utc ASC;",
            (incident_id,),
        )

        return incident

    def create(
        self,
        incident_id: str,
        camera_id: str,
        severity: str,
        start_utc: str,
        peak_weapon_score: float = 0.0,
        peak_violence_score: float = 0.0,
        summary: str = "",
        status: str = "new",
    ) -> Dict[str, Any]:
        self.execute_non_query(
            """
            INSERT INTO incidents (
                id, camera_id, severity, status, start_utc,
                peak_weapon_score, peak_violence_score, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                incident_id,
                camera_id,
                severity,
                status,
                start_utc,
                peak_weapon_score,
                peak_violence_score,
                summary,
            ),
        )
        return self.get_by_id(incident_id) or {}

    def link_evidence(self, incident_id: str, evidence_id: str) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        self.execute_non_query(
            """
            INSERT OR IGNORE INTO incident_evidence (incident_id, evidence_id, attached_utc)
            VALUES (?, ?, ?);
            """,
            (incident_id, evidence_id, now_utc),
        )

    def update_status(
        self,
        incident_id: str,
        status: str,
        user_id: str,
        note: str = "",
    ) -> None:
        now_utc = datetime.now(timezone.utc).isoformat()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE incidents SET status = ? WHERE id = ?;", (status, incident_id))
            action_id = f"act-{datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}"
            cursor.execute(
                """
                INSERT INTO review_actions (id, incident_id, user_id, action, note, ts_utc)
                VALUES (?, ?, ?, ?, ?, ?);
                """,
                (action_id, incident_id, user_id, f"mark_{status}", note, now_utc),
            )

    def get_real_kpis(self) -> Dict[str, Any]:
        """Compute real security KPIs strictly from actual database rows."""
        total_incidents = self.execute_scalar("SELECT COUNT(*) FROM incidents;") or 0
        open_incidents = (
            self.execute_scalar("SELECT COUNT(*) FROM incidents WHERE status IN ('new', 'in_review');") or 0
        )
        confirmed_threats = (
            self.execute_scalar("SELECT COUNT(*) FROM incidents WHERE status = 'reviewed' OR severity = 'HIGH';")
            or 0
        )
        false_alarms = self.execute_scalar("SELECT COUNT(*) FROM incidents WHERE status = 'false_alarm';") or 0

        # Calculate False Alarm Rate (FAR) strictly from reviewed volume
        reviewed_total = confirmed_threats + false_alarms
        far_pct = round((false_alarms / reviewed_total * 100.0), 1) if reviewed_total > 0 else 0.0

        return {
            "total_incidents": total_incidents,
            "open_incidents": open_incidents,
            "confirmed_threats": confirmed_threats,
            "false_alarms": false_alarms,
            "false_alarm_rate_pct": far_pct,
        }

    def get_severity_distribution(self) -> List[Dict[str, Any]]:
        """Compute real incident counts grouped by severity."""
        rows = self.execute_query(
            """
            SELECT severity, COUNT(*) as count
            FROM incidents
            GROUP BY severity
            ORDER BY count DESC;
            """
        )
        return rows

    def get_hourly_activity(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Compute hourly incident detection volume from database."""
        return self.execute_query(
            """
            SELECT strftime('%Y-%m-%d %H:00', start_utc) as hour_bucket,
                   COUNT(*) as incident_count,
                   AVG(peak_weapon_score) as avg_weapon_score,
                   AVG(peak_violence_score) as avg_violence_score
            FROM incidents
            GROUP BY hour_bucket
            ORDER BY hour_bucket DESC
            LIMIT ?;
            """,
            (hours,),
        )
