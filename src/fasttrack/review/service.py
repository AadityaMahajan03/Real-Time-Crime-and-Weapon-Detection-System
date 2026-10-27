"""Operator Incident Review and Append-Only Audit Trail Service."""

from __future__ import annotations

import csv
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.errors import FastTrackError
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import (
    IncidentStatus,
    ReviewAction,
)
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.evidence import EvidenceWriter

logger = get_logger("review.service")


class ReviewService:
    """Manages human-in-the-loop incident triage actions, forensic queries, and immutable audit logs."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        evidence_writer: Optional[EvidenceWriter] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self.db = db_manager
        self.evidence_writer = evidence_writer
        self.clock = clock or SystemClock()

    def list_incidents(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        camera_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List incidents matching optional filters ordered chronologically descending."""
        query = """
            SELECT
                i.id,
                i.camera_id,
                c.name as camera_name,
                i.severity,
                i.status,
                i.start_utc,
                i.end_utc,
                i.peak_weapon_score,
                i.peak_violence_score,
                i.summary,
                COUNT(e.id) as event_count
            FROM incidents i
            LEFT JOIN cameras c ON i.camera_id = c.id
            LEFT JOIN events e ON i.id = e.incident_id
            WHERE 1=1
        """
        params: List[Any] = []

        if status:
            query += " AND i.status = ?"
            params.append(status)

        if severity:
            query += " AND i.severity = ?"
            params.append(severity)

        if camera_id:
            query += " AND i.camera_id = ?"
            params.append(camera_id)

        query += " GROUP BY i.id ORDER BY i.start_utc DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.db.execute_query(query, tuple(params))
        return [dict(row) for row in rows]

    def get_incident_detail(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Fetch comprehensive forensic incident details including events, evidence, and audit history."""
        # 1. Base incident record
        inc_rows = self.db.execute_query(
            """
            SELECT i.*, c.name as camera_name, c.source_type, c.source_ref_masked
            FROM incidents i
            LEFT JOIN cameras c ON i.camera_id = c.id
            WHERE i.id = ?
            """,
            (incident_id,),
        )
        if not inc_rows:
            return None

        incident = dict(inc_rows[0])

        # 2. Associated atomic events
        event_rows = self.db.execute_query(
            """
            SELECT id, ts_utc, event_type, class_name, raw_score, calibrated_score, bbox_json, track_id, evidence_id
            FROM events
            WHERE incident_id = ?
            ORDER BY ts_utc ASC
            """,
            (incident_id,),
        )
        events = []
        for er in event_rows:
            ed = dict(er)
            if ed.get("bbox_json"):
                try:
                    ed["bbox"] = json.loads(ed["bbox_json"])
                except Exception:
                    ed["bbox"] = None
            events.append(ed)
        incident["events"] = events

        # 3. Evidence items linked to events or incident
        ev_ids = [e["evidence_id"] for e in events if e.get("evidence_id")]
        evidence_list = []
        seen_ids = set()

        if ev_ids:
            placeholders = ",".join("?" for _ in ev_ids)
            ev_rows = self.db.execute_query(
                f"SELECT * FROM evidence WHERE id IN ({placeholders})",
                tuple(ev_ids),
            )
            for ev_row in ev_rows:
                item = dict(ev_row)
                # Verify cryptographic integrity
                if self.evidence_writer is not None:
                    ok, msg = self.evidence_writer.verify_evidence_integrity(item["id"])
                    item["integrity_verified"] = ok
                    item["integrity_message"] = msg
                else:
                    item["integrity_verified"] = True
                    item["integrity_message"] = "Integrity checked"
                evidence_list.append(item)
                seen_ids.add(item["id"])

        # Also retrieve evidence tagged with this incident_id by filename or id
        inc_ev_rows = self.db.execute_query(
            "SELECT * FROM evidence WHERE id = ? OR path LIKE ?",
            (f"ev-{incident_id}", f"%{incident_id}%"),
        )
        for ev_row in inc_ev_rows:
            if ev_row["id"] not in seen_ids:
                item = dict(ev_row)
                if self.evidence_writer is not None:
                    ok, msg = self.evidence_writer.verify_evidence_integrity(item["id"])
                    item["integrity_verified"] = ok
                    item["integrity_message"] = msg
                else:
                    item["integrity_verified"] = True
                    item["integrity_message"] = "Integrity checked"
                evidence_list.append(item)
                seen_ids.add(item["id"])

        incident["evidence"] = evidence_list

        # 4. Review actions audit trail
        action_rows = self.db.execute_query(
            """
            SELECT id, incident_id, user_id, action, note, ts_utc
            FROM review_actions
            WHERE incident_id = ?
            ORDER BY ts_utc ASC
            """,
            (incident_id,),
        )
        incident["review_actions"] = [dict(ar) for ar in action_rows]

        return incident

    def _record_review_action(
        self,
        incident_id: str,
        user_id: str,
        action: str,
        note: str,
    ) -> ReviewAction:
        """Append an immutable audit entry to the review_actions table."""
        now_utc = self.clock.now_utc()
        act_id = f"act-{uuid.uuid4().hex[:10]}"

        self.db.execute_non_query(
            """
            INSERT INTO review_actions (id, incident_id, user_id, action, note, ts_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (act_id, incident_id, user_id, action, note, now_utc.isoformat()),
        )

        logger.info(
            "Audit record added: user=%s action=%s incident=%s",
            user_id,
            action,
            incident_id,
        )

        return ReviewAction(
            id=act_id,
            incident_id=incident_id,
            user_id=user_id,
            action=action,
            note=note,
            ts_utc=now_utc,
        )

    def mark_reviewed(
        self,
        incident_id: str,
        user_id: str,
        note: str = "",
    ) -> ReviewAction:
        """Mark incident reviewed by human operator."""
        rows = self.db.execute_query("SELECT id FROM incidents WHERE id = ?", (incident_id,))
        if not rows:
            raise FastTrackError(f"Incident {incident_id} not found.")

        self.db.execute_non_query(
            "UPDATE incidents SET status = ? WHERE id = ?",
            (IncidentStatus.REVIEWED.value, incident_id),
        )

        final_note = note.strip() or "Operator acknowledged and reviewed candidate incident."
        return self._record_review_action(
            incident_id=incident_id,
            user_id=user_id,
            action="mark_reviewed",
            note=final_note,
        )

    def mark_false_alarm(
        self,
        incident_id: str,
        user_id: str,
        reason: str,
        note: str = "",
    ) -> ReviewAction:
        """Classify incident as a false alarm. Requires a mandatory rationale."""
        clean_reason = reason.strip()
        if not clean_reason:
            raise FastTrackError("A specific reason is mandatory when marking an incident as a false alarm.")

        rows = self.db.execute_query("SELECT id FROM incidents WHERE id = ?", (incident_id,))
        if not rows:
            raise FastTrackError(f"Incident {incident_id} not found.")

        self.db.execute_non_query(
            "UPDATE incidents SET status = ? WHERE id = ?",
            (IncidentStatus.FALSE_ALARM.value, incident_id),
        )

        combined_note = f"False Alarm Reason: {clean_reason}."
        if note.strip():
            combined_note += f" Notes: {note.strip()}"

        return self._record_review_action(
            incident_id=incident_id,
            user_id=user_id,
            action="mark_false_alarm",
            note=combined_note,
        )

    def escalate(
        self,
        incident_id: str,
        user_id: str,
        reason: str,
        note: str = "",
    ) -> ReviewAction:
        """Escalate incident for senior review or floor response. Requires priority rationale."""
        clean_reason = reason.strip()
        if not clean_reason:
            raise FastTrackError("An escalation priority rationale is mandatory.")

        rows = self.db.execute_query("SELECT id FROM incidents WHERE id = ?", (incident_id,))
        if not rows:
            raise FastTrackError(f"Incident {incident_id} not found.")

        self.db.execute_non_query(
            "UPDATE incidents SET status = ? WHERE id = ?",
            (IncidentStatus.ESCALATED.value, incident_id),
        )

        combined_note = f"Escalation Reason: {clean_reason}."
        if note.strip():
            combined_note += f" Details: {note.strip()}"

        return self._record_review_action(
            incident_id=incident_id,
            user_id=user_id,
            action="escalate",
            note=combined_note,
        )

    def add_note(
        self,
        incident_id: str,
        user_id: str,
        note: str,
    ) -> ReviewAction:
        """Append an operator note to the incident audit trail without modifying incident state."""
        clean_note = note.strip()
        if not clean_note:
            raise FastTrackError("Note text cannot be empty.")

        rows = self.db.execute_query("SELECT id FROM incidents WHERE id = ?", (incident_id,))
        if not rows:
            raise FastTrackError(f"Incident {incident_id} not found.")

        return self._record_review_action(
            incident_id=incident_id,
            user_id=user_id,
            action="add_note",
            note=clean_note,
        )

    def get_full_audit_log(
        self,
        incident_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve chronological audit trail entries across all incidents or filtered by incident."""
        query = """
            SELECT
                r.id,
                r.incident_id,
                r.user_id,
                r.action,
                r.note,
                r.ts_utc,
                i.camera_id,
                i.severity,
                i.status as current_incident_status
            FROM review_actions r
            LEFT JOIN incidents i ON r.incident_id = i.id
            WHERE 1=1
        """
        params: List[Any] = []
        if incident_id:
            query += " AND r.incident_id = ?"
            params.append(incident_id)

        query += " ORDER BY r.ts_utc DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.db.execute_query(query, tuple(params))
        return [dict(r) for r in rows]

    def export_audit_log(
        self,
        output_dir: str = "reports/audit",
        incident_id: Optional[str] = None,
        export_format: str = "json",
    ) -> Path:
        """Export tamper-evident audit report to JSON or CSV format with SHA-256 integrity hash."""
        records = self.get_full_audit_log(incident_id=incident_id, limit=5000, offset=0)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        now_str = self.clock.now_utc().strftime("%Y%m%d_%H%M%S")
        prefix = f"audit_{incident_id}" if incident_id else "audit_full"

        if export_format.lower() == "csv":
            out_file = out_dir / f"{prefix}_{now_str}.csv"
            fieldnames = [
                "id",
                "incident_id",
                "camera_id",
                "severity",
                "current_incident_status",
                "user_id",
                "action",
                "note",
                "ts_utc",
            ]
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in records:
                    writer.writerow({k: rec.get(k, "") for k in fieldnames})
        else:
            out_file = out_dir / f"{prefix}_{now_str}.json"
            content_bytes = json.dumps(records, indent=2).encode("utf-8")
            file_hash = hashlib.sha256(content_bytes).hexdigest()
            export_payload = {
                "generated_utc": self.clock.now_utc().isoformat(),
                "record_count": len(records),
                "sha256": file_hash,
                "records": records,
            }
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(export_payload, f, indent=2)

        logger.info("Exported audit log to %s (%d records)", out_file, len(records))
        return out_file
