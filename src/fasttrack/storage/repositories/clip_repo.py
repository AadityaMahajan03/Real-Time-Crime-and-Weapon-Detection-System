"""Repository for incident forensic video clips (H.264 verified artifacts)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.fasttrack.storage.repositories.base import BaseRepository


class ClipRepository(BaseRepository):
    """Data access layer for forensic incident clips."""

    def get_by_incident_id(self, incident_id: str) -> List[Dict[str, Any]]:
        return self.execute_query(
            "SELECT * FROM clips WHERE incident_id = ? ORDER BY created_utc DESC;",
            (incident_id,),
        )

    def get_by_id(self, clip_id: str) -> Optional[Dict[str, Any]]:
        rows = self.execute_query("SELECT * FROM clips WHERE id = ?;", (clip_id,))
        return rows[0] if rows else None

    def create(
        self,
        clip_id: str,
        incident_id: str,
        camera_id: str,
        raw_clip_path: str,
        annotated_clip_path: str,
        thumbnail_path: str,
        sha256: str,
        duration_sec: float,
        size_bytes: int,
        codec: str = "h264",
        verification_status: str = "verified",
        keyframe_strip_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        now_utc = datetime.now(timezone.utc).isoformat()
        self.execute_non_query(
            """
            INSERT INTO clips (
                id, incident_id, camera_id, raw_clip_path, annotated_clip_path,
                thumbnail_path, keyframe_strip_path, sha256, duration_sec,
                size_bytes, codec, verification_status, created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                clip_id,
                incident_id,
                camera_id,
                raw_clip_path,
                annotated_clip_path,
                thumbnail_path,
                keyframe_strip_path,
                sha256,
                duration_sec,
                size_bytes,
                codec,
                verification_status,
                now_utc,
            ),
        )
        return self.get_by_id(clip_id) or {}

    def update_verification_status(self, clip_id: str, status: str) -> None:
        self.execute_non_query(
            "UPDATE clips SET verification_status = ? WHERE id = ?;",
            (status, clip_id),
        )
