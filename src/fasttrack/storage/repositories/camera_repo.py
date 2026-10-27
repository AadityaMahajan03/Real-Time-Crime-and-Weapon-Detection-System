"""Camera repository managing surveillance hardware sources, states, and ROI zones."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.fasttrack.storage.repositories.base import BaseRepository


class CameraRepository(BaseRepository):
    """Data access layer for surveillance cameras fleet."""

    def get_all(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        query = "SELECT * FROM cameras"
        params = ()
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY created_utc ASC;"
        return self.execute_query(query, params)

    def get_by_id(self, camera_id: str) -> Optional[Dict[str, Any]]:
        rows = self.execute_query("SELECT * FROM cameras WHERE id = ?;", (camera_id,))
        return rows[0] if rows else None

    def create(
        self,
        camera_id: str,
        name: str,
        source_type: str,
        source_ref_masked: str,
        enabled: bool = True,
        resolution: str = "1920x1080",
        fps: float = 30.0,
        zones: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        now_utc = datetime.now(timezone.utc).isoformat()
        zones_json = json.dumps(zones or [])
        self.execute_non_query(
            """
            INSERT INTO cameras (id, name, source_type, source_ref_masked, enabled, zones_json, created_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (camera_id, name, source_type, source_ref_masked, 1 if enabled else 0, zones_json, now_utc),
        )
        return self.get_by_id(camera_id) or {}

    def update_enabled(self, camera_id: str, enabled: bool) -> None:
        self.execute_non_query(
            "UPDATE cameras SET enabled = ? WHERE id = ?;",
            (1 if enabled else 0, camera_id),
        )

    def set_all_enabled(self, enabled: bool) -> int:
        return self.execute_non_query(
            "UPDATE cameras SET enabled = ?;",
            (1 if enabled else 0,),
        )

    def set_subset_enabled(self, active_camera_ids: List[str]) -> None:
        """Atomically enable the specified camera IDs and disable all others."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE cameras SET enabled = 0;")
            if active_camera_ids:
                placeholders = ",".join(["?"] * len(active_camera_ids))
                cursor.execute(f"UPDATE cameras SET enabled = 1 WHERE id IN ({placeholders});", tuple(active_camera_ids))

    def delete(self, camera_id: str) -> None:
        self.execute_non_query("DELETE FROM cameras WHERE id = ?;", (camera_id,))
