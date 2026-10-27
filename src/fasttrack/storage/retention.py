"""Automated evidence retention cleanup job and storage lifecycle enforcement."""

from __future__ import annotations

import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.logging import get_logger
from src.fasttrack.storage.db import DatabaseManager

logger = get_logger("storage.retention")


def cleanup_expired_evidence(
    db_manager: DatabaseManager,
    evidence_dir: str = "evidence",
    retention_days: Optional[int] = None,
    clock: Optional[Clock] = None,
) -> Dict[str, Any]:
    """Scan and delete expired evidence artifacts from disk and database with audit recording.

    Removes files whose expires_utc <= now_utc, or created_utc older than retention_days.
    Cleans up orphaned .tmp files left over from interrupted write operations.
    """
    clk = clock or SystemClock()
    now_utc = clk.now_utc()
    now_iso = now_utc.isoformat()

    stats: Dict[str, Any] = {
        "total_scanned": 0,
        "total_purged": 0,
        "bytes_freed": 0,
        "tmp_files_cleaned": 0,
        "errors": [],
    }

    # 1. Identify expired evidence records
    query = """
        SELECT id, kind, path, size_bytes, created_utc, expires_utc
        FROM evidence
        WHERE (expires_utc IS NOT NULL AND expires_utc <= ?)
    """
    params: List[Any] = [now_iso]

    if retention_days is not None and retention_days > 0:
        cutoff_utc = now_utc - timedelta(days=retention_days)
        query += " OR (created_utc <= ?)"
        params.append(cutoff_utc.isoformat())

    try:
        records = db_manager.execute_query(query, tuple(params))
        stats["total_scanned"] = len(records)
    except Exception as e:
        logger.error("Failed to query expired evidence: %s", e)
        stats["errors"].append(str(e))
        return stats

    # 2. Delete each expired artifact
    for rec in records:
        ev_id = rec["id"]
        ev_path = Path(rec["path"])
        size_bytes = rec["size_bytes"]
        kind = rec["kind"]

        # Check for associated incident to log review action
        incident_id = None
        try:
            evt_rows = db_manager.execute_query(
                "SELECT incident_id FROM events WHERE evidence_id = ? AND incident_id IS NOT NULL LIMIT 1",
                (ev_id,),
            )
            if evt_rows:
                incident_id = evt_rows[0]["incident_id"]
        except Exception:
            pass

        # Unlink file from disk
        file_deleted = False
        try:
            if ev_path.exists():
                ev_path.unlink()
                file_deleted = True
                stats["bytes_freed"] += size_bytes
            else:
                logger.warning("Evidence file already missing on disk: %s", ev_path)
                file_deleted = True
        except Exception as e:
            err_msg = f"Failed to delete evidence file {ev_path}: {e}"
            logger.error(err_msg)
            stats["errors"].append(err_msg)
            continue  # Do not purge DB row if file couldn't be deleted

        # Purge DB record
        if file_deleted:
            try:
                db_manager.execute_non_query(
                    "UPDATE events SET evidence_id = NULL WHERE evidence_id = ?",
                    (ev_id,),
                )
                db_manager.execute_non_query("DELETE FROM evidence WHERE id = ?", (ev_id,))
                stats["total_purged"] += 1

                # If incident was linked, record audit review action
                if incident_id:
                    act_id = f"act-{uuid.uuid4().hex[:10]}"
                    db_manager.execute_non_query(
                        """
                        INSERT INTO review_actions (id, incident_id, user_id, action, note, ts_utc)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            act_id,
                            incident_id,
                            "system_retention",
                            "evidence_purged",
                            f"Automated retention purge: removed {kind} ({size_bytes} bytes).",
                            now_iso,
                        ),
                    )
            except Exception as e:
                err_msg = f"Failed to purge evidence {ev_id} from database: {e}"
                logger.error(err_msg)
                stats["errors"].append(err_msg)

    # 3. Clean up orphaned .tmp files in evidence directory
    ev_dir_path = Path(evidence_dir)
    if ev_dir_path.exists():
        for sub in ("snapshots", "clips"):
            sub_path = ev_dir_path / sub
            if not sub_path.exists():
                continue
            for tmp_file in sub_path.glob("*tmp*"):
                try:
                    # Clean up tmp file
                    tmp_file.unlink(missing_ok=True)
                    stats["tmp_files_cleaned"] += 1
                except Exception as e:
                    stats["errors"].append(f"Failed to remove tmp file {tmp_file}: {e}")

    logger.info(
        "Evidence retention cleanup completed: %d purged, %.2f MB freed, %d tmp cleaned",
        stats["total_purged"],
        stats["bytes_freed"] / (1024 * 1024),
        stats["tmp_files_cleaned"],
    )

    return stats
