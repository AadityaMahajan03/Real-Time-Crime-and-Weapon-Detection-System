"""Surveillance footage retention policy service protecting incident evidence."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.fasttrack.core.logging import get_logger
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.repositories import AuditRepository, RecordingRepository

logger = get_logger("services.retention")


class RetentionService:
    """Enforces disk retention quotas, safely recycling unlinked background recordings while protecting evidence."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        max_retention_days: int = 7,
        max_storage_gb: float = 20.0,
        cleanup_interval_sec: float = 3600.0,
    ) -> None:
        self.db = db_manager
        self.rec_repo = RecordingRepository(db_manager)
        self.audit_repo = AuditRepository(db_manager)
        self.max_retention_days = max_retention_days
        self.max_storage_gb = max_storage_gb
        self.cleanup_interval_sec = cleanup_interval_sec

        self._is_running = True
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run_loop, name="RetentionWorker", daemon=True
            )
            self._thread.start()

    def _run_loop(self) -> None:
        while self._is_running:
            try:
                self.enforce_retention()
            except Exception as e:
                logger.error("Error during retention cleanup pass: %s", e)
            time.sleep(self.cleanup_interval_sec)

    def enforce_retention(self) -> int:
        """Purge recordings exceeding retention age, strictly preserving incident-locked footage."""
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=self.max_retention_days)
        cutoff_iso = cutoff_dt.isoformat()

        unlocked_candidates = self.rec_repo.get_unlocked_older_than(cutoff_iso)
        if not unlocked_candidates:
            logger.debug("Retention check completed: 0 expired unlocked segments.")
            return 0

        purged_count = 0
        reclaimed_bytes = 0

        for seg in unlocked_candidates:
            seg_id = seg["id"]
            path_str = seg["segment_path"]
            seg_p = Path(path_str)

            # Safety check: ensure segment is not locked
            if seg["is_locked"] != 0:
                continue

            try:
                if seg_p.exists():
                    sz = seg_p.stat().st_size
                    seg_p.unlink(missing_ok=True)
                    reclaimed_bytes += sz

                self.rec_repo.delete_segment_record(seg_id)
                purged_count += 1
            except Exception as e:
                logger.warning("Could not delete expired segment %s: %s", seg_id, e)

        if purged_count > 0:
            reclaimed_mb = reclaimed_bytes / (1024 * 1024)
            self.audit_repo.log_action(
                actor="RetentionService",
                action="RETENTION_PURGE",
                entity_type="recordings",
                entity_id="bulk",
                details=f"Recycled {purged_count} unlocked segments ({reclaimed_mb:.1f} MB reclaimed)",
            )
            logger.info("Retention pass complete: purged %d segments, freed %.1f MB", purged_count, reclaimed_mb)

        return purged_count

    def stop(self) -> None:
        self._is_running = False
