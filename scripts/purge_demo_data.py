"""Safe demo-data purge script with automatic database backup.

Removes synthetic '[DEMO DATA]' artifacts, demo cameras, mock incidents,
and resets the system to clean operational state.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.fasttrack.core.config import get_config
from src.fasttrack.core.logging import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("scripts.purge_demo")


def purge_demo_data(create_backup: bool = True) -> None:
    config = get_config()
    db_path = Path(config.storage.db_path)

    if not db_path.exists():
        logger.warning("Database at %s does not exist. Nothing to purge.", db_path)
        return

    # 1. Automatic Backup
    if create_backup:
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"fasttrack_backup_pre_purge_{timestamp}.db"
        shutil.copy2(db_path, backup_path)
        logger.info("Created pre-purge database backup at: %s", backup_path)

    # 2. Execute safe purge
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    purged_counts = {}

    try:
        # A. Delete demo incidents & cascading events
        cur.execute("SELECT id FROM incidents WHERE id LIKE 'inc-demo%' OR summary LIKE '%[DEMO DATA]%';")
        demo_inc_ids = [r[0] for r in cur.fetchall()]

        if demo_inc_ids:
            ph = ",".join(["?"] * len(demo_inc_ids))
            cur.execute(f"DELETE FROM review_actions WHERE incident_id IN ({ph});", demo_inc_ids)
            cur.execute(f"DELETE FROM events WHERE incident_id IN ({ph});", demo_inc_ids)
            cur.execute(f"DELETE FROM detections WHERE incident_id IN ({ph});", demo_inc_ids)
            cur.execute(f"DELETE FROM clips WHERE incident_id IN ({ph});", demo_inc_ids)
            cur.execute(f"DELETE FROM alarms WHERE incident_id IN ({ph});", demo_inc_ids)
            cur.execute(f"DELETE FROM incident_evidence WHERE incident_id IN ({ph});", demo_inc_ids)
            cur.execute(f"DELETE FROM incidents WHERE id IN ({ph});", demo_inc_ids)
            purged_counts["incidents"] = len(demo_inc_ids)

        # B. Delete demo evidence items
        cur.execute("SELECT id, path FROM evidence WHERE id LIKE 'ev-demo%' OR path LIKE '%demo%';")
        demo_evs = cur.fetchall()
        for ev_id, path_str in demo_evs:
            cur.execute("DELETE FROM incident_evidence WHERE evidence_id = ?;", (ev_id,))
            cur.execute("DELETE FROM evidence WHERE id = ?;", (ev_id,))
        purged_counts["evidence"] = len(demo_evs)

        # C. Delete demo cameras
        cur.execute("SELECT id FROM cameras WHERE id LIKE 'cam-demo%' OR name LIKE '%Demo%';")
        demo_cams = [r[0] for r in cur.fetchall()]
        if demo_cams:
            ph_c = ",".join(["?"] * len(demo_cams))
            cur.execute(f"DELETE FROM cameras WHERE id IN ({ph_c});", demo_cams)
            purged_counts["cameras"] = len(demo_cams)

        # D. Delete demo recordings
        cur.execute("DELETE FROM recordings WHERE camera_id LIKE 'cam-demo%';")
        purged_counts["recordings"] = cur.rowcount

        conn.commit()

        # E. Re-index and optimize database
        cur.execute("VACUUM;")
        conn.commit()

        logger.info("Purge completed successfully! Purged summary: %s", purged_counts)

    except Exception as e:
        conn.rollback()
        logger.error("Purge encountered an error: %s. Rolled back.", e)
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    purge_demo_data(create_backup=True)
