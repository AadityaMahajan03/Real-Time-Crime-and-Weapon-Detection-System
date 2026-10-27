"""Seed realistic sample data and a synthetic video clip for demo mode.

All generated records are clearly watermarked or tagged with '[DEMO DATA]'.
Enables immediate inspection of UI dashboards without live cameras or GPU models.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.fasttrack.core.config import get_config
from src.fasttrack.core.logging import get_logger, setup_logging
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner

setup_logging(level="INFO")
logger = get_logger("scripts.seed_demo")


def generate_synthetic_demo_video(output_path: Path, num_frames: int = 150, fps: int = 25) -> None:
    """Generate a clean synthetic MP4 video clip with timestamp and DEMO DATA watermark."""
    try:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        width, height = 640, 480
        writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))

        for i in range(num_frames):
            # Create a dark surveillance hallway background
            frame = np.full((height, width, 3), 25, dtype=np.uint8)

            # Draw simulated architectural corridor lines
            cv2.line(frame, (0, 0), (200, 180), (60, 60, 60), 2)
            cv2.line(frame, (640, 0), (440, 180), (60, 60, 60), 2)
            cv2.line(frame, (0, 480), (200, 300), (60, 60, 60), 2)
            cv2.line(frame, (640, 480), (440, 300), (60, 60, 60), 2)
            cv2.rectangle(frame, (200, 180), (440, 300), (45, 45, 45), -1)

            # Simulated moving object / candidate detection box in the center
            obj_x = int(270 + 40 * np.sin(i / 10.0))
            obj_y = int(220 + 20 * np.cos(i / 10.0))
            cv2.rectangle(frame, (obj_x, obj_y), (obj_x + 50, obj_y + 80), (50, 150, 255), 2)

            # Burn camera header & timestamp
            ts_str = (datetime.now(timezone.utc) - timedelta(seconds=(num_frames - i) / fps)).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            cv2.putText(
                frame,
                f"CAM-01 [SOUTH CORRIDOR] | {ts_str}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 220, 240),
                1,
                cv2.LINE_AA,
            )

            # Burn high-visibility DEMO DATA watermark
            cv2.putText(
                frame,
                "[DEMO DATA - SIMULATED FOOTAGE]",
                (15, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)

        writer.release()
        logger.info("Generated synthetic demo video at %s (%d frames).", output_path, num_frames)
    except Exception as e:
        logger.warning("Could not generate synthetic MP4 via OpenCV: %s", e)


def generate_synthetic_demo_snapshot(output_path: Path) -> None:
    """Generate a high-fidelity synthetic weapon detection keyframe snapshot."""
    try:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = 640, 480
        frame = np.full((height, width, 3), 25, dtype=np.uint8)

        # Draw corridor walls and floor
        cv2.line(frame, (0, 0), (200, 180), (60, 60, 60), 2)
        cv2.line(frame, (640, 0), (440, 180), (60, 60, 60), 2)
        cv2.line(frame, (0, 480), (200, 300), (60, 60, 60), 2)
        cv2.line(frame, (640, 480), (440, 300), (60, 60, 60), 2)
        cv2.rectangle(frame, (200, 180), (440, 300), (45, 45, 45), -1)

        # Bounding box on detected threat
        cv2.rectangle(frame, (270, 210), (350, 330), (0, 0, 230), 2)
        # Tactical corners
        cv2.line(frame, (270, 210), (285, 210), (255, 255, 255), 2)
        cv2.line(frame, (270, 210), (270, 225), (255, 255, 255), 2)
        cv2.line(frame, (350, 210), (335, 210), (255, 255, 255), 2)
        cv2.line(frame, (350, 210), (350, 225), (255, 255, 255), 2)
        cv2.line(frame, (270, 330), (285, 330), (255, 255, 255), 2)
        cv2.line(frame, (270, 330), (270, 315), (255, 255, 255), 2)
        cv2.line(frame, (350, 330), (335, 330), (255, 255, 255), 2)
        cv2.line(frame, (350, 330), (350, 315), (255, 255, 255), 2)

        # Label tag
        cv2.rectangle(frame, (270, 188), (360, 210), (0, 0, 230), -1)
        cv2.putText(frame, "KNIFE 0.78", (274, 204), cv2.FONT_HERSHEY_DUPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Header HUD
        cv2.putText(frame, "FAST-TRACK AI | FORENSIC EVIDENCE KEYFRAME [DEMO]", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, "[DEMO DATA - TAMPER-EVIDENT RECORD]", (15, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1, cv2.LINE_AA)

        cv2.imwrite(str(output_path), frame)
        logger.info("Generated synthetic demo snapshot at %s.", output_path)
    except Exception as e:
        logger.warning("Could not generate synthetic snapshot via OpenCV: %s", e)


def compute_file_sha256(path: Path) -> tuple[str, int]:
    """Return (sha256_hex, file_size_bytes) for file."""
    import hashlib
    if not path.exists():
        return "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", 0
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest(), path.stat().st_size


def seed_database(db: DatabaseManager) -> None:
    """Populate database tables with mock cameras, incidents, events, and users."""
    now = datetime.now(timezone.utc)

    # 1. Cameras
    cameras = [
        (
            "cam-demo-01",
            "Main Gate - South Entrance [DEMO DATA]",
            "file",
            "data/demo_corridor.mp4",
            1,
            json.dumps(
                [
                    {
                        "name": "Entrance Perimeter",
                        "zone_type": "include",
                        "polygon": [[0.05, 0.10], [0.95, 0.10], [0.95, 0.90], [0.05, 0.90]],
                    }
                ]
            ),
            json.dumps({"weapon_confidence": 0.50, "violence_confidence": 0.65}),
            json.dumps({"active_24_7": True}),
            (now - timedelta(days=5)).isoformat(),
        ),
        (
            "cam-demo-02",
            "East Parking Perimeter [DEMO DATA]",
            "webcam",
            "0",
            1,
            "[]",
            json.dumps({"weapon_confidence": 0.55, "violence_confidence": 0.70}),
            json.dumps({"active_24_7": True}),
            (now - timedelta(days=4)).isoformat(),
        ),
        (
            "cam-demo-03",
            "Server Room Vault [DEMO DATA]",
            "rtsp",
            "rtsp://***@192.168.1.150:554/live",
            0,
            "[]",
            json.dumps({"weapon_confidence": 0.60, "violence_confidence": 0.60}),
            json.dumps({"active_24_7": False, "active_hours": {"start": "18:00", "end": "08:00"}}),
            (now - timedelta(days=3)).isoformat(),
        ),
    ]

    for cam in cameras:
        db.execute_non_query(
            """
            INSERT OR REPLACE INTO cameras
            (id, name, source_type, source_ref_masked, enabled, zones_json, thresholds_json, schedule_json, created_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            cam,
        )

    # 2. Evidence with authentic SHA-256 and size from disk
    snap_path = Path("evidence/snapshots/demo_knife_snapshot.jpg")
    snap_sha, snap_size = compute_file_sha256(snap_path)

    clip_path = Path("data/demo_corridor.mp4")
    clip_sha, clip_size = compute_file_sha256(clip_path)

    evidence_items = [
        (
            "ev-001",
            "snapshot",
            "evidence/snapshots/demo_knife_snapshot.jpg",
            snap_sha,
            snap_size,
            0,
            (now - timedelta(minutes=45)).isoformat(),
            (now + timedelta(days=29)).isoformat(),
        ),
        (
            "ev-002",
            "clip",
            "data/demo_corridor.mp4",
            clip_sha,
            clip_size,
            0,
            (now - timedelta(minutes=20)).isoformat(),
            (now + timedelta(days=29)).isoformat(),
        ),
    ]

    for ev in evidence_items:
        db.execute_non_query(
            """
            INSERT OR REPLACE INTO evidence
            (id, kind, path, sha256, size_bytes, encrypted, created_utc, expires_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            ev,
        )

    # 3. Incidents
    incidents = [
        (
            "inc-demo-101",
            "cam-demo-01",
            "HIGH",
            "new",
            (now - timedelta(minutes=22)).isoformat(),
            (now - timedelta(minutes=21, seconds=35)).isoformat(),
            0.78,
            0.84,
            "[DEMO DATA] Co-occurring possible violence and detected object: knife (0.78). Immediate operator triage required.",
        ),
        (
            "inc-demo-102",
            "cam-demo-01",
            "MEDIUM",
            "reviewed",
            (now - timedelta(hours=2, minutes=15)).isoformat(),
            (now - timedelta(hours=2, minutes=14, seconds=40)).isoformat(),
            0.72,
            0.20,
            "[DEMO DATA] Detected object: knife (0.72) with track persistence across 5 frames.",
        ),
        (
            "inc-demo-103",
            "cam-demo-02",
            "LOW",
            "false_alarm",
            (now - timedelta(hours=5)).isoformat(),
            (now - timedelta(hours=4, minutes=59)).isoformat(),
            0.35,
            0.52,
            "[DEMO DATA] Unconfirmed rapid movement flagged as possible violence (0.52). Human review: operator confirmed maintenance crew.",
        ),
    ]

    for inc in incidents:
        db.execute_non_query(
            """
            INSERT OR REPLACE INTO incidents
            (id, camera_id, severity, status, start_utc, end_utc, peak_weapon_score, peak_violence_score, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            inc,
        )

    # 4. Events
    events = [
        (
            "evt-001",
            "inc-demo-101",
            "cam-demo-01",
            (now - timedelta(minutes=21, seconds=55)).isoformat(),
            "weapon_detection",
            "knife",
            0.78,
            0.76,
            json.dumps([280.0, 230.0, 330.0, 310.0]),
            12,
            "threat-yolo-v1",
            "ev-001",
        ),
        (
            "evt-002",
            "inc-demo-101",
            "cam-demo-01",
            (now - timedelta(minutes=21, seconds=50)).isoformat(),
            "possible_violence",
            "physical_altercation",
            0.84,
            0.82,
            None,
            None,
            "s3d-kinetics-v1",
            "ev-002",
        ),
        (
            "evt-003",
            "inc-demo-102",
            "cam-demo-01",
            (now - timedelta(hours=2, minutes=15)).isoformat(),
            "weapon_detection",
            "knife",
            0.72,
            0.70,
            json.dumps([290.0, 220.0, 340.0, 305.0]),
            7,
            "threat-yolo-v1",
            "ev-001",
        ),
        (
            "evt-004",
            "inc-demo-103",
            "cam-demo-02",
            (now - timedelta(hours=5)).isoformat(),
            "possible_violence",
            "rapid_movement",
            0.52,
            0.48,
            None,
            None,
            "s3d-kinetics-v1",
            "ev-002",
        ),
    ]

    for evt in events:
        db.execute_non_query(
            """
            INSERT OR REPLACE INTO events
            (id, incident_id, camera_id, ts_utc, event_type, class_name, raw_score, calibrated_score, bbox_json, track_id, model_version_id, evidence_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            evt,
        )

    # 5. Review Actions
    reviews = [
        (
            "rev-001",
            "inc-demo-102",
            "operator_john",
            "mark_reviewed",
            "[DEMO DATA] Verified snapshot. Object carried appears to be a tool; notified floor supervisor.",
            (now - timedelta(hours=2, minutes=5)).isoformat(),
        ),
        (
            "rev-002",
            "inc-demo-103",
            "operator_sarah",
            "mark_false_alarm",
            "[DEMO DATA] Maintenance ladder movement triggered optical motion; marked false alarm.",
            (now - timedelta(hours=4, minutes=50)).isoformat(),
        ),
    ]

    for rev in reviews:
        db.execute_non_query(
            """
            INSERT OR REPLACE INTO review_actions
            (id, incident_id, user_id, action, note, ts_utc)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            rev,
        )

    # 6. Users (admin, reviewer, viewer) with salted dummy hashes
    users = [
        ("u-admin", "admin", "$argon2id$v=19$m=65536,t=3,p=4$demo$demo_hash_admin", "admin", 1, now.isoformat()),
        ("u-rev", "reviewer", "$argon2id$v=19$m=65536,t=3,p=4$demo$demo_hash_rev", "reviewer", 1, now.isoformat()),
        ("u-view", "viewer", "$argon2id$v=19$m=65536,t=3,p=4$demo$demo_hash_view", "viewer", 1, None),
    ]

    for u in users:
        db.execute_non_query(
            """
            INSERT OR REPLACE INTO users (id, username, password_hash, role, active, last_login_utc)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            u,
        )

    logger.info("Successfully seeded demo database tables.")


def main() -> None:
    config = get_config()
    db_path = config.storage.db_path
    db = DatabaseManager(db_path)

    # Ensure schema migrations are applied first
    runner = MigrationRunner(db)
    runner.run_migrations()

    # Generate sample video and evidence snapshot
    video_path = Path("data/demo_corridor.mp4")
    generate_synthetic_demo_video(video_path)

    snapshot_path = Path("evidence/snapshots/demo_knife_snapshot.jpg")
    generate_synthetic_demo_snapshot(snapshot_path)

    # Seed mock database items
    seed_database(db)
    print("Demo data successfully generated and seeded into:", db_path)


if __name__ == "__main__":
    main()
