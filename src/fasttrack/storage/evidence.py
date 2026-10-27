"""Cryptographic evidence writer with atomic persistence, SHA-256 integrity, and disk protection."""

from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import Detection, EvidenceItem
from src.fasttrack.storage.db import DatabaseManager

logger = get_logger("storage.evidence")

# Default minimum disk space before degrading to snapshot-only mode (2.0 GB)
DEFAULT_MIN_FREE_DISK_BYTES = 2 * 1024 * 1024 * 1024
# Critical minimum disk space before blocking all writes (50 MB)
CRITICAL_MIN_FREE_DISK_BYTES = 50 * 1024 * 1024


class EvidenceWriter:
    """Manages secure, atomic persistence of snapshots and clips with cryptographic verification."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        base_dir: str = "evidence",
        min_free_disk_bytes: int = DEFAULT_MIN_FREE_DISK_BYTES,
        clock: Optional[Clock] = None,
    ) -> None:
        self.db_manager = db_manager
        self.base_dir = Path(base_dir)
        self.min_free_disk_bytes = min_free_disk_bytes
        self.clock = clock or SystemClock()

        self.snapshots_dir = self.base_dir / "snapshots"
        self.clips_dir = self.base_dir / "clips"

        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        self._face_cascade: Optional[cv2.CascadeClassifier] = None

    def _get_face_cascade(self) -> Optional[cv2.CascadeClassifier]:
        """Lazy loader for OpenCV frontal face cascade classifier."""
        if self._face_cascade is None:
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    self._face_cascade = cascade
            except Exception as e:
                logger.warning("Failed to load OpenCV face cascade for privacy blurring: %s", e)
        return self._face_cascade

    def get_free_disk_bytes(self) -> int:
        """Query available disk space on the evidence partition."""
        try:
            return shutil.disk_usage(self.base_dir).free
        except Exception as e:
            logger.warning("Could not query disk usage on %s: %s", self.base_dir, e)
            return 10 * 1024 * 1024 * 1024  # Default fallback 10GB

    def is_clip_writing_permitted(self) -> bool:
        """Check if remaining disk space exceeds safety threshold for video clip encoding."""
        free_bytes = self.get_free_disk_bytes()
        if free_bytes < self.min_free_disk_bytes:
            logger.warning(
                "Disk space low (%.2f MB remaining < %.2f MB threshold). Degraded to snapshot-only mode.",
                free_bytes / (1024 * 1024),
                self.min_free_disk_bytes / (1024 * 1024),
            )
            return False
        return True

    def _apply_privacy_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur over detected facial regions to prevent biometric identification."""
        cascade = self._get_face_cascade()
        if cascade is None:
            return image

        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            for x, y, w, h in faces:
                roi = image[y : y + h, x : x + w]
                # Apply strong Gaussian blur
                ksize = max(31, (w // 2) | 1)
                blurred = cv2.GaussianBlur(roi, (ksize, ksize), 30)
                image[y : y + h, x : x + w] = blurred
        except Exception as e:
            logger.warning("Privacy face blur pass failed: %s", e)

        return image

    def _burn_annotations(
        self,
        image: np.ndarray,
        camera_id: str,
        camera_name: str,
        timestamp_utc: datetime,
        detections: Optional[List[Detection]] = None,
    ) -> np.ndarray:
        """Annotate snapshot image with bounding boxes, labels, and tamper-resistant metadata banner."""
        h, w = image.shape[:2]

        # 1. Burn bounding boxes
        if detections:
            for det in detections:
                x1, y1, x2, y2 = det.box.to_int_tuple()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                # Color: Orange/Red for weapon, Cyan for other
                color = (0, 80, 255) if "knife" in det.class_name.lower() or "gun" in det.class_name.lower() else (255, 180, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Label pill
                label = f"{det.class_name} ({det.confidence:.2f})"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (lw, lh), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Pill background
                pill_y1 = max(0, y1 - lh - 6)
                pill_y2 = y1
                cv2.rectangle(image, (x1, pill_y1), (x1 + lw + 6, pill_y2), color, -1)
                cv2.putText(
                    image,
                    label,
                    (x1 + 3, pill_y2 - baseline),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        # 2. Burn operational banner at top
        banner_h = 28
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (11, 15, 23), -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

        cam_text = f"FAST-TRACK AI  |  CAMERA: {camera_name or camera_id}  |  {timestamp_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        cv2.putText(
            image,
            cam_text,
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (34, 211, 238),  # Cyan
            1,
            cv2.LINE_AA,
        )

        return image

    def write_snapshot(
        self,
        frame: np.ndarray,
        camera_id: str,
        camera_name: str = "",
        detections: Optional[List[Detection]] = None,
        privacy_blur: bool = False,
        expires_utc: Optional[datetime] = None,
    ) -> EvidenceItem:
        """Atomically persist an annotated evidence snapshot with SHA-256 cryptographic hash."""
        free_bytes = self.get_free_disk_bytes()
        if free_bytes < CRITICAL_MIN_FREE_DISK_BYTES:
            raise RuntimeError(f"Critical disk space starvation ({free_bytes} bytes free). Snapshot write blocked.")

        now_utc = self.clock.now_utc()
        ev_id = f"ev-snap-{uuid.uuid4().hex[:12]}"
        final_path = self.snapshots_dir / f"{ev_id}.jpg"
        temp_path = self.snapshots_dir / f"{ev_id}.tmp"

        # Copy image for safe processing
        img_copy = frame.copy()

        # Privacy face blurring (if enabled)
        if privacy_blur:
            img_copy = self._apply_privacy_blur(img_copy)

        # Bounding box & banner annotation
        img_copy = self._burn_annotations(
            img_copy,
            camera_id=camera_id,
            camera_name=camera_name,
            timestamp_utc=now_utc,
            detections=detections,
        )

        # Encode to JPEG bytes
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        success, encoded_buf = cv2.imencode(".jpg", img_copy, encode_params)
        if not success:
            raise RuntimeError(f"Failed to encode frame into JPEG for evidence {ev_id}")

        img_bytes = encoded_buf.tobytes()
        sha256_hash = hashlib.sha256(img_bytes).hexdigest()
        size_bytes = len(img_bytes)

        # Atomic write: write to .tmp file, flush, then replace
        try:
            with open(temp_path, "wb") as f:
                f.write(img_bytes)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, final_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Failed to write snapshot evidence atomically: {e}")

        # Persist to DB
        exp_iso = expires_utc.isoformat() if expires_utc else None
        if self.db_manager:
            self.db_manager.execute_non_query(
                """
                INSERT INTO evidence (id, kind, path, sha256, size_bytes, encrypted, created_utc, expires_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ev_id,
                    "snapshot",
                    str(final_path),
                    sha256_hash,
                    size_bytes,
                    0,
                    now_utc.isoformat(),
                    exp_iso,
                ),
            )

        logger.info(
            "Evidence snapshot written: %s (SHA-256: %s, %d bytes)",
            ev_id,
            sha256_hash[:12],
            size_bytes,
        )

        return EvidenceItem(
            id=ev_id,
            kind="snapshot",
            path=str(final_path),
            sha256=sha256_hash,
            size_bytes=size_bytes,
            encrypted=False,
            created_utc=now_utc,
            expires_utc=expires_utc,
        )

    def write_clip(
        self,
        frames: List[np.ndarray],
        camera_id: str,
        fps: float = 15.0,
        privacy_blur: bool = False,
        expires_utc: Optional[datetime] = None,
    ) -> Optional[EvidenceItem]:
        """Atomically persist a video evidence clip if disk space permits."""
        if not frames:
            logger.warning("Empty frame list supplied to write_clip; skipping.")
            return None

        if not self.is_clip_writing_permitted():
            return None

        now_utc = self.clock.now_utc()
        ev_id = f"ev-clip-{uuid.uuid4().hex[:12]}"
        final_path = self.clips_dir / f"{ev_id}.mp4"

        try:
            from src.fasttrack.storage.video_encoder import FFmpegVideoEncoder
            encoder = FFmpegVideoEncoder()
            frames_to_write = [self._apply_privacy_blur(f) if privacy_blur else f for f in frames]
            meta = encoder.encode_h264_clip(frames_to_write, final_path, fps=max(1.0, float(fps)))
            sha256_hash = meta["sha256"]
            size_bytes = meta["size_bytes"]
        except Exception as e:
            logger.warning("FFmpeg encoder failed for %s, falling back to basic video writer: %s", ev_id, e)
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            temp_path = self.clips_dir / f"{ev_id}.tmp.mp4"
            writer = cv2.VideoWriter(str(temp_path), fourcc, max(1.0, float(fps)), (w, h))
            try:
                for fr in frames:
                    writer.write(self._apply_privacy_blur(fr) if privacy_blur else fr)
                writer.release()
                if not temp_path.exists() or temp_path.stat().st_size == 0:
                    return None
                os.replace(temp_path, final_path)
                hasher = hashlib.sha256()
                with open(final_path, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        hasher.update(chunk)
                sha256_hash = hasher.hexdigest()
                size_bytes = final_path.stat().st_size
            except Exception as e2:
                writer.release()
                logger.error("Failed to encode fallback clip %s: %s", ev_id, e2)
                return None

        exp_iso = expires_utc.isoformat() if expires_utc else None
        if self.db_manager:
            self.db_manager.execute_non_query(
                """
                INSERT INTO evidence (id, kind, path, sha256, size_bytes, encrypted, created_utc, expires_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ev_id,
                    "clip",
                    str(final_path),
                    sha256_hash,
                    size_bytes,
                    0,
                    now_utc.isoformat(),
                    exp_iso,
                ),
            )

        logger.info(
            "Evidence video clip written: %s (SHA-256: %s, %d bytes, %d frames)",
            ev_id,
            sha256_hash[:12],
            size_bytes,
            len(frames),
        )

        return EvidenceItem(
            id=ev_id,
            kind="clip",
            path=str(final_path),
            sha256=sha256_hash,
            size_bytes=size_bytes,
            encrypted=False,
            created_utc=now_utc,
            expires_utc=expires_utc,
        )

    def verify_evidence_integrity(self, evidence_id: str) -> Tuple[bool, str]:
        """Verify cryptographic integrity of evidence on disk against stored SHA-256 hash."""
        if not self.db_manager:
            return False, "Database manager not attached."

        rows = self.db_manager.execute_query(
            "SELECT path, sha256, size_bytes FROM evidence WHERE id = ?",
            (evidence_id,),
        )
        if not rows:
            return False, f"Evidence {evidence_id} not found in database."

        record = rows[0]
        file_path = Path(record["path"])
        stored_sha = record["sha256"]

        if not file_path.exists():
            return False, f"Evidence file on disk missing: {file_path}"

        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        actual_sha = hasher.hexdigest()

        if hmac.compare_digest(stored_sha, actual_sha):
            return True, f"Integrity verified (SHA-256: {actual_sha[:12]}... matches stored record)."
        else:
            return False, f"Integrity failure: computed SHA-256 {actual_sha} does NOT match stored {stored_sha}."
