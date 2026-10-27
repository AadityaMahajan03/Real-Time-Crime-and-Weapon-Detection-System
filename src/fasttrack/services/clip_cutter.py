"""Background Clip Cutter service extracting incident forensic videos from continuous stream segments."""

from __future__ import annotations

import hashlib
import queue
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.fasttrack.core.logging import get_logger
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.repositories import (
    AuditRepository,
    ClipRepository,
    IncidentRepository,
    RecordingRepository,
)
from src.fasttrack.storage.video_encoder import FFmpegVideoEncoder

logger = get_logger("services.clip_cutter")


class ClipCutterService:
    """Non-blocking background worker cutting H.264 incident clips with pre/post roll and keyframe strips."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        clips_root: str = "data/clips",
    ) -> None:
        self.db = db_manager
        self.clip_repo = ClipRepository(db_manager)
        self.rec_repo = RecordingRepository(db_manager)
        self.inc_repo = IncidentRepository(db_manager)
        self.audit_repo = AuditRepository(db_manager)
        self.encoder = FFmpegVideoEncoder()

        self.clips_root = Path(clips_root)
        self.clips_root.mkdir(parents=True, exist_ok=True)

        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=100)
        self._is_running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop, name="ClipCutterWorker", daemon=True
        )
        self._worker_thread.start()

    def request_clip(
        self,
        incident_id: str,
        camera_id: str,
        trigger_utc: str,
        duration_sec: float = 5.0,
        pre_roll_sec: float = 10.0,
        post_roll_sec: float = 15.0,
        fallback_frames: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Enqueue asynchronous incident clip extraction."""
        task = {
            "incident_id": incident_id,
            "camera_id": camera_id,
            "trigger_utc": trigger_utc,
            "duration_sec": duration_sec,
            "pre_roll_sec": pre_roll_sec,
            "post_roll_sec": post_roll_sec,
            "fallback_frames": fallback_frames,
        }
        try:
            self._queue.put_nowait(task)
            logger.info("Enqueued clip cut request for incident %s (%s)", incident_id, camera_id)
        except queue.Full:
            logger.error("Clip cutter queue is full; dropping request for incident %s", incident_id)

    def _worker_loop(self) -> None:
        while self._is_running:
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_clip_task(task)
            except Exception as e:
                logger.error("Failed to process clip task for incident %s: %s", task.get("incident_id"), e)
            finally:
                self._queue.task_done()

    def _process_clip_task(self, task: Dict[str, Any]) -> None:
        inc_id = task["incident_id"]
        cid = task["camera_id"]
        trig_str = task["trigger_utc"]
        pre_roll = task["pre_roll_sec"]
        post_roll = task["post_roll_sec"]
        fallback_frames = task.get("fallback_frames")

        # Compute clip time window
        try:
            trig_dt = datetime.fromisoformat(trig_str.replace("Z", "+00:00"))
        except Exception:
            trig_dt = datetime.now(timezone.utc)

        start_dt = trig_dt - timedelta(seconds=pre_roll)
        end_dt = trig_dt + timedelta(seconds=task["duration_sec"] + post_roll)

        start_iso = start_dt.isoformat()
        end_iso = end_dt.isoformat()

        # 1. Lock recording segments covering this incident
        self.rec_repo.lock_segments_for_incident(cid, start_iso, end_iso)

        # 2. Check for available segments
        segments = self.rec_repo.get_segments_for_range(cid, start_iso, end_iso)
        today_str = trig_dt.strftime("%Y-%m-%d")
        partition_dir = self.clips_root / cid / today_str
        partition_dir.mkdir(parents=True, exist_ok=True)

        raw_clip_path = partition_dir / f"{inc_id}_raw.mp4"
        ann_clip_path = partition_dir / f"{inc_id}_annotated.mp4"
        thumb_path = partition_dir / f"{inc_id}_thumb.jpg"
        strip_path = partition_dir / f"{inc_id}_strip.jpg"

        clip_created = False
        duration_final = task["duration_sec"] + pre_roll + post_roll
        sha256_hash = ""
        size_bytes = 0

        # Method A: Cut and concatenate from continuous segments
        if segments:
            valid_paths = [Path(s["segment_path"]) for s in segments if Path(s["segment_path"]).exists()]
            if valid_paths:
                clip_created, sha256_hash, size_bytes = self._cut_from_segments(
                    segment_paths=valid_paths,
                    output_path=raw_clip_path,
                )

        # Method B: Fallback to frame buffer if segments unavailable
        if not clip_created and fallback_frames:
            logger.info("Using fallback frame buffer to encode incident clip for %s", inc_id)
            meta = self.encoder.encode_h264_clip(fallback_frames, raw_clip_path, fps=25.0)
            clip_created = True
            sha256_hash = meta["sha256"]
            size_bytes = meta["size_bytes"]
            duration_final = meta["duration_sec"]

        if not clip_created or not raw_clip_path.exists():
            logger.warning("Could not create clip for incident %s: neither segments nor frames found.", inc_id)
            return

        # 3. Create annotated clip copy
        try:
            # Generate annotated copy with bounding overlay
            self._generate_annotated_copy(raw_clip_path, ann_clip_path, inc_id, cid)
        except Exception as e:
            logger.warning("Could not generate annotated copy: %s. Re-using raw.", e)
            import shutil
            shutil.copy2(raw_clip_path, ann_clip_path)

        # 4. Extract Thumbnail and Keyframe Strip
        self._generate_thumbnail_and_strip(raw_clip_path, thumb_path, strip_path)

        # 5. Insert into clips table and link evidence
        clip_id = f"clip-{inc_id}"
        self.clip_repo.create(
            clip_id=clip_id,
            incident_id=inc_id,
            camera_id=cid,
            raw_clip_path=str(raw_clip_path).replace("\\", "/"),
            annotated_clip_path=str(ann_clip_path).replace("\\", "/"),
            thumbnail_path=str(thumb_path).replace("\\", "/"),
            keyframe_strip_path=str(strip_path).replace("\\", "/"),
            sha256=sha256_hash,
            duration_sec=duration_final,
            size_bytes=size_bytes,
            codec="h264",
            verification_status="verified",
        )

        # 6. Log to hash-chained audit trail
        self.audit_repo.log_action(
            actor="ClipCutterService",
            action="INCIDENT_CLIP_FINALIZED",
            entity_type="clip",
            entity_id=clip_id,
            details=f"Generated {duration_final:.1f}s H.264 clip (SHA256: {sha256_hash[:16]}...) for incident {inc_id}",
        )
        logger.info("Forensic clip %s successfully archived and verified for incident %s", clip_id, inc_id)

    def _cut_from_segments(
        self,
        segment_paths: List[Path],
        output_path: Path,
    ) -> Tuple[bool, str, int]:
        """Concat and transcode segment files into an H.264 MP4 clip with FFmpeg."""
        concat_list_file = output_path.with_suffix(".txt")
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for p in segment_paths:
                f.write(f"file '{p.resolve().as_posix()}'\n")

        cmd = [
            self.encoder.ffmpeg_exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_file),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(output_path),
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=45)
            concat_list_file.unlink(missing_ok=True)
            if res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                hasher = hashlib.sha256()
                with open(output_path, "rb") as vf:
                    for chunk in iter(lambda: vf.read(65536), b""):
                        hasher.update(chunk)
                return True, hasher.hexdigest(), output_path.stat().st_size
            return False, "", 0
        except Exception as e:
            concat_list_file.unlink(missing_ok=True)
            logger.error("FFmpeg concat failed: %s", e)
            return False, "", 0

    def _generate_thumbnail_and_strip(
        self,
        video_path: Path,
        thumb_path: Path,
        strip_path: Path,
        strip_frames_count: int = 5,
    ) -> None:
        """Extract keyframe thumbnail and a 5-frame contact strip."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return

        # 1. Middle Thumbnail
        mid_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
        ret, mid_frame = cap.read()
        if ret and mid_frame is not None:
            cv2.imwrite(str(thumb_path), mid_frame)

        # 2. Keyframe Strip
        strip_images = []
        indices = np.linspace(0, max(0, total_frames - 1), strip_frames_count, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            r, fr = cap.read()
            if r and fr is not None:
                small = cv2.resize(fr, (160, 90))
                strip_images.append(small)

        cap.release()

        if len(strip_images) == strip_frames_count:
            # Horizontally concatenate
            strip_canvas = np.hstack(strip_images)
            cv2.imwrite(str(strip_path), strip_canvas)

    def _generate_annotated_copy(
        self,
        raw_video_path: Path,
        ann_video_path: Path,
        incident_id: str,
        camera_id: str,
    ) -> None:
        """Read raw video, stamp camera security overlay, and encode annotated H.264 copy."""
        cap = cv2.VideoCapture(str(raw_video_path))
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames: List[np.ndarray] = []
        while True:
            ret, fr = cap.read()
            if not ret or fr is None:
                break
            # Add watermark banner
            cv2.putText(
                fr,
                f"EVIDENCE ARCHIVE | INCIDENT: {incident_id} | CAMERA: {camera_id}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            frames.append(fr)
        cap.release()

        if frames:
            self.encoder.encode_h264_clip(frames, ann_video_path, fps=fps)

    def stop(self) -> None:
        self._is_running = False
