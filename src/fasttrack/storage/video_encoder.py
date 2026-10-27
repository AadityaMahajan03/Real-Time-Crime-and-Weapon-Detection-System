"""H.264 Video Encoder and verification engine for surveillance evidence clips."""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imageio_ffmpeg
import numpy as np

from src.fasttrack.core.errors import FastTrackError
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import Detection

logger = get_logger("storage.video_encoder")


class FFmpegVideoEncoder:
    """Encodes surveillance frame sequences into verified, web-playable H.264 MP4 clips."""

    def __init__(self, ffmpeg_path: Optional[str] = None) -> None:
        try:
            self.ffmpeg_exe = ffmpeg_path or imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:
            raise FastTrackError(f"Failed to resolve FFmpeg binary: {e}")

    def verify_clip_playability(self, video_path: Path) -> Tuple[bool, str]:
        """Verify video container and stream validity using FFmpeg null demuxer."""
        if not video_path.exists() or video_path.stat().st_size == 0:
            return False, "File is missing or 0 bytes."

        cmd = [
            self.ffmpeg_exe,
            "-v",
            "error",
            "-i",
            str(video_path),
            "-f",
            "null",
            "-",
        ]
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                return True, "H.264 stream verified playable."
            return False, f"FFmpeg validation failed: {result.stderr.strip()[:200]}"
        except subprocess.TimeoutExpired:
            return False, "FFmpeg verification timed out."
        except Exception as e:
            return False, f"Verification execution error: {e}"

    def encode_h264_clip(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: float = 25.0,
        crf: int = 23,
    ) -> Dict[str, Any]:
        """Encode RGB/BGR frames to H.264 MP4 with yuv420p and +faststart."""
        if not frames:
            raise FastTrackError("Empty frame list provided for video encoding.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_suffix(".tmp.mp4")

        height, width = frames[0].shape[:2]

        # Ensure width and height are divisible by 2 for yuv420p
        width = width if width % 2 == 0 else width - 1
        height = height if height % 2 == 0 else height - 1

        cmd = [
            self.ffmpeg_exe,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "bgr24",
            "-r",
            f"{max(1.0, float(fps))}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(temp_path),
        ]

        try:
            raw_bytes = bytearray()
            for f in frames:
                if f.shape[1] != width or f.shape[0] != height:
                    f = cv2.resize(f, (width, height))
                raw_bytes.extend(f.tobytes())

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            _, stderr = proc.communicate(input=bytes(raw_bytes), timeout=45)

            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace") if stderr else ""
                raise FastTrackError(f"FFmpeg encoding error (code {proc.returncode}): {err_msg[:200]}")

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise FastTrackError(f"Video encoding failed: {e}")

        # Post-encode verification
        is_valid, msg = self.verify_clip_playability(temp_path)
        if not is_valid:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise FastTrackError(f"Encoded clip failed validation: {msg}")

        # Atomic move to final destination
        os.replace(temp_path, output_path)

        # Calculate SHA-256 hash
        hasher = hashlib.sha256()
        with open(output_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        sha256_hash = hasher.hexdigest()
        size_bytes = output_path.stat().st_size
        duration_sec = len(frames) / max(1.0, float(fps))

        return {
            "path": str(output_path).replace("\\", "/"),
            "sha256": sha256_hash,
            "size_bytes": size_bytes,
            "duration_sec": duration_sec,
            "codec": "h264",
            "verification_status": "verified",
        }

    def generate_incident_clips(
        self,
        frames: List[np.ndarray],
        incident_id: str,
        camera_id: str,
        detections_per_frame: Optional[List[List[Detection]]] = None,
        fps: float = 25.0,
        base_dir: str = "data/clips",
    ) -> Dict[str, Any]:
        """Generate both RAW and ANNOTATED H.264 clips in hierarchical partition."""
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        partition_dir = Path(base_dir) / camera_id / today_str
        partition_dir.mkdir(parents=True, exist_ok=True)

        raw_path = partition_dir / f"{incident_id}_raw.mp4"
        ann_path = partition_dir / f"{incident_id}_annotated.mp4"
        thumb_path = partition_dir / f"{incident_id}_thumb.jpg"

        # 1. Encode RAW clip
        raw_meta = self.encode_h264_clip(frames, raw_path, fps=fps)

        # 2. Render and encode ANNOTATED clip
        annotated_frames: List[np.ndarray] = []
        for idx, frame in enumerate(frames):
            ann = frame.copy()
            if detections_per_frame and idx < len(detections_per_frame):
                for det in detections_per_frame[idx]:
                    b = det.box
                    color = (0, 0, 255) if getattr(det, "is_threat", True) else (255, 200, 0)
                    cv2.rectangle(ann, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), color, 2)
                    label = f"{det.class_name.upper()} {det.confidence*100:.0f}%"
                    cv2.putText(
                        ann,
                        label,
                        (int(b.x1), max(20, int(b.y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
            # Timestamp banner
            cv2.putText(
                ann,
                f"INCIDENT {incident_id} | {camera_id}",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            annotated_frames.append(ann)

        ann_meta = self.encode_h264_clip(annotated_frames, ann_path, fps=fps)

        # 3. Save thumbnail keyframe (middle frame)
        mid_idx = len(annotated_frames) // 2
        cv2.imwrite(str(thumb_path), annotated_frames[mid_idx])

        return {
            "incident_id": incident_id,
            "camera_id": camera_id,
            "raw_clip_path": raw_meta["path"],
            "annotated_clip_path": ann_meta["path"],
            "thumbnail_path": str(thumb_path).replace("\\", "/"),
            "sha256": raw_meta["sha256"],
            "duration_sec": raw_meta["duration_sec"],
            "size_bytes": raw_meta["size_bytes"],
            "codec": "h264",
            "verification_status": "verified",
        }
