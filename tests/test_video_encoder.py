"""Unit tests verifying H.264 video encoding, playability, and forensic clip generation."""

from pathlib import Path

import numpy as np

from src.fasttrack.core.types import BoundingBox, Detection
from src.fasttrack.storage.video_encoder import FFmpegVideoEncoder


def test_ffmpeg_encoder_playable_h264(tmp_path: Path):
    encoder = FFmpegVideoEncoder()

    # Generate 30 dummy frames (1.2 seconds at 25 fps)
    frames = [
        np.full((240, 320, 3), fill_value=i * 8 % 255, dtype=np.uint8)
        for i in range(30)
    ]
    out_file = tmp_path / "test_playable.mp4"

    meta = encoder.encode_h264_clip(frames, out_file, fps=25.0)

    # Assertions on metadata
    assert out_file.exists()
    assert out_file.stat().st_size > 0
    assert meta["codec"] == "h264"
    assert meta["verification_status"] == "verified"
    assert len(meta["sha256"]) == 64

    # Direct FFprobe playability assertion
    is_valid, msg = encoder.verify_clip_playability(out_file)
    assert is_valid is True
    assert "verified playable" in msg.lower()


def test_incident_clips_generation_raw_and_annotated(tmp_path: Path):
    encoder = FFmpegVideoEncoder()

    frames = [
        np.zeros((300, 400, 3), dtype=np.uint8)
        for _ in range(25)
    ]

    detections_per_frame = [
        [
            Detection(
                box=BoundingBox(x1=50, y1=50, x2=150, y2=150),
                class_name="knife",
                confidence=0.88,
                is_threat=True,
            )
        ]
        for _ in range(25)
    ]

    result = encoder.generate_incident_clips(
        frames=frames,
        incident_id="inc-test-video-01",
        camera_id="cam-sector-4",
        detections_per_frame=detections_per_frame,
        fps=25.0,
        base_dir=str(tmp_path / "clips"),
    )

    raw_path = Path(result["raw_clip_path"])
    ann_path = Path(result["annotated_clip_path"])
    thumb_path = Path(result["thumbnail_path"])

    assert raw_path.exists()
    assert ann_path.exists()
    assert thumb_path.exists()

    # Both clips must pass FFprobe playability
    assert encoder.verify_clip_playability(raw_path)[0] is True
    assert encoder.verify_clip_playability(ann_path)[0] is True
