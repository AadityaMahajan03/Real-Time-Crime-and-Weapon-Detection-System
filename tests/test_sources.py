"""Comprehensive test suite for video sources: File, Webcam, RTSP, and Reconnection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.fasttrack.core.errors import SourceReadError, ValidationError
from src.fasttrack.core.types import CameraConfig, CameraSourceType
from src.fasttrack.sources import (
    FileVideoSource,
    ReconnectManager,
    RTSPVideoSource,
    WebcamVideoSource,
    create_video_source,
)


def _generate_test_video(path: Path, width: int = 320, height: int = 240, num_frames: int = 25, fps: int = 25) -> None:
    """Helper to generate a valid synthetic test video using OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    for i in range(num_frames):
        # Frame with changing grayscale pattern
        frame = np.full((height, width, 3), (i * 10) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


# -----------------------------------------------------------------------------
# File Source Tests
# -----------------------------------------------------------------------------


def test_file_source_normal_read(temp_dir: Path) -> None:
    video_path = temp_dir / "valid_test.mp4"
    _generate_test_video(video_path, width=320, height=240, num_frames=10, fps=25)

    source = FileVideoSource(source_id="file-01", file_path=video_path, loop=False)
    assert source.open() is True
    assert source.is_alive is True

    frame = source.read()
    assert frame is not None
    assert frame.image.shape == (240, 320, 3)
    assert frame.frame_idx == 0
    assert frame.source_id == "file-01"
    assert frame.source_ts == 0.0

    # Read remaining frames
    count = 1
    while source.read() is not None:
        count += 1

    assert count == 10
    source.close()
    assert source.is_alive is False


def test_file_source_nonexistent_file(temp_dir: Path) -> None:
    source = FileVideoSource(source_id="file-missing", file_path=temp_dir / "does_not_exist.mp4")
    with pytest.raises(SourceReadError, match="does not exist"):
        source.open()


def test_file_source_zero_byte_file(temp_dir: Path) -> None:
    empty_file = temp_dir / "empty.mp4"
    empty_file.write_bytes(b"")

    source = FileVideoSource(source_id="file-empty", file_path=empty_file)
    with pytest.raises(SourceReadError, match="empty"):
        source.open()


def test_file_source_invalid_extension(temp_dir: Path) -> None:
    text_file = temp_dir / "malicious.txt"
    text_file.write_text("not a video")

    source = FileVideoSource(source_id="file-ext", file_path=text_file)
    with pytest.raises(ValidationError, match="Unsupported video file extension"):
        source.open()


def test_file_source_odd_resolution(temp_dir: Path) -> None:
    odd_path = temp_dir / "odd_res.mp4"
    # Video container encoders round odd dimensions to nearest even for YUV420 subsampling
    _generate_test_video(odd_path, width=333, height=277, num_frames=5, fps=25)

    source = FileVideoSource(source_id="file-odd", file_path=odd_path)
    try:
        assert source.open() is True
        frame = source.read()
        assert frame is not None
        # Codec produces (276, 332, 3) from 277x333 input
        assert frame.image.shape[0] in (276, 277)
        assert frame.image.shape[1] in (332, 333)

        # Directly verify preprocessing accepts raw odd arrays
        odd_numpy = np.zeros((277, 333, 3), dtype=np.uint8)
        processed = source._preprocess_frame(odd_numpy)
        assert processed.shape == (277, 333, 3)
    finally:
        source.close()


def test_file_source_4k_downscaled_early(temp_dir: Path) -> None:
    # 4K resolution (3840 x 2160)
    video_4k = temp_dir / "test_4k.mp4"
    _generate_test_video(video_4k, width=3840, height=2160, num_frames=2, fps=10)

    source = FileVideoSource(source_id="file-4k", file_path=video_4k)
    assert source.open() is True

    frame = source.read()
    assert frame is not None
    h, w = frame.image.shape[:2]
    # Maximum dimension must be scaled down to <= 1920
    assert max(h, w) <= 1920
    assert frame.metadata["is_downscaled"] is True
    source.close()


def test_file_source_unicode_and_spaced_path(temp_dir: Path) -> None:
    special_folder = temp_dir / "CCTV Folder with Spaces"
    special_file = special_folder / "test_тест_clip.mp4"
    _generate_test_video(special_file, width=160, height=120, num_frames=5)

    source = FileVideoSource(source_id="file-unicode", file_path=special_file)
    assert source.open() is True
    frame = source.read()
    assert frame is not None
    source.close()


def test_file_source_short_clip_insufficient_frames(temp_dir: Path) -> None:
    # 10 frames at 25 fps = 0.4 seconds (< 2.0s required for violence clip buffer)
    short_video = temp_dir / "short_clip.mp4"
    _generate_test_video(short_video, width=160, height=120, num_frames=10, fps=25)

    source = FileVideoSource(source_id="file-short", file_path=short_video, min_clip_duration_sec=2.0)
    assert source.open() is True
    assert source.metadata["insufficient_frames"] is True

    frame = source.read()
    assert frame is not None
    assert frame.metadata["insufficient_frames"] is True
    source.close()


def test_file_source_looping_behavior(temp_dir: Path) -> None:
    loop_video = temp_dir / "loop_test.mp4"
    _generate_test_video(loop_video, width=160, height=120, num_frames=3, fps=25)

    # With loop=True, reading beyond frame 3 should loop seamlessly back to start
    source = FileVideoSource(source_id="file-loop", file_path=loop_video, loop=True)
    assert source.open() is True

    frames_read = [source.read() for _ in range(7)]
    assert all(f is not None for f in frames_read)
    assert len(frames_read) == 7
    source.close()


# -----------------------------------------------------------------------------
# Reconnection Manager Tests
# -----------------------------------------------------------------------------


def test_reconnect_manager_exponential_backoff() -> None:
    mgr = ReconnectManager(initial_delay=1.0, max_delay=10.0, multiplier=2.0, jitter_factor=0.0)

    delay1 = mgr.get_next_delay()
    assert delay1 == 1.0

    delay2 = mgr.get_next_delay()
    assert delay2 == 2.0

    delay3 = mgr.get_next_delay()
    assert delay3 == 4.0

    delay4 = mgr.get_next_delay()
    assert delay4 == 8.0

    delay5 = mgr.get_next_delay()
    assert delay5 == 10.0  # Capped at max_delay

    mgr.record_success()
    assert mgr.current_attempt == 0
    assert mgr.get_next_delay() == 1.0


def test_reconnect_manager_max_attempts() -> None:
    mgr = ReconnectManager(initial_delay=0.1, max_attempts=3)
    assert mgr.can_retry() is True
    mgr.get_next_delay()
    assert mgr.can_retry() is True
    mgr.get_next_delay()
    assert mgr.can_retry() is True
    mgr.get_next_delay()
    assert mgr.can_retry() is False


# -----------------------------------------------------------------------------
# RTSP Source Tests
# -----------------------------------------------------------------------------


def test_rtsp_credential_masking() -> None:
    rtsp_url = "rtsp://security_admin:UltraSecret999@192.168.1.100:554/h264/ch1/main"
    source = RTSPVideoSource(source_id="cam-rtsp-01", rtsp_url=rtsp_url)

    masked = source.masked_url
    assert "UltraSecret999" not in masked
    assert "security_admin" not in masked
    assert masked == "rtsp://***@192.168.1.100:554/h264/ch1/main"


def test_rtsp_mock_read_and_watchdog() -> None:
    rtsp_url = "rtsp://localhost:554/test"
    source = RTSPVideoSource(source_id="cam-rtsp-mock", rtsp_url=rtsp_url, timeout_seconds=1.0)

    # Mock cv2.VideoCapture so test does not require live RTSP server
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: 25.0 if prop == cv2.CAP_PROP_FPS else 640.0

    fake_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    mock_cap.read.return_value = (True, fake_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        assert source.open() is True
        frame = source.read()
        assert frame is not None
        assert frame.image.shape == (480, 640, 3)
        assert frame.source_id == "cam-rtsp-mock"

        # Simulate watchdog timeout by manually setting last_frame_time_monotonic in past
        source._last_frame_time_monotonic = 0.0
        with patch.object(source, "_reconnect", return_value=True) as mock_reconn:
            source.read()
            assert mock_reconn.called

        source.close()


# -----------------------------------------------------------------------------
# Webcam Source Tests
# -----------------------------------------------------------------------------


def test_webcam_enumerate_cameras() -> None:
    # Verifies device enumeration does not raise uncaught exceptions
    devices = WebcamVideoSource.enumerate_cameras(max_to_check=2)
    assert isinstance(devices, list)


def test_webcam_mock_warmup_and_read() -> None:
    source = WebcamVideoSource(source_id="cam-webcam-mock", device_index=0, warmup_frames=3)

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.side_effect = lambda prop: 30.0 if prop == cv2.CAP_PROP_FPS else 640.0

    fake_frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    mock_cap.read.return_value = (True, fake_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        assert source.open() is True
        # Verify 3 warm-up reads occurred during open
        assert mock_cap.read.call_count == 3

        frame = source.read()
        assert frame is not None
        assert frame.image.shape == (480, 640, 3)
        assert frame.metadata["is_black_frame"] is True  # uniform 100 has std=0 < 2.0
        source.close()


# -----------------------------------------------------------------------------
# Source Factory Tests
# -----------------------------------------------------------------------------


def test_create_video_source_factory(temp_dir: Path) -> None:
    video_path = temp_dir / "factory_test.mp4"
    _generate_test_video(video_path, num_frames=5)

    cam_file = CameraConfig(id="c1", name="File Cam", source_type=CameraSourceType.FILE, source_ref=str(video_path))
    src1 = create_video_source(cam_file)
    assert isinstance(src1, FileVideoSource)

    cam_web = CameraConfig(id="c2", name="Webcam", source_type=CameraSourceType.WEBCAM, source_ref="0")
    src2 = create_video_source(cam_web)
    assert isinstance(src2, WebcamVideoSource)

    cam_rtsp = CameraConfig(id="c3", name="RTSP", source_type=CameraSourceType.RTSP, source_ref="rtsp://***@host")
    src3 = create_video_source(cam_rtsp)
    assert isinstance(src3, RTSPVideoSource)
