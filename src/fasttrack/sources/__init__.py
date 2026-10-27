"""FastTrack video source adapters package."""

from __future__ import annotations

from src.fasttrack.core.errors import ConfigError
from src.fasttrack.core.types import CameraConfig, CameraSourceType
from src.fasttrack.sources.base import VideoSource
from src.fasttrack.sources.file_source import FileVideoSource
from src.fasttrack.sources.reconnect import ReconnectManager
from src.fasttrack.sources.rtsp_source import RTSPVideoSource
from src.fasttrack.sources.webcam_source import WebcamVideoSource


def create_video_source(camera: CameraConfig) -> VideoSource:
    """Factory creating appropriate video source adapter based on CameraConfig."""
    if camera.source_type == CameraSourceType.FILE:
        loop = camera.schedule.get("loop", True) if isinstance(camera.schedule, dict) else True
        return FileVideoSource(source_id=camera.id, file_path=camera.source_ref, loop=loop)

    elif camera.source_type == CameraSourceType.WEBCAM:
        fps_override = camera.thresholds.get("fps_override") if isinstance(camera.thresholds, dict) else None
        return WebcamVideoSource(
            source_id=camera.id,
            device_index=camera.source_ref,
            fps_override=fps_override,
        )

    elif camera.source_type == CameraSourceType.RTSP:
        return RTSPVideoSource(source_id=camera.id, rtsp_url=camera.source_ref)

    else:
        raise ConfigError(f"Unknown camera source type: {camera.source_type}")


__all__ = [
    "VideoSource",
    "FileVideoSource",
    "WebcamVideoSource",
    "RTSPVideoSource",
    "ReconnectManager",
    "create_video_source",
]
