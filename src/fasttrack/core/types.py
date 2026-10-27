"""Core domain data types, enums, and structures for FastTrack AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CameraSourceType(str, Enum):
    """Supported video source adapters."""

    FILE = "file"
    WEBCAM = "webcam"
    RTSP = "rtsp"


class StreamStatus(str, Enum):
    """Operational lifecycle state of a camera streaming pipeline."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class IncidentSeverity(str, Enum):
    """Standardized operational triage severity levels."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

    @property
    def display_badge(self) -> str:
        """Icon and text descriptor for UI rendering."""
        badges = {
            "INFO": "[INFO] Operational Notice",
            "LOW": "[LOW] Unconfirmed / Low Confidence",
            "MEDIUM": "[MED] Confirmed Single Signal",
            "HIGH": "[HIGH] Co-occurring Weapon & Violence",
        }
        return badges.get(self.value, self.value)


class IncidentStatus(str, Enum):
    """Review status lifecycle for an incident."""

    NEW = "new"
    REVIEWED = "reviewed"
    FALSE_ALARM = "false_alarm"
    ESCALATED = "escalated"


class EventType(str, Enum):
    """Categorization of individual detection events."""

    WEAPON_DETECTION = "weapon_detection"
    POSSIBLE_VIOLENCE = "possible_violence"
    SYSTEM_ALERT = "system_alert"


class ZoneType(str, Enum):
    """Region of interest filtering type."""

    INCLUDE = "include"
    IGNORE = "ignore"


class UserRole(str, Enum):
    """User access control roles."""

    ADMIN = "admin"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


@dataclass
class BoundingBox:
    """Bounding box in normalized or pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        h = self.height
        return (self.width / h) if h > 1e-6 else 0.0

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) with another box."""
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)

        intersection_w = max(0.0, ix2 - ix1)
        intersection_h = max(0.0, iy2 - iy1)
        intersection = intersection_w * intersection_h

        union = self.area + other.area - intersection
        if union <= 1e-6:
            return 0.0
        return intersection / union

    def to_list(self) -> List[float]:
        return [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)]

    def to_int_tuple(self) -> Tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))


@dataclass
class Detection:
    """Single weapon or object detection in an image."""

    box: BoundingBox
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    source_id: str = "camera-0"
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_threat: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": self.box.to_list(),
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "track_id": self.track_id,
            "source_id": self.source_id,
            "timestamp_utc": self.timestamp_utc.isoformat(),
            "is_threat": self.is_threat,
        }


@dataclass
class Frame:
    """Unified video frame representation passed through pipelines."""

    image: np.ndarray  # BGR numpy array
    ts_utc: datetime
    source_ts: float  # monotonic seconds or presentation timestamp
    frame_idx: int
    source_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


@dataclass
class Zone:
    """Geometrical Region of Interest defined by normalized polygon points [0.0 to 1.0]."""

    name: str
    zone_type: ZoneType
    polygon: List[Tuple[float, float]]

    def contains_point(self, nx: float, ny: float) -> bool:
        """Ray-casting algorithm to determine if normalized point is within polygon."""
        if len(self.polygon) < 3:
            return True
        inside = False
        n = len(self.polygon)
        p1x, p1y = self.polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = self.polygon[i % n]
            if ny > min(p1y, p2y):
                if ny <= max(p1y, p2y):
                    if nx <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (ny - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or nx <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside


@dataclass
class CameraConfig:
    """Configuration structure for a registered camera."""

    id: str
    name: str
    source_type: CameraSourceType
    source_ref: str
    enabled: bool = True
    zones: List[Zone] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    schedule: Dict[str, Any] = field(default_factory=dict)
    created_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def masked_source_ref(self) -> str:
        """Sanitize passwords and access tokens from display strings."""
        if self.source_type == CameraSourceType.RTSP and "@" in self.source_ref:
            prefix, rest = self.source_ref.split("://", 1)
            _, host = rest.split("@", 1)
            return f"{prefix}://***@{host}"
        return self.source_ref


@dataclass
class Event:
    """An individual atomic alert detection generated by pipeline workers."""

    id: str
    incident_id: Optional[str]
    ts_utc: datetime
    event_type: EventType
    class_name: str
    raw_score: float
    calibrated_score: float
    bbox: Optional[BoundingBox] = None
    track_id: Optional[int] = None
    model_version_id: Optional[str] = None
    evidence_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceItem:
    """Cryptographic evidence artifact (snapshot or clip) linked to detections."""

    id: str
    kind: str  # 'snapshot', 'clip'
    path: str
    sha256: str
    size_bytes: int
    encrypted: bool = False
    created_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_utc: Optional[datetime] = None


@dataclass
class Incident:
    """An aggregated incident grouping related events across time on a camera."""

    id: str
    camera_id: str
    severity: IncidentSeverity
    status: IncidentStatus
    start_utc: datetime
    end_utc: Optional[datetime] = None
    peak_weapon_score: float = 0.0
    peak_violence_score: float = 0.0
    summary: str = ""
    events: List[Event] = field(default_factory=list)


@dataclass
class ReviewAction:
    """Audit trail record of an operator interaction with an incident."""

    id: str
    incident_id: str
    user_id: str
    action: str
    note: str
    ts_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class User:
    """System user with role-based access control permissions."""

    id: str
    username: str
    password_hash: str
    role: UserRole
    active: bool = True
    last_login_utc: Optional[datetime] = None


@dataclass
class HealthSample:
    """Diagnostic sample measuring real-time camera and model pipeline metrics."""

    ts_utc: datetime
    camera_id: str
    fps: float
    latency_ms: float
    dropped_frames: int
    gpu_util: float = 0.0
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
