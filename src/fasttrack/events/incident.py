"""Incident aggregate state machine (OPEN -> ACTIVE -> CLOSING -> CLOSED)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from src.fasttrack.core.types import Event, EventType, Incident, IncidentSeverity, IncidentStatus
from src.fasttrack.events.severity import calculate_severity


class IncidentLifecycle(str, Enum):
    """Temporal lifecycle states of an incident."""

    OPEN = "OPEN"          # Initial event triggered
    ACTIVE = "ACTIVE"      # Ongoing detections actively arriving
    CLOSING = "CLOSING"    # Quiet period started, awaiting closure
    CLOSED = "CLOSED"      # Inactivity threshold reached, finalized


@dataclass
class IncidentAggregate:
    """Stateful aggregate tracking temporal events and lifecycle on a single camera."""

    id: str
    camera_id: str
    lifecycle: IncidentLifecycle = IncidentLifecycle.OPEN
    severity: IncidentSeverity = IncidentSeverity.INFO
    status: IncidentStatus = IncidentStatus.NEW
    start_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    end_utc: Optional[datetime] = None
    last_event_mono: float = 0.0
    peak_weapon_score: float = 0.0
    peak_violence_score: float = 0.0
    summary: str = ""
    events: List[Event] = field(default_factory=list)
    merge_gap_sec: float = 10.0

    @classmethod
    def create(
        cls,
        camera_id: str,
        start_mono: float,
        start_utc: datetime,
        merge_gap_sec: float = 10.0,
    ) -> IncidentAggregate:
        """Factory creating a new open incident aggregate."""
        inc_id = f"inc-{uuid.uuid4().hex[:10]}"
        return cls(
            id=inc_id,
            camera_id=camera_id,
            lifecycle=IncidentLifecycle.OPEN,
            start_utc=start_utc,
            last_event_mono=start_mono,
            merge_gap_sec=merge_gap_sec,
        )

    def add_event(self, event: Event, mono_ts: float) -> None:
        """Ingest event into this incident, update peak metrics, and transition state."""
        event.incident_id = self.id
        self.events.append(event)
        self.last_event_mono = mono_ts

        if self.lifecycle in (IncidentLifecycle.OPEN, IncidentLifecycle.CLOSING):
            self.lifecycle = IncidentLifecycle.ACTIVE

        # Update peak scores per signal independently
        if event.event_type == EventType.WEAPON_DETECTION:
            self.peak_weapon_score = max(self.peak_weapon_score, event.calibrated_score)
        elif event.event_type == EventType.POSSIBLE_VIOLENCE:
            self.peak_violence_score = max(self.peak_violence_score, event.calibrated_score)

        # Re-evaluate severity
        has_weapon = self.peak_weapon_score >= 0.45
        has_violence = self.peak_violence_score >= 0.60
        self.severity = calculate_severity(has_weapon, has_violence, has_weak_signal=True)

        # Update summary wording
        parts = []
        if self.peak_weapon_score > 0:
            parts.append(f"detected weapon ({self.peak_weapon_score:.2f})")
        if self.peak_violence_score > 0:
            parts.append(f"possible violence ({self.peak_violence_score:.2f})")
        self.summary = f"Possible incident on camera {self.camera_id}: " + " and ".join(parts) + "."

    def tick_lifecycle(self, now_mono: float, now_utc: datetime) -> IncidentLifecycle:
        """Advance state machine based on elapsed quiet period."""
        if self.lifecycle == IncidentLifecycle.CLOSED:
            return IncidentLifecycle.CLOSED

        elapsed = now_mono - self.last_event_mono

        if elapsed >= self.merge_gap_sec:
            self.lifecycle = IncidentLifecycle.CLOSED
            self.end_utc = now_utc
        elif elapsed >= (self.merge_gap_sec / 2.0) and self.lifecycle == IncidentLifecycle.ACTIVE:
            self.lifecycle = IncidentLifecycle.CLOSING

        return self.lifecycle

    def to_incident_record(self) -> Incident:
        """Convert aggregate state into database Incident entity."""
        return Incident(
            id=self.id,
            camera_id=self.camera_id,
            severity=self.severity,
            status=self.status,
            start_utc=self.start_utc,
            end_utc=self.end_utc,
            peak_weapon_score=round(self.peak_weapon_score, 4),
            peak_violence_score=round(self.peak_violence_score, 4),
            summary=self.summary,
            events=list(self.events),
        )
