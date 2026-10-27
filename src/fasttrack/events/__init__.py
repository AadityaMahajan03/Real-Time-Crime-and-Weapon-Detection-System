"""FastTrack event engine, triage severity, rate limiting, and incident lifecycle."""

from src.fasttrack.events.dedup import AlertStormLimiter, ViolenceHysteresis
from src.fasttrack.events.engine import EventEngine
from src.fasttrack.events.incident import IncidentAggregate, IncidentLifecycle
from src.fasttrack.events.severity import calculate_severity, get_severity_rationale
from src.fasttrack.events.zones import ZoneFilter

__all__ = [
    "AlertStormLimiter",
    "EventEngine",
    "IncidentAggregate",
    "IncidentLifecycle",
    "ViolenceHysteresis",
    "ZoneFilter",
    "calculate_severity",
    "get_severity_rationale",
]
