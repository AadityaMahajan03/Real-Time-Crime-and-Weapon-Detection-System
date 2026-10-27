"""Notification dispatcher for multi-channel incident alerting with quiet-hours suppression."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import Incident, IncidentSeverity
from src.fasttrack.notify.webhook import WebhookChannel

logger = get_logger("notify.dispatcher")

SEVERITY_RANKS = {
    IncidentSeverity.INFO: 0,
    IncidentSeverity.LOW: 1,
    IncidentSeverity.MEDIUM: 2,
    IncidentSeverity.HIGH: 3,
}


class NotificationDispatcher:
    """Dispatches real-time security alerts to authorized internal channels."""

    def __init__(
        self,
        channels: Optional[List[WebhookChannel]] = None,
        quiet_hours: Optional[tuple[int, int]] = None,  # (start_hour_utc, end_hour_utc), e.g. (22, 6)
        clock: Optional[Clock] = None,
    ) -> None:
        self.channels = channels or []
        self.quiet_hours = quiet_hours
        self.clock = clock or SystemClock()
        self._sent_alert_keys: set[str] = set()

    def add_channel(self, channel: WebhookChannel) -> None:
        """Register an alert delivery channel."""
        self.channels.append(channel)

    def is_in_quiet_hours(self) -> bool:
        """Check whether current UTC time falls within configured quiet hours."""
        if not self.quiet_hours:
            return False

        start_h, end_h = self.quiet_hours
        current_h = self.clock.now_utc().hour

        if start_h <= end_h:
            return start_h <= current_h < end_h
        else:
            # Over midnight (e.g. 22:00 to 06:00)
            return current_h >= start_h or current_h < end_h

    def _should_suppress(self, severity: IncidentSeverity) -> bool:
        """Evaluate if notification should be silenced by quiet-hours policy."""
        # HIGH severity always bypasses quiet hours
        if severity == IncidentSeverity.HIGH:
            return False

        if self.is_in_quiet_hours():
            logger.info("Notification for severity %s silenced by quiet hours policy.", severity.value)
            return True

        return False

    def build_payload(
        self,
        incident: Incident,
        camera_name: str = "",
        evidence_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate structured incident payload for external receivers."""
        return {
            "incident_id": incident.id,
            "camera_id": incident.camera_id,
            "camera_name": camera_name or incident.camera_id,
            "severity": incident.severity.value,
            "status": incident.status.value,
            "start_utc": incident.start_utc.isoformat(),
            "peak_weapon_score": incident.peak_weapon_score,
            "peak_violence_score": incident.peak_violence_score,
            "summary": incident.summary,
            "evidence_url": evidence_url,
            "disclaimer": "Probabilistic AI detection. Human review required before physical action.",
            "dispatched_utc": self.clock.now_utc().isoformat(),
        }

    def dispatch_incident_alert(
        self,
        incident: Incident,
        camera_name: str = "",
        evidence_url: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Dispatch incident notification to all qualifying channels."""
        if self._should_suppress(incident.severity):
            return {}

        payload = self.build_payload(incident, camera_name=camera_name, evidence_url=evidence_url)
        results: Dict[str, bool] = {}

        inc_rank = SEVERITY_RANKS.get(incident.severity, 0)

        for i, channel in enumerate(self.channels):
            chan_name = f"webhook_{i}_{channel.url}"
            chan_min_rank = SEVERITY_RANKS.get(channel.min_severity, 0)

            if inc_rank < chan_min_rank:
                logger.debug(
                    "Skipping channel %s: incident rank %d < channel min %d",
                    chan_name,
                    inc_rank,
                    chan_min_rank,
                )
                continue

            success = channel.send(payload)
            results[chan_name] = success

        return results

    def dispatch_escalation_alert(
        self,
        incident_id: str,
        camera_name: str,
        severity: str,
        reason: str,
        user_id: str,
    ) -> Dict[str, bool]:
        """Dispatch high-priority escalation notice when an operator escalates an incident."""
        payload = {
            "alert_type": "OPERATOR_ESCALATION",
            "incident_id": incident_id,
            "camera_name": camera_name,
            "severity": severity,
            "escalated_by": user_id,
            "escalation_reason": reason,
            "dispatched_utc": self.clock.now_utc().isoformat(),
            "disclaimer": "Security team escalation notice. Immediate human attention requested.",
        }

        results: Dict[str, bool] = {}
        for i, channel in enumerate(self.channels):
            chan_name = f"escalation_webhook_{i}"
            results[chan_name] = channel.send(payload)

        return results
