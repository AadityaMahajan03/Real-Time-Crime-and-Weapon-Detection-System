"""Unit tests for WebhookChannel and NotificationDispatcher with HMAC-SHA256 signing."""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.fasttrack.core.clock import SimulatedClock
from src.fasttrack.core.types import Incident, IncidentSeverity, IncidentStatus
from src.fasttrack.notify.dispatcher import NotificationDispatcher
from src.fasttrack.notify.webhook import WebhookChannel


def make_dummy_incident(severity: IncidentSeverity = IncidentSeverity.HIGH) -> Incident:
    """Helper to generate dummy incident entity."""
    return Incident(
        id="inc-test-notif-01",
        camera_id="cam-notif",
        severity=severity,
        status=IncidentStatus.NEW,
        start_utc=datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc),
        peak_weapon_score=0.88,
        peak_violence_score=0.72,
        summary="Detected knife and possible violence",
    )


def test_webhook_channel_hmac_signing_and_delivery():
    """Verify WebhookChannel generates valid HMAC-SHA256 header and handles 200 responses."""
    secret = "TopSecretWebhookKey123"
    channel = WebhookChannel(
        url="https://alerts.surveillance.example.com/webhook",
        secret=secret,
        min_severity=IncidentSeverity.MEDIUM,
    )

    payload = {"incident_id": "inc-01", "severity": "HIGH", "summary": "Threat detected"}

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        ok = channel.send(payload)
        assert ok is True
        assert mock_post.called

        # Inspect headers
        call_kwargs = mock_post.call_args[1]
        headers = call_kwargs["headers"]
        assert "X-FastTrack-Signature" in headers
        assert headers["X-FastTrack-Signature"].startswith("sha256=")

        # Verify signature math
        sent_body = call_kwargs["data"]
        expected_sig = hmac.new(secret.encode("utf-8"), sent_body, hashlib.sha256).hexdigest()
        assert headers["X-FastTrack-Signature"] == f"sha256={expected_sig}"


def test_webhook_channel_retry_backoff():
    """Verify WebhookChannel retries with exponential backoff on server errors."""
    channel = WebhookChannel(
        url="https://alerts.surveillance.example.com/webhook",
        max_retries=2,
        timeout_sec=1.0,
    )

    with patch("requests.post") as mock_post, patch("time.sleep") as mock_sleep:
        # First attempt 500, second attempt 200
        resp1 = MagicMock(status_code=500)
        resp2 = MagicMock(status_code=200)
        mock_post.side_effect = [resp1, resp2]

        ok = channel.send({"test": "data"})
        assert ok is True
        assert mock_post.call_count == 2
        assert mock_sleep.called


def test_notification_dispatcher_severity_gating():
    """Verify lower-severity incidents are filtered when channel requires higher severity."""
    channel_high_only = WebhookChannel(
        url="https://alerts.example.com/high",
        min_severity=IncidentSeverity.HIGH,
    )
    dispatcher = NotificationDispatcher(channels=[channel_high_only])

    with patch.object(channel_high_only, "send", return_value=True) as mock_send:
        # 1. Medium severity incident -> filtered out
        med_inc = make_dummy_incident(severity=IncidentSeverity.MEDIUM)
        results = dispatcher.dispatch_incident_alert(med_inc)
        assert len(results) == 0
        assert not mock_send.called

        # 2. High severity incident -> dispatched
        high_inc = make_dummy_incident(severity=IncidentSeverity.HIGH)
        results2 = dispatcher.dispatch_incident_alert(high_inc)
        assert len(results2) == 1
        assert mock_send.called


def test_notification_dispatcher_quiet_hours_suppression():
    """Verify quiet hours suppress MEDIUM/LOW alerts but allow HIGH alerts."""
    clock = SimulatedClock(start_utc=datetime(2026, 10, 18, 23, 30, 0, tzinfo=timezone.utc))  # 23:30 UTC
    channel = WebhookChannel(url="https://alerts.example.com/feed", min_severity=IncidentSeverity.LOW)

    # Quiet hours from 22:00 to 06:00
    dispatcher = NotificationDispatcher(
        channels=[channel],
        quiet_hours=(22, 6),
        clock=clock,
    )

    assert dispatcher.is_in_quiet_hours() is True

    with patch.object(channel, "send", return_value=True) as mock_send:
        # Medium incident during quiet hours -> suppressed
        med_inc = make_dummy_incident(severity=IncidentSeverity.MEDIUM)
        results = dispatcher.dispatch_incident_alert(med_inc)
        assert len(results) == 0
        assert not mock_send.called

        # High incident during quiet hours -> BYPASSES quiet hours
        high_inc = make_dummy_incident(severity=IncidentSeverity.HIGH)
        results_high = dispatcher.dispatch_incident_alert(high_inc)
        assert len(results_high) == 1
        assert mock_send.called


def test_notification_dispatcher_escalation_alert():
    """Verify operator escalation alert formatting and broadcast."""
    channel = WebhookChannel(url="https://alerts.example.com/escalation")
    dispatcher = NotificationDispatcher(channels=[channel])

    with patch.object(channel, "send", return_value=True) as mock_send:
        results = dispatcher.dispatch_escalation_alert(
            incident_id="inc-99",
            camera_name="Main Gate",
            severity="HIGH",
            reason="Confirmed blade detected",
            user_id="operator_john",
        )
        assert len(results) == 1
        assert mock_send.called
        sent_payload = mock_send.call_args[0][0]
        assert sent_payload["alert_type"] == "OPERATOR_ESCALATION"
        assert sent_payload["incident_id"] == "inc-99"
        assert sent_payload["escalated_by"] == "operator_john"
