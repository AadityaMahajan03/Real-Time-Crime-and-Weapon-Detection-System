"""Webhook alert delivery channel with HMAC-SHA256 signing and exponential backoff retry."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

import requests

from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import IncidentSeverity

logger = get_logger("notify.webhook")

DEFAULT_TIMEOUT_SEC = 5.0
DEFAULT_MAX_RETRIES = 3


class WebhookChannel:
    """Delivers structured security incident alerts to external HTTP endpoints."""

    def __init__(
        self,
        url: str,
        secret: Optional[str] = None,
        min_severity: IncidentSeverity = IncidentSeverity.MEDIUM,
        enabled: bool = True,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    ) -> None:
        self.url = url
        self.secret = secret
        self.min_severity = min_severity
        self.enabled = enabled
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec

    def _generate_hmac_signature(self, body_bytes: bytes) -> str:
        """Compute HMAC-SHA256 signature for payload verification."""
        if not self.secret:
            return ""
        return hmac.new(self.secret.encode("utf-8"), body_bytes, hashlib.sha256).hexdigest()

    def send(self, payload: Dict[str, Any]) -> bool:
        """Deliver JSON payload with HMAC signature and exponential backoff retry."""
        if not self.enabled or not self.url:
            return False

        body_str = json.dumps(payload, sort_keys=True)
        body_bytes = body_str.encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "FastTrack-AI-Surveillance/1.0",
        }

        if self.secret:
            sig = self._generate_hmac_signature(body_bytes)
            headers["X-FastTrack-Signature"] = f"sha256={sig}"

        backoff = 0.5
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.url,
                    data=body_bytes,
                    headers=headers,
                    timeout=self.timeout_sec,
                )
                if 200 <= resp.status_code < 300:
                    logger.info("Webhook delivered successfully to %s (status %d)", self.url, resp.status_code)
                    return True

                logger.warning(
                    "Webhook to %s returned status %d on attempt %d/%d",
                    self.url,
                    resp.status_code,
                    attempt,
                    self.max_retries,
                )
            except requests.RequestException as e:
                logger.warning(
                    "Webhook request to %s failed on attempt %d/%d: %s",
                    self.url,
                    attempt,
                    self.max_retries,
                    e,
                )

            if attempt < self.max_retries:
                time.sleep(backoff)
                backoff *= 2.0

        logger.error("Webhook delivery to %s failed after %d attempts.", self.url, self.max_retries)
        return False
