"""FastTrack notification package: webhooks and dispatch coordinator."""

from src.fasttrack.notify.dispatcher import NotificationDispatcher
from src.fasttrack.notify.webhook import WebhookChannel

__all__ = ["NotificationDispatcher", "WebhookChannel"]
