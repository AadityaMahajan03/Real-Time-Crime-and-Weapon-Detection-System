"""Standardized exception hierarchy for FastTrack AI."""

from __future__ import annotations


class FastTrackError(Exception):
    """Base exception for all FastTrack domain errors."""

    def __init__(self, message: str, details: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} [Details: {self.details}]"
        return self.message


class ConfigError(FastTrackError):
    """Raised when configuration files are missing, malformed, or invalid."""


class CameraError(FastTrackError):
    """Base exception for camera connectivity and streaming issues."""


class SourceReadError(CameraError):
    """Raised when a video frame cannot be read or decoded from a source."""


class ModelLoadError(FastTrackError):
    """Raised when model weights cannot be loaded, verified, or initialized."""


class InferenceError(FastTrackError):
    """Raised during neural network forward passes, CUDA OOM, or preprocessing."""


class DatabaseError(FastTrackError):
    """Raised on database connection, migration, or query failures."""


class EvidenceError(FastTrackError):
    """Raised on evidence capture, file integrity check, or disk space failures."""


class AuthError(FastTrackError):
    """Raised on authentication failures, unauthorized roles, or session expiry."""


class ValidationError(FastTrackError):
    """Raised on input boundary or parameter validation failures."""
