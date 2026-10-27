"""Enterprise logging subsystem with secret sanitization and rotating file handlers."""

from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

# Regex pattern to match credentials in URLs (e.g. rtsp://user:pass@host:port)
_CREDENTIAL_URL_PATTERN = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^:/@]+:[^/@]+@)")


class SanitizedFormatter(logging.Formatter):
    """Log formatter that automatically masks secrets and passwords in messages."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return _mask_secrets(original)


def _mask_secrets(text: str) -> str:
    """Mask credentials in connection strings."""
    return _CREDENTIAL_URL_PATTERN.sub(r"\1***:***@", text)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = "logs/fasttrack.log",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Configure root logger for FastTrack AI with both console and rotating file outputs.

    Args:
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Relative or absolute path to log file. If None, file logging is disabled.
        max_bytes: Max file size before rotation.
        backup_count: Number of rotated log archives to maintain.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("fasttrack")
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if logger.handlers:
        return logger

    log_format = "%(asctime)s | %(levelname)-7s | [%(name)s:%(filename)s:%(lineno)d] | %(message)s"
    formatter = SanitizedFormatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a namespaced logger child under the 'fasttrack' root."""
    if not name:
        return logging.getLogger("fasttrack")
    if name.startswith("fasttrack"):
        return logging.getLogger(name)
    return logging.getLogger(f"fasttrack.{name}")
