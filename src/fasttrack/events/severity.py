"""Rule-based operational severity triage calculator (never linear score summation)."""

from __future__ import annotations

from src.fasttrack.core.types import IncidentSeverity


def calculate_severity(
    has_confirmed_weapon: bool,
    has_confirmed_violence: bool,
    has_weak_signal: bool = False,
) -> IncidentSeverity:
    """Compute rule-based incident severity triage descriptor.

    Rules (strictly decoupled modalities; scores are never linearly combined):
    - HIGH: Confirmed weapon track temporally overlapping with confirmed violence.
    - MEDIUM: Confirmed weapon detection OR confirmed spatiotemporal violence.
    - LOW: Single unconfirmed or weak signal.
    - INFO: System notification or operational baseline.
    """
    if has_confirmed_weapon and has_confirmed_violence:
        return IncidentSeverity.HIGH

    if has_confirmed_weapon or has_confirmed_violence:
        return IncidentSeverity.MEDIUM

    if has_weak_signal:
        return IncidentSeverity.LOW

    return IncidentSeverity.INFO


def get_severity_rationale(severity: IncidentSeverity) -> str:
    """Return plain-language explanation of why this severity level was assigned."""
    explanations = {
        IncidentSeverity.HIGH: "Co-occurring confirmed weapon track and confirmed violent motion on the same camera.",
        IncidentSeverity.MEDIUM: "Single confirmed modality (confirmed weapon track with K-of-N persistence OR confirmed spatiotemporal violence).",
        IncidentSeverity.LOW: "Unconfirmed candidate or low-confidence transient detection.",
        IncidentSeverity.INFO: "Operational system notice or camera diagnostic update.",
    }
    return explanations.get(severity, "Triage descriptor.")
