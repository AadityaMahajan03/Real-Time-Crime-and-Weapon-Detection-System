"""Reusable UI components for FastTrack AI Security Command Center."""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from typing import Any, Optional

import streamlit as st


def render_top_bar(
    active_cameras: int = 1,
    total_cameras: int = 1,
    username: str = "Operator",
    role: str = "reviewer",
    is_demo: bool = False,
) -> None:
    """Render persistent command center top bar with live UTC clock and human-review disclaimer."""
    now_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    status_badge = (
        '<span style="background: rgba(245, 158, 11, 0.2); color: #FBBF24; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; margin-left: 8px;">[DEMO]</span>'
        if is_demo
        else '<span style="background: rgba(34, 197, 94, 0.2); color: #22C55E; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; margin-left: 8px;">● LIVE</span>'
    )

    html = f"""<div class="ft-topbar">
    <div style="display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 1.15rem; font-weight: 700; color: var(--ft-text); letter-spacing: -0.01em;">
            FAST-TRACK <span style="color: var(--ft-primary);">AI</span>
        </span>
        {status_badge}
        <div class="ft-disclaimer-chip">
            <span>⚠️</span> AI PREDICTIONS — HUMAN REVIEW REQUIRED
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 18px; font-size: 0.85rem;">
        <div>
            <span style="color: var(--ft-muted);">Cameras:</span>
            <strong style="color: var(--ft-text); font-family: 'JetBrains Mono', monospace;">{active_cameras}/{total_cameras} Active</strong>
        </div>
        <div>
            <span style="color: var(--ft-muted);">Clock:</span>
            <span style="color: var(--ft-text); font-family: 'JetBrains Mono', monospace;">{now_utc_str}</span>
        </div>
        <div style="background: var(--ft-elevated); padding: 4px 10px; border-radius: 6px; border: 1px solid var(--ft-border);">
            👤 <strong style="color: var(--ft-text);">{username}</strong>
            <span style="color: var(--ft-muted); font-size: 0.75rem;">({role.upper()})</span>
        </div>
    </div>
</div>"""
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    accent_color: Optional[str] = None,
    delta: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Render a clean metric card matching command center theme."""
    val_style = f"color: {accent_color};" if accent_color else "color: var(--ft-text);"
    if subtitle and delta:
        sub_text = f"{subtitle} • {delta}"
    else:
        sub_text = subtitle or delta or ""
    sub_html = f'<div class="ft-kpi-subtitle" style="font-size: 0.8rem; color: var(--ft-muted); margin-top: 4px;">{sub_text}</div>' if sub_text else ""
    html = f"""<div class="ft-kpi-card">
    <div class="ft-kpi-title">{title}</div>
    <div class="ft-kpi-value" style="{val_style}">{value}</div>
    {sub_html}
</div>"""
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def severity_chip(severity: str) -> str:
    """Return HTML markup for severity chip with icon and text label."""
    icons = {
        "HIGH": "🔴 HIGH",
        "MEDIUM": "🟡 MEDIUM",
        "LOW": "🟣 LOW",
        "INFO": "🔵 INFO",
    }
    sev_upper = severity.upper()
    label = icons.get(sev_upper, f"⚪ {sev_upper}")
    return f'<span class="ft-severity-chip ft-severity-{sev_upper}">{label}</span>'


def status_pill(status: str) -> str:
    """Return HTML markup for incident review status pill."""
    labels = {
        "new": "NEW",
        "reviewed": "REVIEWED",
        "false_alarm": "FALSE ALARM",
        "escalated": "ESCALATED",
    }
    lbl = labels.get(status.lower(), status.upper())
    return f'<span class="ft-status-pill ft-status-{status.lower()}">{lbl}</span>'


def section_header(title: str, subtitle: str = "") -> None:
    """Render a consistent section header."""
    sub_html = f'<p style="color: var(--ft-muted); font-size: 0.88rem; margin-top: 2px;">{subtitle}</p>' if subtitle else ""
    html = f"""<div style="margin-top: 14px; margin-bottom: 14px;">
    <h2 style="font-size: 1.35rem; font-weight: 700; color: var(--ft-text); margin: 0;">{title}</h2>
    {sub_html}
</div>"""
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def empty_state(title: str, message: str, icon: str = "📋") -> None:
    """Render empty state feedback with guidance."""
    html = f"""<div style="background: var(--ft-surface); border: 1px dashed var(--ft-border); border-radius: 12px; padding: 36px 20px; text-align: center; margin: 16px 0;">
    <div style="font-size: 2.2rem; margin-bottom: 10px;">{icon}</div>
    <div style="font-size: 1.05rem; font-weight: 600; color: var(--ft-text); margin-bottom: 6px;">{title}</div>
    <div style="font-size: 0.86rem; color: var(--ft-muted); max-width: 500px; margin: 0 auto;">{message}</div>
</div>"""
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


def score_gauge_html(score: float, threshold: float, label: str) -> str:
    """Return HTML markup representing a linear confidence gauge with threshold mark."""
    pct = min(100.0, max(0.0, score * 100))
    thresh_pct = min(100.0, max(0.0, threshold * 100))
    bar_color = "var(--ft-critical)" if score >= threshold else "var(--ft-primary)"

    html = f"""<div style="margin: 8px 0;">
    <div style="display: flex; justify-content: space-between; font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; margin-bottom: 4px;">
        <span style="color: var(--ft-muted);">{label}</span>
        <span style="color: var(--ft-text); font-weight: 600;">{score:.2f} (Thresh: {threshold:.2f})</span>
    </div>
    <div style="position: relative; width: 100%; height: 8px; background: var(--ft-border); border-radius: 4px; overflow: visible;">
        <div style="width: {pct}%; height: 100%; background: {bar_color}; border-radius: 4px; transition: width 200ms ease;"></div>
        <div style="position: absolute; left: {thresh_pct}%; top: -3px; bottom: -3px; width: 2px; background: #FFFFFF; border-radius: 1px;" title="Trigger Threshold: {threshold:.2f}"></div>
    </div>
</div>"""
    return textwrap.dedent(html).strip()
