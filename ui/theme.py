"""Command Center UI design system, color tokens, and global CSS injection."""

from __future__ import annotations

import streamlit as st

# Design System Color Tokens (Security Operations Command Center)
TOKENS_DARK = {
    "background": "#0B0F17",
    "surface": "#121826",
    "elevated": "#1A2236",
    "border": "#263049",
    "border_highlight": "#38476B",
    "text": "#E6EAF2",
    "muted": "#93A0B8",
    "primary": "#22D3EE",       # Cyan
    "primary_hover": "#06B6D4",
    "success": "#22C55E",       # Green
    "warning": "#F59E0B",       # Amber
    "critical": "#EF4444",      # Red
    "info": "#8B5CF6",          # Violet
}

TOKENS_LIGHT = {
    "background": "#F8FAFC",
    "surface": "#FFFFFF",
    "elevated": "#F1F5F9",
    "border": "#CBD5E1",
    "border_highlight": "#94A3B8",
    "text": "#0F172A",
    "muted": "#64748B",
    "primary": "#0891B2",
    "primary_hover": "#0E7490",
    "success": "#16A34A",
    "warning": "#D97706",
    "critical": "#DC2626",
    "info": "#7C3AED",
}


def get_current_theme_tokens() -> dict:
    """Return active token dictionary based on session state."""
    theme_mode = st.session_state.get("theme_mode", "dark")
    return TOKENS_DARK if theme_mode == "dark" else TOKENS_LIGHT


def inject_theme(theme_mode: str = "dark") -> None:
    """Inject polished Security Operations Command Center CSS stylesheet into Streamlit."""
    t = TOKENS_DARK if theme_mode == "dark" else TOKENS_LIGHT

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {{
        --ft-bg: {t['background']};
        --ft-surface: {t['surface']};
        --ft-elevated: {t['elevated']};
        --ft-border: {t['border']};
        --ft-border-hi: {t['border_highlight']};
        --ft-text: {t['text']};
        --ft-muted: {t['muted']};
        --ft-primary: {t['primary']};
        --ft-success: {t['success']};
        --ft-warning: {t['warning']};
        --ft-critical: {t['critical']};
        --ft-info: {t['info']};
    }}

    /* Global Body & Main Container Styling */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: var(--ft-bg) !important;
        color: var(--ft-text) !important;
    }}

    [data-testid="stHeader"] {{
        background-color: var(--ft-bg) !important;
        border-bottom: 1px solid var(--ft-border);
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: var(--ft-surface) !important;
        border-right: 1px solid var(--ft-border) !important;
    }}

    /* Monospace for timestamps, IDs, scores */
    .mono-text, .stCode, code, [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    }}

    /* Security Command Center KPI Card */
    .ft-kpi-card {{
        background: var(--ft-surface);
        border: 1px solid var(--ft-border);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: transform 160ms ease, border-color 160ms ease;
    }}
    .ft-kpi-card:hover {{
        border-color: var(--ft-border-hi);
        transform: translateY(-2px);
    }}
    .ft-kpi-title {{
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--ft-muted);
        margin-bottom: 6px;
    }}
    .ft-kpi-value {{
        font-size: 2.1rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: var(--ft-text);
        line-height: 1.1;
    }}
    .ft-kpi-subtitle {{
        font-size: 0.8rem;
        color: var(--ft-muted);
        margin-top: 6px;
    }}

    /* Top Status Bar */
    .ft-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: var(--ft-surface);
        border: 1px solid var(--ft-border);
        border-radius: 10px;
        padding: 10px 18px;
        margin-bottom: 20px;
    }}
    .ft-disclaimer-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(34, 211, 238, 0.12);
        color: var(--ft-primary);
        border: 1px solid rgba(34, 211, 238, 0.35);
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}

    /* Severity Badges (Never relying on color alone: text + icon) */
    .ft-severity-chip {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 0.76rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.03em;
    }}
    .ft-severity-HIGH {{
        background: rgba(239, 68, 68, 0.16);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.45);
    }}
    .ft-severity-MEDIUM {{
        background: rgba(245, 158, 11, 0.16);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.45);
    }}
    .ft-severity-LOW {{
        background: rgba(139, 92, 246, 0.16);
        color: #A78BFA;
        border: 1px solid rgba(139, 92, 246, 0.45);
    }}
    .ft-severity-INFO {{
        background: rgba(34, 211, 238, 0.14);
        color: #67E8F9;
        border: 1px solid rgba(34, 211, 238, 0.4);
    }}

    /* Status Pills */
    .ft-status-pill {{
        display: inline-block;
        border-radius: 14px;
        padding: 3px 9px;
        font-size: 0.75rem;
        font-weight: 600;
    }}
    .ft-status-new {{ background: #991B1B; color: #FEE2E2; }}
    .ft-status-reviewed {{ background: #065F46; color: #D1FAE5; }}
    .ft-status-false_alarm {{ background: #374151; color: #E5E7EB; }}
    .ft-status-escalated {{ background: #7C2D12; color: #FFEDD5; }}

    /* Respect prefers-reduced-motion */
    @media (prefers-reduced-motion: reduce) {{
        *, *::before, *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
