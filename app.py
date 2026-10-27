"""FAST-TRACK AI: Multi-Camera Weapon and Violence Detection Command Center.

Main entry point for Streamlit web interface.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.fasttrack.core.config import get_config
from src.fasttrack.core.logging import setup_logging
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.migrations import MigrationRunner
from ui.components import render_top_bar
from ui.theme import inject_theme

# Initialize application logging
setup_logging(level=os.getenv("FASTTRACK_LOG_LEVEL", "INFO"))

# Initialize Streamlit page configuration
st.set_page_config(
    page_title="FAST-TRACK AI | Security Operations",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_app_state() -> None:
    """Initialize session state defaults and verify database schema migrations."""
    if "initialized" not in st.session_state:
        config = get_config()
        # Verify database migrations
        try:
            db = DatabaseManager(config.storage.db_path)
            runner = MigrationRunner(db)
            runner.run_migrations()
        except Exception as e:
            st.error(f"Database initialization warning: {e}")

        # Set session defaults
        st.session_state["theme_mode"] = config.ui.theme
        st.session_state["demo_mode"] = False
        st.session_state["current_user"] = "Operator_Lead"
        st.session_state["current_role"] = "reviewer"
        st.session_state["initialized"] = True


initialize_app_state()

# Inject theme CSS
theme_mode = st.session_state.get("theme_mode", "dark")
inject_theme(theme_mode)

# Render Global Header & Top Bar
render_top_bar(
    active_cameras=2,
    total_cameras=3,
    username=st.session_state.get("current_user", "Operator"),
    role=st.session_state.get("current_role", "reviewer"),
    is_demo=False,
)

# Sidebar controls & system status
with st.sidebar:
    st.markdown("### 🛡️ FAST-TRACK AI")
    st.caption("Live AI Surveillance Command Center")

    # Hardware & Model acceleration status badges
    import textwrap
    import torch

    has_cuda = torch.cuda.is_available()
    dev_name = torch.cuda.get_device_name(0) if has_cuda else "CPU Host"

    sb_html = f"""<div style="background: var(--ft-surface); border: 1px solid var(--ft-border); border-radius: 8px; padding: 10px; margin-bottom: 12px; font-size: 0.8rem;">
    <div style="color: #22C55E; font-weight: 700; margin-bottom: 4px;">● SYSTEM OPERATIONAL</div>
    <div style="color: var(--ft-muted);">Weapon Detector: <span style="color: var(--ft-text); font-weight: 600;">YOLO11n</span></div>
    <div style="color: var(--ft-muted);">Action Model: <span style="color: var(--ft-text); font-weight: 600;">S3D Video CNN</span></div>
    <div style="color: var(--ft-muted);">Acceleration: <span style="color: #22D3EE; font-weight: 600;">{dev_name}</span></div>
</div>"""
    st.markdown(textwrap.dedent(sb_html).strip(), unsafe_allow_html=True)

    # Theme toggle
    dark_mode = st.toggle("Dark Theme (Command Center)", value=(st.session_state.get("theme_mode") == "dark"))
    st.session_state["theme_mode"] = "dark" if dark_mode else "light"

    st.divider()

# Navigation setup: Exactly 7 Consolidated Operational Pages (Per P5 Architecture)
pages = [
    st.Page("ui/pages/01_overview.py", title="Dashboard", icon="📊", default=True),
    st.Page("ui/pages/02_live_monitor.py", title="Live Monitor", icon="📹"),
    st.Page("ui/pages/03_incidents.py", title="Incidents", icon="🚨"),
    st.Page("ui/pages/07_cameras_zones.py", title="Cameras & Zones", icon="🎥"),
    st.Page("ui/pages/06_evaluation.py", title="Models & Evaluation", icon="🔬"),
    st.Page("ui/pages/10_users_audit.py", title="Settings & Audit Log", icon="⚙️"),
    st.Page("ui/pages/11_help.py", title="Help", icon="ℹ️"),
]

nav = st.navigation(pages)
nav.run()
