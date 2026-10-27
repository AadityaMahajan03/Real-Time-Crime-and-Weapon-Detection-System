"""Incidents Page: Searchable, Filterable Incident Management & Triage Table with Integrated Forensics."""

from __future__ import annotations

import importlib
import sqlite3

import pandas as pd
import streamlit as st

from src.fasttrack.core.config import get_config
from ui.components import empty_state


def render_incidents_page() -> None:
    config = get_config()
    db_path = config.storage.db_path

    st.markdown("### Incident Registry & Forensic Triage")
    st.caption("Review, filter, inspect H.264 evidence playback, and audit all candidate weapon and violence detections.")

    tab_registry, tab_forensic = st.tabs(["Incident Registry", "Forensic Detail & Video Playback"])

    with tab_registry:
        # Filter toolbar
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            filter_status = st.selectbox("Status Filter", ["All Statuses", "new", "reviewed", "false_alarm", "escalated"])
        with c_f2:
            filter_sev = st.selectbox("Severity Filter", ["All Severities", "HIGH", "MEDIUM", "LOW", "INFO"])
        with c_f3:
            search_query = st.text_input("Search Summary / ID", placeholder="e.g. knife, cam-01...")

        rows = []
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            query = """
                SELECT i.id, i.camera_id, c.name as camera_name, i.severity, i.status,
                       i.start_utc, i.peak_weapon_score, i.peak_violence_score, i.summary
                FROM incidents i
                LEFT JOIN cameras c ON i.camera_id = c.id
                WHERE 1=1
            """
            params = []
            if filter_status != "All Statuses":
                query += " AND i.status = ?"
                params.append(filter_status)
            if filter_sev != "All Severities":
                query += " AND i.severity = ?"
                params.append(filter_sev)
            if search_query:
                query += " AND (i.summary LIKE ? OR i.id LIKE ?)"
                params.extend([f"%{search_query}%", f"%{search_query}%"])

            query += " ORDER BY i.start_utc DESC LIMIT 50;"
            cur.execute(query, params)
            rows = cur.fetchall()
            conn.close()
        except Exception as e:
            st.error(f"Error querying incidents: {e}")

        if rows:
            # Convert to displayable dataframe
            data = []
            for r in rows:
                data.append(
                    {
                        "Incident ID": r["id"],
                        "Camera": r["camera_name"] or r["camera_id"],
                        "Severity": r["severity"],
                        "Status": r["status"],
                        "Detected At (UTC)": r["start_utc"],
                        "Peak Weapon": f"{r['peak_weapon_score']:.2f}",
                        "Peak Violence": f"{r['peak_violence_score']:.2f}",
                        "Summary": r["summary"],
                    }
                )
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            empty_state("No Incidents Found", "No incidents recorded yet - monitoring active.", "🔍")

    with tab_forensic:
        # Load and render forensic detail page inside tab
        try:
            detail_module = importlib.import_module("ui.pages.04_incident_detail")
            detail_module.render_incident_detail_page()
        except Exception as e:
            st.error(f"Error rendering forensic detail: {e}")


if __name__ == "__main__":
    render_incidents_page()
