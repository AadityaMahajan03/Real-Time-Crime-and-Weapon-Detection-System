"""Analytics Page: Temporal Trends, MTTR, False Alarm Pareto, and Fleet Health Metrics."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.fasttrack.analytics.engine import AnalyticsEngine
from src.fasttrack.core.config import get_config
from src.fasttrack.storage.db import DatabaseManager
from ui.components import kpi_card, section_header


def render_analytics_page() -> None:
    config = get_config()
    db_mgr = DatabaseManager(db_path=config.storage.db_path)
    analytics = AnalyticsEngine(db_manager=db_mgr)

    st.markdown("### Operational Intelligence & Reliability Analytics")
    st.caption("Temporal alert distributions, operator MTTR triage latency, false alarm root-cause Pareto analysis, and camera fleet health.")

    # 1. Top-Level Operational KPIs
    kpi_data = analytics.get_kpis()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card(
            title="Total Incidents",
            value=str(kpi_data["total_incidents"]),
            delta=f"{kpi_data['high_severity_count']} High-Severity",
            accent_color="#22D3EE",
        )
    with c2:
        far_val = kpi_data["false_alarm_rate_pct"]
        far_color = "#22C55E" if far_val < 35.0 else ("#F59E0B" if far_val < 60.0 else "#EF4444")
        kpi_card(
            title="False Alarm Rate",
            value=f"{far_val:.1f}%",
            delta="Human verified triage",
            accent_color=far_color,
        )
    with c3:
        mttr_sec = kpi_data["mttr_sec"]
        mttr_display = f"{mttr_sec / 60.0:.1f} min" if mttr_sec >= 60 else f"{mttr_sec:.0f} sec"
        mttr_color = "#22C55E" if mttr_sec < 180 else "#F59E0B"
        kpi_card(
            title="Mean Time to Review",
            value=mttr_display,
            delta=f"Median: {kpi_data['mttr_median_sec']:.0f}s",
            accent_color=mttr_color,
        )
    with c4:
        kpi_card(
            title="Monitored Cameras",
            value=str(kpi_data["active_cameras_count"]),
            delta="Online & healthy",
            accent_color="#22D3EE",
        )

    st.markdown("---")

    # 2. Incident Frequency Trends by Severity
    section_header("Daily Incident Trends by Triage Severity", "Volume progression over time")
    trends = analytics.get_incident_trends()
    if trends:
        df_trends = pd.DataFrame(trends)
        fig_trend = px.bar(
            df_trends,
            x="date_str",
            y=["high_count", "medium_count", "low_count", "info_count"],
            title="Daily Incidents by Severity",
            labels={"value": "Incident Count", "date_str": "Date", "variable": "Severity Level"},
            color_discrete_map={
                "high_count": "#EF4444",
                "medium_count": "#F59E0B",
                "low_count": "#22D3EE",
                "info_count": "#64748B",
            },
        )
        fig_trend.update_layout(
            barmode="stack",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#E6EAF2"),
            height=320,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No historical incident trends recorded yet.")

    # 3. Two Columns: Hourly Diurnal Distribution & False Alarm Pareto Analysis
    c_left, c_right = st.columns(2)

    with c_left:
        section_header("24-Hour Diurnal Activity", "Incident distribution across time of day (UTC)")
        hourly = analytics.get_hourly_distribution()
        df_hourly = pd.DataFrame(hourly)
        fig_hourly = px.bar(
            df_hourly,
            x="hour_label",
            y="count",
            labels={"hour_label": "Hour (UTC)", "count": "Incidents"},
            color="count",
            color_continuous_scale="Viridis",
        )
        fig_hourly.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#E6EAF2"),
            height=300,
            margin=dict(l=10, r=10, t=20, b=10),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_hourly, use_container_width=True)

    with c_right:
        section_header("False Alarm Pareto Root-Cause Analysis", "Categorized reasons for operator dismissals")
        pareto = analytics.get_false_alarm_pareto()
        if pareto:
            df_p = pd.DataFrame(pareto)
            fig_p = go.Figure()
            # Bar chart for counts
            fig_p.add_trace(
                go.Bar(
                    x=df_p["category"],
                    y=df_p["count"],
                    name="Count",
                    marker_color="#F59E0B",
                    yaxis="y",
                )
            )
            # Line chart for cumulative %
            fig_p.add_trace(
                go.Scatter(
                    x=df_p["category"],
                    y=df_p["cumulative_percentage"],
                    name="Cumulative %",
                    marker_color="#22D3EE",
                    yaxis="y2",
                    mode="lines+markers",
                )
            )
            fig_p.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#E6EAF2"),
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
                yaxis=dict(title="Count"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 105]),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig_p, use_container_width=True)
        else:
            st.info("No false alarm dismissals categorized yet.")

    # 4. Camera Fleet Breakdown & Telemetry Diagnostics
    tab_threats, tab_health = st.tabs(["Camera Threat Summary", "Streaming Pipeline Telemetry"])

    with tab_threats:
        cams = analytics.get_camera_threat_summary()
        if cams:
            df_cams = pd.DataFrame(cams)
            df_cams = df_cams.rename(
                columns={
                    "camera_id": "Camera ID",
                    "camera_name": "Location / Name",
                    "incident_count": "Total Incidents",
                    "high_severity_count": "High Severity",
                    "max_weapon_score": "Peak Weapon Score",
                    "max_violence_score": "Peak Violence Score",
                }
            )
            st.dataframe(df_cams, use_container_width=True, hide_index=True)
        else:
            st.info("No camera threat records found.")

    with tab_health:
        telemetry_rows = analytics.get_telemetry_health_summary()
        if telemetry_rows:
            df_tel = pd.DataFrame(telemetry_rows)
            df_tel = df_tel.rename(
                columns={
                    "camera_id": "Camera ID",
                    "avg_fps": "Average FPS",
                    "avg_latency_ms": "Avg Latency (ms)",
                    "max_dropped_frames": "Dropped Frames",
                    "avg_cpu_pct": "Avg CPU %",
                    "avg_ram_pct": "Avg RAM %",
                    "sample_count": "Diagnostic Samples",
                }
            )
            st.dataframe(df_tel, use_container_width=True, hide_index=True)
        else:
            st.info("No streaming telemetry diagnostic samples recorded yet.")


if __name__ == "__main__":
    render_analytics_page()
