"""Cameras & Zones Page: Camera Fleet Management, Multi-Worker Controls, and ROI Zones."""

from __future__ import annotations

import streamlit as st

from src.fasttrack.core.config import get_config
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.repositories import CameraRepository
from ui.components import section_header


def render_cameras_zones_page() -> None:
    config = get_config()
    db = DatabaseManager(config.storage.db_path)
    camera_repo = CameraRepository(db)

    st.markdown("### Surveillance Camera Fleet & Multi-Worker Controls")
    st.caption("Manage active hardware streams, configure worker lifecycles, and persist operational fleet activation in SQLite.")

    # Top Fleet Action Bar
    btn_col1, btn_col2, info_col = st.columns([1, 1, 2])
    with btn_col1:
        if st.button("🟢 Activate All Cameras", use_container_width=True):
            camera_repo.set_all_enabled(True)
            st.toast("All cameras marked ACTIVE in fleet.", icon="🟢")
            st.rerun()

    with btn_col2:
        if st.button("⏹️ Deactivate All", use_container_width=True):
            camera_repo.set_all_enabled(False)
            st.toast("All cameras DEACTIVATED.", icon="⚪")
            st.rerun()

    cameras = camera_repo.get_all()
    enabled_count = sum(1 for c in cameras if c.get("enabled", 1))

    with info_col:
        st.markdown(
            f"""
            <div style="background: var(--ft-surface); border: 1px solid var(--ft-border); border-radius: 8px; padding: 6px 12px; text-align: right; font-size: 0.85rem;">
                <strong>Fleet Status:</strong> <span style="color: #22C55E; font-weight: 700;">{enabled_count} Active</span> / {len(cameras)} Registered
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    tab_fleet, tab_multi, tab_add = st.tabs(["📹 Camera Fleet List", "🎛️ Multi-Select Activation", "➕ Register New Camera"])

    # TAB 1: Individual Camera Cards with Toggles
    with tab_fleet:
        if not cameras:
            st.info("No surveillance cameras registered. Click 'Register New Camera' to connect a webcam, RTSP stream, or video file.")
        else:
            for c in cameras:
                cid = c["id"]
                is_enabled = bool(c.get("enabled", 1))

                status_badge = "🟢 ONLINE" if is_enabled else "⚪ DISABLED"
                status_color = "#22C55E" if is_enabled else "#94A3B8"

                with st.expander(f"{c['name']} ({c['source_type'].upper()}) — {status_badge}", expanded=True):
                    c1, c2, c3 = st.columns([2, 1, 1])

                    with c1:
                        st.markdown(f"**Camera ID:** `{cid}`")
                        st.markdown(f"**Source URI (Masked):** `{c['source_ref_masked']}`")
                        st.markdown(
                            f"**Worker State:** <span style='color: {status_color}; font-weight: 700;'>{status_badge}</span>",
                            unsafe_allow_html=True,
                        )

                    with c2:
                        toggle_val = st.toggle(
                            "Worker Active",
                            value=is_enabled,
                            key=f"toggle_cam_{cid}",
                        )
                        if toggle_val != is_enabled:
                            camera_repo.update_enabled(cid, toggle_val)
                            st.toast(f"Camera '{c['name']}' set to {'Active' if toggle_val else 'Disabled'}.")
                            st.rerun()

                    with c3:
                        if st.button("🗑️ Remove Camera", key=f"del_cam_{cid}", use_container_width=True):
                            camera_repo.delete(cid)
                            st.toast(f"Camera '{c['name']}' removed.", icon="🗑️")
                            st.rerun()

    # TAB 2: Multi-Select Subset Activation
    with tab_multi:
        section_header("Batch Multi-Select Camera Activation", "Select exactly which cameras should run in workers simultaneously")
        if cameras:
            all_options = {c["id"]: f"{c['name']} ({c['source_type'].upper()})" for c in cameras}
            currently_active = [c["id"] for c in cameras if c.get("enabled", 1)]

            selected_subset = st.multiselect(
                "Select active cameras:",
                options=list(all_options.keys()),
                default=currently_active,
                format_func=lambda x: all_options.get(x, x),
            )

            if st.button("💾 Apply Fleet Selection", type="primary"):
                camera_repo.set_subset_enabled(selected_subset)
                st.success(f"Fleet updated! {len(selected_subset)} camera(s) activated in database.")
                st.rerun()
        else:
            st.info("No cameras to select.")

    # TAB 3: Add / Register Camera Form
    with tab_add:
        section_header("Register New Surveillance Source", "Connect local USB/MSMF webcams, RTSP IP cameras, or pre-recorded test files")
        with st.form("register_camera_form"):
            new_id = st.text_input("Unique Camera ID", value=f"cam-0{len(cameras)+1}")
            new_name = st.text_input("Camera Name / Location", value="Main Entry Checkpoint")
            new_type = st.selectbox("Source Type", ["webcam", "file", "rtsp"])
            new_ref = st.text_input(
                "Source Reference",
                value="0",
                help="For webcam: '0' or device index. For file: path to MP4. For RTSP: rtsp://user:pass@ip:port/live",
            )
            initial_enabled = st.checkbox("Activate worker immediately upon registration", value=True)

            submitted = st.form_submit_button("Connect & Save Camera", use_container_width=True)
            if submitted:
                # Mask RTSP credentials
                masked_uri = new_ref
                if new_type == "rtsp" and "@" in new_ref:
                    prefix, rest = new_ref.split("://", 1)
                    _, host = rest.split("@", 1)
                    masked_uri = f"{prefix}://***@{host}"

                try:
                    camera_repo.create(
                        camera_id=new_id.strip(),
                        name=new_name.strip(),
                        source_type=new_type,
                        source_ref_masked=masked_uri.strip(),
                        enabled=initial_enabled,
                    )
                    st.success(f"Camera '{new_name}' successfully registered in database!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to register camera: {e}")


if __name__ == "__main__":
    render_cameras_zones_page()
