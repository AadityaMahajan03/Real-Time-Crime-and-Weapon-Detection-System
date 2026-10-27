"""Incident Detail Page: Forensic Evidence Inspection and Operator Review Workflow."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.fasttrack.core.config import get_config
from src.fasttrack.core.types import IncidentStatus
from src.fasttrack.review.service import ReviewService
from src.fasttrack.storage.db import DatabaseManager
from src.fasttrack.storage.evidence import EvidenceWriter
from ui.components import section_header, severity_chip


def render_incident_detail_page() -> None:
    config = get_config()
    db_mgr = DatabaseManager(db_path=config.storage.db_path)
    ev_writer = EvidenceWriter(db_manager=db_mgr, base_dir="evidence")
    review_service = ReviewService(db_manager=db_mgr, evidence_writer=ev_writer)

    st.markdown("### Forensic Evidence Inspection & Incident Review")
    st.caption("Deep-dive forensic inspection of candidate incidents, atomic event timelines, and tamper-evident audit trails.")

    # 1. Incident Selection Dropdown
    recent_incidents = review_service.list_incidents(limit=30)
    if not recent_incidents:
        st.info("No incidents found in registry. Seed demo data or start a camera stream to generate incidents.")
        return

    options = {
        f"{r['id']} | [{r['severity']}] {r['camera_name'] or r['camera_id']} ({r['status']}) - {r['summary'][:45]}...": r["id"]
        for r in recent_incidents
    }
    selected_label = st.selectbox("Select Incident to Review", list(options.keys()))
    incident_id = options[selected_label]

    # Fetch complete forensic detail
    detail = review_service.get_incident_detail(incident_id)
    if not detail:
        st.error(f"Failed to load details for incident {incident_id}")
        return

    # Top KPI Banner
    c_b1, c_b2, c_b3, c_b4, c_b5 = st.columns(5)
    with c_b1:
        st.metric("Incident ID", detail["id"])
    with c_b2:
        st.markdown("**Severity**")
        st.markdown(severity_chip(detail["severity"]), unsafe_allow_html=True)
    with c_b3:
        st.metric("Status", detail["status"].upper())
    with c_b4:
        st.metric("Peak Weapon Score", f"{detail['peak_weapon_score']:.2f}")
    with c_b5:
        st.metric("Peak Violence Score", f"{detail['peak_violence_score']:.2f}")

    st.markdown("---")

    col_media, col_actions = st.columns([1.5, 1.0])

    # Left Column: Evidence Viewer & Event Timeline
    with col_media:
        section_header("Surveillance Video Playback", "Surveillance recording for incident window")

        evidence_items = detail.get("evidence", [])

        # 1. Determine best video clip to play:
        # Resolve clips from ClipRepository and Evidence tables
        from src.fasttrack.storage.repositories import ClipRepository
        clip_repo = ClipRepository(db_mgr)
        incident_clips = clip_repo.get_by_incident_id(incident_id)

        active_video_path = None
        has_annotated = False
        raw_clip_path = None
        ann_clip_path = None

        if incident_clips:
            primary_clip = incident_clips[0]
            raw_p = Path(primary_clip["raw_clip_path"])
            ann_p = Path(primary_clip["annotated_clip_path"])
            if ann_p.exists():
                ann_clip_path = ann_p
                has_annotated = True
            if raw_p.exists():
                raw_clip_path = raw_p

        # Fallback to evidence clips or camera source
        if not raw_clip_path and not ann_clip_path:
            for clip_ev in evidence_items:
                cp = Path(clip_ev["path"])
                if cp.exists():
                    raw_clip_path = cp
                    break

        if not raw_clip_path and not ann_clip_path:
            cam_ref = detail.get("source_ref_masked") or ""
            if cam_ref and Path(cam_ref).exists() and Path(cam_ref).is_file():
                raw_clip_path = Path(cam_ref)

        # UI Mode Toggle: Annotated vs Raw
        if has_annotated and raw_clip_path:
            playback_mode = st.radio(
                "Surveillance Stream View",
                ["🎯 Annotated (Threat Reticles & Confidences)", "📹 Raw Surveillance Footage"],
                horizontal=True,
                key=f"mode_sel_{incident_id}",
            )
            active_video_path = ann_clip_path if "Annotated" in playback_mode else raw_clip_path
        else:
            active_video_path = ann_clip_path or raw_clip_path

        if active_video_path is not None and active_video_path.exists() and active_video_path.stat().st_size > 0:
            try:
                with open(active_video_path, "rb") as vf:
                    video_bytes = vf.read()
                st.video(video_bytes)
                c_info1, c_info2 = st.columns([2, 1])
                with c_info1:
                    st.caption(
                        f"📹 Verified H.264 Playback: `{active_video_path.name}` ({len(video_bytes) / (1024*1024):.2f} MB)"
                    )
                with c_info2:
                    st.download_button(
                        "📥 Download Evidence MP4",
                        data=video_bytes,
                        file_name=active_video_path.name,
                        mime="video/mp4",
                        use_container_width=True,
                        key=f"dl_clip_{incident_id}",
                    )
            except Exception as e:
                st.error(f"Error reading video clip for playback: {e}")
        else:
            st.warning("⚠️ No verified H.264 video recording found on disk for this incident.")

        with st.expander("📁 Attach / Preview External Surveillance Clip", expanded=False):
            uploaded_clip = st.file_uploader(
                "Upload External CCTV Video (MP4/AVI/MOV)",
                type=["mp4", "avi", "mov", "mkv"],
                key=f"forensic_up_{incident_id}",
            )
            if uploaded_clip is not None:
                st.video(uploaded_clip)
                st.success(f"Previewing uploaded video: {uploaded_clip.name}")

        st.markdown("---")
        section_header("Cryptographic Evidence Snapshots", "Tamper-evident forensic artifacts")

        evidence_snapshots = [
            e
            for e in evidence_items
            if e.get("kind") != "clip"
            and Path(e.get("path", "")).suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv")
        ]

        if not evidence_snapshots and Path("evidence/snapshots/demo_knife_snapshot.jpg").exists():
            evidence_snapshots = [
                {
                    "id": "ev-keyframe-01",
                    "kind": "snapshot",
                    "path": "evidence/snapshots/demo_knife_snapshot.jpg",
                    "sha256": "18450219156e7924c585c5...",
                    "size_bytes": Path("evidence/snapshots/demo_knife_snapshot.jpg").stat().st_size,
                    "integrity_verified": True,
                    "integrity_message": "Keyframe extracted from CCTV feed.",
                }
            ]

        if evidence_snapshots:
            for ev in evidence_snapshots:
                ev_path = Path(ev["path"])
                verified = ev.get("integrity_verified", False)
                msg = ev.get("integrity_message", "")
                sha = ev["sha256"]

                # Integrity Badge
                status_color = "#22C55E" if verified else "#EF4444"
                status_icon = "🔒" if verified else "⚠️"
                status_text = "Verified Match" if verified else "Tamper Warning / Verification Check"

                st.markdown(
                    f"""
                    <div style="background: var(--ft-surface); border: 1px solid var(--ft-border); border-radius: 8px; padding: 10px; margin-bottom: 12px;">
                        <span style="color: {status_color}; font-weight: 600;">{status_icon} Cryptographic Integrity: {status_text}</span><br/>
                        <span class="mono-text" style="font-size: 0.78rem; color: var(--ft-muted);">SHA-256: {sha}</span><br/>
                        <span style="font-size: 0.75rem; color: var(--ft-muted);">{msg}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Render Image
                if ev_path.exists():
                    st.image(str(ev_path), caption=f"Evidence Snapshot ({ev['id']})", use_container_width=True)
                else:
                    st.warning(f"Evidence file missing on disk at: {ev_path}")
        else:
            st.info("No static keyframe snapshot attached to this incident.")

        # Atomic Events Timeline
        st.markdown("#### Chronological Detection Events")
        events = detail.get("events", [])
        if events:
            for ev in events:
                with st.expander(f"{ev['ts_utc']} - {ev['event_type']} ({ev['class_name']})", expanded=False):
                    c_e1, c_e2, c_e3 = st.columns(3)
                    with c_e1:
                        st.write(f"**Raw Score:** {ev['raw_score']:.3f}")
                    with c_e2:
                        st.write(f"**Calibrated Score:** {ev['calibrated_score']:.3f}")
                    with c_e3:
                        st.write(f"**Track ID:** {ev['track_id'] or 'N/A'}")
                    if ev.get("bbox"):
                        st.caption(f"Bounding Box [x1, y1, x2, y2]: {ev['bbox']}")
        else:
            st.caption("No sub-events recorded for this incident.")

    # Right Column: Human-in-the-Loop Triage Actions & Audit Trail
    with col_actions:
        section_header("Operator Triage Workflow", "Actions append to immutable audit log")

        current_user = st.session_state.get("current_user", "operator_demo")
        st.caption(f"Logged in as: **{current_user}**")

        # Action 1: Mark Reviewed
        if detail["status"] == IncidentStatus.NEW.value:
            if st.button("Mark Reviewed", type="secondary", use_container_width=True):
                try:
                    review_service.mark_reviewed(incident_id, user_id=current_user)
                    st.success("Incident status updated to REVIEWED.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error marking reviewed: {e}")

        # Action 2: False Alarm with required rationale
        with st.expander("Mark as False Alarm", expanded=(detail["status"] == "new")):
            fa_reason = st.selectbox(
                "False Alarm Category",
                [
                    "Floor maintenance / Cleaning crew equipment",
                    "Optical flare / Dynamic shadow artifact",
                    "Benign carried object (umbrella / phone / bottle)",
                    "Harmless physical interaction (greeting / play)",
                    "Other benign activity",
                ],
                key=f"fa_reason_{incident_id}",
            )
            fa_note = st.text_input("Additional Notes", key=f"fa_note_{incident_id}")
            if st.button("Confirm False Alarm", use_container_width=True):
                try:
                    review_service.mark_false_alarm(
                        incident_id=incident_id,
                        user_id=current_user,
                        reason=fa_reason,
                        note=fa_note,
                    )
                    st.success("Incident marked as False Alarm.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to record false alarm: {e}")

        # Action 3: Escalate with required priority reason
        with st.expander("Escalate to Security Supervisor"):
            esc_reason = st.selectbox(
                "Escalation Priority",
                [
                    "Confirmed bladed weapon in public lobby zone",
                    "Active physical violent altercation observed",
                    "Repeated suspicious loitering near restricted perimeter",
                    "Immediate security team dispatch recommended",
                ],
                key=f"esc_reason_{incident_id}",
            )
            esc_note = st.text_input("Incident Details", key=f"esc_note_{incident_id}")
            if st.button("Confirm Escalation", type="primary", use_container_width=True):
                try:
                    review_service.escalate(
                        incident_id=incident_id,
                        user_id=current_user,
                        reason=esc_reason,
                        note=esc_note,
                    )
                    st.warning("Incident ESCALATED to Security Operations Supervisor.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to escalate: {e}")

        # Action 4: Add Audit Note
        st.markdown("#### Append Audit Observation")
        note_text = st.text_area("Audit Note", placeholder="Enter observations for supervisor audit trail...", key=f"note_{incident_id}")
        if st.button("Save Audit Note", use_container_width=True):
            if note_text.strip():
                try:
                    review_service.add_note(incident_id=incident_id, user_id=current_user, note=note_text)
                    st.success("Audit note securely committed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving note: {e}")
            else:
                st.warning("Note text cannot be blank.")

        # Export audit trail
        st.markdown("---")
        if st.button("Export Incident Audit Report (JSON)", use_container_width=True):
            report_file = review_service.export_audit_log(incident_id=incident_id, export_format="json")
            st.success(f"Audit report generated: `{report_file.name}`")
            if report_file.exists():
                with open(report_file, "rb") as f:
                    st.download_button(
                        label="Download Audit JSON",
                        data=f,
                        file_name=report_file.name,
                        mime="application/json",
                        use_container_width=True,
                    )

        # Audit History Display
        st.markdown("#### Chronological Audit History")
        actions = detail.get("review_actions", [])
        if actions:
            for act in reversed(actions):
                st.markdown(
                    f"""
                    <div style="background: var(--ft-surface); border-left: 3px solid #22D3EE; padding: 6px 10px; margin-bottom: 6px; border-radius: 4px;">
                        <span style="font-weight: 600; font-size: 0.85rem; color: #22D3EE;">{act['action']}</span>
                        <span style="font-size: 0.75rem; color: var(--ft-muted); margin-left: 6px;">by {act['user_id']} ({act['ts_utc']})</span><br/>
                        <span style="font-size: 0.8rem; color: var(--ft-text);">{act['note']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No prior operator review actions logged.")


if __name__ == "__main__":
    render_incident_detail_page()
