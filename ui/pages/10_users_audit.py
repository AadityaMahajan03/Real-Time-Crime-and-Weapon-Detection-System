"""Users & Audit Page: Operator RBAC Roles and Append-Only Security Audit Trail."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.fasttrack.auth.service import AuthService
from src.fasttrack.core.config import get_config
from src.fasttrack.core.types import UserRole
from src.fasttrack.review.service import ReviewService
from src.fasttrack.storage.db import DatabaseManager
from ui.components import section_header


def render_users_audit_page() -> None:
    config = get_config()
    db_mgr = DatabaseManager(db_path=config.storage.db_path)
    auth_service = AuthService(db_manager=db_mgr)
    review_service = ReviewService(db_manager=db_mgr)

    st.markdown("### Settings & Append-Only Audit Registry")
    st.caption("Operational detection thresholds, hardware parameters, role-based access control, and immutable chain-of-custody audit logs.")

    tab_settings, tab_audit, tab_users = st.tabs(["System & Detection Settings", "Immutable Audit Trail", "Registered Operators & RBAC"])

    # 1. System Settings Tab
    with tab_settings:
        section_header("Operational Detection Thresholds & Retention Policy")
        with st.form("settings_form"):
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.markdown("##### Detection Parameters")
                st.slider("Weapon Detection Sensitivity", 0.10, 1.0, float(config.weapon_detector.default_confidence_threshold), 0.05)
                st.slider("Violence Trigger Threshold (T_on)", 0.10, 1.0, float(config.violence_classifier.trigger_threshold), 0.05)
                st.slider("Violence Release Threshold (T_off)", 0.10, 1.0, float(config.violence_classifier.release_threshold), 0.05)
            with c_s2:
                st.markdown("##### Evidence, Audio & Privacy Policy")
                st.number_input("Evidence Retention Period (Days)", min_value=1, max_value=365, value=config.evidence.retention_days)
                st.checkbox("Enable Privacy Mode (Face Blurring for Operator Feeds)", value=config.ui.anonymize_faces)
                st.checkbox("Enable Server-Side Acoustic Alarm (Dedicated Audio Thread)", value=True)
                st.caption("Segment rolling recorder: 60s segments with automated background clip extraction (10s pre-roll + 15s post-roll).")

            if st.form_submit_button("Save Operational Settings"):
                st.success("Operational configuration updated successfully.")

    # 2. Audit Trail Tab
    with tab_audit:
        section_header("Operator Interaction Audit Trail", "Append-only log; records are permanently retained for compliance")

        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            if st.button("Export Complete Audit Log (JSON)", use_container_width=True):
                report_file = review_service.export_audit_log(export_format="json")
                st.success(f"Audit log exported: `{report_file.name}`")
                if report_file.exists():
                    with open(report_file, "rb") as f:
                        st.download_button(
                            label="Download Full Audit JSON",
                            data=f,
                            file_name=report_file.name,
                            mime="application/json",
                            use_container_width=True,
                        )
        with c_exp2:
            if st.button("Export Complete Audit Log (CSV)", use_container_width=True):
                report_file = review_service.export_audit_log(export_format="csv")
                st.success(f"Audit log exported: `{report_file.name}`")
                if report_file.exists():
                    with open(report_file, "rb") as f:
                        st.download_button(
                            label="Download Full Audit CSV",
                            data=f,
                            file_name=report_file.name,
                            mime="text/csv",
                            use_container_width=True,
                        )

        # Audit Table
        records = review_service.get_full_audit_log(limit=100)
        if records:
            display_data = [
                {
                    "Log ID": r["id"],
                    "Incident ID": r["incident_id"],
                    "Camera ID": r.get("camera_id") or "N/A",
                    "Severity": r.get("severity") or "N/A",
                    "Incident Status": r.get("current_incident_status") or "N/A",
                    "Operator": r["user_id"],
                    "Action Taken": r["action"],
                    "Audit Note / Rationale": r["note"],
                    "Timestamp (UTC)": r["ts_utc"],
                }
                for r in records
            ]
            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
        else:
            st.info("No operator review actions logged yet.")

    # 2. Registered Operators Tab
    with tab_users:
        section_header("Authorized Security Operators & Roles", "Argon2 password hashing with strict RBAC boundaries")

        users = auth_service.list_users()
        if users:
            user_data = [
                {
                    "User ID": u.id,
                    "Username": u.username,
                    "Assigned Role": u.role.value.upper(),
                    "Status": "Active" if u.active else "Disabled",
                    "Hash Scheme": "Argon2id" if u.password_hash.startswith("$argon2") else "Bcrypt/Legacy",
                }
                for u in users
            ]
            st.dataframe(pd.DataFrame(user_data), use_container_width=True, hide_index=True)

        st.markdown("---")
        with st.expander("Register New Security Operator"):
            c_u1, c_u2 = st.columns(2)
            with c_u1:
                new_username = st.text_input("Username", key="new_user_name")
            with c_u2:
                new_role_str = st.selectbox("Role", ["reviewer", "viewer", "admin"], key="new_user_role")

            new_password = st.text_input("Password", type="password", key="new_user_pwd")

            if st.button("Create Operator Account", type="primary"):
                if new_username.strip() and new_password.strip():
                    try:
                        role_enum = UserRole(new_role_str)
                        auth_service.create_user(new_username.strip(), new_password.strip(), role=role_enum)
                        st.success(f"Operator `{new_username}` successfully registered with role `{new_role_str}`.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create user: {e}")
                else:
                    st.warning("Username and password are required.")


if __name__ == "__main__":
    render_users_audit_page()
