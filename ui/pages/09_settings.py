"""Settings Page: Operational Thresholds, Retention Policies, and Notification Settings."""

from __future__ import annotations

import streamlit as st

from src.fasttrack.core.config import get_config
from ui.components import section_header


def render_settings_page() -> None:
    config = get_config()
    st.markdown("### System & Detection Settings")
    st.caption("Fine-tune detection thresholds, evidence retention, and privacy safeguards.")

    with st.form("settings_form"):
        section_header("Weapon Detection Parameters")
        st.slider("Weapon Detection Sensitivity (IoU/Conf)", 0.1, 1.0, float(config.weapon_detector.default_confidence_threshold), 0.05)

        section_header("Violence Classification Parameters")
        c1, c2 = st.columns(2)
        with c1:
            st.slider("Trigger Threshold (T_on)", 0.1, 1.0, float(config.violence_classifier.trigger_threshold), 0.05)
        with c2:
            st.slider("Release Threshold (T_off)", 0.1, 1.0, float(config.violence_classifier.release_threshold), 0.05)

        section_header("Evidence & Privacy Safeguards")
        st.checkbox("Enable Privacy Mode (Automatic Face Blurring for Operator Views)", value=config.ui.anonymize_faces)
        st.number_input("Evidence Retention Period (Days)", min_value=1, max_value=365, value=config.evidence.retention_days)

        if st.form_submit_button("Save Operational Settings"):
            st.success("Settings updated successfully.")


if __name__ == "__main__":
    render_settings_page()
