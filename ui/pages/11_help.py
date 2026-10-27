"""Help & Limitations Page: Operator Manual, Operational Guidance, and Ethical Boundaries."""

from __future__ import annotations

import streamlit as st

from ui.components import section_header


def render_help_page() -> None:
    st.markdown("### Operational Manual & Known Limitations")
    st.caption("Plain-language interpretation of prediction scores, failure modes, and lawful use policy.")

    section_header("1. Score Interpretation Guide")
    st.markdown(
        """
        - **Weapon Score (`0.00` to `1.00`):** Represents the bounding-box classifier confidence that an object matching a weapon class (e.g. knife) appears within the frame. *Candidate alerts require persistent track confirmation across multiple consecutive frames ($K$-of-$N$).*
        - **Violence Score (`0.00` to `1.00`):** Temperature-calibrated and EMA-smoothed probability output from the spatiotemporal 3D CNN across a rolling 2.0-second video buffer.
        - **Important:** Outputs are *predictions*, never legal proof of a crime.
        """
    )

    section_header("2. Documented Edge Cases and Failure Modes")
    st.markdown(
        """
        Operators should be aware of known failure modes where false positives or false negatives may occur:
        - **Weapon Look-alikes:** Elongated smartphones, TV remotes, umbrellas, power drills, and kitchen cutlery can trigger false candidate boxes.
        - **Occlusion and Motion Blur:** Fast hand movements or weapons concealed under coats or bags degrade detection accuracy.
        - **Violence Misclassifications:** High-energy benign actions (sports, celebrations, dancing, playful contact, hugging) can cause elevated spatiotemporal violence scores.
        - **Camera Shake:** Rapid panning or vibration of camera mounts can artificially mimic violent motion in the optical field.
        """
    )

    section_header("3. Lawful Use & Ethical Policy")
    st.markdown(
        """
        - **Authorized Feeds Only:** FAST-TRACK AI must only be connected to video streams authorized under applicable local surveillance and institutional policies.
        - **No Automated Dispatch:** All dispatches must be initiated manually by a qualified human security operator.
        - **Privacy Protection:** Face blurring is provided to protect bystander privacy during operator triage.
        """
    )


if __name__ == "__main__":
    render_help_page()
