"""Models Page: Model Versions, Checkpoints, and Architecture Comparison."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import torch

from src.fasttrack.core.config import get_config
from src.fasttrack.ml.benchmark import BenchmarkService
from src.fasttrack.storage.db import DatabaseManager
from ui.components import kpi_card, section_header


def render_models_page() -> None:
    config = get_config()
    db_mgr = DatabaseManager(db_path=config.storage.db_path)
    benchmark_svc = BenchmarkService(eval_dir="reports/eval", db_manager=db_mgr)

    st.markdown("### Neural Network Model Management & Registry")
    st.caption("Active model checkpoints, cryptographic weight integrity, architecture comparisons, and hardware acceleration.")

    # 1. Hardware Environment & Execution Mode
    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU (Host)"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Inference Engine", "CUDA 12.8" if has_cuda else "CPU", device_name, "#22D3EE")
    with c2:
        kpi_card("PyTorch Version", torch.__version__, "Active Runtime", "#8B5CF6")
    with c3:
        kpi_card("GPU Mutex Lock", "ENABLED", "Prevents VRAM Spikes", "#22C55E")
    with c4:
        kpi_card("Model Registrations", str(len(benchmark_svc.list_model_versions())), "SQLite Catalog", "#F59E0B")

    st.markdown("---")

    # 2. Registered Model Versions from SQLite
    section_header("Model Version Catalog & Active Switching", "Tracked model weights, dataset provenance, and SHA-256 integrity.")

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        if st.button("🔄 Sync Catalog to DB", use_container_width=True):
            count = benchmark_svc.sync_model_versions_to_db()
            st.success(f"Synced {count} model versions into database.")
            st.rerun()

    models = benchmark_svc.list_model_versions()
    if not models:
        st.info("No models registered in SQLite yet. Click 'Sync Catalog to DB' above to register checkpoints and evaluation runs.")
    else:
        # Display table of models
        table_rows = []
        for m in models:
            is_active = "🟢 ACTIVE" if m.get("active") else "⚪ Inactive"
            sha_trunc = (m.get("sha256") or "")[:12] + "..." if m.get("sha256") else "N/A"
            table_rows.append(
                {
                    "ID": m.get("id"),
                    "Modality": (m.get("kind") or "").upper(),
                    "Model Name": m.get("name"),
                    "Status": is_active,
                    "Weights Path": m.get("path"),
                    "SHA-256": sha_trunc,
                    "Dataset Split": m.get("dataset_version"),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

        # Active Model Selector
        st.markdown("##### Switch Active Checkpoint")
        c_sel_w, c_sel_v = st.columns(2)

        with c_sel_w:
            weapon_models = [m for m in models if m.get("kind") == "weapon"]
            if weapon_models:
                w_options = {f"{m['name']} ({'ACTIVE' if m['active'] else 'Inactive'})": m["id"] for m in weapon_models}
                current_active_w = next((k for k, v in w_options.items() if "ACTIVE" in k), list(w_options.keys())[0])
                sel_w = st.selectbox("Active Weapon Model", list(w_options.keys()), index=list(w_options.keys()).index(current_active_w))
                sel_w_id = w_options[sel_w]
                if st.button("Apply Weapon Model", key="btn_apply_w"):
                    benchmark_svc.set_active_model_version(sel_w_id)
                    st.success(f"Switched active weapon detector to {sel_w_id}")
                    st.rerun()

        with c_sel_v:
            violence_models = [m for m in models if m.get("kind") == "violence"]
            if violence_models:
                v_options = {f"{m['name']} ({'ACTIVE' if m['active'] else 'Inactive'})": m["id"] for m in violence_models}
                current_active_v = next((k for k, v in v_options.items() if "ACTIVE" in k), list(v_options.keys())[0])
                sel_v = st.selectbox("Active Violence Model", list(v_options.keys()), index=list(v_options.keys()).index(current_active_v))
                sel_v_id = v_options[sel_v]
                if st.button("Apply Violence Model", key="btn_apply_v"):
                    benchmark_svc.set_active_model_version(sel_v_id)
                    st.success(f"Switched active violence classifier to {sel_v_id}")
                    st.rerun()

    st.markdown("---")

    # 3. Active Pipeline Configurations
    section_header("Active Pipeline Configurations", "Runtime parameters for spatial detection and spatiotemporal inference.")
    c_w, c_v = st.columns(2)
    with c_w:
        st.markdown(
            f"""
            <div style="background: var(--ft-surface); border: 1px solid var(--ft-border); border-radius: 8px; padding: 18px;">
                <h4 style="margin: 0 0 10px 0; color: #22D3EE;">🗡️ Spatial Weapon Pipeline</h4>
                <ul style="color: var(--ft-text); font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 18px;">
                    <li><strong>Architecture:</strong> Ultralytics YOLO Nano (Configurable)</li>
                    <li><strong>Active Weights:</strong> <code>{config.weapon_detector.model_checkpoint}</code></li>
                    <li><strong>Fallback Weights:</strong> <code>{config.weapon_detector.fallback_checkpoint}</code></li>
                    <li><strong>Confidence Threshold (Knife):</strong> <code>{config.weapon_detector.per_class_thresholds.get('knife', 0.45)}</code></li>
                    <li><strong>Tracking Strategy:</strong> ByteTrack with <em>K-of-N confirmation</em> (K=3, N=5)</li>
                    <li><strong>Inference Modality:</strong> PyTorch Native FP16 with ONNX Runtime fallback</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c_v:
        st.markdown(
            f"""
            <div style="background: var(--ft-surface); border: 1px solid var(--ft-border); border-radius: 8px; padding: 18px;">
                <h4 style="margin: 0 0 10px 0; color: #8B5CF6;">💥 Spatiotemporal Violence Pipeline</h4>
                <ul style="color: var(--ft-text); font-size: 0.88rem; line-height: 1.6; margin: 0; padding-left: 18px;">
                    <li><strong>Architecture:</strong> S3D Separable 3D CNN (Baseline: R3D-18)</li>
                    <li><strong>Active Weights:</strong> <code>{config.violence_classifier.weights_path}</code></li>
                    <li><strong>Temporal Window:</strong> <code>{config.violence_classifier.clip_seconds}s ({config.violence_classifier.clip_frames} frames @ 8 Hz)</code></li>
                    <li><strong>Hysteresis Trigger / Release:</strong> <code>{config.violence_classifier.trigger_threshold} / {config.violence_classifier.release_threshold}</code></li>
                    <li><strong>Calibration:</strong> Temperature Scaled (T=1.28)</li>
                    <li><strong>Explainability:</strong> Temporal Grad-CAM Heatmap Overlay</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    render_models_page()
