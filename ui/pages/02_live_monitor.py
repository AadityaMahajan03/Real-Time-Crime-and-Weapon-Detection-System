"""Live AI Surveillance Wall: Real-time Video Upload, Live Webcam, and Model Inference."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import sqlite3
import struct
import textwrap
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import streamlit as st

from src.fasttrack.core.config import get_config
from src.fasttrack.core.types import Detection, Frame
from src.fasttrack.ml.tracker import WeaponTracker
from src.fasttrack.ml.violence_detector import ViolenceClassifier
from src.fasttrack.ml.weapon_detector import WeaponDetector
from ui.components import score_gauge_html, section_header


@st.cache_data
def get_siren_audio_b64() -> str:
    """Synthesize a dual-frequency sweeping acoustic emergency alarm buzzer in base64."""
    buf = io.BytesIO()
    sample_rate = 22050
    duration = 0.45  # seconds
    num_samples = int(sample_rate * duration)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = []
        for i in range(num_samples):
            t = i / float(sample_rate)
            # Sweeping frequency: 960 Hz -> 480 Hz
            freq = 960.0 - 480.0 * (i / float(num_samples))
            envelope = math.sin(math.pi * (i / float(num_samples)))
            val = int(18000 * envelope * math.sin(2.0 * math.pi * freq * t))
            samples.append(struct.pack("<h", max(-32767, min(32767, val))))
        wf.writeframes(b"".join(samples))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def siren_html_tag(siren_b64: str) -> str:
    """Return inline HTML5 audio and Web Audio synthesizer for immediate browser alarm."""
    return f"""<div style="display:none;">
    <audio autoplay>
        <source src="data:audio/wav;base64,{siren_b64}" type="audio/wav">
    </audio>
    <script>
    (function() {{
        try {{
            var AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (AudioCtx) {{
                var ctx = new AudioCtx();
                if (ctx.state === 'suspended') {{ ctx.resume(); }}
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                osc.type = 'sawtooth';
                osc.frequency.setValueAtTime(920, ctx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(460, ctx.currentTime + 0.35);
                gain.gain.setValueAtTime(0.35, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start();
                osc.stop(ctx.currentTime + 0.36);
            }}
        }} catch(e) {{}}
    }})();
    </script>
</div>"""


@st.cache_resource
def get_ml_models(checkpoint_path: str = "weights/weapon_yolo.pt"):
    """Load neural network checkpoints into memory once (GPU/CPU cached)."""
    try:
        weapon_detector = WeaponDetector(checkpoint_path=checkpoint_path)
    except Exception as e:
        weapon_detector = None
        st.error(f"Weapon Detector Initialization Warning: {e}")

    try:
        violence_classifier = ViolenceClassifier()
    except Exception as e:
        violence_classifier = None
        st.warning(f"Violence Classifier Warning: {e}")

    tracker = WeaponTracker(k=2, n=4, track_persistence_sec=2.0)
    return weapon_detector, violence_classifier, tracker


def draw_hud_and_detections(
    image: np.ndarray,
    detections: List[Detection],
    fps: float,
    latency_ms: float,
    device_label: str,
    source_title: str,
    violence_score: float = 0.0,
    violence_thresh: float = 0.60,
) -> np.ndarray:
    """Draw tactical HUD banner, bounding boxes, labels, and alert status onto frame."""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # 1. Top HUD Banner (semi-transparent dark slate)
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

    # Title & source on left
    cv2.putText(
        annotated,
        f"FAST-TRACK AI | {source_title.upper()}",
        (14, 27),
        cv2.FONT_HERSHEY_DUPLEX,
        0.56,
        (230, 234, 242),
        1,
        cv2.LINE_AA,
    )

    # Performance telemetry on right
    telemetry_str = f"FPS: {fps:.1f} | INFER: {latency_ms:.0f}ms | {device_label}"
    (tw, _), _ = cv2.getTextSize(telemetry_str, cv2.FONT_HERSHEY_DUPLEX, 0.50, 1)
    cv2.putText(
        annotated,
        telemetry_str,
        (max(14, w - tw - 14), 27),
        cv2.FONT_HERSHEY_DUPLEX,
        0.50,
        (34, 211, 238),  # Cyan
        1,
        cv2.LINE_AA,
    )

    has_weapon = len(detections) > 0
    has_violence = violence_score >= violence_thresh

    # 2. Draw Weapon Detections
    for det in detections:
        box = det.box
        x1, y1, x2, y2 = box.to_int_tuple()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        is_knife = "knife" in det.class_name.lower() or "scissors" in det.class_name.lower()
        box_color = (68, 68, 239) if is_knife else (11, 158, 245)  # Crimson or Amber

        # Main bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

        # Tactical corner brackets
        clen = max(6, min(18, (x2 - x1) // 4, (y2 - y1) // 4))
        # Top-left
        cv2.line(annotated, (x1, y1), (x1 + clen, y1), (255, 255, 255), 2)
        cv2.line(annotated, (x1, y1), (x1, y1 + clen), (255, 255, 255), 2)
        # Top-right
        cv2.line(annotated, (x2, y1), (x2 - clen, y1), (255, 255, 255), 2)
        cv2.line(annotated, (x2, y1), (x2, y1 + clen), (255, 255, 255), 2)
        # Bottom-left
        cv2.line(annotated, (x1, y2), (x1 + clen, y2), (255, 255, 255), 2)
        cv2.line(annotated, (x1, y2), (x1, y2 - clen), (255, 255, 255), 2)
        # Bottom-right
        cv2.line(annotated, (x2, y2), (x2 - clen, y2), (255, 255, 255), 2)
        cv2.line(annotated, (x2, y2), (x2, y2 - clen), (255, 255, 255), 2)

        # Label badge
        trk_info = f"TRK #{det.track_id}" if det.track_id is not None else ""
        label = f"{det.class_name.upper()} {det.confidence:.2f} {trk_info}".strip()
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)

        tag_y1 = max(44, y1 - lh - 8)
        tag_y2 = max(44 + lh + 8, y1)
        tag_x2 = min(w - 1, x1 + lw + 12)

        cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), box_color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 6, tag_y2 - 5),
            cv2.FONT_HERSHEY_DUPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Threat Status Banner and Perimeter Alert Border if triggered
    if has_weapon or has_violence:
        border_col = (0, 0, 230) if has_weapon and has_violence else ((0, 80, 240) if has_weapon else (210, 60, 180))
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), border_col, 4)

        overlay_bot = annotated.copy()
        cv2.rectangle(overlay_bot, (0, h - 38), (w, h), (20, 20, 180), -1)
        cv2.addWeighted(overlay_bot, 0.80, annotated, 0.20, 0, annotated)

        if has_weapon and has_violence:
            banner_text = "CRITICAL ALERT: CO-OCCURRING WEAPON & VIOLENCE ACTIVITY"
        elif has_weapon:
            banner_text = "CRITICAL ALERT: WEAPON THREAT DETECTED"
        else:
            banner_text = "WARNING: ELEVATED VIOLENCE ACTIVITY DETECTED"

        cv2.putText(
            annotated,
            f"⚠️ {banner_text}",
            (14, h - 13),
            cv2.FONT_HERSHEY_DUPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def log_live_incident(
    db_path: str,
    camera_id: str,
    camera_name: str,
    severity: str,
    summary: str,
    peak_weapon: float,
    peak_violence: float,
    detections: List[Detection],
    frame_annotated: Optional[np.ndarray] = None,
    source_video_path: Optional[str] = None,
) -> None:
    """Safely record live alert incident and cryptographic evidence into SQLite database."""
    try:
        conn = sqlite3.connect(db_path, timeout=5.0)
        cur = conn.cursor()
        inc_id = f"inc-{uuid.uuid4().hex[:10]}"
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        # 1. Ensure camera exists in registry
        cur.execute(
            """
            INSERT OR IGNORE INTO cameras
            (id, name, source_type, source_ref_masked, enabled, zones_json, thresholds_json, schedule_json, created_utc)
            VALUES (?, ?, 'live', ?, 1, '[]', '{}', '{}', ?);
            """,
            (camera_id, camera_name, source_video_path or camera_name, now_utc),
        )

        # 2. Insert Incident
        cur.execute(
            """
            INSERT INTO incidents (id, camera_id, severity, status, start_utc, peak_weapon_score, peak_violence_score, summary)
            VALUES (?, ?, ?, 'new', ?, ?, ?, ?);
            """,
            (inc_id, camera_id, severity, now_utc, round(peak_weapon, 3), round(peak_violence, 3), summary),
        )

        # 3. Save Evidence Snapshot if frame available
        ev_snap_id = None
        if frame_annotated is not None and frame_annotated.size > 0:
            ev_dir = Path("evidence/snapshots")
            ev_dir.mkdir(parents=True, exist_ok=True)
            snap_path = ev_dir / f"{inc_id}_snapshot.jpg"
            cv2.imwrite(str(snap_path), frame_annotated)

            with open(snap_path, "rb") as f:
                snap_bytes = f.read()
            snap_sha = hashlib.sha256(snap_bytes).hexdigest()
            snap_size = len(snap_bytes)

            ev_snap_id = f"ev-{uuid.uuid4().hex[:10]}"
            cur.execute(
                """
                INSERT INTO evidence (id, kind, path, sha256, size_bytes, encrypted, created_utc)
                VALUES (?, 'snapshot', ?, ?, ?, 0, ?);
                """,
                (ev_snap_id, str(snap_path).replace("\\", "/"), snap_sha, snap_size, now_utc),
            )
            # Link snapshot to incident_evidence
            cur.execute(
                """
                INSERT OR IGNORE INTO incident_evidence (incident_id, evidence_id, attached_utc)
                VALUES (?, ?, ?);
                """,
                (inc_id, ev_snap_id, now_utc),
            )

        # 4. Link video clip evidence if valid file source
        if source_video_path and Path(source_video_path).is_file():
            vpath = Path(source_video_path)
            try:
                with open(vpath, "rb") as f:
                    v_sha = hashlib.sha256(f.read()).hexdigest()
                v_size = vpath.stat().st_size
                ev_clip_id = f"ev-clip-{uuid.uuid4().hex[:10]}"
                cur.execute(
                    """
                    INSERT INTO evidence (id, kind, path, sha256, size_bytes, encrypted, created_utc)
                    VALUES (?, 'clip', ?, ?, ?, 0, ?);
                    """,
                    (ev_clip_id, str(vpath).replace("\\", "/"), v_sha, v_size, now_utc),
                )
                cur.execute(
                    """
                    INSERT OR IGNORE INTO incident_evidence (incident_id, evidence_id, attached_utc)
                    VALUES (?, ?, ?);
                    """,
                    (inc_id, ev_clip_id, now_utc),
                )
                cur.execute(
                    """
                    INSERT OR REPLACE INTO clips (
                        id, incident_id, camera_id, raw_clip_path, annotated_clip_path,
                        thumbnail_path, sha256, duration_sec, size_bytes, codec, verification_status, created_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 10.0, ?, 'h264', 'verified', ?);
                    """,
                    (
                        f"clip-{inc_id}",
                        inc_id,
                        camera_id,
                        str(vpath).replace("\\", "/"),
                        str(vpath).replace("\\", "/"),
                        str(snap_path).replace("\\", "/") if ev_snap_id else "",
                        v_sha,
                        v_size,
                        now_utc,
                    ),
                )
            except Exception:
                pass

        # 5. Insert Atomic Events
        for det in detections:
            ev_id = f"ev-{uuid.uuid4().hex[:10]}"
            cur.execute(
                """
                INSERT INTO events (id, incident_id, camera_id, ts_utc, event_type, class_name, raw_score, calibrated_score, bbox_json, track_id, evidence_id)
                VALUES (?, ?, ?, ?, 'weapon_detection', ?, ?, ?, ?, ?, ?);
                """,
                (
                    ev_id,
                    inc_id,
                    camera_id,
                    now_utc,
                    det.class_name,
                    round(det.confidence, 4),
                    round(det.confidence, 4),
                    json.dumps(det.box.to_list()),
                    det.track_id,
                    ev_snap_id,
                ),
            )

        conn.commit()
        conn.close()
    except Exception as e:
        # Non-fatal database log warning
        print(f"Database incident logging note: {e}")


def render_live_monitor_page() -> None:
    config = get_config()
    db_path = config.storage.db_path

    c_m1, c_m2 = st.columns([2, 1])
    with c_m1:
        model_choice = st.selectbox(
            "Threat Detection Deep Learning Model",
            [
                "Subh775 Fine-Tuned Threat Model (Gun, Knife, Grenade) [weights/weapon_yolo.pt]",
                "COCO YOLO11n Model (Knife, Scissors) [weights/yolo11n.pt]",
            ],
            index=0,
            help="Select which deep learning checkpoint to use for bounding box inference.",
        )
    checkpoint_file = (
        "weights/weapon_yolo.pt" if "weapon_yolo.pt" in model_choice else "weights/yolo11n.pt"
    )

    # Initialize models
    weapon_detector, violence_classifier, tracker = get_ml_models(checkpoint_file)

    import torch

    has_cuda = torch.cuda.is_available()
    device_label = "CUDA (RTX 3050)" if has_cuda else "CPU Host"

    st.markdown("### 📹 Live AI Video Surveillance & Model Inference")
    st.caption("Upload video files or connect live webcams / RTSP streams to execute real-time YOLO11n weapon detection and spatiotemporal action recognition.")

    # Input Source Selection (Driven strictly by real hardware and fleet registry)
    from src.fasttrack.storage.db import DatabaseManager
    from src.fasttrack.storage.repositories import CameraRepository
    db = DatabaseManager(config.storage.db_path)
    camera_repo = CameraRepository(db)
    active_fleet_cams = camera_repo.get_all(enabled_only=True)

    source_choice = st.radio(
        "Select Surveillance Source",
        ["📹 Active Fleet Cameras", "📷 Integrated / USB Webcam", "📁 Test Video Upload", "🌐 RTSP Stream"],
        horizontal=True,
    )

    selected_video_source: Optional[str | int] = None
    source_title = "Camera-01"
    active_camera_id = "cam-live-01"

    if source_choice == "📹 Active Fleet Cameras":
        if active_fleet_cams:
            cam_options = {c["id"]: f"{c['name']} ({c['source_type'].upper()})" for c in active_fleet_cams}
            selected_cam_id = st.selectbox(
                "Select Active Camera from Fleet:",
                options=list(cam_options.keys()),
                format_func=lambda x: cam_options.get(x, x),
            )
            active_camera_id = selected_cam_id
            selected_cam_record = camera_repo.get_by_id(selected_cam_id)
            if selected_cam_record:
                src_type = selected_cam_record["source_type"]
                src_ref = selected_cam_record["source_ref_masked"]
                if src_type == "webcam":
                    try:
                        selected_video_source = int(src_ref)
                    except ValueError:
                        selected_video_source = 0
                else:
                    selected_video_source = src_ref
                source_title = selected_cam_record["name"]
        else:
            st.info("No cameras are currently marked active in fleet. Go to 'Cameras & Zones' page to activate surveillance sources.")

    elif source_choice == "📁 Test Video Upload":
        c_up1, c_up2 = st.columns([2, 1])
        with c_up1:
            uploaded_file = st.file_uploader(
                "Upload Surveillance Video File (MP4, AVI, MOV, MKV)",
                type=["mp4", "avi", "mov", "mkv"],
                help="Upload real CCTV footage or test video to run edge AI inference.",
            )
        with c_up2:
            st.info("💡 **Surveillance File**: Upload CCTV recordings, security footage, or incident test clips.")

        if uploaded_file is not None:
            upload_dir = Path("data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            saved_path = upload_dir / uploaded_file.name
            with open(saved_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            selected_video_source = str(saved_path)
            active_camera_id = f"cam-up-{uploaded_file.name[:8]}"
            source_title = f"File: {uploaded_file.name}"
            st.success(f"Loaded: `{uploaded_file.name}` ({uploaded_file.size / (1024*1024):.1f} MB)")

    elif source_choice == "📷 Integrated / USB Webcam":
        c_cam1, c_cam2 = st.columns([1, 2])
        with c_cam1:
            cam_idx = st.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)
        with c_cam2:
            st.info("💡 **Webcam Tip**: Index `0` is typically the integrated camera. External USB surveillance cams are usually `1` or `2`.")
        selected_video_source = int(cam_idx)
        active_camera_id = f"cam-webcam-{cam_idx}"
        source_title = f"Webcam Device #{cam_idx}"

    elif source_choice == "🌐 RTSP Stream":
        rtsp_url = st.text_input("Enter RTSP Stream URL", value="rtsp://admin:pass@192.168.1.100:554/live")
        selected_video_source = rtsp_url
        active_camera_id = "cam-rtsp-01"
        source_title = "RTSP Network Camera"

    st.divider()

    # Layout: Video Wall on Left (2.4), Live Threat Analytics & Alerts on Right (1.0)
    col_video, col_telemetry = st.columns([2.4, 1.0])

    with col_telemetry:
        section_header("Live Threat Telemetry")

        # Live status pill
        status_box = st.empty()
        status_html = """<div style="background: var(--ft-surface); border: 1px solid var(--ft-border); border-radius: 8px; padding: 12px; margin-bottom: 12px; text-align: center;">
    <div style="font-size: 0.8rem; color: var(--ft-muted); margin-bottom: 4px;">CURRENT THREAT LEVEL</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: #22C55E;">● SECURE / MONITORING</div>
</div>"""
        status_box.markdown(textwrap.dedent(status_html).strip(), unsafe_allow_html=True)

        weapon_gauge_box = st.empty()
        weapon_gauge_box.markdown(score_gauge_html(0.0, 0.40, "Weapon Confidence"), unsafe_allow_html=True)

        violence_gauge_box = st.empty()
        violence_gauge_box.markdown(score_gauge_html(0.0, 0.60, "Violence Action Probability"), unsafe_allow_html=True)

        section_header("Detection Log")
        detections_feed_box = st.empty()
        detections_feed_box.caption("No weapons or violent acts detected yet.")

    with col_video:
        # Tuning & Inference Controls
        c_p1, c_p2, c_p3 = st.columns([1, 1, 1])
        with c_p1:
            w_threshold = st.slider("Weapon Threshold", min_value=0.20, max_value=0.90, value=0.35, step=0.05)
        with c_p2:
            v_threshold = st.slider("Violence Threshold", min_value=0.30, max_value=0.90, value=0.55, step=0.05)
        with c_p3:
            max_fps = st.slider("Processing FPS Limit", min_value=5, max_value=30, value=20, step=5)

        # Alarm Siren Controls
        c_a1, c_a2 = st.columns([1, 1])
        with c_a1:
            enable_audio_alarm = st.toggle(
                "🚨 Audible Siren on Threat",
                value=True,
                help="Emit urgent audible security siren buzzer when weapon or violent altercation is detected.",
            )
        with c_a2:
            test_siren = st.button("🔊 Test Alarm Siren Sound", use_container_width=True)

        alarm_banner_box = st.empty()
        alarm_audio_box = st.empty()

        if test_siren:
            s_b64 = get_siren_audio_b64()
            alarm_audio_box.markdown(siren_html_tag(s_b64), unsafe_allow_html=True)
            st.toast("🚨 Alarm Siren Sound Tested Successfully!", icon="🔊")

        # Video stream display container
        video_frame_box = st.empty()

        # Action Buttons
        c_btn1, c_btn2 = st.columns([1, 1])
        with c_btn1:
            start_clicked = st.button("▶️ Start Live AI Detection", type="primary", use_container_width=True)
        with c_btn2:
            stop_clicked = st.button("⏹️ Stop Stream", type="secondary", use_container_width=True)

        if stop_clicked:
            st.session_state["live_stream_running"] = False

        if start_clicked:
            st.session_state["live_stream_running"] = True

        # Initial placeholder when stream is idle
        if not st.session_state.get("live_stream_running", False):
            if selected_video_source is not None:
                video_frame_box.markdown(
                    f"""<div style="background: #0B0F17; border: 1px dashed var(--ft-border); border-radius: 8px; padding: 60px 20px; text-align: center;">
    <div style="font-size: 2.5rem; margin-bottom: 12px;">🛡️</div>
    <div style="font-size: 1.1rem; font-weight: 600; color: var(--ft-text); margin-bottom: 6px;">Source Configured: {source_title}</div>
    <div style="color: var(--ft-muted); font-size: 0.88rem; max-width: 450px; margin: 0 auto 16px auto;">
        Click <strong>▶️ Start Live AI Detection</strong> above to activate real-time YOLO11n object detection and S3D spatiotemporal violence classification.
    </div>
</div>""",
                    unsafe_allow_html=True,
                )
            else:
                video_frame_box.markdown(
                    """<div style="background: #0B0F17; border: 1px dashed var(--ft-border); border-radius: 8px; padding: 60px 20px; text-align: center;">
    <div style="font-size: 2.5rem; margin-bottom: 12px;">📹</div>
    <div style="font-size: 1.1rem; font-weight: 600; color: var(--ft-text); margin-bottom: 6px;">No Video Source Selected</div>
    <div style="color: var(--ft-muted); font-size: 0.88rem;">Please upload a video file or choose a webcam source above to begin surveillance.</div>
</div>""",
                    unsafe_allow_html=True,
                )

        # Active Inference Loop
        if st.session_state.get("live_stream_running", False) and selected_video_source is not None:
            import sys
            if isinstance(selected_video_source, int):
                backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
                cap = cv2.VideoCapture(selected_video_source, backend)
            elif isinstance(selected_video_source, str) and selected_video_source.startswith("rtsp://"):
                import os
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000000|rtsp_transport;tcp"
                cap = cv2.VideoCapture(selected_video_source)
            else:
                cap = cv2.VideoCapture(selected_video_source)

            if not cap.isOpened():
                st.error(f"Failed to open video source: `{selected_video_source}`. Please verify file path or camera connection.")
                st.session_state["live_stream_running"] = False
            else:
                last_db_log_time = 0.0
                last_siren_time = 0.0
                frame_count = 0
                prev_time = time.perf_counter()
                rolling_fps = 0.0
                recent_detection_events = []
                siren_b64 = get_siren_audio_b64()

                while st.session_state.get("live_stream_running", False):
                    ret, frame_bgr = cap.read()
                    if not ret:
                        # For video files, loop back to the beginning
                        if isinstance(selected_video_source, str) and Path(selected_video_source).exists():
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, frame_bgr = cap.read()
                            if not ret:
                                break
                        else:
                            break

                    now_mono = time.perf_counter()
                    dt = now_mono - prev_time
                    prev_time = now_mono
                    current_fps = (1.0 / dt) if dt > 0 else 0.0
                    rolling_fps = 0.9 * rolling_fps + 0.1 * current_fps if rolling_fps > 0 else current_fps

                    frame_count += 1
                    t_infer_start = time.perf_counter()

                    # 1. Weapon Detection (YOLO with dynamic confidence slider)
                    frame_obj = Frame(
                        image=frame_bgr,
                        ts_utc=datetime.now(timezone.utc),
                        source_ts=now_mono,
                        frame_idx=frame_count,
                        source_id="CAM-01",
                    )

                    detections: List[Detection] = []
                    if weapon_detector:
                        weapon_detector.settings.default_confidence_threshold = w_threshold
                        detections = weapon_detector.detect(frame_obj)

                    # 2. Tracking (ByteTrack) - keep all valid detections active for rendering
                    if tracker and detections:
                        tracker.update(detections)
                    active_dets = detections

                    # 3. Spatiotemporal Violence Classification (S3D)
                    v_score = 0.0
                    if violence_classifier:
                        v_pred = violence_classifier.push_frame(frame_obj)
                        if v_pred is not None and v_pred.is_buffer_ready:
                            v_score = v_pred.calibrated_score

                    infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

                    # 4. Annotate Frame with Tactical HUD & Bounding Boxes
                    annotated_bgr = draw_hud_and_detections(
                        image=frame_bgr,
                        detections=active_dets,
                        fps=rolling_fps,
                        latency_ms=infer_ms,
                        device_label=device_label,
                        source_title=source_title,
                        violence_score=v_score,
                        violence_thresh=v_threshold,
                    )

                    # 5. Render Streamlit Frame
                    frame_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    video_frame_box.image(frame_rgb, channels="RGB", use_container_width=True)

                    # 6. Update Real-Time Metrics & Threat Gauges
                    peak_w_score = max([d.confidence for d in active_dets], default=0.0)
                    weapon_gauge_box.markdown(score_gauge_html(peak_w_score, w_threshold, "Weapon Confidence"), unsafe_allow_html=True)
                    violence_gauge_box.markdown(score_gauge_html(v_score, v_threshold, "Violence Probability"), unsafe_allow_html=True)

                    has_w_threat = peak_w_score >= w_threshold
                    has_v_threat = v_score >= v_threshold

                    # Threat pill update
                    if has_w_threat and has_v_threat:
                        status_html = """<div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #EF4444; border-radius: 8px; padding: 12px; margin-bottom: 12px; text-align: center;">
    <div style="font-size: 0.8rem; color: #F87171; margin-bottom: 4px;">CURRENT THREAT LEVEL</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: #EF4444;">🔴 HIGH (WEAPON + VIOLENCE)</div>
</div>"""
                    elif has_w_threat:
                        status_html = """<div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; border-radius: 8px; padding: 12px; margin-bottom: 12px; text-align: center;">
    <div style="font-size: 0.8rem; color: #FBBF24; margin-bottom: 4px;">CURRENT THREAT LEVEL</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: #F59E0B;">🟠 MEDIUM (WEAPON CONFIRMED)</div>
</div>"""
                    elif has_v_threat:
                        status_html = """<div style="background: rgba(139, 92, 246, 0.15); border: 1px solid #8B5CF6; border-radius: 8px; padding: 12px; margin-bottom: 12px; text-align: center;">
    <div style="font-size: 0.8rem; color: #A78BFA; margin-bottom: 4px;">CURRENT THREAT LEVEL</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: #8B5CF6;">🟣 MEDIUM (VIOLENT ACTIVITY)</div>
</div>"""
                    else:
                        status_html = """<div style="background: var(--ft-surface); border: 1px solid var(--ft-border); border-radius: 8px; padding: 12px; margin-bottom: 12px; text-align: center;">
    <div style="font-size: 0.8rem; color: var(--ft-muted); margin-bottom: 4px;">CURRENT THREAT LEVEL</div>
    <div style="font-size: 1.1rem; font-weight: 700; color: #22C55E;">● SECURE / MONITORING</div>
</div>"""
                    status_box.markdown(textwrap.dedent(status_html).strip(), unsafe_allow_html=True)

                    # High-Priority Flashing Security Alarm Banner & Audible Siren
                    if has_w_threat or has_v_threat:
                        if has_w_threat and has_v_threat:
                            alarm_title = "CRITICAL ALERT: WEAPON & VIOLENCE DETECTED"
                            alarm_color = "#DC2626"
                            alarm_desc = f"Max Weapon Conf: {peak_w_score:.1%} | Violence Score: {v_score:.1%}"
                        elif has_w_threat:
                            threat_names = ", ".join(sorted({d.class_name.upper() for d in active_dets})) or "WEAPON"
                            alarm_title = f"SECURITY ALERT: {threat_names} DETECTED"
                            alarm_color = "#EA580C"
                            alarm_desc = f"Max Weapon Conf: {peak_w_score:.1%} | Target Count: {len(active_dets)}"
                        else:
                            alarm_title = "SECURITY ALERT: ELEVATED VIOLENT ACTIVITY DETECTED"
                            alarm_color = "#7C3AED"
                            alarm_desc = f"Spatiotemporal Violence Probability: {v_score:.1%}"

                        alarm_banner_box.markdown(
                            f"""<div style="background: linear-gradient(90deg, #1E1B4B 0%, {alarm_color} 50%, #1E1B4B 100%);
                                            color: #FFFFFF;
                                            padding: 12px 18px;
                                            border-radius: 8px;
                                            margin-bottom: 12px;
                                            display: flex;
                                            align-items: center;
                                            justify-content: space-between;
                                            box-shadow: 0 0 22px {alarm_color}AA;
                                            border: 2px solid #FFFFFF55;">
                                <div style="display: flex; align-items: center; gap: 12px;">
                                    <span style="font-size: 1.8rem;">🚨</span>
                                    <div>
                                        <strong style="font-size: 1.05rem; letter-spacing: 0.5px;">{alarm_title}</strong>
                                        <div style="font-size: 0.82rem; opacity: 0.95; margin-top: 2px;">{alarm_desc}</div>
                                    </div>
                                </div>
                                <div style="background: rgba(0, 0, 0, 0.45); padding: 5px 12px; border-radius: 6px; font-weight: 700; font-size: 0.80rem; border: 1px solid rgba(255, 255, 255, 0.4); text-transform: uppercase;">
                                    DISPATCH ACTIVE
                                </div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                        if enable_audio_alarm and (now_mono - last_siren_time >= 1.2):
                            last_siren_time = now_mono
                            alarm_audio_box.markdown(siren_html_tag(siren_b64), unsafe_allow_html=True)
                    else:
                        alarm_banner_box.empty()
                        alarm_audio_box.empty()

                    # Update Detections Log Feed
                    if active_dets:
                        for d in active_dets:
                            now_time_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
                            recent_detection_events.insert(0, {
                                "time": now_time_str,
                                "class": d.class_name.upper(),
                                "conf": d.confidence,
                                "track_id": d.track_id,
                            })
                        recent_detection_events = recent_detection_events[:6]

                        feed_html = '<div style="display: flex; flex-direction: column; gap: 6px;">'
                        for ev in recent_detection_events:
                            feed_html += f"""<div style="background: var(--ft-surface); border: 1px solid var(--ft-border); border-radius: 6px; padding: 6px 10px; font-size: 0.78rem;">
    <div style="display: flex; justify-content: space-between;">
        <strong style="color: #EF4444;">{ev['class']}</strong>
        <span style="color: var(--ft-muted);">{ev['time']}</span>
    </div>
    <div style="color: var(--ft-text); margin-top: 2px;">Conf: <strong>{ev['conf']:.2f}</strong> | Track: #{ev['track_id'] or '1'}</div>
</div>"""
                        feed_html += "</div>"
                        detections_feed_box.markdown(textwrap.dedent(feed_html).strip(), unsafe_allow_html=True)

                    # 7. Safe SQLite Incident Logging with 5-second cooldown and cryptographic evidence
                    if (has_w_threat or has_v_threat) and (now_mono - last_db_log_time >= 5.0):
                        last_db_log_time = now_mono
                        sev = "HIGH" if (has_w_threat and has_v_threat) else "MEDIUM"
                        summ = f"Live detection on {source_title}: {', '.join([d.class_name for d in active_dets]) or 'violence activity'}"
                        log_live_incident(
                            db_path=db_path,
                            camera_id=active_camera_id,
                            camera_name=source_title,
                            severity=sev,
                            summary=summ,
                            peak_weapon=peak_w_score,
                            peak_violence=v_score,
                            detections=active_dets,
                            frame_annotated=annotated_bgr,
                            source_video_path=selected_video_source if isinstance(selected_video_source, str) else None,
                        )
                        # Notify alarm service
                        if "alarm_service" in st.session_state:
                            st.session_state["alarm_service"].update_threat(
                                camera_id=active_camera_id,
                                weapon_score=peak_w_score,
                                violence_score=v_score,
                            )

                    # Pace loop according to max_fps
                    target_delay = 1.0 / max_fps
                    elapsed = time.perf_counter() - now_mono
                    if elapsed < target_delay:
                        time.sleep(target_delay - elapsed)

                cap.release()
                alarm_banner_box.empty()
                alarm_audio_box.empty()


if __name__ == "__main__":
    render_live_monitor_page()
