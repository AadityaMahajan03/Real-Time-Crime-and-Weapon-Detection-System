"""Combined Real-Time Weapon & Violence Threat Analysis Engine.

Integrates:
1. João Assalim's Weapon Detection:
   - 2D Discrete Wavelet Transform (Haar / db2 / sym2) + YOLOv8.
   - Robust detection of firearms and knives under severe CCTV compression.
2. Abduulrahman Khalid's Violence Detection:
   - MobileNetV2 spatial feature extractor + Bidirectional LSTM (BiLSTM).
   - Spatiotemporal sequence classification for assault, striking, and physical altercations.
3. Unified Threat Fusion & HUD Overlay:
   - Live video evaluation with bounding boxes, high-frequency wavelet edge diagnostic,
     violence probability progression, and fused incident level.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.fasttrack.core.config import AppConfig, ViolenceClassifierSettings, WeaponDetectorSettings
from src.fasttrack.core.types import Frame
from src.fasttrack.ml.violence_detector import ViolenceClassifier
from src.fasttrack.ml.wavelet_enhancer import WaveletEnhancer
from src.fasttrack.ml.weapon_detector import WeaponDetector


def analyze_combined(
    video_path: str,
    output_path: Optional[str] = None,
    wavelet_mode: str = "haar",
    violence_arch: str = "mobilenet_bilstm",
    max_frames: Optional[int] = None,
    render_video: bool = True,
) -> Dict[str, Any]:
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(vpath))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / max(fps, 1.0)

    print("\n=======================================================")
    print(f"COMBINED THREAT DETECTION: {vpath.name}")
    print(f"Resolution: {width}x{height} | FPS: {fps:.1f} | Duration: {duration_sec:.2f}s")
    print(f"Weapon Technique: YOLOv8 + 2D DWT Wavelet ({wavelet_mode.upper()})")
    print(f"Violence Technique: MobileNetV2 + BiLSTM ({violence_arch.upper()})")
    print("=======================================================\n")

    # 1. Initialize Weapon Detector
    w_settings = WeaponDetectorSettings(
        model_checkpoint="weights/joao_models/normal_compressed_best.pt",
        wavelet_checkpoint=f"weights/joao_models/{wavelet_mode}_compressed_best.pt"
        if wavelet_mode == "haar"
        else f"weights/joao_models/{wavelet_mode}_best.pt",
        wavelet_mode=wavelet_mode if wavelet_mode != "standard" else "none",
        enable_wavelet_fusion=(wavelet_mode == "fusion"),
        default_confidence_threshold=0.45,
    )
    weapon_detector = WeaponDetector(settings=w_settings)
    wavelet_enhancer = WaveletEnhancer(default_wavelet="haar" if wavelet_mode in ("standard", "fusion") else wavelet_mode)

    # 2. Initialize Violence Classifier
    v_settings = ViolenceClassifierSettings(
        model_architecture=violence_arch,
        clip_seconds=1.5,
        clip_frames=16,
        stride_seconds=0.2,
        trigger_threshold=0.60,
    )
    violence_classifier = ViolenceClassifier(settings=v_settings)

    out_writer = None
    if render_video and output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Side-by-side output: [Annotated Detection | Wavelet Detail Map]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width * 2, height))

    frame_idx = 0
    weapon_frames_count = 0
    violence_frames_count = 0
    all_weapons_count = 0
    weapon_classes: Dict[str, int] = {}
    violence_scores: List[float] = []
    weapon_confidences: List[float] = []
    latencies: List[float] = []

    t0 = time.perf_counter()

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        cur_ts = frame_idx / fps
        frame_obj = Frame(
            image=frame_bgr,
            ts_utc=time.time(),
            source_ts=cur_ts,
            frame_idx=frame_idx,
            source_id="threat_cam_01",
        )

        t_start = time.perf_counter()

        # Step A: Weapon Detection Pass
        weapon_dets = weapon_detector.detect(frame_obj)

        # Step B: Violence Spatiotemporal Pass
        person_boxes = weapon_detector.last_detected_persons
        v_pred = violence_classifier.push_frame(frame_obj, person_boxes=person_boxes)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(elapsed_ms)

        # Statistics
        if weapon_dets:
            weapon_frames_count += 1
            all_weapons_count += len(weapon_dets)
            for d in weapon_dets:
                weapon_classes[d.class_name] = weapon_classes.get(d.class_name, 0) + 1
                weapon_confidences.append(d.confidence)

        current_v_score = v_pred.fused_score if v_pred else 0.0
        if current_v_score > 0.0:
            violence_scores.append(current_v_score)
        if current_v_score >= v_settings.trigger_threshold:
            violence_frames_count += 1

        if out_writer:
            annotated = frame_bgr.copy()

            # Draw weapon bounding boxes
            for d in weapon_dets:
                box = d.box
                x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
                color = (0, 0, 255) if "gun" in d.class_name else (0, 140, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                tag = f"{d.class_name.upper()} {d.confidence:.2f}"
                cv2.putText(annotated, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Determine Threat Level
            has_weapon = len(weapon_dets) > 0
            is_violence = current_v_score >= v_settings.trigger_threshold

            if has_weapon and is_violence:
                threat_label = "CRITICAL: ARMED VIOLENT INCIDENT"
                threat_color = (0, 0, 255)
            elif has_weapon:
                threat_label = "ALERT: WEAPON DETECTED"
                threat_color = (0, 69, 255)
            elif is_violence:
                threat_label = "ALERT: VIOLENT ALTERCATION DETECTED"
                threat_color = (0, 165, 255)
            else:
                threat_label = "SECURE / NORMAL"
                threat_color = (0, 255, 0)

            # Draw HUD Top Bar
            cv2.rectangle(annotated, (0, 0), (width, 40), (20, 20, 20), -1)
            cv2.putText(annotated, threat_label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, threat_color, 2)

            hud_metrics = f"Weapons: {len(weapon_dets)} | Violence Score: {current_v_score:.2f} | {elapsed_ms:.1f}ms"
            cv2.putText(annotated, hud_metrics, (width - 380, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            # Draw Wavelet Sub-band visualizer
            lh_map = wavelet_enhancer.extract_subband(
                frame_bgr,
                wavelet="haar" if wavelet_mode in ("standard", "fusion") else wavelet_mode,
                subband="LH",
                target_size=(width, height),
            )
            cv2.rectangle(lh_map, (0, 0), (width, 40), (20, 20, 20), -1)
            cv2.putText(
                lh_map,
                f"2D DWT Wavelet LH Detail (Assalim 2025) | Frame {frame_idx}/{total_frames}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Side-by-side stitch
            combined = np.hstack([annotated, lh_map])
            out_writer.write(combined)

        if frame_idx % 50 == 0:
            print(
                f"  Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | "
                f"Avg Latency: {np.mean(latencies):.1f}ms | Weapons: {all_weapons_count} | "
                f"Peak Violence Score: {np.max(violence_scores) if violence_scores else 0.0:.2f}"
            )

    cap.release()
    if out_writer:
        out_writer.release()

    total_time = time.perf_counter() - t0
    avg_fps = frame_idx / max(total_time, 0.001)

    results = {
        "video": vpath.name,
        "wavelet_technique": f"2D DWT ({wavelet_mode.upper()}) + YOLOv8 (Almeida & Assalim 2025)",
        "violence_technique": f"MobileNetV2 + BiLSTM ({violence_arch}) (Khalid 2022)",
        "total_frames_analyzed": frame_idx,
        "duration_sec": round(duration_sec, 2),
        "total_processing_time_sec": round(total_time, 2),
        "pipeline_fps": round(avg_fps, 2),
        "mean_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "weapon_analysis": {
            "frames_with_weapon": weapon_frames_count,
            "weapon_frame_rate_pct": round(weapon_frames_count / max(frame_idx, 1) * 100.0, 2),
            "total_weapon_detections": all_weapons_count,
            "class_breakdown": weapon_classes,
            "mean_weapon_confidence": round(float(np.mean(weapon_confidences)), 3) if weapon_confidences else 0.0,
        },
        "violence_analysis": {
            "frames_with_violence_trigger": violence_frames_count,
            "mean_violence_score": round(float(np.mean(violence_scores)), 3) if violence_scores else 0.0,
            "peak_violence_score": round(float(np.max(violence_scores)), 3) if violence_scores else 0.0,
        },
        "annotated_video_path": str(output_path) if output_path else None,
    }

    print("\n----------------- COMBINED SUMMARY -----------------")
    print(json.dumps(results, indent=2))
    print("----------------------------------------------------\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combined Weapon & Violence Detection Engine")
    parser.add_argument("--video", type=str, default="data/joao_detected_objects_video.mp4")
    parser.add_argument("--output", type=str, default="data/combined_threat_analysis.mp4")
    parser.add_argument("--wavelet_mode", type=str, default="haar", choices=["standard", "haar", "db2", "sym2", "fusion"])
    parser.add_argument("--violence_arch", type=str, default="mobilenet_bilstm")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_render", action="store_true")

    args = parser.parse_args()
    analyze_combined(
        video_path=args.video,
        output_path=args.output,
        wavelet_mode=args.wavelet_mode,
        violence_arch=args.violence_arch,
        max_frames=args.max_frames,
        render_video=not args.no_render,
    )
