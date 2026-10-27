"""Video Analysis and Wavelet Weapon Detection Benchmark Script.

Applies the research technique from:
'FIREARMS AND KNIVES DETECTION USING YOLO AND WAVELET FOR SURVEILLANCE SYSTEM'
(Almeida & Assalim, 2025).

Features:
- Analyzes surveillance videos frame-by-frame.
- Compares Standard YOLO vs 2D DWT Wavelet YOLO (Haar/Db2/Symlet) vs Wavelet Fusion.
- Computes latency, detection counts, confidence distributions, and tracking stats.
- Generates a side-by-side diagnostic visualization (RGB Feed | Wavelet LH Subband | Detection Annotations).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.fasttrack.core.config import WeaponDetectorSettings
from src.fasttrack.core.types import Frame
from src.fasttrack.ml.weapon_detector import WeaponDetector
from src.fasttrack.ml.wavelet_enhancer import WaveletEnhancer


def analyze_video(
    video_path: str,
    output_path: Optional[str] = None,
    mode: str = "haar",
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

    print(f"\n=======================================================")
    print(f"ANALYZING VIDEO: {vpath.name}")
    print(f"Resolution: {width}x{height} | FPS: {fps:.1f} | Frames: {total_frames} | Duration: {duration_sec:.2f}s")
    print(f"Wavelet Mode: {mode}")
    print(f"=======================================================\n")

    settings = WeaponDetectorSettings(
        model_checkpoint="weights/joao_models/normal_compressed_best.pt",
        wavelet_checkpoint=f"weights/joao_models/{mode}_compressed_best.pt" if mode == "haar" else f"weights/joao_models/{mode}_best.pt",
        wavelet_mode=mode if mode != "standard" else "none",
        enable_wavelet_fusion=(mode == "fusion"),
        default_confidence_threshold=0.45,
    )

    detector = WeaponDetector(settings=settings)
    wavelet_enhancer = WaveletEnhancer(default_wavelet="haar" if mode in ("standard", "fusion") else mode)

    out_writer = None
    if render_video and output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Side-by-side video: (width * 2, height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width * 2, height))

    frame_idx = 0
    detected_frames_count = 0
    all_detections_count = 0
    class_counts: Dict[str, int] = {}
    confidences: List[float] = []
    latencies: List[float] = []

    t0 = time.perf_counter()

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        frame_obj = Frame(
            image=frame_bgr,
            ts_utc=time.time(),
            source_ts=frame_idx / fps,
            frame_idx=frame_idx,
            source_id="analysis_cam",
        )

        t_det_start = time.perf_counter()
        detections = detector.detect(frame_obj)
        lat_ms = (time.perf_counter() - t_det_start) * 1000.0
        latencies.append(lat_ms)

        if detections:
            detected_frames_count += 1
            all_detections_count += len(detections)
            for d in detections:
                class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1
                confidences.append(d.confidence)

        if out_writer:
            # Generate Wavelet diagnostic subband
            lh_vis = wavelet_enhancer.extract_subband(
                frame_bgr,
                wavelet="haar" if mode == "standard" else mode,
                subband="LH",
                target_size=(width, height),
            )

            # Draw annotations on original frame
            annotated = frame_bgr.copy()
            for d in detections:
                box = d.box
                x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
                color = (0, 0, 255) if "gun" in d.class_name else (0, 165, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{d.class_name.upper()} {d.confidence:.2f}"
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                )

            # Add HUD banner
            hud = f"Frame {frame_idx}/{total_frames} | Mode: {mode.upper()} | Detections: {len(detections)} | {lat_ms:.1f}ms"
            cv2.rectangle(annotated, (0, 0), (width, 28), (20, 20, 20), -1)
            cv2.putText(annotated, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)

            cv2.rectangle(lh_vis, (0, 0), (width, 28), (20, 20, 20), -1)
            cv2.putText(lh_vis, f"DWT LH Sub-band Detail ({mode})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            combined = np.hstack([annotated, lh_vis])
            out_writer.write(combined)

        if frame_idx % 50 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%) | "
                  f"Avg Latency: {np.mean(latencies):.2f}ms | Weapons found so far: {all_detections_count}")

    cap.release()
    if out_writer:
        out_writer.release()

    total_proc_time = time.perf_counter() - t0
    avg_fps = frame_idx / max(total_proc_time, 0.001)

    results = {
        "video": vpath.name,
        "mode": mode,
        "total_frames": frame_idx,
        "video_duration_sec": round(duration_sec, 2),
        "processing_time_sec": round(total_proc_time, 2),
        "average_fps": round(avg_fps, 2),
        "mean_inference_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
        "p95_inference_latency_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
        "detected_frames_count": detected_frames_count,
        "detection_frame_rate_pct": round(detected_frames_count / max(frame_idx, 1) * 100.0, 2),
        "total_weapon_detections": all_detections_count,
        "class_breakdown": class_counts,
        "confidence_stats": {
            "mean": round(float(np.mean(confidences)), 3) if confidences else 0.0,
            "min": round(float(np.min(confidences)), 3) if confidences else 0.0,
            "max": round(float(np.max(confidences)), 3) if confidences else 0.0,
        },
        "annotated_video_path": str(output_path) if output_path else None,
    }

    print("\n----------------- SUMMARY -----------------")
    print(json.dumps(results, indent=2))
    print("-------------------------------------------\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze video with Wavelet YOLO Weapon Detector")
    parser.add_argument("--video", type=str, default="data/joao_detected_objects_video.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="data/analysis_wavelet_sidebyside.mp4", help="Path to output video")
    parser.add_argument("--mode", type=str, choices=["standard", "haar", "db2", "sym2", "fusion"], default="haar")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_render", action="store_true")

    args = parser.parse_args()
    analyze_video(
        video_path=args.video,
        output_path=args.output,
        mode=args.mode,
        max_frames=args.max_frames,
        render_video=not args.no_render,
    )
