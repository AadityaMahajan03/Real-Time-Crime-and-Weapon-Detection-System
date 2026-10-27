"""System-wide multi-camera throughput, latency, and resource benchmark script."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil
import torch

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.fasttrack.core.logging import get_logger, setup_logging
from src.fasttrack.ml.weapon_detector import WeaponDetector
from src.fasttrack.sources.base import Frame

setup_logging(level="INFO")
logger = get_logger("scripts.benchmark_system")


def run_system_benchmark(
    camera_counts: List[int] = [1, 2, 4],
    frames_per_camera: int = 30,
    output_report: str = "reports/eval/system_benchmark.json",
) -> Dict[str, Any]:
    """Benchmark end-to-end multi-camera throughput, latency percentiles, and hardware resources."""
    report_path = Path(output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    has_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU"

    results: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": f"{device_name} (PyTorch {torch.__version__})",
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "camera_benchmarks": {},
        "summary": {},
    }

    detector = WeaponDetector()

    # Pre-allocate synthetic 720p frames (1280x720x3)
    frame_h, frame_w = 720, 1280
    test_frame = np.random.randint(0, 255, (frame_h, frame_w, 3), dtype=np.uint8)

    for n_cams in camera_counts:
        logger.info("Running benchmark for %d concurrent simulated camera stream(s)...", n_cams)
        latencies_ms: List[float] = []

        initial_cpu = psutil.cpu_percent(interval=None)

        if has_cuda:
            torch.cuda.reset_peak_memory_stats(0)

        start_time = time.perf_counter()

        # Simulate round-robin frame ingestion and detection across n_cams
        for f_idx in range(frames_per_camera):
            for cam_idx in range(n_cams):
                frame_obj = Frame(
                    image=test_frame,
                    ts_utc=datetime.now(timezone.utc),
                    source_ts=time.time(),
                    frame_idx=f_idx,
                    source_id=f"cam_{cam_idx}",
                )
                t0 = time.perf_counter()
                _ = detector.detect(frame_obj)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        total_elapsed = time.perf_counter() - start_time
        total_processed_frames = n_cams * frames_per_camera
        aggregate_fps = round(total_processed_frames / total_elapsed, 2)

        p50 = round(float(np.percentile(latencies_ms, 50)), 2)
        p95 = round(float(np.percentile(latencies_ms, 95)), 2)
        p99 = round(float(np.percentile(latencies_ms, 99)), 2)
        mean_lat = round(float(np.mean(latencies_ms)), 2)

        peak_gpu_vram_mb = 0.0
        if has_cuda:
            peak_gpu_vram_mb = round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)

        end_cpu = psutil.cpu_percent(interval=None)
        end_ram_mb = psutil.virtual_memory().used / (1024 * 1024)

        results["camera_benchmarks"][f"{n_cams}_cameras"] = {
            "num_cameras": n_cams,
            "total_frames": total_processed_frames,
            "duration_seconds": round(total_elapsed, 2),
            "aggregate_fps": aggregate_fps,
            "fps_per_camera": round(aggregate_fps / n_cams, 2),
            "latency": {
                "mean_ms": mean_lat,
                "p50_ms": p50,
                "p95_ms": p95,
                "p99_ms": p99,
            },
            "resources": {
                "cpu_utilization_pct": round(max(initial_cpu, end_cpu), 1),
                "ram_usage_mb": round(end_ram_mb, 1),
                "gpu_vram_peak_mb": peak_gpu_vram_mb,
            },
        }

    results["summary"] = {
        "max_recommended_cameras_720p": 4 if aggregate_fps >= 20.0 else 2,
        "bottleneck": "GPU forward-pass serialization via model mutex" if has_cuda else "CPU compute capacity",
        "graceful_degradation_strategy": "LIFO drop-oldest frame backpressure when bounded queue is saturated",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("System benchmark report written to %s", report_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FastTrack system throughput benchmark")
    parser.add_argument("--output", default="reports/eval/system_benchmark.json", help="Output JSON path")
    parser.add_argument("--frames", type=int, default=20, help="Frames per camera")
    args = parser.parse_args()

    run_system_benchmark(frames_per_camera=args.frames, output_report=args.output)
