"""Environment and hardware diagnostics reporting script for FastTrack AI.

Collects and records hardware, CUDA, PyTorch, TorchVision, Ultralytics, and
system runtime versions into a verifiable JSON artifact for viva defense.
"""

from __future__ import annotations

import json
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict

import psutil


def collect_environment_report() -> Dict[str, Any]:
    """Inspect and compile comprehensive runtime hardware and library information."""
    report: Dict[str, Any] = {
        "timestamp_utc": None,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "ml_frameworks": {},
        "sqlite": {
            "version": sqlite3.sqlite_version,
            "threadsafety": sqlite3.threadsafety,
        },
    }

    from datetime import datetime, timezone

    report["timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    # PyTorch and CUDA diagnostics
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_info = []
        if cuda_available:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append(
                    {
                        "index": i,
                        "name": props.name,
                        "total_vram_gb": round(props.total_memory / (1024**3), 2),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )

        report["ml_frameworks"]["pytorch"] = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
            "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
            "gpus": gpu_info,
        }
    except ImportError:
        report["ml_frameworks"]["pytorch"] = "Not Installed"

    # Torchvision
    try:
        import torchvision

        report["ml_frameworks"]["torchvision"] = {"version": torchvision.__version__}
    except ImportError:
        report["ml_frameworks"]["torchvision"] = "Not Installed"

    # Ultralytics
    try:
        import ultralytics

        report["ml_frameworks"]["ultralytics"] = {"version": ultralytics.__version__}
    except ImportError:
        report["ml_frameworks"]["ultralytics"] = "Not Installed"

    # OpenCV
    try:
        import cv2

        report["ml_frameworks"]["opencv"] = {
            "version": cv2.__version__,
            "cuda_enabled": cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, "cuda") else False,
        }
    except ImportError:
        report["ml_frameworks"]["opencv"] = "Not Installed"

    # Streamlit
    try:
        import streamlit

        report["ml_frameworks"]["streamlit"] = {"version": streamlit.__version__}
    except ImportError:
        report["ml_frameworks"]["streamlit"] = "Not Installed"

    return report


def main() -> None:
    print("=" * 60)
    print("FAST-TRACK AI - Environment & Hardware Diagnostics Report")
    print("=" * 60)

    report = collect_environment_report()

    # Pretty print summary to console
    plat = report["platform"]
    print(f"OS: {plat['system']} {plat['release']} ({plat['machine']})")
    print(f"CPU: {plat['processor']} ({plat['cpu_logical_cores']} threads)")
    print(f"RAM: {plat['total_ram_gb']} GB total, {plat['available_ram_gb']} GB available")
    print(f"Python: {report['python']['version']} ({report['python']['executable']})")
    print(f"SQLite: v{report['sqlite']['version']}")

    ml = report["ml_frameworks"]
    print("-" * 60)
    if isinstance(ml.get("pytorch"), dict):
        torch_info = ml["pytorch"]
        print(f"PyTorch: {torch_info['version']} | CUDA: {torch_info['cuda_version']}")
        print(f"CUDA Available: {torch_info['cuda_available']}")
        for gpu in torch_info.get("gpus", []):
            print(f"  -> GPU [{gpu['index']}]: {gpu['name']} ({gpu['total_vram_gb']} GB VRAM)")
    else:
        print(f"PyTorch: {ml.get('pytorch')}")

    if isinstance(ml.get("ultralytics"), dict):
        print(f"Ultralytics: {ml['ultralytics']['version']}")
    if isinstance(ml.get("torchvision"), dict):
        print(f"TorchVision: {ml['torchvision']['version']}")
    if isinstance(ml.get("opencv"), dict):
        print(f"OpenCV: {ml['opencv']['version']}")
    if isinstance(ml.get("streamlit"), dict):
        print(f"Streamlit: {ml['streamlit']['version']}")
    print("-" * 60)

    # Save artifact
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "env_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report artifact written to: {out_file.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
