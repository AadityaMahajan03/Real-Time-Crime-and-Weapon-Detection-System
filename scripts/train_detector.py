"""CCTV-Realistic YOLO11n Weapon Detector training script with hard negatives and augmentations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.fasttrack.core.logging import get_logger, setup_logging

setup_logging(level="INFO")
logger = get_logger("scripts.train_detector")


def train_weapon_detector(
    data_yaml: str = "config/weapons_dataset.yaml",
    epochs: int = 30,
    imgsz: int = 960,
    batch_size: int = 8,
    base_checkpoint: str = "weights/weapon_yolo.pt",
    output_dir: str = "weights/trained_detector",
) -> None:
    """Train YOLO11n detector with CCTV augmentations, multi-scale training, and hard negative ratios."""
    from ultralytics import YOLO

    logger.info("Initializing detector training from base: %s (imgsz=%d)", base_checkpoint, imgsz)

    model = YOLO(base_checkpoint if Path(base_checkpoint).exists() else "yolo11n.pt")

    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "project": output_dir,
        "name": f"weapon_yolo11n_imgsz{imgsz}",
        "exist_ok": True,
        "seed": 42,
        "deterministic": True,
        # CCTV-Realistic Data Augmentations
        "mosaic": 1.0,
        "mixup": 0.20,
        "copy_paste": 0.15,
        "hsv_h": 0.015,
        "hsv_s": 0.70,
        "hsv_v": 0.40,
        "degrees": 15.0,
        "translate": 0.10,
        "scale": 0.50,
        "fliplr": 0.50,
        "perspective": 0.0005,
    }

    logger.info("Training configuration: %s", train_args)
    if Path(data_yaml).exists():
        model.train(**train_args)
        logger.info("Training completed! Artifacts saved in %s", output_dir)
    else:
        logger.warning(
            "Dataset YAML '%s' not found on disk. Download/mount weapon dataset before running full training.",
            data_yaml,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO11n weapon detector with CCTV augmentations")
    parser.add_argument("--data", type=str, default="config/weapons_dataset.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    train_weapon_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
    )
