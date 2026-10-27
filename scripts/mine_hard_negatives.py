"""Hard-negative mining script collecting false alarms on CCTV clutter (shoes, phones, pens)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.fasttrack.core.logging import get_logger, setup_logging
from src.fasttrack.core.types import Frame
from src.fasttrack.ml.weapon_detector import WeaponDetector

setup_logging(level="INFO")
logger = get_logger("scripts.mine_negatives")


def mine_hard_negatives(
    input_dir: str = "data",
    output_dir: str = "data/hard_negatives",
    conf_thresh: float = 0.20,
) -> int:
    """Scan negative footage/images, capture false positives, and format as YOLO background images."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_images = out_path / "images"
    out_labels = out_path / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    detector = WeaponDetector()
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.mp4", "*.avi")
    files = []
    for ext in extensions:
        files.extend(list(in_path.glob(ext)))

    logger.info("Starting hard-negative mining across %d candidate files in %s", len(files), in_path)

    mined_count = 0

    for f in files:
        if f.suffix.lower() in (".mp4", ".avi", ".mov"):
            cap = cv2.VideoCapture(str(f))
            frame_idx = 0
            while cap.isOpened() and frame_idx < 300:
                ret, img = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % 10 != 0:
                    continue

                fr = Frame(image=img, source_id="negative_stream")
                dets = detector.detect(fr)
                if dets:
                    # False positive found on negative stream!
                    out_img_name = f"mined_fp_{f.stem}_f{frame_idx}.jpg"
                    cv2.imwrite(str(out_images / out_img_name), img)
                    # Create empty label file indicating pure background image for YOLO
                    (out_labels / f"mined_fp_{f.stem}_f{frame_idx}.txt").write_text("")
                    mined_count += 1
            cap.release()

        else:
            img = cv2.imread(str(f))
            if img is None:
                continue
            fr = Frame(image=img, source_id="negative_img")
            dets = detector.detect(fr)
            if dets:
                out_img_name = f"mined_fp_{f.stem}.jpg"
                cv2.imwrite(str(out_images / out_img_name), img)
                (out_labels / f"mined_fp_{f.stem}.txt").write_text("")
                mined_count += 1

    logger.info("Hard-negative mining completed! Mined %d false positive background samples into %s", mined_count, out_path)
    return mined_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine hard negatives for weapon detector retraining")
    parser.add_argument("--input-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="data/hard_negatives")
    args = parser.parse_args()
    mine_hard_negatives(args.input_dir, args.output_dir)
