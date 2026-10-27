"""Comprehensive test suite for the weapon detection pipeline and tracking engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from scripts.audit_weapon_dataset import audit_yolo_dataset
from scripts.eval_weapon import evaluate_weapon_model
from src.fasttrack.core.clock import SimulatedClock
from src.fasttrack.core.config import WeaponDetectorSettings
from src.fasttrack.core.errors import ModelLoadError
from src.fasttrack.core.types import BoundingBox, Detection
from src.fasttrack.ml.tracker import WeaponTracker
from src.fasttrack.ml.weapon_detector import WeaponDetector


def test_tracker_k_of_n_confirmation_lifecycle(sim_clock: SimulatedClock) -> None:
    """Test that a weapon track requires K detections in N frames before confirmation."""
    tracker = WeaponTracker(k=3, n=5, track_persistence_sec=2.0, clock=sim_clock)

    box = BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=200.0)
    det = Detection(box=box, class_name="knife", confidence=0.75, source_id="cam-01")

    # Frame 1: 1 of 5 -> unconfirmed
    confirmed, unconfirmed = tracker.update([det])
    assert len(confirmed) == 0
    assert len(unconfirmed) == 1
    tid = unconfirmed[0].track_id
    assert tid is not None

    # Advance time 0.1s
    sim_clock.advance(0.1)

    # Frame 2: 2 of 5 -> unconfirmed
    det2 = Detection(box=box, class_name="knife", confidence=0.76, track_id=tid, source_id="cam-01")
    confirmed, unconfirmed = tracker.update([det2])
    assert len(confirmed) == 0
    assert len(unconfirmed) == 1

    sim_clock.advance(0.1)

    # Frame 3: 3 of 5 (meets K=3) -> CONFIRMED!
    det3 = Detection(box=box, class_name="knife", confidence=0.78, track_id=tid, source_id="cam-01")
    confirmed, unconfirmed = tracker.update([det3])
    assert len(confirmed) == 1
    assert len(unconfirmed) == 0
    assert confirmed[0].track_id == tid

    # Advance beyond track_persistence_sec (2.0s)
    sim_clock.advance(2.5)
    # Empty frame update triggers pruning of stale track
    tracker.update([])
    assert tid not in tracker._tracks


def test_weapon_detector_loud_class_mismatch() -> None:
    """Validate checkpoint classes against configured weapon classes and fail loudly on mismatch."""
    mock_model = MagicMock()
    # Mock model containing unrelated classes (e.g. animals)
    mock_model.names = {0: "dog", 1: "cat", 2: "horse"}

    with patch("ultralytics.YOLO", return_value=mock_model):
        with pytest.raises(ModelLoadError, match="Class map mismatch"):
            WeaponDetector(checkpoint_path="weights/unrelated_model.pt")


def test_weapon_detector_geometric_and_confidence_filtering() -> None:
    """Verify minimum area, aspect ratio, and per-class confidence filters."""
    settings = WeaponDetectorSettings(
        default_confidence_threshold=0.50,
        per_class_thresholds={"knife": 0.45, "pistol": 0.55},
        min_box_area_px=200.0,
        max_aspect_ratio=5.0,
    )

    mock_model = MagicMock()
    mock_model.names = {0: "knife", 1: "pistol"}

    with patch("ultralytics.YOLO", return_value=mock_model):
        detector = WeaponDetector(settings=settings, checkpoint_path="weights/mock_yolo.pt", device="cpu")

        # 1. Valid knife detection
        valid_box = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0)  # Area = 1600, AR = 1.0
        assert detector._filter_detection(valid_box, "knife", 0.50) is True

        # 2. Confidence below class threshold (0.40 < 0.45)
        assert detector._filter_detection(valid_box, "knife", 0.40) is False

        # 3. Tiny box below min_box_area_px (Area = 50 < 200)
        tiny_box = BoundingBox(x1=10.0, y1=10.0, x2=15.0, y2=20.0)
        assert detector._filter_detection(tiny_box, "knife", 0.90) is False

        # 4. Extreme aspect ratio (width=100, height=5 -> AR = 20.0 > 5.0)
        extreme_ar_box = BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=5.0)
        assert detector._filter_detection(extreme_ar_box, "knife", 0.90) is False


def test_weapon_detector_tiled_nms() -> None:
    """Test Non-Maximum Suppression filtering overlapping tile candidates."""
    mock_model = MagicMock()
    mock_model.names = {0: "knife"}

    with patch("ultralytics.YOLO", return_value=mock_model):
        detector = WeaponDetector(checkpoint_path="weights/mock_yolo.pt", device="cpu")

        # Two overlapping boxes for same class
        b1 = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0)
        b2 = BoundingBox(x1=105.0, y1=105.0, x2=205.0, y2=205.0)  # High IoU with b1

        d1 = Detection(box=b1, class_name="knife", confidence=0.85)
        d2 = Detection(box=b2, class_name="knife", confidence=0.70)

        kept = detector._nms_filter([d1, d2], iou_thresh=0.45)
        # Higher confidence d1 should suppress d2
        assert len(kept) == 1
        assert kept[0].confidence == 0.85


def test_audit_weapon_dataset(temp_dir: Path) -> None:
    """Test dataset audit tool parsing YOLO annotations, image hashing, and small/medium/large categorization."""
    dataset_root = temp_dir / "audit_dataset"
    (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_root / "labels" / "train").mkdir(parents=True, exist_ok=True)

    # Create dummy image
    img_path = dataset_root / "images" / "train" / "img_01.jpg"
    cv2.imwrite(str(img_path), np.zeros((480, 640, 3), dtype=np.uint8))

    # Create standard YOLO label: class 0 (knife), center (0.5, 0.5), w=0.2, h=0.3
    lbl_path = dataset_root / "labels" / "train" / "img_01.txt"
    lbl_path.write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")

    report = audit_yolo_dataset(dataset_root)

    assert report["total_images"] == 1
    assert report["total_annotations"] == 1
    assert "0" in report["class_counts"]
    assert report["class_counts"]["0"] == 1
    assert report["box_size_distribution"]["large_coco"] == 1  # 0.2 * 0.3 = 0.06 > 0.0225


def test_eval_weapon_script(temp_dir: Path) -> None:
    """Test weapon evaluation runner generates valid schema in reports/eval/weapon_eval.json."""
    eval_output = temp_dir / "eval_test.json"
    metrics = evaluate_weapon_model(output_path=str(eval_output))

    assert eval_output.exists()
    assert "mAP_50" in metrics
    assert "mAP_50_95" in metrics
    assert "per_class_metrics" in metrics
    assert "knife" in metrics["per_class_metrics"]
    assert metrics["mAP_50"] > 0.5


def test_weapon_crop_verifier_rejection() -> None:
    """Test that WeaponCropVerifier scores candidate crops and returns validity flag."""
    from src.fasttrack.ml.verifier import WeaponCropVerifier

    verifier = WeaponCropVerifier(checkpoint_path="weights/weapon_verifier.pt", device="cpu")
    # Synthetic image crop
    synthetic_crop = np.zeros((128, 128, 3), dtype=np.uint8)

    is_valid, score, probs = verifier.verify_crop(synthetic_crop, candidate_class="knife", detector_conf=0.75)
    assert isinstance(is_valid, bool)
    assert isinstance(score, float)
    assert isinstance(probs, dict)
    assert "not_weapon" in probs


def test_spatial_pose_prior_shoe_defense() -> None:
    """Test SpatialPosePrior classifies shoe/foot boxes vs hand-held weapons."""
    from src.fasttrack.ml.pose_prior import SpatialPosePrior

    pose_prior = SpatialPosePrior(foot_region_ratio=0.82)

    # Person box: top 100, bottom 500, left 100, right 300 (height = 400)
    person_box = BoundingBox(x1=100.0, y1=100.0, x2=300.0, y2=500.0)

    # False positive knife detection on person's shoe:
    # y1=450, y2=495 (within bottom 18% of person height [428 to 500])
    shoe_box = BoundingBox(x1=150.0, y1=450.0, x2=220.0, y2=495.0)

    # Legitimate knife held in hand:
    # y1=280, y2=330 (waist/chest area)
    hand_knife_box = BoundingBox(x1=250.0, y1=280.0, x2=310.0, y2=330.0)

    # Validate shoe box is rejected when inside the foot region
    is_valid_shoe, reason_shoe = pose_prior.evaluate_candidate(
        weapon_box=shoe_box,
        weapon_class="knife",
        person_boxes=[person_box],
    )
    assert is_valid_shoe is False
    assert "shoe" in reason_shoe.lower()

    # Validate hand knife is NOT rejected
    is_valid_hand, reason_hand = pose_prior.evaluate_candidate(
        weapon_box=hand_knife_box,
        weapon_class="knife",
        person_boxes=[person_box],
    )
    assert is_valid_hand is True

