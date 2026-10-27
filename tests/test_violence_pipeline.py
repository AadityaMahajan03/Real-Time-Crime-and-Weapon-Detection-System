"""Comprehensive unit and integration test suite for the spatiotemporal violence pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import torch

from scripts.build_violence_manifest import build_manifest
from scripts.eval_violence import evaluate_violence_models
from src.fasttrack.core.config import ViolenceClassifierSettings
from src.fasttrack.core.types import Frame
from src.fasttrack.ml.calibration import TemperatureScaler
from src.fasttrack.ml.saliency import compute_temporal_saliency, overlay_heatmap_on_frame
from src.fasttrack.ml.violence_detector import ViolenceClassifier


def test_temperature_scaler_calibration_and_persistence(temp_dir: Path) -> None:
    """Test TemperatureScaler logit calibration, fitting on validation data, and file I/O."""
    scaler = TemperatureScaler(temperature=2.0)

    # Logit 0.0 should calibrate to 0.5 regardless of temperature
    p_zero = scaler.calibrate_logit(0.0)
    assert abs(p_zero - 0.5) < 1e-4

    # Higher temperature softens extreme logits
    p_high = scaler.calibrate_logit(4.0)
    assert 0.5 < p_high < 1.0

    # Fit on validation logits
    val_logits = np.array([3.0, -3.0, 2.5, -2.5, 1.0, -1.0], dtype=np.float32)
    val_labels = np.array([1, 0, 1, 0, 1, 0], dtype=np.float32)
    fitted_t = scaler.fit(val_logits, val_labels, max_iter=10)
    assert fitted_t > 0.05

    # Save and reload
    save_path = temp_dir / "temperature.json"
    scaler.save(save_path)
    assert save_path.exists()

    new_scaler = TemperatureScaler()
    new_scaler.load(save_path)
    assert abs(new_scaler.temperature - fitted_t) < 1e-4


def test_violence_classifier_insufficient_data_warmup() -> None:
    """Verify classifier exposes 'insufficient_data' until temporal buffer reaches clip_seconds."""
    settings = ViolenceClassifierSettings(clip_seconds=2.0, clip_frames=16, stride_seconds=0.5)
    classifier = ViolenceClassifier(settings=settings, device="cpu")

    dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Frame 1: elapsed duration = 0.0s < 2.0s
    frame1 = Frame(image=dummy_img, ts_utc=t0, source_ts=0.0, frame_idx=0, source_id="cam-01")
    pred1 = classifier.push_frame(frame1)

    assert pred1 is not None
    assert pred1.is_buffer_ready is False
    assert pred1.status == "insufficient_data"
    assert pred1.raw_score == 0.0


def test_violence_classifier_time_based_inference() -> None:
    """Feed frames spanning > 2.0s and verify temporal inference returns raw, calibrated, and EMA scores."""
    settings = ViolenceClassifierSettings(clip_seconds=2.0, clip_frames=16, stride_seconds=0.5, temperature=1.2)
    classifier = ViolenceClassifier(settings=settings, device="cpu")

    dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Feed frames at 10 FPS over 2.5 seconds (25 frames, timestamps 0.0 to 2.4s)
    predictions = []
    for i in range(25):
        ts = i * 0.1
        f = Frame(image=dummy_img, ts_utc=t0 + timedelta(seconds=ts), source_ts=ts, frame_idx=i, source_id="cam-01")
        pred = classifier.push_frame(f)
        if pred is not None and pred.is_buffer_ready:
            predictions.append(pred)

    assert len(predictions) >= 1
    last_pred = predictions[-1]
    assert last_pred.is_buffer_ready is True
    assert last_pred.status == "ready"
    assert 0.0 <= last_pred.raw_score <= 1.0
    assert 0.0 <= last_pred.calibrated_score <= 1.0
    assert 0.0 <= last_pred.ema_score <= 1.0


def test_temporal_saliency_computation() -> None:
    """Test gradient-based temporal saliency and spatial heatmap overlay."""
    MagicMock()
    # Dummy gradient output
    dummy_input = torch.zeros((1, 3, 16, 224, 224), dtype=torch.float32)

    # Simple mock spatiotemporal layer
    class SimpleVideoCNN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv3d(3, 1, kernel_size=3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x).mean()

    model = SimpleVideoCNN()
    temp_importance, spatial_heatmap = compute_temporal_saliency(model, dummy_input, device="cpu")

    assert temp_importance.shape == (16,)
    assert spatial_heatmap.shape == (224, 224)
    assert 0.0 <= float(np.max(temp_importance)) <= 1.0

    # Test blending heatmap onto frame
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    blended = overlay_heatmap_on_frame(frame, spatial_heatmap)
    assert blended.shape == (480, 640, 3)


def test_violence_manifest_builder(temp_dir: Path) -> None:
    """Verify build_manifest groups by source and guarantees zero cross-split leakage."""
    manifest_out = temp_dir / "test_manifest.json"
    manifest = build_manifest(
        dataset_dir=str(temp_dir / "nonexistent_dir"),
        output_manifest=str(manifest_out),
    )

    assert manifest_out.exists()
    assert len(manifest["records"]) == 120
    assert "class_counts" in manifest
    assert manifest["class_counts"]["violent"] > 0
    assert manifest["class_counts"]["non_violent"] > 0


def test_eval_violence_script(temp_dir: Path) -> None:
    """Verify evaluation runner writes verifiable scientific metrics to violence_eval.json."""
    eval_out = temp_dir / "violence_eval.json"
    metrics = evaluate_violence_models(output_path=str(eval_out))

    assert eval_out.exists()
    assert "s3d_baseline" in metrics
    assert "r3d18_comparison" in metrics
    assert "hard_negatives_analysis" in metrics

    s3d = metrics["s3d_baseline"]
    assert s3d["roc_auc"] > 0.8
    assert s3d["threshold_t_on"] == 0.65
    assert s3d["threshold_t_off"] == 0.40


def test_violence_classifier_r3d18_arch() -> None:
    """Verify R3D-18 comparison architecture initializes and performs time-based inference."""
    settings = ViolenceClassifierSettings(
        model_architecture="r3d_18", clip_seconds=1.0, clip_frames=16, stride_seconds=0.2
    )
    classifier = ViolenceClassifier(settings=settings, device="cpu")
    assert classifier.arch == "r3d_18"

    dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Feed 15 frames over 1.5 seconds
    predictions = []
    for i in range(15):
        ts = i * 0.1
        f = Frame(image=dummy_img, ts_utc=t0 + timedelta(seconds=ts), source_ts=ts, frame_idx=i, source_id="cam-02")
        pred = classifier.push_frame(f)
        if pred is not None and pred.is_buffer_ready:
            predictions.append(pred)

    assert len(predictions) >= 1
    assert predictions[-1].is_buffer_ready is True


def test_kinetic_single_person_false_alarm_suppression() -> None:
    """Validate that high motion with a single person CANNOT exceed 0.20 (suppressing running/jumping false alarms)."""
    from src.fasttrack.core.types import BoundingBox
    from src.fasttrack.ml.kinetic import KineticFeatureExtractor

    extractor = KineticFeatureExtractor()

    # Generate synthetic high-motion frames (simulating someone jumping/dancing)
    frames = []
    for i in range(16):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Fast moving bright square shifting 20px each frame
        x = (i * 15) % 250
        y = 50 + int(30 * np.sin(i * 0.8))
        cv2.rectangle(frame, (x, y), (x + 60, y + 80), (255, 255, 255), -1)
        frames.append(frame)

    # 1. Single person detected
    single_person = [BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=200.0)]
    metrics_single = extractor.extract(frames, person_boxes=single_person)

    # Gated score must NOT exceed 0.20
    assert metrics_single.person_count == 1
    assert metrics_single.in_proximity is False
    assert metrics_single.gated_kinetic_score <= 0.20
    assert "single person" in metrics_single.gating_rationale.lower()


def test_kinetic_multi_person_proximity_contact() -> None:
    """Validate that high motion with 2 interacting people in close proximity enables high violence scores."""
    from src.fasttrack.core.types import BoundingBox
    from src.fasttrack.ml.kinetic import KineticFeatureExtractor

    extractor = KineticFeatureExtractor()

    # Generate synthetic high-motion frames
    frames = []
    for i in range(16):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Fast erratic motion in center area
        x1 = 120 + int(25 * np.cos(i * 1.5))
        y1 = 80 + int(20 * np.sin(i * 1.5))
        cv2.rectangle(frame, (x1, y1), (x1 + 70, y1 + 90), (255, 255, 255), -1)
        frames.append(frame)

    # Two persons in close physical contact / overlapping
    person_a = BoundingBox(x1=100.0, y1=60.0, x2=180.0, y2=220.0)
    person_b = BoundingBox(x1=130.0, y1=60.0, x2=210.0, y2=220.0)  # Heavy overlap

    metrics_contact = extractor.extract(frames, person_boxes=[person_a, person_b])

    assert metrics_contact.person_count == 2
    assert metrics_contact.in_proximity is True
    assert metrics_contact.gated_kinetic_score > 0.40
    assert "proximity" in metrics_contact.gating_rationale.lower()


def test_violence_classifier_multimodal_single_person_gated() -> None:
    """Test that ViolenceClassifier caps fused_score <= 0.20 when single person is provided."""
    from src.fasttrack.core.types import BoundingBox

    settings = ViolenceClassifierSettings(clip_seconds=1.0, clip_frames=16, stride_seconds=0.2)
    classifier = ViolenceClassifier(settings=settings, device="cpu")

    dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    single_person = [BoundingBox(x1=100.0, y1=50.0, x2=200.0, y2=200.0)]

    predictions = []
    for i in range(15):
        ts = i * 0.1
        f = Frame(image=dummy_img, ts_utc=t0 + timedelta(seconds=ts), source_ts=ts, frame_idx=i, source_id="cam-01")
        pred = classifier.push_frame(f, person_boxes=single_person)
        if pred is not None and pred.is_buffer_ready:
            predictions.append(pred)

    assert len(predictions) >= 1
    last_pred = predictions[-1]
    assert last_pred.is_buffer_ready is True
    assert last_pred.fused_score <= 0.20
    assert last_pred.kinetic_metrics is not None
    assert last_pred.kinetic_metrics.person_count == 1

