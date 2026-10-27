"""Unit tests for MobileNetV2 + BiLSTM (MoBiLSTM) Violence Classifier."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.fasttrack.core.config import ViolenceClassifierSettings
from src.fasttrack.core.types import Frame
from src.fasttrack.ml.mobilenet_bilstm import MoBiLSTM, preprocess_frame_mobilstm
from src.fasttrack.ml.violence_detector import ViolenceClassifier


def test_mobilstm_architecture():
    model = MoBiLSTM(num_classes=2, pretrained_backbone=False)
    # Input sequence of shape (B=2, T=16, C=3, H=64, W=64)
    x = torch.rand(2, 16, 3, 64, 64)
    logits = model(x)
    assert logits.shape == (2, 2)

    probs = torch.softmax(logits, dim=-1)
    assert probs.shape == (2, 2)
    # Check probabilities sum to 1
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_preprocess_frame_mobilstm():
    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    proc = preprocess_frame_mobilstm(dummy, target_size=(64, 64))
    assert proc.shape == (64, 64, 3)
    assert proc.dtype == np.float32
    assert 0.0 <= proc.min() and proc.max() <= 1.0


def test_violence_classifier_with_mobilstm():
    settings = ViolenceClassifierSettings(
        model_architecture="mobilenet_bilstm",
        clip_seconds=1.0,
        clip_frames=16,
        stride_seconds=0.2,
    )
    clf = ViolenceClassifier(settings=settings)
    assert clf.arch == "mobilenet_bilstm"

    # Push frames into rolling buffer
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    pred = None
    for i in range(25):
        t = i * 0.05
        f = Frame(image=dummy_frame, ts_utc=t, source_ts=t, frame_idx=i, source_id="cam_test")
        res = clf.push_frame(f)
        if res is not None and res.is_buffer_ready:
            pred = res

    assert pred is not None
    assert pred.is_buffer_ready is True
    assert 0.0 <= pred.raw_score <= 1.0
    assert 0.0 <= pred.calibrated_score <= 1.0
