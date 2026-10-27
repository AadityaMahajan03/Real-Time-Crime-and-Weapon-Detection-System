"""Unit tests for Wavelet Feature Enhancer and Wavelet YOLO Weapon Detection."""

from __future__ import annotations

import numpy as np
import pytest

from src.fasttrack.core.config import WeaponDetectorSettings
from src.fasttrack.core.types import Frame
from src.fasttrack.ml.wavelet_enhancer import WaveletEnhancer, enhance_contrast
from src.fasttrack.ml.weapon_detector import WeaponDetector


def test_wavelet_enhancer_decomposition():
    enhancer = WaveletEnhancer(default_wavelet="haar")
    dummy_bgr = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

    LL, LH, HL, HH = enhancer.decompose(dummy_bgr, wavelet="haar")
    assert LL.shape[:2] == (120, 160)
    assert LH.shape[:2] == (120, 160)
    assert HL.shape[:2] == (120, 160)
    assert HH.shape[:2] == (120, 160)


def test_wavelet_subband_extraction_and_contrast():
    enhancer = WaveletEnhancer(default_wavelet="haar")
    dummy_bgr = np.zeros((200, 200, 3), dtype=np.uint8)
    dummy_bgr[51:150, 51:150] = 255

    lh_norm = enhancer.extract_subband(dummy_bgr, wavelet="haar", subband="LH", target_size=(640, 640))
    assert lh_norm.shape == (640, 640, 3)
    assert lh_norm.dtype == np.uint8
    assert lh_norm.max() > 0


def test_wavelet_cctv_rgb_enhancement():
    enhancer = WaveletEnhancer(default_wavelet="haar")
    dummy_bgr = np.random.randint(20, 80, (100, 100, 3), dtype=np.uint8)

    enhanced = enhancer.enhance_cctv_wavelet_rgb(dummy_bgr, blend_alpha=0.3)
    assert enhanced.shape == dummy_bgr.shape
    assert enhanced.dtype == np.uint8


def test_weapon_detector_wavelet_modes():
    # Test initialization with haar wavelet mode
    settings = WeaponDetectorSettings(
        model_checkpoint="weights/joao_models/normal_compressed_best.pt",
        wavelet_checkpoint="weights/joao_models/haar_compressed_best.pt",
        wavelet_mode="haar",
    )
    detector = WeaponDetector(settings=settings)

    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = Frame(image=dummy, ts_utc=0.0, source_ts=0.0, frame_idx=1, source_id="test")

    dets = detector.detect(frame)
    assert isinstance(dets, list)
