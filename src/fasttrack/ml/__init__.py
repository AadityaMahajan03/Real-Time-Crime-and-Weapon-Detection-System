"""FastTrack Machine Learning and Neural Network Inference Package."""

from __future__ import annotations

from src.fasttrack.ml.benchmark import BenchmarkService
from src.fasttrack.ml.calibration import TemperatureScaler
from src.fasttrack.ml.saliency import compute_temporal_saliency, overlay_heatmap_on_frame
from src.fasttrack.ml.tracker import TrackState, WeaponTracker
from src.fasttrack.ml.violence_detector import ViolenceClassifier, ViolencePrediction
from src.fasttrack.ml.weapon_detector import WeaponDetector
from src.fasttrack.ml.wavelet_enhancer import WaveletEnhancer
from src.fasttrack.ml.mobilenet_bilstm import MoBiLSTM, preprocess_frame_mobilstm

__all__ = [
    "WeaponDetector",
    "WaveletEnhancer",
    "MoBiLSTM",
    "preprocess_frame_mobilstm",
    "WeaponTracker",
    "TrackState",
    "ViolenceClassifier",
    "ViolencePrediction",
    "TemperatureScaler",
    "compute_temporal_saliency",
    "overlay_heatmap_on_frame",
    "BenchmarkService",
]
