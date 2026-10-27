"""Spatiotemporal violence action classifier using rolling time buffers and TorchVision 3D CNNs."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models.video as video_models

from src.fasttrack.core.config import ViolenceClassifierSettings
from src.fasttrack.core.errors import ModelLoadError
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import BoundingBox, Frame
from src.fasttrack.ml.calibration import TemperatureScaler
from src.fasttrack.ml.kinetic import KineticFeatureExtractor, KineticMetrics

logger = get_logger("ml.violence")


@dataclass
class ViolencePrediction:
    """Output prediction metrics from spatiotemporal video buffer classification."""

    raw_score: float
    calibrated_score: float
    ema_score: float
    window_duration_sec: float
    is_buffer_ready: bool
    status: str  # 'ready', 'insufficient_data', 'degraded'
    latency_ms: float = 0.0
    fused_score: float = 0.0
    kinetic_metrics: Optional[KineticMetrics] = None


class ViolenceClassifier:
    """Spatiotemporal video classifier for detecting possible violent altercations.

    Features:
    - Uniform time-based sampling (in seconds, not frame indices) across rolling buffer
    - Dual architecture support: TorchVision S3D (baseline) and R3D-18 (comparison)
    - Full tensor format adherence: (B, C, T, H, W) layout with ImageNet/Kinetics normalization
    - 3 output values: raw probability, temperature-calibrated probability, and EMA smoothing
    - Explicit 'insufficient data' warm-up state exposure
    """

    # ImageNet / Kinetics spatiotemporal normalization constants
    NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(
        self,
        settings: Optional[ViolenceClassifierSettings] = None,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.settings = settings or ViolenceClassifierSettings()
        self.weights_path = weights_path or self.settings.weights_path
        self.arch = self.settings.model_architecture.lower()

        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Rolling timestamped frame buffer: stores (timestamp_sec, BGR image)
        self._buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=200)
        self._last_infer_ts: float = 0.0
        self._prev_ema_score: float = 0.0
        self._last_latency_ms: float = 0.0
        self._is_ready: bool = False
        self._recent_person_boxes: Optional[List[BoundingBox]] = None
        self.kinetic_extractor = KineticFeatureExtractor()

        self.scaler = TemperatureScaler(temperature=self.settings.temperature)
        self.model: nn.Module = self._initialize_model()

    def _initialize_model(self) -> nn.Module:
        """Instantiate MoBiLSTM (Khalid 2022) or TorchVision S3D / R3D-18 model."""
        logger.info("Initializing %s spatiotemporal violence model on %s...", self.arch.upper(), self.device)
        try:
            if self.arch in ("mobilenet_bilstm", "mobilenet_v2_bilstm", "mobilstm"):
                from src.fasttrack.ml.mobilenet_bilstm import MoBiLSTM
                net = MoBiLSTM(num_classes=2, pretrained_backbone=True)
            elif self.arch == "r3d_18":
                net = video_models.r3d_18(weights=None, num_classes=1)
            else:
                # Default S3D Kinetics baseline
                net = video_models.s3d(weights=None, num_classes=1)

            # Load checkpoint if available on disk
            w_path = Path(self.weights_path)
            if w_path.is_file():
                try:
                    state_dict = torch.load(str(w_path), map_location="cpu")
                    net.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded custom violence checkpoint from %s", w_path)
                except Exception as e:
                    logger.warning("Could not load weights from %s: %s. Using initialized model.", w_path, e)
            else:
                logger.info("Operating with initialized %s spatiotemporal model.", self.arch.upper())

            net.eval().to(self.device)
            return net

        except Exception as e:
            raise ModelLoadError(f"Failed to initialize violence model {self.arch}: {e}")

    def reset_buffer(self) -> None:
        """Flush the rolling buffer and reset smoothing state."""
        self._buffer.clear()
        self._last_infer_ts = 0.0
        self._prev_ema_score = 0.0
        self._is_ready = False

    @property
    def is_buffer_ready(self) -> bool:
        """Return True if rolling buffer spans at least clip_seconds."""
        if len(self._buffer) < 2:
            return False
        span = self._buffer[-1][0] - self._buffer[0][0]
        return span >= self.settings.clip_seconds

    def push_frame(
        self,
        frame: Any,
        image: Optional[np.ndarray] = None,
        person_boxes: Optional[List[BoundingBox]] = None,
    ) -> Optional[ViolencePrediction]:
        """Ingest new video frame into the time-based rolling buffer.

        Accepts either a Frame dataclass, or (timestamp_sec, image_bgr).
        Executes inference only when buffer duration reaches clip_seconds and
        stride_seconds has elapsed since the last inference pass.
        """
        if person_boxes is not None:
            self._recent_person_boxes = person_boxes

        if isinstance(frame, Frame):
            now_ts = frame.source_ts
            img = frame.image
        elif image is not None:
            now_ts = float(frame)
            img = image
        else:
            now_ts = time.time()
            img = frame

        self._buffer.append((now_ts, img))

        # Prune frames older than clip window + 1.0s to bound memory
        min_allowed_ts = now_ts - self.settings.clip_seconds - 1.0
        while self._buffer and self._buffer[0][0] < min_allowed_ts:
            self._buffer.popleft()

        # Check if buffer has accumulated sufficient temporal duration
        if not self.is_buffer_ready:
            self._is_ready = False
            return ViolencePrediction(
                raw_score=0.0,
                calibrated_score=0.0,
                ema_score=self._prev_ema_score,
                window_duration_sec=self._buffer[-1][0] - self._buffer[0][0],
                is_buffer_ready=False,
                status="insufficient_data",
            )

        self._is_ready = True

        # Check stride pacing: only infer when stride interval has elapsed
        if (now_ts - self._last_infer_ts) < self.settings.stride_seconds:
            return None

        # Execute spatiotemporal inference pass
        return self._infer_window(now_ts, person_boxes=self._recent_person_boxes)

    def _sample_frames_by_time(self, end_ts: float) -> List[np.ndarray]:
        """Sample T frames uniformly spaced over the time window [end_ts - clip_seconds, end_ts]."""
        start_ts = end_ts - self.settings.clip_seconds
        sample_times = np.linspace(start_ts, end_ts, self.settings.clip_frames)

        # Buffer timestamps
        timestamps = np.array([item[0] for item in self._buffer])
        sampled_images: List[np.ndarray] = []

        for target_t in sample_times:
            # Find closest buffered frame to the target timestamp
            idx = int(np.argmin(np.abs(timestamps - target_t)))
            sampled_images.append(self._buffer[idx][1])

        return sampled_images

    def _infer_window(
        self,
        end_ts: float,
        person_boxes: Optional[List[BoundingBox]] = None,
    ) -> ViolencePrediction:
        """Sample temporal frames, preprocess tensor to (1, C, T, H, W), and predict score."""
        t_start = time.perf_counter()
        sampled = self._sample_frames_by_time(end_ts)

        is_mobilstm = self.arch in ("mobilenet_bilstm", "mobilenet_v2_bilstm", "mobilstm")
        if is_mobilstm:
            from src.fasttrack.ml.mobilenet_bilstm import preprocess_frame_mobilstm
            processed_frames = [preprocess_frame_mobilstm(img, target_size=(64, 64)) for img in sampled]
            stacked = np.stack(processed_frames, axis=0)  # (16, 64, 64, 3)
            tensor_np = np.transpose(stacked, (0, 3, 1, 2))  # (16, 3, 64, 64)
            input_tensor = torch.tensor(tensor_np, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 16, 3, 64, 64)
        else:
            # Preprocess frames: BGR -> RGB, resize to 224x224, normalize
            processed_frames = []
            for img in sampled:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                norm = (resized - self.NORM_MEAN) / self.NORM_STD
                processed_frames.append(norm)

            # Stack into (Time, Height, Width, Channels) -> transpose to (Channels, Time, Height, Width)
            stacked = np.stack(processed_frames, axis=0)  # (16, 224, 224, 3)
            tensor_np = np.transpose(stacked, (3, 0, 1, 2))  # (3, 16, 224, 224)
            input_tensor = torch.tensor(tensor_np, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, 3, 16, 224, 224)

        try:
            with torch.no_grad():
                out = self.model(input_tensor)
                if is_mobilstm:
                    logits = out.squeeze()
                    probs = torch.softmax(logits, dim=-1)
                    raw_prob = float(probs[1].item())  # Class 1 is Violence
                    raw_logit = float((logits[1] - logits[0]).item())
                else:
                    raw_logit = out.squeeze().item()
                    raw_prob = float(1.0 / (1.0 + np.exp(-np.clip(raw_logit, -30.0, 30.0))))
        except torch.cuda.OutOfMemoryError as e:
            logger.warning("CUDA OOM in violence classifier. Clearing cache and retrying on CPU: %s", e)
            torch.cuda.empty_cache()
            self.device = "cpu"
            self.model.to("cpu")
            with torch.no_grad():
                out = self.model(input_tensor.to("cpu"))
                if is_mobilstm:
                    logits = out.squeeze()
                    probs = torch.softmax(logits, dim=-1)
                    raw_prob = float(probs[1].item())
                    raw_logit = float((logits[1] - logits[0]).item())
                else:
                    raw_logit = out.squeeze().item()
                    raw_prob = float(1.0 / (1.0 + np.exp(-np.clip(raw_logit, -30.0, 30.0))))

        # 1. Spatiotemporal Model Scores
        raw_prob = float(1.0 / (1.0 + np.exp(-np.clip(raw_logit, -30.0, 30.0))))
        calibrated_video_score = float(self.scaler.calibrate_logit(raw_logit))

        # 2. Extract Kinetic Motion & Multi-Person Proximity Features
        kinetic = self.kinetic_extractor.extract(sampled, person_boxes=person_boxes)

        # 3. Multimodal Fusion: Video CNN (40%) + Kinetic Motion (40%) + Proximity Prior (20%)
        prox_prior = 1.0 if kinetic.in_proximity else (0.05 if (person_boxes and len(person_boxes) <= 1) else 0.30)
        fused = (0.40 * calibrated_video_score) + (0.40 * kinetic.gated_kinetic_score) + (0.20 * prox_prior)

        # Multi-Person Proximity Gating:
        # Solitary individuals (running, jumping, dancing, falling) CANNOT exceed 0.20
        if person_boxes is not None:
            if len(person_boxes) <= 1:
                fused = min(fused, 0.20)
            elif not kinetic.in_proximity:
                fused = min(fused, 0.35)

        fused_score = float(np.clip(fused, 0.0, 1.0))
        calibrated_prob = fused_score

        # 4. EMA Smoothing across successive windows: S_t = alpha * score + (1 - alpha) * S_{t-1}
        alpha = self.settings.ema_smoothing_alpha
        ema_prob = (alpha * calibrated_prob) + ((1.0 - alpha) * self._prev_ema_score)
        self._prev_ema_score = ema_prob

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self._last_latency_ms = latency_ms
        self._last_infer_ts = end_ts

        return ViolencePrediction(
            raw_score=round(raw_prob, 4),
            calibrated_score=round(calibrated_prob, 4),
            ema_score=round(ema_prob, 4),
            window_duration_sec=self.settings.clip_seconds,
            is_buffer_ready=True,
            status="ready",
            latency_ms=round(latency_ms, 2),
            fused_score=round(fused_score, 4),
            kinetic_metrics=kinetic,
        )
