"""MobileNetV2 + Bidirectional LSTM (MoBiLSTM) Violence Detection Architecture.

Based on the research and methodology from:
'Real Time Violence Detection using MobileNet and Bi-directional LSTM'
(Abduulrahman Khalid, 2022).
Architecture:
- MobileNetV2 pretrained on ImageNet as spatial feature extractor per frame.
- Bidirectional LSTM (32 forward units, 32 backward units = 64 hidden units)
  to learn spatiotemporal dynamics and sequential combat/striking/assault context.
- Multi-layer Dense MLP classification head:
  Dense(256) -> ReLU -> Dropout(0.25) ->
  Dense(128) -> ReLU -> Dropout(0.25) ->
  Dense(64)  -> ReLU -> Dropout(0.25) ->
  Dense(32)  -> ReLU -> Dropout(0.25) ->
  Dense(2)   -> Softmax [P(NonViolence), P(Violence)]
- Target accuracy: ~97% on Real Life Violence Situations Dataset.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional, Tuple, Union
from pathlib import Path

from src.fasttrack.core.logging import get_logger

logger = get_logger("ml.violence.mobilstm")


class MoBiLSTM(nn.Module):
    """PyTorch implementation of the exact MobileNetV2 + BiLSTM Violence Classifier."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        dropout_rate: float = 0.25,
        lstm_units: int = 32,
    ) -> None:
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained_backbone else None
        mobilenet = models.mobilenet_v2(weights=weights)

        # Headless spatial feature extractor
        self.features = mobilenet.features

        # For 64x64 inputs, MobileNetV2 features yield 1280 channels x 2 x 2 = 5120 flattened features per frame
        self.feature_dim = 1280 * 2 * 2  # 5120

        # Bidirectional LSTM to capture past and future context across sequence
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_units,
            batch_first=True,
            bidirectional=True,
        )

        # Classification Head
        hidden_dim = lstm_units * 2  # 64 (32 forward + 32 backward)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, C, H, W) or (B, C, T, H, W) with pixel values in [0, 1].

        Returns:
            Logits of shape (B, 2) where class 0 is NonViolence and class 1 is Violence.
        """
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input tensor, got shape {x.shape}")

        # Standardize layout to (B, T, C, H, W)
        if x.shape[1] == 3 and x.shape[2] != 3:
            # (B, C, T, H, W) -> (B, T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4)

        b, t, c, h, w = x.shape
        # Flatten temporal and batch dimensions for MobileNetV2
        x_flat = x.reshape(b * t, c, h, w)
        spatial_feats = self.features(x_flat)  # (B*T, 1280, 2, 2)

        # Flatten spatial feature maps
        spatial_flat = spatial_feats.reshape(b, t, -1)  # (B, T, 5120)

        # Temporal sequence modeling via Bidirectional LSTM
        lstm_out, _ = self.lstm(spatial_flat)  # (B, T, 64)

        # Take final temporal state (combining forward final + backward initial contexts)
        seq_representation = lstm_out[:, -1, :]  # (B, 64)

        # Dense classification head
        logits = self.classifier(seq_representation)  # (B, 2)
        return logits


def preprocess_frame_mobilstm(image_bgr: np.ndarray, target_size: Tuple[int, int] = (64, 64)) -> np.ndarray:
    """Preprocess a single video frame according to Khalid (2022) methodology.

    1. Convert BGR to RGB.
    2. Resize to 64x64.
    3. Normalize pixel values by dividing by 255.0 to [0.0, 1.0].
    """
    if image_bgr is None or image_bgr.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, target_size, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def build_keras_mobilstm(
    sequence_length: int = 16,
    image_height: int = 64,
    image_width: int = 64,
    weights_path: Optional[str] = None,
):
    """Optional Keras/TensorFlow builder for loading .h5 weights if available."""
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
        from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Flatten, Input, LSTM, TimeDistributed
        from tensorflow.keras.models import Sequential

        mobilenet = MobileNetV2(include_top=False, weights="imagenet", input_shape=(image_height, image_width, 3))
        mobilenet.trainable = True
        for layer in mobilenet.layers[:-40]:
            layer.trainable = False

        model = Sequential()
        model.add(Input(shape=(sequence_length, image_height, image_width, 3)))
        model.add(TimeDistributed(mobilenet))
        model.add(Dropout(0.25))
        model.add(TimeDistributed(Flatten()))

        lstm_fw = LSTM(units=32)
        lstm_bw = LSTM(units=32, go_backwards=True)
        model.add(Bidirectional(lstm_fw, backward_layer=lstm_bw))
        model.add(Dropout(0.25))

        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(2, activation="softmax"))

        if weights_path and Path(weights_path).is_file():
            model.load_weights(weights_path)
            logger.info("Loaded Keras MoBiLSTM weights from %s", weights_path)

        return model
    except Exception as e:
        logger.warning("TensorFlow/Keras model creation skipped or unavailable: %s", e)
        return None
