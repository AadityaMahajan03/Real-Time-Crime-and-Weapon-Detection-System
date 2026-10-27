"""Spatiotemporal saliency and temporal attention analysis for violence classification."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


def compute_temporal_saliency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute frame-level temporal saliency and spatial gradient heatmap.

    Args:
        model: TorchVision spatiotemporal CNN (e.g. S3D or R3D-18).
        input_tensor: Tensor of shape (1, 3, T, H, W).
        device: Device to execute computation on.

    Returns:
        Tuple of:
            - temporal_importance: 1D array of length T with normalized frame-level energy.
            - spatial_heatmap: (H, W) array representing peak spatial activation.
    """
    model.eval()
    x = input_tensor.clone().detach().to(device).requires_grad_(True)

    output = model(x)
    target_score = output.sum()
    target_score.backward()

    # Gradients with shape (1, 3, T, H, W)
    grads = x.grad.detach().cpu().numpy()[0]  # (3, T, H, W)

    # 1. Temporal importance across T frames (mean absolute gradient magnitude)
    temporal_importance = np.mean(np.abs(grads), axis=(0, 2, 3))  # Shape (T,)
    if np.max(temporal_importance) > 1e-7:
        temporal_importance = temporal_importance / np.max(temporal_importance)

    # 2. Spatial heatmap across all frames
    spatial_raw = np.mean(np.abs(grads), axis=(0, 1))  # Shape (H, W)
    if np.max(spatial_raw) > 1e-7:
        spatial_heatmap = spatial_raw / np.max(spatial_raw)
    else:
        spatial_heatmap = spatial_raw

    return temporal_importance, spatial_heatmap


def overlay_heatmap_on_frame(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Blend a 2D normalized saliency heatmap onto an original BGR frame."""
    h, w = frame.shape[:2]
    resized_map = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * resized_map), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(frame, 1.0 - alpha, heatmap_colored, alpha, 0)
    return blended
