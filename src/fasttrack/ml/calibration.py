"""Probability calibration using Temperature Scaling fitted on validation set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.fasttrack.core.logging import get_logger

logger = get_logger("ml.calibration")


class TemperatureScaler:
    """Post-hoc probability calibration via Platt / Temperature Scaling.

    Scales raw network logits z as:
        p = sigmoid(z / T)
    where T > 0 is learned on held-out validation predictions to minimize
    Negative Log-Likelihood (NLL) and prevent overconfident probabilities.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        self.temperature = max(0.01, float(temperature))

    def calibrate_logit(self, logit: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray]:
        """Apply temperature scaling to raw logit and return calibrated probability."""
        if isinstance(logit, torch.Tensor):
            scaled = logit / self.temperature
            return float(torch.sigmoid(scaled).item()) if logit.ndim == 0 else torch.sigmoid(scaled).cpu().numpy()

        arr = np.asarray(logit, dtype=np.float32)
        scaled = arr / self.temperature
        prob = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30.0, 30.0)))
        return float(prob) if np.isscalar(logit) or arr.ndim == 0 else prob

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray, max_iter: int = 50) -> float:
        """Optimize temperature T on validation logits using NLL loss."""
        logits_t = torch.tensor(val_logits, dtype=torch.float32).view(-1, 1)
        labels_t = torch.tensor(val_labels, dtype=torch.float32).view(-1, 1)

        temp_param = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = optim.LBFGS([temp_param], lr=0.01, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def _eval() -> torch.Tensor:
            optimizer.zero_grad()
            clamped_temp = torch.clamp(temp_param, min=0.05, max=10.0)
            loss = criterion(logits_t / clamped_temp, labels_t)
            loss.backward()
            return loss

        optimizer.step(_eval)
        self.temperature = float(torch.clamp(temp_param, min=0.05, max=10.0).item())
        logger.info("Fitted temperature scaling parameter: T = %.4f", self.temperature)
        return self.temperature

    def save(self, file_path: str | Path) -> None:
        """Save fitted calibration temperature to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"temperature": self.temperature}, f, indent=2)

    def load(self, file_path: str | Path) -> None:
        """Load calibration temperature from JSON."""
        path = Path(file_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.temperature = float(data.get("temperature", 1.0))
            logger.info("Loaded calibration temperature T = %.4f from %s", self.temperature, path)
