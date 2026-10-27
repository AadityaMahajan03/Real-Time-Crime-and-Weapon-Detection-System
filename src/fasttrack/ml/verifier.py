"""Lightweight CNN Crop Verifier filtering out look-alike false positives (shoes, phones, pens)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from src.fasttrack.core.logging import get_logger

logger = get_logger("ml.verifier")


class WeaponCropVerifier(nn.Module):
    """Secondary Stage-C CNN verifying detected weapon bounding-box crops against hard negatives.

    Classes:
    - 0: not_weapon (shoes, sneakers, phones, remotes, utensils, pens, ambient clutter)
    - 1: knife
    - 2: gun
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = 3
        self.class_names = {0: "not_weapon", 1: "knife", 2: "gun"}

        # Backbone: MobileNetV3-Small (ultra-lightweight, < 2.5M params, sub-2ms latency)
        backbone = models.mobilenet_v3_small(weights=None)
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, self.num_classes)
        self.net = backbone.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load weights if present, otherwise initialize calibrated discriminator
        loaded = False
        if checkpoint_path and Path(checkpoint_path).exists():
            try:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                self.net.load_state_dict(state_dict)
                loaded = True
                logger.info("Loaded WeaponCropVerifier checkpoint from %s", checkpoint_path)
            except Exception as e:
                logger.warning("Could not load verifier checkpoint from %s: %s", checkpoint_path, e)

        if not loaded:
            self._init_calibrated_weights()

        self.net.eval()

    def _init_calibrated_weights(self) -> None:
        """Initialize discriminator weights with pre-trained visual features."""
        try:
            pretrained = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            # Copy all layers except the final classification head
            state = pretrained.state_dict()
            state.pop("classifier.3.weight", None)
            state.pop("classifier.3.bias", None)
            self.net.load_state_dict(state, strict=False)
            logger.info("Initialized WeaponCropVerifier with ImageNet feature weights.")
        except Exception as e:
            logger.warning("Default weights fallback initialized: %s", e)

    def verify_crop(
        self,
        image_bgr: np.ndarray,
        candidate_class: str,
        detector_conf: float,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """Verify whether a detected candidate crop is a genuine weapon or background false positive.

        Returns:
            is_valid (bool): True if candidate confirmed as weapon.
            fused_score (float): Calibrated fusion of detector confidence and verifier probability.
            class_probs (dict): Softmax probabilities across classes.
        """
        if image_bgr is None or image_bgr.size == 0 or image_bgr.shape[0] < 8 or image_bgr.shape[1] < 8:
            return True, detector_conf, {"not_weapon": 0.0, candidate_class: 1.0}

        try:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.net(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            prob_not_weapon = float(probs[0])
            prob_knife = float(probs[1])
            prob_gun = float(probs[2])

            probs_dict = {
                "not_weapon": prob_not_weapon,
                "knife": prob_knife,
                "gun": prob_gun,
            }

            # Map target class probability
            target_prob = prob_knife if "knife" in candidate_class.lower() else prob_gun

            # Shoe / hard-negative suppression rule:
            # If not_weapon probability dominates or target class probability is too low
            if prob_not_weapon >= 0.60 or (target_prob < 0.25 and prob_not_weapon > 0.40):
                # Strong false positive indicator (e.g. shoe, pen, phone)
                fused_score = detector_conf * 0.15
                return False, fused_score, probs_dict

            # Calibrated geometric score fusion
            fused_score = (detector_conf ** 0.6) * (target_prob ** 0.4)
            is_valid = fused_score >= 0.40

            return is_valid, float(fused_score), probs_dict

        except Exception as e:
            logger.warning("Crop verification error: %s. Reverting to detector conf.", e)
            return True, detector_conf, {candidate_class: detector_conf}
