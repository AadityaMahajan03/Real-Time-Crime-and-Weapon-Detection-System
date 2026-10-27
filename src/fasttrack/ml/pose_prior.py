"""Spatial Hand/Wrist Prior Engine filtering out context errors and shoe false positives."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import BoundingBox

logger = get_logger("ml.pose_prior")


class SpatialPosePrior:
    """Anatomical context engine verifying weapon candidates against human pose and spatial priors."""

    def __init__(self, foot_region_ratio: float = 0.82) -> None:
        self.foot_region_ratio = foot_region_ratio

    def evaluate_candidate(
        self,
        weapon_box: BoundingBox,
        weapon_class: str,
        person_boxes: List[BoundingBox],
        wrist_keypoints: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[bool, str]:
        """Evaluate if a weapon candidate conforms to human spatial priors.

        Returns:
            is_valid (bool): False if confirmed as a shoe or floor clutter false positive.
            rationale (str): Diagnostic explanation of the spatial filtering verdict.
        """
        # If no persons in scene, check if weapon is plausible
        if not person_boxes:
            # An isolated object on the floor without any human present
            return True, "No persons detected in scene; passed with standard thresholding."

        # Compute weapon center and dimensions
        w_cx = (weapon_box.x1 + weapon_box.x2) / 2.0
        w_cy = (weapon_box.y1 + weapon_box.y2) / 2.0

        # Find closest person
        closest_person = None
        min_dist = float("inf")

        for p in person_boxes:
            p_cx = (p.x1 + p.x2) / 2.0
            p_cy = (p.y1 + p.y2) / 2.0
            dist = math.hypot(w_cx - p_cx, w_cy - p_cy)
            if dist < min_dist:
                min_dist = dist
                closest_person = p

        if closest_person is None:
            return True, "Valid spatial candidate."

        p_h = closest_person.height
        p_w = closest_person.width

        # 1. SHOE / FOOT REJECTION RULE:
        # Check if candidate overlaps the bottom foot/shoe region of the person
        foot_y_threshold = closest_person.y1 + (self.foot_region_ratio * p_h)
        if weapon_box.y1 >= foot_y_threshold:
            # If weapon is inside the bottom 18% of the person box (the feet/shoes zone)
            if closest_person.x1 - 20 <= w_cx <= closest_person.x2 + 20:
                logger.info("SpatialPosePrior: Suppressed candidate in foot/shoe zone (y: %.1f >= %.1f)", weapon_box.y1, foot_y_threshold)
                return False, "Filtered: candidate overlaps foot/shoe region (shoe false-positive)."

        # 2. ASPECT RATIO & ANATOMICAL SIZE CHECK:
        # A knife or pistol cannot be larger than 55% of the person's total height
        if weapon_box.height > (p_h * 0.55):
            return False, "Filtered: candidate exceeds plausible weapon size relative to person height."

        # 3. HAND / WRIST KEYPOINT PRIOR (if pose keypoints provided):
        if wrist_keypoints:
            has_wrist_proximity = False
            for wx, wy in wrist_keypoints:
                wrist_dist = math.hypot(w_cx - wx, w_cy - wy)
                # Max hand radius: 25% of person width
                if wrist_dist <= max(40.0, p_w * 0.45):
                    has_wrist_proximity = True
                    break

            if not has_wrist_proximity and min_dist > (p_w * 1.5):
                return False, "Filtered: candidate is distant from both person and hand/wrist keypoints."

        return True, "Spatial prior confirmed valid weapon candidate."
