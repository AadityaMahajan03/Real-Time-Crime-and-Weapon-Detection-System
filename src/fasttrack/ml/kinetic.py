"""Kinetic feature extraction and multi-person proximity gating for violence detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import BoundingBox

logger = get_logger("ml.kinetic")


@dataclass
class KineticMetrics:
    """Quantitative kinetic motion and proximity features extracted from video frames."""

    kinetic_energy: float
    burst_acceleration: float
    direction_entropy: float
    person_count: int
    in_proximity: bool
    raw_kinetic_score: float
    gated_kinetic_score: float
    gating_rationale: str


class KineticFeatureExtractor:
    """Spatiotemporal kinetic analyzer extracting velocity, acceleration, and contact dynamics.

    Implements multi-person proximity gating to completely suppress false positives
    from solitary individuals running, jumping, exercising, or gesturing.
    """

    def __init__(
        self,
        proximity_pixel_distance: float = 180.0,
        energy_scale: float = 12.0,
        acceleration_scale: float = 15.0,
    ) -> None:
        self.proximity_pixel_distance = proximity_pixel_distance
        self.energy_scale = energy_scale
        self.acceleration_scale = acceleration_scale

    def extract(
        self,
        frames: List[np.ndarray],
        person_boxes: Optional[List[BoundingBox]] = None,
    ) -> KineticMetrics:
        """Extract kinetic metrics from a sequence of video frames.

        Args:
            frames: Sequence of BGR image frames (typically 8 to 16 sampled frames).
            person_boxes: Detected person bounding boxes in the current scene (if available).

        Returns:
            KineticMetrics detailing motion energy, burst acceleration, proximity, and gated score.
        """
        if len(frames) < 2:
            return KineticMetrics(
                kinetic_energy=0.0,
                burst_acceleration=0.0,
                direction_entropy=0.0,
                person_count=len(person_boxes) if person_boxes else 0,
                in_proximity=False,
                raw_kinetic_score=0.0,
                gated_kinetic_score=0.0,
                gating_rationale="Insufficient frames for optical flow analysis.",
            )

        # Downsample frames to 160x120 grayscale for real-time optical flow computation (<3ms)
        gray_frames = []
        for f in frames:
            if f.ndim == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            else:
                gray = f
            resized = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)
            gray_frames.append(resized)

        # Compute optical flow magnitudes and angles across successive frame pairs
        flow_magnitudes: List[float] = []
        flow_angles: List[np.ndarray] = []

        for i in range(1, len(gray_frames)):
            flow = cv2.calcOpticalFlowFarneback(
                prev=gray_frames[i - 1],
                next=gray_frames[i],
                flow=None,
                pyr_scale=0.5,
                levels=2,
                winsize=11,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            fx, fy = flow[..., 0], flow[..., 1]
            mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
            # 95th percentile magnitude to capture rapid limb strikes rather than background
            p95_mag = float(np.percentile(mag, 95))
            flow_magnitudes.append(p95_mag)
            flow_angles.append(ang.flatten())

        # 1. Kinetic Energy (average top-percentile motion velocity)
        avg_mag = float(np.mean(flow_magnitudes)) if flow_magnitudes else 0.0
        kinetic_energy = float(1.0 / (1.0 + np.exp(-0.8 * (avg_mag - 3.5))))

        # 2. Burst Acceleration (jerk: rate of change in motion magnitude)
        if len(flow_magnitudes) >= 2:
            accels = [abs(flow_magnitudes[j] - flow_magnitudes[j - 1]) for j in range(1, len(flow_magnitudes))]
            max_accel = float(max(accels))
            burst_acceleration = float(1.0 / (1.0 + np.exp(-1.0 * (max_accel - 2.8))))
        else:
            burst_acceleration = 0.0

        # 3. Directional Entropy (erratic turbulent motion vs laminar single-direction movement)
        if flow_angles:
            all_angles = np.concatenate(flow_angles)
            hist, _ = np.histogram(all_angles, bins=8, range=(0, 360), density=True)
            hist = hist[hist > 0]
            entropy = -float(np.sum(hist * np.log(hist))) if len(hist) > 0 else 0.0
            direction_entropy = min(1.0, entropy / np.log(8.0))
        else:
            direction_entropy = 0.0

        # Compute Raw Kinetic Score (composite of energy, burst acceleration, and entropy)
        raw_kinetic_score = float(
            0.45 * kinetic_energy + 0.40 * burst_acceleration + 0.15 * direction_entropy
        )
        raw_kinetic_score = float(np.clip(raw_kinetic_score, 0.0, 1.0))

        # 4. Multi-Person Proximity Gating
        person_count = len(person_boxes) if person_boxes is not None else 0
        in_proximity = False
        gated_score = raw_kinetic_score
        rationale = "Standard kinetic evaluation."

        if person_boxes is not None:
            if person_count <= 1:
                # STRICT REQUIREMENT: If only 1 person in scene, score CANNOT exceed 0.20
                gated_score = min(raw_kinetic_score, 0.20)
                in_proximity = False
                rationale = (
                    f"Single person in frame ({person_count}); score strictly capped at <= 0.20 "
                    "to eliminate false positives from running, jumping, or exercising."
                )
            else:
                # Check pairwise proximity between all detected persons
                has_close_contact = False
                min_center_dist = float("inf")

                for a_idx in range(len(person_boxes)):
                    for b_idx in range(a_idx + 1, len(person_boxes)):
                        box_a = person_boxes[a_idx]
                        box_b = person_boxes[b_idx]

                        # Check bounding-box overlap (IoU / intersection)
                        inter_x1 = max(box_a.x1, box_b.x1)
                        inter_y1 = max(box_a.y1, box_b.y1)
                        inter_x2 = min(box_a.x2, box_b.x2)
                        inter_y2 = min(box_a.y2, box_b.y2)

                        inter_w = max(0.0, inter_x2 - inter_x1)
                        inter_h = max(0.0, inter_y2 - inter_y1)
                        inter_area = inter_w * inter_h

                        # Center distance
                        ca_x, ca_y = (box_a.x1 + box_a.x2) / 2.0, (box_a.y1 + box_a.y2) / 2.0
                        cb_x, cb_y = (box_b.x1 + box_b.x2) / 2.0, (box_b.y1 + box_b.y2) / 2.0
                        dist = math.hypot(ca_x - cb_x, ca_y - cb_y)
                        if dist < min_center_dist:
                            min_center_dist = dist

                        # If overlapping or close center distance (e.g. within 1.5 person widths)
                        avg_width = (box_a.width + box_b.width) / 2.0
                        if inter_area > 0 or dist < max(self.proximity_pixel_distance, avg_width * 1.6):
                            has_close_contact = True
                            break
                    if has_close_contact:
                        break

                in_proximity = has_close_contact
                if in_proximity:
                    gated_score = raw_kinetic_score
                    rationale = (
                        f"Multi-person close physical proximity confirmed ({person_count} persons; "
                        f"min separation: {min_center_dist:.1f}px). Close contact altercation dynamics active."
                    )
                else:
                    # Multiple people, but widely separated across the frame
                    gated_score = min(raw_kinetic_score, 0.25)
                    rationale = (
                        f"Multiple persons present ({person_count}) but separated by distance "
                        f"({min_center_dist:.1f}px). No physical contact detected."
                    )

        return KineticMetrics(
            kinetic_energy=round(kinetic_energy, 4),
            burst_acceleration=round(burst_acceleration, 4),
            direction_entropy=round(direction_entropy, 4),
            person_count=person_count,
            in_proximity=in_proximity,
            raw_kinetic_score=round(raw_kinetic_score, 4),
            gated_kinetic_score=round(gated_score, 4),
            gating_rationale=rationale,
        )
