"""Geometrical Region of Interest (ROI) zone filtering for detection candidates."""

from __future__ import annotations

from typing import List

from src.fasttrack.core.types import Detection, Zone, ZoneType


class ZoneFilter:
    """Filters candidate detections against camera Include and Ignore polygon zones."""

    def __init__(self, zones: List[Zone]) -> None:
        self.include_zones = [z for z in zones if z.zone_type == ZoneType.INCLUDE]
        self.ignore_zones = [z for z in zones if z.zone_type == ZoneType.IGNORE]

    def _is_point_in_zone(self, px: float, py: float, zone: Zone) -> bool:
        """Evaluate point containment within zone polygon via ray casting."""
        return zone.contains_point(px, py)

    def is_detection_allowed(
        self,
        det: Detection,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """Determine whether a detection is within permitted include zones and not in ignore zones.

        Evaluates the bottom-center or centroid of the bounding box normalized to [0.0, 1.0].
        """
        # Calculate normalized box centroid
        cx = ((det.box.x1 + det.box.x2) / 2.0) / float(frame_width)
        cy = ((det.box.y1 + det.box.y2) / 2.0) / float(frame_height)
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))

        # 1. If any Ignore zone contains the point -> Disallowed (suppressed)
        for zone in self.ignore_zones:
            if self._is_point_in_zone(cx, cy, zone):
                return False

        # 2. If Include zones are defined, the point must be in at least one Include zone
        if self.include_zones:
            in_any_include = any(self._is_point_in_zone(cx, cy, zone) for zone in self.include_zones)
            return in_any_include

        # If no include zones are configured, all non-ignored areas are permitted
        return True

    def filter_detections(
        self,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
    ) -> List[Detection]:
        """Filter list of detections, returning only those in allowed regions."""
        if not self.include_zones and not self.ignore_zones:
            return detections

        return [
            det
            for det in detections
            if self.is_detection_allowed(det, frame_width, frame_height)
        ]
