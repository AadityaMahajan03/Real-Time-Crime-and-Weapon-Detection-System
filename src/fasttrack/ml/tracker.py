"""Temporal object tracking and K-of-N confirmation engine for weapon candidates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.fasttrack.core.clock import Clock, SystemClock
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import BoundingBox, Detection

logger = get_logger("ml.tracker")


@dataclass
class TrackState:
    """Internal state tracking a detected object across frames."""

    track_id: int
    class_name: str
    last_box: BoundingBox
    last_confidence: float
    last_seen_ts: float  # monotonic seconds
    window: deque[bool] = field(default_factory=deque)  # True if detected in sampled frame
    history_detections: List[Detection] = field(default_factory=list)

    @property
    def confirmation_count(self) -> int:
        """Count of successful detections in the current rolling window."""
        return sum(1 for d in self.window if d)


class WeaponTracker:
    """Manages multi-object tracking associations and K-of-N candidate confirmation.

    Requires an object to be detected in at least K out of the last N sampled frames
    to be promoted from an unconfirmed transient candidate to a confirmed alert candidate.
    """

    def __init__(
        self,
        k: int = 3,
        n: int = 5,
        track_persistence_sec: float = 2.0,
        iou_threshold: float = 0.35,
        clock: Optional[Clock] = None,
    ) -> None:
        self.k = k
        self.n = n
        self.track_persistence_sec = track_persistence_sec
        self.iou_threshold = iou_threshold
        self.clock = clock or SystemClock()
        self._tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1

    def _associate_detections(self, detections: List[Detection]) -> List[Tuple[Detection, Optional[int]]]:
        """Associate detections to existing tracks using explicit track_id or IoU matching."""
        assigned: List[Tuple[Detection, Optional[int]]] = []
        unmatched_dets = []

        # 1. Match by explicit track_id (e.g. from ByteTrack)
        for det in detections:
            if det.track_id is not None and det.track_id in self._tracks:
                assigned.append((det, det.track_id))
            elif det.track_id is not None:
                # New track with given ID
                assigned.append((det, det.track_id))
            else:
                unmatched_dets.append(det)

        # 2. For detections without track_id, match to existing active tracks using IoU
        now_mono = self.clock.monotonic()
        used_tracks = {tid for _, tid in assigned if tid is not None}

        for det in unmatched_dets:
            best_iou = 0.0
            best_tid = None

            for tid, track in self._tracks.items():
                if tid in used_tracks:
                    continue
                if track.class_name != det.class_name:
                    continue
                if now_mono - track.last_seen_ts > self.track_persistence_sec:
                    continue

                iou = det.box.iou(track.last_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                det.track_id = best_tid
                assigned.append((det, best_tid))
                used_tracks.add(best_tid)
            else:
                # Assign brand new track ID
                new_tid = self._next_track_id
                self._next_track_id += 1
                det.track_id = new_tid
                assigned.append((det, new_tid))
                used_tracks.add(new_tid)

        return assigned

    def update(
        self,
        detections: List[Detection],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Detection], List[Detection]]:
        """Update tracker with detections from current sampled frame.

        Returns:
            Tuple of (confirmed_detections, unconfirmed_candidates).
        """
        now_mono = self.clock.monotonic()
        associations = self._associate_detections(detections)
        active_tids = set()

        confirmed: List[Detection] = []
        unconfirmed: List[Detection] = []

        # Update matched tracks
        for det, tid in associations:
            active_tids.add(tid)
            if tid not in self._tracks:
                self._tracks[tid] = TrackState(
                    track_id=tid,
                    class_name=det.class_name,
                    last_box=det.box,
                    last_confidence=det.confidence,
                    last_seen_ts=now_mono,
                    window=deque(maxlen=self.n),
                )

            track = self._tracks[tid]
            track.last_box = det.box
            track.last_confidence = det.confidence
            track.last_seen_ts = now_mono
            track.window.append(True)
            track.history_detections.append(det)

            # Evaluate K-of-N criteria
            if track.confirmation_count >= self.k:
                confirmed.append(det)
            else:
                unconfirmed.append(det)

        # For existing tracks that received no detection this frame, append False to window
        for tid, track in list(self._tracks.items()):
            if tid not in active_tids:
                if now_mono - track.last_seen_ts <= self.track_persistence_sec:
                    track.window.append(False)
                else:
                    # Prune stale track
                    logger.debug("Pruning stale weapon track ID %d", tid)
                    del self._tracks[tid]

        return confirmed, unconfirmed

    def reset(self) -> None:
        """Clear all active tracks."""
        self._tracks.clear()
        self._next_track_id = 1
