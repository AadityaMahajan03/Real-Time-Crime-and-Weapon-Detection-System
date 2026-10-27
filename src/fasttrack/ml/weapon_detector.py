"""Weapon detection engine wrapping Ultralytics YOLO with ByteTrack and tiled inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.fasttrack.core.config import WeaponDetectorSettings
from src.fasttrack.core.errors import ModelLoadError
from src.fasttrack.core.logging import get_logger
from src.fasttrack.core.types import BoundingBox, Detection, Frame

logger = get_logger("ml.weapon")


class WeaponDetector:
    """Production weapon detector wrapping Ultralytics YOLO with safety guards.

    Features:
    - Explicit class map validation at initialization (fails loudly on mismatch)
    - Per-class confidence thresholds, min-box area and aspect-ratio geometric filters
    - CUDA Out-of-Memory catch with automatic CPU fallback
    - Bounded coordinate rescaling returning boxes in original frame coordinates
    - Optional tiled inference for detecting small or distant weapons
    - Precise per-frame inference latency tracking
    """

    def __init__(
        self,
        settings: Optional[WeaponDetectorSettings] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.settings = settings or WeaponDetectorSettings()
        self.checkpoint = checkpoint_path or self.settings.model_checkpoint
        self.fallback_checkpoint = self.settings.fallback_checkpoint

        # Device selection: prefer CUDA if available, fallback to CPU
        if device:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model: Any = None
        self.class_indices: Dict[int, str] = {}
        self.wavelet_model: Any = None
        self.wavelet_class_indices: Dict[int, str] = {}
        self._last_inference_ms: float = 0.0
        self._is_degraded_cpu = False
        self._last_detected_persons: List[BoundingBox] = []

        from src.fasttrack.ml.pose_prior import SpatialPosePrior
        from src.fasttrack.ml.verifier import WeaponCropVerifier
        from src.fasttrack.ml.wavelet_enhancer import WaveletEnhancer
        self.pose_prior = SpatialPosePrior()
        self.crop_verifier = WeaponCropVerifier(device=self.device)
        self.wavelet_enhancer = WaveletEnhancer(
            default_wavelet=self.settings.wavelet_mode if self.settings.wavelet_mode in ("haar", "db2", "sym2") else "haar"
        )

        self._load_model()
        self._run_warmup()

    def _load_model(self) -> None:
        """Load YOLO model checkpoint, validating weights and target weapon class maps."""
        from ultralytics import YOLO

        target_path = Path(self.checkpoint)
        if not target_path.exists():
            fallback_path = Path(self.fallback_checkpoint)
            if fallback_path.exists():
                logger.warning(
                    "Primary weapon checkpoint %s not found. Falling back to %s.",
                    self.checkpoint,
                    self.fallback_checkpoint,
                )
                target_path = fallback_path
            else:
                # If neither local weight exists, attempt standard yolo checkpoint
                logger.warning(
                    "Local checkpoint %s not found. Initializing Ultralytics model %s.",
                    self.checkpoint,
                    self.checkpoint,
                )
                target_path = Path(self.checkpoint)

        try:
            self.model = YOLO(str(target_path))
        except Exception as e:
            raise ModelLoadError(f"Failed to load YOLO model from {target_path}: {e}")

        # Validate model class names against configured weapon classes
        model_names = self.model.names if hasattr(self.model, "names") else {}
        if not model_names:
            raise ModelLoadError(f"Model at {target_path} exposes no class names dictionary.")

        # Invert or search for weapon classes
        configured_targets = set(self.settings.per_class_thresholds.keys())
        # Also include standard weapon aliases
        configured_targets.update(
            {"knife", "pistol", "rifle", "gun", "weapon", "firearm", "grenade", "scissors", "baseball bat"}
        )

        matched_classes = {}
        for idx, name in model_names.items():
            name_lower = name.lower()
            if name_lower in ("guns", "firearms"):
                matched_classes[int(idx)] = "gun"
            elif name_lower in ("knives",):
                matched_classes[int(idx)] = "knife"
            elif name_lower in configured_targets or any(tgt in name_lower for tgt in configured_targets):
                matched_classes[int(idx)] = name_lower

        if not matched_classes:
            available_summary = ", ".join(list(model_names.values())[:10])
            raise ModelLoadError(
                f"Class map mismatch: Checkpoint at {target_path} contains no recognized weapon classes. "
                f"Configured: {configured_targets}. Available in model: [{available_summary}...]. "
                f"Never merging labels silently."
            )

        self.class_indices = matched_classes
        logger.info(
            "WeaponDetector initialized on device '%s' with %d active weapon classes: %s",
            self.device,
            len(self.class_indices),
            list(self.class_indices.values()),
        )

        # Load optional dedicated Wavelet YOLO checkpoint (Assalim & Almeida 2025)
        if self.settings.wavelet_mode != "none" or self.settings.enable_wavelet_fusion:
            wv_path_str = self.settings.wavelet_checkpoint
            if wv_path_str:
                wv_path = Path(wv_path_str)
                if wv_path.exists():
                    try:
                        self.wavelet_model = YOLO(str(wv_path))
                        wv_names = self.wavelet_model.names if hasattr(self.wavelet_model, "names") else {}
                        wv_matched = {}
                        for idx, name in wv_names.items():
                            nl = name.lower()
                            if nl in ("guns", "firearms"):
                                wv_matched[int(idx)] = "gun"
                            elif nl in ("knives",):
                                wv_matched[int(idx)] = "knife"
                            elif nl in configured_targets or any(tgt in nl for tgt in configured_targets):
                                wv_matched[int(idx)] = nl
                        self.wavelet_class_indices = wv_matched
                        logger.info(
                            "Loaded dedicated Wavelet YOLO checkpoint %s (%d classes: %s)",
                            wv_path,
                            len(self.wavelet_class_indices),
                            list(self.wavelet_class_indices.values()),
                        )
                    except Exception as e:
                        logger.warning("Could not load wavelet checkpoint %s: %s", wv_path, e)

    def _run_warmup(self) -> None:
        """Run warm-up forward pass to allocate CUDA buffers and avoid first-frame latency spike."""
        try:
            dummy = np.zeros((self.settings.input_size, self.settings.input_size, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                device=self.device,
                imgsz=self.settings.input_size,
                verbose=False,
                half=self.settings.use_fp16 and "cuda" in self.device,
            )
            logger.info("WeaponDetector warm-up pass completed on %s.", self.device)
        except Exception as e:
            logger.warning("WeaponDetector warm-up warning (non-fatal): %s", e)

    @property
    def last_inference_ms(self) -> float:
        """Return execution latency of the most recent inference pass in milliseconds."""
        return self._last_inference_ms

    @property
    def last_detected_persons(self) -> List[BoundingBox]:
        """Return the list of detected person bounding boxes from the most recent inference pass."""
        return list(self._last_detected_persons)

    def _filter_detection(
        self,
        box: BoundingBox,
        class_name: str,
        confidence: float,
        frame_image: Optional[np.ndarray] = None,
    ) -> bool:
        """Apply geometric filters, per-class confidence thresholds, and black-space noise rejection."""
        # 1. Per-class confidence threshold (dynamic with UI slider)
        class_thresh = self.settings.per_class_thresholds.get(
            class_name, self.settings.default_confidence_threshold
        )
        thresh = min(class_thresh, self.settings.default_confidence_threshold)
        if confidence < thresh:
            return False

        # 2. Minimum box area filter (eliminates noisy sub-pixel artifacts)
        if box.area < self.settings.min_box_area_px:
            return False

        # 3. Maximum aspect ratio filter (e.g. extremely thin line boxes)
        ar = box.aspect_ratio
        if ar > self.settings.max_aspect_ratio or (ar > 0 and ar < (1.0 / self.settings.max_aspect_ratio)):
            return False

        # 4. Reject full-screen hallucinations (>75% of both frame dimensions)
        if frame_image is not None and frame_image.size > 0:
            h_f, w_f = frame_image.shape[:2]
            if (box.width / max(1, w_f)) > 0.75 and (box.height / max(1, h_f)) > 0.75:
                return False

            # 5. Reject black space / flat dark background false positives
            crop = frame_image[int(max(0, box.y1)):int(min(h_f, box.y2)), int(max(0, box.x1)):int(min(w_f, box.x2))]
            if crop.size > 0:
                mean_lum = float(np.mean(crop))
                std_lum = float(np.std(crop))
                if mean_lum < 20.0 and std_lum < 14.0:
                    return False

        return True

    def detect(self, frame: Frame) -> List[Detection]:
        """Execute weapon detection on input Frame with OOM safety and coordinate rescaling."""
        if frame.image is None or frame.image.size == 0:
            return []

        h_orig, w_orig = frame.height, frame.width
        t_start = time.perf_counter()

        if self.settings.enable_wavelet_fusion:
            # Dual-stream: Standard RGB stream + Wavelet detail stream
            std_dets = self._detect_standard(frame)
            wv_dets = self._detect_wavelet(frame)
            detections = self._nms_filter(std_dets + wv_dets, self.settings.nms_iou_threshold)
        elif self.settings.wavelet_mode != "none":
            # Dedicated Wavelet inference mode (Assalim & Almeida 2025)
            detections = self._detect_wavelet(frame)
        elif self.settings.tiled_inference and min(h_orig, w_orig) >= 720:
            detections = self._detect_tiled(frame)
        else:
            detections = self._detect_standard(frame)

        self._last_inference_ms = (time.perf_counter() - t_start) * 1000.0
        return detections

    def _detect_wavelet(self, frame: Frame) -> List[Detection]:
        """Wavelet transform inference extracting directional high-frequency weapon details.

        Decomposes input frame into 2D DWT sub-bands (Haar, db2, sym2) and evaluates
        weapon candidates with sub-band contrast normalization.
        """
        h_orig, w_orig = frame.height, frame.width
        target_size = (self.settings.input_size, self.settings.input_size)

        wavelet_name = (
            self.settings.wavelet_mode
            if self.settings.wavelet_mode in ("haar", "db2", "sym2", "composite")
            else "haar"
        )
        # Extract LH (horizontal detail) sub-band as established in Almeida & Assalim
        subband_type = "COMPOSITE" if wavelet_name == "composite" else "LH"
        actual_wv = "haar" if wavelet_name == "composite" else wavelet_name

        # Skip wavelet detail inference if frame has no edge energy (e.g. black space or dark field)
        _, LH, _, _ = self.wavelet_enhancer.decompose(frame.image, wavelet=actual_wv)
        if float(np.std(LH)) < 3.2 or float(np.mean(frame.image)) < 12.0:
            return []

        wavelet_img = self.wavelet_enhancer.extract_subband(
            frame.image,
            wavelet=actual_wv,
            subband=subband_type,
            target_size=target_size,
        )

        active_model = self.wavelet_model or self.model
        active_classes = (
            list(self.wavelet_class_indices.keys())
            if self.wavelet_model
            else list(self.class_indices.keys())
        )
        active_map = self.wavelet_class_indices if self.wavelet_model else self.class_indices

        try:
            results = active_model.predict(
                wavelet_img,
                device=self.device,
                imgsz=self.settings.input_size,
                conf=0.15,
                iou=self.settings.nms_iou_threshold,
                classes=active_classes,
                verbose=False,
                half=self.settings.use_fp16 and "cuda" in self.device,
            )
        except Exception as e:
            logger.error("Wavelet detection inference error, falling back to CPU: %s", e)
            results = active_model.predict(
                wavelet_img,
                device="cpu",
                imgsz=self.settings.input_size,
                conf=0.15,
                iou=self.settings.nms_iou_threshold,
                classes=active_classes,
                verbose=False,
            )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return detections

        xyxy = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        classes = boxes_obj.cls.cpu().numpy()
        track_ids = (
            boxes_obj.id.int().cpu().numpy()
            if hasattr(boxes_obj, "id") and boxes_obj.id is not None
            else [None] * len(xyxy)
        )

        # Scale coordinates from (target_size[0], target_size[1]) back to (w_orig, h_orig)
        scale_x = float(w_orig) / float(target_size[0])
        scale_y = float(h_orig) / float(target_size[1])

        raw_candidates = []
        person_boxes: List[BoundingBox] = list(self._last_detected_persons)

        for box_arr, conf, cls_idx, tid in zip(xyxy, confs, classes, track_ids):
            class_name = active_map.get(int(cls_idx), "weapon")
            confidence = float(conf)

            x1 = max(0.0, min(float(box_arr[0]) * scale_x, float(w_orig)))
            y1 = max(0.0, min(float(box_arr[1]) * scale_y, float(h_orig)))
            x2 = max(0.0, min(float(box_arr[2]) * scale_x, float(w_orig)))
            y2 = max(0.0, min(float(box_arr[3]) * scale_y, float(h_orig)))
            box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

            if class_name.lower() in ("person", "human"):
                person_boxes.append(box)
            else:
                raw_candidates.append((box, class_name, confidence, tid))

        for box, class_name, confidence, tid in raw_candidates:
            if not self._filter_detection(box, class_name, confidence, frame_image=frame.image):
                continue

            # Stage B: Spatial Hand/Wrist Prior
            if person_boxes:
                is_spatial_valid, rationale = self.pose_prior.evaluate_candidate(
                    weapon_box=box,
                    weapon_class=class_name,
                    person_boxes=person_boxes,
                )
                if not is_spatial_valid:
                    logger.debug("Filtered wavelet candidate via spatial prior: %s (%s)", class_name, rationale)
                    continue

            # Stage C: Crop Verifier CNN
            crop_h = int(box.y2 - box.y1)
            crop_w = int(box.x2 - box.x1)
            fused_conf = confidence
            if crop_h >= 12 and crop_w >= 12 and frame.image is not None:
                crop = frame.image[int(box.y1):int(box.y2), int(box.x1):int(box.x2)]
                is_ver_valid, fused_conf, _ = self.crop_verifier.verify_crop(
                    image_bgr=crop,
                    candidate_class=class_name,
                    detector_conf=confidence,
                )
                if not is_ver_valid:
                    logger.debug("Filtered candidate via crop verifier: %s (fused: %.2f)", class_name, fused_conf)
                    continue

            det = Detection(
                box=box,
                class_name=class_name,
                confidence=fused_conf,
                track_id=int(tid) if tid is not None else None,
                source_id=frame.source_id,
                timestamp_utc=frame.ts_utc,
            )
            detections.append(det)

        return detections

    def _detect_standard(self, frame: Frame) -> List[Detection]:
        """Standard full-frame YOLO prediction pass."""
        h_orig, w_orig = frame.height, frame.width

        try:
            results = self.model.predict(
                frame.image,
                device=self.device,
                imgsz=self.settings.input_size,
                conf=0.15,  # Low base threshold; per-class thresholding applied downstream
                iou=self.settings.nms_iou_threshold,
                classes=list(self.class_indices.keys()),
                verbose=False,
                half=self.settings.use_fp16 and "cuda" in self.device,
            )
        except torch.cuda.OutOfMemoryError as oom:
            logger.error(
                "CUDA Out-of-Memory during weapon detection forward pass. Retrying on CPU: %s", oom
            )
            torch.cuda.empty_cache()
            self.device = "cpu"
            self._is_degraded_cpu = True
            results = self.model.predict(
                frame.image,
                device="cpu",
                imgsz=self.settings.input_size,
                conf=0.15,
                iou=self.settings.nms_iou_threshold,
                classes=list(self.class_indices.keys()),
                verbose=False,
            )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return detections

        # Extract box coordinates, confidences, classes, and optional ByteTrack tracking IDs
        xyxy = boxes_obj.xyxy.cpu().numpy()
        confs = boxes_obj.conf.cpu().numpy()
        classes = boxes_obj.cls.cpu().numpy()
        track_ids = (
            boxes_obj.id.int().cpu().numpy()
            if hasattr(boxes_obj, "id") and boxes_obj.id is not None
            else [None] * len(xyxy)
        )

        # First pass: collect person boxes for spatial prior context
        person_boxes: List[BoundingBox] = []
        raw_candidates = []

        for box_arr, conf, cls_idx, tid in zip(xyxy, confs, classes, track_ids):
            class_name = self.class_indices.get(int(cls_idx), "weapon")
            confidence = float(conf)

            # Clamp coordinates to original image bounds
            x1 = max(0.0, min(float(box_arr[0]), float(w_orig)))
            y1 = max(0.0, min(float(box_arr[1]), float(h_orig)))
            x2 = max(0.0, min(float(box_arr[2]), float(w_orig)))
            y2 = max(0.0, min(float(box_arr[3]), float(h_orig)))
            box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

            if class_name.lower() in ("person", "human"):
                person_boxes.append(box)
            else:
                raw_candidates.append((box, class_name, confidence, tid))

        self._last_detected_persons = person_boxes

        # Second pass: evaluate weapon candidates against spatial priors and crop verifier
        for box, class_name, confidence, tid in raw_candidates:
            if not self._filter_detection(box, class_name, confidence, frame_image=frame.image):
                continue

            # Stage B: Spatial Hand/Wrist Prior (filters out shoe false positives)
            is_spatial_valid, rationale = self.pose_prior.evaluate_candidate(
                weapon_box=box,
                weapon_class=class_name,
                person_boxes=person_boxes,
            )
            if not is_spatial_valid:
                logger.debug("Filtered candidate via spatial prior: %s (%s)", class_name, rationale)
                continue

            # Stage C: Crop Verifier CNN (discriminates weapons vs shoes/phones/pens)
            crop_h = int(box.y2 - box.y1)
            crop_w = int(box.x2 - box.x1)
            fused_conf = confidence
            if crop_h >= 12 and crop_w >= 12 and frame.image is not None:
                crop = frame.image[int(box.y1):int(box.y2), int(box.x1):int(box.x2)]
                is_ver_valid, fused_conf, _ = self.crop_verifier.verify_crop(
                    image_bgr=crop,
                    candidate_class=class_name,
                    detector_conf=confidence,
                )
                if not is_ver_valid:
                    logger.debug("Filtered candidate via crop verifier: %s (fused: %.2f)", class_name, fused_conf)
                    continue

            det = Detection(
                box=box,
                class_name=class_name,
                confidence=fused_conf,
                track_id=int(tid) if tid is not None else None,
                source_id=frame.source_id,
                timestamp_utc=frame.ts_utc,
            )
            detections.append(det)

        return detections

    def _detect_tiled(self, frame: Frame) -> List[Detection]:
        """Tiled inference slicing frame into overlapping quadrants for distant objects."""
        h, w = frame.height, frame.width
        half_w, half_h = w // 2, h // 2
        overlap_w, overlap_h = int(w * 0.1), int(h * 0.1)

        # 4 overlapping tiles (x1, y1, x2, y2)
        tiles = [
            (0, 0, half_w + overlap_w, half_h + overlap_h),
            (half_w - overlap_w, 0, w, half_h + overlap_h),
            (0, half_h - overlap_h, half_w + overlap_w, h),
            (half_w - overlap_w, half_h - overlap_h, w, h),
        ]

        raw_candidates: List[Detection] = []
        for tx1, ty1, tx2, ty2 in tiles:
            tile_crop = frame.image[ty1:ty2, tx1:tx2]
            tile_frame = Frame(
                image=tile_crop,
                ts_utc=frame.ts_utc,
                source_ts=frame.source_ts,
                frame_idx=frame.frame_idx,
                source_id=frame.source_id,
            )
            tile_dets = self._detect_standard(tile_frame)

            # Offset coordinates back to full-frame space
            for det in tile_dets:
                det.box.x1 += tx1
                det.box.y1 += ty1
                det.box.x2 += tx1
                det.box.y2 += ty1
                raw_candidates.append(det)

        # Apply Non-Maximum Suppression across aggregated tile candidates
        return self._nms_filter(raw_candidates, self.settings.nms_iou_threshold)

    def _nms_filter(self, detections: List[Detection], iou_thresh: float) -> List[Detection]:
        """Apply NMS over aggregated detections."""
        if not detections:
            return []

        # Sort descending by confidence
        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        kept: List[Detection] = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            kept.append(best)
            sorted_dets = [
                d
                for d in sorted_dets
                if d.class_name != best.class_name or best.box.iou(d.box) < iou_thresh
            ]

        return kept
