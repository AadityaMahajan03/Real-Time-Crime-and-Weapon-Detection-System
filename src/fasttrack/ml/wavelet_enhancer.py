"""Wavelet-based Image Processing and Feature Enhancement for Weapon Detection.

Based on the research and methodology from:
'FIREARMS AND KNIVES DETECTION USING YOLO AND WAVELET FOR SURVEILLANCE SYSTEM'
(Almeida & Assalim, Fatec Ourinhos, 2025).

Techniques:
1. 2D Discrete Wavelet Transform (DWT) using Haar ('haar'), Daubechies-2 ('db2'), or Symlet-2 ('sym2').
2. Extraction of directional high-frequency sub-bands (LH horizontal detail, HL, HH, and Composite Fusion).
3. Contrast-enhanced sub-band normalization for structural edge detection.
4. Dual-mode processing:
   - Direct sub-band feeding into wavelet-trained YOLO models.
   - Spatial luminance edge-injection for standard RGB YOLO inference under low CCTV quality.
5. Surveillance bandwidth compression simulator (JPEG Q20).
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional, Tuple

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def _numpy_haar_2d(gray_or_ch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy fallback for 1-level 2D Haar Discrete Wavelet Transform."""
    h, w = gray_or_ch.shape[:2]
    # Ensure even dimensions
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    crop = gray_or_ch[:h_even, :w_even].astype(np.float32)

    # 2x2 block decomposition
    top_left = crop[0::2, 0::2]
    top_right = crop[0::2, 1::2]
    bottom_left = crop[1::2, 0::2]
    bottom_right = crop[1::2, 1::2]

    # Haar coefficients
    LL = (top_left + top_right + bottom_left + bottom_right) * 0.5
    LH = (top_left + top_right - bottom_left - bottom_right) * 0.5  # Horizontal detail
    HL = (top_left - top_right + bottom_left - bottom_right) * 0.5  # Vertical detail
    HH = (top_left - top_right - bottom_left + bottom_right) * 0.5  # Diagonal detail

    return LL, LH, HL, HH


def enhance_contrast(image: np.ndarray, min_dynamic_range: float = 12.0) -> np.ndarray:
    """Normalize array values to full dynamic range [0, 255] uint8.
    
    If the image or sub-band has minimal dynamic range (e.g. flat black space or camera sensor noise),
    it is suppressed to zero to prevent artificial contrast hallucinations.
    """
    if image is None or image.size == 0:
        return image
    img_f = image.astype(np.float32)
    mi, ma = float(np.min(img_f)), float(np.max(img_f))
    spread = ma - mi
    if spread < min_dynamic_range:
        return np.zeros_like(image, dtype=np.uint8)
    norm = ((img_f - mi) / spread) * 255.0
    return np.clip(norm, 0, 255).astype(np.uint8)


class WaveletEnhancer:
    """Wavelet feature extraction and CCTV enhancement pipeline."""

    SUPPORTED_WAVELETS = ("haar", "db2", "sym2")

    def __init__(self, default_wavelet: str = "haar"):
        self.wavelet = default_wavelet if default_wavelet in self.SUPPORTED_WAVELETS else "haar"

    def decompose(
        self,
        image_bgr: np.ndarray,
        wavelet: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform 2D DWT on 3-channel BGR image or grayscale.

        Returns:
            (LL, LH, HL, HH) sub-bands.
        """
        wname = wavelet or self.wavelet

        if HAS_PYWT:
            try:
                # pywt dwt2 supports 3D array along spatial axes (0, 1)
                coeffs = pywt.dwt2(image_bgr, wname, axes=(0, 1))
                LL, (LH, HL, HH) = coeffs
                return LL, LH, HL, HH
            except Exception:
                pass

        # Fallback to pure numpy Haar if pywt fails or is absent
        if image_bgr.ndim == 3:
            b_ll, b_lh, b_hl, b_hh = _numpy_haar_2d(image_bgr[:, :, 0])
            g_ll, g_lh, g_hl, g_hh = _numpy_haar_2d(image_bgr[:, :, 1])
            r_ll, r_lh, r_hl, r_hh = _numpy_haar_2d(image_bgr[:, :, 2])
            LL = np.dstack([b_ll, g_ll, r_ll])
            LH = np.dstack([b_lh, g_lh, r_lh])
            HL = np.dstack([b_hl, g_hl, r_hl])
            HH = np.dstack([b_hh, g_hh, r_hh])
            return LL, LH, HL, HH
        else:
            return _numpy_haar_2d(image_bgr)

    def extract_subband(
        self,
        image_bgr: np.ndarray,
        wavelet: Optional[str] = None,
        subband: str = "LH",
        target_size: Optional[Tuple[int, int]] = (640, 640),
    ) -> np.ndarray:
        """Extract and normalize a specific wavelet sub-band.

        Args:
            image_bgr: Input BGR frame.
            wavelet: 'haar', 'db2', or 'sym2'.
            subband: 'LH' (horizontal detail, recommended by Assalim), 'HL', 'HH',
                     or 'composite' (maximum magnitude of all high frequencies).
            target_size: Optional (width, height) resize for neural model input.

        Returns:
            Contrast-normalized uint8 image ready for YOLO inference.
        """
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        LL, LH, HL, HH = self.decompose(image_bgr, wavelet=wavelet)

        if subband.upper() == "LH":
            detail = LH
        elif subband.upper() == "HL":
            detail = HL
        elif subband.upper() == "HH":
            detail = HH
        elif subband.upper() == "COMPOSITE":
            # Multi-directional high-frequency energy fusion
            detail = np.maximum(np.abs(LH), np.maximum(np.abs(HL), np.abs(HH)))
        elif subband.upper() == "LL":
            detail = LL
        else:
            detail = LH

        normalized = enhance_contrast(detail)

        if target_size is not None and target_size[0] > 0 and target_size[1] > 0:
            normalized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_LINEAR)

        return normalized

    def enhance_cctv_wavelet_rgb(
        self,
        image_bgr: np.ndarray,
        wavelet: Optional[str] = "haar",
        blend_alpha: float = 0.30,
    ) -> np.ndarray:
        """Inject high-frequency wavelet edge details back into luminance channel.

        Accentuates sharp metallic edges, knife points, and gun contours while
        preserving true color context for dual verification.
        """
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr

        h, w = image_bgr.shape[:2]
        _, LH, HL, HH = self.decompose(image_bgr, wavelet=wavelet)

        # Composite high-frequency detail across all orientations
        high_freq = np.sqrt(LH.astype(np.float32)**2 + HL.astype(np.float32)**2 + HH.astype(np.float32)**2)
        if high_freq.ndim == 3:
            high_freq = np.mean(high_freq, axis=2)

        high_norm = enhance_contrast(high_freq)
        high_resized = cv2.resize(high_norm, (w, h), interpolation=cv2.INTER_LINEAR)

        # Blend with LAB Luminance channel
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)

        # Add scaled high-frequency detail to L channel
        l_enhanced = cv2.addWeighted(l_chan, 1.0, high_resized, blend_alpha, 0)
        lab_enhanced = cv2.merge((l_enhanced, a_chan, b_chan))
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    @staticmethod
    def compress_frame_jpeg(image_bgr: np.ndarray, quality: int = 20) -> np.ndarray:
        """Simulate low-bandwidth CCTV network compression as used in Almeida & Assalim study."""
        if image_bgr is None or image_bgr.size == 0:
            return image_bgr
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(quality, 5, 100))]
        success, enc = cv2.imencode(".jpg", image_bgr, encode_param)
        if not success:
            return image_bgr
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
