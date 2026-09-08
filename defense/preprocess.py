"""Generate independent image views for consistency checking."""

from typing import Any

import cv2
import numpy as np


def build_views(
    image: np.ndarray,
    resize_shape: tuple[int, int] = (224, 224),
    jpeg_quality: int = 75,
) -> dict[str, np.ndarray]:
    """Return the standard set of normalized image views."""
    original = _as_uint8_rgb(image)
    height, width = original.shape[:2]
    resized = cv2.resize(original, (resize_shape[1], resize_shape[0]), interpolation=cv2.INTER_AREA)
    views: dict[str, np.ndarray] = {
        "original": original.copy(),
        "resized_renormalized": _renormalize(resized),
        "jpeg_recompressed": _jpeg_recompress(original, jpeg_quality),
        "median_filtered": cv2.medianBlur(original, 3),
        "mild_denoised": cv2.fastNlMeansDenoisingColored(original, None, 3, 3, 7, 21),
    }
    crop_scale = 0.9
    crop_height, crop_width = max(1, round(height * crop_scale)), max(1, round(width * crop_scale))
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    crop = original[top : top + crop_height, left : left + crop_width]
    views["center_crop_scaled"] = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    return views


def preprocess(image: np.ndarray, **kwargs: Any) -> dict[str, np.ndarray]:
    """Compatibility alias for callers that prefer a verb-style API."""
    return build_views(image, **kwargs)


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3}:
        raise ValueError("image must be a 2D or 3D numpy array")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] != 3:
        raise ValueError("image must have one or three channels")
    return np.clip(image, 0, 255).astype(np.uint8)


def _renormalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    minimum = image.min(axis=(0, 1), keepdims=True)
    maximum = image.max(axis=(0, 1), keepdims=True)
    diff = maximum - minimum
    scale = np.divide(255.0, diff, out=np.ones_like(diff), where=diff > 0)
    return np.clip((image - minimum) * scale, 0, 255).astype(np.uint8)


def _jpeg_recompress(image: np.ndarray, quality: int) -> np.ndarray:
    quality = max(1, min(100, int(quality)))
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("JPEG decoding failed")
    return decoded