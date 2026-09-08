"""Benign-but-stressing image transforms for evaluation."""

from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def apply_transform(image: np.ndarray, transform_name: str, **params: Any) -> np.ndarray:
    """Apply one named stress transform and return a uint8 image array."""
    transforms = {
        "resize": resize,
        "jpeg": jpeg_recompress,
        "jpeg_recompression": jpeg_recompress,
        "blur": gaussian_blur,
        "brightness_contrast": brightness_contrast,
        "rotation": rotate,
        "crop": crop,
        "print_recapture": print_and_recapture,
    }
    if transform_name not in transforms:
        raise ValueError(f"unknown transform: {transform_name}")
    return transforms[transform_name](image, **params)


def resize(image: np.ndarray, width: int | None = None, height: int | None = None, scale: float = 1.0) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    width = width or max(1, round(source_width * scale))
    height = height or max(1, round(source_height * scale))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def jpeg_recompress(image: np.ndarray, quality: int = 70) -> np.ndarray:
    quality = max(1, min(100, int(quality)))
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("JPEG decoding failed")
    return decoded


def gaussian_blur(image: np.ndarray, radius: float = 1.2) -> np.ndarray:
    kernel = max(3, int(round(radius * 4)) | 1)
    return cv2.GaussianBlur(image, (kernel, kernel), radius)


def brightness_contrast(image: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    adjusted = image.astype(np.float32) * contrast
    adjusted = (adjusted - 127.5) + 127.5 * brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def rotate(image: np.ndarray, angle: float = 3.0) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)


def crop(image: np.ndarray, fraction: float = 0.9) -> np.ndarray:
    fraction = max(0.01, min(1.0, fraction))
    height, width = image.shape[:2]
    crop_height, crop_width = max(1, round(height * fraction)), max(1, round(width * fraction))
    top, left = (height - crop_height) // 2, (width - crop_width) // 2
    return image[top : top + crop_height, left : left + crop_width]


def print_and_recapture(image: np.ndarray, noise_std: float = 2.0, seed: int | None = None) -> np.ndarray:
    height, width = image.shape[:2]
    source = image.astype(np.float32)
    source = cv2.GaussianBlur(source, (3, 3), 0.8)
    source_corners = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])
    shift = min(height, width) * 0.02
    destination = source_corners + np.float32([[shift, 0], [-shift, shift], [0, -shift], [-shift, 0]])
    matrix = cv2.getPerspectiveTransform(source_corners, destination)
    warped = cv2.warpPerspective(source, matrix, (width, height), borderMode=cv2.BORDER_REFLECT)
    noise = np.random.default_rng(seed).normal(0, noise_std, size=warped.shape)
    return np.clip(warped + noise, 0, 255).astype(np.uint8)