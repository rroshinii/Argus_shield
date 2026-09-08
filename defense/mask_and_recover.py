"""Mask suspicious regions and create recovery variants for re-inference."""

import cv2
import numpy as np


def mask_and_recover(
    image: np.ndarray,
    suspicious_regions: list[dict[str, float | int]],
    method: str = "inpaint",
) -> np.ndarray:
    """Recover suspicious regions with inpainting or a blurred replacement."""
    if method not in {"inpaint", "blur"}:
        raise ValueError("method must be 'inpaint' or 'blur'")
    source = np.clip(image, 0, 255).astype(np.uint8)
    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    for region in suspicious_regions:
        x, y = int(region["x"]), int(region["y"])
        width, height = int(region["width"]), int(region["height"])
        mask[max(0, y) : y + max(0, height), max(0, x) : x + max(0, width)] = 255
    if not np.any(mask):
        return source.copy()
    if method == "inpaint":
        return cv2.inpaint(source, mask, 3, cv2.INPAINT_TELEA)
    blurred = cv2.GaussianBlur(source, (0, 0), 5)
    return np.where(mask[..., None] == 255, blurred, source).astype(np.uint8)