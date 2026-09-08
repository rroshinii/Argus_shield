"""Evaluation-only localized synthetic patch generation."""

from typing import Sequence

import numpy as np
from PIL import Image


def generate_patch(
    image: np.ndarray,
    patch_family: str = "patch_solid",
    size: int | tuple[int, int] = 16,
    location: tuple[int, int] | None = None,
    color: Sequence[int] = (255, 0, 0),
    source_image: np.ndarray | None = None,
    object_box: tuple[float, float, float, float] | None = None,
    near_object: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Paste a labeled baseline or naturalistic patch into an image array."""
    if image.ndim not in {2, 3}:
        raise ValueError("image must be a 2D or 3D numpy array")
    result = _as_rgb_uint8(image).copy()
    height, width = result.shape[:2]
    patch_height, patch_width = _size_pair(size, height, width)
    top, left = _location(location, height, width, patch_height, patch_width, seed, object_box, near_object)

    if patch_family == "patch_solid":
        patch = np.empty((patch_height, patch_width, 3), dtype=np.uint8)
        patch[:] = np.asarray(color, dtype=np.uint8)[:3]
    elif patch_family == "patch_naturalistic":
        if source_image is None:
            raise ValueError("source_image is required for patch_naturalistic")
        patch = _crop_source(source_image, patch_height, patch_width, seed)
    elif patch_family == "patch_checkerboard":
        patch = _checkerboard(patch_height, patch_width, color)
    else:
        raise ValueError("patch_family must be patch_solid, patch_checkerboard, or patch_naturalistic")

    result[top : top + patch_height, left : left + patch_width] = patch
    return result if image.ndim == 3 else result[:, :, 0]


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] != 3:
        raise ValueError("image must have one or three channels")
    return np.clip(image, 0, 255).astype(np.uint8)


def _size_pair(size: int | tuple[int, int], height: int, width: int) -> tuple[int, int]:
    if isinstance(size, int):
        size = (size, size)
    patch_height = min(height, max(1, int(size[0])))
    patch_width = min(width, max(1, int(size[1])))
    return patch_height, patch_width


def _location(
    location: tuple[int, int] | None,
    height: int,
    width: int,
    patch_height: int,
    patch_width: int,
    seed: int | None,
    object_box: tuple[float, float, float, float] | None = None,
    near_object: bool = False,
) -> tuple[int, int]:
    if location is not None:
        top, left = location
        return max(0, min(int(top), height - patch_height)), max(0, min(int(left), width - patch_width))
    rng = np.random.default_rng(seed)
    if object_box is not None:
        x1, y1, x2, y2 = (int(value) for value in object_box)
        if near_object:
            top = max(0, min(height - patch_height, y1 - patch_height // 2))
            left = max(0, min(width - patch_width, x2 - patch_width // 2))
        else:
            top = max(0, min(height - patch_height, (y1 + y2 - patch_height) // 2))
            left = max(0, min(width - patch_width, (x1 + x2 - patch_width) // 2))
        return top, left
    return (
        int(rng.integers(0, height - patch_height + 1)),
        int(rng.integers(0, width - patch_width + 1)),
    )


def _checkerboard(height: int, width: int, color: Sequence[int]) -> np.ndarray:
    primary = np.asarray(color, dtype=np.uint8)[:3]
    secondary = np.asarray([255 - value for value in primary], dtype=np.uint8)
    rows, columns = np.indices((height, width))
    return np.where(((rows + columns) % 2)[..., None] == 0, primary, secondary).astype(np.uint8)


def _crop_source(source_image: np.ndarray, height: int, width: int, seed: int | None) -> np.ndarray:
    source = _as_rgb_uint8(source_image)
    rng = np.random.default_rng(seed)
    top = int(rng.integers(0, max(1, source.shape[0] - height + 1)))
    left = int(rng.integers(0, max(1, source.shape[1] - width + 1)))
    crop = source[top : top + height, left : left + width]
    if crop.shape[:2] != (height, width):
        crop = np.asarray(Image.fromarray(crop).resize((width, height)))
    return crop