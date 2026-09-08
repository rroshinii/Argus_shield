"""Evaluation-only bounded black-box pixel perturbations."""

from typing import Any

import numpy as np


def generate_perturbation(
    image: np.ndarray,
    epsilon: float = 8.0,
    norm: str = "linf",
    model: Any | None = None,
    target: Any | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Return a bounded random perturbation without white-box model assumptions.

    ``model`` and ``target`` remain accepted for compatibility but are ignored:
    real ARGUS access is currently mock or black-box only.
    """
    _validate_image(image)
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if norm not in {"linf", "l2"}:
        raise ValueError("norm must be 'linf' or 'l2'")

    rng = np.random.default_rng(seed)
    noise = rng.uniform(-epsilon, epsilon, size=image.shape).astype(np.float32)
    if norm == "l2":
        noise = _normalize_l2_noise(noise, epsilon)
    return _clip_image(image.astype(np.float32) + noise)


def _normalize_l2_noise(noise: np.ndarray, epsilon: float) -> np.ndarray:
    magnitude = float(np.linalg.norm(noise.ravel(), ord=2))
    if magnitude == 0:
        return noise
    return noise * min(1.0, epsilon / magnitude)


def _clip_image(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).round().astype(np.uint8)


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.ndim not in {2, 3}:
        raise ValueError("image must be a 2D or 3D numpy array")