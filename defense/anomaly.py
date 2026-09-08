"""Cheap, interpretable patch/anomaly heuristics."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AnomalyResult:
    anomaly_map: np.ndarray
    anomaly_score: float
    suspicious_regions: list[dict[str, float | int]]


def detect_anomalies(
    image: np.ndarray,
    threshold: float = 0.45,
    block_size: int = 8,
) -> AnomalyResult:
    """Combine DCT high-frequency energy, blur reconstruction error, and variance."""
    gray = _gray_float(image)
    frequency = _dct_high_frequency(gray, block_size)
    reconstruction = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.2))
    mean = cv2.GaussianBlur(gray, (0, 0), 2.0)
    squared_mean = cv2.GaussianBlur(gray * gray, (0, 0), 2.0)
    texture = np.maximum(squared_mean - mean * mean, 0)
    heatmap = np.clip(
        0.40 * _normalize(frequency)
        + 0.35 * _normalize(reconstruction)
        + 0.25 * _normalize(texture),
        0,
        1,
    ).astype(np.float32)
    regions = _regions_from_heatmap(heatmap, threshold)
    return AnomalyResult(
        anomaly_map=heatmap,
        anomaly_score=float(np.percentile(heatmap, 99)),
        suspicious_regions=regions,
    )


def _gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32) / 255.0


def _make_dct_matrix(n: int) -> np.ndarray:
    d = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            alpha = np.sqrt(1.0 / n) if i == 0 else np.sqrt(2.0 / n)
            d[i, j] = alpha * np.cos(np.pi * (2 * j + 1) * i / (2 * n))
    return d


_DCT_8_MATRIX = _make_dct_matrix(8)


def _dct_high_frequency(gray: np.ndarray, block_size: int = 8) -> np.ndarray:
    height, width = gray.shape
    pad_h = (block_size - (height % block_size)) % block_size
    pad_w = (block_size - (width % block_size)) % block_size
    if pad_h > 0 or pad_w > 0:
        padded_gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="edge")
    else:
        padded_gray = gray

    ph, pw = padded_gray.shape
    bh, bw = ph // block_size, pw // block_size
    blocks = padded_gray.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)

    dct_mat = _DCT_8_MATRIX if block_size == 8 else _make_dct_matrix(block_size)
    dct_blocks = np.matmul(dct_mat, np.matmul(blocks, dct_mat.T))
    high_freq_blocks = np.mean(np.abs(dct_blocks[:, :, 2:, 2:]), axis=(2, 3))
    response = np.repeat(np.repeat(high_freq_blocks, block_size, axis=0), block_size, axis=1)
    return response[:height, :width]


def _normalize(values: np.ndarray) -> np.ndarray:
    minimum, maximum = float(values.min()), float(values.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)


def _regions_from_heatmap(heatmap: np.ndarray, threshold: float) -> list[dict[str, float | int]]:
    binary = (heatmap >= threshold).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    regions: list[dict[str, float | int]] = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < 4:
            continue
        region = heatmap[y : y + height, x : x + width]
        regions.append(
            {
                "x": int(x),
                "y": int(y),
                "width": int(width),
                "height": int(height),
                "area": int(area),
                "score": float(region.mean()),
                "center_x": float(centroids[index][0]),
                "center_y": float(centroids[index][1]),
            }
        )
    return regions