"""Multi-view ARGUS inference and prediction agreement scoring."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from adapter.base import ArgusAdapter, ArgusResult, Detection


@dataclass
class ConsistencyResult:
    predictions: dict[str, ArgusResult]
    agreement_score: float
    per_detection_agreement: list[float]
    representative: ArgusResult


def evaluate_consistency(
    adapter: ArgusAdapter,
    views: Mapping[str, np.ndarray],
    recovered: np.ndarray | None = None,
    iou_threshold: float = 0.5,
) -> ConsistencyResult:
    """Run every view and match original detections by label and IoU."""
    names = list(views)
    images = [views[name] for name in names]
    if recovered is not None:
        names.append("recovered")
        images.append(recovered)
    results = adapter.batch_predict(images)
    predictions = dict(zip(names, results, strict=True))
    original = predictions.get("original", results[0])
    representative = original
    rep_name = "original"
    if not original.detections and "recovered" in predictions and predictions["recovered"].detections:
        representative = predictions["recovered"]
        rep_name = "recovered"
    original_shape = views[rep_name].shape[:2] if rep_name in views else (recovered.shape[:2] if recovered is not None and rep_name == "recovered" else images[0].shape[:2])
    other_results = [
        (result, views[name].shape[:2] if name in views else recovered.shape[:2] if recovered is not None and name == "recovered" else original_shape)
        for name, result in predictions.items()
        if name != rep_name
    ]
    agreements = [
        sum(
            any(
                det.label == candidate.label
                and iou(_normalize_box(det.box, original_shape), _normalize_box(candidate.box, shape)) >= iou_threshold
                for candidate in result.detections
            )
            for result, shape in other_results
        )
        / len(other_results)
        if other_results
        else 1.0
        for det in representative.detections
    ]
    agreement = sum(agreements) / len(agreements) if agreements else 1.0
    return ConsistencyResult(predictions, float(agreement), agreements, representative)


def iou(first: list[float], second: list[float]) -> float:
    """Compute intersection-over-union for [x1, y1, x2, y2] boxes."""
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _normalize_box(box: list[float], shape: tuple[int, int]) -> list[float]:
    height, width = shape
    return [box[0] / width, box[1] / height, box[2] / width, box[3] / height]