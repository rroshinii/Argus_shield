"""End-to-end two-stage ARGUS Shield pipeline."""

import base64
import time
from io import BytesIO
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

import cv2
from adapter.base import ArgusAdapter, ArgusResult, Detection

from .anomaly import AnomalyResult, detect_anomalies
from .consistency import evaluate_consistency
from .decision import DecisionOutcome, decide_detection, load_thresholds
from .mask_and_recover import mask_and_recover
from .preprocess import build_views


class ShieldDetection(Detection):
    decision: DecisionOutcome
    agreement_score: float = 0.0


class ShieldResult(BaseModel):
    detections: list[ShieldDetection] = Field(default_factory=list)
    anomaly_map: str | None = None
    anomaly_score: float
    agreement_score: float
    suspicious_regions: list[dict[str, Any]] = Field(default_factory=list)
    per_view_detections: dict[str, list[Detection]] = Field(default_factory=dict)
    total_latency_ms: float

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def decision(self) -> DecisionOutcome:
        if not self.detections:
            return DecisionOutcome.ABSTAIN
        decisions = [d.decision for d in self.detections]
        if DecisionOutcome.ABSTAIN in decisions:
            return DecisionOutcome.ABSTAIN
        if DecisionOutcome.FLAGGED_WITH_WARNING in decisions:
            return DecisionOutcome.FLAGGED_WITH_WARNING
        return DecisionOutcome.TRUSTED


class ShieldPipeline:
    """Two-stage defense: cheap anomaly gate, then optional ensemble."""

    def __init__(
        self,
        adapter: ArgusAdapter,
        thresholds: dict[str, Any] | str | None = None,
        anomaly_detector: Any = detect_anomalies,
    ) -> None:
        self.adapter = adapter
        self.thresholds = load_thresholds(thresholds) if thresholds else load_thresholds()
        self.anomaly_detector = anomaly_detector

    def run(self, image: np.ndarray) -> ShieldResult:
        started = time.perf_counter()
        anomaly: AnomalyResult = self.anomaly_detector(
            image,
            threshold=float(self.thresholds["anomaly_threshold"]),
        )
        original = self.adapter.predict(image)
        run_ensemble = bool(self.thresholds["always_full_ensemble"]) or anomaly.anomaly_score >= float(
            self.thresholds["prefilter_threshold"]
        )
        if run_ensemble:
            views = build_views(image)
            recovered = mask_and_recover(image, anomaly.suspicious_regions)
            consistency = evaluate_consistency(self.adapter, views, recovered)
            predictions = consistency.predictions
            agreement = consistency.agreement_score
            representative = consistency.representative
            per_detection_agreement = consistency.per_detection_agreement
        else:
            predictions = {"original": original}
            agreement = 1.0 if original.error is None else 0.0
            representative = original
            per_detection_agreement = [agreement] * len(original.detections)
        detections = [
            ShieldDetection(
                **det.model_dump(),
                agreement_score=per_detection_agreement[index] if index < len(per_detection_agreement) else agreement,
                decision=decide_detection(
                    anomaly.anomaly_score,
                    per_detection_agreement[index] if index < len(per_detection_agreement) else agreement,
                    det.score,
                    self.thresholds,
                ),
            )
            for index, det in enumerate(representative.detections)
        ]
        anomaly_map = _encode_anomaly_map(image, anomaly.anomaly_map)
        return ShieldResult(
            detections=detections,
            anomaly_map=anomaly_map,
            anomaly_score=anomaly.anomaly_score,
            agreement_score=agreement,
            suspicious_regions=anomaly.suspicious_regions,
            per_view_detections={name: result.detections for name, result in predictions.items()},
            total_latency_ms=(time.perf_counter() - started) * 1000,
        )


def _encode_anomaly_map(image: np.ndarray, anomaly_map: np.ndarray) -> str:
    heatmap = cv2.normalize(anomaly_map.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    overlay = cv2.addWeighted(image.astype(np.uint8), 0.55, heatmap, 0.45, 0)
    success, encoded = cv2.imencode(".png", overlay)
    return base64.b64encode(encoded.tobytes()).decode("ascii") if success else ""