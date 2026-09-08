"""Typed HTTP request and response schemas for the shield boundary."""

from typing import Any

from pydantic import BaseModel, Field

from adapter.base import Detection
from defense.pipeline import ShieldDetection, ShieldResult


class Base64ImageRequest(BaseModel):
    image_base64: str = Field(min_length=1)


class ShieldResponse(BaseModel):
    detections: list[ShieldDetection] = Field(default_factory=list)
    anomaly_score: float
    anomaly_heatmap_base64: str | None = None
    agreement_score: float
    suspicious_regions: list[dict[str, Any]] = Field(default_factory=list)
    per_view_detections: dict[str, list[Detection]] = Field(default_factory=dict)
    total_latency_ms: float

    @classmethod
    def from_result(cls, result: ShieldResult, heatmap_base64: str | None) -> "ShieldResponse":
        return cls(
            detections=result.detections,
            anomaly_score=result.anomaly_score,
            anomaly_heatmap_base64=result.anomaly_map or heatmap_base64,
            agreement_score=result.agreement_score,
            suspicious_regions=result.suspicious_regions,
            per_view_detections=result.per_view_detections,
            total_latency_ms=result.total_latency_ms,
        )