"""Typed adapter contract shared by all ARGUS implementations."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: str
    score: float
    box: list[float]


class ArgusResult(BaseModel):
    """Normalized object-detection result returned by ARGUS."""

    detections: list[Detection] = Field(default_factory=list)
    raw_output: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    error: str | None = None


class ArgusAdapter(ABC):
    """Stable interface for the rest of the shield pipeline."""

    @abstractmethod
    def predict(self, image: np.ndarray) -> ArgusResult:
        """Run one image through ARGUS and return a normalized result."""

    def batch_predict(self, images: list[np.ndarray]) -> list[ArgusResult]:
        """Run the default sequential implementation for a batch."""
        return [self.predict(image) for image in images]