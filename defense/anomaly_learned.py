"""Reserved interface for a learned anomaly detector."""

import numpy as np

from .anomaly import AnomalyResult


def detect_anomalies(
    image: np.ndarray,
    threshold: float = 0.45,
    block_size: int = 8,
) -> AnomalyResult:
    """Match the heuristic detector signature until a learned model is selected."""
    raise NotImplementedError("learned anomaly detection is not implemented yet")