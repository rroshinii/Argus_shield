"""Threshold-configured policy for shield outcomes."""

from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml


class DecisionOutcome(str, Enum):
    TRUSTED = "TRUSTED"
    ABSTAIN = "ABSTAIN"
    FLAGGED_WITH_WARNING = "FLAGGED_WITH_WARNING"


DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parent.parent / "configs" / "defense_thresholds.yaml"


def load_thresholds(
    source: Mapping[str, Any] | str | Path = DEFAULT_THRESHOLDS_PATH,
) -> dict[str, Any]:
    if isinstance(source, Mapping):
        thresholds = dict(source)
    else:
        with Path(source).open("r", encoding="utf-8") as stream:
            thresholds = yaml.safe_load(stream) or {}

    required = {
        "prefilter_threshold",
        "anomaly_threshold",
        "agreement_threshold",
        "confidence_threshold",
        "always_full_ensemble",
    }

    missing = required - set(thresholds)
    if missing:
        raise ValueError(f"threshold config missing: {sorted(missing)}")

    return thresholds


def decide_detection(
    anomaly_score: float,
    agreement_score: float,
    confidence: float | None,
    thresholds: Mapping[str, Any] | str | Path = DEFAULT_THRESHOLDS_PATH,
) -> DecisionOutcome:
    config = load_thresholds(thresholds) if not isinstance(thresholds, Mapping) else thresholds
    confidence_value = confidence if confidence is not None else 0.0
    if confidence_value < float(config["confidence_threshold"]):
        return DecisionOutcome.ABSTAIN
    if anomaly_score >= float(config["anomaly_threshold"]) or agreement_score < float(config["agreement_threshold"]):
        return DecisionOutcome.FLAGGED_WITH_WARNING
    return DecisionOutcome.TRUSTED


def decide(
    anomaly_score: float,
    agreement_score: float,
    confidence: float | None,
    thresholds: Mapping[str, Any] | str | Path = DEFAULT_THRESHOLDS_PATH,
) -> DecisionOutcome:
    """Backward-compatible alias for the per-detection policy."""
    return decide_detection(anomaly_score, agreement_score, confidence, thresholds)