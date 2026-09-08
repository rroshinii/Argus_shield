"""Metrics for raw-versus-shield evaluation result tables."""

from typing import Any

import numpy as np
import pandas as pd


def compute_metrics(results: pd.DataFrame, max_false_positive_rate: float = 0.05) -> dict[str, Any]:
    """Compute accuracy, abstention, trusted operating point, and latency metrics."""
    clean = results[results["attack_family"] == "clean"]
    attacked = results[results["attack_family"] != "clean"]
    return {
        "clean_accuracy": {
            "raw": _mean(clean["raw_correct"]),
            "shield": _mean(clean["shield_correct"]),
        },
        "attacked_accuracy": {
            family: {"raw": _mean(group["raw_correct"]), "shield": _mean(group["shield_correct"])}
            for family, group in attacked.groupby("attack_family")
        },
        "trusted_accuracy_at_fixed_false_positive_rate": trusted_accuracy_at_fixed_fpr(
            results, max_false_positive_rate
        ),
        "abstention_rate": {
            "overall": _mean(results["shield_decision"] == "ABSTAIN"),
            "per_attack_family": {
                family: _mean(group["shield_decision"] == "ABSTAIN")
                for family, group in results.groupby("attack_family")
            },
        },
        "false_positive_flag_rate_clean": _mean(clean["shield_decision"] == "FLAGGED_WITH_WARNING"),
        "latency_ms": {
            "raw": _latency_stats(results.get("raw_latency_ms", pd.Series(dtype=float))),
            "shield": _latency_stats(results["latency_ms"]),
        },
    }


def trusted_accuracy_at_fixed_fpr(
    results: pd.DataFrame,
    max_false_positive_rate: float = 0.05,
) -> dict[str, float | int | None]:
    """Measure accuracy at the clean operating point whose false flags stay below X."""
    if not 0 <= max_false_positive_rate <= 1:
        raise ValueError("max_false_positive_rate must be between 0 and 1")
    frame = results.copy()
    frame["_risk"] = np.maximum(frame["anomaly_score"].astype(float), 1.0 - frame["agreement_score"].astype(float))
    clean = frame[frame["attack_family"] == "clean"]
    if clean.empty:
        return {"accuracy": None, "threshold": None, "false_positive_rate": None, "coverage": 0.0, "count": 0}
    candidates = sorted(set(clean["_risk"].tolist()) | {1.0})
    valid = [threshold for threshold in candidates if _false_positive_rate(clean, threshold) <= max_false_positive_rate]
    threshold = min(valid) if valid else min(candidates)
    eligible = frame[frame["_risk"] <= threshold]
    non_abstained = eligible[eligible["shield_decision"] != "ABSTAIN"]
    return {
        "accuracy": _mean(non_abstained["shield_correct"]),
        "threshold": float(threshold),
        "false_positive_rate": float(_false_positive_rate(clean, threshold)),
        "coverage": float(len(non_abstained) / len(frame)) if len(frame) else 0.0,
        "count": int(len(non_abstained)),
    }


def summary_rows(results: pd.DataFrame, max_false_positive_rate: float = 0.05) -> pd.DataFrame:
    """Flatten headline metrics into a compact summary table."""
    metrics = compute_metrics(results, max_false_positive_rate)
    rows = [
        {"metric": "clean_accuracy_raw", "value": metrics["clean_accuracy"]["raw"]},
        {"metric": "clean_accuracy_shield", "value": metrics["clean_accuracy"]["shield"]},
        {"metric": "trusted_accuracy_at_fixed_fpr", "value": metrics["trusted_accuracy_at_fixed_false_positive_rate"]["accuracy"]},
        {"metric": "abstention_rate_overall", "value": metrics["abstention_rate"]["overall"]},
    ]
    for family, values in metrics["attacked_accuracy"].items():
        rows.extend(
            [
                {"metric": f"attacked_accuracy_{family}_raw", "value": values["raw"]},
                {"metric": f"attacked_accuracy_{family}_shield", "value": values["shield"]},
            ]
        )
    for mode, values in metrics["latency_ms"].items():
        rows.extend(
            [
                {"metric": f"latency_{mode}_p50_ms", "value": values["p50"]},
                {"metric": f"latency_{mode}_p95_ms", "value": values["p95"]},
            ]
        )
    return pd.DataFrame(rows)


def _mean(values: Any) -> float | None:
    return float(values.mean()) if len(values) else None


def _false_positive_rate(clean: pd.DataFrame, threshold: float) -> float:
    return float((clean["_risk"] > threshold).mean()) if len(clean) else 0.0


def _latency_stats(values: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"p50": None, "p95": None}
    return {"p50": float(numeric.quantile(0.50)), "p95": float(numeric.quantile(0.95))}