"""Validation-only threshold sweep for the defense policy."""

import argparse
import json
import re
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import yaml

from defense.decision import DecisionOutcome


def load_validation_rows(results_path: str | Path) -> list[dict[str, Any]]:
    """Load validation rows only; non-validation rows are discarded immediately."""
    rows: list[dict[str, Any]] = []
    with Path(results_path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("partition") == "validation":
                rows.append(row)
    if not rows:
        raise ValueError("results contain no validation rows")
    return rows


def read_contract_budgets(contract_path: str | Path) -> tuple[float, float]:
    """Read latency and clean false-positive budgets from the Markdown contract."""
    text = Path(contract_path).read_text(encoding="utf-8")
    latency = _read_number(text, "latency_budget_ms")
    false_positive = _read_number(text, "false_positive_rate_budget", "false_positive_rate")
    if latency is None or false_positive is None:
        raise ValueError("task contract must define latency_budget_ms and false_positive_rate_budget")
    if latency <= 0 or not 0 <= false_positive <= 1:
        raise ValueError("task contract budgets are out of range")
    return latency, false_positive


def tune_thresholds(
    results_path: str | Path,
    thresholds_path: str | Path,
    contract_path: str | Path,
    anomaly_candidates: Iterable[float] | None = None,
    agreement_candidates: Iterable[float] | None = None,
    confidence_candidates: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Choose the best feasible operating point using validation rows only."""
    rows = load_validation_rows(results_path)
    latency_budget, false_positive_budget = read_contract_budgets(contract_path)
    current = _load_thresholds(thresholds_path)
    candidates = product(
        _with_current(anomaly_candidates or _default_candidates(0.10, 0.80, 8), current["anomaly_threshold"]),
        _with_current(agreement_candidates or _default_candidates(0.40, 1.00, 7), current["agreement_threshold"]),
        _with_current(confidence_candidates or _default_candidates(0.10, 0.90, 9), current["confidence_threshold"]),
    )
    feasible: list[dict[str, Any]] = []
    for anomaly_threshold, agreement_threshold, confidence_threshold in candidates:
        metrics = _measure(
            rows,
            anomaly_threshold,
            agreement_threshold,
            confidence_threshold,
            false_positive_budget,
            latency_budget,
        )
        if metrics["feasible"]:
            feasible.append(
                {
                    "anomaly_threshold": anomaly_threshold,
                    "agreement_threshold": agreement_threshold,
                    "confidence_threshold": confidence_threshold,
                    **metrics,
                }
            )
    if not feasible:
        raise ValueError("no threshold candidate satisfies the contract budgets")
    chosen = max(
        feasible,
        key=lambda item: (
            item["trusted_accuracy"],
            item["trusted_coverage"],
            -item["false_positive_rate"],
        ),
    )
    _write_thresholds(thresholds_path, current, chosen)
    return chosen


def _measure(
    rows: list[dict[str, Any]],
    anomaly_threshold: float,
    agreement_threshold: float,
    confidence_threshold: float,
    false_positive_budget: float,
    latency_budget: float,
) -> dict[str, Any]:
    decisions = []
    for row in rows:
        confidence = float(row.get("shield_confidence", row.get("shield_score", 0.0) or 0.0))
        if confidence < confidence_threshold:
            decision = DecisionOutcome.ABSTAIN.value
        elif float(row["anomaly_score"]) >= anomaly_threshold or float(row["agreement_score"]) < agreement_threshold:
            decision = DecisionOutcome.FLAGGED_WITH_WARNING.value
        else:
            decision = DecisionOutcome.TRUSTED.value
        decisions.append((row, decision))
    clean = [item for item in decisions if item[0].get("attack_family") == "clean"]
    false_positive_rate = sum(decision != DecisionOutcome.TRUSTED.value for _, decision in clean) / len(clean) if clean else 1.0
    trusted = [(row, decision) for row, decision in decisions if decision == DecisionOutcome.TRUSTED.value]
    trusted_accuracy = sum(bool(row.get("shield_correct")) for row, _ in trusted) / len(trusted) if trusted else 0.0
    trusted_coverage = len(trusted) / len(decisions) if decisions else 0.0
    latency_values = sorted(float(row["latency_ms"]) for row, _ in decisions)
    latency_p95 = _percentile(latency_values, 0.95)
    return {
        "trusted_accuracy": trusted_accuracy,
        "trusted_coverage": trusted_coverage,
        "false_positive_rate": false_positive_rate,
        "latency_p95_ms": latency_p95,
        "feasible": false_positive_rate <= false_positive_budget and latency_p95 <= latency_budget,
    }


def _load_thresholds(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        thresholds = yaml.safe_load(stream) or {}
    required = {"anomaly_threshold", "agreement_threshold", "confidence_threshold"}
    missing = required - set(thresholds)
    if missing:
        raise ValueError(f"threshold config missing: {sorted(missing)}")
    return thresholds


def _write_thresholds(path: str | Path, thresholds: dict[str, Any], chosen: dict[str, Any]) -> None:
    updated = dict(thresholds)
    for key in ("anomaly_threshold", "agreement_threshold", "confidence_threshold"):
        updated[key] = float(chosen[key])
    comment = (
        f"# Tuned on validation only ({date.today().isoformat()}): "
        f"trusted_accuracy={chosen['trusted_accuracy']:.4f}, "
        f"trusted_coverage={chosen['trusted_coverage']:.4f}, "
        f"false_positive_rate={chosen['false_positive_rate']:.4f}, "
        f"latency_p95_ms={chosen['latency_p95_ms']:.2f}."
    )
    Path(path).write_text(comment + "\n" + yaml.safe_dump(updated, sort_keys=False), encoding="utf-8")


def _read_number(text: str, *names: str) -> float | None:
    for name in names:
        match = re.search(rf"(?:^|[-`* ]+)`?{re.escape(name)}`?\s*:\s*([-+]?\d*\.?\d+)", text, re.MULTILINE)
        if match:
            return float(match.group(1))
    return None


def _default_candidates(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    step = (stop - start) / (count - 1)
    return [round(start + index * step, 6) for index in range(count)]


def _with_current(values: Iterable[float], current: float) -> list[float]:
    return sorted(set(float(value) for value in values) | {float(current)})


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("inf")
    position = (len(values) - 1) * quantile
    lower, upper = int(position), min(int(position) + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune defense thresholds on validation rows only.")
    parser.add_argument("--results", required=True)
    parser.add_argument("--thresholds", default="configs/defense_thresholds.yaml")
    parser.add_argument("--task-contract", default="docs/task_contract.md")
    args = parser.parse_args()
    chosen = tune_thresholds(args.results, args.thresholds, args.task_contract)
    print(json.dumps(chosen, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()