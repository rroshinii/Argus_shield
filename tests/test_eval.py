import pandas as pd
import pytest

from eval.metrics import compute_metrics
from eval.report import load_results
from eval.run_eval import split_manifest


def result_rows():
    return pd.DataFrame(
        [
            {
                "image_path": "a.png",
                "partition": "validation",
                "attack_family": "clean",
                "raw_correct": True,
                "shield_correct": True,
                "shield_decision": "TRUSTED",
                "anomaly_score": 0.1,
                "agreement_score": 1.0,
                "latency_ms": 10.0,
                "raw_latency_ms": 4.0,
            },
            {
                "image_path": "b.png",
                "partition": "validation",
                "attack_family": "patch_solid",
                "raw_correct": False,
                "shield_correct": True,
                "shield_decision": "FLAGGED_WITH_WARNING",
                "anomaly_score": 0.8,
                "agreement_score": 0.5,
                "latency_ms": 20.0,
                "raw_latency_ms": 5.0,
            },
        ]
    )


def test_split_manifest_keeps_variants_together():
    rows = [
        {"image_path": "a.png", "attack_family": "clean"},
        {"image_path": "a.png", "attack_family": "patch_solid"},
        {"image_path": "b.png", "attack_family": "clean"},
    ]
    split = split_manifest(rows, train_fraction=0.5, validation_fraction=0.0)
    partitions = {row["image_path"]: row["partition"] for row in split}
    assert split[0]["partition"] == split[1]["partition"]
    assert set(partitions.values()) <= {"train", "test"}


def test_metrics_include_accuracy_abstention_and_latency():
    metrics = compute_metrics(result_rows(), max_false_positive_rate=0.05)
    assert metrics["clean_accuracy"]["shield"] == 1.0
    assert metrics["attacked_accuracy"]["patch_solid"]["shield"] == 1.0
    assert metrics["latency_ms"]["shield"]["p50"] == 15.0


def test_report_refuses_test_partition_before_final_run(tmp_path):
    results_path = tmp_path / "results.jsonl"
    rows = result_rows().assign(partition="test").to_dict(orient="records")
    results_path.write_text("\n".join(__import__("json").dumps(row) for row in rows) + "\n")
    with pytest.raises(PermissionError):
        load_results(results_path, partition="test")