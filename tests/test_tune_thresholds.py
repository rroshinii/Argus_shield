import json

from eval.tune_thresholds import tune_thresholds


def test_tuner_uses_validation_only_and_records_metrics(tmp_path):
    results_path = tmp_path / "results.jsonl"
    rows = [
        {
            "partition": "validation",
            "attack_family": "clean",
            "anomaly_score": 0.1,
            "agreement_score": 1.0,
            "shield_confidence": 0.9,
            "shield_correct": True,
            "latency_ms": 10,
        },
        {
            "partition": "validation",
            "attack_family": "patch_solid",
            "anomaly_score": 0.8,
            "agreement_score": 0.5,
            "shield_confidence": 0.9,
            "shield_correct": True,
            "latency_ms": 10,
        },
        {
            "partition": "test",
            "attack_family": "clean",
            "anomaly_score": 0.0,
            "agreement_score": 1.0,
            "shield_confidence": 0.0,
            "shield_correct": False,
            "latency_ms": 9999,
        },
    ]
    results_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    thresholds_path = tmp_path / "defense_thresholds.yaml"
    thresholds_path.write_text(
        "anomaly_threshold: 0.45\nagreement_threshold: 0.6\nconfidence_threshold: 0.35\nalways_full_ensemble: false\n"
    )
    contract_path = tmp_path / "task_contract.md"
    contract_path.write_text("- `latency_budget_ms`: 250\n- `false_positive_rate_budget`: 0.05\n")
    chosen = tune_thresholds(results_path, thresholds_path, contract_path)
    content = thresholds_path.read_text()
    assert chosen["latency_p95_ms"] == 10.0
    assert "Tuned on validation only" in content
    assert "9999" not in content