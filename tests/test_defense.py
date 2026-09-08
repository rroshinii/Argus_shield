import numpy as np

from adapter.base import ArgusAdapter, ArgusResult, Detection
from attacks.patch import generate_patch
from defense.anomaly import detect_anomalies
from defense.consistency import evaluate_consistency
from defense.decision import DecisionOutcome, decide
from defense.mask_and_recover import mask_and_recover
from defense.pipeline import ShieldPipeline, ShieldResult
from defense.preprocess import build_views


class FakeAdapter(ArgusAdapter):
    def predict(self, image: np.ndarray) -> ArgusResult:
        height, width = image.shape[:2]
        return ArgusResult(
            detections=[Detection(label="sample", score=0.9, box=[0, 0, width / 2, height / 2])],
            raw_output={"mock": True},
            latency_ms=0.1,
        )


def sample_images() -> tuple[np.ndarray, np.ndarray]:
    clean = np.full((48, 48, 3), 128, dtype=np.uint8)
    patched = generate_patch(clean, "patch_solid", size=12, location=(16, 18))
    return clean, patched


def test_preprocess_produces_all_views_for_clean_and_patched_images():
    for image in sample_images():
        views = build_views(image)
        assert set(views) == {
            "original",
            "resized_renormalized",
            "jpeg_recompressed",
            "median_filtered",
            "mild_denoised",
            "center_crop_scaled",
        }
        assert all(isinstance(view, np.ndarray) and view.size for view in views.values())


def test_anomaly_detector_returns_heatmap_and_regions_for_both_images():
    for image in sample_images():
        result = detect_anomalies(image)
        assert result.anomaly_map.shape == image.shape[:2]
        assert 0.0 <= result.anomaly_score <= 1.0
        assert isinstance(result.suspicious_regions, list)


def test_mask_and_recover_handles_clean_and_patched_images():
    for image in sample_images():
        recovered = mask_and_recover(
            image,
            [{"x": 16, "y": 18, "width": 12, "height": 12}],
            method="inpaint",
        )
        assert recovered.shape == image.shape
        assert recovered.dtype == np.uint8


def test_consistency_runs_clean_and_patched_views():
    adapter = FakeAdapter()
    for image in sample_images():
        result = evaluate_consistency(adapter, build_views(image), image)
        assert result.agreement_score == 1.0
        assert len(result.predictions) == 7
        assert result.representative.detections[0].label == "sample"


def test_decision_policy_uses_configured_thresholds():
    thresholds = {
        "prefilter_threshold": 0.15,
        "anomaly_threshold": 0.45,
        "agreement_threshold": 0.60,
        "confidence_threshold": 0.35,
        "always_full_ensemble": False,
    }
    assert decide(0.1, 1.0, 0.9, thresholds) == DecisionOutcome.TRUSTED
    assert decide(0.8, 1.0, 0.9, thresholds) == DecisionOutcome.FLAGGED_WITH_WARNING
    assert decide(0.1, 1.0, 0.1, thresholds) == DecisionOutcome.ABSTAIN


def test_pipeline_runs_end_to_end_on_clean_and_patched_images():
    thresholds = {
        "prefilter_threshold": 0.15,
        "anomaly_threshold": 0.45,
        "agreement_threshold": 0.60,
        "confidence_threshold": 0.35,
        "always_full_ensemble": True,
    }
    pipeline = ShieldPipeline(FakeAdapter(), thresholds=thresholds)
    for image in sample_images():
        result = pipeline.run(image)
        assert isinstance(result, ShieldResult)
        assert result.detections[0].label == "sample"
        assert result.detections[0].decision in set(DecisionOutcome)
        assert result.anomaly_map is not None
        assert result.total_latency_ms >= 0
        assert result.per_view_detections