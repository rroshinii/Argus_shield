import base64
from io import BytesIO

import cv2
import numpy as np
from fastapi.testclient import TestClient

from adapter.base import ArgusResult, Detection
from defense.decision import DecisionOutcome
from defense.pipeline import ShieldResult


class FakePipeline:
    def run(self, image: np.ndarray) -> ShieldResult:
        return ShieldResult(
            detections=[{"label": "sample", "score": 0.95, "box": [0, 0, 12, 12], "decision": DecisionOutcome.TRUSTED}],
            anomaly_map="aGVsbG8=",
            anomaly_score=0.0,
            agreement_score=1.0,
            suspicious_regions=[],
            per_view_detections={"original": [Detection(label="sample", score=0.95, box=[0, 0, 12, 12])]},
            total_latency_ms=1.0,
        )


def sample_png() -> bytes:
    image = np.full((24, 24, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return encoded.tobytes()


def test_health_returns_status(monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "get_adapter_type", lambda: "mock")
    client = TestClient(main.app)
    response = client.get("/shield/health")
    assert response.status_code == 200
    assert response.json()["adapter_type"] == "mock"


def test_predict_clean_image_returns_valid_decision(monkeypatch):
    import api.main as main

    monkeypatch.setenv("SHIELD_API_KEY", "test-key")
    monkeypatch.setattr(main, "get_pipeline", lambda: FakePipeline())
    client = TestClient(main.app)
    response = client.post(
        "/shield/detect",
        files={"image": ("sample.png", BytesIO(sample_png()), "image/png")},
        headers={"X-Shield-Key": "test-key"},
    )
    assert response.status_code == 200
    assert response.json()["detections"][0]["decision"] == "TRUSTED"
    assert response.json()["anomaly_heatmap_base64"]


def test_predict_without_api_key_returns_401(monkeypatch):
    import api.main as main

    monkeypatch.setenv("SHIELD_API_KEY", "test-key")
    client = TestClient(main.app)
    response = client.post(
        "/shield/detect",
        json={"image_base64": base64.b64encode(sample_png()).decode("ascii")},
    )
    assert response.status_code == 401


def test_predict_large_image_rescales_boxes(monkeypatch):
    import api.main as main

    class BoxCheckPipeline:
        def run(self, image: np.ndarray) -> ShieldResult:
            # Check downscaled dimension
            assert max(image.shape[:2]) <= 800
            return ShieldResult(
                detections=[{"label": "person", "score": 0.9, "box": [10.0, 20.0, 50.0, 100.0], "decision": DecisionOutcome.TRUSTED}],
                anomaly_map="aGVsbG8=",
                anomaly_score=0.1,
                agreement_score=1.0,
                suspicious_regions=[],
                per_view_detections={},
                total_latency_ms=10.0,
            )

    large_img = np.full((1200, 1600, 3), 128, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", large_img)
    assert success

    monkeypatch.setenv("SHIELD_API_KEY", "test-key")
    monkeypatch.setattr(main, "get_pipeline", lambda: BoxCheckPipeline())
    client = TestClient(main.app)
    response = client.post(
        "/shield/detect",
        files={"image": ("large.png", BytesIO(encoded.tobytes()), "image/png")},
        headers={"X-Shield-Key": "test-key"},
    )
    assert response.status_code == 200
    det = response.json()["detections"][0]
    # Since 1600 was scaled to 800, scale factor is 2.0
    assert det["box"] == [20.0, 40.0, 100.0, 200.0]


def test_predict_timeout_triggers_504_when_enforced(monkeypatch):
    import time
    import api.main as main

    class SlowPipeline:
        def run(self, image: np.ndarray) -> ShieldResult:
            time.sleep(0.1)
            return ShieldResult(
                detections=[],
                anomaly_score=0.0,
                agreement_score=1.0,
                total_latency_ms=100.0,
            )

    monkeypatch.setenv("SHIELD_API_KEY", "test-key")
    monkeypatch.setenv("SHIELD_TIMEOUT_SECONDS", "0.02")
    monkeypatch.setattr(main, "get_pipeline", lambda: SlowPipeline())
    client = TestClient(main.app)
    response = client.post(
        "/shield/detect",
        files={"image": ("sample.png", BytesIO(sample_png()), "image/png")},
        headers={"X-Shield-Key": "test-key"},
    )
    assert response.status_code == 504
    assert response.json()["detail"] == "shield request timed out"