import numpy as np
import pytest
from urllib.error import URLError

from adapter.base import Detection
from adapter.factory import get_adapter
from adapter.http_argus import HttpArgusAdapter
from adapter.mock_argus import MockArgusAdapter


def test_mock_adapter_runs_on_sample_image():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    result = MockArgusAdapter(pretrained=False).predict(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )
    assert result.error is None
    assert all(isinstance(detection, Detection) for detection in result.detections)
    assert result.latency_ms >= 0


def test_http_adapter_returns_connection_error(monkeypatch):
    def fail(*args, **kwargs):
        raise URLError("connection refused")

    monkeypatch.setattr("adapter.http_argus.urlopen", fail)
    result = HttpArgusAdapter("http://argus.invalid", retries=1).predict(
        np.zeros((4, 4, 3), dtype=np.uint8)
    )
    assert result.error is not None
    assert "HTTP request failed" in result.error
    assert result.latency_ms >= 0


def test_factory_picks_adapter_from_config(monkeypatch):
    mock = get_adapter({"adapter": "mock", "pretrained": False})
    assert isinstance(mock, MockArgusAdapter)

    monkeypatch.setenv("ARGUS_API_KEY", "test-only-key")
    http = get_adapter(
        {"adapter": "http", "base_url": "http://argus.invalid", "access_level": "gray-box"}
    )
    assert isinstance(http, HttpArgusAdapter)
    assert http.access_level == "gray-box"