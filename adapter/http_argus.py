"""Black-box HTTP adapter for a remotely hosted ARGUS endpoint."""

import base64
import json
import os
import time
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from .base import ArgusAdapter, ArgusResult, Detection


class HttpArgusAdapter(ArgusAdapter):
    """Call ARGUS as a black-box REST service with bounded retries."""

    def __init__(
        self,
        base_url: str,
        api_key_env: str = "ARGUS_API_KEY",
        timeout: float = 10.0,
        retries: int = 2,
        access_level: str = "black-box",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = os.getenv(api_key_env)
        self.timeout = timeout
        self.retries = max(0, retries)
        self.access_level = access_level

    def predict(self, image: np.ndarray) -> ArgusResult:
        started = time.perf_counter()
        try:
            payload = json.dumps(
                {"image_base64": self._encode_image(image), "access_level": self.access_level}
            ).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            request = Request(self.base_url, data=payload, headers=headers, method="POST")

            last_error = "request failed"
            for attempt in range(self.retries + 1):
                try:
                    with urlopen(request, timeout=self.timeout) as response:
                        output = json.loads(response.read().decode("utf-8"))
                    return self._result_from_output(output, started)
                except (HTTPError, URLError, TimeoutError, OSError) as exc:
                    last_error = f"HTTP request failed: {exc}"
                    if attempt < self.retries:
                        time.sleep(0.05 * (attempt + 1))
                except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
                    last_error = f"invalid ARGUS response: {exc}"
                    break
            return ArgusResult(latency_ms=self._latency_ms(started), error=last_error)
        except Exception as exc:
            return ArgusResult(
                latency_ms=self._latency_ms(started),
                error=f"HTTP adapter error: {exc}",
            )

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        if not isinstance(image, np.ndarray):
            raise ValueError("image must be a numpy array")
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _result_from_output(output: Any, started: float) -> ArgusResult:
        if not isinstance(output, dict):
            raise ValueError("ARGUS response must be a JSON object")
        detections = [Detection.model_validate(item) for item in output.get("detections", [])]
        return ArgusResult(
            detections=detections,
            raw_output=output,
            latency_ms=HttpArgusAdapter._latency_ms(started),
        )

    @staticmethod
    def _latency_ms(started: float) -> float:
        return (time.perf_counter() - started) * 1000