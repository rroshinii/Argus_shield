"""API authentication and rotating request logging middleware."""

import hmac
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

LOG_PATH = os.getenv("SHIELD_API_LOG", "logs/shield_api.log")
logger = logging.getLogger("argus_shield.api")
if not logger.handlers:
    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_PATH, maxBytes=5 * 1024 * 1024, backupCount=3)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ShieldMiddleware(BaseHTTPMiddleware):
    """Require the configured API key and log request metadata only."""

    async def dispatch(self, request: Request, call_next) -> Response:
        started = time.perf_counter()
        status_code = 500
        decision = "-"
        public_paths = {"/shield/health", "/shield/eval-summary", "/docs", "/openapi.json", "/", "/ui", "/favicon.ico"}
        is_public = (
            request.url.path in public_paths
            or request.url.path.startswith(("/static/", "/ui/"))
        )
        if not is_public:
            expected = os.getenv("SHIELD_API_KEY")
            supplied = request.headers.get("X-Shield-Key")
            if not expected or not supplied or not hmac.compare_digest(supplied, expected):
                response = JSONResponse(status_code=401, content={"detail": "invalid or missing API key"})
                status_code = response.status_code
                self._log(request, started, status_code, decision)
                return response
        try:
            response = await call_next(request)
            status_code = response.status_code
            decision = response.headers.get("X-Shield-Decision", "-")
            return response
        finally:
            self._log(request, started, status_code, decision)

    @staticmethod
    def _log(request: Request, started: float, status_code: int, decision: str) -> None:
        logger.info(
            "path=%s method=%s status=%s latency_ms=%.2f decision=%s",
            request.url.path,
            request.method,
            status_code,
            (time.perf_counter() - started) * 1000,
            decision,
        )