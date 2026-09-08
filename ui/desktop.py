"""Desktop UI launcher using pywebview and local FastAPI service."""

import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
import webview

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def wait_for_server(url: str, timeout: float = 15.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def start_server_in_thread(host: str, port: int) -> None:
    # Set default local API key
    os.environ.setdefault("SHIELD_API_KEY", "argus-shield-local")
    from api.main import app

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def launch_desktop(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    health_url = f"http://{host}:{port}/shield/health"
    app_url = f"http://{host}:{port}/"

    # Start server if not already running
    if not wait_for_server(health_url, timeout=0.8):
        thread = threading.Thread(target=start_server_in_thread, args=(host, port), daemon=True)
        thread.start()
        if not wait_for_server(health_url, timeout=15.0):
            print("ERROR: ARGUS Shield server failed to start in time.", file=sys.stderr)
            sys.exit(1)

    window = webview.create_window(
        title="ARGUS Shield — Defense Boundary",
        url=app_url,
        width=1280,
        height=840,
        min_size=(980, 640),
        background_color="#0d0d0f",
    )
    webview.start()


if __name__ == "__main__":
    launch_desktop()
