"""Standalone executable entry point for ARGUS Shield desktop application."""

import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import uvicorn
import webview


def find_free_port(default_port: int = 8765) -> int:
    """Attempt default port first; fallback to any free OS port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", default_port)) != 0:
            return default_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_health(url: str, timeout: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def start_api_server(host: str, port: int) -> None:
    os.environ.setdefault("SHIELD_API_KEY", "argus-shield-local")
    from api.main import app

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


def main() -> None:
    # Handle PyInstaller frozen path
    if getattr(sys, "frozen", False):
        bundle_dir = sys._MEIPASS
        os.environ["ARGUS_SHIELD_BUNDLE"] = bundle_dir
    
    host = "127.0.0.1"
    port = find_free_port(8765)
    health_url = f"http://{host}:{port}/shield/health"
    app_url = f"http://{host}:{port}/"

    server_thread = threading.Thread(
        target=start_api_server,
        args=(host, port),
        daemon=True,
    )
    server_thread.start()

    if not wait_for_health(health_url, timeout=25.0):
        print("ERROR: Background API server failed to start.", file=sys.stderr)
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
    sys.exit(0)


if __name__ == "__main__":
    main()
