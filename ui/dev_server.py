"""Development server runner for ARGUS Shield web interface."""

import os
import sys
import webbrowser

import uvicorn

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def run_dev_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    os.environ.setdefault("SHIELD_API_KEY", "argus-shield-local")
    url = f"http://{host}:{port}/"
    print(f"[*] Starting ARGUS Shield Dev Server at {url}")
    print(f"[*] Client API Key set to: {os.environ['SHIELD_API_KEY']}")
    print("[*] Opening browser...")
    webbrowser.open(url)
    uvicorn.run("api.main:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    run_dev_server()
