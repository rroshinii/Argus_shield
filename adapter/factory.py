"""Construct the configured ARGUS adapter."""

from pathlib import Path
from typing import Any

import yaml

from .base import ArgusAdapter
from .http_argus import HttpArgusAdapter
from .mock_argus import MockArgusAdapter

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "task_contract.yaml"


def get_adapter(config: dict[str, Any] | str | Path | None = None) -> ArgusAdapter:
    """Build an adapter from a mapping or YAML config, defaulting to mock mode."""
    settings = _load_config(config)
    adapter_type = str(settings.get("adapter", "mock")).lower()
    if adapter_type == "mock":
        return MockArgusAdapter(
            pretrained=bool(settings.get("pretrained", True)),
            score_threshold=float(settings.get("score_threshold", 0.5)),
        )
    if adapter_type in {"http", "remote"}:
        base_url = settings.get("base_url")
        if not base_url:
            raise ValueError("HTTP ARGUS adapter requires base_url")
        return HttpArgusAdapter(
            base_url=base_url,
            api_key_env=str(settings.get("api_key_env", "ARGUS_API_KEY")),
            timeout=float(settings.get("timeout", 10.0)),
            retries=int(settings.get("retries", 2)),
            access_level=str(settings.get("access_level", "black-box")),
        )
    raise ValueError(f"unsupported ARGUS adapter: {adapter_type}")


def _load_config(config: dict[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        config = DEFAULT_CONFIG_PATH
    if isinstance(config, dict):
        return config
    with Path(config).open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    return loaded if isinstance(loaded, dict) else {}