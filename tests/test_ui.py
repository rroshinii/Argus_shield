"""Tests for UI assets, endpoints, and architectural boundaries."""

import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_ui_index_served():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "ARGUS SHIELD" in response.text
    assert "MOCK MODE" in response.text
    assert "inspectTab" in response.text
    assert "evalTab" in response.text
    assert "settingsTab" in response.text


def test_ui_static_assets_served():
    client = TestClient(app)
    css_res = client.get("/static/styles.css")
    assert css_res.status_code == 200
    assert "--bg-color: #0d0d0f" in css_res.text

    js_res = client.get("/static/app.js")
    assert js_res.status_code == 200
    assert "ARGUS Shield" in js_res.text


def test_eval_summary_endpoint():
    client = TestClient(app)
    response = client.get("/shield/eval-summary")
    # Should either return 200 with rows or 404 if eval hasn't written results yet
    assert response.status_code in {200, 404}
    if response.status_code == 200:
        data = response.json()
        assert "rows" in data
        assert "latency_series" in data


def test_architectural_boundary_no_attacks_imported_by_defense_api_or_ui():
    """Verify that attack-generation code is never imported outside attacks/ and eval/."""
    forbidden_targets = [
        PROJECT_ROOT / "defense",
        PROJECT_ROOT / "api",
        PROJECT_ROOT / "ui",
        PROJECT_ROOT / "adapter",
    ]

    violating_imports = []
    for directory in forbidden_targets:
        if not directory.exists():
            continue
        for py_file in directory.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "attacks" or alias.name.startswith("attacks."):
                            violating_imports.append((str(py_file), alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module == "attacks" or (node.module and node.module.startswith("attacks.")):
                        violating_imports.append((str(py_file), node.module))

    assert violating_imports == [], f"Architectural violation: attacks imported in forbidden modules: {violating_imports}"
