"""JSONL manifest generation for attack-family evaluation datasets."""

import json
from pathlib import Path
from typing import Any, Mapping

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ATTACK_FAMILIES = {"clean", "perturbation", "patch_solid", "patch_naturalistic", "transform"}


def build_manifest(
    clean_dir: str | Path,
    output_path: str | Path,
    ground_truth_labels: Mapping[str, str] | None = None,
    ground_truth_boxes: Mapping[str, list[dict[str, Any]]] | None = None,
    attack_configs: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    """Write one explicit clean and attack-family row per clean image."""
    clean_dir = Path(clean_dir)
    output_path = Path(output_path)
    configs = attack_configs or {
        "perturbation": {"epsilon": 8.0, "norm": "linf"},
        "patch_solid": {"size": 16, "location": "random"},
        "patch_naturalistic": {"size": 16, "location": "random"},
        "transform": {"name": "jpeg_recompression", "quality": 70},
    }
    invalid = set(configs) - ATTACK_FAMILIES
    if invalid:
        raise ValueError(f"unsupported attack families: {sorted(invalid)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output_path.open("w", encoding="utf-8") as manifest:
        for image_path in sorted(path for path in clean_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            relative_path = image_path.relative_to(clean_dir).as_posix()
            label = (ground_truth_labels or {}).get(relative_path)
            rows = [("clean", {})] + [(family, dict(params)) for family, params in configs.items() if family != "clean"]
            for attack_family, params in rows:
                manifest.write(
                    json.dumps(
                        {
                            "image_path": relative_path,
                            "attack_family": attack_family,
                            "attack_params": params,
                            "ground_truth_boxes": (ground_truth_boxes or {}).get(relative_path, []),
                            "ground_truth_label": label,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                rows_written += 1
    return rows_written