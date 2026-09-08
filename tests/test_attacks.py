import json

import numpy as np

from attacks.manifest import build_manifest
from attacks.patch import generate_patch
from attacks.perturbation import generate_perturbation
from attacks.transform import apply_transform


def sample_image() -> np.ndarray:
    return np.full((32, 40, 3), 128, dtype=np.uint8)


def test_perturbation_generator_runs():
    result = generate_perturbation(sample_image(), epsilon=8, seed=1)
    assert result.shape == sample_image().shape
    assert result.dtype == np.uint8
    assert np.max(np.abs(result.astype(np.int16) - sample_image())) <= 8


def test_patch_generators_run():
    image = sample_image()
    solid = generate_patch(image, "patch_solid", size=8, location=(2, 3))
    checkerboard = generate_patch(image, "patch_checkerboard", size=8, location=(2, 3))
    natural = generate_patch(image, "patch_naturalistic", size=8, source_image=image, seed=1)
    assert all(output.shape == image.shape and output.dtype == np.uint8 for output in (solid, checkerboard, natural))


def test_transforms_run():
    image = sample_image()
    outputs = [
        apply_transform(image, "resize", scale=0.5),
        apply_transform(image, "jpeg", quality=60),
        apply_transform(image, "blur"),
        apply_transform(image, "brightness_contrast"),
        apply_transform(image, "rotation"),
        apply_transform(image, "crop"),
        apply_transform(image, "print_recapture", seed=1),
    ]
    assert all(isinstance(output, np.ndarray) and output.size for output in outputs)


def test_manifest_writes_attack_family_rows(tmp_path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    (clean_dir / "sample.png").write_bytes(b"placeholder")
    output_path = tmp_path / "manifest.jsonl"
    count = build_manifest(clean_dir, output_path, {"sample.png": "cat"})
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert count == len(rows) == 5
    assert {row["attack_family"] for row in rows} == {
        "clean",
        "perturbation",
        "patch_solid",
        "patch_naturalistic",
        "transform",
    }
    assert all(row["image_path"] == "sample.png" and row["ground_truth_label"] == "cat" for row in rows)