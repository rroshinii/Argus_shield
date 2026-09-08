"""Reproducible raw-versus-shield evaluation CLI."""

import argparse
import base64
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd

from adapter.base import ArgusAdapter
from adapter.factory import get_adapter
from attacks.patch import generate_patch
from attacks.perturbation import generate_perturbation
from attacks.transform import apply_transform
from defense.pipeline import ShieldPipeline
from defense.consistency import iou

from .metrics import summary_rows


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def split_manifest(
    rows: Iterable[dict[str, Any]],
    seed: int = 42,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> list[dict[str, Any]]:
    """Assign partitions by base image path so variants cannot leak across splits."""
    if train_fraction < 0 or validation_fraction < 0 or train_fraction + validation_fraction > 1:
        raise ValueError("partition fractions must be non-negative and sum to at most one")
    output = [dict(row) for row in rows]
    image_paths = sorted({str(row["image_path"]) for row in output})
    random.Random(seed).shuffle(image_paths)
    train_end = round(len(image_paths) * train_fraction)
    validation_end = train_end + round(len(image_paths) * validation_fraction)
    partitions = {
        path: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, path in enumerate(image_paths)
    }
    for row in output:
        row["partition"] = partitions[str(row["image_path"])]
    return output


def evaluate_rows(
    rows: Iterable[dict[str, Any]],
    image_root: str | Path,
    adapter: ArgusAdapter,
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    image_root = Path(image_root)
    output_dir = Path(output_dir)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    pipeline = ShieldPipeline(adapter)
    evaluated: list[dict[str, Any]] = []
    for index, manifest_row in enumerate(rows):
        clean_image = _read_image(image_root / str(manifest_row["image_path"]))
        image = _materialize_attack(clean_image, manifest_row)
        raw_started = time.perf_counter()
        raw_result = adapter.predict(image)
        raw_latency = (time.perf_counter() - raw_started) * 1000
        shield_result = pipeline.run(image)
        ground_truth = manifest_row.get("ground_truth_boxes", [])
        heatmap_path = heatmap_dir / f"{index:06d}.png"
        _save_heatmap_overlay(image, shield_result.anomaly_map, heatmap_path)
        shield_decision = getattr(shield_result.decision, "value", shield_result.decision)
        evaluated.append(
            {
                "image_path": str(manifest_row["image_path"]),
                "partition": manifest_row.get("partition", "unspecified"),
                "attack_family": manifest_row["attack_family"],
                "ground_truth_boxes": ground_truth,
                "raw_detections": [det.model_dump() for det in raw_result.detections],
                "raw_correct": _detections_correct(raw_result.detections, ground_truth),
                "shield_decision": shield_decision,
                "shield_detections": [det.model_dump() for det in shield_result.detections],
                "shield_correct": _detections_correct(shield_result.detections, ground_truth),
                "shield_confidence": max((det.score for det in shield_result.detections), default=0.0),
                "anomaly_score": float(shield_result.anomaly_score),
                "agreement_score": shield_result.agreement_score,
                "latency_ms": shield_result.total_latency_ms,
                "raw_latency_ms": raw_latency,
                "anomaly_heatmap_path": str(heatmap_path),
            }
        )
    return evaluated


def write_outputs(
    rows: list[dict[str, Any]],
    jsonl_path: str | Path,
    summary_path: str | Path,
    max_false_positive_rate: float = 0.05,
) -> Path:
    jsonl_path = Path(jsonl_path)
    summary_path = Path(summary_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    summary = summary_rows(pd.DataFrame(rows), max_false_positive_rate)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = summary_path.with_suffix(".csv")
    summary.to_csv(csv_path, index=False)
    if summary_path.suffix.lower() == ".parquet":
        try:
            summary.to_parquet(summary_path, index=False)
        except (ImportError, ValueError):
            return csv_path
    else:
        summary.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw ARGUS and ShieldPipeline predictions.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", default="eval/results")
    parser.add_argument("--adapter-config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition", choices=["all", "train", "validation", "test"], default="all")
    parser.add_argument("--max-fpr", type=float, default=0.05)
    args = parser.parse_args()
    rows = split_manifest(read_manifest(args.manifest), seed=args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "partitioned_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    selected = rows if args.partition == "all" else [row for row in rows if row["partition"] == args.partition]
    adapter = get_adapter(args.adapter_config) if args.adapter_config else get_adapter()
    evaluated = evaluate_rows(selected, args.image_root, adapter, output_dir)
    write_outputs(evaluated, output_dir / "results.jsonl", output_dir / "summary.parquet", args.max_fpr)


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read image: {path}")
    return image


def _materialize_attack(image: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    family = row["attack_family"]
    params = dict(row.get("attack_params") or {})
    if family == "clean":
        return image
    if family == "perturbation":
        return generate_perturbation(image, **params)
    if family in {"patch_solid", "patch_naturalistic"}:
        if params.get("location") == "random":
            params["location"] = None
        if family == "patch_naturalistic":
            params["source_image"] = image
        boxes = row.get("ground_truth_boxes") or []
        if boxes:
            first_box = boxes[0].get("box", boxes[0]) if isinstance(boxes[0], dict) else boxes[0]
            params.setdefault("object_box", tuple(first_box))
        params["patch_family"] = family
        return generate_patch(image, **params)
    if family == "transform":
        name = params.pop("name", "jpeg_recompression")
        return apply_transform(image, name, **params)
    raise ValueError(f"unsupported attack family: {family}")


def _detections_correct(detections: Any, ground_truth: list[dict[str, Any]], threshold: float = 0.5) -> bool:
    if not ground_truth:
        return len(detections) == 0
    return all(
        any(
            (not truth.get("label") or detection.label == truth["label"])
            and iou(detection.box, truth.get("box", truth)) >= threshold
            for detection in detections
        )
        for truth in ground_truth
    )


def _save_heatmap_overlay(image: np.ndarray, anomaly_map: Any, path: Path) -> None:
    if anomaly_map is None:
        return
    if isinstance(anomaly_map, str):
        path.write_bytes(base64.b64decode(anomaly_map))
        return
    heatmap = np.asarray(anomaly_map, dtype=np.float32)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    cv2.imwrite(str(path), cv2.addWeighted(image, 0.55, heatmap, 0.45, 0))


if __name__ == "__main__":
    main()