"""Generate sample clean images and build initial evaluation manifest."""

import json
from pathlib import Path

import cv2
import numpy as np

from attacks.manifest import build_manifest


def generate_sample_images() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    clean_dir = data_dir / "clean"
    manifest_dir = data_dir / "manifests"
    clean_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_boxes = {}
    ground_truth_labels = {}

    # Image 1: Simulated vehicle (rectangular body, wheels, windshield)
    img1 = np.full((256, 256, 3), 220, dtype=np.uint8)
    cv2.rectangle(img1, (0, 180), (256, 256), (100, 100, 100), -1)  # road
    cv2.rectangle(img1, (40, 100), (210, 190), (30, 40, 180), -1)
    cv2.rectangle(img1, (70, 60), (170, 100), (30, 40, 180), -1)
    cv2.rectangle(img1, (80, 65), (160, 95), (200, 220, 240), -1)
    cv2.circle(img1, (75, 190), 22, (20, 20, 20), -1)
    cv2.circle(img1, (175, 190), 22, (20, 20, 20), -1)
    p1 = clean_dir / "car_sample.png"
    cv2.imwrite(str(p1), img1)
    ground_truth_boxes["car_sample.png"] = [{"box": [40, 60, 210, 212], "label": "car"}]
    ground_truth_labels["car_sample.png"] = "car"

    # Image 2: Simulated person silhouette
    img2 = np.full((256, 256, 3), 235, dtype=np.uint8)
    cv2.circle(img2, (128, 50), 22, (60, 70, 90), -1)
    cv2.rectangle(img2, (108, 72), (148, 160), (40, 60, 120), -1)
    cv2.rectangle(img2, (108, 160), (124, 230), (30, 30, 40), -1)
    cv2.rectangle(img2, (132, 160), (148, 230), (30, 30, 40), -1)
    p2 = clean_dir / "person_sample.png"
    cv2.imwrite(str(p2), img2)
    ground_truth_boxes["person_sample.png"] = [{"box": [108, 28, 148, 230], "label": "person"}]
    ground_truth_labels["person_sample.png"] = "person"

    # Image 3: Stop sign
    img3 = np.full((256, 256, 3), 200, dtype=np.uint8)
    cv2.rectangle(img3, (124, 110), (132, 240), (110, 110, 110), -1)
    cv2.circle(img3, (128, 70), 45, (40, 40, 210), -1)
    cv2.putText(img3, "STOP", (95, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    p3 = clean_dir / "stop_sample.png"
    cv2.imwrite(str(p3), img3)
    ground_truth_boxes["stop_sample.png"] = [{"box": [83, 25, 173, 115], "label": "traffic light"}]
    ground_truth_labels["stop_sample.png"] = "traffic light"

    # Image 4: Complex multi-object scene
    img4 = np.full((256, 256, 3), 210, dtype=np.uint8)
    cv2.rectangle(img4, (20, 20), (110, 220), (150, 140, 130), -1)
    cv2.rectangle(img4, (130, 140), (230, 210), (50, 120, 60), -1)
    cv2.circle(img4, (150, 210), 16, (20, 20, 20), -1)
    cv2.circle(img4, (210, 210), 16, (20, 20, 20), -1)
    p4 = clean_dir / "scene_sample.png"
    cv2.imwrite(str(p4), img4)
    ground_truth_boxes["scene_sample.png"] = [{"box": [130, 140, 230, 226], "label": "car"}]
    ground_truth_labels["scene_sample.png"] = "car"

    manifest_path = manifest_dir / "eval.jsonl"
    count = build_manifest(
        clean_dir=clean_dir,
        output_path=manifest_path,
        ground_truth_labels=ground_truth_labels,
        ground_truth_boxes=ground_truth_boxes,
    )
    print(f"Generated sample images in {clean_dir} and {count} manifest entries in {manifest_path}")


if __name__ == "__main__":
    generate_sample_images()
