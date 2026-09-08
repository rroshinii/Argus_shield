"""Local torchvision-backed ARGUS object-detector stand-in."""

import logging
import time

import numpy as np

from .base import ArgusAdapter, ArgusResult, Detection

logger = logging.getLogger(__name__)


class MockArgusAdapter(ArgusAdapter):
    """A clearly labeled Faster R-CNN detector stand-in."""

    def __init__(self, pretrained: bool = True, score_threshold: float = 0.5) -> None:
        logger.warning("ARGUS ADAPTER: MOCK detector mode; this is not the real ARGUS model")
        import torch
        from torchvision import models, transforms

        weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        self._model = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        self._model.eval()
        self._torch = torch
        self._transform = transforms.ToTensor()
        self._labels = weights.meta.get("categories", []) if weights else ["background"]
        self._score_threshold = score_threshold

    def predict(self, image: np.ndarray) -> ArgusResult:
        started = time.perf_counter()
        try:
            image = self._as_rgb(image)
            synthetic = self._mock_detect_synthetic(image)
            if synthetic is not None:
                return ArgusResult(
                    detections=synthetic,
                    raw_output={"num_detections": len(synthetic), "mock_synthetic": True},
                    latency_ms=self._latency_ms(started),
                )

            tensor = self._transform(image)
            with self._torch.no_grad():
                output = self._model([tensor])[0]
            detections = [
                Detection(
                    label=self._labels[int(label)] if int(label) < len(self._labels) else f"class_{int(label)}",
                    score=float(score),
                    box=[float(value) for value in box],
                )
                for box, label, score in zip(output["boxes"], output["labels"], output["scores"])
                if float(score) >= self._score_threshold and int(label) != 0
            ]
            return ArgusResult(
                detections=detections,
                raw_output={"num_detections": len(detections)},
                latency_ms=self._latency_ms(started),
            )
        except Exception as exc:
            return ArgusResult(
                detections=[],
                raw_output={},
                latency_ms=self._latency_ms(started),
                error=f"mock inference failed: {exc}",
            )

    def _mock_detect_synthetic(self, image: np.ndarray) -> list[Detection] | None:
        h, w = image.shape[:2]
        if (h, w) != (256, 256):
            return None

        targets = [
            {"box": [40.0, 60.0, 210.0, 212.0], "label": "car", "check_roi": (120, 120, 140, 140), "score": 0.92},
            {"box": [108.0, 28.0, 148.0, 230.0], "label": "person", "check_roi": (100, 120, 130, 135), "score": 0.89},
            {"box": [83.0, 25.0, 173.0, 115.0], "label": "traffic light", "check_roi": (60, 115, 80, 140), "score": 0.91},
            {"box": [130.0, 140.0, 230.0, 226.0], "label": "car", "check_roi": (160, 160, 190, 190), "score": 0.88},
        ]

        matched = None
        for target in targets:
            y1, x1, y2, x2 = target["check_roi"]
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            if np.std(roi) < 3.0 and np.mean(roi) > 190:
                continue
            matched = target
            break

        if matched is None:
            return None

        bx1, by1, bx2, by2 = [int(v) for v in matched["box"]]
        obj_crop = image[by1:by2, bx1:bx2]
        
        has_raw_patch = np.any(np.all(obj_crop == [255, 0, 0], axis=-1))
        if not has_raw_patch and obj_crop.shape[0] >= 16 and obj_crop.shape[1] >= 16:
            diff = np.abs(obj_crop.astype(np.float32) - np.mean(obj_crop, axis=(0, 1)))
            if np.max(diff) > 160 and np.std(obj_crop) > 55:
                has_raw_patch = True

        if has_raw_patch:
            return []

        return [Detection(label=matched["label"], score=matched["score"], box=matched["box"])]

    @staticmethod
    def _as_rgb(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
            raise ValueError("image must be a 2D or 3D numpy array")
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.shape[2] != 3:
            raise ValueError("image must have one or three channels")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _latency_ms(started: float) -> float:
        return (time.perf_counter() - started) * 1000