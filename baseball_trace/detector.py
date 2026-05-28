from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


BALL_CLASS_NAMES = ("baseball", "sports ball", "ball")


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)


class BaseballDetector:
    """YOLO-first baseball detector with an OpenCV fallback."""

    def __init__(self, model_path: str | None = None, confidence: float = 0.25) -> None:
        self.confidence = confidence
        self.backend = "opencv"
        self.model = None
        if model_path:
            self.model = self._load_yolo(model_path)
            if self.model is not None:
                self.backend = "yolo"

    def detect(self, image: np.ndarray) -> list[Detection]:
        if self.backend == "yolo" and self.model is not None:
            detections = self._detect_with_yolo(image)
            if detections:
                return detections
        return self._detect_with_opencv(image)

    def _load_yolo(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            return None
        return YOLO(model_path)

    def _detect_with_yolo(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(image, conf=self.confidence, verbose=False)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = str(names.get(cls_id, cls_id)).lower()
                if not any(accepted in class_name for accepted in BALL_CLASS_NAMES):
                    continue
                x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                detections.append(Detection(class_name=class_name, confidence=confidence, bbox=(x1, y1, x2, y2)))
        return sorted(detections, key=lambda item: item.confidence, reverse=True)

    def _detect_with_opencv(self, image: np.ndarray) -> list[Detection]:
        bgr = image
        bright = cv2.inRange(bgr, np.array([160, 160, 150], dtype=np.uint8), np.array([255, 255, 255], dtype=np.uint8))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_saturation = cv2.inRange(hsv, np.array([0, 0, 170], dtype=np.uint8), np.array([179, 80, 255], dtype=np.uint8))
        mask = cv2.bitwise_and(bright, low_saturation)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Detection] = []
        image_area = image.shape[0] * image.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 80 or area > image_area * 0.006:
                continue
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / max(h, 1)
            if circularity < 0.55 or not 0.65 <= aspect_ratio <= 1.35:
                continue
            radius = max(w, h) / 2
            if not 5 <= radius <= 34:
                continue

            confidence = float(min(0.99, 0.42 + circularity * 0.38 + min(radius / 34, 1) * 0.19))
            padding = max(4, int(radius * 0.18))
            bbox = (
                max(0, x - padding),
                max(0, y - padding),
                min(image.shape[1] - 1, x + w + padding),
                min(image.shape[0] - 1, y + h + padding),
            )
            candidates.append(Detection(class_name="baseball", confidence=confidence, bbox=bbox))

        if candidates:
            return sorted(candidates, key=lambda item: item.confidence, reverse=True)[:1]
        return self._detect_red_seams(image)

    def _detect_red_seams(self, image: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        red_low = cv2.inRange(hsv, np.array([0, 110, 130], dtype=np.uint8), np.array([12, 255, 255], dtype=np.uint8))
        red_high = cv2.inRange(hsv, np.array([168, 110, 130], dtype=np.uint8), np.array([179, 255, 255], dtype=np.uint8))
        red_mask = cv2.bitwise_or(red_low, red_high)

        kernel = np.ones((7, 7), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 40 or area > image.shape[0] * image.shape[1] * 0.004:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 5 or h < 5:
                continue
            if max(w, h) > 58:
                continue
            center_x = x + w // 2
            center_y = y + h // 2
            radius = max(7, int(max(w, h) * 0.72))
            bbox = (
                max(0, center_x - radius),
                max(0, center_y - radius),
                min(image.shape[1] - 1, center_x + radius),
                min(image.shape[0] - 1, center_y + radius),
            )
            confidence = float(min(0.98, 0.76 + min(max(w, h) / 70, 1) * 0.18))
            candidates.append(Detection(class_name="baseball", confidence=confidence, bbox=bbox))

        return sorted(candidates, key=lambda item: item.confidence, reverse=True)[:1]


def annotate_image(
    image: np.ndarray,
    detections: Iterable[Detection],
    *,
    trajectory: Iterable[tuple[int, int]] = (),
) -> np.ndarray:
    output = image.copy()
    points = list(trajectory)
    if len(points) >= 2:
        cv2.polylines(output, [np.array(points, dtype=np.int32)], False, (58, 229, 255), 3, cv2.LINE_AA)
        for point in points:
            cv2.circle(output, point, 4, (58, 229, 255), -1, cv2.LINE_AA)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(output, (x1, y1), (x2, y2), (48, 236, 94), 3, cv2.LINE_AA)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        label_y = max(24, y1 - 9)
        cv2.rectangle(
            output,
            (x1, label_y - label_size[1] - 10),
            (x1 + label_size[0] + 12, label_y + 5),
            (28, 35, 32),
            -1,
            cv2.LINE_AA,
        )
        cv2.putText(output, label, (x1 + 6, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (232, 255, 232), 2, cv2.LINE_AA)
        cv2.circle(output, detection.center, 5, (43, 226, 255), -1, cv2.LINE_AA)
    return output


def process_images(
    samples_dir: Path,
    results_dir: Path,
    *,
    model_path: str | None = None,
    confidence: float = 0.25,
) -> dict[str, object]:
    results_dir.mkdir(parents=True, exist_ok=True)
    detector = BaseballDetector(model_path=model_path, confidence=confidence)

    rows: list[dict[str, object]] = []
    trajectory: list[tuple[int, int]] = []
    result_paths: list[Path] = []
    image_paths = sorted(samples_dir.glob("frame_*.png"))

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        detections = detector.detect(image)
        if detections:
            trajectory.append(detections[0].center)
        annotated = annotate_image(image, detections, trajectory=trajectory)
        result_path = results_dir / f"{image_path.stem}_detected.png"
        cv2.imwrite(str(result_path), annotated)
        result_paths.append(result_path)

        if detections:
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                rows.append(
                    {
                        "image": image_path.name,
                        "backend": detector.backend,
                        "class_name": detection.class_name,
                        "confidence": f"{detection.confidence:.4f}",
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": detection.center[0],
                        "center_y": detection.center[1],
                    }
                )
        else:
            rows.append(
                {
                    "image": image_path.name,
                    "backend": detector.backend,
                    "class_name": "",
                    "confidence": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "center_x": "",
                    "center_y": "",
                }
            )

    _write_detection_csv(results_dir / "detections.csv", rows)
    _write_trajectory_json(results_dir / "trajectory.json", trajectory)

    return {
        "backend": detector.backend,
        "image_count": len(image_paths),
        "result_count": len(result_paths),
        "detections": rows,
        "result_paths": result_paths,
    }


def _write_detection_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["image", "backend", "class_name", "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_trajectory_json(path: Path, trajectory: list[tuple[int, int]]) -> None:
    payload = {"points": [{"x": x, "y": y} for x, y in trajectory]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
