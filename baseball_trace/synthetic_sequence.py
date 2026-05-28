from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameSpec:
    index: int
    center: tuple[int, int]
    radius: int
    bbox: tuple[int, int, int, int]


def generate_sequence(
    samples_dir: Path,
    labels_dir: Path,
    *,
    frame_count: int = 10,
    width: int = 1280,
    height: int = 720,
) -> list[FrameSpec]:
    """Generate umpire-view pitch frames and YOLO labels."""
    samples_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    specs = _frame_specs(frame_count, width, height)
    for spec in specs:
        image = _draw_scene(width, height, spec)
        image_path = samples_dir / f"frame_{spec.index:02d}.png"
        cv2.imwrite(str(image_path), image)
        _write_yolo_label(labels_dir / f"frame_{spec.index:02d}.txt", spec.bbox, width, height)

    _write_dataset_yaml(samples_dir, labels_dir)
    _write_metadata(samples_dir / "metadata.json", specs, width, height)
    return specs


def _frame_specs(frame_count: int, width: int, height: int) -> list[FrameSpec]:
    specs: list[FrameSpec] = []
    for idx in range(1, frame_count + 1):
        t = (idx - 1) / max(frame_count - 1, 1)
        x = int(width * 0.56 - width * 0.115 * t + 24 * np.sin(np.pi * t))
        y = int(height * 0.34 + height * 0.43 * (t**1.18))
        radius = int(round(10 + 16 * t))
        bbox = (x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3)
        specs.append(FrameSpec(index=idx, center=(x, y), radius=radius, bbox=bbox))
    return specs


def _draw_scene(width: int, height: int, spec: FrameSpec) -> np.ndarray:
    image = _field_background(width, height)
    _draw_diamond_guides(image)
    _draw_mound_and_pitcher(image, spec.index)
    _draw_catcher_foreground(image)
    _draw_strike_zone(image)
    _draw_ball(image, spec.center, spec.radius, spec.index)
    _draw_frame_badge(image, spec.index)
    return image


def _field_background(width: int, height: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(height * 0.32)
    sky_top = np.array([205, 188, 154], dtype=np.uint8)
    sky_bottom = np.array([235, 220, 186], dtype=np.uint8)
    grass_top = np.array([70, 112, 63], dtype=np.uint8)
    grass_bottom = np.array([40, 78, 48], dtype=np.uint8)

    for y in range(height):
        if y < horizon:
            alpha = y / max(horizon, 1)
            color = (1 - alpha) * sky_top + alpha * sky_bottom
        else:
            alpha = (y - horizon) / max(height - horizon, 1)
            color = (1 - alpha) * grass_top + alpha * grass_bottom
        image[y, :, :] = color

    plate_dirt = np.array(
        [
            (int(width * 0.37), height),
            (int(width * 0.47), int(height * 0.47)),
            (int(width * 0.55), int(height * 0.47)),
            (int(width * 0.66), height),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [plate_dirt], (72, 112, 143))
    cv2.ellipse(
        image,
        (int(width * 0.51), int(height * 0.46)),
        (125, 34),
        0,
        0,
        360,
        (82, 126, 154),
        -1,
        cv2.LINE_AA,
    )
    return image


def _draw_diamond_guides(image: np.ndarray) -> None:
    height, width = image.shape[:2]
    center_x = int(width * 0.51)
    home_y = int(height * 0.88)
    mound_y = int(height * 0.46)

    cv2.line(image, (center_x, home_y), (int(width * 0.42), mound_y), (210, 216, 206), 2, cv2.LINE_AA)
    cv2.line(image, (center_x, home_y), (int(width * 0.60), mound_y), (210, 216, 206), 2, cv2.LINE_AA)

    home_plate = np.array(
        [
            (center_x - 34, home_y - 20),
            (center_x + 34, home_y - 20),
            (center_x + 26, home_y + 18),
            (center_x, home_y + 40),
            (center_x - 26, home_y + 18),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [home_plate], (222, 225, 218))
    cv2.polylines(image, [home_plate], True, (86, 88, 82), 2, cv2.LINE_AA)


def _draw_mound_and_pitcher(image: np.ndarray, frame_index: int) -> None:
    height, width = image.shape[:2]
    x = int(width * 0.51)
    y = int(height * 0.39)

    cv2.ellipse(image, (x, y + 58), (110, 25), 0, 0, 360, (62, 103, 133), -1, cv2.LINE_AA)
    cv2.ellipse(image, (x, y + 55), (72, 13), 0, 0, 360, (92, 142, 168), -1, cv2.LINE_AA)

    stride = min(frame_index - 1, 6)
    cv2.circle(image, (x, y - 35), 17, (58, 48, 41), -1, cv2.LINE_AA)
    cv2.ellipse(image, (x, y - 19), (22, 11), 0, 0, 360, (25, 42, 125), -1, cv2.LINE_AA)
    cv2.line(image, (x, y - 16), (x - 8, y + 30), (62, 86, 142), 12, cv2.LINE_AA)
    cv2.line(image, (x - 4, y - 4), (x + 40, y + 4 + stride), (142, 122, 105), 7, cv2.LINE_AA)
    cv2.line(image, (x - 10, y - 2), (x - 45, y + 16), (142, 122, 105), 7, cv2.LINE_AA)
    cv2.line(image, (x - 5, y + 27), (x - 32, y + 64), (38, 50, 104), 8, cv2.LINE_AA)
    cv2.line(image, (x + 4, y + 28), (x + 38 + stride, y + 62), (38, 50, 104), 8, cv2.LINE_AA)
    cv2.ellipse(image, (x + 46 + stride, y + 10 + stride), (12, 8), 25, 0, 360, (52, 42, 34), -1, cv2.LINE_AA)


def _draw_catcher_foreground(image: np.ndarray) -> None:
    height, width = image.shape[:2]
    cx = int(width * 0.50)
    base_y = int(height * 0.83)

    cv2.ellipse(image, (cx, base_y + 85), (170, 68), 0, 0, 360, (25, 28, 38), -1, cv2.LINE_AA)
    cv2.ellipse(image, (cx, base_y + 18), (86, 94), 0, 0, 360, (32, 43, 92), -1, cv2.LINE_AA)
    cv2.ellipse(image, (cx, base_y - 95), (62, 72), 0, 0, 360, (19, 26, 40), -1, cv2.LINE_AA)

    for offset in (-30, 0, 30):
        cv2.line(image, (cx + offset, base_y - 150), (cx + offset, base_y - 45), (104, 117, 130), 3, cv2.LINE_AA)
    cv2.line(image, (cx - 56, base_y - 95), (cx + 56, base_y - 95), (104, 117, 130), 3, cv2.LINE_AA)
    cv2.line(image, (cx - 50, base_y - 125), (cx + 50, base_y - 125), (104, 117, 130), 3, cv2.LINE_AA)

    cv2.line(image, (cx - 78, base_y - 14), (cx - 145, base_y + 26), (31, 40, 82), 22, cv2.LINE_AA)
    cv2.ellipse(image, (cx - 162, base_y + 28), (48, 36), -18, 0, 360, (44, 36, 28), -1, cv2.LINE_AA)
    cv2.ellipse(image, (cx - 162, base_y + 28), (35, 25), -18, 0, 360, (76, 58, 40), 5, cv2.LINE_AA)

    cv2.line(image, (cx + 78, base_y - 10), (cx + 132, base_y + 46), (31, 40, 82), 22, cv2.LINE_AA)
    cv2.ellipse(image, (cx + 142, base_y + 54), (32, 22), 12, 0, 360, (41, 34, 28), -1, cv2.LINE_AA)


def _draw_strike_zone(image: np.ndarray) -> None:
    height, width = image.shape[:2]
    x1, x2 = int(width * 0.45), int(width * 0.56)
    y1, y2 = int(height * 0.46), int(height * 0.72)
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (210, 218, 220), 2, cv2.LINE_AA)
    cv2.line(overlay, ((x1 + x2) // 2, y1), ((x1 + x2) // 2, y2), (210, 218, 220), 1, cv2.LINE_AA)
    cv2.line(overlay, (x1, (y1 + y2) // 2), (x2, (y1 + y2) // 2), (210, 218, 220), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)


def _draw_ball(image: np.ndarray, center: tuple[int, int], radius: int, frame_index: int) -> None:
    x, y = center
    if frame_index >= 5:
        blur_offset = max(3, radius // 2)
        cv2.ellipse(
            image,
            (x + blur_offset, y - blur_offset // 2),
            (radius + blur_offset, max(4, radius // 3)),
            -24,
            0,
            360,
            (128, 133, 126),
            -1,
            cv2.LINE_AA,
        )

    cv2.circle(image, center, radius + 3, (28, 32, 34), -1, cv2.LINE_AA)
    cv2.circle(image, center, radius, (239, 238, 225), -1, cv2.LINE_AA)
    cv2.ellipse(image, (x - radius // 3, y), (max(2, radius // 3), radius), 14, -65, 65, (42, 38, 188), 2, cv2.LINE_AA)
    cv2.ellipse(image, (x + radius // 3, y), (max(2, radius // 3), radius), 14, 115, 245, (42, 38, 188), 2, cv2.LINE_AA)
    cv2.circle(image, (x - max(2, radius // 3), y - max(2, radius // 3)), max(2, radius // 5), (255, 255, 248), -1, cv2.LINE_AA)


def _draw_frame_badge(image: np.ndarray, frame_index: int) -> None:
    cv2.rectangle(image, (24, 24), (148, 62), (30, 37, 46), -1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"Frame {frame_index:02d}",
        (38, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (226, 231, 225),
        2,
        cv2.LINE_AA,
    )


def _write_yolo_label(path: Path, bbox: tuple[int, int, int, int], width: int, height: int) -> None:
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    path.write_text(f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n", encoding="utf-8")


def _write_dataset_yaml(samples_dir: Path, labels_dir: Path) -> None:
    dataset_path = labels_dir.parent / "dataset.yaml"
    dataset_path.write_text(
        "\n".join(
            [
                f"path: {labels_dir.parent.resolve()}",
                f"train: {samples_dir.name}",
                f"val: {samples_dir.name}",
                "names:",
                "  0: baseball",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_metadata(path: Path, specs: list[FrameSpec], width: int, height: int) -> None:
    payload = {
        "width": width,
        "height": height,
        "frames": [
            {
                "index": spec.index,
                "center": list(spec.center),
                "radius": spec.radius,
                "bbox": list(spec.bbox),
            }
            for spec in specs
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
