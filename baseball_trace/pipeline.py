from __future__ import annotations

import argparse
from pathlib import Path

from .detector import process_images
from .synthetic_sequence import generate_sequence


def run_pipeline(
    samples_dir: Path,
    labels_dir: Path,
    results_dir: Path,
    *,
    frame_count: int = 10,
    width: int = 1280,
    height: int = 720,
    model_path: str | None = None,
    confidence: float = 0.25,
) -> dict[str, object]:
    specs = generate_sequence(samples_dir, labels_dir, frame_count=frame_count, width=width, height=height)
    detection_summary = process_images(results_dir=results_dir, samples_dir=samples_dir, model_path=model_path, confidence=confidence)
    return {
        "sample_count": len(specs),
        "label_count": len(list(labels_dir.glob("frame_*.txt"))),
        **detection_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and detect a synthetic baseball pitch sequence.")
    parser.add_argument("--samples-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--model", type=str, default=None, help="Optional YOLO .pt model path.")
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    summary = run_pipeline(
        samples_dir=args.samples_dir,
        labels_dir=args.labels_dir,
        results_dir=args.results_dir,
        frame_count=args.frames,
        width=args.width,
        height=args.height,
        model_path=args.model,
        confidence=args.confidence,
    )

    print(f"Generated samples: {summary['sample_count']} -> {args.samples_dir}")
    print(f"Generated YOLO labels: {summary['label_count']} -> {args.labels_dir}")
    print(f"Detected images: {summary['result_count']} -> {args.results_dir}")
    print(f"Detection backend: {summary['backend']}")
    print(f"Detection CSV: {args.results_dir / 'detections.csv'}")


if __name__ == "__main__":
    main()
