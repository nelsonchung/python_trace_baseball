from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HAS_CV2 = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_CV2, "OpenCV is required for the image pipeline")
class PipelineTest(unittest.TestCase):
    def test_pipeline_generates_ten_detected_frames(self) -> None:
        from baseball_trace.pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = run_pipeline(
                samples_dir=root / "samples",
                labels_dir=root / "labels",
                results_dir=root / "results",
                frame_count=10,
                width=960,
                height=540,
            )

            self.assertEqual(summary["sample_count"], 10)
            self.assertEqual(summary["label_count"], 10)
            self.assertEqual(summary["result_count"], 10)
            self.assertEqual(len(list((root / "samples").glob("frame_*.png"))), 10)
            self.assertEqual(len(list((root / "results").glob("frame_*_detected.png"))), 10)
            self.assertTrue((root / "results" / "detections.csv").exists())
            self.assertTrue((root / "results" / "trajectory.json").exists())

            detected_rows = [row for row in summary["detections"] if row["class_name"] == "baseball"]
            self.assertEqual(len(detected_rows), 10)

            metadata = json.loads((root / "samples" / "metadata.json").read_text(encoding="utf-8"))
            expected_centers = [tuple(frame["center"]) for frame in metadata["frames"]]
            detected_centers = [(int(row["center_x"]), int(row["center_y"])) for row in detected_rows]
            for detected, expected in zip(detected_centers, expected_centers, strict=True):
                self.assertLessEqual(abs(detected[0] - expected[0]), 6)
                self.assertLessEqual(abs(detected[1] - expected[1]), 6)


if __name__ == "__main__":
    unittest.main()
