"""Unit tests for DetectionService."""

import unittest
from pathlib import Path
import numpy as np

from src.config import DEFAULT_MODEL_PATH, STATE_NORMAL
from src.services.detection_service import DetectionService, FrameAnalysisResult


class TestDetectionService(unittest.TestCase):
    """Test DetectionService initialization, class mapping, and detection."""

    def test_missing_model_raises_filenotfound(self):
        non_existent = Path("models/does_not_exist.pt")
        with self.assertRaises(FileNotFoundError):
            DetectionService(model_path=non_existent)

    def test_load_real_model(self):
        if not DEFAULT_MODEL_PATH.exists():
            self.skipTest("Model best.pt does not exist, skipping live test.")

        service = DetectionService(model_path=DEFAULT_MODEL_PATH)
        self.assertIsNotNone(service.model)
        self.assertIn(0, service.class_names)
        self.assertIn(1, service.class_names)
        self.assertTrue(len(service.fire_class_ids) > 0)
        self.assertTrue(len(service.smoke_class_ids) > 0)

    def test_detect_blank_frame(self):
        if not DEFAULT_MODEL_PATH.exists():
            self.skipTest("Model best.pt does not exist, skipping live test.")

        service = DetectionService(model_path=DEFAULT_MODEL_PATH)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = service.detect_frame(blank_frame, frame_index=1, timestamp_sec=0.04)

        self.assertIsInstance(result, FrameAnalysisResult)
        self.assertEqual(result.frame_index, 1)
        self.assertEqual(result.timestamp_sec, 0.04)
        self.assertEqual(result.alert_state, STATE_NORMAL)
        self.assertEqual(len(result.detections), 0)
        self.assertGreaterEqual(result.inference_time_ms, 0)

    def test_set_conf_threshold(self):
        if not DEFAULT_MODEL_PATH.exists():
            self.skipTest("Model best.pt does not exist, skipping live test.")

        service = DetectionService(model_path=DEFAULT_MODEL_PATH)
        service.set_conf_threshold(0.55)
        self.assertAlmostEqual(service.conf_threshold, 0.55)


if __name__ == "__main__":
    unittest.main()
