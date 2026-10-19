"""Unit tests for DetectionService with dual-model and temporal smoothing."""

import unittest
from pathlib import Path
import numpy as np

from src.config import (
    MODEL_CU_PATH,
    MODEL_MOI_PATH,
    STATE_FIRE,
    STATE_NORMAL,
    TEN_KHOI,
    TEN_LUA,
)
from src.services.detection_service import DetectionService, FrameAnalysisResult


class TestDetectionService(unittest.TestCase):
    """Test DetectionService dual-model loading, temporal smoothing, and class mapping."""

    def test_missing_both_models_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            DetectionService(
                model_path=Path("models/non_existent_1.pt"),
                model_moi_path=Path("models/non_existent_2.pt")
            )

    def test_load_models_success(self):
        if not MODEL_MOI_PATH.exists() and not MODEL_CU_PATH.exists():
            self.skipTest("No models found.")

        service = DetectionService()
        self.assertTrue(service.model_moi is not None or service.model_cu is not None)
        if service.model_moi:
            self.assertIn("fire", [name.lower() for name in service.names_moi.values()])

    def test_synonyms_constants(self):
        self.assertIn("fire", TEN_LUA)
        self.assertIn("flame", TEN_LUA)
        self.assertIn("smoke", TEN_KHOI)

    def test_detect_blank_frame(self):
        if not MODEL_MOI_PATH.exists():
            self.skipTest("fire_yolo.pt does not exist.")

        service = DetectionService()
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = service.detect_frame(blank_frame, frame_index=1, timestamp_sec=0.04)

        self.assertIsInstance(result, FrameAnalysisResult)
        self.assertEqual(result.frame_index, 1)
        self.assertEqual(result.alert_state, STATE_NORMAL)
        self.assertFalse(result.is_alert_confirmed)

    def test_temporal_filter_smoothing(self):
        service = DetectionService()
        service.fire_history.clear()

        # Khởi tạo 10 lần gần nhất: 6 lần không có lửa (cũ hơn), 4 lần có lửa (mới hơn)
        service.fire_history.extend([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

        fire_count = sum(service.fire_history)
        self.assertEqual(fire_count, 4)
        self.assertFalse(fire_count >= 5)

        # Lần tiếp theo có lửa -> đẩy 0 cũ ra ngoài, tổng số lửa tăng lên 5/10 -> kích hoạt
        service.fire_history.append(1)
        fire_count = sum(service.fire_history)
        self.assertEqual(fire_count, 5)
        self.assertTrue(fire_count >= 5)


if __name__ == "__main__":
    unittest.main()
