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

    def test_fusion_agreement_and_penalty(self):
        from src.services.detection_service import Detection, fuse_model_detections

        # Box 1 xuất hiện ở cả 2 model (trùng vị trí, cùng class fire)
        d_moi_1 = Detection(class_id=0, class_name="fire", confidence=0.80, bbox=(100, 100, 200, 200))
        d_cu_1 = Detection(class_id=0, class_name="Fire", confidence=0.60, bbox=(105, 105, 205, 205))

        # Box 2 chỉ xuất hiện ở Model Cũ (ví dụ xe đỏ báo giả)
        d_cu_solo = Detection(class_id=0, class_name="Fire", confidence=0.80, bbox=(400, 400, 500, 500))

        fused = fuse_model_detections([d_moi_1], [d_cu_1, d_cu_solo], iou_thresh=0.40, agree_bonus=0.10, single_penalty=0.70)
        self.assertEqual(len(fused), 2)

        # Box đồng thuận
        box_agree = next(b for b in fused if b.n_votes == 2)
        # avg = (0.8*0.8 + 0.6*0.6) / (0.8 + 0.6) = 1.0 / 1.4 ≈ 0.7143 + 0.10 = 0.8143
        self.assertAlmostEqual(box_agree.confidence, 0.8143, places=3)
        self.assertEqual(box_agree.n_votes, 2)

        # Box đơn lẻ: bị phạt conf * 0.70 = 0.80 * 0.70 = 0.56
        box_solo = next(b for b in fused if b.n_votes == 1)
        self.assertAlmostEqual(box_solo.confidence, 0.56, places=2)
        self.assertEqual(box_solo.n_votes, 1)

    def test_night_detection_brightness_vs_daylight(self):
        from src.services.detection_service import is_night_or_ir_frame

        # Khung hình ban ngày sáng nhưng ít màu (ví dụ sương mù/tường bê tông: lum=120, sat=5) -> KHÔNG phải đêm
        day_foggy = np.full((100, 100, 3), 120, dtype=np.uint8)
        self.assertFalse(is_night_or_ir_frame(day_foggy))

        # Khung hình ban đêm tối thực sự (lum=40) -> LÀ đêm
        night_dark = np.full((100, 100, 3), 40, dtype=np.uint8)
        self.assertTrue(is_night_or_ir_frame(night_dark))


if __name__ == "__main__":
    unittest.main()

