"""Unit tests cho Bộ điều phối suy luận thích ứng (Adaptive Model Scheduler) và Dynamic Stride."""

import unittest
import shutil
import json
from pathlib import Path
import numpy as np

from src.services.detection_service import DetectionService, Detection
from src.services.static_light_filter import LocNguonSangTinh, TrackedLightBox
from src.services.spatio_temporal_tracker import TrackedObject
from src.config import PROJECT_ROOT, HARD_NEGATIVES_DIR


class TestAdaptiveScheduler(unittest.TestCase):
    """Kiểm tra Dynamic Stride, Adaptive Cascade và Hard Negative Curation."""

    def setUp(self):
        self.service = DetectionService(
            enable_decision_logger=False,
            enable_adaptive_cascade=True
        )

    def test_dynamic_stride_tightens_on_hazard(self):
        """Khi bình thường stride = recommended (2 trên CPU); khi có nguy cơ siết ngay về 1."""
        # 1. Trạng thái bình thường
        self.service.reset_tracking()
        self.assertEqual(self.service.get_dynamic_stride(), self.service.hw_mgr.profile.recommended_stride)

        # 2. Tạo đối tượng nghi vấn trong Tracker
        track = TrackedObject(
            track_id=99,
            class_name="fire",
            bbox=(100, 100, 200, 200),
            confidence=0.55,
            first_seen_time=1.0,
            last_seen_time=1.5,
            last_frame_idx=10,
            state="SUSPECT"
        )
        self.service.threat_analyzer.tracker.tracks[99] = track

        # Stride phải lập tức siết về 1
        self.assertTrue(self.service.has_active_hazard())
        self.assertEqual(self.service.get_dynamic_stride(), 1)

        # 3. Khi đối tượng lên ALARM
        track.state = "ALARM"
        self.assertEqual(self.service.get_dynamic_stride(), 1)

        # 4. Khi xóa track
        self.service.reset_tracking()
        self.assertFalse(self.service.has_active_hazard())
        self.assertEqual(self.service.get_dynamic_stride(), self.service.hw_mgr.profile.recommended_stride)

    def test_adaptive_cascade_on_blank_frame(self):
        """Khung hình trống không có đối tượng: Adaptive Cascade bỏ qua model PyTorch nặng."""
        blank_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        self.service.reset_tracking()

        # Frame 2 (không phải frame đầu, không phải frame chu kỳ 25)
        result = self.service.detect_frame(blank_frame, frame_index=2, timestamp_sec=0.08)

        # best_pt phải được bỏ qua (thời gian = 0.0) nếu chạy adaptive cascade
        if self.service.net_onnx is not None:
            self.assertEqual(result.timing_ms.get("best_pt", 0.0), 0.0)
            self.assertEqual(len(result.detections), 0)

    def test_periodic_scan_forces_full_ensemble(self):
        """Cứ mỗi 25 frame (1 giây), hệ thống bắt buộc quét Full Ensemble để tránh điểm mù."""
        blank_frame = np.zeros((320, 320, 3), dtype=np.uint8)
        self.service.reset_tracking()

        # Frame 25 là chu kỳ quét định kỳ
        result = self.service.detect_frame(blank_frame, frame_index=25, timestamp_sec=1.0)
        # Tại frame chu kỳ quét, best_pt phải được gọi (thời gian > 0.0)
        self.assertGreater(result.timing_ms.get("best_pt", 0.0), 0.0)

    def test_hard_negative_curation_metadata(self):
        """Lưu hard negative tạo ra bộ 3: ảnh full, ảnh crop, và metadata JSON chi tiết."""
        test_dir = PROJECT_ROOT / "hard_negatives_test"
        if test_dir.exists():
            shutil.rmtree(test_dir)
        test_dir.mkdir(parents=True, exist_ok=True)

        filter_svc = LocNguonSangTinh()
        # Mock cấu hình lưu
        import src.services.static_light_filter as slf_mod
        orig_save = slf_mod.SAVE_HARD_NEGATIVES
        orig_dir = slf_mod.HARD_NEGATIVES_DIR
        try:
            slf_mod.SAVE_HARD_NEGATIVES = True
            slf_mod.HARD_NEGATIVES_DIR = test_dir

            frame = np.full((200, 200, 3), 150, dtype=np.uint8)
            track = TrackedLightBox(
                track_id=5,
                bbox=(50, 50, 100, 100),
                class_name="fire",
                confidence=0.45,
                last_frame_idx=10,
                first_seen_time=1.0,
                last_seen_time=2.0
            )
            track.last_drift = 0.02
            track.last_motion = 1.5
            track.flicker_std = 0.8
            track.core_saturation_ratio = 0.25

            filter_svc._maybe_save_hard_negative(
                frame=frame,
                bbox=(50, 50, 100, 100),
                track=track,
                reason="test_static_light"
            )

            # Kiểm tra các file được tạo
            full_files = list(test_dir.glob("*_full.jpg"))
            crop_files = list(test_dir.glob("*_crop.jpg"))
            meta_files = list(test_dir.glob("*_meta.json"))

            self.assertEqual(len(full_files), 1)
            self.assertEqual(len(crop_files), 1)
            self.assertEqual(len(meta_files), 1)

            # Đọc và xác thực metadata JSON
            with open(meta_files[0], "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["track_id"], 5)
            self.assertEqual(meta["reason"], "test_static_light")
            self.assertEqual(meta["bbox"], [50, 50, 100, 100])
            self.assertEqual(meta["core_sat_ratio"], 0.25)

        finally:
            slf_mod.SAVE_HARD_NEGATIVES = orig_save
            slf_mod.HARD_NEGATIVES_DIR = orig_dir
            if test_dir.exists():
                shutil.rmtree(test_dir)


if __name__ == "__main__":
    unittest.main()
