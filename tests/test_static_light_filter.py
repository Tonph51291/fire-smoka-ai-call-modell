"""Unit tests for LocNguonSangTinh (Static Light Source Filter)."""

import unittest
import numpy as np

from src.services.detection_service import Detection
from src.services.static_light_filter import LocNguonSangTinh, compute_iou_boxes


class TestStaticLightFilter(unittest.TestCase):
    """Test static light source identification, spatial drift, and temporal motion calculation."""

    def test_compute_iou(self):
        box1 = (0, 0, 10, 10)
        box2 = (0, 0, 10, 10)
        self.assertAlmostEqual(compute_iou_boxes(box1, box2), 1.0)

        box3 = (20, 20, 30, 30)
        self.assertAlmostEqual(compute_iou_boxes(box1, box3), 0.0)

        box4 = (5, 0, 15, 10)  # 50% overlap
        self.assertAlmostEqual(compute_iou_boxes(box1, box4), 50.0 / 150.0)

    def test_static_streetlight_filtered_out(self):
        """Đèn đường tĩnh (vị trí cố định, độ sáng ổn định) phải bị phát hiện và loại bỏ."""
        filter_svc = LocNguonSangTinh(
            cua_so_frame=30,
            nguong_dich=4.0,
            nguong_dong=5.0,
            min_frames_xac_nhan=5
        )

        base_frame = np.zeros((120, 120, 3), dtype=np.uint8)
        # Giả lập đèn đường cố định tại (40, 40, 60, 60)
        base_frame[40:60, 40:60] = 230

        was_filtered = False
        for f in range(12):
            # Nhiễu camera nhẹ +- 1 đơn vị
            noise = np.random.randint(-1, 2, (120, 120, 3)).astype(np.int16)
            noisy_frame = np.clip(base_frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            det = Detection(class_id=0, class_name="fire", confidence=0.45, bbox=(40, 40, 60, 60))

            valid, static_dets = filter_svc.filter_detections(noisy_frame, [det], frame_index=f)
            if len(static_dets) > 0:
                was_filtered = True
                break

        self.assertTrue(was_filtered, "Đèn đường tĩnh phải bị bộ lọc loại bỏ sau đủ số frame!")

    def test_real_flickering_fire_preserved(self):
        """Lửa thật bập bùng (nội dung biến đổi mạnh > nguong_dong) KHÔNG được bị loại bỏ."""
        filter_svc = LocNguonSangTinh(
            cua_so_frame=30,
            nguong_dich=4.0,
            nguong_dong=5.0,
            min_frames_xac_nhan=5
        )

        was_filtered = False
        for f in range(12):
            fire_frame = np.zeros((120, 120, 3), dtype=np.uint8)
            # Lửa bập bùng thay đổi ngẫu nhiên từ 50 đến 255
            fire_frame[40:60, 40:60] = np.random.randint(50, 255, (20, 20, 3), dtype=np.uint8)
            # Tọa độ cũng hơi xê dịch 1-2 pixel
            shift = f % 3
            det = Detection(class_id=0, class_name="fire", confidence=0.85, bbox=(40 + shift, 40, 60 + shift, 60))

            valid, static_dets = filter_svc.filter_detections(fire_frame, [det], frame_index=f)
            if len(static_dets) > 0:
                was_filtered = True

        self.assertFalse(was_filtered, "Lửa thật bập bùng không được bị bộ lọc coi là đèn tĩnh!")

    def test_smoke_not_affected_by_static_light_filter(self):
        """Khói (smoke) không phải đèn nên không bị lọc bởi bộ lọc đèn tĩnh."""
        filter_svc = LocNguonSangTinh(min_frames_xac_nhan=3)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        det = Detection(class_id=1, class_name="smoke", confidence=0.70, bbox=(20, 20, 50, 50))

        for f in range(5):
            valid, static_dets = filter_svc.filter_detections(frame, [det], frame_index=f)
            self.assertEqual(len(valid), 1)
            self.assertEqual(len(static_dets), 0)


if __name__ == "__main__":
    unittest.main()
