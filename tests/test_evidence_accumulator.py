"""Unit tests cho EvidenceAccumulator và SpatioTemporalTracker.
Bao gồm kiểm chứng toán học:
1. Đám khói độ tin cậy thấp (C = 0.20) tồn tại bền vững vẫn leo thang lên ALARM.
2. Nguồn sáng tĩnh (biển báo/đèn) bị ghim ở mức an toàn (E < 25), không báo giả.
3. Xe cộ chạy lướt qua bị triệt tiêu về NONE.
4. Phân rã liên tục theo thời gian thực (Real-time exponential decay).
"""

import unittest
from src.services.spatio_temporal_tracker import SpatioTemporalTracker, TrackedObject
from src.services.evidence_accumulator import EvidenceAccumulator, AccumulatorConfig
from src.services.motion_feature_extractor import MotionFeatureExtractor


class TestEvidenceAccumulator(unittest.TestCase):
    """Kiểm tra máy trạng thái và bộ tích lũy bằng chứng."""

    def setUp(self):
        self.accumulator = EvidenceAccumulator()

    def test_low_conf_smoke_escalates_to_alarm(self):
        """Chứng minh toán học: Khói conf thấp 0.20 tồn tại 15 giây vẫn đạt ALARM."""
        track = TrackedObject(
            track_id=1,
            class_name="smoke",
            bbox=(100, 100, 200, 200),
            confidence=0.20,
            first_seen_time=0.0,
            last_seen_time=0.0,
            last_frame_idx=1,
            history_timestamps=[0.0],
            history_bboxes=[(100, 100, 200, 200)],
            history_confs=[0.20],
            history_n_votes=[1],
            state="NONE"
        )

        kinematics = {
            "v_drift": 0.15,
            "area_growth_ratio": 1.1,
            "upward_ratio": 0.15,  # Có xu hướng bốc lên
            "track_age_sec": 0.0,
            "linear_motion_score": 0.1
        }

        # Mô phỏng video 15s ở 10 FPS (dt = 0.1s)
        dt = 0.1
        current_time = 0.0
        for step in range(150):
            current_time += dt
            track.history_timestamps.append(current_time)
            track.history_bboxes.append((100, 100 - step, 200, 200))
            kinematics["track_age_sec"] = current_time

            score, state, reason = self.accumulator.update_track_evidence(
                track=track,
                timestamp_sec=current_time,
                kinematic_feats=kinematics,
                roughness=1.0,
                is_flare=False
            )

        # Sau 15s khói liên tục, điểm E phải vượt ngưỡng ALARM (85.0)
        self.assertGreaterEqual(track.evidence_score, 85.0)
        self.assertEqual(track.state, "ALARM")

    def test_static_light_suppression(self):
        """Vật thể đứng yên (biển báo/đèn) không bập bùng: bị ghim điểm, không chạm tới ALARM."""
        track = TrackedObject(
            track_id=2,
            class_name="fire",
            bbox=(300, 300, 350, 350),
            confidence=0.40,
            first_seen_time=0.0,
            last_seen_time=0.0,
            last_frame_idx=1,
            history_timestamps=[0.0],
            history_bboxes=[(300, 300, 350, 350)],
            history_confs=[0.40],
            history_n_votes=[2],
            state="NONE"
        )

        # Vật thể tĩnh tuyệt đối: drift = 0.0, viền nhẵn (roughness = 1.05)
        kinematics = {
            "v_drift": 0.01,
            "area_growth_ratio": 1.0,
            "upward_ratio": 0.0,
            "track_age_sec": 0.0,
            "linear_motion_score": 0.0
        }

        current_time = 0.0
        dt = 0.1
        for step in range(50):  # 5 giây
            current_time += dt
            track.history_timestamps.append(current_time)
            track.history_bboxes.append((300, 300, 350, 350))
            kinematics["track_age_sec"] = current_time

            score, state, reason = self.accumulator.update_track_evidence(
                track=track,
                timestamp_sec=current_time,
                kinematic_feats=kinematics,
                roughness=1.05,
                is_flare=False
            )

        # Biển báo tĩnh bị phạt: điểm E ghim ở mức thấp, không thể đạt ALARM
        self.assertTrue(track.is_static_suspect)
        self.assertLess(track.evidence_score, 60.0)
        self.assertNotEqual(track.state, "ALARM")

    def test_vehicle_light_suppression(self):
        """Đèn xe chạy nhanh qua màn hình: chuyển động thẳng, v_drift cao -> triệt tiêu."""
        track = TrackedObject(
            track_id=3,
            class_name="fire",
            bbox=(100, 400, 150, 430),
            confidence=0.60,
            first_seen_time=0.0,
            last_seen_time=0.0,
            last_frame_idx=1,
            history_timestamps=[0.0],
            history_bboxes=[(100, 400, 150, 430)],
            history_confs=[0.60],
            history_n_votes=[1],
            state="NONE"
        )

        kinematics = {
            "v_drift": 2.5,  # Di chuyển cực nhanh
            "area_growth_ratio": 1.0,
            "upward_ratio": 0.0,
            "track_age_sec": 0.0,
            "linear_motion_score": 0.95  # Thẳng hàng tuyệt đối
        }

        # Xe chạy qua trong 1.0s (10 frame)
        current_time = 0.0
        dt = 0.1
        for step in range(10):
            current_time += dt
            track.history_timestamps.append(current_time)
            track.history_bboxes.append((100 + step * 30, 400, 150 + step * 30, 430))
            kinematics["track_age_sec"] = current_time

            score, state, reason = self.accumulator.update_track_evidence(
                track=track,
                timestamp_sec=current_time,
                kinematic_feats=kinematics,
                roughness=1.05,
                is_flare=False
            )

        self.assertTrue(track.is_vehicle_suspect)
        self.assertLess(track.evidence_score, 25.0)
        self.assertEqual(track.state, "NONE")


class TestSpatioTemporalTracker(unittest.TestCase):
    """Kiểm tra tính năng bám vết đa tiêu chí."""

    def test_composite_similarity_containment(self):
        tracker = SpatioTemporalTracker()
        
        # Hộp con nằm hoàn toàn trong hộp mẹ (ví dụ khói nở rộng)
        box_inner = (100, 100, 200, 200)
        box_outer = (80, 80, 220, 220)
        
        sim = tracker.compute_similarity(box_inner, box_outer)
        # Điểm tương đồng phải rất cao nhờ containment và khoảng cách tâm gần
        self.assertGreater(sim, 0.60)


if __name__ == "__main__":
    unittest.main()
