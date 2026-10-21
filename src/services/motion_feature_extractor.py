"""Bộ trích xuất đặc trưng động học chuỗi thời gian (Motion Feature Extractor).
Đo đạc các đặc tính vật lý của từng Track:
1. Vận tốc trôi chuẩn hóa (Normalized Drift Velocity): Phân biệt vật thể tĩnh, xe chạy qua, và lửa/khói.
2. Vector mở rộng hướng lên (Upward Smoke Expansion): Kiểm tra khói bốc lên cao và tỏa rộng.
3. Độ dao động quầng viền (Boundary Jitter): Phân biệt ngọn lửa bập bùng với đèn tĩnh / đèn 50Hz aliasing.
4. Phát hiện chuyển động tịnh tiến của phương tiện giao thông (Vehicle Linear Motion).
"""

from typing import Dict, List, Tuple, Optional
import math
import cv2
import numpy as np

from src.services.spatio_temporal_tracker import TrackedObject
from src.utils.optical_features import extract_optical_metrics


class MotionFeatureExtractor:
    """Trích xuất các chỉ số động học từ Track và khung hình raw."""

    def __init__(self, enable_optical_flow: bool = False) -> None:
        self.enable_optical_flow = enable_optical_flow

    @staticmethod
    def compute_kinematic_features(track: TrackedObject) -> Dict[str, float]:
        """
        Tính toán các chỉ số động học dựa trên chuỗi bounding box và timestamps.
        """
        feats = {
            "v_drift": 0.0,
            "area_growth_ratio": 1.0,
            "upward_ratio": 0.0,
            "aspect_ratio": 1.0,
            "track_age_sec": 0.0,
            "linear_motion_score": 0.0
        }
        
        timestamps = track.history_timestamps
        bboxes = track.history_bboxes
        
        if len(timestamps) < 2 or len(bboxes) < 2:
            return feats

        dt = float(timestamps[-1] - timestamps[0])
        track_age = float(timestamps[-1] - track.first_seen_time)
        feats["track_age_sec"] = track_age

        if dt <= 1e-4:
            return feats

        # 1. Tọa độ tâm và kích thước
        b_first = bboxes[0]
        b_curr = bboxes[-1]
        
        c_first_x = (b_first[0] + b_first[2]) / 2.0
        c_first_y = (b_first[1] + b_first[3]) / 2.0
        c_curr_x = (b_curr[0] + b_curr[2]) / 2.0
        c_curr_y = (b_curr[1] + b_curr[3]) / 2.0

        w_curr = max(1.0, float(b_curr[2] - b_curr[0]))
        h_curr = max(1.0, float(b_curr[3] - b_curr[1]))
        area_curr = w_curr * h_curr
        area_first = max(1.0, float((b_first[2] - b_first[0]) * (b_first[3] - b_first[1])))
        
        feats["aspect_ratio"] = float(h_curr / w_curr)
        feats["area_growth_ratio"] = float(area_curr / area_first)

        # 2. Vận tốc trôi chuẩn hóa (Normalized Drift Velocity theo body-length/sec)
        disp = math.hypot(c_curr_x - c_first_x, c_curr_y - c_first_y)
        char_len = math.sqrt(area_curr)
        v_drift = disp / (max(1.0, char_len) * dt)
        feats["v_drift"] = float(v_drift)

        # 3. Vector mở rộng hướng lên (Upward Smoke Expansion)
        # Khói: đỉnh trên dâng cao (y1_curr < y1_first) và độ rộng tăng
        delta_top = float(b_first[1] - b_curr[1])  # Dương nếu đỉnh trên dâng cao
        delta_w = float(w_curr - (b_first[2] - b_first[0]))
        if delta_top > 0 and delta_w >= 0:
            feats["upward_ratio"] = float(delta_top / max(1.0, h_curr))

        # 4. Kiểm tra độ thẳng hàng chuyển động (Linear Motion Score của xe cộ)
        if len(bboxes) >= 4:
            dxs = [(bboxes[i][0] + bboxes[i][2])/2.0 - (bboxes[i-1][0] + bboxes[i-1][2])/2.0 for i in range(1, len(bboxes))]
            dys = [(bboxes[i][1] + bboxes[i][3])/2.0 - (bboxes[i-1][1] + bboxes[i-1][3])/2.0 for i in range(1, len(bboxes))]
            signs_x = [1 if d > 1.0 else (-1 if d < -1.0 else 0) for d in dxs]
            signs_y = [1 if d > 1.0 else (-1 if d < -1.0 else 0) for d in dys]
            consistency_x = abs(sum(signs_x)) / max(1, len(signs_x))
            consistency_y = abs(sum(signs_y)) / max(1, len(signs_y))
            feats["linear_motion_score"] = float(max(consistency_x, consistency_y))

        return feats

    @staticmethod
    def extract_boundary_jitter(raw_gray: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Đo độ dao động quầng viền (Boundary Jitter) trên vùng crop:
        - Cắt crop quanh bounding box.
        - Tìm contour sáng Otsu.
        - Đo tỷ lệ giữa chu vi thực tế và chu vi lồi (Perimeter Roughness):
          Lửa thật: Biên dạng hữu cơ, răng cưa uốn lượn -> Roughness >= 1.25.
          Đèn / Biển báo / LED 50Hz: Cạnh thẳng, hình học cố định -> Roughness < 1.15.
        """
        return extract_optical_metrics(raw_gray, bbox).roughness

    @staticmethod
    def is_vehicle_motion(kinematics: Dict[str, float]) -> bool:
        """
        Nhận diện chuyển động tịnh tiến có hướng của phương tiện giao thông (xe máy / ô tô):
        - linear_motion_score >= 0.80 (chuyển động thẳng hàng nhất quán qua 4+ frame).
        - v_drift >= 0.35 body-lengths/sec (đang di chuyển rõ rệt, không đứng yên bập bùng).
        - area_growth_ratio nằm trong khoảng [0.65, 1.45] (kích thước vật rắn không nở rộng như khói/lửa).
        """
        linear_score = kinematics.get("linear_motion_score", 0.0)
        v_drift = kinematics.get("v_drift", 0.0)
        growth = kinematics.get("area_growth_ratio", 1.0)
        return bool(linear_score >= 0.80 and v_drift >= 0.35 and 0.65 <= growth <= 1.45)

    @staticmethod
    def is_reflective_vest_or_rigid_worker(kinematics: Dict[str, float], roughness: float) -> bool:
        """
        Nhận diện áo phản quang công nhân hoặc vật thể cứng màu vàng/cam di chuyển:
        - aspect_ratio trong khoảng [1.5, 3.2] (tỷ lệ khung hình đứng của người).
        - v_drift >= 0.15 (có bước chuyển động).
        - area_growth_ratio ổn định [0.80, 1.25].
        - roughness < 1.20 (quầng viền thẳng/phẳng của vải áo hoặc vật thể, không phải viền răng cưa ngọn lửa).
        """
        ar = kinematics.get("aspect_ratio", 1.0)
        v_drift = kinematics.get("v_drift", 0.0)
        growth = kinematics.get("area_growth_ratio", 1.0)
        return bool(1.5 <= ar <= 3.2 and v_drift >= 0.15 and 0.80 <= growth <= 1.25 and roughness < 1.20)
