"""Bộ tự động cấu hình theo ngoại cảnh video và camera thời gian thực (SceneAutoConfigurator).
Loại bỏ hoàn toàn các hằng số fix cứng trong config.py.

Nhiệm vụ cốt lõi:
1. Phân tích trắc quang động: Độ sáng, độ bão hòa, độ tương phản và mức nhiễu cảm biến (sensor noise).
2. Đo độ rung lắc nền tổng thể của camera (Camera Shake / Global Motion).
3. Tự động tính toán & nội suy thích ứng các tham số:
   - dynamic_conf_fire & dynamic_conf_smoke
   - dynamic_weights cho Ensemble WBF
   - dynamic_e_alarm & dynamic_tau dựa trên mức nhiễu nền
   - dynamic_drift_threshold bù trừ rung lắc camera cho bộ lọc nguồn sáng tĩnh
4. Cung cấp báo cáo tóm tắt bối cảnh phục vụ hiển thị trực quan trên UI.
"""

from dataclasses import dataclass, field
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.core.entities import EnvironmentType

logger = logging.getLogger(__name__)


class EnvironmentClassifier:
    """Phân loại môi trường ánh sáng từ frame camera có cơ chế Hysteresis."""

    def __init__(self, hysteresis_sec: float = 2.0) -> None:
        self.hysteresis_sec = hysteresis_sec
        self.current_env: str = EnvironmentType.DAY_NORMAL
        self.candidate_env: str = EnvironmentType.DAY_NORMAL
        self.candidate_start_time: float = 0.0
        self.is_first_frame: bool = True

    def reset(self) -> None:
        """Đặt lại bộ phân loại môi trường (khi đổi video hoặc tua vị trí)."""
        self.is_first_frame = True
        self.current_env = EnvironmentType.DAY_NORMAL
        self.candidate_env = EnvironmentType.DAY_NORMAL
        self.candidate_start_time = 0.0

    def extract_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Trích xuất nhanh các chỉ số màu sắc và độ sáng trên ảnh thu nhỏ (160x120)."""
        if frame is None or frame.size == 0:
            return {"val_mean": 128.0, "sat_mean": 50.0, "color_diff": 20.0, "highlight_ratio": 0.0, "contrast": 30.0}

        h_frame = frame.shape[0]
        scene_roi = frame[int(h_frame * 0.08):int(h_frame * 0.70), :] if h_frame >= 100 else frame
        small = cv2.resize(scene_roi, (160, 120), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        sat_mean = float(np.mean(sat))
        val_mean = float(np.mean(val))
        contrast = float(np.std(val))

        b, g, r = small[:, :, 0], small[:, :, 1], small[:, :, 2]
        diff_rg = np.abs(r.astype(np.float32) - g.astype(np.float32))
        diff_gb = np.abs(g.astype(np.float32) - b.astype(np.float32))
        color_diff = float((np.mean(diff_rg) + np.mean(diff_gb)) / 2.0)

        highlight_pixels = np.sum(val >= 250)
        highlight_ratio = float(highlight_pixels / val.size)

        hue = hsv[:, :, 0]
        warm_mask = (hue >= 10) & (hue <= 25) & (sat >= 40)
        warm_ratio = float(np.sum(warm_mask) / hue.size)

        min_c = np.min(small, axis=2)
        dcp_mean = float(np.mean(min_c))

        return {
            "val_mean": val_mean,
            "sat_mean": sat_mean,
            "contrast": contrast,
            "color_diff": color_diff,
            "highlight_ratio": highlight_ratio,
            "warm_ratio": warm_ratio,
            "dcp_mean": dcp_mean
        }

    def _determine_raw_env(self, feats: Dict[str, float]) -> str:
        """Quyết định môi trường tức thời từ đặc trưng frame đơn lẻ."""
        if feats["sat_mean"] < 8.0 and feats["color_diff"] < 3.5 and feats["val_mean"] < 155.0:
            return EnvironmentType.NIGHT_IR
        if feats["highlight_ratio"] > 0.12 and feats["val_mean"] > 160.0:
            return EnvironmentType.GLARE_SUN
        if feats["val_mean"] < 70.0:
            return EnvironmentType.NIGHT_COLOR
        if feats["warm_ratio"] > 0.22 and 65.0 <= feats["val_mean"] <= 155.0:
            return EnvironmentType.DUSK_SUNSET
        if feats["contrast"] < 26.0 and feats["sat_mean"] < 28.0 and feats["val_mean"] > 85.0:
            return EnvironmentType.FOG_HAZE
        return EnvironmentType.DAY_NORMAL

    def classify(self, frame: np.ndarray, timestamp_sec: float) -> Tuple[str, Dict[str, float]]:
        """Phân loại môi trường có bộ đệm Hysteresis chống rung trạng thái."""
        feats = self.extract_features(frame)
        raw_env = self._determine_raw_env(feats)

        if self.is_first_frame:
            self.is_first_frame = False
            self.current_env = raw_env
            self.candidate_env = raw_env
            self.candidate_start_time = timestamp_sec
            return self.current_env, feats

        if raw_env == self.current_env:
            self.candidate_env = self.current_env
            self.candidate_start_time = timestamp_sec
            return self.current_env, feats

        if raw_env != self.candidate_env:
            self.candidate_env = raw_env
            self.candidate_start_time = timestamp_sec
        else:
            if timestamp_sec - self.candidate_start_time >= self.hysteresis_sec:
                self.current_env = self.candidate_env

        return self.current_env, feats


@dataclass
class DynamicSceneProfile:
    """Hồ sơ tham số thích ứng được tính toán tự động cho từng ngoại cảnh."""
    environment: str = EnvironmentType.DAY_NORMAL
    confidence_fire: float = 0.25
    confidence_smoke: float = 0.30
    model_weights: Dict[str, float] = field(default_factory=lambda: {
        "best.pt": 1.0,
        "fire_yolo.pt": 1.0,
        "best.onnx": 0.8
    })
    tau_fire: float = 2.5
    tau_smoke: float = 8.0
    e_alarm: float = 75.0
    e_confirmed: float = 45.0
    e_suspect: float = 20.0
    drift_threshold: float = 4.0
    noise_level: float = 0.0          # 0.0 (rất sạch) -> 1.0 (rất nhiễu)
    camera_shake: float = 0.0         # Pixels dịch chuyển nền
    status_text: str = "Đang tối ưu ngoại cảnh..."
    is_warmup_done: bool = False


class SceneAutoConfigurator:
    """Bộ điều phối tự động cấu hình ngoại cảnh camera & video."""

    def __init__(self, warmup_frames: int = 30, env_hysteresis_sec: float = 2.0) -> None:
        self.warmup_frames = max(10, warmup_frames)
        self.env_classifier = EnvironmentClassifier(hysteresis_sec=env_hysteresis_sec)
        
        # Bộ đệm tích lũy frame đầu
        self.frame_count: int = 0
        self.prev_gray_small: Optional[np.ndarray] = None
        
        # Thống kê đo lường nền
        self.noise_samples: List[float] = []
        self.shake_samples: List[float] = []
        
        # Hồ sơ động hiện tại
        self.current_profile = DynamicSceneProfile()
        
    def reset(self) -> None:
        """Đặt lại bộ đo đạc khi người dùng nạp video mới hoặc tua vị trí."""
        self.frame_count = 0
        self.prev_gray_small = None
        self.noise_samples.clear()
        self.shake_samples.clear()
        self.env_classifier.reset()
        self.current_profile = DynamicSceneProfile()
        logger.info("SceneAutoConfigurator: Đã đặt lại trạng thái để chuẩn bị phân tích ngoại cảnh mới.")

    def _estimate_sensor_noise(self, gray_small: np.ndarray) -> float:
        """
        Đo mức nhiễu cảm biến hạt (sensor noise) thông qua độ biến thiên Laplacian
        trên ảnh xám thu nhỏ. Giá trị chuẩn hóa về [0.0, 1.0].
        """
        if gray_small is None or gray_small.size == 0:
            return 0.0
        # Phương sai Laplacian phản ánh độ xước/nhiễu hạt
        lap_var = float(cv2.Laplacian(gray_small, cv2.CV_64F).var())
        # lap_var thông thường từ 20 (ảnh mịn) đến 400+ (nhiễu hạt đêm nặng)
        norm_noise = max(0.0, min(1.0, (lap_var - 30.0) / 300.0))
        return norm_noise

    def _estimate_camera_shake(self, gray_small: np.ndarray) -> float:
        """
        Đo độ rung lắc camera (Global Motion) giữa 2 frame liên tiếp bằng sai khác vị trí.
        """
        if self.prev_gray_small is None or gray_small is None:
            self.prev_gray_small = gray_small.copy()
            return 0.0

        # Tính độ lệch pha Phase Correlation giữa 2 frame thu nhỏ 160x120
        try:
            f1 = np.float32(self.prev_gray_small)
            f2 = np.float32(gray_small)
            shift, _ = cv2.phaseCorrelate(f1, f2)
            dx, dy = shift
            shake_px = math.sqrt(dx * dx + dy * dy)
            self.prev_gray_small = gray_small.copy()
            # Giới hạn giá trị hợp lý (tránh nhiễu cắt cảnh)
            return min(25.0, float(shake_px))
        except Exception:
            self.prev_gray_small = gray_small.copy()
            return 0.0

    def update(
        self,
        frame: np.ndarray,
        timestamp_sec: float,
        native_fps: float = 25.0
    ) -> DynamicSceneProfile:
        """
        Phân tích frame và cập nhật cấu hình ngoại cảnh thích ứng thời gian thực.
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # 1. Thu nhỏ frame để tính toán siêu nhanh (< 0.5ms)
        small_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 120), interpolation=cv2.INTER_AREA)
        
        # 2. Đo đạc môi trường quang học
        env, feats = self.env_classifier.classify(frame, timestamp_sec)
        
        # 3. Đo đạc độ nhiễu và độ rung camera
        noise = self._estimate_sensor_noise(small_gray)
        shake = self._estimate_camera_shake(small_gray)
        
        self.noise_samples.append(noise)
        self.shake_samples.append(shake)
        if len(self.noise_samples) > 60:
            self.noise_samples.pop(0)
            self.shake_samples.pop(0)
            
        avg_noise = float(np.mean(self.noise_samples)) if self.noise_samples else 0.0
        avg_shake = float(np.mean(self.shake_samples)) if self.shake_samples else 0.0
        
        # 4. Tự động tính toán các tham số thích ứng
        profile = DynamicSceneProfile()
        profile.environment = env
        profile.noise_level = avg_noise
        profile.camera_shake = avg_shake
        profile.is_warmup_done = (self.frame_count >= self.warmup_frames)
        
        # A. Tính ngưỡng tin cậy Lửa & Khói thích ứng
        val_mean = feats.get("val_mean", 100.0)
        highlight_ratio = feats.get("highlight_ratio", 0.0)
        contrast = feats.get("contrast", 30.0)
        
        if env == EnvironmentType.NIGHT_IR:
            # Hồng ngoại ban đêm: Tăng nhạy phát hiện ngọn lửa sớm (0.18), khói giữ ngưỡng 0.40 tránh bụi đêm
            profile.confidence_fire = 0.18
            profile.confidence_smoke = 0.40
            profile.model_weights = {"best.pt": 0.80, "fire_yolo.pt": 1.40, "best.onnx": 0.80}
            profile.status_text = "Tự thích ứng: ĐÊM HỒNG NGOẠI (Tối ưu model IR, nhạy lửa nhen nhóm sớm)"
            
        elif env == EnvironmentType.NIGHT_COLOR:
            # Đêm có màu (đèn đường, bãi xe): Giữ conf cân bằng
            profile.confidence_fire = 0.28
            profile.confidence_smoke = 0.35
            profile.model_weights = {"best.pt": 0.90, "fire_yolo.pt": 1.20, "best.onnx": 0.80}
            profile.status_text = "Tự thích ứng: BAN ĐÊM CÓ ĐÈN (Tăng nhạy lửa bập bùng)"
            
        elif env == EnvironmentType.GLARE_SUN:
            # Ngược sáng chói chang: Nâng ngưỡng lửa để chống bóng mặt trời
            profile.confidence_fire = min(0.38, 0.30 + highlight_ratio * 0.4)
            profile.confidence_smoke = 0.42
            profile.model_weights = {"best.pt": 1.00, "fire_yolo.pt": 1.00, "best.onnx": 0.60}
            profile.status_text = f"Tự thích ứng: NGƯỢC SÁNG / CHÓI (Nâng ngưỡng Lửa {profile.confidence_fire:.2f})"
            
        elif env == EnvironmentType.FOG_HAZE:
            # Sương mù mờ mịt: Nâng khói tránh báo nhầm hơi nước
            profile.confidence_fire = 0.25
            profile.confidence_smoke = 0.42
            profile.model_weights = {"best.pt": 1.00, "fire_yolo.pt": 1.00, "best.onnx": 0.70}
            profile.status_text = "Tự thích ứng: SƯƠNG MÙ / MỜ MỊT (Kháng báo nhầm hơi nước)"
            
        elif env == EnvironmentType.DUSK_SUNSET:
            # Hoàng hôn: Sắc vàng cam nhiều
            profile.confidence_fire = 0.30
            profile.confidence_smoke = 0.35
            profile.model_weights = {"best.pt": 1.10, "fire_yolo.pt": 0.90, "best.onnx": 0.80}
            profile.status_text = "Tự thích ứng: HOÀNG HÔN / ÁNH CAM (Cân bằng tương phản vàng)"
            
        else:
            # BAN NGÀY TIÊU CHUẨN (DAY_NORMAL)
            # Nếu trong phòng tối hoặc kho sạch: Tự động hạ ngưỡng để bắt cực nhạy đốm lửa nhỏ
            if val_mean < 80.0:
                profile.confidence_fire = 0.22
                profile.confidence_smoke = 0.28
                profile.status_text = "Tự thích ứng: TRONG NHÀ / KHO HÀNG (Nhạy tối đa đốm lửa nhỏ)"
            else:
                profile.confidence_fire = 0.25
                profile.confidence_smoke = 0.30
                profile.status_text = "Tự thích ứng: BAN NGÀY CHUẨN (Cân bằng toàn diện)"
            profile.model_weights = {"best.pt": 1.20, "fire_yolo.pt": 0.80, "best.onnx": 0.80}

        # B. Tự động tính toán chu kỳ phân rã tau theo FPS
        # Chuẩn hóa về thời gian thực tế: FPS thấp thì cho phép tau nhỉnh hơn chút để chống đứt track
        fps_factor = max(0.8, min(1.3, 25.0 / max(1.0, native_fps)))
        profile.tau_fire = round(2.5 * fps_factor, 2)
        profile.tau_smoke = round(8.0 * fps_factor, 2)

        # C. Tự động tính toán điểm cảnh báo E_ALARM neo theo mức nhiễu nền
        # Môi trường cực kỳ sạch (nhiễu = 0) -> E_ALARM hạ xuống 60 để báo động nhanh
        # Môi trường nhiều nhiễu hạt (nhiễu cao) -> E_ALARM nâng lên 80-85 để chắc chắn
        profile.e_alarm = round(min(85.0, max(58.0, 58.0 + avg_noise * 27.0)), 1)
        profile.e_confirmed = round(profile.e_alarm * 0.55, 1)
        profile.e_suspect = round(profile.e_alarm * 0.28, 1)

        # D. Tự động bù trừ độ rung lắc camera cho ngưỡng dịch tâm đèn tĩnh
        # Nếu camera bị rung lắc 2.0px -> drift_threshold tự cộng thêm 2.0px
        profile.drift_threshold = round(max(3.5, 4.0 + avg_shake * 1.2), 1)

        self.current_profile = profile
        return self.current_profile

    def get_profile(self) -> DynamicSceneProfile:
        """Lấy hồ sơ ngoại cảnh động hiện tại."""
        return self.current_profile
